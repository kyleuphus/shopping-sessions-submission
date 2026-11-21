//! The pipeline. Stage order here is load-bearing and comes straight from
//! `rules.md`; re-read the relevant section before changing any of it.
//!
//! # Shape: parallel, sharded, no global hash map over events
//!
//! 1. **Lookups** (sequential, small): products, campaigns, window.
//!    **Links** (one sequential task): the link graph and its union-find. Every
//!    ID that appears in `links/` gets a root (its person) and the person's name
//!    (the byte-wise smallest ID in the component). Links runs *concurrently*
//!    with step 2 (`rayon::join`): parse/validate never reads a person, the first
//!    read is dedup routing in step 3. Links itself must stay sequential and in
//!    file/line order -- roots decide shards and row order (see `build_people`).
//!    Afterwards both are read-only and shared by all threads.
//! 2. **Event parse** (parallel over newline-aligned chunks of every file): each
//!    line is split, parsed and *validated* on the spot, and a record goes into
//!    one of `SHARDS` buckets chosen by the hash of its `event_id`. Validating
//!    every copy before dedup is harmless: validity is a pure function of the
//!    copy, and step 3 only ever *uses* the validity of the kept copy, so the
//!    rules.md order "dedup, then validate the kept copy" is preserved exactly
//!    (fixture E5: an invalid kept copy is not rescued by a valid later one).
//! 3. **Dedup** (parallel over event_id shards): probe an open-addressing
//!    table, comparing full event IDs on hash matches and keeping the earliest
//!    ingest copy. Mark the winners, then emit valid winners in record order
//!    to their *person* shards.
//! 4. **Clicks** (parallel over chunks): parsed, validated and routed to the same
//!    person shards. No dedup is needed (see below).
//! 5. **Per person shard** (parallel): sort events by (person, ts), sessionize,
//!    attribute against that person's clicks, aggregate `user_daily` rows and a
//!    shard-local `campaign_daily` map. A person never spans two shards, so only
//!    the small `campaign_daily` maps need merging.
//!
//! ## Person identity across threads
//! An ID in `links/` belongs to the person `root` (a union-find root). An ID not in
//! `links/` is a person on its own, named by itself (rules.md, Linked IDs step 3).
//! Linked users are sharded by their root, unlinked users by the hash of their
//! bytes, so every event and click of one person lands in one shard. Unlinked
//! users are then interned per shard, comparing real bytes.

use std::borrow::Cow;
use std::error::Error;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use rayon::prelude::*;

use crate::csvline::{data_lines, split_fields};
use crate::hash::{fmix64, hash_bytes, FastMap};
use crate::intern::Interner;
use crate::manifest;
use crate::out::{self, CampaignDailyRow, UserDailyRow};
use crate::ts;
use crate::unionfind::UnionFind;

const MS_PER_DAY: i64 = 86_400_000;
const SESSION_GAP_MS: i64 = 1_800_000; // 30 minutes
const LATE_CUTOFF_MS: i64 = 86_400_000; // 24 hours
const SEVEN_DAYS_MS: i64 = 604_800_000;

const TYPE_VIEW: u8 = 0;
const TYPE_CART: u8 = 1;
const TYPE_PURCHASE: u8 = 2;

/// Number of shards for both the dedup and the per-person stage. Many more than
/// cores, so whale persons and Zipf-sized files still balance across threads.
const SHARD_BITS: u32 = 8;
const SHARDS: usize = 1 << SHARD_BITS;

/// Target bytes per parse chunk. Chunks end on a `\n`, so they can be larger.
const CHUNK_BYTES: usize = 8 << 20;

/// Number of user_daily Parquet files, written in parallel.
const OUTPUT_PARTS: usize = 16;

/// `root` value meaning "this user ID is not in links/".
const NO_ROOT: u32 = u32::MAX;

#[inline(always)]
fn shard_of(h: u64) -> usize {
    (h >> (64 - SHARD_BITS)) as usize
}

/// Parse a `price_cents` field: ASCII digits only, per `rules.md`.
/// Leading zeros are fine (`0250` is 250). Returns None on any non-digit or on
/// an empty field. Saturates rather than wrapping, so an absurdly long run of
/// digits fails the range check instead of overflowing into a valid value.
fn parse_digits(b: &[u8]) -> Option<i64> {
    if b.is_empty() {
        return None;
    }
    let mut value: i64 = 0;
    for &byte in b {
        let digit = byte.wrapping_sub(b'0');
        if digit > 9 {
            return None; // catches Unicode digits, signs, dots, exponents
        }
        value = value.saturating_mul(10).saturating_add(digit as i64);
    }
    Some(value)
}

/// Every `*.csv` directly in `dir`. A missing directory yields nothing.
fn csv_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "csv") && p.is_file())
        .collect();
    files.sort();
    files
}

/// mmap every file, skipping empty ones (mmap of a zero-length file errors).
fn map_all(files: &[PathBuf]) -> Result<Vec<Mmap>, Box<dyn Error>> {
    let mut maps = Vec::with_capacity(files.len());
    for path in files {
        let file = std::fs::File::open(path)?;
        if file.metadata()?.len() == 0 {
            continue;
        }
        // Safety: /data is mounted read-only and nothing else writes it during
        // the run, so the mapping cannot be mutated underneath us.
        maps.push(unsafe { Mmap::map(&file)? });
    }
    Ok(maps)
}

/// A newline-aligned byte range of one file. `start == 0` means this chunk holds
/// the file's header line, which is skipped by position (never by content:
/// fixture L3 has a data row identical to the header).
struct Chunk<'a> {
    bytes: &'a [u8],
    has_header: bool,
}

/// Cut every file into chunks of about `CHUNK_BYTES`, each ending just after a
/// `\n` (or at end of file). The largest generated file holds ~1/5 of all rows,
/// so per-file parallelism alone would leave most cores idle.
fn chunk_files<'a>(files: &[&'a [u8]]) -> Vec<Chunk<'a>> {
    let mut chunks = Vec::new();
    for &bytes in files {
        let mut start = 0;
        while start < bytes.len() {
            let mut end = (start + CHUNK_BYTES).min(bytes.len());
            if end < bytes.len() {
                end = match memchr::memchr(b'\n', &bytes[end..]) {
                    Some(p) => end + p + 1,
                    None => bytes.len(),
                };
            }
            chunks.push(Chunk {
                bytes: &bytes[start..end],
                has_header: start == 0,
            });
            start = end;
        }
    }
    chunks
}

/// Call `f` on every non-empty data line of a chunk, split ONLY on `\n` (a `\r`
/// stays part of the line). Empty lines are skipped: rules.md makes them
/// malformed, so skipping them here gives exactly the same answer as handing
/// them to `split_fields` (which would reject them). This matches
/// `csvline::data_lines` line for line otherwise.
#[inline]
fn for_each_line<'a>(chunk: &Chunk<'a>, mut f: impl FnMut(&'a [u8])) {
    let mut s = chunk.bytes;
    if chunk.has_header {
        match memchr::memchr(b'\n', s) {
            Some(p) => s = &s[p + 1..],
            None => return, // header only, no newline: no data
        }
    }
    let mut pos = 0;
    for nl in memchr::memchr_iter(b'\n', s) {
        if nl > pos {
            f(&s[pos..nl]);
        }
        pos = nl + 1;
    }
    if pos < s.len() {
        f(&s[pos..]);
    }
}

/// A nonempty, already unquoted ID or user field in its source chunk.
#[derive(Clone, Copy)]
struct EventField(u32);

impl EventField {
    const OFFSET_BITS: u32 = 24;
    const COMPACT_CHUNK_BYTES: usize = 1 << Self::OFFSET_BITS;
    const OFFSET_MASK: u32 = (1 << Self::OFFSET_BITS) - 1;

    #[inline]
    fn new(bytes: &[u8], field: &[u8]) -> Self {
        // split_fields returns a subslice of the line, itself a subslice of bytes.
        // The parse-loop assertion bounds the offset before this conversion.
        let offset = (field.as_ptr() as usize - bytes.as_ptr() as usize) as u32;
        let len = if bytes.len() <= Self::COMPACT_CHUNK_BYTES && field.len() <= u8::MAX as usize {
            (field.len() as u32) << Self::OFFSET_BITS
        } else {
            0
        };
        Self(offset | len)
    }

    #[inline]
    fn bytes(self, chunk: &[u8]) -> &[u8] {
        let offset = if chunk.len() <= Self::COMPACT_CHUNK_BYTES {
            let len = (self.0 >> Self::OFFSET_BITS) as usize;
            let offset = (self.0 & Self::OFFSET_MASK) as usize;
            if len != 0 {
                return &chunk[offset..offset + len];
            }
            offset
        } else {
            self.0 as usize
        };
        // A long field (or large chunk) stores only the offset. Both fields
        // precede a comma, and neither can contain a comma or quote. Starting
        // after any opening quote, strip the closing quote if present.
        let rest = &chunk[offset..];
        let end = memchr::memchr(b',', rest).expect("parsed event field has a delimiter");
        let field = &rest[..end];
        field.strip_suffix(b"\"").unwrap_or(field)
    }
}

/// One parsed copy of an event. Invalid copies stay through dedup so that an
/// invalid winner still drops the event.
#[derive(Clone, Copy)]
struct ERec {
    h: u64,
    ingest: i64,
    ts: i64,
    id: EventField,
    user: EventField,
    category: u32,
    // Events rule 3 caps purchase prices at 10,000,000 (< 2^24).
    // Other types use price zero, leaving the upper byte for the event type.
    price_and_type: u32,
}

impl ERec {
    const INVALID: u32 = u32::MAX;

    fn valid(&self) -> bool {
        self.price_and_type != Self::INVALID
    }

    fn price(&self) -> u32 {
        self.price_and_type & 0x00FF_FFFF
    }

    fn etype(&self) -> u8 {
        (self.price_and_type >> 24) as u8
    }
}

const _: () = assert!(std::mem::size_of::<ERec>() <= 40);

#[cfg(test)]
#[test]
fn event_field_preserves_long_quoted_id() {
    let id = vec![b'x'; 300];
    let mut line = vec![b'"'];
    line.extend(&id);
    line.extend(b"\",user,ts,ingest,view,product,");
    let mut fields = Vec::new();
    assert!(split_fields(&line, 7, &mut fields));
    assert_eq!(EventField::new(&line, fields[0]).bytes(&line), id);
    assert_eq!(EventField::new(&line, fields[1]).bytes(&line), b"user");
}

/// A surviving (deduped, valid, not late) event on its way to a person shard.
#[derive(Clone, Copy)]
struct PEv<'a> {
    user: &'a [u8],
    root: u32,
    ts: i64,
    price: u32,
    category: u32,
    etype: u8,
}

/// A valid click on its way to a person shard.
#[derive(Clone, Copy)]
struct PCl<'a> {
    user: &'a [u8],
    root: u32,
    ts: i64,
    campaign: u32,
    id: &'a [u8],
}

/// Event within a person shard, keyed by a shard-local person id.
struct Ev {
    pid: u64,
    ts: i64,
    price: u32,
    category: u32,
    etype: u8,
}

struct Cl<'a> {
    pid: u64,
    ts: i64,
    campaign: u32,
    id: &'a [u8],
}

/// Per `(campaign, day)` accumulator for `campaign_daily`.
#[derive(Default, Clone, Copy)]
struct CampaignDay {
    sessions: i64,
    purchasing_sessions: i64,
    revenue: i64,
    assisted_sessions: i64,
}

/// Stage timing, enabled with `FAST_TIMING=1`.
struct Stages {
    enabled: bool,
    last: std::time::Instant,
}

impl Stages {
    fn new() -> Self {
        Self {
            enabled: std::env::var_os("FAST_TIMING").is_some(),
            last: std::time::Instant::now(),
        }
    }

    fn mark(&mut self, label: &str) {
        if self.enabled {
            eprintln!("  {label:<28} {:>8.3}s", self.last.elapsed().as_secs_f64());
        }
        self.last = std::time::Instant::now();
    }

    /// Print a duration measured elsewhere (e.g. on another thread) without
    /// resetting the stage clock.
    fn note(&self, label: &str, d: std::time::Duration) {
        if self.enabled {
            eprintln!("  {label:<28} {:>8.3}s", d.as_secs_f64());
        }
    }
}

/// Read-only lookups for parsing and validating events and clicks: the window,
/// products and campaigns. Built before any parallel stage; needs nothing from
/// `links/`.
struct Lookups<'a> {
    window_start: i64,
    window_end: i64,
    product_category: FastMap<&'a [u8], u32>,
    unknown_category: u32,
    campaigns: Interner<'a>,
    real_campaign_count: u32,
}

/// The resolved link graph: who belongs to which person, and each linked
/// person's name. Read-only once built.
struct People<'a> {
    /// Every ID that appears in a kept links line -> its handle in `link_users`.
    link_users: Interner<'a>,
    /// Handle -> union-find root (its person).
    roots: Vec<u32>,
    /// Root -> handle of the byte-wise smallest ID in that person's component.
    root_name: Vec<u32>,
}

impl People<'_> {
    /// Which person a user ID belongs to, and which person shard that is.
    #[inline]
    fn person(&self, user: &[u8]) -> (u32, usize) {
        match self.link_users.get(user) {
            Some(id) => {
                let root = self.roots[id as usize];
                (root, shard_of(fmix64(root as u64 ^ 0x5555)))
            }
            None => (NO_ROOT, shard_of(hash_bytes(user))),
        }
    }
}

/// Resolve `links/` into persons. Returns the time it took, for FAST_TIMING
/// (it runs concurrently with event parse, so the stage clock can't see it).
///
/// MUST stay sequential, in this exact order (files sorted by `csv_files`, lines
/// in file order): handles are assigned in first-seen order, union-find roots
/// depend on those handles, and the root picks the person shard
/// (`fmix64(root ^ 0x5555)`) and the row order inside it. Any reordering -- e.g.
/// interning in parallel -- changes which part file a user_daily row lands in,
/// and breaks byte-identity of the output.
fn build_people(link_maps: &[Mmap]) -> (People<'_>, std::time::Duration) {
    let started = std::time::Instant::now();
    let mut fields: Vec<&[u8]> = Vec::with_capacity(8);

    // rules.md: "Links come before everything else: an event or click is the
    // person's as soon as it is parsed, whatever its own ID." Logically, yes; in
    // this program nothing reads a person before dedup routing, so links can be
    // built concurrently with event parse (see `run`).
    let mut link_users = Interner::new();
    let mut uf = UnionFind::new(0);
    for map in link_maps {
        for line in data_lines(map) {
            if !split_fields(line, 2, &mut fields) {
                continue;
            }
            // Links-specific rule: a line with either field empty is dropped.
            if fields[0].is_empty() || fields[1].is_empty() {
                continue;
            }
            let a = link_users.intern(fields[0]);
            let b = link_users.intern(fields[1]);
            uf.grow_to(link_users.len());
            uf.union(a, b);
        }
    }
    let n_link = link_users.len();
    let roots: Vec<u32> = (0..n_link as u32).map(|id| uf.find(id)).collect();
    // Name each linked person by the byte-wise smallest of their IDs. This
    // considers every linked ID, including ones never used in an event or click
    // (rules.md, Linked IDs step 3; fixtures L2 and L3). An ID outside links/ is
    // a person on its own and is named by itself, handled in the person stage.
    let mut root_name: Vec<u32> = vec![u32::MAX; n_link];
    for id in 0..n_link as u32 {
        let r = roots[id as usize] as usize;
        let current = root_name[r];
        if current == u32::MAX || link_users.less(id, current) {
            root_name[r] = id;
        }
    }

    (
        People {
            link_users,
            roots,
            root_name,
        },
        started.elapsed(),
    )
}

/// Events, step 1: parse + validate every copy, bucket by event_id hash.
/// Reads only `Lookups`, so it can overlap with `build_people`.
fn parse_events(lk: &Lookups<'_>, event_chunks: &[Chunk<'_>]) -> Vec<Vec<Vec<ERec>>> {
    event_chunks
        .par_iter()
        .map(|chunk| {
            // chunk_files targets 8 MiB but extends to the next newline or EOF.
            // With no line-length bound its only intrinsic ceiling is file size.
            // Enforce the 32-bit offset limit on the actual event chunk.
            assert!(
                chunk.bytes.len() <= u32::MAX as usize,
                "event chunk exceeds u32 offset limit"
            );
            let mut buckets: Vec<Vec<ERec>> = empty_buckets();
            let mut fields: Vec<&[u8]> = Vec::with_capacity(8);
            for_each_line(chunk, |line| {
                if !split_fields(line, 7, &mut fields) {
                    return;
                }
                let id = fields[0];
                if id.is_empty() {
                    return;
                }
                // rules.md: ingest_ts is always valid F1.
                let Some(ingest) = ts::parse_ts(fields[3]) else {
                    return;
                };
                let h = hash_bytes(id);
                let id = EventField::new(chunk.bytes, id);
                let rec = match lk.validate_event(&fields, ingest) {
                    Some((ts, etype, category, price)) => ERec {
                        h,
                        ingest,
                        id,
                        user: EventField::new(chunk.bytes, fields[1]),
                        ts,
                        category,
                        price_and_type: price | (u32::from(etype) << 24),
                    },
                    None => ERec {
                        h,
                        ingest,
                        id,
                        user: EventField(0), // Never read for an invalid winner.
                        ts: 0,
                        category: 0,
                        price_and_type: ERec::INVALID,
                    },
                };
                buckets[shard_of(h)].push(rec);
            });
            buckets
        })
        .collect()
}

impl<'a> Lookups<'a> {
    /// Validate one event copy (rules.md Events steps 3 and 4). Returns
    /// `(ts, etype, category, price)` when the copy would survive.
    #[inline]
    fn validate_event(&self, f: &[&'a [u8]], ingest: i64) -> Option<(i64, u8, u32, u32)> {
        let (user, ts_field, etype_field, product, price_field) = (f[1], f[2], f[4], f[5], f[6]);
        if user.is_empty() || product.is_empty() {
            return None;
        }
        let etype = match etype_field {
            b"view" => TYPE_VIEW,
            b"cart" => TYPE_CART,
            b"purchase" => TYPE_PURCHASE,
            _ => return None, // exact and case-sensitive; ` view` and `Purchase` are out
        };
        let event_ts = ts::parse_ts(ts_field)?;
        if event_ts < self.window_start || event_ts >= self.window_end {
            return None;
        }
        // Late: strictly over 24h. Exactly 24h is kept, and so is an ingest_ts
        // before ts (fixture E6).
        if ingest - event_ts > LATE_CUTOFF_MS {
            return None;
        }
        // Views and carts still require a product ID, checked above. Only
        // purchases contribute price or category to the daily aggregates.
        if etype != TYPE_PURCHASE {
            return Some((event_ts, etype, self.unknown_category, 0));
        }
        let price = parse_digits(price_field)?;
        if !(1..=10_000_000).contains(&price) {
            return None;
        }
        let category = self
            .product_category
            .get(product)
            .copied()
            .unwrap_or(self.unknown_category);
        Some((event_ts, etype, category, price as u32))
    }
}

fn empty_buckets<T>() -> Vec<Vec<T>> {
    (0..SHARDS).map(|_| Vec::new()).collect()
}

/// Turn per-producer bucket arrays into per-shard lists of pieces, moving the
/// vectors (no element copies).
fn transpose<T>(produced: Vec<Vec<Vec<T>>>) -> Vec<Vec<Vec<T>>> {
    let mut by_shard: Vec<Vec<Vec<T>>> = (0..SHARDS).map(|_| Vec::new()).collect();
    for buckets in produced {
        for (s, bucket) in buckets.into_iter().enumerate() {
            if !bucket.is_empty() {
                by_shard[s].push(bucket);
            }
        }
    }
    by_shard
}

/// One session of one person, kept just long enough to attribute it.
struct Session {
    start_ts: i64,
    day: i32,
    revenue: i64,
    has_purchase: bool,
}

/// Count the session starts (sorted) inside the closed interval `[a, b]`, split
/// by the UTC day each start falls on, and add them to `campaign`'s
/// `assisted_sessions`. Days with no such session create no row.
fn add_assists(
    starts: &[i64],
    a: i64,
    b: i64,
    campaign: u32,
    campaign_days: &mut FastMap<(u32, i32), CampaignDay>,
) {
    let mut lo = starts.partition_point(|&s| s < a);
    let hi = starts.partition_point(|&s| s <= b);
    while lo < hi {
        let day = starts[lo].div_euclid(MS_PER_DAY);
        let day_end = (day + 1) * MS_PER_DAY; // first ms of the next day
        let next = lo + starts[lo..hi].partition_point(|&s| s < day_end);
        campaign_days
            .entry((campaign, day as i32))
            .or_default()
            .assisted_sessions += (next - lo) as i64;
        lo = next;
    }
}

/// A person's name (rules.md, Linked IDs step 3): for a linked person the
/// precomputed byte-wise smallest ID of their component; for an unlinked ID,
/// itself.
#[inline]
fn person_name<'a>(
    pid: u64,
    link_users: &Interner<'a>,
    root_name: &[u32],
    local_names: &[&'a [u8]],
) -> &'a [u8] {
    if pid >> 32 == 0 {
        link_users.bytes(root_name[pid as usize])
    } else {
        local_names[(pid & 0xFFFF_FFFF) as usize]
    }
}

/// `top_category`: highest summed purchase price_cents wins; a tie goes to the
/// byte-wise smallest category name; null when there were no purchases.
fn top_category<'c>(cat_rev: &[(u32, i64)], categories: &Interner<'c>) -> Option<Cow<'c, str>> {
    let mut top: Option<(u32, i64)> = None;
    for &(c, r) in cat_rev {
        top = match top {
            None => Some((c, r)),
            Some((bc, br)) => {
                if r > br || (r == br && categories.less(c, bc)) {
                    Some((c, r))
                } else {
                    Some((bc, br))
                }
            }
        };
    }
    top.map(|(c, _)| String::from_utf8_lossy(categories.bytes(c)))
}

/// Shard-local person id: a linked person is its root (< 2^32); an unlinked user
/// ID gets 2^32 + a local index, interned by its exact bytes.
#[inline]
fn local_pid<'d>(
    local: &mut FastMap<&'d [u8], u32>,
    names: &mut Vec<&'d [u8]>,
    user: &'d [u8],
    root: u32,
) -> u64 {
    if root != NO_ROOT {
        return root as u64;
    }
    let next = names.len() as u32;
    let idx = *local.entry(user).or_insert_with(|| {
        names.push(user);
        next
    });
    (1u64 << 32) | idx as u64
}

pub fn run(data: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let mut stages = Stages::new();
    let manifest = manifest::read(data)?;
    // One timestamp parser in the program, so the window bounds cannot drift
    // from the event/click parsing. Fixtures E12 and A10 test these edges.
    let window_start =
        ts::parse_ts(manifest.window_start.as_bytes()).ok_or("manifest: bad window_start")?;
    let window_end =
        ts::parse_ts(manifest.window_end.as_bytes()).ok_or("manifest: bad window_end")?;

    // ---- Map all input files up front.
    let link_maps = map_all(&csv_files(&data.join("links")))?;
    let event_maps = map_all(&csv_files(&data.join("events")))?;
    let click_maps = map_all(&csv_files(&data.join("clicks")))?;

    // ---- Small lookup tables. rules.md says both are clean and uniquely keyed.
    let products_bytes = std::fs::read(data.join("products.csv"))?;
    let campaigns_bytes = std::fs::read(data.join("campaigns.csv"))?;

    let mut categories = Interner::new();
    // `unknown` is the category of any product not in products.csv, and rules.md
    // guarantees no real category uses that name. It competes like any other.
    let unknown_category = categories.intern(b"unknown");

    let mut product_category: FastMap<&[u8], u32> = FastMap::default();
    let mut fields: Vec<&[u8]> = Vec::with_capacity(8);
    for line in data_lines(&products_bytes) {
        if split_fields(line, 2, &mut fields) {
            let category = categories.intern(fields[1]);
            product_category.insert(fields[0], category);
        }
    }

    let mut campaigns = Interner::new();
    let mut channels = Interner::new();
    let mut campaign_channel: Vec<u32> = Vec::new();
    for line in data_lines(&campaigns_bytes) {
        if split_fields(line, 2, &mut fields) {
            let campaign = campaigns.intern(fields[0]);
            let channel = channels.intern(fields[1]);
            if campaign as usize >= campaign_channel.len() {
                campaign_channel.resize(campaign as usize + 1, 0);
            }
            campaign_channel[campaign as usize] = channel;
        }
    }
    // Everything interned above this point is a real campaign. `organic` is
    // interned after, so `campaign >= real_campaign_count` means "not in
    // campaigns.csv" -- which is also how a click naming `organic` literally
    // gets dropped as an unknown campaign.
    let real_campaign_count = campaigns.len() as u32;
    let organic = campaigns.intern(b"organic");
    let organic_channel = channels.intern(b"organic");
    if organic as usize >= campaign_channel.len() {
        campaign_channel.resize(organic as usize + 1, 0);
    }
    campaign_channel[organic as usize] = organic_channel;

    stages.mark("lookups + mmap");

    let lk = Lookups {
        window_start,
        window_end,
        product_category,
        unknown_category,
        campaigns,
        real_campaign_count,
    };

    // ---- Links ‖ event parse, run concurrently.
    // Safe because event parse/validate reads only `lk`: the first read of a
    // person is `people.person()` in dedup routing, after the join. Links stays
    // one sequential task to preserve handle order (see `build_people`). Both
    // closures use the existing Rayon pool.
    let event_files: Vec<&[u8]> = event_maps.iter().map(|m| &m[..]).collect();
    let event_chunks = chunk_files(&event_files);
    let ((people, links_time), parsed) = rayon::join(
        || build_people(&link_maps),
        || parse_events(&lk, &event_chunks),
    );

    stages.mark("links ‖ event parse");
    stages.note("  (links alone)", links_time);

    // Attach the source chunk to each nonempty piece during transpose. Keep
    // the same chunk order as transpose so the winner bitset and emit order agree.
    let mut event_pieces = empty_buckets();
    for (chunk, buckets) in event_chunks.iter().zip(parsed) {
        for (shard, piece) in buckets.into_iter().enumerate() {
            if !piece.is_empty() {
                event_pieces[shard].push((chunk.bytes, piece));
            }
        }
    }

    // ---- Events, step 2: dedup per event_id shard, route survivors by person.
    let deduped: Vec<Vec<Vec<PEv>>> = event_pieces
        .into_par_iter()
        .map(|pieces| {
            // Open-addressing table over this shard's copies, keyed by event_id.
            // A slot holds (hash, location) of the best copy seen so far for one
            // distinct event_id. Two copies are the same event only if their ids
            // are byte-equal: a hash match alone is never trusted, so a 64-bit
            // collision just probes on to another slot (a different event).
            let mut piece_bases = Vec::with_capacity(pieces.len());
            let mut n = 0usize;
            for (_, piece) in &pieces {
                piece_bases.push(n);
                n += piece.len();
            }
            // Round up from a 2/3 worst-case load. Starting at twice the
            // copy count can double the allocation and final slot scan when
            // the shard size sits just above half a power of two.
            let cap = (n + n.div_ceil(2)).next_power_of_two().max(16);
            let mask = cap - 1;
            const EMPTY: u64 = u64::MAX;
            // location = piece << 32 | offset
            let mut slots: Vec<(u64, u64)> = vec![(0, EMPTY); cap];
            for (pi, (chunk, piece)) in pieces.iter().enumerate() {
                for (oi, r) in piece.iter().enumerate() {
                    let loc = ((pi as u64) << 32) | oi as u64;
                    // Low bits pick the slot; the top bits already picked the shard.
                    let mut slot = (r.h as usize) & mask;
                    loop {
                        let (sh, sloc) = slots[slot];
                        if sloc == EMPTY {
                            slots[slot] = (r.h, loc);
                            break;
                        }
                        if sh == r.h {
                            let (kept_chunk, kept_piece) = &pieces[(sloc >> 32) as usize];
                            let kept = &kept_piece[sloc as u32 as usize];
                            if kept.id.bytes(kept_chunk) == r.id.bytes(chunk) {
                                // Same event: keep the smallest ingest_ts. Equal
                                // ingest_ts copies are identical (rules.md), so a
                                // tie keeps either.
                                if r.ingest < kept.ingest {
                                    slots[slot].1 = loc;
                                }
                                break;
                            }
                        }
                        slot = (slot + 1) & mask;
                    }
                }
            }
            // Mark final slot locations without loading the records in hash
            // order. Then stream the pieces using one bit per input record.
            let mut winners = vec![0u64; n.div_ceil(64)];
            for (_, loc) in slots {
                if loc != EMPTY {
                    let piece = (loc >> 32) as usize;
                    let offset = loc as u32 as usize;
                    let index = piece_bases[piece] + offset;
                    winners[index / 64] |= 1u64 << (index % 64);
                }
            }
            let records = pieces
                .iter()
                .flat_map(|(chunk, piece)| piece.iter().map(move |r| (*chunk, r)));
            let survivors = records.enumerate().filter_map(|(index, (chunk, r))| {
                // Validity is that of the KEPT copy: an invalid kept copy drops
                // the event, and a later valid copy does not rescue it (E5).
                let won = winners[index / 64] & (1u64 << (index % 64)) != 0;
                (won && r.valid()).then_some((chunk, r))
            });
            let mut counts = [0usize; SHARDS];
            let mut routes: Vec<(u32, u32)> = Vec::with_capacity(n);
            for (chunk, r) in survivors.clone() {
                let user = r.user.bytes(chunk);
                let (root, shard) = people.person(user);
                // Shard indices fit in u32, keeping each cached route at 8 bytes.
                routes.push((root, shard as u32));
                counts[shard] += 1;
            }
            let mut out: Vec<Vec<PEv>> = counts.into_iter().map(Vec::with_capacity).collect();
            // Both passes visit the same survivors in record order, so the
            // cached routes line up with their records and bucket order is kept.
            for ((chunk, r), (root, shard)) in survivors.zip(routes) {
                out[shard as usize].push(PEv {
                    user: r.user.bytes(chunk),
                    root,
                    ts: r.ts,
                    price: r.price(),
                    category: r.category,
                    etype: r.etype(),
                });
            }
            out
        })
        .collect();

    stages.mark("event dedup");

    // ---- Clicks: parse + validate, route by person.
    // No dedup is needed. rules.md: copies of a click have identical values. A
    // copy changes neither the credit (latest ts, then smallest click_id: the copy
    // has the same ts, id and campaign, so the same winner) nor the assists (a set
    // of campaigns). So duplicates cannot affect any output.
    let click_window_start = window_start - SEVEN_DAYS_MS;
    let click_files: Vec<&[u8]> = click_maps.iter().map(|m| &m[..]).collect();
    let click_chunks = chunk_files(&click_files);
    let clicks: Vec<Vec<Vec<PCl>>> = click_chunks
        .par_iter()
        .map(|chunk| {
            let mut buckets: Vec<Vec<PCl>> = empty_buckets();
            let mut fields: Vec<&[u8]> = Vec::with_capacity(8);
            for_each_line(chunk, |line| {
                if !split_fields(line, 4, &mut fields) {
                    return;
                }
                let (click_id, user, campaign_field, ts_field) =
                    (fields[0], fields[1], fields[2], fields[3]);
                if click_id.is_empty() || user.is_empty() {
                    return;
                }
                let Some(click_ts) = ts::parse_ts(ts_field) else {
                    return;
                };
                // The click window reaches 7 days before the event window.
                if click_ts < click_window_start || click_ts >= window_end {
                    return;
                }
                // Must name a campaign that is actually in campaigns.csv.
                let Some(campaign) = lk.campaigns.get(campaign_field) else {
                    return;
                };
                if campaign >= lk.real_campaign_count {
                    return;
                }
                let (root, shard) = people.person(user);
                buckets[shard].push(PCl {
                    user,
                    root,
                    ts: click_ts,
                    campaign,
                    id: click_id,
                });
            });
            buckets
        })
        .collect();

    stages.mark("clicks");

    // ---- Per person shard: sessionize, attribute, aggregate.
    let event_shards = transpose(deduped);
    let click_shards = transpose(clicks);
    let link_users = &people.link_users;
    let root_name = &people.root_name;
    let shard_results: Vec<(Vec<UserDailyRow>, FastMap<(u32, i32), CampaignDay>)> = event_shards
        .into_par_iter()
        .zip(click_shards.into_par_iter())
        .map(|(ev_pieces, cl_pieces)| {
            // Shard-local person ids: a linked person is its root (< 2^32); an
            // unlinked user ID gets 2^32 + a local index, interned by its bytes.
            let mut local: FastMap<&[u8], u32> = FastMap::default();
            let mut local_names: Vec<&[u8]> = Vec::new();
            let n_ev: usize = ev_pieces.iter().map(|p| p.len()).sum();
            let mut evs: Vec<Ev> = Vec::with_capacity(n_ev);
            for piece in &ev_pieces {
                for e in piece {
                    evs.push(Ev {
                        pid: local_pid(&mut local, &mut local_names, e.user, e.root),
                        ts: e.ts,
                        price: e.price,
                        category: e.category,
                        etype: e.etype,
                    });
                }
            }
            drop(ev_pieces);
            let n_cl: usize = cl_pieces.iter().map(|p| p.len()).sum();
            let mut cls: Vec<Cl> = Vec::with_capacity(n_cl);
            for piece in &cl_pieces {
                for c in piece {
                    cls.push(Cl {
                        pid: local_pid(&mut local, &mut local_names, c.user, c.root),
                        ts: c.ts,
                        campaign: c.campaign,
                        id: c.id,
                    });
                }
            }
            drop(cl_pieces);
            // Assists view: (person, campaign, ts), so each campaign's clicks for a
            // person are contiguous and time-ordered.
            let mut assist_clicks: Vec<(u64, u32, i64)> =
                cls.iter().map(|c| (c.pid, c.campaign, c.ts)).collect();
            assist_clicks.sort_unstable();

            // Sessionize: sort by (person, ts). A gap OVER 30 min starts a session.
            // The gap is measured between distinct timestamps, so events sharing a
            // ts always land in the same session whatever their order (E16).
            evs.sort_unstable_by_key(|e| (e.pid, e.ts));
            // Clicks by (person, ts, click_id) so the crediting click can be found
            // by position: the FIRST entry at the max qualifying ts has the
            // smallest click_id (fixture A5).
            cls.sort_unstable_by(|a, b| {
                a.pid
                    .cmp(&b.pid)
                    .then(a.ts.cmp(&b.ts))
                    .then_with(|| a.id.cmp(b.id))
            });

            let mut user_rows: Vec<UserDailyRow> = Vec::new();
            let mut campaign_days: FastMap<(u32, i32), CampaignDay> = FastMap::default();
            let mut sess: Vec<Session> = Vec::new();
            let mut starts: Vec<i64> = Vec::new();
            let mut ai = 0;
            let mut cat_rev: Vec<(u32, i64)> = Vec::new();

            let mut ci = 0;
            let mut i = 0;
            while i < evs.len() {
                let pid = evs[i].pid;
                let mut j = i + 1;
                while j < evs.len() && evs[j].pid == pid {
                    j += 1;
                }
                while ci < cls.len() && cls[ci].pid < pid {
                    ci += 1;
                }
                let mut ce = ci;
                while ce < cls.len() && cls[ce].pid == pid {
                    ce += 1;
                }
                let pclicks = &cls[ci..ce];
                ci = ce;
                while ai < assist_clicks.len() && assist_clicks[ai].0 < pid {
                    ai += 1;
                }
                let mut ae = ai;
                while ae < assist_clicks.len() && assist_clicks[ae].0 == pid {
                    ae += 1;
                }
                let passist = &assist_clicks[ai..ae];
                ai = ae;
                sess.clear();

                // One user-day group at a time. Sessions come out in start_ts
                // order, so a person's days are non-decreasing and each
                // (person, day) group is contiguous.
                let mut cur_day = i32::MIN;
                let (mut g_sessions, mut g_events, mut g_revenue) = (0i64, 0i64, 0i64);
                cat_rev.clear();

                let mut k = i;
                while k < j {
                    let start_ts = evs[k].ts;
                    let day = start_ts.div_euclid(MS_PER_DAY) as i32;
                    if day != cur_day {
                        if g_sessions > 0 {
                            user_rows.push(UserDailyRow {
                                user_id: String::from_utf8_lossy(person_name(pid, link_users, root_name, &local_names)),
                                day: cur_day,
                                sessions: g_sessions,
                                events: g_events,
                                revenue_cents: g_revenue,
                                top_category: top_category(&cat_rev, &categories),
                            });
                        }
                        cur_day = day;
                        g_sessions = 0;
                        g_events = 0;
                        g_revenue = 0;
                        cat_rev.clear();
                    }
                    let mut previous_ts = start_ts;
                    let mut events = 0i64;
                    let mut revenue = 0i64;
                    let mut has_purchase = false;
                    while k < j && evs[k].ts - previous_ts <= SESSION_GAP_MS {
                        let e = &evs[k];
                        events += 1;
                        if e.etype == TYPE_PURCHASE {
                            let p = e.price as i64;
                            revenue += p;
                            has_purchase = true;
                            match cat_rev.iter_mut().find(|(c, _)| *c == e.category) {
                                Some(slot) => slot.1 += p,
                                None => cat_rev.push((e.category, p)),
                            }
                        }
                        previous_ts = e.ts;
                        k += 1;
                    }
                    g_sessions += 1;
                    g_events += events;
                    g_revenue += revenue;

                    sess.push(Session {
                        start_ts,
                        day,
                        revenue,
                        has_purchase,
                    });
                }
                if g_sessions > 0 {
                    user_rows.push(UserDailyRow {
                        user_id: String::from_utf8_lossy(person_name(pid, link_users, root_name, &local_names)),
                        day: cur_day,
                        sessions: g_sessions,
                        events: g_events,
                        revenue_cents: g_revenue,
                        top_category: top_category(&cat_rev, &categories),
                    });
                }

                // ---- Credit (rules.md Clicks 4-5). Among the person's clicks with
                // start - 7 days <= ts <= start, the latest ts wins, then the
                // smallest click_id. Sessions and clicks are both in time order,
                // so one forward pointer serves every session. `run_start` is the
                // first click sharing the latest ts seen so far; clicks are sorted
                // by (ts, click_id), so it holds the smallest click_id (A5).
                let mut p = 0;
                let mut run_start = 0;
                for s in &sess {
                    while p < pclicks.len() && pclicks[p].ts <= s.start_ts {
                        if p == 0 || pclicks[p].ts != pclicks[p - 1].ts {
                            run_start = p;
                        }
                        p += 1;
                    }
                    // A click exactly at the start counts; one exactly 7 days
                    // before counts; 1 ms more does not (A4, A10).
                    let credited = if p > 0 && s.start_ts - pclicks[p - 1].ts <= SEVEN_DAYS_MS {
                        pclicks[run_start].campaign
                    } else {
                        organic
                    };
                    let entry = campaign_days.entry((credited, s.day)).or_default();
                    entry.sessions += 1;
                    entry.revenue += s.revenue;
                    if s.has_purchase {
                        entry.purchasing_sessions += 1;
                    }
                }

                // ---- Assists (rules.md Clicks 7). Campaign c assists a session
                // iff some click of c has start - 7d <= click.ts <= start, i.e.
                // start lies in [click.ts, click.ts + 7d]. Per campaign, merge
                // those intervals so overlapping clicks are not double counted,
                // then count the session starts inside each merged interval,
                // split at UTC midnights because a session counts on the day it
                // starts. Never sessions x clicks: whales and shared-device
                // persons would make that quadratic. `organic` never assists: it
                // is never a real campaign, so it never appears in a click.
                if !passist.is_empty() {
                    starts.clear();
                    starts.extend(sess.iter().map(|s| s.start_ts));
                    let mut x = 0;
                    while x < passist.len() {
                        let campaign = passist[x].1;
                        let mut a = passist[x].2;
                        let mut b = a + SEVEN_DAYS_MS;
                        x += 1;
                        loop {
                            let more = x < passist.len() && passist[x].1 == campaign;
                            if more && passist[x].2 <= b {
                                b = passist[x].2 + SEVEN_DAYS_MS;
                                x += 1;
                                continue;
                            }
                            add_assists(&starts, a, b, campaign, &mut campaign_days);
                            if !more {
                                break;
                            }
                            a = passist[x].2;
                            b = a + SEVEN_DAYS_MS;
                            x += 1;
                        }
                    }
                }
                i = j;
            }
            (user_rows, campaign_days)
        })
        .collect();

    stages.mark("sessionize + attribution");

    // ---- Merge shard results. user_daily rows are disjoint across shards (a
    // person lives in exactly one); campaign_daily needs summing.
    let mut user_parts: Vec<Vec<UserDailyRow>> = Vec::with_capacity(SHARDS);
    let mut campaign_days: FastMap<(u32, i32), CampaignDay> = FastMap::default();
    for (rows, days) in shard_results {
        user_parts.push(rows);
        for (key, d) in days {
            let e = campaign_days.entry(key).or_default();
            e.sessions += d.sessions;
            e.purchasing_sessions += d.purchasing_sessions;
            e.revenue += d.revenue;
            e.assisted_sessions += d.assisted_sessions;
        }
    }

    let mut campaign_rows: Vec<CampaignDailyRow> = Vec::with_capacity(campaign_days.len());
    for ((campaign, day), accumulated) in &campaign_days {
        let channel = campaign_channel[*campaign as usize];
        campaign_rows.push(CampaignDailyRow {
            campaign_id: String::from_utf8_lossy(lk.campaigns.bytes(*campaign)),
            channel: String::from_utf8_lossy(channels.bytes(channel)),
            day: *day,
            sessions: accumulated.sessions,
            purchasing_sessions: accumulated.purchasing_sessions,
            revenue_cents: accumulated.revenue,
            assisted_sessions: accumulated.assisted_sessions,
        });
    }

    // Cheap invariants straight out of rules.md. They catch whole classes of bug
    // with no diffing, so they are checked on every run.
    let user_sessions: i64 = user_parts.iter().flatten().map(|r| r.sessions).sum();
    let campaign_sessions: i64 = campaign_rows.iter().map(|r| r.sessions).sum();
    let user_revenue: i64 = user_parts.iter().flatten().map(|r| r.revenue_cents).sum();
    let campaign_revenue: i64 = campaign_rows.iter().map(|r| r.revenue_cents).sum();
    if user_sessions != campaign_sessions || user_revenue != campaign_revenue {
        eprintln!(
            "INVARIANT VIOLATED: sessions {user_sessions} vs {campaign_sessions}, \
             revenue {user_revenue} vs {campaign_revenue}"
        );
    }

    stages.mark("merge + build output rows");

    // user_daily is written as OUTPUT_PARTS files in parallel, each holding a
    // contiguous group of person shards (rules.md: one or more *.parquet files,
    // file count does not matter). Part 0 is always written, so an empty table
    // still has a file carrying the schema (fixtures E5, E13).
    let user_dir = output.join("user_daily");
    std::fs::create_dir_all(&user_dir)?;
    let per_part = SHARDS.div_ceil(OUTPUT_PARTS);
    let write_results: Vec<Result<(), String>> = user_parts
        .par_chunks(per_part)
        .enumerate()
        .map(|(part, group)| {
            let n: usize = group.iter().map(|g| g.len()).sum();
            if n == 0 && part != 0 {
                return Ok(());
            }
            out::write_user_daily(&user_dir, part, group.iter().flatten())
                .map_err(|e| e.to_string())
        })
        .collect();
    for r in write_results {
        r?;
    }
    out::write_campaign_daily(&output.join("campaign_daily"), &campaign_rows)?;
    stages.mark("write parquet");
    Ok(())
}
