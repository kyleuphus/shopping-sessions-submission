//! v3 pipeline core, selected with `FAST_V3=1`. Work order: `PLAN-v3.md`.
//!
//! The shape (see PLAN-v3 §3/§4):
//!
//! * **L** links -> `PersonMap` (reused from `links2`).
//! * **P** one parallel pass over every event and click chunk. Each line is
//!   split, its IDs packed to `IdKey`, its person resolved, and its validity
//!   decided *per copy*. It emits small fixed-size records into thread-local
//!   staging, then each chunk does one counting scatter into one contiguous
//!   buffer ordered by partition (with a prefix-sum offset table):
//!     - `Tok` (16 B) per event copy with a non-empty `event_id`, ordered by
//!       *event* partition. Its `rec` points at the copy's `Ev`, or is `NONE`
//!       when the copy is invalid or late.
//!     - `Ev` (16 B) per valid, non-late event copy, ordered by *person*
//!       partition.
//!     - `Ck` (24 B) per valid click, ordered by person partition.
//! * **D** dedup per event partition: group tokens by event key, the copy
//!   with the smallest ingest wins. If the winner is invalid every copy's `Ev`
//!   is marked dead (dedup comes before validation, rules Events 2); otherwise
//!   every loser's `Ev` is marked dead. Marks go into per-chunk dead bitmaps.
//! * **S** person stage per person partition, largest first: gather the live
//!   `Ev`s and the `Ck`s, sort, then pipeline2's sessionize / credit / assist /
//!   aggregate walk (semantics copied exactly).
//! * **O** output rows -> Parquet via `out.rs`.
//!
//! Validity computed at parse does not break "dedup before validation":
//! dedup only *selects* one copy; validity only decides whether the selected
//! copy contributes. Both are per-copy facts, so the order of computing them
//! does not matter, as long as an invalid copy still takes part in dedup
//! (it does: it still emits a `Tok`, with `rec = NONE`).
//!
//! Narrow fields (u32 relative ms, u8 category, u16 campaign, u32 seconds,
//! packed record refs) are protected by width guards. If any guard fails the
//! run falls back to v1 (`crate::pipeline::run`) *before anything is written*
//! (PLAN-v3 §2). Generator data never trips a guard.

use std::borrow::Cow;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use memmap2::Mmap;
use rayon::prelude::*;

use crate::csvline::{data_lines, split_fields};
use crate::hash::{fmix64, FastMap};
use crate::intern::Interner;
use crate::key::{key_bytes, key_cmp, key_of, IdInterner, IdKey};
use crate::links2::{self, PersonMap};
use crate::manifest;
use crate::out::{self, CampaignDailyRow, UserDailyRow};
use crate::ts;

const MS_PER_DAY: i64 = 86_400_000;
const SESSION_GAP_MS: i64 = 1_800_000;
const LATE_CUTOFF_MS: i64 = 86_400_000;
const SEVEN_DAYS_MS: i64 = 604_800_000;

const CHUNK_BYTES: usize = 8 << 20;
const MAX_DENSE_CELLS: usize = 8 << 20;
/// Aim for this many bytes of event input per person/event partition, which
/// puts a partition's `Ev` + `Ck` working set at a few hundred KB (L2-sized).
const INPUT_BYTES_PER_PARTITION: u64 = 3 << 19; // 1.5 MiB
const MIN_PARTITION_BITS: u32 = 4;
const MAX_PARTITION_BITS: u32 = 12; // partition ids must fit the u16 staging tags

/// `Tok::rec` value meaning "this copy is invalid or late: it has no `Ev`".
const NONE: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

/// One event copy's dedup token.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
struct Tok {
    /// The `event_id`, packed (see key.rs). Raw equality is exact for packed
    /// and fallback keys alike, which is all grouping needs.
    ekey: IdKey,
    /// `ingest_ts` in whole epoch seconds (guarded to fit u32).
    ingest_s: u32,
    /// `(event chunk id << k) | index of this copy's Ev in that chunk's buffer`,
    /// or `NONE` for an invalid / late copy.
    rec: u32,
}

/// One valid, non-late event copy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
struct Ev {
    /// The person's name key: `PersonMap::get(user).unwrap_or(user)`.
    person: IdKey,
    /// `ts - base`, with `base = window_start - 7 days` (guarded to fit u32).
    ts_rel: u32,
    /// `price << 8 | category`. View and cart events store 0, so "is a purchase"
    /// is exactly `price > 0` (a valid purchase has price >= 1). Price is at
    /// most 10,000,000 < 2^24, so it fits in the top 24 bits.
    pc: u32,
}

impl Ev {
    #[inline(always)]
    fn price(self) -> u32 {
        self.pc >> 8
    }
    #[inline(always)]
    fn category(self) -> u8 {
        self.pc as u8
    }
}

/// One valid click (no click dedup: identical copies cannot change which click
/// gets credit or the set of assisting campaigns).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
struct Ck {
    person: IdKey,
    click: IdKey,
    ts_rel: u32,
    camp: u16,
}

const _: () = assert!(std::mem::size_of::<Tok>() == 16);
const _: () = assert!(std::mem::size_of::<Ev>() == 16);
const _: () = assert!(std::mem::size_of::<Ck>() <= 24);

// ---------------------------------------------------------------------------
// Escape hatch (PLAN-v3 §2)
// ---------------------------------------------------------------------------

/// Set when any width guard fails. Owned by one run (not a process global) so
/// unit tests can exercise it in parallel; it plays the role of the plan's
/// global `FALLBACK` flag.
#[derive(Default)]
struct Fallback {
    tripped: AtomicBool,
    reason: OnceLock<String>,
}

impl Fallback {
    fn trip(&self, reason: impl Into<String>) {
        let _ = self.reason.set(reason.into());
        self.tripped.store(true, Ordering::Relaxed);
    }
    fn is_set(&self) -> bool {
        self.tripped.load(Ordering::Relaxed)
    }
    fn reason(&self) -> String {
        self.reason.get().cloned().unwrap_or_default()
    }
}

/// Guard: every stored `ts_rel` lies in `[0, window_end - base)`; that range
/// must fit u32. Returns the failure reason, if any.
fn window_guard(window_start: i64, window_end: i64) -> Option<String> {
    // Overflow anywhere counts as "does not fit" (span = None).
    let span = window_start
        .checked_sub(SEVEN_DAYS_MS)
        .and_then(|base| window_end.checked_sub(base));
    match span {
        Some(span) if span <= i64::from(u32::MAX) => None,
        _ => Some(format!(
            "window_end - (window_start - 7d) = {span:?} ms does not fit u32"
        )),
    }
}

/// Bits for the in-chunk index part of `Tok::rec`, given the number of event
/// chunks. Chunk ids `0..n` take the top `32 - k` bits; requiring
/// `n + 1 <= 2^(32-k)` keeps the largest chunk id below all-ones, so no real
/// `rec` can equal `NONE`. Each chunk's `Ev` count must then be `<= 2^k`
/// (checked per chunk at parse).
fn rec_index_bits(event_chunks: usize) -> u32 {
    let chunk_bits = usize::BITS - event_chunks.leading_zeros(); // bits to hold n
    // n < 2^chunk_bits; we need n + 1 <= 2^(32-k) i.e. 32 - k >= bits(n).
    // bits(n) is enough: n <= 2^b - 1 implies n - 1 <= 2^b - 2 < all-ones.
    32 - chunk_bits.max(1)
}

// ---------------------------------------------------------------------------
// Files and chunks (copied from pipeline2)
// ---------------------------------------------------------------------------

fn csv_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<_> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "csv") && path.is_file())
        .collect();
    files.sort();
    files
}

fn map_all(files: &[PathBuf]) -> Result<Vec<Mmap>, Box<dyn Error>> {
    let mut maps = Vec::with_capacity(files.len());
    for path in files {
        let file = std::fs::File::open(path)?;
        if file.metadata()?.len() != 0 {
            // SAFETY: scorer input is read-only for the lifetime of the run.
            maps.push(unsafe { Mmap::map(&file)? });
        }
    }
    Ok(maps)
}

#[derive(Clone, Copy)]
struct Chunk<'a> {
    bytes: &'a [u8],
    has_header: bool,
}

/// Split files into ~8 MiB chunks ending on a newline. Only a file's first
/// chunk carries its header line.
fn chunk_files<'a>(files: &[&'a [u8]]) -> Vec<Chunk<'a>> {
    let mut chunks = Vec::new();
    for &bytes in files {
        let mut start = 0;
        while start < bytes.len() {
            let mut end = (start + CHUNK_BYTES).min(bytes.len());
            if end < bytes.len() {
                end = memchr::memchr(b'\n', &bytes[end..])
                    .map_or(bytes.len(), |offset| end + offset + 1);
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

/// Call `f` on every non-empty line (empty lines are malformed: skipped).
#[inline]
fn for_each_line<'a>(chunk: &Chunk<'a>, mut f: impl FnMut(&'a [u8])) {
    let mut bytes = chunk.bytes;
    if chunk.has_header {
        match memchr::memchr(b'\n', bytes) {
            Some(newline) => bytes = &bytes[newline + 1..],
            None => return,
        }
    }
    let mut start = 0;
    for newline in memchr::memchr_iter(b'\n', bytes) {
        if newline > start {
            f(&bytes[start..newline]);
        }
        start = newline + 1;
    }
    if start < bytes.len() {
        f(&bytes[start..]);
    }
}

fn parse_digits(bytes: &[u8]) -> Option<i64> {
    if bytes.is_empty() {
        return None;
    }
    let mut value = 0i64;
    for &byte in bytes {
        let digit = byte.wrapping_sub(b'0');
        if digit > 9 {
            return None;
        }
        value = value.saturating_mul(10).saturating_add(digit as i64);
    }
    Some(value)
}

// ---------------------------------------------------------------------------
// Timers
// ---------------------------------------------------------------------------

struct Stages {
    enabled: bool,
    last: Instant,
}

impl Stages {
    fn new() -> Self {
        Self {
            enabled: std::env::var_os("FAST_TIMING").is_some(),
            last: Instant::now(),
        }
    }

    fn mark(&mut self, label: &str) {
        if self.enabled {
            eprintln!("  v3 {label:<25} {:>8.3}s", self.last.elapsed().as_secs_f64());
        }
        self.last = Instant::now();
    }
}

// ---------------------------------------------------------------------------
// Partitioning
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Parts {
    /// log2 of the partition count (same count for event and person sides).
    bits: u32,
}

impl Parts {
    fn for_input(event_bytes: u64) -> Self {
        let want = (event_bytes / INPUT_BYTES_PER_PARTITION).max(1);
        let bits = (64 - (want - 1).leading_zeros()).clamp(MIN_PARTITION_BITS, MAX_PARTITION_BITS);
        Self { bits }
    }
    fn count(self) -> usize {
        1 << self.bits
    }
    /// Top bits of a salted fmix64, as in v2.
    #[inline(always)]
    fn event(self, key: IdKey) -> usize {
        (fmix64(key ^ 0x6576_656e_745f_7633) >> (64 - self.bits)) as usize
    }
    #[inline(always)]
    fn person(self, key: IdKey) -> usize {
        (fmix64(key ^ 0x7065_7273_6f6e_7633) >> (64 - self.bits)) as usize
    }
}

/// Stable counting scatter: put `items[i]` into group `tags[i]`, groups in
/// order, items within a group in input order. Returns the scattered items,
/// the offset table (`groups + 1` entries; group g is `off[g]..off[g+1]`), and
/// each input item's final position.
fn counting_scatter<T: Copy + Default>(items: &[T], tags: &[u16], groups: usize) -> (Vec<T>, Vec<u32>, Vec<u32>) {
    let mut off = vec![0u32; groups + 1];
    for &t in tags {
        off[t as usize + 1] += 1;
    }
    for g in 0..groups {
        off[g + 1] += off[g];
    }
    let mut cursor = off[..groups].to_vec();
    let mut out = vec![T::default(); items.len()];
    let mut pos = Vec::with_capacity(items.len());
    for (item, &t) in items.iter().zip(tags) {
        let at = &mut cursor[t as usize];
        out[*at as usize] = *item;
        pos.push(*at);
        *at += 1;
    }
    (out, off, pos)
}

// ---------------------------------------------------------------------------
// Parse (stage P)
// ---------------------------------------------------------------------------

/// Read-only state shared by every parse task.
struct Ctx<'a> {
    window_start: i64,
    window_end: i64,
    /// `window_start - 7 days`: the zero of every `ts_rel`.
    base: i64,
    /// product_id bytes -> index into the byte-sorted category name list.
    product_cat: FastMap<&'a [u8], u8>,
    unknown_cat: u8,
    campaigns: Interner<'a>,
    real_campaign_count: u32,
    people: &'a PersonMap,
    ids: &'a IdInterner,
    parts: Parts,
    /// See [`rec_index_bits`].
    k: u32,
    fallback: &'a Fallback,
}

impl Ctx<'_> {
    /// rules Events 3-4 for one copy: `Some((ts, pc))` if valid and not late.
    /// (Same checks as pipeline2's `validate_event`.)
    #[inline]
    fn validate_event(&self, fields: &[&[u8]], ingest_ms: i64) -> Option<(i64, u32)> {
        let user = fields[1];
        let product = fields[5];
        if user.is_empty() || product.is_empty() {
            return None;
        }
        let is_purchase = match fields[4] {
            b"view" | b"cart" => false,
            b"purchase" => true,
            _ => return None,
        };
        let event_ts = ts::parse_ts(fields[2])?;
        if event_ts < self.window_start || event_ts >= self.window_end {
            return None;
        }
        // Late: "over 24 hours" is dropped; exactly 24 h, or ingest before ts, kept.
        if ingest_ms - event_ts > LATE_CUTOFF_MS {
            return None;
        }
        if !is_purchase {
            return Some((event_ts, 0));
        }
        let price = parse_digits(fields[6])?;
        if !(1..=10_000_000).contains(&price) {
            return None;
        }
        let cat = self.product_cat.get(product).copied().unwrap_or(self.unknown_cat);
        Some((event_ts, (price as u32) << 8 | u32::from(cat)))
    }
}

/// One event chunk after its scatter.
struct EvChunk {
    /// Valid copies, ordered by person partition.
    evs: Vec<Ev>,
    /// Person partition p is `evs[ev_off[p]..ev_off[p+1]]`.
    ev_off: Vec<u32>,
    /// Tokens, ordered by event partition.
    toks: Vec<Tok>,
    tok_off: Vec<u32>,
    /// One bit per `evs` entry; set = killed by dedup.
    dead: Vec<AtomicU64>,
}

/// One click chunk after its scatter.
struct CkChunk {
    cks: Vec<Ck>,
    ck_off: Vec<u32>,
    /// Some click key in this chunk is a fallback (unpacked) key, whose raw
    /// u64 order is not byte order. See `sort_clicks`.
    has_fallback_click: bool,
}

fn parse_event_chunk(ctx: &Ctx<'_>, chunk: &Chunk<'_>, chunk_id: usize) -> EvChunk {
    let parts = ctx.parts;
    let mut evs = Vec::new();
    let mut ev_tag = Vec::new();
    let mut toks = Vec::new();
    let mut tok_tag = Vec::new();
    let mut fields = Vec::with_capacity(8);
    for_each_line(chunk, |line| {
        // rules Events 1: malformed lines and empty event_id produce nothing.
        if !split_fields(line, 7, &mut fields) || fields[0].is_empty() {
            return;
        }
        // rules: "ingest_ts is always valid F1", so use the F1-only parser.
        // Anything else drops the line (never happens on spec-conforming data).
        let Some(ingest_ms) = ts::parse_ingest_ts(fields[3]) else {
            return;
        };
        // Width guard: ingest is stored as u32 seconds. F1 is whole seconds by
        // construction, so only the range needs checking. One unsigned compare:
        // a negative ingest_ms casts to a huge u64 and fails too.
        if ingest_ms as u64 > u64::from(u32::MAX) * 1000 {
            ctx.fallback.trip(format!("ingest_ts {ingest_ms} ms is not whole u32 seconds"));
            return;
        }
        let ekey = key_of(fields[0], ctx.ids);
        // `rec` here is the staging index; it is remapped after the scatter.
        let rec = match ctx.validate_event(&fields, ingest_ms) {
            Some((event_ts, pc)) => {
                let user = key_of(fields[1], ctx.ids);
                let person = ctx.people.get(user).unwrap_or(user);
                let index = evs.len() as u32;
                evs.push(Ev {
                    person,
                    // In [0, window_end - base) by the window check + window guard.
                    ts_rel: (event_ts - ctx.base) as u32,
                    pc,
                });
                ev_tag.push(parts.person(person) as u16);
                index
            }
            None => NONE,
        };
        toks.push(Tok {
            ekey,
            ingest_s: (ingest_ms / 1000) as u32,
            rec,
        });
        tok_tag.push(parts.event(ekey) as u16);
    });

    if evs.len() as u64 > 1u64 << ctx.k {
        ctx.fallback
            .trip(format!("event chunk {chunk_id} has {} records > 2^{}", evs.len(), ctx.k));
    }

    let groups = parts.count();
    let (evs, ev_off, pos) = counting_scatter(&evs, &ev_tag, groups);
    // Point each token at its Ev's final slot in this chunk's buffer.
    let chunk_tag = (chunk_id as u32).wrapping_shl(ctx.k);
    for tok in &mut toks {
        if tok.rec != NONE {
            tok.rec = chunk_tag | pos[tok.rec as usize];
        }
    }
    let (toks, tok_off, _) = counting_scatter(&toks, &tok_tag, groups);
    let dead = (0..evs.len().div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
    EvChunk {
        evs,
        ev_off,
        toks,
        tok_off,
        dead,
    }
}

fn parse_click_chunk(ctx: &Ctx<'_>, chunk: &Chunk<'_>) -> CkChunk {
    let parts = ctx.parts;
    let click_start = ctx.base;
    let mut cks = Vec::new();
    let mut tags = Vec::new();
    let mut has_fallback_click = false;
    let mut fields = Vec::with_capacity(4);
    for_each_line(chunk, |line| {
        if !split_fields(line, 4, &mut fields) {
            return;
        }
        let (click_id, user, campaign_id, ts_field) = (fields[0], fields[1], fields[2], fields[3]);
        // rules Clicks 1 and 3.
        if click_id.is_empty() || user.is_empty() {
            return;
        }
        let Some(click_ts) = ts::parse_ts(ts_field) else {
            return;
        };
        if click_ts < click_start || click_ts >= ctx.window_end {
            return;
        }
        let Some(campaign) = ctx.campaigns.get(campaign_id) else {
            return;
        };
        if campaign >= ctx.real_campaign_count {
            return; // `organic` is interned after the real campaigns
        }
        let user = key_of(user, ctx.ids);
        let person = ctx.people.get(user).unwrap_or(user);
        let click = key_of(click_id, ctx.ids);
        // A fallback key has length nibble 0 (key.rs).
        has_fallback_click |= click & 0x0f == 0;
        cks.push(Ck {
            person,
            click,
            ts_rel: (click_ts - ctx.base) as u32,
            camp: campaign as u16,
        });
        tags.push(parts.person(person) as u16);
    });
    let (cks, ck_off, _) = counting_scatter(&cks, &tags, parts.count());
    CkChunk {
        cks,
        ck_off,
        has_fallback_click,
    }
}

// ---------------------------------------------------------------------------
// Dedup (stage D)
// ---------------------------------------------------------------------------

/// Dead marks, addressed by `Tok::rec`.
struct DeadView<'a> {
    chunks: &'a [EvChunk],
    k: u32,
}

impl DeadView<'_> {
    #[inline]
    fn mark(&self, rec: u32) {
        if rec == NONE {
            return;
        }
        let chunk = (rec as u64 >> self.k) as usize;
        let index = rec as u64 & ((1u64 << self.k) - 1);
        self.chunks[chunk].dead[(index / 64) as usize].fetch_or(1 << (index % 64), Ordering::Relaxed);
    }
}

#[inline(always)]
fn is_dead(dead: &[AtomicU64], index: usize) -> bool {
    dead[index / 64].load(Ordering::Relaxed) & (1 << (index % 64)) != 0
}

/// Dedup one event partition. `keys` holds `(ekey, ingest_s, rec)` packed in a
/// u128 so that one integer sort orders by event, then ingest, then rec. Within
/// an event the first entry is the winner (smallest ingest; on an ingest tie a
/// valid copy sorts before `NONE`, as in v2 - equal-ingest copies are identical
/// per the rules, so this tie order never decides anything).
/// Winner `NONE` => kill every copy; otherwise kill every other copy.
fn dedup_keys(keys: &mut [u128], dead: &DeadView<'_>) {
    keys.sort_unstable();
    let mut start = 0;
    while start < keys.len() {
        let event = keys[start] >> 64;
        let mut end = start + 1;
        while end < keys.len() && keys[end] >> 64 == event {
            end += 1;
        }
        let winner_invalid = keys[start] as u32 == NONE;
        let first_loser = if winner_invalid { start } else { start + 1 };
        for &key in &keys[first_loser..end] {
            dead.mark(key as u32);
        }
        start = end;
    }
}

#[inline(always)]
fn tok_key(tok: &Tok) -> u128 {
    (u128::from(tok.ekey) << 64) | (u128::from(tok.ingest_s) << 32) | u128::from(tok.rec)
}

fn dedup_partition(chunks: &[EvChunk], partition: usize, k: u32) {
    let mut keys = Vec::new();
    for chunk in chunks {
        let slice = &chunk.toks[chunk.tok_off[partition] as usize..chunk.tok_off[partition + 1] as usize];
        keys.extend(slice.iter().map(tok_key));
    }
    dedup_keys(&mut keys, &DeadView { chunks, k });
}

// ---------------------------------------------------------------------------
// Person stage (stage S) — semantics copied from pipeline2::process_partition
// ---------------------------------------------------------------------------

#[derive(Default, Clone, Copy)]
struct CampaignDay {
    sessions: i64,
    purchasing_sessions: i64,
    revenue: i64,
    assisted_sessions: i64,
}

impl CampaignDay {
    fn add(&mut self, other: &Self) {
        self.sessions += other.sessions;
        self.purchasing_sessions += other.purchasing_sessions;
        self.revenue += other.revenue;
        self.assisted_sessions += other.assisted_sessions;
    }

    fn is_row(self) -> bool {
        self.sessions != 0 || self.assisted_sessions != 0
    }
}

/// Per-(campaign, day) accumulator: a dense grid over the window's days when
/// small enough, else a map. (Copied from pipeline2.)
enum CampaignDays {
    Dense {
        cells: Vec<CampaignDay>,
        min_day: i32,
        days: usize,
    },
    Map(FastMap<(u32, i32), CampaignDay>),
}

impl CampaignDays {
    fn new(campaigns: usize, window_start: i64, window_end: i64) -> Self {
        if window_end <= window_start {
            return Self::Map(FastMap::default());
        }
        let min_day = window_start.div_euclid(MS_PER_DAY);
        let max_day = (window_end - 1).div_euclid(MS_PER_DAY);
        let days = (max_day - min_day + 1) as usize;
        match days.checked_mul(campaigns) {
            Some(cells) if cells <= MAX_DENSE_CELLS && i32::try_from(min_day).is_ok() => Self::Dense {
                cells: vec![CampaignDay::default(); cells],
                min_day: min_day as i32,
                days,
            },
            _ => Self::Map(FastMap::default()),
        }
    }

    #[inline]
    fn at(&mut self, campaign: u32, day: i32) -> &mut CampaignDay {
        match self {
            Self::Dense { cells, min_day, days } => {
                let day_index = day.wrapping_sub(*min_day) as u32 as usize;
                assert!(day_index < *days, "v3 session day outside manifest window");
                &mut cells[campaign as usize * *days + day_index]
            }
            Self::Map(map) => map.entry((campaign, day)).or_default(),
        }
    }

    fn merge(mut self, other: Self) -> Self {
        match (&mut self, other) {
            (Self::Dense { cells: left, .. }, Self::Dense { cells: right, .. }) => {
                for (left, right) in left.iter_mut().zip(&right) {
                    left.add(right);
                }
            }
            (Self::Map(left), Self::Map(right)) => {
                for (key, value) in right {
                    left.entry(key).or_default().add(&value);
                }
            }
            _ => unreachable!("campaign accumulator shapes differ"),
        }
        self
    }

    fn for_each_row(&self, mut f: impl FnMut(u32, i32, &CampaignDay)) {
        match self {
            Self::Dense { cells, min_day, days } => {
                for (index, cell) in cells.iter().enumerate() {
                    if cell.is_row() {
                        f((index / days) as u32, *min_day + (index % days) as i32, cell);
                    }
                }
            }
            Self::Map(map) => {
                for (&(campaign, day), cell) in map {
                    if cell.is_row() {
                        f(campaign, day, cell);
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Session {
    start_ts: i64,
    day: i32,
    revenue: i64,
    has_purchase: bool,
}

/// Highest summed revenue; tie -> byte-wise smallest name. Category indices are
/// positions in the byte-sorted name list, so a smaller index is a byte-wise
/// smaller name.
fn top_category<'a>(revenues: &[(u8, i64)], names: &[&'a [u8]]) -> Option<Cow<'a, str>> {
    let mut top: Option<(u8, i64)> = None;
    for &(category, revenue) in revenues {
        top = match top {
            None => Some((category, revenue)),
            Some((best_category, best_revenue)) => {
                if revenue > best_revenue || (revenue == best_revenue && category < best_category) {
                    Some((category, revenue))
                } else {
                    Some((best_category, best_revenue))
                }
            }
        };
    }
    top.map(|(category, _)| String::from_utf8_lossy(names[category as usize]))
}

/// Count session starts in `[first, last]`, by start day, as assists.
fn add_assists(starts: &[i64], first: i64, last: i64, campaign: u32, days: &mut CampaignDays) {
    let mut low = starts.partition_point(|&start| start < first);
    let high = starts.partition_point(|&start| start <= last);
    while low < high {
        let day = starts[low].div_euclid(MS_PER_DAY);
        let next_day = (day + 1) * MS_PER_DAY;
        let next = low + starts[low..high].partition_point(|&start| start < next_day);
        days.at(campaign, day as i32).assisted_sessions += (next - low) as i64;
        low = next;
    }
}

/// Sort clicks by `(person, ts, click_id bytes)`.
///
/// Fast path: one integer sort with the raw click key as the last tiebreak.
/// Raw u64 order equals byte order only when both keys are packed (key.rs);
/// a fallback key (low nibble 0) breaks that. So afterwards, every run of
/// equal `(person, ts)` that holds a fallback key is re-sorted with the exact
/// byte comparison `key_cmp`. Generator IDs are all packed, so the repair
/// almost never runs.
///
/// `any_fallback` is false when no click chunk held a fallback key (tracked at
/// parse); then raw order is byte order everywhere and the repair scan is
/// skipped, so generator data pays no extra pass.
fn sort_clicks(cks: &mut [Ck], ids: &IdInterner, any_fallback: bool) {
    cks.sort_unstable_by_key(|c| ((u128::from(c.person) << 32) | u128::from(c.ts_rel), c.click));
    if !any_fallback {
        return;
    }
    let mut start = 0;
    while start < cks.len() {
        let (person, ts) = (cks[start].person, cks[start].ts_rel);
        let mut end = start + 1;
        let mut has_fallback = cks[start].click & 0x0f == 0;
        while end < cks.len() && cks[end].person == person && cks[end].ts_rel == ts {
            has_fallback |= cks[end].click & 0x0f == 0;
            end += 1;
        }
        if has_fallback && end - start > 1 {
            cks[start..end].sort_unstable_by(|a, b| key_cmp(a.click, b.click, ids));
        }
        start = end;
    }
}

/// Reusable per-worker buffers for the person stage.
struct Scratch {
    evs: Vec<Ev>,
    cks: Vec<Ck>,
    sessions: Vec<Session>,
    starts: Vec<i64>,
    category_revenue: Vec<(u8, i64)>,
    /// Per campaign: the open merged assist interval `[first, last]`.
    open: Vec<(i64, i64)>,
    touched: Vec<u32>,
    name_bytes: Vec<u8>,
    /// Summed seconds in gather / sort / walk (reported under FAST_TIMING).
    phase: [f64; 3],
}

impl Scratch {
    fn new(real_campaign_count: u32) -> Self {
        Self {
            evs: Vec::new(),
            cks: Vec::new(),
            sessions: Vec::new(),
            starts: Vec::new(),
            category_revenue: Vec::new(),
            open: vec![(i64::MIN, i64::MIN); real_campaign_count as usize],
            touched: Vec::new(),
            name_bytes: Vec::new(),
            phase: [0.0; 3],
        }
    }
}

struct PersonCtx<'a> {
    base: i64,
    ids: &'a IdInterner,
    cat_names: &'a [&'a [u8]],
    organic: u32,
    /// Any click key anywhere is a fallback key (see `sort_clicks`).
    any_fallback_click: bool,
}

/// Gather one person partition's live events and clicks, sort them, and run
/// pipeline2's per-person walk. Appends user rows to `rows`.
fn process_partition<'a>(
    partition: usize,
    ev_chunks: &[EvChunk],
    ck_chunks: &[CkChunk],
    pc: &PersonCtx<'a>,
    scratch: &mut Scratch,
    campaign_days: &mut CampaignDays,
    rows: &mut Vec<UserDailyRow<'a>>,
) {
    let base = pc.base;
    let ids = pc.ids;
    let t_start = Instant::now();
    scratch.evs.clear();
    for chunk in ev_chunks {
        let (lo, hi) = (chunk.ev_off[partition] as usize, chunk.ev_off[partition + 1] as usize);
        for index in lo..hi {
            if !is_dead(&chunk.dead, index) {
                scratch.evs.push(chunk.evs[index]);
            }
        }
    }
    scratch.cks.clear();
    for chunk in ck_chunks {
        let (lo, hi) = (chunk.ck_off[partition] as usize, chunk.ck_off[partition + 1] as usize);
        scratch.cks.extend_from_slice(&chunk.cks[lo..hi]);
    }
    let t_gathered = Instant::now();
    // Events by (person, ts). Raw u64 person order is only used to group (any
    // consistent order works); order among equal-ts events cannot affect any
    // output (they share a session, and all aggregates are sums).
    scratch
        .evs
        .sort_unstable_by_key(|e| (u128::from(e.person) << 32) | u128::from(e.ts_rel));
    // Clicks by (person, ts, click_id bytes): within an equal-ts run the first
    // click has the byte-wise smallest click_id (rules Clicks 4 tie-break).
    sort_clicks(&mut scratch.cks, ids, pc.any_fallback_click);
    let t_sorted = Instant::now();
    // Phase CPU-time totals for FAST_TIMING: gather, sort, walk.
    scratch.phase[0] += (t_gathered - t_start).as_secs_f64();
    scratch.phase[1] += (t_sorted - t_gathered).as_secs_f64();

    let Scratch {
        evs,
        cks,
        sessions,
        starts,
        category_revenue,
        open,
        touched,
        name_bytes,
        phase,
    } = scratch;
    let phase_walk = &mut phase[2];

    let mut ev_index = 0;
    let mut ck_index = 0;
    while ev_index < evs.len() {
        let person = evs[ev_index].person;
        let mut ev_end = ev_index + 1;
        while ev_end < evs.len() && evs[ev_end].person == person {
            ev_end += 1;
        }
        let events = &evs[ev_index..ev_end];
        ev_index = ev_end;
        // Clicks of persons without events are skipped: they can credit or
        // assist nothing.
        while ck_index < cks.len() && cks[ck_index].person < person {
            ck_index += 1;
        }
        let ck_start = ck_index;
        while ck_index < cks.len() && cks[ck_index].person == person {
            ck_index += 1;
        }
        let person_clicks = &cks[ck_start..ck_index];

        let person_name = String::from_utf8_lossy(key_bytes(person, ids, name_bytes)).into_owned();

        // --- Sessionize (rules Events 5-6), as pipeline2 ---
        sessions.clear();
        category_revenue.clear();
        let mut current_day = i32::MIN;
        let mut day_sessions = 0i64;
        let mut day_events = 0i64;
        let mut day_revenue = 0i64;
        let event_end = events.len();
        let ts_of = |e: &Ev| base + i64::from(e.ts_rel);

        let mut index = 0;
        while index < event_end {
            let start_ts = ts_of(&events[index]);
            let day = start_ts.div_euclid(MS_PER_DAY) as i32;
            if day != current_day {
                if day_sessions != 0 {
                    rows.push(UserDailyRow {
                        user_id: Cow::Owned(person_name.clone()),
                        day: current_day,
                        sessions: day_sessions,
                        events: day_events,
                        revenue_cents: day_revenue,
                        top_category: top_category(category_revenue, pc.cat_names),
                    });
                }
                current_day = day;
                day_sessions = 0;
                day_events = 0;
                day_revenue = 0;
                category_revenue.clear();
            }

            let mut previous_ts = start_ts;
            let mut session_events = 0i64;
            let mut session_revenue = 0i64;
            let mut has_purchase = false;
            while index < event_end && ts_of(&events[index]) - previous_ts <= SESSION_GAP_MS {
                let event = events[index];
                session_events += 1;
                if event.price() > 0 {
                    let price = i64::from(event.price());
                    session_revenue += price;
                    has_purchase = true;
                    let category = event.category();
                    match category_revenue.iter_mut().find(|(c, _)| *c == category) {
                        Some(entry) => entry.1 += price,
                        None => category_revenue.push((category, price)),
                    }
                }
                previous_ts = ts_of(&event);
                index += 1;
            }

            day_sessions += 1;
            day_events += session_events;
            day_revenue += session_revenue;
            sessions.push(Session {
                start_ts,
                day,
                revenue: session_revenue,
                has_purchase,
            });
        }
        if day_sessions != 0 {
            rows.push(UserDailyRow {
                user_id: Cow::Owned(person_name),
                day: current_day,
                sessions: day_sessions,
                events: day_events,
                revenue_cents: day_revenue,
                top_category: top_category(category_revenue, pc.cat_names),
            });
        }

        // --- Credit (rules Clicks 4-5): latest click with ts <= start and
        // within 7 days; an equal-ts run is byte-sorted, so its first click
        // has the smallest ID. ---
        let click_ts = |c: &Ck| base + i64::from(c.ts_rel);
        let mut click_index = 0;
        let mut latest_run_start = 0;
        for session in sessions.iter() {
            while click_index < person_clicks.len() && click_ts(&person_clicks[click_index]) <= session.start_ts {
                if click_index == 0 || person_clicks[click_index].ts_rel != person_clicks[click_index - 1].ts_rel {
                    latest_run_start = click_index;
                }
                click_index += 1;
            }
            let credited = if click_index != 0
                && session.start_ts - click_ts(&person_clicks[click_index - 1]) <= SEVEN_DAYS_MS
            {
                u32::from(person_clicks[latest_run_start].camp)
            } else {
                pc.organic
            };
            let cell = campaign_days.at(credited, session.day);
            cell.sessions += 1;
            cell.revenue += session.revenue;
            if session.has_purchase {
                cell.purchasing_sessions += 1;
            }
        }

        // --- Assists (rules Clicks 7): merge each campaign's
        // [click.ts, click.ts + 7d] intervals, then count session starts in the
        // merged intervals by start day. ---
        if !person_clicks.is_empty() {
            starts.clear();
            starts.extend(sessions.iter().map(|session| session.start_ts));
            for click in person_clicks {
                let campaign = u32::from(click.camp);
                let ts = click_ts(click);
                let slot = &mut open[campaign as usize];
                if slot.0 == i64::MIN {
                    *slot = (ts, ts + SEVEN_DAYS_MS);
                    touched.push(campaign);
                } else if ts <= slot.1 {
                    slot.1 = ts + SEVEN_DAYS_MS;
                } else {
                    add_assists(starts, slot.0, slot.1, campaign, campaign_days);
                    *slot = (ts, ts + SEVEN_DAYS_MS);
                }
            }
            for &campaign in touched.iter() {
                let (first, last) = open[campaign as usize];
                add_assists(starts, first, last, campaign, campaign_days);
                open[campaign as usize] = (i64::MIN, i64::MIN);
            }
            touched.clear();
        }
    }
    *phase_walk += t_sorted.elapsed().as_secs_f64();
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

pub fn run(data: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let fallback = Fallback::default();
    run_v3(data, output, &fallback)?;
    if fallback.is_set() {
        // Nothing has been written to `output` yet (run_v3 returns before
        // writing whenever the flag is set), so v1 starts from a clean slate.
        eprintln!("v3: fallback to v1 because {}", fallback.reason());
        return crate::pipeline::run(data, output);
    }
    Ok(())
}

/// Runs v3. Returns early, *without writing anything*, as soon as `fallback`
/// is set at a check point; the caller then runs v1.
fn run_v3(data: &Path, output: &Path, fallback: &Fallback) -> Result<(), Box<dyn Error>> {
    let mut stages = Stages::new();
    let manifest = manifest::read(data)?;
    let window_start = ts::parse_ts(manifest.window_start.as_bytes()).ok_or("manifest: bad window_start")?;
    let window_end = ts::parse_ts(manifest.window_end.as_bytes()).ok_or("manifest: bad window_end")?;
    if let Some(reason) = window_guard(window_start, window_end) {
        fallback.trip(reason);
        return Ok(());
    }
    let base = window_start - SEVEN_DAYS_MS;
    ts::set_window_hint(window_start, window_end);

    let link_maps = map_all(&csv_files(&data.join("links")))?;
    let event_maps = map_all(&csv_files(&data.join("events")))?;
    let click_maps = map_all(&csv_files(&data.join("clicks")))?;
    let products_bytes = std::fs::read(data.join("products.csv"))?;
    let campaigns_bytes = std::fs::read(data.join("campaigns.csv"))?;

    // Categories: byte-sorted unique names incl. `unknown`, so u8 index order
    // is byte order (top_category tie-break). If the file itself uses the name
    // `unknown` it is the same category, as in v1/v2's interner.
    let mut fields = Vec::with_capacity(8);
    let mut product_rows: Vec<(&[u8], &[u8])> = Vec::new();
    for line in data_lines(&products_bytes) {
        if split_fields(line, 2, &mut fields) {
            product_rows.push((fields[0], fields[1]));
        }
    }
    let mut cat_names: Vec<&[u8]> = product_rows.iter().map(|&(_, c)| c).collect();
    cat_names.push(b"unknown");
    cat_names.sort_unstable();
    cat_names.dedup();
    if cat_names.len() > 256 {
        fallback.trip(format!("{} categories do not fit u8", cat_names.len()));
        return Ok(());
    }
    let cat_index = |name: &[u8]| cat_names.binary_search(&name).expect("category listed") as u8;
    let unknown_cat = cat_index(b"unknown");
    let mut product_cat = FastMap::default();
    for &(product, category) in &product_rows {
        // Later duplicate rows overwrite earlier ones, as in v2.
        product_cat.insert(product, cat_index(category));
    }

    let mut campaigns = Interner::new();
    let mut channels = Interner::new();
    let mut campaign_channel = Vec::<u32>::new();
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
    let real_campaign_count = campaigns.len() as u32;
    if real_campaign_count > 65_535 {
        fallback.trip(format!("{real_campaign_count} campaigns do not fit u16"));
        return Ok(());
    }
    let organic = campaigns.intern(b"organic");
    let organic_channel = channels.intern(b"organic");
    if organic as usize >= campaign_channel.len() {
        campaign_channel.resize(organic as usize + 1, 0);
    }
    campaign_channel[organic as usize] = organic_channel;
    stages.mark("lookups + mmap");

    // --- L: links ---
    let ids = IdInterner::new();
    let link_files: Vec<&[u8]> = link_maps.iter().map(|map| &map[..]).collect();
    let people = links2::build(&link_files, &ids);
    stages.mark("L links");

    // --- P: one parallel pass over all event and click chunks ---
    let event_files: Vec<&[u8]> = event_maps.iter().map(|map| &map[..]).collect();
    let click_files: Vec<&[u8]> = click_maps.iter().map(|map| &map[..]).collect();
    let event_chunks = chunk_files(&event_files);
    let click_chunks = chunk_files(&click_files);
    let event_bytes: u64 = event_files.iter().map(|f| f.len() as u64).sum();
    let parts = Parts::for_input(event_bytes);
    let k = rec_index_bits(event_chunks.len());

    let ctx = Ctx {
        window_start,
        window_end,
        base,
        product_cat,
        unknown_cat,
        campaigns,
        real_campaign_count,
        people: &people,
        ids: &ids,
        parts,
        k,
        fallback,
    };

    // Tasks: (is_event, index), biggest chunks first so the tail is short.
    let mut tasks: Vec<(bool, usize, usize)> = event_chunks
        .iter()
        .enumerate()
        .map(|(i, c)| (true, i, c.bytes.len()))
        .chain(click_chunks.iter().enumerate().map(|(i, c)| (false, i, c.bytes.len())))
        .collect();
    tasks.sort_unstable_by_key(|&(_, _, len)| std::cmp::Reverse(len));
    enum Parsed {
        E(usize, EvChunk),
        C(usize, CkChunk),
    }
    let parsed: Vec<Parsed> = tasks
        .par_iter()
        .with_max_len(1)
        .map(|&(is_event, i, _)| {
            if is_event {
                Parsed::E(i, parse_event_chunk(&ctx, &event_chunks[i], i))
            } else {
                Parsed::C(i, parse_click_chunk(&ctx, &click_chunks[i]))
            }
        })
        .collect();
    let mut ev_slots: Vec<Option<EvChunk>> = (0..event_chunks.len()).map(|_| None).collect();
    let mut ck_slots: Vec<Option<CkChunk>> = (0..click_chunks.len()).map(|_| None).collect();
    for p in parsed {
        match p {
            Parsed::E(i, c) => ev_slots[i] = Some(c),
            Parsed::C(i, c) => ck_slots[i] = Some(c),
        }
    }
    let ev_chunks: Vec<EvChunk> = ev_slots.into_iter().map(|c| c.expect("parsed")).collect();
    let ck_chunks: Vec<CkChunk> = ck_slots.into_iter().map(|c| c.expect("parsed")).collect();
    stages.mark("P parse (events+clicks)");
    if fallback.is_set() {
        return Ok(());
    }

    // --- D: dedup per event partition ---
    (0..parts.count())
        .into_par_iter()
        .for_each(|partition| dedup_partition(&ev_chunks, partition, k));
    stages.mark("D dedup");

    // --- S: person stage, largest partitions first, dynamic queue ---
    let mut order: Vec<(usize, usize)> = (0..parts.count())
        .map(|p| {
            let evs: usize = ev_chunks.iter().map(|c| (c.ev_off[p + 1] - c.ev_off[p]) as usize).sum();
            let cks: usize = ck_chunks.iter().map(|c| (c.ck_off[p + 1] - c.ck_off[p]) as usize).sum();
            (p, evs + cks)
        })
        .collect();
    order.sort_unstable_by_key(|&(_, size)| std::cmp::Reverse(size));
    let person_ctx = PersonCtx {
        base,
        ids: &ids,
        cat_names: &cat_names,
        organic,
        any_fallback_click: ck_chunks.iter().any(|c| c.has_fallback_click),
    };
    let campaign_count = ctx.campaigns.len();
    let next = AtomicUsize::new(0);
    let workers = rayon::current_num_threads().max(1);
    // Each worker drains the shared queue; it owns its rows and campaign grid.
    type WorkerOut<'r> = (Vec<UserDailyRow<'r>>, CampaignDays, Vec<f64>, [f64; 3]);
    let worker_out: Vec<WorkerOut> = (0..workers)
        .into_par_iter()
        .with_max_len(1)
        .map(|_| {
            let mut rows = Vec::new();
            let mut days = CampaignDays::new(campaign_count, window_start, window_end);
            let mut scratch = Scratch::new(real_campaign_count);
            let mut times = Vec::new();
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= order.len() {
                    break;
                }
                let started = Instant::now();
                process_partition(order[i].0, &ev_chunks, &ck_chunks, &person_ctx, &mut scratch, &mut days, &mut rows);
                times.push(started.elapsed().as_secs_f64());
            }
            (rows, days, times, scratch.phase)
        })
        .collect();
    let mut user_parts = Vec::with_capacity(worker_out.len());
    let mut campaign_days = CampaignDays::new(campaign_count, window_start, window_end);
    let mut part_times = Vec::new();
    let mut phase = [0.0f64; 3];
    for (rows, days, times, worker_phase) in worker_out {
        user_parts.push(rows);
        campaign_days = campaign_days.merge(days);
        part_times.extend(times);
        for (total, add) in phase.iter_mut().zip(worker_phase) {
            *total += add;
        }
    }
    stages.mark("S person stage");
    if stages.enabled && !part_times.is_empty() {
        part_times.sort_unstable_by(f64::total_cmp);
        eprintln!(
            "  v3   person partitions: {} max {:.4}s median {:.4}s",
            part_times.len(),
            part_times[part_times.len() - 1],
            part_times[part_times.len() / 2]
        );
        // Summed over workers, divided by worker count: per-thread share.
        let w = workers as f64;
        eprintln!(
            "  v3   S phases per thread: gather {:.4}s sort {:.4}s walk {:.4}s",
            phase[0] / w,
            phase[1] / w,
            phase[2] / w
        );
    }

    // --- O: output ---
    let mut campaign_rows = Vec::<CampaignDailyRow>::new();
    campaign_days.for_each_row(|campaign, day, accumulated| {
        let channel = campaign_channel[campaign as usize];
        campaign_rows.push(CampaignDailyRow {
            campaign_id: String::from_utf8_lossy(ctx.campaigns.bytes(campaign)),
            channel: String::from_utf8_lossy(channels.bytes(channel)),
            day,
            sessions: accumulated.sessions,
            purchasing_sessions: accumulated.purchasing_sessions,
            revenue_cents: accumulated.revenue,
            assisted_sessions: accumulated.assisted_sessions,
        });
    });
    let user_sessions: i64 = user_parts.iter().flatten().map(|row| row.sessions).sum();
    let campaign_sessions: i64 = campaign_rows.iter().map(|row| row.sessions).sum();
    let user_revenue: i64 = user_parts.iter().flatten().map(|row| row.revenue_cents).sum();
    let campaign_revenue: i64 = campaign_rows.iter().map(|row| row.revenue_cents).sum();
    if user_sessions != campaign_sessions || user_revenue != campaign_revenue {
        eprintln!(
            "V3 INVARIANT VIOLATED: sessions {user_sessions} vs {campaign_sessions}, \
             revenue {user_revenue} vs {campaign_revenue}"
        );
    }
    // Final check point: nothing has been written yet.
    if fallback.is_set() {
        return Ok(());
    }

    let user_dir = output.join("user_daily");
    std::fs::create_dir_all(&user_dir)?;
    // Split the workers' row vectors into ~equal slices, at least 16 and at
    // least 2 per thread, so the Parquet encoding runs in parallel without
    // one big part setting the wall time. Part 0 is always written (possibly
    // empty) so the folder always holds a Parquet file.
    let total_rows: usize = user_parts.iter().map(Vec::len).sum();
    let target_parts = (2 * workers).max(16);
    let per_part = total_rows.div_ceil(target_parts).max(1);
    let mut slices: Vec<&[UserDailyRow]> = user_parts
        .iter()
        .flat_map(|rows| rows.chunks(per_part))
        .collect();
    if slices.is_empty() {
        slices.push(&[]);
    }
    let campaign_dir = output.join("campaign_daily");
    let (user_writes, campaign_write) = rayon::join(
        || {
            slices
                .par_iter()
                .enumerate()
                .map(|(part, rows)| {
                    out::write_user_daily(&user_dir, part, rows.iter()).map_err(|error| error.to_string())
                })
                .collect::<Vec<Result<(), String>>>()
        },
        || out::write_campaign_daily(&campaign_dir, &campaign_rows).map_err(|error| error.to_string()),
    );
    for write in user_writes {
        write?;
    }
    campaign_write?;
    stages.mark("O output");

    // main exits right after success; skip destructors of the big buffers.
    std::mem::forget(user_parts);
    std::mem::forget(campaign_rows);
    std::mem::forget(campaign_days);
    std::mem::forget(ev_chunks);
    std::mem::forget(ck_chunks);
    std::mem::forget(ctx);
    std::mem::forget(people);
    std::mem::forget(ids);
    std::mem::forget(event_maps);
    std::mem::forget(click_maps);
    std::mem::forget(link_maps);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An EvChunk with `n` Evs and nothing else, for dead-bit tests.
    fn chunk_with(n: usize) -> EvChunk {
        EvChunk {
            evs: vec![Ev::default(); n],
            ev_off: vec![0, n as u32],
            toks: Vec::new(),
            tok_off: vec![0, 0],
            dead: (0..n.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    const K: u32 = 20;
    fn rec(chunk: u32, index: u32) -> u32 {
        (chunk << K) | index
    }
    fn tok(ekey: IdKey, ingest_s: u32, rec: u32) -> u128 {
        tok_key(&Tok { ekey, ingest_s, rec })
    }
    fn dead(chunks: &[EvChunk], chunk: usize, index: usize) -> bool {
        is_dead(&chunks[chunk].dead, index)
    }

    #[test]
    fn invalid_earliest_copy_kills_later_valid_copy() {
        // E5: the earliest copy is invalid (NONE); the later valid copy must die.
        let chunks = vec![chunk_with(3), chunk_with(2)];
        let mut keys = vec![tok(17, 200, rec(1, 1)), tok(17, 100, NONE), tok(18, 50, rec(0, 2))];
        dedup_keys(&mut keys, &DeadView { chunks: &chunks, k: K });
        assert!(dead(&chunks, 1, 1));
        assert!(!dead(&chunks, 0, 2)); // a different event, alone: lives
        assert!(!dead(&chunks, 1, 0));
    }

    #[test]
    fn later_invalid_copy_does_not_kill_earliest_valid_copy() {
        let chunks = vec![chunk_with(1)];
        let mut keys = vec![tok(5, 300, NONE), tok(5, 100, rec(0, 0))];
        dedup_keys(&mut keys, &DeadView { chunks: &chunks, k: K });
        assert!(!dead(&chunks, 0, 0));
    }

    #[test]
    fn equal_ingest_copies_keep_exactly_one() {
        // Equal ingest = identical copies: exactly one survives.
        let chunks = vec![chunk_with(70), chunk_with(4)];
        let mut keys = vec![tok(9, 100, rec(1, 3)), tok(9, 100, rec(0, 69)), tok(9, 100, rec(1, 0))];
        dedup_keys(&mut keys, &DeadView { chunks: &chunks, k: K });
        let alive = [(1, 3), (0, 69), (1, 0)].iter().filter(|&&(c, i)| !dead(&chunks, c, i)).count();
        assert_eq!(alive, 1);
    }

    #[test]
    fn different_user_copies_only_winner_survives() {
        // E4: copies of one event name different users, so their Evs sit under
        // different persons (here: different chunks/slots). Only the
        // earliest-ingest copy's Ev survives; a later copy never adds an event
        // to its own user.
        let chunks = vec![chunk_with(2), chunk_with(2)];
        let mut keys = vec![
            tok(42, 1_000, rec(0, 1)), // user A, earliest
            tok(42, 1_060, rec(1, 0)), // user B
            tok(42, 1_600, rec(1, 1)), // user C
        ];
        dedup_keys(&mut keys, &DeadView { chunks: &chunks, k: K });
        assert!(!dead(&chunks, 0, 1));
        assert!(dead(&chunks, 1, 0));
        assert!(dead(&chunks, 1, 1));
        assert!(!dead(&chunks, 0, 0));
    }

    #[test]
    fn rec_refs_never_collide_with_none() {
        for n in [1usize, 2, 3, 255, 256, 257, 4095, 70_000] {
            let k = rec_index_bits(n);
            let max_chunk = (n - 1) as u64;
            let max_rec = (max_chunk << k) | ((1u64 << k) - 1);
            assert!(max_rec < u64::from(NONE), "n={n} k={k}");
            // Also the top-bit layout must fit u32.
            assert!(max_rec <= u64::from(u32::MAX));
        }
    }

    #[test]
    fn window_guard_trips_beyond_u32_ms() {
        let day = MS_PER_DAY;
        let start = 1_790_000_000_000i64;
        // A 14-day window (generator) passes.
        assert!(window_guard(start, start + 14 * day).is_none());
        // 7 + 42 days = 49 days < 2^32 ms (49.7 days): passes.
        assert!(window_guard(start, start + 42 * day).is_none());
        // A 60-day window cannot be stored in u32 relative ms: falls back.
        assert!(window_guard(start, start + 60 * day).is_some());
    }

    #[test]
    fn fallback_flag_records_first_reason() {
        let fallback = Fallback::default();
        assert!(!fallback.is_set());
        fallback.trip("first");
        fallback.trip("second");
        assert!(fallback.is_set());
        assert_eq!(fallback.reason(), "first");
    }

    #[test]
    fn sort_clicks_orders_ties_by_exact_bytes_with_fallback_keys() {
        let ids = IdInterner::new();
        // Mix packed IDs (<= 10 bytes of [-0-9A-Z_a-z]) and fallback IDs
        // (too long, or a byte outside the alphabet) at equal (person, ts).
        let names: [&[u8]; 6] = [b"c_b", b"c_aaaaaaaaaaaaaa", b"c_a", b"c.z", b"C_A", b"c_a "];
        let mut cks: Vec<Ck> = Vec::new();
        for (i, name) in names.iter().enumerate() {
            for ts in [5u32, 3] {
                cks.push(Ck { person: 7, click: key_of(name, &ids), ts_rel: ts, camp: i as u16 });
            }
        }
        cks.push(Ck { person: 2, click: key_of(b"zz", &ids), ts_rel: 9, camp: 99 });
        sort_clicks(&mut cks, &ids, true);
        let mut expected: Vec<(u64, u32, Vec<u8>)> = cks
            .iter()
            .map(|c| (c.person, c.ts_rel, key_bytes(c.click, &ids, &mut Vec::new()).to_vec()))
            .collect();
        expected.sort();
        let got: Vec<(u64, u32, Vec<u8>)> = cks
            .iter()
            .map(|c| (c.person, c.ts_rel, key_bytes(c.click, &ids, &mut Vec::new()).to_vec()))
            .collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn counting_scatter_is_stable_and_positions_match() {
        let items: Vec<u32> = (0..100).collect();
        let tags: Vec<u16> = items.iter().map(|&i| (i * 7 % 5) as u16).collect();
        let (out, off, pos) = counting_scatter(&items, &tags, 5);
        assert_eq!(off[5] as usize, items.len());
        for (i, &p) in pos.iter().enumerate() {
            assert_eq!(out[p as usize], items[i]);
        }
        for g in 0..5 {
            let slice = &out[off[g] as usize..off[g + 1] as usize];
            assert!(slice.windows(2).all(|w| w[0] < w[1]));
            assert!(slice.iter().all(|&i| (i * 7 % 5) as usize == g));
        }
    }
}
