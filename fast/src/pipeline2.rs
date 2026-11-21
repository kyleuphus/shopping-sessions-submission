//! Integer-key v2 pipeline, selected with `FAST_V2=1`.
//!
//! This is intentionally separate from `pipeline.rs`: v1 remains the default
//! behavioral oracle.  The stage order is the order mandated by `rules.md`:
//! lookups and links, parse, event-id dedup, per-person sessionization and
//! attribution, then output.

use std::borrow::Cow;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

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

const TYPE_VIEW: u8 = 0;
const TYPE_CART: u8 = 1;
const TYPE_PURCHASE: u8 = 2;

const PARTITION_BITS: u32 = 8;
const PARTITIONS: usize = 1 << PARTITION_BITS;
const CHUNK_BYTES: usize = 8 << 20;
const OUTPUT_PARTS: usize = 16;
/// A valid event's record reference: `(chunk index << 32) | chunk-local valid
/// record index`.  Each parse task numbers its own valid records, so there is
/// no shared counter; the global dead-bitmap position is `bases[chunk] + local`
/// (see [`DeadMarks`]).  `INVALID_REC` marks a token whose copy failed
/// validation; it sorts after every real reference.
type RecRef = u64;
const INVALID_REC: RecRef = u64::MAX;
const MAX_DENSE_CELLS: usize = 8 << 20;

#[inline(always)]
fn partition_of(hash: u64) -> usize {
    (hash >> (64 - PARTITION_BITS)) as usize
}

#[inline(always)]
fn event_partition(key: IdKey) -> usize {
    partition_of(fmix64(key ^ 0x6576_656e_745f_7632))
}

#[inline(always)]
fn person_partition(key: IdKey) -> usize {
    partition_of(fmix64(key ^ 0x7065_7273_6f6e_7632))
}

fn empty_buckets<T>() -> Vec<Vec<T>> {
    (0..PARTITIONS).map(|_| Vec::new()).collect()
}

/// Partition buckets pre-sized from an estimate of the chunk's line count
/// (newlines in the first 64 KiB, scaled), so per-row pushes rarely realloc.
fn sized_buckets<T>(chunk: &Chunk<'_>) -> Vec<Vec<T>> {
    const SAMPLE: usize = 64 << 10;
    let sample = &chunk.bytes[..chunk.bytes.len().min(SAMPLE)];
    let lines = memchr::memchr_iter(b'\n', sample).count().max(1);
    let estimate = lines * chunk.bytes.len() / sample.len().max(1);
    let per_bucket = estimate / PARTITIONS;
    let per_bucket = per_bucket + per_bucket / 8 + 8;
    (0..PARTITIONS).map(|_| Vec::with_capacity(per_bucket)).collect()
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
            eprintln!("  v2 {label:<25} {:>8.3}s", self.last.elapsed().as_secs_f64());
        }
        self.last = std::time::Instant::now();
    }
}

struct Lookups<'a> {
    window_start: i64,
    window_end: i64,
    product_category: FastMap<&'a [u8], u32>,
    unknown_category: u32,
    campaigns: Interner<'a>,
    real_campaign_count: u32,
}

impl<'a> Lookups<'a> {
    #[inline]
    fn validate_event(&self, fields: &[&'a [u8]], ingest: i64) -> Option<(i64, u32, u32)> {
        let user = fields[1];
        let product = fields[5];
        if user.is_empty() || product.is_empty() {
            return None;
        }
        let event_type = match fields[4] {
            b"view" => TYPE_VIEW,
            b"cart" => TYPE_CART,
            b"purchase" => TYPE_PURCHASE,
            _ => return None,
        };
        let event_ts = ts::parse_ts(fields[2])?;
        if event_ts < self.window_start || event_ts >= self.window_end {
            return None;
        }
        if ingest - event_ts > LATE_CUTOFF_MS {
            return None;
        }
        if event_type != TYPE_PURCHASE {
            return Some((event_ts, u32::from(event_type) << 24, self.unknown_category));
        }
        let price = parse_digits(fields[6])?;
        if !(1..=10_000_000).contains(&price) {
            return None;
        }
        let category = self
            .product_category
            .get(product)
            .copied()
            .unwrap_or(self.unknown_category);
        Some((
            event_ts,
            price as u32 | (u32::from(event_type) << 24),
            category,
        ))
    }
}

#[derive(Clone, Copy, Debug)]
struct EventToken {
    event: IdKey,
    ingest: i64,
    rec: RecRef,
}

#[derive(Clone, Copy, Debug, Default)]
struct EventRec {
    person: IdKey,
    ts: i64,
    rec: RecRef,
    price_and_type: u32,
    category: u32,
}

impl EventRec {
    #[inline]
    fn price(self) -> u32 {
        self.price_and_type & 0x00ff_ffff
    }

    #[inline]
    fn event_type(self) -> u8 {
        (self.price_and_type >> 24) as u8
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ClickRec {
    person: IdKey,
    ts: i64,
    campaign: u32,
    click: IdKey,
}

const _: () = assert!(std::mem::size_of::<EventToken>() <= 24);
const _: () = assert!(std::mem::size_of::<EventRec>() <= 32);
const _: () = assert!(std::mem::size_of::<ClickRec>() <= 32);

struct ParsedEvents {
    tokens: Vec<Vec<EventToken>>,
    records: Vec<Vec<EventRec>>,
    /// Number of valid records this chunk produced (local indices 0..valid).
    valid: u32,
}

fn parse_events(
    chunks: &[Chunk<'_>],
    lookups: &Lookups<'_>,
    people: &PersonMap,
    ids: &IdInterner,
) -> Vec<ParsedEvents> {
    chunks
        .par_iter()
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let chunk_tag = (chunk_index as u64) << 32;
            let mut valid = 0u32;
            let mut tokens = sized_buckets(chunk);
            let mut records = sized_buckets(chunk);
            let mut fields = Vec::with_capacity(8);
            for_each_line(chunk, |line| {
                if !split_fields(line, 7, &mut fields) || fields[0].is_empty() {
                    return;
                }
                // v1 parses ingest_ts with the general parser.  Although the
                // spec guarantees F1, a bad value is dropped before token emit.
                let Some(ingest) = ts::parse_ts(fields[3]) else {
                    return;
                };
                let event = key_of(fields[0], ids);
                let rec = if let Some((event_ts, price_and_type, category)) =
                    lookups.validate_event(&fields, ingest)
                {
                    let user = key_of(fields[1], ids);
                    let person = people.get(user).unwrap_or(user);
                    let rec = chunk_tag | u64::from(valid);
                    valid = valid.checked_add(1).expect("v2 chunk record index overflow");
                    records[person_partition(person)].push(EventRec {
                        person,
                        ts: event_ts,
                        rec,
                        price_and_type,
                        category,
                    });
                    rec
                } else {
                    INVALID_REC
                };
                tokens[event_partition(event)].push(EventToken { event, ingest, rec });
            });
            ParsedEvents {
                tokens,
                records,
                valid,
            }
        })
        .collect()
}

fn parse_clicks(
    chunks: &[Chunk<'_>],
    lookups: &Lookups<'_>,
    people: &PersonMap,
    ids: &IdInterner,
) -> Vec<Vec<Vec<ClickRec>>> {
    let click_start = lookups.window_start - SEVEN_DAYS_MS;
    chunks
        .par_iter()
        .map(|chunk| {
            let mut records = sized_buckets(chunk);
            let mut fields = Vec::with_capacity(4);
            for_each_line(chunk, |line| {
                if !split_fields(line, 4, &mut fields) {
                    return;
                }
                let (click_id, user, campaign_id, ts_field) =
                    (fields[0], fields[1], fields[2], fields[3]);
                if click_id.is_empty() || user.is_empty() {
                    return;
                }
                let Some(click_ts) = ts::parse_ts(ts_field) else {
                    return;
                };
                if click_ts < click_start || click_ts >= lookups.window_end {
                    return;
                }
                let Some(campaign) = lookups.campaigns.get(campaign_id) else {
                    return;
                };
                if campaign >= lookups.real_campaign_count {
                    return;
                }
                let user = key_of(user, ids);
                let person = people.get(user).unwrap_or(user);
                let click = key_of(click_id, ids);
                records[person_partition(person)].push(ClickRec {
                    person,
                    ts: click_ts,
                    campaign,
                    click,
                });
            });
            records
        })
        .collect()
}

/// Dead marks for valid event records.  One global bitmap; chunk `c`'s records
/// occupy bits `bases[c] .. bases[c] + valid[c]` (prefix sum of the per-chunk
/// valid counts, computed after parsing and before dedup).
struct DeadMarks {
    bits: Vec<AtomicU64>,
    bases: Vec<u64>,
}

impl DeadMarks {
    fn new(valid_per_chunk: impl IntoIterator<Item = u32>) -> Self {
        let mut bases = Vec::new();
        let mut total = 0u64;
        for valid in valid_per_chunk {
            bases.push(total);
            total += u64::from(valid);
        }
        let bits = (0..total.div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
        Self { bits, bases }
    }

    #[inline(always)]
    fn bit(&self, rec: RecRef) -> u64 {
        self.bases[(rec >> 32) as usize] + (rec & 0xffff_ffff)
    }

    #[inline]
    fn mark(&self, rec: RecRef) {
        if rec != INVALID_REC {
            let bit = self.bit(rec);
            self.bits[(bit / 64) as usize].fetch_or(1u64 << (bit % 64), Ordering::Relaxed);
        }
    }

    #[inline]
    fn is_dead(&self, rec: RecRef) -> bool {
        let bit = self.bit(rec);
        self.bits[(bit / 64) as usize].load(Ordering::Relaxed) & (1u64 << (bit % 64)) != 0
    }
}

/// Dedup one exact-key partition.  A record starts live; every valid copy other
/// than the minimum-ingest winner is marked dead.  If that winner is invalid,
/// every valid copy is marked dead, implementing dedup-before-validation.
fn dedup_tokens(tokens: &mut [EventToken], dead: &DeadMarks) {
    tokens.par_sort_unstable_by_key(|token| (token.event, token.ingest, token.rec));
    let mut start = 0;
    while start < tokens.len() {
        let event = tokens[start].event;
        let mut end = start + 1;
        while end < tokens.len() && tokens[end].event == event {
            end += 1;
        }
        let winner = tokens[start].rec;
        for (offset, token) in tokens[start..end].iter().enumerate() {
            if winner == INVALID_REC || offset != 0 {
                dead.mark(token.rec);
            }
        }
        start = end;
    }
}

fn take_token_partitions(parsed: &mut [ParsedEvents]) -> Vec<Vec<EventToken>> {
    let mut by_partition: Vec<Vec<EventToken>> = empty_buckets();
    let mut counts = vec![0usize; PARTITIONS];
    for producer in parsed.iter() {
        for (partition, bucket) in producer.tokens.iter().enumerate() {
            counts[partition] += bucket.len();
        }
    }
    for (bucket, count) in by_partition.iter_mut().zip(counts) {
        bucket.reserve(count);
    }
    for producer in parsed.iter_mut() {
        for (partition, bucket) in producer.tokens.iter_mut().enumerate() {
            by_partition[partition].append(bucket);
        }
    }
    by_partition
}

fn collect_person_partitions(
    parsed_events: Vec<ParsedEvents>,
    parsed_clicks: Vec<Vec<Vec<ClickRec>>>,
    dead: &DeadMarks,
) -> Vec<(usize, Vec<EventRec>, Vec<ClickRec>)> {
    (0..PARTITIONS)
        .into_par_iter()
        .map(|partition| {
            let event_count: usize = parsed_events
                .iter()
                .map(|producer| producer.records[partition].len())
                .sum();
            let click_count: usize = parsed_clicks
                .iter()
                .map(|producer| producer[partition].len())
                .sum();
            let mut events = Vec::with_capacity(event_count);
            for producer in &parsed_events {
                events.extend(
                    producer.records[partition]
                        .iter()
                        .copied()
                        .filter(|event| !dead.is_dead(event.rec)),
                );
            }
            let mut clicks = Vec::with_capacity(click_count);
            for producer in &parsed_clicks {
                clicks.extend_from_slice(&producer[partition]);
            }
            (partition, events, clicks)
        })
        .collect()
}

const NO_PERSON: u32 = u32::MAX;

/// Open-addressing map from a person key to a dense per-partition index
/// (assigned in first-insertion order).  Linear probing, load factor <= 1/2,
/// `NO_PERSON` marks an empty slot.  Keys within one partition share their top
/// `PARTITION_BITS` of `person_partition`'s hash, so slots use a differently
/// salted hash's low bits.
struct PersonIndex {
    slots: Vec<(IdKey, u32)>,
    mask: usize,
    persons: Vec<IdKey>,
}

impl PersonIndex {
    fn with_capacity(expected: usize) -> Self {
        let cap = (expected.max(8) * 2).next_power_of_two();
        Self {
            slots: vec![(0, NO_PERSON); cap],
            mask: cap - 1,
            persons: Vec::with_capacity(expected),
        }
    }

    #[inline(always)]
    fn slot_of(&self, key: IdKey) -> usize {
        fmix64(key ^ 0x6465_6e73_655f_7632) as usize & self.mask
    }

    #[inline]
    fn insert(&mut self, key: IdKey) -> u32 {
        let mut slot = self.slot_of(key);
        loop {
            let (k, v) = self.slots[slot];
            if v == NO_PERSON {
                if (self.persons.len() + 1) * 2 > self.slots.len() {
                    self.grow();
                    return self.insert(key);
                }
                let dense = self.persons.len() as u32;
                self.slots[slot] = (key, dense);
                self.persons.push(key);
                return dense;
            }
            if k == key {
                return v;
            }
            slot = (slot + 1) & self.mask;
        }
    }

    #[inline]
    fn get(&self, key: IdKey) -> u32 {
        let mut slot = self.slot_of(key);
        loop {
            let (k, v) = self.slots[slot];
            if v == NO_PERSON || k == key {
                return v;
            }
            slot = (slot + 1) & self.mask;
        }
    }

    fn grow(&mut self) {
        let cap = self.slots.len() * 2;
        self.slots = vec![(0, NO_PERSON); cap];
        self.mask = cap - 1;
        for (dense, &key) in self.persons.iter().enumerate() {
            let mut slot = self.slot_of(key);
            while self.slots[slot].1 != NO_PERSON {
                slot = (slot + 1) & self.mask;
            }
            self.slots[slot] = (key, dense as u32);
        }
    }
}

/// One stable counting-sort pass: scatter `items` by `dense[i]` (skipping
/// `NO_PERSON`) into contiguous per-person groups.  Returns the grouped items
/// and `offsets` with group `p` at `offsets[p]..offsets[p + 1]`.
fn counting_sort_by_dense<T: Copy + Default>(
    items: &[T],
    dense: &[u32],
    groups: usize,
) -> (Vec<T>, Vec<u32>) {
    let mut offsets = vec![0u32; groups + 1];
    for &d in dense {
        if d != NO_PERSON {
            offsets[d as usize + 1] += 1;
        }
    }
    for p in 0..groups {
        offsets[p + 1] += offsets[p];
    }
    let mut cursor = offsets.clone();
    let mut out = vec![T::default(); offsets[groups] as usize];
    for (item, &d) in items.iter().zip(dense) {
        if d != NO_PERSON {
            let at = &mut cursor[d as usize];
            out[*at as usize] = *item;
            *at += 1;
        }
    }
    (out, offsets)
}

/// Events and clicks of one person partition, grouped by person.  Only persons
/// with at least one surviving event get an index; clicks of any other person
/// can never credit or assist a session and are dropped here.  Group slices are
/// left in input order; the caller sorts each person's slice by time.
struct PersonGroups {
    persons: Vec<IdKey>,
    events: Vec<EventRec>,
    event_offsets: Vec<u32>,
    clicks: Vec<ClickRec>,
    click_offsets: Vec<u32>,
}

fn group_by_person(events: &[EventRec], clicks: &[ClickRec]) -> PersonGroups {
    let mut index = PersonIndex::with_capacity(events.len() / 8);
    let event_dense: Vec<u32> = events.iter().map(|event| index.insert(event.person)).collect();
    let groups = index.persons.len();
    let click_dense: Vec<u32> = clicks.iter().map(|click| index.get(click.person)).collect();
    let (events, event_offsets) = counting_sort_by_dense(events, &event_dense, groups);
    let (clicks, click_offsets) = counting_sort_by_dense(clicks, &click_dense, groups);
    PersonGroups {
        persons: index.persons,
        events,
        event_offsets,
        clicks,
        click_offsets,
    }
}

/// Click order the credit rule needs: by `ts`, ties by exact click_id bytes,
/// so the first click of an equal-ts run has the smallest ID.
fn sort_person_clicks(clicks: &mut [ClickRec], ids: &IdInterner) {
    clicks.sort_unstable_by(|left, right| {
        left.ts
            .cmp(&right.ts)
            .then_with(|| key_cmp(left.click, right.click, ids))
    });
}

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
            Some(cells) if cells <= MAX_DENSE_CELLS && i32::try_from(min_day).is_ok() => {
                Self::Dense {
                    cells: vec![CampaignDay::default(); cells],
                    min_day: min_day as i32,
                    days,
                }
            }
            _ => Self::Map(FastMap::default()),
        }
    }

    #[inline]
    fn at(&mut self, campaign: u32, day: i32) -> &mut CampaignDay {
        match self {
            Self::Dense {
                cells,
                min_day,
                days,
            } => {
                let day_index = day.wrapping_sub(*min_day) as u32 as usize;
                assert!(day_index < *days, "v2 session day outside manifest window");
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
            Self::Dense {
                cells,
                min_day,
                days,
            } => {
                for (index, cell) in cells.iter().enumerate() {
                    if cell.is_row() {
                        f(
                            (index / days) as u32,
                            *min_day + (index % days) as i32,
                            cell,
                        );
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

fn top_category<'a>(revenues: &[(u32, i64)], categories: &Interner<'a>) -> Option<Cow<'a, str>> {
    let mut top: Option<(u32, i64)> = None;
    for &(category, revenue) in revenues {
        top = match top {
            None => Some((category, revenue)),
            Some((best_category, best_revenue)) => {
                if revenue > best_revenue
                    || (revenue == best_revenue && categories.less(category, best_category))
                {
                    Some((category, revenue))
                } else {
                    Some((best_category, best_revenue))
                }
            }
        };
    }
    top.map(|(category, _)| String::from_utf8_lossy(categories.bytes(category)))
}

fn add_assists(
    starts: &[i64],
    first: i64,
    last: i64,
    campaign: u32,
    days: &mut CampaignDays,
) {
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

fn process_partition<'c>(
    events: Vec<EventRec>,
    clicks: Vec<ClickRec>,
    ids: &IdInterner,
    categories: &'c Interner<'c>,
    real_campaign_count: u32,
    organic: u32,
    campaign_days: &mut CampaignDays,
) -> Vec<UserDailyRow<'c>> {
    // Group by person with one counting-sort pass each (dense per-partition
    // person index), then sort each person's small slice by time.  Persons are
    // visited in first-seen order; output row order does not matter.
    let PersonGroups {
        persons,
        events: mut grouped_events,
        event_offsets,
        clicks: mut grouped_clicks,
        click_offsets,
    } = group_by_person(&events, &clicks);
    drop(events);
    drop(clicks);

    let mut rows = Vec::new();
    let mut sessions = Vec::<Session>::new();
    let mut starts = Vec::<i64>::new();
    let mut category_revenue = Vec::<(u32, i64)>::new();
    let mut open = vec![(i64::MIN, i64::MIN); real_campaign_count as usize];
    let mut touched = Vec::<u32>::new();
    let mut name_bytes = Vec::new();

    for (dense, &person) in persons.iter().enumerate() {
        let events = &mut grouped_events
            [event_offsets[dense] as usize..event_offsets[dense + 1] as usize];
        // Order among equal-ts events cannot affect any output: they always
        // share a session and every per-session/day aggregate is a sum.
        events.sort_unstable_by_key(|event| event.ts);
        let events: &[EventRec] = events;
        let person_clicks = &mut grouped_clicks
            [click_offsets[dense] as usize..click_offsets[dense + 1] as usize];
        sort_person_clicks(person_clicks, ids);
        let person_clicks: &[ClickRec] = person_clicks;
        let event_end = events.len();
        let person_name = String::from_utf8_lossy(key_bytes(person, ids, &mut name_bytes)).into_owned();

        sessions.clear();
        category_revenue.clear();
        let mut current_day = i32::MIN;
        let mut day_sessions = 0i64;
        let mut day_events = 0i64;
        let mut day_revenue = 0i64;

        let mut index = 0;
        while index < event_end {
            let start_ts = events[index].ts;
            let day = start_ts.div_euclid(MS_PER_DAY) as i32;
            if day != current_day {
                if day_sessions != 0 {
                    rows.push(UserDailyRow {
                        user_id: Cow::Owned(person_name.clone()),
                        day: current_day,
                        sessions: day_sessions,
                        events: day_events,
                        revenue_cents: day_revenue,
                        top_category: top_category(&category_revenue, categories),
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
            while index < event_end && events[index].ts - previous_ts <= SESSION_GAP_MS {
                let event = events[index];
                session_events += 1;
                if event.event_type() == TYPE_PURCHASE {
                    let price = i64::from(event.price());
                    session_revenue += price;
                    has_purchase = true;
                    match category_revenue
                        .iter_mut()
                        .find(|(category, _)| *category == event.category)
                    {
                        Some(entry) => entry.1 += price,
                        None => category_revenue.push((event.category, price)),
                    }
                }
                previous_ts = event.ts;
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
                top_category: top_category(&category_revenue, categories),
            });
        }

        // Credit: latest click no later than the session start; because an
        // equal-ts run is byte-sorted, its first click has the smallest ID.
        let mut click_index = 0;
        let mut latest_run_start = 0;
        for session in &sessions {
            while click_index < person_clicks.len()
                && person_clicks[click_index].ts <= session.start_ts
            {
                if click_index == 0
                    || person_clicks[click_index].ts != person_clicks[click_index - 1].ts
                {
                    latest_run_start = click_index;
                }
                click_index += 1;
            }
            let credited = if click_index != 0
                && session.start_ts - person_clicks[click_index - 1].ts <= SEVEN_DAYS_MS
            {
                person_clicks[latest_run_start].campaign
            } else {
                organic
            };
            let cell = campaign_days.at(credited, session.day);
            cell.sessions += 1;
            cell.revenue += session.revenue;
            if session.has_purchase {
                cell.purchasing_sessions += 1;
            }
        }

        // Assists: merge each campaign's [click.ts, click.ts + 7d] intervals,
        // then count session starts in the merged intervals by UTC start day.
        if !person_clicks.is_empty() {
            starts.clear();
            starts.extend(sessions.iter().map(|session| session.start_ts));
            for click in person_clicks {
                let slot = &mut open[click.campaign as usize];
                if slot.0 == i64::MIN {
                    *slot = (click.ts, click.ts + SEVEN_DAYS_MS);
                    touched.push(click.campaign);
                } else if click.ts <= slot.1 {
                    slot.1 = click.ts + SEVEN_DAYS_MS;
                } else {
                    add_assists(&starts, slot.0, slot.1, click.campaign, campaign_days);
                    *slot = (click.ts, click.ts + SEVEN_DAYS_MS);
                }
            }
            for &campaign in &touched {
                let (first, last) = open[campaign as usize];
                add_assists(&starts, first, last, campaign, campaign_days);
                open[campaign as usize] = (i64::MIN, i64::MIN);
            }
            touched.clear();
        }
    }

    rows
}

pub fn run(data: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let mut stages = Stages::new();
    let manifest = manifest::read(data)?;
    let window_start =
        ts::parse_ts(manifest.window_start.as_bytes()).ok_or("manifest: bad window_start")?;
    let window_end =
        ts::parse_ts(manifest.window_end.as_bytes()).ok_or("manifest: bad window_end")?;
    ts::set_window_hint(window_start, window_end);

    let link_maps = map_all(&csv_files(&data.join("links")))?;
    let event_maps = map_all(&csv_files(&data.join("events")))?;
    let click_maps = map_all(&csv_files(&data.join("clicks")))?;
    let products_bytes = std::fs::read(data.join("products.csv"))?;
    let campaigns_bytes = std::fs::read(data.join("campaigns.csv"))?;

    let mut categories = Interner::new();
    let unknown_category = categories.intern(b"unknown");
    let mut product_category = FastMap::default();
    let mut fields = Vec::with_capacity(8);
    for line in data_lines(&products_bytes) {
        if split_fields(line, 2, &mut fields) {
            let category = categories.intern(fields[1]);
            product_category.insert(fields[0], category);
        }
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
    let organic = campaigns.intern(b"organic");
    let organic_channel = channels.intern(b"organic");
    if organic as usize >= campaign_channel.len() {
        campaign_channel.resize(organic as usize + 1, 0);
    }
    campaign_channel[organic as usize] = organic_channel;

    let lookups = Lookups {
        window_start,
        window_end,
        product_category,
        unknown_category,
        campaigns,
        real_campaign_count,
    };
    stages.mark("lookups + mmap");

    let ids = IdInterner::new();
    let link_files: Vec<&[u8]> = link_maps.iter().map(|map| &map[..]).collect();
    let people = links2::build(&link_files, &ids);
    stages.mark("links");

    let event_files: Vec<&[u8]> = event_maps.iter().map(|map| &map[..]).collect();
    let event_chunks = chunk_files(&event_files);
    let mut parsed_events = parse_events(&event_chunks, &lookups, &people, &ids);
    stages.mark("event parse");

    let dead = DeadMarks::new(parsed_events.iter().map(|chunk| chunk.valid));
    let token_partitions = take_token_partitions(&mut parsed_events);
    token_partitions
        .into_par_iter()
        .for_each(|mut tokens| dedup_tokens(&mut tokens, &dead));
    stages.mark("event dedup");

    // Click duplicates are identical values.  Keeping all copies cannot change
    // either the chosen click or the per-campaign set of assists, matching v1.
    let click_files: Vec<&[u8]> = click_maps.iter().map(|map| &map[..]).collect();
    let click_chunks = chunk_files(&click_files);
    let parsed_clicks = parse_clicks(&click_chunks, &lookups, &people, &ids);
    stages.mark("click parse");

    let partitions = collect_person_partitions(parsed_events, parsed_clicks, &dead);
    let campaign_count = lookups.campaigns.len();
    let new_campaign_days = || CampaignDays::new(campaign_count, window_start, window_end);
    let (mut shard_rows, campaign_days): (Vec<(usize, Vec<UserDailyRow>)>, CampaignDays) =
        partitions
            .into_par_iter()
            .fold(
                || (Vec::new(), new_campaign_days()),
                |(mut all_rows, mut days), (partition, events, clicks)| {
                    let rows = process_partition(
                        events,
                        clicks,
                        &ids,
                        &categories,
                        real_campaign_count,
                        organic,
                        &mut days,
                    );
                    all_rows.push((partition, rows));
                    (all_rows, days)
                },
            )
            .reduce(
                || (Vec::new(), new_campaign_days()),
                |(mut left_rows, left_days), (right_rows, right_days)| {
                    left_rows.extend(right_rows);
                    (left_rows, left_days.merge(right_days))
                },
            );
    stages.mark("person group + aggregate");

    shard_rows.sort_unstable_by_key(|(partition, _)| *partition);
    let user_parts: Vec<Vec<UserDailyRow>> =
        shard_rows.into_iter().map(|(_, rows)| rows).collect();
    let mut campaign_rows = Vec::<CampaignDailyRow>::new();
    campaign_days.for_each_row(|campaign, day, accumulated| {
        let channel = campaign_channel[campaign as usize];
        campaign_rows.push(CampaignDailyRow {
            campaign_id: String::from_utf8_lossy(lookups.campaigns.bytes(campaign)),
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
    let user_revenue: i64 = user_parts
        .iter()
        .flatten()
        .map(|row| row.revenue_cents)
        .sum();
    let campaign_revenue: i64 = campaign_rows.iter().map(|row| row.revenue_cents).sum();
    if user_sessions != campaign_sessions || user_revenue != campaign_revenue {
        eprintln!(
            "V2 INVARIANT VIOLATED: sessions {user_sessions} vs {campaign_sessions}, \
             revenue {user_revenue} vs {campaign_revenue}"
        );
    }
    stages.mark("build output rows");

    let user_dir = output.join("user_daily");
    std::fs::create_dir_all(&user_dir)?;
    let per_part = PARTITIONS.div_ceil(OUTPUT_PARTS);
    let writes: Vec<Result<(), String>> = user_parts
        .par_chunks(per_part)
        .enumerate()
        .map(|(part, group)| {
            let rows: usize = group.iter().map(Vec::len).sum();
            if rows == 0 && part != 0 {
                return Ok(());
            }
            out::write_user_daily(&user_dir, part, group.iter().flatten())
                .map_err(|error| error.to_string())
        })
        .collect();
    for write in writes {
        write?;
    }
    out::write_campaign_daily(&output.join("campaign_daily"), &campaign_rows)?;
    stages.mark("write parquet");

    // main exits immediately after success.  Avoid destructing the remaining
    // large mmaps, tables, and row vectors; all writers are already closed.
    std::mem::forget(user_parts);
    std::mem::forget(campaign_rows);
    std::mem::forget(campaign_days);
    std::mem::forget(dead);
    std::mem::forget(event_chunks);
    std::mem::forget(click_chunks);
    std::mem::forget(event_files);
    std::mem::forget(click_files);
    std::mem::forget(link_files);
    std::mem::forget(categories);
    std::mem::forget(channels);
    std::mem::forget(campaign_channel);
    std::mem::forget(lookups);
    std::mem::forget(people);
    std::mem::forget(ids);
    std::mem::forget(products_bytes);
    std::mem::forget(campaigns_bytes);
    std::mem::forget(link_maps);
    std::mem::forget(event_maps);
    std::mem::forget(click_maps);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(chunk: u64, local: u64) -> RecRef {
        (chunk << 32) | local
    }

    #[test]
    fn invalid_earliest_copy_kills_later_valid_copy() {
        let dead = DeadMarks::new([0, 1]);
        let mut tokens = vec![
            EventToken {
                event: 17,
                ingest: 20,
                rec: rec(1, 0),
            },
            EventToken {
                event: 17,
                ingest: 10,
                rec: INVALID_REC,
            },
        ];
        dedup_tokens(&mut tokens, &dead);
        assert!(dead.is_dead(rec(1, 0)));
    }

    #[test]
    fn earliest_valid_copy_alone_survives() {
        // Three chunks with 70, 0 and 3 valid records: chunk 2 starts at bit 70,
        // so the marks cross a word boundary.
        let dead = DeadMarks::new([70, 0, 3]);
        let mut tokens = vec![
            EventToken {
                event: 3,
                ingest: 30,
                rec: rec(2, 2),
            },
            EventToken {
                event: 3,
                ingest: 10,
                rec: rec(0, 69),
            },
            EventToken {
                event: 3,
                ingest: 20,
                rec: rec(2, 0),
            },
        ];
        dedup_tokens(&mut tokens, &dead);
        assert!(!dead.is_dead(rec(0, 69)));
        assert!(dead.is_dead(rec(2, 0)));
        assert!(dead.is_dead(rec(2, 2)));
        assert!(!dead.is_dead(rec(2, 1)));
        assert!(!dead.is_dead(rec(0, 0)));
    }

    #[test]
    fn invalid_ref_sorts_after_valid_refs() {
        assert!(rec(u32::MAX as u64 - 1, u32::MAX as u64) < INVALID_REC);
    }

    #[test]
    fn person_index_grows_and_keeps_dense_ids() {
        let mut index = PersonIndex::with_capacity(1);
        let keys: Vec<IdKey> = (0..1000u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
        for (i, &key) in keys.iter().enumerate() {
            assert_eq!(index.insert(key), i as u32);
        }
        for (i, &key) in keys.iter().enumerate() {
            assert_eq!(index.insert(key), i as u32);
            assert_eq!(index.get(key), i as u32);
        }
        assert_eq!(index.get(12345), NO_PERSON);
        assert_eq!(index.persons, keys);
    }

    #[test]
    fn grouping_matches_lexicographic_person_time_sort() {
        let ids = IdInterner::new();
        let mut events = Vec::new();
        let mut clicks = Vec::new();
        let mut state = 0x1234_5678u64;
        let mut next = || {
            state = fmix64(state.wrapping_add(0x9E37_79B9_7F4A_7C15));
            state
        };
        for i in 0..5000u64 {
            let person = next() % 97;
            let ts = (next() % 50) as i64 - 25;
            events.push(EventRec {
                person,
                ts,
                rec: i,
                ..EventRec::default()
            });
        }
        for _ in 0..3000 {
            // Some click persons (97..120) have no events and must be dropped.
            let person = next() % 120;
            let ts = (next() % 20) as i64 - 10;
            let click = key_of(format!("c{}", next() % 40).as_bytes(), &ids);
            clicks.push(ClickRec {
                person,
                ts,
                campaign: (next() % 5) as u32,
                click,
            });
        }

        let groups = group_by_person(&events, &clicks);
        let mut got_events = Vec::new();
        let mut got_clicks = Vec::new();
        let mut persons = groups.persons.clone();
        let mut grouped_events = groups.events.clone();
        let mut grouped_clicks = groups.clicks.clone();
        for (dense, &person) in groups.persons.iter().enumerate() {
            let slice = &mut grouped_events[groups.event_offsets[dense] as usize
                ..groups.event_offsets[dense + 1] as usize];
            assert!(!slice.is_empty());
            slice.sort_unstable_by_key(|event| event.ts);
            for event in slice.iter() {
                assert_eq!(event.person, person);
                got_events.push((event.person, event.ts, event.rec));
            }
            let slice = &mut grouped_clicks[groups.click_offsets[dense] as usize
                ..groups.click_offsets[dense + 1] as usize];
            sort_person_clicks(slice, &ids);
            for click in slice.iter() {
                assert_eq!(click.person, person);
                got_clicks.push((click.person, click.ts, click.click, click.campaign));
            }
        }

        // Reference: (person, ts) for events; (person, ts, click bytes) for
        // clicks of persons that have events.  Compare as sorted multisets plus
        // the per-person order checks below.
        persons.sort_unstable();
        persons.dedup();
        assert_eq!(persons.len(), groups.persons.len());
        let mut expected_events: Vec<_> =
            events.iter().map(|event| (event.person, event.ts, event.rec)).collect();
        expected_events.sort_unstable();
        let mut sorted_got = got_events.clone();
        sorted_got.sort_unstable();
        assert_eq!(sorted_got, expected_events);
        for pair in got_events.windows(2) {
            if pair[0].0 == pair[1].0 {
                assert!(pair[0].1 <= pair[1].1);
            }
        }

        let mut expected_clicks: Vec<_> = clicks
            .iter()
            .filter(|click| persons.binary_search(&click.person).is_ok())
            .map(|click| (click.person, click.ts, click.click, click.campaign))
            .collect();
        expected_clicks.sort_unstable();
        let mut sorted_got = got_clicks.clone();
        sorted_got.sort_unstable();
        assert_eq!(sorted_got, expected_clicks);
        let mut buf_a = Vec::new();
        let mut buf_b = Vec::new();
        for pair in got_clicks.windows(2) {
            if pair[0].0 == pair[1].0 {
                assert!(pair[0].1 <= pair[1].1);
                if pair[0].1 == pair[1].1 {
                    let a = key_bytes(pair[0].2, &ids, &mut buf_a).to_vec();
                    let b = key_bytes(pair[1].2, &ids, &mut buf_b).to_vec();
                    assert!(a <= b);
                }
            }
        }
    }
}
