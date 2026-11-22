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
//! `FAST_DECODE=0` selects the previous timestamp, event-key, and validation
//! paths in P. Set `FAST_RAW_EVENT_HEX=1` to include source-backed event keys
//! in a `FAST_PARSE_LEVEL` profiling run below level 6.
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
use crate::key::{key_bytes, key_cmp, key_of, user_key_bytes, user_key_of, IdInterner, IdKey};
use crate::links2::{self, alias_hash, AliasMap, PersonMap};
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

// Token-only tags: IdKey uses low nibbles 0 through 10. For source IDs the low
// 24 bits above the tag locate the unquoted 33-byte ID in the event chunk.
// The upper 36 bits route and screen comparisons, never decide equality.
const SHORT_EVENT_TAG: u64 = 14;
const RAW_EVENT_TAG: u64 = 15;
const RAW_EVENT_OFFSET_BITS: u32 = 24;
const RAW_EVENT_OFFSET_MAX: u64 = (1 << RAW_EVENT_OFFSET_BITS) - 1;

#[inline(always)]
fn raw_event_id(key: IdKey) -> bool {
    key & 15 == RAW_EVENT_TAG
}

#[inline(always)]
fn routed_event_key(key: IdKey) -> IdKey {
    if raw_event_id(key) {
        key & !(RAW_EVENT_OFFSET_MAX << 4)
    } else {
        key
    }
}

#[inline(always)]
fn source_event_id(bytes: &[u8], offset: u64) -> &[u8] {
    let offset = (offset & RAW_EVENT_OFFSET_MAX) as usize;
    &bytes[offset..offset + crate::hexid::HEX_ID_LEN]
}

#[inline(always)]
fn short_event_id(id: &[u8]) -> Option<IdKey> {
    // The generator's most frequent ID is e followed by eight ASCII bytes.
    // Dedup needs an exact identity but never orders event IDs by their bytes.
    const ASCII: u64 = 0x7f7f_7f7f_7f7f_7f7f;
    if id.len() == 9 && id[0] == b'e' {
        let suffix = u64::from_le_bytes(id[1..].try_into().unwrap());
        if suffix & !ASCII == 0 {
            #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
            // SAFETY: BMI2 is enabled by the build target.
            let packed = unsafe { std::arch::x86_64::_pext_u64(suffix, ASCII) } << 4;
            #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
            let packed = {
                // Save the low nibble in the unused high bits of the first
                // four ASCII bytes (bits 7, 15, 23, 31). The four shifted
                // copies of the nibble are disjoint before masking.
                let spread = ((suffix & 15) * 0x1020_4080) & 0x8080_8080;
                (suffix & !15) | spread
            };
            return Some(packed | SHORT_EVENT_TAG);
        }
    }
    if id.len() == 8 && id[0] == b'e' {
        let suffix = u64::from_le_bytes(id.try_into().unwrap()) >> 8;
        return Some((1 << 63) | (suffix << 4) | SHORT_EVENT_TAG);
    }
    None
}

#[inline(always)]
fn event_fingerprint(id: &[u8; crate::hexid::HEX_ID_LEN]) -> u64 {
    // All 33 bytes are opaque here. Two unaligned loads and AES rounds avoid
    // validating hexadecimal characters or packing nibbles to then hash them.
    // A match on the retained 36 bits always goes through a full byte comparison.
    #[cfg(all(target_arch = "x86_64", target_feature = "aes"))]
    {
        use std::arch::x86_64::*;
        // SAFETY: AES is enabled, and bytes 1..17 and 17..33 are both in bounds.
        unsafe {
            let a = _mm_loadu_si128(id.as_ptr().add(1).cast());
            let b = _mm_loadu_si128(id.as_ptr().add(17).cast());
            let seed = _mm_set1_epi8(id[0] as i8);
            let state = _mm_aesenc_si128(_mm_xor_si128(a, seed), b);
            let state = _mm_aesenc_si128(state, _mm_setzero_si128());
            _mm_cvtsi128_si64(_mm_aesenc_si128(state, seed)) as u64
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "aes")))]
    crate::hash::hash_bytes(id)
}

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

/// One event copy's dedup token.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
struct Tok {
    /// Exact token key or a tagged source reference for a 33-byte ID.
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
    /// An interned key or a (chunk, byte offset) reference, selected for the run.
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
#[cfg_attr(not(test), allow(dead_code))]
fn chunk_files<'a>(files: &[&'a [u8]]) -> Vec<Chunk<'a>> {
    chunk_files_sized(files, CHUNK_BYTES)
}

/// P chunk target size for real runs: 15 MiB. Fewer chunks means fewer
/// per-chunk partition segments for D and S to walk (2^bits partitions x
/// chunks); on the 8-vCPU x86 box 16 MiB beat 8 MiB by ~25 ms at d20 and
/// ~75 ms at d60 with the P tail still ~1 chunk. NOT 16: raw event-ID
/// offsets must fit 24 bits, and a chunk's last line can start at exactly
/// the target size, so 16 MiB can (rarely) trip the v1 fallback (+3 s);
/// 32 MiB always does. 15 MiB keeps every line start below 2^24 - 1 MiB.
/// (`CHUNK_BYTES` stays 8 MiB for the tests' chunk-boundary cases.)
const RUN_CHUNK_BYTES: usize = 15 << 20;

/// `FAST_CHUNK_MB=n` (tuning knob, 1..=15): P chunk target size in MiB.
/// FAST_CHUNK_MB=8 restores the old chunking.
fn chunk_bytes_setting() -> usize {
    std::env::var("FAST_CHUNK_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| (1..=15).contains(&n))
        .map_or(RUN_CHUNK_BYTES, |n| n << 20)
}

fn chunk_files_sized<'a>(files: &[&'a [u8]], chunk_bytes: usize) -> Vec<Chunk<'a>> {
    let mut chunks = Vec::new();
    for &bytes in files {
        let mut start = 0;
        while start < bytes.len() {
            let mut end = (start + chunk_bytes).min(bytes.len());
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

// Canonical p1..p99999 and c_1..c_99999 have direct lookup slots. Other
// spellings, including leading zeroes and arbitrary fixture IDs, use the map.
const NUMBERED_IDS: usize = 100_000;

#[inline(always)]
fn numbered_id(id: &[u8], prefix: &[u8]) -> Option<usize> {
    let digits = id.strip_prefix(prefix)?;
    if !(1..=5).contains(&digits.len()) || digits[0] == b'0' {
        return None;
    }
    ts::short_decimal(digits).map(|n| n as usize)
}

#[inline(always)]
fn purchase_type(kind: &[u8]) -> Option<bool> {
    match kind.len() {
        4 => {
            let word = u32::from_le_bytes(kind.try_into().unwrap());
            ((word == u32::from_le_bytes(*b"view")) | (word == u32::from_le_bytes(*b"cart"))).then_some(false)
        }
        8 => (u64::from_le_bytes(kind.try_into().unwrap()) == u64::from_le_bytes(*b"purchase")).then_some(true),
        _ => None,
    }
}

#[inline(always)]
fn purchase_price(bytes: &[u8]) -> Option<u32> {
    let value = if bytes.len() <= 8 {
        i64::from(ts::short_decimal(bytes)?)
    } else {
        // Even a very long field can be valid when it has leading zeroes.
        parse_digits(bytes)?
    };
    (1..=10_000_000).contains(&value).then_some(value as u32)
}

// ---------------------------------------------------------------------------
// Timers
// ---------------------------------------------------------------------------

struct Stages {
    enabled: bool,
    /// `FAST_TIMING=2`: also print each stage's start and end as raw
    /// CLOCK_MONOTONIC nanoseconds plus the pid, so a `perf record` taken on
    /// the x86 profiler can be sliced by stage.
    raw: bool,
    last: Instant,
    last_ns: u64,
    /// Process CPU seconds (user + sys) at the last mark, for per-stage
    /// core utilisation under FAST_TIMING.
    last_cpu: f64,
}

/// Process CPU seconds (user + sys) from getrusage.
fn process_cpu_secs() -> f64 {
    // SAFETY: zeroed rusage is a valid out-parameter for getrusage.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    tv(ru.ru_utime) + tv(ru.ru_stime)
}

/// FAST_TIMING: summarise a task trace `(thread, start, end, size)` for a
/// stage that took `total` seconds: per-thread busy and finish times, idle
/// core-seconds, and the last few tasks to finish.
fn report_trace(stage: &str, trace: &[(usize, f64, f64, usize)], total: f64) {
    let threads = trace.iter().map(|t| t.0 + 1).max().unwrap_or(0);
    let mut busy = vec![0.0f64; threads];
    let mut end = vec![0.0f64; threads];
    let mut first = vec![f64::MAX; threads];
    for &(t, a, b, _) in trace {
        busy[t] += b - a;
        end[t] = end[t].max(b);
        first[t] = first[t].min(a);
    }
    let idle: f64 = busy.iter().map(|b| total - b).sum();
    let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(" ");
    eprintln!("  v3   trace {stage}: tasks {} total {total:.3}s idle {idle:.3} core-s", trace.len());
    eprintln!("  v3   trace {stage}: start {}", fmt(&first));
    eprintln!("  v3   trace {stage}: busy  {}", fmt(&busy));
    eprintln!("  v3   trace {stage}: end   {}", fmt(&end));
    let mut last: Vec<_> = trace.to_vec();
    last.sort_by(|a, b| b.2.total_cmp(&a.2));
    for &(t, a, b, size) in last.iter().take(6) {
        eprintln!("  v3   trace {stage}: late task thr {t} {a:.3}-{b:.3} ({:.3}s) size {size}", b - a);
    }
}

/// `FAST_PAR_EXIT=1`: release big buffers in parallel before exit. Off by
/// default: measured on the 8-vCPU x86 box it reaches only ~1.5 cores (the
/// kernel serialises munmap/madvise on mmap_lock + TLB shootdowns) and costs
/// as much (~50 ms d20, ~95 ms d60) as the exit teardown it saves.
fn par_exit_enabled() -> bool {
    std::env::var_os("FAST_PAR_EXIT").as_deref() == Some(std::ffi::OsStr::new("1"))
}

/// The byte range a Vec's live elements occupy.
fn byte_range<T>(v: &[T]) -> (usize, usize) {
    (v.as_ptr() as usize, std::mem::size_of_val(v))
}

/// `madvise(MADV_DONTNEED)` the whole pages strictly inside `[ptr, ptr+len)`.
/// Only for buffers that are never read again (their owners are forgotten);
/// partial edge pages are left alone so allocator metadata is never touched.
fn release_pages(ptr: usize, len: usize) {
    const PAGE: usize = 4096;
    let lo = ptr.next_multiple_of(PAGE);
    let hi = (ptr + len) & !(PAGE - 1);
    if hi > lo + 16 * PAGE {
        // SAFETY: the range lies inside a live allocation nobody reads again;
        // DONTNEED on private anonymous memory just drops the pages.
        unsafe { libc::madvise(lo as *mut libc::c_void, hi - lo, libc::MADV_DONTNEED) };
    }
}

/// FAST_TIMING: resident memory split, from /proc/self/status.
fn print_rss() {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        let pick: Vec<&str> = status
            .lines()
            .filter(|l| l.starts_with("VmHWM") || l.starts_with("RssAnon") || l.starts_with("RssFile"))
            .map(|l| l.split_whitespace().collect::<Vec<_>>()[1])
            .collect();
        eprintln!("  v3   mem kB VmHWM/RssAnon/RssFile {}", pick.join(" "));
    }
}

/// CLOCK_MONOTONIC in nanoseconds (the clock `perf` timestamps use on Linux).
fn monotonic_ns() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: valid pointer to a timespec; CLOCK_MONOTONIC always exists.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

impl Stages {
    fn new() -> Self {
        let timing = std::env::var("FAST_TIMING").ok();
        Self {
            enabled: timing.is_some(),
            raw: timing.as_deref() == Some("2"),
            last: Instant::now(),
            last_ns: monotonic_ns(),
            last_cpu: process_cpu_secs(),
        }
    }

    fn mark(&mut self, label: &str) {
        if self.enabled {
            let wall = self.last.elapsed().as_secs_f64();
            eprintln!("  v3 {label:<25} {:>8.3}s", wall);
            let cpu = process_cpu_secs();
            eprintln!("  v3   cpu {label:<25} {:>8.3}s = {:.2} cores", cpu - self.last_cpu, (cpu - self.last_cpu) / wall.max(1e-9));
            self.last_cpu = cpu;
        }
        if self.raw {
            let now = monotonic_ns();
            eprintln!(
                "  v3stage pid={} label=\"{label}\" start_ns={} end_ns={}",
                std::process::id(),
                self.last_ns,
                now
            );
            self.last_ns = now;
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
        self.person_hashed(fmix64(key ^ 0x7065_7273_6f6e_7633))
    }
    #[inline(always)]
    fn person_hashed(self, hash: u64) -> usize {
        (hash >> (64 - self.bits)) as usize
    }
}

/// Stable counting scatter: put `items[i]` into group `tags[i]`, groups in
/// order, items within a group in input order. Returns the scattered items,
/// the offset table (`groups + 1` entries; group g is `off[g]..off[g+1]`), and
/// each input item's final position.
fn counting_scatter<T: Copy>(items: &[T], tags: &[u16], groups: usize) -> (Vec<T>, Vec<u32>, Vec<u32>) {
    let mut pos = Vec::with_capacity(items.len());
    let (out, off) = scatter_map(items, tags, groups, |item| item, |at| pos.push(at));
    (out, off, pos)
}

/// The scatter itself: like [`counting_scatter`] but writes `f(item)` (so a
/// remap can be fused into the copy), reports each item's final position to
/// `on_pos`, and writes straight into uninitialised memory (no zero-fill pass).
fn scatter_map<T: Copy, U: Copy>(
    items: &[T],
    tags: &[u16],
    groups: usize,
    f: impl Fn(T) -> U,
    mut on_pos: impl FnMut(u32),
) -> (Vec<U>, Vec<u32>) {
    assert_eq!(items.len(), tags.len());
    let mut off = vec![0u32; groups + 1];
    for &t in tags {
        off[t as usize + 1] += 1;
    }
    for g in 0..groups {
        off[g + 1] += off[g];
    }
    let mut cursor = off[..groups].to_vec();
    let mut out: Vec<U> = Vec::with_capacity(items.len());
    let slots = out.spare_capacity_mut();
    for (item, &t) in items.iter().zip(tags) {
        let at = &mut cursor[t as usize];
        slots[*at as usize].write(f(*item));
        on_pos(*at);
        *at += 1;
    }
    // SAFETY: the offsets are a prefix sum of the per-group counts, so the
    // cursors hand out every index in 0..items.len() exactly once: all
    // `items.len()` slots were written above.
    unsafe { out.set_len(items.len()) };
    (out, off)
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
    dense_products: Vec<u8>,
    unknown_cat: u8,
    campaigns: Interner<'a>,
    dense_campaigns: Vec<u16>,
    real_campaign_count: u32,
    decode: Option<ts::DecodeCache>,
    people: &'a PersonLookup,
    ids: &'a IdInterner,
    user_ids: &'a IdInterner,
    parts: Parts,
    /// See [`rec_index_bits`].
    k: u32,
    fallback: &'a Fallback,
    /// Profiling only (`FAST_PARSE_LEVEL=n`, default 6 = full work): stop each
    /// line after stage n (0 line scan only, 1 split, 2 ingest + event key, 3 validate,
    /// 4 user key, 5 staging push + the per-chunk resolve_persons loop; 6 adds
    /// the scatter), then stop the run after P without writing output. Stage
    /// cost = difference between consecutive levels. Clicks: 1-2 split,
    /// 3 validate, 4 user + click keys, 5 push + resolve, 6 scatter.
    parse_level: u8,
    /// `FAST_SCAN=0` selects the legacy event and click scanner.
    fast_scan: bool,
}

enum PersonLookup {
    Legacy(PersonMap),
    Aliases(AliasMap),
}

impl Ctx<'_> {
    #[inline]
    fn user_key<const RAW_USERS: bool>(&self, user: &[u8]) -> IdKey {
        if RAW_USERS {
            user_key_of(user, self.user_ids)
        } else {
            key_of(user, self.user_ids)
        }
    }

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

    #[inline(always)]
    fn validate_event_fast(&self, decode: &ts::DecodeCache, fields: &[&[u8]; 7], ingest_s: u32) -> Option<(i64, u32)> {
        let product = fields[5];
        if fields[1].is_empty() | product.is_empty() {
            return None;
        }
        let is_purchase = purchase_type(fields[4])?;
        let event_ts = decode.timestamp(fields[2])?;
        let late_floor = i64::from(ingest_s) * 1000 - LATE_CUTOFF_MS;
        if (event_ts < self.window_start) | (event_ts >= self.window_end) | (event_ts < late_floor) {
            return None;
        }
        if !is_purchase {
            return Some((event_ts, 0));
        }
        let price = purchase_price(fields[6])?;
        let cat = if let Some(index) = numbered_id(product, b"p") {
            self.dense_products[index]
        } else {
            self.product_cat.get(product).copied().unwrap_or(self.unknown_cat)
        };
        Some((event_ts, (price << 8) | u32::from(cat)))
    }

    #[inline(always)]
    fn campaign_fast(&self, id: &[u8]) -> Option<u32> {
        if let Some(index) = numbered_id(id, b"c_") {
            let campaign = self.dense_campaigns[index];
            return (campaign != u16::MAX).then_some(u32::from(campaign));
        }
        self.campaigns
            .get(id)
            .filter(|&campaign| campaign < self.real_campaign_count)
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

#[cold]
fn general_event_key(id: &[u8], ids: &IdInterner) -> IdKey {
    key_of(id, ids)
}

fn parse_event_chunk<const RAW_HEX: bool, const DECODE: bool, const RAW_USERS: bool>(
    ctx: &Ctx<'_>,
    chunk: &Chunk<'_>,
    chunk_id: usize,
) -> EvChunk {
    let parts = ctx.parts;
    let level = ctx.parse_level;
    let decode = if DECODE {
        Some(ctx.decode.as_ref().unwrap())
    } else {
        None
    };
    // Pre-size staging from the chunk size (~70-byte event lines) so the
    // pushes rarely reallocate and copy.
    let estimate = chunk.bytes.len() / 64 + 16;
    let mut evs = Vec::with_capacity(estimate);
    let mut toks = Vec::with_capacity(estimate);
    let mut tok_tag = Vec::with_capacity(estimate);
    // Level 0 (profiling only): the old memchr line scan, no fields.
    if level < 1 {
        for_each_line(chunk, |line| {
            std::hint::black_box(line);
        });
    }
    let scan_bytes: &[u8] = if level < 1 { &[] } else { chunk.bytes };
    // One SIMD pass per chunk; malformed lines never reach the closure.
    crate::scan3::for_each_record::<7>(scan_bytes, chunk.has_header, ctx.fast_scan, |fields| {
        let fields: &[&[u8]; 7] = fields;
        // rules Events 1: empty event_id produces nothing.
        if fields[0].is_empty() {
            return;
        }
        if level < 2 {
            std::hint::black_box(&fields);
            return;
        }
        // rules: "ingest_ts is always valid F1", so use the F1-only parser.
        // Anything else drops the line (never happens on spec-conforming data).
        let ingest = if DECODE {
            decode.unwrap().ingest_seconds(fields[3])
        } else {
            ts::parse_ingest_ts(fields[3])
        };
        let Some(ingest) = ingest else {
            return;
        };
        // The unsigned comparison also rejects negative epoch seconds.
        let max_ingest = u64::from(u32::MAX) * if DECODE { 1 } else { 1000 };
        if ingest as u64 > max_ingest {
            let ingest_ms = if DECODE { ingest * 1000 } else { ingest };
            ctx.fallback.trip(format!("ingest_ts {ingest_ms} ms is not whole u32 seconds"));
            return;
        }
        let event_id = fields[0];
        let short = if DECODE { short_event_id(event_id) } else { None };
        let ekey = if let Some(key) = short {
            key
        } else {
            let fingerprint = if !RAW_HEX {
                None
            } else if DECODE {
                <&[u8; crate::hexid::HEX_ID_LEN]>::try_from(event_id)
                    .ok()
                    .map(event_fingerprint)
            } else {
                crate::hexid::try_parse(event_id).map(|(prefix, value)| crate::hexid::fingerprint(prefix, value))
            };
            if let Some(fingerprint) = fingerprint {
                // scan3's normalized fields borrow the original chunk.
                let offset = event_id.as_ptr() as usize - chunk.bytes.as_ptr() as usize;
                debug_assert!(offset + event_id.len() <= chunk.bytes.len());
                if offset as u64 > RAW_EVENT_OFFSET_MAX {
                    ctx.fallback.trip("raw event ID offset exceeds 24 bits");
                    return;
                }
                (fingerprint << (RAW_EVENT_OFFSET_BITS + 4)) | ((offset as u64) << 4) | RAW_EVENT_TAG
            } else if DECODE {
                general_event_key(event_id, ctx.ids)
            } else {
                key_of(event_id, ctx.ids)
            }
        };
        if level < 3 {
            std::hint::black_box(ekey);
            return;
        }
        let valid = if DECODE {
            ctx.validate_event_fast(decode.unwrap(), fields, ingest as u32)
        } else {
            ctx.validate_event(fields, ingest)
        };
        if level < 4 {
            std::hint::black_box(valid);
            return;
        }
        // `rec` here is the staging index; it is remapped after the scatter.
        let rec = match valid {
            Some((event_ts, pc)) => {
                let user = ctx.user_key::<RAW_USERS>(fields[1]);
                if level < 5 {
                    std::hint::black_box(user);
                    return;
                }
                let index = evs.len() as u32;
                evs.push(Ev {
                    // The raw user key for now; resolve_persons turns it into
                    // the person key after the line loop.
                    person: user,
                    // In [0, window_end - base) by the window check + window guard.
                    ts_rel: (event_ts - ctx.base) as u32,
                    pc,
                });
                index
            }
            None => NONE,
        };
        toks.push(Tok {
            ekey,
            ingest_s: if DECODE { ingest as u32 } else { (ingest / 1000) as u32 },
            rec,
        });
        let group_key = if RAW_HEX { routed_event_key(ekey) } else { ekey };
        tok_tag.push(parts.event(group_key) as u16);
    });

    let ev_tag = resolve_persons(ctx, &mut evs, |e| &mut e.person);
    if level < 6 {
        std::hint::black_box((&evs, &toks, &ev_tag, &tok_tag));
        return EvChunk {
            evs: Vec::new(),
            ev_off: vec![0; parts.count() + 1],
            toks: Vec::new(),
            tok_off: vec![0; parts.count() + 1],
            dead: Vec::new(),
        };
    }
    if evs.len() as u64 > 1u64 << ctx.k {
        ctx.fallback
            .trip(format!("event chunk {chunk_id} has {} records > 2^{}", evs.len(), ctx.k));
    }

    let groups = parts.count();
    let (evs, ev_off, pos) = counting_scatter(&evs, &ev_tag, groups);
    // Scatter the tokens, pointing each one at its Ev's final slot in this
    // chunk's buffer as it is copied (the remap is fused into the scatter).
    let chunk_tag = (chunk_id as u32).wrapping_shl(ctx.k);
    let remap = |mut tok: Tok| {
        if tok.rec != NONE {
            tok.rec = chunk_tag | pos[tok.rec as usize];
        }
        tok
    };
    let (toks, tok_off) = scatter_map(&toks, &tok_tag, groups, remap, |_| {});
    let dead = (0..evs.len().div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
    EvChunk {
        evs,
        ev_off,
        toks,
        tok_off,
        dead,
    }
}

/// Replace raw user keys with person names, and return their partition tags.
fn resolve_persons<T>(
    ctx: &Ctx<'_>,
    records: &mut [T],
    person: impl Fn(&mut T) -> &mut IdKey,
) -> Vec<u16> {
    match ctx.people {
        PersonLookup::Legacy(people) => resolve_persons_old(ctx.parts, people, records, person),
        PersonLookup::Aliases(people) => resolve_aliases(ctx.parts, people, records, person),
    }
}

fn resolve_persons_old<T>(
    parts: Parts,
    people: &PersonMap,
    records: &mut [T],
    person: impl Fn(&mut T) -> &mut IdKey,
) -> Vec<u16> {
    let mut tags = Vec::with_capacity(records.len());
    const PREFETCH_AHEAD: usize = 16;
    let n = records.len();
    for i in 0..n {
        if i + PREFETCH_AHEAD < n {
            people.prefetch(*person(&mut records[i + PREFETCH_AHEAD]));
        }
        let key = person(&mut records[i]);
        *key = people.get(*key).unwrap_or(*key);
        tags.push(parts.person(*key) as u16);
    }
    tags
}

/// Route each record as itself first. Only possible aliases enter the
/// prefetch queue. The lookup's low hash bits address the bucket, while the
/// high bits already give the partition for a key that names itself.
fn resolve_aliases<T>(
    parts: Parts,
    people: &AliasMap,
    records: &mut [T],
    person: impl Fn(&mut T) -> &mut IdKey,
) -> Vec<u16> {
    // At this size the queue still fits in L1, and the bucket reads get
    // enough lead time even after the filter rejects most records.
    const BATCH: usize = 256;
    let mut tags = Vec::with_capacity(records.len());
    let mut pending = [(0, 0); BATCH];
    for batch in records.chunks_mut(BATCH) {
        let start = tags.len();
        let mut count = 0;
        for (i, record) in batch.iter_mut().enumerate() {
            let hash = alias_hash(*person(record));
            tags.push(parts.person_hashed(hash) as u16);
            // Advance the queue only for possible aliases, without a branch.
            pending[count] = (i, hash);
            count += usize::from(people.may_contain(hash));
        }
        let pending = &pending[..count];
        for &(_, hash) in pending {
            people.prefetch(hash);
        }
        for &(i, hash) in pending {
            let key = person(&mut batch[i]);
            let name = people.resolve(*key, hash);
            if name != *key {
                *key = name;
                tags[start + i] = parts.person_hashed(alias_hash(name)) as u16;
            }
        }
    }
    tags
}

fn parse_click_chunk<const SOURCE_IDS: bool, const DECODE: bool, const RAW_USERS: bool>(
    ctx: &Ctx<'_>,
    chunk: &Chunk<'_>,
    chunk_id: usize,
) -> CkChunk {
    let parts = ctx.parts;
    let level = ctx.parse_level;
    let click_start = ctx.base;
    let click_span = (ctx.window_end - click_start).max(0) as u64;
    let decode = if DECODE {
        Some(ctx.decode.as_ref().unwrap())
    } else {
        None
    };
    let mut cks = Vec::with_capacity(chunk.bytes.len() / 48 + 16); // ~55-byte click lines
    let mut has_fallback_click = false;
    if level < 1 {
        for_each_line(chunk, |line| {
            std::hint::black_box(line);
        });
    }
    let scan_bytes: &[u8] = if level < 1 { &[] } else { chunk.bytes };
    crate::scan3::for_each_record::<4>(scan_bytes, chunk.has_header, ctx.fast_scan, |fields| {
        let fields: &[&[u8]; 4] = fields;
        if level < 3 {
            std::hint::black_box(&fields);
            return;
        }
        let (click_id, user, campaign_id, ts_field) = (fields[0], fields[1], fields[2], fields[3]);
        // rules Clicks 1 and 3.
        let empty_id = if DECODE {
            click_id.is_empty() | user.is_empty()
        } else {
            click_id.is_empty() || user.is_empty()
        };
        if empty_id {
            return;
        }
        let click_ts = if DECODE {
            decode.unwrap().timestamp(ts_field)
        } else {
            ts::parse_ts(ts_field)
        };
        let Some(click_ts) = click_ts else {
            return;
        };
        let outside = if DECODE {
            (click_ts as u64).wrapping_sub(click_start as u64) >= click_span
        } else {
            click_ts < click_start || click_ts >= ctx.window_end
        };
        if outside {
            return;
        }
        let campaign = if DECODE {
            ctx.campaign_fast(campaign_id)
        } else {
            // `organic` is interned after the real campaigns.
            ctx.campaigns
                .get(campaign_id)
                .filter(|&campaign| campaign < ctx.real_campaign_count)
        };
        let Some(campaign) = campaign else {
            return;
        };
        if level < 4 {
            std::hint::black_box((click_ts, campaign));
            return;
        }
        let user = ctx.user_key::<RAW_USERS>(user);
        let click = if SOURCE_IDS {
            // The scanner returns a nonempty subslice of this chunk. The driver
            // checks the chunk count and byte lengths before enabling source IDs.
            let offset = click_id.as_ptr() as usize - chunk.bytes.as_ptr() as usize;
            (chunk_id as u64) << 32 | offset as u64
        } else {
            key_of(click_id, ctx.ids)
        };
        if level < 5 {
            std::hint::black_box((user, click));
            return;
        }
        // A fallback key has length nibble 0 (key.rs).
        if !SOURCE_IDS {
            has_fallback_click |= click & 0x0f == 0;
        }
        cks.push(Ck {
            person: user, // resolved after the line loop (resolve_persons)
            click,
            ts_rel: (click_ts - ctx.base) as u32,
            camp: campaign as u16,
        });
    });
    let tags = resolve_persons(ctx, &mut cks, |c| &mut c.person);
    if level < 6 {
        std::hint::black_box((&cks, &tags));
        return CkChunk {
            cks: Vec::new(),
            ck_off: vec![0; parts.count() + 1],
            has_fallback_click,
        };
    }
    let (cks, ck_off) = scatter_map(&cks, &tags, parts.count(), |c| c, |_| {});
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

/// The old gather + sort path for one partition: kept as the fallback for a
/// partition too big for the streaming table's scratch, and as the reference
/// the streaming kernel is differentially tested against.
fn dedup_partition_sorted(chunks: &[EvChunk], partition: usize, k: u32) {
    let mut keys = Vec::new();
    for chunk in chunks {
        let slice = &chunk.toks[chunk.tok_off[partition] as usize..chunk.tok_off[partition + 1] as usize];
        keys.extend(slice.iter().map(tok_key));
    }
    dedup_keys(&mut keys, &DeadView { chunks, k });
}

/// Per-worker scratch cap for the streaming dedup table (slots + stamps).
const DEDUP_SCRATCH_BYTES: usize = 8 << 20;
/// Salt for the table index hash. Deliberately different from the salt in
/// `Parts::event`: every key in a partition shares the top bits of *that*
/// hash, so reusing it would pile the table onto a fraction of its slots.
/// (We also take the LOW bits here, the partition id uses the HIGH bits.)
const DEDUP_SALT: u64 = 0x6465_6475_705f_7633; // "dedup_v3"

/// One winner-table entry: the best copy seen so far of event `ekey`.
#[derive(Clone, Copy, Default)]
struct WinSlot {
    ekey: IdKey,
    ingest_s: u32,
    rec: u32,
}

/// Bytes per table slot: the slot itself plus its epoch stamp.
const WIN_SLOT_BYTES: usize = std::mem::size_of::<WinSlot>() + std::mem::size_of::<u32>();

/// Reusable per-worker winner table. A slot is occupied iff
/// `stamps[i] == epoch`; bumping `epoch` empties the whole table without
/// touching memory. Slots past the current table size keep stale stamps, which
/// are always `< epoch` (stamps only ever take epoch values already used), so
/// growing is safe. When `epoch` would wrap, every stamp is zeroed first.
#[derive(Default)]
struct DedupScratch {
    slots: Vec<WinSlot>,
    stamps: Vec<u32>,
    sources: Vec<u64>,
    epoch: u32,
}

std::thread_local! {
    static DEDUP_SCRATCH: std::cell::RefCell<DedupScratch> = std::cell::RefCell::new(DedupScratch::default());
}

/// `(ingest_s, rec)` as one integer, ordered exactly like the low 64 bits of
/// `tok_key` (ingest first, then rec; `NONE` = u32::MAX sorts last on a tie).
#[inline(always)]
fn ingest_rec(ingest_s: u32, rec: u32) -> u64 {
    (u64::from(ingest_s) << 32) | u64::from(rec)
}

/// Streaming dedup of one event partition: one pass over every chunk's token
/// slice, keeping per event (exact `ekey` equality, never hash identity) the
/// copy with the smallest `(ingest_s, rec)`; every copy that loses a
/// comparison is marked dead at that moment.
///
/// Equivalence with `dedup_keys` (gather + sort): for one event with copies
/// `C`, the sort path picks `w = min(C)` by `(ingest_s, rec)` and marks every
/// copy except `w` (or every copy, if `w.rec == NONE`; marking `NONE` is a
/// no-op). Here, each comparison marks the larger of the stored copy and the
/// arriving one (the arriving one on an exact tie) and keeps the smaller, so
/// after the pass the table holds `min(C)` = `w`, and every other copy was
/// marked exactly once (when it lost - a copy is only ever compared while it
/// is the stored or arriving one, and once it loses it leaves the table).
/// `w` itself is never marked: it never compares larger than anything, and an
/// exact tie with `w` is only possible for another `NONE` copy with the same
/// ingest (valid `rec`s are unique), where both are no-op marks anyway. So the
/// set of marked valid recs is "all valid copies but `w`", or "all valid
/// copies" if `w.rec == NONE` - identical to the sort path, whatever order the
/// copies arrive in (the min of a set does not depend on order, and marks are
/// idempotent `fetch_or`s).
///
/// Returns false, having marked nothing, if this partition's table would
/// exceed `cap_bytes`; the caller must then use the sort path.
fn dedup_stream(chunks: &[EvChunk], partition: usize, k: u32, scratch: &mut DedupScratch, cap_bytes: usize) -> bool {
    dedup_stream_inner::<false>(chunks, &[], partition, k, scratch, cap_bytes)
}

fn dedup_stream_inner<const RAW_HEX: bool>(chunks: &[EvChunk], input: &[Chunk<'_>], partition: usize,
    k: u32, scratch: &mut DedupScratch, cap_bytes: usize) -> bool {
    let n: usize = chunks
        .iter()
        .map(|c| (c.tok_off[partition + 1] - c.tok_off[partition]) as usize)
        .sum();
    if n == 0 {
        return true;
    }
    // Load factor <= 1/2, so a probe always finds an empty slot or its key.
    let size = n.saturating_mul(2).next_power_of_two();
    let entry_bytes = WIN_SLOT_BYTES + if RAW_HEX { std::mem::size_of::<u64>() } else { 0 };
    if size.saturating_mul(entry_bytes) > cap_bytes {
        return false;
    }
    if scratch.slots.len() < size {
        scratch.slots.resize(size, WinSlot::default());
        scratch.stamps.resize(size, 0); // 0 is never a live epoch (see below)
    }
    if RAW_HEX && scratch.sources.len() < size {
        scratch.sources.resize(size, 0);
    }
    if scratch.epoch == u32::MAX {
        scratch.stamps.fill(0);
        scratch.epoch = 0;
    }
    scratch.epoch += 1; // live epochs are 1..=u32::MAX
    let epoch = scratch.epoch;
    let mask = size - 1;
    let slots = &mut scratch.slots[..size];
    let stamps = &mut scratch.stamps[..size];
    let dead = DeadView { chunks, k };
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        let slice = &chunk.toks[chunk.tok_off[partition] as usize..chunk.tok_off[partition + 1] as usize];
        for tok in slice {
            let from_source = RAW_HEX && raw_event_id(tok.ekey);
            let key = if RAW_HEX { routed_event_key(tok.ekey) } else { tok.ekey };
            let mut i = fmix64(key ^ DEDUP_SALT) as usize & mask;
            loop {
                if stamps[i] != epoch {
                    // First copy of this event: it is the winner so far.
                    stamps[i] = epoch;
                    slots[i] = WinSlot {
                        ekey: key,
                        ingest_s: tok.ingest_s,
                        rec: tok.rec,
                    };
                    if from_source {
                        scratch.sources[i] = ((chunk_index as u64) << RAW_EVENT_OFFSET_BITS)
                            | ((tok.ekey >> 4) & RAW_EVENT_OFFSET_MAX);
                    }
                    break;
                }
                let slot = &mut slots[i];
                if slot.ekey == key && (!from_source || {
                    let source = scratch.sources[i];
                    let previous_chunk = (source >> RAW_EVENT_OFFSET_BITS) as usize;
                    source_event_id(input[chunk_index].bytes, tok.ekey >> 4)
                        == source_event_id(input[previous_chunk].bytes, source)
                }) {
                    if ingest_rec(tok.ingest_s, tok.rec) < ingest_rec(slot.ingest_s, slot.rec) {
                        dead.mark(slot.rec); // no-op if the old winner is NONE
                        slot.ingest_s = tok.ingest_s;
                        slot.rec = tok.rec;
                    } else {
                        dead.mark(tok.rec); // no-op if NONE
                    }
                    break;
                }
                i = (i + 1) & mask;
            }
        }
    }
    true
}

fn dedup_raw_sorted(chunks: &[EvChunk], input: &[Chunk<'_>], partition: usize, k: u32) {
    let mut keys = Vec::new();
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        let range = chunk.tok_off[partition] as usize..chunk.tok_off[partition + 1] as usize;
        keys.extend(chunk.toks[range].iter().map(|tok| (tok, chunk_index)));
    }
    let group_cmp = |a: &(&Tok, usize), b: &(&Tok, usize)| {
        routed_event_key(a.0.ekey).cmp(&routed_event_key(b.0.ekey)).then_with(|| {
            if raw_event_id(a.0.ekey) {
                source_event_id(input[a.1].bytes, a.0.ekey >> 4)
                    .cmp(source_event_id(input[b.1].bytes, b.0.ekey >> 4))
            } else {
                std::cmp::Ordering::Equal
            }
        })
    };
    keys.sort_unstable_by(|a, b| group_cmp(a, b).then_with(|| {
        ingest_rec(a.0.ingest_s, a.0.rec).cmp(&ingest_rec(b.0.ingest_s, b.0.rec))
    }));
    let dead = DeadView { chunks, k };
    let mut winner = None;
    for entry in keys {
        if winner.as_ref().is_some_and(|first| group_cmp(first, &entry).is_eq()) {
            dead.mark(entry.0.rec);
        } else {
            winner = Some(entry);
        }
    }
}

/// Streaming dedup with the sort path as a fallback for oversized partitions.
fn dedup_partition_with(chunks: &[EvChunk], partition: usize, k: u32, scratch: &mut DedupScratch, cap_bytes: usize) {
    if !dedup_stream(chunks, partition, k, scratch, cap_bytes) {
        dedup_partition_sorted(chunks, partition, k);
    }
}

fn dedup_partition(chunks: &[EvChunk], partition: usize, k: u32) {
    DEDUP_SCRATCH.with(|scratch| {
        dedup_partition_with(chunks, partition, k, &mut scratch.borrow_mut(), DEDUP_SCRATCH_BYTES);
    });
}

std::thread_local! {
    static DEDUP2_SCRATCH: std::cell::RefCell<Vec<u32>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

/// FAST_D2=1 opts in to the compact winner table. Default OFF: on the x86
/// bench (ss-bench, d20, 8 threads) it measured neutral against the streaming
/// table (D 0.066-0.069 s either way), so the proven kernel stays the default.
fn dedup2_enabled() -> bool {
    std::env::var("FAST_D2").map_or(false, |v| v == "1")
}

/// Compact winner table. A slot holds a hash tag and a reference back to the
/// original token (chunk and index in its slice). The unique-ID fast path
/// writes just four bytes, and a probe only fetches the previous token on a
/// hash tag match. A tag match still checks the entire event ID.
fn dedup_compact<const RAW_HEX: bool>(
    chunks: &[EvChunk],
    input: &[Chunk<'_>],
    partition: usize,
    k: u32,
    scratch: &mut Vec<u32>,
    cap_bytes: usize,
) -> bool {
    let mut n = 0;
    let mut max_len = 0;
    for chunk in chunks {
        let len = (chunk.tok_off[partition + 1] - chunk.tok_off[partition]) as usize;
        n += len;
        max_len = max_len.max(len);
    }
    if n <= 1 {
        return true;
    }
    let Some(size) = n.saturating_mul(2).checked_next_power_of_two() else {
        return false;
    };
    // At most half full. Keep at least one tag bit for the occupied marker.
    let index_bits = usize::BITS - (max_len - 1).leading_zeros();
    let chunk_bits = usize::BITS - (chunks.len() - 1).leading_zeros();
    let ref_bits = index_bits + chunk_bits;
    if ref_bits >= u32::BITS || size > cap_bytes / std::mem::size_of::<u32>() {
        return false;
    }
    let index_mask = (1u32 << index_bits) - 1;
    let tag_mask = !((1u32 << ref_bits) - 1);
    if scratch.len() < size {
        scratch.resize(size, 0);
    }
    let table = &mut scratch[..size];
    table.fill(0);
    let table_mask = size - 1;
    let dead = DeadView { chunks, k };
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        let start = chunk.tok_off[partition] as usize;
        let end = chunk.tok_off[partition + 1] as usize;
        let chunk_ref = (chunk_index as u32) << index_bits;
        for (i, tok) in chunk.toks[start..end].iter().enumerate() {
            let key = if RAW_HEX { routed_event_key(tok.ekey) } else { tok.ekey };
            let hash = fmix64(key ^ DEDUP_SALT);
            let tag = (hash >> 32) as u32 & tag_mask | (1 << 31);
            let entry = tag | chunk_ref | i as u32;
            let mut at = hash as usize & table_mask;
            loop {
                let slot = &mut table[at];
                if *slot == 0 {
                    *slot = entry;
                    break;
                }
                if *slot & tag_mask == tag {
                    let reference = *slot & !tag_mask;
                    let previous_chunk = (reference >> index_bits) as usize;
                    let previous = &chunks[previous_chunk];
                    let winner = &previous.toks[previous.tok_off[partition] as usize + (reference & index_mask) as usize];
                    let winner_key = if RAW_HEX { routed_event_key(winner.ekey) } else { winner.ekey };
                    if winner_key == key && (!RAW_HEX || !raw_event_id(tok.ekey)
                        || source_event_id(input[chunk_index].bytes, tok.ekey >> 4)
                            == source_event_id(input[previous_chunk].bytes, winner.ekey >> 4))
                    {
                        if ingest_rec(tok.ingest_s, tok.rec) < ingest_rec(winner.ingest_s, winner.rec) {
                            dead.mark(winner.rec);
                            *slot = entry;
                        } else {
                            dead.mark(tok.rec);
                        }
                        break;
                    }
                }
                at = (at + 1) & table_mask;
            }
        }
    }
    true
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

#[derive(Default, Clone, Copy)]
struct CampaignCredit {
    revenue: i64,
    sessions: u32,
    purchasing_sessions: u32,
}

/// Per-(campaign, day) accumulator: a dense grid over the window's days when
/// small enough, else a map. (Copied from pipeline2.)
enum CampaignDays {
    Dense {
        cells: Vec<CampaignDay>,
        min_day: i32,
        days: usize,
    },
    /// FAST_WALK keeps the frequently updated assist counters separate from
    /// credit. Both count widths are safe: at most one per session per
    /// campaign, and v3's record-reference space fits fewer than 2^32 events.
    Compact {
        credits: Vec<CampaignCredit>,
        assists: Vec<u32>,
        min_day: i32,
        days: usize,
    },
    Map(FastMap<(u32, i32), CampaignDay>),
}

impl CampaignDays {
    fn new(campaigns: usize, window_start: i64, window_end: i64, compact: bool) -> Self {
        if window_end <= window_start {
            return Self::Map(FastMap::default());
        }
        let min_day = window_start.div_euclid(MS_PER_DAY);
        let max_day = (window_end - 1).div_euclid(MS_PER_DAY);
        let days = (max_day - min_day + 1) as usize;
        match days.checked_mul(campaigns) {
            Some(cells) if compact && cells <= MAX_DENSE_CELLS && i32::try_from(min_day).is_ok() => Self::Compact {
                credits: vec![CampaignCredit::default(); cells],
                assists: vec![0; cells],
                min_day: min_day as i32,
                days,
            },
            Some(cells) if cells <= MAX_DENSE_CELLS && i32::try_from(min_day).is_ok() => Self::Dense {
                cells: vec![CampaignDay::default(); cells],
                min_day: min_day as i32,
                days,
            },
            _ => Self::Map(FastMap::default()),
        }
    }

    #[inline(always)]
    fn index(campaign: u32, day: i32, min_day: i32, days: usize) -> usize {
        let day_index = day.wrapping_sub(min_day) as u32 as usize;
        assert!(day_index < days, "v3 session day outside manifest window");
        campaign as usize * days + day_index
    }

    #[inline]
    fn at(&mut self, campaign: u32, day: i32) -> &mut CampaignDay {
        match self {
            Self::Dense { cells, min_day, days } => {
                &mut cells[Self::index(campaign, day, *min_day, *days)]
            }
            Self::Map(map) => map.entry((campaign, day)).or_default(),
            Self::Compact { .. } => unreachable!("compact cells are accessed by credit or assist"),
        }
    }

    #[inline]
    fn credit(&mut self, campaign: u32, day: i32, revenue: i64, assisted: bool) {
        if let Self::Compact { credits, assists, min_day, days } = self {
            let index = Self::index(campaign, day, *min_day, *days);
            let cell = &mut credits[index];
            cell.sessions += 1;
            cell.revenue += revenue;
            cell.purchasing_sessions += u32::from(revenue != 0);
            if assisted {
                assists[index] += 1;
            }
        } else {
            let cell = self.at(campaign, day);
            cell.sessions += 1;
            cell.revenue += revenue;
            cell.purchasing_sessions += i64::from(revenue != 0);
            cell.assisted_sessions += i64::from(assisted);
        }
    }

    #[inline]
    fn assist(&mut self, campaign: u32, day: i32, count: u32) {
        if let Self::Compact { assists, min_day, days, .. } = self {
            assists[Self::index(campaign, day, *min_day, *days)] += count;
        } else {
            self.at(campaign, day).assisted_sessions += i64::from(count);
        }
    }

    fn merge(mut self, other: Self) -> Self {
        match (&mut self, other) {
            (Self::Dense { cells: left, .. }, Self::Dense { cells: right, .. }) => {
                for (left, right) in left.iter_mut().zip(&right) {
                    left.add(right);
                }
            }
            (Self::Compact { credits, assists, .. }, Self::Compact { credits: rhs_credits, assists: rhs_assists, .. }) => {
                for (left, right) in credits.iter_mut().zip(rhs_credits) {
                    left.revenue += right.revenue;
                    left.sessions += right.sessions;
                    left.purchasing_sessions += right.purchasing_sessions;
                }
                for (left, right) in assists.iter_mut().zip(rhs_assists) {
                    *left += right;
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
            Self::Compact { credits, assists, min_day, days } => {
                for (index, (credit, &assist)) in credits.iter().zip(assists).enumerate() {
                    if credit.sessions != 0 || assist != 0 {
                        let cell = CampaignDay {
                            sessions: i64::from(credit.sessions),
                            purchasing_sessions: i64::from(credit.purchasing_sessions),
                            revenue: credit.revenue,
                            assisted_sessions: i64::from(assist),
                        };
                        f((index / days) as u32, *min_day + (index % days) as i32, &cell);
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

fn click_source_id<'a>(reference: u64, chunks: &[Chunk<'a>]) -> &'a [u8] {
    let chunk = &chunks[(reference >> 32) as usize];
    let rest = &chunk.bytes[reference as u32 as usize..];
    let comma = memchr::memchr(b',', rest).expect("a parsed click ID precedes a comma");
    let id = &rest[..comma];
    // The offset already skips the opening quote. A valid field can have a
    // closing quote here only if the scanner stripped a wrapping pair.
    id.strip_suffix(b"\"").unwrap_or(id)
}

/// Clicks are already grouped by person. Only the first click in each
/// timestamp run can get credit. Assists count each campaign independently.
fn sort_clicks_with_source_ids(cks: &mut [Ck], chunks: &[Chunk<'_>]) {
    cks.sort_unstable_by_key(|c| c.ts_rel);
    let mut start = 0;
    while start + 1 < cks.len() {
        let ts = cks[start].ts_rel;
        if cks[start + 1].ts_rel != ts {
            start += 1;
            continue;
        }
        let mut best = start;
        let mut best_id = click_source_id(cks[best].click, chunks);
        let mut next = start + 1;
        while next < cks.len() && cks[next].ts_rel == ts {
            let id = click_source_id(cks[next].click, chunks);
            // The rules give identical copies of a click identical values.
            if id < best_id {
                best = next;
                best_id = id;
            }
            next += 1;
        }
        cks.swap(start, best);
        start = next;
    }
}

const NO_PERSON: u32 = u32::MAX;

/// Person key -> dense id (first-insertion order) for one partition.
/// Open addressing, linear probing, load <= 1/2; `NO_PERSON` marks an empty
/// slot. Reused across partitions (`reset`). A partition holds a few hundred
/// persons at 20M, so the table stays in L1/L2.
#[derive(Default)]
struct DenseIndex {
    slots: Vec<(IdKey, u32)>,
    mask: usize,
    persons: Vec<IdKey>,
}

impl DenseIndex {
    fn reset(&mut self, expected: usize) {
        let cap = (expected.max(8) * 2).next_power_of_two();
        self.slots.clear();
        self.slots.resize(cap, (0, NO_PERSON));
        self.mask = cap - 1;
        self.persons.clear();
    }

    #[inline(always)]
    fn slot_of(&self, key: IdKey) -> usize {
        // Salted differently from the partition hash, whose top bits are
        // equal for every key in this partition.
        fmix64(key ^ 0x6465_6e73_655f_7633) as usize & self.mask
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
        self.slots.clear();
        self.slots.resize(cap, (0, NO_PERSON));
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

/// Stable counting sort of `items` by `dense[i]` into `out`; records whose id
/// is `NO_PERSON` are dropped. Group `g` is `out[off[g]..off[g+1]]`. Buffers
/// are reused across partitions.
fn group_by_dense<T: Copy>(items: &[T], dense: &[u32], groups: usize, out: &mut Vec<T>, off: &mut Vec<u32>) {
    off.clear();
    off.resize(groups + 1, 0);
    for &d in dense {
        if d != NO_PERSON {
            off[d as usize + 1] += 1;
        }
    }
    for g in 0..groups {
        off[g + 1] += off[g];
    }
    let total = off[groups] as usize;
    out.clear();
    out.reserve(total);
    let slots = &mut out.spare_capacity_mut()[..total];
    // Use off[g] as group g's write cursor; afterwards off[g] == old off[g+1].
    for (item, &d) in items.iter().zip(dense) {
        if d != NO_PERSON {
            let at = &mut off[d as usize];
            slots[*at as usize].write(*item);
            *at += 1;
        }
    }
    // Shift the cursors back into start offsets.
    for g in (1..=groups).rev() {
        off[g] = off[g - 1];
    }
    off[0] = 0;
    // SAFETY: the start offsets partition 0..total and each group's cursor
    // wrote exactly its count, so all `total` slots were written.
    unsafe { out.set_len(total) };
}

/// Insertion sort by key: fastest for the short per-person slices.
#[inline]
fn insertion_sort_by_key<T: Copy, K: Ord>(slice: &mut [T], key: impl Fn(&T) -> K) {
    for i in 1..slice.len() {
        let item = slice[i];
        let k = key(&item);
        let mut j = i;
        while j > 0 && key(&slice[j - 1]) > k {
            slice[j] = slice[j - 1];
            j -= 1;
        }
        slice[j] = item;
    }
}

/// Reusable per-worker buffers for the person stage.
struct Scratch {
    evs: Vec<Ev>,
    /// Ping-pong buffer for `radix_sort_evs_by_ts`.
    ev_tmp: Vec<Ev>,
    cks: Vec<Ck>,
    sessions: Vec<Session>,
    starts: Vec<i64>,
    walk_starts: Vec<WalkStart>,
    walk_days: Vec<WalkDay>,
    walk_ranges: Vec<WalkRange>,
    category_revenue: Vec<(u8, i64)>,
    /// Per campaign: the open merged assist interval `[first, last]`.
    open: Vec<(i64, i64)>,
    touched: Vec<u32>,
    name_bytes: Vec<u8>,
    /// Person grouping (see process_partition): dense ids, per-record ids,
    /// and the grouped copies with their offset tables.
    index: DenseIndex,
    dense: Vec<u32>,
    evs_by: Vec<Ev>,
    ev_off: Vec<u32>,
    cks_by: Vec<Ck>,
    ck_off: Vec<u32>,
    /// Summed seconds in gather / sort / walk (reported under FAST_TIMING).
    phase: [f64; 5],
}

impl Scratch {
    fn new(real_campaign_count: u32) -> Self {
        Self {
            evs: Vec::new(),
            ev_tmp: Vec::new(),
            cks: Vec::new(),
            sessions: Vec::new(),
            starts: Vec::new(),
            walk_starts: Vec::new(),
            walk_days: Vec::new(),
            walk_ranges: Vec::new(),
            category_revenue: Vec::new(),
            open: vec![(i64::MIN, i64::MIN); real_campaign_count as usize],
            touched: Vec::new(),
            name_bytes: Vec::new(),
            index: DenseIndex::default(),
            dense: Vec::new(),
            evs_by: Vec::new(),
            ev_off: Vec::new(),
            cks_by: Vec::new(),
            ck_off: Vec::new(),
            phase: [0.0; 5],
        }
    }
}

/// Partitions with fewer live events than this keep the per-person sort: the
/// radix pass's fixed histogram cost would dominate.
const RADIX_MIN_EVENTS: usize = 64;

/// `FAST_S_RADIX=0` selects the old per-person event sort; anything else (or
/// unset) pre-sorts the partition's events by `ts_rel` with an LSD radix sort.
fn s_radix_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FAST_S_RADIX").map_or(true, |v| v != "0"))
}

/// Stable LSD radix sort of `evs` by `ts_rel`: 3 digits of 11/11/10 bits.
/// A digit that is constant across all keys is skipped (its pass would be the
/// identity). `tmp` is a reusable ping-pong buffer; on return the sorted data
/// is in `evs` (the two Vecs may have been swapped).
fn radix_sort_evs_by_ts(evs: &mut Vec<Ev>, tmp: &mut Vec<Ev>) {
    const BITS: u32 = 11;
    const BUCKETS: usize = 1 << BITS;
    const MASK: u32 = (BUCKETS - 1) as u32;
    let n = evs.len();
    if n < 2 {
        return;
    }
    // One read pass builds all three histograms and the varying-bit mask.
    let first = evs[0].ts_rel;
    let mut diff = 0u32;
    let mut hist = [[0u32; BUCKETS]; 3];
    for e in evs.iter() {
        let k = e.ts_rel;
        diff |= k ^ first;
        hist[0][(k & MASK) as usize] += 1;
        hist[1][((k >> BITS) & MASK) as usize] += 1;
        hist[2][(k >> (2 * BITS)) as usize] += 1; // top 10 bits
    }
    for (pass, counts) in hist.iter_mut().enumerate() {
        let shift = BITS * pass as u32;
        if (diff >> shift) & MASK == 0 {
            continue; // every key has the same digit here
        }
        // Exclusive prefix sums: counts[b] becomes bucket b's write cursor.
        let mut sum = 0u32;
        for c in counts.iter_mut() {
            let here = *c;
            *c = sum;
            sum += here;
        }
        tmp.clear();
        tmp.reserve(n);
        let slots = &mut tmp.spare_capacity_mut()[..n];
        // Scatter in input order: equal digits keep their relative order, so
        // the pass is stable and the LSD composition sorts by the full key.
        for e in evs.iter() {
            let at = &mut counts[((e.ts_rel >> shift) & MASK) as usize];
            slots[*at as usize].write(*e);
            *at += 1;
        }
        // SAFETY: the cursors partition 0..n and each wrote exactly its count,
        // so all n slots were written.
        unsafe { tmp.set_len(n) };
        std::mem::swap(evs, tmp);
    }
}

#[derive(Clone, Copy)]
enum ClickIds<'a> {
    Interned { ids: &'a IdInterner, any_fallback: bool },
    Source(&'a [Chunk<'a>]),
}

/// Group a gathered partition by person and order each person's records.
///
/// Group by person without a comparison sort over the whole partition: give
/// each person with events a dense id (first-seen order) through a small
/// L1/L2-resident hash table, counting-sort events and clicks by that id, then
/// order each person's short slice by time. Clicks of persons with no events
/// get no id and are dropped: they can credit or assist nothing.
///
/// With `radix`, the whole partition's events are first stably radix-sorted by
/// `ts_rel`; `group_by_dense` is a stable scatter, so each person's event slice
/// then comes out already in ts order and needs no per-person sort. This also
/// changes the dense-id (first-seen) order, i.e. only the order of
/// `index.persons` and hence of pushed rows: row order is free in the output
/// and campaign_days are sums, so output is identical. Returns the instant the
/// grouping (before per-person sorts) finished.
fn group_partition(scratch: &mut Scratch, click_ids: ClickIds<'_>, radix: bool) -> Instant {
    let Scratch { evs, ev_tmp, cks, index, dense, evs_by, ev_off, cks_by, ck_off, .. } = scratch;
    if radix {
        radix_sort_evs_by_ts(evs, ev_tmp);
    }
    index.reset(evs.len() / 16 + 8);
    dense.clear();
    dense.extend(evs.iter().map(|e| index.insert(e.person)));
    let groups = index.persons.len();
    group_by_dense(evs, dense, groups, evs_by, ev_off);
    dense.clear();
    dense.extend(cks.iter().map(|c| index.get(c.person)));
    group_by_dense(cks, dense, groups, cks_by, ck_off);
    let t_grouped = Instant::now();
    for d in 0..groups {
        if !radix {
            // Events by ts. Order among equal-ts events cannot affect any
            // output (they share a session; all aggregates are sums).
            let slice = &mut evs_by[ev_off[d] as usize..ev_off[d + 1] as usize];
            if slice.len() <= 24 {
                insertion_sort_by_key(slice, |e| e.ts_rel);
            } else {
                slice.sort_unstable_by_key(|e| e.ts_rel);
            }
        }
        // Clicks by (ts, click_id bytes): within an equal-ts run the first
        // click has the byte-wise smallest click_id (rules Clicks 4).
        let slice = &mut cks_by[ck_off[d] as usize..ck_off[d + 1] as usize];
        if slice.len() > 1 {
            match click_ids {
                ClickIds::Interned { ids, any_fallback } => sort_clicks(slice, ids, any_fallback),
                ClickIds::Source(chunks) => sort_clicks_with_source_ids(slice, chunks),
            }
        }
    }
    t_grouped
}

struct PersonCtx<'a> {
    base: i64,
    ids: &'a IdInterner,
    ascii_users: bool,
    cat_names: &'a [&'a [u8]],
    organic: u32,
    click_ids: ClickIds<'a>,
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
    let radix = s_radix_enabled() && scratch.evs.len() >= RADIX_MIN_EVENTS;
    let t_grouped = group_partition(scratch, pc.click_ids, radix);
    let t_sorted = Instant::now();
    scratch.phase[3] += (t_grouped - t_gathered).as_secs_f64();
    scratch.phase[4] += (t_sorted - t_grouped).as_secs_f64();
    // Phase CPU-time totals for FAST_TIMING: gather, sort, walk.
    scratch.phase[0] += (t_gathered - t_start).as_secs_f64();
    scratch.phase[1] += (t_sorted - t_gathered).as_secs_f64();

    let Scratch {
        index,
        evs_by,
        ev_off,
        cks_by,
        ck_off,
        sessions,
        starts,
        category_revenue,
        open,
        touched,
        name_bytes,
        phase,
        ..
    } = scratch;
    let phase_walk = &mut phase[2];

    for (d, &person) in index.persons.iter().enumerate() {
        let events = &evs_by[ev_off[d] as usize..ev_off[d + 1] as usize];
        let person_clicks = &cks_by[ck_off[d] as usize..ck_off[d + 1] as usize];

        let name = if pc.ascii_users {
            user_key_bytes(person, ids, name_bytes)
        } else {
            key_bytes(person, ids, name_bytes)
        };
        let person_name = String::from_utf8_lossy(name).into_owned();

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
        // within 7 days. Each equal-ts run starts with its smallest ID. ---
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

/// `FAST_S2=0` selects the original person stage + output path
/// (`process_partition` with one owned `String` per row, all Parquet encoding
/// after S). Anything else (or unset) selects `process_partition2`: columnar
/// rows, bitmap gather, and Parquet parts encoded by the S workers themselves
/// as their row buffers fill, overlapping O with S.
fn s2_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FAST_S2").map_or(true, |v| v != "0"))
}

/// Append the live (not dedup-killed) entries of `evs[lo..hi]` to `out`, in
/// order. Reads the dead bitmap one 64-bit word at a time: a word with no dead
/// bit in range copies its whole run with one `extend_from_slice` (a memcpy),
/// otherwise only the live bits are visited (`trailing_zeros` walk).
#[inline]
fn gather_live(evs: &[Ev], dead: &[AtomicU64], lo: usize, hi: usize, out: &mut Vec<Ev>) {
    let mut i = lo;
    while i < hi {
        let word_index = i / 64;
        let bit = i % 64;
        // This step covers indices i..end, all inside one bitmap word.
        let end = ((word_index + 1) * 64).min(hi);
        let span = end - i;
        let mask = if span == 64 { u64::MAX } else { (1u64 << span) - 1 };
        // After the shift, bit j of `dead_here` is index i + j.
        let dead_here = (dead[word_index].load(Ordering::Relaxed) >> bit) & mask;
        if dead_here == 0 {
            out.extend_from_slice(&evs[i..end]);
        } else {
            let mut live = !dead_here & mask;
            while live != 0 {
                out.push(evs[i + live.trailing_zeros() as usize]);
                live &= live - 1; // clear the lowest set bit
            }
        }
        i = end;
    }
}

/// Like `top_category`, but returns the winning category's index (or
/// `out::TOP_NULL` with no purchases). Same rule: highest summed revenue, tie
/// -> smallest index = byte-wise smallest name.
#[inline]
fn top_category_index(revenues: &[(u8, i64)]) -> u16 {
    let mut top: Option<(u8, i64)> = None;
    for &(category, revenue) in revenues {
        top = match top {
            Some((best_category, best_revenue))
                if !(revenue > best_revenue || (revenue == best_revenue && category < best_category)) =>
            {
                Some((best_category, best_revenue))
            }
            _ => Some((category, revenue)),
        };
    }
    top.map_or(out::TOP_NULL, |(category, _)| u16::from(category))
}

fn walk_enabled() -> bool {
    std::env::var("FAST_WALK").map_or(true, |v| v != "0")
}

/// Index purchases by category in O(1), and only clear categories that
/// received revenue. Non-purchasing days need neither a scan nor a reset.
/// The full u8 range is valid, even when the input has more than 20 categories.
struct WalkCategories {
    revenue: [i64; 256],
    touched: [u8; 256],
    len: usize,
}

impl WalkCategories {
    #[inline]
    fn add(&mut self, category: u8, price: i64) {
        let revenue = &mut self.revenue[category as usize];
        if *revenue == 0 {
            self.touched[self.len] = category;
            self.len += 1;
        }
        *revenue += price;
    }

    #[inline]
    fn take_top(&mut self) -> u16 {
        let mut best_revenue = 0;
        let mut best = out::TOP_NULL;
        for &category in &self.touched[..self.len] {
            let revenue = std::mem::take(&mut self.revenue[category as usize]);
            let category = u16::from(category);
            if revenue > best_revenue || (revenue == best_revenue && category < best) {
                best_revenue = revenue;
                best = category;
            }
        }
        self.len = 0;
        best
    }
}

#[derive(Clone, Copy)]
struct WalkStart {
    ts: u32,
    day_run: u32,
}

struct WalkDay {
    day: i32,
    past: u32,
}

/// A u32 timestamp span holds fewer than 2,400 sessions (each starts over
/// 30 minutes after the previous start). Zero past means no open range.
#[derive(Clone, Copy, Default)]
struct WalkRange {
    first: u16,
    past: u16,
}

/// Count the sessions in the ordinal range [first, past), by day. Each
/// start points directly to its day run, including at a partial-day edge.
#[inline]
fn add_walk_assists(
    starts: &[WalkStart],
    day_runs: &[WalkDay],
    mut first: usize,
    past: usize,
    campaign: u32,
    days: &mut CampaignDays,
) {
    while first < past {
        let run = &day_runs[starts[first].day_run as usize];
        let end = past.min(run.past as usize);
        days.assist(campaign, run.day, (end - first) as u32);
        first = end;
    }
}

/// For long click histories, project each click's inclusive seven-day reach
/// onto session ordinals with two monotone cursors. Each campaign's pending
/// union can then merge adjacent or overlapping ranges. Counting a union by
/// day uses the day runs produced during sessionization.
fn long_history_assists(
    clicks: &[Ck],
    starts: &[WalkStart],
    day_runs: &[WalkDay],
    open: &mut [WalkRange],
    touched: &mut Vec<u32>,
    days: &mut CampaignDays,
) {
    let (mut first, mut past) = (0, 0);
    for click in clicks {
        while first < starts.len() && starts[first].ts < click.ts_rel {
            first += 1;
        }
        if first == starts.len() {
            break;
        }
        if starts[first].ts - click.ts_rel > SEVEN_DAYS_MS as u32 {
            continue;
        }
        let end = click.ts_rel.saturating_add(SEVEN_DAYS_MS as u32);
        past = past.max(first);
        while past < starts.len() && starts[past].ts <= end {
            past += 1;
        }
        let campaign = u32::from(click.camp);
        let slot = &mut open[campaign as usize];
        if slot.past == 0 {
            touched.push(campaign);
        } else if first <= usize::from(slot.past) {
            slot.past = past as u16;
            continue;
        } else {
            add_walk_assists(starts, day_runs, usize::from(slot.first), usize::from(slot.past), campaign, days);
        }
        *slot = WalkRange { first: first as u16, past: past as u16 };
    }
    for campaign in touched.drain(..) {
        let range = std::mem::take(&mut open[campaign as usize]);
        add_walk_assists(starts, day_runs, usize::from(range.first), usize::from(range.past), campaign, days);
    }
}

/// A sorted block that ends within one gap of the previous event cannot
/// contain a session boundary. Skip the per-event time and price branches
/// for a block of non-purchases.
#[inline]
fn walk_session(
    events: &[Ev],
    index: &mut usize,
    categories: &mut WalkCategories,
) -> i64 {
    let first = events[*index];
    let mut revenue = i64::from(first.price());
    if revenue != 0 {
        categories.add(first.category(), revenue);
    }
    let mut previous = first.ts_rel;
    let mut rest = &events[*index + 1..];
    if events.len() >= 24 {
        while let Some(block) = rest.first_chunk::<4>() {
            if block[3].ts_rel - previous > SESSION_GAP_MS as u32 {
                break;
            }
            if (block[0].pc | block[1].pc | block[2].pc | block[3].pc) >> 8 != 0 {
                for &event in block {
                    let price = i64::from(event.price());
                    if price != 0 {
                        categories.add(event.category(), price);
                        revenue += price;
                    }
                }
            }
            previous = block[3].ts_rel;
            rest = &rest[4..];
        }
    }
    while let Some((event, remaining)) = rest.split_first() {
        if event.ts_rel - previous > SESSION_GAP_MS as u32 {
            break;
        }
        let price = i64::from(event.price());
        if price != 0 {
            categories.add(event.category(), price);
            revenue += price;
        }
        previous = event.ts_rel;
        rest = remaining;
    }
    *index = events.len() - rest.len();
    revenue
}

/// Consume already-sorted person slices. Short click histories use the two
/// endpoints of the eligible range as session starts increase. Long histories
/// keep the interval union so a whale does not scan its active clicks on each
/// session. Only the latter case needs a buffer of session starts.
fn walk_sorted_partition(
    pc: &PersonCtx<'_>,
    scratch: &mut Scratch,
    days: &mut CampaignDays,
    cols: &mut out::UserDailyCols,
) {
    let Scratch {
        index, evs_by, ev_off, cks_by, ck_off, walk_starts, walk_days, walk_ranges, open, touched, name_bytes, ..
    } = scratch;
    if walk_ranges.len() < open.len() {
        walk_ranges.resize(open.len(), WalkRange::default());
    }
    let base_day = pc.base.div_euclid(MS_PER_DAY);
    let base_rem = pc.base.rem_euclid(MS_PER_DAY) as u64;
    let day_ms = MS_PER_DAY as u64;
    let mut categories = WalkCategories { revenue: [0; 256], touched: [0; 256], len: 0 };
    const SHORT_CLICKS: usize = 8;
    for (d, &person) in index.persons.iter().enumerate() {
        let events = &evs_by[ev_off[d] as usize..ev_off[d + 1] as usize];
        let clicks = &cks_by[ck_off[d] as usize..ck_off[d + 1] as usize];
        let name = if pc.ascii_users {
            user_key_bytes(person, pc.ids, name_bytes)
        } else {
            key_bytes(person, pc.ids, name_bytes)
        };
        let name = String::from_utf8_lossy(name);
        let long_history = clicks.len() > SHORT_CLICKS;
        if long_history {
            walk_starts.clear();
            walk_days.clear();
        }
        let mut left = 0;
        let mut upcoming = clicks;
        let mut latest_ts = 0;
        let mut latest_campaign = pc.organic;
        let (mut day, mut day_end) = (0, 0);
        let (mut day_sessions, mut day_events, mut day_revenue) = (0, 0, 0);
        let mut i = 0;
        while i < events.len() {
            let first = i;
            let start = events[first].ts_rel;
            // For multiple sessions on one UTC day, only the first needs the
            // division. The next midnight may exceed the range of ts_rel.
            if u64::from(start) >= day_end {
                if day_sessions != 0 {
                    cols.push(&name, day, day_sessions, day_events, day_revenue, categories.take_top());
                    if long_history {
                        walk_days.push(WalkDay { day, past: walk_starts.len() as u32 });
                    }
                }
                let day_index = (base_rem + u64::from(start)) / day_ms;
                day = (base_day + day_index as i64) as i32;
                day_end = (day_index + 1) * day_ms - base_rem;
                day_sessions = 0;
                day_events = 0;
                day_revenue = 0;
            }
            let revenue = walk_session(events, &mut i, &mut categories);
            day_sessions += 1;
            day_events += (i - first) as i64;
            day_revenue += revenue;

            while let Some((click, rest)) = upcoming.split_first() {
                if click.ts_rel > start {
                    break;
                }
                // Clicks with equal timestamps are ordered by ID. The first
                // in a run owns credit, but every campaign can assist.
                if latest_campaign == pc.organic || click.ts_rel != latest_ts {
                    latest_ts = click.ts_rel;
                    latest_campaign = u32::from(click.camp);
                }
                upcoming = rest;
            }
            let lower = start.saturating_sub(SEVEN_DAYS_MS as u32);
            let has_click = latest_campaign != pc.organic && latest_ts >= lower;
            let credited = if has_click { latest_campaign } else { pc.organic };
            days.credit(credited, day, revenue, has_click && !long_history);
            if long_history {
                walk_starts.push(WalkStart { ts: start, day_run: walk_days.len() as u32 });
            } else if has_click {
                while clicks[left].ts_rel < lower {
                    left += 1;
                }
                let active = &clicks[left..clicks.len() - upcoming.len()];
                for (j, click) in active.iter().enumerate() {
                    let campaign = u32::from(click.camp);
                    if campaign != credited && !active[..j].iter().any(|prev| prev.camp == click.camp) {
                        days.assist(campaign, day, 1);
                    }
                }
            }
        }
        cols.push(&name, day, day_sessions, day_events, day_revenue, categories.take_top());

        if long_history {
            walk_days.push(WalkDay { day, past: walk_starts.len() as u32 });
            long_history_assists(clicks, walk_starts, walk_days, walk_ranges, touched, days);
        }
    }
}

/// The `FAST_S2` person stage for one partition. Same rules and results as
/// `process_partition` (unit test `s2_matches_old_person_stage` compares them
/// on random inputs), with these cost cuts:
/// - gather reads the dead bitmap a word at a time (`gather_live`);
/// - rows go into columnar buffers (`out::UserDailyCols`): the person's name
///   is rendered once and its bytes appended per row, instead of one heap
///   `String` per row;
/// - session/day arithmetic runs on the u32 `ts_rel` values: the day of a
///   session is `base_day + (base_rem + ts_rel) / MS_PER_DAY` (all
///   non-negative, so a plain unsigned division instead of `div_euclid` on
///   i64), and the 30-minute gap test is a u32 subtraction (events are sorted,
///   so it never underflows).
fn process_partition2(
    partition: usize,
    ev_chunks: &[EvChunk],
    ck_chunks: &[CkChunk],
    pc: &PersonCtx<'_>,
    scratch: &mut Scratch,
    campaign_days: &mut CampaignDays,
    cols: &mut out::UserDailyCols,
    fast_walk: bool,
) {
    let base = pc.base;
    let ids = pc.ids;
    let t_start = Instant::now();
    let total: usize = ev_chunks
        .iter()
        .map(|c| (c.ev_off[partition + 1] - c.ev_off[partition]) as usize)
        .sum();
    scratch.evs.clear();
    scratch.evs.reserve(total);
    for chunk in ev_chunks {
        let (lo, hi) = (chunk.ev_off[partition] as usize, chunk.ev_off[partition + 1] as usize);
        gather_live(&chunk.evs, &chunk.dead, lo, hi, &mut scratch.evs);
    }
    scratch.cks.clear();
    for chunk in ck_chunks {
        let (lo, hi) = (chunk.ck_off[partition] as usize, chunk.ck_off[partition + 1] as usize);
        scratch.cks.extend_from_slice(&chunk.cks[lo..hi]);
    }
    let t_gathered = Instant::now();
    let radix = s_radix_enabled() && scratch.evs.len() >= RADIX_MIN_EVENTS;
    let t_grouped = group_partition(scratch, pc.click_ids, radix);
    let t_sorted = Instant::now();
    scratch.phase[3] += (t_grouped - t_gathered).as_secs_f64();
    scratch.phase[4] += (t_sorted - t_grouped).as_secs_f64();
    scratch.phase[0] += (t_gathered - t_start).as_secs_f64();
    scratch.phase[1] += (t_sorted - t_gathered).as_secs_f64();

    if fast_walk {
        walk_sorted_partition(pc, scratch, campaign_days, cols);
        scratch.phase[2] += t_sorted.elapsed().as_secs_f64();
        return;
    }

    let Scratch { index, evs_by, ev_off, cks_by, ck_off, sessions, starts, category_revenue, open, touched, name_bytes, phase, .. } =
        scratch;
    // base = base_day * MS_PER_DAY + base_rem with 0 <= base_rem < MS_PER_DAY,
    // so (base + t).div_euclid(MS_PER_DAY) == base_day + (base_rem + t) / MS_PER_DAY
    // for every t >= 0.
    let base_day = base.div_euclid(MS_PER_DAY);
    let base_rem = base.rem_euclid(MS_PER_DAY) as u64;
    let day_of = |ts_rel: u32| (base_day + ((base_rem + u64::from(ts_rel)) / MS_PER_DAY as u64) as i64) as i32;
    const GAP: u32 = SESSION_GAP_MS as u32;
    const _: () = assert!(SESSION_GAP_MS <= u32::MAX as i64);

    for (d, &person) in index.persons.iter().enumerate() {
        let events = &evs_by[ev_off[d] as usize..ev_off[d + 1] as usize];
        let person_clicks = &cks_by[ck_off[d] as usize..ck_off[d + 1] as usize];
        // Borrowed (no allocation) whenever the name is valid UTF-8, which
        // the old path's from_utf8_lossy would also have kept unchanged.
        // Person keys come from the user-ID table when FAST_PERSON is on
        // (atk-person), so decode them with the matching function.
        let name = if pc.ascii_users {
            user_key_bytes(person, ids, name_bytes)
        } else {
            key_bytes(person, ids, name_bytes)
        };
        let person_name = String::from_utf8_lossy(name);

        // --- Sessionize (rules Events 5-6) ---
        sessions.clear();
        category_revenue.clear();
        let mut current_day = i32::MIN;
        let mut day_sessions = 0i64;
        let mut day_events = 0i64;
        let mut day_revenue = 0i64;
        let mut index = 0;
        while index < events.len() {
            let start_rel = events[index].ts_rel;
            let day = day_of(start_rel);
            if day != current_day {
                if day_sessions != 0 {
                    cols.push(&person_name, current_day, day_sessions, day_events, day_revenue,
                        top_category_index(category_revenue));
                }
                current_day = day;
                day_sessions = 0;
                day_events = 0;
                day_revenue = 0;
                category_revenue.clear();
            }
            let mut previous = start_rel;
            let first = index;
            let mut session_revenue = 0i64;
            let mut has_purchase = false;
            // Equal timestamps give gap 0, so they always share the session.
            while index < events.len() && events[index].ts_rel - previous <= GAP {
                let event = events[index];
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
                previous = event.ts_rel;
                index += 1;
            }
            day_sessions += 1;
            day_events += (index - first) as i64;
            day_revenue += session_revenue;
            sessions.push(Session { start_ts: base + i64::from(start_rel), day, revenue: session_revenue, has_purchase });
        }
        if day_sessions != 0 {
            cols.push(&person_name, current_day, day_sessions, day_events, day_revenue,
                top_category_index(category_revenue));
        }

        // --- Credit (rules Clicks 4-5): identical to process_partition. ---
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

        // --- Assists (rules Clicks 7): identical to process_partition. ---
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
    phase[2] += t_sorted.elapsed().as_secs_f64();
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
    let decode = (std::env::var_os("FAST_DECODE").as_deref() != Some(std::ffi::OsStr::new("0")))
        .then(|| ts::DecodeCache::new(window_start, window_end));
    if decode.is_none() {
        ts::set_window_hint(window_start, window_end);
    }

    let link_maps = map_all(&csv_files(&data.join("links")))?;
    let event_maps = map_all(&csv_files(&data.join("events")))?;
    let click_maps = map_all(&csv_files(&data.join("clicks")))?;
    let products_bytes = std::fs::read(data.join("products.csv"))?;
    let campaigns_bytes = std::fs::read(data.join("campaigns.csv"))?;

    // Lookups (products, campaigns) do not depend on L, so they run beside
    // it on the pool (FAST_OVERLAP_LOOKUPS=0: in sequence, the old order).
    // A width-guard failure comes back as Err and trips the fallback after.
    let lookups = || -> Result<_, String> {
    let mut fields = Vec::with_capacity(8);
        // Categories: byte-sorted unique names incl. `unknown`, so u8 index order
        // is byte order (top_category tie-break). If the file itself uses the name
        // `unknown` it is the same category, as in v1/v2's interner.
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
            return Err(format!("{} categories do not fit u8", cat_names.len()));
        }
        let cat_index = |name: &[u8]| cat_names.binary_search(&name).expect("category listed") as u8;
        let unknown_cat = cat_index(b"unknown");
        let mut product_cat = FastMap::default();
        let mut dense_products = if decode.is_some() {
            vec![unknown_cat; NUMBERED_IDS]
        } else {
            Vec::new()
        };
        for &(product, category) in &product_rows {
            // Later duplicate rows overwrite earlier ones, as in v2.
            let category = cat_index(category);
            if let Some(index) = decode.as_ref().and_then(|_| numbered_id(product, b"p")) {
                dense_products[index] = category;
            } else {
                product_cat.insert(product, category);
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
        if real_campaign_count > 65_535 {
            return Err(format!("{real_campaign_count} campaigns do not fit u16"));
        }
        let mut dense_campaigns = if decode.is_some() {
            vec![u16::MAX; NUMBERED_IDS]
        } else {
            Vec::new()
        };
        if decode.is_some() {
            for campaign in 0..real_campaign_count {
                if let Some(index) = numbered_id(campaigns.bytes(campaign), b"c_") {
                    dense_campaigns[index] = campaign as u16;
                }
            }
        }
        let organic = campaigns.intern(b"organic");
        let organic_channel = channels.intern(b"organic");
        if organic as usize >= campaign_channel.len() {
            campaign_channel.resize(organic as usize + 1, 0);
        }
        campaign_channel[organic as usize] = organic_channel;
        Ok((cat_names, unknown_cat, product_cat, dense_products, campaigns, channels,
            campaign_channel, real_campaign_count, dense_campaigns, organic))
    };
    stages.mark("lookups + mmap");

    let parse_level = std::env::var("FAST_PARSE_LEVEL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    let raw_event_setting = std::env::var_os("FAST_RAW_EVENT_HEX");
    let raw_hex = match raw_event_setting.as_deref().and_then(std::ffi::OsStr::to_str) {
        Some("0") => false,
        Some("1") => true, // Also profile the full pipeline's event keys at levels 2–5.
        _ => parse_level == 6,
    };
    // FAST_PERSON=0 restores the original shared ID interner and person map.
    let fast_person = std::env::var_os("FAST_PERSON").as_deref() != Some(std::ffi::OsStr::new("0"));

    // --- L: links ---
    let compact_hex = raw_hex && std::env::var_os("FAST_COMPACT_HEX_TABLE").as_deref()
        != Some(std::ffi::OsStr::new("0"));
    let ids = if compact_hex {
        // Source-backed event IDs leave mainly users and clicks in this table.
        // Full regions still use the exact overflow interner.
        IdInterner::with_hex_log2(23)
    } else {
        IdInterner::new()
    };
    let link_files: Vec<&[u8]> = link_maps.iter().map(|map| &map[..]).collect();
    let compact_user_ids = if fast_person {
        // Isolate users from event/click IDs so the user hex working set can
        // be sized for repeated IDs. Full regions still intern exactly.
        let bytes: usize = event_maps
            .iter()
            .chain(&click_maps)
            .chain(&link_maps)
            .map(|map| map.len())
            .sum();
        let slots = (bytes / 4096).clamp(64, 1 << 28).next_power_of_two();
        Some(IdInterner::with_hex_log2(slots.ilog2()))
    } else {
        None
    };
    let user_ids = compact_user_ids.as_ref().unwrap_or(&ids);
    let links = || {
        if fast_person {
            PersonLookup::Aliases(links2::build_aliases(&link_files, user_ids))
        } else {
            PersonLookup::Legacy(links2::build(&link_files, user_ids))
        }
    };
    let overlap = std::env::var_os("FAST_OVERLAP_LOOKUPS").as_deref() != Some(std::ffi::OsStr::new("0"));
    let (looked_up, people) = if overlap {
        rayon::join(lookups, links)
    } else {
        let looked_up = lookups();
        (looked_up, links())
    };
    let (cat_names, unknown_cat, product_cat, dense_products, campaigns, channels,
        campaign_channel, real_campaign_count, dense_campaigns, organic) = match looked_up {
        Ok(tables) => tables,
        Err(reason) => {
            fallback.trip(reason);
            return Ok(());
        }
    };
    stages.mark("L links");

    // --- P: one parallel pass over all event and click chunks ---
    let event_files: Vec<&[u8]> = event_maps.iter().map(|map| &map[..]).collect();
    let click_files: Vec<&[u8]> = click_maps.iter().map(|map| &map[..]).collect();
    let chunk_bytes = chunk_bytes_setting();
    let event_chunks = chunk_files_sized(&event_files, chunk_bytes);
    let click_chunks = chunk_files_sized(&click_files, chunk_bytes);
    let event_bytes: u64 = event_files.iter().map(|f| f.len() as u64).sum();
    let mut parts = Parts::for_input(event_bytes);
    // FAST_PART_BITS=n (tuning knob): force 2^n partitions.
    if let Some(bits) = std::env::var("FAST_PART_BITS").ok().and_then(|v| v.parse::<u32>().ok()) {
        parts.bits = bits.clamp(MIN_PARTITION_BITS, MAX_PARTITION_BITS);
    }
    let k = rec_index_bits(event_chunks.len());

    let ctx = Ctx {
        window_start,
        window_end,
        base,
        product_cat,
        dense_products,
        unknown_cat,
        campaigns,
        dense_campaigns,
        real_campaign_count,
        decode,
        people: &people,
        ids: &ids,
        user_ids,
        parts,
        k,
        fallback,
        parse_level,
        fast_scan: std::env::var_os("FAST_SCAN").as_deref() != Some(std::ffi::OsStr::new("0")),
    };
    let source_click_ids = ctx.parse_level == 6
        && std::env::var_os("FAST_SOURCE_CLICK_IDS").as_deref() != Some(std::ffi::OsStr::new("0"))
        && u32::try_from(click_chunks.len()).is_ok()
        && click_chunks.iter().all(|chunk| u32::try_from(chunk.bytes.len()).is_ok());

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
    // FAST_TIMING: per-task (thread, start, end) seconds from P start, to
    // report each thread's idle tail.
    let p_start = Instant::now();
    let p_trace = std::sync::Mutex::new(Vec::<(usize, f64, f64, usize)>::new());
    let parsed: Vec<Parsed> = tasks
        .par_iter()
        .with_max_len(1)
        .map(|&(is_event, i, len)| {
            let t0 = stages.enabled.then(|| p_start.elapsed().as_secs_f64());
            let out = if is_event {
                let chunk = match (raw_hex, ctx.decode.is_some(), fast_person) {
                    (true, true, true) => parse_event_chunk::<true, true, true>(&ctx, &event_chunks[i], i),
                    (true, true, false) => parse_event_chunk::<true, true, false>(&ctx, &event_chunks[i], i),
                    (true, false, true) => parse_event_chunk::<true, false, true>(&ctx, &event_chunks[i], i),
                    (true, false, false) => parse_event_chunk::<true, false, false>(&ctx, &event_chunks[i], i),
                    (false, true, true) => parse_event_chunk::<false, true, true>(&ctx, &event_chunks[i], i),
                    (false, true, false) => parse_event_chunk::<false, true, false>(&ctx, &event_chunks[i], i),
                    (false, false, true) => parse_event_chunk::<false, false, true>(&ctx, &event_chunks[i], i),
                    (false, false, false) => parse_event_chunk::<false, false, false>(&ctx, &event_chunks[i], i),
                };
                Parsed::E(i, chunk)
            } else {
                let chunk = match (source_click_ids, ctx.decode.is_some(), fast_person) {
                    (true, true, true) => parse_click_chunk::<true, true, true>(&ctx, &click_chunks[i], i),
                    (true, true, false) => parse_click_chunk::<true, true, false>(&ctx, &click_chunks[i], i),
                    (true, false, true) => parse_click_chunk::<true, false, true>(&ctx, &click_chunks[i], i),
                    (true, false, false) => parse_click_chunk::<true, false, false>(&ctx, &click_chunks[i], i),
                    (false, true, true) => parse_click_chunk::<false, true, true>(&ctx, &click_chunks[i], i),
                    (false, true, false) => parse_click_chunk::<false, true, false>(&ctx, &click_chunks[i], i),
                    (false, false, true) => parse_click_chunk::<false, false, true>(&ctx, &click_chunks[i], i),
                    (false, false, false) => parse_click_chunk::<false, false, false>(&ctx, &click_chunks[i], i),
                };
                Parsed::C(i, chunk)
            };
            if let Some(t0) = t0 {
                let t1 = p_start.elapsed().as_secs_f64();
                p_trace.lock().unwrap().push((rayon::current_thread_index().unwrap_or(0), t0, t1, len));
            }
            out
        })
        .collect();
    if stages.enabled {
        report_trace("P", &p_trace.into_inner().unwrap(), p_start.elapsed().as_secs_f64());
    }
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
    if ctx.parse_level < 6 {
        eprintln!("v3: FAST_PARSE_LEVEL={} profiling run, stopping after P", ctx.parse_level);
        return Ok(());
    }
    if fallback.is_set() {
        return Ok(());
    }

    // --- D: dedup per event partition ---
    let d2 = dedup2_enabled();
    (0..parts.count())
        .into_par_iter()
        .for_each(|partition| {
            if d2 && DEDUP2_SCRATCH.with(|scratch| {
                let mut scratch = scratch.borrow_mut();
                if raw_hex {
                    dedup_compact::<true>(&ev_chunks, &event_chunks, partition, k, &mut scratch, DEDUP_SCRATCH_BYTES)
                } else {
                    dedup_compact::<false>(&ev_chunks, &[], partition, k, &mut scratch, DEDUP_SCRATCH_BYTES)
                }
            }) {
                return;
            }
            if raw_hex {
                let handled = DEDUP_SCRATCH.with(|scratch| {
                    dedup_stream_inner::<true>(&ev_chunks, &event_chunks, partition, k,
                        &mut scratch.borrow_mut(), DEDUP_SCRATCH_BYTES)
                });
                if !handled {
                    dedup_raw_sorted(&ev_chunks, &event_chunks, partition, k);
                }
            } else {
                dedup_partition(&ev_chunks, partition, k);
            }
        });
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
        ids: user_ids,
        ascii_users: fast_person,
        cat_names: &cat_names,
        organic,
        click_ids: if source_click_ids {
            ClickIds::Source(&click_chunks)
        } else {
            ClickIds::Interned { ids: &ids, any_fallback: ck_chunks.iter().any(|c| c.has_fallback_click) }
        },
    };
    let campaign_count = ctx.campaigns.len();
    let next = AtomicUsize::new(0);
    let workers = rayon::current_num_threads().max(1);
    let s2 = s2_enabled();
    let fast_walk = s2 && walk_enabled();
    let user_dir = output.join("user_daily");
    // S2: user_daily part files are written by the S workers (see below), so
    // the folder must exist before S. Safe: no fallback check point can trip
    // after P (all `fallback.trip` calls are in P and earlier), and P's check
    // point is above, so v3 is committed to writing output by now.
    if s2 {
        std::fs::create_dir_all(&user_dir)?;
    }
    // Category index -> output string (as `top_category` renders it).
    let cat_strs: Vec<Cow<str>> = cat_names.iter().map(|n| String::from_utf8_lossy(n)).collect();
    // S2 flush size: a worker encodes a Parquet part as soon as it holds this
    // many rows, overlapping the encode with other workers' person work.
    let flush_rows: usize = std::env::var("FAST_S2_FLUSH")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or(65_536);
    let next_part = AtomicUsize::new(0);
    // Encode one part from `cols` (leaving it empty) under a fresh part number.
    let flush = |cols: &mut out::UserDailyCols, sums: &mut (i64, i64), encode: &mut f64| -> Result<(), String> {
        let started = Instant::now();
        let (sessions, revenue) = cols.sums();
        sums.0 += sessions;
        sums.1 += revenue;
        let part = next_part.fetch_add(1, Ordering::Relaxed);
        let part_cols = std::mem::replace(cols, out::UserDailyCols::for_walk(fast_walk));
        let result = out::write_user_daily_cols(&user_dir, part, part_cols, &cat_strs);
        *encode += started.elapsed().as_secs_f64();
        result.map_err(|e| e.to_string())
    };
    // Each S2 worker returns (sessions, revenue) summed over its rows, for
    // the cross-table invariant, and its first write error if any.
    type S2Out = ((i64, i64), f64, Result<(), String>);
    let s2_out: std::sync::Mutex<Vec<S2Out>> = std::sync::Mutex::new(Vec::new());
    // Each worker drains the shared queue; it owns its rows and campaign grid.
    type WorkerOut<'r> = (Vec<UserDailyRow<'r>>, CampaignDays, Vec<f64>, [f64; 5]);
    let s_start = Instant::now();
    let trace_s = stages.enabled;
    let s_trace = std::sync::Mutex::new(Vec::<(usize, f64, f64, usize)>::new());
    let worker_out: Vec<WorkerOut> = (0..workers)
        .into_par_iter()
        .with_max_len(1)
        .map(|_| {
            let mut rows = Vec::new();
            let mut days = CampaignDays::new(campaign_count, window_start, window_end, fast_walk);
            let mut scratch = Scratch::new(real_campaign_count);
            let mut times = Vec::new();
            let mut cols = out::UserDailyCols::for_walk(fast_walk);
            let mut sums = (0i64, 0i64);
            let mut encode = 0.0;
            let mut result: Result<(), String> = Ok(());
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= order.len() {
                    break;
                }
                let started = Instant::now();
                let t0 = s_start.elapsed().as_secs_f64();
                if s2 {
                    process_partition2(order[i].0, &ev_chunks, &ck_chunks, &person_ctx, &mut scratch, &mut days, &mut cols, fast_walk);
                    if result.is_ok() && (cols.len() >= flush_rows || cols.name_bytes() >= 1 << 30) {
                        result = flush(&mut cols, &mut sums, &mut encode);
                    }
                } else {
                    process_partition(order[i].0, &ev_chunks, &ck_chunks, &person_ctx, &mut scratch, &mut days, &mut rows);
                }
                times.push(started.elapsed().as_secs_f64());
                if trace_s {
                    s_trace.lock().unwrap().push((rayon::current_thread_index().unwrap_or(0), t0, s_start.elapsed().as_secs_f64(), order[i].1));
                }
            }
            if trace_s && s2 {
                // Account the final flush as a task of size 0.
                let t0 = s_start.elapsed().as_secs_f64();
                if result.is_ok() && !cols.is_empty() {
                    result = flush(&mut cols, &mut sums, &mut encode);
                }
                s_trace.lock().unwrap().push((rayon::current_thread_index().unwrap_or(0), t0, s_start.elapsed().as_secs_f64(), 0));
            }
            if s2 {
                if result.is_ok() && !cols.is_empty() {
                    result = flush(&mut cols, &mut sums, &mut encode);
                }
                s2_out.lock().unwrap().push((sums, encode, result));
            }
            (rows, days, times, scratch.phase)
        })
        .collect();
    let mut user_parts = Vec::with_capacity(worker_out.len());
    let mut campaign_days = CampaignDays::new(campaign_count, window_start, window_end, fast_walk);
    let mut part_times = Vec::new();
    let mut phase = [0.0f64; 5];
    for (rows, days, times, worker_phase) in worker_out {
        user_parts.push(rows);
        campaign_days = campaign_days.merge(days);
        part_times.extend(times);
        for (total, add) in phase.iter_mut().zip(worker_phase) {
            *total += add;
        }
    }
    if stages.enabled {
        report_trace("S", &s_trace.into_inner().unwrap(), s_start.elapsed().as_secs_f64());
    }
    let s2_out = s2_out.into_inner().unwrap();
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
            "  v3   S phases per thread: gather {:.4}s sort {:.4}s walk {:.4}s (sort = group {:.4}s + per-person {:.4}s)",
            phase[0] / w,
            phase[1] / w,
            phase[2] / w, phase[3] / w, phase[4] / w
        );
        if s2 {
            let encode = s2_out.iter().map(|(_, time, _)| time).sum::<f64>() / w;
            eprintln!(
                "  v3   S encode per thread: {:.4}s (walk + encode {:.4}s)",
                encode, phase[2] / w + encode
            );
        }
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
    let (mut user_sessions, mut user_revenue) = (0i64, 0i64);
    for ((sessions, revenue), _, _) in &s2_out {
        user_sessions += sessions;
        user_revenue += revenue;
    }
    user_sessions += user_parts.iter().flatten().map(|row| row.sessions).sum::<i64>();
    user_revenue += user_parts.iter().flatten().map(|row| row.revenue_cents).sum::<i64>();
    let campaign_sessions: i64 = campaign_rows.iter().map(|row| row.sessions).sum();
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

    if s2 {
        for (_, _, result) in s2_out {
            result?;
        }
        // Part 0 is always written (possibly empty) so the folder always
        // holds a Parquet file.
        if next_part.load(Ordering::Relaxed) == 0 {
            out::write_user_daily_cols(&user_dir, 0, out::UserDailyCols::for_walk(fast_walk), &cat_strs)?;
        }
        out::write_campaign_daily(&output.join("campaign_daily"), &campaign_rows)?;
        stages.mark("O output");
        if stages.enabled {
            print_rss();
        }
        if par_exit_enabled() {
            // Process exit frees every page on one thread (exit_mmap). Hand
            // the biggest buffers back to the kernel from all workers first,
            // so the serial teardown after `done` is short. Output is fully
            // written; these buffers are forgotten below and never read again.
            let mut ranges: Vec<(usize, usize)> = Vec::new();
            for c in &ev_chunks {
                ranges.push(byte_range(&c.evs));
                ranges.push(byte_range(&c.toks));
            }
            for c in &ck_chunks {
                ranges.push(byte_range(&c.cks));
            }
            // Input file mappings: unmapped in parallel too. The Mmap values
            // (and everything borrowing them) are forgotten below, never used.
            let maps: Vec<(usize, usize)> =
                event_maps.iter().chain(&click_maps).chain(&link_maps).map(|m| byte_range(&m[..])).collect();
            rayon::join(
                || ranges.par_iter().with_max_len(1).for_each(|&(ptr, len)| release_pages(ptr, len)),
                || {
                    maps.par_iter().with_max_len(1).for_each(|&(ptr, len)| {
                        if len > 0 {
                            // SAFETY: a whole mapping created by map_all; no
                            // reference into it is dereferenced after this.
                            unsafe { libc::munmap(ptr as *mut libc::c_void, len) };
                        }
                    })
                },
            );
            stages.mark("X release");
            if stages.enabled {
                print_rss();
            }
        }
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
        return Ok(());
    }
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
    std::mem::forget(compact_user_ids);
    std::mem::forget(ids);
    std::mem::forget(event_maps);
    std::mem::forget(click_maps);
    std::mem::forget(link_maps);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn person_paths_agree_through_scatter_dedup_and_click_ties() {
        let links = b"a,b\n\
\" aa root \",\"u12345678\"\n\
u12345678,u00000000000000000000000000000000\n\
u12345678,zzzzzzzzzz\n\
u12345678,\"p123\r5678\"\n\
u12345678,u12345678\n\
u12345678,\"\"\n\
u12345678,bad\"quote\n\
u12345678,wouldlink,extra\n";
        let mut events = b"event,user,ts,ingest,type,product,price\n\
e1,\"u12345678\",2026-10-01T00:00:00Z,2026-10-01T00:00:00Z,view,p,\n\
dup,\"\",2026-10-01T00:00:00Z,2026-10-01T00:00:00Z,view,p,\n\
e2,u00000000000000000000000000000000,2026-10-01T00:00:00Z,2026-10-01T00:00:00Z,purchase,p,\"0250\"\n"
            .to_vec();
        // A malformed line straddles the byte target. The following chunk
        // starts with a real row, and the invalid dedup winner is in chunk 0.
        events.resize(CHUNK_BYTES + 11, b'x');
        events.extend_from_slice(b"\n\
dup,q00000000,2026-10-01T00:00:00Z,2026-10-01T00:00:01Z,view,p,\n\
e3,zzzzzzzzzz,2026-10-01T00:00:00Z,2026-10-01T00:00:00Z,view,p,\n\
e4,\"p123\r5678\",2026-10-01T00:00:00Z,2026-10-01T00:00:00Z,cart,p,\n\
e5,q00000000,2026-10-01T00:00:00Z,2026-10-01T00:00:00Z,view,p,\r\n\
bad-price,u12345678,2026-10-01T00:00:00Z,2026-10-01T00:00:00Z,purchase,p,10\r\n\
bad\"quote,q00000000,2026-10-01T00:00:00Z,2026-10-01T00:00:00Z,view,p,\n\
extra,q00000000,2026-10-01T00:00:00Z,2026-10-01T00:00:00Z,view,p,,extra");
        let event_input = chunk_files(&[&events]);
        assert_eq!(event_input.len(), 2);
        let click_input = [
            Chunk {
                bytes: b"click,user,campaign,ts\n\
z00000000000000000000000000000000,u12345678,c2,2026-10-01T00:00:00Z\n\
bad,q00000000,c1,2026-10-01T00:00:00Z\r\n",
                has_header: true,
            },
            Chunk {
                bytes: b"\"A generic click\",u00000000000000000000000000000000,c1,2026-10-01T00:00:00Z\n\
empty,\"\",c1,2026-10-01T00:00:00Z",
                has_header: false,
            },
        ];

        let run = |ascii_users| {
            let ids = IdInterner::with_hex_log2(6);
            let compact_users = IdInterner::with_hex_log2(6);
            let user_ids = if ascii_users { &compact_users } else { &ids };
            let people = if ascii_users {
                PersonLookup::Aliases(links2::build_aliases(&[links], user_ids))
            } else {
                PersonLookup::Legacy(links2::build(&[links], user_ids))
            };
            let window_start = ts::parse_ts(b"2026-10-01T00:00:00Z").unwrap();
            let window_end = window_start + MS_PER_DAY;
            let mut product_cat = FastMap::default();
            product_cat.insert(b"p".as_slice(), 0);
            let mut campaigns = Interner::new();
            campaigns.intern(b"c1");
            campaigns.intern(b"c2");
            let organic = campaigns.intern(b"organic");
            let fallback = Fallback::default();
            let ctx = Ctx {
                window_start,
                window_end,
                base: window_start - SEVEN_DAYS_MS,
                product_cat,
                // No canonical p<n> / c_<n> IDs here: dense tables say "absent".
                dense_products: vec![1; NUMBERED_IDS],
                unknown_cat: 1,
                campaigns,
                dense_campaigns: vec![u16::MAX; NUMBERED_IDS],
                real_campaign_count: 2,
                // This test isolates the person paths; it keeps the old decoder.
                decode: None,
                people: &people,
                ids: &ids,
                user_ids,
                parts: Parts { bits: MIN_PARTITION_BITS },
                k: rec_index_bits(event_input.len()),
                fallback: &fallback,
                parse_level: 6,
                fast_scan: true,
            };
            let evs: Vec<_> = event_input.iter().enumerate()
                .map(|(i, chunk)| if ascii_users {
                    parse_event_chunk::<true, false, true>(&ctx, chunk, i)
                } else {
                    parse_event_chunk::<true, false, false>(&ctx, chunk, i)
                })
                .collect();
            let cks: Vec<_> = click_input.iter().enumerate()
                .map(|(i, chunk)| if ascii_users {
                    parse_click_chunk::<true, false, true>(&ctx, chunk, i)
                } else {
                    parse_click_chunk::<true, false, false>(&ctx, chunk, i)
                })
                .collect();
            assert!(!fallback.is_set());
            let pc = PersonCtx {
                base: ctx.base,
                ids: user_ids,
                ascii_users,
                cat_names: &[b"cat", b"unknown"],
                organic,
                click_ids: ClickIds::Source(&click_input),
            };
            let mut rows = Vec::new();
            let mut days = CampaignDays::new(3, window_start, window_end, false);
            let mut scratch = Scratch::new(2);
            let compact = reset(&evs);
            let mut compact_table = Vec::new();
            for part in 0..ctx.parts.count() {
                dedup_raw_sorted(&evs, &event_input, part, ctx.k);
                assert!(dedup_compact::<true>(
                    &compact, &event_input, part, ctx.k, &mut compact_table, DEDUP_SCRATCH_BYTES,
                ));
            }
            assert_eq!(dead_bits(&compact), dead_bits(&evs));
            for part in 0..ctx.parts.count() {
                process_partition(part, &evs, &cks, &pc, &mut scratch, &mut days, &mut rows);
            }
            let mut users: Vec<_> = rows.into_iter().map(|r| (
                r.user_id.into_owned(), r.sessions, r.events, r.revenue_cents,
                r.top_category.map(Cow::into_owned),
            )).collect();
            users.sort_unstable();
            let mut credits = Vec::new();
            days.for_each_row(|id, _, day| {
                credits.push((id, day.sessions, day.revenue, day.assisted_sessions));
            });
            credits.sort_unstable();
            (users, credits)
        };
        let old = run(false);
        assert_eq!(run(true), old);
        assert_eq!(old.0, vec![
            (" aa root ".into(), 1, 4, 250, Some("cat".into())),
            ("q00000000".into(), 1, 1, 0, None),
        ]);
        assert_eq!(old.1, vec![(0, 1, 250, 1), (1, 0, 0, 1), (2, 1, 0, 0)]);
    }

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
    fn short_event_keys_are_exact_on_ascii_and_keep_widths_distinct() {
        let mut seen = std::collections::HashMap::new();
        for len in [8, 9] {
            let mut id = *b"e01234567";
            for pos in 1..len {
                let original = id[pos];
                for byte in 0..=255 {
                    id[pos] = byte;
                    let key = short_event_id(&id[..len]);
                    assert_eq!(key.is_some(), len == 8 || byte.is_ascii());
                    if let Some(key) = key {
                        assert_eq!(key & 15, SHORT_EVENT_TAG);
                        let bytes = id[..len].to_vec();
                        if let Some(previous) = seen.insert(key, bytes.clone()) {
                            assert_eq!(previous, bytes);
                        }
                    }
                }
                id[pos] = original;
            }
            id[0] = b'E';
            assert!(short_event_id(&id[..len]).is_none());
        }
    }

    #[test]
    fn fast_prices_and_numbered_ids_match_exact_bytes() {
        let oracle = |p: &[u8]| {
            parse_digits(p)
                .filter(|n| (1..=10_000_000).contains(n))
                .map(|n| n as u32)
        };
        for price in [
            b"".as_slice(),
            b"0",
            b"1",
            b"0099",
            b"10000000",
            b"10000001",
            b"000000000000010000000",
            b"99999999999999999999999",
            b"250\r",
            b"+250",
            b"25.0",
            "２５０".as_bytes(),
            "٢٥٠".as_bytes(),
        ] {
            assert_eq!(purchase_price(price), oracle(price), "{price:?}");
        }
        for len in 1..=8 {
            let mut price = *b"12345678";
            for i in 0..len {
                let original = price[i];
                for byte in 0..=255 {
                    price[i] = byte;
                    assert_eq!(purchase_price(&price[..len]), oracle(&price[..len]), "{price:?}");
                }
                price[i] = original;
            }
        }
        for n in 1..NUMBERED_IDS {
            let product = format!("p{n}");
            assert_eq!(numbered_id(product.as_bytes(), b"p"), Some(n));
        }
        for id in [
            b"p".as_slice(),
            b"p0",
            b"p01",
            b"p100000",
            b"P1",
            b"p1\r",
            b"p 1",
            "p１".as_bytes(),
        ] {
            assert_eq!(numbered_id(id, b"p"), None, "{id:?}");
        }
    }

    #[test]
    fn decode_modes_match_events_and_clicks_at_chunk_edges() {
        let begin = ts::parse_ts(b"2028-02-29T00:00:00Z").unwrap();
        let end = ts::parse_ts(b"2028-03-01T00:00:00Z").unwrap();
        let ids = IdInterner::with_hex_log2(6);
        let people = PersonLookup::Legacy(links2::build(&[], &ids));
        let fallback = Fallback::default();
        let mut campaigns = Interner::new();
        let one = campaigns.intern(b"c_1");
        campaigns.intern(b"c_01"); // Distinct from the canonical spelling.
        campaigns.intern(b"organic");
        let mut dense_campaigns = vec![u16::MAX; NUMBERED_IDS];
        dense_campaigns[1] = one as u16;
        let mut dense_products = vec![2; NUMBERED_IDS];
        dense_products[1] = 0;
        let ctx = Ctx {
            window_start: begin,
            window_end: end,
            base: begin - SEVEN_DAYS_MS,
            product_cat: [(b"p1".as_slice(), 0), (b"p01".as_slice(), 1)].into_iter().collect(),
            dense_products,
            unknown_cat: 2,
            campaigns,
            dense_campaigns,
            real_campaign_count: 2,
            decode: Some(ts::DecodeCache::new(begin, end)),
            people: &people,  // wrapped as PersonLookup below
            user_ids: &ids,
            ids: &ids,
            parts: Parts { bits: 4 },
            k: K,
            fallback: &fallback,
            parse_level: 6,
            fast_scan: true,
        };
        let mut input = b"header\n\
            e12345678,u,2028-02-29T00:00:00Z,2028-02-29T00:00:00Z,purchase,p1,\n\
            E00000000000000000000000000000000,u,2028-02-29T00:00:00Z,2028-03-01T00:00:00Z,purchase,p01,0250\n\
            e00000000000000000000000000000000,u,2028-02-29T00:00:00Z,2028-03-01T00:00:01Z,view,p1,ignored\n\
            e76543210,u,2028-02-29T00:00:00Z,2028-02-29T00:00:01Z,purchase,p1,250\r\n\
            e76543211,u,2028-02-29T00:00:00Z,2028-02-29T00:00:01Z,cart,p1,250\r\n\
            e76543212,,2028-02-29T00:00:00Z,2028-02-29T00:00:01Z,view,p1,\n\
            e76543213,u,2028-02-29T00:00:00Z,2028-02-29T00:00:01Z,view,,\n\
            e76543214,u,2028-03-01T00:00:00Z,2028-02-29T00:00:01Z,view,p1,\n\
            e76543215,u,2028-02-29T00:00:00Z,2028-02-29T00:00:01Z,View,p1,\n\
            e76543216,u,2028-02-29T00:00:00Z,2028-02-29T00:00:01Z,view,p1,,extra\n"
            .to_vec();
        // The malformed padding straddles the target boundary. The following
        // record becomes the first record of the second chunk, without a header.
        input.resize(CHUNK_BYTES + 4, b'x');
        input.extend_from_slice(
            b"\n\
            \"e12345678\",u,2028-02-29T00:00:00Z,2028-02-29T00:00:01Z,view,p1,\n\
            \"E00000000000000000000000000000000\",u,2028-02-29T00:00:00Z,2028-03-01T00:00:00Z,purchase,p01,0250\n\
            bad\"quote,u,2028-02-29T00:00:00Z,2028-02-29T00:00:00Z,view,p1,\n\
            ebcdefghi,\"u\",2028-02-29T00:00:00-00:00,2028-02-28T23:59:59Z,purchase,p99999,000000000500",
        );
        let chunks = chunk_files(&[&input]);
        assert_eq!(chunks.len(), 2);
        let parse_events = |fast| {
            chunks
                .iter()
                .enumerate()
                .map(|(i, chunk)| {
                    if fast {
                        parse_event_chunk::<true, true, false>(&ctx, chunk, i)
                    } else {
                        parse_event_chunk::<true, false, false>(&ctx, chunk, i)
                    }
                })
                .collect::<Vec<_>>()
        };
        let winners = |fast, sorted| {
            let parsed = parse_events(fast);
            let mut scratch = DedupScratch::default();
            for part in 0..ctx.parts.count() {
                if sorted {
                    dedup_raw_sorted(&parsed, &chunks, part, ctx.k);
                } else {
                    assert!(dedup_stream_inner::<true>(
                        &parsed,
                        &chunks,
                        part,
                        ctx.k,
                        &mut scratch,
                        DEDUP_SCRATCH_BYTES
                    ));
                }
            }
            let mut keys = Vec::new();
            for (i, chunk) in parsed.iter().enumerate() {
                for (j, event) in chunk.evs.iter().enumerate() {
                    if !dead(&parsed, i, j) {
                        keys.push((event.person, event.ts_rel, event.pc));
                    }
                }
            }
            keys.sort_unstable();
            keys
        };
        let old = winners(false, false);
        assert_eq!(old.len(), 3);
        assert_eq!(
            old.iter().map(|row| row.2).collect::<Vec<_>>(),
            [0, (250 << 8) | 1, (500 << 8) | 2]
        );
        assert_eq!(winners(true, false), old);
        assert_eq!(winners(true, true), old);
        assert!(!fallback.is_set());

        let clicks = [Chunk {
            has_header: true,
            bytes: b"header\n\
                \"z\",u,c_1,2028-02-22T00:00:00Z\n\
                \"a\",\"u\",\"c_01\",2028-02-22T00:00:00-00:00\n\
                a,u,c_01,2028-02-22T00:00:00Z\n\
                early,u,c_1,2028-02-21T23:59:59Z\n\
                end,u,c_1,2028-03-01T00:00:00Z\n\
                cr,u,c_1,2028-02-22T00:00:00Z\r\n\
                extra,u,c_1,2028-02-22T00:00:00Z,\n\
                ,u,c_1,2028-02-22T00:00:00Z\n\
                no-user,,c_1,2028-02-22T00:00:00Z\n\
                hole,u,c_2,2028-02-22T00:00:00Z\n\
                organic,u,organic,2028-02-22T00:00:00Z",
        }];
        let mut new = parse_click_chunk::<true, true, false>(&ctx, &clicks[0], 0).cks;
        let old = parse_click_chunk::<true, false, false>(&ctx, &clicks[0], 0).cks;
        assert_eq!(new, old);
        sort_clicks_with_source_ids(&mut new, &clicks);
        // The source sort places the winner first within each timestamp tie.
        assert_eq!(new[0].camp, 1);
        assert_eq!(new.iter().filter(|click| click.camp == 1).count(), 2);
        assert!(new.iter().all(|click| click.ts_rel == 0));
    }

    #[test]
    fn raw_event_fingerprint_collisions_preserve_exact_winners_in_both_kernels() {
        let raw_a = crate::hexid::render(b'e', 7);
        let raw_b = *b"E00000000000000000000000000000007";
        let mut bytes = b"id,user,ts,ingest,type,product,price\n".to_vec();
        bytes.push(b'"');
        bytes.extend_from_slice(&raw_a);
        bytes.extend_from_slice(b"\",u,t,i,view,p,\n\"malformed,u,t,i,view,p,\n");
        bytes.extend_from_slice(&raw_b);
        bytes.extend_from_slice(b",u,t,i,view,p,");
        let input = [Chunk { bytes: &bytes, has_header: true }; 2];
        let scanned_keys = |fast| {
            let mut keys = Vec::new();
            crate::scan3::for_each_record::<7>(&bytes, true, fast, |fields| {
                let offset = fields[0].as_ptr() as usize - bytes.as_ptr() as usize;
                // Force a collision in the routing fingerprint. Dedup must
                // read the full source ID at the scanner's unquoted offset.
                keys.push((11 << (RAW_EVENT_OFFSET_BITS + 4)) | (offset as u64) << 4 | RAW_EVENT_TAG);
            });
            keys
        };
        let keys = scanned_keys(true);
        assert_eq!(keys, scanned_keys(false));
        let [key_a, key_b] = keys[..] else { panic!("expected two well-formed events") };
        assert_eq!(source_event_id(&bytes, key_a >> 4), raw_a);
        assert_eq!(source_event_id(&bytes, key_b >> 4), raw_b);
        for kernel in 0..3 {
            let mut chunks = [chunk_with(1), chunk_with(2)];
            chunks[0].toks = vec![
                Tok { ekey: key_a, ingest_s: 1, rec: NONE },
                Tok { ekey: key_b, ingest_s: 2, rec: rec(0, 0) },
            ];
            chunks[1].toks = vec![
                Tok { ekey: key_a, ingest_s: 3, rec: rec(1, 0) },
                Tok { ekey: key_b, ingest_s: 4, rec: rec(1, 1) },
            ];
            for chunk in &mut chunks {
                chunk.tok_off = vec![0, 2];
            }
            match kernel {
                0 => assert!(dedup_stream_inner::<true>(
                    &chunks, &input, 0, K, &mut DedupScratch::default(), DEDUP_SCRATCH_BYTES,
                )),
                1 => dedup_raw_sorted(&chunks, &input, 0, K),
                _ => assert!(dedup_compact::<true>(
                    &chunks, &input, 0, K, &mut Vec::new(), DEDUP_SCRATCH_BYTES,
                )),
            }
            assert!(!dead(&chunks, 0, 0));
            assert!(dead(&chunks, 1, 0));
            assert!(dead(&chunks, 1, 1));
        }
    }

    #[test]
    fn raw_event_kernels_match_exact_oracle_with_collisions_and_epoch_wrap() {
        let ids = IdInterner::new();
        let mut rng = Rng(0x0e71_8501_1275_d4ac);
        let mut scratch = DedupScratch::default();
        let mut compact_scratch = Vec::new();
        for _ in 0..100 {
            let mut input_bytes = Vec::new();
            let mut raw = Vec::new();
            let mut exact = Vec::new();
            for chunk_index in 0..5 {
                // The same normalized ID appears at different offsets and
                // sometimes has a surrounding quote in the source.
                let mut bytes = vec![b'\n'; chunk_index + 1];
                let mut raw_chunk = chunk_with(85);
                let mut exact_chunk = chunk_with(85);
                for row in 0..85 {
                    let number = rng.below(24);
                    let prefix = if number % 2 == 0 { b'e' } else { b'u' };
                    let rendered = crate::hexid::render(prefix, u128::from(number));
                    let id: &[u8] = match number % 6 {
                        0 => b"short-id",
                        1 => b"E00000000000000000000000000000005",
                        _ => &rendered,
                    };
                    let key = key_of(id, &ids);
                    let raw_key = if crate::hexid::try_parse(id).is_some() {
                        let quoted = rng.below(2) == 0;
                        if quoted {
                            bytes.push(b'"');
                        }
                        let offset = bytes.len() as u64;
                        bytes.extend_from_slice(id);
                        if quoted {
                            bytes.push(b'"');
                        }
                        (u64::from(number % 3) << (RAW_EVENT_OFFSET_BITS + 4))
                            | (offset << 4) | RAW_EVENT_TAG
                    } else {
                        key
                    };
                    let rec = if rng.below(5) == 0 { NONE } else { rec(chunk_index as u32, row) };
                    let ingest_s = rng.below(90) as u32;
                    let tok = Tok { ekey: key, ingest_s, rec };
                    exact_chunk.toks.push(tok);
                    raw_chunk.toks.push(Tok { ekey: raw_key, ..tok });
                }
                raw_chunk.tok_off = vec![0, 85];
                exact_chunk.tok_off = vec![0, 85];
                input_bytes.push(bytes);
                raw.push(raw_chunk);
                exact.push(exact_chunk);
            }
            let input: Vec<_> = input_bytes.iter()
                .map(|bytes| Chunk { bytes, has_header: false })
                .collect();
            dedup_partition_sorted(&exact, 0, K);
            let sorted = reset(&raw);
            assert!(!dedup_stream_inner::<true>(&sorted, &input, 0, K, &mut scratch, 1));
            assert!(!dedup_compact::<true>(&sorted, &input, 0, K, &mut compact_scratch, 1));
            assert!(dead_bits(&sorted).iter().flatten().all(|&word| word == 0));
            dedup_raw_sorted(&sorted, &input, 0, K);
            let compact = reset(&raw);
            assert!(dedup_compact::<true>(
                &compact, &input, 0, K, &mut compact_scratch, DEDUP_SCRATCH_BYTES,
            ));
            scratch.epoch = u32::MAX;
            assert!(dedup_stream_inner::<true>(
                &raw, &input, 0, K, &mut scratch, DEDUP_SCRATCH_BYTES,
            ));
            assert_eq!(dead_bits(&raw), dead_bits(&exact));
            assert_eq!(dead_bits(&sorted), dead_bits(&exact));
            assert_eq!(dead_bits(&compact), dead_bits(&exact));
        }
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

    /// Tiny xorshift64* for the randomized dedup tests (no rand dependency).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Random event chunks with `parts` event partitions. Event keys come from
    /// a small pool (forcing many copies per event, including extreme values),
    /// ingest from a tiny range (many ties), ~20% of copies are `NONE`, and
    /// every valid copy gets its own unique rec. A key's partition is a fixed
    /// function of the key, as in the pipeline.
    fn random_chunks(rng: &mut Rng, parts: usize) -> Vec<EvChunk> {
        let pool_size = 1 + rng.below(40) as usize;
        let pool: Vec<IdKey> = (0..pool_size)
            .map(|i| match i {
                0 => 0,
                1 => u64::MAX,
                2 => 1u64 << 63,
                _ => rng.next(),
            })
            .collect();
        let ingest_span = 1 + rng.below(6) as u32;
        let n_chunks = 1 + rng.below(4) as usize;
        (0..n_chunks)
            .map(|c| {
                let n_toks = rng.below(120) as usize;
                let mut by_part: Vec<Vec<Tok>> = vec![Vec::new(); parts];
                let mut n_evs = 0u32;
                for _ in 0..n_toks {
                    let ekey = pool[rng.below(pool.len() as u64) as usize];
                    let ingest_s = 1_000 + rng.below(u64::from(ingest_span)) as u32;
                    let r = if rng.below(5) == 0 {
                        NONE
                    } else {
                        n_evs += 1;
                        rec(c as u32, n_evs - 1)
                    };
                    by_part[(fmix64(ekey) % parts as u64) as usize].push(Tok { ekey, ingest_s, rec: r });
                }
                // Shuffle within each partition so copy order is arbitrary.
                for toks in &mut by_part {
                    for i in (1..toks.len()).rev() {
                        let j = rng.below(i as u64 + 1) as usize;
                        toks.swap(i, j);
                    }
                }
                let mut tok_off = vec![0u32];
                let mut toks = Vec::new();
                for part in by_part {
                    toks.extend(part);
                    tok_off.push(toks.len() as u32);
                }
                let mut chunk = chunk_with(n_evs as usize);
                chunk.toks = toks;
                chunk.tok_off = tok_off;
                chunk
            })
            .collect()
    }

    /// A fresh copy of `chunks` with every dead bit cleared.
    fn reset(chunks: &[EvChunk]) -> Vec<EvChunk> {
        chunks
            .iter()
            .map(|c| EvChunk {
                evs: c.evs.clone(),
                ev_off: c.ev_off.clone(),
                toks: c.toks.clone(),
                tok_off: c.tok_off.clone(),
                dead: (0..c.dead.len()).map(|_| AtomicU64::new(0)).collect(),
            })
            .collect()
    }

    fn dead_bits(chunks: &[EvChunk]) -> Vec<Vec<u64>> {
        chunks
            .iter()
            .map(|c| c.dead.iter().map(|w| w.load(Ordering::Relaxed)).collect())
            .collect()
    }

    /// Run the sort path and the streaming kernel on the same chunks and
    /// compare the dead bitmaps exactly. Returns the streaming kernel's
    /// "handled" flag for every partition.
    fn diff_against_sorted(chunks: &[EvChunk], parts: usize, scratch: &mut DedupScratch, cap: usize) -> bool {
        let sorted = reset(chunks);
        for p in 0..parts {
            dedup_partition_sorted(&sorted, p, K);
        }
        let streamed = reset(chunks);
        let mut all_streamed = true;
        for p in 0..parts {
            all_streamed &= dedup_stream(&streamed, p, K, scratch, cap);
        }
        assert_eq!(dead_bits(&sorted), dead_bits(&streamed));
        all_streamed
    }

    #[test]
    fn streaming_dedup_matches_sort_path_randomized() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut scratch = DedupScratch::default();
        let mut some_killed = false;
        for _ in 0..3_000 {
            let parts = 1 + rng.below(4) as usize;
            let chunks = random_chunks(&mut rng, parts);
            assert!(diff_against_sorted(&chunks, parts, &mut scratch, DEDUP_SCRATCH_BYTES));
            // Sanity: the generator really produces kills.
            let s = reset(&chunks);
            (0..parts).for_each(|p| dedup_partition_sorted(&s, p, K));
            some_killed |= dead_bits(&s).iter().flatten().any(|&w| w != 0);
        }
        assert!(some_killed);
    }

    #[test]
    fn compact_dedup_matches_sort_with_variable_chunk_slices() {
        let mut rng = Rng(0x435f_1202_47f3);
        let mut scratch = Vec::new();
        for _ in 0..500 {
            let parts = 1 + rng.below(8) as usize;
            let chunks = random_chunks(&mut rng, parts);
            let compact = reset(&chunks);
            for partition in (0..parts).rev() {
                dedup_partition_sorted(&chunks, partition, K);
                assert!(dedup_compact::<false>(
                    &compact, &[], partition, K, &mut scratch, DEDUP_SCRATCH_BYTES,
                ));
            }
            assert_eq!(dead_bits(&compact), dead_bits(&chunks));
        }
    }

    #[test]
    fn streaming_dedup_handles_none_winner_and_ties() {
        // Event 7: a NONE copy strictly earliest kills both valid copies.
        // Event 8: three valid copies tied on ingest - exactly one survives.
        // Event 9: a valid and a NONE copy tied on ingest - the valid one wins.
        let mut chunk = chunk_with(6);
        chunk.toks = vec![
            Tok { ekey: 7, ingest_s: 200, rec: rec(0, 0) },
            Tok { ekey: 8, ingest_s: 50, rec: rec(0, 3) },
            Tok { ekey: 7, ingest_s: 100, rec: NONE },
            Tok { ekey: 8, ingest_s: 50, rec: rec(0, 1) },
            Tok { ekey: 7, ingest_s: 300, rec: rec(0, 2) },
            Tok { ekey: 8, ingest_s: 50, rec: rec(0, 4) },
            Tok { ekey: 9, ingest_s: 10, rec: rec(0, 5) },
            Tok { ekey: 9, ingest_s: 10, rec: NONE },
        ];
        chunk.tok_off = vec![0, chunk.toks.len() as u32];
        let chunks = vec![chunk];
        let mut scratch = DedupScratch::default();
        assert!(dedup_stream(&chunks, 0, K, &mut scratch, DEDUP_SCRATCH_BYTES));
        assert!(dead(&chunks, 0, 0) && dead(&chunks, 0, 2)); // NONE winner kills all
        assert!(!dead(&chunks, 0, 1)); // smallest rec among the ingest tie wins
        assert!(dead(&chunks, 0, 3) && dead(&chunks, 0, 4));
        assert!(!dead(&chunks, 0, 5)); // valid beats NONE on an ingest tie
        assert!(diff_against_sorted(&chunks, 1, &mut scratch, DEDUP_SCRATCH_BYTES));
    }

    #[test]
    fn streaming_dedup_falls_back_before_marking() {
        let mut rng = Rng(42);
        let mut scratch = DedupScratch::default();
        let mut fell_back = false;
        for _ in 0..200 {
            let parts = 1 + rng.below(3) as usize;
            let chunks = random_chunks(&mut rng, parts);
            // Too small for anything but an empty partition.
            let fresh = reset(&chunks);
            for p in 0..parts {
                let n: u32 = fresh.iter().map(|c| c.tok_off[p + 1] - c.tok_off[p]).sum();
                let handled = dedup_stream(&fresh, p, K, &mut scratch, WIN_SLOT_BYTES - 1);
                assert_eq!(handled, n == 0);
                fell_back |= !handled;
            }
            assert!(dead_bits(&fresh).iter().flatten().all(|&w| w == 0), "marked before falling back");
            // The wrapper with the tiny cap must still give the exact answer.
            let sorted = reset(&chunks);
            let wrapped = reset(&chunks);
            for p in 0..parts {
                dedup_partition_sorted(&sorted, p, K);
                dedup_partition_with(&wrapped, p, K, &mut scratch, WIN_SLOT_BYTES - 1);
            }
            assert_eq!(dead_bits(&sorted), dead_bits(&wrapped));
        }
        assert!(fell_back);
    }

    #[test]
    fn streaming_dedup_survives_epoch_wrap() {
        let mut rng = Rng(7);
        let mut scratch = DedupScratch::default();
        // Leave stale stamps at epoch 1 from one dataset ...
        while scratch.epoch == 0 {
            let first = random_chunks(&mut rng, 1);
            diff_against_sorted(&first, 1, &mut scratch, DEDUP_SCRATCH_BYTES);
        }
        assert_eq!(scratch.epoch, 1);
        // ... then wrap: the next live epoch is 1 again, so those stamps must
        // have been cleared or they would read as occupied slots.
        scratch.epoch = u32::MAX - 3;
        for _ in 0..50 {
            let parts = 1 + rng.below(3) as usize;
            let chunks = random_chunks(&mut rng, parts);
            assert!(diff_against_sorted(&chunks, parts, &mut scratch, DEDUP_SCRATCH_BYTES));
        }
        assert!(scratch.epoch < 100, "epoch wrapped");
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
    fn source_click_ids_preserve_tie_winners_and_all_assist_campaigns() {
        for fast_scan in [false, true] {
            check_source_click_ties(fast_scan);
        }
    }

    #[test]
    fn scanners_agree_across_pipeline_chunks() {
        let mut bytes = b"id,user,campaign,ts\n".to_vec();
        bytes.resize(CHUNK_BYTES - 7, b'x');
        bytes.extend_from_slice(b"\n\"edge\",u,c,t\n\"\",u,c,t\n\"bad,u,c,t\nnext,u,c,t\r");
        let chunks = chunk_files(&[&bytes]);
        assert_eq!(chunks.len(), 2);
        let spans = |fast| {
            let mut records = Vec::new();
            for chunk in &chunks {
                crate::scan3::for_each_record::<4>(chunk.bytes, chunk.has_header, fast, |fields| {
                    records.push(fields.map(|field| (field.as_ptr() as usize, field.len())));
                });
            }
            records
        };
        let records = spans(true);
        assert_eq!(records, spans(false));
        assert_eq!(records.len(), 3);
    }

    fn check_source_click_ties(fast_scan: bool) {
        let files: [&[u8]; 2] = [
            b"id,user,campaign,ts\nz,u,1,5\n\"A \"  ,u,99,5\n\"A \"\",u,99,5\n\"B\",u,2,2\n\"Aa\",u,5,5\n",
            b"id,user,campaign,ts\n\"A \",\"u\",3,5\nA ,u,3,5\n0,u,4,2\nvery-long-fallback-key,u,6,7",
        ];
        let chunks = chunk_files(&files);
        let ids = IdInterner::new();
        let mut interned = Vec::new();
        let mut source = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            crate::scan3::for_each_record::<4>(chunk.bytes, chunk.has_header, fast_scan, |fields| {
                let id = fields[0];
                let camp = std::str::from_utf8(fields[2]).unwrap().parse().unwrap();
                let ts_rel = std::str::from_utf8(fields[3]).unwrap().parse().unwrap();
                let offset = id.as_ptr() as usize - chunk.bytes.as_ptr() as usize;
                let click = (i as u64) << 32 | offset as u64;
                assert_eq!(click_source_id(click, &chunks), id);
                source.push(Ck { person: 7, click, ts_rel, camp });
                interned.push(Ck { person: 7, click: key_of(id, &ids), ts_rel, camp });
            });
        }
        assert_eq!(source.len(), 7);
        sort_clicks_with_source_ids(&mut source, &chunks);
        sort_clicks(&mut interned, &ids, true);
        let winners = |clicks: &[Ck]| {
            clicks.chunk_by(|a, b| a.ts_rel == b.ts_rel)
                .map(|run| (run[0].ts_rel, run[0].camp))
                .collect::<Vec<_>>()
        };
        assert_eq!(winners(&source), vec![(2, 4), (5, 3), (7, 6)]);
        assert_eq!(winners(&source), winners(&interned));
        let campaigns = |clicks: &[Ck]| {
            let mut values: Vec<_> = clicks.iter().map(|c| (c.ts_rel, c.camp)).collect();
            values.sort_unstable();
            values
        };
        assert_eq!(campaigns(&source), campaigns(&interned));
    }

    /// Differential: radix pre-sort + stable grouping vs the old per-person
    /// sort. For every person the event slice must be the same multiset and
    /// sorted by ts_rel; the person set must match (dense order may differ).
    /// FAST_S2 vs the original person stage on random multi-chunk partitions:
    /// dead bits at word edges, equal timestamps, sessions across midnight
    /// with a base that is not day-aligned, ties in category revenue, click
    /// ties, packed / long (interner) / non-UTF-8 names, persons without
    /// clicks and clicks without events. User rows (as multisets) and the
    /// campaign grid must be identical.
    #[test]
    fn s2_matches_old_person_stage() {
        let ids = IdInterner::new();
        let mut rng = Rng(0x5_2_a77ac);
        let cat_names: Vec<&[u8]> = vec![b"a", b"b", b"cat", b"unknown", b"zz"];
        let names: Vec<Vec<u8>> = (0..40)
            .map(|i| match i % 4 {
                0 => format!("u{i}").into_bytes(),
                1 => format!("U{i}-a-rather-long-user-identifier-{i}").into_bytes(),
                2 => vec![b'x', 0xff, b'0' + (i % 10) as u8, 0xfe], // invalid UTF-8
                _ => format!("p {i}\r").into_bytes(),
            })
            .collect();
        let keys: Vec<IdKey> = names.iter().map(|n| key_of(n, &ids)).collect();
        let click_keys: Vec<IdKey> = (0..50).map(|i| key_of(format!("c{:03}", i).as_bytes(), &ids)).collect();
        let window_start = 20_000 * MS_PER_DAY + 12_345_678;
        let window_end = window_start + 6 * MS_PER_DAY;
        let base = window_start - SEVEN_DAYS_MS;
        let campaigns = 4u32; // 0..3 real, 4 = organic
        let (mut total_rows, mut total_assists, mut total_tops) = (0usize, 0i64, 0usize);
        for round in 0..200 {
            let parts = 2usize;
            let nchunks = 1 + rng.below(4) as usize;
            let mut ev_chunks = Vec::new();
            let mut ck_chunks = Vec::new();
            let span = (window_end - window_start) as u64;
            let narrow = round % 3 == 0; // few distinct ts -> many ties / short gaps
            for _ in 0..nchunks {
                let mut evs: Vec<Vec<Ev>> = vec![Vec::new(); parts];
                let mut cks: Vec<Vec<Ck>> = vec![Vec::new(); parts];
                for _ in 0..rng.below(300) {
                    let person = keys[rng.below(if round % 2 == 0 { 6 } else { 40 }) as usize];
                    let off = if narrow { rng.below(40) * 600_000 + (span / 2) } else { rng.below(span) };
                    let ts_rel = (SEVEN_DAYS_MS as u64 + off) as u32;
                    let cat = rng.below(cat_names.len() as u64) as u32;
                    let price = if rng.below(3) == 0 { [1u32, 5, 100, 10_000_000][rng.below(4) as usize] } else { 0 };
                    evs[(person % parts as u64) as usize].push(Ev { person, ts_rel, pc: price << 8 | cat });
                }
                for _ in 0..rng.below(120) {
                    let person = keys[rng.below(if round % 2 == 0 { 8 } else { 40 }) as usize];
                    let ts_rel = if narrow {
                        (SEVEN_DAYS_MS as u64 + span / 2 - rng.below(20) * 600_000) as u32
                    } else {
                        rng.below(SEVEN_DAYS_MS as u64 + span) as u32
                    };
                    let click = click_keys[rng.below(click_keys.len() as u64) as usize];
                    let camp = rng.below(u64::from(campaigns)) as u16;
                    cks[(person % parts as u64) as usize].push(Ck { person, click, ts_rel, camp });
                }
                let mut ev_off = vec![0u32];
                let mut all_evs = Vec::new();
                for p in &evs {
                    all_evs.extend_from_slice(p);
                    ev_off.push(all_evs.len() as u32);
                }
                let mut ck_off = vec![0u32];
                let mut all_cks = Vec::new();
                for p in &cks {
                    all_cks.extend_from_slice(p);
                    ck_off.push(all_cks.len() as u32);
                }
                let dead: Vec<AtomicU64> = (0..all_evs.len().div_ceil(64))
                    .map(|_| {
                        AtomicU64::new(match rng.below(4) {
                            0 => 0,
                            1 => u64::MAX,
                            _ => rng.next() & rng.next(),
                        })
                    })
                    .collect();
                ev_chunks.push(EvChunk { evs: all_evs, ev_off, toks: Vec::new(), tok_off: vec![0, 0], dead });
                ck_chunks.push(CkChunk { cks: all_cks, ck_off, has_fallback_click: false });
            }
            let pc = PersonCtx {
                ascii_users: false,
                base,
                ids: &ids,
                cat_names: &cat_names,
                organic: campaigns,
                click_ids: ClickIds::Interned { ids: &ids, any_fallback: true },
            };
            let mut old_scratch = Scratch::new(campaigns);
            let mut new_scratch = Scratch::new(campaigns);
            let mut old_days = CampaignDays::new(campaigns as usize + 1, window_start, window_end, false);
            let mut new_days = CampaignDays::new(campaigns as usize + 1, window_start, window_end, true);
            let mut old_rows = Vec::new();
            let mut cols = out::UserDailyCols::for_walk(true);
            for partition in 0..parts {
                process_partition(partition, &ev_chunks, &ck_chunks, &pc, &mut old_scratch, &mut old_days, &mut old_rows);
                process_partition2(partition, &ev_chunks, &ck_chunks, &pc, &mut new_scratch, &mut new_days, &mut cols, true);
                // gather_live vs the per-bit loop, on every sub-range edge.
                for chunk in &ev_chunks {
                    let (lo, hi) = (chunk.ev_off[partition] as usize, chunk.ev_off[partition + 1] as usize);
                    for (a, b) in [(lo, hi), (lo + (hi - lo) / 3, hi), (lo, lo + (hi - lo) / 2)] {
                        let mut fast = Vec::new();
                        gather_live(&chunk.evs, &chunk.dead, a, b, &mut fast);
                        let slow: Vec<Ev> = (a..b).filter(|&i| !is_dead(&chunk.dead, i)).map(|i| chunk.evs[i]).collect();
                        assert_eq!(fast, slow, "round {round}: gather_live");
                    }
                }
            }
            let mut old: Vec<_> = old_rows
                .iter()
                .map(|r| (r.user_id.to_string(), r.day, r.sessions, r.events, r.revenue_cents, r.top_category.as_ref().map(|c| c.to_string())))
                .collect();
            let mut new: Vec<_> = (0..cols.len())
                .map(|i| {
                    let top = (cols.top[i] != out::TOP_NULL).then(|| String::from_utf8_lossy(cat_names[cols.top[i] as usize]).into_owned());
                    let (sessions, events) = cols.counts(i);
                    (cols.user_name(i), cols.day[i], sessions, events, cols.revenue_cents[i], top)
                })
                .collect();
            old.sort();
            new.sort();
            assert_eq!(old, new, "round {round}: user rows differ");
            let grid = |d: &CampaignDays| {
                let mut v = Vec::new();
                d.for_each_row(|c, day, x| v.push((c, day, x.sessions, x.purchasing_sessions, x.revenue, x.assisted_sessions)));
                v
            };
            assert_eq!(grid(&old_days), grid(&new_days), "round {round}: campaign grid differs");
            total_rows += new.len();
            total_tops += new.iter().filter(|r| r.5.is_some()).count();
            total_assists += grid(&new_days).iter().map(|r| r.5).sum::<i64>();
        }
        // The random inputs must actually exercise rows, purchases and assists.
        assert!(total_rows > 2000 && total_tops > 500 && total_assists > 500, "{total_rows} {total_tops} {total_assists}");
    }

    #[test]
    fn walk_matches_s2_for_exact_session_and_click_boundaries() {
        let ids = IdInterner::new();
        let person = key_of("long UTF-8 name 🦀".as_bytes(), &ids);
        let day = 20_000;
        let midnight = i64::from(day + 1) * MS_PER_DAY;
        let window_start = midnight - MS_PER_DAY + 123;
        let window_end = midnight + MS_PER_DAY;
        let base = window_start - SEVEN_DAYS_MS;
        let first = (midnight - 1 - base) as u32;
        let gap = SESSION_GAP_MS as u32;
        let week = SEVEN_DAYS_MS as u32;
        let mut events = vec![
            Ev { person, ts_rel: first, pc: 9 << 8 },
            Ev { person, ts_rel: first, pc: 9 << 8 | 1 },
            Ev { person, ts_rel: first + 1, pc: 0 },
            Ev { person, ts_rel: first + gap + 1, pc: 0 },
            Ev { person, ts_rel: first + 2 * (gap + 1), pc: 2 << 8 | 255 },
        ];
        // Exercise the short-gap block shortcut as well as the scalar boundary.
        events.extend((0..25).map(|_| Ev { person, ts_rel: first, pc: 0 }));
        events.reverse();
        let mut ev_chunk = chunk_with(events.len());
        ev_chunk.ev_off = vec![0, events.len() as u32];
        ev_chunk.evs = events;
        let ev_chunks = [ev_chunk];
        let clicks = [
            (first - week - 1, 0, &b"expired"[..]),
            (first - week, 1, &b"boundary"[..]),
            (first, 3, &b"Z"[..]),
            (first, 2, &b"A a long source key"[..]),
            (first + 1, 0, &b"next"[..]),
        ].map(|(ts_rel, camp, id)| Ck { person, ts_rel, camp, click: key_of(id, &ids) });
        let pc = PersonCtx {
            base,
            ids: &ids,
            ascii_users: false,
            cat_names: &[],
            organic: 4,
            click_ids: ClickIds::Interned { ids: &ids, any_fallback: true },
        };
        let grid = |days: &CampaignDays| {
            let mut out = Vec::new();
            days.for_each_row(|c, day, s| {
                out.push((c, day, s.sessions, s.purchasing_sessions, s.revenue, s.assisted_sessions));
            });
            out.sort_unstable();
            out
        };
        for copies in [1, 3] {
            let ck_chunks = [CkChunk {
                cks: clicks.repeat(copies),
                ck_off: vec![0, (clicks.len() * copies) as u32],
                has_fallback_click: true,
            }];
            let run = |fast| {
                let mut scratch = Scratch::new(4);
                let mut days = CampaignDays::new(5, window_start, window_end, fast);
                let mut cols = out::UserDailyCols::for_walk(fast);
                process_partition2(0, &ev_chunks, &ck_chunks, &pc, &mut scratch, &mut days, &mut cols, fast);
                (cols, grid(&days))
            };
            let (old, old_grid) = run(false);
            let (new, new_grid) = run(true);
            assert_eq!(new_grid, old_grid);
            assert_eq!(new_grid, vec![
                (0, day + 1, 1, 1, 2, 1),
                (1, day, 0, 0, 0, 1),
                (2, day, 1, 1, 18, 1),
                (2, day + 1, 0, 0, 0, 1),
                (3, day, 0, 0, 0, 1),
                (3, day + 1, 0, 0, 0, 1),
            ]);
            assert_eq!(new.day, vec![day, day + 1]);
            assert_eq!(new.day, old.day);
            assert_eq!([new.counts(0), new.counts(1)], [(1, 29), (1, 1)]);
            assert_eq!([new.counts(0), new.counts(1)], [old.counts(0), old.counts(1)]);
            assert_eq!(new.revenue_cents, old.revenue_cents);
            assert_eq!(new.top, vec![0, 255]);
            assert_eq!(new.top, old.top);
            assert_eq!(new.user_name(0), old.user_name(0));
            assert_eq!(new.user_name(1), old.user_name(1));
        }
    }

    #[test]
    fn radix_grouping_matches_per_person_sort() {
        let ids = IdInterner::new();
        let mut rng = Rng(0x5eed_1234_abcd_0001);
        let mut old = Scratch::new(0);
        let mut new = Scratch::new(0);
        for round in 0..300 {
            let persons = 1 + rng.below(if round % 3 == 0 { 4 } else { 60 }) as usize;
            // Timestamp regimes: full u32 range, narrow, tiny (many equal),
            // and high-bits-only variation (low digits constant).
            let regime = round % 4;
            let mut evs = Vec::new();
            for p in 0..persons {
                let person = (p as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
                let count = 1 + rng.below(200) as usize;
                for _ in 0..count {
                    let ts_rel = match regime {
                        0 => rng.next() as u32,
                        1 => 1_000_000 + rng.below(100_000) as u32,
                        2 => 7 + rng.below(4) as u32,
                        _ => (rng.below(8) as u32) << 25 | 0x00AB_CDEF,
                    };
                    evs.push(Ev { person, ts_rel, pc: rng.next() as u32 });
                }
            }
            // Shuffle so persons interleave.
            for i in (1..evs.len()).rev() {
                evs.swap(i, rng.below(i as u64 + 1) as usize);
            }
            old.evs.clone_from(&evs);
            old.cks.clear();
            new.evs.clone_from(&evs);
            new.cks.clear();
            let click_ids = ClickIds::Interned { ids: &ids, any_fallback: false };
            group_partition(&mut old, click_ids, false);
            group_partition(&mut new, click_ids, true);
            assert_eq!(old.index.persons.len(), new.index.persons.len());
            let by_person = |s: &Scratch| -> std::collections::HashMap<IdKey, Vec<Ev>> {
                s.index
                    .persons
                    .iter()
                    .enumerate()
                    .map(|(d, &p)| (p, s.evs_by[s.ev_off[d] as usize..s.ev_off[d + 1] as usize].to_vec()))
                    .collect()
            };
            let (a, b) = (by_person(&old), by_person(&new));
            for (person, old_slice) in &a {
                let new_slice = &b[person];
                assert!(new_slice.windows(2).all(|w| w[0].ts_rel <= w[1].ts_rel), "round {round}: not sorted");
                assert!(old_slice.windows(2).all(|w| w[0].ts_rel <= w[1].ts_rel));
                let key = |e: &Ev| (e.ts_rel, e.pc, e.person);
                let mut x: Vec<_> = old_slice.iter().map(key).collect();
                let mut y: Vec<_> = new_slice.iter().map(key).collect();
                x.sort_unstable();
                y.sort_unstable();
                assert_eq!(x, y, "round {round}: multiset differs");
            }
        }
    }

    #[test]
    fn radix_sort_evs_is_stable_and_sorted() {
        let mut rng = Rng(0x0dd_ba11);
        let mut tmp = Vec::new();
        for n in [0usize, 1, 2, 3, 64, 1000, 5000] {
            let mut evs: Vec<Ev> =
                (0..n).map(|i| Ev { person: 0, ts_rel: (rng.next() as u32) % 97 * 0x0101_0101, pc: i as u32 }).collect();
            let mut expected = evs.clone();
            expected.sort_by_key(|e| e.ts_rel); // stable
            radix_sort_evs_by_ts(&mut evs, &mut tmp);
            assert_eq!(evs, expected);
        }
    }

    #[test]
    fn group_by_dense_is_stable_and_drops_unindexed() {
        let mut index = DenseIndex::default();
        index.reset(1);
        let persons: Vec<IdKey> = (0..500u64).map(|i| (i % 37).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1).collect();
        let dense: Vec<u32> = persons.iter().map(|&p| index.insert(p)).collect();
        assert_eq!(index.persons.len(), 37);
        assert_eq!(index.get(12345 << 4), NO_PERSON);
        let mut ids = dense.clone();
        ids[3] = NO_PERSON; // dropped
        let items: Vec<u32> = (0..500).collect();
        let (mut out, mut off) = (Vec::new(), Vec::new());
        group_by_dense(&items, &ids, 37, &mut out, &mut off);
        assert_eq!(out.len(), 499);
        for g in 0..37 {
            let slice = &out[off[g] as usize..off[g + 1] as usize];
            assert!(slice.windows(2).all(|w| w[0] < w[1])); // stable
            assert!(slice.iter().all(|&i| ids[i as usize] == g as u32));
        }
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
