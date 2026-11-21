//! Timestamp parsing for the Shopping Sessions pipeline.
//!
//! `rules.md` ("Timestamps") accepts exactly three forms for a `ts` field, and nothing
//! else:
//!
//! | Form | Example                     | Meaning                                       |
//! | ---- | --------------------------- | --------------------------------------------- |
//! | F1   | `2026-10-03T23:58:10Z`      | UTC, whole seconds                            |
//! | F2   | `2026-10-03T19:58:10-04:00` | offset `+HH:MM` / `-HH:MM`, subtract for UTC   |
//! | F3   | `1791071890000`             | digits only, epoch milliseconds               |
//!
//! All three examples above are the *same instant*; `tests::three_way_equivalence`
//! asserts that, because it is the strongest available check on the epoch arithmetic.
//!
//! Design notes that matter for correctness:
//!
//! * **"Digits" means ASCII `0`..=`9` only.** `rules.md` ("CSV dialect") says so
//!   explicitly: `٢٥٠` (Arabic-Indic) and `２５０` (full-width) are *not* digits. This
//!   module therefore never decodes UTF-8 and never calls any `char` classifier; it only
//!   ever compares raw bytes against `b'0'..=b'9'`. Any multi-byte digit look-alike is
//!   just a non-digit byte and is rejected.
//! * **No trimming.** The input is the field bytes *after* CSV quote-stripping only.
//!   Spaces are part of a value (`rules.md`, "CSV dialect"), so ` 2026-...Z` is invalid.
//! * **`\r` is an ordinary character**, never a line ending, so `2026-10-03T09:00:00Z\r`
//!   is an invalid timestamp (fixture E20). That falls out of the strict length checks.
//! * **No window check here.** `parse_ts` answers only "is this a syntactically and
//!   calendrically valid instant, and which instant is it". The event/click window
//!   comparisons live in the caller.
//!
//! Zero dependencies, zero allocation, no UTF-8 decoding: this runs ~75M times.

/// Field layout shared by F1 and F2, as byte offsets:
///
/// ```text
///   0123456789012345678901234
///   2026-10-03T19:58:10-04:00
///   ^^^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^
///   yyyy-mm-ddThh:mi:ss+oh:om
/// ```
use std::sync::atomic::{AtomicPtr, Ordering};

const F1_LEN: usize = 20; // `2026-10-03T23:58:10Z`
const F2_LEN: usize = 25; // `2026-10-03T19:58:10-04:00`

const MS_PER_SEC: i64 = 1_000;
const MS_PER_MIN: i64 = 60 * MS_PER_SEC;
const MS_PER_HOUR: i64 = 60 * MS_PER_MIN;
const MS_PER_DAY: i64 = 24 * MS_PER_HOUR;

// ---------------------------------------------------------------- date-prefix cache
//
// Almost every F1/F2 `ts` falls within a few days of the manifest window, so it starts
// with one of ~30 distinct `YYYY-MM-DD` prefixes. `DateCache` memoises "those 10 bytes
// -> days-since-epoch" for the dates around the window, so the hot path skips the
// year/month/day digit parsing, the calendar validation and `days_from_civil`.
//
// Why this cannot change any answer:
//
// * An entry is only ever created from a real, valid calendar date formatted exactly as
//   `{y:04}-{m:02}-{d:02}`, with its day count computed by the same `days_from_civil`
//   the exact path uses. A lookup hits only if all 10 input bytes equal the stored key,
//   and every such 10-byte string is one `parse_civil` would accept with that same
//   day count. So a hit is just a memoised `parse_civil` date half.
// * The cache is a pure memo of a pure function, so *which* window it was built for
//   affects only the hit rate, never a result. That is what makes calling
//   `set_window_hint` repeatedly (tests calling `run` many times, or two different
//   windows in one process) safe: an old or "wrong" cache is still exact.
// * Everything that is not a hit (out-of-window dates, malformed or impossible dates,
//   non-ASCII bytes, collisions in the table) takes the original exact path.

/// One direct-mapped slot: the 10-byte key `YYYY-MM-DD` split as bytes 0..8 (u64) and
/// 8..10 (u16), plus the day number. An empty slot is all zeros; its `k8` has byte 7
/// (the second `-`) equal to 0, and a lookup is only attempted after checking that the
/// input has `-` at byte 7, so an empty slot can never match.
#[derive(Clone, Copy, Default)]
struct DateEntry {
    k8: u64,
    k2: u16,
    days: i32,
}

/// Slots are indexed by (low nibble of the month's second digit, day-of-month): byte 6
/// low nibble (4 bits), byte 8 low 2 bits, byte 9 low nibble. For valid dates that is
/// unique per (month-mod-10, DD), so two dates collide only if they share DD and their
/// months share a last digit (e.g. Jan 05 / Nov 05, or the same date a year apart). On a
/// collision the first date inserted keeps the slot; the other one just misses.
const SLOTS: usize = 1024;

#[inline(always)]
fn slot_of(b: &[u8]) -> usize {
    ((b[6] as usize & 0x0F) << 6) | ((b[8] as usize & 0x03) << 4) | (b[9] as usize & 0x0F)
}

/// Precomputed `YYYY-MM-DD -> days since epoch` table for dates around a window.
pub struct DateCache {
    window: (i64, i64),
    table: Box<[DateEntry; SLOTS]>,
}

/// Upper bound on the number of days a cache will cover, so a silly manifest window
/// cannot make building it expensive. Days beyond it just miss.
const MAX_CACHE_DAYS: i64 = 4096;

impl DateCache {
    /// Cache covering every date in `[window_start - 8 days, window_end + 7 days]`.
    /// The margins cover clicks (up to 7 days before the window) and F2 offsets, whose
    /// local date can be a day either side of the UTC date.
    fn for_window(window_start_ms: i64, window_end_ms: i64) -> Self {
        let mut table = Box::new([DateEntry::default(); SLOTS]);
        let start_day = window_start_ms.div_euclid(MS_PER_DAY).saturating_sub(8);
        let end_day = window_end_ms.div_euclid(MS_PER_DAY).saturating_add(7);
        let end_day = end_day.min(start_day.saturating_add(MAX_CACHE_DAYS));
        let mut d = start_day;
        while d <= end_day {
            let (y, m, day) = civil_from_days(d);
            // Only 4-digit years are expressible in F1/F2; i32 holds any such day count.
            if (0..=9999).contains(&y) {
                let s = format!("{y:04}-{m:02}-{day:02}");
                let k = s.as_bytes();
                debug_assert_eq!(days_from_civil(y, m, day), d);
                let e = DateEntry {
                    k8: u64::from_le_bytes(k[0..8].try_into().unwrap()),
                    k2: u16::from_le_bytes(k[8..10].try_into().unwrap()),
                    days: d as i32,
                };
                let slot = &mut table[slot_of(k)];
                if slot.k8 == 0 {
                    *slot = e;
                }
            }
            d += 1;
        }
        DateCache {
            window: (window_start_ms, window_end_ms),
            table,
        }
    }

    /// Day number for the 10-byte date prefix of `b`, if it is cached. Caller
    /// guarantees `b.len() >= 10` and `b[7] == b'-'`.
    #[inline(always)]
    fn lookup(&self, b: &[u8]) -> Option<i64> {
        let e = &self.table[slot_of(b)];
        let k8 = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let k2 = u16::from_le_bytes(b[8..10].try_into().unwrap());
        if e.k8 == k8 && e.k2 == k2 {
            Some(e.days as i64)
        } else {
            None
        }
    }
}

/// The process-global cache. Null means "no hint given": `parse_ts` then behaves exactly
/// like the original code path. Each `set_window_hint` with a new window installs a new,
/// leaked cache (16 KiB) — it is never freed, so a reader on another thread can never
/// see a dangling pointer, and any cache it does see is exact (see above).
static DATE_CACHE: AtomicPtr<DateCache> = AtomicPtr::new(std::ptr::null_mut());

/// Tell the parser which window the data is for, so it can precompute the dates around
/// it. Purely a speed hint: results never depend on it. Cheap to call repeatedly with the
/// same window (no-op); a different window installs a new cache.
pub fn set_window_hint(window_start_ms: i64, window_end_ms: i64) {
    let cur = DATE_CACHE.load(Ordering::Acquire);
    // SAFETY: non-null pointers in DATE_CACHE come from Box::into_raw and are never freed.
    if !cur.is_null() && unsafe { (*cur).window } == (window_start_ms, window_end_ms) {
        return;
    }
    let cache = Box::new(DateCache::for_window(window_start_ms, window_end_ms));
    DATE_CACHE.store(Box::into_raw(cache), Ordering::Release);
}

#[inline(always)]
fn current_cache() -> Option<&'static DateCache> {
    let p = DATE_CACHE.load(Ordering::Acquire);
    // SAFETY: see DATE_CACHE — pointers are leaked boxes, valid for 'static.
    if p.is_null() {
        None
    } else {
        Some(unsafe { &*p })
    }
}

/// Parse a timestamp field into integer milliseconds since the Unix epoch, UTC.
///
/// Returns `None` if the value is not a valid timestamp per `rules.md`.
///
/// Input is the raw field bytes AFTER CSV quote-stripping (so no surrounding quotes),
/// and is NOT trimmed — leading or trailing whitespace makes it invalid.
///
/// Behaviour is identical to [`parse_ts_uncached`] (the original implementation) on
/// every input; this version only adds fast paths: a date-prefix cache for F1/F2 (see
/// [`set_window_hint`]) and an 8-bytes-at-a-time digit parse for F3. The F3 overflow
/// rules (saturate at `i64::MAX`) are documented on [`parse_ts_uncached`].
#[inline]
pub fn parse_ts(b: &[u8]) -> Option<i64> {
    parse_ts_with(current_cache(), b)
}

#[inline(always)]
fn parse_ts_with(cache: Option<&DateCache>, b: &[u8]) -> Option<i64> {
    match b.len() {
        F1_LEN if b[19] == b'Z' => match cached_civil(cache, b) {
            Some(r) => r,
            None => parse_civil(b).map(|(ms, _)| ms),
        },
        F2_LEN if b[19] == b'+' || b[19] == b'-' => {
            let ms = match cached_civil(cache, b) {
                Some(r) => r?,
                None => parse_civil(b)?.0,
            };
            // Offset handling copied verbatim from parse_ts_uncached.
            if b[22] != b':' {
                return None;
            }
            let oh = two_digits(b, 20)?;
            let om = two_digits(b, 23)?;
            if oh > 23 || om > 59 {
                return None;
            }
            let off = oh * MS_PER_HOUR + om * MS_PER_MIN;
            Some(if b[19] == b'+' { ms - off } else { ms + off })
        }
        0 => None,
        8..=18 => parse_epoch_millis_swar(b),
        _ => parse_epoch_millis(b),
    }
}

/// Cache-assisted version of `parse_civil(b).map(|(ms, _)| ms)` for `b.len() >= 19`.
///
/// Returns `None` ("not decided here, use the exact path") on a cache miss, and
/// `Some(result)` when the date prefix hit the cache — where `result` is exactly what
/// `parse_civil` would have produced: the date half is already known valid, so only
/// the `T`, the two `:` and the `hh`,`mi`,`ss` digits/ranges remain to check.
#[inline(always)]
fn cached_civil(cache: Option<&DateCache>, b: &[u8]) -> Option<Option<i64>> {
    let cache = cache?;
    // Guards the empty-slot sentinel (see DateEntry). A real hit implies it anyway.
    if b[7] != b'-' {
        return None;
    }
    let days = cache.lookup(b)?;
    if b[10] != b'T' {
        return Some(None);
    }
    Some(swar_hms(b).map(|tod| days * MS_PER_DAY + tod))
}

/// Parse and validate `hh:mi:ss` at `b[11..19]` in one u64, returning milliseconds into
/// the day, or `None` if any of the six digit bytes is not an ASCII digit, either colon
/// is missing, or hour > 23 / minute > 59 / second > 59 (same rules as `parse_civil`).
///
/// Bytes are loaded little-endian, so byte `i` of the slice is bits `8i..8i+8`.
/// XOR with `"00:00:00"` maps an ASCII digit to its value 0..=9 and a `:` to 0. A byte
/// `c` maps into 0..=9 under `^ 0x30` iff `c` is in `0x30..=0x39` (high nibble must be 3,
/// low nibble 0..=9), so this is an exact ASCII-digit test, never a Unicode one.
#[inline(always)]
fn swar_hms(b: &[u8]) -> Option<i64> {
    let v = u64::from_le_bytes(b[11..19].try_into().unwrap());
    let x = v ^ u64::from_le_bytes(*b"00:00:00");
    // Every byte <= 9: a byte >= 0x80 is caught by `| x`; for a byte < 0x80, adding
    // 0x76 sets bit 7 iff the byte >= 10, and cannot carry into the next byte.
    if ((x.wrapping_add(0x7676_7676_7676_7676) | x) & 0x8080_8080_8080_8080) != 0 {
        return None;
    }
    // The colon positions (bytes 2 and 5) must be exactly 0, i.e. exactly `:`.
    if x & 0x0000_FF00_00FF_0000 != 0 {
        return None;
    }
    let d = |i: u32| ((x >> (8 * i)) & 0xFF) as i64;
    let hour = d(0) * 10 + d(1);
    let min = d(3) * 10 + d(4);
    let sec = d(6) * 10 + d(7);
    if hour > 23 || min > 59 || sec > 59 {
        return None;
    }
    Some(hour * MS_PER_HOUR + min * MS_PER_MIN + sec * MS_PER_SEC)
}

/// F3 for lengths 8..=18: the first 8 bytes via SWAR, the rest scalar. At most 18 ASCII
/// digits is at most 999_999_999_999_999_999 < i64::MAX, so no overflow handling is
/// needed here (longer values go to `parse_epoch_millis`, which saturates). Same result
/// as `parse_epoch_millis` on every input of these lengths.
#[inline(always)]
fn parse_epoch_millis_swar(b: &[u8]) -> Option<i64> {
    debug_assert!((8..=18).contains(&b.len()));
    let v = u64::from_le_bytes(b[0..8].try_into().unwrap());
    let x = v ^ 0x3030_3030_3030_3030;
    // All 8 bytes are ASCII digits (see swar_hms for why this test is exact).
    if ((x.wrapping_add(0x7676_7676_7676_7676) | x) & 0x8080_8080_8080_8080) != 0 {
        return None;
    }
    // Combine 8 digit values (most significant digit in the lowest byte) into a number:
    // pairs -> 2-digit values in 16-bit lanes, -> 4-digit in 32-bit lanes, -> 8-digit.
    let x = x.wrapping_mul(10).wrapping_add(x >> 8) & 0x00FF_00FF_00FF_00FF;
    let x = x.wrapping_mul(100).wrapping_add(x >> 16) & 0x0000_FFFF_0000_FFFF;
    let x = x.wrapping_mul(10_000).wrapping_add(x >> 32) & 0x0000_0000_FFFF_FFFF;
    let mut n = x as i64;
    for &c in &b[8..] {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        n = n * 10 + d as i64;
    }
    Some(n)
}

/// Inverse of `days_from_civil` (Howard Hinnant's `civil_from_days`): day number since
/// 1970-01-01 -> proleptic Gregorian (year, month, day). Only used to build the cache.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg_attr(not(test), allow(dead_code))]
/// The original, cache-free parser, kept unchanged as the oracle for the differential
/// tests: `parse_ts` must agree with it on every input. (Not called at runtime.)
///
/// Parse a timestamp field into integer milliseconds since the Unix epoch, UTC.
///
/// Returns `None` if the value is not a valid timestamp per `rules.md`.
///
/// Input is the raw field bytes AFTER CSV quote-stripping (so no surrounding quotes),
/// and is NOT trimmed — leading or trailing whitespace makes it invalid.
///
/// # F3 overflow
///
/// `rules.md`: "An F3 value is just a number, so one that is very short (`86400000` is
/// 1970-01-02) or too long for a 64-bit integer is not malformed, only outside the
/// window." So a 21-digit F3 value is *valid*, just absurd. We represent it by
/// **saturating at [`i64::MAX`]** — never wrapping, never panicking. Since an F3 value is
/// digits-only it can never be negative, so saturation is one-sided. Any real window end
/// is astronomically below `i64::MAX`, so the caller's `ts < window_end` test rejects it,
/// which is exactly the required behaviour (fixture E18, event `e4`).
#[inline]
fn parse_ts_uncached(b: &[u8]) -> Option<i64> {
    // Dispatch is unambiguous on (length, byte 19): F1 and F2 both contain `-`, `T` and
    // `:`, so a value that is all digits can never have `Z` at index 19 (F1) nor `+`/`-`
    // there (F2). Anything that is not shaped like F1 or F2 must therefore be F3 — and
    // the F3 path rejects every non-digit byte, so garbage still returns None.
    match b.len() {
        F1_LEN if b[19] == b'Z' => parse_civil(b).map(|(ms, _)| ms),
        F2_LEN if b[19] == b'+' || b[19] == b'-' => {
            let (ms, _) = parse_civil(b)?;
            // Offset hours 00..=23, minutes 00..=59, colon REQUIRED (`+0200` is invalid,
            // and is in any case the wrong length and so never reaches here).
            if b[22] != b':' {
                return None;
            }
            let oh = two_digits(b, 20)?;
            let om = two_digits(b, 23)?;
            if oh > 23 || om > 59 {
                return None;
            }
            let off = oh * MS_PER_HOUR + om * MS_PER_MIN;
            // "subtract it to get UTC". `-00:00` is UTC, which falls out for free.
            Some(if b[19] == b'+' { ms - off } else { ms + off })
        }
        0 => None, // empty field is a bad timestamp (and would otherwise parse as F3 = 0)
        _ => parse_epoch_millis(b),
    }
}

/// Fast path for `ingest_ts`, which `rules.md` guarantees is "always valid F1".
///
/// Accepts **only** F1 (`2026-10-03T23:58:10Z`) and returns `None` for anything else,
/// including otherwise-valid F2 and F3 values. Use [`parse_ts`] for `ts` fields.
///
/// Kept as a separate function rather than a flag so the hot ingest path skips the
/// dispatch entirely; it shares all validation with [`parse_ts`], so the two cannot drift.
#[inline]
pub fn parse_ingest_ts(b: &[u8]) -> Option<i64> {
    if b.len() != F1_LEN || b[19] != b'Z' {
        return None;
    }
    parse_civil(b).map(|(ms, _)| ms)
}

/// Parse and validate the fixed 19-byte civil part `yyyy-mm-ddThh:mi:ss` at the start of
/// `b` (callers have already checked `b.len() >= 19`).
///
/// Returns `(milliseconds as if the civil time were UTC, days since the epoch)`. The
/// caller applies any zone offset. The day count is returned because it is free here.
#[inline]
fn parse_civil(b: &[u8]) -> Option<(i64, i64)> {
    // Separators must be exactly these bytes: this is what rejects a space instead of
    // `T`, a lowercase `t`, and fractional seconds (which change the length anyway).
    if b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
        return None;
    }

    let year = four_digits(b, 0)?;
    let month = two_digits(b, 5)?;
    let day = two_digits(b, 8)?;
    let hour = two_digits(b, 11)?;
    let min = two_digits(b, 14)?;
    let sec = two_digits(b, 17)?;

    // Real calendar validation: month 13, February 30, February 29 in a non-leap year,
    // hour 24 and second 60 are all "impossible calendar values" per rules.md. There are
    // no leap seconds in this model, so 60 is rejected.
    if month < 1 || month > 12 {
        return None;
    }
    if day < 1 || day > days_in_month(year, month) {
        return None;
    }
    if hour > 23 || min > 59 || sec > 59 {
        return None;
    }

    let days = days_from_civil(year, month, day);
    let ms = days * MS_PER_DAY + hour * MS_PER_HOUR + min * MS_PER_MIN + sec * MS_PER_SEC;
    Some((ms, days))
}

/// Parse an F3 value: digits only, epoch milliseconds, any length, leading zeros allowed.
///
/// Rejects any non-ASCII-digit byte, which covers `1791071890000.0`, `1.79e12`, `-5`,
/// a trailing `\r`, and every Unicode digit look-alike. Saturates at [`i64::MAX`] on
/// overflow (see [`parse_ts`] docs) instead of wrapping, but still scans the whole value
/// so that a non-digit byte after the overflow point is still caught.
#[inline]
fn parse_epoch_millis(b: &[u8]) -> Option<i64> {
    if b.is_empty() {
        return None;
    }
    let mut v: i64 = 0;
    let mut overflowed = false;
    for &c in b {
        // wrapping_sub on u8: any byte below b'0' wraps to a large value, so the single
        // `> 9` test catches both ends. No char decoding, so multi-byte digit
        // look-alikes are simply non-digit bytes.
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        if !overflowed {
            match v.checked_mul(10).and_then(|x| x.checked_add(d as i64)) {
                Some(n) => v = n,
                None => overflowed = true,
            }
        }
    }
    Some(if overflowed { i64::MAX } else { v })
}

/// Two ASCII digits at `b[i..i+2]` as a number, or `None` if either byte is not an ASCII
/// digit.
#[inline]
fn two_digits(b: &[u8], i: usize) -> Option<i64> {
    let hi = b[i].wrapping_sub(b'0');
    let lo = b[i + 1].wrapping_sub(b'0');
    if hi > 9 || lo > 9 {
        return None;
    }
    Some(hi as i64 * 10 + lo as i64)
}

/// Four ASCII digits at `b[i..i+4]` as a number, or `None` if any byte is not an ASCII
/// digit. Used for the year, so `２026` (full-width `2`) is rejected — and in fact a
/// full-width digit is 3 UTF-8 bytes, which already changes the field length.
#[inline]
fn four_digits(b: &[u8], i: usize) -> Option<i64> {
    Some(two_digits(b, i)? * 100 + two_digits(b, i + 2)?)
}

/// Proleptic Gregorian leap year: every 4th year, except every 100th, except every 400th.
#[inline]
fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

#[inline]
fn days_in_month(y: i64, m: i64) -> i64 {
    const LEN: [i64; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if m == 2 && is_leap(y) {
        29
    } else {
        LEN[(m - 1) as usize]
    }
}

/// Days since 1970-01-01 for a proleptic Gregorian date. Howard Hinnant's
/// `days_from_civil` (the algorithm behind C++20's `std::chrono::year_month_day`).
///
/// How it works, since it is not self-evident:
///
/// 1. **Shift the year to start in March** (`y -= m <= 2`). With March as month 1, the
///    leap day lands at the *end* of the year, so a leap year just makes the final month
///    one day longer and no month index has to shift.
/// 2. **Work in 400-year "eras"** (`era`), because the Gregorian calendar repeats exactly
///    every 400 years = 146097 days. `yoe` is the year-of-era, 0..=399.
/// 3. **Day-of-year** (`doy`, 0..=365) from the March-based month: successive month
///    lengths from March are 31,30,31,30,31,31,30,31,30,31,31,28/29, whose running total
///    is exactly `(153 * mp + 2) / 5` for `mp` = 0..=11. That integer-division identity is
///    the whole trick — the 153/5 ratio is the average 31,30,31 month length.
/// 4. **Day-of-era** (`doe`, 0..=146096) = whole years of the era in days, plus that era's
///    leap days (`yoe/4 - yoe/100`; no `+ yoe/400` term because year-of-era 0 is itself
///    the 400-divisible year and its leap day is already counted at the end of the shifted
///    year), plus `doy`.
/// 5. **Rebase onto 1970-01-01** by subtracting 719468, the day-of-era offset of the Unix
///    epoch (which sits in the era starting 1600-03-01).
#[inline]
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    // Floor-divide towards negative infinity so years before 1 land in the right era.
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = if m > 2 { m - 3 } else { m + 9 }; // March-based month index [0, 11]
    let doy = (153 * mp + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

#[cfg(test)]
mod tests {
    use super::*;

    // Written from rules.md and the fixtures, not from the implementation.

    // ---------------------------------------------------------------- the three forms

    #[test]
    fn three_way_equivalence() {
        // rules.md, "Timestamps": the F1, F2 and F3 examples are the same instant.
        let f1 = parse_ts(b"2026-10-03T23:58:10Z");
        let f2 = parse_ts(b"2026-10-03T19:58:10-04:00");
        let f3 = parse_ts(b"1791071890000");
        assert_eq!(f1, Some(1_791_071_890_000));
        assert_eq!(f2, Some(1_791_071_890_000));
        assert_eq!(f3, Some(1_791_071_890_000));
        assert_eq!(f1, f2);
        assert_eq!(f2, f3);
    }

    #[test]
    fn f1_basic() {
        assert_eq!(parse_ts(b"1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_ts(b"1970-01-02T00:00:00Z"), Some(86_400_000));
        assert_eq!(parse_ts(b"1970-01-01T00:00:01Z"), Some(1_000));
        assert_eq!(parse_ts(b"1970-01-01T23:59:59Z"), Some(86_399_000));
        // Pre-epoch instants must come out negative.
        assert_eq!(parse_ts(b"1969-12-31T23:59:59Z"), Some(-1_000));
        assert_eq!(parse_ts(b"1969-12-31T00:00:00Z"), Some(-86_400_000));
    }

    #[test]
    fn f1_known_epoch_values() {
        // Spot checks against independently known epoch values.
        assert_eq!(parse_ts(b"2000-01-01T00:00:00Z"), Some(946_684_800_000));
        assert_eq!(parse_ts(b"2001-09-09T01:46:40Z"), Some(1_000_000_000_000));
        assert_eq!(parse_ts(b"2026-10-01T00:00:00Z"), Some(1_790_812_800_000)); // fixture window start
        assert_eq!(parse_ts(b"2026-10-15T00:00:00Z"), Some(1_792_022_400_000)); // fixture window end
        // The two differ by exactly 14 days (rules.md: "A 14-day window").
        assert_eq!(1_792_022_400_000i64 - 1_790_812_800_000i64, 14 * MS_PER_DAY);
        // Fixture E18 `e1`: 0001791021600000 is 2026-10-03T10:00:00Z.
        assert_eq!(parse_ts(b"2026-10-03T10:00:00Z"), Some(1_791_021_600_000));
    }

    #[test]
    fn f3_basic() {
        assert_eq!(parse_ts(b"0"), Some(0));
        assert_eq!(parse_ts(b"1"), Some(1));
        // rules.md: "one that is very short (86400000 is 1970-01-02)".
        assert_eq!(parse_ts(b"86400000"), Some(86_400_000));
        assert_eq!(parse_ts(b"86400000"), parse_ts(b"1970-01-02T00:00:00Z"));
        // Millisecond precision: rules.md says a ts with milliseconds can only be F3.
        assert_eq!(parse_ts(b"1791021600001"), Some(1_791_021_600_001));
    }

    #[test]
    fn f3_leading_zeros() {
        // rules.md: "Leading zeros are allowed (0001791071890000 is the same instant)".
        assert_eq!(parse_ts(b"0001791071890000"), Some(1_791_071_890_000));
        assert_eq!(parse_ts(b"0001791071890000"), parse_ts(b"1791071890000"));
        // Fixture E18 `e1`.
        assert_eq!(parse_ts(b"0001791021600000"), Some(1_791_021_600_000));
        assert_eq!(parse_ts(b"00000000"), Some(0));
        assert_eq!(parse_ts(b"0000000000000000000000000000001"), Some(1));
    }

    #[test]
    fn f3_overflow_saturates_and_stays_out_of_window() {
        // Fixture E18 `e4`: 21 digits. rules.md: "too long for a 64-bit integer is not
        // malformed, only outside the window" — so it parses, but to a value no window
        // can contain. We saturate at i64::MAX.
        assert_eq!(parse_ts(b"179102160000000000000"), Some(i64::MAX));
        assert_eq!(parse_ts(b"9223372036854775807"), Some(i64::MAX)); // exactly i64::MAX
        assert_eq!(parse_ts(b"9223372036854775808"), Some(i64::MAX)); // one past: saturates
        assert_eq!(parse_ts(b"99999999999999999999999999999999"), Some(i64::MAX));
        // Leading zeros must not trigger overflow on an otherwise fine value.
        let padded = b"00000000000000000000000001791071890000";
        assert_eq!(parse_ts(padded), Some(1_791_071_890_000));
        // A saturated value is rejected by any realistic window check.
        let window_end = 1_792_022_400_000i64;
        assert!(parse_ts(b"179102160000000000000").unwrap() >= window_end);
        // ...and a non-digit after the overflow point is still caught.
        assert_eq!(parse_ts(b"99999999999999999999999999999999x"), None);
        assert_eq!(parse_ts(b"179102160000000000000.0"), None);
    }

    // ---------------------------------------------------------------- F2 offsets

    #[test]
    fn f2_offsets_apply_in_the_right_direction() {
        // "subtract it to get UTC"
        assert_eq!(
            parse_ts(b"2026-10-03T19:58:10-04:00"),
            parse_ts(b"2026-10-03T23:58:10Z")
        );
        assert_eq!(
            parse_ts(b"2026-10-04T01:58:10+02:00"),
            parse_ts(b"2026-10-03T23:58:10Z")
        );
        // Fixture E7: ts 2026-10-04T01:30:00+02:00 is Oct 3 23:30 UTC.
        assert_eq!(
            parse_ts(b"2026-10-04T01:30:00+02:00"),
            parse_ts(b"2026-10-03T23:30:00Z")
        );
        // An offset that moves the instant across a month boundary, both ways.
        assert_eq!(
            parse_ts(b"2026-11-01T00:30:00+01:00"),
            parse_ts(b"2026-10-31T23:30:00Z")
        );
        assert_eq!(
            parse_ts(b"2026-10-31T23:30:00-01:00"),
            parse_ts(b"2026-11-01T00:30:00Z")
        );
        // rules.md, generator: "Offsets from -10:00 to +12:00".
        assert_eq!(
            parse_ts(b"2026-10-03T13:58:10-10:00"),
            parse_ts(b"2026-10-03T23:58:10Z")
        );
        assert_eq!(
            parse_ts(b"2026-10-04T11:58:10+12:00"),
            parse_ts(b"2026-10-03T23:58:10Z")
        );
        // A half-hour offset exercises the minutes field.
        assert_eq!(
            parse_ts(b"2026-10-04T05:28:10+05:30"),
            parse_ts(b"2026-10-03T23:58:10Z")
        );
    }

    #[test]
    fn f2_minus_zero_zero_is_utc() {
        // rules.md: "-00:00 is UTC." Fixture E17 `e4`.
        assert_eq!(
            parse_ts(b"2026-10-03T12:00:00-00:00"),
            parse_ts(b"2026-10-03T12:00:00Z")
        );
        assert_eq!(
            parse_ts(b"2026-10-03T12:00:00+00:00"),
            parse_ts(b"2026-10-03T12:00:00Z")
        );
    }

    #[test]
    fn f2_boundary_offsets_valid() {
        // Fixture E17 `e3`: +23:59 is accepted and 2026-10-04T09:59:00+23:59 is
        // 2026-10-03T10:00:00Z.
        assert_eq!(
            parse_ts(b"2026-10-04T09:59:00+23:59"),
            parse_ts(b"2026-10-03T10:00:00Z")
        );
        assert_eq!(
            parse_ts(b"2026-10-02T10:01:00-23:59"),
            parse_ts(b"2026-10-03T10:00:00Z")
        );
        assert!(parse_ts(b"2026-10-03T12:00:00+00:01").is_some());
        assert!(parse_ts(b"2026-10-03T12:00:00-00:59").is_some());
        assert!(parse_ts(b"2026-10-03T12:00:00+23:00").is_some());
    }

    #[test]
    fn f2_bad_offsets_rejected() {
        // Fixture E17: +24:00, +05:60 and +0200 are all dropped.
        assert_eq!(parse_ts(b"2026-10-03T10:00:00+24:00"), None); // hour > 23
        assert_eq!(parse_ts(b"2026-10-03T10:00:00-24:00"), None);
        assert_eq!(parse_ts(b"2026-10-03T10:00:00+99:00"), None);
        assert_eq!(parse_ts(b"2026-10-03T10:00:00+05:60"), None); // minute > 59
        assert_eq!(parse_ts(b"2026-10-03T10:00:00+05:99"), None);
        assert_eq!(parse_ts(b"2026-10-03T10:00:00+0200"), None); // no colon
        assert_eq!(parse_ts(b"2026-10-03T10:00:00-0200"), None);
        assert_eq!(parse_ts(b"2026-10-03T10:00:00+02-00"), None); // wrong separator
        assert_eq!(parse_ts(b"2026-10-03T10:00:00+02:0"), None); // too short
        assert_eq!(parse_ts(b"2026-10-03T10:00:00+2:00"), None);
        assert_eq!(parse_ts(b"2026-10-03T10:00:00+02:000"), None); // too long
        assert_eq!(parse_ts(b"2026-10-03T10:00:00 +02:00"), None); // space before offset
        assert_eq!(parse_ts(b"2026-10-03T10:00:00+02:00Z"), None); // both zone forms
        assert_eq!(parse_ts(b"2026-10-03T10:00:00+aa:bb"), None);
        assert_eq!(parse_ts(b"2026-10-03T10:00:00*02:00"), None); // sign is + or - only
    }

    // ---------------------------------------------------------------- rejections

    #[test]
    fn empty_is_invalid() {
        // rules.md: a bad timestamp includes "empty". An empty field is null, not 0.
        assert_eq!(parse_ts(b""), None);
    }

    #[test]
    fn fractional_seconds_rejected() {
        assert_eq!(parse_ts(b"2026-10-03T23:58:10.500Z"), None);
        assert_eq!(parse_ts(b"2026-10-03T23:58:10.000Z"), None);
        assert_eq!(parse_ts(b"2026-10-03T23:58:10.5Z"), None);
        assert_eq!(parse_ts(b"2026-10-03T19:58:10.500-04:00"), None);
        assert_eq!(parse_ts(b"2026-10-03T23:58:10,500Z"), None);
    }

    #[test]
    fn wrong_case_and_separators_rejected() {
        assert_eq!(parse_ts(b"2026-10-03T23:58:10z"), None); // lowercase z
        assert_eq!(parse_ts(b"2026-10-03t23:58:10Z"), None); // lowercase t
        assert_eq!(parse_ts(b"2026-10-03t23:58:10z"), None);
        // Fixture E5's bad ts: a space instead of T (and no zone).
        assert_eq!(parse_ts(b"2026-10-03 10:00:00"), None);
        assert_eq!(parse_ts(b"2026-10-03 23:58:10Z"), None); // space instead of T
        assert_eq!(parse_ts(b"2026/10/03T23:58:10Z"), None); // wrong date separator
        assert_eq!(parse_ts(b"2026-10-03T23-58-10Z"), None); // wrong time separator
        assert_eq!(parse_ts(b"20261003T235810Z"), None); // basic ISO format
    }

    #[test]
    fn missing_or_extra_zone_rejected() {
        assert_eq!(parse_ts(b"2026-10-03T23:58:10"), None); // no zone
        assert_eq!(parse_ts(b"2026-10-03T23:58:10 Z"), None);
        assert_eq!(parse_ts(b"2026-10-03T23:58:10UTC"), None);
        assert_eq!(parse_ts(b"2026-10-03T23:58:10+Z"), None);
        assert_eq!(parse_ts(b"2026-10-03"), None); // date only
        assert_eq!(parse_ts(b"23:58:10Z"), None); // time only
    }

    #[test]
    fn non_digits_in_f3_rejected() {
        // rules.md names these three explicitly.
        assert_eq!(parse_ts(b"1791071890000.0"), None);
        assert_eq!(parse_ts(b"1.79e12"), None);
        assert_eq!(parse_ts(b"-5"), None);
        // and the obvious neighbours
        assert_eq!(parse_ts(b"+1791071890000"), None);
        assert_eq!(parse_ts(b"-1791071890000"), None);
        assert_eq!(parse_ts(b"1791071890000 "), None);
        assert_eq!(parse_ts(b" 1791071890000"), None);
        assert_eq!(parse_ts(b"1 791071890000"), None);
        assert_eq!(parse_ts(b"1791071890000x"), None);
        assert_eq!(parse_ts(b"0x1791071890"), None);
        assert_eq!(parse_ts(b"1_791_071_890_000"), None);
        assert_eq!(parse_ts(b"NaN"), None);
        assert_eq!(parse_ts(b"null"), None);
        assert_eq!(parse_ts(b"-"), None);
        assert_eq!(parse_ts(b"."), None);
        // Bytes on either side of the ASCII digit block: '/' is b'0'-1, ':' is b'9'+1.
        assert_eq!(parse_ts(b"123/456"), None);
        assert_eq!(parse_ts(b"123:456"), None);
    }

    #[test]
    fn untrimmed_whitespace_is_invalid() {
        // rules.md, "CSV dialect": spaces are part of a value, quoted or not.
        assert_eq!(parse_ts(b" 2026-10-03T23:58:10Z"), None);
        assert_eq!(parse_ts(b"2026-10-03T23:58:10Z "), None);
        assert_eq!(parse_ts(b"\t2026-10-03T23:58:10Z"), None);
        assert_eq!(parse_ts(b" "), None);
        assert_eq!(parse_ts(b"   "), None);
    }

    #[test]
    fn trailing_carriage_return_is_invalid() {
        // Fixture E20: a click whose ts becomes `...Z\r` is dropped. `\r` is an ordinary
        // character of the field, never a line ending.
        assert_eq!(parse_ts(b"2026-10-03T09:00:00Z\r"), None);
        assert_eq!(parse_ts(b"2026-10-03T19:58:10-04:00\r"), None);
        assert_eq!(parse_ts(b"1791071890000\r"), None);
        assert_eq!(parse_ts(b"\r"), None);
        assert_eq!(parse_ts(b"\r2026-10-03T09:00:00Z"), None);
        // A `\r` in place of the Z keeps the F1 length but is still invalid.
        assert_eq!(parse_ts(b"2026-10-03T09:00:00\r"), None);
    }

    // ---------------------------------------------------------------- calendar

    #[test]
    fn impossible_calendar_values_rejected() {
        // rules.md names: month 13, February 30, February 29 outside a leap year,
        // hour 24, second 60.
        assert_eq!(parse_ts(b"2026-13-01T00:00:00Z"), None); // month 13
        assert_eq!(parse_ts(b"2026-00-01T00:00:00Z"), None); // month 0
        assert_eq!(parse_ts(b"2026-99-01T00:00:00Z"), None);
        assert_eq!(parse_ts(b"2026-02-30T00:00:00Z"), None); // February 30
        assert_eq!(parse_ts(b"2026-02-29T00:00:00Z"), None); // non-leap February 29
        assert_eq!(parse_ts(b"2026-10-00T00:00:00Z"), None); // day 0
        assert_eq!(parse_ts(b"2026-10-32T00:00:00Z"), None); // day 32
        assert_eq!(parse_ts(b"2026-10-03T24:00:00Z"), None); // hour 24
        assert_eq!(parse_ts(b"2026-10-03T25:00:00Z"), None);
        assert_eq!(parse_ts(b"2026-10-03T23:60:00Z"), None); // minute 60
        assert_eq!(parse_ts(b"2026-10-03T23:58:60Z"), None); // second 60 (no leap seconds)
        assert_eq!(parse_ts(b"2026-10-03T23:58:61Z"), None);
        // The same checks must apply on the F2 path.
        assert_eq!(parse_ts(b"2026-02-30T00:00:00+02:00"), None);
        assert_eq!(parse_ts(b"2026-10-03T24:00:00+02:00"), None);
    }

    #[test]
    fn month_lengths() {
        // 30-day months: April, June, September, November.
        for (m, last) in [
            (1u32, 31),
            (2, 28),
            (3, 31),
            (4, 30),
            (5, 31),
            (6, 30),
            (7, 31),
            (8, 31),
            (9, 30),
            (10, 31),
            (11, 30),
            (12, 31),
        ] {
            let ok = format!("2026-{:02}-{:02}T00:00:00Z", m, last);
            let bad = format!("2026-{:02}-{:02}T00:00:00Z", m, last + 1);
            assert!(parse_ts(ok.as_bytes()).is_some(), "{} should be valid", ok);
            assert!(parse_ts(bad.as_bytes()).is_none(), "{} should be invalid", bad);
        }
    }

    #[test]
    fn leap_years() {
        // rules.md, generator: "some windows hold February 29, 2028".
        assert!(parse_ts(b"2028-02-29T00:00:00Z").is_some());
        assert_eq!(parse_ts(b"2027-02-29T00:00:00Z"), None);
        assert_eq!(parse_ts(b"2026-02-29T00:00:00Z"), None);
        assert_eq!(parse_ts(b"2025-02-29T00:00:00Z"), None);
        assert!(parse_ts(b"2024-02-29T00:00:00Z").is_some());
        // Century rules: divisible by 100 is not a leap year unless also by 400.
        assert!(parse_ts(b"2000-02-29T00:00:00Z").is_some()); // /400 -> leap
        assert_eq!(parse_ts(b"1900-02-29T00:00:00Z"), None); // /100 not /400
        assert_eq!(parse_ts(b"2100-02-29T00:00:00Z"), None);
        assert_eq!(parse_ts(b"1800-02-29T00:00:00Z"), None);
        assert!(parse_ts(b"1600-02-29T00:00:00Z").is_some()); // /400 -> leap
        assert!(parse_ts(b"2400-02-29T00:00:00Z").is_some());
        assert!(parse_ts(b"0000-02-29T00:00:00Z").is_some()); // year 0 is /400
        assert_eq!(parse_ts(b"0100-02-29T00:00:00Z"), None);
    }

    #[test]
    fn leap_day_epoch_arithmetic() {
        // 2028-02-29T00:00:00Z should be exactly one day after Feb 28 and one before Mar 1.
        let feb28 = parse_ts(b"2028-02-28T00:00:00Z").unwrap();
        let feb29 = parse_ts(b"2028-02-29T00:00:00Z").unwrap();
        let mar01 = parse_ts(b"2028-03-01T00:00:00Z").unwrap();
        assert_eq!(feb29 - feb28, MS_PER_DAY);
        assert_eq!(mar01 - feb29, MS_PER_DAY);
        // In a non-leap year Feb 28 and Mar 1 are adjacent.
        let n28 = parse_ts(b"2027-02-28T00:00:00Z").unwrap();
        let n01 = parse_ts(b"2027-03-01T00:00:00Z").unwrap();
        assert_eq!(n01 - n28, MS_PER_DAY);
        // 2028-02-29 as an independently known epoch value.
        assert_eq!(feb29, 1_835_395_200_000);
    }

    #[test]
    fn year_lengths_and_monotonicity() {
        // A leap year is 366 days, a common year 365.
        let y2028 = parse_ts(b"2028-01-01T00:00:00Z").unwrap();
        let y2029 = parse_ts(b"2029-01-01T00:00:00Z").unwrap();
        assert_eq!((y2029 - y2028) / MS_PER_DAY, 366);
        let y2026 = parse_ts(b"2026-01-01T00:00:00Z").unwrap();
        let y2027 = parse_ts(b"2027-01-01T00:00:00Z").unwrap();
        assert_eq!((y2027 - y2026) / MS_PER_DAY, 365);
        // Every day of the generator's possible window span must be strictly increasing
        // and exactly one day apart (this walks 2025-01-01 .. 2028-12-31, leap day
        // included).
        let mut prev = parse_ts(b"2025-01-01T00:00:00Z").unwrap();
        for y in 2025..=2028 {
            for m in 1..=12u32 {
                let dim = days_in_month(y as i64, m as i64);
                for d in 1..=dim {
                    let s = format!("{:04}-{:02}-{:02}T00:00:00Z", y, m, d);
                    let v = parse_ts(s.as_bytes()).unwrap();
                    if !(y == 2025 && m == 1 && d == 1) {
                        assert_eq!(v - prev, MS_PER_DAY, "gap before {}", s);
                    }
                    prev = v;
                }
            }
        }
    }

    #[test]
    fn time_of_day_arithmetic() {
        let base = parse_ts(b"2026-10-03T00:00:00Z").unwrap();
        assert_eq!(parse_ts(b"2026-10-03T00:00:01Z").unwrap() - base, 1_000);
        assert_eq!(parse_ts(b"2026-10-03T00:01:00Z").unwrap() - base, 60_000);
        assert_eq!(parse_ts(b"2026-10-03T01:00:00Z").unwrap() - base, 3_600_000);
        assert_eq!(
            parse_ts(b"2026-10-03T23:59:59Z").unwrap() - base,
            86_399_000
        );
        // 30 minutes and 30 minutes + 1 ms, the sessionization boundary (fixtures E1/E2).
        let a = parse_ts(b"2026-10-03T10:00:00Z").unwrap();
        let b = parse_ts(b"2026-10-03T10:30:00Z").unwrap();
        assert_eq!(b - a, 1_800_000);
        assert_eq!(parse_ts(b"1791023400001").unwrap() - a, 1_800_001);
    }

    // ---------------------------------------------------------------- Unicode digits

    #[test]
    fn non_ascii_digits_rejected_in_f3() {
        // Fixture E18 `e3`: 1791021600000 written in Arabic-Indic digits is dropped.
        assert_eq!(
            parse_ts("١٧٩١٠٢١٦٠٠٠٠٠".as_bytes()),
            None,
            "Arabic-Indic digits are not digits"
        );
        assert_eq!(parse_ts("٢٥٠".as_bytes()), None);
        assert_eq!(parse_ts("２５０".as_bytes()), None); // full-width
        assert_eq!(parse_ts("１７９１０２１６０００００".as_bytes()), None);
        // Mixed ASCII and non-ASCII must also fail.
        assert_eq!(parse_ts("179102160000٠".as_bytes()), None);
        assert_eq!(parse_ts("٠179102160000".as_bytes()), None);
        // Other Unicode numeric look-alikes.
        assert_eq!(parse_ts("٠٠٠".as_bytes()), None); // Arabic-Indic zeros
        assert_eq!(parse_ts("۱۲۳".as_bytes()), None); // Extended Arabic-Indic
        assert_eq!(parse_ts("१२३".as_bytes()), None); // Devanagari
        assert_eq!(parse_ts("¹²³".as_bytes()), None); // superscripts
        assert_eq!(parse_ts("Ⅻ".as_bytes()), None); // Roman numeral
        assert_eq!(parse_ts("½".as_bytes()), None);
    }

    #[test]
    fn non_ascii_digits_rejected_in_datetime() {
        // Fixture E18 `e8`: a year with a full-width digit is dropped.
        assert_eq!(parse_ts("２026-10-03T10:00:00Z".as_bytes()), None);
        assert_eq!(parse_ts("202６-10-03T10:00:00Z".as_bytes()), None);
        assert_eq!(parse_ts("٢026-10-03T10:00:00Z".as_bytes()), None);
        // ...and in every other numeric position.
        assert_eq!(parse_ts("2026-１0-03T10:00:00Z".as_bytes()), None); // month
        assert_eq!(parse_ts("2026-10-０3T10:00:00Z".as_bytes()), None); // day
        assert_eq!(parse_ts("2026-10-03T１0:00:00Z".as_bytes()), None); // hour
        assert_eq!(parse_ts("2026-10-03T10:０0:00Z".as_bytes()), None); // minute
        assert_eq!(parse_ts("2026-10-03T10:00:０0Z".as_bytes()), None); // second
        assert_eq!(parse_ts("2026-10-03T10:00:00+０2:00".as_bytes()), None); // offset hour
        assert_eq!(parse_ts("2026-10-03T10:00:00+02:０0".as_bytes()), None); // offset minute
    }

    // ---------------------------------------------------------------- robustness

    #[test]
    fn garbage_never_panics() {
        // Every length from 0 to 30 of a few repeated bytes, plus some adversarial
        // shapes. The point is only that nothing panics or indexes out of bounds.
        for filler in [b'0', b'Z', b'-', b'+', b':', b'T', 0x00, 0xFF, b' '] {
            for len in 0..=30usize {
                let v = vec![filler; len];
                let _ = parse_ts(&v);
                let _ = parse_ingest_ts(&v);
            }
        }
        for s in [
            &b"ZZZZZZZZZZZZZZZZZZZZ"[..],
            &b"++++++++++++++++++++++++"[..],
            &b"--------------------Z"[..],
            &b"0000000000000000000Z"[..],
            &b"0000000000000000000+00:00"[..],
            &[0xFF; 20][..],
            &[0xFF; 25][..],
            &[0x00; 20][..],
        ] {
            let _ = parse_ts(s);
            let _ = parse_ingest_ts(s);
        }
    }

    #[test]
    fn twenty_and_twentyfive_digit_f3_values_still_parse_as_f3() {
        // These lengths collide with F1 (20) and F2 (25); dispatch must not be fooled.
        assert_eq!(parse_ts(b"00000000001791071890"), Some(1_791_071_890));
        assert_eq!(parse_ts(b"00000000000001791071890000"), Some(1_791_071_890_000));
        let twenty_five = b"0000000000001791071890000";
        assert_eq!(twenty_five.len(), 25);
        assert_eq!(parse_ts(twenty_five), Some(1_791_071_890_000));
        let twenty = b"00000001791071890000";
        assert_eq!(twenty.len(), 20);
        assert_eq!(parse_ts(twenty), Some(1_791_071_890_000));
    }

    // ---------------------------------------------------------------- ingest_ts helper

    #[test]
    fn parse_ingest_ts_accepts_only_f1() {
        // rules.md: "ingest_ts is always valid F1."
        assert_eq!(parse_ingest_ts(b"2026-10-03T23:58:10Z"), Some(1_791_071_890_000));
        assert_eq!(parse_ingest_ts(b"1970-01-01T00:00:00Z"), Some(0));
        // It agrees with parse_ts on every F1 value.
        for s in [
            &b"2026-10-01T00:00:00Z"[..],
            &b"2028-02-29T12:34:56Z"[..],
            &b"1969-12-31T23:59:59Z"[..],
            &b"2026-10-15T00:00:00Z"[..],
        ] {
            assert_eq!(parse_ingest_ts(s), parse_ts(s));
        }
        // Strictly F1: F2 and F3 are not accepted by the ingest fast path.
        assert_eq!(parse_ingest_ts(b"2026-10-03T19:58:10-04:00"), None);
        assert_eq!(parse_ingest_ts(b"1791071890000"), None);
        // ...and it still rejects everything parse_ts rejects.
        assert_eq!(parse_ingest_ts(b""), None);
        assert_eq!(parse_ingest_ts(b"2026-02-29T00:00:00Z"), None);
        assert_eq!(parse_ingest_ts(b"2026-10-03T23:58:10Z\r"), None);
        assert_eq!(parse_ingest_ts(b"2026-10-03T23:58:10z"), None);
        assert_eq!(parse_ingest_ts(b"2026-10-03 23:58:10Z"), None);
    }

    // ---------------------------------------------------------------- internals

    #[test]
    fn days_from_civil_known_values() {
        // Independently known day numbers relative to 1970-01-01.
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1970, 1, 2), 1);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        assert_eq!(days_from_civil(1972, 2, 29), 789); // first leap day after the epoch
        assert_eq!(days_from_civil(2000, 1, 1), 10957);
        assert_eq!(days_from_civil(2000, 3, 1), 11017);
        // Start of the 400-year era containing the epoch. Derived by hand: 1600-03-01 to
        // 1970-03-01 is 370 years = 370*365 = 135050 days plus leap days — the years
        // 1604..1968 divisible by 4 are 92, less the non-leap centuries 1700/1800/1900,
        // so 89 — giving 135139; 1970-01-01 is 59 days (31 + 28) earlier, so 135080.
        assert_eq!(days_from_civil(1600, 3, 1), -135080);
        // Equivalently: exactly 4 eras before the algorithm's origin 0000-03-01.
        assert_eq!(days_from_civil(1600, 3, 1), 4 * 146097 - 719468);
        assert_eq!(days_from_civil(1, 1, 1), -719162); // proleptic Gregorian day 1
        assert_eq!(days_from_civil(0, 3, 1), -719468); // the algorithm's era origin
    }

    #[test]
    fn is_leap_rules() {
        for y in [1600, 2000, 2400, 2024, 2028, 1996, 0] {
            assert!(is_leap(y), "{} should be leap", y);
        }
        for y in [1700, 1800, 1900, 2100, 2200, 2300, 2025, 2026, 2027, 1] {
            assert!(!is_leap(y), "{} should not be leap", y);
        }
    }

    // ---------------------------------------------------------------- W2b: cache + SWAR
    //
    // Differential tests: the cached / SWAR `parse_ts` must return exactly what the
    // original `parse_ts_uncached` returns, on every input, for every window.

    /// Every literal used anywhere in this test module (extracted mechanically).
    fn existing_test_inputs() -> Vec<&'static [u8]> {
        vec![
        &b""[..],
        &b" "[..],
        &b"   "[..],
        &b" 1791071890000"[..],
        &b" 2026-10-03T23:58:10Z"[..],
        &b"++++++++++++++++++++++++"[..],
        &b"+1791071890000"[..],
        &b"-"[..],
        &b"--------------------Z"[..],
        &b"-1791071890000"[..],
        &b"-5"[..],
        &b"."[..],
        &b"0"[..],
        &b"0000-02-29T00:00:00Z"[..],
        &b"00000000"[..],
        &b"0000000000000000000+00:00"[..],
        &b"0000000000000000000000000000001"[..],
        &b"00000000000000000000000001791071890000"[..],
        &b"0000000000000000000Z"[..],
        &b"00000000000001791071890000"[..],
        &b"0000000000001791071890000"[..],
        &b"00000000001791071890"[..],
        &b"00000001791071890000"[..],
        &b"0001791021600000"[..],
        &b"0001791071890000"[..],
        &b"0100-02-29T00:00:00Z"[..],
        &b"0x1791071890"[..],
        &b"1"[..],
        &b"1 791071890000"[..],
        &b"1.79e12"[..],
        &b"123/456"[..],
        &b"123:456"[..],
        &b"1600-02-29T00:00:00Z"[..],
        &b"179102160000000000000"[..],
        &b"179102160000000000000.0"[..],
        &b"1791021600001"[..],
        &b"1791023400001"[..],
        &b"1791071890000"[..],
        &b"1791071890000 "[..],
        &b"1791071890000.0"[..],
        &b"1791071890000\r"[..],
        &b"1791071890000x"[..],
        &b"1800-02-29T00:00:00Z"[..],
        &b"1900-02-29T00:00:00Z"[..],
        &b"1969-12-31T00:00:00Z"[..],
        &b"1969-12-31T23:59:59Z"[..],
        &b"1970-01-01T00:00:00Z"[..],
        &b"1970-01-01T00:00:01Z"[..],
        &b"1970-01-01T23:59:59Z"[..],
        &b"1970-01-02T00:00:00Z"[..],
        &b"1_791_071_890_000"[..],
        &b"2000-01-01T00:00:00Z"[..],
        &b"2000-02-29T00:00:00Z"[..],
        &b"2001-09-09T01:46:40Z"[..],
        &b"2024-02-29T00:00:00Z"[..],
        &b"2025-01-01T00:00:00Z"[..],
        &b"2025-02-29T00:00:00Z"[..],
        &b"2026-00-01T00:00:00Z"[..],
        &b"2026-01-01T00:00:00Z"[..],
        &b"2026-02-29T00:00:00Z"[..],
        &b"2026-02-30T00:00:00+02:00"[..],
        &b"2026-02-30T00:00:00Z"[..],
        &b"2026-10-00T00:00:00Z"[..],
        &b"2026-10-01T00:00:00Z"[..],
        &b"2026-10-02T10:01:00-23:59"[..],
        &b"2026-10-03"[..],
        &b"2026-10-03 10:00:00"[..],
        &b"2026-10-03 23:58:10Z"[..],
        &b"2026-10-03T00:00:00Z"[..],
        &b"2026-10-03T00:00:01Z"[..],
        &b"2026-10-03T00:01:00Z"[..],
        &b"2026-10-03T01:00:00Z"[..],
        &b"2026-10-03T09:00:00Z\r"[..],
        &b"2026-10-03T09:00:00\r"[..],
        &b"2026-10-03T10:00:00 +02:00"[..],
        &b"2026-10-03T10:00:00*02:00"[..],
        &b"2026-10-03T10:00:00+02-00"[..],
        &b"2026-10-03T10:00:00+0200"[..],
        &b"2026-10-03T10:00:00+02:0"[..],
        &b"2026-10-03T10:00:00+02:000"[..],
        &b"2026-10-03T10:00:00+02:00Z"[..],
        &b"2026-10-03T10:00:00+05:60"[..],
        &b"2026-10-03T10:00:00+05:99"[..],
        &b"2026-10-03T10:00:00+24:00"[..],
        &b"2026-10-03T10:00:00+2:00"[..],
        &b"2026-10-03T10:00:00+99:00"[..],
        &b"2026-10-03T10:00:00+aa:bb"[..],
        &b"2026-10-03T10:00:00-0200"[..],
        &b"2026-10-03T10:00:00-24:00"[..],
        &b"2026-10-03T10:00:00Z"[..],
        &b"2026-10-03T10:30:00Z"[..],
        &b"2026-10-03T12:00:00+00:00"[..],
        &b"2026-10-03T12:00:00+00:01"[..],
        &b"2026-10-03T12:00:00+23:00"[..],
        &b"2026-10-03T12:00:00-00:00"[..],
        &b"2026-10-03T12:00:00-00:59"[..],
        &b"2026-10-03T12:00:00Z"[..],
        &b"2026-10-03T13:58:10-10:00"[..],
        &b"2026-10-03T19:58:10-04:00"[..],
        &b"2026-10-03T19:58:10-04:00\r"[..],
        &b"2026-10-03T19:58:10.500-04:00"[..],
        &b"2026-10-03T23-58-10Z"[..],
        &b"2026-10-03T23:30:00Z"[..],
        &b"2026-10-03T23:58:10"[..],
        &b"2026-10-03T23:58:10 Z"[..],
        &b"2026-10-03T23:58:10+Z"[..],
        &b"2026-10-03T23:58:10,500Z"[..],
        &b"2026-10-03T23:58:10.000Z"[..],
        &b"2026-10-03T23:58:10.500Z"[..],
        &b"2026-10-03T23:58:10.5Z"[..],
        &b"2026-10-03T23:58:10UTC"[..],
        &b"2026-10-03T23:58:10Z"[..],
        &b"2026-10-03T23:58:10Z "[..],
        &b"2026-10-03T23:58:10Z\r"[..],
        &b"2026-10-03T23:58:10z"[..],
        &b"2026-10-03T23:58:60Z"[..],
        &b"2026-10-03T23:58:61Z"[..],
        &b"2026-10-03T23:59:59Z"[..],
        &b"2026-10-03T23:60:00Z"[..],
        &b"2026-10-03T24:00:00+02:00"[..],
        &b"2026-10-03T24:00:00Z"[..],
        &b"2026-10-03T25:00:00Z"[..],
        &b"2026-10-03t23:58:10Z"[..],
        &b"2026-10-03t23:58:10z"[..],
        &b"2026-10-04T01:30:00+02:00"[..],
        &b"2026-10-04T01:58:10+02:00"[..],
        &b"2026-10-04T05:28:10+05:30"[..],
        &b"2026-10-04T09:59:00+23:59"[..],
        &b"2026-10-04T11:58:10+12:00"[..],
        &b"2026-10-15T00:00:00Z"[..],
        &b"2026-10-31T23:30:00-01:00"[..],
        &b"2026-10-31T23:30:00Z"[..],
        &b"2026-10-32T00:00:00Z"[..],
        &b"2026-11-01T00:30:00+01:00"[..],
        &b"2026-11-01T00:30:00Z"[..],
        &b"2026-13-01T00:00:00Z"[..],
        &b"2026-99-01T00:00:00Z"[..],
        &b"2026/10/03T23:58:10Z"[..],
        &b"20261003T235810Z"[..],
        &b"2027-01-01T00:00:00Z"[..],
        &b"2027-02-28T00:00:00Z"[..],
        &b"2027-02-29T00:00:00Z"[..],
        &b"2027-03-01T00:00:00Z"[..],
        &b"2028-01-01T00:00:00Z"[..],
        &b"2028-02-28T00:00:00Z"[..],
        &b"2028-02-29T00:00:00Z"[..],
        &b"2028-02-29T12:34:56Z"[..],
        &b"2028-03-01T00:00:00Z"[..],
        &b"2029-01-01T00:00:00Z"[..],
        &b"2100-02-29T00:00:00Z"[..],
        &b"23:58:10Z"[..],
        &b"2400-02-29T00:00:00Z"[..],
        &b"86400000"[..],
        &b"9223372036854775807"[..],
        &b"9223372036854775808"[..],
        &b"99999999999999999999999999999999"[..],
        &b"99999999999999999999999999999999x"[..],
        &b"NaN"[..],
        &b"ZZZZZZZZZZZZZZZZZZZZ"[..],
        &b"\r"[..],
        &b"\r2026-10-03T09:00:00Z"[..],
        &b"\t2026-10-03T23:58:10Z"[..],
        &b"null"[..],
        "179102160000٠".as_bytes(),
        "2026-10-03T10:00:00+02:０0".as_bytes(),
        "2026-10-03T10:00:00+０2:00".as_bytes(),
        "2026-10-03T10:00:０0Z".as_bytes(),
        "2026-10-03T10:０0:00Z".as_bytes(),
        "2026-10-03T１0:00:00Z".as_bytes(),
        "2026-10-０3T10:00:00Z".as_bytes(),
        "2026-１0-03T10:00:00Z".as_bytes(),
        "202６-10-03T10:00:00Z".as_bytes(),
        "¹²³".as_bytes(),
        "½".as_bytes(),
        "٠179102160000".as_bytes(),
        "٠٠٠".as_bytes(),
        "١٧٩١٠٢١٦٠٠٠٠٠".as_bytes(),
        "٢026-10-03T10:00:00Z".as_bytes(),
        "٢٥٠".as_bytes(),
        "۱۲۳".as_bytes(),
        "१२३".as_bytes(),
        "Ⅻ".as_bytes(),
        "１７９１０２１６０００００".as_bytes(),
        "２026-10-03T10:00:00Z".as_bytes(),
        "２５０".as_bytes(),
        ]
    }

    fn ms(s: &str) -> i64 {
        parse_ts_uncached(s.as_bytes()).unwrap()
    }

    /// Windows that stress the cache: the fixture window, leap-day windows, month and
    /// year boundaries, a long window (forces direct-mapped slot collisions), the epoch,
    /// pre-epoch, and the ends of the 4-digit-year range.
    fn test_windows() -> Vec<(i64, i64)> {
        vec![
            (ms("2026-10-01T00:00:00Z"), ms("2026-10-15T00:00:00Z")), // fixtures
            (ms("2028-02-20T00:00:00Z"), ms("2028-03-05T00:00:00Z")), // holds Feb 29 2028
            (ms("2028-02-29T00:00:00Z"), ms("2028-03-14T00:00:00Z")),
            (ms("2027-02-20T00:00:00Z"), ms("2027-03-06T00:00:00Z")), // no Feb 29
            (ms("2100-02-20T00:00:00Z"), ms("2100-03-06T00:00:00Z")), // century non-leap
            (ms("2026-10-25T00:00:00Z"), ms("2026-11-08T00:00:00Z")), // month boundary
            (ms("2026-12-25T00:00:00Z"), ms("2027-01-08T00:00:00Z")), // year boundary
            (ms("2026-12-31T12:34:56Z"), ms("2027-01-01T00:00:01Z")), // unaligned
            (ms("2026-01-01T00:00:00Z"), ms("2027-06-01T00:00:00Z")), // collisions
            (ms("1970-01-01T00:00:00Z"), ms("1970-01-15T00:00:00Z")), // epoch
            (ms("1969-12-20T00:00:00Z"), ms("1970-01-03T00:00:00Z")), // pre-epoch
            (ms("0000-01-01T00:00:00Z"), ms("0000-01-15T00:00:00Z")), // year 0 edge
            (ms("9999-12-20T00:00:00Z"), ms("9999-12-31T00:00:00Z")), // year 9999 edge
            (-5 * MS_PER_DAY * 365 * 3000, -5 * MS_PER_DAY * 365 * 3000 + 14 * MS_PER_DAY),
            (0, i64::MAX), // absurd: capped, must not panic
        ]
    }

    fn assert_same(cache: Option<&DateCache>, b: &[u8]) {
        assert_eq!(
            parse_ts_with(cache, b),
            parse_ts_uncached(b),
            "mismatch on {:?} ({:?})",
            String::from_utf8_lossy(b),
            b
        );
    }

    #[test]
    fn civil_from_days_inverts_days_from_civil() {
        let lo = days_from_civil(0, 1, 1) - 400;
        let hi = days_from_civil(9999, 12, 31) + 400;
        let mut d = lo;
        while d <= hi {
            let (y, m, day) = civil_from_days(d);
            assert!((1..=12).contains(&m) && day >= 1 && day <= days_in_month(y, m));
            assert_eq!(days_from_civil(y, m, day), d, "day {}", d);
            d += 1;
        }
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        assert_eq!(
            civil_from_days(ms("2028-02-29T00:00:00Z") / MS_PER_DAY),
            (2028, 2, 29)
        );
    }

    #[test]
    fn cache_entries_are_exact_and_cover_the_window() {
        for (ws, we) in test_windows() {
            let c = DateCache::for_window(ws, we);
            let mut n = 0;
            for e in c.table.iter().filter(|e| e.k8 != 0) {
                n += 1;
                let mut key = [0u8; 10];
                key[0..8].copy_from_slice(&e.k8.to_le_bytes());
                key[8..10].copy_from_slice(&e.k2.to_le_bytes());
                let mut full = key.to_vec();
                full.extend_from_slice(b"T00:00:00Z");
                let (_, days) = parse_civil(&full).expect("cached key must be a valid date");
                assert_eq!(days, e.days as i64);
                assert_eq!(slot_of(&key), slot_of(&full));
                assert!(std::ptr::eq(&c.table[slot_of(&key)], e), "entry in wrong slot");
            }
            // For a normal ~2-week window, every date in [start-8d, end+7d] is cached.
            if we - ws <= 31 * MS_PER_DAY && ws >= 0 && we < ms("9999-12-01T00:00:00Z") {
                let (a, b) = (ws.div_euclid(MS_PER_DAY) - 8, we.div_euclid(MS_PER_DAY) + 7);
                assert_eq!(n as i64, b - a + 1, "window {:?}", (ws, we));
                for d in a..=b {
                    let (y, m, dd) = civil_from_days(d);
                    let s = format!("{y:04}-{m:02}-{dd:02}T00:00:00Z");
                    assert_eq!(c.lookup(s.as_bytes()), Some(d), "{}", s);
                }
            }
        }
    }

    #[test]
    fn empty_slots_never_match() {
        let c = DateCache::for_window(ms("2026-10-01T00:00:00Z"), ms("2026-10-15T00:00:00Z"));
        let mut z = [0u8; 20];
        z[19] = b'Z';
        assert_same(Some(&c), &z);
        z[4] = b'-';
        z[7] = b'-';
        z[10] = b'T';
        z[13] = b':';
        z[16] = b':';
        assert_same(Some(&c), &z);
        let mut z2 = [0u8; 25];
        z2[19] = b'+';
        z2[22] = b':';
        assert_same(Some(&c), &z2);
    }

    #[test]
    fn differential_existing_inputs_all_windows() {
        let inputs = existing_test_inputs();
        assert!(inputs.len() > 150);
        for (ws, we) in test_windows() {
            let c = DateCache::for_window(ws, we);
            for b in &inputs {
                assert_same(Some(&c), b);
            }
        }
        for b in &inputs {
            assert_same(None, b);
        }
    }

    #[test]
    fn differential_every_date_near_each_window() {
        let times = [
            "00:00:00", "23:59:59", "12:34:56", "24:00:00", "23:60:00", "23:59:60",
            "99:99:99", "1a:00:00", "12-00:00", "12:00-00", "0:000:00", "/0:00:00",
            "::::::::",
        ];
        let zones = [
            "Z", "+00:00", "-00:00", "+23:59", "-23:59", "+24:00", "+05:60", "+0500x",
            "-10:00", "+12:00", "z", "*02:00",
        ];
        for (ws, we) in test_windows() {
            let c = DateCache::for_window(ws, we);
            let a = ws.div_euclid(MS_PER_DAY).saturating_sub(40);
            let b = a.saturating_add(we.div_euclid(MS_PER_DAY).saturating_sub(a).min(120) + 80);
            for d in a..=b {
                let (y, m, dd) = civil_from_days(d);
                if !(0..=9999).contains(&y) {
                    continue;
                }
                for t in times {
                    for z in zones {
                        let s = format!("{y:04}-{m:02}-{dd:02}T{t}{z}");
                        assert_same(Some(&c), s.as_bytes());
                        let s = format!("{y:04}-{m:02}-{dd:02}t{t}{z}");
                        assert_same(Some(&c), s.as_bytes());
                    }
                }
                // Impossible days adjacent to real ones (Feb 29/30, day 31/32, day 00).
                for bad in ["29", "30", "31", "32", "00", "3a", "٢"] {
                    let s = format!("{y:04}-{m:02}-{bad}T10:00:00Z");
                    assert_same(Some(&c), s.as_bytes());
                }
            }
        }
    }

    #[test]
    fn differential_non_ascii_digits() {
        let c = DateCache::for_window(ms("2026-10-01T00:00:00Z"), ms("2026-10-15T00:00:00Z"));
        let base = "2026-10-03T10:00:00Z";
        let base2 = "2026-10-03T10:00:00+02:00";
        let f3 = "1791021600000";
        for look in ["٢", "２", "۲", "२", "²", "\u{0}", "\r", " "] {
            for src in [base, base2, f3] {
                for i in 0..src.len() {
                    // Replace byte i by the look-alike (changes length if multi-byte),
                    // and also overwrite in place to keep the length.
                    let mut v = src.as_bytes().to_vec();
                    v.splice(i..i + 1, look.bytes());
                    assert_same(Some(&c), &v);
                    let mut w = src.as_bytes().to_vec();
                    for (k, byte) in look.bytes().enumerate() {
                        if i + k < w.len() {
                            w[i + k] = byte;
                        }
                    }
                    assert_same(Some(&c), &w);
                }
            }
        }
    }

    #[test]
    fn swar_f3_matches_scalar_on_all_lengths() {
        let c = DateCache::for_window(ms("2026-10-01T00:00:00Z"), ms("2026-10-15T00:00:00Z"));
        // Every byte value in every position, for every length 1..=26.
        for len in 1..=26usize {
            let digits: Vec<u8> = (0..len).map(|i| b"1791021600000987654321"[i % 22]).collect();
            assert_same(Some(&c), &digits);
            for pos in 0..len {
                for byte in 0..=255u8 {
                    let mut v = digits.clone();
                    v[pos] = byte;
                    assert_same(Some(&c), &v);
                }
            }
        }
        for s in [
            "00000000", "99999999", "999999999999999999", "000000000000000000",
            "0000000000000000001", "9223372036854775807", "9223372036854775808",
            "1791021600000", "0001791021600000", "12345678",
        ] {
            assert_same(Some(&c), s.as_bytes());
            assert_same(None, s.as_bytes());
        }
        assert_eq!(parse_ts_with(None, b"999999999999999999"), Some(999_999_999_999_999_999));
        assert_eq!(parse_ts_with(None, b"12345678"), Some(12_345_678));
        assert_eq!(parse_ts_with(None, b"0001791021600000"), Some(1_791_021_600_000));
    }

    #[test]
    fn swar_hms_every_byte_in_every_position() {
        let c = DateCache::for_window(ms("2026-10-01T00:00:00Z"), ms("2026-10-15T00:00:00Z"));
        for src in ["2026-10-03T19:58:10Z", "2026-10-03T19:58:10-04:00"] {
            for pos in 0..src.len() {
                for byte in 0..=255u8 {
                    let mut v = src.as_bytes().to_vec();
                    v[pos] = byte;
                    assert_same(Some(&c), &v);
                }
            }
        }
        // Every hh:mi:ss combination 00..=99 on a cached date.
        for h in 0..100 {
            for m in (0..100).step_by(7).chain([59, 60]) {
                for s in [0, 1, 59, 60, 99] {
                    let t = format!("2026-10-03T{h:02}:{m:02}:{s:02}Z");
                    assert_same(Some(&c), t.as_bytes());
                }
            }
        }
    }

    /// xorshift64 — deterministic, dependency-free.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    #[test]
    fn differential_fuzz_mutated_timestamps() {
        let windows = test_windows();
        let caches: Vec<DateCache> =
            windows.iter().map(|&(a, b)| DateCache::for_window(a, b)).collect();
        let pool: &[u8] = b"0123456789-T:Z+tz /.,\r\x00\xff\xd9\xa2\xef\xbc\x92\x3a\x2f\xb0\xb9\x40";
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut hits = 0u64;
        const N: usize = 400_000;
        for _ in 0..N {
            let wi = rng.below(windows.len() as u64) as usize;
            let (ws, we) = windows[wi];
            let c = &caches[wi];
            let span = (we.saturating_sub(ws) / MS_PER_DAY).clamp(1, 400);
            let day = ws.div_euclid(MS_PER_DAY) - 12 + rng.below(span as u64 + 24) as i64;
            let (y, m, d) = civil_from_days(day);
            let y = y.clamp(0, 9999);
            // Mostly-valid time fields, sometimes just out of range.
            let h = if rng.below(20) == 0 { 24 } else { rng.below(24) };
            let mi = if rng.below(20) == 0 { 60 } else { rng.below(60) };
            let se = if rng.below(20) == 0 { 60 } else { rng.below(60) };
            let mut s: Vec<u8> = match rng.below(3) {
                0 => format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{se:02}Z").into_bytes(),
                1 => {
                    let sign = if rng.below(2) == 0 { '+' } else { '-' };
                    let oh = rng.below(25);
                    let om = if rng.below(10) == 0 { 60 } else { rng.below(60) };
                    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{se:02}{sign}{oh:02}:{om:02}")
                        .into_bytes()
                }
                _ => {
                    if rng.below(4) == 0 {
                        // random digit string of random length (overflow, 20/25 lengths)
                        let len = 1 + rng.below(30) as usize;
                        (0..len).map(|_| b'0' + rng.below(10) as u8).collect()
                    } else {
                        let v = ws
                            .saturating_add((rng.next() % (30 * MS_PER_DAY as u64)) as i64)
                            .saturating_sub(10 * MS_PER_DAY)
                            .max(0);
                        let zeros = if rng.below(3) == 0 { rng.below(9) as usize } else { 0 };
                        format!("{}{}", "0".repeat(zeros), v).into_bytes()
                    }
                }
            };
            // Mutate: 0..=3 operations.
            let nmut = match rng.below(8) {
                0..=2 => 0,
                3..=5 => 1,
                6 => 2,
                _ => 3,
            };
            for _ in 0..nmut {
                match rng.below(6) {
                    0..=2 if !s.is_empty() => {
                        let i = rng.below(s.len() as u64) as usize;
                        s[i] = if rng.below(4) == 0 {
                            rng.next() as u8
                        } else {
                            pool[rng.below(pool.len() as u64) as usize]
                        };
                    }
                    3 if !s.is_empty() => {
                        // Replace a byte with a multi-byte digit look-alike.
                        let i = rng.below(s.len() as u64) as usize;
                        let look: &[u8] =
                            if rng.below(2) == 0 { "٢".as_bytes() } else { "２".as_bytes() };
                        s.splice(i..i + 1, look.iter().copied());
                    }
                    4 if !s.is_empty() => {
                        let i = rng.below(s.len() as u64) as usize;
                        s.remove(i);
                    }
                    _ => {
                        let i = rng.below(s.len() as u64 + 1) as usize;
                        s.insert(i, pool[rng.below(pool.len() as u64) as usize]);
                    }
                }
            }
            let got = parse_ts_with(Some(c), &s);
            assert_eq!(
                got,
                parse_ts_uncached(&s),
                "fuzz mismatch on {:?} ({:?}) window {:?}",
                String::from_utf8_lossy(&s),
                s,
                (ws, we)
            );
            if s.len() >= 19 && s[7] == b'-' && c.lookup(&s).is_some() {
                hits += 1;
            }
        }
        // The fuzz must actually exercise the cache-hit path, not only misses.
        assert!(hits > (N / 10) as u64, "only {} cache hits", hits);
    }

    #[test]
    fn global_hint_path_matches_uncached() {
        // Other tests may install other windows concurrently; any installed cache is exact,
        // so this holds regardless of interleaving.
        let (ws, we) = (ms("2026-10-01T00:00:00Z"), ms("2026-10-15T00:00:00Z"));
        set_window_hint(ws, we);
        set_window_hint(ws, we); // same window: no-op
        for b in existing_test_inputs() {
            assert_eq!(parse_ts(b), parse_ts_uncached(b));
        }
        set_window_hint(ms("2028-02-20T00:00:00Z"), ms("2028-03-05T00:00:00Z"));
        for b in existing_test_inputs() {
            assert_eq!(parse_ts(b), parse_ts_uncached(b));
        }
        assert_eq!(parse_ts(b"2028-02-29T12:00:00Z"), Some(ms("2028-02-29T12:00:00Z")));
        assert_eq!(parse_ts(b"2027-02-29T12:00:00Z"), None);
    }
}
