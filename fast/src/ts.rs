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
const F1_LEN: usize = 20; // `2026-10-03T23:58:10Z`
const F2_LEN: usize = 25; // `2026-10-03T19:58:10-04:00`

const MS_PER_SEC: i64 = 1_000;
const MS_PER_MIN: i64 = 60 * MS_PER_SEC;
const MS_PER_HOUR: i64 = 60 * MS_PER_MIN;
const MS_PER_DAY: i64 = 24 * MS_PER_HOUR;

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
pub fn parse_ts(b: &[u8]) -> Option<i64> {
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
}
