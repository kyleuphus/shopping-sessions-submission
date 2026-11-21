//! Byte-level CSV line and field splitting, implementing the `rules.md` dialect.
//!
//! This module is deliberately dependency-free and allocation-free per line: it
//! only ever hands back subslices of the caller's bytes.
//!
//! # The dialect, straight from `rules.md`
//!
//! > Comma-separated, UTF-8, `\n` line endings. Only `\n` ends a line: a `\r` is
//! > an ordinary character of the field it is in [...]. Any field may be wrapped
//! > in double quotes, and the quotes are syntax, not part of the value: `"u1"`
//! > is `u1`, and `""` is an empty field. No value ever holds a comma, a quote
//! > or a newline, quoted or not: every comma separates fields. An empty field
//! > means null. A line with the wrong number of fields, or with a quote
//! > anywhere except around a whole field, is malformed and is dropped. An empty
//! > line is malformed too. Spaces are part of a value, quoted or not [...].
//!
//! Three consequences drive the whole implementation:
//!
//! 1. **Every comma separates fields.** Because no *value* can contain a comma —
//!    quoted or not — there is no need for a quote-aware state machine. We split
//!    on every comma, unconditionally. This is the single biggest simplification
//!    available and it is stated explicitly in the spec.
//! 2. **Quotes are syntax only, and only around a *whole* field.** A field is
//!    either entirely unquoted and quote-free, or it is `"`...`"` with a
//!    quote-free interior. Anything else makes the *whole line* malformed
//!    (`u1,u5"` is malformed; so is `"u1`; so is a lone `"`).
//! 3. **Nothing is trimmed, folded or normalised.** Not spaces, not `\r`, not
//!    case. A `\r` left over from a `\r\n` line ending belongs to the last field
//!    and must survive (fixture E20 relies on this: a purchase whose price
//!    becomes `1000\r` has to be rejected *downstream* by the price validator,
//!    which means the `\r` has to still be there).
//!
//! # Empty fields are null
//!
//! `rules.md`: "An empty field means null", and `""` (quoted empty) "is an empty
//! field". This module represents both as a **zero-length subslice**. So after a
//! successful [`split_fields`], a field is null exactly when `field.is_empty()`.
//! [`is_null`] is provided to make that explicit at call sites. There is no other
//! null representation and no distinction between `` and `""` — per the spec they
//! are the same thing.
//!
//! Note that *which* empty fields are fatal is not this module's business. For
//! links, "lines where either field is empty" are dropped; for events, an empty
//! `event_id` is dropped; but `price_cents` may legitimately be empty on a
//! `view`. [`split_fields`] therefore accepts empty fields and leaves the
//! emptiness policy to each caller.

/// Returns true when a field returned by [`split_fields`] is null.
///
/// Per `rules.md` an empty field means null, and `""` is an empty field, so both
/// arrive here as a zero-length slice. Nothing is trimmed first: a field holding
/// a single space is **not** null.
#[inline(always)]
pub fn is_null(field: &[u8]) -> bool {
    field.is_empty()
}

/// Strip the surrounding quotes from one already-comma-split field.
///
/// Returns `None` if the field holds a quote anywhere other than as a matched
/// pair wrapping the entire field — which, per `rules.md`, makes the whole line
/// malformed.
///
/// Cases, all derived from the spec:
/// - no `"` at all          -> the field is its own value (spaces included)
/// - `"abc"`                -> `abc`
/// - `""`                   -> empty field (null)
/// - `"`                     -> malformed (a lone quote is not a matched pair)
/// - `"abc`, `abc"`, `a"bc` -> malformed (quote not around a whole field)
/// - `"a"b"`                -> malformed (quote inside the value)
#[inline]
pub(crate) fn unquote<'a>(field: &'a [u8]) -> Option<&'a [u8]> {
    // `first()` rather than indexing: the field may be empty (a null), which is
    // fine and needs no unquoting.
    if field.first() == Some(&b'"') {
        let n = field.len();
        // n >= 2 rules out the lone `"` field, which has an opening quote and no
        // closing one and so is malformed.
        if n >= 2 && field[n - 1] == b'"' {
            let inner = &field[1..n - 1];
            // No value ever holds a quote, so a quote inside the wrapper is not
            // data; the line is malformed.
            if inner.contains(&b'"') {
                None
            } else {
                Some(inner)
            }
        } else {
            None
        }
    } else if field.contains(&b'"') {
        // A quote that is not at the start cannot be a whole-field wrapper, so
        // `u1,u5"` and `u"1` are both malformed.
        None
    } else {
        Some(field)
    }
}

/// Split one CSV line into its fields, applying the `rules.md` dialect.
///
/// `line` is the bytes of one line, WITHOUT the terminating `\n`. (Any `\r` is
/// *not* a terminator and must be left in place by the caller — see
/// [`data_lines`].)
///
/// On success, `out` is filled with one subslice per field, with surrounding
/// quotes stripped, and `true` is returned. The subslices borrow from `line`;
/// nothing is copied or allocated beyond growing `out`.
///
/// Returns `false` if the line is malformed per `rules.md`, in which case the
/// caller drops the line. `out` contents are unspecified on failure.
///
/// `expected_fields` is the field count required for this file type (7 for
/// events, 4 for clicks, 2 for links and the two lookup tables); a line with any
/// other count is malformed.
///
/// Malformed means any of:
/// - an empty line (zero bytes) — explicitly malformed per the spec, even though
///   it would otherwise read as a single null field;
/// - a field count other than `expected_fields`;
/// - a quote anywhere except as a matched pair wrapping a whole field.
///
/// `out` is cleared on entry, so one buffer can be reused for every line of a
/// multi-gigabyte scan without reallocating.
///
/// # Implementation: SIMD comma scan
///
/// The hot loop finds every comma and notes whether any `"` occurs. On x86_64
/// with AVX2 (compile-time `target_feature`) it does that 32 bytes at a time; on
/// aarch64 (NEON is always present) 16 bytes at a time. Each chunk yields a
/// bitmask with one bit per comma; the bits are visited lowest-first, which is
/// exactly left-to-right order, and each one is handled with the *same* logic
/// as the scalar loop (same push order, same too-many-fields bail-out). The
/// bytes left over after the last whole chunk go through the scalar loop, and
/// so does the whole line on any other target. Quote handling (`unquote`) is
/// shared and untouched. So the result — the bool AND the contents of `out`,
/// even on failure — is identical to [`split_fields_scalar`], which is kept as
/// the reference and differentially tested against this function.
pub fn split_fields<'a>(line: &'a [u8], expected_fields: usize, out: &mut Vec<&'a [u8]>) -> bool {
    out.clear();

    if line.is_empty() || expected_fields == 0 {
        return false;
    }
    let need_commas = expected_fields - 1;
    let n = line.len();
    let mut start = 0usize;
    let mut saw_quote = false;

    // SIMD prefix: processes whole chunks, returns the index where the scalar
    // tail must resume, or None if the line already has too many fields.
    let mut i = match simd_prefix(line, need_commas, out, &mut start, &mut saw_quote) {
        Some(j) => j,
        None => return false,
    };

    // Scalar tail (or the whole line on targets without a SIMD path). Same
    // logic as `split_fields_scalar`.
    while i < n {
        let b = line[i];
        if b == b',' {
            if out.len() == need_commas {
                return false;
            }
            out.push(&line[start..i]);
            start = i + 1;
        } else if b == b'"' {
            saw_quote = true;
        }
        i += 1;
    }

    if out.len() != need_commas {
        return false;
    }
    out.push(&line[start..n]);

    if saw_quote {
        for f in out.iter_mut() {
            match unquote(*f) {
                Some(v) => *f = v,
                None => return false,
            }
        }
    }

    true
}

/// Dispatch to the SIMD scanner compiled for this target (exactly one of these
/// three definitions exists). The fallback scans nothing and returns 0, leaving
/// the whole line to the scalar loop.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn simd_prefix<'a>(
    line: &'a [u8],
    need_commas: usize,
    out: &mut Vec<&'a [u8]>,
    start: &mut usize,
    saw_quote: &mut bool,
) -> Option<usize> {
    // SAFETY: avx2 is enabled at compile time (target_feature cfg).
    unsafe { scan_avx2(line, need_commas, out, start, saw_quote) }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn simd_prefix<'a>(
    line: &'a [u8],
    need_commas: usize,
    out: &mut Vec<&'a [u8]>,
    start: &mut usize,
    saw_quote: &mut bool,
) -> Option<usize> {
    // SAFETY: NEON is mandatory on aarch64.
    unsafe { scan_neon(line, need_commas, out, start, saw_quote) }
}

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    target_arch = "aarch64"
)))]
#[inline(always)]
fn simd_prefix<'a>(
    _line: &'a [u8],
    _need_commas: usize,
    _out: &mut Vec<&'a [u8]>,
    _start: &mut usize,
    _saw_quote: &mut bool,
) -> Option<usize> {
    Some(0)
}

/// Handle all commas of one chunk. `mask` has exactly one set bit per comma,
/// at bit position `(byte offset within chunk) * shift_unit` — 1 for AVX2,
/// 4 for the NEON nibble mask. Returns false on too many fields.
#[cfg(any(all(target_arch = "x86_64", target_feature = "avx2"), target_arch = "aarch64"))]
#[inline(always)]
fn push_commas<'a>(
    line: &'a [u8],
    base: usize,
    mut mask: u64,
    shift: u32,
    need_commas: usize,
    out: &mut Vec<&'a [u8]>,
    start: &mut usize,
) -> bool {
    while mask != 0 {
        let pos = base + (mask.trailing_zeros() >> shift) as usize;
        mask &= mask - 1; // clear lowest set bit
        if out.len() == need_commas {
            return false;
        }
        out.push(&line[*start..pos]);
        *start = pos + 1;
    }
    true
}

/// AVX2: 32-byte chunks. Returns the index of the first unprocessed byte, or
/// None when the line has more than `need_commas` commas.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn scan_avx2<'a>(
    line: &'a [u8],
    need_commas: usize,
    out: &mut Vec<&'a [u8]>,
    start: &mut usize,
    saw_quote: &mut bool,
) -> Option<usize> {
    use std::arch::x86_64::*;
    let n = line.len();
    let ptr = line.as_ptr();
    let v_comma = _mm256_set1_epi8(b',' as i8);
    let v_quote = _mm256_set1_epi8(b'"' as i8);
    // OR of all quote-compare results; tested once after the loop.
    let mut qacc = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 32 <= n {
        // In bounds: i + 32 <= n. Unaligned load.
        let chunk = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let cmask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, v_comma)) as u32;
        qacc = _mm256_or_si256(qacc, _mm256_cmpeq_epi8(chunk, v_quote));
        if !push_commas(line, i, cmask as u64, 0, need_commas, out, start) {
            return None;
        }
        i += 32;
    }
    if _mm256_testz_si256(qacc, qacc) == 0 {
        *saw_quote = true;
    }
    Some(i)
}

/// NEON: 16-byte chunks. `vceqq_u8` gives 0xFF/0x00 per byte; narrowing each
/// u16 lane right-shifted by 4 (`shrn #4`) packs that into a u64 with 4 bits
/// per input byte. Masking with 0x8888... keeps one bit per byte (bit 4k+3 for
/// byte k), so `trailing_zeros() >> 2` is the byte offset.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn scan_neon<'a>(
    line: &'a [u8],
    need_commas: usize,
    out: &mut Vec<&'a [u8]>,
    start: &mut usize,
    saw_quote: &mut bool,
) -> Option<usize> {
    use std::arch::aarch64::*;
    let n = line.len();
    let ptr = line.as_ptr();
    let v_comma = vdupq_n_u8(b',');
    let v_quote = vdupq_n_u8(b'"');
    let mut qacc = vdupq_n_u8(0);
    let mut i = 0usize;
    while i + 16 <= n {
        // In bounds: i + 16 <= n.
        let chunk = vld1q_u8(ptr.add(i));
        let ceq = vceqq_u8(chunk, v_comma);
        qacc = vorrq_u8(qacc, vceqq_u8(chunk, v_quote));
        let nib = vshrn_n_u16::<4>(vreinterpretq_u16_u8(ceq));
        let cmask = vget_lane_u64::<0>(vreinterpret_u64_u8(nib)) & 0x8888_8888_8888_8888;
        if !push_commas(line, i, cmask, 2, need_commas, out, start) {
            return None;
        }
        i += 16;
    }
    if vmaxvq_u8(qacc) != 0 {
        *saw_quote = true;
    }
    Some(i)
}

/// The original byte-at-a-time implementation, kept verbatim as the reference
/// for the differential tests of [`split_fields`].
#[cfg_attr(not(test), allow(dead_code))]
fn split_fields_scalar<'a>(line: &'a [u8], expected_fields: usize, out: &mut Vec<&'a [u8]>) -> bool {
    out.clear();

    // "An empty line is malformed too." Checked before anything else, because a
    // zero-byte line does contain exactly one (empty) field and would otherwise
    // be accepted when expected_fields == 1.
    if line.is_empty() {
        return false;
    }
    // Defensive: no real file type has zero columns, and every line has at least
    // one field, so this could never match.
    if expected_fields == 0 {
        return false;
    }

    // Single pass over the line: cut at every comma (no quote-awareness needed,
    // since no value can contain a comma) and note whether the line contains a
    // quote at all. 80% of fields are unquoted, so the `saw_quote` flag lets the
    // common line skip the per-field quote handling entirely.
    let n = line.len();
    let mut start = 0usize;
    let mut saw_quote = false;
    let mut i = 0usize;
    while i < n {
        let b = line[i];
        if b == b',' {
            // Bail out as soon as there are too many fields rather than pushing
            // them all; `out` is unspecified on failure so this is safe.
            if out.len() == expected_fields - 1 {
                return false;
            }
            out.push(&line[start..i]);
            start = i + 1;
        } else if b == b'"' {
            saw_quote = true;
        }
        i += 1;
    }

    // The text after the last comma is the final field. If we have not seen
    // exactly expected_fields - 1 commas by now, the count is wrong.
    if out.len() != expected_fields - 1 {
        return false;
    }
    out.push(&line[start..n]);

    if saw_quote {
        for f in out.iter_mut() {
            match unquote(*f) {
                Some(v) => *f = v,
                None => return false,
            }
        }
    }

    true
}

/// Iterate the lines of a file's bytes, splitting ONLY on `\n`.
///
/// The first line is the header and is skipped **by position**, never by
/// matching its content: fixture L3's `links/part-00000.csv` contains a data row
/// byte-identical to the header (`user_id_a,user_id_b`), and that row is real
/// data that joins the ID `user_id_a` to the ID `user_id_b`. Content-matching
/// would silently drop it and change the answer.
///
/// `\r` is never treated as a line ending, so a file written with `\r\n` yields
/// lines whose last field ends in `\r`. That is deliberate (fixture E20).
///
/// Robust to the shapes a generated dataset actually contains:
/// - an empty file yields nothing (no underflow, no panic);
/// - a header-only file yields nothing (1 to 3 such files per dataset);
/// - a file whose last line has no trailing `\n` still yields that last line;
/// - a trailing `\n` at end of file does not produce a phantom empty last line,
///   but a genuine blank line in the middle of the file *is* yielded (it is a
///   malformed line, which is the caller's business, not ours).
pub fn data_lines(bytes: &[u8]) -> impl Iterator<Item = &[u8]> {
    // Drop one trailing newline, if present, so that "a\n" is one line rather
    // than a line plus an empty tail. Only one: "a\n\n" is two lines, the second
    // of them empty (and malformed).
    let body = match bytes.last() {
        Some(&b'\n') => &bytes[..bytes.len() - 1],
        _ => bytes,
    };

    // `split` on an empty slice yields a single empty item, so `skip(1)` leaves
    // nothing for an empty or header-only file. No arithmetic that can underflow.
    body.split(|&b| b == b'\n').skip(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience: `Some(fields)` when the line is well-formed, `None` when it
    /// is malformed and the caller must drop it.
    ///
    /// Every call is also a differential test: the SIMD `split_fields` must
    /// agree exactly with `split_fields_scalar` on this input.
    fn split(line: &[u8], expected_fields: usize) -> Option<Vec<&[u8]>> {
        assert_same(line, expected_fields);
        let mut out = Vec::new();
        if split_fields(line, expected_fields, &mut out) {
            Some(out)
        } else {
            None
        }
    }

    // ---------------------------------------------------------------- basics

    #[test]
    fn plain_two_field_line() {
        assert_eq!(split(b"u4,u1", 2), Some(vec![&b"u4"[..], &b"u1"[..]]));
    }

    #[test]
    fn plain_single_field_line() {
        assert_eq!(split(b"u1", 1), Some(vec![&b"u1"[..]]));
    }

    #[test]
    fn full_seven_field_event_line() {
        // Shape of events/*.csv:
        // event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents
        let line = b"e2,u1,2026-10-03T10:05:00Z,2026-10-03T12:00:00Z,purchase,p1,1000";
        assert_eq!(
            split(line, 7),
            Some(vec![
                &b"e2"[..],
                &b"u1"[..],
                &b"2026-10-03T10:05:00Z"[..],
                &b"2026-10-03T12:00:00Z"[..],
                &b"purchase"[..],
                &b"p1"[..],
                &b"1000"[..],
            ])
        );
    }

    #[test]
    fn full_four_field_click_line() {
        // Shape of clicks/*.csv: click_id,user_id,campaign_id,ts
        let line = b"k1,u1,c_1,2026-10-03T09:00:00Z";
        assert_eq!(
            split(line, 4),
            Some(vec![
                &b"k1"[..],
                &b"u1"[..],
                &b"c_1"[..],
                &b"2026-10-03T09:00:00Z"[..]
            ])
        );
    }

    #[test]
    fn fields_are_subslices_of_the_input_not_copies() {
        // Zero-copy is a hard requirement: assert the returned slices point into
        // the original buffer.
        let line = b"abc,de".to_vec();
        let mut out = Vec::new();
        assert!(split_fields(&line, 2, &mut out));
        let base = line.as_ptr() as usize;
        assert_eq!(out[0].as_ptr() as usize, base);
        assert_eq!(out[1].as_ptr() as usize, base + 4);
    }

    #[test]
    fn out_buffer_is_reusable_across_lines() {
        let mut out = Vec::new();
        assert!(split_fields(b"a,b", 2, &mut out));
        assert_eq!(out.len(), 2);
        // A second call must not append to the first call's results.
        assert!(split_fields(b"c,d", 2, &mut out));
        assert_eq!(out, vec![&b"c"[..], &b"d"[..]]);
        // And a failing call must not leave stale state that a later call sees.
        assert!(!split_fields(b"e,f,g", 2, &mut out));
        assert!(split_fields(b"h,i", 2, &mut out));
        assert_eq!(out, vec![&b"h"[..], &b"i"[..]]);
    }

    // ------------------------------------------------- field count (fixture L3)

    #[test]
    fn too_many_fields_is_malformed() {
        // L3: `u1,u2,u3` in links/ — three fields where two are required.
        assert_eq!(split(b"u1,u2,u3", 2), None);
    }

    #[test]
    fn too_few_fields_is_malformed() {
        // L3: a bare `u1` line in links/.
        assert_eq!(split(b"u1", 2), None);
    }

    #[test]
    fn event_line_with_six_fields_is_malformed() {
        // E13: `e2,u1,<ts>,<ingest_ts>,view,p1` — six fields, seven required.
        let line = b"e2,u1,2026-10-03T10:05:00Z,2026-10-03T10:06:00Z,view,p1";
        assert_eq!(split(line, 7), None);
    }

    #[test]
    fn event_line_with_eight_fields_is_malformed() {
        let line = b"e2,u1,2026-10-03T10:05:00Z,2026-10-03T10:06:00Z,view,p1,100,extra";
        assert_eq!(split(line, 7), None);
    }

    #[test]
    fn empty_line_is_malformed_at_every_expected_count() {
        // "An empty line is malformed too." Including when one field is expected,
        // where it would otherwise parse as a single null field.
        assert_eq!(split(b"", 1), None);
        assert_eq!(split(b"", 2), None);
        assert_eq!(split(b"", 7), None);
    }

    // ------------------------------------------------- empty fields are null

    #[test]
    fn empty_field_on_the_right_is_a_valid_null_field() {
        // L3 has `u1,` in links/, and that line IS dropped — but NOT by this
        // splitter. It is a perfectly well-formed 2-field line whose second
        // field is empty. The links stage drops it under its own rule ("Drop
        // [...] lines where either field is empty"), which is deliberately not
        // implemented here because other file types allow empty fields (an empty
        // `price_cents` on a `view` is fine). Asserting the splitter's behaviour,
        // not the links rule.
        let f = split(b"u1,", 2).expect("well-formed 2-field line");
        assert_eq!(f, vec![&b"u1"[..], &b""[..]]);
        assert!(!is_null(f[0]));
        assert!(is_null(f[1]));
    }

    #[test]
    fn empty_field_on_the_left_is_a_valid_null_field() {
        // L3 has `,u3`; same reasoning as above.
        let f = split(b",u3", 2).expect("well-formed 2-field line");
        assert_eq!(f, vec![&b""[..], &b"u3"[..]]);
        assert!(is_null(f[0]));
        assert!(!is_null(f[1]));
    }

    #[test]
    fn empty_event_id_is_a_valid_null_field() {
        // E13's first data row has an empty event_id. The line parses fine; the
        // events stage is what drops it.
        let line = b",u1,2026-10-03T10:00:00Z,2026-10-03T10:01:00Z,view,p1,";
        let f = split(line, 7).expect("well-formed 7-field line");
        assert!(is_null(f[0]));
        assert!(is_null(f[6]));
        assert_eq!(f[1], b"u1");
    }

    #[test]
    fn a_line_of_only_commas_is_all_null_fields() {
        let f = split(b",", 2).expect("two empty fields");
        assert_eq!(f, vec![&b""[..], &b""[..]]);
        let f = split(b",,,,,,", 7).expect("seven empty fields");
        assert_eq!(f.len(), 7);
        assert!(f.iter().all(|x| is_null(x)));
    }

    #[test]
    fn a_field_of_one_space_is_not_null() {
        let f = split(b" ,u3", 2).unwrap();
        assert!(!is_null(f[0]));
        assert_eq!(f[0], b" ");
    }

    // -------------------------------------------- quotes are syntax, not value

    #[test]
    fn quoted_fields_yield_the_unquoted_value() {
        // L3: `"u4","u1"` is kept and links u4 to u1.
        assert_eq!(split(b"\"u4\",\"u1\"", 2), Some(vec![&b"u4"[..], &b"u1"[..]]));
    }

    #[test]
    fn mixed_quoted_and_unquoted_fields() {
        // E15: one view by `"u1"` (quoted), one by u1 — same user.
        let quoted = b"e1,\"u1\",2026-10-03T10:00:00Z,2026-10-03T10:01:00Z,view,p1,";
        let plain = b"e1,u1,2026-10-03T10:00:00Z,2026-10-03T10:01:00Z,view,p1,";
        assert_eq!(split(quoted, 7).unwrap()[1], b"u1");
        assert_eq!(split(quoted, 7).unwrap(), split(plain, 7).unwrap());
    }

    #[test]
    fn quoted_empty_field_is_empty_and_null() {
        // rules.md: `""` is an empty field, and an empty field means null. So
        // `""` and `` are indistinguishable after splitting.
        let f = split(b"u1,\"\"", 2).expect("quoted empty is a valid empty field");
        assert_eq!(f, vec![&b"u1"[..], &b""[..]]);
        assert!(is_null(f[1]));
        assert_eq!(split(b"u1,\"\"", 2), split(b"u1,", 2));
    }

    #[test]
    fn quoted_empty_field_alone_is_valid() {
        assert_eq!(split(b"\"\"", 1), Some(vec![&b""[..]]));
    }

    #[test]
    fn every_field_quoted_differently_is_the_same_row() {
        // Click duplicates are "identical values, quoted differently", so the
        // splitter must flatten the difference away.
        let a = b"k1,u1,c_1,2026-10-03T09:00:00Z";
        let b = b"\"k1\",u1,\"c_1\",\"2026-10-03T09:00:00Z\"";
        assert_eq!(split(a, 4).unwrap(), split(b, 4).unwrap());
    }

    // ------------------------------------- a stray quote makes the LINE malformed

    #[test]
    fn trailing_quote_without_opening_quote_is_malformed() {
        // L3: `u1,u5"` is dropped, which is why u5 stays a person of its own.
        assert_eq!(split(b"u1,u5\"", 2), None);
    }

    #[test]
    fn opening_quote_without_closing_quote_is_malformed() {
        assert_eq!(split(b"\"u1", 1), None);
        assert_eq!(split(b"\"u1,u2", 2), None);
    }

    #[test]
    fn a_lone_quote_field_is_malformed() {
        // A single `"` is an opening quote with no closing quote. It must NOT be
        // read as a matched pair or as an empty field.
        assert_eq!(split(b"\"", 1), None);
        assert_eq!(split(b"\",u2", 2), None);
        assert_eq!(split(b"u1,\"", 2), None);
    }

    #[test]
    fn a_quote_inside_an_unquoted_field_is_malformed() {
        assert_eq!(split(b"u\"1,u2", 2), None);
        assert_eq!(split(b"u1,u\"2", 2), None);
    }

    #[test]
    fn a_quote_inside_a_quoted_field_is_malformed() {
        // No value ever holds a quote, so the inner quote cannot be data.
        assert_eq!(split(b"\"a\"b\"", 1), None);
        assert_eq!(split(b"\"u1\"\",u2", 2), None);
    }

    #[test]
    fn a_quoted_field_adjacent_to_text_is_malformed() {
        assert_eq!(split(b"x\"u1\",u2", 2), None);
        assert_eq!(split(b"\"u1\"x,u2", 2), None);
    }

    #[test]
    fn three_quotes_as_a_whole_field_is_malformed() {
        assert_eq!(split(b"\"\"\"", 1), None);
    }

    #[test]
    fn a_malformed_quote_anywhere_kills_the_whole_line() {
        // Not just the offending field: the line is dropped.
        let line = b"e1,u1,2026-10-03T10:00:00Z,2026-10-03T12:00:00Z,view,p1,10\"";
        assert_eq!(split(line, 7), None);
    }

    #[test]
    fn a_quote_is_not_a_field_separator() {
        // Guard against a quote-aware state machine creeping in: `"a,b"` is two
        // fields (`"a` and `b"`), both malformed, NOT one field `a,b`.
        assert_eq!(split(b"\"a,b\"", 1), None);
        assert_eq!(split(b"\"a,b\"", 2), None);
    }

    // ------------------------------------------- spaces are part of the value

    #[test]
    fn leading_and_trailing_spaces_are_never_trimmed_unquoted() {
        // E19: users `u1`, ` u1`, `u1 ` are three different people.
        assert_eq!(split(b" u1,u1 ", 2), Some(vec![&b" u1"[..], &b"u1 "[..]]));
    }

    #[test]
    fn leading_and_trailing_spaces_are_never_trimmed_inside_quotes() {
        // E19: `" u1"` is the same ID as ` u1`, and neither is `u1`.
        assert_eq!(split(b"\" u1\",\"u1 \"", 2), Some(vec![&b" u1"[..], &b"u1 "[..]]));
    }

    #[test]
    fn quoted_spaced_id_equals_unquoted_spaced_id_but_not_the_bare_id() {
        let quoted = split(b"e4,\" u1\"", 2).unwrap();
        let plain = split(b"e4, u1", 2).unwrap();
        let bare = split(b"e4,u1", 2).unwrap();
        assert_eq!(quoted[1], plain[1]);
        assert_ne!(quoted[1], bare[1]);
        assert_eq!(quoted[1], b" u1");
    }

    #[test]
    fn spaces_are_kept_in_every_column() {
        // E19 relies on ` view` being a bad event_type, ` p1` being an unknown
        // product, ` c_1` an unknown campaign and `e1 ` a distinct event_id.
        let line = b"e1 ,u1,2026-10-03T10:00:00Z,2026-10-03T12:00:00Z, view, p1,400 ";
        let f = split(line, 7).unwrap();
        assert_eq!(f[0], b"e1 ");
        assert_eq!(f[4], b" view");
        assert_ne!(f[4], b"view");
        assert_eq!(f[5], b" p1");
        assert_eq!(f[6], b"400 ");
    }

    #[test]
    fn a_field_of_only_spaces_is_preserved() {
        assert_eq!(split(b"  ,u1", 2), Some(vec![&b"  "[..], &b"u1"[..]]));
        assert_eq!(split(b"\"  \",u1", 2), Some(vec![&b"  "[..], &b"u1"[..]]));
    }

    // ------------------------------------------- \r is an ordinary character

    #[test]
    fn carriage_return_is_kept_in_the_last_field() {
        // E20: a purchase whose price becomes `1000\r` must be dropped by the
        // price validator, which can only happen if the `\r` survives here.
        let line = b"e2,u1,2026-10-03T10:05:00Z,2026-10-03T12:00:00Z,purchase,p1,1000\r";
        let f = split(line, 7).unwrap();
        assert_eq!(f[6], b"1000\r");
        assert_ne!(f[6], b"1000");
    }

    #[test]
    fn carriage_return_alone_is_a_non_empty_last_field() {
        // E20's first row is `...,view,p1,\r`: price_cents is `\r`, not empty.
        // It does not matter for a view (price is ignored), but it must not be
        // mistaken for null.
        let line = b"e1,u1,2026-10-03T10:00:00Z,2026-10-03T12:00:00Z,view,p1,\r";
        let f = split(line, 7).unwrap();
        assert_eq!(f[6], b"\r");
        assert!(!is_null(f[6]));
    }

    #[test]
    fn carriage_return_in_a_click_ts_is_kept() {
        // E20: a click whose ts becomes `...Z\r` is dropped downstream.
        let line = b"k1,u1,c_1,2026-10-03T09:00:00Z\r";
        let f = split(line, 4).unwrap();
        assert_eq!(f[3], b"2026-10-03T09:00:00Z\r");
    }

    #[test]
    fn carriage_return_mid_line_belongs_to_its_own_field() {
        // E20: event IDs `e3` and `e3\r` are different IDs, so they do not dedup
        // against each other.
        let line = b"e3\r,u1,2026-10-03T10:15:00Z,2026-10-03T12:00:00Z,view,p1,";
        let f = split(line, 7).unwrap();
        assert_eq!(f[0], b"e3\r");
        assert_ne!(f[0], b"e3");
    }

    #[test]
    fn carriage_return_inside_quotes_is_kept() {
        let f = split(b"\"e3\r\",u1", 2).unwrap();
        assert_eq!(f[0], b"e3\r");
    }

    #[test]
    fn a_line_of_just_a_carriage_return_is_not_an_empty_line() {
        // `\r\n` on an otherwise blank line leaves a one-byte line, which is a
        // single non-empty field, not the "empty line" case.
        assert_eq!(split(b"\r", 1), Some(vec![&b"\r"[..]]));
        assert_eq!(split(b"\r", 2), None); // wrong field count, though
    }

    // ------------------------------------------- no folding, no normalisation

    #[test]
    fn no_case_folding() {
        // E11 / L3: `U1` and `u1` are different IDs.
        let f = split(b"U1,u1", 2).unwrap();
        assert_eq!(f[0], b"U1");
        assert_ne!(f[0], f[1]);
    }

    #[test]
    fn tabs_and_other_control_bytes_are_ordinary_characters() {
        // L3's last data row is `"A<TAB>B",user_id_a`, and `A<TAB>B` ends up
        // being the *name* of that person because it is the byte-wise smallest
        // ID. So the tab has to come through intact.
        let f = split(b"\"A\tB\",user_id_a", 2).unwrap();
        assert_eq!(f[0], b"A\tB");
        assert_eq!(f[1], b"user_id_a");
    }

    #[test]
    fn non_ascii_utf8_bytes_pass_through_unchanged() {
        // E18: `٢٥٠` and `２５０` are not digits, but they are valid field
        // content and must reach the validator byte-for-byte.
        let line = "e1,u1,2026-10-03T10:00:00Z,2026-10-03T12:00:00Z,purchase,p1,٢٥٠".as_bytes();
        let f = split(line, 7).unwrap();
        assert_eq!(f[6], "٢٥٠".as_bytes());
        assert_ne!(f[6], b"250");
    }

    #[test]
    fn no_whitespace_or_dot_interpretation_of_timestamps() {
        // E13: `1791071890000.0` is a bad ts, but a well-formed field.
        let line = b"e4,u1,1791071890000.0,2026-10-03T23:59:00Z,view,p1,";
        assert_eq!(split(line, 7).unwrap()[2], b"1791071890000.0");
    }

    // --------------------------------------------------- data_lines: iteration

    fn lines(bytes: &[u8]) -> Vec<&[u8]> {
        data_lines(bytes).collect()
    }

    #[test]
    fn data_lines_skips_exactly_one_header_line() {
        let f = b"user_id_a,user_id_b\nu4,u1\nu6,u6\n";
        assert_eq!(lines(f), vec![&b"u4,u1"[..], &b"u6,u6"[..]]);
    }

    #[test]
    fn data_lines_on_an_empty_file_yields_nothing() {
        // Must not underflow or panic.
        assert_eq!(lines(b""), Vec::<&[u8]>::new());
    }

    #[test]
    fn data_lines_on_a_header_only_file_yields_nothing() {
        // 1 to 3 files per generated dataset hold only the header.
        assert_eq!(lines(b"user_id_a,user_id_b\n"), Vec::<&[u8]>::new());
        // ...and the same file without its trailing newline.
        assert_eq!(lines(b"user_id_a,user_id_b"), Vec::<&[u8]>::new());
    }

    #[test]
    fn data_lines_on_a_file_of_just_a_newline_yields_nothing() {
        assert_eq!(lines(b"\n"), Vec::<&[u8]>::new());
    }

    #[test]
    fn data_lines_yields_a_last_line_with_no_trailing_newline() {
        let f = b"hdr\na,b\nc,d";
        assert_eq!(lines(f), vec![&b"a,b"[..], &b"c,d"[..]]);
    }

    #[test]
    fn data_lines_does_not_invent_an_empty_trailing_line() {
        let f = b"hdr\na,b\n";
        assert_eq!(lines(f), vec![&b"a,b"[..]]);
    }

    #[test]
    fn data_lines_yields_a_genuine_blank_line_in_the_middle() {
        // L3 contains a blank line, which the splitter then rejects as malformed.
        let f = b"hdr\na,b\n\nc,d\n";
        assert_eq!(lines(f), vec![&b"a,b"[..], &b""[..], &b"c,d"[..]]);
        assert_eq!(split(b"", 2), None);
    }

    #[test]
    fn data_lines_yields_a_blank_final_line_from_a_double_newline() {
        // Only ONE trailing newline is the terminator; "a\n\n" is two lines.
        let f = b"hdr\na,b\n\n";
        assert_eq!(lines(f), vec![&b"a,b"[..], &b""[..]]);
    }

    #[test]
    fn data_lines_keeps_a_data_row_identical_to_the_header() {
        // THE critical case (fixture L3). The header is skipped by POSITION. A
        // later row that happens to equal the header is real data: it links the
        // ID `user_id_a` to the ID `user_id_b`.
        let f = b"user_id_a,user_id_b\nu4,u1\nuser_id_a,user_id_b\n";
        let got = lines(f);
        assert_eq!(got.len(), 2);
        assert_eq!(got[1], b"user_id_a,user_id_b");
        assert_eq!(
            split(got[1], 2),
            Some(vec![&b"user_id_a"[..], &b"user_id_b"[..]])
        );
    }

    #[test]
    fn data_lines_splits_only_on_newline_never_on_carriage_return() {
        let f = b"hdr\na,b\r\nc,d\r\n";
        assert_eq!(lines(f), vec![&b"a,b\r"[..], &b"c,d\r"[..]]);
    }

    #[test]
    fn data_lines_does_not_strip_a_trailing_cr_from_the_file() {
        // A file ending `...\r\n`: the \r belongs to the last line.
        let f = b"hdr\na,b\r\n";
        assert_eq!(lines(f), vec![&b"a,b\r"[..]]);
    }

    #[test]
    fn data_lines_returns_subslices_of_the_input() {
        let f = b"hdr\nabc\n".to_vec();
        let got: Vec<&[u8]> = data_lines(&f).collect();
        assert_eq!(got[0].as_ptr() as usize, f.as_ptr() as usize + 4);
    }

    // --------------------------------- the whole of fixture L3's links file

    #[test]
    fn fixture_l3_links_file_end_to_end() {
        // Byte-for-byte the content of
        // starter-duckdb/fixtures/L3/input/links/part-00000.csv.
        // Expected: the 5 malformed lines are rejected by the splitter; the
        // empty-side lines (`u1,` and `,u3`) parse fine here and are dropped by
        // the links stage instead; the duplicate-of-header row is real data.
        let file: &[u8] = b"user_id_a,user_id_b\n\
                            u1,u2,u3\n\
                            u1,\n\
                            ,u3\n\
                            u1,u5\"\n\
                            \"u4\",\"u1\"\n\
                            u4,u1\n\
                            u6,u6\n\
                            \n\
                            u1\n\
                            user_id_a,user_id_b\n\
                            u4,user_id_b\n\
                            \"A\tB\",user_id_a\n";

        let got: Vec<&[u8]> = data_lines(file).collect();
        assert_eq!(got.len(), 12, "12 data rows after skipping the header");

        let parsed: Vec<Option<Vec<&[u8]>>> = got.iter().map(|l| split(l, 2)).collect();

        // Malformed per the CSV dialect -> dropped by the splitter.
        assert_eq!(parsed[0], None, "u1,u2,u3 : wrong field count");
        assert_eq!(parsed[3], None, "u1,u5\" : stray quote");
        assert_eq!(parsed[7], None, "empty line");
        assert_eq!(parsed[8], None, "u1 : wrong field count");

        // Well-formed for the splitter; dropped later by the links-specific
        // "either field empty" rule, NOT here.
        assert_eq!(parsed[1], Some(vec![&b"u1"[..], &b""[..]]), "u1,");
        assert_eq!(parsed[2], Some(vec![&b""[..], &b"u3"[..]]), ",u3");

        // Kept, and the quotes are stripped.
        assert_eq!(parsed[4], Some(vec![&b"u4"[..], &b"u1"[..]]), "\"u4\",\"u1\"");
        assert_eq!(parsed[5], Some(vec![&b"u4"[..], &b"u1"[..]]), "u4,u1 (repeat)");
        assert_eq!(parsed[6], Some(vec![&b"u6"[..], &b"u6"[..]]), "u6,u6 (self)");

        // The row equal to the header, plus the two rows that chain off it.
        assert_eq!(
            parsed[9],
            Some(vec![&b"user_id_a"[..], &b"user_id_b"[..]]),
            "a data row identical to the header is real data"
        );
        assert_eq!(parsed[10], Some(vec![&b"u4"[..], &b"user_id_b"[..]]));
        assert_eq!(parsed[11], Some(vec![&b"A\tB"[..], &b"user_id_a"[..]]));
    }

    #[test]
    fn fixture_e20_events_file_end_to_end() {
        // Byte-for-byte the content of
        // starter-duckdb/fixtures/E20/input/events/part-00000.csv.
        let file: &[u8] = b"event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n\
            e1,u1,2026-10-03T10:00:00Z,2026-10-03T12:00:00Z,view,p1,\r\n\
            e2,u1,2026-10-03T10:05:00Z,2026-10-03T12:00:00Z,purchase,p1,1000\r\n\
            e3,u1,2026-10-03T10:10:00Z,2026-10-03T12:00:00Z,purchase,p2,500\n\
            e3\r,u1,2026-10-03T10:15:00Z,2026-10-03T12:00:00Z,view,p1,\n";

        let got: Vec<&[u8]> = data_lines(file).collect();
        assert_eq!(got.len(), 4);

        let rows: Vec<Vec<&[u8]>> = got
            .iter()
            .map(|l| split(l, 7).expect("all four lines are well-formed"))
            .collect();

        assert_eq!(rows[0][6], b"\r", "a \\r-only price, not null");
        assert_eq!(rows[1][6], b"1000\r", "the \\r makes this price bad");
        assert_eq!(rows[2][6], b"500", "no \\r on this line");
        assert_eq!(rows[3][0], b"e3\r", "e3\\r is a different event_id from e3");
        assert_ne!(rows[3][0], rows[2][0]);
    }

    // ------------------------------------- SIMD vs scalar differential tests

    /// (ptr, len) of each field, so "identical" means the very same subslice,
    /// not merely equal bytes.
    fn spans(v: &[&[u8]]) -> Vec<(usize, usize)> {
        v.iter().map(|f| (f.as_ptr() as usize, f.len())).collect()
    }

    /// Assert `split_fields` and `split_fields_scalar` agree on the bool and on
    /// the exact contents of `out` (also on failure, where the push order of
    /// both implementations is the same).
    fn assert_same(line: &[u8], expected_fields: usize) {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let ra = split_fields(line, expected_fields, &mut a);
        let rb = split_fields_scalar(line, expected_fields, &mut b);
        assert_eq!(ra, rb, "result differs for {:?} / {}", line, expected_fields);
        assert_eq!(spans(&a), spans(&b), "fields differ for {:?} / {}", line, expected_fields);
    }

    struct XorShift(u64);
    impl XorShift {
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
    fn simd_matches_scalar_on_random_lines() {
        const ALPHA: &[u8] = &[b'a', b'1', b',', b'"', b' ', b'\r', 0xC3, 0xA9];
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        let mut line = Vec::with_capacity(256);
        let mut accepted = 0usize;
        for iter in 0..300_000u32 {
            line.clear();
            let len = rng.below(200) as usize;
            // Vary the density of special bytes so long lines are not always
            // rejected for having too many commas.
            let mode = rng.below(4);
            for _ in 0..len {
                let b = match mode {
                    0 => ALPHA[rng.below(ALPHA.len() as u64) as usize],
                    1 => {
                        if rng.below(16) == 0 { b',' } else { b'a' }
                    }
                    2 => {
                        let r = rng.below(32);
                        if r == 0 { b',' } else if r == 1 { b'"' } else { ALPHA[(r % 8) as usize] }
                    }
                    _ => [b'a', b'1', b' ', b'\r', 0xC3, 0xA9][rng.below(6) as usize],
                };
                line.push(b);
            }
            // Mode 3 lines are comma/quote free; place commas exactly on and
            // around 16/32-byte chunk boundaries.
            if mode == 3 && len > 0 {
                for &p in &[15usize, 16, 31, 32, 33, 47, 48, 63, 64, 95, 96, 127, 128] {
                    if p < len && rng.below(2) == 0 {
                        line[p] = b',';
                    }
                }
                if rng.below(4) == 0 {
                    line[rng.below(len as u64) as usize] = b'"';
                }
            }
            let expected = 1 + (iter as usize % 9);
            let mut tmp = Vec::new();
            if split_fields_scalar(&line, expected, &mut tmp) {
                accepted += 1;
            }
            assert_same(&line, expected);
            // Also try the field count the line actually has, so the success
            // path is exercised heavily.
            let commas = line.iter().filter(|&&b| b == b',').count();
            assert_same(&line, commas + 1);
        }
        assert!(accepted > 1000, "fuzz should hit the success path ({accepted})");
    }

    #[test]
    fn simd_matches_scalar_on_chunk_boundaries_exhaustively() {
        // Every line of length 1..=100 with a single comma at every position,
        // two commas at every pair of positions straddling 16/32 boundaries,
        // and quoted/unquoted variants.
        for len in 1..=100usize {
            for p in 0..len {
                let mut l = vec![b'x'; len];
                l[p] = b',';
                for e in 1..=3 {
                    assert_same(&l, e);
                }
                for q in 0..len {
                    let mut m = l.clone();
                    if q != p {
                        m[q] = b'"';
                    }
                    assert_same(&m, 2);
                }
            }
        }
        // Wrapped fields crossing chunk boundaries.
        for w in 0..80usize {
            let mut l = Vec::new();
            l.push(b'"');
            l.extend(std::iter::repeat(b'a').take(w));
            l.extend_from_slice(b"\",\"");
            l.extend(std::iter::repeat(b'b').take(w / 2));
            l.push(b'"');
            assert_same(&l, 2);
            let f = split(&l, 2).expect("two wrapped fields");
            assert_eq!(f[0].len(), w);
        }
    }

    #[test]
    fn simd_matches_scalar_on_long_realistic_lines() {
        let lines: [&[u8]; 6] = [
            b"e2,u1,2026-10-03T10:05:00Z,2026-10-03T12:00:00Z,purchase,p1,1000",
            b"\"e2\",\"u1\",\"2026-10-03T10:05:00Z\",\"2026-10-03T12:00:00Z\",\"purchase\",\"p1\",\"1000\"",
            b"e2,u1,2026-10-03T10:05:00Z,2026-10-03T12:00:00Z,purchase,p1,1000\r",
            b"e2,u1,2026-10-03T10:05:00Z,2026-10-03T12:00:00Z,purchase,p1,10\"00",
            b"e2,u1,2026-10-03T10:05:00Z,2026-10-03T12:00:00Z,purchase,p1,1000,",
            b"e2,u1,2026-10-03T10:05:00Z,2026-10-03T12:00:00Z,purchase,p1",
        ];
        for l in lines {
            for e in 1..=9 {
                assert_same(l, e);
            }
        }
    }
}
