//! v3 chunk scanner: CSV lines -> fixed arrays of fields, in one SIMD pass.
//!
//! What it replaces: `for_each_line` (memchr per line) + `csvline::split_fields`
//! (a per-line SIMD scan into a `Vec`). Here one pass over the chunk builds, per
//! 64-byte block, three bitmasks — commas, quotes, newlines — and walks the
//! comma/newline bits in order with `trailing_zeros`. A line's fields are
//! located by its comma positions; nothing is allocated per line.
//!
//! The result is defined to be exactly what the old path produced (see the
//! differential tests at the bottom, which compare against `split_fields` on
//! random adversarial input):
//! - only `\n` ends a line; `\r` and every other byte is ordinary data;
//! - empty lines are skipped (malformed);
//! - a line whose comma count is not `N - 1` is skipped (malformed);
//! - quotes: same result as `csvline::unquote` on every field (strip one
//!   wrapping pair; any other quote makes the line malformed), computed from
//!   the line's quote count (see `finish`);
//! - a last line without a trailing `\n` is still a line.


/// Bitmasks of one 64-byte block: bit i set = byte i is that character.
#[derive(Clone, Copy)]
struct Masks {
    comma: u64,
    quote: u64,
    newline: u64,
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn block_masks(block: &[u8; 64]) -> Masks {
    use std::arch::aarch64::*;
    // NEON has no movemask. Standard trick (as in simdjson): AND each 0xFF/0x00
    // compare result with the weights 1,2,4..128 per 8 bytes, then three rounds
    // of pairwise adds collapse 64 bytes into 64 bits, byte 8k+j -> bit 8k+j.
    // SAFETY: NEON is mandatory on aarch64; the loads read exactly 64 bytes.
    unsafe {
        const WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let weights = vld1q_u8(WEIGHTS.as_ptr());
        let p = block.as_ptr();
        let v = [vld1q_u8(p), vld1q_u8(p.add(16)), vld1q_u8(p.add(32)), vld1q_u8(p.add(48))];
        let mask_of = |c: u8| -> u64 {
            let needle = vdupq_n_u8(c);
            let t0 = vandq_u8(vceqq_u8(v[0], needle), weights);
            let t1 = vandq_u8(vceqq_u8(v[1], needle), weights);
            let t2 = vandq_u8(vceqq_u8(v[2], needle), weights);
            let t3 = vandq_u8(vceqq_u8(v[3], needle), weights);
            let s0 = vpaddq_u8(t0, t1);
            let s1 = vpaddq_u8(t2, t3);
            let s = vpaddq_u8(s0, s1);
            let s = vpaddq_u8(s, s);
            vgetq_lane_u64::<0>(vreinterpretq_u64_u8(s))
        };
        Masks {
            comma: mask_of(b','),
            quote: mask_of(b'"'),
            newline: mask_of(b'\n'),
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn block_masks(block: &[u8; 64]) -> Masks {
    use std::arch::x86_64::*;
    // SAFETY: avx2 is enabled at compile time; the loads read exactly 64 bytes.
    unsafe {
        let p = block.as_ptr() as *const __m256i;
        let lo = _mm256_loadu_si256(p);
        let hi = _mm256_loadu_si256(p.add(1));
        let mask_of = |c: u8| -> u64 {
            let needle = _mm256_set1_epi8(c as i8);
            let a = _mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, needle)) as u32 as u64;
            let b = _mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, needle)) as u32 as u64;
            a | (b << 32)
        };
        Masks {
            comma: mask_of(b','),
            quote: mask_of(b'"'),
            newline: mask_of(b'\n'),
        }
    }
}

#[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx2"))))]
#[inline(always)]
fn block_masks(block: &[u8; 64]) -> Masks {
    block_masks_scalar(block)
}

/// Reference / fallback implementation of `block_masks`.
#[inline]
#[allow(dead_code)]
fn block_masks_scalar(block: &[u8; 64]) -> Masks {
    let mut m = Masks {
        comma: 0,
        quote: 0,
        newline: 0,
    };
    for (i, &b) in block.iter().enumerate() {
        m.comma |= u64::from(b == b',') << i;
        m.quote |= u64::from(b == b'"') << i;
        m.newline |= u64::from(b == b'\n') << i;
    }
    m
}

/// Call `f` with the `N` fields of every well-formed line of `bytes`, in order.
/// Malformed lines (see the module doc) are skipped silently. If `skip_header`
/// the first line (up to and including its `\n`) is skipped; a header with no
/// `\n` means there are no data lines.
#[inline]
pub fn for_each_record<'a, const N: usize>(bytes: &'a [u8], skip_header: bool, mut f: impl FnMut(&[&'a [u8]; N])) {
    assert!(N >= 1 && N <= 16);
    let buf = if skip_header {
        match memchr::memchr(b'\n', bytes) {
            Some(newline) => &bytes[newline + 1..],
            None => return,
        }
    } else {
        bytes
    };

    // Per-line state.
    let mut line_start = 0usize;
    let mut commas = [0usize; 16]; // positions of the first N-1 commas
    let mut n_commas = 0usize;
    let mut line_quotes = 0u32; // number of `"` bytes in the current line

    // Finish the line `[line_start, end)`; emits it if well-formed.
    let mut finish = |line_start: usize, end: usize, commas: &[usize; 16], n_commas: usize, quotes: u32| {
        if end == line_start || n_commas != N - 1 {
            return; // empty line, or wrong field count: malformed
        }
        // Field k is the byte range [lo(k), hi(k)) of `buf`:
        //   lo(0) = line_start, lo(k) = commas[k-1] + 1;  hi(k) = commas[k], hi(N-1) = end.
        // Each field slice is built exactly once, from these bounds, already
        // unquoted (the old code built every slice, then re-sliced it when the
        // line had quotes). The result is the same subslice of `buf` as before:
        // same bytes and same address, which matters because source-backed IDs
        // store offsets into the chunk.
        //
        // Why the unchecked slicing is in bounds: `n_commas == N - 1` here, so
        // `commas[..N-1]` holds every comma of this line, recorded in scan order
        // at positions strictly inside it: line_start <= commas[0] < commas[1]
        // < ... < commas[N-2] < end <= buf.len() (`end` is this line's newline,
        // or buf.len() for a last line without one). Hence for every k,
        // lo(k) <= hi(k) <= buf.len(). With w in {0,1} and w == 1 only when the
        // field has length >= 2, lo(k) + w <= hi(k) - w as well.
        debug_assert!(line_start <= end && end <= buf.len());
        debug_assert!((0..N - 1).all(|k| commas[k] >= line_start && commas[k] < end));
        debug_assert!((1..N - 1).all(|k| commas[k - 1] < commas[k]));
        let bounds = |k: usize| -> (usize, usize) {
            let lo = if k == 0 { line_start } else { commas[k - 1] + 1 };
            let hi = if k == N - 1 { end } else { commas[k] };
            (lo, hi)
        };
        if quotes == 0 {
            // No quote anywhere on the line (the common case): plain fields.
            // SAFETY: lo <= hi <= buf.len(), shown above.
            let fields: [&'a [u8]; N] = std::array::from_fn(|k| {
                let (lo, hi) = bounds(k);
                unsafe { buf.get_unchecked(lo..hi) }
            });
            f(&fields);
            return;
        }
        // Equivalent to `csvline::unquote` on every field, using the line's
        // quote count from the bitmasks: strip each field that is wrapped in a
        // pair (`"..."`, length >= 2). The line is well-formed iff every quote
        // in it was one end of such a pair, i.e. iff the stripped pairs account
        // for all `quotes` quote bytes. Any other quote (lone, inside, at one
        // end only) leaves the counts unequal.
        //
        // Branch-free: the wrap test is a coin flip on real data (~20% of
        // fields quoted), so it is an integer w in {0,1}. `first()`/`last()`
        // with `unwrap_or(0)` never read out of bounds (an empty field yields
        // 0, not a quote) and lower to selects; `n >= 2` rules out a lone `"`
        // counting as both ends.
        let mut stripped = 0u32;
        let fields: [&'a [u8]; N] = std::array::from_fn(|k| {
            let (lo, hi) = bounds(k);
            // SAFETY: lo <= hi <= buf.len(), shown above.
            let raw = unsafe { buf.get_unchecked(lo..hi) };
            let first = raw.first().copied().unwrap_or(0);
            let last = raw.last().copied().unwrap_or(0);
            let w = usize::from(raw.len() >= 2) & usize::from(first == b'"') & usize::from(last == b'"');
            stripped += 2 * w as u32;
            // SAFETY: w == 1 only if hi - lo >= 2, so lo + w <= hi - w <= buf.len().
            unsafe { buf.get_unchecked(lo + w..hi - w) }
        });
        if stripped != quotes {
            return; // a quote not wrapping a whole field: malformed
        }
        f(&fields);
    };

    let len = buf.len();
    let mut base = 0usize;
    while base < len {
        let masks = if base + 64 <= len {
            block_masks(buf[base..base + 64].try_into().expect("64 bytes"))
        } else {
            // Tail: zero padding contains none of the three characters.
            let mut tail = [0u8; 64];
            tail[..len - base].copy_from_slice(&buf[base..]);
            block_masks(&tail)
        };
        let mut quote = masks.quote;
        let mut events = masks.comma | masks.newline;
        while events != 0 {
            let bit = events.trailing_zeros();
            events &= events - 1;
            let pos = base + bit as usize;
            if masks.newline >> bit & 1 != 0 {
                // Quotes strictly before this newline (and, after the previous
                // newline's clearing, at or after this line's start) belong to
                // this line. Then drop every quote bit up to the newline.
                let below = (1u64 << bit) - 1;
                line_quotes += (quote & below).count_ones();
                quote &= !(below | (1u64 << bit));
                finish(line_start, pos, &commas, n_commas, line_quotes);
                line_start = pos + 1;
                n_commas = 0;
                line_quotes = 0;
            } else {
                if n_commas < N - 1 {
                    commas[n_commas] = pos;
                }
                // Saturating keeps "too many commas" detectable without overflow.
                n_commas = n_commas.saturating_add(1);
            }
        }
        // Remaining quote bits lie after the last newline: the current line's.
        line_quotes += quote.count_ones();
        base += 64;
    }
    if line_start < len {
        finish(line_start, len, &commas, n_commas, line_quotes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csvline::split_fields;

    /// The old path: per-line iteration as pipeline3's `for_each_line`, then
    /// `split_fields`.
    fn reference<const N: usize>(bytes: &[u8], skip_header: bool) -> Vec<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        let mut body = bytes;
        if skip_header {
            match memchr::memchr(b'\n', body) {
                Some(n) => body = &body[n + 1..],
                None => return out,
            }
        }
        let mut handle = |line: &[u8]| {
            let mut fields = Vec::new();
            if split_fields(line, N, &mut fields) {
                out.push(fields.iter().map(|f| f.to_vec()).collect());
            }
        };
        let mut start = 0;
        for nl in memchr::memchr_iter(b'\n', body) {
            if nl > start {
                handle(&body[start..nl]);
            }
            start = nl + 1;
        }
        if start < body.len() {
            handle(&body[start..]);
        }
        out
    }

    fn scanned<const N: usize>(bytes: &[u8], skip_header: bool) -> Vec<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        for_each_record::<N>(bytes, skip_header, |fields| {
            out.push(fields.iter().map(|f| f.to_vec()).collect());
        });
        out
    }

    fn check<const N: usize>(bytes: &[u8]) {
        for skip in [false, true] {
            assert_eq!(scanned::<N>(bytes, skip), reference::<N>(bytes, skip), "N={N} skip={skip} input={:?}", String::from_utf8_lossy(bytes));
        }
    }

    #[test]
    fn simd_masks_match_scalar() {
        let mut rng = 0x1234_5678_9abc_def0u64;
        for _ in 0..2000 {
            let mut block = [0u8; 64];
            for b in block.iter_mut() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                *b = b",\"\na\r 0\xff"[(rng % 8) as usize];
            }
            let (a, b) = (block_masks(&block), block_masks_scalar(&block));
            assert_eq!((a.comma, a.quote, a.newline), (b.comma, b.quote, b.newline));
        }
    }

    #[test]
    fn hand_cases() {
        for case in [
            &b""[..],
            b"\n",
            b"\n\n\n",
            b"h\na,b,c,d",
            b"h\na,b,c,d\n",
            b"h\na,b,c\na,b,c,d,e\n,,,\n\"a\",\"\",c,d\n",
            b"h\n\"a,b\",c,d\n\"a\"b\",c,d,e\n a ,b ,c\r,d\r\n",
            b"h\"x\nq\"\",b,c,d\n\"\"\"\",b,c,d\n\"a,b,c,\"",
            b"no newline header",
            // Empty / lone-quote fields at EOF with no trailing newline.
            b"h\na,b,c,\"\"",
            b"h\na,b,c,\"",
            b"h\na,b,c,",
            b"h\n\"\"",
            b"h\n\"",
            b"h\n\"\"\"\"",
            b"h\n\"\"\"\",b,c,d",
            b"h\n\",\",\",\"",
            b"h\n\"\",\"\",\"\",\"\"",
            b"h\n\"\",\"\",\"\",\"\",\"\",\"\",\"\"",
            b"h\n\",b,c,d\n",
            b"h\na,b,c,\"\n",
        ] {
            check::<4>(case);
            check::<7>(case);
            check::<1>(case);
        }
    }

    /// Quoted fields whose quotes straddle 64-byte block boundaries: lines of
    /// every length 62..=130 with quoted first and/or last fields, preceded by
    /// a prefix that shifts the line so quotes land at bytes 63/64, 127/128.
    /// Also lines ending exactly at the buffer end (no `\n`) with an empty or
    /// quoted-empty last field.
    #[test]
    fn block_boundary_quotes() {
        let mut cases: Vec<Vec<u8>> = Vec::new();
        // Field opening quote at byte 63, closing quote at byte 64.
        let mut b = vec![b'x'; 62];
        b.push(b'\n'); // line 1 = bytes 0..62, newline at 62
        b.extend_from_slice(b"\"\",b,c,d\n"); // `"` at 63 and 64
        cases.push(b.clone());
        let mut b = vec![b'x'; 63];
        b.extend_from_slice(b",b,\"c\""); // last field `"c"` with quotes at 66 and 68
        cases.push(b);
        let mut b = Vec::new();
        b.extend_from_slice(&[b'a'; 60]);
        b.extend_from_slice(b",b,\"c\",\"d\""); // quotes near/over 64
        cases.push(b);
        for prefix in [0usize, 1, 2, 3, 30, 61, 62, 63] {
            for line_len in 62usize..=130 {
                for (qfirst, qlast, nl) in [(true, false, true), (false, true, true), (true, true, true), (true, true, false), (false, true, false)] {
                    for n in [4usize, 7] {
                        let mut buf = Vec::new();
                        if prefix > 0 {
                            buf.extend(std::iter::repeat(b'p').take(prefix - 1));
                            buf.push(b'\n');
                        }
                        // Fields: first and last get the padding, middle are short.
                        let fixed = (n - 1) + (n - 2) + 2 * usize::from(qfirst) + 2 * usize::from(qlast);
                        let pad = line_len.saturating_sub(fixed);
                        let (pf, pl) = (pad / 2, pad - pad / 2);
                        if qfirst {
                            buf.push(b'"');
                        }
                        buf.extend(std::iter::repeat(b'f').take(pf));
                        if qfirst {
                            buf.push(b'"');
                        }
                        for _ in 0..n - 2 {
                            buf.extend_from_slice(b",m");
                        }
                        buf.push(b',');
                        if qlast {
                            buf.push(b'"');
                        }
                        buf.extend(std::iter::repeat(b'l').take(pl));
                        if qlast {
                            buf.push(b'"');
                        }
                        if nl {
                            buf.push(b'\n');
                        }
                        cases.push(buf.clone());
                        // Same line twice, plus a trailing line ending exactly at
                        // buffer end with an empty / `""` / `"` last field.
                        let body = buf.clone();
                        for tail in [&b"a,b,c,"[..], b"a,b,c,\"\"", b"a,b,c,\"", b"\"\""] {
                            let mut c = body.clone();
                            if !nl {
                                c.push(b'\n');
                            }
                            c.extend_from_slice(&body);
                            if !nl {
                                c.push(b'\n');
                            }
                            c.extend_from_slice(tail);
                            cases.push(c);
                        }
                    }
                }
            }
        }
        let mut accepted = 0usize;
        for c in &cases {
            check::<4>(c);
            check::<7>(c);
            check::<1>(c);
            accepted += scanned::<4>(c, false).len() + scanned::<7>(c, false).len();
        }
        assert!(accepted > 1000, "only {accepted} accepted records");
    }

    /// Randomised adversarial lines: quotes in odd places, `\r`, spaces,
    /// non-ASCII (incl. unicode digits), empty fields and lines, wrong field
    /// counts, and lines of every length around the 64-byte block edges.
    #[test]
    fn random_differential() {
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let atoms: [&[u8]; 16] = [
            b",", b"\"", b"\"\"", b"\r", b" ", b"a", b"u_12", b"2026-10-03T23:58:10Z", b"\xd9\xa2", b"\xef\xbc\x92",
            b"1791071890000", b"-04:00", b"\"x\"", b"purchase", b"", b"\n",
        ];
        let mut accepted = 0usize;
        for round in 0..3000 {
            let mut buf = Vec::new();
            let lines = 1 + next() % 12;
            for _ in 0..lines {
                // Sometimes pad so the next line starts a few bytes before a
                // 64-byte block edge, so its opening quote / commas straddle it.
                if next() % 3 == 0 {
                    let target = 64 - (1 + (next() % 4) as usize); // start at 60..63 mod 64
                    let fill = (target + 64 - (buf.len() + 1) % 64) % 64;
                    buf.extend(std::iter::repeat(b'z').take(fill));
                    buf.push(b'\n');
                }
                // Mostly well-formed-ish lines with N-1 commas, sometimes noise.
                let n = [4usize, 7][(next() % 2) as usize];
                let noisy = next() % 4 == 0;
                for k in 0..n {
                    let quoted = next() % 5 == 0;
                    if quoted {
                        buf.push(b'"');
                    }
                    let parts = next() % 4;
                    for _ in 0..parts {
                        let atom = atoms[(next() % 16) as usize];
                        if !noisy && (atom == b"," || atom == b"\n" || atom.contains(&b'"')) {
                            buf.extend_from_slice(b"pad_pad_pad");
                        } else {
                            buf.extend_from_slice(atom);
                        }
                    }
                    if quoted {
                        buf.push(b'"');
                    }
                    if k + 1 < n {
                        buf.push(b',');
                    }
                }
                if next() % 8 != 0 || round % 2 == 0 {
                    buf.push(b'\n');
                }
            }
            check::<4>(&buf);
            check::<7>(&buf);
            accepted += scanned::<4>(&buf, false).len() + scanned::<7>(&buf, false).len();
        }
        // The generator must exercise accepted lines, not only rejections.
        assert!(accepted > 3000, "only {accepted} accepted records");
    }

    /// Every line of a real dataset: `SCAN3_DATA=~/bt-data/d20 cargo test --release
    /// -- --ignored scan3`.
    #[test]
    #[ignore]
    fn dataset_differential() {
        let Ok(dir) = std::env::var("SCAN3_DATA") else { return };
        let mut checked = 0usize;
        for (sub, n) in [("events", 7usize), ("clicks", 4)] {
            let mut files: Vec<_> = std::fs::read_dir(format!("{dir}/{sub}")).unwrap().map(|e| e.unwrap().path()).collect();
            files.sort();
            for path in files {
                let bytes = std::fs::read(&path).unwrap();
                let (a, b) = if n == 7 {
                    (scanned::<7>(&bytes, true), reference::<7>(&bytes, true))
                } else {
                    (scanned::<4>(&bytes, true), reference::<4>(&bytes, true))
                };
                assert!(a == b, "mismatch in {path:?}");
                checked += a.len();
            }
        }
        eprintln!("scan3 dataset_differential: {checked} records identical");
    }

    /// Frozen copy of `for_each_record` as of public 341a296 / private 865c931
    /// (field slices built first, then re-sliced when the line has quotes).
    /// The fused `finish` must emit exactly the same subslices of the input.
    fn for_each_record_v0<'a, const N: usize>(bytes: &'a [u8], skip_header: bool, mut f: impl FnMut(&[&'a [u8]; N])) {
        let buf = if skip_header {
            match memchr::memchr(b'\n', bytes) {
                Some(newline) => &bytes[newline + 1..],
                None => return,
            }
        } else {
            bytes
        };
        let mut line_start = 0usize;
        let mut commas = [0usize; 16];
        let mut n_commas = 0usize;
        let mut line_quotes = 0u32;
        let mut finish = |line_start: usize, end: usize, commas: &[usize; 16], n_commas: usize, quotes: u32| {
            if end == line_start || n_commas != N - 1 {
                return;
            }
            let mut fields: [&'a [u8]; N] = [&[]; N];
            let mut start = line_start;
            for (k, field) in fields.iter_mut().enumerate().take(N - 1) {
                *field = &buf[start..commas[k]];
                start = commas[k] + 1;
            }
            fields[N - 1] = &buf[start..end];
            if quotes != 0 {
                let mut stripped = 0u32;
                for field in fields.iter_mut() {
                    let n = field.len();
                    let first = field.first().copied().unwrap_or(0);
                    let last = field.last().copied().unwrap_or(0);
                    let w = usize::from(n >= 2) & usize::from(first == b'"') & usize::from(last == b'"');
                    *field = &field[w..n - w];
                    stripped += 2 * w as u32;
                }
                if stripped != quotes {
                    return;
                }
            }
            f(&fields);
        };
        let len = buf.len();
        let mut base = 0usize;
        while base < len {
            let masks = if base + 64 <= len {
                block_masks_scalar(buf[base..base + 64].try_into().expect("64 bytes"))
            } else {
                let mut tail = [0u8; 64];
                tail[..len - base].copy_from_slice(&buf[base..]);
                block_masks_scalar(&tail)
            };
            let mut quote = masks.quote;
            let mut events = masks.comma | masks.newline;
            while events != 0 {
                let bit = events.trailing_zeros();
                events &= events - 1;
                let pos = base + bit as usize;
                if masks.newline >> bit & 1 != 0 {
                    let below = (1u64 << bit) - 1;
                    line_quotes += (quote & below).count_ones();
                    quote &= !(below | (1u64 << bit));
                    finish(line_start, pos, &commas, n_commas, line_quotes);
                    line_start = pos + 1;
                    n_commas = 0;
                    line_quotes = 0;
                } else {
                    if n_commas < N - 1 {
                        commas[n_commas] = pos;
                    }
                    n_commas = n_commas.saturating_add(1);
                }
            }
            line_quotes += quote.count_ones();
            base += 64;
        }
        if line_start < len {
            finish(line_start, len, &commas, n_commas, line_quotes);
        }
    }

    /// (offset into `bytes`, length) of every emitted field, per record.
    fn spans<const N: usize>(bytes: &[u8], skip: bool, v0: bool) -> Vec<[(usize, usize); N]> {
        let base = bytes.as_ptr() as usize;
        let mut out = Vec::new();
        let mut push = |fields: &[&[u8]; N]| {
            out.push(std::array::from_fn(|k| (fields[k].as_ptr() as usize - base, fields[k].len())));
        };
        if v0 {
            for_each_record_v0::<N>(bytes, skip, &mut push);
        } else {
            for_each_record::<N>(bytes, skip, &mut push);
        }
        out
    }

    fn check_spans(bytes: &[u8]) -> usize {
        let mut accepted = 0;
        for skip in [false, true] {
            macro_rules! one {
                ($n:literal) => {{
                    let (new, old) = (spans::<$n>(bytes, skip, false), spans::<$n>(bytes, skip, true));
                    assert_eq!(new, old, "N={} skip={skip} input={:?}", $n, String::from_utf8_lossy(bytes));
                    accepted += new.len();
                }};
            }
            one!(1);
            one!(2);
            one!(4);
            one!(7);
            one!(16);
        }
        accepted
    }

    /// The fused field construction emits the same subslices (same offsets
    /// into the input, not just equal bytes) as the 341a296 code, on the hand
    /// cases plus random adversarial buffers: wrapping, interior and lone
    /// quotes, empty fields, spaces and `\r`, wrong comma counts, lines placed
    /// across 64-byte block edges, and last lines with and without `\n`.
    #[test]
    fn fused_finish_matches_v0_offsets() {
        let mut accepted = 0usize;
        for case in [
            &b""[..], b"\n", b"h\na,b,c,d", b"h\n\"a\",\"\",c,d\n", b"h\n\"a\"b\",c,d,e\n a ,b ,c\r,d\r\n",
            b"h\na,b,c,\"\"", b"h\na,b,c,\"", b"h\n\"\"\"\",b,c,d", b"h\n\",\",\",\"", b"\"\",\"\",\"\",\"\"",
            b"\"x\",\"\",\"\"\"\",\" \",\"\r\",\"a,b\"\n", b",,,\n,,,,,,\n\"\",,\"\",\n",
        ] {
            accepted += check_spans(case);
        }
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let atoms: [&[u8]; 12] = [b",", b"\"", b"\"\"", b"\r", b" ", b"a", b"u_12", b"\xef\xbc\x92", b"", b"\n", b"\"x\"", b"e0123456789abcdef0123456789abcdef"];
        for _ in 0..20_000 {
            let mut buf = Vec::new();
            let target = (next() % 200) as usize;
            while buf.len() < target {
                let fields = 1 + (next() % 8) as usize;
                for k in 0..fields {
                    let quoted = next() % 4 == 0;
                    if quoted {
                        buf.push(b'"');
                    }
                    for _ in 0..next() % 3 {
                        buf.extend_from_slice(atoms[(next() % atoms.len() as u64) as usize]);
                    }
                    if quoted {
                        buf.push(b'"');
                    }
                    if k + 1 < fields {
                        buf.push(b',');
                    }
                }
                if next() % 6 != 0 {
                    buf.push(b'\n');
                }
            }
            accepted += check_spans(&buf);
        }
        assert!(accepted > 10_000, "only {accepted} accepted records");
    }
}
