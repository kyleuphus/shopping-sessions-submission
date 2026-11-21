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
        let mut fields: [&'a [u8]; N] = [&[]; N];
        let mut start = line_start;
        for (k, field) in fields.iter_mut().enumerate().take(N - 1) {
            *field = &buf[start..commas[k]];
            start = commas[k] + 1;
        }
        fields[N - 1] = &buf[start..end];
        if quotes != 0 {
            // Equivalent to `csvline::unquote` on every field, using the
            // line's quote count from the bitmasks: strip each field that is
            // wrapped in a pair (`"..."`, length >= 2). The line is well-formed
            // iff every quote in it was one end of such a pair, i.e. iff the
            // stripped pairs account for all `quotes` quote bytes. Any other
            // quote (lone, inside, at one end only) leaves the counts unequal.
            let mut stripped = 0u32;
            for field in fields.iter_mut() {
                let n = field.len();
                if n >= 2 && field[0] == b'"' && field[n - 1] == b'"' {
                    *field = &field[1..n - 1];
                    stripped += 2;
                }
            }
            if stripped != quotes {
                return; // a quote not wrapping a whole field: malformed
            }
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
        ] {
            check::<4>(case);
            check::<7>(case);
            check::<1>(case);
        }
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
}
