//! Clock field decoder shared by F1, F2 and zero-padded F2 offsets.
#![cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]

use std::arch::x86_64::*;

/// Input is little-endian HH:MM:SS XORed with ASCII `00:00:00`.
/// With CHECK, a nonzero second result rejects the clock (ignore the value).
/// Without CHECK the caller guarantees a valid clock and the flag is zero.
#[inline(always)]
pub(crate) fn clock_seconds<const CHECK: bool>(digits: u64) -> (i64, u64) {
    // SAFETY: the module requires SSSE3. All operations are register-only.
    unsafe {
        let bytes = _mm_cvtsi64_si128(digits as i64);
        // Work with the natural adjacent pairs: hour, ten times minute tens,
        // minute units, second. Both separator bytes have weight zero.
        let decimal = _mm_setr_epi8(10, 1, 0, 10, 1, 0, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0);
        let pairs = _mm_maddubs_epi16(bytes, decimal);
        let parts = _mm_madd_epi16(pairs, _mm_setr_epi16(3600, 60, 60, 1, 0, 0, 0, 0));
        let total = _mm_add_epi32(parts, _mm_srli_si128::<4>(parts));

        let bad = if CHECK {
            // Unsigned saturating addition sets the sign bit iff a byte
            // exceeds its limit. The normalized separators must be zero.
            let bias = _mm_setr_epi8(125, 118, 127, 122, 118, 127, 122, 118, 0, 0, 0, 0, 0, 0, 0, 0);
            let invalid = _mm_movemask_epi8(_mm_adds_epu8(bytes, bias));
            // Compare the two digit bytes lexicographically against 23.
            invalid as u64 | u64::from((digits as u16).swap_bytes() > 0x0203)
        } else {
            0
        };
        (i64::from(_mm_cvtsi128_si32(total)), bad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MASK: u64 = u64::from_le_bytes(*b"00:00:00");

    fn check(raw: [u8; 8]) {
        let [a, b, sep1, c, d, sep2, e, f] = raw;
        let pair = |x: u8, y: u8| (i64::from(x) - 48) * 10 + i64::from(y) - 48;
        let (h, m, s) = (pair(a, b), pair(c, d), pair(e, f));
        let valid = sep1 == b':'
            && sep2 == b':'
            && [a, b, c, d, e, f].iter().all(u8::is_ascii_digit)
            && h < 24
            && m < 60
            && s < 60;
        let digits = u64::from_le_bytes(raw) ^ MASK;
        let (value, bad) = clock_seconds::<true>(digits);
        assert_eq!(bad == 0, valid, "{raw:?}");
        if valid {
            assert_eq!(value, h * 3600 + m * 60 + s);
            assert_eq!(clock_seconds::<false>(digits), (value, 0));
        }
    }

    #[test]
    fn every_clock() {
        for h in 0..24 {
            for m in 0..60 {
                for s in 0..60 {
                    check([
                        b'0' + h / 10, b'0' + h % 10, b':',
                        b'0' + m / 10, b'0' + m % 10, b':',
                        b'0' + s / 10, b'0' + s % 10,
                    ]);
                }
            }
        }
    }

    #[test]
    fn arbitrary_bytes() {
        for base in [*b"00:00:00", *b"19:59:59", *b"23:59:59"] {
            for pos in 0..8 {
                for byte in 0..=255 {
                    let mut raw = base;
                    raw[pos] = byte;
                    check(raw);
                }
            }
        }
        let mut x = 0x93ab_f741_7d62_8591u64;
        for _ in 0..200_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            check(x.to_le_bytes());
        }
    }
}
