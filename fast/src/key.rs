//! Order-preserving packing for the short, generator-shaped ID fast path.
//!
//! The 64 accepted bytes are assigned six-bit codes in ASCII order. Ten codes
//! fill bits 63..4, left aligned, and the byte length occupies bits 3..0.
//! Consequently, the first differing byte decides integer order exactly as it
//! decides slice order. If one input is a prefix of the other, its zero padding
//! can only tie actual `'-'` bytes (code zero); a later nonzero code makes the
//! longer input larger, and otherwise the low length nibble breaks the tie in
//! favour of the shorter input.

use std::cmp::Ordering;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::hash::{hash_bytes, FastMap};

/// The integer representation used by the v2 pipeline for every nonempty ID.
///
/// Packed keys have a length nibble in `1..=10`. A zero length nibble marks an
/// exact fallback and the upper 60 bits hold its index in [`IdInterner`].
pub type IdKey = u64;

/// log2 of the number of interner shards. 1024 shards keep lock contention
/// negligible when 1 in 8 distinct IDs is a (33-byte, unpackable) fallback.
const SHARD_BITS: u32 = 10;
const FALLBACK_SHARDS: usize = 1 << SHARD_BITS;
/// Shard-local indices must fit in the bits above the shard number of a u32.
const MAX_LOCAL: u32 = 1 << (32 - SHARD_BITS);

/// One shard: forward map (exact bytes -> local index) and reverse vector
/// (local index -> bytes). Both are only ever touched under the shard mutex.
#[derive(Default)]
struct Shard {
    map: FastMap<Arc<[u8]>, u32>,
    rev: Vec<Arc<[u8]>>,
}

/// Concurrent, exact storage for IDs which [`pack`] cannot represent.
///
/// The hash of the bytes selects one of 1024 shards; equality inside a shard is
/// always on the complete byte string, so the hash never affects correctness.
/// A fallback's global index is `(local << SHARD_BITS) | shard`, so reverse
/// lookups decode the shard directly and lock only that shard. There is no
/// global lock and no global counter on the insert path.
pub struct IdInterner {
    shards: Box<[Mutex<Shard>]>,
    /// Number of distinct fallbacks. Informational (tests only), Relaxed.
    count: AtomicUsize,
}

impl Default for IdInterner {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn encode_index(shard: usize, local: u32) -> u32 {
    debug_assert!(shard < FALLBACK_SHARDS);
    (local << SHARD_BITS) | shard as u32
}

#[inline]
fn decode_index(index: u32) -> (usize, usize) {
    (
        (index as usize) & (FALLBACK_SHARDS - 1),
        (index >> SHARD_BITS) as usize,
    )
}

impl IdInterner {
    pub fn new() -> Self {
        Self {
            shards: (0..FALLBACK_SHARDS)
                .map(|_| Mutex::new(Shard::default()))
                .collect(),
            count: AtomicUsize::new(0),
        }
    }

    #[inline]
    fn lock(&self, shard: usize) -> MutexGuard<'_, Shard> {
        self.shards[shard]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn intern(&self, bytes: &[u8]) -> u32 {
        // Top bits: the map's own hashing uses the low bits of a different
        // (FastHasher-combined) value, but taking the top bits here keeps the
        // shard choice independent of in-shard bucket choice regardless.
        let shard_index = (hash_bytes(bytes) >> (64 - SHARD_BITS)) as usize;
        let mut shard = self.lock(shard_index);
        if let Some(&local) = shard.map.get(bytes) {
            return encode_index(shard_index, local);
        }

        let local = shard.rev.len() as u32;
        assert!(
            local < MAX_LOCAL,
            "fallback ID interner shard {shard_index} exceeded {MAX_LOCAL} entries"
        );
        let owned: Arc<[u8]> = Arc::from(bytes);
        // Both the forward and reverse entries are installed under the same
        // shard lock, so any index ever returned (to this or another thread)
        // has its bytes visible to a later reader that locks this shard.
        shard.rev.push(owned.clone());
        shard.map.insert(owned, local);
        drop(shard);
        self.count.fetch_add(1, AtomicOrdering::Relaxed);
        encode_index(shard_index, local)
    }

    /// The interned bytes for `index` (shard lock held only for the clone).
    #[inline]
    fn get_arc(&self, index: u32) -> Arc<[u8]> {
        let (shard, local) = decode_index(index);
        let guard = self.lock(shard);
        guard
            .rev
            .get(local)
            .cloned()
            .unwrap_or_else(|| panic!("fallback key has no interner entry: {index}"))
    }

    fn copy_bytes(&self, index: u32, out: &mut Vec<u8>) {
        let (shard, local) = decode_index(index);
        let guard = self.lock(shard);
        let bytes = guard
            .rev
            .get(local)
            .unwrap_or_else(|| panic!("fallback key has no interner entry: {index}"));
        out.clear();
        out.extend_from_slice(bytes);
    }

    pub fn len(&self) -> usize {
        self.count.load(AtomicOrdering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Return the exact key for one nonempty ID.
///
/// Empty fields have stage-specific meaning and must be handled by the caller.
#[inline]
pub fn key_of(bytes: &[u8], interner: &IdInterner) -> IdKey {
    assert!(!bytes.is_empty(), "empty IDs are handled by the caller");
    pack(bytes).unwrap_or_else(|| (interner.intern(bytes) as u64) << 4)
}

/// Materialize the bytes represented by `key` into `buf`.
///
/// The returned slice borrows `buf`; this keeps lookup safe while other threads
/// add fallback IDs to the interner. `key` must have been made with this same
/// interner.
pub fn key_bytes<'a>(key: IdKey, interner: &IdInterner, buf: &'a mut Vec<u8>) -> &'a [u8] {
    let len = (key & 0x0f) as usize;
    if len == 0 {
        interner.copy_bytes((key >> 4) as u32, buf);
    } else {
        assert!(len <= 10, "invalid packed ID length nibble: {len}");
        let mut packed = [0; 10];
        let written = unpack(key, &mut packed);
        buf.clear();
        buf.extend_from_slice(&packed[..written]);
    }
    buf
}

/// The bytes of one key without heap allocation: packed keys unpack onto the
/// stack, fallbacks share the interner's `Arc`.
enum KeyBytes {
    Packed([u8; 10], usize),
    Fallback(Arc<[u8]>),
}

impl KeyBytes {
    #[inline]
    fn new(key: IdKey, interner: &IdInterner) -> Self {
        let len = (key & 0x0f) as usize;
        if len == 0 {
            KeyBytes::Fallback(interner.get_arc((key >> 4) as u32))
        } else {
            assert!(len <= 10, "invalid packed ID length nibble: {len}");
            let mut buf = [0; 10];
            let written = unpack(key, &mut buf);
            KeyBytes::Packed(buf, written)
        }
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        match self {
            KeyBytes::Packed(buf, len) => &buf[..*len],
            KeyBytes::Fallback(bytes) => bytes,
        }
    }
}

/// Compare the underlying ID bytes exactly.
///
/// Packing is order preserving, so the overwhelmingly common packed/packed
/// case is one integer comparison. Any comparison involving a fallback
/// materializes the real bytes (stack or shared `Arc`, no allocation) and uses
/// slice ordering.
#[inline]
pub fn key_cmp(a: IdKey, b: IdKey, interner: &IdInterner) -> Ordering {
    if a & 0x0f != 0 && b & 0x0f != 0 {
        return a.cmp(&b);
    }
    if a == b {
        return Ordering::Equal;
    }
    let a_bytes = KeyBytes::new(a, interner);
    let b_bytes = KeyBytes::new(b, interner);
    a_bytes.as_slice().cmp(b_bytes.as_slice())
}

const INVALID: u8 = 0xff;

const fn make_codes() -> [u8; 256] {
    let mut codes = [INVALID; 256];
    codes[b'-' as usize] = 0;

    let mut i = 0;
    while i < 10 {
        codes[(b'0' + i as u8) as usize] = 1 + i as u8;
        i += 1;
    }

    i = 0;
    while i < 26 {
        codes[(b'A' + i as u8) as usize] = 11 + i as u8;
        i += 1;
    }

    codes[b'_' as usize] = 37;

    i = 0;
    while i < 26 {
        codes[(b'a' + i as u8) as usize] = 38 + i as u8;
        i += 1;
    }

    codes
}

const CODES: [u8; 256] = make_codes();
const ALPHABET: [u8; 64] = *b"-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";

/// Pack an ID into an order-preserving integer.
///
/// Returns `Some` exactly for nonempty inputs of at most ten bytes whose every
/// byte is in `-0-9A-Z_a-z`. All other inputs must stay on the exact-byte path.
#[inline]
pub fn pack(b: &[u8]) -> Option<u64> {
    #[cfg(all(target_arch = "x86_64", target_feature = "ssse3", target_feature = "bmi2"))]
    if (8..=10).contains(&b.len()) {
        return pack_long_x86(b);
    }

    pack_scalar(b)
}

#[inline]
fn pack_scalar(b: &[u8]) -> Option<u64> {
    let len = b.len();
    if len == 0 || len > 10 {
        return None;
    }

    let mut payload = 0_u64;
    for &byte in b {
        let code = CODES[byte as usize];
        if code == INVALID {
            return None;
        }
        payload = (payload << 6) | code as u64;
    }

    Some((payload << ((10 - len) * 6 + 4)) | len as u64)
}

#[cfg(all(target_arch = "x86_64", target_feature = "ssse3", target_feature = "bmi2"))]
#[inline]
fn pack_long_x86(b: &[u8]) -> Option<u64> {
    use std::arch::x86_64::*;

    const VALID_HIGH_BY_LOW: [u8; 16] = {
        let mut masks = [0; 16];
        let mut byte = 0;
        while byte < 128 {
            if CODES[byte] != INVALID {
                masks[byte & 15] |= 1 << (byte >> 4);
            }
            byte += 1;
        }
        masks
    };

    let len = b.len();
    let prefix = u64::from_be_bytes(b[..8].try_into().unwrap());
    let tail = u16::from_be_bytes(b[len - 2..].try_into().unwrap());
    // SAFETY: SSSE3 and BMI2 are enabled by the cfg. The SIMD input is built
    // from two in-bounds reads, with overlap when there are fewer than ten bytes.
    unsafe {
        let bytes = _mm_set_epi64x(tail as i64, prefix as i64);
        let nibble_mask = _mm_set1_epi8(15);
        let low = _mm_and_si128(bytes, nibble_mask);
        let high = _mm_and_si128(_mm_srli_epi16::<4>(bytes), nibble_mask);
        let allowed = _mm_shuffle_epi8(
            _mm_loadu_si128(VALID_HIGH_BY_LOW.as_ptr().cast()),
            low,
        );
        let high_bit = _mm_shuffle_epi8(
            _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 0, 0, 0, 0, 0, 0, 0, 0),
            high,
        );
        let invalid = _mm_cmpeq_epi8(_mm_and_si128(allowed, high_bit), _mm_setzero_si128());
        if _mm_movemask_epi8(invalid) & 0x03ff != 0 {
            return None;
        }

        // Each ASCII range has a constant byte-minus-code offset. Underscore
        // shares a high nibble with P..Z but has an offset four larger.
        let offsets = _mm_shuffle_epi8(
            _mm_setr_epi8(0, 0, 45, 47, 54, 54, 59, 59, 0, 0, 0, 0, 0, 0, 0, 0),
            high,
        );
        let underscore = _mm_cmpeq_epi8(bytes, _mm_set1_epi8(b'_' as i8));
        let codes = _mm_sub_epi8(
            _mm_sub_epi8(bytes, offsets),
            _mm_and_si128(underscore, _mm_set1_epi8(4)),
        );

        // The big-endian reads put the first byte in the highest lane of
        // each chunk. PEXT preserves that order as it removes lane padding.
        let prefix = _pext_u64(_mm_cvtsi128_si64(codes) as u64, 0x3f3f_3f3f_3f3f_3f3f);
        let tail = _pext_u64(_mm_cvtsi128_si64(_mm_srli_si128::<8>(codes)) as u64, 0x3f3f);
        let padding = (10 - len) * 6;
        let tail = (tail & (0xfff >> padding)) << (padding + 4);
        Some((prefix << 16) | tail | len as u64)
    }
}

/// Write the bytes represented by a key returned from [`pack`] into `buf`.
///
/// Returns the number of bytes written.
#[inline]
pub fn unpack(k: u64, buf: &mut [u8; 10]) -> usize {
    let len = (k & 0x0f) as usize;
    assert!(len <= 10, "key was not produced by pack");

    let mut i = 0;
    while i < len {
        let code = ((k >> (58 - i * 6)) & 0x3f) as usize;
        buf[i] = ALPHABET[code];
        i += 1;
    }
    len
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    const SUBSET: [u8; 10] = [b'-', b'0', b'9', b'A', b'Z', b'_', b'a', b'z', b'5', b'm'];
    const MIXED: &[u8] =
        b"-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz \"\r\xc3\xff.+";

    fn unpacked(k: u64) -> Vec<u8> {
        let mut buf = [0; 10];
        let len = unpack(k, &mut buf);
        buf[..len].to_vec()
    }

    fn reference_packable(b: &[u8]) -> bool {
        (1..=10).contains(&b.len())
            && b.iter().all(|byte| {
                matches!(
                    byte,
                    b'-' | b'0'..=b'9' | b'A'..=b'Z' | b'_' | b'a'..=b'z'
                )
            })
    }

    fn append_strings(out: &mut Vec<Vec<u8>>, prefix: &mut Vec<u8>, remaining: usize) {
        if remaining == 0 {
            out.push(prefix.clone());
            return;
        }
        for &byte in &SUBSET {
            prefix.push(byte);
            append_strings(out, prefix, remaining - 1);
            prefix.pop();
        }
    }

    #[test]
    fn exhaustive_len_zero_through_three_and_all_pair_orderings() {
        let mut strings = Vec::with_capacity(1 + 10 + 100 + 1_000);
        for len in 0..=3 {
            append_strings(&mut strings, &mut Vec::with_capacity(len), len);
        }
        assert_eq!(strings.len(), 1_111);
        assert_eq!(pack(&strings[0]), None);

        let packable: Vec<(&[u8], u64)> = strings
            .iter()
            .filter_map(|s| pack(s).map(|k| (s.as_slice(), k)))
            .collect();
        assert_eq!(packable.len(), 1_110);

        for &(bytes, key) in &packable {
            assert_eq!(unpacked(key), bytes);
        }

        // 1,232,100 ordered pairs prove ordering and injectivity over this set.
        for &(a, ka) in &packable {
            for &(b, kb) in &packable {
                assert_eq!(ka.cmp(&kb), a.cmp(b), "ordering mismatch: {a:?}, {b:?}");
                assert_eq!(ka == kb, a == b, "injectivity mismatch: {a:?}, {b:?}");
            }
        }
    }

    struct XorShift64(u64);

    impl XorShift64 {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn fill(&mut self, out: &mut [u8]) {
            for byte in out {
                *byte = MIXED[self.next() as usize % MIXED.len()];
            }
        }
    }

    #[test]
    fn deterministic_two_million_pair_fuzz() {
        let mut rng = XorShift64(0x4d59_5df4_d0f3_3173);
        let mut a_buf = [0; 12];
        let mut b_buf = [0; 12];

        for pair_index in 0..2_000_000 {
            let a_len = (rng.next() % 12 + 1) as usize;
            let mut b_len = (rng.next() % 12 + 1) as usize;
            rng.fill(&mut a_buf[..a_len]);
            rng.fill(&mut b_buf[..b_len]);

            // Exercise equality regularly instead of leaving it to chance.
            if pair_index % 1_024 == 0 {
                b_buf = a_buf;
                b_len = a_len;
            }

            let a = &a_buf[..a_len];
            let b = &b_buf[..b_len];
            let ka = pack(a);
            let kb = pack(b);

            assert_eq!(ka.is_some(), reference_packable(a), "packability: {a:?}");
            assert_eq!(kb.is_some(), reference_packable(b), "packability: {b:?}");

            if let Some(key) = ka {
                assert_eq!(unpacked(key), a, "round trip: {a:?}");
            }
            if let Some(key) = kb {
                assert_eq!(unpacked(key), b, "round trip: {b:?}");
            }
            if let (Some(a_key), Some(b_key)) = (ka, kb) {
                assert_eq!(a_key.cmp(&b_key), a.cmp(b), "ordering: {a:?}, {b:?}");
                assert_eq!(a_key == b_key, a == b, "injectivity: {a:?}, {b:?}");
            }
        }
    }

    #[test]
    fn edge_cases_and_exact_byte_semantics() {
        assert_eq!(pack(b""), None);
        assert!(pack(b"0123456789").is_some());
        assert_eq!(pack(b"0123456789a"), None);

        let upper = pack(b"U1").unwrap();
        let lower = pack(b"u1").unwrap();
        assert!(upper < lower);
        assert_eq!(upper.cmp(&lower), b"U1"[..].cmp(&b"u1"[..]));

        for id in [b"a1b2c3d4e".as_slice(), b"0000abc42", b"zzzzzzzzz"] {
            let key = pack(id).expect("nine-byte lowercase base36 ID should pack");
            assert_eq!(unpacked(key), id);
        }

        assert_eq!(pack(b" u1"), None);
        assert_eq!(pack(b"u1 "), None);
        assert_eq!(pack(b"\"u1\""), None);
        assert_eq!(pack(b"u1\r"), None);
        assert_eq!(pack(b"u1\xff"), None);
    }

    #[test]
    fn mixed_packed_and_fallback_keys_are_exact_and_ordered() {
        let interner = IdInterner::new();
        let ids: Vec<&[u8]> = vec![
            b"-",
            b"0",
            b"U1",
            b"u1",
            b" u1",
            b"u1 ",
            b"u1\r",
            b"a,b",
            b"\xc3\xa9",
            b"\xff",
            b"0123456789",
            b"0123456789a",
            b"a very long fallback id",
        ];
        let keys: Vec<IdKey> = ids.iter().map(|id| key_of(id, &interner)).collect();

        for (id, &key) in ids.iter().zip(&keys) {
            let mut buf = Vec::new();
            assert_eq!(key_bytes(key, &interner, &mut buf), *id);
        }
        for (a_index, a) in ids.iter().enumerate() {
            for (b_index, b) in ids.iter().enumerate() {
                assert_eq!(keys[a_index] == keys[b_index], a == b);
                assert_eq!(key_cmp(keys[a_index], keys[b_index], &interner), a.cmp(b));
            }
        }
    }

    #[test]
    fn fallback_interning_is_consistent_across_threads() {
        let interner = IdInterner::new();
        let ids: Vec<Vec<u8>> = (0..1_000)
            .map(|n| format!("long fallback id number {n:04}").into_bytes())
            .collect();
        std::thread::scope(|scope| {
            let mut joins = Vec::new();
            for _ in 0..8 {
                joins.push(scope.spawn(|| {
                    ids.iter()
                        .map(|id| key_of(id, &interner))
                        .collect::<Vec<_>>()
                }));
            }
            let expected = joins.remove(0).join().unwrap();
            for join in joins {
                assert_eq!(join.join().unwrap(), expected);
            }
        });
        assert_eq!(interner.len(), ids.len());
    }

    #[test]
    fn mixed_key_fuzz_agrees_with_byte_equality_and_order() {
        let interner = IdInterner::new();
        let mut rng = XorShift64(0xd1b5_4a32_d192_ed03);
        let alphabet = b"-09AZ_az \r\t\x01\x7f\x80\xc3\xff.+";
        let mut a_buf = [0; 32];
        let mut b_buf = [0; 32];

        for pair_index in 0..200_000 {
            let a_len = (rng.next() % 32 + 1) as usize;
            let mut b_len = (rng.next() % 32 + 1) as usize;
            for byte in &mut a_buf[..a_len] {
                *byte = alphabet[rng.next() as usize % alphabet.len()];
            }
            for byte in &mut b_buf[..b_len] {
                *byte = alphabet[rng.next() as usize % alphabet.len()];
            }
            if pair_index % 257 == 0 {
                b_buf = a_buf;
                b_len = a_len;
            }

            let a = &a_buf[..a_len];
            let b = &b_buf[..b_len];
            let a_key = key_of(a, &interner);
            let b_key = key_of(b, &interner);
            assert_eq!(a_key == b_key, a == b, "equality mismatch: {a:?}, {b:?}");
            assert_eq!(
                key_cmp(a_key, b_key, &interner),
                a.cmp(b),
                "order mismatch: {a:?}, {b:?}"
            );
        }
    }

    #[test]
    fn zero_code_prefix_tiebreak_examples() {
        let ids: [&[u8]; 6] = [b"u1", b"u1-", b"u1--", b"u1-0", b"u10", b"u2"];
        for pair in ids.windows(2) {
            assert!(pair[0] < pair[1]);
            assert!(pack(pair[0]).unwrap() < pack(pair[1]).unwrap());
        }
    }

    #[test]
    fn pack_scalar_differential() {
        let mut rng = XorShift64(0x9e37_79b9_7f4a_7c15);
        let mut buf = [0; 16];
        let check = |b: &[u8]| assert_eq!(pack(b), pack_scalar(b), "ID {b:?}");
        for _ in 0..1_700_000 {
            let len = (rng.next() % 17) as usize;
            let b = &mut buf[..len];
            for byte in b.iter_mut() {
                *byte = rng.next() as u8;
            }
            check(b);

            for byte in b.iter_mut() {
                *byte = ALPHABET[(rng.next() & 63) as usize];
            }
            check(b);
            if len != 0 {
                b[rng.next() as usize % len] ^= (rng.next() % 255 + 1) as u8;
            }
            check(b);
        }

        for len in 0..=16 {
            let mut dashes = [b'-'; 16];
            check(&dashes[..len]);
            for pos in 0..len {
                for byte in 0..=255 {
                    dashes[pos] = byte;
                    check(&dashes[..len]);
                }
                dashes[pos] = b'-';
            }
        }
    }

    /// One random ID: 33-char generator-style fallback, short packable, or
    /// odd bytes (spaces, `\r`, non-UTF-8). Never empty.
    fn random_id(rng: &mut XorShift64) -> Vec<u8> {
        const HEX: &[u8] = b"0123456789abcdef";
        const ODD: &[u8] = b"-0aZ_ \r\t,\"\x00\x7f\x80\xc3\xff";
        match rng.next() % 5 {
            0 | 1 => {
                let mut id = vec![b"uec"[(rng.next() % 3) as usize]];
                id.extend((0..32).map(|_| HEX[(rng.next() & 15) as usize]));
                id
            }
            2 => {
                let len = (rng.next() % 10 + 1) as usize;
                (0..len).map(|_| ALPHABET[(rng.next() & 63) as usize]).collect()
            }
            3 => {
                // Short hex-ish, often a prefix of a 33-char ID.
                let len = (rng.next() % 6 + 1) as usize;
                let mut id = vec![b"uec"[(rng.next() % 3) as usize]];
                id.extend((1..len).map(|_| HEX[(rng.next() & 15) as usize]));
                id
            }
            _ => {
                let len = (rng.next() % 14 + 1) as usize;
                (0..len).map(|_| ODD[rng.next() as usize % ODD.len()]).collect()
            }
        }
    }

    #[test]
    fn concurrent_stress_overlapping_sets_are_exact() {
        use std::collections::HashMap;
        let mut rng = XorShift64(0x1234_5678_9abc_def1);
        let universe: Vec<Vec<u8>> = (0..50_000).map(|_| random_id(&mut rng)).collect();
        let interner = IdInterner::new();
        let results: Vec<Vec<(usize, IdKey)>> = std::thread::scope(|scope| {
            let joins: Vec<_> = (0..8u64)
                .map(|t| {
                    let universe = &universe;
                    let interner = &interner;
                    scope.spawn(move || {
                        // Each thread visits an overlapping, shuffled subset.
                        let mut rng = XorShift64(0x9e37_79b9_7f4a_7c15 ^ (t + 1) * 0x1000_0001);
                        (0..50_000)
                            .map(|_| {
                                let i = rng.next() as usize % universe.len();
                                (i, key_of(&universe[i], interner))
                            })
                            .collect()
                    })
                })
                .collect();
            joins.into_iter().map(|j| j.join().unwrap()).collect()
        });

        let mut key_by_bytes: HashMap<&[u8], IdKey> = HashMap::new();
        let mut bytes_by_key: HashMap<IdKey, &[u8]> = HashMap::new();
        let mut buf = Vec::new();
        for (i, key) in results.into_iter().flatten() {
            let bytes = universe[i].as_slice();
            assert_eq!(*key_by_bytes.entry(bytes).or_insert(key), key, "same bytes, different keys: {bytes:?}");
            assert_eq!(*bytes_by_key.entry(key).or_insert(bytes), bytes, "same key, different bytes");
            assert_eq!(key_bytes(key, &interner, &mut buf), bytes);
            assert_eq!(key & 0x0f == 0, pack(bytes).is_none());
        }
        let fallbacks = key_by_bytes.keys().filter(|b| pack(b).is_none()).count();
        assert_eq!(interner.len(), fallbacks);
        assert!(fallbacks > 10_000, "stress should exercise many fallbacks: {fallbacks}");
        // Interning again on one thread returns the same keys.
        for (bytes, &key) in &key_by_bytes {
            assert_eq!(key_of(bytes, &interner), key);
        }
    }

    #[test]
    fn key_cmp_matches_bytes_for_packed_fallback_mixes() {
        let interner = IdInterner::new();
        let mut rng = XorShift64(0x0bad_cafe_dead_beef);
        let mut pool: Vec<Vec<u8>> = (0..4_000).map(|_| random_id(&mut rng)).collect();
        // Explicit prefix relations between fallbacks and packed keys.
        let long = b"e0123456789abcdef0123456789abcdef".to_vec();
        for cut in 1..=long.len() {
            pool.push(long[..cut].to_vec());
        }
        pool.push(b"e123".to_vec());
        pool.push(b"e123 ".to_vec());
        pool.push(b"e1230000000000000000000000000000".to_vec());
        let keys: Vec<IdKey> = pool.iter().map(|b| key_of(b, &interner)).collect();
        // All pairs among the explicit tail, plus random pairs.
        let tail = pool.len() - long.len() - 3;
        for a in tail..pool.len() {
            for b in tail..pool.len() {
                assert_eq!(key_cmp(keys[a], keys[b], &interner), pool[a].cmp(&pool[b]));
            }
        }
        for _ in 0..300_000 {
            let a = rng.next() as usize % pool.len();
            let b = rng.next() as usize % pool.len();
            assert_eq!(
                key_cmp(keys[a], keys[b], &interner),
                pool[a].cmp(&pool[b]),
                "{:?} vs {:?}",
                pool[a],
                pool[b]
            );
            assert_eq!(keys[a] == keys[b], pool[a] == pool[b]);
        }
    }

    #[test]
    fn fallback_index_encoding_round_trips() {
        for shard in [0, 1, 7, FALLBACK_SHARDS - 2, FALLBACK_SHARDS - 1] {
            for local in [0u32, 1, 2, 12_345, MAX_LOCAL - 1] {
                let index = encode_index(shard, local);
                assert_eq!(decode_index(index), (shard, local as usize));
                let key = (index as u64) << 4;
                assert_eq!(key & 0x0f, 0);
                assert_eq!((key >> 4) as u32, index);
            }
        }
        let interner = IdInterner::new();
        let mut rng = XorShift64(0x5555_aaaa_1234_9876);
        let mut buf = Vec::new();
        for _ in 0..20_000 {
            let id = random_id(&mut rng);
            let key = key_of(&id, &interner);
            if pack(&id).is_none() {
                assert_eq!(key & 0x0f, 0);
                let (shard, local) = decode_index((key >> 4) as u32);
                assert!(shard < FALLBACK_SHARDS);
                assert_eq!(&*interner.lock(shard).rev[local], id.as_slice());
            }
            assert_eq!(key_bytes(key, &interner, &mut buf), id.as_slice());
        }
    }

    #[test]
    #[ignore = "microbenchmark: ten million mixed ID packs"]
    fn pack_cost_probe() {
        const COUNT: usize = 10_000_000;
        let mut rng = XorShift64(0x4d59_5df4_d0f3_3173);
        let ids: Vec<Vec<u8>> = (0..4_000)
            .map(|i| {
                let mut id = match i % 7 {
                    0 | 1 => format!("u{:07}", rng.next() % 10_000_000).into_bytes(),
                    2 | 3 => format!("e{:08}", rng.next() % 100_000_000).into_bytes(),
                    4 => format!("c_{}", rng.next() % 100).into_bytes(),
                    5 => format!("p{}", rng.next() % 1_000).into_bytes(),
                    _ => (0..2 + rng.next() % 9)
                        .map(|_| ALPHABET[(rng.next() & 63) as usize])
                        .collect(),
                };
                if i % 20 == 0 {
                    let pos = rng.next() as usize % id.len();
                    id[pos] = b' ';
                }
                id
            })
            .collect();

        let started = Instant::now();
        for id in ids.iter().cycle().take(COUNT) {
            black_box(pack(black_box(id.as_slice())));
        }
        eprintln!(
            "pack_cost_probe: {:.3} ns/call",
            started.elapsed().as_nanos() as f64 / COUNT as f64
        );
    }

    #[test]
    #[ignore = "microbenchmark: packs ten million nine-byte generator-like IDs"]
    fn benchmark_pack_10m() {
        const COUNT: usize = 10_000_000;
        const BASE36: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
        let mut ids = [[b'0'; 9]; 1_024];

        for (i, id) in ids.iter_mut().enumerate() {
            let mut n = (i as u64).wrapping_mul(6_364_136_223_846_793_005);
            for byte in id.iter_mut().rev() {
                *byte = BASE36[(n % 36) as usize];
                n /= 36;
            }
        }

        let started = Instant::now();
        for i in 0..COUNT {
            let id = black_box(&ids[i & (ids.len() - 1)]);
            black_box(pack(id).unwrap());
        }
        let elapsed = started.elapsed();
        let ns_per_pack = elapsed.as_nanos() as f64 / COUNT as f64;
        eprintln!(
            "packed {COUNT} generator-like 9-byte IDs in {elapsed:?}: {ns_per_pack:.3} ns/pack"
        );
    }
}
