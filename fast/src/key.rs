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
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

use crate::hash::{hash_bytes, FastMap};

/// The integer representation used by the v2 pipeline for every nonempty ID.
///
/// Packed keys have a length nibble in `1..=10`. A zero length nibble marks an
/// exact fallback and the upper 60 bits hold its index in [`IdInterner`].
pub type IdKey = u64;

const FALLBACK_SHARDS: usize = 64;

/// Concurrent, exact storage for IDs which [`pack`] cannot represent.
///
/// The hash selects one of 64 mutexes, but equality inside a shard is always on
/// the complete byte string. The reverse vector makes fallback keys reversible.
/// Fallbacks are rare in generated data, so this favours simple correctness over
/// a more complicated lock-free allocation scheme.
pub struct IdInterner {
    shards: [Mutex<FastMap<Arc<[u8]>, u32>>; FALLBACK_SHARDS],
    bytes: Mutex<Vec<Option<Arc<[u8]>>>>,
    next: AtomicU32,
}

impl Default for IdInterner {
    fn default() -> Self {
        Self::new()
    }
}

impl IdInterner {
    pub fn new() -> Self {
        Self {
            shards: std::array::from_fn(|_| Mutex::new(FastMap::default())),
            bytes: Mutex::new(Vec::new()),
            next: AtomicU32::new(0),
        }
    }

    fn intern(&self, bytes: &[u8]) -> u32 {
        let hash = hash_bytes(bytes);
        let shard_index = (hash as usize) & (FALLBACK_SHARDS - 1);
        let mut shard = self.shards[shard_index]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(&index) = shard.get(bytes) {
            return index;
        }

        let mut index = self.next.load(AtomicOrdering::Relaxed);
        loop {
            let next = index
                .checked_add(1)
                .unwrap_or_else(|| panic!("fallback ID interner exhausted u32 indices"));
            match self.next.compare_exchange_weak(
                index,
                next,
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => index = actual,
            }
        }
        let owned: Arc<[u8]> = Arc::from(bytes);
        shard.insert(owned.clone(), index);

        // Keep the shard locked until the reverse entry is visible. A second
        // thread interning the same bytes can therefore never observe a key
        // whose reverse entry has not been installed yet.
        let mut reverse = self
            .bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if reverse.len() <= index as usize {
            reverse.resize(index as usize + 1, None);
        }
        debug_assert!(reverse[index as usize].is_none());
        reverse[index as usize] = Some(owned);
        index
    }

    fn copy_bytes(&self, index: u32, out: &mut Vec<u8>) {
        let reverse = self
            .bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let bytes = reverse
            .get(index as usize)
            .and_then(Option::as_deref)
            .unwrap_or_else(|| panic!("fallback key has no interner entry: {index}"));
        out.clear();
        out.extend_from_slice(bytes);
    }

    pub fn len(&self) -> usize {
        self.next.load(AtomicOrdering::Relaxed) as usize
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

/// Compare the underlying ID bytes exactly.
///
/// Packing is order preserving, so the overwhelmingly common packed/packed
/// case is one integer comparison. Any comparison involving a fallback
/// materializes the real bytes and uses slice ordering.
#[inline]
pub fn key_cmp(a: IdKey, b: IdKey, interner: &IdInterner) -> Ordering {
    if a & 0x0f != 0 && b & 0x0f != 0 {
        return a.cmp(&b);
    }
    if a == b {
        return Ordering::Equal;
    }
    let mut a_buf = Vec::new();
    let mut b_buf = Vec::new();
    key_bytes(a, interner, &mut a_buf).cmp(key_bytes(b, interner, &mut b_buf))
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
