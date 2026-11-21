//! Maps byte-string IDs to dense `u32` handles.
//!
//! IDs in `rules.md` are compared exactly and byte-wise — never trimmed, never
//! case-folded — so they are opaque byte strings, not text. Interning them to
//! `u32` lets the hot stages (sort, group-by, union-find) work on integers while
//! the original bytes are kept only for output, where a person's name must be
//! emitted verbatim (it can contain tabs, spaces and uppercase — see fixture L3).
//!
//! Borrow the original bytes from the mapped files and lookup buffers. Both
//! directions of the lookup can share the same slice for the life of the run.

use crate::hash::hash_bytes;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};

/// Map key: the ID's bytes plus their precomputed `hash_bytes`. Hashing a key
/// just hands over the stored hash, so a caller that already hashed the ID
/// (person routing needs `hash_bytes(user)` for unlinked IDs anyway) can probe
/// without hashing twice. Equality is still on the bytes: a hash match alone
/// never identifies an ID.
#[derive(Clone, Copy)]
struct Key<'a> {
    h: u64,
    b: &'a [u8],
}

impl Hash for Key<'_> {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.h);
    }
}

impl PartialEq for Key<'_> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        // Comparing the hash first is only a fast reject; the bytes decide.
        self.h == other.h && self.b == other.b
    }
}

impl Eq for Key<'_> {}

/// Pass-through hasher for `Key`: `hash_bytes` already ends in `fmix64`, so
/// its bits are well mixed for the table's bucket and tag bits.
#[derive(Default, Clone, Copy)]
struct PassHasher(u64);

impl Hasher for PassHasher {
    fn write(&mut self, _: &[u8]) {
        unreachable!("Key hashes only via write_u64")
    }
    #[inline(always)]
    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }
}

type KeyMap<'a> = HashMap<Key<'a>, u32, BuildHasherDefault<PassHasher>>;

#[derive(Default)]
pub struct Interner<'a> {
    map: KeyMap<'a>,
    keys: Vec<&'a [u8]>,
}

impl<'a> Interner<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            map: KeyMap::with_capacity_and_hasher(n, Default::default()),
            keys: Vec::with_capacity(n),
        }
    }

    /// Intern `b`, returning its handle. Repeated calls with equal bytes return
    /// the same handle.
    pub fn intern(&mut self, b: &'a [u8]) -> u32 {
        let key = Key { h: hash_bytes(b), b };
        *self.map.entry(key).or_insert_with(|| {
            let id = self.keys.len() as u32;
            self.keys.push(b);
            id
        })
    }

    /// Look up an existing handle without interning.
    #[inline]
    pub fn get(&self, b: &[u8]) -> Option<u32> {
        self.get_hashed(b, hash_bytes(b))
    }

    /// `get` with `h == hash_bytes(b)` already computed by the caller, so a
    /// caller that needs the hash anyway hashes once. Passing any other `h` is
    /// a logic error (the lookup would miss); debug builds check it.
    #[inline]
    pub fn get_hashed(&self, b: &[u8], h: u64) -> Option<u32> {
        debug_assert_eq!(h, hash_bytes(b));
        // HashMap is covariant in its key type, so the map can be viewed with
        // the probe's shorter lifetime; nothing is stored through this view.
        let map: &KeyMap<'_> = &self.map;
        map.get(&Key { h, b }).copied()
    }

    /// The bytes behind a handle.
    pub fn bytes(&self, id: u32) -> &'a [u8] {
        self.keys[id as usize]
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// `true` if `a`'s bytes sort before `b`'s. Byte-wise, as `rules.md` requires
    /// everywhere it says "smallest".
    pub fn less(&self, a: u32, b: u32) -> bool {
        self.bytes(a) < self.bytes(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_bytes_same_handle_different_bytes_different_handle() {
        let mut i = Interner::new();
        let a = i.intern(b"u1");
        let b = i.intern(b"u1");
        let c = i.intern(b"u2");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(i.len(), 2);
    }

    #[test]
    fn ids_are_never_trimmed_or_case_folded() {
        // rules.md: ` u1`, `u1 `, `u1` and `U1` are four different IDs.
        let mut i = Interner::new();
        let ids = [
            i.intern(b"u1"),
            i.intern(b" u1"),
            i.intern(b"u1 "),
            i.intern(b"U1"),
        ];
        let mut sorted = ids.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 4, "all four must be distinct IDs");
    }

    #[test]
    fn bytes_round_trip_including_tabs() {
        // Fixture L3's person is named with a tab-containing ID.
        let mut i = Interner::new();
        let id = i.intern(b"A\tB");
        assert_eq!(i.bytes(id), b"A\tB");
    }

    #[test]
    fn less_is_byte_wise_so_uppercase_sorts_before_lowercase() {
        // 'A' = 0x41 < 'u' = 0x75, which is why fixture L3's person is named "A\tB".
        let mut i = Interner::new();
        let a = i.intern(b"A\tB");
        let u = i.intern(b"u1");
        assert!(i.less(a, u));
        assert!(!i.less(u, a));
    }

    #[test]
    fn get_does_not_intern() {
        let mut i = Interner::new();
        i.intern(b"u1");
        assert!(i.get(b"u1").is_some());
        assert!(i.get(b"u2").is_none());
        assert_eq!(i.len(), 1);
    }

    #[test]
    fn get_hashed_matches_get() {
        // get_hashed(b, hash_bytes(b)) must answer exactly like get(b), for
        // present and absent IDs, including near-miss spellings and long IDs.
        let long = vec![b'x'; 300];
        let present: Vec<&[u8]> = vec![b"u1", b" u1", b"U1", b"A\tB", b"", b"12345678", &long];
        let absent: Vec<&[u8]> = vec![b"u1 ", b"u2", b"a\tb", b"\0", b"123456789", &long[1..]];
        let mut i = Interner::new();
        let ids: Vec<u32> = present.iter().map(|b| i.intern(b)).collect();
        for (b, id) in present.iter().zip(&ids) {
            assert_eq!(i.get(b), Some(*id));
            assert_eq!(i.get_hashed(b, hash_bytes(b)), Some(*id));
        }
        for b in &absent {
            assert_eq!(i.get(b), None);
            assert_eq!(i.get_hashed(b, hash_bytes(b)), None);
        }
        // A probe borrowed for less than the interner's lifetime works too.
        let tmp = b"u1".to_vec();
        assert_eq!(i.get_hashed(&tmp, hash_bytes(&tmp)), Some(ids[0]));
    }

    #[test]
    fn get_hashed_agrees_with_get_on_many_ids() {
        let owned: Vec<Vec<u8>> = (0..5000u32).map(|n| format!("user_{n:06}").into_bytes()).collect();
        let mut i = Interner::with_capacity(16);
        for b in owned.iter().step_by(2) {
            i.intern(b);
        }
        for (n, b) in owned.iter().enumerate() {
            let got = i.get_hashed(b, hash_bytes(b));
            assert_eq!(got, i.get(b));
            assert_eq!(got.is_some(), n % 2 == 0);
        }
        assert_eq!(i.len(), 2500);
    }
}
