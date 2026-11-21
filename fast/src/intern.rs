//! Maps byte-string IDs to dense `u32` handles.
//!
//! IDs in `rules.md` are compared exactly and byte-wise — never trimmed, never
//! case-folded — so they are opaque byte strings, not text. Interning them to
//! `u32` lets the hot stages (sort, group-by, union-find) work on integers while
//! the original bytes are kept only for output, where a person's name must be
//! emitted verbatim (it can contain tabs, spaces and uppercase — see fixture L3).
//!
//! Correctness-first implementation: one allocation per distinct ID. This will
//! show up in profiles at 60M events and is a known optimisation target.

use crate::hash::FastMap;

#[derive(Default)]
pub struct Interner {
    map: FastMap<Box<[u8]>, u32>,
    keys: Vec<Box<[u8]>>,
}

impl Interner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            map: FastMap::with_capacity_and_hasher(n, Default::default()),
            keys: Vec::with_capacity(n),
        }
    }

    /// Intern `b`, returning its handle. Repeated calls with equal bytes return
    /// the same handle.
    pub fn intern(&mut self, b: &[u8]) -> u32 {
        if let Some(&id) = self.map.get(b) {
            return id;
        }
        let id = self.keys.len() as u32;
        let boxed: Box<[u8]> = b.to_vec().into_boxed_slice();
        self.keys.push(boxed.clone());
        self.map.insert(boxed, id);
        id
    }

    /// Look up an existing handle without interning.
    pub fn get(&self, b: &[u8]) -> Option<u32> {
        self.map.get(b).copied()
    }

    /// The bytes behind a handle.
    pub fn bytes(&self, id: u32) -> &[u8] {
        &self.keys[id as usize]
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
}
