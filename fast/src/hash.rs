//! A fast, non-cryptographic hash for byte-string IDs and small integer keys.
//!
//! `std`'s default SipHash is DoS-resistant and slow. There is no adversary here,
//! so we use an FxHash-style word-at-a-time combine with a murmur3 `fmix64`
//! finaliser. The finaliser matters: shards are chosen from the TOP bits of the
//! hash, and a raw Fx combine has weak high bits on short keys.
//!
//! **Correctness never depends on this hash.** Every place that groups by a hash
//! (event dedup, person sharding, hash maps) also compares the real bytes, so a
//! collision costs time, never a wrong answer.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const K: u64 = 0x9E37_79B9_7F4A_7C15;

#[inline(always)]
pub fn fmix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
    k ^= k >> 33;
    k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    k ^= k >> 33;
    k
}

/// Hash a byte string. Length is mixed in so `a` and `a\0` differ.
#[inline]
pub fn hash_bytes(b: &[u8]) -> u64 {
    let mut h: u64 = (b.len() as u64).wrapping_mul(K);
    let mut chunks = b.chunks_exact(8);
    for c in &mut chunks {
        let w = u64::from_le_bytes(c.try_into().unwrap());
        h = (h.rotate_left(5) ^ w).wrapping_mul(K);
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut buf = [0u8; 8];
        buf[..rem.len()].copy_from_slice(rem);
        h = (h.rotate_left(5) ^ u64::from_le_bytes(buf)).wrapping_mul(K);
    }
    fmix64(h)
}

/// `Hasher` adapter so the same hash backs `HashMap`s.
#[derive(Default, Clone, Copy)]
pub struct FastHasher(u64);

impl FastHasher {
    #[inline(always)]
    fn add(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(K);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.add(hash_bytes(bytes));
    }
    #[inline]
    fn write_u8(&mut self, n: u8) {
        self.add(n as u64);
    }
    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.add(n as u64);
    }
    #[inline]
    fn write_i32(&mut self, n: i32) {
        self.add(n as u32 as u64);
    }
    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.add(n);
    }
    #[inline]
    fn write_i64(&mut self, n: i64) {
        self.add(n as u64);
    }
    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.add(n as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        fmix64(self.0)
    }
}

pub type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_inputs_hash_differently() {
        assert_ne!(hash_bytes(b"a"), hash_bytes(b"a\0"));
        assert_ne!(hash_bytes(b""), hash_bytes(b"\0"));
        assert_ne!(hash_bytes(b"u1"), hash_bytes(b"U1"));
        assert_ne!(hash_bytes(b"12345678"), hash_bytes(b"123456789"));
        assert_eq!(hash_bytes(b"evt_000123"), hash_bytes(b"evt_000123"));
    }

    #[test]
    fn fast_map_works_with_byte_keys() {
        let mut m: FastMap<&[u8], u32> = FastMap::default();
        m.insert(b"x", 1);
        m.insert(b"y", 2);
        assert_eq!(m.get(&b"x"[..]), Some(&1));
        assert_eq!(m.get(&b"z"[..]), None);
    }
}
