//! Lock-free exact table for generator-shaped 33-byte hex IDs.
//!
//! The v5 generator makes about 1 in 8 distinct user/event/click IDs a 33-byte
//! ID: one lowercase letter followed by 32 lowercase hex digits. `key::pack`
//! cannot hold them (more than 10 bytes), and the general mutex-sharded
//! interner costs ~3 dependent cache misses plus a mutex per lookup. This module
//! stores exactly that shape inline as `(prefix: u8, value: u128)`.
//!
//! **Shape accepted** (and nothing else): `^[a-z][0-9a-f]{32}$`, exactly 33
//! bytes. Uppercase hex, any other byte, or any other length is rejected and
//! stays on the general byte-string path, so the representation is a
//! bijection between accepted byte strings and `(prefix, value)` pairs.
//!
//! **Table design.** A fixed array of `2^log2` slots (24 B each), split into
//! regions of 64 consecutive slots. An ID hashes to one region and a start
//! offset inside it, and linear-probes *only within that region* (wrapping).
//! Slots are never freed or overwritten once published. If an ID is absent
//! and its region has no empty slot, `intern` returns `None` and the caller
//! (key.rs) uses the exact sharded interner instead. Because a full region
//! stays full forever, "absent from a full region" is a permanent fact, so
//! every thread makes the same choice for the same ID: there is no global
//! counter, no global "table full" flag, and no race between the two homes.
//! The table therefore never fills in a harmful way and never panics; an
//! overloaded table only degrades speed. The slot number itself is the ID's
//! index (no separate index allocator, no reverse array): `get(index)` reads
//! the published slot.
//!
//! **Publish protocol** (per slot, `meta` word):
//! `EMPTY (0) --CAS--> BUSY (1) --store Release--> PUBLISHED | prefix`.
//! The claimer writes `hi`/`lo` (Relaxed) between the CAS and the Release
//! store. A reader loads `meta` with Acquire; only when it sees a PUBLISHED
//! value does it read `hi`/`lo` (Relaxed), and the Release/Acquire pair makes
//! the claimer's `hi`/`lo` writes visible. A reader that sees BUSY spins
//! (the claimer is between two adjacent stores, nanoseconds) and yields after
//! a few dozen spins in case the claimer was preempted. `hi`/`lo` are written
//! exactly once, before publish, so a reader never observes a torn key.
//!
//! **Why linear probing inside a region is exact under concurrency.** Slots go
//! EMPTY -> non-empty once and never back. A thread that finds X's first EMPTY
//! slot `k` on X's probe sequence and wins the CAS on it knows every earlier
//! slot on the sequence holds a published non-X key (it waited out any BUSY
//! ones and compared). Any other thread looking for X walks the same sequence,
//! finds those same final keys, then reaches `k`, which is now BUSY or
//! PUBLISHED with X; it cannot get past `k` without seeing X. Two threads
//! racing on `k` are serialized by the CAS; the loser waits for publish and
//! compares.

use std::sync::atomic::{AtomicU64, Ordering};

/// Bytes in an accepted hex ID: one prefix letter + 32 hex digits.
pub const HEX_ID_LEN: usize = 33;

const L: u64 = 0x0101_0101_0101_0101;
const H: u64 = 0x8080_8080_8080_8080;

/// Validate 8 ASCII bytes (big-endian loaded: first byte most significant) as
/// lowercase hex `[0-9a-f]` and convert them to a 32-bit value.
///
/// SWAR range check: with every byte's high bit forced on (`h = x | H`, so a
/// byte is `0x80 | c` for 7-bit `c`), subtracting `lo` (< 0x80) from every byte
/// can never borrow across bytes, and the byte's high bit stays set iff
/// `c >= lo`. So `(h - lo) & !(h - (hi + 1)) & H` marks bytes with `lo <= c <= hi`.
/// A byte with its real high bit set (`x & H != 0`) is rejected separately.
#[inline(always)]
fn hex8(x: u64) -> (u64, bool) {
    let h = x | H;
    let digit = h.wrapping_sub(0x30 * L) & !h.wrapping_sub(0x3a * L);
    let alpha = h.wrapping_sub(0x61 * L) & !h.wrapping_sub(0x67 * L);
    let ok = (x & H) == 0 && ((digit | alpha) & H) == H;
    // '0'..'9' -> low nibble 0..9; 'a'..'f' -> low nibble 1..6, +9 = 10..15.
    // Per-byte values stay <= 24, so no carry crosses bytes even when !ok.
    let nib = (x & 0x0f0f_0f0f_0f0f_0f0f) + ((alpha & H) >> 7) * 9;
    // Each 16-bit lane: high byte = earlier digit d0, low byte = d1.
    // `nib | nib >> 4` puts d0 << 4 | d1 in the lane's low byte.
    let t = (nib | (nib >> 4)) & 0x00ff_00ff_00ff_00ff;
    let t = (t | (t >> 8)) & 0x0000_ffff_0000_ffff;
    let t = (t | (t >> 16)) & 0xffff_ffff;
    (t, ok)
}

/// Parse an ID matching exactly `^[a-z][0-9a-f]{32}$` into `(prefix, value)`,
/// with the first hex digit most significant. Anything else returns `None`.
#[inline]
pub fn try_parse(bytes: &[u8]) -> Option<(u8, u128)> {
    if bytes.len() != HEX_ID_LEN {
        return None;
    }
    let prefix = bytes[0];
    if !prefix.is_ascii_lowercase() {
        return None;
    }
    let chunk = |i: usize| u64::from_be_bytes(bytes[i..i + 8].try_into().unwrap());
    let (a, ok_a) = hex8(chunk(1));
    let (b, ok_b) = hex8(chunk(9));
    let (c, ok_c) = hex8(chunk(17));
    let (d, ok_d) = hex8(chunk(25));
    if !(ok_a & ok_b & ok_c & ok_d) {
        return None;
    }
    let hi = (a << 32) | b;
    let lo = (c << 32) | d;
    Some((prefix, ((hi as u128) << 64) | lo as u128))
}

/// The exact 33 bytes for `(prefix, value)`; inverse of [`try_parse`].
#[inline]
pub fn render(prefix: u8, value: u128) -> [u8; HEX_ID_LEN] {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = [0u8; HEX_ID_LEN];
    out[0] = prefix;
    let mut i = 0;
    while i < 32 {
        out[1 + i] = DIGITS[((value >> (124 - 4 * i)) & 15) as usize];
        i += 1;
    }
    out
}

/// Slots per region (probe window). 64 x 24 B = 1.5 KiB.
const REGION_BITS: u32 = 6;
const REGION: usize = 1 << REGION_BITS;

const EMPTY: u64 = 0;
const BUSY: u64 = 1;
/// Published meta = `PUBLISHED | prefix` (prefix is a lowercase letter, so
/// the value never equals EMPTY or BUSY).
const PUBLISHED: u64 = 1 << 63;

#[repr(C)]
struct Slot {
    meta: AtomicU64,
    hi: AtomicU64,
    lo: AtomicU64,
}

/// Zero-initialised slot storage. On unix it is a private anonymous mapping:
/// the kernel supplies zero pages lazily, so untouched parts of the table
/// cost no RSS and no zeroing time.
struct Slots {
    ptr: *mut Slot,
    len: usize,
}

// SAFETY: the storage is only accessed through `&Slot`, whose fields are
// atomics; the raw pointer is owned and freed exactly once in Drop.
unsafe impl Send for Slots {}
unsafe impl Sync for Slots {}

impl Slots {
    fn new(len: usize) -> Self {
        let bytes = len * std::mem::size_of::<Slot>();
        #[cfg(unix)]
        {
            // SAFETY: plain anonymous mapping; checked for MAP_FAILED.
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            if p == libc::MAP_FAILED {
                std::alloc::handle_alloc_error(std::alloc::Layout::array::<Slot>(len).unwrap());
            }
            // SAFETY: the mapping is `bytes` long, zeroed, and exclusively ours.
            let slice = unsafe { std::slice::from_raw_parts_mut(p.cast::<Slot>(), len) };
            crate::sysmem::advise_hugepage(slice);
            Self { ptr: p.cast(), len }
        }
        #[cfg(not(unix))]
        {
            let layout = std::alloc::Layout::array::<Slot>(len).unwrap();
            // SAFETY: nonzero layout; all-zero bytes are a valid Slot (EMPTY).
            let p = unsafe { std::alloc::alloc_zeroed(layout) };
            if p.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            let _ = bytes;
            Self { ptr: p.cast(), len }
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> &Slot {
        assert!(i < self.len);
        // SAFETY: in bounds; all-zero bytes are a valid Slot; shared access
        // goes through atomics only.
        unsafe { &*self.ptr.add(i) }
    }
}

impl Drop for Slots {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: unmapping exactly the mapping created in `new`.
        unsafe {
            libc::munmap(self.ptr.cast(), self.len * std::mem::size_of::<Slot>());
        }
        #[cfg(not(unix))]
        // SAFETY: same layout as the allocation in `new`.
        unsafe {
            std::alloc::dealloc(self.ptr.cast(), std::alloc::Layout::array::<Slot>(self.len).unwrap());
        }
    }
}

/// Concurrent exact set of hex IDs; see the module docs.
pub struct HexTable {
    slots: Slots,
    region_mask: usize,
}

/// Cheap, well-mixed hash of the inline key (folded 64x64->128 multiply, as in
/// rapidhash's `mum`, but over two words instead of a byte loop). Only
/// performance depends on it; equality is always on all 136 key bits.
#[inline(always)]
fn hash(prefix: u8, hi: u64, lo: u64) -> u64 {
    let a = hi ^ 0x243f_6a88_85a3_08d3 ^ (prefix as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let b = lo ^ 0x1319_8a2e_0370_7344;
    let m = (a as u128).wrapping_mul(b as u128);
    let x = (m >> 64) as u64 ^ m as u64;
    // One more multiply so low bits (offset) and middle bits (region) are
    // both well mixed even for structured inputs.
    (x ^ (x >> 32)).wrapping_mul(0xd6e8_feb8_6659_fd93)
}

/// Wait for a BUSY slot to be published; returns the published meta.
#[inline(never)]
fn wait_published(slot: &Slot) -> u64 {
    let mut spins = 0u32;
    loop {
        let m = slot.meta.load(Ordering::Acquire);
        if m != BUSY {
            return m;
        }
        spins += 1;
        if spins < 64 {
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
}

impl HexTable {
    /// A table of `2^log2_slots` slots (`log2_slots >= 6`, at most 28 so that
    /// an index fits comfortably in an `IdKey`).
    pub fn new(log2_slots: u32) -> Self {
        assert!((REGION_BITS..=28).contains(&log2_slots));
        let len = 1usize << log2_slots;
        Self {
            slots: Slots::new(len),
            region_mask: (len >> REGION_BITS) - 1,
        }
    }

    /// Return the index of `(prefix, value)`, inserting it if absent, or
    /// `None` if it is absent and its region is full. Both answers are final:
    /// the same ID always gets the same answer from every thread.
    #[inline]
    pub fn intern(&self, prefix: u8, value: u128) -> Option<u32> {
        let hi = (value >> 64) as u64;
        let lo = value as u64;
        let want = PUBLISHED | prefix as u64;
        let h = hash(prefix, hi, lo);
        let base = (((h >> REGION_BITS) as usize) & self.region_mask) << REGION_BITS;
        let start = h as usize;
        for i in 0..REGION {
            let index = base + ((start + i) & (REGION - 1));
            let slot = self.slots.get(index);
            let mut m = slot.meta.load(Ordering::Acquire);
            if m == EMPTY {
                // Claim. Success ordering can be Relaxed: nothing is read
                // through this edge, and readers only look at hi/lo after
                // Acquire-observing the Release publish below.
                match slot
                    .meta
                    .compare_exchange(EMPTY, BUSY, Ordering::Relaxed, Ordering::Acquire)
                {
                    Ok(_) => {
                        slot.hi.store(hi, Ordering::Relaxed);
                        slot.lo.store(lo, Ordering::Relaxed);
                        slot.meta.store(want, Ordering::Release);
                        return Some(index as u32);
                    }
                    Err(current) => m = current,
                }
            }
            if m == BUSY {
                m = wait_published(slot);
            }
            // `m` is a PUBLISHED value obtained with Acquire, so hi/lo are visible.
            if m == want
                && slot.hi.load(Ordering::Relaxed) == hi
                && slot.lo.load(Ordering::Relaxed) == lo
            {
                return Some(index as u32);
            }
        }
        None
    }

    /// `(prefix, value)` stored at an index returned by [`intern`](Self::intern).
    #[inline]
    pub fn get(&self, index: u32) -> (u8, u128) {
        let slot = self.slots.get(index as usize);
        let m = slot.meta.load(Ordering::Acquire);
        assert!(m & PUBLISHED != 0, "hex ID index {index} is not published");
        let hi = slot.hi.load(Ordering::Relaxed);
        let lo = slot.lo.load(Ordering::Relaxed);
        (m as u8, ((hi as u128) << 64) | lo as u128)
    }

    /// Number of published IDs. O(capacity) scan; tests/diagnostics only.
    pub fn count(&self) -> usize {
        (0..self.slots.len)
            .filter(|&i| self.slots.get(i).meta.load(Ordering::Acquire) & PUBLISHED != 0)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar oracle for `^[a-z][0-9a-f]{32}$`.
    fn oracle(b: &[u8]) -> Option<(u8, u128)> {
        if b.len() != 33 || !(b'a'..=b'z').contains(&b[0]) {
            return None;
        }
        let mut v = 0u128;
        for &c in &b[1..] {
            let d = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                _ => return None,
            };
            v = (v << 4) | d as u128;
        }
        Some((b[0], v))
    }

    #[test]
    fn try_parse_every_byte_at_every_position() {
        let bases: [&[u8; 33]; 3] = [
            b"e0123456789abcdef0123456789abcdef",
            b"a00000000000000000000000000000000",
            b"zffffffffffffffffffffffffffffffff",
        ];
        for base in bases {
            assert_eq!(try_parse(base), oracle(base));
            assert!(try_parse(base).is_some());
            for pos in 0..33 {
                for byte in 0..=255u8 {
                    let mut id = *base;
                    id[pos] = byte;
                    assert_eq!(try_parse(&id), oracle(&id), "pos {pos} byte {byte:#x}");
                }
            }
            // Lengths 0..=40 by truncation / extension.
            for len in 0..=40 {
                let mut id = base.to_vec();
                id.resize(len, b'0');
                assert_eq!(try_parse(&id), oracle(&id), "len {len}");
            }
        }
        assert_eq!(try_parse(b"E0123456789abcdef0123456789abcdef"), None);
        assert_eq!(try_parse(b"e0123456789ABCDEF0123456789abcdef"), None);
        assert_eq!(try_parse(b"e0123456789abcdef0123456789abcdeg"), None);
        assert_eq!(try_parse(b"e0123456789abcdef0123456789abcde"), None);
        assert_eq!(try_parse(b"e0123456789abcdef0123456789abcdef0"), None);
    }

    #[test]
    fn try_parse_random_fuzz_and_render_round_trip() {
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        const ALPH: &[u8] = b"0123456789abcdefgABCDEFG/:@`\x00\x7f\x80\xb0\xe1\xff";
        for _ in 0..500_000 {
            let mut id = [0u8; 33];
            id[0] = if next() % 8 == 0 { next() as u8 } else { b'a' + (next() % 26) as u8 };
            let mode = next() % 4;
            for b in &mut id[1..] {
                *b = if mode == 0 {
                    ALPH[(next() % ALPH.len() as u64) as usize]
                } else {
                    b"0123456789abcdef"[(next() & 15) as usize]
                };
            }
            if mode == 1 {
                id[1 + (next() % 32) as usize] = next() as u8;
            }
            let got = try_parse(&id);
            assert_eq!(got, oracle(&id), "{id:?}");
            if let Some((p, v)) = got {
                assert_eq!(render(p, v), id);
            }
        }
        for v in [0u128, 1, u128::MAX, 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210] {
            let r = render(b'q', v);
            assert_eq!(try_parse(&r), Some((b'q', v)));
        }
    }

    #[test]
    fn table_dedups_and_overflows_exactly() {
        let t = HexTable::new(6); // one region of 64 slots
        let mut idx = Vec::new();
        for v in 0..200u128 {
            idx.push(t.intern(b'e', v * 0x1_0000_0001));
        }
        assert_eq!(idx.iter().filter(|i| i.is_some()).count(), 64);
        for (v, &i) in idx.iter().enumerate() {
            let v = v as u128 * 0x1_0000_0001;
            assert_eq!(t.intern(b'e', v), i);
            if let Some(i) = i {
                assert_eq!(t.get(i), (b'e', v));
            }
        }
        // Prefix participates in equality.
        assert_eq!(t.intern(b'u', 0), None);
        assert_eq!(t.count(), 64);
    }
}
