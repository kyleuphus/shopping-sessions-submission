//! Parallel v2 links stage over exact integer ID keys.
//!
//! Every phase is parallel:
//!
//! 1. **Parse + insert.** Newline-aligned chunks are parsed with the same
//!    `csvline::split_fields` as v1. Each ID key is inserted into a concurrent
//!    open-addressing table (CAS on the key word); the slot index becomes the
//!    node's union-find handle. Each chunk keeps its edges as slot pairs.
//! 2. **Union.** A lock-free union-find over an `AtomicU32` parent array. A
//!    union always links the root with the *larger* key (exact `key_cmp`
//!    order) under the root with the smaller key, so every parent pointer goes
//!    from a larger key to a smaller one. That invariant makes cycles
//!    impossible even when threads read stale parents, and the root of every
//!    component is its byte-wise smallest ID: the person name.
//! 3. **Name map.** Every occupied slot is resolved to its root key and
//!    inserted into a compact final table sized for the real distinct count.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use rayon::prelude::*;

use crate::csvline::split_fields;
use crate::hash::fmix64;
use crate::key::{key_cmp, key_of, IdInterner, IdKey};

const CHUNK_BYTES: usize = 1 << 18;

#[derive(Clone, Copy)]
struct Chunk<'a> {
    bytes: &'a [u8],
    has_header: bool,
}

/// Split every file at a newline at or after the target size. Only the first
/// chunk of each file owns and skips that file's header.
fn chunks<'a>(files: &[&'a [u8]]) -> Vec<Chunk<'a>> {
    let mut out = Vec::new();
    for &bytes in files {
        let mut start = 0;
        while start < bytes.len() {
            let mut end = (start + CHUNK_BYTES).min(bytes.len());
            if end < bytes.len() {
                end = memchr::memchr(b'\n', &bytes[end..])
                    .map_or(bytes.len(), |offset| end + offset + 1);
            }
            out.push(Chunk {
                bytes: &bytes[start..end],
                has_header: start == 0,
            });
            start = end;
        }
    }
    out
}

#[inline]
fn for_each_line<'a>(chunk: &Chunk<'a>, mut f: impl FnMut(&'a [u8])) {
    let mut bytes = chunk.bytes;
    if chunk.has_header {
        match memchr::memchr(b'\n', bytes) {
            Some(newline) => bytes = &bytes[newline + 1..],
            None => return,
        }
    }

    let mut start = 0;
    for newline in memchr::memchr_iter(b'\n', bytes) {
        if newline > start {
            f(&bytes[start..newline]);
        }
        start = newline + 1;
    }
    if start < bytes.len() {
        f(&bytes[start..]);
    }
}

#[inline]
fn home_slot(key: IdKey, mask: usize) -> usize {
    fmix64(key ^ 0x6c69_6e6b_7332_5f6b) as usize & mask
}

fn table_capacity(len: usize) -> usize {
    (len + len.div_ceil(3)).next_power_of_two().max(16)
}

/// A zero-filled vector, allocated with `alloc_zeroed` so large tables come
/// straight from fresh (lazily faulted) pages instead of being written.
///
/// Safety: all-zero bytes must be a valid `T` (true for the atomics and
/// `Slot` used here).
unsafe fn zeroed_vec<T>(len: usize) -> Vec<T> {
    if len == 0 || std::mem::size_of::<T>() == 0 {
        return Vec::new();
    }
    let layout = std::alloc::Layout::array::<T>(len).expect("links table size overflow");
    let ptr = std::alloc::alloc_zeroed(layout);
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    // Same allocator and layout that `Vec<T>` uses to free it.
    Vec::from_raw_parts(ptr.cast(), len, len)
}

/// Concurrent key -> slot table. A slot word holds `key + 1`; 0 means empty.
/// `key + 1` never wraps: a key's low nibble is 0..=10, so it is never
/// `u64::MAX`.
struct KeyTable {
    keys: Vec<AtomicU64>,
    mask: usize,
}

impl KeyTable {
    /// Return the slot holding `key`, inserting it if absent, and whether this
    /// call inserted it.
    #[inline]
    fn insert(&self, key: IdKey) -> (u32, bool) {
        let stored = key + 1;
        let mut slot = home_slot(key, self.mask);
        loop {
            let current = self.keys[slot].load(Relaxed);
            if current == stored {
                return (slot as u32, false);
            }
            if current == 0 {
                match self.keys[slot].compare_exchange(0, stored, Relaxed, Relaxed) {
                    Ok(_) => return (slot as u32, true),
                    // Another thread claimed it; it may have claimed it for us.
                    Err(actual) if actual == stored => return (slot as u32, false),
                    Err(_) => {}
                }
            }
            slot = (slot + 1) & self.mask;
        }
    }

    #[inline]
    fn key(&self, slot: u32) -> IdKey {
        self.keys[slot as usize].load(Relaxed) - 1
    }
}

/// Lock-free union-find over table slots. `parent[x]` holds `p + 1`, or 0 when
/// `x` is a root. Invariant: `key(parent) < key(child)` in exact byte order.
struct Forest<'t> {
    parent: Vec<AtomicU32>,
    table: &'t KeyTable,
}

impl Forest<'_> {
    /// Find the root of `x`, path-halving along the way. Halving only ever
    /// replaces a parent with one of its ancestors, preserving the invariant.
    #[inline]
    fn find(&self, mut x: u32) -> u32 {
        loop {
            let p = self.parent[x as usize].load(Relaxed);
            if p == 0 {
                return x;
            }
            let p = p - 1;
            let gp = self.parent[p as usize].load(Relaxed);
            if gp == 0 {
                return p;
            }
            // Best effort: a failed CAS means someone already moved it upward.
            let _ = self.parent[x as usize].compare_exchange_weak(p + 1, gp, Relaxed, Relaxed);
            x = gp - 1;
        }
    }

    fn union(&self, a: u32, b: u32, interner: &IdInterner) {
        let (mut a, mut b) = (a, b);
        loop {
            a = self.find(a);
            b = self.find(b);
            if a == b {
                return;
            }
            let (lo, hi) = if key_cmp(self.table.key(a), self.table.key(b), interner).is_lt() {
                (a, b)
            } else {
                (b, a)
            };
            // Link only if `hi` is still a root. A stale view of `lo` is
            // harmless: `lo` can never sit under `hi` (keys only decrease
            // towards roots), so linking cannot form a cycle.
            if self.parent[hi as usize]
                .compare_exchange(0, lo + 1, Relaxed, Relaxed)
                .is_ok()
            {
                return;
            }
        }
    }
}

#[repr(C, align(16))]
struct Slot {
    /// `key + 1`, or 0 for an empty slot. Atomic only so the table can be
    /// filled in parallel; after `build` returns it is read-only and relaxed
    /// loads compile to plain loads.
    key: AtomicU64,
    value: AtomicU64,
}

/// Linked ID -> byte-wise smallest ID in its transitive component.
///
/// Only IDs appearing in kept link lines are present. Lookups hash the already
/// computed integer key and then compare one `u64`; packed IDs never incur byte
/// hashing on this path.
pub struct PersonMap {
    slots: Vec<Slot>,
    mask: usize,
    len: usize,
}

impl PersonMap {
    /// Return the person's name key, or `None` when the ID appeared in no kept
    /// links line.
    #[inline]
    pub fn get(&self, key: IdKey) -> Option<IdKey> {
        let stored = key + 1;
        let mut slot = home_slot(key, self.mask);
        loop {
            let entry = &self.slots[slot];
            let current = entry.key.load(Relaxed);
            if current == stored {
                return Some(entry.value.load(Relaxed));
            }
            if current == 0 {
                return None;
            }
            slot = (slot + 1) & self.mask;
        }
    }

    /// Hint the CPU to start loading the cache line that `get(key)` will probe
    /// first. Purely a performance hint: it reads nothing and changes nothing,
    /// so it cannot affect any result. Callers issue it some records ahead of
    /// the matching `get` so the memory latency overlaps other work.
    #[inline(always)]
    pub fn prefetch(&self, key: IdKey) {
        let slot = home_slot(key, self.mask);
        // `wrapping_add` + a prefetch never dereference: even an out-of-range
        // address would just be an ignored hint. `slot <= mask < len` anyway.
        let ptr = self.slots.as_ptr().wrapping_add(slot) as *const u8;
        prefetch_read(ptr);
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Software prefetch of one cache line into L1 (no-op on other targets).
#[inline(always)]
fn prefetch_read(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch is a hint; it never faults or reads into the program.
    unsafe {
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: as above; `prfm` is a hint instruction.
    unsafe {
        core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) ptr, options(nostack, readonly, preserves_flags));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = ptr;
}

struct Timer {
    enabled: bool,
    last: Instant,
}

impl Timer {
    fn mark(&mut self, what: &str) {
        if self.enabled {
            let now = Instant::now();
            eprintln!(
                "  links2 {what:<14} {:8.2} ms",
                (now - self.last).as_secs_f64() * 1e3
            );
            self.last = now;
        }
    }
}

/// Build the v2 person map from mmapped link-file contents.
///
/// `interner` must also be used for keys passed to `PersonMap::get` and for
/// decoding returned fallback keys.
pub fn build(link_files: &[&[u8]], interner: &IdInterner) -> PersonMap {
    let mut timer = Timer {
        enabled: std::env::var_os("FAST_TIMING").is_some(),
        last: Instant::now(),
    };
    let file_chunks = chunks(link_files);

    // Every kept line holds at most two new IDs, and every line ends with a
    // newline or the end of its file, so 2 * (newlines + files) bounds the
    // distinct count.
    let lines: usize = file_chunks
        .par_iter()
        .map(|chunk| memchr::memchr_iter(b'\n', chunk.bytes).count() + 1)
        .sum();
    let table = KeyTable {
        keys: unsafe { zeroed_vec(table_capacity(2 * lines)) },
        mask: table_capacity(2 * lines) - 1,
    };
    assert!(
        u32::try_from(table.keys.len()).is_ok(),
        "links table exceeds u32 slots"
    );
    timer.mark("count");

    // Phase 1: parse and insert. Self-links are dropped here (no-ops).
    let distinct = AtomicUsize::new(0);
    let edges: Vec<Vec<(u32, u32)>> = file_chunks
        .par_iter()
        .map(|chunk| {
            let mut edges = Vec::with_capacity(chunk.bytes.len() / 20);
            let mut fields = Vec::with_capacity(2);
            let mut inserted = 0usize;
            for_each_line(chunk, |line| {
                if !split_fields(line, 2, &mut fields)
                    || fields[0].is_empty()
                    || fields[1].is_empty()
                {
                    return;
                }
                let (a, new_a) = table.insert(key_of(fields[0], interner));
                let (b, new_b) = table.insert(key_of(fields[1], interner));
                inserted += new_a as usize + new_b as usize;
                if a != b {
                    edges.push((a, b));
                }
            });
            distinct.fetch_add(inserted, Relaxed);
            edges
        })
        .collect();
    let len = distinct.into_inner();
    timer.mark("parse+insert");

    // Phase 2: lock-free union.
    let forest = Forest {
        parent: unsafe { zeroed_vec(table.keys.len()) },
        table: &table,
    };
    edges.par_iter().for_each(|chunk_edges| {
        for &(a, b) in chunk_edges {
            forest.union(a, b, interner);
        }
    });
    drop(edges);
    timer.mark("union");

    // Phase 3: resolve every occupied slot to its root's key and insert into a
    // compact final table. Final keys are unique, so each insert claims the
    // first empty slot it meets; the value is published before `build`
    // returns (the parallel loop is a barrier).
    let capacity = table_capacity(len);
    let mask = capacity - 1;
    // Safety: all-zero bytes are a valid (empty) `Slot`.
    let slots: Vec<Slot> = unsafe { zeroed_vec(capacity) };
    table
        .keys
        .par_iter()
        .enumerate()
        .with_min_len(4096)
        .for_each(|(slot, word)| {
            let stored = word.load(Relaxed);
            if stored == 0 {
                return;
            }
            let name = table.key(forest.find(slot as u32));
            let mut at = home_slot(stored - 1, mask);
            loop {
                let target = &slots[at];
                if target
                    .key
                    .compare_exchange(0, stored, Relaxed, Relaxed)
                    .is_ok()
                {
                    target.value.store(name, Relaxed);
                    return;
                }
                at = (at + 1) & mask;
            }
        });
    timer.mark("name map");

    PersonMap { slots, mask, len }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key::key_bytes;

    fn named(map: &PersonMap, interner: &IdInterner, id: &[u8]) -> Option<Vec<u8>> {
        let name = map.get(key_of(id, interner))?;
        let mut buf = Vec::new();
        Some(key_bytes(name, interner, &mut buf).to_vec())
    }

    #[test]
    fn csv_rules_transitivity_and_minimum_name_are_exact() {
        let first = b"user_id_a,user_id_b\na,b\nb,c\nc,c\n\" long id \",x\nx,b\n,z\nq,\nbad\"quote,z\nonly_one\ntail,carriage\r\n";
        let second = b"user_id_a,user_id_b\nuser_id_a,user_id_b\nc,a\n";
        let interner = IdInterner::new();
        let map = build(&[first, second], &interner);

        for id in [b"a".as_slice(), b"b", b"c", b"x", b" long id "] {
            assert_eq!(
                named(&map, &interner, id).as_deref(),
                Some(b" long id ".as_slice())
            );
        }
        assert_eq!(
            named(&map, &interner, b"tail").as_deref(),
            Some(b"carriage\r".as_slice())
        );
        assert_eq!(
            named(&map, &interner, b"carriage\r").as_deref(),
            Some(b"carriage\r".as_slice())
        );
        assert_eq!(
            named(&map, &interner, b"user_id_a").as_deref(),
            Some(b"user_id_a".as_slice())
        );
        assert_eq!(
            named(&map, &interner, b"user_id_b").as_deref(),
            Some(b"user_id_a".as_slice())
        );
        for dropped in [b"z".as_slice(), b"q", b"bad\"quote", b"only_one"] {
            assert_eq!(named(&map, &interner, dropped), None);
        }
    }

    #[test]
    fn long_chain_and_large_shared_component_are_exact() {
        let mut file = b"user_id_a,user_id_b\n".to_vec();
        for id in 1..5_000 {
            file.extend_from_slice(format!("id{id:05},id{:05}\n", id - 1).as_bytes());
        }
        // Repeats, reverse repeats, and self-links must be no-ops.
        file.extend_from_slice(b"id00000,id00001\nid00001,id00000\nid04999,id04999\n");

        let interner = IdInterner::new();
        let map = build(&[&file], &interner);
        assert_eq!(map.len(), 5_000);
        for id in [0, 1, 127, 2_048, 4_999] {
            let bytes = format!("id{id:05}");
            assert_eq!(
                named(&map, &interner, bytes.as_bytes()).as_deref(),
                Some(b"id00000".as_slice())
            );
        }
    }

    #[test]
    fn balanced_union_tree_maps_every_depth_to_the_minimum() {
        // Pair equal-sized components repeatedly to force a multi-level tree;
        // this catches finalization that only path-halves instead of recording
        // the returned root for every dense ID.
        let mut file = b"user_id_a,user_id_b\n".to_vec();
        for width in [1, 2, 4, 8, 16, 32] {
            for left in (0..64).step_by(width * 2) {
                file.extend_from_slice(
                    format!("node{left:02},node{:02}\n", left + width).as_bytes(),
                );
            }
        }

        let interner = IdInterner::new();
        let map = build(&[&file], &interner);
        assert_eq!(map.len(), 64);
        for id in 0..64 {
            let bytes = format!("node{id:02}");
            assert_eq!(
                named(&map, &interner, bytes.as_bytes()).as_deref(),
                Some(b"node00".as_slice())
            );
        }
    }

    #[test]
    fn empty_and_header_only_files_make_an_empty_map() {
        let interner = IdInterner::new();
        let map = build(&[b"", b"user_id_a,user_id_b\n"], &interner);
        assert!(map.is_empty());
        assert_eq!(map.get(key_of(b"absent", &interner)), None);
    }

    /// Deterministic Fisher-Yates shuffle (no rand dependency).
    fn shuffle<T>(items: &mut [T], mut state: u64) {
        for i in (1..items.len()).rev() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            items.swap(i, (state >> 33) as usize % (i + 1));
        }
    }

    fn file_of(lines: &[String]) -> Vec<u8> {
        let mut file = b"user_id_a,user_id_b\n".to_vec();
        for line in lines {
            file.extend_from_slice(line.as_bytes());
            file.push(b'\n');
        }
        file
    }

    #[test]
    fn chain_of_400_in_random_order() {
        for seed in 1..=5u64 {
            // Pseudo-random distinct IDs so the minimum is not the chain head.
            let ids: Vec<String> = (0..400u64)
                .map(|i| format!("c{:08x}", fmix64(i ^ (seed << 32)) as u32))
                .collect();
            let mut lines: Vec<String> = (1..ids.len())
                .map(|i| {
                    if i % 2 == 0 {
                        format!("{},{}", ids[i - 1], ids[i])
                    } else {
                        format!("{},{}", ids[i], ids[i - 1])
                    }
                })
                .collect();
            shuffle(&mut lines, seed);
            // One line per file: every line is its own parallel chunk.
            let files: Vec<Vec<u8>> = lines.chunks(1).map(file_of).collect();
            let refs: Vec<&[u8]> = files.iter().map(Vec::as_slice).collect();
            let interner = IdInterner::new();
            let map = build(&refs, &interner);
            let mut unique = ids.clone();
            unique.sort();
            unique.dedup();
            assert_eq!(map.len(), unique.len());
            let min = unique[0].as_bytes();
            for id in &ids {
                assert_eq!(named(&map, &interner, id.as_bytes()).as_deref(), Some(min));
            }
        }
    }

    #[test]
    fn star_of_5000() {
        let mut lines: Vec<String> = (0..5_000).map(|i| format!("s{i:05},hub")).collect();
        shuffle(&mut lines, 7);
        // Split across many files so several chunks union concurrently.
        let files: Vec<Vec<u8>> = lines.chunks(50).map(file_of).collect();
        let refs: Vec<&[u8]> = files.iter().map(Vec::as_slice).collect();
        let interner = IdInterner::new();
        let map = build(&refs, &interner);
        assert_eq!(map.len(), 5_001);
        // 'h' < 's', so the hub is the minimum.
        for id in [b"hub".as_slice(), b"s00000", b"s02500", b"s04999"] {
            assert_eq!(
                named(&map, &interner, id).as_deref(),
                Some(b"hub".as_slice())
            );
        }
    }

    #[test]
    fn upper_case_sorts_first_and_becomes_the_name() {
        let file = b"user_id_a,user_id_b\nabc,xyz\nxyz,Zed\nZed,aaa\nq,Q\n";
        let interner = IdInterner::new();
        let map = build(&[file], &interner);
        for id in [b"abc".as_slice(), b"xyz", b"Zed", b"aaa"] {
            assert_eq!(
                named(&map, &interner, id).as_deref(),
                Some(b"Zed".as_slice())
            );
        }
        assert_eq!(
            named(&map, &interner, b"q").as_deref(),
            Some(b"Q".as_slice())
        );
        assert_eq!(
            named(&map, &interner, b"Q").as_deref(),
            Some(b"Q".as_slice())
        );
    }

    #[test]
    fn unpackable_ids_mixed_with_packed_use_exact_byte_minimum() {
        // Unpackable: space, >10 bytes, '\r', non-ASCII. Byte order:
        // " sp" < "0" < "A" < "a" < "aaaaaaaaaaaaaa" < "b\r" < "\xc3\xa9".
        let file = "user_id_a,user_id_b\n\
                    b\r,\u{e9}\n\
                    \u{e9},aaaaaaaaaaaaaa\n\
                    aaaaaaaaaaaaaa,a\n\
                    a,A\n\
                    A,0\n\
                    0, sp\n\
                    \u{e9}2,b\r\n\
                    lone,\u{e9}3\n\
                    \u{e9}3,\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\n"
            .as_bytes();
        let interner = IdInterner::new();
        let map = build(&[file], &interner);
        for id in [
            "b\r".as_bytes(),
            "\u{e9}".as_bytes(),
            b"aaaaaaaaaaaaaa",
            b"a",
            b"A",
            b"0",
            b" sp",
            "\u{e9}2".as_bytes(),
        ] {
            assert_eq!(
                named(&map, &interner, id).as_deref(),
                Some(b" sp".as_slice())
            );
        }
        // Packed "lone" (0x6c) < non-ASCII fallbacks (0xc3).
        for id in [
            "lone",
            "\u{e9}3",
            "\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}",
        ] {
            assert_eq!(
                named(&map, &interner, id.as_bytes()).as_deref(),
                Some(b"lone".as_slice())
            );
        }
        // Packed vs fallback with a shared prefix: "abcdefghij" (packed, 10
        // bytes) < "abcdefghijk" (fallback, 11 bytes).
        let file = b"user_id_a,user_id_b\nabcdefghijk,abcdefghij\n";
        let interner = IdInterner::new();
        let map = build(&[file], &interner);
        assert_eq!(
            named(&map, &interner, b"abcdefghijk").as_deref(),
            Some(b"abcdefghij".as_slice())
        );
    }

    #[test]
    fn self_links_and_repeats_change_nothing() {
        let file = b"user_id_a,user_id_b\nm,m\nm,n\nn,m\nm,n\nn,n\no,o\n";
        let interner = IdInterner::new();
        let map = build(&[file], &interner);
        assert_eq!(map.len(), 3);
        assert_eq!(
            named(&map, &interner, b"n").as_deref(),
            Some(b"m".as_slice())
        );
        assert_eq!(
            named(&map, &interner, b"m").as_deref(),
            Some(b"m".as_slice())
        );
        // A self-link alone still makes the ID a (singleton) linked person.
        assert_eq!(
            named(&map, &interner, b"o").as_deref(),
            Some(b"o".as_slice())
        );
    }

    #[test]
    fn empty_sides_and_malformed_lines_are_dropped() {
        let file = b"user_id_a,user_id_b\n,e1\ne2,\n,\nm1\nm2,m3,m4\n\"m5,m6\nm7\",m8\nm9,\"m10\nok1,ok2\n";
        let interner = IdInterner::new();
        let map = build(&[file], &interner);
        for dropped in [
            b"e1".as_slice(),
            b"e2",
            b"m1",
            b"m2",
            b"m3",
            b"m4",
            b"m5",
            b"m6",
            b"m7",
            b"m8",
            b"m9",
            b"m10",
        ] {
            assert_eq!(named(&map, &interner, dropped), None, "{dropped:?}");
        }
        assert_eq!(
            named(&map, &interner, b"ok2").as_deref(),
            Some(b"ok1".as_slice())
        );
        assert_eq!(map.len(), 2);
    }
}
