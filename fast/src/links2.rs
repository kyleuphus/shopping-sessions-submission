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
use crate::key::{key_cmp, key_of, user_key_cmp, user_key_of, IdInterner, IdKey};

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

/// One folded multiply gives the alias lookup its index and supplies the high
/// bits for person routing. Every candidate still compares the full key.
#[inline(always)]
pub(crate) fn alias_hash(key: IdKey) -> u64 {
    let product = u128::from(key ^ 0x7065_7273_6f6e_7633) * 0xa076_1d64_78bd_642f;
    product as u64 ^ (product >> 64) as u64
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
/// `key + 1` never wraps: the regular key format uses low nibbles 0..=10,
/// and the ASCII user format's five-bit prefix code is always at most 25.
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

    fn union<const ALIASES_ONLY: bool>(&self, a: u32, b: u32, interner: &IdInterner) -> bool {
        let (mut a, mut b) = (a, b);
        loop {
            a = self.find(a);
            b = self.find(b);
            if a == b {
                return false;
            }
            let a_key = self.table.key(a);
            let b_key = self.table.key(b);
            let order = if ALIASES_ONLY {
                user_key_cmp(a_key, b_key, interner)
            } else {
                key_cmp(a_key, b_key, interner)
            };
            let (lo, hi) = if order.is_lt() {
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
                return true;
            }
        }
    }

    fn for_each_name<const ALIASES_ONLY: bool>(&self, insert: impl Fn(u64, IdKey) + Send + Sync) {
        self.table
            .keys
            .par_iter()
            .enumerate()
            .with_min_len(4096)
            .for_each(|(slot, word)| {
                let stored = word.load(Relaxed);
                if stored != 0 {
                    let root = self.find(slot as u32);
                    if !ALIASES_ONLY || root as usize != slot {
                        insert(stored, self.table.key(root));
                    }
                }
            });
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

#[repr(C, align(64))]
struct Bucket {
    keys: [AtomicU64; 4],
    values: [AtomicU64; 4],
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

/// Read-only after links finish. Roots and unlinked IDs already name
/// themselves, so this table stores only IDs that actually change names.
pub(crate) struct AliasMap {
    buckets: Vec<Bucket>,
    filter: Vec<AtomicU64>,
    mask: usize,
}

impl AliasMap {
    #[inline(always)]
    fn filter_bit(&self, hash: u64) -> (&AtomicU64, u64) {
        let word = &self.filter[hash as usize & (self.filter.len() - 1)];
        let bit = 1u64 << ((hash >> 32) & 63);
        (word, bit)
    }

    /// A missing bit rules out an alias. A set bit still requires a full key
    /// comparison in the name table.
    #[inline(always)]
    pub(crate) fn may_contain(&self, hash: u64) -> bool {
        let (word, bit) = self.filter_bit(hash);
        word.load(Relaxed) & bit != 0
    }

    #[inline(always)]
    pub(crate) fn prefetch(&self, hash: u64) {
        let index = hash as usize & self.mask;
        prefetch_read(self.buckets.as_ptr().wrapping_add(index).cast());
    }

    #[inline(always)]
    pub(crate) fn resolve(&self, key: IdKey, hash: u64) -> IdKey {
        let stored = key + 1;
        let mut index = hash as usize & self.mask;
        loop {
            let bucket = &self.buckets[index];
            if let Some(lane) = bucket.find(stored) {
                return bucket.values[lane].load(Relaxed);
            }
            // Build fills lanes from left to right. An empty last lane means
            // that no key probed past this bucket.
            if bucket.keys[3].load(Relaxed) == 0 {
                return key;
            }
            index = (index + 1) & self.mask;
        }
    }

    fn insert(&self, stored: u64, name: IdKey) {
        let hash = alias_hash(stored - 1);
        let (word, bit) = self.filter_bit(hash);
        word.fetch_or(bit, Relaxed);
        let mut index = hash as usize & self.mask;
        loop {
            let bucket = &self.buckets[index];
            for (key, value) in bucket.keys.iter().zip(&bucket.values) {
                if key.load(Relaxed) == 0
                    && key.compare_exchange(0, stored, Relaxed, Relaxed).is_ok()
                {
                    value.store(name, Relaxed);
                    return;
                }
            }
            index = (index + 1) & self.mask;
        }
    }
}

impl Bucket {
    /// Reader-only: after the parallel build joins, none of the key words are
    /// ever written again. Four keys fit in one AVX2 load.
    #[inline(always)]
    fn find(&self, stored: u64) -> Option<usize> {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        // SAFETY: this is an aligned read of the bucket's four immutable keys.
        unsafe {
            use core::arch::x86_64::*;
            let keys = _mm256_load_si256(self.keys.as_ptr().cast());
            let matching = _mm256_cmpeq_epi64(keys, _mm256_set1_epi64x(stored as i64));
            let mask = _mm256_movemask_pd(_mm256_castsi256_pd(matching)) as u32;
            (mask != 0).then(|| mask.trailing_zeros() as usize)
        }
        #[cfg(target_arch = "aarch64")]
        // SAFETY: both vector loads are within the four immutable key words.
        unsafe {
            use core::arch::aarch64::*;
            let ptr = self.keys.as_ptr().cast::<u64>();
            let target = vdupq_n_u64(stored);
            let a = vmovn_u64(vceqq_u64(vld1q_u64(ptr), target));
            let b = vmovn_u64(vceqq_u64(vld1q_u64(ptr.add(2)), target));
            // Each comparison occupies 16 bits in the final lane mask.
            let matches = vreinterpret_u64_u16(vshrn_n_u32::<16>(vcombine_u32(a, b)));
            let mask = vget_lane_u64::<0>(matches);
            (mask != 0).then(|| mask.trailing_zeros() as usize / 16)
        }
        #[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx2"))))]
        {
            self.keys.iter().position(|key| key.load(Relaxed) == stored)
        }
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
    build_names::<false, _>(link_files, interner, |forest, len| {
        let capacity = table_capacity(len);
        let mask = capacity - 1;
        // SAFETY: all-zero bytes represent an empty slot.
        let slots: Vec<Slot> = unsafe { zeroed_vec(capacity) };
        forest.for_each_name::<false>(|stored, name| {
            let mut at = home_slot(stored - 1, mask);
            loop {
                let target = &slots[at];
                if target.key.compare_exchange(0, stored, Relaxed, Relaxed).is_ok() {
                    target.value.store(name, Relaxed);
                    return;
                }
                at = (at + 1) & mask;
            }
        });
        PersonMap { slots, mask, len }
    })
}

/// Build the alias-only map with the v3 user key encoding. Queries and output
/// use the same interner and encoding as the link endpoints.
pub(crate) fn build_aliases(link_files: &[&[u8]], interner: &IdInterner) -> AliasMap {
    build_names::<true, _>(link_files, interner, |forest, aliases| {
        let capacity = table_capacity(aliases);
        let bucket_count = capacity / 4;
        // Four filter bits per slot. The bitset rejects most unchanged IDs
        // without loading a bucket, and collisions only cause extra probes.
        // SAFETY: zero denotes an empty bucket and an unset filter word.
        let map = AliasMap {
            buckets: unsafe { zeroed_vec(bucket_count) },
            filter: unsafe { zeroed_vec((capacity / 16).max(1)) },
            mask: bucket_count - 1,
        };
        forest.for_each_name::<true>(|stored, name| map.insert(stored, name));
        map
    })
}

fn build_names<const ALIASES_ONLY: bool, T>(
    link_files: &[&[u8]],
    interner: &IdInterner,
    finalize: impl FnOnce(&Forest<'_>, usize) -> T,
) -> T {
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
    let batch_scan = cfg!(all(target_arch = "x86_64", target_feature = "avx2"))
        && std::env::var_os("FAST_LINK_SCAN").as_deref() != Some(std::ffi::OsStr::new("0"));
    let distinct = AtomicUsize::new(0);
    let edges: Vec<Vec<(u32, u32)>> = file_chunks
        .par_iter()
        .map(|chunk| {
            let mut edges = Vec::with_capacity(chunk.bytes.len() / 20);
            let mut inserted = 0usize;
            let mut insert = |left: &[u8], right: &[u8]| {
                if left.is_empty() || right.is_empty() {
                    return;
                }
                let encode = |id| {
                    if ALIASES_ONLY {
                        user_key_of(id, interner)
                    } else {
                        key_of(id, interner)
                    }
                };
                let (a, new_a) = table.insert(encode(left));
                let (b, new_b) = table.insert(encode(right));
                inserted += new_a as usize + new_b as usize;
                if a != b {
                    edges.push((a, b));
                }
            };
            if batch_scan {
                crate::scan3::for_each_record::<2>(chunk.bytes, chunk.has_header, true, |fields| {
                    insert(fields[0], fields[1]);
                });
            } else {
                let mut fields = Vec::with_capacity(2);
                for_each_line(chunk, |line| {
                    if split_fields(line, 2, &mut fields) {
                        insert(fields[0], fields[1]);
                    }
                });
            }
            if !ALIASES_ONLY {
                distinct.fetch_add(inserted, Relaxed);
            }
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
    // Every successful union installs one new non-root parent. This counts the
    // exact aliases without rescanning the forest to size the final table.
    let aliases: usize = edges
        .par_iter()
        .map(|chunk_edges| {
            let mut joined = 0;
            for &(a, b) in chunk_edges {
                joined += forest.union::<ALIASES_ONLY>(a, b, interner) as usize;
            }
            joined
        })
        .sum();
    drop(edges);
    timer.mark("union");

    // Phase 3 publishes the exact names after all union threads have joined.
    let names = finalize(&forest, if ALIASES_ONLY { aliases } else { len });
    timer.mark("name map");
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key::{key_bytes, user_key_bytes};

    #[test]
    fn alias_buckets_check_full_keys_when_the_last_bucket_wraps() {
        let old_ids = IdInterner::with_hex_log2(6);
        let new_ids = IdInterner::with_hex_log2(6);
        let colliders: Vec<_> = (0..1000)
            .map(|i| format!("q{i:08}"))
            .filter(|id| alias_hash(user_key_of(id.as_bytes(), &new_ids)) & 3 == 3)
            .take(7)
            .collect();
        assert_eq!(colliders.len(), 7);
        let mut links = String::from("a,b\n");
        for alias in &colliders[..6] {
            links.push_str(&format!("{alias},a_root\n"));
        }
        links.push_str("a_root,a_root\n");
        let input = [links.as_bytes()];
        let old = build(&input, &old_ids);
        let new = build_aliases(&input, &new_ids);
        assert_eq!(new.mask, 3);
        let mut old_bytes = Vec::new();
        let mut new_bytes = Vec::new();
        for id in colliders.iter().map(String::as_bytes).chain([b"a_root".as_slice()]) {
            let legacy = key_of(id, &old_ids);
            let legacy = old.get(legacy).unwrap_or(legacy);
            let key = user_key_of(id, &new_ids);
            let hash = alias_hash(key);
            let resolved = if new.may_contain(hash) { new.resolve(key, hash) } else { key };
            assert_eq!(
                user_key_bytes(resolved, &new_ids, &mut new_bytes),
                key_bytes(legacy, &old_ids, &mut old_bytes),
            );
        }
    }

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
