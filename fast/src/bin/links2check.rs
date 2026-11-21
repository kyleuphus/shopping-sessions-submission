//! Differential and timing harness for the v2 links stage.
//!
//! Usage: `links2check [--repeat N] DATA_DIR...`
//! Each data directory must contain `links/*.csv`. The harness rebuilds v1's
//! exact sequential interner/union-find result, builds `links2::PersonMap`, and
//! compares the person-name bytes for every ID in every kept link line. It also
//! builds the previous sequential-union `links2` (embedded below as
//! `links2_old`) and checks that old and new `get()` agree, as bytes, for every
//! ID token appearing in any link line (kept or dropped).

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use fast::csvline::{data_lines, split_fields};
use fast::intern::Interner;
use fast::key::{key_bytes, key_of, IdInterner};
use fast::links2;
use fast::unionfind::UnionFind;
use memmap2::Mmap;

/// Verbatim copy of `fast/src/links2.rs` before the parallel union-find
/// rewrite (commit fbfdcfb), kept as the differential reference.
#[allow(dead_code)]
mod links2_old {
    // Previous (sequential union-find) v2 links stage over exact integer ID keys.
    //
    // Parsing is parallel over newline-aligned chunks and uses the same
    // `csvline::split_fields` implementation as v1. The parsed graph is resolved
    // sequentially over dense integer slots; union-by-size and path halving make
    // long chains and shared-device components effectively linear. The final
    // open-addressing table maps every ID in a kept link directly to the byte-wise
    // smallest key in its component.

    use rayon::prelude::*;

    use fast::csvline::split_fields;
    use fast::hash::fmix64;
    use fast::key::{key_cmp, key_of, IdInterner, IdKey};

    const CHUNK_BYTES: usize = 1 << 20;
    const EMPTY: u64 = u64::MAX;

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

    fn parse_chunk(chunk: &Chunk<'_>, interner: &IdInterner) -> Vec<(IdKey, IdKey)> {
        let mut pairs = Vec::with_capacity(chunk.bytes.len() / 24);
        let mut fields = Vec::with_capacity(2);
        for_each_line(chunk, |line| {
            if !split_fields(line, 2, &mut fields) || fields[0].is_empty() || fields[1].is_empty() {
                return;
            }
            pairs.push((key_of(fields[0], interner), key_of(fields[1], interner)));
        });
        pairs
    }

    #[derive(Clone, Copy)]
    #[repr(C, align(16))]
    struct Slot {
        key: IdKey,
        value: u64,
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
            let mut slot = home_slot(key, self.mask);
            loop {
                let entry = self.slots[slot];
                if entry.value == EMPTY {
                    return None;
                }
                if entry.key == key {
                    return Some(entry.value);
                }
                slot = (slot + 1) & self.mask;
            }
        }

        pub fn len(&self) -> usize {
            self.len
        }

        pub fn is_empty(&self) -> bool {
            self.len == 0
        }
    }

    #[inline]
    fn home_slot(key: IdKey, mask: usize) -> usize {
        fmix64(key ^ 0x6c69_6e6b_7332_5f6b) as usize & mask
    }

    fn table_capacity(len: usize) -> usize {
        (len + len.div_ceil(3)).next_power_of_two().max(16)
    }

    fn insert_index(slots: &mut [Slot], mask: usize, key: IdKey, index: u32) {
        let mut slot = home_slot(key, mask);
        while slots[slot].value != EMPTY {
            debug_assert_ne!(slots[slot].key, key, "unique ID inserted twice");
            slot = (slot + 1) & mask;
        }
        slots[slot] = Slot {
            key,
            value: index as u64,
        };
    }

    #[inline]
    fn index_of(slots: &[Slot], mask: usize, key: IdKey) -> u32 {
        let mut slot = home_slot(key, mask);
        loop {
            let entry = slots[slot];
            assert_ne!(
                entry.value, EMPTY,
                "parsed link ID missing from dense table"
            );
            if entry.key == key {
                return entry.value as u32;
            }
            slot = (slot + 1) & mask;
        }
    }

    struct UnionFind {
        parent: Vec<u32>,
        size: Vec<u32>,
        name: Vec<IdKey>,
    }

    impl UnionFind {
        fn new(keys: &[IdKey]) -> Self {
            assert!(
                u32::try_from(keys.len()).is_ok(),
                "more than u32::MAX linked IDs"
            );
            Self {
                parent: (0..keys.len() as u32).collect(),
                size: vec![1; keys.len()],
                name: keys.to_vec(),
            }
        }

        #[inline]
        fn find(&mut self, mut id: u32) -> u32 {
            while self.parent[id as usize] != id {
                let grandparent = self.parent[self.parent[id as usize] as usize];
                self.parent[id as usize] = grandparent;
                id = grandparent;
            }
            id
        }

        #[inline]
        fn union(&mut self, a: u32, b: u32, interner: &IdInterner) {
            let mut a_root = self.find(a);
            let mut b_root = self.find(b);
            if a_root == b_root {
                return;
            }
            let name = if key_cmp(
                self.name[a_root as usize],
                self.name[b_root as usize],
                interner,
            )
            .is_le()
            {
                self.name[a_root as usize]
            } else {
                self.name[b_root as usize]
            };
            if self.size[a_root as usize] < self.size[b_root as usize] {
                std::mem::swap(&mut a_root, &mut b_root);
            }
            self.parent[b_root as usize] = a_root;
            self.size[a_root as usize] += self.size[b_root as usize];
            self.name[a_root as usize] = name;
        }
    }

    /// Build the v2 person map from mmapped link-file contents.
    ///
    /// Parsing is parallel; the compact union-find is sequential after all IDs have
    /// integer keys. `interner` must also be used for keys passed to `PersonMap::get`
    /// and for decoding returned fallback keys.
    pub fn build(link_files: &[&[u8]], interner: &IdInterner) -> PersonMap {
        let file_chunks = chunks(link_files);
        let pairs: Vec<(IdKey, IdKey)> = file_chunks
            .par_iter()
            .map(|chunk| parse_chunk(chunk, interner))
            .reduce(Vec::new, |mut left, mut right| {
                if left.len() < right.len() {
                    std::mem::swap(&mut left, &mut right);
                }
                left.append(&mut right);
                left
            });

        let mut ids: Vec<IdKey> = pairs.par_iter().flat_map_iter(|&(a, b)| [a, b]).collect();
        ids.par_sort_unstable();
        ids.dedup();

        let capacity = table_capacity(ids.len());
        let mask = capacity - 1;
        let mut slots = vec![
            Slot {
                key: 0,
                value: EMPTY
            };
            capacity
        ];
        for (index, &key) in ids.iter().enumerate() {
            insert_index(&mut slots, mask, key, index as u32);
        }

        let mut union_find = UnionFind::new(&ids);
        for &(a, b) in &pairs {
            union_find.union(
                index_of(&slots, mask, a),
                index_of(&slots, mask, b),
                interner,
            );
        }

        // Tree roots are chosen for balance, while `name[root]` was maintained as
        // the exact byte minimum on every successful union. Rewrite dense indices
        // to those final person-name keys.
        for index in 0..ids.len() as u32 {
            let root = union_find.find(index);
            union_find.parent[index as usize] = root;
        }
        let parents = &union_find.parent;
        let names = &union_find.name;
        slots.par_iter_mut().for_each(|slot| {
            if slot.value != EMPTY {
                let root = parents[slot.value as usize] as usize;
                slot.value = names[root];
            }
        });

        PersonMap {
            slots,
            mask,
            len: ids.len(),
        }
    }
}

fn csv_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().is_some_and(|extension| extension == "csv") && path.is_file()
        })
        .collect();
    files.sort();
    files
}

fn map_all(paths: &[PathBuf]) -> Result<Vec<Mmap>, Box<dyn Error>> {
    let mut maps = Vec::with_capacity(paths.len());
    for path in paths {
        let file = std::fs::File::open(path)?;
        if file.metadata()?.len() == 0 {
            continue;
        }
        // The harness only maps input files and never writes them.
        maps.push(unsafe { Mmap::map(&file)? });
    }
    Ok(maps)
}

/// Reproduce `pipeline.rs::build_people` exactly through person naming.
fn build_v1<'a>(files: &[&'a [u8]]) -> (Interner<'a>, Vec<u32>) {
    let mut fields = Vec::with_capacity(2);
    let mut users = Interner::new();
    let mut union_find = UnionFind::new(0);
    for &file in files {
        for line in data_lines(file) {
            if !split_fields(line, 2, &mut fields) || fields[0].is_empty() || fields[1].is_empty() {
                continue;
            }
            let a = users.intern(fields[0]);
            let b = users.intern(fields[1]);
            union_find.grow_to(users.len());
            union_find.union(a, b);
        }
    }

    let roots: Vec<u32> = (0..users.len() as u32)
        .map(|id| union_find.find(id))
        .collect();
    let mut root_names = vec![u32::MAX; users.len()];
    for id in 0..users.len() as u32 {
        let root = roots[id as usize] as usize;
        let current = root_names[root];
        if current == u32::MAX || users.less(id, current) {
            root_names[root] = id;
        }
    }
    let names = roots
        .into_iter()
        .map(|root| root_names[root as usize])
        .collect();
    (users, names)
}

fn median(times: &mut [Duration]) -> Duration {
    times.sort_unstable();
    times[times.len() / 2]
}

fn check_dir(data_dir: &Path, repeats: usize) -> Result<(), Box<dyn Error>> {
    let paths = csv_files(&data_dir.join("links"));
    if paths.is_empty() {
        return Err(format!("{} has no links/*.csv files", data_dir.display()).into());
    }
    let maps = map_all(&paths)?;
    let files: Vec<&[u8]> = maps.iter().map(|map| &map[..]).collect();

    let v1_started = Instant::now();
    let (v1_users, v1_names) = build_v1(&files);
    let v1_elapsed = v1_started.elapsed();

    // Initialize Rayon and fault the link pages before the measured builds.
    let warm_interner = IdInterner::new();
    drop(links2::build(&files, &warm_interner));

    let mut times = Vec::with_capacity(repeats);
    let mut final_build = None;
    for _ in 0..repeats {
        let interner = IdInterner::new();
        let started = Instant::now();
        let people = links2::build(&files, &interner);
        times.push(started.elapsed());
        final_build = Some((interner, people));
    }
    let (interner, people) = final_build.expect("repeat count is nonzero");

    let mut mismatches = 0usize;
    let mut actual_buf = Vec::new();
    let mut fields_buf = Vec::with_capacity(2);
    for id in 0..v1_users.len() as u32 {
        let bytes = v1_users.bytes(id);
        let expected = v1_users.bytes(v1_names[id as usize]);
        let actual = people
            .get(key_of(bytes, &interner))
            .map(|key| key_bytes(key, &interner, &mut actual_buf));
        if actual != Some(expected) {
            mismatches += 1;
            if mismatches <= 20 {
                eprintln!(
                    "{}: mismatch id={:?} expected={:?} actual={:?}",
                    data_dir.display(),
                    String::from_utf8_lossy(bytes),
                    String::from_utf8_lossy(expected),
                    actual.map(String::from_utf8_lossy)
                );
            }
        }
    }

    // Old vs new links2: every comma-separated token of every data line, so IDs
    // from dropped lines are checked too (both must say None).
    let old_interner = IdInterner::new();
    let old_people = links2_old::build(&files, &old_interner);
    let mut tokens: Vec<&[u8]> = Vec::new();
    for &file in &files {
        for line in data_lines(file) {
            tokens.extend(line.split(|&byte| byte == b',').filter(|t| !t.is_empty()));
            if split_fields(line, 2, &mut fields_buf) {
                tokens.extend(fields_buf.iter().copied().filter(|t| !t.is_empty()));
            }
        }
    }
    tokens.sort_unstable();
    tokens.dedup();
    let mut old_buf = Vec::new();
    let mut diffs = 0usize;
    for &token in &tokens {
        let new = people
            .get(key_of(token, &interner))
            .map(|key| key_bytes(key, &interner, &mut actual_buf).to_vec());
        let old = old_people
            .get(key_of(token, &old_interner))
            .map(|key| key_bytes(key, &old_interner, &mut old_buf).to_vec());
        if new != old {
            diffs += 1;
            if diffs <= 20 {
                eprintln!(
                    "{}: old/new differ id={:?} old={:?} new={:?}",
                    data_dir.display(),
                    String::from_utf8_lossy(token),
                    old.as_deref().map(String::from_utf8_lossy),
                    new.as_deref().map(String::from_utf8_lossy)
                );
            }
        }
    }
    if old_people.len() != people.len() {
        diffs += 1;
        eprintln!(
            "{}: len old={} new={}",
            data_dir.display(),
            old_people.len(),
            people.len()
        );
    }
    if diffs == 0 {
        println!(
            "links2 diff: IDENTICAL {} ids ({})",
            tokens.len(),
            data_dir.display()
        );
    } else {
        println!(
            "links2 diff: DIFFERENT {diffs} of {} ids ({})",
            tokens.len(),
            data_dir.display()
        );
        return Err(format!("{} old/new links2 differ", data_dir.display()).into());
    }

    let median = median(&mut times);
    println!(
        "{}: compared={} mismatches={} v1_reference={:.3}ms v2_median_{}={:.3}ms fallbacks={}",
        data_dir.display(),
        v1_users.len(),
        mismatches,
        v1_elapsed.as_secs_f64() * 1_000.0,
        repeats,
        median.as_secs_f64() * 1_000.0,
        interner.len()
    );
    if mismatches != 0 {
        return Err(format!("{} had {mismatches} mismatches", data_dir.display()).into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let mut repeats = 1usize;
    let mut paths = Vec::new();
    while let Some(arg) = args.next() {
        if arg == "--repeat" {
            let value = args.next().ok_or("--repeat requires a positive integer")?;
            repeats = value.to_string_lossy().parse()?;
            if repeats == 0 {
                return Err("--repeat must be positive".into());
            }
        } else {
            paths.push(PathBuf::from(arg));
        }
    }
    if paths.is_empty() {
        return Err("usage: links2check [--repeat N] DATA_DIR...".into());
    }
    for path in paths {
        check_dir(&path, repeats)?;
    }
    Ok(())
}
