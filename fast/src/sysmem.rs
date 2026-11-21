//! Best-effort Linux memory hints. Wiring for `pipeline3.rs` (four lines):
//! Replace `Buckets::new`'s `parts` initializer with the first line;
//! put the remaining three lines just before `std::process::exit(0)`, after
//! all output writers are closed and every Rayon reader has joined.
//! ```ignore
//! parts: (0..partitions).map(|_| { let mut v = Vec::with_capacity(capacity_each); crate::sysmem::advise_hugepage(v.spare_capacity_mut()); v }).collect(),
//! fn region<T>(v: &[T]) -> (*const u8, usize) { (v.as_ptr().cast(), std::mem::size_of_val(v)) }
//! let regions: Vec<_> = event_maps.iter().chain(&click_maps).chain(&link_maps).map(|m| region(&m[..])).chain(event_buckets.iter().flat_map(|b| &b.parts).map(|v| region(&v[..]))).chain(dedup_workers.iter().flat_map(|w| &w.out.parts).map(|v| region(&v[..]))).chain(click_workers.iter().flat_map(|w| &w.out.parts).map(|v| region(&v[..]))).collect();
//! crate::sysmem::release_parallel(&regions);
//! ```

/// Hint that the complete 2 MiB-aligned interior may use transparent huge pages.
///
/// An interior smaller than 2 MiB is skipped. Errors are ignored; this neither
/// guarantees huge pages nor changes the slice's contents. Non-Linux: no-op.
/// For an empty Vec's allocation, pass `v.spare_capacity_mut()` rather than
/// `&mut v[..]`; the former is a valid slice of `MaybeUninit<T>`.
pub fn advise_hugepage<T>(v: &mut [T]) {
    #[cfg(target_os = "linux")]
    linux::advise(v.as_mut_ptr().cast(), std::mem::size_of_val(v));
    #[cfg(not(target_os = "linux"))]
    let _ = v;
}

/// Reserve at least `len` elements, returning a Vec whose **length is zero**.
///
/// Uses the permitted ordinary-Vec fallback, including for allocations >= 4 MiB:
/// `Vec<T>` frees through the global allocator with `T`'s alignment. An allocation
/// made with a 2 MiB-aligned Layout cannot soundly be handed to `Vec::from_raw_parts`
/// with a different layout; the default Vec cannot carry a custom deallocator.
/// Instead hint its complete huge-page interior through `MaybeUninit<T>` spare
/// capacity. No uninitialized `T` is ever exposed, and normal growth/drop work.
/// Reapply the hint after a reallocation if necessary.
pub fn hugepage_vec<T: Copy>(len: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(len);
    advise_hugepage(v.spare_capacity_mut());
    v
}

/// Discard complete pages within these regions in parallel, just before exit.
///
/// **Caller contract:** pass only live, stable anonymous-buffer payloads or file
/// mappings whose contents will NEVER be read again, including by other threads
/// or destructors. Join readers and close output files first. Do not include
/// allocator metadata, stacks, or memory used by this call/Rayon. Keep mappings
/// alive until this function returns; do not unmap/reallocate them concurrently.
/// Anonymous pages subsequently read as zeros; private file mappings refault the
/// file contents (private modifications are lost). Objects with references or
/// destructors can become invalid, so exit immediately or forget those objects.
/// An ordinary byte Vec can still be dropped without inspecting its payload.
///
/// Partial boundary pages are preserved to avoid discarding neighboring objects.
/// Work is split at 2 MiB boundaries with page-aligned head/tail pieces. Syscall
/// errors are ignored. Non-Linux: no-op.
pub fn release_parallel(regions: &[(*const u8, usize)]) {
    #[cfg(target_os = "linux")]
    linux::release(regions);
    #[cfg(not(target_os = "linux"))]
    let _ = regions;
}

/// Return Linux's THP configuration text, including the bracketed active mode.
/// Returns `None` if unavailable or on other operating systems.
pub fn thp_mode() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled").ok()
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use rayon::prelude::*;

    use std::os::raw::{c_int, c_long, c_void};

    const HUGE: usize = 2 * 1024 * 1024;
    // Linux x86_64/aarch64 ABI constants; no direct libc dependency in Cargo.toml.
    const MADV_HUGEPAGE: c_int = 14;
    const MADV_DONTNEED: c_int = 4;
    const SC_PAGESIZE: c_int = 30;

    unsafe extern "C" {
        fn madvise(addr: *mut c_void, length: usize, advice: c_int) -> c_int;
        fn sysconf(name: c_int) -> c_long;
    }

    // All intervals are rounded INWARD; overflow and empty intervals are skipped.
    fn interior(start: usize, len: usize, align: usize) -> Option<(usize, usize)> {
        let end = start.checked_add(len)?;
        let lo = start.checked_add(align - 1)? & !(align - 1);
        let hi = end & !(align - 1);
        (lo < hi).then_some((lo, hi))
    }

    pub(super) fn advise(ptr: *mut u8, len: usize) {
        if let Some((lo, hi)) = interior(ptr as usize, len, HUGE) {
            // The entire advised range belongs to the supplied slice; advice
            // preserves its values. The return code is deliberately ignored.
            unsafe { madvise(lo as *mut c_void, hi - lo, MADV_HUGEPAGE) };
        }
    }

    fn pieces(start: usize, end: usize) -> impl Iterator<Item = (usize, usize)> {
        let mut next = start;
        std::iter::from_fn(move || {
            if next >= end {
                return None;
            }
            let lo = next;
            // The first/last piece may be short; middle pieces are exactly 2 MiB.
            next = (lo | (HUGE - 1)).checked_add(1).unwrap_or(end).min(end);
            Some((lo, next - lo))
        })
    }

    pub(super) fn release(regions: &[(*const u8, usize)]) {
        // x86_64 uses 4 KiB; query instead of hardcoding for aarch64 Linux builds
        // whose base pages may be 16/64 KiB. A failed query means no advice.
        let page = unsafe { sysconf(SC_PAGESIZE) };
        if page <= 0 || !(page as usize).is_power_of_two() || HUGE % page as usize != 0 {
            return;
        }
        // Send integer ranges, not raw pointers, to Rayon. No worker dereferences
        // a pointer in Rust; the caller guarantees all these mappings stay live.
        let chunks: Vec<_> = regions
            .iter()
            .filter(|&&(ptr, _)| !ptr.is_null())
            .filter_map(|&(ptr, len)| interior(ptr as usize, len, page as usize))
            .flat_map(|(lo, hi)| pieces(lo, hi))
            .collect();
        chunks.into_par_iter().for_each(|(lo, len)| {
            // Caller contract: payload is dead and no other thread accesses it.
            // Inward rounding keeps allocator headers/adjacent objects intact.
            unsafe { madvise(lo as *mut c_void, len, MADV_DONTNEED) };
        });
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn inward_rounding_and_chunk_coverage() {
            assert_eq!(interior(1, HUGE, HUGE), None);
            assert_eq!(
                interior(HUGE + 1, 3 * HUGE, HUGE),
                Some((2 * HUGE, 4 * HUGE))
            );
            assert_eq!(interior(usize::MAX - 1, 4, HUGE), None);
            let (lo, hi) = interior(1, 4 * HUGE + 123, 4096).unwrap();
            let chunks: Vec<_> = pieces(lo, hi).collect();
            assert_eq!(chunks[0].0, lo);
            assert_eq!(chunks.iter().map(|&(_, n)| n).sum::<usize>(), hi - lo);
            for (i, &(p, n)) in chunks.iter().enumerate() {
                assert_eq!(p % 4096, 0);
                assert_eq!(n % 4096, 0);
                assert!(n > 0 && n <= HUGE);
                if i > 0 {
                    assert_eq!(p % HUGE, 0);
                    assert_eq!(chunks[i - 1].0 + chunks[i - 1].1, p);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_vec_is_empty_and_usable_for_eight_mib() {
        let len = 8 * 1024 * 1024 / std::mem::size_of::<u64>();
        let mut v = hugepage_vec::<u64>(len);
        assert!(v.is_empty());
        assert!(v.capacity() >= len);
        for i in 0..len {
            v.push((i as u64).wrapping_mul(17));
        }
        advise_hugepage(&mut v);
        for (i, value) in v.iter_mut().enumerate() {
            assert_eq!(*value, (i as u64).wrapping_mul(17));
            *value ^= 0x1234;
        }
        for (i, &value) in v.iter().enumerate() {
            assert_eq!(value, (i as u64).wrapping_mul(17) ^ 0x1234);
        }
        v.push(42); // Normal reallocation remains valid too.
        assert_eq!(v.pop(), Some(42));
        assert!(hugepage_vec::<u8>(0).is_empty());
        let mut zst = hugepage_vec::<()>(123);
        zst.push(());
        assert_eq!(zst.len(), 1);
    }

    #[test]
    fn small_empty_and_unaligned_slices_preserve_data() {
        let mut small = [37u8; 1024];
        advise_hugepage(&mut small[1..1023]);
        advise_hugepage(&mut small[..0]);
        assert!(small.iter().all(|&b| b == 37));
        let mut large = vec![91u8; 8 * 1024 * 1024 + 2];
        advise_hugepage(&mut large[1..]);
        assert!(large.iter().all(|&b| b == 91));
    }

    #[test]
    fn discard_sixty_four_mib_and_keep_running() {
        let v = vec![0xa5u8; 64 * 1024 * 1024];
        // Ensure the initialized payload is materialized before discarding it.
        std::hint::black_box(&v[..]);
        release_parallel(&[(v.as_ptr(), v.len())]);
        // Do not read the discarded payload. Dropping a byte Vec reads only its
        // allocation descriptor, which is outside the discarded interior.
        drop(v);
        let mut fresh = vec![7u8; 8192];
        fresh[4096] = 13;
        assert_eq!(
            fresh.iter().map(|&b| b as usize).sum::<usize>(),
            7 * 8192 + 6
        );
        release_parallel(&[]);
        release_parallel(&[(std::ptr::null(), 0)]);
    }
}
