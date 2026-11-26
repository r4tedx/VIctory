//! Copy-back traffic by the original console reservation flags.

use std::collections::BTreeMap;
use std::sync::Mutex;

use windows_sys::Win32::System::Memory::{PAGE_NOCACHE, PAGE_WRITECOMBINE};

// The console adds GPU access permissions in protection bits 11 through 18.
const GPU_PROTECTION: u32 = 0x7f800;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Flags {
    kind: u32,
    flags: u64,
    protect: u32,
}

#[derive(Default)]
struct Census {
    // Non-overlapping ranges, keyed by base with an exclusive end.
    reservations: BTreeMap<usize, (usize, Flags)>,
    bytes: BTreeMap<Option<Flags>, usize>,
}

static CENSUS: Mutex<Census> = Mutex::new(Census {
    reservations: BTreeMap::new(),
    bytes: BTreeMap::new(),
});

impl Census {
    fn reserve(&mut self, base: usize, end: usize, flags: Flags) {
        // A title reservation can be released through a native import and
        // its address reused. Replace any old metadata for the new extent.
        if let Some((&start, &(old_end, old_flags))) = self.reservations.range(..base).next_back()
            && old_end > base
        {
            self.reservations.insert(start, (base, old_flags));
            if old_end > end {
                self.reservations.insert(end, (old_end, old_flags));
            }
        }
        while let Some((&start, &(old_end, old_flags))) = self.reservations.range(base..end).next()
        {
            self.reservations.remove(&start);
            if old_end > end {
                self.reservations.insert(end, (old_end, old_flags));
            }
        }
        self.reservations.insert(base, (end, flags));
    }

    fn record(&mut self, base: usize, len: usize) {
        let end = base + len;
        let mut at = base;
        if let Some((_, &(limit, flags))) = self.reservations.range(..base).next_back()
            && limit > base
        {
            at = limit.min(end);
            *self.bytes.entry(Some(flags)).or_default() += at - base;
        }
        for (&start, &(limit, flags)) in self.reservations.range(base..end) {
            if start > at {
                *self.bytes.entry(None).or_default() += start - at;
            }
            at = limit.min(end);
            *self.bytes.entry(Some(flags)).or_default() += at - start;
        }
        if at < end {
            *self.bytes.entry(None).or_default() += end - at;
        }
    }
}

/// Remember a successful title reservation, including the bits stripped by HLE.
pub fn title_memory_reserved(base: usize, len: usize, kind: u32, flags: u64, protect: u32) {
    if let Some(end) = base.checked_add(len)
        && len != 0
    {
        CENSUS.lock().unwrap_or_else(|e| e.into_inner()).reserve(
            base,
            end,
            Flags {
                kind,
                flags,
                protect,
            },
        );
    }
}

/// Release a whole reservation through HLE. Hold the census lock across
/// release so a concurrent allocation at the same address registers after it.
pub fn release_title_reservation(base: usize, release: impl FnOnce() -> bool) {
    let mut census = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
    if release() {
        census.reservations.remove(&base);
    }
}

/// Count bytes reported transferred by a copy-back call, including retries.
pub(crate) fn record(base: usize, len: usize) {
    if len != 0 {
        CENSUS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .record(base, len);
    }
}

pub(crate) fn log(frames: f64) {
    use std::fmt::Write;

    let bytes = std::mem::take(&mut CENSUS.lock().unwrap_or_else(|e| e.into_inner()).bytes);
    let mut buckets = String::new();
    for (flags, bytes) in bytes {
        let average = bytes as f64 / frames;
        if !buckets.is_empty() {
            buckets.push_str(", ");
        }
        if let Some(Flags {
            kind,
            flags,
            protect,
        }) = flags
        {
            let wc = u8::from(protect & PAGE_WRITECOMBINE != 0);
            let uncached = u8::from(protect & PAGE_NOCACHE != 0);
            let gpu = protect & GPU_PROTECTION;
            let _ = write!(
                buckets,
                "kind={kind:#x} flags={flags:#x} protect={protect:#x} wc={wc} uncached={uncached} gpu={gpu:#x}: {average:.1}"
            );
        } else {
            let _ = write!(buckets, "unknown: {average:.1}");
        }
    }
    tracing::info!("copy-back bytes per frame ({frames:.0} frames): [{buckets}]");
}

#[cfg(test)]
mod tests {
    use super::*;

    const WC: Flags = Flags {
        kind: 0x2000,
        flags: 3,
        protect: 0x4404,
    };
    const UNCACHED: Flags = Flags {
        kind: 0x3000,
        flags: 7,
        protect: 0x20204,
    };

    #[test]
    fn splits_bytes_at_reservation_boundaries_and_gaps() {
        let mut census = Census::default();
        census.reserve(100, 200, WC);
        census.reserve(200, 250, UNCACHED);
        census.reserve(300, 400, WC);
        census.record(175, 150);
        assert_eq!(
            census.bytes,
            BTreeMap::from([(None, 50), (Some(WC), 50), (Some(UNCACHED), 50),])
        );
    }

    #[test]
    fn reuse_replaces_only_the_overlapping_reservation_metadata() {
        let mut census = Census::default();
        census.reserve(100, 200, WC);
        census.reserve(125, 175, UNCACHED);
        census.record(90, 130);
        assert_eq!(
            census.bytes,
            BTreeMap::from([(None, 30), (Some(WC), 50), (Some(UNCACHED), 50),])
        );
    }

    #[test]
    fn mapped_writeback_counts_transfers_under_the_registered_flags() {
        const KIND: u32 = 0x3000;
        const FLAGS: u64 = 0xc700_0000;
        const PROTECT: u32 = 0x4404;
        let mut dst = [0u8; 16];
        let base = dst.as_mut_ptr() as usize;
        let known = dst.len() / 2;
        title_memory_reserved(base, known, KIND, FLAGS, PROTECT);
        let src = [7u8; 16];
        // The first half is known and the second half has no reservation
        // metadata. Both halves are backed by writable stack memory.
        crate::host::write_mapped(dst.as_mut_ptr(), src.as_ptr(), dst.len());
        release_title_reservation(base, || true);
        let mut census = CENSUS.lock().unwrap();
        assert!(!census.reservations.contains_key(&base));
        assert_eq!(
            census.bytes.remove(&Some(Flags {
                kind: KIND,
                flags: FLAGS,
                protect: PROTECT,
            })),
            Some(known),
        );
        assert_eq!(dst, [7; 16]);
    }
}
