use std::sync::Mutex;
use std::time::{Duration, Instant};

const BAND_SLOTS: u32 = 1 << 16;
const REPORT_INTERVAL: Duration = Duration::from_secs(10);

pub(super) struct OverflowWrites {
    cap: u32,
    title_count: u32,
    state: Mutex<Writes>,
}

struct Writes {
    /// The same three dwords that mirroring uses to recognize a descriptor.
    records: Vec<[u32; 3]>,
    /// One bit per overflow slot written during this reporting interval.
    seen: Vec<u64>,
    /// Counts in absolute 64 Ki-slot bands, starting with the band at the cap.
    bands: Vec<u64>,
    written: u64,
    distinct: u32,
    since: Instant,
}

impl OverflowWrites {
    pub(super) fn new(cap: u32, title_count: u32) -> Option<Self> {
        let slots = title_count.checked_sub(cap)?;
        (slots != 0).then(|| Self {
            cap,
            title_count,
            state: Mutex::new(Writes {
                records: vec![[0; 3]; slots as usize],
                seen: vec![0; slots.div_ceil(u64::BITS) as usize],
                bands: vec![0; (title_count.div_ceil(BAND_SLOTS) - cap / BAND_SLOTS) as usize],
                written: 0,
                distinct: 0,
                since: Instant::now(),
            }),
        })
    }

    pub(super) fn contains(&self, index: u32) -> bool {
        (self.cap..self.title_count).contains(&index)
    }

    pub(super) fn observe_write(&self, index: u32, record: Option<[u32; 3]>) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(record) = record {
            state.records[(index - self.cap) as usize] = record;
        }
        state.count(self.cap, index);
    }

    pub(super) fn observe_sync(&self, first: u32, records: &[[u32; 3]]) {
        let skip = self.cap.saturating_sub(first) as usize;
        let end = records.len().min(self.title_count.saturating_sub(first) as usize);
        if skip >= end {
            return;
        }
        // A dirty page gives no per-slot write events. Count only the
        // changed records, and use explicit writes as the baseline so the
        // next dirty-page sync does not count them a second time.
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        for (i, &record) in records[skip..end].iter().enumerate() {
            let index = first + (skip + i) as u32;
            let previous = &mut state.records[(index - self.cap) as usize];
            if *previous != record {
                *previous = record;
                state.count(self.cap, index);
            }
        }
    }

    pub(super) fn report(&self, heap: u64, now: Instant) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if now.duration_since(state.since) < REPORT_INTERVAL {
            return;
        }
        tracing::info!(
            heap = format_args!("{heap:#x}"),
            cap = self.cap,
            first_band = self.cap / BAND_SLOTS,
            "title slots past the host heap cap: written={} distinct={} bands={:?}",
            state.written,
            state.distinct,
            state.bands,
        );
        state.written = 0;
        state.distinct = 0;
        state.seen.fill(0);
        state.bands.fill(0);
        state.since = now;
    }
}

impl Writes {
    fn count(&mut self, cap: u32, index: u32) {
        self.written += 1;
        self.bands[(index / BAND_SLOTS - cap / BAND_SLOTS) as usize] += 1;
        let offset = index - cap;
        let word = &mut self.seen[(offset / u64::BITS) as usize];
        let mask = 1 << (offset % u64::BITS);
        self.distinct += u32::from(*word & mask == 0);
        *word |= mask;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_writes_at_cap_and_across_band_boundary() {
        let cap = BAND_SLOTS - 1;
        let overflow = OverflowWrites::new(cap, cap + 3).unwrap();
        assert!(OverflowWrites::new(3, 3).is_none());
        assert!(!overflow.contains(cap - 1));
        assert!(overflow.contains(cap));
        assert!(!overflow.contains(cap + 3));

        let a = [1, 2, 3];
        let b = [1, 3, 3];
        let empty = [0; 3];
        overflow.observe_write(cap, Some(a));
        overflow.observe_write(cap, None); // a queued write to a now-unmapped slot
        overflow.observe_sync(cap - 1, &[b, a, b, empty, b]);
        overflow.observe_sync(cap, &[a, b, empty]);
        overflow.observe_sync(cap + 1, &[empty]); // cleared by the title

        let since = overflow.state.lock().unwrap().since;
        overflow.report(0, since + Duration::from_secs(9));
        let totals = || {
            let state = overflow.state.lock().unwrap();
            (state.written, state.distinct, state.bands.clone())
        };
        assert_eq!(totals(), (4, 2, vec![2, 2]));

        overflow.report(0, since + REPORT_INTERVAL);
        overflow.observe_sync(cap, &[a, empty, empty]);
        overflow.observe_write(cap, Some(a));
        assert_eq!(totals(), (1, 1, vec![1, 0]));
    }
}
