//! Submissions the host GPU may still be running, oldest first, and what is
//! done once each has run: its work (copying what shaders wrote back into
//! title memory, then the bottom-of-pipe writes that tell the title the GPU
//! is done), then the title's fence signals queued behind it. Finished in
//! submission order, one thread at a time.
//!
//! Each submission also names the title memory its GPU work writes (copied
//! back when it finishes) and reads (host copies brought up to date for
//! it), so that a later submission's CPU-side steps can find the earlier
//! ones they must wait for ([`Timeline::conflict`]).

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, MutexGuard};

/// Title memory ranges `[from, to)`, sorted and merged.
#[derive(Default)]
pub(crate) struct Ranges(Vec<(usize, usize)>);

impl Ranges {
    pub(crate) fn new(mut ranges: Vec<(usize, usize)>) -> Self {
        ranges.retain(|&(a, b)| a < b);
        ranges.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
        for (a, b) in ranges {
            match merged.last_mut() {
                Some(last) if a <= last.1 => last.1 = last.1.max(b),
                _ => merged.push((a, b)),
            }
        }
        Ranges(merged)
    }

    /// Whether `[from, to)` overlaps one of the ranges.
    pub(crate) fn overlaps(&self, from: usize, to: usize) -> bool {
        let i = self.0.partition_point(|&(_, b)| b <= from);
        self.0.get(i).is_some_and(|&(a, _)| a < to)
    }
}

type Signal = Box<dyn FnOnce() + Send>;

/// The title memory a submission's GPU work touches.
#[derive(Default)]
pub(crate) struct Touches {
    /// What its shaders write: the host copy (by address) and the range.
    pub(crate) shaders: Vec<(usize, usize, usize)>,
    /// What its bottom-of-pipe writes write.
    pub(crate) labels: Vec<(usize, usize)>,
    /// What it reads (host copies and textures brought up to date for it).
    pub(crate) reads: Vec<(usize, usize)>,
    /// Shared texture upload buffers that the GPU may still read.
    pub(crate) staging: std::collections::HashSet<usize>,
    /// Title-memory page tables (by view heap key) its shaders read.
    pub(crate) tables: std::collections::HashSet<usize>,
}

struct Entry<W> {
    /// The host fence value the submission signals.
    value: u64,
    /// Everything it writes, shaders' and labels'.
    writes: Ranges,
    labels: Ranges,
    shaders: Vec<(usize, usize, usize)>,
    reads: Ranges,
    staging: std::collections::HashSet<usize>,
    tables: std::collections::HashSet<usize>,
    /// Title fence signals the title queued after it.
    signals: Vec<Signal>,
    /// Taken while it is being finished.
    work: Option<W>,
}

pub(crate) struct Timeline<W> {
    entries: Mutex<VecDeque<Entry<W>>>,
    added: Condvar,
    /// Held while submissions are finished: in order, by one thread.
    finishing: Mutex<()>,
}

impl<W> Timeline<W> {
    pub(crate) const fn new() -> Self {
        Timeline {
            entries: Mutex::new(VecDeque::new()),
            added: Condvar::new(),
            finishing: Mutex::new(()),
        }
    }

    fn entries(&self) -> MutexGuard<'_, VecDeque<Entry<W>>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A submission that signals `value` (higher than any before it), the
    /// title memory its GPU work touches, and its work once it ran.
    pub(crate) fn push(&self, value: u64, touches: Touches, work: W) {
        let Touches {
            shaders,
            labels,
            reads,
            staging,
            tables,
        } = touches;
        let writes = shaders
            .iter()
            .map(|&(_, a, b)| (a, b))
            .chain(labels.iter().copied());
        self.entries().push_back(Entry {
            value,
            writes: Ranges::new(writes.collect()),
            labels: Ranges::new(labels),
            shaders,
            reads: Ranges::new(reads),
            staging,
            tables,
            signals: Vec::new(),
            work: Some(work),
        });
        self.added.notify_all();
    }

    /// Submissions not finished yet.
    pub(crate) fn len(&self) -> usize {
        self.entries().len()
    }

    /// The values of the oldest and newest submissions not finished yet.
    pub(crate) fn span(&self) -> Option<(u64, u64)> {
        let entries = self.entries();
        Some((entries.front()?.value, entries.back()?.value))
    }

    /// Runs `signal` once every submission pushed so far has finished: now
    /// if none is left.
    pub(crate) fn after(&self, signal: impl FnOnce() + Send + 'static) {
        let mut entries = self.entries();
        match entries.back_mut() {
            Some(last) => last.signals.push(Box::new(signal)),
            // Under the lock, so it stays behind signals still running.
            None => signal(),
        }
    }

    /// The newest submission not finished yet whose GPU work writes
    /// `[from, to)`, or (with `writing`, for a step that writes it) reads
    /// it.
    pub(crate) fn conflict(&self, from: usize, to: usize, writing: bool) -> Option<u64> {
        if from >= to {
            return None;
        }
        self.entries()
            .iter()
            .rev()
            .find(|e| e.writes.overlaps(from, to) || writing && e.reads.overlaps(from, to))
            .map(|e| e.value)
    }

    /// The most recent submission still reading this host staging allocation.
    pub(crate) fn staging_conflict(&self, staging: usize) -> Option<u64> {
        self.entries()
            .iter()
            .rev()
            .find(|e| e.staging.contains(&staging))
            .map(|e| e.value)
    }

    /// The most recent submission still reading view heap `key`'s page
    /// table.
    pub(crate) fn table_conflict(&self, key: usize) -> Option<u64> {
        self.entries()
            .iter()
            .rev()
            .find(|e| e.tables.contains(&key))
            .map(|e| e.value)
    }

    /// For a refresh of `[from, to)` through the host copy `copy`: the
    /// newest submission not finished yet it must wait for. One whose
    /// bottom-of-pipe writes overlap the range counts, and so does one whose
    /// shaders write it through another copy; one whose shaders write it
    /// through this copy counts only if `stale` says the title wrote some of
    /// the overlap since the copy took it. Otherwise the copy already holds
    /// what the refresh would bring, and those shaders' writes reach it in
    /// queue order.
    pub(crate) fn refresh_conflict(
        &self,
        copy: usize,
        from: usize,
        to: usize,
        stale: impl Fn(usize, usize) -> bool,
    ) -> Option<u64> {
        if from >= to {
            return None;
        }
        let (mut must, mut parts) = (None, Vec::new());
        for e in self.entries().iter() {
            if e.labels.overlaps(from, to) {
                must = Some(e.value);
            }
            for &(c, a, b) in e.shaders.iter().filter(|&&(_, a, b)| a < to && b > from) {
                if c == copy {
                    parts.push((e.value, a.max(from), b.min(to)));
                } else {
                    must = Some(e.value);
                }
            }
        }
        // Asked outside the lock, newest first.
        parts.sort_unstable_by_key(|p| std::cmp::Reverse(p.0));
        parts
            .into_iter()
            .take_while(|&(v, ..)| must.is_none_or(|m| v > m))
            .find(|&(_, a, b)| stale(a, b))
            .map(|(v, ..)| v)
            .or(must)
    }

    /// Submissions still in flight whose shaders may write `[from, to)` of
    /// host copy `copy`: `(fence_value, overlap_from, overlap_to)`.
    pub(crate) fn in_flight_writers(
        &self,
        copy: usize,
        from: usize,
        to: usize,
    ) -> Vec<(u64, usize, usize)> {
        if from >= to {
            return Vec::new();
        }
        self.entries()
            .iter()
            .flat_map(|e| {
                e.shaders
                    .iter()
                    .filter(move |&&(c, a, b)| c == copy && a < to && b > from)
                    .map(move |&(_, a, b)| (e.value, a.max(from), b.min(to)))
            })
            .collect()
    }

    /// Whether any submission still in flight behind the oldest one (the
    /// one currently completing) reads title memory overlapping `[from, to)`.
    pub(crate) fn has_subsequent_reader(&self, from: usize, to: usize) -> bool {
        if from >= to {
            return false;
        }
        self.entries()
            .iter()
            .skip(1)
            .any(|e| e.reads.overlaps(from, to))
    }

    /// Blocks until a submission is left; its value.
    pub(crate) fn next(&self) -> u64 {
        let mut entries = self.entries();
        loop {
            if let Some(first) = entries.front() {
                return first.value;
            }
            entries = self.added.wait(entries).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Finishes, oldest first, the submissions whose values are at most
    /// `reached`: `finish` takes each one's work, then its signals run.
    /// Returns once those are all finished, by this thread or another.
    pub(crate) fn finish(&self, reached: u64, mut finish: impl FnMut(W)) {
        let _finishing = self.finishing.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            let work = {
                let mut entries = self.entries();
                match entries.front_mut() {
                    Some(first) if first.value <= reached => first.work.take(),
                    _ => return,
                }
            };
            if let Some(work) = work {
                finish(work);
            }
            let mut entries = self.entries();
            if let Some(done) = entries.pop_front() {
                // Under the lock: a signal queued now goes behind these.
                done.signals.into_iter().for_each(|s| s());
            }
        }
    }

    /// Calls `f` with the work of the submissions not finished yet (not
    /// those being finished).
    pub(crate) fn each_work(&self, mut f: impl FnMut(&W)) {
        self.entries()
            .iter()
            .filter_map(|e| e.work.as_ref())
            .for_each(&mut f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    type Log = Arc<Mutex<Vec<String>>>;

    fn note(log: &Log, what: &str) {
        log.lock().unwrap().push(what.to_owned());
    }

    #[test]
    fn submissions_finish_in_order_and_signals_follow_their_work() {
        let log: Log = Arc::default();
        let t: Timeline<&str> = Timeline::new();
        // Nothing in flight: a signal is at once.
        let l = log.clone();
        t.after(move || note(&l, "signal 0"));
        t.push(1, Touches::default(), "work 1");
        t.push(2, Touches::default(), "work 2");
        // Queued after both: behind the second's work.
        let l = log.clone();
        t.after(move || note(&l, "signal 2"));
        let finish = |w: &str| note(&log, w);
        t.finish(0, finish);
        assert_eq!(*log.lock().unwrap(), ["signal 0"]);
        t.finish(1, finish);
        assert_eq!(*log.lock().unwrap(), ["signal 0", "work 1"]);
        assert_eq!(t.span(), Some((2, 2)));
        t.finish(5, finish);
        assert_eq!(
            *log.lock().unwrap(),
            ["signal 0", "work 1", "work 2", "signal 2"]
        );
        assert_eq!(t.len(), 0);
        // Finished again: nothing more.
        t.finish(5, finish);
        assert_eq!(log.lock().unwrap().len(), 4);
    }

    #[test]
    fn steps_conflict_with_the_newest_submission_touching_their_memory() {
        let t: Timeline<()> = Timeline::new();
        // Writes 0x1000..0x2000 and a 4-byte word at 0x9000; reads 0x5000..0x6000.
        let touches = |shaders: Vec<(usize, usize, usize)>, labels, reads| Touches {
            shaders,
            labels,
            reads,
            ..Touches::default()
        };
        t.push(
            1,
            touches(
                vec![(1, 0x1000, 0x2000)],
                vec![(0x9000, 0x9004)],
                vec![(0x5000, 0x6000)],
            ),
            (),
        );
        t.push(2, touches(vec![(1, 0x1800, 0x1900)], vec![], vec![]), ());
        // Reading what either writes waits for the newest that does.
        assert_eq!(t.conflict(0x1880, 0x1890, false), Some(2));
        assert_eq!(t.conflict(0x1000, 0x1010, false), Some(1));
        assert_eq!(t.conflict(0x9002, 0x9003, false), Some(1));
        // Their GPU reads matter only to steps that write.
        assert_eq!(t.conflict(0x5000, 0x5001, false), None);
        assert_eq!(t.conflict(0x5fff, 0x7000, true), Some(1));
        // Next to, not in.
        assert_eq!(t.conflict(0x2000, 0x5000, true), None);
        assert_eq!(t.conflict(0x9004, 0x9008, true), None);
        // A refresh through copy 1 waits only where the title wrote the
        // overlap; labels and other copies always count.
        let clean = |_: usize, _: usize| false;
        let dirty_at = |at: usize| move |a: usize, b: usize| (a..b).contains(&at);
        assert_eq!(t.refresh_conflict(1, 0x1000, 0x2000, clean), None);
        assert_eq!(
            t.refresh_conflict(1, 0x1000, 0x2000, dirty_at(0x1100)),
            Some(1)
        );
        assert_eq!(
            t.refresh_conflict(1, 0x1000, 0x2000, dirty_at(0x1880)),
            Some(2)
        );
        assert_eq!(t.refresh_conflict(2, 0x1000, 0x1010, clean), Some(1));
        assert_eq!(t.refresh_conflict(1, 0x8000, 0x9001, clean), Some(1));
        t.finish(2, |()| {});
        assert_eq!(t.conflict(0x1000, 0x2000, true), None);

        // Shared texture staging buffers track the newest in-flight reader.
        let staging_a = 0xDEAD_0000usize;
        let staging_b = 0xDEAD_1000usize;
        let mut t1 = Touches::default();
        t1.staging.insert(staging_a);
        let mut t2 = Touches::default();
        t2.staging.insert(staging_a);
        t2.staging.insert(staging_b);
        t.push(3, t1, ());
        t.push(4, t2, ());
        assert_eq!(t.staging_conflict(staging_a), Some(4));
        assert_eq!(t.staging_conflict(staging_b), Some(4));
        assert_eq!(t.staging_conflict(0xDEAD_2000), None);
        t.finish(4, |()| {});
        assert_eq!(t.staging_conflict(staging_a), None);
    }
}
