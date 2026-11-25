//! What each host copy of a title buffer costs and would save in video
//! memory, and the `placement:` line (diagnostics, with
//! `VICTORY_PLACEMENT_STATS=1`).
//!
//! Each host copy ([`crate::shadow`]) counts, on paths that already run
//! under the host's submission lock:
//! - the views draws bind of it, and their bytes (an upper bound of what
//!   shaders read);
//! - the bytes heap-indexed views reach, per list that runs shaders
//!   indexing the heap (counted once per heap and list, and spread over
//!   the heap's ranges at the fold, [`crate::bindless::fold_reach`]);
//! - the pages refreshes take and the bytes they copy in;
//! - the pages shaders mark written, the bytes of ranges they declare
//!   without marks, and the submissions that wrote it;
//! - the pages and bytes copied back into title memory;
//! - the bytes of what shaders wrote that the CPU reads back, and those of
//!   them a GPU copy brought from video memory.
//!
//! Its tracker counts the pages the title wrote ([`crate::watch`]). At the
//! first submission after each present the counts go into the copy's
//! window. Every 5 s the `placement:` line gives the totals by class, then
//! the copies that would save the most time per MiB of video memory
//! ([`gain_ms`]). Nothing acts on the score.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering::Relaxed};

use crate::vram::Class;

/// What a copy counts.
#[derive(Clone, Copy)]
pub(crate) enum N {
    /// Views draws bound, and their bytes.
    Binds,
    Bound,
    /// Bytes heap-indexed views reach.
    Heap,
    /// Submissions whose draws bound it.
    Uses,
    /// Bytes refreshes copied in, and the pages they took.
    Refreshed,
    Taken,
    /// Pages the title wrote (from its tracker).
    TitleWrote,
    /// Pages shaders marked written, bytes of ranges declared without
    /// marks, and submissions that wrote it either way.
    Marked,
    Declared,
    Writes,
    /// Pages and bytes copied back into title memory.
    BackPages,
    BackBytes,
    /// Bytes the CPU read of what shaders wrote, and bytes a GPU copy
    /// brought from video memory.
    ReadBack,
    Pulled,
}

/// How many counters a copy has.
pub(crate) const COUNTS: usize = N::Pulled as usize + 1;

/// A copy's counters.
#[derive(Default)]
pub(crate) struct Use {
    /// Counted since the last fold. Written only under the host's
    /// submission lock (a load and a store, no locked add): a racing
    /// writer could lose a count, never more.
    live: [AtomicU64; COUNTS],
    /// Folded since the last line.
    window: [AtomicU64; COUNTS],
    /// The submission that last used it, and that last wrote it.
    used_in: AtomicU64,
    written_in: AtomicU64,
}

/// Adds `by` to a counter only the submitting thread writes.
fn bump(a: &AtomicU64, by: u64) {
    a.store(a.load(Relaxed).wrapping_add(by), Relaxed);
}

/// `VICTORY_PLACEMENT_STATS`: 0 not read yet, 1 off, 2 on.
static ON: AtomicU8 = AtomicU8::new(0);

/// Whether copies count (`VICTORY_PLACEMENT_STATS=1`).
pub(crate) fn on() -> bool {
    match ON.load(Relaxed) {
        2 => true,
        1 => false,
        _ => {
            let on = crate::switch::on("VICTORY_PLACEMENT_STATS");
            ON.store(if on { 2 } else { 1 }, Relaxed);
            on
        }
    }
}

/// The submission being prepared (counted from 1, so a new copy's first
/// use counts).
static SUBMISSION: AtomicU64 = AtomicU64::new(1);
/// A present happened since the last fold, and folds since the last line.
static DUE: AtomicBool = AtomicBool::new(false);
static FOLDS: AtomicU64 = AtomicU64::new(0);
/// What copies gone since the last line counted, by class.
static GONE: [[AtomicU64; COUNTS]; 3] = [const { [const { AtomicU64::new(0) }; COUNTS] }; 3];

impl Use {
    /// Counts `by` in `n`.
    pub(crate) fn add(&self, n: N, by: u64) {
        if on() {
            bump(&self.live[n as usize], by);
        }
    }

    /// A draw binds `len` bytes of the copy.
    pub(crate) fn bound(&self, len: usize) {
        if on() {
            self.used();
            bump(&self.live[N::Binds as usize], 1);
            bump(&self.live[N::Bound as usize], len as u64);
        }
    }

    /// Heap-indexed views of lists folded now reached `len` bytes of the
    /// copy (at the fold).
    pub(crate) fn reached(&self, len: u64) {
        self.window[N::Heap as usize].fetch_add(len, Relaxed);
    }

    /// Shaders marked `by` pages (`N::Marked`) or declared `by` bytes with
    /// no marks (`N::Declared`) of the copy.
    pub(crate) fn wrote(&self, n: N, by: u64) {
        if on() {
            let now = SUBMISSION.load(Relaxed);
            if self.written_in.load(Relaxed) != now {
                self.written_in.store(now, Relaxed);
                bump(&self.live[N::Writes as usize], 1);
            }
            bump(&self.live[n as usize], by);
        }
    }

    /// The submission being prepared uses the copy.
    fn used(&self) {
        let now = SUBMISSION.load(Relaxed);
        if self.used_in.load(Relaxed) != now {
            self.used_in.store(now, Relaxed);
            bump(&self.live[N::Uses as usize], 1);
        }
    }

    /// Moves the counts into the window, with `title_wrote` pages the title
    /// wrote (under the submission lock).
    pub(crate) fn fold(&self, title_wrote: u64) {
        for (live, window) in self.live.iter().zip(&self.window) {
            let v = live.load(Relaxed);
            if v != 0 {
                live.store(0, Relaxed);
                window.fetch_add(v, Relaxed);
            }
        }
        if title_wrote != 0 {
            self.window[N::TitleWrote as usize].fetch_add(title_wrote, Relaxed);
        }
    }

    /// The window, emptied.
    fn take(&self) -> [u64; COUNTS] {
        std::array::from_fn(|i| self.window[i].swap(0, Relaxed))
    }

    /// The copy goes: what it counted stays in its class's totals.
    pub(crate) fn gone(&self, class: Class, title_wrote: u64) {
        if !on() {
            return;
        }
        self.fold(title_wrote);
        for (total, v) in GONE[class as usize].iter().zip(self.take()) {
            total.fetch_add(v, Relaxed);
        }
    }
}

/// A present happened: the next submission folds the counts.
pub(crate) fn presented() {
    if on() {
        DUE.store(true, Relaxed);
    }
}

/// At the start of a submission, under the submission lock: a new
/// submission number, and the fold after a present.
pub(crate) fn submission() {
    if !on() {
        return;
    }
    bump(&SUBMISSION, 1);
    if DUE.swap(false, Relaxed) {
        fold();
    }
}

/// Every copy's counts into its window.
fn fold() {
    FOLDS.fetch_add(1, Relaxed);
    crate::shadow::each_counted(|copy, _| copy.used.fold(copy.title_wrote()));
    crate::bindless::fold_reach();
}

/// A copy's counts per frame, as the score reads them (bytes or pages).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Rates {
    /// Bytes draws bound (R) and heap-indexed views reached (H).
    pub(crate) bound: f64,
    pub(crate) heap: f64,
    /// Bytes refreshes copied in (F).
    pub(crate) refreshed: f64,
    /// Bytes of shaders' writes the CPU reads back (B): what a copy in
    /// video memory brings back by a GPU copy first.
    pub(crate) back: f64,
    /// Pages shaders marked (W), bytes declared with no marks (Wd), and
    /// submissions that wrote the copy.
    pub(crate) marked: f64,
    pub(crate) declared: f64,
    pub(crate) writes: f64,
}

impl Rates {
    fn of(c: &[u64; COUNTS], frames: f64) -> Rates {
        let f = |n: N| c[n as usize] as f64 / frames;
        Rates {
            bound: f(N::Bound),
            heap: f(N::Heap),
            refreshed: f(N::Refreshed),
            back: f(N::ReadBack).max(f(N::Pulled)),
            marked: f(N::Marked),
            declared: f(N::Declared),
            writes: f(N::Writes),
        }
    }
}

/// Milliseconds per frame a copy would save with its GPU side in video
/// memory, from its counts per frame. Bytes over GB/s are ms at 1e6.
/// - Reads: draws' bytes and heap views' bytes (8 times: scattered reads,
///   damped as the views' reach is an upper bound) at 22.9 GB/s across
///   the bus against 700 GB/s in video memory (#2120, 5070 Ti).
/// - Read-write traffic: each marked page at 1.3 GB/s against 215 GB/s
///   (#2130, synthetic).
/// - CPU: refreshes write the system memory side at 28.7 GB/s and video
///   memory at 22.7 GB/s too; what shaders wrote comes back at 25 GB/s;
///   each submission that wrote the copy waits ~0.05 ms for that copy.
///
/// Page tables' reach is left out: a table names every copy there is.
pub(crate) fn gain_ms(r: &Rates) -> f64 {
    const MS: f64 = 1e6;
    let read = (r.bound + 8.0 * r.heap) * (1.0 / 22.9 - 1.0 / 700.0) / MS;
    let rw = r.marked * 65536.0 * (1.0 / 1.3 - 1.0 / 215.0) / MS;
    let cpu = r.refreshed / 22.7 / MS + r.back / 25.0 / MS + 0.05 * r.writes;
    read + rw - cpu
}

/// Why a copy could not go to video memory however it scores: under
/// 1 MiB (the per-copy costs dominate), 4 GiB or more, or writes coming
/// unmarked (5 % or more of what shaders write: each submission would
/// bring the declared ranges back).
fn barred(size: u64, r: &Rates) -> Option<&'static str> {
    let written = r.marked * 65536.0 + r.declared;
    if size < 1 << 20 {
        Some("under 1 MiB")
    } else if size >= 4 << 30 {
        Some("4 GiB or more")
    } else if written > 0.0 && r.declared / written >= 0.05 {
        Some("unmarked writes")
    } else {
        None
    }
}

/// A class's copies and their counts per frame.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Totals {
    pub(crate) copies: u64,
    pub(crate) bytes: u64,
    pub(crate) counts: [u64; COUNTS],
}

/// A copy in the line: its title range, video memory size, class, rates.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ranked {
    pub(crate) start: u64,
    pub(crate) len: u64,
    pub(crate) size: u64,
    pub(crate) class: Class,
    pub(crate) rates: Rates,
}

impl Ranked {
    /// Ms per frame saved per MiB of video memory.
    fn score(&self) -> f64 {
        gain_ms(&self.rates) / (self.size as f64 / f64::from(1 << 20))
    }
}

/// Copies in the line.
const TOP: usize = 20;

/// The `placement:` line over `frames` frames: totals by class, then up to
/// [`TOP`] copies by score.
pub(crate) fn line(frames: u64, classes: &[Totals; 3], mut ranked: Vec<Ranked>) -> String {
    use std::fmt::Write;
    let frames_f = frames.max(1) as f64;
    let mb = |b: f64| b / 1e6;
    let mut s = format!("placement: per frame over {frames} frames: ");
    for (i, (name, t)) in ["System", "VramRead", "VramWritten"]
        .iter()
        .zip(classes)
        .enumerate()
    {
        let c = |n: N| t.counts[n as usize] as f64 / frames_f;
        let _ = write!(
            s,
            "{}{name} {} copies {:.1} MB: bound {:.1} MB in {:.0} views by {:.0} submissions, \
             heap views {:.1} MB; title wrote {:.0} pages, took {:.0}, \
             refreshed {:.2} MB; shaders marked {:.0} pages, declared {:.2} MB unmarked, \
             wrote in {:.0} submissions; copied back {:.0} pages {:.2} MB; read back {:.2} MB, \
             {:.2} MB of it from video memory",
            if i > 0 { "; " } else { "" },
            t.copies,
            mb(t.bytes as f64),
            mb(c(N::Bound)),
            c(N::Binds),
            c(N::Uses),
            mb(c(N::Heap)),
            c(N::TitleWrote),
            c(N::Taken),
            mb(c(N::Refreshed)),
            c(N::Marked),
            mb(c(N::Declared)),
            c(N::Writes),
            c(N::BackPages),
            mb(c(N::BackBytes)),
            mb(c(N::ReadBack)),
            mb(c(N::Pulled)),
        );
    }
    ranked.sort_by(|a, b| b.score().total_cmp(&a.score()));
    ranked.truncate(TOP);
    let _ = write!(
        s,
        "; top {} copies by ms/frame saved per MiB of video memory (not acted on): ",
        ranked.len()
    );
    for (i, r) in ranked.iter().enumerate() {
        let v = &r.rates;
        let _ = write!(
            s,
            "{}{:#x}+{:#x} {:?} {:.1} MiB {:.4} ({:.3} ms; R {:.2} H {:.2} F {:.2} B {:.2} \
             Wd {:.2} MB, W {:.1} pages){}",
            if i > 0 { " | " } else { "" },
            r.start,
            r.len,
            r.class,
            r.size as f64 / f64::from(1 << 20),
            r.score(),
            gain_ms(v),
            mb(v.bound),
            mb(v.heap),
            mb(v.refreshed),
            mb(v.back),
            mb(v.declared),
            v.marked,
            barred(r.size, v).map_or(String::new(), |why| format!(" [{why}]")),
        );
    }
    s
}

/// Logs the `placement:` line, with every copy's window since the last
/// (every 5 s, with the stats).
pub(crate) fn log() {
    if !on() {
        return;
    }
    let frames = FOLDS.swap(0, Relaxed);
    if frames == 0 {
        return;
    }
    let mut classes = [Totals::default(); 3];
    let mut ranked = Vec::new();
    crate::shadow::each_counted(|copy, loose| {
        let (start, len) = copy.range();
        let (class, size) = (copy.class(), len.next_multiple_of(1 << 16));
        let counts = copy.used.take();
        let t = &mut classes[class as usize];
        t.copies += 1;
        t.bytes += size;
        t.counts.iter_mut().zip(counts).for_each(|(t, v)| *t += v);
        // Loose windows stay in system memory, whatever they score.
        if !loose && counts.iter().any(|&v| v != 0) {
            let rates = Rates::of(&counts, frames as f64);
            ranked.push(Ranked {
                start,
                len,
                size,
                class,
                rates,
            });
        }
    });
    for (t, gone) in classes.iter_mut().zip(&GONE) {
        t.counts
            .iter_mut()
            .zip(gone)
            .for_each(|(t, v)| *t += v.swap(0, Relaxed));
    }
    tracing::info!("{}", line(frames, &classes, ranked));
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Copies count from now on in this process (or not), whatever the
    /// switch says.
    pub(crate) fn count(on: bool) {
        ON.store(if on { 2 } else { 1 }, Relaxed);
    }

    /// Every copy's counts into its window, as after a present.
    pub(crate) fn fold_all() {
        fold();
    }

    /// The copy's window after a fold now, emptied.
    pub(crate) fn folded(copy: &crate::shadow::Shadow) -> [u64; COUNTS] {
        copy.used.fold(copy.title_wrote());
        copy.used.take()
    }

    /// The score: reads across the bus saved, read-write traffic saved,
    /// less the CPU's extra work; per MiB of video memory.
    #[test]
    fn the_score_is_ms_saved_per_mib_of_video_memory() {
        const MIB: f64 = (1 << 20) as f64;
        let read = Rates {
            bound: 100.0 * MIB,
            ..Rates::default()
        };
        let gain = gain_ms(&read);
        // 100 MiB across the bus at 22.9 GB/s, against 700 GB/s.
        assert!((gain - 4.429).abs() < 0.01, "{gain}");
        let r = Ranked {
            start: 0,
            len: 100 << 20,
            size: 100 << 20,
            class: Class::System,
            rates: read,
        };
        assert!((r.score() - 0.04429).abs() < 1e-4, "{}", r.score());
        let heap = Rates {
            heap: MIB,
            ..Rates::default()
        };
        assert!(
            (gain_ms(&heap)
                - 8.0
                    * gain_ms(&Rates {
                        bound: MIB,
                        ..Rates::default()
                    }))
            .abs()
                < 1e-9
        );
        // Refreshed every frame and never read: a loss.
        let written = Rates {
            refreshed: 10.0 * MIB,
            ..Rates::default()
        };
        assert!(gain_ms(&written) < 0.0);
        // One marked page a frame about pays for its submission's wait;
        // four do.
        let marked = |pages: f64| Rates {
            marked: pages,
            writes: 1.0,
            back: pages * 65536.0,
            ..Rates::default()
        };
        assert!(gain_ms(&marked(1.0)).abs() < 0.01);
        assert!(gain_ms(&marked(4.0)) > 0.1);
        assert_eq!(barred(1 << 19, &read), Some("under 1 MiB"));
        assert_eq!(barred(4 << 30, &read), Some("4 GiB or more"));
        let snap = Rates {
            marked: 1.0,
            declared: 65536.0,
            ..Rates::default()
        };
        assert_eq!(barred(1 << 20, &snap), Some("unmarked writes"));
        assert_eq!(barred(1 << 20, &read), None);
    }

    /// The line: totals by class, per frame, then copies by score.
    #[test]
    fn the_placement_line_formats() {
        let mut system = Totals {
            copies: 2,
            bytes: 3 << 20,
            ..Totals::default()
        };
        system.counts[N::Bound as usize] = 4_000_000;
        system.counts[N::Binds as usize] = 8;
        system.counts[N::TitleWrote as usize] = 6;
        system.counts[N::Marked as usize] = 2;
        system.counts[N::BackPages as usize] = 2;
        system.counts[N::BackBytes as usize] = 2 << 16;
        let read = Totals {
            copies: 1,
            bytes: 1 << 20,
            ..Totals::default()
        };
        let classes = [system, read, Totals::default()];
        let mib = f64::from(1 << 20);
        let ranked = vec![
            Ranked {
                start: 0x1000_0000,
                len: 0x10_0000,
                size: 1 << 20,
                class: Class::System,
                rates: Rates {
                    bound: mib,
                    ..Rates::default()
                },
            },
            Ranked {
                start: 0x2000_0000,
                len: 0x8000,
                size: 1 << 16,
                class: Class::VramRead,
                rates: Rates {
                    bound: 4.0 * mib,
                    ..Rates::default()
                },
            },
        ];
        let s = line(2, &classes, ranked);
        assert!(
            s.starts_with("placement: per frame over 2 frames: System 2 copies 3.1 MB: "),
            "{s}"
        );
        assert!(s.contains("bound 2.0 MB in 4 views"), "{s}");
        assert!(s.contains("title wrote 3 pages"), "{s}");
        assert!(s.contains("shaders marked 1 pages"), "{s}");
        assert!(s.contains("copied back 1 pages 0.07 MB"), "{s}");
        assert!(s.contains("; VramRead 1 copies 1.0 MB: "), "{s}");
        assert!(s.contains("; VramWritten 0 copies 0.0 MB: "), "{s}");
        // The small copy saves more per MiB, and is barred for its size.
        let top = s.split("(not acted on): ").nth(1).expect("the top copies");
        let entries: Vec<&str> = top.split(" | ").collect();
        assert_eq!(entries.len(), 2, "{s}");
        assert!(
            entries[0].starts_with("0x20000000+0x8000 VramRead 0.1 MiB 2.8346 (0.177 ms; R 4.19"),
            "{s}"
        );
        assert!(entries[0].ends_with(" [under 1 MiB]"), "{s}");
        assert!(
            entries[1].starts_with("0x10000000+0x100000 System 1.0 MiB 0.0443 "),
            "{s}"
        );
        assert!(!entries[1].contains('['), "{s}");
        let many = vec![
            Ranked {
                start: 0,
                len: 1,
                size: 1 << 20,
                class: Class::System,
                rates: Rates::default()
            };
            30
        ];
        let s = line(1, &classes, many);
        assert!(s.contains("top 20 copies"), "{s}");
        assert_eq!(s.matches(" | ").count(), 19, "{s}");
    }
}
