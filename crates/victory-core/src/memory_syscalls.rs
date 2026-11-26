//! Process-wide Windows memory call counters for the GPU frame report.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub enum Call {
    GetWriteWatch,
    ResetWriteWatch,
    VirtualQuery,
    VirtualQueryEx,
    VirtualProtect,
    ReadProcessMemory,
    NtWriteVirtualMemory,
    WriteProcessMemory,
}

// Keep the reset and write APIs in the report even when the copy path does
// not call them.
const CALLS: [Call; 8] = [
    Call::GetWriteWatch,
    Call::ResetWriteWatch,
    Call::VirtualQuery,
    Call::VirtualQueryEx,
    Call::VirtualProtect,
    Call::ReadProcessMemory,
    Call::NtWriteVirtualMemory,
    Call::WriteProcessMemory,
];
static COUNTERS: [Counter; CALLS.len()] = [const { Counter::new() }; CALLS.len()];

impl Call {
    /// Count an attempted call and the requested range size, if applicable.
    #[inline]
    pub fn record(self, bytes: usize) {
        COUNTERS[self as usize].record(bytes);
    }

    /// Count an attempted call with elapsed time and a caller-provided byte count.
    pub fn record_timed(self, bytes: usize, started: Instant) {
        let ns = started.elapsed().as_nanos() as u64;
        let counter = &COUNTERS[self as usize];
        counter.record(bytes);
        counter.ns.fetch_add(ns, Relaxed);
    }
}

struct Counter {
    calls: AtomicU64,
    bytes: AtomicU64,
    ns: AtomicU64,
}

impl Counter {
    const fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            ns: AtomicU64::new(0),
        }
    }

    fn record(&self, bytes: usize) {
        self.calls.fetch_add(1, Relaxed);
        if bytes != 0 {
            self.bytes.fetch_add(bytes as u64, Relaxed);
        }
    }

    fn take(&self) -> Sample {
        Sample {
            calls: self.calls.swap(0, Relaxed),
            bytes: self.bytes.swap(0, Relaxed),
            ns: self.ns.swap(0, Relaxed),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Sample {
    calls: u64,
    bytes: u64,
    ns: u64,
}

/// Drain the counters for this frame window. Calls racing with a drain can
/// contribute their byte and time totals to an adjacent window.
pub fn take(frames: f64) -> impl fmt::Display {
    Report {
        frames,
        samples: CALLS.map(|call| COUNTERS[call as usize].take()),
    }
}

struct Report {
    frames: f64,
    samples: [Sample; CALLS.len()],
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let frames = self.frames;
        for (i, (call, sample)) in CALLS.into_iter().zip(&self.samples).enumerate() {
            if i != 0 {
                f.write_str("; ")?;
            }
            write!(f, "{call:?} {:.2}", sample.calls as f64 / frames)?;
            match call {
                Call::GetWriteWatch | Call::VirtualProtect => {
                    write!(f, " ({:.0} B requested", sample.bytes as f64 / frames)?;
                }
                Call::ReadProcessMemory => {
                    write!(f, " ({:.0} B transferred", sample.bytes as f64 / frames)?;
                }
                _ => continue,
            }
            if matches!(call, Call::GetWriteWatch | Call::ReadProcessMemory) {
                write!(f, ", {:.3} ms", sample.ns as f64 / frames / 1e6)?;
            }
            f.write_str(")")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_drains_per_frame_counts() {
        let counters = [const { Counter::new() }; CALLS.len()];
        let read = &counters[Call::ReadProcessMemory as usize];
        read.record(4096);
        read.record(0);
        read.ns.fetch_add(4_000_000, Relaxed);
        counters[Call::VirtualQuery as usize].record(0);

        let report = Report {
            frames: 2.0,
            samples: counters.each_ref().map(Counter::take),
        };
        assert_eq!(
            report.to_string(),
            "GetWriteWatch 0.00 (0 B requested, 0.000 ms); ResetWriteWatch 0.00; \
             VirtualQuery 0.50; VirtualQueryEx 0.00; VirtualProtect 0.00 (0 B requested); \
             ReadProcessMemory 1.00 (2048 B transferred, 2.000 ms); \
             NtWriteVirtualMemory 0.00; WriteProcessMemory 0.00"
        );
        assert_eq!(
            read.take(),
            Sample {
                calls: 0,
                bytes: 0,
                ns: 0
            }
        );
    }
}
