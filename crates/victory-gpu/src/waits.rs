//! Diagnostics: time threads spend waiting for locks that recording threads
//! and the submitting thread share (the view registry, a heap's kept
//! resources, the retired resources), split by the thread that waited.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

/// The locks timed.
#[derive(Clone, Copy)]
pub(crate) enum Lock {
    Views = 0,
    Kept = 1,
    Retired = 2,
}

/// Nanoseconds waited, by [submitting thread, other threads] and [`Lock`].
static WAITS: [[AtomicU64; 3]; 2] = [const { [const { AtomicU64::new(0) }; 3] }; 2];

thread_local! {
    /// The thread is inside a host submission (`Host::execute`).
    static SUBMITTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` as the submitting thread.
pub(crate) fn submitting<R>(f: impl FnOnce() -> R) -> R {
    let was = SUBMITTING.with(|s| s.replace(true));
    let r = f();
    SUBMITTING.with(|s| s.set(was));
    r
}

/// Locks `m` (poisoned or not), timing the wait in diagnostics builds.
pub(crate) fn lock<T>(m: &Mutex<T>, which: Lock) -> MutexGuard<'_, T> {
    #[cfg(feature = "diagnostics")]
    {
        match m.try_lock() {
            Ok(guard) => return guard,
            Err(std::sync::TryLockError::Poisoned(e)) => return e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {}
        }
        let started = std::time::Instant::now();
        let guard = m.lock().unwrap_or_else(|e| e.into_inner());
        let thread = usize::from(!SUBMITTING.with(std::cell::Cell::get));
        WAITS[thread][which as usize].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        guard
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = which;
        m.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Nanoseconds waited since the last call: [submitting, others][lock].
pub(crate) fn take() -> [[u64; 3]; 2] {
    WAITS.each_ref().map(|t| t.each_ref().map(|a| a.swap(0, Ordering::Relaxed)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wait for a held lock is counted for the thread that waited.
    #[test]
    fn a_wait_for_a_held_lock_is_counted() {
        let m = std::sync::Arc::new(Mutex::new(0));
        take();
        let held = m.lock().unwrap();
        let other = {
            let m = m.clone();
            std::thread::spawn(move || {
                *lock(&m, Lock::Kept) += 1;
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(20));
        drop(held);
        other.join().unwrap();
        assert_eq!(*m.lock().unwrap(), 1);
        let waits = take();
        #[cfg(feature = "diagnostics")]
        assert!(waits[1][Lock::Kept as usize] >= 10_000_000, "{waits:?}");
        assert_eq!(waits[0], [0; 3], "not the submitting thread");
    }
}
