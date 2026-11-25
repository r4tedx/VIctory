//! The video memory budget the system gives this process (DXGI's
//! `QueryVideoMemoryInfo`), for VIctory's own pools to size themselves from.
//!
//! Past its budget, the system's video memory manager demotes the
//! process's allocations to system memory. It picks them by residency
//! priority (`crate::host::keep_resident`), not by what the next frame
//! needs, so a pool that grows into the budget can push out the title's
//! render targets: frames then take seconds. A pool that has somewhere
//! else to put its bytes (host copies in GPU upload heaps fall back to
//! system memory) asks [`Budget::limit`] how much it may hold now instead
//! of using a fixed size, and gives way when the title's own textures need
//! the room.
//!
//! DXGI is asked at most every [`MAX_AGE`], and again as soon as the
//! system signals a budget change, so asking is cheap. Bytes pools take
//! between readings ([`Budget::charge`]) count as used until the next one.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// How old a reading may be before the next ask reads again.
pub(crate) const MAX_AGE: Duration = Duration::from_millis(250);

/// A segment group's numbers, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Segment {
    /// What the process may use before the system starts demoting.
    pub(crate) budget: u64,
    /// What it uses now.
    pub(crate) usage: u64,
}

impl Segment {
    /// Whether use is past nine tenths of the budget.
    pub(crate) fn nearly_full(self) -> bool {
        self.usage > self.budget / 10 * 9
    }
}

/// Video memory on the card (local) and the system memory the GPU reads
/// across the bus (non-local).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Reading {
    pub(crate) local: Segment,
    pub(crate) non_local: Segment,
}

/// Where readings come from: DXGI for the host's adapter, or a test's
/// numbers.
pub(crate) trait Source: Send {
    /// The numbers now; `None` if the adapter cannot say.
    fn read(&mut self) -> Option<Reading>;

    /// Whether the system signalled a budget change since the last call.
    fn changed(&mut self) -> bool {
        false
    }
}

/// A pool's share of the local budget: what it may hold, in all, at a
/// time.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pool {
    /// The part (numerator, denominator) of what the rest of the process
    /// leaves of the budget.
    pub(crate) part: (u64, u64),
    /// At most this, or [`Pool::large`] of the budget if that is more;
    /// all of it without a reading (the fixed size the pool had before).
    pub(crate) ceiling: u64,
    /// The part of the whole budget the ceiling grows to on large cards.
    pub(crate) large: (u64, u64),
}

/// The budget, as last read.
pub(crate) struct Budget {
    state: Mutex<State>,
    max_age: Duration,
}

struct State {
    source: Box<dyn Source>,
    /// When it was last read, and what that gave.
    last: Option<(Instant, Option<Reading>)>,
    /// Bytes pools took in video memory since `last` was read, which its
    /// usage does not have yet.
    charged: u64,
    /// Times use was found past nine tenths of the local budget and logged
    /// (once a run).
    warned: u32,
}

impl Budget {
    pub(crate) fn new(source: Box<dyn Source>, max_age: Duration) -> Self {
        Budget {
            state: Mutex::new(State {
                source,
                last: None,
                charged: 0,
                warned: 0,
            }),
            max_age,
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The numbers, read again if the last reading is older than the
    /// maximum age or the budget changed since; with the bytes charged
    /// since the reading added to local use.
    pub(crate) fn reading(&self) -> Option<Reading> {
        let mut s = self.state();
        let stale = match s.last {
            None => true,
            Some((at, _)) => at.elapsed() >= self.max_age,
        };
        // Asked every time, so a signal is taken even while fresh.
        if s.source.changed() || stale {
            let reading = s.source.read();
            s.last = Some((Instant::now(), reading));
            s.charged = 0;
            if let Some(r) = reading
                && r.local.nearly_full()
                && s.warned == 0
            {
                s.warned += 1;
                warn(r);
            }
        }
        let charged = s.charged;
        let (_, reading) = s.last?;
        reading.map(|mut r| {
            r.local.usage += charged;
            r
        })
    }

    /// Bytes a pool just took in video memory: used, as far as this budget
    /// knows, until the next reading has them.
    pub(crate) fn charge(&self, bytes: u64) {
        self.state().charged += bytes;
    }

    /// Bytes `pool` may hold now, given it holds `own` (counted in the
    /// reading's use): its part of what the rest of the process leaves of
    /// the local budget, up to its ceiling. Use by others shrinks it; its
    /// own bytes do not. The ceiling alone without a reading.
    pub(crate) fn limit(&self, pool: &Pool, own: u64) -> u64 {
        let Some(r) = self.reading() else {
            return pool.ceiling;
        };
        let others = r.local.usage.saturating_sub(own);
        let left = r.local.budget.saturating_sub(others);
        let share =
            (u128::from(left) * u128::from(pool.part.0) / u128::from(pool.part.1.max(1))) as u64;
        let large = (u128::from(r.local.budget) * u128::from(pool.large.0)
            / u128::from(pool.large.1.max(1))) as u64;
        share.min(pool.ceiling.max(large))
    }

    /// Times use was logged as past nine tenths of the budget.
    #[cfg(test)]
    fn warned(&self) -> u32 {
        self.state().warned
    }
}

/// Logs, once a run, that use passed nine tenths of the local budget, with
/// what VIctory knows takes it.
fn warn(r: Reading) {
    let mb = |b: u64| b as f64 / 1e6;
    let (textures, texture_bytes, used, ..) = crate::resource::texture_counts();
    let copies = crate::shadow::copy_bytes();
    let upload = crate::host::upload_bytes();
    tracing::warn!(
        "video memory in use ({:.0} MB) is past nine tenths of the budget the system gives this process ({:.0} MB): past it, the system moves allocations to system memory and frames slow down. \
         Host textures: {textures}, {:.0} MB ({:.0} MB of them used by lists); host copies of title memory {:.0} MB, {:.0} MB of them in video memory (GPU upload heaps). \
         System memory the GPU reads: {:.0} of {:.0} MB",
        mb(r.local.usage),
        mb(r.local.budget),
        mb(texture_bytes),
        mb(used),
        mb(copies),
        mb(upload),
        mb(r.non_local.usage),
        mb(r.non_local.budget),
    );
}

/// The budget of the host device's adapter; `None` without a host device
/// or an adapter that reports one.
pub(crate) fn get() -> Option<&'static Budget> {
    static BUDGET: OnceLock<Option<Budget>> = OnceLock::new();
    BUDGET
        .get_or_init(|| {
            let source = Dxgi::new()?;
            Some(Budget::new(Box::new(source), MAX_AGE))
        })
        .as_ref()
}

/// Reads the host's budget if the last reading is old: logs use past nine
/// tenths of it (once a run), whether or not a pool asks. Called with each
/// frame-rate line.
pub(crate) fn watch() {
    if let Some(b) = get() {
        b.reading();
    }
}

/// DXGI's numbers for the host device's adapter, and the event the system
/// signals when the budget changes.
struct Dxgi {
    adapter: windows::Win32::Graphics::Dxgi::IDXGIAdapter3,
    event: Option<windows::Win32::Foundation::HANDLE>,
}

// SAFETY: DXGI adapters are free-threaded; the event is only waited on.
unsafe impl Send for Dxgi {}

impl Dxgi {
    fn new() -> Option<Self> {
        use windows::Win32::Graphics::Dxgi::*;
        let host = crate::host::host()?;
        // SAFETY: plain DXGI queries and event creation.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let adapter: IDXGIAdapter3 = factory
                .EnumAdapterByLuid(host.device.GetAdapterLuid())
                .ok()?;
            // Auto-reset: one signal, one read. Registered for the run.
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None)
                .ok()
                .filter(|&e| {
                    adapter
                        .RegisterVideoMemoryBudgetChangeNotificationEvent(e)
                        .is_ok()
                });
            Some(Dxgi { adapter, event })
        }
    }
}

impl Source for Dxgi {
    fn read(&mut self) -> Option<Reading> {
        use windows::Win32::Graphics::Dxgi::*;
        let query = |group| {
            let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            // SAFETY: a live adapter and a structure of ours.
            unsafe { self.adapter.QueryVideoMemoryInfo(0, group, &mut info) }.ok()?;
            Some(Segment {
                budget: info.Budget,
                usage: info.CurrentUsage,
            })
        };
        Some(Reading {
            local: query(DXGI_MEMORY_SEGMENT_GROUP_LOCAL)?,
            non_local: query(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)?,
        })
    }

    fn changed(&mut self) -> bool {
        use windows::Win32::Foundation::WAIT_OBJECT_0;
        use windows::Win32::System::Threading::WaitForSingleObject;
        // SAFETY: our event, polled.
        self.event
            .is_some_and(|e| unsafe { WaitForSingleObject(e, 0) } == WAIT_OBJECT_0)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    /// Numbers a test sets, the reads made and a budget-change signal.
    #[derive(Clone, Default)]
    pub(crate) struct Fake {
        pub(crate) now: Arc<Mutex<Option<Reading>>>,
        pub(crate) reads: Arc<AtomicU32>,
        pub(crate) signal: Arc<AtomicBool>,
    }

    impl Fake {
        pub(crate) fn set(&self, budget: u64, usage: u64) {
            *self.now.lock().unwrap() = Some(Reading {
                local: Segment { budget, usage },
                non_local: Segment::default(),
            });
        }
    }

    impl Source for Fake {
        fn read(&mut self) -> Option<Reading> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            *self.now.lock().unwrap()
        }

        fn changed(&mut self) -> bool {
            self.signal.swap(false, Ordering::Relaxed)
        }
    }

    /// A budget read on every ask, over `fake`'s numbers.
    pub(crate) fn always_read(fake: &Fake) -> Budget {
        Budget::new(Box::new(fake.clone()), Duration::ZERO)
    }

    const GIB: u64 = 1 << 30;
    const POOL: Pool = Pool {
        part: (1, 2),
        ceiling: 2 * GIB,
        large: (1, 4),
    };

    /// The pool's limit shrinks as others' use rises, to nothing at the
    /// budget; its own bytes do not shrink it.
    #[test]
    fn the_limit_shrinks_as_use_rises() {
        let fake = Fake::default();
        let budget = always_read(&fake);
        let mut last = u64::MAX;
        for usage in [0, 2 * GIB, 4 * GIB, 6 * GIB, 7 * GIB, 8 * GIB, 9 * GIB] {
            fake.set(8 * GIB, usage);
            let limit = budget.limit(&POOL, 0);
            assert!(limit <= last, "{usage}: {limit} > {last}");
            last = limit;
        }
        assert_eq!(last, 0, "nothing past the budget");
        fake.set(8 * GIB, 6 * GIB);
        assert_eq!(budget.limit(&POOL, 0), GIB, "half of the 2 GiB left");
        // A pool holding 1 GiB of those 6: the others leave it 3 GiB.
        assert_eq!(budget.limit(&POOL, GIB), GIB + GIB / 2);
    }

    /// The ceiling is the fixed size of before, higher on large budgets,
    /// and all of it without a reading.
    #[test]
    fn the_ceiling_grows_only_with_large_budgets() {
        let fake = Fake::default();
        let budget = always_read(&fake);
        assert_eq!(budget.limit(&POOL, 0), 2 * GIB, "no reading");
        fake.set(8 * GIB, 0);
        assert_eq!(budget.limit(&POOL, 0), 2 * GIB);
        fake.set(24 * GIB, 0);
        assert_eq!(budget.limit(&POOL, 0), 6 * GIB);
    }

    /// Asked often, DXGI is read once per maximum age, and again at once
    /// when the system signals a budget change. Bytes charged in between
    /// count as used until the next reading.
    #[test]
    fn readings_are_cached_until_old_or_changed() {
        let fake = Fake::default();
        fake.set(8 * GIB, 4 * GIB);
        let budget = Budget::new(Box::new(fake.clone()), Duration::from_secs(3600));
        for _ in 0..100 {
            assert_eq!(budget.reading().unwrap().local.usage, 4 * GIB);
        }
        assert_eq!(fake.reads.load(Ordering::Relaxed), 1);
        budget.charge(GIB);
        assert_eq!(budget.reading().unwrap().local.usage, 5 * GIB);
        assert_eq!(budget.limit(&POOL, 0), GIB + GIB / 2);
        fake.set(4 * GIB, 4 * GIB);
        assert_eq!(budget.reading().unwrap().local.budget, 8 * GIB, "cached");
        fake.signal.store(true, Ordering::Relaxed);
        let r = budget.reading().unwrap();
        assert_eq!(
            (r.local.budget, r.local.usage),
            (4 * GIB, 4 * GIB),
            "read again, charges gone"
        );
        assert_eq!(fake.reads.load(Ordering::Relaxed), 2);
    }

    /// The host's adapter answers (where there is a host GPU).
    #[test]
    fn the_host_adapter_reports_its_budget() {
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        let budget = get().expect("the host adapter's budget");
        assert!(budget.reading().is_some());
    }

    /// Use past nine tenths of the budget is logged once a run, not at
    /// each reading or each time it passes again.
    #[test]
    fn nine_tenths_is_logged_once() {
        let fake = Fake::default();
        let budget = always_read(&fake);
        fake.set(10 * GIB, 9 * GIB);
        budget.reading();
        assert_eq!(budget.warned(), 0, "nine tenths is not past");
        fake.set(10 * GIB, 9 * GIB + 1);
        budget.reading();
        budget.reading();
        assert_eq!(budget.warned(), 1);
        fake.set(10 * GIB, GIB);
        budget.reading();
        fake.set(10 * GIB, 10 * GIB);
        budget.reading();
        assert_eq!(budget.warned(), 1);
    }
}
