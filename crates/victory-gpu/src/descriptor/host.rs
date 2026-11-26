//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, view_at};
use crate::host::host;

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots past `count` (none in sampler heaps): [`DYNAMIC`]
    /// where the host takes them, else [`MIN_DYNAMIC`].
    dynamic: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from.
    made: Mutex<Vec<u64>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    /// With it the view's own copy of title memory, if it has one.
    kept: Mutex<Vec<Kept>>,
    /// Which dynamic slots are taken (by lists not yet reset).
    next: Mutex<Slots>,
    /// Slot pairs kept for host copies' views (`crate::memory`), past the
    /// dynamic slots and below the null buffer views: the next never
    /// taken, those given back, and where they end.
    pairs: Mutex<(u32, Vec<u32>)>,
    pairs_end: u32,
    /// Dynamic slots title memory copies' view pairs hold, where the heap
    /// has no room past the dynamic slots (a capped heap).
    pair_slots: std::sync::atomic::AtomicU32,
    /// View heaps: the view-stride table translated shaders read
    /// (`victory_shader::emit::VIEW_STRIDES`), a dword per title slot: the
    /// byte stride of the buffer view there, 0 for none.
    strides: Option<crate::tables::Table>,
    /// View heaps: the view-mark table translated shaders read
    /// (`victory_shader::emit`, "Write marks"), two dwords per slot as the
    /// stride table has one: the mark bit and phase of the unordered buffer
    /// view there, `VIEW_MARK_NONE` for none.
    marks: Option<crate::tables::Table>,
    /// Per title slot and per dynamic slot (tables made afresh lie there,
    /// and shaders index the view addresses at the slots their tables are
    /// in), the title address of the buffer view it holds, as a buffer
    /// descriptor's dwords 0 and 1 (shaders reading a descriptor's address
    /// as data, `crate::memory`), made on first use.
    addresses: std::sync::OnceLock<Option<crate::tables::Table>>,
    /// What the title slots view that title memory feeds (`crate::bindless`).
    pub(crate) viewed: std::sync::Arc<crate::bindless::Viewed>,
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// The size of a capped view heap if `device` takes no larger
/// shader-visible view heap (NVIDIA), else `None`. Translated shaders keep
/// their view indices below it (`set_view_heap_limit`), and a capped heap
/// keeps its last slot null for them.
pub(crate) fn view_heap_cap(device: &ID3D12Device) -> Option<u32> {
    // SAFETY: plain heap creation, released at once.
    let larger = unsafe {
        device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: PORTABLE_VIEWS + 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })
    };
    larger.is_err().then_some(PORTABLE_VIEWS)
}

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59).
const DYNAMIC: u32 = 1 << 18;

/// The dynamic slots of a heap whose host would not take [`DYNAMIC`] more
/// (a capped heap, NVIDIA): the title's slots keep their room first.
const MIN_DYNAMIC: u32 = 1 << 16;

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // Every slot a shader can index (title slots, also past a capped
        // heap, and dynamic ones): shaders read the table as a root view,
        // which has no bounds, at indices clamped below `HEAP_VIEWS`.
        let slots = count.max(victory_shader::emit::HEAP_VIEWS) as usize;
        let strides = (!sampler)
            .then(|| crate::tables::Table::new(slots * 4, "view strides"))
            .flatten();
        let marks = (!sampler)
            .then(|| {
                let table = crate::tables::Table::new(slots * 8, "view marks")?;
                // Every slot unmarked (all ones) until a view is made there.
                table.fill(0xFF);
                Some(table)
            })
            .flatten();
        // The larger dynamic range only where the host takes it on top of
        // every title slot; else the smaller one, and the cap after that.
        // Translated shaders clamp below the host cap (or `HEAP_VIEWS`),
        // samplers below `HEAP_SAMPLERS`: every such slot must exist. The
        // cap's last slot stays null, including for dynamic and pair views.
        let cap = victory_shader::emit::view_heap_limit();
        let views = cap
            .unwrap_or(victory_shader::emit::HEAP_VIEWS)
            .min(victory_shader::emit::HEAP_VIEWS);
        let samplers = victory_shader::emit::HEAP_SAMPLERS;
        let (heap, count, extra) = if sampler {
            match make(count.max(samplers)) {
                Ok(heap) => (heap, count, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if let Ok(heap) = make((count + DYNAMIC).max(views)) {
            (heap, count, DYNAMIC - u32::from(cap == Some(count + DYNAMIC)))
        } else if let Ok(heap) = make((count + DYNAMIC).max(PORTABLE_VIEWS)) {
            // shortcut: a host that caps its heaps below `HEAP_VIEWS` (NVIDIA,
            // at `PORTABLE_VIEWS`) faults on an index past them, as before
            // the clamp. It still takes every title slot, the larger dynamic
            // range (the smaller one runs out in the world, #59) and, past
            // them, room for host copies' view pairs (`crate::memory`).
            (heap, count, DYNAMIC - u32::from(cap == Some(count + DYNAMIC)))
        } else {
            match make((count + MIN_DYNAMIC).max(views)) {
                Ok(heap) => (heap, count, MIN_DYNAMIC - u32::from(cap == Some(count + MIN_DYNAMIC))),
                Err(e) if count + MIN_DYNAMIC > PORTABLE_VIEWS => {
                    // shortcut: descriptors past the host's limit are left
                    // out; a title that reaches them needs remapping. A
                    // shader's index past the heap reads its last slot,
                    // which no view takes (translated shaders clamp to it).
                    tracing::warn!(error = %e, count, "host view heap capped at {PORTABLE_VIEWS}");
                    (
                        make(PORTABLE_VIEWS).ok()?,
                        PORTABLE_VIEWS - MIN_DYNAMIC,
                        MIN_DYNAMIC - 1,
                    )
                }
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        };
        // Draws read the heap's tables every frame. Not the heap itself: a
        // priority on a shader-visible descriptor heap removed the device at
        // the next submission (RTX 5070 Ti, driver 621.24).
        for table in strides.iter().chain(&marks) {
            crate::host::keep_resident(table.resource(), crate::host::Residency::Frame);
        }
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let (gpu, total) = unsafe {
            (heap.GetGPUDescriptorHandleForHeapStart().ptr, heap.GetDesc().NumDescriptors)
        };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(total) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            sampler,
            made: Mutex::new(vec![0; count as usize]),
            kept: Mutex::new(vec![None; (count + extra) as usize]),
            next: Mutex::new(Slots::new(extra)),
            pairs: Mutex::new((count + extra, Vec::new())),
            pair_slots: std::sync::atomic::AtomicU32::new(0),
            pairs_end: match victory_shader::emit::NULL_BUFFER_VIEW {
                null if !sampler && null > count + extra && null < total => null - 1,
                _ if !sampler && cap == Some(total) => total - 1,
                _ => total,
            },
            strides,
            marks,
            addresses: std::sync::OnceLock::new(),
            viewed: Default::default(),
        };
        if !sampler {
            crate::bindless::register(&h.viewed);
        }
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, total) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        // What shaders read instead of a slot without a buffer view where
        // they take one at a computed index (`victory_shader::emit`, "Buffer
        // views at computed heap indices"): a null raw SRV, and UAV below it.
        let null = victory_shader::emit::NULL_BUFFER_VIEW;
        if !sampler && null > count + extra && null < total {
            h.raw_view_at(null, None, 0, 0, false);
            h.raw_view_at(null - 1, None, 0, 0, true);
        } else if !sampler {
            tracing::warn!(count, total, "no room for the null buffer views: computed indices of slots without a buffer view read what is there");
        }
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

/// A sampler heap of our own, for the lists that set a root signature whose
/// sampler heap is indexed directly (`SAMPLER_HEAP_DIRECTLY_INDEXED`) while
/// the title binds no sampler heap: D3D12 requires one to be bound first
/// (the debug layer's id 1419), and breaking that rule removes the device on
/// NVIDIA. Slot 0 holds a point sampler, so an indexed read of it returns
/// something defined rather than whatever the memory held.
pub(crate) fn fallback_samplers() -> Option<&'static HostHeap> {
    static HEAP: std::sync::OnceLock<Option<HostHeap>> = std::sync::OnceLock::new();
    HEAP.get_or_init(|| {
        let heap = HostHeap::new(true, 1)?;
        null_sampler(heap.cpu(0));
        Some(heap)
    })
    .as_ref()
}

/// What [`HostHeap::dynamic_stats`] tells.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct DynamicStats {
    pub(crate) free: u32,
    pub(crate) total: u32,
    pub(crate) low: u32,
    pub(crate) pairs: u32,
}

/// A heap's dynamic slots: which are taken, how many are free, and where
/// the search for free ones starts (past the last run taken).
pub(crate) struct Slots {
    next: u32,
    taken: Vec<bool>,
    free: u32,
    /// No free run is longer: a search that found none of a length is not
    /// walked again until a release makes a longer run (based on Belobug
    /// 788ea467: once the slots ran out, every table of a frame walked them
    /// all again, and frames took seconds).
    longest: usize,
    /// The fewest free since the last [`Slots::take_low`]: the pool's
    /// high-water mark is its size less this.
    low: u32,
}

impl Slots {
    pub(crate) fn new(n: u32) -> Self {
        Slots { next: 0, taken: vec![false; n as usize], free: n, longest: n as usize, low: n }
    }

    /// The fewest slots free since the last call; from now on, those free
    /// now.
    pub(crate) fn take_low(&mut self) -> u32 {
        std::mem::replace(&mut self.low, self.free)
    }

    /// `n` consecutive free slots, the last such run from the end, taken
    /// until released: the first. For what is held long (title memory
    /// copies' view pairs, for as long as a copy lives), so that it lies
    /// together at the top and does not break up the runs lists take from
    /// the cursor on.
    pub(crate) fn take_top(&mut self, n: u32) -> Option<u32> {
        let n = n as usize;
        if n == 0 || (self.free as usize) < n || self.longest < n {
            return None;
        }
        let mut run = 0;
        let first = (0..self.taken.len()).rev().find(|&at| {
            run = if self.taken[at] { 0 } else { run + 1 };
            run == n
        })?;
        self.taken[first..first + n].fill(true);
        self.free -= n as u32;
        self.low = self.low.min(self.free);
        Some(first as u32)
    }

    /// The first of `want` consecutive free slots; else the longest free
    /// run if it has at least `least`. Taken until released: the first and
    /// how many. One search either way, and none when fewer than `least`
    /// are free or no run that long is.
    pub(crate) fn take(&mut self, want: u32, least: u32) -> Option<(u32, u32)> {
        let len = self.taken.len();
        let least = least.max(1) as usize;
        if least > (want as usize).min(len) || (self.free as usize) < least || self.longest < least {
            return None;
        }
        // No run longer than `longest`: search for one that long.
        let want = (want as usize).min(len).min(self.longest);
        // Search from `next` on, then from the start (extending past the
        // cursor by want - 1 so runs crossing the cursor are not missed).
        let cursor = self.next as usize;
        let second_end = cursor.saturating_add(want - 1).min(len);
        let starts = (cursor..len).chain(0..second_end);
        let mut run = 0;
        let mut found = None;
        // The longest run seen: its first slot and length.
        let mut best = (0, 0);
        let mut last = usize::MAX;
        for at in starts {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if self.taken[at] { 0 } else { run + 1 };
            if run > best.1 {
                best = (at + 1 - run, run);
            }
            if run == want {
                found = Some((at + 1 - want, want));
                break;
            }
        }
        if found.is_none() {
            // Every slot was passed: no free run is longer than the best.
            self.longest = best.1;
        }
        let (first, n) = found.or((best.1 >= least).then_some(best))?;
        self.taken[first..first + n].fill(true);
        self.free -= n as u32;
        self.low = self.low.min(self.free);
        self.next = ((first + n) % len) as u32;
        Some((first as u32, n as u32))
    }

    /// Gives back `n` slots from `first` (taken or not).
    pub(crate) fn release(&mut self, first: u32, n: u32) {
        let len = self.taken.len();
        let (first, end) = ((first as usize).min(len), (first as usize).saturating_add(n as usize).min(len));
        let mut freed = false;
        for slot in self.taken[first..end].iter_mut().filter(|s| **s) {
            *slot = false;
            self.free += 1;
            freed = true;
        }
        if freed {
            // The free run the slots joined, which may be the longest now.
            let left = self.taken[..first].iter().rev().take_while(|t| !**t).count();
            let right = self.taken[end..].iter().take_while(|t| !**t).count();
            self.longest = self.longest.max(left + (end - first) + right);
        }
    }
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index. Lists take theirs in runs
    /// (`crate::draw`), title memory copies' view pairs from the top
    /// ([`HostHeap::pair`]): tests take them so.
    #[cfg(test)]
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let found = self.dynamic_run(n, n).map(|(first, _)| first);
        if found.is_none() {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(n, "no free dynamic descriptor slots: a table is not made"));
        }
        found
    }

    /// `want` consecutive free dynamic slots (view heaps only), or if there
    /// is no such run the longest free one of at least `least`, taken until
    /// [`HostHeap::release`]: the first's index and how many.
    pub(crate) fn dynamic_run(&self, want: u32, least: u32) -> Option<(u32, u32)> {
        if self.sampler {
            return None;
        }
        let mut slots = self.next.lock().unwrap_or_else(|e| e.into_inner());
        slots.take(want, least).map(|(first, n)| (self.count + first, n))
    }

    /// The dynamic slots free and in all.
    pub(crate) fn dynamic_free(&self) -> (u32, u32) {
        let slots = self.next.lock().unwrap_or_else(|e| e.into_inner());
        (slots.free, self.dynamic)
    }

    /// The dynamic slots free now, in all, the fewest free since the last
    /// call, and how many title memory copies' view pairs hold.
    pub(crate) fn dynamic_stats(&self) -> DynamicStats {
        let mut slots = self.next.lock().unwrap_or_else(|e| e.into_inner());
        DynamicStats {
            free: slots.free,
            total: self.dynamic,
            low: slots.take_low(),
            pairs: self.pair_slots.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Two consecutive slots (view heaps only) kept until
    /// [`HostHeap::free_pair`]: past the dynamic slots where the heap has
    /// room (a host copy's views, `crate::memory`, for as long as it
    /// lives), else dynamic ones.
    pub(crate) fn pair(&self) -> Option<u32> {
        if self.sampler {
            return None;
        }
        {
            let mut pairs = self.pairs.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(index) = pairs.1.pop() {
                return Some(index);
            }
            if pairs.0 + 2 <= self.pairs_end {
                pairs.0 += 2;
                return Some(pairs.0 - 2);
            }
        }
        // From the top of the dynamic slots: a pair lives as long as its
        // copy, and scattered among the runs lists take it would leave holes
        // too short for a table when they come back.
        let first = self.next.lock().unwrap_or_else(|e| e.into_inner()).take_top(2)?;
        self.pair_slots.fetch_add(2, std::sync::atomic::Ordering::Relaxed);
        Some(self.count + first)
    }

    /// Gives back a pair [`HostHeap::pair`] returned.
    pub(crate) fn free_pair(&self, index: u32) {
        if index >= self.count + self.dynamic {
            self.pairs.lock().unwrap_or_else(|e| e.into_inner()).1.push(index);
        } else {
            self.pair_slots.fetch_sub(2, std::sync::atomic::Ordering::Relaxed);
            self.release(index, 2);
        }
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        self.next.lock().unwrap_or_else(|e| e.into_inner()).release(at, n);
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let (kept, mark) = make_marked(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
        self.set_stride(index, Some(view));
        self.set_mark(index, mark);
        // A table made afresh: shaders reading a view's address there (e.g.
        // the BVH builder's base) index the view addresses at this slot.
        self.note_address(index, Some(view));
    }

    /// Makes slot `index` (a dynamic one) a raw view, shader resource or
    /// unordered access (`uav`), of `bytes` of `buffer` from byte `first`
    /// (both multiples of 4), which it holds; a null one for no buffer, or
    /// for unordered access to a buffer shaders cannot write.
    pub(crate) fn raw_view_at(&self, index: u32, buffer: Option<&ID3D12Resource>, first: u64, bytes: u64, uav: bool) {
        let Some(host) = host() else { return };
        let writable = |b: &ID3D12Resource| {
            // SAFETY: a live buffer.
            unsafe { b.GetDesc() }.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
        };
        let buffer = buffer.filter(|&b| !uav || writable(b));
        // shortcut: a buffer view holds at most 2^27 elements (512 MiB raw).
        let (first, elements) = match buffer {
            Some(_) => (first / 4, (bytes / 4).clamp(1, 1 << 27) as u32),
            // A null view still names its size.
            None => (0, 1),
        };
        // SAFETY: a slot of our heap; the buffer is live (or none: a null view).
        unsafe {
            if uav {
                let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            FirstElement: first,
                            NumElements: elements,
                            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                            ..Default::default()
                        },
                    },
                };
                host.device
                    .CreateUnorderedAccessView(buffer, None, Some(&desc), self.cpu(index));
            } else {
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: first,
                            NumElements: elements,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                };
                host.device
                    .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index));
            }
        }
        self.keep(index, buffer.map(|b| (b.clone(), None)));
        self.write_stride(index, 4);
        self.set_mark(index, None);
        // Not a title view: no entry an earlier table left there stays.
        self.note_address(index, None);
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
        self.set_stride(index, None);
        self.set_mark(index, None);
        self.note_address(index, None);
        if index < self.count {
            self.viewed.set(index, None, false);
        }
    }

    /// Enters the byte stride of `view` (0 for none or not a buffer view)
    /// for slot `index` in the view-stride table.
    fn set_stride(&self, index: u32, view: Option<&View>) {
        self.write_stride(index, view.map_or(0, view_stride));
    }

    fn write_stride(&self, index: u32, stride: u32) {
        let Some(table) = &self.strides else { return };
        if index >= self.count.max(victory_shader::emit::HEAP_VIEWS) {
            return;
        }
        // The table has a dword per slot below the larger of the title's
        // count and `HEAP_VIEWS`.
        table.write(index as usize, &[stride]);
    }

    /// The view-stride table, for root SRV `VIEW_STRIDES`.
    pub(crate) fn strides(&self) -> Option<&ID3D12Resource> {
        self.strides.as_ref().map(crate::tables::Table::resource)
    }

    /// Enters slot `index`'s view-mark entry: the mark bit and phase of the
    /// unordered buffer view there ([`crate::shadow::Shadow::mark_entry`]),
    /// none for anything else.
    pub(crate) fn set_mark(&self, index: u32, entry: Option<(u32, u32)>) {
        let Some(table) = &self.marks else { return };
        if index >= self.count.max(victory_shader::emit::HEAP_VIEWS) {
            return;
        }
        let entry = entry.unwrap_or((victory_shader::emit::VIEW_MARK_NONE, 0));
        // The table has two dwords per slot below the larger of the title's
        // count and `HEAP_VIEWS`.
        table.write(2 * index as usize, &[entry.0, entry.1]);
    }

    /// The view-mark table, for the root SRV of shaders marking their writes.
    pub(crate) fn marks(&self) -> Option<&ID3D12Resource> {
        self.marks.as_ref().map(crate::tables::Table::resource)
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Kept) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = kept.get_mut(index as usize) {
            let old = std::mem::replace(slot, resource);
            #[cfg(feature = "diagnostics")]
            if let Some((old, _)) = &old
                && !self.sampler
                && slot.as_ref().map(|s| windows::core::Interface::as_raw(&s.0)) != Some(windows::core::Interface::as_raw(old))
                // SAFETY: a live resource.
                && unsafe { old.GetDesc() }.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
            {
                use windows::core::Interface;
                let left = crate::resource::life::holders(old);
                crate::resource::life::note(old.as_raw() as usize, crate::resource::life::SLOT, u64::from(index), left);
            }
            // Title threads rewrite slots while the GPU runs a submission;
            // one of its shaders may hold the old descriptor (or read the
            // slot at an index from wrong data): what it views is freed
            // only once that work has run.
            if let Some((old, _)) = old {
                crate::host::retire(old);
            }
        }
    }

    /// The host address of the slots' view addresses (see the field), made
    /// on first use from the slots made so far.
    pub(crate) fn view_addresses(&self) -> Option<u64> {
        let made = self.addresses.get_or_init(|| {
            // The view is a root view, which has no bounds: an entry for
            // every slot a table can lie in, the dynamic ones too (a table
            // made afresh on a host whose heap is smaller than the title's
            // read past the title slots' entries: whatever followed the
            // buffer, or a GPU fault).
            let table = crate::tables::Table::new(self.address_slots() as usize * 8, "view addresses")?;
            crate::host::keep_resident(table.resource(), crate::host::Residency::Frame);
            let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            for (i, &key) in made.iter().enumerate() {
                let view = (key != 0).then(|| view_at(key as u32, (key >> 32) as u32)).flatten();
                write_address(&table, i, view.as_ref());
            }
            // Dynamic slots are entered as they are made ([`Self::make_at`],
            // [`Self::make_null`], raw views).
            for i in self.count..self.address_slots() {
                write_address(&table, i as usize, None);
            }
            Some(table)
        });
        made.as_ref().map(crate::tables::Table::gpu)
    }

    /// The slots the view addresses have an entry for: the title's and the
    /// dynamic ones.
    fn address_slots(&self) -> u32 {
        self.count + self.dynamic
    }

    /// Slot `index` now holds `view` (or no view): its view address.
    fn note_address(&self, index: u32, view: Option<&View>) {
        if index < self.address_slots()
            && let Some(Some(table)) = self.addresses.get()
        {
            write_address(table, index as usize, view);
        }
    }

    /// The view address entered for slot `index`; none for a slot past the
    /// view addresses' buffer (tests).
    #[cfg(test)]
    fn address_at(&self, index: u32) -> Option<u64> {
        let table = self.addresses.get()?.as_ref()?;
        let (lo, hi) = (table.read(2 * index as usize)?, table.read(2 * index as usize + 1)?);
        Some(u64::from(hi) << 32 | u64::from(lo))
    }

    /// Makes slots `index` and `index + 1` (dynamic ones) a raw shader
    /// resource view and a raw unordered access view of the first `bytes`
    /// (a multiple of 4) of `buffer`, which they hold; null views for no
    /// buffer.
    pub(crate) fn raw_pair_at(&self, index: u32, buffer: Option<&ID3D12Resource>, bytes: u64) {
        let Some(host) = host() else { return };
        // shortcut: a buffer view holds at most 2^27 elements (512 MiB raw);
        // a longer copy's rest reads 0 through it.
        let elements = (bytes / 4).min(1 << 27) as u32;
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: elements,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        // SAFETY: slots of our heap; the buffer is live (or none: null views).
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&srv), self.cpu(index));
            host.device
                .CreateUnorderedAccessView(buffer, None, Some(&uav), self.cpu(index + 1));
        }
        self.keep(index, buffer.map(|b| (b.clone(), None)));
        self.keep(index + 1, buffer.map(|b| (b.clone(), None)));
        self.write_stride(index, 4);
        self.write_stride(index + 1, 4);
        // Writes through these are title memory's, marked in its page table.
        self.set_mark(index, None);
        self.set_mark(index + 1, None);
        // Not a title view: no entry an earlier table left there stays.
        self.note_address(index, None);
        self.note_address(index + 1, None);
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live shader-visible heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start.ptr + u64::from(index) * u64::from(self.step),
        }
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    /// Makes the view of slot `index` (at `handle` in title memory) if it
    /// is a placed texture's left null for later ([`Self::mirror`]), as a
    /// draw reading it through a table would: true if it made it.
    pub(crate) fn make_placed(&self, index: u32, handle: u64) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        if index >= self.count || self.sampler {
            PLACED_WHY[0].fetch_add(1, Relaxed);
            return false;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        let key = u64::from(generation) << 32 | u64::from(i);
        {
            let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            if made[index as usize] == key {
                PLACED_WHY[1].fetch_add(1, Relaxed);
                return false;
            }
        }
        let Some(view) = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten() else {
            PLACED_WHY[2].fetch_add(1, Relaxed);
            return false;
        };
        if !is_placed_texture(&view) {
            PLACED_WHY[3].fetch_add(1, Relaxed);
            return false;
        }
        self.mirror(index, handle, true);
        true
    }

    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if made[index as usize] != 0 {
                made[index as usize] = 0;
                self.make_null(index);
                self.note_address(index, None);
            }
            return None;
        };
        if made[index as usize] != key {
            self.set_stride(index, Some(&view));
            self.note_address(index, Some(&view));
            // Shaders index the heap itself (`crate::bindless`): buffer views
            // are made when written.
            // shortcut: views of textures the title placed itself are made
            // when a draw reads them through a table: a host texture (and a
            // staging copy) for each of the ~20,000 the title has views of
            // ran the host out of memory (run b2).
            let later = !for_draw && is_placed_texture(&view);
            made[index as usize] = if later { 0 } else { key };
            let (kept, mark) = if later {
                (None, None)
            } else {
                make_marked(&view, self.cpu(index))
            };
            self.set_mark(index, mark);
            if kept.is_none() && !self.sampler {
                if !later && is_placed_texture(&view) {
                    PLACED_FAILED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                null_for(&view, self.cpu(index));
            }
            if !self.sampler {
                self.viewed.set(index, Some(&view), kept.is_some());
            }
            self.keep(index, kept.flatten());
        }
        Some(view)
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self.made.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize)?;
        // A view the host could not make leaves a null view, its record
        // noted all the same; it keeps no resource.
        let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize).is_some_and(Option::is_some);
        if key == 0 || (!kept && !self.sampler) {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }

    /// Whether slot `index` was made from a record (tests).
    #[cfg(test)]
    pub(crate) fn made(&self, index: u32) -> bool {
        self.made.lock().unwrap_or_else(|e| e.into_inner())[index as usize] != 0
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// The byte stride of a buffer view (its element size, at least 1), 0 for
/// other views: what the view-stride table holds for it.
fn view_stride(view: &View) -> u32 {
    let (unordered, words): (bool, &[u32; 10]) = match view {
        // Read as raw buffers (`make`).
        View::Constant(..) => return 4,
        View::Shader(_, Some(desc)) => (false, desc),
        View::Unordered(_, _, Some(desc)) => (true, desc),
        View::Placed(unordered, resource, desc) if resource[0] == 1 => {
            match desc[..10].try_into() {
                Ok(words) => (*unordered, words),
                Err(_) => return 0,
            }
        }
        _ => return 0,
    };
    // The view dimension: buffer (1) for shader and unordered views alike.
    if words[1] != 1 {
        return 0;
    }
    let element = if unordered {
        let desc = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let buffer = unsafe { desc.Anonymous.Buffer };
        if buffer.StructureByteStride > 0 {
            u64::from(buffer.StructureByteStride)
        } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            4
        } else {
            crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        }
    } else {
        element_size(&srv_desc(words))
    };
    // Nonzero: shaders taking views at computed indices read the null buffer
    // view where the table holds 0 (`victory_shader::emit`).
    element.max(1) as u32
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// Writes slot `index`'s view address into the view addresses `table`
/// (the caller keeps `index` below [`HostHeap::address_slots`]):
/// a buffer view's title address, or a ray tracing acceleration
/// structure's (dword 0 low, dword 1 bits 0-15 high); zeros for anything
/// else.
fn write_address(table: &crate::tables::Table, index: usize, view: Option<&View>) {
    let address = view.map_or(0, view_address);
    let words = [address as u32, (address >> 32) as u32 & 0xFFFF];
    // The view addresses hold 8 bytes per title and dynamic slot (below
    // `address_slots`).
    table.write(2 * index, &words);
}

/// The title address a program reads as a view's address (dwords 0 and 1
/// of its descriptor). A ray tracing acceleration structure's shader view
/// has no resource: its description holds the structure's address
/// (`Location`, dwords 4 and 5), the BVH the program walks with
/// `image_bvh64_intersect_ray` from that address. Any other view: its
/// buffer's first byte, or 0.
fn view_address(view: &View) -> u64 {
    match view {
        View::Shader(_, Some(desc))
            if desc[1] == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32 =>
        {
            u64::from(desc[4]) | u64::from(desc[5]) << 32
        }
        _ => buffer_range(view).map_or(0, |(address, _)| address),
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host copy a buffer view's bytes are in (the one its host view
/// reads, [`crate::shadow::for_view`]) and the view's title range.
pub(crate) fn view_copy(view: &View) -> Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    let (address, len) = buffer_range(view)?;
    let (shadow, _) = crate::shadow::for_view(address, len, view_align(view))?;
    Some((shadow, address, len))
}

/// Where the host copy a buffer view reads starts ([`copy_align`]), as
/// [`buffer_srv`] and [`buffer_uav`] ask for it, so that draws find and
/// refresh the copy the slot shows.
fn view_align(view: &View) -> u64 {
    let words: Option<(bool, &[u32; 10])> = match view {
        View::Shader(_, Some(desc)) => Some((false, desc)),
        View::Unordered(_, _, Some(desc)) => Some((true, desc)),
        View::Placed(unordered, resource, desc) if resource[0] == 1 => {
            desc[..10].try_into().ok().map(|w| (*unordered, w))
        }
        _ => None,
    };
    let element = u64::from(view_stride(view));
    match words {
        Some((false, w)) if w[1] == 1 => {
            let desc = srv_desc(w);
            // SAFETY: a buffer view's union member.
            copy_align(desc.Format, unsafe { desc.Anonymous.Buffer }.StructureByteStride, element)
        }
        Some((true, w)) if w[1] == 1 => {
            let desc = uav_desc(w);
            // SAFETY: a buffer view's union member.
            copy_align(desc.Format, unsafe { desc.Anonymous.Buffer }.StructureByteStride, element)
        }
        _ => element,
    }
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, if any, and the view's
/// own copy of title memory ([`crate::shadow::for_view`]): the slot holds
/// it, so that draws reading the view find and refresh that copy.
type Kept = Option<(ID3D12Resource, Option<std::sync::Arc<crate::shadow::Shadow>>)>;

/// Buffer views refused so far (diagnostics: the first are logged).
#[cfg(feature = "diagnostics")]
static REFUSED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Where a raw view may start in its buffer, in bytes.
const RAW_ALIGN: u64 = D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT as u64;

/// The stride of a structured buffer view the host makes raw: its
/// structure stride when it is a whole number of dwords (what translated
/// shaders read by index), else 0 (left as it is).
///
/// Translated shaders read untyped buffers as raw bytes (scalar loads at a
/// byte address, raw loads at `element * stride + offset`), and indexed
/// reads of structured buffers through raw handles too. NVIDIA addresses a
/// structured view by its own stride whatever the read (an RTX 5090 read
/// zeros at byte 20 of a 16-byte-stride view; a raw read at byte `b` of a
/// 12-byte view lands at `b / 4 * 12`), so only a raw view reads the same
/// everywhere. The view-stride table keeps the title's stride. Ported from
/// fxceless/victory-gta6 (GPL-3.0), commit 578d459c, `raw_window`.
fn raw_stride(format: DXGI_FORMAT, stride: u32) -> u32 {
    if format == DXGI_FORMAT_UNKNOWN && stride > 0 && stride.is_multiple_of(4) {
        stride
    } else {
        0
    }
}

/// `count` elements of `stride` bytes (a whole number of dwords) from byte
/// `start` of a host copy as a raw view: its first dword and its dwords, if
/// they start on 16 bytes (`D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT`) and are not
/// too many for one view.
fn raw_window(start: u64, count: u32, stride: u32) -> Option<(u64, u32)> {
    if stride == 0 || !start.is_multiple_of(RAW_ALIGN) {
        return None;
    }
    let dwords = u64::from(count) * u64::from(stride / 4);
    let dwords = u32::try_from(dwords).ok().filter(|&d| d <= 1 << 27)?;
    Some((start / 4, dwords))
}

/// Where the host copy of a buffer view's bytes must start: on 16 bytes if
/// the host makes the view raw ([`raw_stride`]), else on one of its
/// elements (`element`). [`crate::shadow::for_view`] makes a copy of the
/// view's own bytes when its buffer's copy does not start it there (the
/// meshlet culling's 456-byte constants: element 1 is not on 16 bytes).
fn copy_align(format: DXGI_FORMAT, stride: u32, element: u64) -> u64 {
    if raw_stride(format, stride) > 0 {
        RAW_ALIGN
    } else {
        element
    }
}

/// Logs a structured view the host leaves structured (a stride that is not
/// a whole number of dwords, or too many dwords for one raw view): raw
/// reads of it may read the wrong bytes on NVIDIA.
fn left_structured(format: DXGI_FORMAT, stride: u32, count: u32) {
    if format == DXGI_FORMAT_UNKNOWN && stride > 0 {
        once!(stride, count, "structured buffer view left structured on the host");
    }
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if element == 0 {
        once!(len, "buffer view of elements without a size");
        return None;
    }
    // The copy its own bytes are in, its first element counted from there
    // (a structured view made raw: on 16 bytes of it).
    let start = address + buffer.FirstElement * element;
    let stride = buffer.StructureByteStride;
    let align = copy_align(desc.Format, stride, element);
    let Some((shadow, own)) =
        crate::shadow::for_view(start, u64::from(buffer.NumElements) * element, align)
    else {
        #[cfg(feature = "diagnostics")]
        if REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::info!(start = format_args!("{start:#x}"), len, element, count = buffer.NumElements, "refused buffer view: no host copy");
        }
        return None;
    };
    let offset = shadow.offset(start);
    let element = match raw_window(offset, buffer.NumElements, raw_stride(desc.Format, stride)) {
        Some((first, count)) => {
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            buffer.FirstElement = first;
            buffer.NumElements = count;
            buffer.StructureByteStride = 0;
            buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            4
        }
        None => {
            left_structured(desc.Format, stride, buffer.NumElements);
            buffer.FirstElement = offset / element;
            element
        }
    };
    let (first, count) = (buffer.FirstElement, buffer.NumElements);
    if !in_buffer(&shadow.buffer, first, count, element) {
        #[cfg(feature = "diagnostics")]
        if REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::info!(start = format_args!("{start:#x}"), first, count, element, "refused buffer view: empty or past its copy");
        }
        once!(first, count, element, "buffer view empty or past its host copy");
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Some((shadow.buffer.clone(), own.then_some(shadow))))
}

/// Whether `count` elements of `element` bytes from element `first` lie in
/// `buffer`. The console takes a view that is empty or runs past its
/// memory; the host removes the device for one.
fn in_buffer(buffer: &ID3D12Resource, first: u64, count: u32, element: u64) -> bool {
    // SAFETY: a live buffer.
    within(unsafe { buffer.GetDesc() }.Width, first, count, element)
}

fn within(width: u64, first: u64, count: u32, element: u64) -> bool {
    count > 0 && first.saturating_add(u64::from(count)).saturating_mul(element) <= width
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    if element == 0 {
        once!(len, "unordered buffer view of elements without a size");
        return None;
    }
    let start = address + buffer.FirstElement * element;
    let stride = buffer.StructureByteStride;
    let align = copy_align(desc.Format, stride, element);
    let (shadow, own) = crate::shadow::for_view(start, u64::from(buffer.NumElements) * element, align)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    let offset = shadow.offset(start);
    buffer.CounterOffsetInBytes = 0;
    let element = match raw_window(offset, buffer.NumElements, raw_stride(desc.Format, stride)) {
        Some((first, count)) => {
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            buffer.FirstElement = first;
            buffer.NumElements = count;
            buffer.StructureByteStride = 0;
            buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            4
        }
        None => {
            left_structured(desc.Format, stride, buffer.NumElements);
            buffer.FirstElement = offset / element;
            element
        }
    };
    let (first, count) = (buffer.FirstElement, buffer.NumElements);
    if !in_buffer(&shadow.buffer, first, count, element) {
        once!(first, count, element, "unordered buffer view empty or past its host copy");
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at)
    };
    MARK.with(|m| m.set(shadow.mark_entry(start)));
    Some(Some((shadow.buffer.clone(), own.then_some(shadow))))
}

thread_local! {
    /// The view-mark entry of the unordered buffer view [`make`] last made
    /// on this thread ([`make_marked`]).
    static MARK: std::cell::Cell<Option<(u32, u32)>> = const { std::cell::Cell::new(None) };
}

/// [`make`], and the view-mark entry of the view if it is an unordered
/// buffer view whose writes shaders mark (`victory_shader::emit`, "Write
/// marks").
fn make_marked(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> (Option<Kept>, Option<(u32, u32)>) {
    MARK.with(|m| m.set(None));
    let kept = make(view, at);
    (kept, MARK.with(std::cell::Cell::take))
}

/// Typed depth formats with the colour formats shaders read them as.
const DEPTH_READABLE: [(u32, u32); 4] = [(20, 21), (40, 41), (45, 46), (55, 56)];

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in a format neither of the
/// resource's family nor among the formats it was made castable to
/// (`castable`, `resource::castable`). A view of a typed texture in another
/// format of its family takes the texture's format unless it is castable.
/// shortcut: such a view (sRGB of UNORM, say, on a texture the host made
/// without castable formats) reads without its conversion.
fn fits(texture: &ID3D12Resource, castable: &[u32], mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let mut view = format.as_ref().map_or(0, |f| f.0 as u32);
    // Shaders read a depth texture through its readable format; the host
    // refuses a typed depth format in a shader resource view.
    if !unordered
        && let Some(&(_, readable)) = DEPTH_READABLE.iter().find(|&&(depth, _)| depth == view)
    {
        view = readable;
        if let Some(format) = format.as_deref_mut() {
            format.0 = readable as i32;
        }
    }
    if view == 0 {
        return !typeless && (!unordered || typed_store(own));
    }
    let cast = castable.contains(&view);
    if family(view) != family(own) && !cast {
        once!(view, own, "view format of another family");
        return false;
    }
    let target = if !typeless && !cast { own } else { view };
    if unordered && !typed_store(target) {
        once!(target, own, "an unordered view in a format the host cannot store");
        return false;
    }
    if let Some(format) = format
        && !typeless
        && !cast
    {
        *format = d.Format;
    }
    true
}

/// Whether a view without a description of `resource` (a title texture)
/// may take its host texture's defaults: not when the host texture has
/// another format than the title's (typeless, or shared with a title
/// texture of another format of the same memory).
/// shortcut: such views are null; spell the description out (dimension,
/// mips) if the title makes them.
fn default_view(resource: crate::object::This) -> bool {
    let other = crate::resource::view_format(resource);
    if other.is_some() {
        once!(format = other, "a view without a description of a texture whose host texture has another format");
    }
    other.is_none()
}

/// Whether the host stores to typed unordered views of `format`.
fn typed_store(format: u32) -> bool {
    static KNOWN: Mutex<Vec<(u32, bool)>> = Mutex::new(Vec::new());
    let mut known = KNOWN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&(_, ok)) = known.iter().find(|(f, _)| *f == format) {
        return ok;
    }
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: DXGI_FORMAT(format as i32),
        ..Default::default()
    };
    let ok = host().is_some_and(|h| {
        // SAFETY: a plain query into a struct of its size.
        unsafe {
            h.device.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                (&raw mut support).cast(),
                size_of_val(&support) as u32,
            )
        }
        .is_ok()
            && support.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 != 0
    });
    known.push((format, ok));
    ok
}

/// Makes the host descriptor for `view` at `at` (in any host view heap):
/// the host resource it views, if it can.
pub(crate) fn make_view(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<ID3D12Resource> {
    make(view, at).flatten().map(|k| k.0)
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            // Constant buffers in the title's slots, through the heap itself
            // or a table, are read as raw buffers by translated shaders
            // (`victory_shader::emit`, "Constant buffers in the heap"): the
            // console needs 16-byte alignment, a host constant buffer view
            // 256, which most of the title's do not have on the host.
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let bytes = u64::from(size).min(shadow.left(address));
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        NumElements: (bytes / 4) as u32,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            buffer_srv(address, bytes, desc, at)
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(copy) = crate::resource::host_copy(resource) {
                let texture = &copy.texture;
                let mut desc = desc.as_ref().map(srv_desc);
                if desc.is_none() && !default_view(resource) {
                    return None;
                }
                if !fits(texture, &copy.castable, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some((texture.clone(), None)))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(copy) = crate::resource::host_copy(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            let texture = &copy.texture;
            // A depth texture's unordered views write its twin.
            // SAFETY: a live resource.
            if unsafe { texture.GetDesc() }.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0 {
                return crate::twin::uav(resource, &copy, desc.as_ref(), at)
                    .map(|twin| Some((twin, None)));
            }
            let mut desc = desc.as_ref().map(uav_desc);
            if desc.is_none() && !default_view(resource) {
                return None;
            }
            if !fits(texture, &copy.castable, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some((texture.clone(), None)))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words, crate::sampler::aniso_point_mip(device));
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(dimension = resource[0], "placed unordered texture view not mirrored");
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// Whether `view` is a view of a texture the title placed in memory itself.
/// Marked slots not made, by reason: past the heap or a sampler heap,
/// already made, no view of ours there, not a placed texture.
/// Placed-texture views the host tried to make and could not (left null
/// although marked made).
pub(crate) static PLACED_FAILED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) static PLACED_WHY: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

fn is_placed_texture(view: &View) -> bool {
    matches!(view, View::Placed(_, resource, _) if resource[0] != 1)
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(false, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    Some(View::Shader(texture as usize, desc[..10].try_into().ok()))
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        // Read as raw buffers, as made (`make`).
        View::Constant(..) => return null_buffer_view(at),
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Which structured views the host makes raw, and where their copies
    /// start: a stride of whole dwords on 16 bytes of its copy becomes the
    /// same bytes in dwords; off 16 bytes it needs a copy of its own (16);
    /// odd strides and typed views stay as they are (their element).
    #[test]
    fn structured_views_become_raw_views_of_the_same_bytes() {
        // 10 elements of 12 bytes at byte 48: dwords 12..42.
        assert_eq!(raw_window(48, 10, raw_stride(DXGI_FORMAT_UNKNOWN, 12)), Some((12, 30)));
        // Element 1 of a 456-byte view is not on 16 bytes.
        assert_eq!(raw_window(456, 4, raw_stride(DXGI_FORMAT_UNKNOWN, 456)), None);
        assert_eq!(copy_align(DXGI_FORMAT_UNKNOWN, 456, 456), RAW_ALIGN);
        // A stride of 6 bytes, a typed view: not made raw.
        assert_eq!(raw_stride(DXGI_FORMAT_UNKNOWN, 6), 0);
        assert_eq!(raw_window(48, 10, raw_stride(DXGI_FORMAT_UNKNOWN, 6)), None);
        assert_eq!(copy_align(DXGI_FORMAT_UNKNOWN, 6, 6), 6);
        assert_eq!(raw_stride(DXGI_FORMAT_R32G32B32A32_FLOAT, 0), 0);
        assert_eq!(copy_align(DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16), 16);
        // Too many dwords for one view.
        assert_eq!(raw_window(0, u32::MAX, 8), None);
    }

    /// A shader resource view of a depth texture the host made typeless,
    /// described in the typed depth format, is made in the readable one.
    #[test]
    fn depth_formats_in_shader_resource_views_are_made_readable() {
        let Some(host) = host() else { return };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R32_TYPELESS,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: plain creation; our out pointer.
        let made = unsafe {
            host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut texture)
        };
        let (Ok(()), Some(texture)) = (made, texture) else { return };
        let mut format = DXGI_FORMAT_D32_FLOAT;
        assert!(fits(&texture, &[], Some(&mut format), false));
        assert_eq!(format, DXGI_FORMAT_R32_FLOAT);
    }

    /// The tables beside view heaps, which draws read every frame, are
    /// kept resident; the heap itself gets no priority (one removed the
    /// device).
    #[test]
    fn view_heap_tables_are_kept_resident() {
        use crate::host::{Residency, kept};
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU here
        };
        assert_eq!(kept(&heap.heap), None);
        for table in heap.strides.iter().chain(&heap.marks) {
            assert_eq!(kept(table.resource()), Some(Residency::Frame));
        }
    }

    /// The world's 16-byte view of 0 elements removed the device.
    #[test]
    fn empty_and_overlong_buffer_views_are_refused() {
        assert!(within(64, 0, 4, 16));
        assert!(within(64, 3, 1, 16));
        assert!(!within(64, 0, 0, 16));
        assert!(!within(64, 3, 2, 16));
        assert!(!within(64, u64::MAX, 1, 16));
    }

    /// The world's ray traced pass (`f607780aa8e502d0`, #1968) takes its
    /// BVH's address from an acceleration structure's view, which has no
    /// resource: the view address is the description's location, not 0. A
    /// buffer view placed in memory keeps its own address.
    #[test]
    fn an_acceleration_structures_view_address_is_its_location() {
        // Words as the title wrote them: format 0, dimension 11, the
        // default component mapping, then the location 0xd_5f1c_0000.
        let mut words = [0u32; 10];
        (words[1], words[2], words[4], words[5]) = (11, 0x1688, 0x5f1c_0000, 0xd);
        let Some(host) = host() else {
            return; // no host GPU: no table to write
        };
        // Two slots' entries, in system memory as without VICTORY_TABLES_VRAM.
        let table = crate::tables::Table::with_mode(host, 16, "test view addresses", crate::tables::Mode::System)
            .expect("a table");
        table.fill(0xFF);
        let entries = |t: &crate::tables::Table| [0, 1, 2, 3].map(|i| t.read(i).unwrap());
        write_address(&table, 1, Some(&View::Shader(0, Some(words))));
        assert_eq!(entries(&table), [u32::MAX, u32::MAX, 0x5f1c_0000, 0xd]);

        // A placed buffer view: dimension 1, 64 elements from element 0,
        // its memory at 0x1_2345_6700 (description dwords 10 and 11).
        let mut resource = [0u32; 14];
        resource[0] = 1;
        let mut desc = [0u32; 16];
        (desc[1], desc[2], desc[6], desc[10], desc[11]) = (1, 0x1688, 64, 0x2345_6700, 1);
        write_address(&table, 0, Some(&View::Placed(false, resource, desc)));
        assert_eq!(entries(&table), [0x2345_6700, 1, 0x5f1c_0000, 0xd]);

        // No view: zeros.
        write_address(&table, 1, None);
        assert_eq!(entries(&table), [0x2345_6700, 1, 0, 0]);
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A view heap the host caps keeps its last slot out of title and
    /// dynamic slots: translated shaders clamp their indices to it.
    #[test]
    fn a_capped_view_heap_keeps_its_last_slot_null() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(cap) = view_heap_cap(&host.device) else {
            return; // the host takes larger heaps
        };
        let heap = HostHeap::new(false, cap + 1).expect("a capped view heap");
        assert_eq!(heap.count + heap.dynamic, cap - 1);
        let last = heap.dynamic(heap.dynamic).unwrap() + heap.dynamic - 1;
        assert!(last < cap - 1);
        assert_eq!(heap.pairs_end, cap - 1);
        // Small title heaps must cover the same shader-visible range too.
        let small = HostHeap::new(false, 4).expect("a small capped view heap");
        // SAFETY: a plain query of a live heap.
        assert_eq!(unsafe { small.heap.GetDesc().NumDescriptors }, cap);
        assert!(small.count + small.dynamic < cap);
        assert_eq!(small.pairs_end, cap - 1);
        // A dynamic range ending exactly at the cap also leaves a guard.
        let exact = HostHeap::new(false, cap - DYNAMIC).expect("an exact capped view heap");
        assert_eq!(exact.count + exact.dynamic, cap - 1);
        assert_eq!(exact.pairs_end, cap - 1);
    }

    /// A table made afresh lies in dynamic slots, and a shader reading a
    /// view's address (e.g. the BVH builder's base) indexes the view
    /// addresses there: they have an entry for each dynamic slot, entered
    /// whenever the slot is made (a view, null, or a raw view of ours).
    /// Giving a slot back leaves its entry: every table made afresh makes
    /// all its slots. (Hosts whose view heaps are smaller than the title's,
    /// NVIDIA, make such tables; the shader read past the buffer.)
    ///
    /// Any heap size the host takes will do: a title heap of 4 slots leaves
    /// dynamic slots past it either way ([`DYNAMIC`] or [`MIN_DYNAMIC`]).
    #[test]
    fn view_addresses_cover_tables_made_afresh() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let end = heap.count() + heap.dynamic;
        assert!(heap.view_addresses().is_some());
        let first = heap.dynamic(2).unwrap();
        assert!(first >= heap.count() && first + 1 < end);
        // A buffer the title placed at 0xd_6013_e000: 16 elements of 4 bytes.
        let address = 0xd_6013_e000u64;
        let mut resource = [0u32; 14];
        (resource[0], resource[4]) = (1, 64);
        let mut desc = [0u32; 16];
        (desc[1], desc[2], desc[6], desc[7]) = (1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, 16, 4);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        let view = View::Placed(false, resource, desc);
        assert_eq!(buffer_range(&view), Some((address, 64)));
        assert_eq!(heap.address_at(first), Some(0), "not made yet");
        heap.make_at(first, &view);
        assert_eq!(heap.address_at(first), Some(address));
        assert_eq!(heap.address_at(first + 1), Some(0), "not made yet");
        // The last dynamic slot has its entry too.
        heap.make_at(end - 1, &view);
        assert_eq!(heap.address_at(end - 1), Some(address));
        // A slot given back keeps its entry until it is made again; made
        // null, it has its entry cleared.
        heap.release(first, 2);
        assert_eq!(
            heap.address_at(first),
            Some(address),
            "given back, not made"
        );
        heap.make_null(first);
        assert_eq!(heap.address_at(first), Some(0));
        assert_eq!(heap.address_at(end - 1), Some(address), "still made");
        // Raw views of ours (root views, host copies' view pairs) in dynamic
        // slots clear what an earlier table entered there.
        heap.make_at(first, &view);
        heap.raw_view_at(first, None, 0, 0, false);
        assert_eq!(heap.address_at(first), Some(0), "a root view");
        heap.make_at(first, &view);
        heap.make_at(first + 1, &view);
        heap.raw_pair_at(first, None, 0);
        assert_eq!(heap.address_at(first), Some(0), "a view pair");
        assert_eq!(heap.address_at(first + 1), Some(0), "a view pair");
        // Slots past the dynamic ones (host copies' view pairs) have no
        // entry, and nothing is written past the last: a mark there (the
        // buffer's allocation reaches past it) stays.
        assert_eq!(heap.address_at(end), None);
        let host = host().unwrap();
        let (buffer, data) = heap.addresses.get().unwrap().as_ref().unwrap().mapped();
        // SAFETY: plain queries of a live buffer.
        let (desc, size) = unsafe {
            let desc = buffer.GetDesc();
            let info = host.device.GetResourceAllocationInfo(0, &[desc]);
            (desc, info.SizeInBytes)
        };
        assert_eq!(desc.Width, 8 * u64::from(end));
        assert!(size >= desc.Width + 8, "room for the mark");
        let past = (data + 8 * end as usize) as *mut u64;
        let mark = 0x5a5a_5a5a_5a5a_5a5a;
        // SAFETY: mapped memory of the buffer's allocation (checked).
        unsafe { past.write_unaligned(mark) };
        heap.note_address(end, Some(&view));
        // SAFETY: as above.
        assert_eq!(unsafe { past.read_unaligned() }, mark);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        // Every index a translated shader can reach exists (clamped below
        // `HEAP_VIEWS`, `HEAP_SAMPLERS`), where the host takes a heap that
        // large; a host that caps its heaps (NVIDIA) still gets the larger
        // dynamic range.
        // SAFETY: plain queries of live heaps.
        unsafe {
            assert!(
                heap.heap.GetDesc().NumDescriptors
                    >= victory_shader::emit::view_heap_limit()
                        .unwrap_or(victory_shader::emit::HEAP_VIEWS)
            );
            assert_eq!(heap.dynamic, DYNAMIC);
            let samplers = HostHeap::new(true, 128).unwrap();
            assert!(samplers.heap.GetDesc().NumDescriptors >= victory_shader::emit::HEAP_SAMPLERS);
        }
        // A slot that stops viewing a resource leaves it alive until the GPU
        // work in flight has run (a read of a freed host texture otherwise).
        // SAFETY: a plain committed buffer; one reference taken and given back.
        unsafe {
            let host = host().unwrap();
            let mut buffer: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .unwrap();
            let buffer = buffer.unwrap();
            let references = |b: &ID3D12Resource| {
                let raw = windows::core::Interface::as_raw(b);
                let unknown = &*(raw as *const *const windows::core::IUnknown_Vtbl);
                ((**unknown).AddRef)(raw);
                ((**unknown).Release)(raw)
            };
            let slot = heap.count + 2;
            heap.raw_view_at(slot, Some(&buffer), 0, 256, false);
            assert_eq!(references(&buffer), 2, "ours and the slot's");
            heap.make_null(slot);
            assert!(references(&buffer) >= 2, "the slot's, retired");
            crate::host::release_retired();
            assert_eq!(references(&buffer), 1, "ours");
        }
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    /// With no free run as long as wanted, the longest free one of at
    /// least the least is taken, in one search; none when fewer are free.
    #[test]
    fn a_shorter_run_is_taken_where_the_long_one_is_not_free() {
        let mut slots = Slots::new(300);
        assert_eq!(slots.take(100, 100), Some((0, 100)));
        assert_eq!(slots.take(100, 100), Some((100, 100)));
        assert_eq!(slots.take(100, 100), Some((200, 100)));
        // A capped heap's pool held by lists: holes of 30 and 12.
        slots.release(120, 30);
        slots.release(260, 12);
        assert_eq!(slots.free, 42);
        assert_eq!(slots.take(256, 256), None, "no 256-slot run");
        assert_eq!(slots.take(256, 40), None, "no run of 40 either");
        assert_eq!(slots.take(256, 1), Some((120, 30)), "the longest hole");
        assert_eq!(slots.take(256, 1), Some((260, 12)));
        assert_eq!(slots.free, 0);
        assert_eq!(slots.take(1, 1), None);
        // A run as long as wanted is still taken whole, and first.
        slots.release(0, 300);
        assert_eq!(slots.free, 300);
        assert_eq!(slots.take(256, 1), Some((0, 256)));
        assert_eq!(slots.take(256, 1), Some((256, 44)));
        // A failed search is remembered: no run of 40 is searched for until
        // a release joins one, here the hole at 120 with the 10 after it.
        slots.release(0, 300);
        assert_eq!(slots.take(300, 300), Some((0, 300)));
        slots.release(120, 30);
        slots.release(200, 10);
        assert_eq!(slots.take(40, 40), None);
        assert_eq!(slots.longest, 30);
        assert_eq!(slots.take(40, 40), None, "not walked again");
        slots.release(150, 10);
        assert_eq!(slots.longest, 40);
        assert_eq!(slots.take(256, 1), Some((120, 40)));
        assert_eq!(slots.take(256, 1), Some((200, 10)));
        assert_eq!(slots.take(1, 1), None);
        // Wanting more than the pool holds takes the pool.
        let mut small = Slots::new(16);
        assert_eq!(small.take(256, 1), Some((0, 16)));
    }

    /// What is held long (title memory copies' view pairs) is taken from
    /// the top, away from the runs taken from the cursor on; the fewest
    /// free since the last look is kept.
    #[test]
    fn long_held_pairs_come_from_the_top() {
        let mut slots = Slots::new(64);
        assert_eq!(slots.take_top(2), Some(62));
        assert_eq!(slots.take_top(2), Some(60));
        assert_eq!(slots.take(16, 16), Some((0, 16)), "runs still start at the bottom");
        slots.release(60, 2);
        assert_eq!(slots.take_top(2), Some(60), "a pair given back is the top's again");
        assert_eq!(slots.take_low(), 44);
        slots.release(0, 16);
        assert_eq!(slots.take_low(), 44, "the fewest since the last look");
        assert_eq!(slots.take_low(), 60, "then from the slots free when last looked");
        // No two free in a row: none.
        let mut full = Slots::new(4);
        assert_eq!(full.take(4, 4), Some((0, 4)));
        full.release(1, 1);
        full.release(3, 1);
        assert_eq!(full.take_top(2), None);
        assert_eq!(full.take_top(1), Some(3));
    }

    /// Slots given back are free again, counted once however often.
    #[test]
    fn released_slots_are_counted_once() {
        let mut slots = Slots::new(64);
        let (first, n) = slots.take(32, 32).unwrap();
        slots.release(first, n);
        slots.release(first, n);
        slots.release(60, 100);
        assert_eq!(slots.free, 64);
        assert_eq!(slots.take(64, 64), Some((0, 64)));
    }

    #[test]
    fn dynamic_search_can_find_a_run_across_the_cursor() {
        let Some(heap) = HostHeap::new(false, 4) else {
            eprintln!("no host descriptor heap; skipped");
            return;
        };
        let first = heap.dynamic(1).unwrap();
        let left = heap.dynamic(2).unwrap();
        let right = heap.dynamic(2).unwrap();
        let rest = heap.dynamic(heap.dynamic - 5).unwrap();
        assert_eq!((first, left, right, rest), (4, 5, 7, 9));

        // Reallocating the left half puts the cursor back in the middle.
        heap.release(left, 2);
        assert_eq!(heap.dynamic(2), Some(left));
        heap.release(left, 2);
        heap.release(right, 2);
        assert_eq!(heap.dynamic(4), Some(left));
        assert_eq!(heap.dynamic(1), None);

        // A free slot at each physical end does not form a two-slot range.
        heap.release(first, 1);
        let last = first + heap.dynamic - 1;
        heap.release(last, 1);
        assert_eq!(heap.dynamic(2), None);
    }

    #[test]
    fn unsupported_typed_uav_format_is_refused() {
        let Some(host) = host() else {
            return;
        };
        let mut tex: Option<ID3D12Resource> = None;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R16_TYPELESS,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        // SAFETY: plain committed typeless texture on the default heap.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    None,
                    &mut tex,
                )
                .unwrap();
        }
        let tex = tex.unwrap();
        const B5G5R5A1_UNORM: u32 = 86;
        let mut fmt = DXGI_FORMAT(B5G5R5A1_UNORM as i32);
        // Unsupported typed UAV format must be refused even when castable.
        assert_eq!(
            super::fits(&tex, &[B5G5R5A1_UNORM], Some(&mut fmt), true),
            super::typed_store(B5G5R5A1_UNORM)
        );
        // SRV with the same castable format remains allowed.
        let mut srv_fmt = DXGI_FORMAT(B5G5R5A1_UNORM as i32);
        assert!(super::fits(&tex, &[B5G5R5A1_UNORM], Some(&mut srv_fmt), false));
    }

    #[test]
    fn supported_typed_uav_formats_are_accepted() {
        let Some(host) = host() else {
            return;
        };
        let make_tex = |format: DXGI_FORMAT| {
            let mut tex: Option<ID3D12Resource> = None;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 4,
                Height: 4,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            // SAFETY: plain committed texture on the default heap.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_DEFAULT,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        None,
                        &mut tex,
                    )
                    .unwrap();
            }
            tex.unwrap()
        };

        let typeless = make_tex(DXGI_FORMAT_R8G8B8A8_TYPELESS);
        assert!(!super::fits(&typeless, &[], None, true));
        for format in [DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UINT] {
            let mut fmt = format;
            assert!(super::fits(&typeless, &[], Some(&mut fmt), true));
            assert_eq!(fmt, format);
        }
        let mut srgb = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        assert!(super::fits(&typeless, &[], Some(&mut srgb), true));
        assert_eq!(srgb, DXGI_FORMAT_R8G8B8A8_UNORM);

        for format in [DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R16G16B16A16_FLOAT] {
            let typed = make_tex(format);
            assert!(super::fits(&typed, &[], None, true));
            let mut fmt = format;
            assert!(super::fits(&typed, &[], Some(&mut fmt), true));
        }
    }
}
