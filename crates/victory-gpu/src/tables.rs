//! The host's lookup tables that translated shaders read as root views on
//! nearly every memory access: a view heap's page table (`crate::memory`),
//! its view-stride table, view-mark table and view addresses
//! (`crate::descriptor::host::HostHeap`). The host writes them from the
//! CPU, title threads among it (heap slots are rewritten while the GPU
//! runs).
//!
//! By default each table is an upload buffer: system memory, so every
//! lookup crosses the bus, and most lookups are dependent (a page-table
//! entry needs its level-1 entry first). With `VICTORY_TABLES_VRAM` set,
//! tables live in video memory instead ([`Mode`]):
//! - `=1`: in a GPU upload heap where the host has one (the CPU writes
//!   video memory in place through the bus, no copies), else as `=copy`;
//! - `=copy`: in a default heap, with the upload buffer kept as the CPU's
//!   copy. What the CPU wrote since the last submission is copied over at
//!   the start of each submission ([`copies`]), in [`CHUNK`]-byte chunks:
//!   shaders read what the CPU wrote before the submission, as they did.
//!
//! What a title thread writes while a submission runs reaches shaders at
//! the next submission in `=copy` mode, at once (as before) otherwise.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;

use crate::host::Host;

/// Where a table lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// An upload buffer, in system memory (the default).
    System,
    /// A GPU upload heap: video memory the CPU writes in place.
    Upload,
    /// A default heap, copied into from an upload buffer at submissions.
    Copy,
}

/// The bytes of a table one dirty bit stands for in [`Mode::Copy`].
pub(crate) const CHUNK: usize = 256;

/// Dirty chunks this close together are copied as one run: fewer copy
/// commands for a few bytes more.
const GAP: usize = 16;

/// The mode `VICTORY_TABLES_VRAM` asks for on this host, decided once.
pub(crate) fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| {
        if !crate::switch::on("VICTORY_TABLES_VRAM") {
            return Mode::System;
        }
        let copy =
            std::env::var("VICTORY_TABLES_VRAM").is_ok_and(|v| v.eq_ignore_ascii_case("copy"));
        let mode = match crate::host::host() {
            Some(host) if !copy && upload_heaps(host) => Mode::Upload,
            _ => Mode::Copy,
        };
        tracing::info!(
            ?mode,
            gpu_upload_heaps = crate::host::host().is_some_and(upload_heaps),
            "host lookup tables in video memory (VICTORY_TABLES_VRAM)"
        );
        mode
    })
}

/// Whether the host has GPU upload heaps (video memory the CPU maps).
pub(crate) fn upload_heaps(host: &Host) -> bool {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    // SAFETY: the query writes its own structure of the given size.
    unsafe {
        host.device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS16,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
            .is_ok()
            && options.GPUUploadHeapSupported.as_bool()
    }
}

/// A lookup table: a buffer shaders read as a root view, and the mapping
/// the host writes it through.
pub(crate) struct Table(Arc<Inner>);

struct Inner {
    /// What shaders read (root views name it; lists keep it).
    gpu: ID3D12Resource,
    /// [`Mode::Copy`]: the upload buffer the host writes, copied into `gpu`.
    staging: Option<ID3D12Resource>,
    /// The host's mapping: of `staging`, else of `gpu`.
    data: usize,
    bytes: usize,
    mode: Mode,
    /// [`Mode::Copy`]: a bit per [`CHUNK`] written since the last copy.
    dirty: Box<[AtomicU64]>,
    /// Whether any bit of `dirty` is set.
    any: AtomicBool,
    /// Bytes copied into `gpu` so far.
    copied: AtomicU64,
}

// SAFETY: `data` is a persistent mapping, written through `Table`'s
// bounds-checked writes only; D3D12 resources are free-threaded.
unsafe impl Send for Inner {}
// SAFETY: as above.
unsafe impl Sync for Inner {}

/// [`Mode::Copy`] tables, for [`copies`].
static LIVE: Mutex<Vec<Weak<Inner>>> = Mutex::new(Vec::new());

/// The kinds of table logged so far (once each, with the first one's size).
static LOGGED: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

/// Bytes and copies of tables copied at submissions, since the last call
/// of [`take_copied`].
static COPIED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

impl Table {
    /// A table of `bytes` (zeros) where [`mode`] puts it.
    pub(crate) fn new(bytes: usize, name: &'static str) -> Option<Table> {
        let host = crate::host::host()?;
        let mode = mode();
        let made = Table::with_mode(host, bytes, name, mode);
        if made.is_none() && mode != Mode::System {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    ?mode,
                    name,
                    bytes,
                    "no host lookup table in video memory: in system memory"
                )
            });
            return Table::with_mode(host, bytes, name, Mode::System);
        }
        if mode != Mode::System {
            let mut logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
            if !logged.contains(&name) {
                logged.push(name);
                tracing::info!(
                    ?mode,
                    name,
                    mb = bytes as f64 / 1e6,
                    "host lookup table in video memory (one per view heap)"
                );
            }
        }
        made
    }

    /// A table of `bytes` (zeros) in `mode`.
    pub(crate) fn with_mode(
        host: &Host,
        bytes: usize,
        name: &'static str,
        mode: Mode,
    ) -> Option<Table> {
        let bytes = bytes.max(4);
        let (gpu, staging, data) = match mode {
            Mode::System => {
                let (buffer, data) = host.upload_buffer(bytes as u64)?;
                (buffer, None, data)
            }
            Mode::Upload => {
                let (buffer, data) = buffer(host, D3D12_HEAP_TYPE_GPU_UPLOAD, bytes, true)?;
                (buffer, None, data)
            }
            Mode::Copy => {
                let (buffer, _) = buffer(host, D3D12_HEAP_TYPE_DEFAULT, bytes, false)?;
                let (staging, data) = host.upload_buffer(bytes as u64)?;
                crate::host::name(&staging, || format!("{name} (the CPU's copy)"));
                (buffer, Some(staging), data)
            }
        };
        crate::host::name(&gpu, || name.into());
        let words = if mode == Mode::Copy {
            bytes.div_ceil(CHUNK).div_ceil(64)
        } else {
            0
        };
        let inner = Arc::new(Inner {
            gpu,
            staging,
            data: data as usize,
            bytes,
            mode,
            dirty: (0..words).map(|_| AtomicU64::new(0)).collect(),
            any: AtomicBool::new(false),
            copied: AtomicU64::new(0),
        });
        if mode == Mode::Copy {
            let mut live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
            live.retain(|t| t.strong_count() > 0);
            live.push(Arc::downgrade(&inner));
        }
        Some(Table(inner))
    }

    /// The buffer shaders read, for root views (lists keep it).
    pub(crate) fn resource(&self) -> &ID3D12Resource {
        &self.0.gpu
    }

    /// Its host GPU address.
    pub(crate) fn gpu(&self) -> u64 {
        // SAFETY: a live buffer.
        unsafe { self.0.gpu.GetGPUVirtualAddress() }
    }

    #[cfg_attr(not(test), expect(dead_code, reason = "tests read it"))]
    pub(crate) fn mode(&self) -> Mode {
        self.0.mode
    }

    /// Bytes copied into the table at submissions so far ([`Mode::Copy`]).
    #[cfg_attr(not(test), expect(dead_code, reason = "tests read it"))]
    pub(crate) fn copied(&self) -> u64 {
        self.0.copied.load(Ordering::Relaxed)
    }

    /// Writes `dwords` from dword `at` on; nothing past the table's end.
    pub(crate) fn write(&self, at: usize, dwords: &[u32]) {
        let Some(end) = at
            .checked_add(dwords.len())
            .filter(|&end| end * 4 <= self.0.bytes)
        else {
            debug_assert!(false, "a write past a lookup table's end");
            return;
        };
        if dwords.is_empty() {
            return;
        }
        // SAFETY: inside the mapping, checked above.
        unsafe {
            let to = (self.0.data as *mut u32).add(at);
            if let [one] = dwords {
                to.write_volatile(*one);
            } else {
                std::ptr::copy_nonoverlapping(dwords.as_ptr(), to, dwords.len());
            }
        }
        self.0.wrote(at * 4, end * 4);
    }

    /// Fills the whole table with `byte`.
    pub(crate) fn fill(&self, byte: u8) {
        // SAFETY: the whole mapping.
        unsafe { std::ptr::write_bytes(self.0.data as *mut u8, byte, self.0.bytes) };
        self.0.wrote(0, self.0.bytes);
    }
}

impl Inner {
    /// Bytes `[start, end)` changed: [`Mode::Copy`] copies their chunks at
    /// the next submission.
    fn wrote(&self, start: usize, end: usize) {
        if self.mode != Mode::Copy || start >= end {
            return;
        }
        set(&self.dirty, start / CHUNK, (end - 1) / CHUNK);
        self.any.store(true, Ordering::Release);
    }
}

/// Sets bits `first..=last` of `bits`.
fn set(bits: &[AtomicU64], first: usize, last: usize) {
    let (w0, w1) = (first / 64, last / 64);
    for (w, word) in bits.iter().enumerate().take(w1 + 1).skip(w0) {
        let lo = if w == w0 { first % 64 } else { 0 };
        let hi = if w == w1 { last % 64 } else { 63 };
        let mask = (u64::MAX >> (63 - hi)) & (u64::MAX << lo);
        word.fetch_or(mask, Ordering::Release);
    }
}

/// The runs of set bits in `words` (first, one past the last), runs at
/// most `gap` clear bits apart joined.
fn runs(words: &[u64], gap: usize) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (w, &word) in words.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let start = w * 64 + bits.trailing_zeros() as usize;
            // The run inside this word.
            let ones = (bits >> (start % 64)).trailing_ones() as usize;
            let end = start + ones;
            bits = if end.is_multiple_of(64) {
                0
            } else {
                bits & (u64::MAX << (end % 64))
            };
            match runs.last_mut() {
                Some(last) if start <= last.1 + gap => last.1 = end,
                _ => runs.push((start, end)),
            }
        }
    }
    runs
}

/// A committed buffer of `bytes` in heap `kind`, in the common state, and
/// its mapping if `mapped`.
fn buffer(
    host: &Host,
    kind: D3D12_HEAP_TYPE,
    bytes: usize,
    mapped: bool,
) -> Option<(ID3D12Resource, *mut u8)> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: kind,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    let mut data = std::ptr::null_mut();
    // SAFETY: plain creation (and mapping) of a buffer.
    let made = unsafe {
        host.device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
            .and_then(|()| {
                let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                if mapped {
                    b.Map(0, None, Some(&mut data))
                } else {
                    Ok(())
                }
            })
    };
    match made {
        Ok(()) => Some((buffer?, data.cast())),
        Err(e) => {
            tracing::warn!(error = %e, bytes, kind = kind.0, "no host lookup table buffer");
            None
        }
    }
}

/// A submission's copies into [`Mode::Copy`] tables: the list to run before
/// the submission's own, and the tables, held until it has run.
pub(crate) struct Copies {
    pub(crate) list: ID3D12CommandList,
    _tables: Vec<Arc<Inner>>,
}

/// The allocator and list the copies are recorded in, reused: one
/// submission runs at a time, and each is waited for
/// ([`crate::host::Host::execute`]).
static COPIER: Mutex<Option<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>> =
    Mutex::new(None);

/// At a submission, with the last one run: a list copying what the CPU
/// wrote into [`Mode::Copy`] tables since the last submission, for the
/// start of this one; None if nothing changed.
pub(crate) fn copies(host: &Host) -> Option<Copies> {
    let tables: Vec<Arc<Inner>> = {
        let live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
        live.iter()
            .filter_map(Weak::upgrade)
            .filter(|t| t.any.load(Ordering::Acquire))
            .collect()
    };
    if tables.is_empty() {
        return None;
    }
    let mut copier = COPIER.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: plain object creation; the allocator's last list has run.
    let list = unsafe {
        match copier.as_ref() {
            Some((allocator, list)) => allocator
                .Reset()
                .and_then(|()| list.Reset(allocator, None))
                .map(|()| list.clone()),
            None => host
                .device
                .CreateCommandAllocator::<ID3D12CommandAllocator>(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .and_then(|allocator| {
                    let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(
                        0,
                        D3D12_COMMAND_LIST_TYPE_DIRECT,
                        &allocator,
                        None,
                    )?;
                    crate::host::name(&list, || "host lookup table copies".into());
                    *copier = Some((allocator, list.clone()));
                    Ok(list)
                }),
        }
    };
    let list = match list {
        Ok(list) => list,
        Err(e) => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(error = %e, "no list for host lookup table copies: shaders read older tables"));
            return None;
        }
    };
    // Each table with its runs of dirty chunks.
    let mut copied = Vec::with_capacity(tables.len());
    for t in tables {
        t.any.store(false, Ordering::Release);
        let words: Vec<u64> = t
            .dirty
            .iter()
            .map(|w| w.swap(0, Ordering::AcqRel))
            .collect();
        let runs = runs(&words, GAP);
        if !runs.is_empty() {
            copied.push((t, runs));
        }
    }
    if copied.is_empty() {
        // SAFETY: an open list of ours.
        let _ = unsafe { list.Close() };
        return None;
    }
    let to = |after: bool| -> Vec<D3D12_RESOURCE_BARRIER> {
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
            | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        let (before, after) = if after {
            (D3D12_RESOURCE_STATE_COPY_DEST, read)
        } else {
            // Buffers are back in the common state after each submission.
            (D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)
        };
        copied
            .iter()
            .map(|(t, _)| crate::host::transition(&t.gpu, before, after))
            .collect()
    };
    let (mut bytes, mut commands) = (0u64, 0u64);
    // SAFETY: an open list; both buffers of each table live until it has
    // run (`Copies` holds them).
    unsafe {
        list.ResourceBarrier(&to(false));
        for (t, runs) in &copied {
            let Some(staging) = &t.staging else { continue };
            let mut table_bytes = 0;
            for &(first, end) in runs {
                let start = first * CHUNK;
                let len = (end * CHUNK).min(t.bytes).saturating_sub(start);
                if len == 0 {
                    continue;
                }
                list.CopyBufferRegion(&t.gpu, start as u64, staging, start as u64, len as u64);
                table_bytes += len as u64;
                commands += 1;
            }
            t.copied.fetch_add(table_bytes, Ordering::Relaxed);
            bytes += table_bytes;
        }
        list.ResourceBarrier(&to(true));
    }
    // SAFETY: an open list of ours.
    if let Err(e) = unsafe { list.Close() } {
        tracing::warn!(error = %e, "host lookup table copies not recorded");
        for (t, runs) in &copied {
            for &(first, end) in runs {
                set(&t.dirty, first, end - 1);
            }
            t.any.store(true, Ordering::Release);
        }
        return None;
    }
    COPIED[0].fetch_add(bytes, Ordering::Relaxed);
    COPIED[1].fetch_add(commands, Ordering::Relaxed);
    Some(Copies {
        list: list.cast().ok()?,
        _tables: copied.into_iter().map(|(t, _)| t).collect(),
    })
}

/// Bytes and copy commands of table copies since the last call.
pub(crate) fn take_copied() -> (u64, u64) {
    (
        COPIED[0].swap(0, Ordering::Relaxed),
        COPIED[1].swap(0, Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_join_close_chunks_and_split_far_ones() {
        assert_eq!(runs(&[0], GAP), []);
        assert_eq!(runs(&[0b1011], 0), [(0, 2), (3, 4)]);
        assert_eq!(runs(&[0b1011], 1), [(0, 4)]);
        // A run across a word boundary stays one run.
        assert_eq!(runs(&[1 << 63, 1], 0), [(63, 65)]);
        assert_eq!(runs(&[u64::MAX, u64::MAX], 0), [(0, 128)]);
        // Far apart: two runs.
        assert_eq!(runs(&[1, 0, 1], GAP), [(0, 1), (128, 129)]);
    }

    #[test]
    fn set_marks_exactly_the_chunks_written() {
        let bits: Vec<AtomicU64> = (0..3).map(|_| AtomicU64::new(0)).collect();
        set(&bits, 62, 66);
        let words: Vec<u64> = bits.iter().map(|w| w.load(Ordering::Relaxed)).collect();
        assert_eq!(runs(&words, 0), [(62, 67)]);
        set(&bits, 0, 191);
        assert!(bits.iter().all(|w| w.load(Ordering::Relaxed) == u64::MAX));
        // Past the end: clipped.
        let small: Vec<AtomicU64> = (0..1).map(|_| AtomicU64::new(0)).collect();
        set(&small, 60, 200);
        assert_eq!(small[0].load(Ordering::Relaxed), 0b1111 << 60);
    }

    /// Copies `n` dwords of table `t0` from dword `from` into `u0`.
    const READ: &str = r#"
#define RS "SRV(t0), UAV(u0), RootConstants(num32BitConstants=2, b0)"
ByteAddressBuffer table : register(t0);
RWByteAddressBuffer dst : register(u0);
cbuffer c : register(b0) { uint from; uint n; }
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x < n) {
        dst.Store(id.x * 4, table.Load((from + id.x) * 4));
    }
}
"#;

    /// Shaders read a table the same in every mode: what the CPU wrote
    /// before a list was recorded, after it was recorded (before its
    /// submission: the page table is rewritten then), and in the
    /// submission right after another (a table update within a frame);
    /// `Mode::Copy` copies only the chunks written.
    #[test]
    fn shaders_read_what_the_cpu_wrote_in_every_mode() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let (root, pso) = crate::memory::tests::pipeline(READ);
        let mut modes = vec![Mode::System, Mode::Copy];
        if upload_heaps(host) {
            modes.push(Mode::Upload);
        } else {
            eprintln!("no GPU upload heaps on this host: that mode is not tested");
        }
        const DWORDS: usize = 64 * 1024;
        const N: u32 = 256;
        let mut seen = Vec::new();
        for mode in modes {
            let table = Table::with_mode(host, DWORDS * 4, "test table", mode).expect("a table");
            assert_eq!(table.mode(), mode);
            let pattern: Vec<u32> = (0..DWORDS as u32)
                .map(|i| i.wrapping_mul(2_654_435_761))
                .collect();
            table.write(0, &pattern);
            let (out, out_cpu) = host.shared_buffer(u64::from(N) * 4).unwrap();
            // SAFETY: a live buffer.
            let out_gpu = unsafe { out.GetGPUVirtualAddress() };
            let list = host.new_list().unwrap();
            let run = |from: u32, before: &dyn Fn()| -> Vec<u32> {
                let constants = [from, N];
                list.keep(table.resource());
                // SAFETY: an open list, live objects of ours.
                unsafe {
                    list.list.SetComputeRootSignature(&root);
                    list.list.SetComputeRootShaderResourceView(0, table.gpu());
                    list.list.SetComputeRootUnorderedAccessView(1, out_gpu);
                    list.list
                        .SetComputeRoot32BitConstants(2, 2, constants.as_ptr().cast(), 0);
                    list.list.SetPipelineState(&pso);
                    list.list.Dispatch(N.div_ceil(64), 1, 1);
                }
                list.close();
                before();
                host.execute(&[&list]);
                list.reset();
                // SAFETY: the mapped output, the GPU done with it.
                unsafe { std::slice::from_raw_parts(out_cpu.cast::<u32>(), N as usize).to_vec() }
            };
            let mut expected = pattern.clone();
            let from = 40_000u32;
            let at = |e: &[u32]| e[from as usize..(from + N) as usize].to_vec();
            assert_eq!(
                run(from, &|| {}),
                at(&expected),
                "{mode:?}: the first submission"
            );
            // Written after the list was recorded, before it is submitted.
            let update = [0xA0A0_0001, 0xA0A0_0002, 0xA0A0_0003];
            expected[from as usize + 7..from as usize + 10].copy_from_slice(&update);
            let copied = table.copied();
            assert_eq!(
                run(from, &|| table.write(from as usize + 7, &update)),
                at(&expected),
                "{mode:?}: after recording"
            );
            if mode == Mode::Copy {
                assert_eq!(table.copied() - copied, CHUNK as u64, "one chunk copied");
            }
            // Submission after submission, a dword each time (one frame's
            // updates), the last dword of the table among them.
            for k in 0..3u32 {
                let i = from as usize + 100 + k as usize;
                expected[i] = 0xB0B0_0000 + k;
                table.write(i, &[0xB0B0_0000 + k]);
                assert_eq!(run(from, &|| {}), at(&expected), "{mode:?}: update {k}");
            }
            let last = DWORDS - 1;
            table.write(last, &[0xC0C0_C0C0]);
            expected[last] = 0xC0C0_C0C0;
            let tail = run(DWORDS as u32 - N, &|| {});
            assert_eq!(
                tail,
                expected[DWORDS - N as usize..],
                "{mode:?}: the table's end"
            );
            // Nothing written: nothing copied.
            let copied = table.copied();
            run(0, &|| {});
            assert_eq!(table.copied(), copied, "{mode:?}: nothing to copy");
            seen.push(tail);
        }
        assert!(
            seen.windows(2).all(|w| w[0] == w[1]),
            "every mode reads the same"
        );
    }
}
