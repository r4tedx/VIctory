//! The loop-cut counter (`victory_shader::emit`, "Loop cuts"): with
//! `VICTORY_LOOP_CUTS=1` in a diagnostics build, every lane that leaves a
//! translated loop at the loop limit adds one to a dword of ours, bound as
//! the root UAV `u2` in the memory space of every pipeline. The frame line
//! says how many lanes were cut since the last one, so a run tells whether
//! the limit (`VICTORY_SHADER_LOOP_LIMIT`) is ever reached, and how often.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::host;

/// Our counter: the buffer, its CPU mapping and its GPU address.
struct Counter {
    _buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
}

// SAFETY: the buffer is a COM object made free-threaded by D3D12; `data` is
// its persistent mapping, read with volatile loads only.
unsafe impl Send for Counter {}
// SAFETY: as above.
unsafe impl Sync for Counter {}

fn counter() -> Option<&'static Counter> {
    static COUNTER: OnceLock<Option<Counter>> = OnceLock::new();
    COUNTER
        .get_or_init(|| {
            // 256 bytes: a root UAV's address needs no more than dword
            // alignment, but a whole small buffer keeps it apart.
            let (buffer, data) = host()?.shared_buffer(256)?;
            crate::host::name(&buffer, || "loop-cut counter".into());
            // SAFETY: our mapped buffer of 256 bytes.
            unsafe { std::ptr::write_bytes(data, 0, 256) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some(Counter {
                _buffer: buffer,
                data: data as usize,
                gpu,
            })
        })
        .as_ref()
}

/// The GPU address to bind at a pipeline's loop-cut parameter.
pub(crate) fn address() -> Option<u64> {
    counter().map(|c| c.gpu)
}

/// Lanes cut since the start: the counter is never reset (shaders may be
/// adding to it at any time), so the wraps of its 32 bits are followed.
fn total() -> u64 {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let Some(c) = counter() else {
        return 0;
    };
    // SAFETY: the first dword of our mapped buffer.
    let now = u64::from(unsafe { std::ptr::read_volatile(c.data as *const u32) });
    let last = LAST.load(Ordering::Relaxed);
    let wraps = last >> 32;
    let mut t = (wraps << 32) | now;
    if t < last {
        t += 1 << 32;
    }
    LAST.store(t, Ordering::Relaxed);
    t
}

/// For the frame line: lanes cut since the last call and since the start,
/// while loop cuts are counted.
pub(crate) fn line() {
    if !victory_shader::emit::loop_cuts() {
        return;
    }
    static SEEN: AtomicU64 = AtomicU64::new(0);
    let all = total();
    let new = all - SEEN.swap(all, Ordering::Relaxed);
    tracing::info!(
        limit = victory_shader::emit::loop_limit(),
        "translated loops left at the limit (VICTORY_LOOP_CUTS): {new} lanes since the last line, {all} since the start"
    );
}
