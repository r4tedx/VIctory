//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with (made in time or
//!    not) are listed by a key known when the title creates them (their
//!    shader containers and graphics state); the next run hands those to
//!    the workers at creation, during loading. A full list keeps the
//!    latest run's first.
//!
//! 3. The pre-build (`victory translate --cache --pipelines`) lists the
//!    title pipelines it has put in the library, by the same key, so a
//!    second pre-build skips them without translating or loading anything.
//!    The list holds only for the library bytes it was saved with and for
//!    what names the pipelines (the translator, this module and `host`).
//!
//! The library is free-threaded but for loads of one pipeline on several
//! threads (`ID3D12Device1::CreatePipelineLibrary`, "Thread Safety"), so
//! pipelines of different names load at the same time. A load is not always
//! quick: the library of an AMD driver holds about 140 bytes per pipeline,
//! and what its own cache lacks is compiled inside the load, for seconds.
//! Loaded one at a time, 650 pipelines kept the eight workers to one core
//! for minutes, while each title frame could wait out its 500 ms for
//! them (`make`). Stores and serialising keep the lock: the size asked for
//! must still be right when the bytes are written.
//!
//! The library's file is mapped, not read into memory ([`Blob`]): the host
//! points into the bytes it was made from for as long as it lives, and read
//! into a buffer they were the process's own memory for the whole run
//! (1,352 MB in a world run on fxceless/victory-fork). Mapped, the pages are
//! the file's: only those the host reads are in memory, and the system can
//! drop them without writing them anywhere. A save serialises the new
//! library straight into a mapping of its new file.
//!
//! The mapped file's handle stays open for as long as the mapping lives. A
//! save renames the new library over the file this run has mapped, and with
//! the handle closed Windows gives up the old file's contents: pages of the
//! mapping not in memory at that moment read as zeros from then on, which
//! the host reads as stored pipelines (the same fork: device removals right
//! after a save). With the handle open they keep reading the old file
//! (`a_replaced_library_file_still_reads_from_the_disk`). Where a system
//! refuses to replace a mapped file, the new library waits beside it
//! ([`NEXT_FILE`]) and takes its place at the next start.
//! `VICTORY_LIBRARY_MAP=0`: read and written through buffers, as before.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);

/// Whether the library's file is mapped ([`Blob`]) and saved through a
/// mapping. On by default; `VICTORY_LIBRARY_MAP=0` uses buffers, as before.
fn mapped() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_MAP").is_none_or(|v| v != "0"))
}

/// A whole file mapped into memory: read-only (the library's file) or
/// writable (the file a save fills). Unmapped when dropped.
struct Mapped {
    at: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
    /// The file, for a mapping made by [`Mapped::open`]: open until after
    /// the view is unmapped, so that a file a rename took the name from
    /// keeps its contents for the view (see the module's notes).
    _file: Option<std::fs::File>,
}

impl Mapped {
    /// Maps `file` (`len` bytes, not zero) for reading, or for reading and
    /// writing.
    fn map(file: &std::fs::File, len: usize, write: bool) -> Option<Mapped> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, FILE_MAP_WRITE, MapViewOfFile, PAGE_READONLY,
            PAGE_READWRITE,
        };
        if len == 0 {
            return None;
        }
        // SAFETY: a file we hold open; the view keeps the mapping alive
        // after its handle is closed.
        let at = unsafe {
            let mapping = CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                if write { PAGE_READWRITE } else { PAGE_READONLY },
                0,
                0,
                std::ptr::null(),
            );
            if mapping.is_null() {
                return None;
            }
            let access = if write { FILE_MAP_WRITE } else { FILE_MAP_READ };
            let at = MapViewOfFile(mapping, access, 0, 0, 0);
            CloseHandle(mapping);
            at
        };
        (!at.Value.is_null()).then_some(Mapped {
            at,
            len,
            _file: None,
        })
    }

    /// The file at `path`, mapped for reading and held open; `None` if it is
    /// not there, is empty or cannot be mapped.
    fn open(path: &std::path::Path) -> Option<Mapped> {
        use std::os::windows::fs::OpenOptionsExt;
        // Others may read it, and rename or delete it (a save renames the
        // new library over it), but not write it: a write in place would
        // change the bytes under the host while it reads them, which is
        // how the mapping removed the device on the fork. Opening it to
        // write or truncate fails with a sharing violation meanwhile.
        const FILE_SHARE_READ: u32 = 1;
        const FILE_SHARE_DELETE: u32 = 4;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_DELETE)
            .open(path)
            .ok()?;
        let len = usize::try_from(file.metadata().ok()?.len()).ok()?;
        let mut mapped = Mapped::map(&file, len, false)?;
        mapped._file = Some(file);
        Some(mapped)
    }

    /// A new file of `len` bytes at `path` (replacing what is there),
    /// mapped for writing.
    fn create(path: &std::path::Path, len: usize) -> Option<Mapped> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .ok()?;
        file.set_len(len as u64).ok()?;
        Mapped::map(&file, len, true)
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the view holds `len` readable bytes until dropped.
        unsafe { std::slice::from_raw_parts(self.at.Value.cast(), self.len) }
    }

    /// Only for a view made by [`Mapped::create`].
    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: a writable view of `len` bytes, ours alone.
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: our view.
        unsafe { windows_sys::Win32::System::Memory::UnmapViewOfFile(self.at) };
    }
}

/// The bytes a library is made from: the host reads them for as long as
/// the library lives, so they are kept beside it.
enum Blob {
    /// No file (an empty library).
    None,
    /// The file read into memory (`VICTORY_LIBRARY_MAP=0`, tests, or a file
    /// that could not be mapped).
    Held(Vec<u8>),
    /// The file itself, mapped.
    Mapped(Mapped),
}

impl Blob {
    /// The library file in `dir`: a library a save left beside it
    /// ([`NEXT_FILE`]) takes its place first.
    fn of(dir: &std::path::Path) -> Blob {
        remove_abandoned(dir);
        let path = dir.join(LIBRARY_FILE);
        let next = dir.join(NEXT_FILE);
        if next.exists() && std::fs::rename(&next, &path).is_err() {
            tracing::warn!(
                "the host pipeline library saved last could not take the old one's place"
            );
        }
        if mapped()
            && let Some(file) = Mapped::open(&path)
        {
            return Blob::Mapped(file);
        }
        match std::fs::read(&path) {
            Ok(bytes) if !bytes.is_empty() => Blob::Held(bytes),
            _ => Blob::None,
        }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Blob::None => &[],
            Blob::Held(bytes) => bytes,
            Blob::Mapped(file) => file.bytes(),
        }
    }
}

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Blob,
    /// Pipelines stored since the last save.
    added: u64,
    /// Title pipelines the pre-build put in the library or found there, by
    /// key, and whether that list changed since the last save.
    prebuilt: HashSet<String>,
    prebuilt_changed: bool,
}

// SAFETY: D3D12 objects are free-threaded (a pipeline library with the
// exception `NAMES` takes care of).
unsafe impl Send for Library {}

/// Names one thread at a time holds.
struct Names {
    held: Mutex<Option<HashSet<String>>>,
    released: Condvar,
}

/// A name held until this is dropped.
struct Held<'a> {
    names: &'a Names,
    name: &'a str,
}

impl Names {
    const fn new() -> Names {
        Names {
            held: Mutex::new(None),
            released: Condvar::new(),
        }
    }

    /// Holds `name`, once no other thread does.
    fn hold<'a>(&'a self, name: &'a str) -> Held<'a> {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        while held.as_ref().is_some_and(|h| h.contains(name)) {
            held = self.released.wait(held).unwrap_or_else(|e| e.into_inner());
        }
        held.get_or_insert_default().insert(name.to_owned());
        Held { names: self, name }
    }
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        let mut held = self.names.held.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = held.as_mut() {
            h.remove(self.name);
        }
        drop(held);
        self.names.released.notify_all();
    }
}

/// Pipelines being loaded, or made and stored.
static NAMES: Names = Names::new();

const LIBRARY_FILE: &str = "host-pipelines.bin";
/// A saved library that could not replace [`LIBRARY_FILE`] while this run
/// had it mapped (a system that refuses to replace a mapped file): the next
/// start renames it over the library first ([`Blob::of`]).
const NEXT_FILE: &str = "host-pipelines.next";
const EARLY_FILE: &str = "early-pipelines.txt";
const PREBUILT_FILE: &str = "prebuilt-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

fn library() -> Option<&'static Mutex<Library>> {
    static LIBRARY: OnceLock<Option<Mutex<Library>>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let read = |file| DIR.get().and_then(|d| std::fs::read(d.join(file)).ok());
            let blob = DIR.get().map_or(Blob::None, |d| Blob::of(d));
            let tag = tag(blob.bytes());
            let mut library = open(&device, blob);
            // The pre-build's list, if it was saved with these bytes.
            if let (Some(l), Some(list)) = (library.as_mut(), read(PREBUILT_FILE))
                && !l._blob.bytes().is_empty()
            {
                l.prebuilt = listed(&list, &tag);
            }
            if library.is_none() {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library.map(Mutex::new)
        })
        .as_ref()
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Blob) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?.cast()
        }
    };
    if !blob.bytes().is_empty() {
        let started = std::time::Instant::now();
        match make(blob.bytes()) {
            Ok(library) => {
                let ms = started.elapsed().as_millis() as u64;
                let (bytes, mapped) = (blob.bytes().len(), matches!(blob, Blob::Mapped(_)));
                tracing::info!(bytes, mapped, ms, "host pipeline library loaded");
                return Some(Library {
                    library,
                    _blob: blob,
                    added: 0,
                    prebuilt: HashSet::new(),
                    prebuilt_changed: false,
                });
            }
            Err(e) => tracing::info!(error = %e, "host pipeline library rejected; starting an empty one"),
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    // The rejected bytes are let go (a mapped file is unmapped) before the
    // empty library is made: nothing reads them any more.
    drop(blob);
    let library = make(&[]).ok()?;
    Some(Library {
        library,
        _blob: Blob::None,
        added: 0,
        prebuilt: HashSet::new(),
        prebuilt_changed: false,
    })
}

/// What the pre-build's list holds for: the library's bytes, and what
/// names its pipelines (the translation, as the shader cache keys it; the
/// host descriptions).
fn tag(blob: &[u8]) -> String {
    let limit = victory_shader::emit::loop_limit().to_le_bytes();
    let translator = victory_shader::SOURCE_HASH.as_bytes();
    let mut h = blake3::Hasher::new();
    for part in [
        blob,
        translator,
        include_bytes!("../shader_cache.rs"),
        &limit,
        include_bytes!("host.rs"),
        include_bytes!("library.rs"),
    ] {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex().to_string()
}

/// The pre-build's list as saved: the tag it holds for, then its keys.
fn list(tag: &str, keys: &HashSet<String>) -> String {
    let keys = keys.iter().map(String::as_str);
    std::iter::once(tag).chain(keys).collect::<Vec<_>>().join("\n")
}

/// The keys of a saved list, if it holds for `tag` (else none).
fn listed(list: &[u8], tag: &str) -> HashSet<String> {
    let list = String::from_utf8_lossy(list);
    let mut lines = list.lines();
    match lines.next() == Some(tag) {
        true => lines.map(str::to_owned).collect(),
        false => HashSet::new(),
    }
}

/// Whether the pre-build has the title pipeline `key` in the library.
pub(crate) fn prebuilt(key: &str) -> bool {
    library().is_some_and(|l| l.lock().unwrap_or_else(|e| e.into_inner()).prebuilt.contains(key))
}

/// The pre-build has the title pipeline `key` in the library now.
pub(crate) fn note_prebuilt(key: String) {
    if let Some(l) = library() {
        let mut l = l.lock().unwrap_or_else(|e| e.into_inner());
        l.prebuilt_changed |= l.prebuilt.insert(key);
    }
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root].into_iter().chain(code.iter().copied()).chain([state.as_bytes()]) {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored. A
/// name asked for by several threads is loaded or made by one at a time
/// (the others then find it in the library); different names overlap.
pub(crate) fn pipeline(
    name: &str,
    load: impl FnOnce(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide = HSTRING::from(name);
    #[cfg(feature = "diagnostics")]
    LAST.set(Some((name.to_owned(), true)));
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    let _held = NAMES.hold(name);
    // The interface alone under the lock: loads take up to seconds. Its
    // blob lives as long as the library, which is never dropped.
    let loads = library
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .library
        .clone();
    if let Ok(pso) = load(&loads, PCWSTR(wide.as_ptr())) {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(pso);
    }
    // Compiled outside the lock: compiles take up to seconds.
    #[cfg(feature = "diagnostics")]
    LAST.set(Some((name.to_owned(), false)));
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a live pipeline; a name already stored (a description that
    // differs) is refused, which leaves the library as it was.
    if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
        l.added += 1;
    }
    Ok(pso)
}

#[cfg(feature = "diagnostics")]
thread_local! {
    /// This thread's last pipeline: its name, and whether it was loaded
    /// (else made), for the pipeline log (diagnostics).
    pub(crate) static LAST: std::cell::Cell<Option<(String, bool)>> =
        const { std::cell::Cell::new(None) };
}

/// When a save during the run is due. Each save serialises and writes the
/// whole library (hundreds of MB to GB), however few pipelines it gained:
/// saved every 20 s, it was rewritten whole three times a minute for a few
/// new pipelines each time. So a run's saves wait for `min` new pipelines
/// (`VICTORY_LIBRARY_SAVE_MIN`, 256), or for `every` to pass since the
/// last save (`VICTORY_LIBRARY_SAVE_SECS`, 60 s) with anything new; the
/// end of a run saves whatever is new ([`save`] `now`). Saves stay
/// [`SavePolicy::FLOOR`] apart, so a load that makes thousands does not
/// save every few hundred.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SavePolicy {
    min: u64,
    every: std::time::Duration,
}

impl SavePolicy {
    /// The least time between two saves of a run, when `every` is longer.
    const FLOOR: std::time::Duration = std::time::Duration::from_secs(20);
    const DEFAULT: SavePolicy = SavePolicy {
        min: 256,
        every: std::time::Duration::from_secs(60),
    };

    /// The policy `VICTORY_LIBRARY_SAVE_MIN` and `VICTORY_LIBRARY_SAVE_SECS`
    /// set (each a whole number; others are ignored).
    fn of(min: Option<&str>, secs: Option<&str>) -> SavePolicy {
        let mut policy = SavePolicy::DEFAULT;
        if let Some(min) = min.and_then(|v| v.trim().parse().ok()) {
            policy.min = min;
        }
        if let Some(secs) = secs.and_then(|v| v.trim().parse().ok()) {
            policy.every = std::time::Duration::from_secs(secs);
        }
        policy
    }

    fn get() -> SavePolicy {
        static POLICY: OnceLock<SavePolicy> = OnceLock::new();
        *POLICY.get_or_init(|| {
            let var = |name| std::env::var(name).ok();
            let policy = SavePolicy::of(
                var("VICTORY_LIBRARY_SAVE_MIN").as_deref(),
                var("VICTORY_LIBRARY_SAVE_SECS").as_deref(),
            );
            if policy != SavePolicy::DEFAULT {
                tracing::info!(
                    min = policy.min,
                    secs = policy.every.as_secs(),
                    "host pipeline library saves"
                );
            }
            policy
        })
    }

    /// Whether a save is due `since` the last (or the first check), with
    /// `added` pipelines stored since and the lists `changed` or not.
    fn due(self, since: std::time::Duration, added: u64, changed: bool) -> bool {
        if added == 0 && !changed {
            return false;
        }
        since >= self.every || (added >= self.min && since >= Self::FLOOR.min(self.every))
    }
}

/// Writes the library and the early list if anything changed since the
/// last save and a save is due ([`SavePolicy`]), or at once (`now`, the
/// end of a run).
pub(crate) fn save(now: bool) {
    /// When the last save started (or the first check, which starts the
    /// time to the first save).
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let (Some(dir), Some(library)) = (DIR.get(), library()) else { return };
    let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        let since = last.get_or_insert_with(std::time::Instant::now).elapsed();
        let changed = l.prebuilt_changed || USED_CHANGED.load(Ordering::Relaxed);
        let due = match now {
            true => l.added > 0 || changed,
            false => SavePolicy::get().due(since, l.added, changed),
        };
        if !due {
            return;
        }
        *last = Some(std::time::Instant::now());
    }
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!(error = %e, "host pipeline library not saved");
        return;
    }
    let stored = l.added;
    let (bytes, prebuilt) = {
        let used_changed = USED_CHANGED.swap(false, Ordering::Relaxed);
        if l.added == 0 && !l.prebuilt_changed {
            if !used_changed {
                return;
            }
            // Only the early list changed: avoid re-serialising and writing a
            // 1+ GiB NVIDIA pipeline library blob during gameplay (#2180).
            (0, None)
        } else {
            // SAFETY: a plain query.
            let size = unsafe { l.library.GetSerializedSize() };
            // SAFETY: a buffer of the size the library asks for.
            let written = write_library(dir, size, |out| unsafe { l.library.Serialize(out) });
            match written {
                Ok(tag) => {
                    l.added = 0;
                    l.prebuilt_changed = false;
                    // Kept beside the bytes it holds for (also when a run adds
                    // pipelines).
                    let prebuilt = (!l.prebuilt.is_empty()).then(|| list(&tag, &l.prebuilt));
                    (size, prebuilt)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "host pipeline library not saved");
                    return;
                }
            }
        }
    };
    drop(l);
    let used = USED.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(Used::list).unwrap_or_default();
    let written =
        write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()).and_then(|()| {
            match &prebuilt {
                Some(list) => write_atomically(&dir.join(PREBUILT_FILE), list.as_bytes()),
                None => Ok(()),
            }
        });
    let (created, distinct) = created();
    let (shaders_cached, shaders_translated) = crate::shader_cache::counts();
    match written {
        Ok(()) => tracing::info!(
            bytes,
            stored,
            early = used.len(),
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            created,
            distinct,
            shaders_cached,
            shaders_translated,
            "host pipeline library saved"
        ),
        Err(e) => tracing::warn!(error = %e, "host pipeline library not saved"),
    }
}

/// Held while a file is written and renamed into place: concurrent worker
/// and shutdown saves do not race on the same staging file or rename.
static WRITING: Mutex<()> = Mutex::new(());

/// A staging file beside `path`, of a name no other write uses.
fn staging(path: &std::path::Path) -> std::path::PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp-{}-{n}", std::process::id()));
    path.with_file_name(name)
}

/// Removes the staging files in `dir` of runs that ended during a save
/// (killed, or crashed): their names carry the process's ID, so no later
/// run reuses or removes them, and the library's are as big as the
/// library. A file of a process still running, or one whose state cannot
/// be told, is left.
fn remove_abandoned(dir: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = name.to_str().and_then(staging_pid) else {
            continue;
        };
        if pid != std::process::id() && ended(pid) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// The process ID in a [`staging`] file's name of one of our files.
fn staging_pid(name: &str) -> Option<u32> {
    let rest = [LIBRARY_FILE, EARLY_FILE, PREBUILT_FILE]
        .iter()
        .find_map(|file| name.strip_prefix(file)?.strip_prefix(".tmp-"))?;
    let (pid, n) = rest.split_once('-')?;
    n.parse::<u64>().ok()?;
    pid.parse().ok()
}

/// Whether process `pid` has ended: no process has that ID ("no such
/// process" is the only refusal taken as that; denied access is a running
/// process of another user), or the process has exited and only a handle
/// to it lives on.
fn ended(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{
        CloseHandle, ERROR_INVALID_PARAMETER, GetLastError, STILL_ACTIVE,
    };
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    // SAFETY: plain queries; the handle is closed.
    unsafe {
        let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if process.is_null() {
            return GetLastError() == ERROR_INVALID_PARAMETER;
        }
        let mut code = 0u32;
        let exited = GetExitCodeProcess(process, &mut code) != 0 && code != STILL_ACTIVE as u32;
        CloseHandle(process);
        exited
    }
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let _guard = WRITING.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = staging(path);
    let res = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path));
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

/// Writes the library's `size` serialised bytes (`serialize` fills them)
/// to a staging file in `dir`, through a mapping of that file where it can
/// (the bytes are then the file's pages, not more of the process's own
/// memory), and renames it over the library's file ([`replace_library`]).
/// Returns the pre-build list's tag of those bytes.
fn write_library(
    dir: &std::path::Path,
    size: usize,
    serialize: impl FnOnce(&mut [u8]) -> windows::core::Result<()>,
) -> std::io::Result<String> {
    let _guard = WRITING.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = staging(&dir.join(LIBRARY_FILE));
    let failed = |e: windows::core::Error| std::io::Error::other(e.to_string());
    let written = if mapped()
        && let Some(mut file) = Mapped::create(&tmp, size)
    {
        serialize(file.bytes_mut())
            .map_err(failed)
            .map(|()| tag(file.bytes()))
    } else {
        zeroed(size).and_then(|mut bytes| {
            serialize(&mut bytes)
                .map_err(failed)
                .and_then(|()| std::fs::write(&tmp, &bytes).map(|()| tag(&bytes)))
        })
    };
    let res = written.and_then(|tag| replace_library(dir, &tmp).map(|()| tag));
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

/// `n` zero bytes, asked for fallibly: a library is gigabytes, and a host
/// without the memory leaves the save for later instead of aborting.
#[allow(clippy::slow_vector_initialization)] // `vec!` would abort instead.
fn zeroed(n: usize) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(n)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, e))?;
    bytes.resize(n, 0);
    Ok(bytes)
}

/// Renames the written library `tmp` over the library's file in `dir`.
/// This run may have that file mapped: Windows replaces a mapped file it
/// opened for sharing, and the mapping goes on reading the old one (its
/// handle is held). Where a system refuses, the new library is left
/// beside it as [`NEXT_FILE`] for the next start.
fn replace_library(dir: &std::path::Path, tmp: &std::path::Path) -> std::io::Result<()> {
    std::fs::rename(tmp, dir.join(LIBRARY_FILE))
        .or_else(|_| std::fs::rename(tmp, dir.join(NEXT_FILE)))
}

/// Title pipelines created, and their distinct keys.
static CREATED: Mutex<(u64, Option<HashSet<String>>)> = Mutex::new((0, None));

/// The title created a pipeline with `key`.
pub(crate) fn note_created(key: &str) {
    let mut created = CREATED.lock().unwrap_or_else(|e| e.into_inner());
    created.0 += 1;
    created.1.get_or_insert_with(HashSet::new).insert(key.to_owned());
}

/// Title pipelines created so far, and how many distinct ones.
fn created() -> (u64, usize) {
    let created = CREATED.lock().unwrap_or_else(|e| e.into_inner());
    (created.0, created.1.as_ref().map_or(0, HashSet::len))
}

/// Title pipeline keys drawn or dispatched with: in earlier runs (the
/// list as loaded), and in this one.
#[derive(Default)]
struct Used {
    before: HashSet<String>,
    now: HashSet<String>,
}

impl Used {
    /// The list to keep: this run's keys, then earlier runs' others, at
    /// most `EARLY_MAX` (the oldest go once it is full).
    fn list(&self) -> Vec<String> {
        let earlier = self.before.iter().filter(|k| !self.now.contains(*k));
        self.now.iter().chain(earlier).take(EARLY_MAX).cloned().collect()
    }
}

static USED: Mutex<Option<Used>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<Used>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let before: HashSet<String> = listed.lines().filter(|l| !l.is_empty()).map(str::to_owned).collect();
        tracing::info!(pipelines = before.len(), "title pipelines to start early");
        *used = Some(Used { before, now: HashSet::new() });
    }
    used
}

/// Reads the early list now if not yet, so a save keeps it (a pre-build
/// asks for no title pipeline).
pub(crate) fn keep_early_list() {
    drop(used());
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|u| u.before.contains(key))
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    if used().get_or_insert_with(Used::default).now.insert(key.to_owned()) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A run's saves wait for enough new pipelines or for time to pass, and
    /// a save with nothing new is never due.
    #[test]
    fn a_save_waits_for_enough_new_pipelines_or_a_minute() {
        let secs = std::time::Duration::from_secs;
        let policy = SavePolicy::DEFAULT;
        assert_eq!((policy.min, policy.every), (256, secs(60)));
        // A few to a few hundred new pipelines 20 s or 59 s after a save:
        // not due.
        for added in [3, 10, 40, 255] {
            assert!(
                !policy.due(secs(20), added, false),
                "{added} new pipelines after 20 s"
            );
            assert!(
                !policy.due(secs(59), added, true),
                "{added} new pipelines after 59 s"
            );
        }
        // A minute with anything new, also only an early list.
        assert!(policy.due(secs(60), 1, false));
        assert!(policy.due(secs(60), 0, true));
        // Enough new pipelines, but not sooner than the floor.
        assert!(!policy.due(secs(5), 1000, false));
        assert!(policy.due(SavePolicy::FLOOR, 256, false));
        // Nothing new, however long it has been.
        assert!(!policy.due(secs(3600), 0, false));

        // The saves of a run checked every 20 s (the workers' checks), with
        // `new[i]` pipelines made before check `i`; and what is left for the
        // end of the run (`save(true)`).
        let saves = |new: &[u64]| {
            let (mut added, mut at, mut saves) = (0, 0, 0);
            for (i, &n) in new.iter().enumerate() {
                let now = 20 * (i as u64 + 1);
                added += n;
                if policy.due(secs(now - at), added, false) {
                    (added, at, saves) = (0, now, saves + 1);
                }
            }
            (saves, added)
        };
        // Thirty new every 20 s for five minutes: fifteen whole rewrites
        // before, five now (each of 90).
        assert_eq!(saves(&[30; 15]), (5, 0));
        // A load that makes 600 every 20 s for a minute, then a few: saved
        // at each check of the load, then once a minute.
        assert_eq!(saves(&[600, 600, 600, 2, 2, 2, 2]), (4, 2));
        // Nothing new after the last save: no more saves.
        assert_eq!(saves(&[300, 0, 0, 0, 0]), (1, 0));
    }

    /// The variables set the count and the time; others leave the defaults.
    #[test]
    fn the_save_policy_comes_from_the_variables() {
        let secs = std::time::Duration::from_secs;
        assert_eq!(SavePolicy::of(None, None), SavePolicy::DEFAULT);
        let set = SavePolicy {
            min: 64,
            every: secs(30),
        };
        assert_eq!(SavePolicy::of(Some("64"), Some(" 30 ")), set);
        let wrong = SavePolicy::of(Some("lots"), Some("-1"));
        assert_eq!(wrong, SavePolicy::DEFAULT);
        // 0 and 0: a save at every check with anything new, for comparison
        // runs.
        let always = SavePolicy::of(Some("0"), Some("0"));
        assert!(always.due(secs(0), 1, false));
        assert!(!always.due(secs(0), 0, false));
        // An `every` shorter than the floor also shortens the floor.
        let quick = SavePolicy::of(Some("1"), Some("5"));
        assert!(quick.due(secs(5), 1, false));
        assert!(!quick.due(secs(4), 300, false));
    }

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    /// This run's keys stay in the list; earlier runs' fill it up to the cap.
    #[test]
    fn the_early_list_keeps_this_run_first() {
        let keys = |r: std::ops::Range<usize>| r.map(|i| format!("{i:032x}")).collect::<HashSet<_>>();
        let used = Used { before: keys(0..EARLY_MAX), now: keys(EARLY_MAX - 10..EARLY_MAX + 10) };
        let list = used.list();
        assert_eq!(list.len(), EARLY_MAX);
        assert!(used.now.iter().all(|k| list.contains(k)));
        assert_eq!(list.iter().collect::<HashSet<_>>().len(), EARLY_MAX, "no key twice");
    }

    /// The pre-build's list holds only for the library bytes it was
    /// saved with.
    #[test]
    fn the_prebuilt_list_holds_for_its_library_only() {
        let keys: HashSet<String> = ["a".repeat(32), "b".repeat(32)].into();
        let saved = list(&tag(b"library"), &keys);
        assert_eq!(listed(saved.as_bytes(), &tag(b"library")), keys);
        assert!(listed(saved.as_bytes(), &tag(b"library2")).is_empty());
        assert!(listed(b"", &tag(b"library")).is_empty());
    }

    #[test]
    fn different_names_are_held_at_the_same_time() {
        static NAMES: Names = Names::new();
        // Each thread holds its name until every thread holds its own.
        let all = std::sync::Barrier::new(8);
        std::thread::scope(|s| {
            for i in 0..8 {
                let all = &all;
                s.spawn(move || {
                    let name = format!("pipeline {i}");
                    let _held = NAMES.hold(&name);
                    all.wait();
                });
            }
        });
        assert!(NAMES.held.lock().unwrap().as_ref().unwrap().is_empty());
    }

    #[test]
    fn one_name_is_held_by_one_thread_at_a_time() {
        static NAMES: Names = Names::new();
        static INSIDE: AtomicU64 = AtomicU64::new(0);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..500 {
                        let _held = NAMES.hold("pipeline");
                        assert_eq!(INSIDE.fetch_add(1, Ordering::SeqCst), 0);
                        std::hint::spin_loop();
                        INSIDE.fetch_sub(1, Ordering::SeqCst);
                    }
                });
            }
        });
        // A holder that panics lets go of it.
        let panicked = std::thread::spawn(|| {
            let _held = NAMES.hold("pipeline");
            panic!("on purpose");
        });
        assert!(panicked.join().is_err());
        drop(NAMES.hold("pipeline"));
    }

    /// A library's pipelines load on many threads at once from its mapped
    /// file, each with its own description (a load with another's is
    /// refused), while others are stored and the library is serialised.
    #[test]
    fn pipelines_load_on_many_threads_at_once() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::core::s;
        const N: usize = 48;
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        let shader = |i: usize| {
            let source = format!(
                "#define RS \"UAV(u0)\"\nRWByteAddressBuffer b : register(u0);\n\
                 [RootSignature(RS)] [numthreads(1, 1, 1)] void main() {{ b.Store(0, b.Load(4) * {i} + {i}); }}"
            );
            let mut blob = None;
            // SAFETY: the source outlives the call; the blob is read within it.
            unsafe {
                D3DCompile(
                    source.as_ptr().cast(),
                    source.len(),
                    None,
                    None,
                    None,
                    s!("main"),
                    s!("cs_5_1"),
                    0,
                    0,
                    &mut blob,
                    None,
                )
                .ok()?;
                let blob = blob?;
                Some(
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer().cast::<u8>(),
                        blob.GetBufferSize(),
                    )
                    .to_vec(),
                )
            }
        };
        let Some(codes) = (0..2 * N).map(shader).collect::<Option<Vec<_>>>() else {
            return; // no shader compiler here
        };
        // SAFETY: the code holds its root signature.
        let root: ID3D12RootSignature =
            unsafe { device.CreateRootSignature(0, &codes[0]) }.unwrap();
        struct Shared(ID3D12Device1, ID3D12RootSignature, ID3D12PipelineLibrary1);
        // SAFETY: D3D12 objects are free-threaded.
        unsafe impl Sync for Shared {}
        impl Shared {
            fn desc(&self, code: &[u8]) -> D3D12_COMPUTE_PIPELINE_STATE_DESC {
                D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    // SAFETY: a borrowed pointer the description never releases.
                    pRootSignature: unsafe { std::mem::transmute_copy(&self.1) },
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                }
            }
            fn store(&self, i: usize, code: &[u8]) {
                let name = HSTRING::from(format!("p{i}"));
                // SAFETY: the description points at live data.
                unsafe {
                    let pso: ID3D12PipelineState =
                        self.0.CreateComputePipelineState(&self.desc(code)).unwrap();
                    self.2.StorePipeline(&name, &pso).unwrap();
                }
            }
            fn load(&self, i: usize, code: &[u8]) -> windows::core::Result<ID3D12PipelineState> {
                let name = HSTRING::from(format!("p{i}"));
                // SAFETY: as above.
                unsafe { self.2.LoadComputePipeline(&name, &self.desc(code)) }
            }
            fn serialized(&self) -> Vec<u8> {
                // SAFETY: a buffer of the size asked for; nothing is stored
                // meanwhile.
                unsafe {
                    let mut blob = vec![0; self.2.GetSerializedSize()];
                    self.2.Serialize(&mut blob).unwrap();
                    blob
                }
            }
        }
        let first = open(&device, Blob::None).expect("an empty library");
        let made = Shared(device.clone(), root.clone(), first.library.clone());
        (0..N).for_each(|i| made.store(i, &codes[i]));
        // The library again, from its file mapped (as a run opens it).
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(LIBRARY_FILE), made.serialized()).unwrap();
        let blob = Blob::of(dir.path());
        assert!(matches!(blob, Blob::Mapped(_)) || !mapped());
        let again = open(&device, blob).expect("the library again");
        assert!(
            !again._blob.bytes().is_empty(),
            "the stored library, not an empty one"
        );
        let shared = Shared(device, root, again.library.clone());
        let stores = Mutex::new(());
        std::thread::scope(|s| {
            for t in 0..8 {
                let (shared, codes, stores) = (&shared, &codes, &stores);
                s.spawn(move || {
                    for i in (t..N).step_by(8) {
                        shared.load(i, &codes[i]).expect("a stored pipeline");
                        assert!(
                            shared.load(i, &codes[(i + 1) % N]).is_err(),
                            "another description"
                        );
                        assert!(shared.load(N + i, &codes[N + i]).is_err(), "not stored yet");
                        let _one = stores.lock().unwrap();
                        shared.store(N + i, &codes[N + i]);
                        shared.serialized();
                    }
                });
            }
        });
        for (i, code) in codes.iter().enumerate() {
            shared.load(i, code).expect("every pipeline");
        }
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return; // no host GPU here
        };
        let library = open(&device, Blob::Held(b"not a pipeline library".repeat(16)))
            .expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(matches!(library._blob, Blob::None));
    }

    /// A mapped file reads as the file's bytes, a save writes the library
    /// through a mapping of a staging file and renames it over the mapped
    /// one, the mapping still reads the old library, and the next start
    /// reads the new one.
    #[test]
    fn the_library_file_is_mapped_and_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        assert!(matches!(Blob::of(dir.path()), Blob::None), "no file yet");
        std::fs::write(&path, b"").unwrap();
        assert!(matches!(Blob::of(dir.path()), Blob::None), "an empty file");
        let old: Vec<u8> = (0..200_000u32).map(|i| i as u8).collect();
        std::fs::write(&path, &old).unwrap();
        let blob = Blob::of(dir.path());
        assert!(matches!(blob, Blob::Mapped(_)) || !mapped());
        assert_eq!(blob.bytes(), &old[..]);
        // A save while the old file is mapped.
        let new: Vec<u8> = (0..300_000u32).map(|i| (i * 7) as u8).collect();
        let tag = write_library(dir.path(), new.len(), |out| {
            out.copy_from_slice(&new);
            Ok(())
        })
        .unwrap();
        assert_eq!(tag, super::tag(&new), "the tag of the bytes written");
        // The mapping still reads the old library; no staging file is left.
        assert_eq!(blob.bytes(), &old[..]);
        let left: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert_eq!(left, [std::ffi::OsString::from(LIBRARY_FILE)], "{left:?}");
        drop(blob);
        // The next start reads the new one, wherever the save left it.
        assert_eq!(Blob::of(dir.path()).bytes(), &new[..]);
        assert!(!dir.path().join(NEXT_FILE).exists());
        // A serialisation that fails leaves the library as it was.
        let refused = write_library(dir.path(), 64, |_| {
            Err(windows::core::Error::from_hresult(windows::core::HRESULT(
                -1,
            )))
        });
        assert!(refused.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), new);
        assert_eq!(
            std::fs::read_dir(dir.path()).unwrap().count(),
            1,
            "no staging file left"
        );
    }

    /// A save buffer the host cannot give is an error, not an abort.
    #[test]
    fn a_save_buffer_the_host_cannot_give_is_an_error() {
        let refused = zeroed(usize::MAX / 2).expect_err("no such memory");
        assert_eq!(refused.kind(), std::io::ErrorKind::OutOfMemory);
        assert_eq!(zeroed(100).unwrap(), [0u8; 100]);
    }

    /// The bytes at `at` of the pattern the test below writes.
    fn pattern(at: usize, out: &mut [u8]) {
        for (i, word) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let v = ((at / 4 + i) as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(1);
            *word = v.to_le_bytes();
        }
    }

    /// A file of `len` bytes (whole MiB) of [`pattern`], written past the
    /// system's cache: none of its pages is in memory afterwards. False
    /// where the folder's file system does not take such writes.
    fn write_uncached(path: &std::path::Path, len: usize) -> bool {
        use std::io::Write;
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH
        const UNCACHED: u32 = 0x2000_0000 | 0x8000_0000;
        const CHUNK: usize = 1 << 20;
        let Ok(mut file) = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(UNCACHED)
            .open(path)
        else {
            return false;
        };
        // Such writes want a buffer on a sector boundary.
        let mut store = vec![0u8; CHUNK + 4096];
        let start = store.as_ptr().align_offset(4096);
        for at in (0..len).step_by(CHUNK) {
            let chunk = &mut store[start..start + CHUNK];
            pattern(at, chunk);
            if file.write_all(chunk).is_err() {
                return false;
            }
        }
        true
    }

    /// The library's file is replaced by a save while it is mapped, and
    /// pages of the mapping that were not in memory then are read after:
    /// with the file's handle held they still read as the old file (closed,
    /// they read as zeros, which the host would take for pipelines).
    #[test]
    fn a_replaced_library_file_still_reads_from_the_disk() {
        const LEN: usize = 8 << 20;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        if !write_uncached(&path, LEN) {
            return; // no uncached writes here: nothing to show
        }
        let mapped = Mapped::open(&path).expect("the file, mapped");
        // A save: the new library, renamed over the mapped one.
        write_library(dir.path(), LEN, |out| {
            out.fill(0xEE);
            Ok(())
        })
        .unwrap();
        assert!(!dir.path().join(NEXT_FILE).exists(), "replaced in place");
        // Every page of the old mapping, read for the first time now.
        let mut expected = vec![0u8; 4096];
        for (page, bytes) in mapped.bytes().chunks(4096).enumerate() {
            pattern(page * 4096, &mut expected);
            assert!(
                bytes == &expected[..],
                "page {page} of the replaced library no longer reads as the old file"
            );
        }
    }

    /// While the library's file is mapped, no one may write it in place or
    /// truncate it (the bytes would change under the host); reading it,
    /// and a save renaming a new library over it, still work.
    #[test]
    fn a_mapped_library_file_cannot_be_written_meanwhile() {
        const SHARING_VIOLATION: i32 = 32;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        let old = vec![0x5Au8; 3 * 4096];
        std::fs::write(&path, &old).unwrap();
        let mapped = Mapped::open(&path).expect("the file, mapped");
        let write = std::fs::OpenOptions::new().write(true).open(&path);
        assert_eq!(
            write.err().and_then(|e| e.raw_os_error()),
            Some(SHARING_VIOLATION),
            "written in place"
        );
        let truncate = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&path);
        assert_eq!(
            truncate.err().and_then(|e| e.raw_os_error()),
            Some(SHARING_VIOLATION),
            "truncated"
        );
        assert_eq!(std::fs::read(&path).unwrap(), old, "others still read it");
        // A save renames over it; the mapping keeps the old bytes.
        write_library(dir.path(), 4096, |out| {
            out.fill(1);
            Ok(())
        })
        .unwrap();
        assert!(!dir.path().join(NEXT_FILE).exists(), "replaced in place");
        assert_eq!(std::fs::read(&path).unwrap(), [1u8; 4096]);
        assert_eq!(mapped.bytes(), &old[..]);
    }

    /// A staging file's name gives the process that wrote it, for our files
    /// only.
    #[test]
    fn staging_names_give_their_process() {
        let dir = std::path::Path::new("d");
        let name = |p: &std::path::Path| p.file_name().unwrap().to_str().unwrap().to_owned();
        for file in [LIBRARY_FILE, EARLY_FILE, PREBUILT_FILE] {
            assert_eq!(
                staging_pid(&name(&staging(&dir.join(file)))),
                Some(std::process::id())
            );
        }
        assert_eq!(staging_pid("host-pipelines.bin.tmp-12-0"), Some(12));
        for other in [
            "host-pipelines.bin",
            "host-pipelines.next",
            "host-pipelines.bin.tmp-12",
            "host-pipelines.bin.tmp-x-0",
            "host-pipelines.bin.tmp-12-x",
            "shaders.bin.tmp-12-0",
        ] {
            assert_eq!(staging_pid(other), None, "{other}");
        }
    }

    /// A load removes the staging files of runs that ended during a save,
    /// and leaves those of running processes, this one's and other files.
    #[test]
    fn staging_files_of_ended_runs_are_removed() {
        let exited = |keep: bool| {
            let mut child = std::process::Command::new("cmd")
                .args(["/c", "exit 0"])
                .spawn()
                .unwrap();
            child.wait().unwrap();
            let pid = child.id();
            // Dropped, no handle keeps the process: no such process. Kept,
            // it has exited but its object lives on.
            if keep {
                (pid, Some(child))
            } else {
                (pid, None)
            }
        };
        let (gone, _) = exited(false);
        let (zombie, _held) = exited(true);
        assert!(ended(gone), "no such process");
        assert!(ended(zombie), "exited, a handle open");
        assert!(!ended(std::process::id()), "this one");
        assert!(!ended(4), "the system, which we may not open");
        let mut running = std::process::Command::new("cmd")
            .args(["/c", "pause"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        assert!(!ended(running.id()), "a running process");
        let dir = tempfile::tempdir().unwrap();
        let file = |name: String| {
            std::fs::write(dir.path().join(&name), b"staged").unwrap();
            name
        };
        let removed = [
            file(format!("{LIBRARY_FILE}.tmp-{gone}-3")),
            file(format!("{EARLY_FILE}.tmp-{gone}-0")),
            file(format!("{PREBUILT_FILE}.tmp-{zombie}-1")),
        ];
        let kept = [
            file(format!("{LIBRARY_FILE}.tmp-{}-0", std::process::id())),
            file(format!("{LIBRARY_FILE}.tmp-{}-0", running.id())),
            file(format!("{LIBRARY_FILE}.tmp-4-0")),
            file(format!("other.tmp-{gone}-0")),
            file(LIBRARY_FILE.to_owned()),
        ];
        let blob = Blob::of(dir.path());
        assert_eq!(blob.bytes(), b"staged", "the library itself");
        for name in &removed {
            assert!(!dir.path().join(name).exists(), "{name} left");
        }
        for name in &kept {
            assert!(dir.path().join(name).exists(), "{name} removed");
        }
        let _ = running.kill();
        let _ = running.wait();
    }

    /// The process's own memory a library file costs: read into memory and
    /// mapped (every page read once).
    /// `cargo test --release -p victory-gpu --lib library_file_memory -- --ignored --nocapture`
    #[test]
    #[ignore = "measurement: run with --ignored --nocapture"]
    fn library_file_memory() {
        use windows_sys::Win32::System::ProcessStatus::{
            GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        fn private() -> f64 {
            let mut c = PROCESS_MEMORY_COUNTERS_EX::default();
            // SAFETY: a plain query into our own counters.
            unsafe {
                GetProcessMemoryInfo(
                    GetCurrentProcess(),
                    (&raw mut c).cast(),
                    size_of_val(&c) as u32,
                );
            }
            c.PrivateUsage as f64 / (1 << 20) as f64
        }
        const LEN: usize = 256 << 20;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        std::fs::write(&path, vec![7u8; LEN]).unwrap();
        let sum = |b: &[u8]| b.iter().step_by(4096).map(|&x| x as u64).sum::<u64>();
        let before = private();
        let read = std::fs::read(&path).unwrap();
        assert_eq!(sum(&read), 7 * (LEN / 4096) as u64);
        let held = private() - before;
        drop(read);
        let before = private();
        let mapped = Mapped::open(&path).unwrap();
        assert_eq!(sum(mapped.bytes()), 7 * (LEN / 4096) as u64);
        let mapped_cost = private() - before;
        drop(mapped);
        eprintln!(
            "256 MiB library file: {held:.1} MiB private read into memory, {mapped_cost:.1} MiB mapped"
        );
        assert!(mapped_cost < 16.0, "{mapped_cost:.1} MiB");
    }

    #[test]
    fn concurrent_write_atomically_never_fails_or_truncates() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("host-pipelines.bin");
        let threads: Vec<_> = (0..4)
            .map(|t| {
                let target = target.clone();
                std::thread::spawn(move || {
                    let payload = vec![b'a' + t as u8; 4096];
                    for _ in 0..30 {
                        write_atomically(&target, &payload).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let final_bytes = std::fs::read(&target).unwrap();
        assert_eq!(final_bytes.len(), 4096);
        assert!(final_bytes.iter().all(|&b| b == final_bytes[0]));
    }
}
