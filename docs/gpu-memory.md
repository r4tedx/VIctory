# GPU memory: bridging unified and split memory

An Xbox Series console has one pool of memory. The CPU and the GPU read
and write the same bytes at the same addresses, and a title is written
for that: it builds vertex data, constants and command buffers in place,
hands the GPU their addresses, and reads back what shaders wrote, all
without copying anything.

A desktop PC (Apple silicon and integrated GPUs aside) has two pools:
system memory next to the CPU and video memory on the graphics card,
joined by PCI Express. Each processor is fast in its own pool and slow in
the other one. This page covers how `victory-gpu` bridges the two, what
each choice costs, and the switches that move data closer to the GPU.

```mermaid
flowchart LR
  subgraph Console["Console: one pool"]
    direction TB
    CCPU[CPU] <--> UM[("unified memory<br/>title data, buffers,<br/>render targets")]
    CGPU[GPU] <--> UM
  end
  subgraph PC["PC: two pools"]
    direction TB
    PCPU[CPU] <--> SYS[("system memory<br/>title memory")]
    PGPU[GPU] <--> VRAM[("video memory")]
    SYS <-. "PCI Express<br/>~23 GB/s" .-> VRAM
  end
```

## Host copies of title memory

Title memory sits in system memory, at the addresses the title chose.
The host GPU cannot use those addresses, so every title buffer a draw or
dispatch uses gets a **host copy** (`crate::shadow::Shadow`): a D3D12
buffer of the same size. A title GPU address inside the buffer maps to
the same offset in its copy. Around each submission:

1. **Refresh.** Before the submission runs, the ranges it reads (vertex,
   index and constant data, buffer views, or everything a heap-indexing
   shader can reach) are brought up to date from title memory. Only the
   64 KiB pages the title wrote since the last copy are copied
   (`crate::watch` tracks them with write-watch and read-only guards).
2. **Run.** The host GPU reads and writes the copies.
3. **Copy-back.** What shaders wrote goes back into title memory, without
   overwriting what the title wrote in the meantime. There are two ways
   to find out what shaders wrote:
   - **Marked pages.** Translated shaders set a bit for each page they
     write through a buffer view (`victory_shader::emit`, "Write marks").
     Only those pages are compared with title memory.
   - **Snapshots.** For a range a list declares as written without marks,
     the copy's bytes are saved before the submission and compared after
     it. Whatever changed is what the GPU wrote.

```mermaid
sequenceDiagram
  participant T as Title (CPU)
  participant M as Title memory
  participant C as Host copy
  participant G as Host GPU
  T->>M: writes vertex data, constants
  Note over M,C: refresh: pages written since the last copy
  M->>C: copy changed pages
  G->>C: shaders read and write the copy
  Note over M,C: copy-back: marked pages, or snapshot compare
  C->>M: bytes the GPU wrote
  T->>M: reads what shaders wrote
```

## Where the copies live

By default every copy is in **system memory**, mapped write-back for the
CPU (`Host::shared_buffer`: a custom heap, `MEMORY_POOL_L0`). The CPU
writes and reads it at full speed, but every shader access crosses PCI
Express.

On a GPU with **Resizable BAR**, D3D12 also offers **GPU upload heaps**
(`D3D12_HEAP_TYPE_GPU_UPLOAD`, reported by
`D3D12_OPTIONS16.GPUUploadHeapSupported`). These are video memory that
the CPU can map and write through the BAR. Shaders access them at
video-memory speed. CPU writes stream across the bus at about the speed
of a system-memory write. CPU reads are uncached and very slow.

Measured on an RTX 5070 Ti (driver 621.24), Ryzen 9 7900X, Windows 11,
256 MiB at a time:

| | GPU reads | CPU writes | CPU reads |
|---|---|---|---|
| system memory (`L0`, the default for copies) | 22.9 GB/s | 28.7 GB/s | 22.7 GB/s |
| GPU upload heap (video memory through the BAR) | 716.8 GB/s | 22.7 GB/s | **0.05 GB/s** |
| video memory, default heap | 721.2 GB/s | (no CPU access) | (no CPU access) |

```mermaid
---
config:
  xyChart:
    width: 640
    height: 170
---
xychart-beta
  title "GPU reading 256 MiB, GB/s"
  x-axis ["system memory (L0)", "GPU upload heap", "video memory"]
  y-axis "GB/s" 0 --> 800
  bar [22.9, 716.8, 721.2]
```

So a copy shaders use heavily belongs in video memory, but the CPU must
never read it there. Two switches act on that, both off by default
(`=1` turns one on). They come after P-0a's video memory copies
(`VICTORY_VRAM_COPIES`, every copy by default, `=1` read-only ones, `=0`
none; `crate::vram`): a new copy tries
video memory under its budget first, then a GPU upload heap if these ask
for one, then system memory. The chosen placement is logged once when
the host device is made.

| switch | copies moved to a GPU upload heap | what the CPU reads |
|---|---|---|
| `VICTORY_GPU_UPLOAD=1` | buffers made without unordered access (shaders only read them) | nothing: the CPU only writes these copies |
| `VICTORY_GPU_UPLOAD_WRITES=1` (with the one above) | buffers shaders may write | a **mirror** in system memory |

Both switches fall back to system memory when the GPU has no GPU upload
heaps, and past these copies' share of the video memory budget
(`UPLOAD_POOL`, see [below](#video-memory-budget-and-residency); a copy's
bytes are given back when D3D12 destroys its buffer). Copies
of loose title memory and of single views (`loose`, `exact`) always stay
in system memory.

```mermaid
flowchart TD
  B[title buffer used by the GPU] --> RO{"made without<br/>unordered access?"}
  RO -- yes --> U1{VICTORY_GPU_UPLOAD}
  U1 -- on --> UH["GPU upload heap<br/>(CPU writes only)"]
  U1 -- off --> SM["system memory (L0)"]
  RO -- no --> U2{"VICTORY_GPU_UPLOAD +<br/>VICTORY_GPU_UPLOAD_WRITES"}
  U2 -- on --> VM["GPU upload heap<br/>+ mirror in system memory"]
  U2 -- off --> SM
```

## Written copies: video memory with a lazy mirror

A copy that shaders write has to be read by the CPU: copy-back compares
it with title memory, and snapshots are taken from it. Reading it from
the upload heap at 50 MB/s is out of the question, so such a copy has two
parts:

- **the copy in video memory**, which shaders use and which always holds
  the current bytes;
- **the mirror** in system memory, which the CPU reads. It may be behind
  the copy in video memory, one 64 KiB page at a time.

The rules that keep them consistent:

- **Refreshes write video memory only** (`Shadow::copy`) and mark the
  pages they touched as stale in the mirror (a bit per page,
  `Vram::stale`). Writing the mirror as well would double the CPU's
  refresh cost, and on the marked path nothing ever reads those bytes.
- **Any other CPU write to the mirror goes on into video memory**
  (`Shadow::push`): copy-back taking the title's bytes, and the update of
  a snapshot under a busy range.
- **Before the CPU reads the mirror, its stale pages come back from video
  memory.** The GPU copies them (`Shadow::freshen` → `bring_back` →
  `Host::copy_now`: one command list, one fence wait, for any number of
  pages). The same goes for pages shaders wrote (`shadow::pull`): the
  marked pages before their copy-back, and the declared ranges with
  snapshots.
- **A submission's snapshots are prepared up front.** Before a
  submission, `Host::execute` collects the ranges its lists declare as
  written without marks (`shadow::snapshots_ahead`). Their stale pages
  come back in one GPU copy, and until the copy-backs, refreshes in those
  pages write the mirror as well. A snapshot then never waits for the
  GPU.
- **Page bits are cleared under the page lock, around the GPU copy.** A
  refresh that races the copy sets its bit again once the lock is
  released, so its bytes are not lost.

```mermaid
stateDiagram-v2
  direction LR
  [*] --> Current
  Current --> Stale: refresh writes video memory only
  Stale --> Stale: more refreshes
  Stale --> Current: CPU reads the page (freshen / pull)
  Current --> Current: CPU writes the mirror + push, or a refresh in a snapshot range
```

```mermaid
sequenceDiagram
  participant T as Title memory
  participant Mi as Mirror (system memory)
  participant V as Copy (video memory)
  participant G as Host GPU
  Note over Mi,V: refresh
  T->>V: changed pages (CPU writes through the BAR)
  Note over Mi: pages marked stale
  Note over Mi,V: snapshot ranges (unmarked lists only)
  V->>Mi: stale pages in the ranges (one GPU copy)
  G->>V: shaders run at video memory speed
  Note over Mi,V: before copy-back
  V->>Mi: marked pages / declared ranges (one GPU copy)
  Mi->>T: bytes the GPU wrote
  Mi->>V: title bytes taken by the copy (push)
```

### What it costs and what it buys

On the GPU side, from the tests `reading_copies_in_video_and_system_memory`
and `written_copies_in_video_and_system_memory` (release, same machine):

| 256 MiB, one copy | system memory | video memory | |
|---|---|---|---|
| shader reads (read-only buffer) | 11.64 ms (23.1 GB/s) | 0.37 ms (722 GB/s) | 31× |
| shader reads and writes (written buffer) | 212.23 ms | 1.25 ms | 170× |

Bringing pages back costs a fixed 0.05 ms (a list and a fence wait),
plus the copy itself: 64 scattered pages (4 MiB) take 0.23 ms.

On the CPU side, a synthetic world frame (#1743's shape: a 174 MB heap,
36 submissions with 24 declared ranges each; median of three interleaved
runs, ms per frame):

| path | copies in system memory | written copies in video memory, mirror always written | lazy mirror (now) |
|---|---|---|---|
| marked pages | 80.7 | 114.6 | **69.0** |
| snapshots of declared ranges | 708 | 910 | 950 |

```mermaid
---
config:
  xyChart:
    width: 640
    height: 170
---
xychart-beta
  title "CPU time per synthetic frame, marked path (ms)"
  x-axis ["system memory", "mirror always written", "lazy mirror"]
  y-axis "ms" 0 --> 130
  bar [80.7, 114.6, 69.0]
```

On the marked path the lazy mirror costs less CPU time than copies in
system memory. A refresh writes each changed page once, through the BAR,
where it used to write system memory, and only the pages shaders marked
come back. The snapshot path pays for bringing every declared range back
before its compare (several GB per frame here): it needs marks to
benefit.

## Video memory budget and residency

Windows gives each process a **video memory budget**
(`IDXGIAdapter3::QueryVideoMemoryInfo`), for video memory (the local
segment group) and for the system memory the GPU reads across the bus
(non-local). It changes while running, as other processes take and free
memory. When a process uses more than its budget, the video memory
manager **demotes** some of its allocations to system memory. Shaders
then reach them across PCI Express, which is 30 times slower or worse
(see the tables above). If a render target or depth buffer is demoted,
every frame waits on the bus, and frames take seconds.

VIctory does two things about it.

**Residency priorities.** The manager demotes allocations with the
lowest residency priority first (`ID3D12Device1::SetResidencyPriority`).
`crate::host::keep_resident` sets it where VIctory makes an allocation:

| allocation | priority | why |
|---|---|---|
| host textures with the render target, depth stencil or unordered access flag | `HIGH` | the GPU writes them every frame, and they hold the only copy of what it wrote |
| view heaps' stride, mark and address tables | `HIGH` | every draw reads them |
| the views' written-pages bitmap | `HIGH` | shaders that mark their writes set it every frame |
| list scratch chunks | `HIGH` | the list's commands read them |
| host copies in GPU upload heaps | `HIGH` | shaders read them every frame; they are budgeted (below) |
| host textures uploaded from title memory (`texture::title_written`) | `NORMAL` | streamed: each is read by some frames only, and sampling one from system memory is slower without stalling the frame |

Descriptor heaps get none: on an RTX 5070 Ti (driver 621.24) a priority
on a shader-visible descriptor heap removed the device at the next
submission.

`NORMAL` is what every allocation starts with, so streamed textures and
everything VIctory does not name go first. `MAXIMUM` is not used: the
system keeps such allocations even past the budget, at the cost of other
processes. `LOW` is not used either: it would demote streamed textures
before allocations that have no priority set (depth twins, upload
staging), which frames need no more than these.

**Pools sized from the budget.** `crate::budget` reads DXGI at most every
250 ms, and again as soon as the system signals a budget change
(`RegisterVideoMemoryBudgetChangeNotificationEvent`), so pools can ask it
as often as they like. Bytes a pool takes between readings
(`Budget::charge`) count as used until the next reading includes them.
A pool asks `Budget::limit` how much it may hold now: a part of what the
rest of the process leaves of the local budget, up to a ceiling. Its own
bytes do not shrink the answer, everyone else's do.

The copies in GPU upload heaps (`UPLOAD_POOL`) take half of what the rest
of the process leaves. The ceiling is 2 GiB (the fixed size of before), or
a quarter of the budget where that is more. On a host that reports no
budget, the ceiling applies. A copy that does not fit stays in system
memory, as before. Examples:

| local budget | used by the rest of the process | copies may hold |
|---|---|---|
| 7.5 GB (8 GB card) | 2 GB | 2 GiB (ceiling) |
| 7.5 GB | 5 GB | 1.25 GB |
| 7.5 GB | 7.5 GB or more | nothing |
| 15 GB (16 GB card) | 4 GB | 3.75 GB (a quarter of the budget) |

The first time use passes nine tenths of the local budget, a warning
names what VIctory knows takes it: host textures (all of them, and those
lists used), host copies of title memory (all of them, and those in GPU
upload heaps), and the non-local segment's use and budget. The check runs
with each frame-rate line, whether or not a pool asks.

## Measuring a title

The tests show the mechanism. Whether it pays off in a title depends on
how much of the title's GPU time goes to shaders reading these copies,
and that needs a run of the title. Compare `VICTORY_GPU_TIMING=1` with
and without the switches, together with the copy counters
(`--features diagnostics`).

## Further steps

- Move copies between pools while running, according to how they are
  used (read-mostly, written by shaders, or read back by the CPU).
- Give snapshot ranges marks, or compare them on the GPU, so they no
  longer need their bytes brought back.
- Detile textures on the GPU, so their upload crosses the bus once,
  still tiled.
