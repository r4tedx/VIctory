# Tester integration branch on main

This branch puts a few open upstream pull requests together on top of the
current upstream `main`, so that testers who run the title can build once and
measure them together. It is **not an official release**; the maintainer has
not merged any of these PRs, and each may still change or be turned down.

It replaces `integration/testers`, whose last update is `4744af9` (update 10).
That tree was based on `ffe310e`, before the maintainer's large drop of
2026-11-24 and the PRs merged after it; it is frozen, not deleted.

- Base: upstream `main` at `940d32f`.
- Every PR comes in as its own merge commit (`git log --first-parent`), so
  you can find, revert or leave out each one. No edits were made beyond the
  PRs themselves. Two conflicts, both text: the README's "missing" row,
  which #158 and #159 both edit, carries both clauses; and the DXIL line of
  `translate`'s summary keeps #1315's "vertex, geometry and pixel programs"
  with #2115's wording for checks that were not asked for.
- Only heads their authors or owners name as green on `main` are taken, and
  a head goes in only when no program that validates on `main` is lost.

## What is in it

| PR | Author | Head | What it does |
|---|---|---|---|
| #158 | r4tedx | `49dd080` | the depth bounds test reaches the host, on adapters that support it |
| #159 | r4tedx | `b7249e5` | occlusion, pipeline statistics and timestamp queries run on the host |
| #2111 | r4tedx | `0ef11e5` | one pipeline wait budget per frame for all recording threads; the frame line prints how long draws waited for pipelines |
| #1762 | claude-GTA6 | `4eb5d1e` | structured buffer views become raw views of the same bytes on the host (NVIDIA reads 0 through raw loads of structured views) |
| #1883 | powplusplus | `a31d8cc` | lane masks a traversal keeps as numbers |
| #2027 | fiestorra | `34d0b4e` | a compute shader samples at level 0, not with derivatives |
| #1315 | r4tedx | `0d45a59` | single-stream triangle geometry programs |
| #2136 | r4tedx | `937f329` | a host copy nothing wrote is not asked for at submission (less CPU time preparing a frame) |
| #2147 | r4tedx | `f869790` | ranges shaders may write are armed as runs of pages, not page by page |
| #2134 | r4tedx | `efb35b5` | a benchmark of a world frame's preparing at the published volumes (tests only) |
| #2115 | r4tedx | `4d34d76` | `translate --count`: the instructions of the emitted shaders, to compare builds |
| #1428 | fuzeofficlal | `0e3796e` | `OMSetRenderTargets` reads no handle for zero targets, and bounds the count |
| #2043 | Deilach | `7686b6f` | a pixel discard in a loop whose exit depends on it kills its lanes, as the console does |
| #1989 | BastienRodz | `0f11320` | a vertex (or mesh, geometry) program writing through views declares the UAV flags the validator requires |
| #1763 | n-romanovich | `50d91b1` | typed UAVs in formats the adapter does not support are refused whatever their family or castability, and the supported ones are pinned by tests (on NVIDIA, `main` removes the device at boot) |
| #1751 | JurajSimicevic | `4521591` | view heaps capped by the host (1,000,000 slots on NVIDIA) keep the full dynamic descriptor range |
| #1133 | rivaldodev | `cae16e6` | translated shaders clamp view indices to the host's cap (1,000,000 on NVIDIA), whose last slot stays null |

#1133 conflicts with #1751 in text only, resolved as integration/play
resolved it (`4dc038b8`; each file the merge touches is the same as
play's): `descriptor/host.rs` keeps #1751's middle branch (a host that
takes `PORTABLE_VIEWS` but not `HEAP_VIEWS`) with #1133's guard slot, and
the dynamic-slot test asserts both #1133's heap size and #1751's dynamic
range; `host.rs` keeps the depth-bounds query with #1133's cap probe.

Left out for now: #2150 (one instruction per value on the finished
module). It rewrites every emitted shader for a module-size gain; it has
had one hardware run by a tester since. It is kept out so that the first
supervised game run uses shader output as close to `main` as possible.

Not in it yet: the wave64 stack (#223, #233, #320, #439: compute programs
written for 64-lane waves, on NVIDIA's 32-lane waves). Its head on `main`
is `5a11cf4`, with its implicit-sample and recursion guards; it goes in
with the next check round. Until then an NVIDIA card refuses those
programs on this branch as on `main`.

## Known risks on NVIDIA (crash safety)

- **Heap indices past a capped heap.** Translated shaders clamp
  descriptor-heap indices below 2^21, but NVIDIA caps a shader-visible
  view heap at 1,000,000, so an index between the two still passes the
  heap's end. Upstream #1777 aims at this but is not taken yet: it opens
  the default hardware adapter in every process that translates, and has
  a slot-overlap case at one heap size (both reported on #1777).
- **Driver-level shader error reporting** (the Nsight Aftermath Monitor
  set to Global) turned every world load into a device hang for a tester
  on upstream #71. Check it is off before a run.
- Running out of per-draw descriptor slots does not bind unwritten slots:
  `main` drops that draw or dispatch (logged as `a draw or dispatch
  dropped: no dynamic slots for its tables`).

## How it was checked

On one Windows 11 PC (NVIDIA RTX 5070, AMD Radeon integrated GPU, WARP):

- `cargo build --release -p victory`, the test builds and
  `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- Checked at `62d46c3` (the #1133 merge). The later commits for this file
  change only text.
- `victory-gpu` tests on WARP (diagnostics build, `VICTORY_D3D12_WARP=1`,
  one test per process): 171 passed, 0 failed (#1763 adds two tests,
  #1751 updates the view-heap test, #1133 adds one, whose capped-heap
  checks run only on a host that caps view heaps).
- `victory-shader` tests with `VICTORY_WARP=1`, one test per process: 269
  passed, 2 failed (#1133 adds one). The two failures are hardware-only checks that
  `VICTORY_WARP=1` sends to WARP (`gathers_at_a_level_match_the_hardware`,
  and `permutes_read_the_lane_named_and_zero_from_lanes_left_out`, which
  needs a device that accepts a wave of 32); they fail the same way on
  `main` under `VICTORY_WARP=1`, and no PR here touches them.
- **Local title-data diagnostic:** `victory translate shaders.rpf --validate
  --list --jobs 6` on this PC's copy of the title (offline, CPU only, counts
  and refusal reasons only), program by program against the same count of
  `main`: **1,842 programs gained, 0 lost**: all 867 vertex programs
  `main` emits but the validator refuses (#1989), all 195 geometry
  programs (#1315), and 780 compute programs (#1883 and #2027 together:
  735 that `main` emits but the validator refuses, 40 ray-intersection
  programs and 5 more). Every program of that archive that does not validate on this
  branch does not validate on `main` either.
- The speed numbers of #2136 and #2111 in their PRs come from synthetic
  benchmarks, not from the game; times from the title are what they need.
- In-game check by us (local title data, RTX 5070, driver 616.92,
  diagnostics build): main 940d32f removes the device at boot after
  9.5-11 s (debug layer: typed UAV of B5G5R5A1_UNORM unsupported, then
  RemoveDevice); main + #1763 ran 10 minutes with no device removal.
- Apart from that check of #1763 alone on `main`, nothing from this branch
  was run on the RTX 5070, and **the title was not run** with this branch.
  It makes no claim about results in the game.
- #1751 held in game on the RTX 5070 as part of integration/play's set (layer-1 run L1b, 10 minutes, lit world, 0 events); not run alone.
- #1133 held in game on the RTX 5070 as part of integration/play's set (layer-1 run L1b, 10 minutes, lit world, 0 events); not run alone.

## What a useful report from the title contains

- the branch head, your GPU and driver, Windows version;
- whether the world loads and whether it is lit, and how long it runs;
- the frame line's times (preparing, on the GPU, waiting for pipelines);
- any device removal with its DRED output, and the System log's display
  driver events (`nvlddmkm`) at that time. On NVIDIA, check first that no
  driver-level shader error reporting (the Nsight Aftermath Monitor set to
  Global) is on: a tester on upstream issue #71 found it turns every world
  load into a device hang.

Counts, messages and hashes only: never game files, shader dumps or
captures.
