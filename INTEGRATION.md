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
- `victory-gpu` tests on WARP (diagnostics build, `VICTORY_D3D12_WARP=1`,
  one test per process): 168 passed, 0 failed.
- `victory-shader` tests with `VICTORY_WARP=1`, one test per process: 268
  passed, 2 failed. The two failures are hardware-only checks that
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
- Nothing from this branch was run on the RTX 5070, and **the title was not
  run** with it. It makes no claim about results in the game.

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

## `integration/play`: playable candidate for NVIDIA

`integration/play` is this branch (`915978c`) plus the NVIDIA boot, heap
and format PRs and the wave64 stack, so that one build on a 32-lane NVIDIA
card gets as far as the ztrx fork does. It is a **candidate**: built and
checked only (`cargo check --locked -p victory --features diagnostics`
after every merge, then `cargo build --locked --profile diag -p victory
--features diagnostics`); no test suite was run on it and the title has
not been run with it yet. Every PR is its own merge commit
(`git log --first-parent`), authors' commits unchanged.

### Merged, in order

| PR | Head | What it does |
|---|---|---|
| #1763 | `50d91b1` | typed UAVs in formats the host does not support are refused, whatever their family or castability |
| #1751 | `4521591` | a host that caps view heaps (NVIDIA, 1,000,000) keeps the larger dynamic range |
| #1133 | `cae16e6` | translated shaders clamp view indices to the host's cap, whose last slot stays null |
| #2180 | `e1bf670` | UAV/render-target flags a format cannot take are stripped or widened; an early-list-only save writes no library |
| #1778 | `ba67548` | a sampler heap of our own is bound when the title binds none |
| #2129 | `2ef0c75` | the pipeline library file is mapped, not read into memory (intermittent device loss) |
| #1906 | `bcd0d3d` | view addresses for tables made afresh in dynamic slots |
| #1849 | `de95e51` | free memory is not the runtime's to watch (world-load deadlock) |
| #454 | `85c04d1` | opt-in `VICTORY_NO_WATCHDOG` hook for the title's watchdog (unset: off) |
| #223, #233, #320, #439 | `5a11cf4` | wave64 compute on 32-lane hosts, in halves and in the console flow, with the implicit-sample and recursion guards (fork `shader/console-flow-lds`; not `831cdb3`) |

Conflicts resolved at the merges (each merge commit says how): #1133 with
#1751 (both heap branches kept, #1133's guard slot on each) and with
#158's depth-bounds query; #2129 with #2180 in the library save (#2129's
mapped write, #2180's skip when only the early list changed); #1906 with
#1133 (two tests at one place); the wave64 stack with #2115 (`--count`
and `--host-lanes` both), #1133 (`view_limit` and `ops`), #1883 (mask
joins: the stack's rule in the console flow, #1883's per-lane cases
elsewhere, a numeric mask's lane bit refused in halves) and #2043 (the
killing form for trapped discards first, then the per-lane form, the
console flow, the killing form for discards).

### Considered and left out

- #89 (packed 16-bit formats 86/115 as R8G8B8A8): based on `35b054e`,
  before the maintainer's drop; eight conflicting hunks in
  `descriptor/host.rs`, `list.rs` and `texture.rs`, two of them large.
  #2180 is taken instead; it strips or widens the flags rather than
  widening the format.
- #1777 and #1779 (views past a capped heap): #1133 is taken. It clamps
  in the shader to a null last slot of the capped heap, keeps the larger
  dynamic range #1751 asks for, and is based on `940d32f`; #1777 is based
  on the older `70564ae`, and #1779 rewrites ten files (884 lines). Like
  #1777, #1133 creates the host device when the shader cache key is first
  computed (also in `victory translate --cache`).

### Ray tracing (not changed here)

The ztrx and fxceless forks run with the title's ray-tracing structure
builder off: the compute pipelines of `XboxBvhBuilder.dll` (the programs
with a spilled user-data table) are dropped at dispatch unless
`VICTORY_BVH_BUILDER=1` (ztrx `crates/victory-gpu/src/pipeline.rs:285`
`builder_runs`, called from `draw.rs:1677` and `draw.rs:2278`; detection
`of_builder`, `pipeline.rs:238`, needs that fork's
`victory_shader::spill`). The programs that walk the structures still
run there. Upstream `main` has no such gate and no `spill` module, so it
is not a small port and is not in this branch. For the hang in compute
`f607780aa8e502d0` (the probe shadow trace), a diagnostics build of this
branch takes `VICTORY_SKIP_PIPELINES=f607780aa8e502d0`, as `main` does.

### Build

`CARGO_TARGET_DIR=C:\Users\PC\.t3\targets\play`, `cargo build --locked
--profile diag -p victory --features diagnostics` at `0ab7ca3`:
`diag\victory.exe`, sha256
`5b8bf42def4777099794b9a9ce2ef64330c0edf0a2a0a9dd507ee3de0e155a30`.
