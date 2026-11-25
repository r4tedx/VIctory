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
  PRs themselves; the only conflict was the README's "missing" row, which
  #158 and #159 both edit, resolved to carry both clauses.
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

Not in it yet, because their owners have not named a head based on `main`:
the wave64 stack (#223, #233, #320, #439: compute programs written for
64-lane waves, on NVIDIA's 32-lane waves). Without it an NVIDIA card still
refuses those programs on this branch as on `main`.

## How it was checked

On one Windows 11 PC (NVIDIA RTX 5070, AMD Radeon integrated GPU, WARP):

- `cargo build --release -p victory`, the test builds and
  `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `victory-gpu` tests on WARP (diagnostics build, `VICTORY_D3D12_WARP=1`,
  one thread): 161 passed, 0 failed, 2 ignored.
- `victory-shader` tests with `VICTORY_WARP=1`, one thread: 259 passed,
  2 failed, 3 ignored. The two failures are hardware-only checks that
  `VICTORY_WARP=1` sends to WARP (`gathers_at_a_level_match_the_hardware`,
  and `permutes_read_the_lane_named_and_zero_from_lanes_left_out`, which
  needs a device that accepts a wave of 32); they fail the same way on
  `main` under `VICTORY_WARP=1`, and no PR here touches them.
- **Local title-data diagnostic:** `victory translate shaders.rpf --validate
  --list --jobs 6` on this PC's copy of the title (offline, CPU only, counts
  and refusal reasons only), program by program against the same count of
  `main`: **975 programs gained, 0 lost**. They are all 195 geometry
  programs (#1315) and 780 compute programs (#1883 and #2027 together: 735
  that `main` emits but the validator refuses, 40 ray-intersection programs
  and 5 more). Every program of that archive that does not validate on this
  branch does not validate on `main` either.
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
