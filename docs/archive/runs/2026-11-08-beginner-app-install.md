# Beginner review and installed app validation

Date: 2026-11-08. Final app source:
`bfd1d6624534c819e9d45e6f012182c90354389d` on
`codex/beginner-hardware-ranking`. The app was built from a clean source tree.

## Review corrections

The review and installed-app check found three issues that were corrected:

- Benchmark selection counted isolated scored models across memory tiers. It
  now counts distinct parents only within tiers having at least two comparable
  parents. A two-tier regression verifies that the usable cohort wins.
- The production Swift stdout reader used `readData(ofLength: 65_536)` on a
  persistent pipe. A model-free probe reproduced its wait for the remaining
  bytes, and the installed app remained at its checking screen. The reader now
  uses `availableData`, retaining the existing JSONL buffering and wire limits.
  The regression exercises the production handler with two short replies while
  the writer stays open, not a substitute test reader.
- The backend's `files` task and BFCL mapping were unreachable from the task
  picker and request guard. Beginner now separates Explain text from Work with
  files. Tests verify both the visible choice/action and `task="files"` transport.

A separate read-only review found no concrete blocker in native coverage,
pinned source/revision matching, installed-copy reuse, storage credit or wheel
inclusion. It did not establish inference quality or measured memory peaks.

## Validation

| Check | Result |
| --- | --- |
| Focused benchmark and guided model tests | 101 passed at ranking fix `8b8c1fed7e526d62ea0e58d4bdde8474c4c5591e` |
| Complete converter suite | 6,735 passed; 96 skipped; 270.94 seconds at `8b8c1fed7e526d62ea0e58d4bdde8474c4c5591e`; Python implementation unchanged in final app source |
| Final controller harness, serial run | 136 passed, including the production pipe reader and file-task transport |
| Final Beginner UI/session/report harness | 12 passed |
| Final menu-bar builder suite | 46 passed |
| Python lint/format, repository policy and whitespace checks | Passed |
| Generation, optimized scoring and template workers | Separate locked release builds with Rust 1.98.1 passed |
| Self-contained app build and packaged runtime checks | Passed |
| Developer ID signature, deep strict verification | Passed; team `S5BT2PRN5B` |

The final controller checks ran serially after a concurrent agent harness run
reported three failures; that concurrent result is not used as passing evidence.
The isolated serial run passed all 136 checks.

The app was built with:

```sh
PATH="$HOME/.cargo/bin:$PATH" uv run --no-project --python 3.13.13 python scripts/build_menubar.py --version 0.0.0-dev.beginner-recommendations
```

The builder verified the native entrypoints, complete current converter wheel,
locked Python dependency overlay, template helper, license/adaptation notices
and pinned OpenWork connector. Bundled Python loaded all five native profiles,
five benchmark cohorts and 54 public aggregate score rows.

Packaged score registry SHA-256:
`b72cd716b75957317b668d9be0029b1b21fc8a520f98169bc3027e265662d799`.
Packaged native profile SHA-256:
`d8ecb8d4b09249ea09268dbb71f7944828b9aa9c35a3ec0e5f65a2354edb9463`.

## Actual-machine recommendation check

The bundled backend checked this Mac's measured resources and existing library
without loading a model. Physical memory was 137,438,953,472 bytes (128 GiB).
Metal reported Apple M5 Max, unified memory and a recommended working-set budget
of 115,448,725,504 bytes. Its bounded scan returned 228 catalog entries. Coding
with Balanced offered:

| Choice | Existing copy | Estimated system / GPU memory |
| --- | --- | --- |
| Flint INT4 | No | 29 / 29 GiB |
| Qwen3.8-Flash-Next MLX-Serve mixed | Yes | 111 / 82 GiB |
| Qwen3.8-Flash-Next Lily Q4-XL | Yes | 111 / 82 GiB |

These are planner estimates and a current recommendation order, not measured
quality or proof that a model will start under memory pressure. Both Qwen
choices exposed their upstream coding reference. Two exports of one parent do
not create independent benchmark coverage. No weights were downloaded, and no
inference or task benchmark was run. MaxxBench remains deferred.

## Installation

The verified bundle was installed through the builder's staging, signature
verification and atomic replacement helper at `~/Applications/Stark.app`,
reusing the existing Developer ID team. The installed manifest reports source
`bfd1d6624534c819e9d45e6f012182c90354389d`, development channel and build number
`20261108203505543075`.

Installed signed executable SHA-256:
`56f9ca3bfbb44012c8d6b0eebc10c5b0c744797378750b6d2e86b2d367862f91`.

Installed manifest SHA-256:
`99e6a80118514397aea20690c5e7da80a7d5be1bd85d890c6611518f0bde8001`.

The final app processed its model inventory and displayed the new Work with
files card. The actual Guided Setup window completed Coding / Balanced
discovery and visibly recommended Flint INT4, showing the Mac's 128 GB memory
and 107.52 GB GPU-budget labels, a 22.58 GB download and 27 GB setup estimate.
The guide was left open on that recommendation; no download or model start was
initiated. The alternative choices in the table above were verified through
the bundled backend; their disclosure remained collapsed in the final window.
