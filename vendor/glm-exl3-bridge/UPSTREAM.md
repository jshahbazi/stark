# GLM EXL3 engine bridge

This directory contains Stark's C ABI wrapper and a bounded patch to a pinned native engine. It contains no server, llama runtime, application integration, model weights, MLX dependency source tree, compiler, or generated binaries.

The supported source composition is mlx-serve `dc92a3e88ad193a82c70f24ef96918b8dc1d4af6`, its matched mlx-stream gitlink `7d29eaa7217ea8d0e1b50000eb23ad37b06f5d80`, MLX `64ea011cb65f14d9ce2737e60db9a4ae91ed7441`, mlx-c `56b2d39fc831f2c0eb5bb94d82ef7191f7b31fa6` with the host's pinned gather-QMM compatibility patch, and sushi `0bc61aa690cadfed63637997f86b5de641d6645c`. Exact compiler and dependency artifact hashes are in `lock.json`. New validation records must identify this complete tuple; it is not a reproduction of a historical executable or the historical 1,093-token/s result.

The MLX source additionally carries `patches/mlx-get-wired-limit.patch`: a read-only public getter for the Metal allocator's current wired capacity. The allocator locks its existing mutex while reading; the getter changes no residency. The owned `src/mlx_wired.cpp` adapter returns only a sanitized integer status. The previous capacity is observed before any setter, because upstream's setter can mutate its limit before a resize allocation throws. The original MLX revision, exact patch and complete Git working-tree diff hashes remain bound in the lock and stage receipt. Existing MLX-C, JACCL and NAX metallib artifacts are retained when preparing this getter-only dependency stage; all artifact and installed-header hashes must match the lock.

`src/host/{mlx,dsa_nax,crt,os_mutex,io_util,mtp_acceptance}.zig` are extracted from that host. Their original hashes are in `imported-files.json`. Stark changes are limited to a silent local logging adapter, mandatory non-exiting MLX error handling, and omission of test-only DSA helpers that import the full server. Preserve the source files' original notices. `licenses/` includes the upstream MIT license/NOTICE files and Apache-2.0 text; host DSA/KDA code includes oMLX Apache-2.0 material and MLX MIT material. The external plugin retains its own MIT/Apple/turboderp notices. Model and pack licensing is separate.

## Explicit local build

Cargo must link an existing, validated stage. It must never invoke this script, download a dependency, or build native code implicitly.

First prepare the exact external source/toolchain/MLX stage under ignored storage. Dependency preparation is separate from this wrapper. Then run the following explicitly with the corresponding absolute local paths:

```sh
uv run --no-project --python 3.13.13 python vendor/glm-exl3-bridge/tools/prepare_stage.py \
  --host /absolute/ignored/mlx-serve \
  --plugin /absolute/ignored/mlx-stream-host823 \
  --mlx-source /absolute/ignored/mlx-wired-getter-source \
  --mlx-stage /absolute/ignored/mlx-wired-getter-stage \
  --zig /absolute/ignored/mlx-serve/.zig-toolchain/zig \
  --output /absolute/stark/runtime/glm-exl3-stage
```

The output must be new and Git-ignored. The script verifies source/compiler/artifact pins and the complete installed MLX/private-MLX-C header inventory and hashes. It verifies the approved MLX-C working-tree patch exactly, including staged changes. It extracts the plugin and sushi directly from their pinned Git commits into private output directories, applies `patches/mlx-stream-prefix.patch` to that private plugin, builds the static library and benchmark, and runs the model-free C ABI smoke test. Working-tree edits or untracked files in the external plugin/sushi checkout are never compiled. It does not load a model or run inference. No shared upstream checkout is modified. Local build caches live under the ignored output; the compiler's shared global cache remains outside the repository. It writes `stark-glm-stage.json`.

The supplied MLX source must be the original pinned commit with only the exact getter patch applied. Preparation checks its complete diff, including staged changes, and rejects untracked source imports. Build the patched MLX in a new ignored source/build/stage using the original Release/shared-library flags, macOS deployment target 26.2, matching SDK, tests/examples disabled and four build jobs. Reuse the pinned dependency sources and unchanged NAX metallib; do not replace an earlier dependency stage or its validation receipt.

The exact underlying build is Zig `build test` followed by Zig `build --prefix <output>`, both with `--cache-dir <output>/build-cache`, `-Doptimize=ReleaseFast`, `-Dhost-dir=<pinned host>`, `-Dplugin-dir=<patched private copy>`, `-Dsushi-dir=<private pinned sushi archive>`, `-Dmlx-stage=<pinned stage>`, and `-Dexperimental-prefix=false`. The deployment floor is macOS arm64 26.2. The link needs the staged MLX/MLX-C pair, matching private headers, libc++, Metal, Foundation, CoreFoundation, IOSurface, and IOKit. MLX's `libjaccl.dylib` and `mlx.metallib` remain in its pinned stage; do not substitute system/Homebrew libraries.

Add `--experimental-prefix` to compile the prefix implementation. Its C capability bits are full F32 readout, serial decode and prefix snapshot (`1 | 2 | 4`); a serial-only stage reports `1 | 2`. Capability metadata reports implementation availability. Artifact qualification and the engine's decision to enable reuse are separate; compilation alone does not qualify restoration.

## ABI and lifecycle

`include/stark_glm.h` is the contract. The ABI uses fixed-width values, bounded borrowed input slices, caller-owned readout storage, stable sanitized statuses, and opaque handles. Load copies the model path; no borrowed path or input token pointer survives a synchronous call. Snapshot capture copies source-prefix tokens privately. Never persist or log snapshots or their token vectors.

Only the atomic cancel setter may run concurrently. Load, forward, reset, snapshot operations, drain and destruction belong to the loading inference thread. The Rust owner must be `!Send` and `!Sync`. Callers must not race destruction with cancellation. A model/MLX reader is single-flight process-wide.

`prefill` is a new request from reset; `step` consumes one token after a successful prefill. Both evaluate the last row's full F32 vocabulary readout before returning; `readout_f32` checks every value is finite before copying any caller-owned word. It does not advance the model. The bridge calls `Module` directly, bypassing upstream's token-substituting sampler wrappers. MTP is disabled. The decoder follows the pack's FP8/BF16/EXL3 numerical route; F32 readout storage does not imply full-F32 inference.

`stark_glm_get_rows` returns the actual admitted prompt and decode expert rows from `Module.rows`, which is set by the completed admission plan and used to construct the expert source. The benchmark prints these as `expert_prefill_rows` and `expert_rows`; requested config pins are not substituted for installed counts.

Load uses the native host's admission census before billing: it counts and evicts this pack's file cache, charges remaining foreign file cache, and retains the non-file memory floor. After loading and draining, the bridge installs a process-local MLX wired capacity bounded by the actual admitted model's process bill, the ceiling less margin/reserve and admitted baseline, and Metal's recommended maximum working set. It does not read a wiring-policy environment variable or change macOS settings. `stark_glm_get_wired_capacity` returns the installed capacity and saved previous setting; the benchmark emits `wired_capacity_bytes` and `previous_wired_capacity_bytes`. The previous setting may exceed this model's budget. Destruction and load-error cleanup release model buffers and clear MLX's free-buffer cache before restoring it. Restoration failure returns stable `DEVICE_FAILURE`; destruction still nulls the released handle.

Request argument failure invalidates prefix state and blocks that request until reset. GPU/I/O execution failure fences the model and requires destruction/reload. An owning-thread operation that observes cancellation invalidates reuse. Reset after a clean cancellation drains and clears cancellation; a fenced context stays fenced. Wrong-thread rejection performs no mutation. Destroy nulls the caller's handle after teardown even if draining returned an error; wrong-thread destroy leaves the handle live. A missing snapshot from `take_prefix_snapshot` is an ordinary `OK` plus null output, not a failed request.

The ceiling is bounded at 112 GiB and a zero margin means a 2 GiB safety margin. Pack row pins are cleared unless explicitly supplied in C load options; over-budget explicit pins are rejected against both phase bills before weight materialization. Serial builds use the native bill's automatic fill. Profile probes compile out, raw log adapters are silent, MTP calibration/dumps and expert tally/usage persistence are disabled. Seeded weight residency can follow the pack; no prompt/token/output diagnostic is emitted.

## Experimental prefix patch

`patches/mlx-stream-prefix.patch` adds `stark_prefix.zig` and narrow module/root hooks. Capture occurs after an existing full 16,384-row prompt call, after evaluating states, flushing expert I/O and synchronizing. It does not split a call or change expert GEMM shapes. The checkpoint holds CPU deep copies of contiguous F32 KDA convolution/recurrent states and BF16 DSA open-pool tails, plus private source tokens, owner identity, epoch, context and boundary. A snapshot retains no aliases to appendable DSA backing arrays. A successful resume restores CPU state and logical lane lengths, validates the full source prefix, then executes the suffix through the existing 16,384-row call schedule. A fresh prompt that can overwrite the prefix increments the owner epoch; failure, cancellation and model destruction invalidate it. Normal request end/reset can preserve only the successful immediately preceding prefix. Every request after capture must retain that prefix.

The patch supports MTP-off only. Recurrent state is about 152.6 MB per checkpoint for Flash. Experimental admission additionally reserves twice the KDA state bytes, twice the tail bytes, and context-sized token ownership to cover host copies and restore peaks. Caller-owned snapshots must be freed explicitly, including after model destruction.

Compilation and model-free ABI checks do not prove restoration equality. Before promoting the capability, qualify fresh versus restored full F32 readouts, greedy sequences, complete KDA/DSA state hashes, changed suffixes, source/model/context mismatch, repeated consecutive requests, failure and cancellation, including DSA pooling/top-k transitions. `stark_glm_validation_state_sha256` is available only in an explicit experimental build: it synchronizes and hashes every KDA conv/recurrent tensor, each DSA latent/index lane's valid logical prefix (including completed-pool storage), open tails and their ends, and context/position. It excludes token IDs and owner epochs/counters, and returns only 32 digest bytes. This expensive operation is outside benchmark timings. The 1,024-row attention span is not a supported reuse boundary: the whole-call MoE batch also matters.

## Benchmark and receipt

`stark-glm-bench` loads once, performs one untimed warmup, then requires at least two measured reset runs. Timings include the final full readout copy; decode includes host argmax and readouts. Each run emits aggregate durations/rates and independent prefill/final FNV-1a64 readout hashes. It never prints model paths, prompts, tokens, logits, or probabilities. Supply U32LE IDs through `--tokens -` on stdin to avoid persisting vectors; input allocation and reads are bounded by context minus decode budget before allocation. Run inference only under the coordinating worker's GPU serialization and memory/quiet-host controls.

`lock.json` contains `owned_sources`, sorted by POSIX relative path, with each file's SHA256. `bridge_source_sha256` is SHA256 of the concatenation, for each entry in that order, of UTF8 path bytes, one NUL byte, raw file bytes, one NUL byte. The owned set is build.zig/build.zig.zon, LICENSE, every file under src/include/tests/tools/patches/licenses excluding `__pycache__`. The lock and this documentation are excluded to avoid circular hashes and doc-only rebuilds.

Stage receipt version 1 carries the ABI/source/dependency/patch pins, `experimental_prefix`, `mlx_lib_dir`, and `artifacts` keys `stark_glm`, `mlx`, `mlxc`, `jaccl`, `metallib`, each with `path` and `sha256`. The static library path is relative to the receipt; external dependency paths are absolute. Cargo validates every hash and all expected pins against this lock before linking. There is no network/build fallback.

`mlx_getter_patch_sha256` and `mlx_worktree_diff_sha256` identify the additional MLX getter patch and actual complete source diff; both must equal the lock, alongside the original `mlx_revision`.
