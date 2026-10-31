# Qwen3.8 GDN recurrence and output norm/gate fusion

Date: 2026-10-31. Frozen model baseline:
`645ad5562f44926341b3a74001ed4243e40db3e9`.
This follows the [prework experiment](../qwen38-gdn-prework-2026-10-31/README.md)
and preserves its sources and results separately. No production dispatch selects
these kernels, and this record supersedes no published benchmark.
Status: strict GPU correctness complete; full-model throughput pending a clean
window because other sessions are repeatedly compiling.

## Measured outcome so far

Both layouts passed all four strict GPU controls (864 whole-array comparisons).
The component control repeated deployed-path exactness before timing and
produced identical state/history/gated-output hashes across all arms.
[Aggregate evidence and source hashes](gpu-results.json) preserve every sample
and interrupted attempt; [offline checks](cpu-validation.json) are separate.

| Component arm | GPU µs per layer, mean | Time change |
| --- | ---: | ---: |
| Production convolution + recurrence + norm/gate | 66.0629 | — |
| Production convolution + fused 256 | 76.9724 | +16.51% |
| Production convolution + fused 512 | 76.7853 | +16.23% |

This is one noisy synthetic window: six measured samples per arm, each over
36 serialized independent layer instances, in mirrored arm order and with
the first full cycle excluded as warmup. Baseline samples span 39.36–94.60 µs;
all samples are retained. Projections are excluded. Both candidate means are
slower; this does not establish a complete-model regression or a precise
causal cost. No component speedup was demonstrated.

Three full-model attempts have been retained separately. The first completed
only a baseline warmup and was interrupted during its first measured call;
the other two were blocked before any request. There is no measured full-model
request or candidate full-model correctness result yet. The requested
24-request comparison remains pending, with workers already built and sealed.

## Experiment

[fused.metal](fused.metal) fuses the existing official ValueKey recurrence with
its output RMS normalization and sigmoid gate. The production convolution,
input and output projections, checkpoint, stored quantization and sampling
stay fixed. This isolates a different hypothesis from the prior fused
convolution/preparation experiment.

One threadgroup owns all 128 value rows of one head. Each lane retains four
contiguous key elements and the original per-row operation and SIMD reduction
order. Lane zero puts each BF16-rounded raw readout into threadgroup memory.
After a uniform barrier, SIMD group zero runs the original four-contiguous-
elements-per-lane normalization and gate. It retains the raw BF16 boundary,
the three subsequent BF16 rounding points, the custom K1 sigmoid and the
original precise inverse square root. State remains F32 `[48,V128,K128]`.

| Per GDN layer | Unchanged baseline | Fused 256 | Fused 512 |
| --- | ---: | ---: | ---: |
| Recurrence threadgroups | 192 | 48 | 48 |
| Threads per recurrence group | 256 | 256 | 512 |
| Rows per SIMD group | 4 | 16 | 8 |
| State scalars in each thread's source tile | 16 | 64 | 32 |
| Convolution/recurrence/norm dispatches | 3 | 2 | 2 |
| Raw device write plus read bytes | 24,576 | 0 | 0 |
| New model scratch bytes | 0 | 0 | 0 |

Both variants use 1,800 bytes of declared threadgroup scratch. The normal
shader build has `RESEARCH_CAPTURE_RAW=0`; its diagnostic raw device buffer is
unused. Capture builds add the raw store solely for intermediate exactness
checks. The model experiments use the same no-capture shader tested by the
deployed-path controls. This saves 36 norm dispatches per decoded token while
reducing recurrence groups fourfold and increasing the source state tile.
Actual register allocation, spills and GPU occupancy were not measured.

## Correctness

The ignored opt-in harness,
[`gdn_fusion_research.rs`](../../../../vendor/lily-core/tests/gdn_fusion_research.rs),
compares against the unchanged production convolution, ValueKey raw recurrence
and norm/gate kernels. Each layout has separate deployed and capture tests;
failure of one does not prevent the other tests from running. All four passed.

Each test covers three synthetic fixtures and 16 chained steps. These include
zero and dense F32 state, non-dyadic inputs, adversarial BF16 gate values,
extreme finite z values, signed zeros and varied F32 normalization weights.
Every step poisons output buffers, checks surrounding canaries and finiteness,
and compares whole arrays byte for byte. Deployed tests compare convolution
history, convolved QKV, F32 state and BF16 gated output; capture tests also
compare BF16 raw output. The unused raw buffer must remain fully poisoned in
deployed builds. No tolerance or golden was changed.

Initial controls passed 864 whole-array comparisons over 192 advances across
the four candidate chains. The component test repeated the two deployed
controls, another 384 comparisons, before timing.

[check_cpu.py](check_cpu.py) independently enumerates all 786,432 state,
6,144 raw and 6,144 gated-output owners for each geometry and compiles both
entrypoints with capture disabled and enabled using Safe Metal 3.1 and warnings
as errors. It does not open a GPU context. Its source-bound aggregate and GPU
results are published separately from the earlier prework evidence.

## Protocol and scope

The [source preparation script](prepare_model_trial.py) archives frozen
`645ad556` into an untouched baseline and two isolated candidate sources.
Candidate patches change only the fixed official ValueKey model route, its
validated wrapper and the new shader. There is no environment selector or
additional model allocation. Each source has its own Rust 1.98.1 locked
release-worker target. Builds finish before timed model work starts.

The [model controller](benchmark_model.py) uses blocks in
baseline/fused256/fused512/fused512/fused256/baseline order. Each block loads
one worker, excludes one warmup and measures three requests, then closes that
worker. This yields six measured requests per arm and 24 total completed
requests when all controls succeed. Every call resets model state; no
speculation or prefix reuse is enabled.

The official `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit` checkpoint is
revision `788c07a8725642bccf13a0a3b8a21954ab84b4f2`. Its stored quantization,
BF16 activation boundaries, F32 recurrent state and Q8 affine64 KV policy
remain fixed. The workload is the same pinned 4,096-token corpus prefix,
256 outputs, temperature 1, top-k 20, top-p .95, seed 42 and context 4,352.
Useful decode throughput counts 255 post-prefill outputs divided by the native
decode timer, excluding prefill and transport.

The controller binds all source files, manifests, workers and tooling before
and after every block. It verifies the official 113-file checkpoint against
the existing payload-hash receipt and current size/mtime/inode identities,
and hashes the actual tokenizer and config. It does not rehash 107 GB of
weights during the experiment. Every output must match the historical seeded
completion hash; no prompts, responses, token vectors or probabilities are
published.

Host: Mac17,7, Apple M5 Max with 40 GPU cores, 128 GiB, macOS 26.7.1
(25G241), AC power. The previously reserved chat was idle/completed before
GPU use. The process watcher checks metadata every second and stops only our
worker if it observes model or benchmark competition or a build. A small
parent-1 Python daemon was reviewed by executable and mapped-library metadata
and monitored with a memory bound. A later optional
`--reviewed-auxiliary-manifest` binds two specifically reviewed Python
processes. The private manifest retains their identities. They had only
standard-library/compression/OpenSSL/SQLite text mappings and no GPU/model
mapping markers. Each has a strict 512 MiB RSS ceiling; the manifest binds
PID, parent, command basename, start time and text-mapping path-set hash/count.
Its SHA-256 is
`236fd9ba9187e110aec796a4f9050f69cc874d63a5b76567bdb90dfa01f0b2f2`.
Every poll rechecks the exact identities, mappings, marker absence, memory
bounds and manifest bytes. Missing processes or failed inspection stop the run.
Only the listed PIDs are exempt; children continue to be scanned. An actual
metadata probe still rejected their cargo/rustc children. Unknown high-memory
Python and compiler rules remain unchanged. The manifest hash is also checked
before and after every model block and published with the results.
Mapping hashes cover paths, not file contents or Python script arguments.
This cannot establish exclusive GPU use
or exclude short/system activity. GPU clock and thermal telemetry were not
captured. Worker readiness does not expose precision metadata; the sealed
sources establish the precision policy.

Synthetic byte identity and a fixed seeded continuation do not establish
full-vocabulary readout identity, broad answer quality or cross-engine parity.
The baseline's existing qualification boundaries remain. Results from this
window must be interpreted with sample variation, process order and baseline
drift; component and whole-model measurements have separate scopes.

The first model attempt completed its baseline warmup with the historical
output hash, then was stopped during the first measured request by the
watcher's high-memory Python threshold. That process vanished before mapping
inspection, so this is unconfirmed competition. No measured request completed;
the attempt contributes no comparative timing. A subsequent preflight quiet
interval was also stopped by the same condition before loading a model.
Those records and the first controller's source bytes are retained separately.
The controller's summary was then corrected to count completed calls separately
from interrupted attempts; sampling, process classification and timing policy
were unchanged in that revision. The second model attempt stopped in preflight
on Python memory thresholds and an active compiler. A metadata-only observer
also inspected one separate transient 294 MiB Python process with standard
library mappings; that does not identify or exonerate the vanished earlier
processes. Its bounded, incomplete trace remains private and hash-bound.
After the explicit two-PID allowance was added and independently reviewed,
96.8 seconds of preflight waiting included a final 45-second clear interval.
The third model controller
then correctly stopped on newly resumed cargo/rustc work before a request.
No failed timing window is pooled or used to claim a speedup.

## Provenance and reproduction

The shader adapts the unchanged native `gdn.metal` ValueKey recurrence and
`gdn_normgate_value_key_mlx`, retaining its source hashes and the Apple and
David Dalcu MIT notices. The helper source hashes are embedded in the shader;
upstream license records remain in `vendor/lily-core/NOTICE` and
`LICENSE.mlx-serve`. No new external kernel or package dependency was added.

Run correctness only with an available GPU slot. The filter runs four separate
tests, and the Rust test runner retains individual results if one fails:

```sh
uv run --python 3.12.13 docs/archive/experiments/qwen38-gdn-fusion-2026-10-31/check_cpu.py
CARGO_TARGET_DIR=runtime/gdn-fusion-test-target cargo +1.98.1 test --manifest-path vendor/lily-core/Cargo.toml --locked --release --test gdn_fusion_research matches_production -- --ignored --nocapture --test-threads=1
CARGO_TARGET_DIR=runtime/gdn-fusion-test-target cargo +1.98.1 test --manifest-path vendor/lily-core/Cargo.toml --locked --release --test gdn_fusion_research fused_component_rotating_after_exactness -- --ignored --nocapture --test-threads=1
```

Source preparation refuses to overwrite prior snapshots. The model controller
requires completed source-bound build receipts. Runtime copies, workers and
private receipts stay ignored under `runtime/gdn-fusion-gpu-20261031/`.
