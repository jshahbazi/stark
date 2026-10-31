# Qwen3.8 GDN recurrence and output norm/gate fusion

Date: 2026-10-31. Frozen model baseline:
`645ad5562f44926341b3a74001ed4243e40db3e9`.
This follows the [prework experiment](../qwen38-gdn-prework-2026-10-31/README.md)
and preserves its sources and results separately. No production dispatch selects
these kernels, and this record supersedes no published benchmark.
Status: GPU correctness and complete-model comparison finished. Neither tested
layout demonstrated a reliable throughput gain; both remain experimental.

## Measured outcome

Both layouts passed all four strict GPU controls (864 whole-array comparisons).
The component control repeated deployed-path exactness before timing and
produced identical state/history/gated-output hashes across all arms.
[Aggregate evidence and source hashes](gpu-results.json) preserve every sample
and the initial interrupted attempts; [offline checks](cpu-validation.json)
are separate. The completed model window and all later attempts are retained
in a [new source-bound model record](model-results-v2.json). The interim
`gpu-results.json` bytes remain unchanged.

| Complete-model arm | Measured requests | Pooled useful decode tok/s | Median tok/s | Observed change |
| --- | ---: | ---: | ---: | ---: |
| Unchanged baseline | 6 | 61.8485 | 61.9783 | — |
| Fused 256 | 6 | 58.5785 | 59.5337 | −5.29% |
| Fused 512 | 6 | 61.6187 | 62.4982 | −0.37% |

All 24 final calls, including six excluded warmups, reached 256 outputs and
matched the historical completion-token hash
`cf67e22bcf0d71a09bd998471f860b6da8fd4383a75c82b0bc5a3a00a2b7dafb`.
All six blocks passed their source, worker, checkpoint and tooling seals before
and after execution. Neither the checkpoint nor model precision changed.

These are descriptive observations under background CPU activity. Baseline
block throughput fell from 65.9176 to 58.2526 tok/s (−11.63%), exceeding both
observed candidate differences. Fused 256's two blocks changed by +4.19%;
fused 512's changed by −2.90%. This single window cannot establish a precise
causal regression, a sustained rate or a small improvement. It provides no
evidence for adopting either layout as an optimization.

The final monitor made 203 polls and recorded 203 distinct CPU compiler
process observations. It also recorded 41 Python observations, including one
inspected process with an `mlx` mapping below the 512 MiB exclusion threshold.
That observation is unconfirmed: memory size and library mappings cannot
establish whether it issued GPU work. There were zero fatal observations under
the final monitor policy. This is not evidence of GPU exclusivity or CPU
isolation. No own builds ran during timing; no clock/thermal or per-process
GPU utilization telemetry was captured.

### Component window

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

Seven earlier model attempts remain separate and excluded from the final
window's summaries. They contain 46 completed calls (13 warmups and 33 measured
calls), all with the historical completion hash, and four interrupted calls.
Three attempts stopped before any request. Every completed request's metrics
and hash are retained with its original order; failures were not discarded or
pooled into the final comparison.

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
worker. This yielded six measured requests per arm and 24 completed
requests in the final window. Every call resets model state; no
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
(25G241), AC power. The user released the GPU before the final comparison;
no other loaded model was observed at launch. The final nominal one-second
process watcher stops only our worker for named model/benchmark clients or
Python processes with GPU/model library mappings and RSS at least 512 MiB.
CPU compiler presence and Python heap size alone are recorded without stopping
the run. Smaller mapped Python and vanished processes are unconfirmed.

The 512 MiB threshold is a heuristic, not an activity measurement: genuine
lightweight GPU work can pass it. The controller's “lightweight library imports”
label describes that heuristic and cannot establish import-only or CPU-only
behavior. The one below-threshold `mlx` observation in the final window remains
an ambient confounder. Sampling can miss short activity or system clients;
the result does not establish exclusive GPU use. No foreign process was
signalled. The final run used only the previously reviewed small parent-1
daemon allowance. Its identity, mappings and memory bound were rechecked;
the earlier optional auxiliary-PID allowance was not used.

The original monitor rejected unknown Python at 64 MiB or any GPU/model mapping
and stopped on CPU compilers. Those rules caused interruptions on processes
without established GPU activity. After the user's instruction to run the test
and explicit GPU release, three separately preserved controller revisions
removed the RAM-only exclusion, recorded CPU compilation and finally
introduced the mapped-Python threshold. The native model sources, binaries,
checkpoint, sampling, precision, strict numerical tests and golden hash did
not change. The current and previous controller hashes are recorded separately.
The identified Planetable adapter was reviewed as a generic dispatcher and was
never whitelisted by name, path or hash.

| Earlier attempt | Complete calls | Warmups | Measured | Interrupted calls | Stop condition |
| --- | ---: | ---: | ---: | ---: | --- |
| v1 | 1 | 1 | 0 | 1 | Earlier Python memory threshold |
| v2 | 0 | 0 | 0 | 0 | Preflight Python/build checks |
| v3 | 0 | 0 | 0 | 0 | Preflight compiler activity |
| v4 | 9 | 3 | 6 | 1 | Earlier Python memory threshold |
| v5 | 0 | 0 | 0 | 0 | Resumed mapped Python and native model worker |
| v6 | 20 | 5 | 15 | 1 | CPU compiler check in the final baseline warmup |
| v7 | 16 | 4 | 12 | 1 | Below-threshold MLX mapping; actual GPU activity unknown |

Additional availability checks and a bounded background wait performed no
inference and are retained separately. The original v1 controller summary
miscounted its interrupted call as completed; the immutable report is retained,
and published counts are derived from row statuses. The separately preserved
v1–v3 report/log hashes and counts still match the interim public record.
No failed or interrupted attempt is used to qualify a speedup.

In the final JSON, protocol qualification means that the requested call matrix,
output hashes, declared monitor policy and source seals passed. It does not
qualify a performance improvement or production dispatch. Worker readiness
does not expose precision metadata; sealed source and binaries bind that
policy. Synthetic byte identity and a fixed seeded continuation do not
establish full-vocabulary readout identity, broad answer quality or
cross-engine parity. The baseline's existing qualification boundaries remain.

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
