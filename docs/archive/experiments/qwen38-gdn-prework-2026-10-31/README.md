# Qwen3.8 GDN prework experiment

Date: 2026-10-31. Baseline: `c3c83fcd851bba751744861d80f45e1b933fbd71`.
Status: GPU-tested research prototype; neither variant demonstrated a
full-model speedup. No production dispatch selects these kernels.
This record does not supersede an earlier throughput or numerical result.

## GPU outcome

Both variants passed the synthetic exactness controls and reproduced the
historical full-model completion on all tested requests. Keep them as research:
the measured complete-model rates were lower than the unchanged baseline.
[Source-bound aggregates and hashes](gpu-results.json) retain every request,
both component windows and the interrupted attempt separately.

| Full-model arm | Measured requests | Pooled useful decode tok/s | Median tok/s | Change from baseline |
| --- | ---: | ---: | ---: | ---: |
| Unchanged baseline | 6 | 65.3825 | 65.6701 | — |
| Prepared, staged recurrence | 6 | 64.7676 | 64.9926 | −0.94% |
| Prepared, direct recurrence | 6 | 62.9889 | 63.2265 | −3.66% |

These are observations from one mirrored, cross-process window. Baseline block
throughput fell from 66.3034 to 64.4868 tok/s (−2.74%); this drift is larger than
the staged variant's difference. Six measured requests per arm do not establish
a precise causal regression or a sustained throughput estimate. There is no
evidence here for admitting either variant as an optimization.

The redundancy identified below is real. Removing it did not improve measured
complete-model throughput. The direct variant adds repeated prepared-input
loads, and both variants change convolution parallelism and scheduling. The
experiments do not isolate which cost offsets the saved preparation work.

## Why this piece

The [optimization ledger](../../../benchmarks/QWEN38_OPTIMIZATION_HISTORY.md)
records the largest historical gains from removing host transfers, repeated
submissions and full-vocabulary CPU sorting. Those fixes already exist. For the
current official checkpoint, the strongest remaining stage evidence is GDN:
36.39% of an intermediate instrumented GPU capture, followed by MoE at 28.42%.
The capture predates final changes, overlaps inclusive/child intervals, and can
perturb scheduling. It does not establish the current bottleneck or the largest
GDN subphase. See the [matched report](../../../benchmarks/QWEN38_MATCHED_OPTIMIZATION_2026_10_30.md#profile-attribution-diagnostic-only).

The official pack uses Q8 GDN projections; the earlier Lily pack uses Q4.
Stark's official Q8 projection already adapts MLX's stock fast QMV. Previously
rejected Q4 packing, split-K, wider-lane and reassociation experiments provide
little reason to repeat them indiscriminately.

Source inspection found avoidable work in the ValueKey recurrence. Four value
partitions each prepare Q/K and gates; three value heads share each key head.
Each distinct Q/K row is normalized twelve times, and each gate pair four times.
The recurrence is already hand-written Metal. Removing this duplicated work is
a more concrete experiment than translating the same expressions again.

## Candidate

[prepared.metal](prepared.metal) replaces convolution plus inline recurrence
preparation with fused convolution/preparation followed by the existing
state-update structure. One SIMD group owns each of 16 Q, 16 K and 48 V heads.
Each lane computes four contiguous channels, retaining channel-major convolution
history and tap-major weights. State stays F32 in `[48,value=128,key=128]` order.

| Per GDN layer | Baseline | Candidate |
| --- | ---: | ---: |
| Q/K row normalizations | 384 | 32 |
| Gate-pair evaluations | 192 | 48 |
| Convolution channel evaluations | 10,240 | 10,240 |
| Convolution/preparation + recurrence + norm dispatches | 3 | 3 |
| QKV/prepared scratch bytes | 20,480 | 20,480 |
| Additional gate scratch bytes | 0 | 384 |

Prepared Q/K values retain the two BF16 rounding boundaries, then remain BF16
in scratch. V is the existing BF16 convolution/SiLU result. Gate addition,
softplus and decay remain F32 with the original intrinsic policy. Stored
weights, checkpoint quantization and recurrent-state precision are unchanged.

Two recurrence entries separate the hypotheses:

- `qwen38_gdn_recurrence_staged_research` retains the existing threadgroup
  staging and barrier, isolating shared preparation.
- `qwen38_gdn_recurrence_research` reads prepared Q/K directly and broadcasts V
  and gates within each SIMD group. It removes threadgroup scratch and the
  barrier, while introducing more repeated device-load instructions.

The intended arithmetic order matches the baseline. Moving code across kernels
can change compiler contraction and scheduling; source correspondence alone
does not establish numerical identity.

## External research and provenance

Newer MLX-Serve source implements the same broad prework decomposition:
[gdn_prework.metal](https://github.com/ddalcu/mlx-serve/blob/1271abe0f1937f0bcdf85c558b2cba12484af65a/src/kernels/gdn/gdn_prework.metal)
and [prepared recurrence](https://github.com/ddalcu/mlx-serve/blob/1271abe0f1937f0bcdf85c558b2cba12484af65a/src/kernels/gdn/decode_k1p.metal).
These are algorithm references. The prototype adapts Stark's existing admitted
K1 expressions and native history layout. Newer generic gate casts and buffer
layouts are not interchangeable with the current contract. MLX-Serve's
[MIT license](https://github.com/ddalcu/mlx-serve/blob/1271abe0f1937f0bcdf85c558b2cba12484af65a/LICENSE)
was inspected; the prototype retains its existing source notices and source
hashes.

The inspected [MLX QMV source](https://github.com/ml-explore/mlx/blob/f7f86bd71edd29bd5073c767b96c0b96e7cdf3c9/mlx/backend/metal/kernels/quantized.h)
does not supply an obvious new single-token replacement. The advertised
few-row improvements in [MLX #4781](https://github.com/ml-explore/mlx/pull/4781)
leave M=1 unchanged; those gains do not support this workload.

## Validation

[check_cpu.py](check_cpu.py) enumerates all channel, history, gate, state and raw
output owners and checks history updates against independent dense streams.
It compiles all three entry points to AIR and LLVM IR with Safe Metal 3.1,
using only the offline compiler. This checks ownership and compilation, not
GPU arithmetic or speed. All three compiled with warnings treated as errors;
the inherited C++17-extension warning is explicitly suppressed. The initial
[CPU aggregate](cpu-validation.json) binds the original shader, reference and
harness hashes. The later GPU aggregate separately binds its updated timing
harness; the tested shader bytes remained unchanged. Rust 1.98.1 compiled the
ignored integration tests with locked dependencies.

```sh
uv run --python 3.12.13 docs/archive/experiments/qwen38-gdn-prework-2026-10-31/check_cpu.py
```

The GPU harness is
[`gdn_prework_research.rs`](../../../../vendor/lily-core/tests/gdn_prework_research.rs).
Its strict control covers three synthetic fixtures, two recurrence variants and
16 steps per chain: 96 advances and 480 complete-array comparisons. It uses
non-dyadic values, zero/dense state, adversarial BF16 gates, NaN-poisoned outputs
and surrounding canaries. Both variants passed this matrix in the initial
control and in the subsequent release-host timing run. The component test
reruns correctness first, then measures 36 serialized
convolution/recurrence pairs in mirrored arm order with six measured samples
per arm; projections and norm/gate are outside that component.

Run these controls only with an available GPU slot:

```sh
CARGO_TARGET_DIR=runtime/research-target cargo +1.98.1 test --manifest-path vendor/lily-core/Cargo.toml --locked --test gdn_prework_research prepared_gdn_matches_materialized_production_chain_exactly -- --ignored --nocapture --test-threads=1
CARGO_TARGET_DIR=runtime/research-target cargo +1.98.1 test --manifest-path vendor/lily-core/Cargo.toml --locked --test gdn_prework_research prepared_gdn_component_abba_after_exactness -- --ignored --nocapture --test-threads=1
```

### Component windows

| Host harness/window | Baseline µs/layer | Staged µs/layer | Direct µs/layer |
| --- | ---: | ---: | ---: |
| Original debug-host control, v1 | 54.7677 | 54.4169 | 52.8389 |
| Release-host control with per-sample output, v2 | 53.1489 | 54.3343 | 55.4304 |

Each window has six measured samples per arm, preceded by an excluded warmup
cycle, and the same exact final state/history/raw-output digest. Their ranking
reverses. The release-host baseline includes a 32.9931 µs sample followed by
54–59 µs samples; no measured sample was discarded. Host gaps and scheduling
differ between the two harness profiles. These windows establish no reliable
component gain and are not pooled together.

### Complete-model window

The source preparation script,
[`prepare_model_trial.py`](prepare_model_trial.py), archived exact base
`645ad5562f44926341b3a74001ed4243e40db3e9` into a pristine baseline and two
isolated source copies. Candidate copies add only the prework wrapper, fixed
recurrence entry, 384-byte F32 gate scratch and guarded official-ValueKey model
route. Prepared BF16 Q/K/V reuses the existing convolution scratch. Each copy
has its own Cargo target and sealed Rust 1.98.1 locked release worker; no
environment selector or default production route was added.

[`benchmark_model.py`](benchmark_model.py) ran blocks in
baseline/staged/direct/direct/staged/baseline order. Each block loaded one
worker, ran one excluded warmup and three measured requests, then closed it
before loading another. This is a comparison across fresh processes and loads,
not a same-loaded-model toggle.

All calls used the official `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`
checkpoint at revision `788c07a8725642bccf13a0a3b8a21954ab84b4f2`, the pinned
4,096-token corpus prefix, 256 outputs, temperature 1, top-k 20, top-p .95 and
seed 42. Stored weights, quantization, primitive rounding, F32 recurrent state
and Q8 affine64 KV policy stayed fixed. Speculation and prefix reuse were
disabled. Rates count 255 useful post-prefill outputs divided by the worker's
decode timer; prefill and transport are excluded. Every measured request,
including the slow ones, remains in the aggregate.

All 24 calls completed at length with the historical output-token hash
`cf67e22bcf0d71a09bd998471f860b6da8fd4383a75c82b0bc5a3a00a2b7dafb`.
Before and after every block, the controller checked source, worker, checkpoint
metadata and tooling identities. The prior complete payload-hash receipt still
matched all 113 files' stat identities; weights were not rehashed during this
experiment. Independent review checked the 24 call records, six block seals,
three source/build/binary bundles and summary/comparison arithmetic.

Host: Apple M5 Max, 40 GPU cores, 128 GiB, macOS 26.7.1 (25G241), AC power.
The prior GPU session had finished before the run. A metadata watcher recorded
198 polls without an observed competing model or build. A small preexisting
Python daemon was explicitly reviewed and monitored; one other Python process
vanished before inspection. One-second process sampling cannot establish
exclusive GPU use or exclude system clients. GPU clock and thermal telemetry
were not captured. Worker readiness exposed no precision metadata; precision
binding comes from the sealed sources, not those readiness responses.

An earlier model attempt was stopped during its first warmup by the initial
watcher's blanket classification of a transient Python process. No model
request completed and no timing or numerical conclusion is taken from it.
The revised watcher checks memory/mapped-library metadata for unknown Python
processes. Both attempts and their controller identities are retained; the
earlier result was not rewritten.

Synthetic byte identity and one fixed seeded continuation do not establish
full-vocabulary readout identity, broader answer quality or cross-engine
equivalence. The baseline's existing numerical qualification boundaries remain.

## Further research

There is no speedup estimate from the twelvefold normalization reduction:
normalization is only part of GDN. Preparation reduces thread count and changes
memory access, occupancy and the overlap between state loads and gate math.
The convolution mapping changes from 10,240 channel threads to 2,560 threads
that each process four channels.
The direct variant can lose despite removing a barrier. Even a component win
can disappear behind Q8 projection bandwidth or whole-model scheduling.

One further aggressive experiment in the same piece would be
one threadgroup per value head with recurrence plus output norm/gate fusion.
That could remove the raw-output round trip and 36 norm dispatches per token.
It also reduces available groups fourfold and multiplies the per-thread state
tile, so register pressure can dominate. It is a distinct experiment and is
not implemented or qualified here.
