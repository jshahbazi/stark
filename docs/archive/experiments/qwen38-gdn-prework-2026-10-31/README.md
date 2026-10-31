# Qwen3.8 GDN prework experiment

Date: 2026-10-31. Baseline: `c3c83fcd851bba751744861d80f45e1b933fbd71`.
Status: research prototype; no production dispatch selects these kernels.
This record does not supersede an earlier throughput or numerical result.

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

## Validation and next decision

[check_cpu.py](check_cpu.py) enumerates all channel, history, gate, state and raw
output owners and checks history updates against independent dense streams.
It compiles all three entry points to AIR and LLVM IR with Safe Metal 3.1,
using only the offline compiler. This checks ownership and compilation, not
GPU arithmetic or speed. All three compiled with warnings treated as errors;
the inherited C++17-extension warning is explicitly suppressed. The
[CPU aggregate](cpu-validation.json) binds shader, reference and harness hashes.
Rust 1.98.1 compiled the two ignored integration tests with `--locked --no-run`.

```sh
uv run --python 3.12.13 docs/archive/experiments/qwen38-gdn-prework-2026-10-31/check_cpu.py
```

The GPU harness is
[`gdn_prework_research.rs`](../../../../vendor/lily-core/tests/gdn_prework_research.rs).
Its strict control covers three synthetic fixtures, two recurrence variants and
16 steps per chain: 96 advances and 480 complete-array comparisons. It uses
non-dyadic values, zero/dense state, adversarial BF16 gates, NaN-poisoned outputs
and surrounding canaries. These are prepared controls, not executed passes.
The component test reruns correctness first, then measures 36 serialized
convolution/recurrence pairs in mirrored arm order with six measured samples
per arm; projections and norm/gate are outside that component.

After the other session releases its GPU reservation, run correctness first:

```sh
CARGO_TARGET_DIR=runtime/research-target cargo +1.98.1 test --manifest-path vendor/lily-core/Cargo.toml --locked --test gdn_prework_research prepared_gdn_matches_materialized_production_chain_exactly -- --ignored --nocapture --test-threads=1
CARGO_TARGET_DIR=runtime/research-target cargo +1.98.1 test --manifest-path vendor/lily-core/Cargo.toml --locked --test gdn_prework_research prepared_gdn_component_abba_after_exactness -- --ignored --nocapture --test-threads=1
```

GPU controls must first compare every history, prepared Q/K/V, gate, F32 state
and BF16 raw-output byte against the original source, over repeated synthetic
steps and non-dyadic/adversarial inputs. Only passing candidates advance to
rotating component timing. A component win then needs the same official
checkpoint, fixed precision, no speculation and paired complete-model controls
before production admission. Preserve failed results separately.

There is no speedup estimate from the twelvefold normalization reduction:
normalization is only part of GDN. Preparation reduces thread count and changes
memory access, occupancy and the overlap between state loads and gate math.
The convolution mapping changes from 10,240 channel threads to 2,560 threads
that each process four channels.
The direct variant can lose despite removing a barrier. Even a component win
can disappear behind Q8 projection bandwidth or whole-model scheduling.

If this candidate fails, the next aggressive experiment in the same piece is
one threadgroup per value head with recurrence plus output norm/gate fusion.
That could remove the raw-output round trip and 36 norm dispatches per token.
It also reduces available groups fourfold and multiplies the per-thread state
tile, so register pressure can dominate. It is a distinct experiment and is
not implemented or qualified here.
