# Official Qwen3.5 vision parity

Date: 2026-10-26. Baseline: `9eadf771f45458027e0d8cb22b9555a1d5cd78f2`.
This record covers the native engine's private official-weight tower replay.
It supersedes the numeric failure in the [earlier vision run](2026-10-26-qwen35-native-vision-preprocessing.md)
for the source fingerprint below; that record remains unchanged.

## Result and fixed gate

CPU and Metal each pass **60/60 checkpoints, bit-exact**: patch projection,
position addition, twelve transformer blocks, and the final merger, for each
of four fixtures. Every checkpoint has relative RMS error `0`, maximum absolute
error `0`, and cosine similarity `1`.

The original gate is unchanged: patch projection relative RMS at most `0.005`;
subsequent checkpoints at most `0.02`; cosine similarity at least `0.999`;
matching shapes and finite values throughout. Neither reference vectors nor
compute dtype were substituted to obtain this result.

| Fixture | Grid | Baseline CPU merger RMS | Baseline Metal merger RMS | Final CPU / Metal merger RMS |
| --- | --- | ---: | ---: | ---: |
| Normal 256 × 256 RGB | 1 × 16 × 16 | 0.0487236151 | 0.0506087117 | 0 / 0 |
| Normal processed rectangular RGB | 1 × 22 × 14 | 0.0405518044 | 0.0402893172 | 0 / 0 |
| Small stress image | 1 × 2 × 2 | 0.2645391190 | 0.0356331085 | 0 / 0 |
| Rectangular stress image | 1 × 2 × 4 | 0.0198568068 | 0.0184553107 | 0 / 0 |

The two small fixtures are below the official processor's minimum image area;
they remain numerical stress cases. The two normal cases use RGB pixels and
production preprocessing, with complete BF16 patch equality checked first.

## Reference identity and independent verification

- Official BF16 checkpoint: `Qwen/Qwen3.5-0.8B` at
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Converted full-weight SHA-256:
  `7c5960e7225b3b36844fa397a85c5e20d281d732e382475062f552c0737c79d4`.
- Transformers source: `2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397`;
  PyTorch `2.14.1`, eager attention, BF16 parameters and FP32 rotary buffers,
  eight reference CPU threads.
- Immutable canonical reference manifest SHA-256:
  `e5dceac11aa5fd42bea86ca50b4aa1a9d48d8bfd99f08cd43c5fe950225330ae`.

An independent actual Hugging Face `from_pretrained` load, assigning the official
BF16 vision tensors and preserving FP32 rotary buffers, matched the pinned
reference's complete projector bit-for-bit on both normal cases. This separately
checks the extracted reference's loading and dtype semantics.

Same-input diagnostics isolated all twelve blocks and the merger, then replayed
individual primitives from captured native inputs. Separate reviewers checked
the reduction schedules, strides, scalar tails, compiler contraction, saturation,
and text-kernel isolation. Final verification additionally includes:

- All 65,280 finite BF16 inputs through both GELUs: zero mismatches on CPU and
  Metal. A shuffled reference audit also found zero BF16 context differences.
- Native host rotary tables: 65,536 FP32 sine values and 65,536 cosine values,
  coordinates 0–4095 across sixteen frequencies, all bit-exact. Manifest SHA-256:
  `428365de73521a66c95c174a2d7b5430c0cb18009e5bc433cef80b7aa8b987e5`.
- Two independent synthetic twelve-layer tower fixtures pass on both backends.
- A real official-artifact image request completes native image encoding,
  M-RoPE fusion, and language generation, including fresh-state repeatability.
- Two greedy text generations and four full-vocabulary scoring digests match
  the baseline workers. Cache hits remain false, true, false, false, with three
  native score calls.

## Changes that closed the gap

The model amplifies small BF16 rounding differences over twelve blocks. Local
relative errors alone did not predict the final projector error. The vision
path now preserves the pinned reference's numerical operations:

1. FP32 Welford moments follow PyTorch's BF16 lane/chunk cascade, including the
   scalar final contraction. Affine operations retain their separate rounding.
2. BF16 dense/QK dot products use the reference's thirty-two FP32 accumulator
   streams and reduction tree. Patch projection and probability/value products
   use its four-stream reduction. Bias precedes the final BF16 conversion.
3. Rotary products use separate FP32 multiply/add operations and pinned SLEEF
   sine/cosine behavior. Softmax uses pinned exponential arithmetic, four-lane
   summation, and reciprocal multiplication before BF16 conversion.
4. Vision GELU follows pinned SLEEF tanh and PyTorch NEON erf arithmetic. An
   exhaustive native check detected Metal FP32 subnormal flushing; a bounded
   integer BF16 half-value operation now preserves the required tiny results.

Metal safe-math and contraction settings are local to these vision kernels.
Text normalization, attention, rotary preparation, and activation kernels are
unchanged. No PyTorch, SLEEF, or MLX runtime dependency was introduced.
Adapted source revisions, hashes, and copyright/license notices are retained in
[upstream provenance](../../../provenance/UPSTREAM.md) and the supplemental
distribution notices, which are hash-checked before packaging.

## Build and regression evidence

Native source manifest SHA-256:
`89c307d9a992fef9c7f452353d706ddf7a5615d7923ac453e01303b7fe288139`.
The source fingerprint includes untracked native source before commit; the
final commit contains that source. Cargo dependencies remain locked:
`2254da0dd179db6b926492f2bec5f63b756f6508132b34f5ac772ec9d877aaa2`.
Rust toolchain file SHA-256:
`72da9f9e503464ba7393560f6f51a69fb06be72633d52ae7725fa0ba59dfda22`.

Engine suites and private replays use `nightly-2026-10-01`. Release workers use
Rust `1.98.1`, with generator and optimized scorer built in separate commands.

| Release executable | SHA-256 |
| --- | --- |
| Generator | `723fc044c965f8b8317a03276c8615616c67eb2de320432b06b720b95305ed48` |
| Optimized scorer | `df55a3fbbe69332877616cf66b5db226d46d504055e5f626a76a5a9d0fdf32ed` |
| Full-vocabulary digest verifier | `e36d4ac38860722fd197c4748b9a226796f8dcf8025db0e8b766646fde078a2f` |

Completed local regression checks:

| Check | Result |
| --- | --- |
| CPU/grammar engine unit suite | 368 passed, 10 ignored |
| CPU/grammar with RoPE cache | 370 passed, 10 ignored |
| Native CPU/Metal engine unit suite after the tiny-value fix | 547 passed, 19 ignored |
| Engine integration tests and doctests, each configuration | 7 integration tests and 3 doctests passed |
| Release generator / optimized scorer tests | 26 / 39 passed |
| Generator / scorer process checks | 6 / 8 passed |
| Distribution license/packaging tests | 19 passed |
| Repository policy, pinned nightly formatting, whitespace check | Passed |

Model-bound and exhaustive private replays listed above were explicitly run
despite their normal ignored status. Private raw references, native vectors,
models, binaries, and logs remain in ignored, owner-only runtime storage.
This document publishes aggregates and hashes only. These are local results;
no hosted CI completion is asserted.

## Other engine inspection and limits

[mlx-serve](https://github.com/ddalcu/mlx-serve/blob/191da71de32a65f69ebceeb6cb2e91036ab50b30/src/qwen_vision.zig)
has its own Zig vision tower and preprocessing. Its attention explicitly uses
FP32 scores/softmax; source inspection does not establish the same BF16 eager
parity gate. [MLX-VLM](https://github.com/Blaizzy/mlx-vlm/blob/6d9ba328abbd6129c7461eda86280aa9fbfd9986/mlx_vlm/models/qwen3_vl/vision.py)
provides the shared Qwen3-VL vision implementation used by its Qwen3.5 model.
No ready native vision tower/preprocessor was found in the inspected
[Uzu engine tree](https://github.com/trymirai/uzu/tree/c541d8e6e7018a3a9a4e1b6d81548131511b0a6a/crates/uzu-engine).
These were read-only comparisons; the numerical adaptations come from the
pinned PyTorch and SLEEF sources.

This is native tower fixture qualification. End-to-end decoder-logit parity,
visual task quality, supported memory/latency budgets, transport, cancellation,
and packaged image serving still require qualification. The Local API continues
to report `images:false`.
