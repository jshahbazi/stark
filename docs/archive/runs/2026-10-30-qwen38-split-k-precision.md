# Qwen3.8 split-K precision investigation, 2026-10-30

This record preserves failed controls from the full Uzu execution integration.
It does not qualify model accuracy or generation throughput. All observations
below precede the F32 split-partial correction. Weights, quantization and
speculation settings were unchanged.

## Preserved failures

The realistic synthetic two-layer graph produced a relative L2 difference of
`0.01670465350` between one-token and 32-token Metal projection geometry over
4,096 readout values. The 16-token and 32-token results matched in that run.
CPU results matched across those geometries. These are observations, with no
quality acceptance threshold.

Replacing the affine coefficients with exactly BF16-representable weights did
not remove the discrepancy: the strict Metal graph control still failed with
relative L2 `0.01841477978`, RMS error `0.006579488502`, and maximum absolute
error `0.08641862869`. No tolerance was relaxed.

An isolated standard projection control then failed for Q4/group64 weights,
M=16, N=68, K=256, BF16 output. Of 1,088 outputs, 136 exceeded the derived
branch arithmetic/output-rounding bounds. The same F32-output control had
zero error. Dense BF16 weights and forced unsplit quantized GEMM passed their
unchanged bounds.

## Attribution

Every one of the 1,088 default BF16 projection outputs exactly matched an
independent host emulation that rounded each of two split-K partial sums to
BF16 before their final reduction. The forced unsplit quantized control had
zero bound failures; its maximum budget ratio was `0.9300539663`.

The implementation stored each partition in the requested final output dtype.
Although the reducer accumulated in F32, it received already-rounded BF16
partials. Cancellation between partitions could therefore lose information
before the final BF16 conversion. The issue was independent of the canonical
GroupOutput tensor-plane ordering.

The correction keeps split-K partition outputs in F32, includes their output
dtype in the kernel cache key, and makes the reducer read F32 partials before
casting once to the requested output dtype. Unsplit execution, projection
geometry, weights and activation precision remain unchanged. Verification of
the correction must be recorded as a separate run.

## Evidence identity

Private aggregate logs are retained under the worktree's ignored `runtime/`:

| Log | SHA-256 |
| --- | --- |
| `uzu-qwen-precision-contracts-v1.log` | `ec0194b1b905db1c2d6c5d81e0d0b06c94ff4ef95522d4052d9aafced6d8b233` |
| `uzu-split-k-attribution-v2.log` | `356b3f600bdf29d66da0fdb41f52e75f9a31ed16e98aa50ab5894e457321623f` |

The isolated controls live in
`crates/stark-engine/src/encodable_block/qwen38/numerical_tests.rs`. Their
independent F64 oracles distinguish exact affine weights, BF16-staged weights,
F32 reduction error and final output rounding. Cancellation controls cover
Q4/Q8, M=16/32, F32/BF16 outputs and forced split counts two and four. A
negative control retains the rejected BF16-partial arithmetic.
