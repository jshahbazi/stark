# Split-K F32 partial correction validation, 2026-10-30

This record is separate from the preserved
[failed precision investigation](2026-10-30-qwen38-split-k-precision.md).
It summarizes existing private aggregate logs from working-tree checks.
It does not establish final source/binary qualification, complete-model
accuracy, sampled-generation quality or throughput. No model quantization or
speculation change is part of the correction.

## Correction and numerical controls

Standard Metal GEMM now stores split-K partial outputs in F32, includes the
actual output dtype in its pipeline cache key, and reduces F32 partials before
the final requested output cast. Unsplit arithmetic is unchanged. The change
also affects ordinary models using standard split-K matrix dispatch.

The independent controls in
`crates/stark-engine/src/encodable_block/qwen38/numerical_tests.rs` distinguish
the F32 affine GEMV policy from GEMM's BF16-staged dequantized weights. Each
uses its matching host F64 dot-product reference. Bounds account for operand
staging, F32 accumulation and final output rounding; no model-quality
tolerance is derived from the observed errors.

The historical native suite contains the following passing correction checks:

- Branch-matched projections cover Q4/Q8, group64, N=68, K=256, M=16/32,
  BF16/F32 outputs, exactly BF16-representable and nonrepresentable affine
  weights, default/forced-unsplit/forced-two-/four-partition GEMM, GEMV and
  explicitly staged dense controls. All branch bounds pass. The largest
  reported default GEMM budget ratio is `0.9593899677`; the largest reported
  GEMM-versus-GEMV combined-budget ratio is `0.5566504336`.
- Cancellation controls cover Q4/Q8, M=16/32, BF16/F32 outputs and two/four
  partitions. Every output retains the exact expected residual, with zero
  bound failures and zero maximum absolute error. The negative control using
  BF16 partials is rejected.
- The native suite also passes existing ordinary matrix checks for affine
  layout, 8-bit split-K, A8 weight/output-bias/RHT routes and independent
  activation groups. These checks cover their existing fixtures and bounds.

The current numerical-test source additionally exercises one kernel instance
through split2, unsplit, split4, unsplit and split2 to check dtype-cache reuse.
That control is absent from the historical native log below; a fresh execution
receipt is still required.

## Broader working-tree checks and observations

| Existing log | Recorded result |
| --- | --- |
| `uzu-full-native-tests-v1.log` | Engine unit suite: 711 passed, 0 failed, 42 ignored; separate C++ string tests 7 passed and documentation groups 1 and 2 passed |
| `uzu-full-cpu-tests-v2.log` | Engine unit suite: 462 passed, 0 failed, 20 ignored; separate C++ string tests 7 passed and documentation groups 1 and 2 passed |
| `uzu-ordinary-split-k-tests-v3.log` | Two ordinary-model controls completed; their distinct meanings are recorded below |

The ordinary dense split-K fixture uses model width 512, 33 prompt rows,
vocabulary 16 and two QKV partitions. CPU/Metal readouts match exactly and
repeated Metal readouts are identical.

The separate realistic ordinary fixture reports maximum absolute CPU/Metal
difference `0.25`, RMS `0.06454183906`, and ratio `2.015748024` against its
original tolerance. It records `quality_acceptance_threshold=none` and exact
same-route repeatability. Completion of this observation does not make the
original tolerance pass or qualify model quality.

The realistic two-layer Qwen precision observation also remains separate from
acceptance: Metal one-token versus 32-token geometry reports maximum absolute
difference `0.01455931365`, RMS `0.003563819018`, and relative L2
`0.01683980534` over 4,096 values. Metal 16-versus-32 and CPU 1-/16-versus-32 are
identical in these logs. No graph tolerance was set from these measurements.
The stricter exact-affine state/order fixtures pass in the native and CPU
suites, within their existing fixture scope.

## Evidence identity and limits

The private logs remain under the worktree's ignored `runtime/` directory.

| Log | SHA-256 |
| --- | --- |
| `uzu-full-native-tests-v1.log` | `573cd87267b59e14a213734eef4611526f8dd9358adad2fe6bc8b0d42c9d81db` |
| `uzu-full-cpu-tests-v2.log` | `08bb67d42b9a686fcc03b9311cc43a01bdea46d4a14841cde13267fb6aac37ea` |
| `uzu-ordinary-split-k-tests-v3.log` | `600bb0192639beece9b9ac5588d0679a4093903cbeacfcdacba6fe54b216b0db` |

The numerical-test source inspected for this record has SHA-256
`5cebced0eda546119ed071ffd7213314883c6a31cd32140873d202db57ef51fe`.
That identifies the inspected source only. These historical logs have no
immutable source manifest or test-binary SHA-256 receipt; their executable
paths do not bind them to the current working tree. Later retained changes
require fresh sealed suites and binary/source identities.

The planned fixed-prefix full-vocabulary comparison at 35 Qwen checkpoints and
three ordinary-model checkpoints has not run and remains pending. Primitive
bounds and synthetic fixtures cannot qualify the complete 48-layer model or
establish a general accuracy threshold. Final numerical and architectural
acceptance remains governed by the
[Uzu execution plan](../../QWEN38_UZU_EXECUTION_PLAN.md).
