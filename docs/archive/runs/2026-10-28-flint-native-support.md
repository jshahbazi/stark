# Recursive Flint native support validation

Date: 2026-10-28

Base commit: `018f89c5028d871c16914107e5b5578a42452380`

Branch: `codex/recursive-flint`

## Scope

This record covers native Rust/Metal execution of Recursive Flint and the
original-format model preparation and worker paths in this branch. It does not
qualify response quality, long-context behavior, production performance, or
all Flint checkpoints.

The pinned upstream reference is `rec/engine/src/rec_mlx/model.py` at
`0e3150ca4262f572dee25c231d9ac7148414d2af`, SHA-256
`56363f7a1caea2b9db317feb192f4bfea32eccb2b6b190dffc3fa93ee09695b1`.
The official MLX INT4 checkpoint is `RecursiveSI/flint-int4` at revision
`3b93f9053b530488405ae9fca51d4900d6ce21f0`; its five Safetensors shards total
22,568,912,600 bytes. The native forward is an independent implementation
based on the pinned reference. It does not run upstream Python or MLX.

## Engine validation

The native hybrid Metal suite passed 51 tests; six existing tests remained
ignored. The final filtered Flint run passed five tests, including the packed
weight regression. Independent F64-oracle Flint tests cover complete forwards
with both banked and separate expert tensors.

Four pinned MLX reference fixtures each evaluated 13 positions. Maximum
absolute logit errors and greedy agreement were:

| Fixture | Maximum absolute error | Greedy agreement |
| --- | ---: | ---: |
| F32 | 5.364418e-7 | 13/13 |
| BF16 | 6.258488e-7 | 13/13 |
| Q4 | 1.087785e-6 | 13/13 |
| Q4 BF16 | 8.344650e-7 | 13/13 |

All four fixtures passed exact reset and failure-reset checks. The captured
reference forwards explicitly use F32 KV caches to match native arithmetic.
Recursive's serving runtime can use quantized KV caches; these fixture results
do not establish equivalence for that cache policy.

## Worker and preparation validation

The generator and scorer were built separately with Rust 1.98.1. Two
source-bound fixtures, F32 and Q4 BF16, passed worker integration. Generator
greedy agreement was 4/4 for each fixture. Scorer maximum absolute errors were
4.470348358154297e-7 for F32 and 6.854534149169922e-7 for Q4 BF16.
Tokenization, EOS union, request isolation, consecutive-input cache behavior,
invalid-request recovery, and clean worker exit passed.

Release-mode worker unit checks passed: 43 generation tests, 19 optimized
scoring tests, the same 19 scoring checks in the digest probe, and the
experiment-protocol test.

Worker SHA-256 values:

| Worker | SHA-256 |
| --- | --- |
| `stark-generate` | `c700541714f348ff0f71b233b212313802d680d8af1a3b8bd6d2efd47acbda73` |
| `stark-score` | `75bc607e048b08d841e27ceb4ad3dd930993a5eda864d441f0a5c8278fb8007f` |

Preparation tests passed 8 tests. Distribution tests passed 19 tests. The
Python suite passed 4,962 tests with 88 skips; one unchanged failure was the
RSS sampler in `test_benchmark.py`. Rerunning all 61 benchmark tests in
isolation passed. Repository policy, Ruff, and formatting checks passed.
Clippy passed with warnings denied for both workers and all targets, including
the scorer's `optimized` features. The full official-checkpoint native smoke
test remains pending.
The official checkpoint download was still in progress when this record was
written; no full-checkpoint execution result is claimed.

## Fix during validation

The synthetic Q4 Safetensors fixture exposed an overly strict alignment check
in the new U8 admission path: MLX can leave its JSON header unpadded. The
existing kernels already address packed planes byte by byte. Removing the
absolute-offset alignment requirement admits this valid storage layout, and
a regression test covers an odd-length header. This fix is included below.

## Source hashes

SHA-256 hashes identify the source snapshot assessed in this record. Update
these values if subsequent fixes change any listed file.

| File | SHA-256 |
| --- | --- |
| `crates/stark-engine/src/engine/hybrid/flint.rs` | `d3df9b5845e05b7d4263cc2222309953f0c2dcff86011804fe80fd45eb25d1bb` |
| `crates/stark-engine/src/engine/hybrid/flint_tests.rs` | `6cb0e6b4fcd35169407b102c127c1f6f02c8aa912729950e717ae8e9f5d1aef0` |
| `crates/stark-engine/src/engine/hybrid/weights.rs` | `2d5d33dcd15365780f4eafa90ec9cf3ca23d9f01456f100245be9996e8191d4d` |
| `crates/stark-engine/src/engine/hybrid/mod.rs` | `4e8c65b35f1a9ac9f44a6e6abf613d295143d538a1ae2eed70e22ae70c1fc6e3` |
| `crates/stark-score/src/scoring.rs` | `14cf3b05cc99341f946ce6ac80be25609aa27c09485823a34f69d88990bdac74` |
| `tools/converter/stark_convert/hybrid_prepare.py` | `0ca85d30d39cf29ffb245ca4c1553a7fc1684729281c897a887c2c25c7209ef1` |
| `scripts/flint_native_fixture.py` | `6a4ef6055204e18ef671af627bf02977eed863c7bac29e57b17be25fdc1d3950` |
