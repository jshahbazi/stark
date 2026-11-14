# Gemma 4 source residual fusions: validation draft, 2026-10-29

This record covers the source PLE residual fusions on `codex/native-olmo3-gemma4`, based on `3394cbe72fed7450c531d33cc333771ad1104cba`. The completed checks below bind the frozen binaries and unchanged model artifacts. The CPU/grammar suite also passes. Final matched timing and push remain pending the quiet GPU window requested by the user.

Follow-up: the [2026-11-02 validation](2026-11-02-olmo3-gemma4-source-fusions.md) records fresh checks and timing qualification separately. The pending status below describes this October checkpoint.

The earlier [repaired Q4 benchmark](../../benchmarks/OLMO3_GEMMA4_Q4_REPAIRED_2026_10_29.md) and [precision investigation](2026-10-28-olmo3-gemma4-q4-precision.md) retain their original results and identities. This draft makes no new speed claim.

## Implementation

Gemma's explicit BF16 source PLE path folds its post-attention residual addition into pre-FFN RMS normalization. The sum rounds to BF16 before statistics. For validated source rows of at most 4,096 elements, each Metal thread retains its four rounded values in registers through the unchanged source reduction and normalization. This avoids reading residual values written by other threads.

The final PLE residual addition and layer scaling use `MaterializedBF16AddScale`, which preserves the BF16 residual sum and final scaled BF16 store in one dispatch. Both host selections require BF16, reference PLE and the explicit source RMS policy. Canonical BF16 and legacy PLE profiles retain their separate operations.

The scheduling investigation inspected local Qwen revision `908ae21f4894d841cf2e1ad44451643439543443`, including its thread-local rounded GDN value. No Lily kernel arithmetic was copied. Existing MLX source pins, attribution and MIT notice remain recorded in [upstream provenance](../../../provenance/UPSTREAM.md).

## Build identity

The private build receipt is `q4-repair/gemma4-gap-v2/build-receipt.json` under the ignored `runtime/olmo3-gemma4/` directory. It records successful locked builds with Rust 1.98.1 for the examples and workers, and `nightly-2026-10-01` for the CPU/Metal unit binary. The scoring worker uses its `optimized` feature.

| Identity | SHA-256 |
| --- | --- |
| Build receipt | `66c5ec93610f15fda818809070bb056a968b745a9528d5920522b0d24b2dcd4b` |
| Recorded source patch | `b77420a37a90802337e834cead5eea14be4fd3c7cba498e38a0f3db81fc9c981` |
| `dense-profile` | `48a4e660035152657e7a19a71696486c68da7f2852d917aa072bdc729079a7c7` |
| `dense-parity` | `bd2f2890c10589f3a8c100b9a3ce8e287ca8e3c8942ff39b30e6fed696c0e34c` |
| CPU/Metal unit binary | `95f0c2d747733712cea67afb19413bd6d9ee1b13ba0ac236ca93d0d66ac648a6` |
| `stark-generate` | `36d477f7670c392da2c2e2bb55516b5d5f07df8d504df36423ce54467dcf421d` |
| Optimized `stark-score` | `d2dcd1ec03f1247b058bc145334f9d073663f370d6b78f1ca50374ff2ce00e25` |

The unchanged native artifacts are OLMo Q4 source BF16 v5, Gemma E2B Q4 source BF16 v5, the two canonical BF16 text artifacts and Gemma E2B BF16 vision v2. Their checkpoint, conversion, weight and configuration identities remain those in the earlier precision record.

## Completed numerical and worker checks

The four strict suites retain the existing source references, fixtures and gates. All 28 cases pass and all 28 greedy choices agree. Every native full-vocabulary output hash equals its corresponding result from the preceding normalization qualification. Gemma remains exact against all 14 source references; OLMo remains exact in 13 of 14 cases, with the unchanged accepted 4,101-token discrepancy documented in the earlier precision record.

| Check | Completed result |
| --- | --- |
| OLMo Q4 original suite | 5/5; all native output hashes unchanged |
| OLMo Q4 sequence suite | 9/9; all native output hashes unchanged |
| Gemma Q4 original suite | 5/5; all native output hashes unchanged |
| Gemma Q4 sequence suite | 9/9; all native output hashes unchanged |
| Canonical BF16 OLMo text | 5/5; all native output hashes unchanged |
| Canonical BF16 Gemma text | 5/5; all native output hashes unchanged |
| Canonical HF Gemma images | 3/3; complete aggregate metrics bytes unchanged |
| OLMo native generation worker | 2/2; restart and replay checks pass |
| OLMo native scoring worker | 2/2; replay and recovery checks pass |
| Gemma native generation worker | 2/2; restart and replay checks pass |
| Gemma native scoring worker | 2/2; replay and recovery checks pass |
| Gemma BF16 image worker | 3/3; fresh/restart replay, malformed-image recovery, image counterfactual and staging cleanup pass |
| Gemma image client | 2/2; cold/cached checks and staging cleanup pass |
| Source RMS residual kernel control | Pass; exact separate/fused comparisons |
| Materialized BF16 add/scale kernel control | Pass; exact separate/fused comparisons |
| Complete native CPU/Metal unit suite | 717 passed, 0 failed, 54 ignored |
| CPU/grammar suite | 476 unit tests, 7 integration tests and 3 doctests passed; 21 ignored, zero failures |
| Repository policy, pinned formatting and diff checks | Pass |

The canonical image metrics SHA-256 remains `64cc154be30bb3984f41f26b932c421d237f996de67645596c109840093e00a1`. This image evidence uses the separate BF16 vision artifact.

## Private evidence identities

Only aggregate results and hashes are published. Raw captures, fixtures, worker artifacts and executables remain ignored and private. Paths below are relative to `runtime/olmo3-gemma4/q4-repair/gemma4-gap-v2/`.

| Evidence | SHA-256 |
| --- | --- |
| `qualification/strict-olmo3-5.log` | `0a8050f75ff89cdf669f2113b4f259b7b57146d7325878426fba60fb3e3fa18c` |
| `qualification/strict-olmo3-9.log` | `af8437b3a2e1fa520d3d76228f013384becc778a45c46b8b2ca88d9dafae47f6` |
| `qualification/strict-gemma4-5.log` | `a37fdc42e3000af76e0179ca3d52925cd9d9cfdaa8fb8426c8a456c59d78563e` |
| `qualification/strict-gemma4-9.log` | `b12c483af51d3cae1538c6e4d0b9bd8d44cbdd805027fe4c317c3e5e7be6b888` |
| `qualification/canonical-text-olmo3.log` | `815b5bd6f619ed0a27d1efe6dccaeb4e18560bc19355273672c4e4c7f69bec8f` |
| `qualification/canonical-text-gemma4.log` | `c956970b74827420cba92fa57d752fbf068255c07b11c93d0a4207805131482d` |
| `qualification-resume/canonical-image3.log` | `f58863cac74474a82212cf3a9c2c0bede8df092b9c13435f9b858b8c7634616b` |
| `qualification-resume/olmo3-workers.log` | `f3fa0683d13a3661b07957974ea95581ce80afd7367fa6bcff8663fa9de9164b` |
| `qualification-resume/gemma4-workers.log` | `7b66edb3cb9770208fa775f9f1c3d548c922b475b9d55f768f810269a904a8ab` |
| `qualification-resume/image-workers.log` | `54ab87f9f233c44ae7f2015ac336d8a9237f70f6de208848aa6f7bb7b1083dab` |
| `qualification-resume/image-client.log` | `55f2fe0a0a6e7411600f6bcc32f4807276736792225903ac1676fe590205a8e6` |
| `targeted-rms.log` | `000c4fb8ccb19cc66f7066941168ed85e2ee8a2fcda696f3b54e2a4b16733597` |
| `targeted-ple.log` | `f7358f07b07fe123f69574d0fdad558b6070aee86e2721b5ae8d899f4cf256c3` |
| `qualification-resume/native-full.log` | `f7f7ae26bd08b27f6e2660a091f5910e85a7be2da4fcc50f455ce119dbab2e73` |
| `qualification-resume/cpu-grammar.log` | `db514cde892c7a9a1557457ceb15ce1c322e4105b7dea545cfcf08b3f96794bc` |
| `qualification-resume/qualification-receipt.json` | `3dbed1d470c865f8a3db930f9ed3eecbf85030693d4d74620c1ed7f0b1046eab` |

## Pending qualification

Preliminary timing screens had variable latency and competing host activity. Their private receipts remain preserved; they provide no accepted performance result.

The user deferred performance timing because another process uses the GPU intermittently. A final guarded comparison with the frozen preceding implementation and pinned MLX remains pending a quiet window. Closing the remaining Gemma performance gap has not been established for this change. Push is pending the remaining qualification and timing decision.

The first canonical image harness invocation selected the text-only artifact and failed during model loading. The corrected invocation used the BF16 vision artifact on a private copy of the canonical fixture; its complete metrics bytes match the preserved prior result. Both captures remain private, with no changes to engine source or gates between invocations.
