# Rejected GLM Q5 integer-fragment MMA experiment

Disposition: **REJECTED as a runtime candidate**, 2026-11-02. The main projection's 0.90% median reduction does not establish a useful improvement over the existing grouped Q5 path. The output projection regressed 7.24%. No production route or model admission was added.

The prototype passed its corrected component-level numerical gates. These results use synthetic payloads and do not qualify the actual checkpoint or establish a model-quality improvement.

## Contents and provenance

- `results.json` contains public aggregate measurements, numerical policies, source/executable hashes, provenance and separately classified v1/v2 validation outcomes.
- `rejected.patch` is the exact test-only prototype, including all four previously untracked test/shader files and nine lines of `cfg(test)` wiring. It is for reproduction in an isolated checkout. It is not applied to the active runtime.

The patch applies to Stark base `13de5d4ee095a77029d38863e58ffca0339051a9`. It adapts the Q5 specialization from [llama.cpp `dde879bc4f954b809892fd0d660515f42536394a`](https://github.com/ggml-org/llama.cpp/blob/dde879bc4f954b809892fd0d660515f42536394a/ggml/src/ggml-metal/kernels/mul_mv_mma.metal), preserving its pin, copyright and MIT notices. Existing Stark code retains its license notices. Upstream terms remain in `LICENSE.ggml-hybrid`. The archived generic v1 shader is preserved modulo a separate namespace; its source archive hash is recorded in the JSON.

Only source and public aggregates are included. Private raw logs, generated payloads, activation arrays, probabilities, model bytes and runtime artifacts were not copied into this archive.

## Arithmetic and numerical admission

The specialization holds exact integers `q` or `16*q` in F16 A fragments, with F32 activations, matrix accumulation and affine reconstruction. Stored scale/min metadata uses the existing byte-safe decoder. It changes multiplication, affine and reduction association and makes no bit-equivalence claim.

The experimental policy limits both affine coefficient terms to magnitude 1 and admits zero or nonzero input magnitudes in `[1e-30,1e20]`. IEEE integer magnitude keys classify tiny/subnormal and nonfinite inputs. Scale and cancellation rejection uniformly replay generic F32 MMA; tiny/extreme/nonfinite input uses the existing Q5 source chronology. Near-zero candidates are rejected against an affine L1 magnitude estimate. This policy is numerically tested and is not presented as a universal error theorem.

The unchanged gates are relative RMS `<=2e-5` and F64 error divided by absolute product sum `<=2e-6`, with both source-decoded F32 and independent mathematical GGUF coefficients. The cancellation suite additionally requires exact zero and uniform fallback.

## Separate validation outcomes

**V1 failed.** Its first bounded nonzero-affine gate passed, then the cancellation test failed because it asserted that the existing grouped GPU baseline returned exact zero. The third gate and timings were not run. An independent CPU inspection verified the synthetic encoding: 13,312 coefficient checks under each F32/F64 policy found zero nonzero coefficients. The grouped baseline already factors affine terms, so its output was an incorrect mathematical zero oracle. The original failed source, receipts and logs remain preserved privately; this archive does not rewrite that run as a pass.

**V2 passed component qualification.** Only the test reference and diagnostics changed. All 32 scale/magnitude/sign cases independently asserted zero CPU F32 and F64 coefficients, comprising 106,496 checks per policy. Generic per-weight F32 MMA also had to return zero. The guarded candidate retained exact-zero, bit-comparison, RRMS/F64 and uniform-fallback requirements. The raw unguarded specialization failed the numerical policy in 18 cases where specialized work actually executed.

The legacy grouped path returned nonzero results in four synthetic cases. The first was case 2, stored scale 65504 and activation magnitude 1e10, with 52 nonzero outputs and maximum absolute error 424,673,280. This matches the magnitude predicted by CPU emulation of one-sided fused affine reconstruction. No device instruction disassembly was performed. These deliberately pathological synthetic cases do not establish an actual-checkpoint quality defect or improvement.

V2 also passed nonzero-affine basis/tail and F32-detail tests, subnormal/extreme/nonfinite tests, metadata poisoning, zero-times-MAX and recovery. Metal pipeline creation and execution succeeded. No production or full-model qualification followed.

## Paired T4 results

Each comparison used eight rotated banks totaling 184,549,376 bytes, three ABBA warmup rounds and eight measured ABBA rounds. The two comparisons retained separate guarded measurements, with 16 measured commands per schedule. GPU timestamps exclude upload. All numerical gates passed before timing aggregates were accepted.

| Projection | Baseline | Baseline GPU us | Guarded GPU us | Guarded time change |
| --- | --- | ---: | ---: | ---: |
| Main, R8192 K4096 | Legacy projection-major | 157.945 | 156.529 | -0.90% |
| Main, R8192 K4096 | Generic F32 MMA | 219.604 | 156.151 | -28.89% |
| Output, R4096 K8192 | Legacy projection-major | 160.417 | 172.036 | +7.24% |
| Output, R4096 K8192 | Generic F32 MMA | 266.477 | 171.943 | -35.48% |

The guarded main/output cancellation fallback fractions were 0.6836% and 0.8789%. Ordinary component runs had zero scale or source-input rejections. Candidate RRMS/F64 normalized errors were approximately `1.97e-7 / 5.09e-8` for main and `2.26e-7 / 8.14e-8` for output. No repeated-run uncertainty interval was qualified.

The component log's generic `precision` field described the candidate on baseline rows as well. The JSON here records the schedule-specific policies correctly: legacy is grouped F32, generic MMA has F32 A/B/accumulators, and the candidate has exact integer F16 A with F32 B/accumulation/reconstruction. Private source-bound raw logs remain unchanged.

## Reproduction only

Apply `rejected.patch` only to an isolated checkout of the pinned base. Use a distinct target directory, the locked dependencies and `nightly-2026-10-01` for engine tests:

```sh
git apply --unidiff-zero /absolute/path/to/rejected.patch
CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR=runtime/target-glm53-q5-mma-repro cargo +nightly-2026-10-01 test --locked -p stark-engine --features cpu,metal --no-run
```

Use the printed engine test executable on a qualified 32-lane Apple Metal device. Serialize GPU work and run the three correctness filters before the ignored timing component:

```sh
TEST_EXECUTABLE q5_special_bounded_nonzero_affine --nocapture --test-threads=1
TEST_EXECUTABLE q5_special_cancellation_raw_rejected --nocapture --test-threads=1
TEST_EXECUTABLE q5_special_extremes_subnormals --nocapture --test-threads=1
TEST_EXECUTABLE q5_special_t4_main_output_rotated_abba_components --ignored --nocapture --test-threads=1
```

Stop on any failed gate. Reproduction on another device/compiler is a new qualification run. The patch has no production selector and must not be treated as runtime admission.
