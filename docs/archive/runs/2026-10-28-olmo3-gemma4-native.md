# OLMo 3 and Gemma 4 Native Validation

Date: 2026-10-28  
Implementation revision: `0f3e855f69b130a3de922fc15fdea760b92d9c92`

This record summarizes validation for native OLMo 3 7B Instruct and Gemma 4 E2B text and image support. Runtime receipts and detailed fixtures live under ignored `runtime/olmo3-gemma4/`; they are not part of the repository. No prompts, generated response text, token vectors, or tensor dumps are published here.

## Pinned sources and converted artifacts

| Artifact | Source | Converted artifact | Size / tensors | SHA-256 |
| --- | --- | --- | ---: | --- |
| OLMo 3 BF16 | `allenai/Olmo-3-7B-Instruct` at `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` | BF16, schema 1 | 14,597,132,448 bytes / 259 | `e2e546baea40e9fc6e0511b3de78c373e7e47b4cdb7c9ed118c328168572f0ec` |
| Gemma 4 E2B BF16 text | `google/gemma-4-E2B-it` at `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | BF16, schema 1 | 9,257,802,526 bytes / 475 | `951ba1dc7075efa1fea77eca113d8925f101735e7deb7bbeecb7dac9abc903ee` |
| Gemma 4 E2B BF16 image-capable | same pinned Google source | BF16, schema 4 profile `gemma4-e2b-bf16-v1` | 9,594,977,710 bytes / 1,134 | `5c26289c1536f6afda47e62616c02ae63df489dc5c37d6034624413b2fd75993` |

The image-capable profile binds the official processor and chat template. The listed source component and tokenizer hashes are recorded in the ignored conversion receipts. These converted outputs are BF16 weights; this describes activation/weight storage for these artifacts and is separate from any affine quantized matrix format.

## Execution environment and reference profiles

The Metal receipts were produced on the Apple Silicon host using the repository's pinned Rust toolchain and locked dependencies. Python tooling ran in the locked converter environment. Image CPU reference generation used Transformers 5.15.1 and the canonical Hugging Face processor/model path. A separate MLX-VLM comparison used MLX 0.32.4 at source revision `53ebd1647cb83549b482a031ea70ba9d9ea60a04`; it is a comparison profile, not the canonical image oracle. Text MLX reference receipts use their separately locked environment. Results from different reference profiles are not combined into one acceptance claim.

## BF16 native parity

Final Metal parity receipts use five cases per model, with context size 1,024 except the long OLMo case (4,133). All cases passed and greedy-token agreement held for all ten cases:

| Model / receipt | Cases | Maximum centered RMSE | Maximum reference-to-native KL | Result |
| --- | ---: | ---: | ---: | --- |
| OLMo 3 (`olmo3-parity-final.json`) | 5 | 0.040281 | 0.002087 | Pass |
| Gemma 4 E2B text (`gemma4-parity-canonical-final.json`) | 5 | 0.106322 | 0.007834 | Pass |

The aggregate receipts report acceptance profiles `bf16-centered-logits-v3` and `gemma4-canonical-bf16-v4` respectively. The OLMo gate requires centered maximum error at most 0.75, absolute mean error at most 0.75, KL at most 0.01, RMSE at most 0.1, and a combined centered RMSE bound of `1/128 + 0.02 × reference centered RMS`. Gemma uses the same centered maximum, mean, and KL caps, with centered RMSE at most 0.15 and `1/128 + 0.08 × reference centered RMS`. Raw relative error and greedy agreement are reported as diagnostics, not acceptance criteria. Centering removes a common additive logit offset, to which the softmax distribution is invariant; the mean-error cap bounds that offset. The Gemma norm budget was calibrated against independent CPU Hugging Face BF16/F32 controls (up to 5.405% centered relative L2) and Hugging Face BF16 versus the separate MLX comparison profile (up to 6.068%). The calibration report covers five prompt lengths under both CPU dtypes and includes a context-reset negative control that fails strongly.

Earlier parity runs failed: OLMo's first case missed the then-active centered-error gate, and Gemma's earlier canonical profile failed one case. Those outcomes remain in the private run history; the final receipts use their recorded acceptance profiles and pass all cases. A Gemma negative control with a deliberately mismatched reference context failed as expected, confirming that the comparison detects a material context mismatch.

## Gemma 4 image path

The canonical Hugging Face CPU reference profile compared native Metal image features and final logits on three bounded image fixtures. The native feature gate passed all three against its 2% relative RMS bound; errors were 1.17%, 1.33%, and 1.50%, with cosine similarity above 0.99988. Patch tensors matched exactly for all fixtures. The HF CPU BF16/F32 control had relative feature differences of 1.70–1.78%; this supports the feature tolerance calibration. Image logits use the same centered-logit criterion as text. No separate image-specific BF16/F32 logit control was run. All three next-token greedy comparisons passed; maximum KL was 0.000028174 and maximum raw absolute logit error was 0.375. The separate MLX-VLM comparison profile passed its three cases, but is not the HF canonical acceptance result.

The final image worker receipt passed fresh replay, restart replay, malformed-image recovery, staging cleanup, and the red/blue image counterfactual check. The local real chat client was also checked with both a cold request and a cached request for the Gemma image profile; both passed and staging cleanup succeeded. This exercises worker lifecycle and image-dependent output, not every desktop interaction or a throughput qualification.

## Generation and scoring workers

Final OLMo and Gemma text worker receipts passed, including exact restart replay. OLMo generated two EOS-terminated cases with 13 and 11 output tokens, exactly matching the pinned MLX-LM greedy sequences. Its score-worker cases used 46 and 4,101 input tokens; maximum absolute score error was 0.125, with cache replay, invalidation, and failure-clearing checks passing. Gemma's combined fixture pairs canonical HF scoring references with a separate MLX generation profile; its exact generation matches contain 9 and 32 output tokens. Aggregate evidence records worker restart/replay, cache behavior, and recovery checks. These bounded checks do not establish response quality beyond their fixtures.

## Quantized conversion artifacts

Affine 4-bit, group-size 64 exports were converted successfully for both model families:

| Family | Source export and immutable revision | Converted size / tensors | SHA-256 |
| --- | --- | ---: | --- |
| OLMo 3 | `mlx-community/Olmo-3-7B-Instruct-4bit` at `d732c91ae02e90cd5d86e810fbeb9741794b4dd9` | 4,107,042,600 bytes / 519 | `80741d14a2d64a740c65c285e2dfca7783f59ce63fa1a2465c12ea31306c3db8` |
| Gemma 4 E2B | `mlx-community/gemma-4-e2b-it-4bit` at `238767527555cb75a05732a84dff5d6ba0dd6809` | 2,604,711,182 bytes / 901 | `0118a58bcb1ec4a0bc97db53172581eef9e1762238e4b733972a29ec3ba2551b` |

Conversion provenance, bounded worker execution, numerical comparisons and diagnostic timings are recorded separately. Neither Q4 export meets all numerical acceptance criteria.

### Q4 parity and worker evidence

The converted affine Q4 artifacts were exercised against new source-specific pinned MLX-LM references, using the same case lengths and unchanged BF16 numerical gates. These references are distinct from the official BF16 and canonical HF references. Every case retained next-token greedy agreement, but the strict numerical gates did not pass for all cases:

| Family | Numeric parity | Failed cases | Diagnostic |
| --- | ---: | --- | --- |
| OLMo 3 Q4 | 4/5 | 1-token case | Raw RMSE 0.127956 exceeded the 0.1 limit; centered RMSE 0.047431 exceeded its case bound 0.047090. Other four cases passed. |
| Gemma 4 E2B Q4 | 3/5 | 23-token and 129-token cases | All five retained greedy agreement; two failed the Gemma BF16 numeric profile. |

These are diagnostic comparisons, not qualification: Q4 results are not promoted by all-case greedy agreement. Both Q4 generation/scoring workers passed two generation cases with exact greedy sequences, replay and restart replay; scoring cache invalidation and error recovery passed. This worker evidence does not change the numerical parity outcome. Serialized Q4 timing measurements are in the [matched benchmark report](../../benchmarks/OLMO3_GEMMA4_NATIVE_MATCHED.md): Stark's combined work rate was 14.0% below MLX for OLMo and 2.4% below for Gemma. The longer forced-128-token benchmark trajectories differ across engines; these remain diagnostic timings.

## Build and test receipts

- CPU Rust test receipt: 442 main tests, 7 and 1 integration tests, and 2 doctests passed; 20 ignored, no failures.
- Metal Rust test receipt: 660 main tests, 7 and 1 integration tests, and 2 doctests passed; 39 ignored, no failures.
- Converter suite: 5,055 passed, 90 skipped.
- Generation-worker unit tests: 44 passed.
- Controller tests: 103 passed.
- Repository policy self-tests: 41 passed.
- Formatting, Ruff, dependency-lock, and repository-policy checks passed.
- Final OLMo/Gemma parity and worker receipts: passed as summarized above.

## Receipt identities

Release generation and scoring worker SHA-256 values are `209a76ea50db5cd5602558cdcd73e3401f4c781c31dd738789b2c962abc85612` and `112620e48291fb3fdf558667ea51b6456e71be3a685d50f3d12eb5eb8339fa99`. Workers were built with Rust 1.98.1; the engine test suites used `nightly-2026-10-01`. Hardware was Apple M5 Max with 128 GiB unified memory, macOS 26.7.1 build 25G241. Detailed artifacts remain private; the hashes bind the aggregate statements above to their local receipts.

| Private receipt | SHA-256 |
| --- | --- |
| `olmo3-parity-final.json` | `e67c033301db428e99e08f61cabdc947b8d2f94316c90f811af9677a4645b0ed` |
| `gemma4-parity-canonical-final.json` | `5dfc797990aaf4ef46086f8239e412df1c8065d5789025466b5cac7a64e6ed74` |
| `olmo3-workers-final.json` | `d4c2c52430cbcd7d2ff3bbf993fe0652a5388ad13abbbc2fd62defa3813c4e3a` |
| `gemma4-workers-final.json` | `998d8892f5dacf10d51ccbe3125fd8a6b5d1a276b40dca57e4fe25723e4bffc6` |
| `gemma4-image-workers.json` | `549d2c96e113b8226ed44a231c1b4ab269756d6fc6dc6d12e1ad8802b9c5977e` |
| `gemma4-image-client.json` | `8297d2a774d414d6ecc25be1fbac7a2b463d0a659831821eafdf619a9d6a7940` |
| `gemma4-image-hf-reference/native-validation-v2-receipt.json` | `9b3b0b2f40787d404288aa0b0a00cbb3b76e28c31fcf4c9c3b1cd72e396c9ae5` |
| `gemma4-image-hf-reference/native-metrics-v2.json` | `b5451789251abdd15cd59857e714341aa3e5a940a580ee4c187e1c78d48ede4f` |
| `olmo3-q4-parity.json` | `241a7bc5ba75e23730b1bddab45b11a7671fa437208fdd19561a207a58af9613` |
| `gemma4-q4-parity.json` | `de215e09042e74f66030a982c4c32628a5477f938d362deaa23af77f1fdf3f79` |
| `olmo3-q4-workers.json` | `1431df5790df1f684c2d68975f455db62fc558c94cbc58f407b96fc89d9945fb` |
| `gemma4-q4-workers.json` | `87ae402449c840ac88518c20e7c6a45361f6b785b7bf681c735d6c57feedbf4e` |

## Remaining qualification limits

- Affine-Q4 numerical qualification remains incomplete. Diagnostic timing does not resolve the failed error gates.
- Native validation is bounded to the recorded BF16 text and Gemma E2B image profiles and fixtures. It does not qualify other Gemma sizes, audio/video, or broader multimodal input types.
- Performance timings in parity and worker receipts are diagnostic samples, not matched benchmark results; no speed claim is made here.
- Runtime receipts and private evaluation fixtures remain ignored local artifacts; only aggregate values and hashes are included here.
