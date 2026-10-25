# Milestone 6: ModernBERT sequence classifiers

This packet continues the [replacement plan](MODEL_CONVERSION_PLAN.md) from `97ca08b7` on `codex/conversion-milestone-4`. Version 0.9.0 independently implements the ModernBERT classifier layout in Lalamo revision `831accb8252a6b6606c378048caeb44178f1a050`. Lalamo, JAX and PyTorch remain outside the installed converter's dependencies. [Independent technical review](MODEL_CONVERSION_MILESTONE_6_REVIEW.md) records the final disposition.

## Implementation and coverage

The catalog now implements **85/110 built-ins plus four extensions**: 43 dense language profiles, 40 MLX profiles and two dense classifier profiles. Twenty-five built-ins remain planned. The two new identities are `trymirai/chat-moderation-router` and `trymirai/health-related-router`. Live authoritative probes returned repository-unavailable responses with current credentials and unauthenticated access; these results do not establish whether the repositories are private or deleted. Both remain **access-blocked and native-unqualified**. Public validation fixtures do not qualify these identities.

The reusable `modernbert` adapter validates the sequence-classification architecture, mean pooling, GELU semantics, local/global noncausal attention, per-layer RoPE, first-layer normalization exception, FP32 subtract-mean normalization, embedding/head/readout parameters, optional attention/head biases, numeric label order and inverse label maps. Despite the source annotation `position_embedding_type=absolute`, the pinned implementation exports per-layer RoPE. The classifier's readout always has bias. ModernBERT's Wi gate/up halves are reversed to match the actual pinned loader; bounded views retain the original complete source tensor signatures.

Dense BF16/F16/F32 source and target precisions are supported. Missing declared biases, inconsistent/duplicate labels, malformed dimensions, unsupported pooling/activation, unexpected tensors and packed layouts fail explicitly. `decoder_bias` is validated as a Boolean but belongs to the unused masked-LM head for this sequence-classification architecture; unexpected masked-LM decoder tensors still fail complete source consumption. Omitted pooling is rejected rather than silently treating the upstream default CLS pooling as mean.

The CLI publishes `ClassifierModelConfig` with `classifier_config` and `token_codec_config`, without a generation policy. The legacy internal `ConversionPlan.decoder_config` slot holds the complete classifier configuration for this adapter; root routing and language-only native mixer checks use the explicit model kind. Classifier sources do not require the inherited ConfigMap generation-file default because the pinned classifier importer never selects it.

Classifier metadata preserves the pinned chat codec and normalized tokenizer, including post-processing and padding. Label maps, vocabulary bounds, added-token identifiers, special-token mappings, padding identifiers and literal post-processor identifiers are validated. BOS/EOS codec fields are optional and do not define classifier generation stops. Metadata errors remain sanitized.

An uncatalogued self-contained checkpoint with `model_type=modernbert` can use `--metadata-profile hf-generic --model-kind classifier`, optionally asserting `--architecture modernbert`. The source profile's architecture binds the root kind for live conversion and schema-2 offline replay, including anonymous local sources. Language/classifier/unsupported-kind mismatches, forged replay architecture and known catalog generic overrides fail. A declared supported chat template is required; none is invented. A present generic generation sidecar remains content-bound but is not interpreted. Generic native status remains unverified and `--require-native-support` is refused.

## Parallel work and validation

Separate workers owned the ModernBERT tensor planner/full-export oracles and classifier metadata/import parity. The coordinator owned CLI/catalog/registry/source-profile integration, typed qualification evidence, version/lock, the private classifier harness and documentation. The reviewer authored no production code, tests, generators, benchmark harness or qualification proofs.

The final synthetic fixture contains **99 actual pinned full-classifier exports**, eleven config variants across all nine BF16/F16/F32 precision pairs. Tests exercise both shard choices, attention/head biases, first-layer normalization, global/local periods, numeric ordering beyond nine labels, mixed FP32 norms, dropout annotations and unused masked-LM bias. Every tensor name, shape, dtype, payload hash, logical matrix spec and complete classifier configuration matches the pinned exporter. Twenty separate synthetic metadata cases match the actual pinned classifier codec importer and tokenizer bytes.

The final integrated suite passed **3,672 tests / 60 skipped** in 46.76 s. The skipped checks are language profiles without supplied private real metadata reference views; the two inaccessible classifier rows are documented separately. Ruff check and format passed for 59 files. The locked dependency check remains 27 packages, with only the editable package version changing. Repository policy, seventeen policy unit tests and whitespace checks passed. No hosted CI run is claimed.

## Public real-weight validation fixtures

The inaccessible Mirai rows were not substituted. Two separately named public ModernBERT sequence classifiers provide real-sized validation and benchmarks:

| Public weights | Immutable original revision | Source precision | Tensor payload bytes |
| --- | --- | --- | ---: |
| `SoFairOA/sofair-modernBERT-base-filter` | `6f5ea32876e6bd942493321ce7025c917394c369` | F32 | 598,425,608 |
| `MoritzLaurer/ModernBERT-base-zeroshot-v2.0` | `d421c4545a438fd006fb43f8b981c5d908faa1e1` | BF16 | 299,212,804 |

Each local composite copies the selected immutable public weights, model config, tokenizer and tokenizer config unchanged and adds an explicitly declared synthetic chat-template sidecar. The public repositories do not supply that template. Local composite manifests truthfully use `primary_revision=null`; separate original-source provenance binds each remote commit, file hashes and overlay recipe. These are public real-weight fixtures with a synthetic codec, not unchanged direct conversions of either public repository and not Mirai router qualifications.

Both BF16 outputs contain 138 tensors and match every pinned reference tensor/spec, complete root config and normalized tokenizer. Each output's tensor payload is 299,283,460 bytes; file headers and matrix metadata add serialization bytes. Whole-file serialization can differ between candidate and reference despite exact payload and logical metadata parity. The reference runner uses actual pinned `ClassifierModelSpec.import_model` and model save/export against the explicitly supplied composite source. Candidate receipts remain `catalogued=false`, native-unverified and execution-unverified.

| Composite | Source-manifest SHA-256 | Candidate weights SHA-256 |
| --- | --- | --- |
| SoFairOA F32 weights + synthetic codec | `27c8115d61924781f8562dc61473de9c0d7f128eb1133c155fb91ca52542bd6b` | `eaaf1204a19f240bd23a15fb72f6db62181032592538b4302e9bf38748197106` |
| MoritzLaurer BF16 weights + synthetic codec | `f2a8eaa7216856829ae36a02da29bdd78ff467aee7630adc433cc8e066b04258` | `d53b68991399c89ffc7948cf318ef1359954f71df0ee36e53ce625aebcba2b4a` |

Both exact composite outputs also passed the private classifier loader's construction/readiness followed by stdin EOF, with **zero classification, scoring or generation calls**. The harness uses Rust 1.98.1, locked release and the explicitly recorded default/optimized features. It suppresses native diagnostics and panic output and accepts no classification requests. Its binary, source, Cargo manifest and private lock are bound separately from engine source/root lock/toolchain hashes; this is a classifier load harness, not the scoring worker. No classification correctness, probabilities, quality or inference performance were evaluated.

| Exact composite proof | Proof SHA-256 | Independent artifact/source/build audit SHA-256 |
| --- | --- | --- |
| SoFairOA | `abced84f2add06a6496ef85548958a3f0d9e158b6ada59d64cd2afced7bf1d08` | `3847f42fd20007cd86fbf6ade125eb89caf68fdf0fdfd76915aaf6937d99de4e` |
| MoritzLaurer | `56aa67b50b725f659c4ca039b8332a769b4de6abf735ee705c10cc4baab696d3` | `cb0fabb8ccd4cf9546ffdd81f9272cf65a73aeea4fbc9bb76bdbe96a0b7ca5ea` |

The public qualification catalog remains **byte-identical to the baseline, with 27 records**. There are **62 implemented identities without exact catalog qualification**. Classifier-specific qualification validation now requires an explicit model kind, classifier-harness loader identity, its three source/build hashes and zero classification/generation calls; language records retain their original exact schema. No private generic proof is added to the public native ledger.

## Live Lalamo comparison

The preregistered run measured **twenty serial fresh complete commands**, five per backend for each public composite. All selected files were warmed before each run; backend order alternated and its starting order reversed across tuples. Other task model/tests were paused, while pre-existing user processes were left running. Measurements were persisted before parity/source/memory checks; no failure, outlier removal or favorable rerun occurred.

| Public-weight composite, BF16 output | Stark median, s | Lalamo median, s | Lower median time | Median sampled tree RSS, MiB, Stark / Lalamo | Lower RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| MoritzLaurer BF16 source + synthetic codec | 0.562 | 1.137 | 50.59% | 102.0 / 675.6 | 84.90% |
| SoFairOA F32 source + synthetic codec | 0.861 | 1.312 | 34.36% | 101.9 / 977.9 | 89.58% |

| Public-weight composite | Stark range, s | Lalamo range, s |
| --- | ---: | ---: |
| MoritzLaurer BF16 | 0.539–0.701 | 1.073–1.626 |
| SoFairOA F32 | 0.785–1.331 | 1.192–1.486 |

All twenty tensor/spec/root-config/tokenizer comparisons passed, and every candidate matched the frozen exact private source/artifact/load proof. All ten candidate receipts remained generic/native-unverified with no catalog qualification. Both raw median latency hypotheses and sampled-memory hypotheses passed. BF16 ranges are separated in this run; F32 ranges overlap. Five observations on one host do not establish statistical significance or family-wide latency acceptance.

Candidate command timing includes startup/imports, full initial/final source verification, metadata, planning, export/output hashing, publication and receipt. The clean pinned Lalamo reference command performs actual classifier import/export with weaker source/receipt checks. Download/setup, file warming, full parity and native loading are excluded. There is no earlier Stark classifier implementation baseline.

Each candidate's transformation workspace was at most 8 MiB and its sampled process-tree RSS below 1 GiB. Ten-millisecond RSS sampling can miss peaks or double-count shared pages; workspace and observed RSS are separate from an enforced total-memory cap. These warm-file measurements establish neither cold-cache behavior nor classifier execution, quality or inference performance. The run occurred **2026-10-12 17:27:43.788–17:28:21.551 UTC** on arm64 macOS 26.7.1, eighteen logical CPUs and 128 GiB RAM. All four frozen weights and original inputs were retained through independent post-run auditing, followed by user-directed cleanup.

## New source and evidence bindings

| Binding | SHA-256 |
| --- | --- |
| Converter sources | `2e99313bf8576ca234f06116af7cde9517fb025fa5fd7de287155a5f7ffed300` |
| Converter lock | `1354c36a99cdc0b27bb3ff3f1d12c54aafb9fb8620a6ed7d633f6552b3d72c65` |
| Unchanged 27-record native ledger | `a08a969c68ec20447e960771af0e1cffe6dabc19f6bdb7c513c9853df19fe607` |
| Final 99-export classifier fixture | `5e809cb14203bfab0bfbabec61bbb039f25ffad1ad5a2d03eb9ca014b37237dc` |
| Final twenty-case metadata oracle | `7af26617fedb0993b15de99e338ef34d2932378da80a3e8b58d38ee9d4bd59cb` |
| Independent final CLI/replay contract audit | `5689fd0898d63a761c09386c80200a88c5f9ee89be1a5d2ab1b50253a60ec9e0` |
| Private classifier loader binary | `a2bada0bf41a4d0daa13f6dd517f3f646699583c8044431f91536b30101c5404` |
| Benchmark objectives | `9705da17191e4cadba84bd99f178ce1a140b82b7b70155d5f6a94f5749f77a7c` |
| Benchmark freeze | `326ba79e360ce434f75dbb612be370f2848f2880c6e83cb30a69439d62251255` |
| Benchmark harness | `07e3c546c90db9441dd1f57e5a1767a32ab9738e19d0968c631cd93afe759ac4` |
| Complete twenty-command benchmark report | `ea5b58c404a3f094ae740f96d05fdcd6dddf097aac40a9c901bad6cd7e4e6bab` |
| Independent final artifact/source/benchmark audit | `3ff7284a91b398ab3fd5ef3f86fe8522e666cd72eedc9bae9093e4b9bc01e74b` |

## Model cleanup after validation

At the user's direction, the four frozen candidate/reference weight files, two composite input weight files, two downloaded public weight blobs and their four cache aliases were removed after independent post-benchmark review. All twelve selected paths were confirmed absent. A separately reviewed unlink-only script checked file identities/signatures, initial link counts, exact runtime/cache routes, unselected cache aliases and active use before deletion. Serving models and unrelated cached weights were excluded.

Cleanup completed at **2026-10-12 17:37:57.322 UTC**. Removed regular weights totaled 2,992,594,888 bytes; total logical sizes including symlinks were 2,992,595,198 bytes. Available volume space increased by an observed 2,992,832,512 bytes, approximately **2.79 GiB**. Shared APFS volume activity can affect the free-space difference. Weights were not rehashed for deletion; complete independent payload/source audits had already passed.

Small metadata/config/tokenizer files, source/original-remote provenance manifests, overlay recipes, receipts, load proofs, all twenty observations, frozen hashes, audit records and scripts remain retained privately. The original measured/qualification records were not rewritten. Manifests alone cannot replay removed weights; future model-dependent checks must reacquire the pinned inputs, recreate the declared overlay and verify recorded hashes. This follows the [artifact cleanup policy](MODEL_CONVERSION_PLAN.md#model-artifact-cleanup).

| Cleanup binding | SHA-256 |
| --- | --- |
| Final twelve-path inventory | `990e730c6ae87f8107c9cd97c366062f4cc88c28450cd1b80145f723ea2b44eb` |
| Unlink-only cleanup script | `838d66e5b502f399703f2ea74ffa79fe506b6f9139e5a5a11e112f0c612acf56` |
| Independent cleanup approval | `006ab1f49e363f8ff1545a80a8b55482daae515073f287711dfb116fd166d44d` |
| Completed cleanup audit | `400cb7838db7f03b6a5e84bff50785c968b7580673269f4b3a5ceb79e339f101` |
| Independent post-cleanup lifecycle audit | `99bbbed5c7f91620288e6f3e393c3bc66e246828832f2bee4b16a7122562060f` |

Earlier source-bound validation records and native ledger objects remain unchanged. Intermediate development oracles/contract reports have separate bindings and are not final-code proof. Three setup helpers printed static tracebacks during access/signature/integration investigation, without credentials, model payload, tokenizer content, prompts or evaluation content. This setup limitation is recorded; subsequent preparation, reference conversion, qualification, audits and benchmarks suppress diagnostic dumps and retain fixed aggregate error categories.

Remaining work includes access to the exact Mirai rows and corresponding qualifications, classifier execution acceptance, other pooling/architectures, compressed classifiers and the remaining twenty-five catalog rows. Existing language latency failures retain their earlier results and status. Models and operational data remain ignored/private; publish aggregate evidence and hashes only. Adapted sources include pinned ModernBERT config, Hugging Face loader, classifier module/model, tokenizer import and Mirai specs. [Lalamo's MIT license](../../../tools/converter/LICENSE.lalamo), upstream copyright and Uzu provenance are preserved.
