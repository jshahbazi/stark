# Milestone 6 technical review

Review baseline: `97ca08b7434415b98e70bbb8098782aaaec7da4c` on `codex/conversion-milestone-4`. The reviewer does not author the adapter, metadata implementation, tests, fixture generator, source resolver, qualification ledger, native harness or benchmark harness.

## Scope

This packet implements the two dense ModernBERT classifier identities in Lalamo revision `831accb8252a6b6606c378048caeb44178f1a050`: `trymirai/chat-moderation-router` and `trymirai/health-related-router`. Authenticated lookup returned 404 and anonymous lookup returned 401 for both identities. Their live access and native qualification remain unresolved; these responses do not establish why the repositories are unavailable.

Real artifact validation uses two separately identified public ModernBERT classifiers: `SoFairOA/sofair-modernBERT-base-filter` and `MoritzLaurer/ModernBERT-base-zeroshot-v2.0`. Each local composite contains unchanged pinned public weights, config and tokenizer files plus an explicitly declared synthetic chat-template sidecar. The composite revision is `null`; the original public revision and every original file hash are bound separately. These fixtures exercise the reusable `hf-generic` classifier route. They do not substitute for either Mirai identity. Generic receipts remain native-unverified even if their exact local artifacts pass a private native construction proof. Conversion acceptance, exact native loading, classification execution and performance remain separate claims.

## Verdict

**Accepted for the scoped ModernBERT classifier conversion, strict source/metadata validation, bounded workspace, and two exact private generic-artifact load-only proofs. No blocking findings remain, and the requested model cleanup is independently verified.** Both selected composite medians were lower than Lalamo; F32 timing ranges overlap. The two Mirai catalog sources remain access-unresolved and have no exact qualification. Classification execution, quality and complete Lalamo replacement remain open.

## Independent checks completed

- Read the pinned ModernBERT bridge, classifier loader/export paths, shared tokenizer importer, and native classifier construction paths.
- Independently ran **258 ModernBERT adapter tests** and **73 classifier metadata tests**, all passing. The frozen oracle contains **99 actual full-classifier exports**: eleven semantic variants across all nine BF16/F16/F32 source/target precision pairs. Both single-file and sharded inputs match the frozen tensor, spec and complete classifier-config hashes. Tests cover biases, local/global attention intervals, numeric label order, F32 normalization, unsupported variants, tensor completeness and bounded `Wi` reshuffling.
- Independently reran twenty synthetic metadata cases through the actual pinned `ClassifierModelSpec` shared codec importer; every codec and normalized tokenizer matched. This invoked no weight loading, classification or scoring.
- Independently ran eleven complete-CLI synthetic checks: anonymous local generic conversion, exact offline replay, classifier root and absent generation policy, native-unverified generic receipts, content-bound unused generation sidecar, wrong model-kind/architecture rejection, forged replay architecture rejection, and generic native-qualification rejection. The provisional earlier source-bound result remains preserved separately.
- The coordinator's final integrated validation passed **3,672 tests / 60 skipped**, with locked dependency checks, formatting, repository policy and its seventeen policy tests passing. The skipped checks require other private real metadata reference views; they are not native qualifications for this packet.

## Real artifact and native proof audit

The reviewer independently parsed and streamed every candidate and reference tensor, validating names, shapes, dtypes, payload hashes and full logical matrix specifications. Complete root config and tokenizer objects matched. It independently rehashed all four retained weight artifacts and their config/tokenizer files, replayed both selected source manifests with full content verification, and checked the original public cache files against the unchanged local copies and separately declared synthetic overlay.

| Exact local composite | Original public revision | Tensors | Private load proof SHA-256 | Independent audit SHA-256 |
| --- | --- | ---: | --- | --- |
| `SoFairOA/sofair-modernBERT-base-filter`, F32 source | `6f5ea32876e6bd942493321ce7025c917394c369` | 138 | `abced84f2add06a6496ef85548958a3f0d9e158b6ada59d64cd2afced7bf1d08` | `3847f42fd20007cd86fbf6ade125eb89caf68fdf0fdfd76915aaf6937d99de4e` |
| `MoritzLaurer/ModernBERT-base-zeroshot-v2.0`, BF16 source | `d421c4545a438fd006fb43f8b981c5d908faa1e1` | 138 | `56aa67b50b725f659c4ca039b8332a769b4de6abf735ee705c10cc4baab696d3` | `cb0fabb8ccd4cf9546ffdd81f9272cf65a73aeea4fbc9bb76bdbe96a0b7ca5ea` |

Both BF16 targets passed native construction/readiness followed by stdin EOF, with zero classification, scoring and generation calls. The private harness was inspected and its binary, source, Cargo manifest and Cargo lock hashes independently checked, together with the complete unchanged native engine source manifest, repository Cargo lock and Rust toolchain. This exact-artifact load proof is separate from catalog qualification. Both generic receipts continue to report native loading and execution as unverified.

The packaged qualification ledger remains **byte-identical to baseline**, with the original twenty-seven records. The two Mirai rows have no exact native qualification. Counts are therefore **85/110 implemented built-ins plus four extensions**, **27 exact catalog-qualified BF16 tuples**, and **62 implemented identities without an exact catalog qualification**. Generic local composites are outside that inventory denominator.

## Evidence bindings before timing

| Binding | SHA-256 |
| --- | --- |
| Final converter source tree | `2e99313bf8576ca234f06116af7cde9517fb025fa5fd7de287155a5f7ffed300` |
| Final converter lock | `1354c36a99cdc0b27bb3ff3f1d12c54aafb9fb8620a6ed7d633f6552b3d72c65` |
| Unchanged native qualification ledger | `a08a969c68ec20447e960771af0e1cffe6dabc19f6bdb7c513c9853df19fe607` |
| Full-classifier 99-export oracle | `5e809cb14203bfab0bfbabec61bbb039f25ffad1ad5a2d03eb9ca014b37237dc` |
| Final twenty-case actual metadata-import comparison | `7af26617fedb0993b15de99e338ef34d2932378da80a3e8b58d38ee9d4bd59cb` |
| Eleven-check final independent CLI/replay audit | `5689fd0898d63a761c09386c80200a88c5f9ee89be1a5d2ab1b50253a60ec9e0` |

## Independent benchmark audit

The final audit recomputed every summary from all **twenty serial fresh complete commands**, five per tuple/backend. It verified alternating backend order with the starting order rotated across tuples, every persisted full-parity record, all ten generic candidate receipts and their exact source/output/config/tokenizer/toolchain bindings, the private exact-artifact load proofs, workspace/RSS gates, and unchanged prior qualification records. No timed failure, dropped observation or favorable rerun occurred. Every observation was persisted before its acceptance checks.

| Exact local composite | Stark median (s), range | Lalamo median (s), range | Median time reduction | Median sampled RSS reduction |
| --- | --- | --- | ---: | ---: |
| MoritzLaurer ModernBERT base, BF16 source | 0.561545, 0.538579–0.701313 | 1.136597, 1.073213–1.626052 | 50.59% | 84.90% |
| SoFairOA ModernBERT base, F32 source | 0.861474, 0.785402–1.331192 | 1.312483, 1.191968–1.486351 | 34.36% | 89.58% |

Both targets are BF16. The BF16 observed ranges are disjoint; F32 ranges overlap. These are descriptive five-repeat, one-host warm-file results, not statistical significance, cold-cache, inference-speed or architecture-wide performance evidence. Setup, downloads, source warming, parity checks and native load-only validation are excluded from command timing. Candidate complete commands include full source and receipt validation; the reference runner provides weaker primary source/receipt checks. All work from this task that could compete with conversion was paused during timing; this is not a claim that the entire host had no background processes.

Each candidate used at most 8 MiB transform workspace and sampled tree RSS below 1 GiB. Ten-millisecond process-tree sampling can miss peaks or double-count shared pages and does not enforce an OS memory limit. Candidate whole-file hashes match the exact privately load-validated artifacts. Lalamo's fresh whole-file weight hashes vary across serialization, while every tensor name, shape, dtype, payload, matrix spec, full logical config and tokenizer matches. The audit retains that variation; reference whole-file byte identity is not a preregistered parity requirement.

The reviewer rehashed all four frozen outputs and both complete selected sources after timing, including original cache files, local copies and overlay. It independently checked the frozen converter/native/harness inputs, clean pinned reference checkout/interpreter/lock, objectives, benchmark coordinator, fixture generator and synthetic input definition. All bindings remained unchanged.

| Final benchmark binding | SHA-256 |
| --- | --- |
| Preregistered settled objectives | `9705da17191e4cadba84bd99f178ce1a140b82b7b70155d5f6a94f5749f77a7c` |
| Frozen serial-run inputs | `326ba79e360ce434f75dbb612be370f2848f2880c6e83cb30a69439d62251255` |
| Complete benchmark coordinator | `07e3c546c90db9441dd1f57e5a1767a32ab9738e19d0968c631cd93afe759ac4` |
| Complete twenty-command result report | `ea5b58c404a3f094ae740f96d05fdcd6dddf097aac40a9c901bad6cd7e4e6bab` |
| Final independent benchmark/artifact/source audit | `3ff7284a91b398ab3fd5ef3f86fe8522e666cd72eedc9bae9093e4b9bc01e74b` |

## Completed model cleanup

After all source, artifact, native and benchmark audits, the user-directed cleanup removed **twelve selected paths**: six local source/frozen output weight files, four selected Hugging Face cache aliases and two backing weight blobs. The reviewer independently inspected the unlink-only script and exact path inventory, checked signatures, link counts and resolved routes, found no unselected cache alias to either blob, and ran a fresh open-file check before execution. No unrelated model, serving model or small evidence record was selected.

The independent post-cleanup check confirms every selected path is absent and no `model.safetensors` remains in the milestone 6 packet. It verifies twenty-six retained source/artifact/metadata records, all relative documentation links, and unchanged benchmark and pre-cleanup review audit hashes. Regular selected weight bytes total **2,992,594,888**; observed free-space increase is **2,992,832,512 bytes**, about **2.79 GiB**. Storage readings are observations and can also reflect other host activity.

Artifact-retention statements above describe their inspection state before cleanup. Historical validation receipts, source/content hashes, native proofs, complete observations, metadata, fixture recipes and scripts remain preserved. Offline source replay now needs the removed weights restored from the recorded pinned sources; the retained records do not constitute available model weights.

| Cleanup binding | SHA-256 |
| --- | --- |
| Twelve-path inventory | `990e730c6ae87f8107c9cd97c366062f4cc88c28450cd1b80145f723ea2b44eb` |
| Unlink-only cleanup script | `838d66e5b502f399703f2ea74ffa79fe506b6f9139e5a5a11e112f0c612acf56` |
| Independent preflight approval | `006ab1f49e363f8ff1545a80a8b55482daae515073f287711dfb116fd166d44d` |
| Completed cleanup audit | `400cb7838db7f03b6a5e84bff50785c968b7580673269f4b3a5ceb79e339f101` |
| Independent post-cleanup audit | `99bbbed5c7f91620288e6f3e393c3bc66e246828832f2bee4b16a7122562060f` |

## Initial contract review

The pinned classifier importer emits `ClassifierModelConfig` with `classifier_config` and a token codec. It does not emit a language decoder root or generation configuration. The ModernBERT loader uses `model`, `head` and `classifier` source roots; its MLP `Wi` row halves are reshuffled before export. First-layer attention normalization is absent, attention is noncausal, local and global RoPE configurations alternate by the configured interval, and GELU is exact. Normalization uses mean subtraction with F32 parameters and no normalization biases in the pinned bridge. Prediction-head dense bias follows `classifier_bias`; the final readout always has a bias.

Output labels must follow contiguous numeric identifiers. Strict inverse `id2label`/`label2id` validation and unique labels prevent ambiguous mappings. Dense input format, matrix dimensions, tensor completeness, recognized extras and source immutability still require validation independently of catalog implementation status.

Stark has a BF16 classifier loader with mean pooling. The language scoring worker does not establish classifier loading. Any new classifier load-only proof must bind a separately named executable, the unchanged engine/build inputs and the complete private harness source/build inputs. Zero classification calls and execution-unverified status must remain explicit. The prior twenty-seven language qualification objects must remain unchanged.

## Evidence and model lifecycle

Real prompt templates, tokenizer payloads, model weights and native artifacts remain ignored and private. Public documentation records aggregate results and hashes. No classification content, token vector, decision probability or exception dump is permitted in validation output. After artifact and benchmark audits, completed generated weights and selected downloaded weights are removed under the user's cleanup instruction; small manifests, receipts, proofs, scripts and complete observations remain retained.

## Findings during implementation

The reviewer found that initial classifier source validation rejected the anonymous local `hf-generic` route because it required a nonempty source repository string. The metadata worker repaired that boundary while preserving explicit registered-profile repository validation. An independent complete-CLI audit passed anonymous conversion and exact offline replay after the repair.

The reviewer independently reproduced acceptance of tokenizer padding whose literal padding identifier exceeded the configured vocabulary. Vocabulary, Hugging Face special-token metadata and post-processor validation did not cover this independent padding field. The metadata worker added a vocabulary-consistency guard and regression checks; the reviewer independently reproduced rejection after the fix, with zero execution calls.

One provisional reviewer helper ran before adapter exports were integrated and emitted a static `ImportError` traceback containing code paths only. It contained no prompt, response, tokenizer payload, model data or credential. This setup output violated the no-exception-dump instruction. Subsequent helpers use a top-level sanitizer and fixed aggregate error categories. This limitation must remain disclosed in the final review.
