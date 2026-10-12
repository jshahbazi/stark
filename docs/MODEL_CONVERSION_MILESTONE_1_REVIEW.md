# Independent conversion milestone 1 review

Date: 2026-10-11. Review disposition: **accept G0, G1, and the selected G2 conversion, load-only, and memory evidence described below**. No unresolved promotion-blocking correctness, source-integrity, provenance, privacy, or native-status finding remains in the final packet. The raw conversion latency objective failed for all four measured checkpoints and remains open. Complete registry replacement and execution qualification remain open.

This review applies the replacement policy in the [current plan](MODEL_CONVERSION_PLAN.md). Lalamo is an attributed external reference; its installed runtime, package/lock, runner, and fallback routing were removed. The [historical plan review](MODEL_CONVERSION_PLAN_REVIEW.md) and prototype results retain their original scope and source versions.

## Review scope and independence

The reviewer inspected contracts, the pinned catalog and exporter, source resolution/replay, metadata profiles, Qwen3 and new dense adapters, bounded transforms, publication, recursive provenance, stored native qualifications, the exact comparator, benchmark subprocess handling, and the final [milestone evidence](MODEL_CONVERSION_MILESTONE_1.md). The pinned reference configuration/loaders were inspected independently. Aggregate hashes, tuple identities, benchmark outcomes, and final converter/lock provenance were checked against the retained evidence.

The reviewer did not author production implementation, download model checkpoints, or run model inference. The reviewer authored [cast regressions](../tools/converter/tests/test_casts.py) and their frozen [external-reference fixture](../tools/converter/tests/fixtures/dense_casts.json), then reviewed the production fixes made by the implementation owners. These tests cover all nine BF16/F16/F32 single-cast combinations, normalization widening, two small workspaces, fused/unfused paths, mixed-dtype manual-operation rejection, and transpose output-hash fallback. Their 49 cases passed. An earlier focused source/provenance/catalog/cast suite also passed independently. The final integrated suite and model/native jobs were run by the designated implementation/validation owners; their results are reviewed evidence, not a claim that the reviewer reran every job.

The cast oracle used pinned external JAX CPU behavior, preserving exact bytes for rounding, overflow, signed zero, subnormals, infinities, and quiet/signaling NaNs. The comparator was not relaxed to accept numerical similarity or altered NaN payloads.

## Source-version binding

| Reviewed input or evidence | SHA-256 |
| --- | --- |
| Final converter sources | `f794a77cee3c8339d7806bd070a0a18389e5e3b973a4b42f7a37225d75bcaa96` |
| Converter lock | `2030e19c3997c05b92df598f4d0d05eb4a88d409189d5f87ff03c96710f6be32` |
| Four-model measured converter sources | `889ab888e44acff52257dfff85219514ec2ee2b8fb81d971fe33b688bcd4d785` |
| Four-model benchmark aggregate | `dd6eff4887c55e3dcb7e84a3adfc1b8ed559832558cf8cdeea7538e02d155071` |
| Final-source isolated Qwen3 aggregate | `7543f2afe85239a53955732d5df0b44d6ef9ba12619dbea0b0506053e0c569e8` |
| Initial benchmark aggregate | `7fd5f019aa982a64e024fdc55239a8402aaa8bc79e50f477eeb89371d40571a3` |
| Six structural-variant aggregate | `4a5cdd1fe8b3dcf5e3d6f7b406b8244b3d2d7c787bdd261a5a6eaac31f4f4592` |
| Catalog | `dd2d86200556c8f020334fdd42625577fc87b4d0acf3d22280b0b70460ddf670` |
| Stored native qualifications | `393c39e7b68962b459d805579a65623da61e0928b4f0cda6c0fc8de39063f1bd` |

The initial measurements retain source `6088a3ed789c9bbd655d1bebede1576813f478b133837400f28ac557d2fd1b4b`. They were not rewritten after optimization. The four-model comparison retains source `889ab888...`; the final change repairs only benchmark receipt draining and subprocess/reader cleanup. CLI, adapter, metadata, and writer behavior did not change between that measured version and the final source. The separate final-source Qwen3 baseline verifies the repaired harness. No live reference ratio is assigned to that baseline, and the earlier overlapping harness smoke contributes no timing claim.

## Gate assessment

| Gate or claim | Decision and evidence boundary |
| --- | --- |
| G0: catalog, contracts, inventory, status semantics | Accepted. All 110 pinned built-ins are accounted for: 107 language, two classifier, one TTS. Three Stark extensions are counted separately. Nineteen built-ins and three extensions have implemented profiles. Unsupported rows remain explicit. Catalog probing performs no source validation and cannot confer native or execution qualification. |
| G1: independent integration and source/publication checks | Accepted. Independent Qwen3, explicit quantized refusal, locked framework-free runtime imports, immutable primary/external source resolution, private offline replay, complete layout validation, mutation checks, owner-only staging, sanitized failures, no-replace publication, provenance, and final-source regression evidence are present. Existing CI adds model-free converter checks without enabling inherited release/cloud infrastructure. |
| Selected G2: exact dense conversion | Accepted for the four exact public BF16 tuples in the milestone ledger and the six recorded structural variants. Tensor names/shapes/dtypes/payloads, tensor metadata/specs, root/nested config, generation/token-codec metadata, and normalized tokenizer all compare exactly against independently generated pinned-reference artifacts. Other implemented profiles remain unqualified. |
| Selected G2: native loading | Accepted only for the four exact stored artifact/source tuples with the named Rust 1.98.1 release build, default plus optimized features. Readiness followed by EOF performed zero native scoring calls. This qualifies recorded construction/loading and supplies no execution result. |
| Selected G2: transformation memory and observed RSS | Accepted for the measured cases. The 8 MiB live transformation-buffer bound is tested separately from process memory. Observed streaming tree RSS stayed below 1 GiB, and the preregistered median-RSS reduction hypothesis passed for all four checkpoints. |
| Raw complete-command latency | Failed for all four checkpoints. Remains open; no speed acceptance or generalized speedup claim is granted. |
| Complete replacement and execution | Open. Ninety-one built-ins still need adapters, and broader source/config/dtype qualification remains outstanding. Scoring, generation, classification, and TTS require separate evidence. |

The accepted load tuples are Qwen/Qwen3-0.6B, Qwen/Qwen2.5-Coder-0.5B-Instruct, the explicit unsloth/Llama-3.2-1B-Instruct extension, and the explicit mistralai/Mistral-7B-Instruct-v0.3 extension, at the immutable revisions recorded in the milestone ledger. No official Meta access or qualification claim is accepted. The Mistral extension's independently constructed LanguageModelSpec is disclosed and is not counted as an upstream registry entry.

The final validation packet reports **418 passed, 18 unsupplied private-reference profiles skipped**, comprising 414 model-free cases plus four supplied real-reference metadata checks. The remaining skips are visible qualification gaps. Ruff, formatting, lock, repository policy, advisory-gate tests, and actionlint checks pass. CI workflow wiring was reviewed; a hosted run was not observed. The cached offline OSV result concerns its database snapshot; it does not establish current online advisory coverage.

## Resolved findings

| Finding | Resolution reviewed |
| --- | --- |
| Exporter omitted from converter provenance by an incorrect directory | Hash the actual converter exporter; nested changes, additions/removals, and exporter mutation/deletion are covered. |
| Native provenance omitted template and vendor inputs | Inventory all relevant files under crates, .cargo, and vendor, plus root Cargo/toolchain inputs. The native receipt binds the tested executable separately. |
| Replayed config root binding depended on a caller-controlled role | Bind the registered config filename to the checkpoint root independently of its role. The regression uses a fully valid alternate file record so it reaches the binding check. |
| Newly added index/shards could redirect planning beyond the manifest | Recompute and compare selected weights/index against the manifest on unchanged-source checks, including the cheap planning mode. |
| Mixed-dtype fusion cast differed from reference promotion for signaling NaNs | Reject mixed dtypes within a fused matrix/bias group after complete source-layout validation; manual writer operations reject before output creation. Frozen exact comparison remains unchanged. |
| Comparator accepted extra tensor-header fields or empty tensor names | Enforce exact header fields, valid nonempty names, strict spans, duplicate-key rejection, finite runtime JSON, and valid tensor-spec metadata. |
| Provenance failure could escape staging cleanup | Place provenance work inside the publication cleanup boundary and verify failure/interruption cleanup. |
| Capability model-kind mapping used the wrong classifier class | Report the actual registered ClassifierModelSpec kind while refusing its unimplemented conversion. |
| Qualification lookup could conceal scope or mutate before publication | Validate/deep-copy exact records, reject duplicates/ambiguity, always emit the full named-build receipt, and recheck toolchain/source identity after lookup. |
| Planning receipt overstated source validation after signature-only checks | Planning retains source_validation=not_performed; successful verified publication reports complete. |
| Repeated input/output hashing added avoidable complete-command cost | Keep full resolver/replay verification and final content rehash; use selection/signatures only for the repeated planning check. Hash sequential writes incrementally and rehash completed seek/transpose output. Whole-file hashes are independently asserted. |
| F16 normalization warnings could expose noisy diagnostics | Suppress expected NumPy invalid warnings locally while preserving exact widened NaN bytes. |
| Large valid subprocess receipts could deadlock benchmark pipes | Drain communicate concurrently with RSS polling; sanitize reader failure; terminate/reap owned process groups and descendants and join readers on interruption. Regressions cover a 1.8 MB receipt and a signal-ignoring descendant holding a pipe. |

The signature-only planning optimization does not replace content verification. Full hashing remains the default, offline replay performs it, and final publication performs it again. A restored-stat persistent content mutation is specifically tested: it can pass the cheaper planning check and still fails replay/final content verification. Source-file selection and config/root binding remain checked in both modes.

## Performance assessment

The four-model source-bound complete-command medians were 1.895/1.250/2.997/15.435 seconds for Stark and 1.389/1.193/2.006/6.568 seconds for the external reference, in Qwen3/Qwen2/Llama/Mistral order. Every raw latency hypothesis failed. Median sampled tree RSS reductions were 90.07%, 92.08%, 92.22%, and 99.33%; all hard parity/workspace/observed-RSS checks passed.

These are complete-command measurements. Stark includes stronger source verification and receipts than the external reference, so the ratios do not isolate equivalent tensor transforms. Both commands produce the accepted artifacts. Initial-to-optimized improvements are descriptive sequential-phase observations. The historical prototype's 1.83× figure does not apply to this implementation.

The final-source isolated Qwen3 run reports three exact comparisons, median complete CLI time **1.980 seconds**, range **1.932–2.018 seconds**, and median sampled tree RSS **206.8 MiB**. It includes the uv wrapper and has no live reference timing. Ten-millisecond RSS sampling can miss short peaks and can count shared pages multiple times. These observations support the stated measurements, not a total-memory guarantee.

## Remaining work and limits

- Pursue the open raw-latency objective while preserving full source integrity, exact byte/spec/config/tokenizer parity, sanitized receipts, and new source-bound measurements. Do not remove acceptance checks to improve a ratio.
- Qualify remaining implemented sizes/configs and implement the 91 outstanding built-in adapters. Official Meta access, the 18 unsupplied metadata-reference profiles, optional later-row dependencies, and unsupported representations remain explicit gaps.
- Add execution qualification separately. Native readiness/EOF is load-only. F16/F32 native support, compressed/quantized formats, hybrid/state-space models, MoE, classifiers, and TTS remain blocked or later work.
- Add defensive caps for very large dimensions/header/config/tokenizer metadata and a platform-specific process-memory failure contract if required. The transform workspace covers live transform buffers; metadata, catalogs, hashing, and interpreter overhead remain additional allocations. Selected-model RSS observations do not close adversarial total-memory limits.
- Keep distribution and host integration separate. Preserve upstream licenses, pinned source revisions, and prior source-bound validation without silently refreshing old results.

No Rust, Metal, Cargo dependency, scoring path, or runtime format changed in this patch. Existing converted Jev/JevK5 artifacts retain their execution path; no new Jev/JevK5 conversion, inference-latency, or judge-quality evidence is claimed. Public review evidence contains public identities, aggregate results, and hashes only. Private source paths and model/native operational artifacts remain ignored, and prohibited execution content was not part of this review.

The first milestone may be promoted with this scoped acceptance. The open latency, full replacement, and execution objectives must remain open in the plan and coverage ledger.
