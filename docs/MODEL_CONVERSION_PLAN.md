# Model conversion integration and coverage plan

Status: implementation plan; expansion work remains pending. Updated: 2026-10-11.

Keep `stark-convert` in this repository as a first-class companion to the Stark engine. Expand its streaming implementation toward the built-in registry of a pinned Lalamo revision while retaining the isolated Lalamo fallback. Source ownership, an independently installable package, and a separate repository are different decisions: source stays here; public distribution is a later scoped task.

This change documents the checklist and delegated technical review. It does not implement the expansion, run new model experiments, or publish a package. The [prototype evidence](MODEL_CONVERSION.md) remains bound to its original source and lockfile hashes. Record subsequent runs separately.

## Baseline and scope

- [x] Prototype committed as `77e9a425`; source lives in `tools/converter`.
- [x] Lalamo fallback pinned to `831accb8252a6b6606c378048caeb44178f1a050`, with its own locked environment and preserved MIT license.
- [x] Dense Qwen3 full-precision conversion, eight synthetic metadata profiles, 122 converter tests, and one real BF16 checkpoint validated.
- [x] Three-run Qwen3-0.6B comparison recorded: 1.83× complete CLI speedup and 89.6% lower sampled process-tree RSS on the recorded host.
- [x] Exact tensor/spec/config/tokenizer comparison and native load-only checks passed for that BF16 checkpoint; one MLX-4bit model exercised automatic fallback and external metadata routing.
- [ ] Make the converter a supported repository tool with documented contracts and model-free CI tests.
- [ ] Audit and classify every built-in registry entry at the pinned Lalamo revision.
- [ ] Extend independently implemented streaming conversion by validated family and format.

The coverage target is the audited built-in registry at the pinned revision, including its required source files and optional dependencies. It does not mean arbitrary Hugging Face repositories, third-party Lalamo plugins, or a moving upstream `main`. Updating the reference pin is a deliberate migration.

Conversion qualification, native model loading, and execution qualification are separate results. Current supported language-model loading selects BF16 dense weights; F16/F32 conversion parity does not qualify native loading. MoE construction is currently rejected. Existing DeltaNet, Mamba2, classifier, or weight-spec types do not establish acceptance for every upstream model. General generation, classification, and TTS each need their own execution evidence; scoring evidence does not establish them.

Host application integration stays outside this repository. Public package registries, signing, release automation, binaries, and distribution hosting are outside this implementation plan. Preserve inherited disabled infrastructure.

## Contracts and capability ledger

Each streaming adapter uses one typed conversion plan to generate both model configuration and tensor output. The Lalamo fallback retains its pinned importer/exporter and writer; it is not required to produce this repository's typed plan. Both backends share source/request/result, qualification, provenance, staging, and publication contracts. Document fallback validation and source-consumption limits rather than implying that its internal accounting is identical to the streaming writer.

| Proposed contract | Required contents |
| --- | --- |
| `SourceSnapshot` | Local primary files; public registry identity when applicable; resolved remote revisions or a hashed local manifest; identities/signatures of every consumed file; external dependency manifest. |
| `ConversionRequest` | Model kind, requested output dtype, backend policy, streaming transform-workspace budget, strict-versus-preferred resource policy, source snapshot, and metadata profile identity. |
| `CapabilityResult` | Supported adapter or bounded fallback route, unsupported option category, known invalid source category, conversion qualification, and independent runtime qualification. Unknown validation status must remain explicit. |
| `ConversionPlan` (streaming) | Complete root model config, output tensor names/shapes/dtypes/specifications, ordered typed operations, consumed/omitted input accounting, source signatures, and enforced transform-workspace bound. Extend the existing streaming plan/writer. Lalamo keeps its reference writer behind the shared backend contracts. |
| `ConversionResult` | Selected backend, exact output manifests, qualification and resource-contract status, aggregate timings, source/reference/lock hashes, and references to permitted private aggregate/hash evidence. |

- [ ] Define typed contracts and adapter registration in `stark_convert/contracts.py`, `registry.py`, and `adapters/`; keep concrete names revisable during contract review.
- [ ] Move the existing Qwen3 planner behind that interface without widening its accepted layouts.
- [ ] Make root model kind explicit: language, classifier, and TTS configurations must not all be wrapped as `LanguageModelConfig`.
- [ ] Keep the fast environment free of imports and dependencies on JAX, Lalamo, and PyTorch. Test fresh-process import isolation for every advertised fast adapter.
- [ ] Define and enforce the streaming transform-workspace budget per operation. The bound covers live transformation buffers; header JSON, tensor catalogs, metadata/tokenizer normalization, hashing, and interpreter overhead are separate. Record whole-process/tree RSS separately with its sampling limits.
- [ ] Mark Lalamo's transform-workspace/resource bound as unbounded or unverified: it materializes a JAX model and cannot inherit the streaming chunk limit. A request requiring a strict bound must reject fallback unless that backend has an independently enforceable, validated contract. A preferred workspace setting may permit fallback only with explicit resource status. A strict total-process memory ceiling is unsupported until a platform-specific enforcement and failure contract is implemented and validated; sampled RSS alone cannot enforce it.
- [ ] Separate capability probing from complete validation. An unsupported probe must not declare the source valid. The selected backend must perform its own complete supported-layout validation before publishing output.
- [ ] Preserve hard errors for malformed or contradictory inputs in a recognized supported layout, source mutations, missing required tensors, and unexpected tensors. Preserve fallback for valid representations outside fast coverage and explicit streaming refusal of fallback.
- [ ] Define a machine-readable, sanitized capability query and receipt fields; use stable categories rather than exception text or source values.
- [ ] Add an optional requirement for a known native-supported output target. Conversion-only results must remain usable and must not be advertised as runnable scoring models.

The coverage ledger records orthogonal dimensions: registry identity and revision, model kind/family/config variant, source container and sharding, source/target dtype, compressed layout, metadata profile, emitted weight spec, conversion backend/status, native loading, execution mode, and performance evidence. Use statuses such as `unverified`, `qualified`, `fallback-only`, `runtime-blocked`, and `access-or-dependency-blocked` for the applicable dimension. Do not collapse them into one `supported` boolean.

Public ledger entries contain public model identities, revisions, aggregate outcomes, and hashes. Private checkpoint identities, paths, tokenizer vocabularies, and templates stay out of reports. Do not serialize or log prompts, responses, token vectors, decision probabilities, credentials, or exception dumps in either public or private validation artifacts, receipts, benchmark output, or diagnostic captures. Compare prohibited execution values in memory and retain permitted aggregate outcomes and hashes only. Existing authorized reference inputs and required private checkpoint/model/tokenizer/config files are operational data, not permission to capture runtime content as logs. Keep permitted private evidence under ignored directories.

## Phase 0 — Reference inventory and qualification contract

Dependencies: existing prototype only. This phase fixes the coverage denominator and acceptance rules before adapter implementation.

- [ ] Export the pinned built-in `ModelRegistry` in the isolated Lalamo environment without loading checkpoints or executing inference. Record identities, original origin types, config adapters, expected source containers, external `FileSpec.repo` dependencies, and required optional dependency groups.
- [ ] Preserve origin semantics, including Hugging Face and NeMo origins; unknown origin types receive an explicit unverified/unsupported status.
- [ ] Generate a versioned declarative catalog for fast dispatch and metadata-profile selection. The main converter must not import the upstream Python registry to read it. Record exporter source, Lalamo revision, catalog hash, and attribution.
- [ ] Keep checkpoint-specific config, chat templates, tokenizer contents, and sampling settings in the resolved source. A catalog profile selects verified parser/reasoning defaults and source-file rules; it does not replace checkpoint metadata with guessed family defaults.
- [ ] Map every registry row to conversion coverage and Stark runtime status. Verify actual loader/kernel constraints for each emitted weight spec; an enum or config class is insufficient evidence.
- [ ] Audit current download filename patterns against all registry-declared primary files and containers. Resolve required files from specs so extension allowlists do not truncate coverage.
- [ ] Define per-adapter acceptance fixtures, exact comparison policy, native target where applicable, and benchmark criteria before implementing or timing that adapter.
- [ ] Document unsupported third-party plugins, gated access, optional dependencies, and runtime gaps as bounded statuses, not completed coverage.

Gate G0: an attributed inventory accounts for every pinned built-in entry, the catalog can be read without heavyweight framework imports, and independent review accepts the contracts, status semantics, and fixture selection. A row blocked by access/dependencies remains blocked rather than being counted as independently qualified.

## Phase 1 — Repository integration and source resolution

Dependencies: G0. Preserve the existing command and backend choices while making the shared infrastructure reusable.

- [ ] Refactor CLI dispatch to registered adapters and the shared contracts. Retain `--backend auto|streaming|lalamo`, BF16 default, and source-versus-runtime qualification.
- [ ] Preserve the pinned Lalamo writer behind the shared source/result/provenance/publication interface. Exercise its observable output and failure contracts independently; do not rewrite it into the streaming plan just to unify internal implementations.
- [ ] Replace the current flat `stark_convert/*.py` provenance glob with a recursive, explicitly inventoried manifest keyed by repository-relative path. Cover every adapter, metadata profile, generated catalog, schema, exporter, fallback resource, and configuration file affecting conversion; exclude caches/runtime artifacts and keep dependency lock hashes explicit. Test that nested adapter changes, profile/catalog changes, and file additions/removals change the hash and invalidate receipts/benchmark runs. Detect collisions or unaccounted conversion inputs rather than silently omitting them.
- [ ] Bind native qualification to the actual Stark source/build receipt, worker binary hash, selected features, Rust toolchain, and dependency locks. Preserve the Uzu upstream revision as a separate attribution field; it is not the revision of the compiled Stark worker.
- [ ] Resolve primary and external dependencies once per request to immutable revisions before conversion. A local checkpoint without a remote revision uses a content manifest; never invent a commit SHA.
- [ ] Add explicit source-manifest input/output for reproducible replay. Validate every manifest reference and required file, including external metadata, before using it.
- [ ] Implement offline replay with a supplied pinned manifest and all required cached/local files. Test missing primary, missing external, and mismatched-hash failures. Current remote-ID resolution and external-repository pin discovery require online access; merely caching files does not establish this future offline feature.
- [ ] Generalize file signatures/content hashes to all consumed metadata and weights on both backends. Include sidecar templates, generation files, archive-derived inputs, and external files; reject source changes before publication.
- [ ] Keep malformed data failures separate from unsupported-fast capability results. Validate entire recognized layouts before dtype fallback or publication.
- [ ] Preserve owner-only staging and evidence permissions, atomic no-replace publication, input/output separation, complete cleanup, interruption handling, and sanitized diagnostics. Test fallback subprocess interruption/termination and ensure it cannot continue writing after staging is removed.
- [ ] Add model-free converter tests, import checks, and synthetic reference checks to existing CI where practical. Keep model downloads, native model-dependent checks, and raw artifact uploads outside public CI.
- [ ] Maintain both dependency locks and their existing OSV coverage. Use `uv` and the pinned managed interpreter; use Rust 1.98.1 for scoring builds.
- [ ] Keep repository usage documentation and an entrypoint that delegates to the Python converter without making JAX part of the Rust engine or host application.

Gate G1: existing Qwen3 and quantized fallback regressions pass; offline manifest replay works without network; automatic backend selection and publication/error contracts pass adversarial synthetic tests. Nested adapter/profile/catalog source mutations and additions/removals must change comprehensive provenance and invalidate a run. Record a new Qwen3 baseline after refactoring with new source/lock hashes; preserve the prototype evidence.

## Phase 2 — Shared transforms and first dense families

Dependencies: G1 and the reviewed catalog/adapter contracts. Develop adapters and metadata parity in parallel.

- [ ] Extend typed operations only where a concrete adapter needs them: bounded copy, row fusion/splitting, tiled transpose/permutation, reshape with explicit byte semantics, and dtype conversion.
- [ ] Validate operation input/output sizes, axis/order semantics, shard boundaries, endian handling, alignment, and workspace accounting. Vary chunk/tile sizes and verify identical output. Never materialize a full tensor merely to fuse or permute it.
- [ ] Implement dtype conversions with the pinned reference's rounding, overflow, subnormal, signed-zero, infinity, and NaN behavior. Matching transforms must retain exact target-byte parity; numerical closeness is insufficient to promote a mismatching adapter.
- [ ] Preserve Qwen3's Q/K/V and up/gate order, F32 normalization scales, tied-head validation, full configuration defaults, and metadata-profile behavior through the refactor.
- [ ] Implement Qwen2/Qwen2.5 dense adapters and verified metadata profiles.
- [ ] Implement explicitly enumerated Llama 3.x dense config/RoPE variants. Unsupported variants stay on fallback; do not accept every Llama model from one checkpoint.
- [ ] Implement dense Mistral variants, including reviewed sliding-window semantics and metadata defaults.
- [ ] For every family, create synthetic fixtures for tied/untied heads, biases where applicable, normalization, projection layouts, variant config defaults, dtype cases, malformed inputs, and cross-shard projection fusion.
- [ ] Validate at least one small accessible public checkpoint and a second checkpoint or structural config variant per family. Record exact revisions, access constraints, metadata sources, and separate evidence for each tuple.
- [ ] Add independent root-config deserialization and native load-only checks for each proposed native-supported output class. Keep conversion-only cases marked separately.
- [ ] Reuse existing private scoring validation only where execution qualification is an explicit implementation deliverable. A readiness/EOF load smoke does not establish scoring or generation correctness.

Gate G2: each promoted family/variant has complete exact output parity, source provenance, bounded-memory evidence, independent review, and accurate runtime status. Existing Qwen3 regressions pass. Any unqualified variant remains fallback-only.

## Phase 3 — Additional dense variants

Dependencies: G1 integration plus the reviewed catalog, metadata contract, and specific shared operators needed by each adapter. Apply G2 acceptance per adapter tuple. These adapters may start when their own dependencies are ready; they need not wait for all Qwen2/Llama/Mistral qualifications.

- [ ] Add Gemma 2 and Gemma 3 text-decoder variants with explicit normalization, scaling, attention, and activation semantics. Treat multimodal components and newer Gemma versions as separate rows.
- [ ] Add verified Phi-3/Phi-4 variants, including fused projection layout and RoPE differences present in the pinned reference.
- [ ] Add SmolLM and Granite families, then prioritize remaining dense registry rows using measured demand and shared operations.
- [ ] Generalize verified metadata profiles, including external tokenizer/config/generation sources, without reintroducing framework imports or family-based guesses.
- [ ] Complete every applicable G2 acceptance item for each variant before automatic selection.

Gate G3: the ledger accounts for remaining dense rows as qualified, fallback-only, runtime-blocked, or access/dependency-blocked. Only reviewed and measured adapter variants are promoted.

## Phase 4 — Compressed weight formats

Dependencies: stable shared operators, metadata resolver, and an inspected converter-to-native weight-spec contract. Start with formats the current engine can actually load.

- [ ] Inventory packed layouts implemented by the pinned upstream loaders, independently of quantization config enums. Do not infer complete GPTQ/AWQ support from a type name.
- [ ] Implement MLX 4-bit and 8-bit adapters first where target kernels and embeddings support the exact group/layout/spec combinations. Use existing quantized Qwen3 fallback evidence as a regression case, not as streaming qualification.
- [ ] Add other integer-packed formats only after specifying bit order, packing divisor, scales, zero points, group axes, fusion order, and padding rules. Test group/row boundaries across chunks and shards.
- [ ] Keep reorder/repack of an existing quantized checkpoint separate from quantizing a floating-point checkpoint. Newly computed quantization is a distinct transformation and quality qualification task.
- [ ] Assess low-rank, hybrid, D4S4, and microfloat outputs individually. Specialized embedding paths or enums must not be treated as universal matrix/kernel support.
- [ ] Require exact packed payload/spec parity for deterministic equivalent conversion. If an upstream optimization is nondeterministic or a proposed transform intentionally differs, keep it fallback-only until a separately specified equivalence/quality protocol is reviewed; do not weaken the general exact comparator.
- [ ] Record dequantized numerical checks and private runtime checks as additional evidence, with metric/tolerance selection fixed before evaluation. Publish aggregates and hashes only.
- [ ] Repeat the G2 provenance, malformed-input, memory, performance, and native-status gates for each format/family combination.

Gate G4: promotion is per quantization scheme, bit width, group size, layout, model variant, and native target. No general compressed-format claim is made from a single passing checkpoint.

## Phase 5 — Specialized models and complete registry accounting

Dependencies: required operators and output contracts from earlier phases. These workstreams may overlap once their specific dependencies pass; they need not wait for unrelated dense or quantized families.

- [ ] Add hybrid/state-space adapters individually: DeltaNet/Qwen hybrids, Mamba/LLaMBA, short-convolution/LFM variants, and other audited rows. Verify tensor permutations, convolution/state parameters, architecture config, and actual native path separately.
- [ ] Add classifier conversion through a classifier root config and metadata contract. Validate load and classification separately from language scoring.
- [ ] Plan MoE conversion as its own adapter family. Valid conversion-only artifacts may be produced, but runnable MoE requires a separately scoped native expert/router/kernel implementation and validation.
- [ ] Classify audio/TTS and auxiliary/speculator models explicitly. Implement conversion where justified; optional dependencies and native execution support remain separate deliverables. Do not use playground audio playback as TTS-engine evidence.
- [ ] Account for every remaining registry tuple, including unsupported origins/containers, dependency/access constraints, and native blocks. Keep fallback available for all retained compatible reference routes.
- [ ] Publish a coverage summary whose denominator and qualification levels are reproducible from the pinned inventory. State the fraction independently streamed and the fraction delegated to fallback; do not count them together as independently implemented support.

Gate G5: every pinned built-in registry row has an evidence-backed status. Claim equivalent conversion reach only for the documented source/dependency/access scope. Claim independently implemented parity only for qualified streaming tuples. New upstream entries require a pin migration and new accounting.

## Evidence and promotion checklist

Apply this checklist to each adapter tuple, not only to each family name.

- [ ] Freeze reference revision, source/dependency manifests, requested dtype, converter/lock hashes, runtime target, and expected emitted specs.
- [ ] Pass all existing model-free regressions and the new tuple's positive/adversarial fixtures.
- [ ] Compare all output tensor names, shapes, dtypes, logical metadata/specs, and payload hashes against pinned Lalamo. Header order, padding, and whole-file hash differences are allowed only when semantic and payload equality is established.
- [ ] Compare complete root/nested config, token codec or classifier/TTS metadata, generation settings where applicable, and normalized tokenizer JSON. Missing defaults such as SiLU's `alpha` must fail the comparison.
- [ ] Independently validate output spans, total byte counts, source consumption/justified omission, and strict schema deserialization. Reject unexpected source/output data in the supported layout.
- [ ] Verify source and toolchain mutation detection, clean failure, sanitized interruption, no-replace publication, and dependency/import isolation.
- [ ] Verify the enforced streaming transform-workspace limit and report all other memory separately. Test that a strict resource request cannot silently route to an unbounded/unverified fallback; never substitute sampled RSS for a memory-enforcement contract.
- [ ] For a claimed native-supported tuple, construct the model and consume all tensors. Require the appropriate private execution tests before claiming scoring, classification, generation, or TTS correctness. Compare execution values in memory; prohibit raw prompt/response/token/probability/credential/exception capture even in private evidence, and retain aggregates/hashes only. Serialize complete decisions against one loaded model and preserve restart/cache clearing rules.
- [ ] Benchmark fresh processes with alternating order and isolated ignored output directories. Separate setup/downloads, conversion-phase time, complete CLI time, hashing, and validation. Run measured conversions serially without concurrent model tests or other agent benchmark jobs.
- [ ] Record warm/cold cache regime accurately, repeat counts/ranges, host/runtime versions, model size, workspace bounds, and sampled RSS caveats. Do not reuse the 0.6B prototype ratios for other tuples.
- [ ] Before measurement, choose and record the promotion objective, workspace limit, and permitted timing/memory regression. Publish all valid runs against that fixed criterion. A tuple that fails its performance objective stays explicit-streaming or fallback-only until selection policy is reviewed; correctness and performance results are separate.
- [ ] Delegate independent technical review of the final adapter and evidence. Promote only after findings are resolved or an explicit scoped disposition is recorded.
- [ ] Add a new validation entry and catalog status tied to exact hashes. Preserve all earlier source-bound evidence and attribution.

## Dependency-aware parallel execution

Use the environment's actual concurrency limit. At the current four-slot limit, the coordinator occupies one slot and at most three worker/reviewer agents run at once. Replace a completed worker with the reviewer; do not start a fifth active agent. Recompute this allocation if the environment changes.

| Work packet | Ownership | Prerequisite | Parallelism |
| --- | --- | --- | --- |
| Registry inventory/catalog | One worker owns inventory/export/schema fixtures. | Existing pinned fallback. | Runs with fallback/provenance investigation and validation-contract work. |
| Fallback/source manifests | One worker owns resolver and offline replay tests. | Reviewed source contract for implementation. | Runs with registry work and model-free harness development. |
| Shared planner/operators | Coordinator or one worker owns shared modules and dispatch. | G0 contract; G1 resolver for integration. | Adapter investigations/fixtures can proceed against reviewed interfaces. |
| Dense adapters | One worker per adapter and its fixtures. | Required shared operations and metadata contract. | Qwen2 and Llama can run together; Mistral or metadata work uses the next free slot. |
| Metadata profiles | One worker owns profile/catalog fragments and reference comparisons. | Registry/source contracts. | Runs with corresponding tensor adapters; shared files have one owner. |
| Independent review | Reviewer did not author the reviewed packet. | Reviewable draft or complete patch plus evidence. | Replaces a finished worker while other independent packets continue. |
| Model qualification/benchmarks | One designated validation owner. | Integrated candidate passes synthetic checks. | Model execution and timed runs are serial; read-only/docs work may continue outside measurement windows. |

- [ ] Assign file ownership and interface versions before dispatch. Workers edit separate adapter/profile/test files; the integration owner serializes shared registry, CLI, lockfile, and documentation changes.
- [ ] Merge/reconcile dependency-complete packets before model-dependent qualification. Preserve each worker's evidence and source revision; rerun only checks affected by integration.
- [ ] Dispatch new independent packets when slots free instead of imposing a sequential family-by-family staffing schedule.
- [ ] Give technical review a concrete patch, capability matrix, and evidence bundle. A review finding is a dependency for affected promotion, not a reason to stop unrelated work.
- [ ] Keep downloads/caches coordinated and source snapshots immutable. Use separate ignored staging/artifact directories; do not share mutable output paths or silently overwrite another run.

Suggested G0 preparation wave: the coordinator drafts contracts; worker A inventories the catalog, worker B audits fallback/source resolution and designs manifest/offline cases, and worker C designs independent synthetic/parity tests. Replace a completed worker with the independent G0 reviewer. After G0 passes, the G1 implementation wave assigns shared Qwen3 migration to the coordinator, catalog/dispatch integration to worker A, manifest/offline implementation to worker B, and contract regression tests to worker C. Replace completed workers with review as packets become ready. After G1, reuse slots for Qwen2, Llama, and metadata parity; slot availability and each adapter's operator dependencies determine when Mistral and additional dense adapters begin.

## Reference updates and later distribution

- [ ] For each Lalamo/dependency pin update, record the old and new pins, update locks/security inventory as necessary, regenerate catalog differences, and review added/removed/changed registry tuples.
- [ ] Requalify changed metadata defaults, tensor transformations, emitted specs, and fallback routing. Rerun a representative existing Qwen3 and MLX regression plus affected tuples; never silently refresh evidence or external revisions.
- [ ] Preserve upstream copyright/license notices and record every adapted source path/revision. Keep model redistribution outside the source repository.
- [ ] Consider a separately installable package only after the in-repository interfaces and coverage gates stabilize. Define packaging of fallback projects/locks/licenses, resource lookup independent of checkout paths, clean-install tests, runtime compatibility/versioning, and optional dependency installation.
- [ ] Treat public release/signing/hosting infrastructure as a later separate scoped implementation. Do not enable inherited upstream infrastructure as a shortcut.

The first implementation milestone is G1 plus qualified Qwen2, explicitly selected Llama 3.x, and dense Mistral variants under G2. This milestone is a priority target, not a global barrier for unrelated adapters whose own integration/operator/metadata prerequisites already pass. The longer-term target is complete pinned-registry accounting under G5 while retaining the fallback.

## Technical review and source pointers

The delegated review is recorded in [MODEL_CONVERSION_PLAN_REVIEW.md](MODEL_CONVERSION_PLAN_REVIEW.md). Implementation checkboxes remain unchecked until their own acceptance evidence exists.

Current repository sources: [CLI orchestration](../tools/converter/stark_convert/cli.py), [streaming plan/writer](../tools/converter/stark_convert/streaming.py), [Qwen3 adapter](../tools/converter/stark_convert/qwen3.py), [metadata](../tools/converter/stark_convert/metadata.py), [fallback resolver](../tools/converter/stark_convert/lalamo_runner.py), [independent comparison/benchmark](../tools/converter/stark_convert/benchmark.py), [converter usage](../tools/converter/README.md), [source-bound validation](MODEL_CONVERSION.md), [native language loading](../crates/stark-engine/src/engine/language_model/mod.rs), and [MoE rejection](../crates/stark-engine/src/encodable_block/mlp/mod.rs).

Pinned reference sources: [Lalamo model specifications](https://github.com/trymirai/lalamo/blob/831accb8252a6b6606c378048caeb44178f1a050/lalamo/model_import/model_spec.py), [Hugging Face transformations](https://github.com/trymirai/lalamo/blob/831accb8252a6b6606c378048caeb44178f1a050/lalamo/model_import/loaders/huggingface.py), [origin/source resolution](https://github.com/trymirai/lalamo/blob/831accb8252a6b6606c378048caeb44178f1a050/lalamo/model_import/origins.py), and [upstream model test tiers](https://github.com/trymirai/lalamo/blob/831accb8252a6b6606c378048caeb44178f1a050/tests/model_test_tiers.toml). These are reference implementations/test inventories, not blanket Stark qualification.
