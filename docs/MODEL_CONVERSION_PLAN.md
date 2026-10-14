# Independent model conversion replacement plan

Current priority: [tool hardening](MODEL_CONVERSION_HARDENING.md). Further conversion coverage is deferred at the owner's direction; existing milestone evidence remains preserved.

Updated: 2026-10-12. Milestone 1's G0/G1/selected G2, milestone 2's scoped G3 and the first Qwen3 MLX packet remain preserved. Milestone 3 includes independent MLX Llama/SmolLM3/Gemma text conversion and six additional exact BF16 load-only qualifications. The new [latency packet](MODEL_CONVERSION_MILESTONE_3_LATENCY.md) reduces six complete-command medians by 16–38% against the frozen prior implementation; five beat Lalamo and SmolLM3 8-bit remains slower. The [dense continuation](MODEL_CONVERSION_MILESTONE_3_DENSE.md) and [original review](MODEL_CONVERSION_MILESTONE_3_DENSE_REVIEW.md) retain their separate source bindings and four failed latency hypotheses. Full G4 and broad latency parity remain open.

[Milestone 6](MODEL_CONVERSION_MILESTONE_6.md) adds ModernBERT classifier roots, validated labels/tokenizers and generic classifier conversion. Both pinned Mirai routers are inaccessible to the current account and remain unqualified; public real-weight fixtures with explicit synthetic codec overlays receive separate validation. The existing twenty-seven native ledger records are unchanged. [Milestone 5](MODEL_CONVERSION_MILESTONE_5.md) retains its dense and 4-bit MLX LLaMBA qualifications and open dense latency objective. [Milestone 4](MODEL_CONVERSION_MILESTONE_4.md) adds dense Qwen3.5 text/DeltaNet and LFM2 short-convolution/attention, plus explicit architecture dispatch for self-contained future checkpoints. Its [MLX continuation](MODEL_CONVERSION_MILESTONE_4_MLX.md) adds eighteen 4/8-bit profiles and four exact group-64 BF16 load-only qualifications. Prior benchmark and qualification packets retain their original source bindings.

Keep `stark-convert` in this repository as a first-class companion to the Stark engine. The user-directed target is to supplant Lalamo with independently implemented conversion and remove the runtime fallback. Lalamo remains an attributed, pinned reference for registry inventory, format semantics, and externally generated qualification artifacts. The installed converter must neither import nor invoke Lalamo, JAX, or PyTorch.

This policy supersedes the permanent-fallback policy in the plan committed at `e51ed763`. The [earlier plan review](MODEL_CONVERSION_PLAN_REVIEW.md) reviewed that historical policy and source state; its acceptance does not approve this replacement policy or the current implementation. The [prototype evidence](MODEL_CONVERSION.md) retains its original source/lock hashes and results. New code and tooling receive separate source-bound evidence.

## Scope and coverage denominator

The target is the **110 built-in registry entries** at Lalamo revision `831accb8252a6b6606c378048caeb44178f1a050`. The [catalog](../tools/converter/stark_convert/catalog.json) also records **four explicit Stark extensions** separately: `unsloth/Llama-3.2-1B-Instruct`, `mistralai/Mistral-7B-Instruct-{v0.2,v0.3}`, and `google/gemma-2-2b-it`. Gemma2 uses the pinned config/loader with a standard LanguageModelSpec; the registry has no Gemma2 built-in. Extensions do not alter the built-in denominator or imply upstream registration.

The current implementation covers **85 built-in profiles and four extensions**, including 43 dense language profiles for Qwen, Llama, LLaMBA, Mistral, Gemma3 text, Phi-4-mini, Granite, SmolLM and LFM2, 40 MLX profiles for the supported language families, and two dense ModernBERT classifier profiles. Gemma2 remains separately counted as a Stark extension. Twenty-seven exact BF16 checkpoint/source/artifact tuples have named-build load-only qualifications. The remaining 25 built-ins need independent adapters; 62 implemented identities remain unqualified, including both inaccessible Mirai classifier rows. Qualification remains per exact source revision, config variant, dtype, metadata source set and root model kind. The pinned replacement denominator excludes arbitrary Hugging Face repositories, third-party Lalamo plugins, and a moving upstream branch. Public classifier benchmark fixtures are outside that denominator and do not expand the four-extension count.

Reusable adapters are tracked by architecture and input format. The catalog tracks checkpoint-specific metadata/source rules and implementation status; the native ledger tracks exact source/artifact tuples and named builds. An explicit `hf-generic` profile can convert a future repository with a supported architecture and self-contained metadata without adding a catalog entry. It retains strict config/layout/source validation, plain metadata defaults and an unqualified native status. It cannot override a known catalog profile or expand the pinned denominator. See [future-checkpoint usage](../tools/converter/README.md#future-checkpoints-and-architecture-tracking).

Conversion, native loading, execution, and performance are separate dimensions. Current Stark dense loading selects BF16; F16/F32 conversion-only outputs remain runtime-blocked. MoE construction is currently rejected. Existing kernel/config enums do not establish acceptance for every model using them. Language scoring, generation, classification, and TTS each require the corresponding execution evidence.

Host application integration belongs in separate repositories. Package publication, model redistribution, signing, binary distribution, release automation, and hosting are separate scoped work. Keep inherited release/cloud infrastructure disabled.

The separate [ColGrep generation work](GENERATION.md) adds a native worker and controlled functional execution evidence for one generic Llama checkpoint. It preserves the converter coverage denominator and catalog load-only ledger. Exact generation token parity, stochastic backend equivalence, broad context quality and live app integration retain their separate open status.

## Contracts and status semantics

| Implemented contract | Required meaning |
| --- | --- |
| `SourceSnapshot` | Selected primary/external files, immutable remote revisions or local content identity, hashes/signatures, and private manifest replay. |
| `ConversionRequest` | Registered identity, model kind, output dtype, independent backend policy, transformation workspace, and requested native-support policy. |
| `CapabilityResult` | Implemented/unsupported adapter, source-validation state, native-loading status, execution status, and sanitized reason category. |
| `PlannedConversion` / `ConversionPlan` | Complete accepted model config, tensor names/shapes/dtypes/specs, ordered typed operations, source consumption/justified omission, signatures, and workspace-bound writing. |
| `ConversionResult` | Independent backend, artifact/source hashes, aggregate timings, tensor/byte counts, transformation workspace and separate RSS. |

- [x] Define shared contracts and registered adapter dispatch without heavyweight model-framework imports.
- [x] Move Qwen3 through shared dispatch while preserving its accepted layouts and metadata profiles.
- [x] Make model kind explicit; refuse unsupported classifier/TTS requests.
- [x] Separate catalog probing from complete accepted-layout validation. A capability probe reports `source_validation=not_performed`.
- [x] Preserve hard failures for malformed/contradictory accepted layouts, missing/unexpected tensors, source mutations, and unsupported representations.
- [x] Enforce the transformation-workspace bound and record whole-process RSS separately.
- [x] Implement exact stored source/artifact qualification selection for `--require-native-support`, bound to the named native build's source/binary/lock hashes, features, and Rust receipt. Catalog probes remain unverified; arbitrary future/current binaries receive no implied qualification.
- [ ] Implement a strict total-process memory ceiling only with a validated platform-specific enforcement/failure contract.

The workspace bound covers live transformation buffers. Header JSON, tensor catalogs, metadata/tokenizer normalization, hashing, and interpreter overhead are additional costs. RSS sampling reports observations and cannot enforce a memory ceiling.

Catalog implementation status, conversion qualification, access, native loading, execution, and performance remain independent. A catalogued adapter does not mean that every checkpoint/variant has passed exact reference comparison or native construction. Native-qualified selection matches registered identity, BF16 dtype, source-manifest hash, weights/config/tokenizer hashes, and returns the complete recorded build receipt; it qualifies that named build's load-only result. Public ledgers contain public identities/revisions, aggregate outcomes, and hashes. Private source paths, tokenizer vocabularies, and templates stay out of public evidence.

Never capture prompts, responses, token vectors, decision probabilities, credentials, or exception dumps in logs, receipts, benchmark reports, or private validation artifacts. Existing operational checkpoint/config/tokenizer files remain private model inputs. Compare execution values in memory and retain permitted aggregates/hashes only.

## Model artifact cleanup

At the user's direction, remove downloaded conversion-only weights and generated model outputs after a packet's validation, benchmark and review are complete. Keep the small source manifests, receipts, aggregate measurements, audit records, scripts and pinned revision/content hashes. Check for active use before deletion; preserve serving models, externally supplied source models and unrelated caches. Record cleanup separately from the original validation evidence, including what was removed and the measured storage change.

Artifact retention statements describe the validation run's state. A later cleanup does not alter its original hashes or results. Future checks that need deleted weights must reacquire the pinned inputs and regenerate outputs, then verify the recorded hashes; existing manifests alone cannot replay deleted files.

- [ ] For each completed future conversion packet, remove its unused model weights and outputs, retain small evidence, and record cleanup before reporting completion.

## Phase 0 — Pinned reference inventory and acceptance contract

Dependencies: the existing prototype and pinned reference source.

- [x] Export every built-in reference entry without loading model checkpoints or executing inference.
- [x] Record original origins, model/config classes, expected containers, metadata source rules, public identities, and source attribution/hashes.
- [x] Store a versioned declarative catalog readable without importing the upstream registry.
- [x] Keep Stark extensions separate from the 110-entry built-in denominator.
- [x] Preserve checkpoint-defined tokenizer, template, special-token, and generation settings; profiles provide explicit upstream parser/reasoning/role/source semantics.
- [x] Retain unsupported statuses for specialized origins, containers, compressed formats, metadata semantics, and runtime gaps.
- [x] Complete independent review of the replacement contracts, source resolver, inventory completeness, and fixture selection.
- [ ] Fully inventory optional reference-generation dependencies and access constraints for later rows before scheduling their qualifications.

Gate G0: every built-in row is accounted for, catalog provenance is reproducible, imports remain isolated, and independent review accepts the contracts and status semantics. Access or dependency blocks remain visible.

## Phase 1 — Repository integration and independent source resolution

Dependencies: G0's reviewed contracts and catalog; implementation packets can proceed against those interfaces while their review is completed.

- [x] Dispatch through independent registered adapters. Retain `--backend auto|streaming` and BF16 default.
- [x] Remove runtime Lalamo execution, runner, dependency environment/lock, and fallback tests/routing.
- [x] Preserve upstream license/copyright and pinned reference provenance.
- [x] Hash recursive conversion-affecting package inputs, catalog/schema resources, exporter, project configuration, and license; record the converter lock separately.
- [x] Test nested changes, additions/removals, catalog/profile changes, and unexpected package resources against provenance accounting.
- [x] Resolve primary and external metadata repositories once to immutable commits before downloading selected files.
- [x] Accept local sources with content identity when no remote commit is available.
- [x] Write private source manifests and replay complete manifests offline without network discovery.
- [x] Validate manifest identity, paths, required primary/external files, sizes, signatures, and hashes; reject incomplete or changed sources before publication.
- [x] Preserve owner-only staging, no-replace atomic publication, cleanup/interruption behavior, input/output separation, and sanitized CLI errors.
- [x] Add model-free import, dispatch, metadata, manifest, source-mutation, provenance, and publication regressions.
- [x] Provide native source/build/binary/lock provenance for separate loader checks.
- [x] Complete the integrated model-free suite, formatting/security/repository checks, and independent review against the final patch.
- [x] Record a new source-bound Qwen3 regression baseline with current converter/lock hashes.
- [x] Wire model-free converter checks into existing CI without enabling release/signing/cloud infrastructure; hosted execution remains unverified for this patch.

Gate G1: independent Qwen3 conversion and explicit unsupported quantized routing pass; offline replay and adversarial publication/error/provenance checks pass; the final patch is reviewed. Existing quantized prototype evidence remains historical. Current quantized inputs must refuse conversion until an independent implementation exists.

## Phase 2 — Shared transforms and first dense families

Dependencies: shared contract/source infrastructure and the concrete operator/metadata requirements for each adapter. Develop independent adapter packets in parallel.

- [x] Reuse bounded copy/cast, projection fusion, tiled transpose, and normalization widening through one tensor/config plan.
- [x] Test new dense-adapter BF16/F16/F32 casts against frozen independent reference hashes for rounding, overflow, subnormal, signed-zero, infinity, and NaN behavior.
- [x] Implement Qwen2 dense conversion for the six registered Qwen2.5-Coder Instruct profiles.
- [x] Implement selected Llama 3.1/3.2 dense/RoPE configurations and explicit metadata profiles.
- [x] Implement dense Codestral/Devstral profiles and Mistral v0.2/v0.3 extensions with reviewed sliding-window/config semantics.
- [x] Preserve Qwen3 Q/K/V and up/gate ordering, F32 normalization, tied-head checks, and all eight metadata defaults.
- [x] Implement external metadata resolution, JSON-field/text templates, registered role/reasoning defaults, and generation configuration/override semantics.
- [x] Add synthetic positive/adversarial fixtures for projections, norms, tied/untied heads, biases where applicable, config/RoPE variants, dtype boundaries, and cross-shard fusion.
- [x] Complete exact tensor/spec/config/tokenizer comparison against independently generated pinned reference artifacts for the four promoted checkpoint tuples and six structural variants.
- [x] Qualify a small accessible public checkpoint and a second checkpoint or structural config variant per family; record explicit extension provenance and access blocks.
- [x] Record bounded-memory evidence and source-bound benchmark objectives/results for the selected tuples, including the failed raw latency hypothesis.
- [x] Independently deserialize emitted root configs and perform native BF16 load-only checks for the four exact native-supported tuples.
- [x] Delegate review of the integrated dense adapters and metadata evidence; resolve every promotion-blocking finding.
- [ ] Improve conversion latency against the external reference while preserving complete source verification and exact artifact parity.

Gate G2: every promoted tuple has exact output parity, pinned source/metadata provenance, bounded transformation memory, appropriate runtime status, and accepted technical review. Unqualified implemented profiles remain visibly unqualified. Unsupported variants fail explicitly.

The **first milestone is G1 plus selected Qwen2/Qwen2.5-Coder, Llama 3.x, and dense Mistral variants under G2**, preserving the original milestone scope with the user's replacement policy. Structural config variants can satisfy the second-variant requirement when their complete output/reference evidence is recorded. Implementation checkboxes alone do not close G1/G2; the milestone evidence identifies the exact accepted gates and remaining work.

## Phase 3 — Additional dense variants

Dependencies: reviewed integration, metadata/source contracts, and the shared operators required by the particular adapter. Work can begin when those dependencies pass.

- [x] Add Gemma 2/3 text decoder variants with explicit normalization, scaling, attention, activation, and metadata semantics; keep multimodal/newer versions separate.
- [x] Add the pinned tied Phi-4-mini/Phi3 layout, fused projection reorder, and partial LongRoPE. Untied Phi3 remains unsupported because the pinned config/loader does not implement it.
- [x] Add SmolLM2/3, Granite, and remaining uncompressed Qwen/Llama inventory profiles using their explicit operator/metadata contracts. Unmeasured profiles remain implemented-unqualified.
- [x] Implement the added dense rows' Granite parser/reasoning field, reasoning boundaries, and reviewed static inline template/system defaults with role/hash binding. Specialized unknown profiles remain unsupported.
- [x] Extend independently implemented BF16/F16/F32 dtype conversion to Qwen3; all nine source/target pairs have complete external synthetic artifact parity and frozen IEEE edge regressions.
- [x] Apply G2 acceptance items to the seven newly qualified real BF16 tuples and 26 synthetic variants; retain load-only and conversion-only status boundaries.
- [x] Implement bounded parallel hashing of independent large shards while preserving complete initial/final content verification and before/after signatures/selection checks.
- [ ] Close the complete-command latency objective against Lalamo for each claimed tuple; milestone 2 records measured outcomes without granting a general speed qualification.

Gate G3: remaining dense rows have reproducible qualified, implemented-unqualified, unsupported, runtime-blocked, or access-blocked statuses. Family names never substitute for tuple evidence.

## Phase 4 — Compressed weight formats

Dependencies: inspected native matrix/kernel/embedding contracts, source metadata, and stable shared operators.

- [x] Inspect the actual MLX loader/exporter, packed bit order, source planes, padding and native matrix/kernel/embedding contracts for the initial Qwen3 packet.
- [x] Independently implement the 12 registered MLX Qwen3 4-bit/8-bit profiles with affine groups 16/32/64/128, both accepted physical layouts, dtype casts and cross-shard fusion; preserve the prototype MLX result as historical evidence.
- [x] Qualify the exact 0.6B 4-bit and 8-bit group-128 BF16 tuples with full reference parity, named-build load-only evidence and source-bound performance; keep other profiles/groups unqualified.
- [x] Freeze 288 external packed/spec/dequantized synthetic cases, exceptional coefficient casts and 12 complete synthetic artifact variants; enforce workspace and malformed-input/publication/source gates.
- [x] Inventory remaining packed schemes from actual reference loaders rather than quantization enums; preserve the [baseline format report](MODEL_CONVERSION_FORMAT_INVENTORY.md) and its source binding. GPTQ enum presence alone does not prove a complete input loader.
- [x] Extend the unchanged MLX helper to five Llama, two SmolLM3 and two Gemma3 text profiles; qualify the selected six group-64 BF16 tuples with exact artifact parity, named-build load-only and source-bound benchmark evidence. Keep other variants unqualified.
- [x] Freeze additional actual-loader decoder exports and complete synthetic CLI metadata/artifact comparisons; preserve family RoPE/norm/embedding semantics and reject mixed/malformed representations.
- [x] Record explicit canonical/transport provenance for Llama's exact public metadata copy, authenticated against the official pinned Git blob; keep direct gated resolution and automatic substitution policies unchanged.
- [x] Profile and optimize MLX coefficient export with sequential output bands, bounded complete-row reads and inline hashing; retain exact artifact bytes, full source verification and the 8 MiB workspace limit. Independently review and benchmark six tuples against the frozen prior implementation and Lalamo.
- [x] Extend affine MLX 4/8-bit conversion to ten dense Qwen3.5 text and eight LFM2 built-ins; qualify the selected four group-64 BF16 tuples with exact pinned full-export parity and named-build load-only evidence. Preserve earlier qualification records and benchmark bindings.
- [ ] Deferred by user choice: close the remaining SmolLM3-3B 8-bit complete-command latency objective and qualify broader latency claims with additional sources/hosts. This does not block the next architecture packet.
- [ ] Extend independent compressed conversion to remaining families/schemes and qualify each additionally claimed tuple.
- [ ] Define bit order, packing divisor, group axis, scales/zero points, padding, fusion, and cross-chunk/shard boundaries for every additional scheme.
- [ ] Keep repacking an existing compressed checkpoint separate from computing new quantization and its quality acceptance.
- [ ] Assess low-rank/hybrid/D4S4/microfloat outputs against their exact native targets.
- [ ] Require deterministic packed payload/spec parity. Intentionally changed or nondeterministic transforms need separately reviewed equivalence/quality protocols.
- [ ] Add dequantized numerical checks and permitted private runtime evidence using metrics/tolerances fixed before evaluation.
- [ ] Apply G2 source/provenance, malformed-input, memory, performance, and native-status gates for every format/family tuple.

Gate G4: qualification is per scheme, bit width, group size, layout, model variant, and native target. Unsupported formats have no delegated execution route.

The **third milestone comprises the initial Qwen3 MLX packet, the Llama/SmolLM3/Gemma text continuation and the sequential coefficient latency packet under scoped G4**. Its exact tuple conversion/load-only/workspace/memory acceptance does not close full Phase 4. The preserved dense continuation records two passing and four failing raw latency medians. The new [five-repeat three-backend comparison](MODEL_CONVERSION_MILESTONE_3_LATENCY.md) improves all six medians against the frozen implementation and records five passing raw Lalamo medians; SmolLM3 8-bit remains slower, and Llama/Gemma 8-bit passing ranges overlap. Specialized/multimodal compressed architectures, other schemes, unmeasured variants and execution quality retain separate gates. The earlier source-bound qualification boundaries remain unchanged.

## Phase 5 — Specialized models and complete replacement accounting

Dependencies: the operators and root-output contracts needed by each row. Independent packets can overlap when those requirements pass.

- [x] Implement dense Qwen3.5 text/DeltaNet and LFM2 attention/short convolution with exact pinned reference exports; preserve unsupported MoE and later-generation variants. The subsequent MLX packet has separate format/tuple evidence.
- [x] Qualify the exact Qwen3.5-0.8B and LFM2-350M BF16 tuples with full tensor/spec/config/tokenizer parity and named-build native readiness/EOF, with zero scoring calls. Preserve the prior nineteen records.
- [x] Add explicit `hf-generic` architecture dispatch, self-contained checkpoint metadata and schema-2 offline replay for future supported checkpoints; keep catalog profiles authoritative and qualification per tuple.
- [x] Run the preregistered twenty-command dense architecture benchmark against live pinned Lalamo; retain every observation and exact qualification. LFM2's raw median latency hypothesis passes; Qwen3.5's fails. Both memory hypotheses pass, with broad overlapping timing ranges.
- [ ] Improve Qwen3.5-0.8B complete-command conversion latency and qualify any improvement against preserved source-bound measurements.
- [x] Extend MLX conversion to Qwen3.5 text and LFM2 and separately qualify four exact bit/group/config/checkpoint tuples; see the [continuation evidence](MODEL_CONVERSION_MILESTONE_4_MLX.md).
- [x] Benchmark the four selected MLX tuples against live pinned Lalamo with forty serial fresh commands. All parity/source/native/workspace/observed-memory gates pass; measured median times are 28–71% lower and sampled RSS 92–95% lower. Preserve all observations and the separate dense latency failure.
- [x] Implement the two pinned LLaMBA dense/4-bit MLX rows and their reusable Mamba2 helper, with separately qualified exact BF16 artifacts, external-config source/replay rules and scoped future-checkpoint support; see [milestone 5](MODEL_CONVERSION_MILESTONE_5.md).
- [x] Benchmark both exact LLaMBA tuples against live pinned Lalamo with twenty serial fresh commands; all parity/native/source/workspace/observed-memory gates pass. MLX raw median passes with overlapping ranges; dense raw median fails. Preserve every observation.
- [ ] Profile and improve dense F32-source LLaMBA complete-command latency against the frozen milestone 5 implementation/reference while preserving full verification and exact artifacts.
- [ ] Inspect other remaining state-space families individually; the pinned inventory has no separate Mamba1 row and this packet grants no Mamba1 conversion claim.
- [x] Implement ModernBERT classifier roots, source/metadata selection, strict label/tokenizer validation and model-kind-bound generic replay, with actual pinned synthetic full-classifier exports; see [milestone 6](MODEL_CONVERSION_MILESTONE_6.md).
- [x] Validate two separate public real-weight classifier fixtures with explicit synthetic codec overlays against complete pinned reference exports and a separately bound native classifier load-only harness. Preserve generic receipt status and the original twenty-seven ledger records.
- [ ] Obtain access to the two pinned Mirai classifier repositories, then qualify their exact source/artifact/build tuples separately; private/unavailable repositories are not qualified by public fixtures.
- [ ] Validate classifier execution correctness and quality with a separately scoped protocol; construction/readiness/EOF proves loading only.
- [ ] Implement conversion-only MoE artifacts as a distinct adapter; native expert/router/kernel work requires a separately scoped implementation.
- [ ] Account for audio/TTS, auxiliary/speculator models, NeMo/archive origins, and their dependency/access requirements; implement each independently where in scope.
- [ ] Complete every remaining pinned row's source, config, tensor, metadata, and failure semantics.
- [ ] Publish an evidence-backed coverage summary using the reproducible 110-entry built-in denominator and separate extension counts.
- [ ] Close every unsupported row or record an explicit bounded access/dependency/runtime disposition; do not call blocked rows independently qualified.

Gate G5: all in-scope pinned conversion rows have independently implemented and qualified parity within documented source/access/dependency limits. Runtime and execution blocks remain separate. Complete registry accounting is a prerequisite for a claim of replacing Lalamo's conversion reach.

The **fourth milestone's dense architecture packet** adds five Qwen3.5 and four LFM2 built-ins. Only the selected 0.8B/350M BF16 tuples gain exact named-build load-only qualification. Qwen3.5-35B-A3B is MoE and remains planned despite sharing the pinned reference config class. The [dense evidence](MODEL_CONVERSION_MILESTONE_4.md) records bounded workspace, fresh external benchmarks, exact hashes and [independent review](MODEL_CONVERSION_MILESTONE_4_REVIEW.md). Its [MLX continuation](MODEL_CONVERSION_MILESTONE_4_MLX.md) adds eighteen profiles and four exact group-64 BF16 qualifications with separate source bindings. [Milestone 5](MODEL_CONVERSION_MILESTONE_5.md) adds the two pinned LLaMBA rows and their Mamba2 mixer with exact independent qualifications and external-config replay support. Generic conversion reuses supported adapters without granting checkpoint-wide or architecture-wide native qualification. Full G4/G5, classifier/MoE/specialized rows, later state-space variants and complete replacement remain open.

## Per-tuple qualification and promotion checklist

- [ ] Freeze reference revision, source/metadata manifests, requested dtype, config variant, converter/lock hashes, runtime target, and expected specs before validation.
- [ ] Pass existing regressions plus independent positive/adversarial fixtures; validate full source consumption and justified omissions.
- [ ] Compare every output name, shape, dtype, payload hash, logical metadata/spec, root/nested config field, generation/token-codec settings, and normalized tokenizer against externally generated reference output. Missing defaults fail parity.
- [ ] Independently check safetensors spans/total bytes and strict root-config deserialization. Different serialization order/padding may explain differing whole-file hashes only when semantic and payload equality passes.
- [ ] Verify source/toolchain mutation detection, private cleanup, sanitized interruption/errors, publication races, manifest replay, and framework-import isolation.
- [ ] Verify the transformation workspace bound separately from whole-process/tree RSS. Fixed scratch allocation does not qualify total-memory enforcement.
- [ ] For claimed native support, construct the exact output with a source-bound Stark build and consume all tensors. Record Rust 1.98.1, selected features, source/lock receipts, and worker hash. `--require-native-support` must match the exact stored source/artifact qualification and emit the complete `native_qualification` build receipt; it cannot confer support on other binaries or variants.
- [ ] Before claiming scoring/generation/classification/TTS, complete corresponding private execution checks. Keep prohibited execution content out of saved output.
- [ ] Choose and record performance objectives, repeat count, workspace limit, cache regime, and permitted regressions before timed runs.
- [ ] Run fresh streaming processes serially with other model-dependent work paused; report setup/download exclusion, complete CLI/conversion-phase time, host versions, ranges, and RSS sampling limits.
- [ ] Compare frozen independent reference artifacts outside measured time. Any performance ratio requires separately recorded live external reference measurements; the current benchmark does not generate such a ratio.
- [ ] Obtain independent review of final code/evidence and resolve or explicitly dispose of findings before promoting affected tuples.
- [ ] Record new source-bound outcomes/hashes in the milestone/coverage ledger. Preserve prior source-bound results and upstream attribution.

## Dependency-aware parallel execution

At the current four-slot limit, the coordinator occupies one slot and up to three workers/reviewers operate concurrently. Replace completed workers with review or the next dependency-ready packet. Recompute allocation when the environment's limit changes.

| Work packet | Owner | Prerequisite | Parallel work |
| --- | --- | --- | --- |
| Catalog and source manifests | One worker owns inventory/schema/resolver/replay tests. | Pinned reference and source contract. | Adapter implementation and metadata fixtures. |
| Shared planner and dense adapters | One owner per shared module or adapter packet. | Required reviewed operations/config contract. | Independent family adapters. |
| Metadata profiles | One worker owns codec/generation/source-rule semantics and tests. | Catalog/source interface. | Tensor adapter packets. |
| Documentation and evidence | One owner records policy, statuses, and source-bound outcomes. | Reviewable integrated interfaces/evidence. | Read-only review and untimed checks. |
| Independent review | Reviewer did not author the reviewed packet. | Concrete diff and evidence. | Dependency-ready unrelated packets. |
| Model qualification/benchmarks | One designated validation owner. | Integrated synthetic checks pass. | Read-only/docs work outside measurement windows. |

- [x] Assign disjoint file ownership and stable interfaces before implementation dispatch.
- [x] Use parallel catalog/source, dense-adapter, and metadata workstreams within the actual concurrency limit.
- [x] Reconcile completed packets and run affected integration checks before model qualification.
- [x] Replace completed workers with independent technical review and record final dispositions.
- [x] Coordinate downloads/caches, immutable snapshots, and separate ignored output paths.

## Reference updates and later distribution

- [ ] For a reference update, record both pins, regenerate catalog differences, and review new/removed/changed tuples.
- [ ] Requalify changed metadata, transforms, specs, source rules, and affected runtime classes without silently refreshing earlier evidence.
- [ ] Retain upstream copyright/license notices and exact adapted source paths/revisions.
- [x] Implement checkout-independent local wheel/sdist packaging with source identity, explicit licensing and clean-install checks; see the [hardening packet](MODEL_CONVERSION_HARDENING.md).
- [ ] Specify engine compatibility/versioning policy before eventual package publication.
- [ ] Treat package publication, signing, releases, and hosting as separate scoped work; do not enable inherited infrastructure.

Implementation sources: [contracts](../tools/converter/stark_convert/contracts.py), [catalog dispatch](../tools/converter/stark_convert/registry.py), [source manifests](../tools/converter/stark_convert/sources.py), [CLI](../tools/converter/stark_convert/cli.py), [streaming writer](../tools/converter/stark_convert/streaming.py), [dense adapters](../tools/converter/stark_convert/adapters/dense.py), [Qwen3](../tools/converter/stark_convert/qwen3.py), [metadata](../tools/converter/stark_convert/metadata.py), [provenance](../tools/converter/stark_convert/provenance.py), [comparison/benchmark](../tools/converter/stark_convert/benchmark.py), and [usage](../tools/converter/README.md).

Pinned reference sources: [model specifications](https://github.com/trymirai/lalamo/blob/831accb8252a6b6606c378048caeb44178f1a050/lalamo/model_import/model_spec.py), [Hugging Face transformations](https://github.com/trymirai/lalamo/blob/831accb8252a6b6606c378048caeb44178f1a050/lalamo/model_import/loaders/huggingface.py), and [origin resolution](https://github.com/trymirai/lalamo/blob/831accb8252a6b6606c378048caeb44178f1a050/lalamo/model_import/origins.py). These sources define reference behavior; they do not qualify Stark tuples.
