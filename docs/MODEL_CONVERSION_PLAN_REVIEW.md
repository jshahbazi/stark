# Model conversion plan technical review

Historical review of the plan at `e51ed763`. Its permanent-fallback policy was superseded by the user-directed [replacement plan](MODEL_CONVERSION_PLAN.md); this review does not approve the current implementation. See the separate [milestone review](MODEL_CONVERSION_MILESTONE_1_REVIEW.md).

Review date: 2026-10-11. Status: accepted as an implementation plan after revision. No unresolved planning blockers were identified in the final document.

## Reviewed artifact and scope

| Item | Identity |
| --- | --- |
| Plan | [MODEL_CONVERSION_PLAN.md](MODEL_CONVERSION_PLAN.md) |
| Final plan SHA256 | `c8dda9134159b5f191031b887f8e7810a8aff6c430560bd6e268ea0892d894d5` |
| Repository source inspected before the documentation commit | `77e9a425ae3d8f0d140911adfb6288e7795e7fd8` |
| Pinned Lalamo source inspected | `831accb8252a6b6606c378048caeb44178f1a050` |
| Review method | Independent static technical review, with delegated read-only source inspection and an independent plan-hash check |

The reviewer did not author the plan. Review covered repository integration, backend contracts, capability accounting, phase dependencies, the four-slot agent allocation, exact output comparison, resource policy, privacy, provenance, and promotion gates. Source inspection included the existing converter, native loader constraints, pinned upstream import paths, repository instructions, and recorded prototype evidence.

This is a planning-only review. It does not certify expanded conversion coverage, native execution, inference quality, or performance. No models were loaded, no model-dependent tests or benchmarks were run, and no checkpoints or environments were downloaded during review. Runtime and converter code were not changed. Historical results were checked against their published source-bound record; those experiments were not repeated.

Acceptance is bound to the plan hash above. Changes to the plan's contracts, dependencies, validation policy, or promotion criteria require review of the changed requirements. Implementation checkboxes and G0–G5 remain pending until their own evidence exists.

## Findings and dispositions

All findings below were resolved by explicit planning requirements. The corresponding runtime changes remain future implementation work.

| Finding | Severity | Final disposition and plan reference |
| --- | --- | --- |
| Private execution evidence could be interpreted as permission to retain prohibited raw values. | P1 | Resolved. The privacy paragraph at line 51 prohibits logging or serialization of prompts, responses, token vectors, decision probabilities, credentials, and exception dumps in public and private artifacts. The execution-promotion item at line 157 requires in-memory comparison and permitted aggregates/hashes. Existing authorized reference inputs remain operational inputs. |
| The first implementation wave started G1 migration and offline implementation while G0 inventory and contract review were incomplete. | P2 | Resolved. Phase 1 retains G0 as its prerequisite at line 70. The wave at line 184 separates G0 preparation, independent review, and the subsequent G1 implementation work. |
| A requested workspace budget lacked separate streaming, process-memory, and fallback semantics. | P2 | Resolved. The contracts and resource items at lines 33–43 distinguish enforced live transformation buffers, additional metadata/interpreter costs, sampled process-tree RSS, and Lalamo's unbounded/unverified resource status. Strict requests reject a fallback without an independently enforceable contract. Promotion checks that behavior at line 156. |
| The single typed plan/writer requirement could require replacing the retained Lalamo writer. | P2 | Resolved. Lines 28 and 35 scope typed plans to streaming adapters. The Phase 1 item at line 73 preserves the pinned fallback writer behind shared source/result/provenance/publication contracts and requires independent checks of its observable behavior. |
| Phase 3 dependencies could create a global barrier behind unrelated Qwen2/Llama/Mistral qualifications. | P2 | Resolved. Line 108 makes dependencies local to each adapter's integration, metadata, and required operators, applying G2 per tuple. The milestone at line 194 explicitly preserves overlap for dependency-ready work. |
| Moving conversion code into nested adapter modules would leave it outside the current flat source-hash manifest. | P1 | Resolved. The Phase 1 item at line 74 requires a comprehensive recursive manifest keyed by relative path, including profiles/catalogs/schemas/exporters/resources/configuration. G1 at line 86 requires mutation, addition, and removal tests that change provenance and invalidate runs. The item at line 75 separately binds native qualification to actual Stark source/build/binary/toolchain/features/locks. |

The source-hash finding is a migration requirement: the existing implementation hashes direct `stark_convert/*.py` files. The current converter was not refactored in this documentation task.

## Source checks supporting acceptance

| Requirement checked | Source evidence and conclusion |
| --- | --- |
| Preserve the fast/fallback boundary and sanitized publication behavior. | [CLI orchestration](../tools/converter/stark_convert/cli.py), `convert` at lines 176–263, distinguishes detected unsupported conversion/metadata from hard conversion failures, stages privately, checks mutation, and publishes without replacement. An unsupported architecture probe is not complete source validation. The plan keeps that distinction explicit. |
| Treat offline manifest replay as future work. | [Primary source resolution](../tools/converter/stark_convert/cli.py), `_source` at lines 54–85, resolves remote IDs through `HfApi.model_info`. [Fallback resolution](https://github.com/jshahbazi/stark/blob/77e9a425ae3d8f0d140911adfb6288e7795e7fd8/tools/converter/stark_convert/lalamo_runner.py), `PinnedSnapshotResolver` at lines 32–112, requires online pin discovery for external repositories and rejects unresolved dependencies when downloads are disabled. Cached files alone do not implement the planned replay contract. |
| Preserve exact reference parity. | [Independent output comparison](../tools/converter/stark_convert/benchmark.py), `compare_outputs` at lines 153–186, compares tensor names/shapes/dtypes/payload hashes, metadata, complete configuration, and normalized tokenizer. The plan retains exact target-byte parity for matching dtype transforms and deterministic packed conversion. Dequantized metrics and intentionally changed transforms have separate protocols. |
| Separate streaming scratch allocation from total memory. | [Streaming writer](../tools/converter/stark_convert/streaming.py), `write_plan` at lines 331–390, reports transformation-buffer peaks. [Benchmark measurement](../tools/converter/stark_convert/benchmark.py), `_measure` at lines 203–239, samples process-tree RSS. These are different measurements. [Lalamo execution](https://github.com/jshahbazi/stark/blob/77e9a425ae3d8f0d140911adfb6288e7795e7fd8/tools/converter/stark_convert/lalamo_runner.py), lines 268–273, materializes and saves the imported model. |
| Require complete provenance after module expansion. | [Toolchain provenance](../tools/converter/stark_convert/cli.py), lines 42–50, currently uses a nonrecursive glob at line 44. The new Phase 1 requirement covers the nested modules and catalog resources planned for expansion. |
| Keep native loading and execution qualifications distinct. | [Native language loading](../crates/stark-engine/src/engine/language_model/mod.rs), lines 78–103, selects BF16 and checks tensor consumption. [Linear loading](../crates/stark-engine/src/encodable_block/linear/matmul.rs), lines 77–80, accepts BF16/F32 and rejects F16. [Embedding loading](../crates/stark-engine/src/encodable_block/embedding_table.rs), lines 99–103, also rejects F16. A load-only smoke establishes construction for the tested artifact. Execution claims require separate tests. |
| Qualify compressed layouts individually. | [Weight-matrix loading](../crates/stark-engine/src/encodable_block/weight_matrix.rs), lines 35–75 and 105–157, supports specific full-precision/MLX/integer specs, 4/8-bit modes, positive group sizes, exact layout, packing divisibility, and typed scale/correction planes. [Embedding special cases](../crates/stark-engine/src/encodable_block/embedding_table.rs), lines 107–131, impose additional D4S4/hybrid constraints. Type-enum presence is insufficient runtime evidence. |
| Preserve the MoE runtime boundary. | [MLP construction](../crates/stark-engine/src/encodable_block/mlp/mod.rs), lines 26–34 and 78–79, explicitly rejects mixture-of-experts construction. The plan requires separately scoped native work before claiming runnable MoE. |
| Preserve source-bound historical evidence and attribution. | [Prototype validation](MODEL_CONVERSION.md) and [converter provenance](../tools/converter/README.md) record the one real BF16 checkpoint, exact comparison, load-only scope, one quantized fallback smoke, source/lock hashes, and preserved Lalamo attribution. The plan records new validation separately and does not reuse those ratios for other tuples. |

The pinned Lalamo registry and loader sources were used as conversion references. No complete registry inventory or qualification fraction was produced by this review; Phase 0 owns that work.

## Remaining implementation risks

- G0 must establish the actual registry denominator, required files, origins, optional dependencies, and tuple-level acceptance fixtures. Blocked rows must retain their status.
- Comprehensive source/dependency manifests, offline replay, fallback source-consumption accounting, strict resource routing, and subprocess cleanup still need implementation and adversarial tests.
- Exact dtype and compressed-format parity may expose nondeterministic or platform-dependent upstream behavior. Such cases need the separately reviewed protocol specified by the plan before promotion.
- Native support remains narrower than conversion reach. Every claimed load or execution mode needs evidence for the exact artifact, Stark build, features, and target.
- Real-model qualification and benchmarks require immutable inputs, private permitted artifacts, serial measured runs, and criteria fixed before evaluation. Four-slot parallel work must respect file ownership and reviewed interface dependencies.

The revised plan provides concrete gates for these risks. Proceed with G0 under the documented scope; this review confers no expanded runtime or coverage certification.
