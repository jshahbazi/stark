# Independent conversion milestone 2 technical review

Date: 2026-10-11. Baseline: `3b069795`. Reviewer was independent of the production implementation and qualification coordinator. This review covers the final conversion packet, source-integrity optimization, and source-bound evidence in [milestone 2](MODEL_CONVERSION_MILESTONE_2.md).

## Disposition

**Accept the scoped G3 conversion, load-only, and bounded-memory promotion.** No unresolved promotion-blocking finding remains. Seven new real BF16 artifact tuples and 26 complete synthetic variants have the required independent parity evidence. The package implements 33 of the 110 pinned built-ins plus four separately counted extensions; unmeasured profiles remain implemented-unqualified and 77 built-ins still require adapters.

The broad complete-command latency objective against Lalamo remains **open**. The controlled Mistral comparison establishes a lower observed median against the prior Stark converter. The recorded FunctionGemma tuple is faster than the external reference. Eight other measured tuples remain slower. This acceptance grants no family-wide execution, quality, inference-speed, quantized-format, or full Lalamo-replacement qualification.

## Reviewed source and evidence binding

| Artifact | SHA-256 |
| --- | --- |
| final converter sources | `0a4a17a1273b0387b0a77ee019473a366726dd286e7eacbd7841ea18b1c55e15` |
| converter lock | `c7a77fc0e6451524d8276c3aba584929ef0e0907a2f255ad1f160af69ac7e0d5` |
| nine-checkpoint official benchmark | `bebac6881dee0a75bf2f60ea359b955abcfe9651188a66a3cbf914c475a755e6` |
| 26 synthetic artifact comparisons | `e116e87fb51995097a4b200068095b2684f7ab4e57c6abb37e8abbf7e1f5a749` |
| seven initial qualification aggregates | `842229ff5c3898faf50bbf52ec845521ae39e6e1eb8db57e25346340678b8f82` |
| eleven exact native qualifications | `1e8aeadd0088204b60db03d79d0e5bd7f63491156007706790b1b182a23aa1a4` |

The reviewer recomputed the current converter/lock provenance and aggregate hashes, checked catalog counts, inspected all benchmark acceptance flags, and matched all 27 final-source candidate receipts to their stored source/weights/config/tokenizer qualification tuples. All candidate receipts contain the exact named-build qualification record. Every recorded native result reports successful loading and zero scoring calls. The earlier untimed source versions remain identified separately in milestone evidence; their measurements were not silently rebound to the final source.

Reference semantics were checked directly against the clean external Lalamo checkout at `831accb8252a6b6606c378048caeb44178f1a050`, including the Gemma2/3, Phi3, Granite, and SmolLM3 configs, Hugging Face loader, and codec/model profiles. MIT attribution and `LICENSE.lalamo` remain preserved. Version 0.3.0 changes the lock's companion-package version only; dependency versions remain unchanged. No Lalamo, JAX, PyTorch, or Transformers runtime dependency or fallback was added.

## Technical assessment

**Tensor operators and arithmetic.** Phi preserves Q/K/V rows and reverses gate/up through contiguous source row views. The complete original input catalog remains bound to file signatures; row slicing introduces no full matrix allocation. Partial LongRoPE retains distinct attention and rotary dimensions, factor counts, original context length, and the pinned full-attention semantics. Unsupported untied/bias/other RoPE variants fail explicitly.

Gemma2 preserves tied embedding scaling, four layer normalization points, alternating windows, approximate GELU, and attention/final softcaps. Gemma3 text adds head-width Q/K norms and periodic local/global attention with separate local and accepted global RoPE configs. Norm offset and embedding scaling remain configuration semantics; stored norm bytes are widened to F32 without folding offsets into weights. Multimodal/newer Gemma and contradictory window/layer annotations remain unsupported.

Granite applies residual multiplication and final-norm division before source-to-output casting. This order matters for BF16/F16 rounding and normalization widening. The implementation reproduces the pinned CPU backend's rounded reciprocal, source-precision rounding, signed-zero, NaN, and flush-to-zero behavior. The reviewer independently reconstructed the synthetic inputs and checked all 126 frozen arithmetic hashes at a 4,096-byte workspace: all matched. BF16/F16 cases cover all 65,536 source bit patterns; F32 cases cover 4,120 edge/permutation patterns. Scalar work buffers fit the conservative accounting: input, F32 work values, and exponent/mask scratch are accounted for; the underflow mask and working F32 array are released before later allocation stages.

SmolLM3 preserves the pinned per-layer RoPE mask despite its misleading field name. Interval, full-attention, and sliding-window annotations must agree with the accepted mask/layout. Positive `pretraining_tp` is training metadata ignored by the pinned SmolLM3 decoder/load path. SmolLM2's browser-export dictionary is admitted only in the Llama profile and cannot select the checkpoint tensor representation. Complete header, geometry, dtype, source-consumption, tied-head, and unsupported-quantization checks remain authoritative. The target's nested Linear/WeightMatrix names deliberately include `.weights.weights` and `.weights.spec`; these match the established qualified schema.

Qwen3 now uses the reviewed bounded casts for all nine uniform BF16/F16/F32 source/target combinations, preserving normalization widening, projection/bias ordering, and tied-head checks. Mixed dtypes within a fused group remain unsupported because concatenation promotion can alter NaN bytes before a cast. Full synthetic comparisons provide tensor/spec/config/tokenizer evidence beyond isolated operator tests.

**Metadata and status boundaries.** Granite parser/reasoning defaults and explicit reasoning boundaries agree with the pinned profiles. Static inline defaults resolve through reviewed role/hash identities and recheck their content hash. Catalog rows do not embed prompt templates or vocabularies. Local and external metadata source identity remains in the source manifest. Unsupported identities/profiles cannot select arbitrary inline data or execute checkpoint code.

The expanded [qualification catalog](../../../tools/converter/stark_convert/native_qualifications.json) binds BF16 repository identity and exact source/artifact hashes to a named unchanged Rust 1.98.1 release/default-plus-optimized build. It confers load-only support on that recorded build. F16/F32 remain runtime-blocked; arbitrary future binaries and unmeasured checkpoint variants receive no qualification. Capability probes still report source validation as not performed until successful publication completes it.

**Source integrity and latency optimization.** Source hashing uses reusable 256 KiB buffers and at most four workers for independent large files. Small metadata hashes finish before worker fanout, bounding hashing buffers to 1 MiB on pinned CPython. Hash ordering remains deterministic. Every worker is joined, including failure and interruption; hashing exceptions remain sanitized.

Initial resolution/replay and final publication retain complete SHA-256 checks. All file signatures are checked before and after the entire batch, including restored-signature content mutations through full hashes. Root/index selection is checked again after payload reads. The optimization does not substitute signature-only publication or remove source verification. Transformation workspace and hash buffers remain separate from whole-process memory; RSS sampling provides observations and cannot enforce a process ceiling.

## Validation and measured outcomes

During code review, the reviewer independently ran the Phi/Gemma/frozen dense-cast subset (202 passed), source-integrity subset (42 passed), and all 126 arithmetic reference combinations. The coordinator's final integrated suite records **858 passed and 26 unsupplied metadata-reference skips**, comprising 847 model-free cases and eleven real-reference metadata checks. Relevant adversarial tests cover malformed layouts/configs, mixed fusion, incomplete scales, conflicting annotations, source/index mutations, cleanup, publication, and provenance. Formatting, lock, repository-policy, advisory-report, and actionlint checks are recorded as passed. The native build and eleven readiness/EOF load results remain load-only evidence.

The official aggregate records three repeats over nine checkpoints, serial fresh processes, warm sources, alternating backend order, and parity outside timed intervals. All 60 timed-output artifact comparisons passed, all hard acceptance flags passed, and every median RSS hypothesis passed. The report retains the source/reference/baseline identities and the limitations of 10 ms RSS sampling. The external Lalamo commands perform weaker verification/receipt work, so these timings do not isolate equivalent transformation work.

Mistral's controlled median decreased from 15.663 seconds to 9.790 seconds, a 37.5% observed reduction against the archived Stark converter; the measured ranges are separated. Qwen3's approximately 0.9% median change is within observed variation and does not establish a single-shard speed improvement. FunctionGemma's 1.047-second median versus Lalamo's 1.338 seconds supports the recorded 1.279× complete-command ratio for that tuple and protocol. The other eight reference latency hypotheses failed. The discarded writer-buffer proposal showed a change within variation and was correctly excluded from production.

## Remaining gates

The accepted milestone completes its additional dense implementation and selected exact conversion/load/memory scope. General Lalamo latency parity, the remaining 77 built-in adapters, broader checkpoint/config qualification, quantized formats, multimodal/newer Gemma, untied Phi3, hybrid/state-space/MoE/classifier/TTS conversion, execution/quality, strict process-memory enforcement, and distribution remain open.

No Rust/Metal/runtime format or scoring-path change is present. Existing converted Jev/JevK5 artifacts retain the same execution path; no Jev conversion, inference-speed, or judge-quality test was performed. Prior source-bound evidence is preserved. Models, operational tokenizer/config inputs, and permitted private aggregate artifacts remain ignored; public review exposes only identities, aggregate outcomes, and hashes. No prohibited native execution content or exception captures are retained. Inherited cloud/signing/release infrastructure remains disabled.
