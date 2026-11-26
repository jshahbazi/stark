# Stark model research source index

This bibliography links every checked-in benchmark report and dated model experiment at the time of indexing. It also links model-focused archived runs, validation records, conversion history and evidence packets. Consult each source for the exact artifact, implementation, result and limitations.

For before/after results, use the [family speed histories](../MODEL_RESEARCH.md). For record status, use the [experiment ledger](README.md) and the [benchmark index](../benchmarks/README.md). The [full archived-run directory](../archive/runs/) also contains product and infrastructure work.

## Current model references

- [Model-family support and installation preflight](../MODEL_SUPPORT.md)
- [Adding model support](../MODEL_ONBOARDING.md)
- [Beginner recommendation evidence](../MODEL_RECOMMENDATIONS.md)
- [Model image capabilities and remaining work](../MODEL_IMAGE_STATUS.md)
- [Default image support: text parity worklist](../IMAGE_DEFAULT_PARITY_PLAN.md)
- [Native GLM-5.3-Flash, Qwen3.8-Flash-Next, Flint and Ripple](../NATIVE_HYBRID_MODELS.md)
- [Native hybrid sampling](../NUMERICAL_SAMPLING.md)
- [Performance playbook](../PERFORMANCE_PLAYBOOK.md)
- [Qwen3.5 and Qwen3.8 checkpoints](../QWEN_TRAINED_CHECKPOINTS.md)
- [Recursive Flint](../FLINT.md)
- [Ripple-65B Preview](../RIPPLE.md)
- [OLMo 3 and Gemma 4 support](../OLMO3_GEMMA4.md)
- [Gemma 3 image artifacts](../GEMMA3_IMAGES.md)
- [Gemma 4 assistant conversion and native text MTP plan](../GEMMA4_ASSISTANT_PLAN.md)
- [Vision model families and local installation checks](../VISION_FAMILY_CONTRACT.md)
- [Decider v1.3](../DECIDER.md)
- [Elephant 1](../ELEPHANT.md)
- [Edge0-145B native support plan](../EDGE0_PLAN.md)
- [Chat terminator and desktop metrics validation](../TERMINATOR_METRICS_VALIDATION_2026_10_25.md)
- [Converter formats and usage](../../tools/converter/README.md)
- [Upstream source revisions and patch provenance](../../provenance/UPSTREAM.md)
- [Exact native qualification records](../../tools/converter/stark_convert/native_qualifications.json) and [vision runtime qualifications](../../tools/converter/stark_convert/vision_runtime_qualifications.json)
- [Frozen scoring validation](../../provenance/SCORING_VALIDATION.json) and [ColGrep generation validation](../../provenance/COLGREP_GENERATION_VALIDATION.json)

## Qwen3.8-Flash-Next and Lily

- [Shared load-time generation context sizing](2026-11-25-shared-generation-context-sizing.md) · Experiment, 2026-11-25, attempt 20261125-CTX-01

### Native decode, kernels and engine comparisons

- [Flash-Next expert gate/up and shared-down experiment](../benchmarks/2026-11-07-flashnext-moe-projections.md) · Benchmark, 2026-11-07
- [Flash-Next optimization pilot](../benchmarks/2026-11-07-flashnext-optimization-pilot.md) · Benchmark, 2026-11-07
- [Flash-Next sustained throughput and idle wake diagnosis](../benchmarks/2026-11-07-flashnext-sustained-diagnosis.md) · Benchmark, 2026-11-07
- [llama.cpp 1.0.0 opportunities and Flash Next Lily baseline](../benchmarks/2026-11-15-llama-v1-flashnext-lily-assessment.md) · Benchmark, 2026-11-15
- [Sushi and Stark on Qwen3.8-Flash-Next: source and record comparison](../benchmarks/2026-11-19-sushi-flashnext-lily-assessment.md) · Benchmark, 2026-11-19
- [Qwen3.8 residual fusion and exact CPU sampling](../benchmarks/QWEN38_FUSION_2026_10_29.md) · Benchmark, 2026-10-29
- [Qwen3.8 iQ-MLX 4.7 bpw investigation](../benchmarks/QWEN38_IQ47_INVESTIGATION_2026_10_29.md) · Benchmark, 2026-10-29
- [Qwen3.8 kernel optimization at fixed quantization](../benchmarks/QWEN38_KERNELS_2026_10_29.md) · Benchmark, 2026-10-29
- [Qwen3.8 matched-checkpoint Metal optimization](../benchmarks/QWEN38_MATCHED_OPTIMIZATION_2026_10_30.md) · Benchmark, 2026-10-30
- [Qwen3.8 Q4-XL Metal performance note](../benchmarks/QWEN38_METAL_HOTPATHS_2026_10_29.md) · Benchmark, 2026-10-29
- [Qwen3.8-Flash-Next native inference optimization](../benchmarks/QWEN38_MIXED_PRECISION_2026_10_28.md) · Benchmark, 2026-10-28
- [Qwen3.8 optimization history](../benchmarks/QWEN38_OPTIMIZATION_HISTORY.md) · Benchmark
- [Qwen3.8 sampled generation with AC power settings](../benchmarks/QWEN38_POWER_MODES_2026_10_29.md) · Benchmark, 2026-10-29
- [Qwen3.8 same-checkpoint maximum throughput](../benchmarks/QWEN38_SAME_CHECKPOINT_STARK_MLX_SERVE_2026_10_30.md) · Benchmark, 2026-10-30
- [Qwen3.8 sampled generation and BF16 profiling](../benchmarks/QWEN38_SAMPLED_BF16_2026_10_28.md) · Benchmark, 2026-10-28
- [Qwen3.8 sampled throughput and desktop contention](../benchmarks/QWEN38_STABILITY_2026_10_29.md) · Benchmark, 2026-10-29
- [Qwen3.8 sampled generation: Stark and MLX-Serve](../benchmarks/QWEN38_STARK_MLX_SERVE_2026_10_30.md) · Benchmark, 2026-10-30
- [Qwen3.8 Uzu conversion and generation comparison](../benchmarks/QWEN38_UZU_FORMAT_2026_10_29.md) · Benchmark, 2026-10-29
- [Qwen3.8 Uzu loading and sampled generation](../benchmarks/QWEN38_UZU_SPEED_2026_10_30.md) · Benchmark, 2026-10-30
- [Qwen3.8 external-table affine import](2026-11-10-qwen38-external-import.md) · Experiment, 2026-11-10
- [Lily fixed-corpus benchmark under Sushi's bench conditions](2026-11-19-lily-fixed-corpus-benchmark.md) · Experiment, 2026-11-19
- [OpenWork Lily bounded workflow acceptance](../archive/runs/2026-11-01-openwork-lily-acceptance.md) · Archived run, 2026-11-01
- [OpenWork Lily validation follow-up](../archive/runs/2026-11-01-openwork-lily-followup.md) · Archived run, 2026-11-01
- [Flash-Next PR 98 integration with main](../archive/runs/2026-11-07-flashnext-pr98-integration.md) · Archived run, 2026-11-07
- [Qwen3.8 external MLX adapter controls](../archive/runs/QWEN38_EXTERNAL_MLX_ADAPTER_CONTROLS_2026_10_30.md) · Archived run, 2026-10-30
- [Qwen3.8-Flash-Next performance literature and engine review](../archive/runs/2026-10-28-qwen38-performance-literature-review.md) · Archived run, 2026-10-28

### Speculation, draft policy and verification

- [Flash Next Lily one-draft recommendation study](../benchmarks/2026-11-15-flashnext-lily-one-draft-default.md) · Benchmark, 2026-11-15
- [Lily long coding response comparison](../benchmarks/2026-11-15-lily-long-coding-response.md) · Benchmark, 2026-11-15
- [Lily projection reuse with a very long public benchmark input](../benchmarks/2026-11-15-lily-long-context-infinitebench.md) · Benchmark, 2026-11-15
- [Lily long business-planning response comparison](../benchmarks/2026-11-15-lily-long-planning-response.md) · Benchmark, 2026-11-15
- [Lily MTP default: comparison reconciliation](../benchmarks/2026-11-15-lily-mtp-default-reconciliation.md) · Benchmark, 2026-11-15
- [Lily sampling-specific MTP recommendation](../benchmarks/2026-11-15-lily-sampling-mtp-policy.md) · Benchmark, 2026-11-15
- [Lily small-row verification on natural long responses](../benchmarks/2026-11-15-lily-small-row-verifier.md) · Benchmark, 2026-11-15
- [Qwen3.8 batched exact MTP verification](../benchmarks/QWEN38_BATCHED_MTP_2026_10_30.md) · Benchmark, 2026-10-30
- [Qwen3.8 exact MTP generation](../benchmarks/QWEN38_EXACT_MTP_2026_10_30.md) · Benchmark, 2026-10-30
- [Qwen3.8 MTP with the optimized Metal branch](../benchmarks/QWEN38_MTP_COMBINATION_2026_10_31.md) · Benchmark, 2026-10-31
- [Qwen3.8 exact MTP sampled generation optimization](../benchmarks/QWEN38_MTP_OPTIMIZATION_2026_10_30.md) · Benchmark, 2026-10-30
- [Qwen3.8 mixed-4-8bit exact batched MTP](../benchmarks/QWEN38_OFFICIAL_BATCHED_MTP_2026_10_31.md) · Benchmark, 2026-10-31
- [Adopt verifier projection reuse](2026-11-16-lily-projection-reuse.md) · Experiment, 2026-11-16
- [Lily GDN verify replay rollback](2026-11-19-lily-gdn-verify-replay.md) · Experiment, 2026-11-19
- [Shared-weight hyper-connection reads in MTP verify](2026-11-19-lily-hc-verify-rows.md) · Experiment, 2026-11-19
- [Share routed experts across verify rows](2026-11-19-lily-moe-verify-rows.md) · Experiment, 2026-11-19
- [Lily MTP adaptive depth and prompt-lookup drafts](2026-11-19-lily-mtp-depth-policy.md) · Experiment, 2026-11-19
- [Fewer host round trips in Lily MTP rounds](2026-11-19-lily-mtp-host-roundtrips.md) · Experiment, 2026-11-19
- [Merged Lily MTP changes, quiet before/after timing](2026-11-19-lily-mtp-merged-quiet-timing.md) · Experiment, 2026-11-19
- [Lily MTP round profile on Qwen3.8-Flash-Next](2026-11-19-lily-mtp-round-profile.md) · Experiment, 2026-11-19
- [Lily MTP schedule, matched-window gate for a default](2026-11-19-lily-mtp-schedule-default-gate.md) · Experiment, 2026-11-19
- [Lily Q4-XL default MTP schedule (adaptive with prompt lookup)](2026-11-20-lily-mtp-schedule-default.md) · Experiment, 2026-11-20
- [MTP integration with current main](../archive/runs/2026-10-31-mtp-main-merge.md) · Archived run, 2026-10-31
- [MTP model-free build follow-up](../archive/runs/2026-10-31-mtp-model-free-build-fix.md) · Archived run, 2026-10-31
- [Model-aware MTP toggle backend](../archive/runs/2026-10-31-mtp-toggle-backend.md) · Archived run, 2026-10-31
- [OpenWork branch integration with MTP main](../archive/runs/2026-10-31-openwork-mtp-main-merge.md) · Archived run, 2026-10-31
- [Lily automatic MTP recommendation restored to two](../archive/runs/2026-11-15-lily-mtp-two-draft-default-restored.md) · Archived run, 2026-11-15
- [Lily verifier projection reuse: final native validation](../archive/runs/2026-11-16-lily-projection-reuse.md) · Archived run, 2026-11-16

### Prefix reuse, prefill, startup and image residency

- [Flash-Next idle wake latency fix](../benchmarks/2026-11-07-flashnext-idle-wake-fix.md) · Benchmark, 2026-11-07
- [Lily text MTP with vision loaded: quick paired check](../benchmarks/2026-11-08-lily-vision-text-mtp.md) · Benchmark, 2026-11-08
- [Retained Lily image startup and Playground attachment](../benchmarks/2026-11-09-lily-image-startup.md) · Benchmark, 2026-11-09
- [Image-resident Lily text controls](2026-11-09-model-launch-double-check.md) · Experiment, 2026-11-09
- [Retained Lily image resolution](2026-11-10-lily-image-size.md) · Experiment, 2026-11-10
- [Pinned Lily midpoint prefix capture](2026-11-17-lily-midpoint-integration.md) · Experiment, 2026-11-17
- [Lily midpoint prefix speed and release record](2026-11-17-lily-midpoint-speed-and-release.md) · Experiment, 2026-11-17
- [Lily startup mode and memory admission](2026-11-17-lily-startup-mode-memory.md) · Experiment, 2026-11-17
- [Lily MTP scheduling selectors under generation prefix reuse](2026-11-19-lily-mtp-selector-prefix-reuse.md) · Experiment, 2026-11-19
- [Qwen3.8 prefix reuse with recurrent-state snapshots](../archive/runs/2026-11-02-hybrid-prefix-reuse.md) · Archived run, 2026-11-02
- [Retained Lily vision source numerics](../archive/runs/2026-11-07-retained-lily-vision-numerics.md) · Archived run, 2026-11-07
- [Text MTP with retained vision loaded](../archive/runs/2026-11-08-lily-vision-text-mtp.md) · Archived run, 2026-11-08
- [Model launch fixes: independent double-check](../archive/runs/2026-11-09-model-launch-double-check.md) · Archived run, 2026-11-09
- [General Flash Next image admission](../archive/runs/2026-11-12-flashnext-general-images.md) · Archived run, 2026-11-12
- [Live OpenWork session with Qwen3.8 prefix reuse](../archive/runs/2026-11-02-openwork-qwen38-live-prefix-reuse.md) · Archived run, 2026-11-02
- [Retained ddalcu default-image memory plan](../archive/runs/2026-11-09-ddalcu-default-image-memory-plan.md) · Archived run, 2026-11-09
- [Qwen3.8 Beginner memory admission](../archive/runs/2026-11-08-qwen-guided-memory.md) · Archived run, 2026-11-08

## Dense Qwen, Qwen MoE and GGUF

- [Shared load-time generation context sizing](2026-11-25-shared-generation-context-sizing.md) · Experiment, 2026-11-25, attempt 20261125-CTX-01

### Qwen3.8 27B, catalog admission and image parity

- [Dense Qwen27 approaches and decisions](../benchmarks/2026-11-06-qwen38-27b-approaches.md) · Benchmark, 2026-11-06
- [Dense Qwen3.8-27B oQ4e MTP on M5 Max](../benchmarks/2026-11-06-qwen38-27b-oq4e-mtp.md) · Benchmark, 2026-11-06
- [Dense Qwen27 packed weights and MTP follow-up](../benchmarks/2026-11-06-qwen38-27b-packed-followup.md) · Benchmark, 2026-11-06
- [Qwen dense image-loaded text parity controls](../benchmarks/2026-11-09-qwen-dense-image-text-parity.md) · Benchmark, 2026-11-09
- [Remaining Qwen image-loaded text controls](../benchmarks/2026-11-09-qwen-remaining-image-text-parity.md) · Benchmark, 2026-11-09
- [Qwen image experiments: retained attempt ledger](2026-11-08-qwen-images.md) · Experiment, 2026-11-08
- [F32 Qwen scalar attention workspace](2026-11-09-qwen-attention-memory.md) · Experiment, 2026-11-09
- [Official dense catalog admission](2026-11-17-qwen38-dense-catalog.md) · Experiment, 2026-11-17
- [Official dense Qwen3.8 template admission](2026-11-17-qwen38-official-template-admission.md) · Experiment, 2026-11-17
- [Dense Qwen27 CI and catalog follow-up](../archive/runs/2026-11-06-qwen38-27b-ci-followup.md) · Archived run, 2026-11-06
- [Qwen dense default-image text parity](../archive/runs/2026-11-09-qwen-dense-image-default-parity.md) · Archived run, 2026-11-09
- [Qwen affine MoE bounded-prefill memory proposal](../archive/runs/2026-11-09-qwen-moe-bounded-prefill-memory-plan.md) · Archived run, 2026-11-09
- [Remaining Qwen default-image text parity](../archive/runs/2026-11-09-qwen-remaining-image-text-parity.md) · Archived run, 2026-11-09
- [Qwen 27B shared-weight fast text with images loaded](../archive/runs/2026-11-09-qwen27-shared-weight-fast-text.md) · Archived run, 2026-11-09

### Qwen2.x, Qwen3, Qwen3.5 and Mistral

- [Qwen3-4B generation prefix reuse and long-prompt prefill profile](../archive/runs/2026-11-01-generation-prefix-reuse.md) · Archived run, 2026-11-01
- [Qwen vision text prefix timing](../benchmarks/2026-11-08-qwen35-vision-text-prefix.md) · Benchmark, 2026-11-08
- [Qwen3.5-2B: Stark and four other Mac inference engines](../benchmarks/QWEN35_ENGINE_COMPARISON_2026_10_22.md) · Benchmark, 2026-10-22
- [Qwen3.5-2B: Stark versus MLX-LM under observed quiet conditions](../benchmarks/QWEN35_MLX_QUIET_COMPARISON_2026_10_20.md) · Benchmark, 2026-10-20
- [Qwen3 GGUF compressed import experiment](../benchmarks/QWEN3_GGUF_IMPORT_EXPERIMENT.md) · Benchmark
- [Qwen3 GGUF prefill optimization](../benchmarks/QWEN3_GGUF_PREFILL_OPTIMIZATION.md) · Benchmark
- [Stock TensorFold and Stark: Qwen3.5-2B MLX 4-bit](../benchmarks/TENSORFOLD_STOCK_QWEN35_COMPARISON.md) · Benchmark
- [Restored Qwen3 0.6B artifact replay](2026-11-10-tool-quality-qwen06-replay.md) · Experiment, 2026-11-10
- [Qwen3.5 qualified conversion and serving replay](../archive/runs/2026-10-26-qwen35-image-activation.md) · Archived run, 2026-10-26
- [Qwen3.5 native image execution qualification](../archive/runs/2026-10-26-qwen35-image-execution.md) · Archived run, 2026-10-26
- [Native projected-image fusion and Qwen3.5 M-RoPE](../archive/runs/2026-10-26-qwen35-native-fusion-mrope.md) · Archived run, 2026-10-26
- [Native Qwen3.5 preprocessing and vision tower](../archive/runs/2026-10-26-qwen35-native-vision-preprocessing.md) · Archived run, 2026-10-26
- [Qwen3.5 vision artifact conversion boundary](../archive/runs/2026-10-26-qwen35-vision-artifact-contract.md) · Archived run, 2026-10-26
- [Qwen3.5 vision artifact follow-up review](../archive/runs/2026-10-26-qwen35-vision-artifact-review.md) · Archived run, 2026-10-26
- [Official Qwen3.5 vision parity](../archive/runs/2026-10-26-qwen35-vision-parity.md) · Archived run, 2026-10-26
- [Config-derived Qwen3.5 families](../archive/runs/2026-10-27-qwen35-config-driven-family.md) · Archived run, 2026-10-27
- [Qwen3.5 family admission and automatic installation checks](../archive/runs/2026-10-27-qwen35-family-installation.md) · Archived run, 2026-10-27
- [Qwen3.5 Developer ID worker execution qualification](../archive/runs/2026-10-27-qwen35-signed-worker.md) · Archived run, 2026-10-27
- [Qwen3.5 Swift attachments and local signed-package qualification](../archive/runs/2026-10-27-qwen35-swift-packaged-app.md) · Archived run, 2026-10-27
- [Qwen retained image execution paths](../archive/runs/2026-11-07-qwen-image-families.md) · Archived run, 2026-11-07
- [Qwen3.5 BF16 MoE image implementation](../archive/runs/2026-11-07-qwen35-bf16-moe-images.md) · Archived run, 2026-11-07
- [Qwen 4 release preparation](../archive/runs/2026-11-08-qwen4-release-readiness.md) · Archived run, 2026-11-08
- [MLX affine import for Qwen2 and Mistral dense](2026-11-12-dense-mlx-qwen2-mistral.md) · Experiment, 2026-11-12
- [MLX affine Qwen2.5 and Mistral numerical qualification](../archive/runs/2026-11-12-dense-mlx-qwen2-mistral-qualification.md) · Archived run, 2026-11-12
- [MLX affine import for Qwen2 and Mistral dense](../archive/runs/2026-11-12-dense-mlx-qwen2-mistral.md) · Archived run, 2026-11-12
- [App installs select native F16 for all-F16 MLX Qwen2 and Mistral sources](../archive/runs/2026-11-12-install-auto-f16.md) · Archived run, 2026-11-12

## GLM, Flint and native hybrids

- [Hy4-flash AngelSlim Q3_K native adoption and llama.cpp parity](2026-11-25-hy4-native-q3-k.md) · Model experiment, 2026-11-25
- [Hy4-flash AngelSlim Q3_K prefill and decode attribution](2026-11-25-hy4-prefill-decode-profile.md) · Model experiment, 2026-11-25
- [Hy4-flash AngelSlim Q3_K routed prefill and expert retention](2026-11-25-hy4-dense-prefill-expert-retention.md) · Model experiment, 2026-11-25
- [Hy4-flash AngelSlim Q3_K context ladder and decode residency](2026-11-25-hy4-context-ladder.md) · Model experiment, 2026-11-25
- [Hy4-flash AngelSlim Q3_K NextN cache construction and verify attention](2026-11-25-hy4-decode-cache-construction.md) · Model experiment, 2026-11-25
- [Hy4-flash AngelSlim Q3_K multi-query MLA prefill](2026-11-25-hy4-multi-query-prefill.md) · Model experiment, 2026-11-25
- [Hy4-flash AngelSlim Q3_K grouped expert fragments](2026-11-25-hy4-grouped-fragments.md) · Model experiment, 2026-11-25
- [Hy4-flash catalog install and packaged generation](2026-11-26-hy4-desktop-catalog.md) · Model experiment, 2026-11-26
- [GLM EXL3 throughput comparison and gap investigation](../benchmarks/2026-11-04-glm-mlx-stream-comparison.md) · Benchmark, 2026-11-04
- [GLM-5.3 Batched Prefill Experiment](../benchmarks/GLM53_BATCHED_PREFILL_2026_11_02.md) · Benchmark, 2026-11-02
- [Native GLM and Qwen GPU decode graphs](../benchmarks/GLM_QWEN_DECODE_GRAPH_2026_10_28.md) · Benchmark, 2026-10-28
- [Native GLM and Qwen GPU graph optimization](../benchmarks/GLM_QWEN_NATIVE_GRAPH_2026_10_28.md) · Benchmark, 2026-10-28
- [Native GLM and Qwen execution optimization](../benchmarks/GLM_QWEN_NATIVE_OPTIMIZATION_2026_10_28.md) · Benchmark, 2026-10-28
- [Experimental GLM batched matrices](../validation/GLM_BATCHED_MATRICES.md) · Validation
- [Optional GLM EXL3/FP8 backend](../validation/GLM_EXL3_BACKEND.md) · Validation
- [Experimental GLM register KDA prefill](../validation/GLM_KDA_PREFILL.md) · Validation
- [Experimental GLM wide prefill](../validation/GLM_WIDE_PREFILL.md) · Validation
- [Recursive Flint native support validation](../archive/runs/2026-10-28-flint-native-support.md) · Archived run, 2026-10-28
- [GLM-5.3-Flash native performance literature review](../archive/runs/2026-10-28-glm53-performance-literature-review.md) · Archived run, 2026-10-28
- [Native hybrid full-checkpoint validation](../archive/runs/2026-10-28-native-hybrid-full-checkpoints.md) · Archived run, 2026-10-28
- [Native GLM5-Next and Qwen4-Exp implementation validation](../archive/runs/2026-10-28-native-hybrid-implementation.md) · Archived run, 2026-10-28
- [GLM batched-matrix research record](../archive/runs/2026-11-03-glm-batched-matrix-research.md) · Archived run, 2026-11-03
- [GLM EXL3 backend implementation and qualification](../archive/runs/2026-11-03-glm-exl3-backend.md) · Archived run, 2026-11-03
- [GLM-5.3-Flash fast-engine research](../archive/runs/2026-11-03-glm-fast-engine-research.md) · Archived run, 2026-11-03
- [GLM KDA segmented reduction candidate: declined numerical qualification](../archive/runs/2026-11-03-glm-kda-segment-tree-declined.md) · Archived run, 2026-11-03
- [GLM native model catalog integration](../archive/runs/2026-11-03-glm-model-catalog.md) · Archived run, 2026-11-03
- [GLM-5.3 and external mixed-affine Qwen3.8 prefix reuse](../archive/runs/2026-11-03-glm-qwen-prefix-reuse.md) · Archived run, 2026-11-03
- [GLM-5.3 Bounded Metal Prefill Validation](../archive/runs/2026-11-03-glm53-prefill-merge-validation.md) · Archived run, 2026-11-03
- [GLM EXL3 native MTP and head-prefix mechanism validation](../archive/runs/2026-11-04-glm-exl3-native-mtp-prefix.md) · Archived run, 2026-11-04
- [GLM EXL3/FP8 artifact qualification](../archive/runs/2026-11-04-glm-exl3-qualification.md) · Archived run, 2026-11-04
- [Retained GLM BF16 native profile](../archive/runs/2026-11-05-glm-retained-profile.md) · Archived run, 2026-11-05
- [GLM Flash, Flint and recent-model tool assessment](../archive/runs/2026-11-07-glm-flint-tool-qualification.md) · Archived run, 2026-11-07
- [Guided native and recurrent model coverage](../archive/runs/2026-11-08-guided-native-hybrid-coverage.md) · Archived run, 2026-11-08

## Gemma and OLMo

- [Gemma 4 E2B/E4B: text with image support loaded](../benchmarks/2026-11-09-gemma4-image-text-parity.md) · Benchmark, 2026-11-09
- [Gemma 4 warmed cache follow-up and suffix-dispatch fix](../benchmarks/2026-11-09-gemma4-warmed-cache.md) · Benchmark, 2026-11-09
- [OLMo 3 and Gemma 4 Mac engine evidence](../benchmarks/OLMO3_GEMMA4_MAC_ENGINES.md) · Benchmark
- [Matched native OLMo 3 and Gemma 4 text benchmark](../benchmarks/OLMO3_GEMMA4_NATIVE_MATCHED.md) · Benchmark
- [Repaired OLMo 3 and Gemma 4 Q4 benchmark](../benchmarks/OLMO3_GEMMA4_Q4_REPAIRED_2026_10_29.md) · Benchmark, 2026-10-29
- [OLMo 3 and Gemma 4 source fusion benchmark](../benchmarks/OLMO3_GEMMA4_SOURCE_FUSIONS_2026_11_02.md) · Benchmark, 2026-11-02
- [Gemma image experiments: retrospective and attempt ledger](2026-11-08-gemma-images.md) · Experiment, 2026-11-08
- [Gemma image-default continuation](2026-11-09-gemma-next-image-parity.md) · Experiment, 2026-11-09
- [Gemma 4 default-image text parity](2026-11-09-gemma4-image-text-parity.md) · Experiment, 2026-11-09
- [Gemma installation repair](2026-11-10-gemma-install-repair.md) · Experiment, 2026-11-10
- [OLMo 3 and Gemma 4 Native Validation](../archive/runs/2026-10-28-olmo3-gemma4-native.md) · Archived run, 2026-10-28
- [OLMo 3 and Gemma 4 Q4 precision investigation](../archive/runs/2026-10-28-olmo3-gemma4-q4-precision.md) · Archived run, 2026-10-28
- [Gemma 4 source residual fusions: validation draft](../archive/runs/2026-10-29-gemma4-source-residual-fusions.md) · Archived run, 2026-10-29
- [OLMo 3 and Gemma 4 source fusion validation](../archive/runs/2026-11-02-olmo3-gemma4-source-fusions.md) · Archived run, 2026-11-02
- [Gemma 3 image validation](../archive/runs/2026-11-07-gemma3-images.md) · Archived run, 2026-11-07
- [Gemma 4 E4B BF16 image qualification](../archive/runs/2026-11-07-gemma4-e4b-images.md) · Archived run, 2026-11-07
- [Gemma 4 image family architecture assessment](../archive/runs/2026-11-07-gemma4-image-family-scope.md) · Archived run, 2026-11-07
- [Larger Gemma vision Flash source controls](../archive/runs/2026-11-07-gemma4-larger-flash-controls.md) · Archived run, 2026-11-07
- [Larger Gemma MLP source controls](../archive/runs/2026-11-07-gemma4-larger-mlp-controls.md) · Archived run, 2026-11-07
- [Pinned Gemma 4 31B vision component qualification](../archive/runs/2026-11-07-gemma4-larger-vision-qualified-component.md) · Archived run, 2026-11-07
- [Larger Gemma vision stage diagnostics](../archive/runs/2026-11-07-gemma4-larger-vision-stage-diagnostics.md) · Archived run, 2026-11-07
- [Gemma 4 MoE component controls](../archive/runs/2026-11-07-gemma4-moe-component-controls.md) · Archived run, 2026-11-07
- [Gemma 4 no-PLE residual controls](../archive/runs/2026-11-07-gemma4-nople-residual-controls.md) · Archived run, 2026-11-07
- [Gemma 4 12B encoder-free vision controls](../archive/runs/2026-11-07-gemma4-unified-vision-controls.md) · Archived run, 2026-11-07

## Inkling 2 Mini

- [Shared load-time generation context sizing](2026-11-25-shared-generation-context-sizing.md) · Experiment, 2026-11-25, attempt 20261125-CTX-01

### References, native milestones and follow-up

- [Original-checkpoint implementation plan](../INKLING_2_MINI_PLAN.md)
- [Native Metal retrospective](../archive/runs/2026-11-21-inkling-mini-native-retrospective.md) · Archived run, 2026-11-21
- [Prefill and decode performance follow-up](../INKLING_2_MINI_PERFORMANCE_FOLLOWUP.md) · Proposals, 2026-11-21
- [Prefill, decode, DSpark and prefix reuse speed history](speed-inkling.md)
- [Stark against MLX-Serve: every difference, what worked, what did not and what is open](inkling-mlx-serve-differences.md) · Maintained list

### Benchmarks and experiment records

- [Opt-in cumulative thinking budget](../archive/runs/2026-11-24-inkling-thinking-budget.md) · Experiment, 2026-11-24
- [Load-time context and live memory admission](2026-11-24-inkling-mini-memory-context.md) · Experiment, 2026-11-24, attempt 20261124-INKCTX-01
- [Request-sized context allocation](2026-11-24-inkling-mini-request-context.md) · Experiment, 2026-11-24, attempt 20261124-INKCTX-02
- [Tool-loop cache diagnostic](2026-11-24-inkling-mini-tool-loop-cache.md) · Experiment, 2026-11-24, attempt 20261124-INKCTX-03
- [M5 Max original and Unsloth baselines](../benchmarks/2026-11-20-inkling-mini-m5-max.md) · Benchmark, 2026-11-20
- [First milestone and real-weight NVFP4](2026-11-20-inkling-mini-milestone-1.md) · Experiment, 2026-11-20
- [Native decode slice](../benchmarks/2026-11-21-inkling-mini-native-slice.md) · Benchmark, 2026-11-21
- [Native slice trials](2026-11-21-inkling-mini-native-slice.md) · Experiment, 2026-11-21
- [Pinned Ollama oracle](../benchmarks/2026-11-21-inkling-mini-pinned-oracle.md) · Benchmark, 2026-11-21
- [Pinned-runtime parity trials](2026-11-21-inkling-mini-pinned-oracle.md) · Experiment, 2026-11-21
- [Short projection parity and budget](../benchmarks/2026-11-21-inkling-mini-short-projection.md) · Benchmark, 2026-11-21
- [Short projection root cause](2026-11-21-inkling-mini-short-projection.md) · Experiment, 2026-11-21
- [Full target and 64-row grouped prefill](2026-11-21-inkling-mini-full-target.md) · Experiment, 2026-11-21
- [Native worker and 4K prefill progression](../benchmarks/2026-11-21-inkling-mini-native-worker.md) · Benchmark, 2026-11-21
- [Worker load, vision, streaming and tool controls](2026-11-21-inkling-mini-worker.md) · Experiment, 2026-11-21
- [Vision tower against Ollama](2026-11-21-inkling-mini-vision.md) · Experiment, 2026-11-21
- [Native DSpark and serial verification](2026-11-21-inkling-mini-dspark.md) · Experiment, 2026-11-21
- [Grouped NVFP4 routes and buckets](2026-11-21-inkling-mini-grouped-nvfp4.md) · Experiment, 2026-11-21
- [Matched 512-row prefill trace](2026-11-21-inkling-mini-matched-prefill.md) · Experiment, 2026-11-21
- [Grouped prefill stage profile](2026-11-21-inkling-mini-prefill-profile.md) · Experiment, 2026-11-21
- [Faster 128-row native prefill](2026-11-21-inkling-mini-faster-prefill.md) · Experiment, 2026-11-21
- [4K prefill budget and batched RMS](2026-11-21-inkling-mini-prefill-budget.md) · Experiment, 2026-11-21
- [Packed prefill, rejected kernels and source-aligned worker](2026-11-21-inkling-mini-packed-prefill.md) · Experiment, 2026-11-21
- [Stock 4K prefill and singleton decode attribution](2026-11-21-inkling-mini-stock-profile.md) · Experiment, 2026-11-21
- [Stock NVFP4 occupancy, scale sharing and one-row decode router](2026-11-21-inkling-mini-nvfp4-router.md) · Experiment, 2026-11-21
- [Wider NVFP4 run tiles on the stock 4K graph](2026-11-21-inkling-mini-nvfp4-run-tiles.md) · Experiment, 2026-11-21
- [Tensor MXFP8 with source-order replay and wider-tile NVFP4 scale sharing](2026-11-21-inkling-mini-mxfp8-tensor.md) · Experiment, 2026-11-21
- [Default fast prefill with flags unset](2026-11-21-inkling-mini-prefill-default.md) · Experiment, 2026-11-21
- [Default one-row router with flags unset](2026-11-21-inkling-mini-decode-default.md) · Experiment, 2026-11-21
- [Decode toward 80 tok/s, verifier cuts, sampled proposal headroom and MLX-Serve comparison](2026-11-22-inkling-mini-decode-80.md) · Experiment, 2026-11-22, through attempts [109](2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-109--expand-the-nvfp4-coefficients-before-verification) and [110](2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-110--shorten-the-dspark-forward)
- [Decode toward 60 tok/s, exact verifier and sampled acceptance](2026-11-21-inkling-mini-decode-60.md) · Experiment, 2026-11-21–22
- [Default fast paths, source admission from 1 token to 32K context and default drafting](2026-11-22-inkling-mini-default-fast-paths.md) · Experiment, 2026-11-22, attempt 20261122-INKDEFAULT-01
- [Verifier long-position gain rounding fix](2026-11-22-inkling-mini-long-position-gain.md) · Experiment, 2026-11-22, attempt 20261122-INKLONGPOS-01
- [Verifier and single-row launch fusion, two sessions](2026-11-22-inkling-mini-launch-fusion.md) · Experiment, 2026-11-22–23, attempts 20261122-INKFUSE-01 and -02
- [Engine matrix against MLX-Serve, runs 1 to 5](../benchmarks/2026-11-22-inkling-mini-engine-matrix.md) · Benchmark, 2026-11-22–23, attempts 20261122-INKENGINE-01 to -03 and 20261123-INKENGINE-04 and -05
- [Standard benchmark against MLX-Serve, three repeats with quality](../benchmarks/2026-11-23-inkling-mini-standard-benchmark.md) · Benchmark (standard result), 2026-11-23, attempt 20261123-INKENGINE-06
- [Exact faster prefill: NVFP4 guard removal, fused NVFP4 kernels, faster MXFP8 replay](2026-11-22-inkling-mini-exact-fast-prefill.md) · Experiment, 2026-11-22, attempt 20261122-INKFASTPRE-01
- [Unguarded MXFP8 prefill default, noise-floor gate and mask-free global attention](2026-11-22-inkling-mini-unguarded-mxfp8-prefill.md) · Experiment, 2026-11-22–23, attempt 20261122-INKFASTPRE-02
- [Prefill attribution, MLX-Serve against Stark](../benchmarks/2026-11-22-inkling-mini-prefill-attribution.md) · Benchmark, 2026-11-22, attempt 20261122-INKPREATTR-01
- [8-bit affine dense weights: feasibility, byte-exact pack reproduction and the Stark 8-bit engine artifact](2026-11-22-inkling-mini-affine8.md) · Experiment, 2026-11-22–23, attempts 20261122-INKAFFINE8-01 to -03
- [The 8-bit artifact becomes the default: preparation, upgrade, catalog and qualification status](2026-11-23-inkling-mini-affine8-default.md) · Compatibility, 2026-11-23, attempt 20261123-INKAFFINE8-04
- [Unquantized NVFP4 expert inputs as the default, and the rejected fold](2026-11-23-inkling-mini-expert-inputs-default.md) · Experiment, 2026-11-23, attempts 20261123-INKROUND-09 and -10
- [Draft-depth planner tuning and the hold rule](2026-11-23-inkling-mini-planner-tuning.md) · Experiment, 2026-11-23, attempts 20261123-INKROUND-11 and -12
- [Gathered NVFP4 prefill kernel for wide commands](2026-11-23-inkling-mini-gathered-nvfp4.md) · Experiment, 2026-11-23, attempt 20261122-INKFASTPRE-07
- [Optimization wind-down: reverted unmeasured attempts, the paired-control hang and lock incidents](2026-11-23-inkling-mini-wind-down.md) · Experiment, 2026-11-23, attempts 20261123-INKROUND-13 and -14 and 20261122-INKFASTPRE-08
- [Session audit: attempts, intermediate figures and incidents the records lacked, and corrections](2026-11-23-inkling-mini-session-audit.md) · Documentation audit, 2026-11-23, attempt 20261123-INKAUDIT-01
- [Prefill round 3: dense kernel, packed draft commit, mask removal, width independence and two rejected kernels](2026-11-23-inkling-mini-prefill-round3.md) · Experiment, 2026-11-23, attempt 20261122-INKFASTPRE-03
- [Prefill round 4: routing, drafting overhead, command width, expert ceiling, rejected stationary and staged kernels, BF16 attention and dense decomposition](2026-11-23-inkling-mini-prefill-round4.md) · Experiment, 2026-11-23, attempts 20261122-INKFASTPRE-04 to -06
- [Decode round: stage table, short-context verifier attention, router, worker loop, depth policy, expert inputs and natural-text quality](2026-11-23-inkling-mini-decode-round.md) · Experiment, 2026-11-23, attempts 20261123-INKROUND-01 to -08
- [Worker prefill command width and the cache on/off fault](2026-11-23-inkling-mini-wide-prefill-worker.md) · Experiment, 2026-11-23, attempts 20261123-INKWIDTH-01 and -02
- [MLX-Serve source comparison](../benchmarks/2026-11-23-inkling-mini-mlx-serve-source.md) · Benchmark (source reading), 2026-11-23, attempt 20261123-INKMLXSRC-01

### Protocols and aggregates

- [Real-weight NVFP4 probe](../archive/experiments/2026-11-20-inkling-mini/NVFP4_PROBE.md)
- [Pinned oracle capture, including stock prefill and seed](../archive/experiments/2026-11-21-inkling-oracle/README.md)
- [Vision oracle capture](../archive/experiments/2026-11-21-inkling-vision/README.md)
- [Reference server and component aggregates](../benchmarks/data/inkling-mini-m5-max-2026-11-20.json)
- [Native slice aggregates](../benchmarks/data/inkling-mini-native-slice-2026-11-21.json)
- [Pinned oracle aggregates](../benchmarks/data/inkling-mini-pinned-oracle-2026-11-21.json)
- [Worker-schedule captures and checks at caller-chosen lengths and contexts](../archive/experiments/2026-11-22-inkling-schedule-oracle/README.md)

## Ripple

- [Ripple-65B native GPU performance](../benchmarks/RIPPLE_65B_GPU_2026_11_05.md) · Benchmark, 2026-11-05
- [Ripple-65B Preview native Mac validation](../benchmarks/RIPPLE_65B_NATIVE_2026_11_04.md) · Benchmark, 2026-11-04
- [Ripple attention decode tuning](../benchmarks/RIPPLE_Q4_ATTENTION_2026_11_12.md) · Benchmark, 2026-11-12
- [Ripple Q4_K_M native validation](../benchmarks/RIPPLE_Q4_K_M_2026_11_12.md) · Benchmark, 2026-11-12
- [Ripple Q4_K_M kernel performance](../benchmarks/RIPPLE_Q4_KERNELS_2026_11_12.md) · Benchmark, 2026-11-12
- [Ripple native implementation validation](../archive/runs/2026-11-04-ripple-native-support.md) · Archived run, 2026-11-04

## Decider, Elephant and Pi

- [Decider HTTP latency diagnosis](../benchmarks/2026-11-16-decider-http-latency.md) · Benchmark, 2026-11-16
- [Regular Decider v1.3 201-token cold timing](../benchmarks/2026-11-18-regular-decider-300-token.md) · Benchmark, 2026-11-18
- [Pi Connections delivery](2026-11-13-pi-connections.md) · Experiment, 2026-11-13
- [Guarded Pi manual qualification](2026-11-13-pi-manual-qualification.md) · Experiment, 2026-11-13
- [Decider Flash app integration](2026-11-16-decider-flash-app.md) · Experiment, 2026-11-16
- [Decider Flash v1.3 native image-capable scoring](2026-11-16-decider-flash-v13.md) · Experiment, 2026-11-16
- [Native Decision HTTP transport and latency profile](2026-11-16-decider-http-latency.md) · Experiment, 2026-11-16
- [Regular Decider v1.3 admission](2026-11-18-decider-v13.md) · Experiment, 2026-11-18
- [Pinned Elephant 1 native Decision admission](2026-11-18-elephant-1.md) · Experiment, 2026-11-18
- [Pi manual foundation](../archive/runs/2026-11-12-pi-manual-foundation.md) · Archived run, 2026-11-12
- [Pi manual foundation recheck](../archive/runs/2026-11-12-pi-manual-recheck.md) · Archived run, 2026-11-12
- [Guarded Pi live qualification](../archive/runs/2026-11-13-pi-guard-live.md) · Archived run, 2026-11-13
- [Persistent manual Pi connection](../archive/runs/2026-11-13-pi-persistent-default.md) · Archived run, 2026-11-13
- [Decider Flash v1.3 packaged-app qualification](../archive/runs/2026-11-16-decider-flash-app.md) · Archived run, 2026-11-16
- [Decider Flash v1.3 native text/image qualification](../archive/runs/2026-11-16-decider-flash-v13.md) · Archived run, 2026-11-16
- [Decider reference environment security update](../archive/runs/2026-11-16-decider-reference-security.md) · Archived run, 2026-11-16

## Cross-engine comparisons and workload studies

- [Chat-template isolation latency](../benchmarks/CHAT_TEMPLATE_ISOLATION_2026_11_03.md) · Benchmark, 2026-11-03
- [DeepInfer adoption study: no runtime change retained](../benchmarks/DEEPINFER_ADOPTION_2026_10_21.md) · Benchmark, 2026-10-21
- [MLX-informed Stark generation optimization](../benchmarks/MLX_GENERATION_OPTIMIZATION_2026_10_20.md) · Benchmark, 2026-10-20
- [Native chat-template rendering](../benchmarks/NATIVE_CHAT_TEMPLATE_2026_11_03.md) · Benchmark, 2026-11-03
- [Stark: native inference on Apple silicon, measured against MLX-LM](../benchmarks/STARK_BENCHMARK_BRIEF_2026_10_21.md) · Benchmark, 2026-10-21
- [Stock inference engine pilot: Qwen3.5-2B and JevK5](../benchmarks/STOCK_ENGINE_BENCHMARK_2026_10_19.md) · Benchmark, 2026-10-19
- [TensorFold and Stark on JevK5](../benchmarks/TENSORFOLD_JEVK5_COMPARISON.md) · Benchmark
- [ColGrep generation validation](../validation/GENERATION_COLGREP.md) · Validation
- [Scoring parity validation](../validation/SCORING_PARITY.md) · Validation
- [OpenWork time to first text](../archive/runs/2026-10-31-openwork-first-text-latency.md) · Archived run, 2026-10-31
- [Beginner published task benchmark references](../archive/runs/2026-11-08-beginner-published-task-benchmarks.md) · Archived run, 2026-11-08
- [Coding benchmark priority and local installation](../archive/runs/2026-11-08-coding-benchmark-priority.md) · Archived run, 2026-11-08
- [llmbench summary-score snapshot](../archive/runs/2026-11-17-llmbench-summary-scores.md) · Archived run, 2026-11-17
- [ColGrep generation review](../archive/runs/COLGREP_GENERATION_REVIEW.md) · Archived run
- [Cross-backend idle wake follow-up](../benchmarks/2026-11-08-cross-backend-idle-wake.md) · Benchmark, 2026-11-08
- [Catalog coding ranking snapshot](../benchmarks/2026-11-14-catalog-coding-ranking.md) · Benchmark, 2026-11-14
- [Stark comparisons: evidence inventory and future benchmark plan](../benchmarks/INFERENCE_COMPARISON_EVIDENCE.md) · Benchmark

## Shared runtime, catalog and image qualification

- [Shared load-time generation context sizing](2026-11-25-shared-generation-context-sizing.md) · Experiment, 2026-11-25, attempt 20261125-CTX-01
- [Image-support work: findings, failed attempts and remaining qualification](2026-11-08-image-support-retrospective.md) · Experiment, 2026-11-08
- [Resource ceilings and installed-app recovery](2026-11-09-resource-admission.md) · Experiment, 2026-11-09
- [Catalogue family admission](2026-11-10-catalog-family-admission.md) · Experiment, 2026-11-10
- [Capped attention and dense sampling correctness](2026-11-10-dense-attention-sampling-correctness.md) · Experiment, 2026-11-10
- [Model-family double-check before pull request](2026-11-10-family-double-check.md) · Experiment, 2026-11-10
- [Image loading and request admission correctness](2026-11-10-family-image-correctness.md) · Experiment, 2026-11-10
- [Text prefix snapshot capability](2026-11-10-text-prefix-snapshot-capability.md) · Experiment, 2026-11-10
- [Bounded tool output and continuation replay](2026-11-10-tool-output-quality-replay.md) · Experiment, 2026-11-10
- [Shared config normalization before family planners](2026-11-12-catalog-config-normalization.md) · Experiment, 2026-11-12
- [Untied Phi3 output head](2026-11-12-phi3-untied-embeddings.md) · Experiment, 2026-11-12
- [Model-family inventory and installation checks](../archive/runs/2026-10-27-model-family-support.md) · Archived run, 2026-10-27
- [Model context capacity validation](../archive/runs/2026-10-31-model-context-capacity.md) · Archived run, 2026-10-31
- [Model context default validation](../archive/runs/2026-10-31-model-context-defaults.md) · Archived run, 2026-10-31
- [Tool-template compatibility validation](../archive/runs/2026-11-01-tool-template-compatibility.md) · Archived run, 2026-11-01
- [Model admission security validation](../archive/runs/2026-11-03-model-admission-security.md) · Archived run, 2026-11-03
- [OpenWork follow-up tool-history rendering fix](../archive/runs/2026-11-03-openwork-tool-history-template.md) · Archived run, 2026-11-03
- [Beginner recommendation review and corrections](../archive/runs/2026-11-05-hardware-model-recommendation-review.md) · Archived run, 2026-11-05
- [Hardware-aware Beginner recommendations](../archive/runs/2026-11-05-hardware-model-recommendations.md) · Archived run, 2026-11-05
- [Model catalog discovery and pagination](../archive/runs/2026-11-05-model-catalog-pagination.md) · Archived run, 2026-11-05
- [Model catalog review and malformed metadata isolation](../archive/runs/2026-11-05-model-catalog-review.md) · Archived run, 2026-11-05
- [Additional trained tool protocols](../archive/runs/2026-11-07-additional-tool-protocols.md) · Archived run, 2026-11-07
- [Image family integration checkpoints](../archive/runs/2026-11-07-image-family-integration.md) · Archived run, 2026-11-07
- [Larger-model tool qualification](../archive/runs/2026-11-07-recommended-model-tool-qualification.md) · Archived run, 2026-11-07
- [Tool-calling coverage and onboarding checks](../archive/runs/2026-11-07-tool-calling-coverage.md) · Archived run, 2026-11-07
- [Tool-calling follow-up review](../archive/runs/2026-11-07-tool-calling-review.md) · Archived run, 2026-11-07
- [Context and resource-budget findings](../archive/runs/2026-11-08-context-resource-budget-findings.md) · Archived run, 2026-11-08
- [Context and Resource Budget Validation](../archive/runs/2026-11-08-context-resource-budget.md) · Archived run, 2026-11-08
- [Fresh-install resident memory planning](../archive/runs/2026-11-08-fresh-guided-memory.md) · Archived run, 2026-11-08
- [Image decoder budget follow-up](../archive/runs/2026-11-08-image-decoder-budget.md) · Archived run, 2026-11-08
- [Automatic output for undeclared contexts](../archive/runs/2026-11-08-unknown-context-auto.md) · Archived run, 2026-11-08
- [Text prefix reuse on vision artifacts](../archive/runs/2026-11-08-vision-text-prefix-reuse.md) · Archived run, 2026-11-08
- [Model launch correctness fixes](../archive/runs/2026-11-09-model-launch-correctness.md) · Archived run, 2026-11-09
- [Model launch and performance triage](../archive/runs/2026-11-09-model-launch-performance-triage.md) · Archived run, 2026-11-09
- [Playground resource admission correction](../archive/runs/2026-11-09-playground-resource-admission.md) · Archived run, 2026-11-09
- [Model catalogue family admission validation](../archive/runs/2026-11-10-catalog-family-admission.md) · Archived run, 2026-11-10
- [Model-family converter correctness](../archive/runs/2026-11-10-family-converter-correctness.md) · Archived run, 2026-11-10
- [Model-family review disposition](../archive/runs/2026-11-10-model-family-triage.md) · Archived run, 2026-11-10
- [RAM-scaled Speed recommendation check](../archive/runs/2026-11-11-speed-recommendation-policy.md) · Archived run, 2026-11-11
- [Whatmodel.sh recommendations snapshot](../archive/runs/2026-11-11-whatmodel-recommendations.md) · Archived run, 2026-11-11
- [Catalogue config normalization](../archive/runs/2026-11-12-catalog-config-normalization.md) · Archived run, 2026-11-12
- [Model catalogue GGUF-only listing screen](../archive/runs/2026-11-12-catalog-screen-gguf.md) · Archived run, 2026-11-12
- [Model catalogue Hub API reuse, verdict cache and streamed rows](../archive/runs/2026-11-12-model-catalog-api-reuse.md) · Archived run, 2026-11-12
- [Model catalogue folded foreign-format exports](../archive/runs/2026-11-12-model-catalog-folded-exports.md) · Archived run, 2026-11-12
- [Model catalogue unavailable-row reasons](../archive/runs/2026-11-12-model-catalog-honest-reasons.md) · Archived run, 2026-11-12
- [Model catalogue top-rated ordering](../archive/runs/2026-11-12-model-catalog-top-rated.md) · Archived run, 2026-11-12
- [Guided recommendations use installed RAM](../archive/runs/2026-11-13-guided-capacity-recommendations.md) · Archived run, 2026-11-13
- [Guided recommendation corner-case review](../archive/runs/2026-11-13-guided-recommendation-review.md) · Archived run, 2026-11-13
- [Converter dependency refresh validation](../archive/runs/CONVERTER_DEPENDENCY_VALIDATION_2026_10_25.md) · Archived run, 2026-10-25
- [Non-Lily resident-memory accounting](2026-11-09-non-lily-admission.md) · Experiment, 2026-11-09

## Conversion plans, milestones and reviews

- [Model conversion prototype validation](../archive/conversion/MODEL_CONVERSION.md)
- [Completed conversion model cleanup](../archive/conversion/MODEL_CONVERSION_ARTIFACT_CLEANUP.md)
- [Phase 4 format inventory](../archive/conversion/MODEL_CONVERSION_FORMAT_INVENTORY.md)
- [Converter hardening](../archive/conversion/MODEL_CONVERSION_HARDENING.md)
- [Converter hardening validation and review](../archive/conversion/MODEL_CONVERSION_HARDENING_REVIEW.md)
- [Independent conversion milestone 1](../archive/conversion/MODEL_CONVERSION_MILESTONE_1.md)
- [Independent conversion milestone 1 review](../archive/conversion/MODEL_CONVERSION_MILESTONE_1_REVIEW.md)
- [Independent conversion milestone 2](../archive/conversion/MODEL_CONVERSION_MILESTONE_2.md)
- [Independent conversion milestone 2 technical review](../archive/conversion/MODEL_CONVERSION_MILESTONE_2_REVIEW.md)
- [Milestone 3: independent MLX Qwen3 conversion](../archive/conversion/MODEL_CONVERSION_MILESTONE_3.md)
- [Milestone 3 continuation: MLX Llama, SmolLM3 and Gemma text](../archive/conversion/MODEL_CONVERSION_MILESTONE_3_DENSE.md)
- [Independent milestone 3 dense MLX technical review](../archive/conversion/MODEL_CONVERSION_MILESTONE_3_DENSE_REVIEW.md)
- [Milestone 3 MLX conversion latency](../archive/conversion/MODEL_CONVERSION_MILESTONE_3_LATENCY.md)
- [Independent milestone 3 latency technical review](../archive/conversion/MODEL_CONVERSION_MILESTONE_3_LATENCY_REVIEW.md)
- [Independent conversion milestone 3 technical review](../archive/conversion/MODEL_CONVERSION_MILESTONE_3_REVIEW.md)
- [Milestone 4: dense hybrid architectures and future checkpoints](../archive/conversion/MODEL_CONVERSION_MILESTONE_4.md)
- [Milestone 4 continuation: Qwen3.5 text and LFM2 affine MLX](../archive/conversion/MODEL_CONVERSION_MILESTONE_4_MLX.md)
- [Milestone 4 MLX continuation: independent technical review](../archive/conversion/MODEL_CONVERSION_MILESTONE_4_MLX_REVIEW.md)
- [Independent milestone 4 technical review](../archive/conversion/MODEL_CONVERSION_MILESTONE_4_REVIEW.md)
- [Milestone 5: LLaMBA and the Mamba2 mixer](../archive/conversion/MODEL_CONVERSION_MILESTONE_5.md)
- [Milestone 5 technical review](../archive/conversion/MODEL_CONVERSION_MILESTONE_5_REVIEW.md)
- [Milestone 6: ModernBERT sequence classifiers](../archive/conversion/MODEL_CONVERSION_MILESTONE_6.md)
- [Milestone 6 technical review](../archive/conversion/MODEL_CONVERSION_MILESTONE_6_REVIEW.md)
- [Independent model conversion replacement plan](../archive/conversion/MODEL_CONVERSION_PLAN.md)
- [Model conversion plan technical review](../archive/conversion/MODEL_CONVERSION_PLAN_REVIEW.md)
- [Pi-assisted conversion adapter builder](../archive/conversion/PI_ADAPTER_BUILDER_PLAN.md)
- [Conversion benchmarks](../benchmarks/CONVERSION_BENCHMARKS.md) · Benchmark

## Experiment packets and aggregate data

These source folders contain experiment protocols, aggregate measurements, patches and harnesses. An interrupted or invalid attempt may still have a receipt.

- [Archived experiment packet index](../archive/experiments/README.md)
- [Benchmark aggregate data directory](../benchmarks/data)
- [Archived-run aggregate data directory](../archive/runs/data)
- [Flash-Next idle-wake patch](../archive/experiments/2026-11-07-flashnext-idle-wake)
- [Flash-Next MoE and cadence tools](../archive/experiments/2026-11-07-flashnext-moe)
- [Flash-Next concurrent encoder pilot patch](../archive/experiments/2026-11-07-flashnext-pilot)
- [Cross-backend idle-wake patch](../archive/experiments/2026-11-08-cross-backend-idle-wake)
- [Lily long-context evidence](../archive/experiments/2026-11-15-lily-long-context)
- [Lily natural-response harnesses](../archive/experiments/2026-11-15-lily-long-responses)
- [Lily MTP reconciliation harnesses](../archive/experiments/2026-11-15-lily-mtp-reconciliation)
- [Lily one-draft comparison harnesses](../archive/experiments/2026-11-15-lily-mtp1)
- [Lily sampling-specific MTP policy harnesses](../archive/experiments/2026-11-15-lily-sampling-mtp-policy)
- [Lily small-row verifier harnesses](../archive/experiments/2026-11-15-lily-verifier)
- [llama.cpp v1 and Lily baseline harnesses](../archive/experiments/2026-11-15-llama-v1-lily)
- [Lily projection reuse adoption harnesses](../archive/experiments/2026-11-16-lily-projection-adoption)
- [Lily verify-row profiling and trial harnesses](../archive/experiments/2026-11-19-lily-verify-rows)
- [Qwen quiet runs, diagnostics and contaminated attempts](../archive/experiments/clean-qwen-2026-10-20)
- [Current-engine admission, interruptions and resumed measurements](../archive/experiments/current-engines-2026-10-21)
- [Decision Playground review and validation](../archive/experiments/decision-playground-2026-10-22)
- [DeepInfer adoption and reverted runtime trials](../archive/experiments/deepinfer-adoption-2026-10-21)
- [MLX generation changes, numerical controls and invalid attempts](../archive/experiments/mlx-optimization-2026-10-20)
- [Native GLM and Qwen hybrid packet](../archive/experiments/native-hybrid-2026-10-28)
- [Qwen3 GGUF import aggregates](../archive/experiments/qwen3-gguf-2026-10-13)
- [Qwen3 GGUF prefill aggregates](../archive/experiments/qwen3-prefill-2026-10-13)
- [Rejected Qwen3.8 F32 hyper-connection fusion](../archive/experiments/qwen38-f32-hc-2026-10-28)
- [Stock engine scoring, generation and serving evidence](../archive/experiments/stock-engines-2026-10-19)
- [TensorFold JevK5 aggregates](../archive/experiments/tensorfold-jevk5-2026-10-16)
- [TensorFold Qwen3.5 aggregates](../archive/experiments/tensorfold-stock-qwen35-2026-10-16)
- [Native chat-template rendering aggregates](../benchmarks/NATIVE_CHAT_TEMPLATE_2026_11_03.json)
- [Matched OLMo 3 and Gemma 4 aggregates](../benchmarks/OLMO3_GEMMA4_MATCHED_2026_10_28.json)
- [Repaired OLMo 3 and Gemma 4 Q4 aggregates](../benchmarks/olmo3-gemma4-q4-repaired-2026-10-29.json)
- [OLMo 3 and Gemma 4 source-fusion aggregates](../benchmarks/olmo3-gemma4-source-fusions-2026-11-02.json)
- [Qwen guided-memory aggregates](../archive/runs/data/2026-11-08-qwen-guided-memory.json)
- [Lily verifier projection reuse aggregates](../archive/runs/data/2026-11-16-lily-projection-reuse.json)
- [Pi persistent manual connection aggregates](../archive/runs/data/2026-11-13-pi-persistent-default.json)

## Branch-only research

- [Model experiments on branches and in pull requests](branch-research.md) · Edge0, Inkling, GLM, Flash-Next, MaxxBench, scoring studies and conversion
- [Individual branch reports by family](branch-sources.md) · Direct commit-pinned links

- [Flash-Next speed review, fresh and agent prompt grid studies, and kernel follow-up](../PERFORMANCE_PLAYBOOK.md#records) · 2026-11-17 branch records linked from the playbook
