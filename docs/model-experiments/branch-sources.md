# Branch research report links

The [branch research inventory](branch-research.md) identified 39 benchmark reports, 46 experiment records and 58 archive documents absent from its main snapshot. Those 143 paths are linked below, along with 10 research documents from the top-level docs directory. Repeated paths across refs appear once. Benchmarks and experiment records with similar titles remain separate where one holds the measurements and the other records qualification or disposition. Retired and superseded attempts keep their original links and measurement boundaries.

Each link points to a commit available on GitHub at the inventory check. The [research summary](branch-research.md) covers source-lineage exceptions, including the local GLM T8 commit whose benchmark already appears on main.

## Research plans, provenance and validation

- **Inkling 2 Mini** · [Original-checkpoint implementation plan](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/INKLING_2_MINI_PLAN.md) · [Performance follow-up proposals and measured baseline](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/INKLING_2_MINI_PERFORMANCE_FOLLOWUP.md)
- **Qwen3.8 typed Uzu execution** · [Execution scope and controls](https://github.com/jshahbazi/stark/blob/174854991e022ff7283dfa6f0f951114de708ac1/docs/QWEN38_UZU_EXECUTION.md) · [Implementation plan](https://github.com/jshahbazi/stark/blob/174854991e022ff7283dfa6f0f951114de708ac1/docs/QWEN38_UZU_EXECUTION_PLAN.md) · [Source and patch provenance](https://github.com/jshahbazi/stark/blob/174854991e022ff7283dfa6f0f951114de708ac1/docs/QWEN38_UZU_PROVENANCE.md) · [Stage profiling procedure](https://github.com/jshahbazi/stark/blob/174854991e022ff7283dfa6f0f951114de708ac1/docs/QWEN38_STAGE_PROFILING.md)
- **Lalamo conversion parity** · [Milestone 7 conversion and timing evidence](https://github.com/jshahbazi/stark/blob/3e63524c649cdff6d654be042afe8496b92d9af5/docs/MODEL_CONVERSION_MILESTONE_7.md) · [Technical review](https://github.com/jshahbazi/stark/blob/3e63524c649cdff6d654be042afe8496b92d9af5/docs/MODEL_CONVERSION_MILESTONE_7_REVIEW.md)
- **OLMo 3 and Gemma 4** · [Historical model experiment register](https://github.com/jshahbazi/stark/blob/466af7bd85733df430502ccb3362b06ba1514583/docs/MODEL_EXPERIMENTS.md)
- **JevK5 BF16** · The [former validation page](https://github.com/jshahbazi/stark/blob/7f56e28b88fe4baec6998728a6447944696dadd2/docs/VALIDATION.md) and the [canonical scoring parity record](../validation/SCORING_PARITY.md) describe the same 243-case run.

The two remaining top-level paths cover GUI advanced options validation and website deployment.

## Edge0-145B

### Benchmarks

- [Arm attribution](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/benchmarks/2026-11-13-edge0-arm-attribution.md)
- [Expert windows](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/benchmarks/2026-11-13-edge0-expert-windows.md)
- [Host interference](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/benchmarks/2026-11-13-edge0-host-interference.md)
- [MTP paging](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/benchmarks/2026-11-13-edge0-mtp-paging.md)
- [Observed speed](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/benchmarks/2026-11-13-edge0-observed-speed.md)
- [Quiet window](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/benchmarks/2026-11-13-edge0-quiet-window.md)
- [Singleton speed](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/benchmarks/2026-11-13-edge0-singleton-speed.md)
- [Worker memory](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/benchmarks/2026-11-13-edge0-worker-memory.md)
- [Worker MTP](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/benchmarks/2026-11-13-edge0-worker-mtp.md)
- [Context loading](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/benchmarks/2026-11-16-edge0-context-loading.md)
- [Direct routes](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/benchmarks/2026-11-17-edge0-direct-routes.md)
- [Prefill batches](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/benchmarks/2026-11-17-edge0-prefill-batches.md)
- [Token rows](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/benchmarks/2026-11-17-edge0-token-rows.md)

### Experiment records

- [Block readers](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-12-edge0-block-readers.md)
- [Compatibility](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-12-edge0-compatibility.md)
- [Metal](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-12-edge0-metal.md)
- [Native tokenizer](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-12-edge0-native-tokenizer.md)
- [Reference forward](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-12-edge0-reference-forward.md)
- [Source preparation](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-12-edge0-source-preparation.md)
- [Arm attribution](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-13-edge0-arm-attribution.md)
- [Expert windows](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-13-edge0-expert-windows.md)
- [Host interference](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-13-edge0-host-interference.md)
- [MTP batched worker](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-13-edge0-mtp-batched-worker.md)
- [MTP head](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-13-edge0-mtp-head.md)
- [MTP integration](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-13-edge0-mtp-integration.md)
- [MTP paging](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-13-edge0-mtp-paging.md)
- [MTP worker](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-13-edge0-mtp-worker.md)
- [MTP](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-13-edge0-mtp.md)
- [Observed speed](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-13-edge0-observed-speed.md)
- [Quiet window](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-13-edge0-quiet-window.md)
- [Singleton speed](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-13-edge0-singleton-speed.md)
- [Worker memory](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-13-edge0-worker-memory.md)
- [Context loading](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-16-edge0-context-loading.md)
- [Direct routes](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-17-edge0-direct-routes.md)
- [Prefill batches](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-17-edge0-prefill-batches.md)
- [Token rows](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/model-experiments/2026-11-17-edge0-token-rows.md)

### Archived runs and protocols

- [License intake](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/archive/runs/2026-11-12-edge0-license-intake.md)
- [Reference performance ideas](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/archive/runs/2026-11-13-edge0-reference-performance-ideas.md)
- [Performance outlook](https://github.com/jshahbazi/stark/blob/5d3681fd4242ba3f98524ade0aedca2161e5b1bc/docs/archive/runs/2026-11-17-edge0-performance-outlook.md)

## Inkling 2 Mini

### Benchmarks

- [M5 Max](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/benchmarks/2026-11-20-inkling-mini-m5-max.md)
- [Native slice](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/benchmarks/2026-11-21-inkling-mini-native-slice.md)
- [Native worker](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/benchmarks/2026-11-21-inkling-mini-native-worker.md)
- [Pinned oracle](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/benchmarks/2026-11-21-inkling-mini-pinned-oracle.md)
- [Short projection](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/benchmarks/2026-11-21-inkling-mini-short-projection.md)

### Experiment records

- [Milestone 1](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/model-experiments/2026-11-20-inkling-mini-milestone-1.md)
- [DSpark](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/model-experiments/2026-11-21-inkling-mini-dspark.md)
- [Faster prefill](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/model-experiments/2026-11-21-inkling-mini-faster-prefill.md)
- [Full target](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/model-experiments/2026-11-21-inkling-mini-full-target.md)
- [Grouped NVFP4](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/model-experiments/2026-11-21-inkling-mini-grouped-nvfp4.md)
- [Matched prefill](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/model-experiments/2026-11-21-inkling-mini-matched-prefill.md)
- [Native slice](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/model-experiments/2026-11-21-inkling-mini-native-slice.md)
- [Packed prefill](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/model-experiments/2026-11-21-inkling-mini-packed-prefill.md)
- [Pinned oracle](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/model-experiments/2026-11-21-inkling-mini-pinned-oracle.md)
- [Prefill budget](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/model-experiments/2026-11-21-inkling-mini-prefill-budget.md)
- [Prefill profile](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/model-experiments/2026-11-21-inkling-mini-prefill-profile.md)
- [Short projection](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/model-experiments/2026-11-21-inkling-mini-short-projection.md)
- [Vision](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/model-experiments/2026-11-21-inkling-mini-vision.md)
- [Worker](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/model-experiments/2026-11-21-inkling-mini-worker.md)

### Archived runs and protocols

- [Inkling mini, NVFP4_PROBE](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/archive/experiments/2026-11-20-inkling-mini/NVFP4_PROBE.md)
- [Inkling oracle](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/archive/experiments/2026-11-21-inkling-oracle/README.md)
- [Inkling vision](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/archive/experiments/2026-11-21-inkling-vision/README.md)
- [Native retrospective](https://github.com/jshahbazi/stark/blob/e667c653caaab47c8d1aa596901297bf42efdada/docs/archive/runs/2026-11-21-inkling-mini-native-retrospective.md)

## GLM-5.3-Flash

### Benchmarks

- [Llama full model tests](https://github.com/jshahbazi/stark/blob/6c040236a580171779dc61af13a828ecf58bff5b/docs/benchmarks/2026-11-15-glm53-llama-full-model-tests.md)
- [Llama v1 GLM-5.3 assessment](https://github.com/jshahbazi/stark/blob/6c040236a580171779dc61af13a828ecf58bff5b/docs/benchmarks/2026-11-15-llama-v1-glm53-assessment.md)
- [Matched partial](https://github.com/jshahbazi/stark/blob/cdee063c51e98857b1e847300700b59f8452fa96/docs/benchmarks/GLM53_MATCHED_PARTIAL_2026_11_03.md)
- [MLX baseline](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/benchmarks/GLM53_MLX_BASELINE_2026_11_01.md)
- [Native optimization](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/benchmarks/GLM53_NATIVE_OPTIMIZATION_2026_11_01.md)
- [Prefill KDA optimization](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/benchmarks/GLM53_PREFILL_KDA_OPTIMIZATION_2026_11_01.md)
- [Website protocol](https://github.com/jshahbazi/stark/blob/cdee063c51e98857b1e847300700b59f8452fa96/docs/benchmarks/GLM53_WEBSITE_PROTOCOL_2026_11_03.md)

### Experiment records

- [Llama v1 GLM-5.3 full model tests](https://github.com/jshahbazi/stark/blob/6c040236a580171779dc61af13a828ecf58bff5b/docs/model-experiments/2026-11-15-llama-v1-glm53-full-model-tests.md)

### Archived runs and protocols

- [Fused control](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/archive/experiments/glm53-control-fused-2026-11-01/README.md)
- [IQ arithmetic](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/archive/experiments/glm53-iq-arithmetic-2026-11-01/README.md)
- [IQ factored](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/archive/experiments/glm53-iq-factored-2026-11-01/README.md)
- [IQ shared bound](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/archive/experiments/glm53-iq-shared-bound-20261102/README.md)
- [MTP feasibility](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/archive/experiments/glm53-mtp-feasibility-20261102/README.md)
- [MTP feasibility, cache-v7c](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/archive/experiments/glm53-mtp-feasibility-20261102/cache-v7c.md)
- [MTP feasibility, profile-v4](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/archive/experiments/glm53-mtp-feasibility-20261102/profile-v4.md)
- [MTP feasibility, reuse-v6](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/archive/experiments/glm53-mtp-feasibility-20261102/reuse-v6.md)
- [MTP feasibility, wait-v5](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/archive/experiments/glm53-mtp-feasibility-20261102/wait-v5.md)
- [Packed finite](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/archive/experiments/glm53-packed-finite-2026-11-01/README.md)
- [Q5 MMA](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/archive/experiments/glm53-q5-mma-20261102/README.md)
- [T2 feasibility](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/archive/experiments/glm53-t2-feasibility-20261102/README.md)
- [T2 feasibility, compact-v2](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/archive/experiments/glm53-t2-feasibility-20261102/compact-v2.md)
- [T2 feasibility, dual-input-v3](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/archive/experiments/glm53-t2-feasibility-20261102/dual-input-v3.md)
- [T2 feasibility, screened-v4c](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/archive/experiments/glm53-t2-feasibility-20261102/screened-v4c.md)
- [Main refresh](https://github.com/jshahbazi/stark/blob/667a7ef69248b0405e4b94ddc2ac9ec41bd6e7ac/docs/archive/runs/2026-11-02-glm53-main-refresh.md)
- [Actual vision components](https://github.com/jshahbazi/stark/blob/327262720406a0b12ed998c99e354b03cf45c284/docs/archive/runs/2026-11-09-glm-actual-vision-components.md)
- [Bounded image execution](https://github.com/jshahbazi/stark/blob/327262720406a0b12ed998c99e354b03cf45c284/docs/archive/runs/2026-11-09-glm-bounded-image-execution.md)
- [First use control diagnostic](https://github.com/jshahbazi/stark/blob/327262720406a0b12ed998c99e354b03cf45c284/docs/archive/runs/2026-11-09-glm-first-use-control-diagnostic.md)
- [First use pressure audit](https://github.com/jshahbazi/stark/blob/327262720406a0b12ed998c99e354b03cf45c284/docs/archive/runs/2026-11-09-glm-first-use-pressure-audit.md)
- [First weight mapping plan](https://github.com/jshahbazi/stark/blob/327262720406a0b12ed998c99e354b03cf45c284/docs/archive/runs/2026-11-09-glm-first-weight-mapping-plan.md)
- [Functional pageout screen](https://github.com/jshahbazi/stark/blob/327262720406a0b12ed998c99e354b03cf45c284/docs/archive/runs/2026-11-09-glm-functional-pageout-screen.md)
- [Hash backend](https://github.com/jshahbazi/stark/blob/327262720406a0b12ed998c99e354b03cf45c284/docs/archive/runs/2026-11-09-glm-hash-backend.md)
- [Image admission](https://github.com/jshahbazi/stark/blob/327262720406a0b12ed998c99e354b03cf45c284/docs/archive/runs/2026-11-09-glm-image-admission.md)
- [Image parity](https://github.com/jshahbazi/stark/blob/327262720406a0b12ed998c99e354b03cf45c284/docs/archive/runs/2026-11-09-glm-image-parity.md)
- [Image worker integration](https://github.com/jshahbazi/stark/blob/327262720406a0b12ed998c99e354b03cf45c284/docs/archive/runs/2026-11-09-glm-image-worker-integration.md)
- [Mapping lifecycle diagnostic](https://github.com/jshahbazi/stark/blob/327262720406a0b12ed998c99e354b03cf45c284/docs/archive/runs/2026-11-09-glm-mapping-lifecycle-diagnostic.md)
- [Metal upstream memory review](https://github.com/jshahbazi/stark/blob/327262720406a0b12ed998c99e354b03cf45c284/docs/archive/runs/2026-11-09-glm-metal-upstream-memory-review.md)
- [Native image-conditioned generation](https://github.com/jshahbazi/stark/blob/327262720406a0b12ed998c99e354b03cf45c284/docs/archive/runs/2026-11-09-glm-native-image-generation.md)
- [Native vision components](https://github.com/jshahbazi/stark/blob/327262720406a0b12ed998c99e354b03cf45c284/docs/archive/runs/2026-11-09-glm-native-vision-components.md)
- [Owned upload source candidate](https://github.com/jshahbazi/stark/blob/327262720406a0b12ed998c99e354b03cf45c284/docs/archive/runs/2026-11-09-glm-owned-upload-source-candidate.md)

## Qwen, Flash-Next and Lily

### Benchmarks

- [Unsloth v3.1 Flash-Next comparison](https://github.com/jshahbazi/stark/blob/9b9eb6d8cc0cbb29269df27d53fb7d3f52d4e331/docs/benchmarks/2026-11-06-unsloth-v31-flash-next-comparison.md)
- [Optimization retrospective](https://github.com/jshahbazi/stark/blob/6ca9c584571048a908cbebc0fe3ae84e6c69ace8/docs/benchmarks/2026-11-08-flashnext-optimization-retrospective.md)
- [Speed review](https://github.com/jshahbazi/stark/blob/2e3058d15fc67ccde7608861a80e5368b926acd8/docs/benchmarks/2026-11-17-flashnext-speed-review.md)
- [Uzu rejected tuning](https://github.com/jshahbazi/stark/blob/174854991e022ff7283dfa6f0f951114de708ac1/docs/benchmarks/QWEN38_UZU_REJECTED_TUNING_2026_10_30.md)
- [Qwen3 long prefill candidate](https://github.com/jshahbazi/stark/blob/0c49a4db82ef3dfaf2a9132668f9ab808f8aa0f8/docs/benchmarks/QWEN3_LONG_PREFILL_CANDIDATE_2026_11_01.md)
- [Qwen3 long prefill timing](https://github.com/jshahbazi/stark/blob/0c49a4db82ef3dfaf2a9132668f9ab808f8aa0f8/docs/benchmarks/QWEN3_LONG_PREFILL_TIMING_2026_11_01.md)

### Experiment records

- [Optimization](https://github.com/jshahbazi/stark/blob/6ca9c584571048a908cbebc0fe3ae84e6c69ace8/docs/model-experiments/2026-11-08-flashnext-optimization.md)
- [Coding agent grid](https://github.com/jshahbazi/stark/blob/2e3058d15fc67ccde7608861a80e5368b926acd8/docs/model-experiments/2026-11-17-flashnext-coding-agent-grid.md)
- [Fresh coding grid](https://github.com/jshahbazi/stark/blob/2e3058d15fc67ccde7608861a80e5368b926acd8/docs/model-experiments/2026-11-17-flashnext-fresh-coding-grid.md)
- [Kernel and prefix follow-up](https://github.com/jshahbazi/stark/blob/2e3058d15fc67ccde7608861a80e5368b926acd8/docs/model-experiments/2026-11-17-flashnext-kernel-and-prefix-follow-up.md)
- [Midpoint attribution](https://github.com/jshahbazi/stark/blob/2e3058d15fc67ccde7608861a80e5368b926acd8/docs/model-experiments/2026-11-17-flashnext-midpoint-attribution.md)
- [Midpoint](https://github.com/jshahbazi/stark/blob/2e3058d15fc67ccde7608861a80e5368b926acd8/docs/model-experiments/2026-11-17-flashnext-midpoint.md)
- [Speed experiments](https://github.com/jshahbazi/stark/blob/2e3058d15fc67ccde7608861a80e5368b926acd8/docs/model-experiments/2026-11-17-flashnext-speed-experiments.md)

### Archived runs and protocols

- [GDN fusion](https://github.com/jshahbazi/stark/blob/635db78cad4247b7b51c94351a55306ad9a78cfd/docs/archive/experiments/qwen38-gdn-fusion-2026-10-31/README.md)
- [GDN next](https://github.com/jshahbazi/stark/blob/635db78cad4247b7b51c94351a55306ad9a78cfd/docs/archive/experiments/qwen38-gdn-next-2026-10-31/README.md)
- [GDN prework](https://github.com/jshahbazi/stark/blob/635db78cad4247b7b51c94351a55306ad9a78cfd/docs/archive/experiments/qwen38-gdn-prework-2026-10-31/README.md)
- [64K admission](https://github.com/jshahbazi/stark/blob/174854991e022ff7283dfa6f0f951114de708ac1/docs/archive/runs/2026-10-30-qwen38-64k-admission.md)
- [Experimental full vocabulary](https://github.com/jshahbazi/stark/blob/174854991e022ff7283dfa6f0f951114de708ac1/docs/archive/runs/2026-10-30-qwen38-experimental-full-vocabulary.md)
- [Split-K correction validation](https://github.com/jshahbazi/stark/blob/174854991e022ff7283dfa6f0f951114de708ac1/docs/archive/runs/2026-10-30-qwen38-split-k-correction-validation.md)
- [Split-K precision](https://github.com/jshahbazi/stark/blob/174854991e022ff7283dfa6f0f951114de708ac1/docs/archive/runs/2026-10-30-qwen38-split-k-precision.md)
- [Uzu paused checkpoint](https://github.com/jshahbazi/stark/blob/174854991e022ff7283dfa6f0f951114de708ac1/docs/archive/runs/2026-10-30-qwen38-uzu-paused-checkpoint.md)
- [MoE image-loaded text parity](https://github.com/jshahbazi/stark/blob/4d8c2f54cfc8707574d7181dc560f5ee53dcd4a0/docs/archive/runs/2026-11-09-qwen35-moe-image-text-parity.md)

## MaxxBench and Flint

### Benchmarks

- [MaxxBench native pilot](https://github.com/jshahbazi/stark/blob/cee1d7691c2fbdc3c242ec02b17148d300ec2237/docs/benchmarks/2026-11-08-maxxbench-native-pilot.md)
- [MaxxBench runtime qualification](https://github.com/jshahbazi/stark/blob/cee1d7691c2fbdc3c242ec02b17148d300ec2237/docs/benchmarks/2026-11-08-maxxbench-runtime-qualification.md)
- [MaxxBench](https://github.com/jshahbazi/stark/blob/cee1d7691c2fbdc3c242ec02b17148d300ec2237/docs/benchmarks/MAXXBENCH.md)

### Archived runs and protocols

- [Cliffy source platform](https://github.com/jshahbazi/stark/blob/cee1d7691c2fbdc3c242ec02b17148d300ec2237/docs/archive/runs/2026-11-08-cliffy-source-platform.md)
- [EICRUD source platform](https://github.com/jshahbazi/stark/blob/cee1d7691c2fbdc3c242ec02b17148d300ec2237/docs/archive/runs/2026-11-08-eicrud-source-platform.md)
- [Flint prefix core qualification](https://github.com/jshahbazi/stark/blob/cee1d7691c2fbdc3c242ec02b17148d300ec2237/docs/archive/runs/2026-11-08-flint-prefix-core-qualification.md)
- [Flint prefix reuse](https://github.com/jshahbazi/stark/blob/cee1d7691c2fbdc3c242ec02b17148d300ec2237/docs/archive/runs/2026-11-08-flint-prefix-reuse.md)
- [GLM runtime baseline](https://github.com/jshahbazi/stark/blob/cee1d7691c2fbdc3c242ec02b17148d300ec2237/docs/archive/runs/2026-11-08-glm-maxxbench-runtime-baseline.md)
- [Lily thinking parser boundary](https://github.com/jshahbazi/stark/blob/cee1d7691c2fbdc3c242ec02b17148d300ec2237/docs/archive/runs/2026-11-08-lily-thinking-parser-boundary.md)
- [MaxxBench verifier platform](https://github.com/jshahbazi/stark/blob/cee1d7691c2fbdc3c242ec02b17148d300ec2237/docs/archive/runs/2026-11-08-maxxbench-verifier-platform.md)

## Decision scoring

### Benchmarks

- [Drex Human100 comparison](https://github.com/jshahbazi/stark/blob/c59721dba7ef7b8612b0b7243410ec2d25e75fe1/docs/benchmarks/2026-11-08-drex-human100-comparison.md)
- [Drex Human100 registration](https://github.com/jshahbazi/stark/blob/c59721dba7ef7b8612b0b7243410ec2d25e75fe1/docs/benchmarks/2026-11-08-drex-human100-registration.md)
- [Drex JevK5 comparison](https://github.com/jshahbazi/stark/blob/c59721dba7ef7b8612b0b7243410ec2d25e75fe1/docs/benchmarks/2026-11-08-drex-jevk5-comparison.md)
- [Mercury JevBench attempt](https://github.com/jshahbazi/stark/blob/c59721dba7ef7b8612b0b7243410ec2d25e75fe1/docs/benchmarks/2026-11-08-mercury-jevbench-attempt.md)
- [Mercury JevBench comparison](https://github.com/jshahbazi/stark/blob/c59721dba7ef7b8612b0b7243410ec2d25e75fe1/docs/benchmarks/2026-11-08-mercury-jevbench-comparison.md)

## Other model research

### Experiment records

- [Maxwell compatibility](https://github.com/jshahbazi/stark/blob/fd1ed2a53cba2d7c8890bde852dcbd956356ce28/docs/model-experiments/2026-11-17-maxwell-compatibility.md)

### Archived runs and protocols

- [Llama moe CPU share assessment](https://github.com/jshahbazi/stark/blob/60e2bc00270515d1604e7cb25196b8ff3a7fc826/docs/archive/runs/2026-11-07-llama-moe-cpu-share-assessment.md)
- [OLMo 3 Gemma 4 findings](https://github.com/jshahbazi/stark/blob/466af7bd85733df430502ccb3362b06ba1514583/docs/archive/runs/2026-11-08-olmo3-gemma4-findings.md)
- [Model agent crash recovery](https://github.com/jshahbazi/stark/blob/2f367d117f59091cbe8f3dca93d4afc61794fa2b/docs/archive/runs/2026-11-09-model-agent-crash-recovery.md)
- [Beginner unsure evidence](https://github.com/jshahbazi/stark/blob/3716e381afbc93c8e0131b22170fdbdc6cd7671f/docs/archive/runs/2026-11-11-beginner-unsure-evidence.md)
