# Model launch and performance triage — 2026-11-09

Initial read-only source and record review on baseline `1ac5b9dc`. A subsequent
scalar-workspace candidate is tracked as [20261109-QMEM-01](../../model-experiments/2026-11-09-qwen-attention-memory.md).
Historical measurements remain bound to their stated source, artifact, recipe
and machine and are not claims about current speed.

Completed correctness and component checks are recorded separately in the
[validation record](2026-11-09-model-launch-correctness.md). GLM default and
optimized prefix restoration passed on the complete synthetic fixture; the
production GLM artifact qualification remains incomplete.

## Findings

| Topic | Disposition | Evidence and blocker / next step |
| --- | --- | --- |
| GLM optimized recipe on catalog launch | **Current gap; experiment proposed.** | The catalog exposes `glm53-q2`; `desktop.py` has native preparation routing, while the optimized recipe is an explicit wrapper (`scripts/with_glm53_optimized.sh`). The Nov. 3 catalog record says no real model load/inference was performed. The prefix run exercises GLM only under that wrapper, so it does not prove that a catalog-launched worker receives or benefits from the recipe. Compare a normal catalog launch's effective environment and loaded backend with an explicitly recipe-configured run before changing defaults. |
| GLM prefix reuse under default settings | **Qualification incomplete.** | `docs/archive/runs/2026-11-03-glm-qwen-prefix-reuse.md` reports 32 lifecycle/output checks under the optimized wrapper; it explicitly says this does not compare the complete recipe numerically with the default graph. Do not transfer that result to default launch. A default-recipe artifact-bound exactness/lifecycle run is required. |
| `qwen38-mixed` catalog execution profile | **Current route; performance experiment proposed.** | The catalog choice is PipeNetwork's `Qwen3.8-Flash-Next-MLX-mixed-4_8bit` at revision `b2c422f3…`; `guided_memory.py` assigns it `_qwen_scalar`, and the engine's scalar path retains mapped F32 activation/state buffers and a F32 split-attention kernel (`qwen_attention.metal`). The similarly named ddalcu MLX-Serve artifact is a separate catalog choice and its prior prefix result cannot qualify PipeNetwork. Establish a matched, artifact-bound baseline before describing this catalog route as fast or slow. |
| `qwen38-mixed` prefix reuse | **Currently blocked in the loaded-backend contract.** | `Forward::Qwen` is the PipeNetwork engine route, but `HybridModel::prefix_chunk_tokens`, `capture_prefix_at`, `take_prefix_snapshot` and `restore_prefix` handle GLM/GLM-EXL3/Lily and fall through to “prefix snapshots unavailable” for Qwen (`crates/stark-engine/src/engine/hybrid/mod.rs`). `docs/GENERATION.md` explicitly keeps the F32 Qwen graph unsupported. This is a concrete implementation blocker: there is no Qwen snapshot type or hook to save/restore its recurrent Delta state, QSA attention KV/indexer cache, PLE/history state at a prefill boundary. The separate Nov. 3 Qwen prefix proof binds ddalcu MLX-Serve mixed, not the PipeNetwork catalog artifact. Keep reuse disabled until those state owners and exact restore/lifecycle tests are implemented for this artifact; do not claim the old result transfers. |
| F32 Qwen attention buffers | **Allocation fix implemented; component validation passed.** | Baseline `AttentionControls::ensure_cache` allocated `scores` and `probabilities` as `capacity * heads` F32 elements with every KV growth. Normal decode stops using them at 256 tokens; split attention uses different command scratch. Candidate `20261109-QMEM-01` gives scalar attention independent lazy owners and its own stride, retaining at most 256 scalar rows per head on the normal split route. The scalar override and oversized-split fallback still grow on demand. Independent growth is reflected in memory admission. Allocation/inventory, kernel stride and synthetic scalar/split/forced-scalar full-forward checks passed. No throughput or observed peak-memory improvement is claimed. |
| Split-attention scratch vs 2,048 selected positions | **Current bounded full-context sizing; compaction is a proposed experiment.** | `qwen_graph.rs` computes `splits = ceil(count/256)` and requests `heads * splits * (head_dim + 2)` F32 partials at lines 285–287. The request is capped at 8,000,000 floats (about 30.5 MiB); above that threshold the route falls back to scalar attention. `qwen_graph_qsa_split_attention` traverses context splits and consults the `visible` mask; the indexer mask limits selected complete blocks to its configured budget (2,048 for the relevant artifact), plus the current partial tail, but does not shrink split count or partials below the cap. This is a bounded memory/performance tradeoff, not an unbounded allocation bug. Reducing it requires compacting selected token rows/positions before attention while preserving positional visibility, ordering, source arithmetic and the causal tail; measure separately and validate full outputs/state. |
| Prefault or prewarm behavior | **Proposed experiment; current behavior not established.** | The reviewed catalog and engine source did not provide evidence of a deliberate model-weight prefault/prewarm phase. Separate load, first-request and warmed-request costs on the exact artifact, and inspect page faults/residency. Do not infer startup policy from prior throughput runs, which often warmed kernels/weight views before timing. |
| Per-token host/dispatch overhead | **Proposed experiment.** | No current matched profile isolates host submission, synchronization, sampler/worker overhead and GPU compute per generated token for these catalog artifacts. Capture aggregate counters/timing on a sealed worker, then compare only like-for-like request and decode settings. Existing historical throughput observations are not current evidence. |
| GLM sort/dispatch overhead | **Proposed experiment, not an established bottleneck.** | Source contains sorting/selection work, but inspection alone does not show that its dispatch cost dominates. Profile the exact GLM catalog path and count/summarize dispatches and host-side sorting before optimizing. Preserve selection order and source-bound numerical behavior in any candidate. |
| Lily finer snapshot grid | **Proposed tradeoff study.** | Current hybrid prefix code and the onboarding contract require exact state restoration at supported boundaries; the existing Qwen/Lily record uses a 4,096-token grid. A finer grid could reduce replay work while increasing retained recurrent snapshot payload and capture/copy cost. Measure both costs and validate exact continuation, lifecycle clearing and memory admission before promotion. |

For now, keep the GLM optimized recipe opt-in. Its historical planning profile
is roughly 109 GB of model weights with a 4,096-token context; neither that
recipe nor its memory/performance observations justify changing the catalog
default without exact launch-path qualification and matched correctness,
memory and performance evidence.

## Safe isolated follow-ups

1. The scalar-workspace controls in `20261109-QMEM-01` passed. Production
   artifact memory/speed measurements remain open. Reducing full-context split
   partials needs a separate compaction design and model-bound parity evidence.
2. A catalog-launch configuration audit can establish whether the GLM recipe's
   environment reaches the native worker. This does not require a model run; any
   behavior/performance conclusion still requires the model-bound comparison.
3. The remaining performance questions need serialized, source-bound model
   experiments. The GLM catalog record explicitly lacks fresh execution evidence;
   the Qwen PipeNetwork artifact also needs separate qualification. No safe
   source-only speed fix is supported by this review.

## Governing records

- [Model onboarding](../../MODEL_ONBOARDING.md): prefix-state inventory,
  exactness/lifecycle qualification, artifact binding and matched cold/warm
  performance measurement.
- [GLM catalog integration](2026-11-03-glm-model-catalog.md): catalog and
  preparation evidence; explicitly no model execution or performance run.
- [GLM/Qwen prefix reuse](2026-11-03-glm-qwen-prefix-reuse.md): GLM evidence is
  under the optimized recipe; Qwen identity is ddalcu MLX-Serve, not PipeNetwork.
- [Qwen image experiment ledger](../../model-experiments/2026-11-08-qwen-images.md):
  artifact-specific distinctions and remaining qualification.
