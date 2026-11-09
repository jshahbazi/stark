# Remaining Qwen default-image text parity

Registered before implementation and model runs on `codex/qwen-image-parity-next`,
base `c18f1d2396d84e7658e855e0331d8e1378da808f`. Related evidence:
[QDIP-07](2026-11-09-qwen27-shared-weight-fast-text.md), which qualifies only the
exact Jundot 27B artifact. Governing requirements:
[default-image plan](../../IMAGE_DEFAULT_PARITY_PLAN.md) and
[model onboarding](../../MODEL_ONBOARDING.md).

## Attempts and fixed scope

| Attempt | Scope | Initial disposition |
| --- | --- | --- |
| 20261109-QNIP-01 | Official Qwen3.5-0.8B BF16 image-loaded ordinary text graph/cache/speed | Existing complete image artifact; inspect exact source head and ordinary text configuration/payload before enabling a shared graph |
| 20261109-QNIP-02 | Qwen3.5 dense affine headless image-loaded text, initially existing MLX 0.8B 4-bit | Existing complete image artifact; 8-bit and other artifacts require separate inventory and proof |
| 20261109-QNIP-03 | Qwen3.5-35B-A3B affine-v4 image-loaded ordinary expert graph/cache/speed | Existing complete image artifact; prior bounded cache evidence does not qualify text speed; prove expert banks share storage |
| 20261109-QNIP-04 | Official Qwen3.5 BF16 MoE expert predictor | Source has an actual expert head; absent full payload or implementation is unfinished support, never MTP N/A |
| 20261109-QNIP-05 | Retained Flash Next Lily and ddalcu default-route/lifecycle audit | Keep artifacts independent; run ddalcu only with complete admitted retained payload, otherwise record the exact missing resources |

Images remain enabled by default. Ordinary text must select the existing text
precision/kernel policy with immutable shared raw weights; image requests keep
the admitted image reference graph, zero drafts/proposals and prefix invalidation.
No full model/expert-bank duplication, image-off workaround or predictor stripping
can satisfy these attempts. Graph-specific controls and state remain independent.

## Fixed checks

- Prove canonical text target/head configuration and tensor-transform equality
  against the exact source's ordinary text conversion. Do not assume BF16 norm
  folding or MoE routing order matches the already-qualified affine 27B graph.
- Inventory actual source heads/configuration. Where an existing text head is
  supported, preserve it with images loaded and verify actual drafts. Missing
  head bytes in an older converted export are not evidence of source absence.
- Prove shared parameter and expert-bank storage, no image-buffer mutation,
  graph-identity rejection before GPU work, and bounded startup/request memory.
- Qualify nonperiodic near-capacity fresh/cache equality with exact aligned reuse,
  seeded same-mode controls, applicable MTP switches, scope/context/reset/error/
  cancellation recovery, and real image → fresh text → cached text transitions.
- Compare matching candidate text-only/default-image arms with the same weights,
  precision, sampler/seed, context and output budget. Start with 64 output tokens,
  context 2,048, greedy seed 42, ABBA loads, excluded warmup and three measured
  requests per policy/load. Record first-token, useful decode and whole-request
  medians/ranges separately. Use the larger of five percent and twice the baseline
  relative range as the registered loss bound; report the strict five-percent
  result too, without treating noisy samples as five-percent measurement precision.
- Record real draft counts, output hashes, paging/thermal status, startup cost and
  memory independently. Preserve failed/rejected/invalid attempts with reasons.
  Bounded/component results do not close untested artifacts or installed behavior.

Builds, model execution and downloads wait for the coordinator's resource slot.
No new full weight download or full export/copy is authorized without its stated
budget. Use existing payloads and honest metadata derivatives where possible.
Keep at least eight GiB disk free; seal and retire only this run's disposable
build output if needed. Models, private inputs/outputs, token vectors and native
artifacts remain ignored. Publish aggregate results and hashes only.

## Results

In progress. No new execution or speed claim is established by this registration.
