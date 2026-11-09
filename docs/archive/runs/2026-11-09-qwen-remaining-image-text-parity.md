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

### Registered bounded MoE timing recipe

Before timing, QNIP-03 fixes 16 output tokens for its short matched affine-MoE
comparison, with the same context, input, greedy seed, ABBA loads and repetitions
above. Prior reference-graph cache controls already showed substantial per-token
expert overhead. This shorter workload bounds resource occupancy without changing
settings after observing candidate speed. Small dense controls retain 64 outputs.
Cache/lifecycle correctness uses eight outputs independently of the speed run.

## Current results and remaining gates

The [source-bound benchmark](../../benchmarks/2026-11-09-qwen-remaining-image-text-parity.md)
records passing short MTP0 speed comparisons for the three exact older headless
exports. Their candidate image/text decode medians are 304.108/303.618 tok/s for
BF16 0.8B, 661.392/659.888 for affine 0.8B, and 39.960/39.988 for affine MoE.
Independent receipt recomputation agrees. This qualifies the recorded speed
workload, not each complete family or its source predictor.

The fast graph now covers BF16 fixed/dynamic, affine dense and affine MoE.
Actual loaded tests prove canonical ordinary-text configuration and immutable
shared storage, including 18,119,393,280 shared MoE expert-bank bytes. Retained
prefixes carry graph identity; changed-graph state cannot be reused. The frozen
27B path and its prior evidence remain separate and intact.

The first nonperiodic 1,792-token/context-2,048 BF16 cache matrix passes eleven
cases. Its next case mistakenly expects reuse after changing scope; the worker
correctly misses. The corrected full recipe is sealed but not yet run. All
three full lifecycle matrices, including real-image transitions, remain open.

### Actual predictor inventory and implementation

| Source | Pinned revision | Index tensors / predictor tensors | Disposition |
| --- | --- | ---: | --- |
| Official Qwen3.5-0.8B BF16 | `2fc06364715b967f1860aea9cf38778875588b17` | 488 / 15 | Complete local source. Retain the 12 canonical head tensors; old headless image export is not source absence. Actual head admission/execute/speed remains open. |
| MLX Qwen3.5-0.8B 4-bit | `5d894f8cc4ef3e6c88537bf3746ed262f549da6a` | 847 / 0 | Exact source has no head payload despite config declaration. No MTP capability is claimed. |
| MLX Qwen3.5-35B-A3B 4-bit | `1e20fd8d42056f870933bf98ca6211024744f7ec` | 2,090 / 0 | Exact source has no head payload; fast expert storage and short speed comparison pass. |
| Official Qwen3.5-35B-A3B BF16 | `59d61f3ce65a6d9863b86d2e96597125219dc754` | 1,811 / 785 | Index fetched within approved one-MiB metadata bound; full payload absent. Implemented per-expert canonical bank planning and strict native contract; runtime qualification blocked. |

BF16 expert index SHA-256:
`d8d0b7ca4e61ae107e3e87a3ff21136b3ac7c789e64bb24267227ca804e04205`.
Its 768 expert matrices are separate per-expert up/gate/down tensors; they cannot
be treated as the dense head or assumed to share the trunk's source layout.
Tiny fixtures preserve all source matrices and raw norm gain offset one.
Incomplete source heads and dedicated predictor embeddings fail closed.

The pinned oMLX implementation at `2165e75726ba83d9651dbaced61c97728da77bba`
uses post-final-norm target features in its actual driver and priming path.
The BF16 candidate uses this proposal feature convention with target rejection
verification; this is not a claim of original-training feature equivalence.
Native raw BF16 scales use F32 gain-plus-one, rather than inheriting the affine
source's pre-folded norm gains or its rounding evidence. New BF16 execution and
performance need their own tests. MTP remains controlled by the request policy.

The first appended BF16 candidate preserved a planner text RoPE in the image
decoder and correctly failed admission. A corrected metadata derivative keeps
the exact reference image decoder while its predictor keeps ordinary text
RoPE. Native header validation then exposed a real bug: its BF16 spec whitelist
accepted decoder matrices but rejected legitimate predictor matrix specs.
The fix binds specs for both namespaces, with missing/changed/extra-spec
negative tests. Conversion of a retained pinned BF16 head also selects schema 3
before component hashing, preserving the frozen older headless schema-2 tuple.
These latest native changes require a new compiled execution check.

Independent source review also found that the MoE converter did not apply the
image target RoPE rewrite. It now rewrites both dense and MoE image targets,
leaving the separately cloned predictor's ordinary text RoPE intact. Tiny CLI
fixtures check headed and headless BF16 MoE outputs. The five affected converter
test modules pass 190 tests; Ruff and package-scoped native formatting checks
pass. Frozen headless reference JSON remains unchanged; explicitly headless
source fixtures still reproduce it, while incomplete-head fixtures now assert
the intended rejection. These source/component checks do not replace the blocked
native build and actual predictor run.

### Resource stop and unqualified work

All model workers are closed. After the speed controls and partial cache run,
available disk fell below the fixed eight-GiB reserve. This run's test and worker
build targets were sealed and retired; bounded explicitly authorized old Qwen
generated outputs were also retired, preserving standalone binaries and all
models. Free space remained below the reserve and varied while heavy work was
stopped. Native builds, model loads and downloads remain stopped until the
reserve plus the next phase's bounded allocation is available.

Actual BF16 head execution/cache/MTP0+3 speed, full cache matrices, dense 8-bit,
other sizes, BF16 MoE full payload, and retained ddalcu execution remain open.
The installed ddalcu source is complete (101 shards, 75,303,252,216 bytes), but
has no admitted image manifest. Any upgrade must use isolated metadata/hardlink
views and preserve the installed artifact; no backbone copy or model deletion
has been performed. Lily's prior proof does not qualify ddalcu.
