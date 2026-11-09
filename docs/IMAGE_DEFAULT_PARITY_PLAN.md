# Default image support: text parity worklist

This is the implementation and qualification worklist derived from the
[model status inventory](MODEL_IMAGE_STATUS.md). Source baseline: main
`af3e37aa`. [PR #110](https://github.com/jshahbazi/stark/pull/110) and
[PR #112](https://github.com/jshahbazi/stark/pull/112) are open at this snapshot.
Their branch results are evidence for their named artifacts, not shipped support.
This document adds requirements and planned work; no model was run to create it.

## Required product behavior

Image support is the default for an image-capable model. The default installed
artifact and loaded worker must preserve the model's supported text functions.
Users must not need to choose a text-only installation to recover correct
caching, text performance or supported text MTP. Keep the existing user/model
choice of whether to use text MTP; loading images must not disable that choice.
Image requests continue to use zero MTP drafts and invalidate retained text state.

Only admit image execution after its normal model-specific checks. If the
implementation is missing or unqualified, report that gap and finish the work;
do not hide a regression by making image support an optional escape hatch or
calling a text-only export the completed default for an image-capable source.
This is a development acceptance requirement, not a claim that a new automated
runtime qualification gate has been implemented.

The three required results are independent:

1. **Cache correctness:** exact fresh/cached outputs, correct reused counts and
   complete state restoration/invalidation on the default image-loaded worker.
2. **Text performance parity:** ordinary text with images loaded performs within
   the predeclared baseline variability/regression bound at matching settings.
   Measure first-token/prefill, useful decode and end-to-end latency separately.
3. **Text MTP parity where supported:** retain/load the head and verify actual
   draft execution with images resident, matching the supported text-only path.
   Image requests still have zero drafts/proposals, followed by valid text MTP.

## Family work queue

IDs identify work items, not experiments already executed. Create source-bound
attempt records before implementing or running each item. Priorities put known
loss of existing text functionality and reusable fixes before larger unqualified
models. A shared fix does not complete every checkpoint/quantization row.

| ID / priority | Family / actual artifact path | Current evidence | Cache work | Text performance with images loaded | Text MTP work | Completion scope |
| --- | --- | --- | --- | --- | --- | --- |
| IMGPAR-00 / P0 | Shared catalog, preparation and host defaults | Automatic image selection merged in #109; vision-worker prefix request is in #110 | Ensure the loaded worker advertises and receives supported cache scope; all image operations clear it | Audit every selectable default route; a faster text-only export must not silently replace the image-capable default | Preserve existing MTP preferences and capability; force zero drafts only for image requests | Merge/review #110/#112, then check the normal install/load/host flow, memory admission and capability reporting |
| IMGPAR-01 / P1 | Qwen3.5 dense BF16, converted vision | #110 has exact cache/lifecycle checks and bounded paired timing on official 0.8B only | Finish delivery of #110; qualify 2B/4B/9B/27B and fine-tunes separately, including transitions through images | Repeat matched loading-mode text checks for each admitted artifact/precision; 0.8B timing does not qualify other sizes | Inventory actual source heads and text support; preserve/implement declared supported heads rather than inferring N/A from an export that omitted them | Exact artifact and supported context/sampling policy; keep the 0.8B result as a reference |
| IMGPAR-02 / P1 | Qwen3.5 dense affine 4/8-bit decoder + BF16 tower | Small source-logit/image evidence exists; affine prefix parity is not qualified by the BF16 0.8B run | Qualify the new converted snapshot route for each quantization/layout | Compare the same packed target, precision and kernels with vision off/on; cached follow-ups separately | Converted image artifacts omit heads. Where the corresponding model/text artifact supports MTP, retain config/weights, expose text execution and qualify predictor/cache state | Each actual packed artifact; unsupported quantizers/towers need implementation first |
| IMGPAR-03 / P1 | Dense Qwen3.8-27B, `Jundot/Qwen3.8-27B-oQ4e-mtp` | Text MTP exists; converted image tower/worker evidence is separate from full decoder parity | Qualify image-loaded target/predictor prefix restoration and required MTP-selection keys | Compare the same 27B mixed-affine target with vision off/on at matching draft depth; check full decoder reference behavior | Implement retention/loading/execution of the existing text head in the vision artifact; image MTP stays off | Full 27B artifact, each admitted head/readout configuration and draft policy; no inheritance from Flash Next |
| IMGPAR-04 / P1 | Qwen3.8 Flash Next, retained Lily Q4-XL schema 6 | #112 passes bounded greedy/seeded/long-prefix, MTP-switch and image→text controls; short paired MTP2 text speed matches | Existing 4,096-token snapshots plus host enablement; finish delivery and cover remaining error/cancellation/host cases | Bounded MTP2 comparison completed on #112; extend to representative prompt/output lengths, MTP-off and normal context workloads before claiming whole-family parity | Resident head and text capability fixed in #112; merge and preserve zero image drafts/counters | Exact pinned Lily artifact; distinguish completed branch checks, broader qualification, release and installed app |
| IMGPAR-05 / P1 | Qwen3.8 Flash Next, retained `ddalcu` mixed 4/8 schema 6 | Image preparation/loading exists; its own full image execution and parity remain unqualified | Qualify its actual retained decoder/predictor snapshots with images resident; test mode switches and image invalidation | Run its own matched loading-mode benchmark; Lily Q4-XL results do not transfer | #112 supplies the loading policy structurally; inventory and qualify this source's declared head, modes and defaults | This distinct source/revision/layout after successful full image execution/admission |
| IMGPAR-06 / P1 | Gemma 3 BF16, 4B; then 12B/27B | 4B has image numerical/lifecycle evidence; larger full-model execution remains unverified | Implement exact sliding-ring KV payload/position snapshots and restoration, then qualify each size | Paired text-only versus image-loaded checks, fresh and cached; do not accept global cache disabling as the completed default | No implemented MTP path in this family; inventory the exact source and document a real absence or concrete unsupported head | 4B first, then independently admitted 12B/27B artifacts; quantized image exports need separate implementation |
| IMGPAR-07 / P1 | Gemma 4 BF16 E2B and E4B | Separate small-profile image execution evidence exists | Implement/qualify sliding-ring restoration and PLE/shared-state ownership per profile | Matched text paths with identical target arithmetic; qualify both profiles independently | Inventory source/draft support; record genuine N/A or implement a supported head | Each named E2B/E4B artifact and its distinct decoder/profile policy |
| IMGPAR-08 / P2 | Qwen3.5 MoE, affine-source v4 | Tower/worker evidence exists; full decoder/source parity remains limited | Qualify recurrent/expert target and any predictor state on the image-loaded artifact | Compare the exact expert quantization/materialization policy and source reference; measure memory and follow-ups | Inventory source/text heads; preserve and qualify supported text MTP instead of stripping it from the default | Each MoE source/layout, starting with the pinned 35B-A3B artifact |
| IMGPAR-09 / P2 | Qwen3.5 MoE, BF16-source v5 | Structural/operator controls; complete official 35B execution remains unverified | Complete full admission/execution first; then exact snapshots and lifecycle | Full model must run before a text parity/performance claim | Assess actual head/config inventory and text route; no inference from affine-source evidence | Separate full BF16 source artifact; resource deferral is recorded, not a pass |
| IMGPAR-10 / P2 | Gemma 4 12B unified | Encoder-free image component checks; complete decoder generation unqualified | Full execution, then sliding-ring and decoder-state snapshots | Match the unified decoder path and account for its patch/embedder residency | Inspect exact source; no supported-head claim from the small Gemma profiles | Full 12B artifact and its own reference/precision policy |
| IMGPAR-11 / P2 | Gemma 4 26B-A4B MoE | Bounded component/MoE controls; actual expert/decoder execution unqualified | Full execution plus exact ring/shared-KV/expert state ownership | Validate source numerics and compare matching complete targets | Inspect source/head support; qualify any future text MTP separately | Actual released expert weights and decoder layout, not synthetic controls |
| IMGPAR-12 / P2 | Gemma 4 31B | Tower/projector feature gates; complete 31B generation unqualified | Full execution plus ring-state restoration | Full decoder parity and paired text measurements; bound remaining numerical controls | Inspect source/head support; document N/A only with evidence | Exact pinned full checkpoint; component similarity is insufficient |
| IMGPAR-13 / P1 route audit | Typed Uzu Lily text export and original PipeNetwork/F32 route | Typed export omits vision/MTP; Pipe route lacks fusion and exact prefix support | Do not select an incomplete route as the completed default for an image-capable model; implement restoration if retaining that route | Qualify the actual chosen default graph; a text-only benchmark does not prove image-loaded parity | Typed format needs head preservation/execution if offered as a full-capability route | Prefer an implemented retained image route where compatible; otherwise implement missing functionality and qualify the distinct format |
| IMGPAR-14 / P3 | Gemma 3 quantized image exports | No admitted/qualified quantized image route; BF16 evidence does not transfer | Add/admit the quantization first, then qualify ring snapshots on its real state layout | Match the same packed target and precision with image support loaded; do not substitute a BF16 benchmark | Inspect the exact source/head contract; no inferred MTP capability from the BF16 or sibling model | Each intended 4/8-bit artifact, complete native image execution and the full applicable parity matrix |

The P1 route audit is a selection/implementation task, not permission to remove
working alternative formats or claim incompatible checkpoints use the same path.
Any remaining route limitation must be explicit in the catalog and support docs.

## Families without a current image path

These registrations remain part of the mandatory model intake. The loaded-image
comparison is N/A only after inspecting the exact source and establishing that
it has no image capability. A missing Stark image implementation is a work item,
not evidence that the source is text-only.

| Current registrations | Required disposition |
| --- | --- |
| Qwen2 dense, Qwen3 dense, Llama dense, Mistral dense | Existing text cache/performance/MTP checks remain. Inspect source architecture; multimodal siblings require separate image onboarding and the full matrix above. |
| Gemma 2, Gemma 3 text and compatible derivatives, OLMo 3 | Distinguish a genuinely text-only source from an export/wrapper that removed vision. If capable, enter the relevant image queue; otherwise record image N/A and retain text gates. |
| Phi-3 dense, Granite dense, SmolLM3 dense, LFM2, LLaMBA | Audit actual selected config/head inventory; no automatic image or MTP claim from brand names. Future vision paths must pass all applicable gates. |
| Native GLM-5.3 Flash and Flint | Existing text state/performance qualifications still apply. Flint's documented prefix blocker stays open; no image route does not make cache correctness a pass. Any image extension must preserve existing text optimizations. Full GLM-5.3 is a different architecture. |
| ModernBERT classifier | No current app generation/image/MTP workflow. Record the classifier-specific contract; do not advertise chat parity or treat it as a language-model pass. |

This covers all 20 registrations from the status table. Planned/null-adapter
catalog entries and all new discoveries require intake before being advertised
as implemented support. A future size, quantization, fine-tune, format, backend
or state-affecting option is a new qualification scope even within a known family.

## Acceptance matrix for every current and future artifact

| Gate | Required comparison/check | Evidence and exit condition |
| --- | --- | --- |
| Default route and capability | Normal catalog install/load has image support enabled for an image-capable source, with honest live image/cache/MTP readiness and local image checks | Verify the actual artifact, worker and host flow; no optional image-off workaround for failed parity |
| Reference arms | Qualified prior text build; candidate text-only load; candidate default image-loaded worker | Same target/head weights, precision, sampler/seed, context, stop/output budget, MTP depth and dispatch options; identify intentional graph differences |
| Cache exactness | Fresh versus cached text on the default image-loaded instance, at its advertised chunk grid | Exact matched-mode greedy/seeded outputs, correct reused counts, one retained state, publish after delivered success; unknown/partial restoration stays disabled and unfinished |
| Cache transitions | Cold/growing/repeated/shrinking/divergent/final-chunk cases; scope/context/MTP option changes; reset/error/cancel/restart/idle/delivery failures | Correct misses/restoration, no stale state or cross-scope reuse; include attention/ring/recurrent/predictor/position state actually read |
| Images between text requests | Text with MTP/cache → real image → text with MTP/cache; include failed/cancelled image recovery | Images use zero drafts, proposals and text reuse; following text is fresh, re-enables the selected text MTP policy, then can cache again |
| Text speed | Matched image-off/on loaded modes with MTP off, and with each supported text MTP policy | Separate cache-cold and reused follow-ups; first-token/prefill, useful decode and end-to-end medians/ranges; repeat/alternate serialized runs, predeclare bounds from baseline variability, fix or record failed/inconclusive parity |
| MTP correctness | Verify declared head tensors/config, loaded capability and actual draft proposals; match text-only policy and required cache keys | Same-mode output controls and correct default/opt-in behavior; do not label a stripped or disabled supported head N/A; no requirement that different stochastic algorithms share token sequences |
| Memory and loading | Resident target+tower+head weights, active/snapshot KV/ring/recurrent/predictor state, prefill/verify scratch, staging and allocator margin | Update loader/startup/request admission for loaded heads even while image MTP is off; distinguish payload/reserved capacity from observed peak RSS; report load cost separately |
| Host and completion | CLI, API and desktop text/image preferences; stale image-check receipts; normal install/load; CI/review | Record code, artifact, model-bound proof, measured speed, merge/release/install separately. No inherited pass from another size/format/profile. |

The existing [onboarding workflow](MODEL_ONBOARDING.md) and
[experiment template](model-experiments/TEMPLATE.md) own these standing gates.
Use existing validators where they fit, including `--vision --reuse boundary`
for supported converted Qwen vision snapshots. Tools need to exercise the actual
default image-loaded worker; a text-only loading test cannot satisfy that gate.

## How to mark an item complete

Link the exact source/worker/artifact identities, attempt IDs, fixed criteria,
all results (including failures), remaining scope and paired observations from
the work item. A quick check may complete its bounded test scope; it cannot
close all sizes, contexts or formats. Mark implementation, qualification and
delivery independently. A concrete blocker remains visible and keeps the item
open for that capability. This worklist schedules no GPU runs by itself.
