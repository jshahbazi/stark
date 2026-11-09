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

Before the recovery below, the first nonperiodic 1,792-token/context-2,048 BF16 cache matrix passes eleven
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

## Resumed qualification registration

External cleanup later restored more than 80 GiB available disk. The planned
main-worktree compiler cleanup did not execute here. Both sealed Qwen binaries
still match their recorded hashes; all three image artifacts, their text controls,
the BF16 headed derivatives and actual indexed BF16 source shard remain present.
Old template helpers were removed externally, so the next slot must build and
hash a matching helper in this worktree's own target. Native work remains
serialized behind Gemma and GLM; source review and tiny metadata reads continue.

The corrected full headless cache recipe is resealed as version 3 solely to add
a validation-only watchdog that terminates its own worker before disk crosses
the eight-GiB reserve. Driver SHA-256 is
`1f35430b376ed162fd91669c92704db1bd243f91da9a3628d7d5dd9fda3027b2`.
It retains the corrected scope expectation and nonperiodic 1,792-token/context-
2,048 case. Its results must remain bound to frozen worker `4010b86e…`, independently
of new predictor admission changes.

Before BF16 head timing, fix a matched text/default-image ABBA comparison at
MTP0 and MTP3, greedy seed 42, context 2,048, 64-token budget with valid EOS
shortening, one warmup plus three measured requests per load/policy. Record
startup separately, actual proposals/acceptances, same-mode output hashes,
decode/first-token/request timings and paging/thermal eligibility. Also compare
MTP0 against the prior headless target graph to detect new predictor overhead.
The complete head lifecycle recipe adds MTP-depth switches, nonperiodic near-
capacity reuse, seeded same-mode fresh controls, context/scope/reset/error/
cancellation recovery, forbidden image MTP options and real image → MTP text.

Reuse the existing pinned common-input head reference runner instead of creating
a new reference implementation. The available locked environment verifies Python
3.13.13, MLX 0.32.4 and MLX-LM `72be8c955d9a0d3099bd01865af288e0e947c3fa`.
The native BF16 gain policy and pinned MLX sanitizer may round gain folding
differently. Predeclare finite outputs, target-norm maximum absolute/relative RMS
error at most 0.03125/1%, predictor hidden 0.0625/2%, logits 0.25/2%, and matching
top-one logits on every replayed row. This is bounded head-component evidence,
not full-model cross-engine equivalence. Private registration hash:
`378b90e4da1c6c6c12b69cd0e2edfd5eb148c22ddcf0194ead251e2de7337049`.

BF16 MoE also gets a small native expert-predictor integration check: actual
engine routing/shared-gate execution, a zero-MLP negative control, nonzero
attention KV, cache-only priming versus full execution, and CPU/Metal comparison.
Synthetic geometry/weights cannot qualify the absent full official checkpoint.

The complete ddalcu retained artifact additionally contains its required
32,000,153,976-byte ngram table: total payload is **107,303,406,192 bytes**.
All 101 shard files are regular, with 333 expected vision tensor names and 72
predictor tensor names in the index. Installed config/source/index identities
exactly match the existing paged-text policy inventory. Use a private hardlink
payload view with copied metadata, then perform the ordinary strict image upgrade
and native payload-hash admission. Never modify the installed source.

Retained image loading currently drops the exact text request-admission plan;
the source fix preserves it for ordinary text with the tower resident. Context,
MTP and exact-artifact bounds remain unchanged; image requests retain separate
admission. Available memory is sampled after loading, and its macOS free-plus-
inactive estimate is not a reservation or residency ceiling. Measure paging and
label any process-memory observations separately from allocation inventories.
For ddalcu, register the same short ABBA timing recipe at MTP0 and recommended
MTP2, plus actual proposals, a complete nonperiodic cache/lifecycle matrix and
image MTP0 followed by fresh/repeated MTP text. MTP3 can receive a separate state-
option switch control without inheriting the MTP2 performance or memory gate.

## Crash recovery and bounded continuation

After the machine restart, HEAD `48c5aa52` and the four unfinished source/record
files were preserved. Recovered worker `c69a6685d153f007276dfc88a37c73e8d7491f13f4df0049f841e1c7a427b404`
and engine test executable
`c6bdaa238d821802576cd2b1891b07af52953bf1be19c922181b5efa1c9b25d4`
match their source seals, including the recovered expert-predictor fixture and
retained text-admission changes. All seven recorded source/lock hashes match.
The official source shard, legacy image artifact and both appended BF16-head
derivatives were freshly hashed and match their prior identities. No model or
prefix state survived or was resumed.

Two complete pre-crash version-four cache receipts survived: official BF16
0.8B and affine 0.8B each pass **30/30** lifecycle controls. The final-chunk
guard correctly misses at 1,025 tokens and reuses 1,024 rows at 1,026 tokens,
because restoration requires two suffix prompt tokens. These supersede the
earlier invalid expectations for those exact headless exports. The affine MoE
version-four run has no completed receipt and remains unfinished. Receipt hashes
are `5943494efafb473421fd0e0ecfd9a761679418736d36399f5de903221982626a`
and `6a15054bc82a09d35847f16201f24c80a85ee320d9e9981c1d098d3aa1af445b`.

The first post-restart small-model sanity run passes 27 text/cache/cancellation
checks, then stops at image preparation because the native template helper was
not discoverable. This is a preserved helper-setup failure, not a completed
matrix. The matching sealed helper was then supplied on PATH. Guards terminate
only their owned worker at 12 GiB RSS, below 32 GiB available memory, below
8 GiB free disk, any pageout/swap-in/swap-out increase, or 180 seconds per load.
Available memory is an observation rather than a reservation.

Actual official BF16-headed 0.8B now passes **26/26** bounded lifecycle checks
on the recovered `c69a6685` worker: MTP3/0 switches, seeded same-mode fresh/cache
equality, scope/context/reset/error/cancel recovery, forbidden image MTP, real
image zero-MTP execution, then fresh/repeated MTP text. Its short cases execute
100 proposals and accept 100. Peak sampled worker RSS is 2,134,360,064 bytes,
minimum observed available memory 102,213,599,232 bytes, with zero paging.
Receipt: `1eb4e6a8eb50a13beecb31d14090e64cd7ba801376120af55dedde78384734a7`.
This does not qualify the independent MLX head-numerics gate.

The registered headed ABBA run completes with matching same-mode output hashes
and zero paging. Text/image-loaded MTP3 decode medians are 147.764/150.135 tok/s;
each arm proposes 468 and accepts 222 measured drafts. MTP0 medians are
157.330/149.369 tok/s, a 5.06% image-loaded loss. The variability-based 8.69%
bound passes, while strict five-percent decode, first-token and request gates
fail. This is not a text-speed completion claim. Receipt:
`542dc13b2b07294249f1de958551f89cd45c96f8321e0fd34df9c74168a94a14`.

The tiny BF16 expert-predictor fixture passes one CPU/Metal integration test,
including cache-only priming and the zero-MLP negative control, on the sealed
`c6bdaa23` executable. It does not qualify absent official BF16 MoE payloads.

### QNIP-01 MTP0 dispatch follow-up

Source inspection explains a separate regression: every loaded dense predictor
enters the predictor-session driver even when the request selects zero drafts.
That maintains unused predictor state and makes the headed ordinary text path
much slower than the older headless stream. The new source candidate selects
the ordinary stream at MTP0, retains the loaded head, and uses an exact target-
only boundary snapshot for hybrid text caching. MTP1/2/3 retain predictor sessions.
Switching between those state representations must miss and discard the prior
entry before allocating; graph identity and exact draft depth remain cache keys.
Ordinary text and image admission exclude inactive predictor request state while
preserving resident head weights; MTP admission still counts its predictor.

This follow-up is registered before candidate execution. Required gates remain
the original same-mode cache/image lifecycle matrix, a matched headed MTP0/3
ABBA comparison, and a same-build headed/headless MTP0 control with the original
five-percent/variability bound. Also run normal text snapshot CPU/Metal controls,
the existing toy predictor memory/dispatch controls, and the independent pinned
MLX head replay. The frozen 27B branch and all earlier binary-bound results stay
unchanged. The new candidate compiles with Rust 1.98.1 and locked release
dependencies in a distinct target. Worker executable:
`1e592b4816b159f230a64d71555dac842a0dcd7d59e68169f95a4e67a908add1`.
All 135 worker unit tests pass (two ignored), and three CPU-only normal/hybrid
snapshot tests pass on pinned nightly; those snapshot tests now use the public
ordinary stream wrapper. Repository policy, package-scoped formatting and diff
checks pass. Independent source review of focused patch
`305d15e68e22b9484cfa688c12903b88069adfeb5b698a8e8d0ad4d0a8871218`
finds no confirmed blocker. It identifies a coverage gap in the old toy test's
MTP0 cases, which still used predictor sessions. The toy test now adds direct
headed ordinary-stream multi-token greedy/seeded fresh-versus-restored controls;
that Metal test has not run. The next actual worker matrix also adds seeded
MTP0 and all positive-depth switches in both loading modes. New-candidate
model-bound correctness, numerical and speed qualification remain pending.

After Qwen closed all owned workers and handed the resource slot to Gemma,
another installed Stark worker appeared outside this task. Further model/Metal
work is held for coordinator review. Its presence does not establish the cause
of the preceding machine crash.
