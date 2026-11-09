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

## Final qualification continuation

The user subsequently unloaded the installed model. The coordinator verified
the machine's quiet state and granted Qwen the exclusive slot after Gemma's
bounded controls completed. This continuation uses the unchanged production
candidate at `e4144be5`, worker `1e592b48…`, and a fresh pinned-nightly native
test target. The existing tiny hybrid fixture is retained; its converted weight
SHA-256 is `156f79e885165762545012fde03c4c453d6631e5c1cc633acb44d8fe4dbc95ab`.

Before execution, register the following remaining gates separately: the newly
extended ordinary-stream toy test; actual headed image/text lifecycle matrices
with seeded MTP0 and 0/1/2/3 switches; same-build headed/headless MTP0 ABBA;
headed image/text MTP0 and MTP3 ABBA; independent pinned MLX head replay; and the
corrected affine MoE lifecycle matrix. Each mode's cached result must equal its
own fresh result, and matched loading-mode timing outputs must match exactly.
Ordinary and predictor drivers may use different floating-point operation order;
cross-mode greedy agreement is recorded separately and is not substituted for
same-mode cache equality.

The MoE image payload was freshly rehashed and still matches `2f762b26…`.
Its bounded private watchdog is registered at 40 GiB worker RSS, at least
64 GiB observed available memory, zero paging, eight GiB free disk and ten
minutes per load. The previously proved 19,495,956,480 shared parameter bytes
are an allocation inventory, not a residency guarantee. This separate recipe
does not relax the small model's 12 GiB RSS guard. The 107 GB retained ddalcu
load remains held for a complete memory plan.

The fresh native test executable is
`daf7b79d7614a61ac990d3d6b71683ddfde5bf5c0bfe2cf87da5acf0ae9b3119`.
The extended tiny hybrid test passes: direct ordinary-stream multi-token
greedy/seeded fresh-versus-restored checks, predictor-depth controls and inactive
predictor request-memory separation all execute. Sampled peak RSS is 73,482,240
bytes, zero paging, with the watchdog unstopped. This closes the source review's
toy-test coverage gap; actual BF16 headed-worker matrices remain separate.

The new ordinary-dispatch worker passes **34/34 image-loaded** and **28/28
text-only** actual BF16-headed lifecycle controls at the registered nonperiodic
1,792-token/context-2,048 input. MTP0 seeded fresh/cached results match exactly;
switches among depths zero, one, two and three miss, and same-mode repeats
reuse exactly 1,024 rows. Image rejection/execution and subsequent text recovery
pass in the image-loaded matrix. Image-loaded peak sampled RSS is 2,156,576,768
bytes, with zero pageout/swap-in/swap-out deltas and no watchdog stops.
Receipts: `4a0ca99e3a1f1954278b6dcd546ba43da9f001aed874e9b3c12961a68063163b`
and `3c36e91dc62ec388f492a8ac877f357c1b85fb7541733e785ae52aa97f1af963`.

The pre-benchmark quiet-process screen then finds another app-build worker
outside this task. All Qwen workers are closed; timing is held again for
coordinator review. No result from that competing-worker interval is promoted.

After the coordinator verifies the other task's worker is closed, same-build
headed/headless image-loaded MTP0 ABBA completes with matching output hashes,
zero paging and no watchdog stops. Decode medians are 294.968/299.031 tok/s
(headed/headless), a 1.36% loss; first-token proxies are 9.540/10.225 ms and
whole-request medians 223.176/222.159 ms. All three strict five-percent median
gates pass. Decode ranges are 278.630–299.565 and 283.816–303.090 tok/s;
the registered variability bound is 12.89%, so this short run is not precise
evidence of a speedup. It closes the earlier roughly twofold inactive-predictor
overhead for this workload and exact BF16 artifact. Receipt:
`88880038813cdfc7b85eb1cb6fa3ff43910cd89d78d85a221a4c246f8abf580f`.

An installed worker subsequently reappears after that completed comparison.
The next loading-mode timing attempt starts despite its printed quiet screen
and exits incomplete after three rows. That launch was a harness oversight;
receipt `db0e90fbd0f336ad7cc9b7f24910e8209f3556959f38b578d854c27dc475f72f`
is excluded. The coordinator unloads the model normally. The validation-only
watchdog now rejects competing workers before launch and continuously watches
for another worker, terminating only its own process if one appears. Active
guard records are retained even when a timing load fails.

The corrected watched loading-mode ABBA passes on the same `1e592b48…` worker:
MTP0 text/image decode medians are **299.377/299.810 tok/s**, first-token proxies
10.481/10.113 ms and request medians 222.205/220.910 ms. MTP3 medians are
**149.586/153.860 tok/s**, 12.175/11.062 ms and 428.730/414.884 ms. Every strict
five-percent median gate passes. Each arm's six measured MTP3 requests propose
468 and accept 222 drafts; each MTP0 arm proposes zero. All outputs contain
62 tokens before valid EOS and match exactly within mode across loading arms.
Every measured row has zero paging and eligible thermal status; no foreign
worker or watchdog stop is observed. Decode ranges remain broad:
278.683–304.024 / 279.764–304.168 at MTP0 and 144.127–152.280 /
151.923–155.110 at MTP3. Registered variability bounds are 16.93% and 10.90%,
so these quick controls establish bounded parity, not a measured speedup.
Receipt: `a5154274e9e9d4f9830de4c4693460424b667d8f8a194bd99308c52c8f7d9d7c`.

The independent pinned MLX head replay completes but **fails the registered
maximum-absolute hidden-state gate**. Target normalization is bit-exact;
normalized-hidden relative RMS is 0.62–0.84%, all replayed top-one logits match,
and logit errors pass their fixed bounds (maximum 0.15625, relative RMS below
one percent). However, hidden maximum absolute errors are 0.25, 0.125 and
0.140625 across the three stages, exceeding the unchanged 0.0625 limit.
This is a real failed numerical attempt, not a passed head qualification.
Norm-gain rounding and predictor eager boundaries require diagnosis. Gate
receipt: `3626be445f0ec47d288fb127fc707fabcb25da62daac34beabc13511c8810dc2`;
raw aggregate comparison: `7d90aa8a29eba87970a0f1e2a1ceeec443e771985ee93ae68d2f9e15e7c8ce99`.

The affine MoE continuation is stopped before its first cache case by the
continuous foreign-worker watch: an installed model reloads again. The owned
MoE worker reaches 20,723,073,024 sampled RSS bytes, minimum available memory
83,612,221,440 bytes and zero paging; no RAM/RSS limit is reached. This is a
resource-deferred foreign-worker attempt, not a model numerical failure or a
cache pass. Receipt: `65f9afac8e10ea71619f3bb7c31bf4c7074bd2e7bc3efd19bbc04f03f97b204d`.
The slot is released for other coordinated builds while first-divergence
diagnostics remain source/CPU-only.

The [ddalcu memory plan](2026-11-09-ddalcu-default-image-memory-plan.md) now
inventories all headers and source owners, including resident target experts,
the actual head/tower, paged n-gram worst case, the 16,384-capacity floor and
growth-inclusive request buffers. Its full run remains held for review.

### QNIP first-divergence trace and second MoE continuation

The private first-layer trace now reproduces the unmodified normal predictor
path bit for bit. Test engine
`998e94ffe9ad70d7bf4992defc187e03a6417996eec094d2551204e7f90fdba6`
contains only test instrumentation over `bd3e781`; production arithmetic is
unchanged. Export receipt
`4ab178f5462f784f03ac181cf08d5aadad48ab0c115e0c8198433d930adcbc30`
passes 1/1 controls with zero paging and no foreign worker. Earlier trace-only
attempts are retained separately: one compile error, a constant-handle lifetime
assertion in the new harness, and foreign-test preflight deferrals are not
numerical passes. Dropping the test constant handle before command completion
resolves that harness assertion.

The unchanged pinned MLX replay reports exact embedding, embedding norm and
hidden norm. Its first mismatch is the head fusion projection: maximum absolute
error `0.001953125`, relative RMS `0.00251255`. Replaying the FC with native norm
inputs gives the same difference. The manual trace's final hidden output is
bit-exact with the actual normal-path first head output, so the diagnostic does
not introduce the discrepancy. Pinned aggregate receipt:
`6f97ce9bb56dee634f5a97417ede3ab46dd978d4e3bdea839b0fad4d40d8a41b`.
The original full-head numerical gate still fails; no tolerance or production
arithmetic policy changes follow from this diagnostic alone. A test-only
parallel reference-FC experiment is prepared for both prefill and single-row
head execution and awaits its coordinated build/model slot.

The second corrected affine MoE continuation also stops before its first cache
case when a separate resource-budget native engine test appears. Receipt
`796f63f3e47bd60c548eed1a5da3d59ac9b775c96819b41e5e47dfdef46aef60`
records peak sampled RSS `20,722,958,336` bytes, minimum available memory
`84,208,844,800` bytes and zero paging. It is a foreign-process deferral with
zero cache cases, not a model or memory-limit failure. The owned worker is
closed and the hardware slot is released to the next coordinated family.

The test-only causal FC candidate now compiles as engine
`12aad35cbc39159499bccb4c6725f459d3644e8961d6fdd18e029407da27942f`.
Its seal is `b0bb102a57fc177404e29f7fb0d1fc113b635538c65fbd59a8aa64654891a8e3`.
It replaces only the predictor fusion projection when the private test selector
is set; target arithmetic, input norms and the downstream transformer are
unchanged. The alternate full BF16 FC has a 4 MiB loading overlap on this exact
small artifact; it is not a production memory or performance result. No extra
FC copy is created without the selector. The first candidate compile fails on
a scratch-buffer borrow in the test adapter; the corrected second compile
passes. Both attempts and the unchanged numerical preregistration are retained.
Independent source review finds no blocker for this bounded experiment. Native
and MLX execution of the candidate remain pending the coordinated model slot.

### QNIP causal FC result: local correction, full head still fails

After the coordinated hardware handoff, engine `12aad35c` passes both the
unchanged baseline and test-only reference-FC exports. All 21 baseline tensor
records are bit-exact with the prior normal-path trace. The independent pinned
replay makes the candidate's FC fusion bit-exact with MLX, confirming a local
arithmetic difference rather than different input norms or weights. Fusion
receipt: `5d22cbfc685c413bff7f959e7dbabf34af0f9e9258d1a822a8154fce7b13f119`.

The unchanged full-head gate still fails: hidden maximum absolute errors are
`0.25`, `0.1875`, `0.1875` over its three stages. All original finite, relative
RMS, logit-error and top-1 checks pass. Gate receipt:
`e2acf10016bb868fbd674456bf2a9e44399ad8e4f4ecd10c6b76a39e5cb705b9`;
full aggregate replay:
`a623312e224ff374b7ee7a43e3f551d27da654e5865f69765a4369671dad8ba3`.
There is no production arithmetic adoption or tolerance change.

Local comparisons on identical captured native operands report exact
pre-attention, pre-MLP and final norms, as well as exact residual addition.
Remaining attention output error is `0.00390625` maximum absolute /
`0.00508196` relative RMS; MLP output is `0.015625` /
`0.00547159`. Local receipt:
`10c43978f98c145c5931bffa30a533e1103241d6816e25966970037eaf237d85`.
Every native/reference process closes with zero paging and no foreign worker.
The next source-only causal candidate selects reference accumulation for head
linears while retaining target/readout, norms, attention and MLP activation
policies. It remains a test-only experiment pending compilation and execution.

### QNIP additional headed cache coverage and quiet MoE stop

The same production worker `1e592b48` passes an additional **48/48 image-loaded**
and **26/26 text-only** controls, independently of the head numerical failure.
Both MTP0 ordinary state and MTP3 predictor state cover shrinking, growing,
divergent prefixes and the 1,024/1,025/1,026-token suffix-position boundaries.
Image controls explicitly prime a retained text prefix, then exercise failed,
cancelled and successful real images before checking fresh/repeated text in
both modes. Images propose zero drafts. Sampled peak RSS is `2,147,172,352`
bytes for images and `1,915,420,672` for text; paging/foreign-worker counts are
zero. These supplement the original 34/28 lifecycle cases and are not new speed
measurements. Payloads are rehashed against their prior exact identities after
the run; binding receipt:
`84cd1243f5f63e7f96bc1134b2a3a9f1cf27cd0c619f049be47484758928d11b`.

| Additional cache scope | Aggregate receipt SHA-256 |
| --- | --- |
| Image-loaded 48 cases | `aeb8a3034c56bc34ffdcf172be105906e9f8951d87fd50100c2e5fdd06c72012` |
| Text-only 26 cases | `cadb6fe0a8f910c5ab674f9cce943644431bb092d1efd38ecdc7c969dc24225f` |

The third affine MoE continuation reaches readiness in 70.155 seconds but stops
before completing its first 1,792-token cache case when the global Pageouts
counter increases by one page (16 KiB). There is no foreign worker or swap
growth; sampled RSS reaches `31,748,620,288` bytes and minimum available memory
is `74,006,904,832`, above the unchanged 64 GiB floor. This is an unqualified
resource stop, without attribution to the model or a memory-limit failure.
Receipt: `adceb2e976070e6abb9417542888a22a606acc0c15d06c6a28abab247fbc688d`.
All owned workers close. The corrected affine MoE cache matrix remains open;
no guard is relaxed and no incomplete case is counted as a pass.

### QNIP all-head-linear experiment: rejected

The next isolated candidate is confined to `cfg(test)`: a scoped head parameter
tree selects the existing full-BF16 reference linear implementation, while
target/readout, norms, attention and MLP activation policies remain unchanged.
It accepts only an incoming Legacy profile and rejects conflicting FC-only /
all-head selectors. The source-scoping CPU control passes 1/1; the model export
passes 1/1. Build time is 1 minute 34 seconds. Test engine:
`7b25eb5595e7b3cfa2b5834ce4941468bc9e9f2543c9d557c0d26c6be9f56f7e`;
source/binary seal:
`450a7ea768ad65eeadbd3b8b4f0113e4d8cd6c1617217ca3f0ecfe3eab2675be`.
Its reference F32 accumulator workspace is explicitly inventoried; this is not
a production admission or performance claim.

The unchanged complete-head numerical gate rejects the candidate. Hidden
maximum errors are `0.1875`, `0.25`, `0.15625`; the final stage also loses its
top-1 match. Relative RMS and absolute logit limits pass. Gate receipt:
`868cbd232975bd209d5c5cc1ad6c46dfc9c1750ef5b73dff3796ea3a018e2b71`;
full aggregate:
`d57901432eac1712f820a03f24d644688327c5cf5ea36a9836eebb4190e56e3c`.

Common-input norms and residuals remain exact. Attention output is
`0.004638671875` maximum / `0.00482802` relative RMS; MLP output is
`0.0078125` / `0.00378840`. Local receipt:
`ae9c89eae2d77de0fa9589ac504e5adaa4611abc0a1043dfa4154beafcee60bd`.
All processes close with zero paging and no foreign worker. No arithmetic is
adopted; the next diagnostic must isolate projection and activation primitives
on identical operands rather than change the whole head profile.
