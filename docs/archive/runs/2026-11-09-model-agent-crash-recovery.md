# Model-family agent crash recovery, 2026-11-09

This is the coordinating checkpoint for the interrupted default-image work.
The machine reboot ended the original agent sessions. The three implementation
worktrees, committed history, unfinished source edits and private validation
receipts survived. Replacement agents resumed those worktrees; no model weights
or previous evidence were deleted. The reboot cause has not been established.

The governing acceptance criteria remain in
[the default-image parity plan](../../IMAGE_DEFAULT_PARITY_PLAN.md): image
support is the default; preserve exact text caching, matched normal text
performance and supported text MTP. Image requests use zero drafts and invalidate
text state. Completion is per artifact, configuration and backend.

## Latest coordinating checkpoint

| Family | Completed resumed gates | Open gate / next action |
| --- | --- | --- |
| Qwen BF16 0.8B | Corrected MTP0 dispatch; 34/34 image-loaded and 28/28 text-only lifecycle checks; headed/headless and text-only/image-loaded MTP0/3 short timing passes | Independent head hidden-feature numerical tolerance failed; trace the first divergence before any precision edit |
| Qwen affine 0.8B / 35B MoE | Original source-bound cache/timing evidence preserved | Corrected full MoE lifecycle matrix was stopped by a foreign worker before case zero; repeat in an exclusive slot |
| Gemma | E2B G4AP-02 unchanged short oracle passes exactly; G4AP-03 compiles and six tiny CPU/Metal committed-row controls pass | Long chained oracle failed at the first 512-token case; batch/cache controls did not execute. Diagnose attention/readout before worker integration |
| GLM | Fresh worker/helper and native engine builds pass at `b76f560d`; arithmetic and live supervisor cleanup controls pass | Both short attempts stopped on global pageouts before text completion; prepare a separately registered functional-only screen after kernel-counter review. Image/cache/performance/MTP remain open |

The installed app was closed without implicitly reopening it. The coordinator
verified no competing model/compiler/app processes, 95% free memory and zero
swap before Gemma's actual E2B gate. Gemma released its processes after the failed
long gate, and GLM completed its builds and two stopped short attempts. Qwen now
owns the next diagnostic build/model window, Gemma's trace is queued afterward,
and GLM prepares a separately registered functional-only supervisor in source.
Repeated external Metal tests from `stark-playground-resource-budget` are caught
by the preflight. The user authorized cross-chat coordination, and that chat
confirmed its current GPU run finished and it is holding new GPU/model runs.
The exclusive queue is Qwen diagnostics/MoE cache, Gemma's bounded trace, then
GLM's functional-only short control. The coordinator must explicitly hand the
GPU back afterward. No PR is authorized here.

## Ownership and source checkpoints

| Owner | Branch and recovered HEAD | Worktree | Next gate |
| --- | --- | --- | --- |
| `qwen_recovery` | `codex/qwen-image-parity-next`, `48c5aa52beacdba117beae7b5b7bc71675f0f702` plus preserved edits | `stark-qwen-image-parity-next` | Fix inactive-head MTP0 dispatch; independently review, then qualify current source cache and speed |
| `gemma_recovery` | `codex/gemma-image-parity-next`, `dccb8ed0cd1e80f3cbe84746ab57a427aad0f8cf` plus preserved edits | `stark-gemma-image-parity-next` | Build the saved assistant arithmetic fix, run controls and repeat the unchanged E2B numerical oracle |
| `glm_recovery` | recovered `fecc27d551773d8197f0f9b2c55c6dba21fb9444`; recovery commits pushed through `31fd48dbd83e28e26a470abecada719362e79dc4` | `stark-glm-image-parity` | Finish peak-memory accounting before any further full-model execution |
| Coordinator | `codex/image-default-model-worklist` | `stark-image-default-model-worklist` | Maintain this checkpoint, serialize resource-heavy validation and integrate independent review |

Worktrees are siblings under `/Users/johnshahbazian/development/`.
The Qwen27 branch `codex/qwen-dense-image-parity` remains frozen at
`67284748f82f71bd60f3a73f57c5fb31120db505` for
[PR #114](https://github.com/jshahbazi/stark/pull/114). Follow-up source fixes
belong on the branches above. No additional PR was opened during recovery.

## Preserved recovery snapshots

Each implementation worktree contains the ignored private directory
`runtime/crash-recovery-unified-20261109T174122Z/`: complete current source-file
copies, status, normal binary-capable unified patches and per-file hashes.
Every working-tree patch passed `git apply --check --reverse` against its saved
source without modifying it. These are source snapshots, not persisted inference
or prefix-cache state.

| Snapshot | Working-tree patch SHA-256 | Bytes / complete files |
| --- | --- | --- |
| Qwen | `a3fbdd845ce7c44cc7a32bad834852244f88f195a04f6ab6cba2cc9da89ff46c` | 19,109 / 4 |
| Gemma | `df5a7f9868bfee7bf8f1fc103907ec6065faa902af6c1f93d5464e9c81049cca` | 40,259 / 9 |
| GLM | `78f2533bfbc7f76f537829d960fc0c012b07f6ba0f13eb41d1d150ae22eb550e` | 3,009 / 4 |

The earlier `runtime/crash-recovery-20261109T173249Z/` diff used an external
display formatter and is not an applyable recovery patch. It is retained for
history; use the verified unified snapshots above. Agents have continued editing
after these snapshots; newer candidate source must receive its own source seal.

## Evidence retained and remaining work

### Qwen

The completed headless paired benchmarks survive: official BF16 0.8B
303.618/304.108 tokens/s, affine 0.8B 659.888/661.392 and affine MoE 35B
39.988/39.960, respectively text-only/image-loaded. These short measurements
qualify their original binaries, artifacts and output budgets only.

Two completed pre-reboot lifecycle receipts were recovered and verified:

| Private receipt | Result | SHA-256 |
| --- | --- | --- |
| `runtime/bf16-08b-cache-04.json` | 30/30 | `5943494efafb473421fd0e0ecfd9a761679418736d36399f5de903221982626a` |
| `runtime/affine4-08b-cache-04.json` | 30/30 | `6a15054bc82a09d35847f16201f24c80a85ee320d9e9981c1d098d3aa1af445b` |

The matrices include real-image transitions and distinguish a 1,025-token
repeat miss from actual 1,024-token reuse on the 1,026-token case. They remain
bound to the original `4010b86e` worker; they do not qualify later source edits.

Post-reboot, the actual BF16-headed 0.8B artifact passed 26/26 bounded lifecycle
controls on the recovered `c69a6685` worker, including MTP3/0 switches, seeded
cache equality, scope/context/reset/error/cancellation, forbidden image MTP and
real image followed by fresh/repeated text MTP. Receipt
`runtime/bf16-08b-head-cache-recovery-01.json` SHA-256:
`1eb4e6a8eb50a13beecb31d14090e64cd7ba801376120af55dedde78384734a7`.

The headed short benchmark exposed an additional problem: MTP0 still entered
the predictor session, approximately halving normal BF16 0.8B throughput versus
the previous headless target. MTP3 text/image-loaded medians were
147.764/150.135 tokens/s; MTP0 was 157.330/149.369. The latter is a 5.06%
loading-mode loss and fails a strict 5% gate. This is an open regression.
The candidate dispatch fix selects the ordinary stream at MTP0 while retaining
head weights and exact target-only boundary snapshots; independent source
review and fresh binary-bound cache/performance checks are still required.

Also open: complete corrected affine MoE lifecycle matrix, independent BF16
head numerical reference, full BF16 MoE/expert-head qualification, other
sizes/quantizations, and the distinct 107.30 GB retained ddalcu artifact.
The ddalcu full-model load remains resource-deferred.
The detailed branch record is
[`2026-11-09-qwen-remaining-image-text-parity.md`](https://github.com/jshahbazi/stark/blob/codex/qwen-image-parity-next/docs/archive/runs/2026-11-09-qwen-remaining-image-text-parity.md).

### Gemma

Gemma 3 4B's 40-case lifecycle run and candidate text/image-loaded medians
68.10/67.04 tokens/s survive. The separate prior-worker baseline attempt had
swap/background activity and remains ineligible. Previous Gemma 4 E2B/E4B cache
evidence remains bound to `7468dbc4`; the earlier `81b85f88` qualification was
withdrawn because it replayed the full prompt.

The official E2B assistant converted and executed native proposals, but its
pinned Transformers oracle failed: masked-logit maximum absolute error
47.28125, relative L2 0.02197. Matching the first greedy proposal does not pass
that gate. Serving MTP remains disabled.

The saved G4AP-02 fix changes assistant-only BF16 query arithmetic and committed
KV handling. Recovery preserved it, added partial-rejection/zero-draft controls
and a conservative draft-position guard above 512, and resumed a distinct build.
An initial test compile failure from an unavailable tempfile helper was corrected
using the existing private test-directory helper; the failed attempt is retained.
Build success, CPU/Metal controls, unchanged E2B oracle, target verification,
cache integration and useful MTP performance remain separate open gates.
E4B and larger assistants require their own qualification.
See the
[Gemma branch ledger](https://github.com/jshahbazi/stark/blob/codex/gemma-image-parity-next/docs/model-experiments/2026-11-09-gemma-next-image-parity.md).

### GLM

The saved full GGUF worker loaded image-ready in 196.30 seconds and completed
an ordinary text control, but actual image generation failed
`resource_budget_exceeded`. A later text request also failed after lazy expert
bank growth. Vision component tests do not establish full image generation.

Recovery preserved that failure and the rejected admission relaxation in commit
`f68697e7`. Commit `31fd48db` isolates the macOS/aarch64 SHA-256 feature change.
Unchanged source identities, worker/helper seals and GGUF hardlink identities
were verified. Five complete metadata/sidecar hashes agreed across hardware,
software and independent references; this does not qualify decoder startup speed.

Header/source accounting finds 99,035,873,280 bytes of rounded expert-bank spans.
The existing resident-bank/static/state admission subtotal at context 4,096 is
approximately 108,692,065,600 bytes, before separately loaded vision, host
workspace and 305,266,688 bytes for two overlapping KDA snapshots. Implementing
the currently dormant predictor adds approximately 2.788 GB of weights.
These estimates do not prove residency or peak RSS. A full-resident retry is held;
bounded execution needs separate review and performance qualification.
Actual image execution, cache/lifecycle, matched text performance and GGUF
source-head MTP remain unfinished. EXL3 has separate backend-specific gates.
See the
[GLM admission record](https://github.com/jshahbazi/stark/blob/codex/glm-image-parity/docs/archive/runs/2026-11-09-glm-image-admission.md).

## Restart and resource rules

1. Preserve original receipts and failed attempts. Never relabel an old binary's
   result as evidence for new source. Seal new candidate source and binaries.
2. Resume unfinished gates; repeat completed checks only when a source change
   affects them or their original measurement was invalid.
3. Serialize Metal/model runs and coordinate memory-heavy builds. Use a distinct
   ignored target directory per candidate; start with small artifacts.
4. During recovery an installed Stark worker outside agent ownership occupied
   approximately 71 GiB RSS. Agents paused model/Metal launches and preserved that
   process. Resolve resource ownership before issuing another hardware benchmark.
5. Keep GLM and ddalcu full-model loads held until complete peak accounting is
   reviewed. Do not weaken admission based on a single free-memory observation.
6. Update family ledgers with source identity, aggregate results, remaining gate
   and the next command/recipe before handing off. Keep this coordinator record
   linked from the plan; store private native receipts under ignored runtime paths.

## Hardware validation resumed after app unload

The user unloaded the installed Stark model and requested continued fixes and
benchmarks on 2026-11-09. The coordinator verified no `stark-generate`, `cargo`
or `rustc` processes, 87% system memory free and zero swap. The external-worker
hold above is resolved. This does not resolve GLM/ddalcu's separate full-model
peak-memory requirements.

| Owner | Pushed restart checkpoint | Resumed gate |
| --- | --- | --- |
| Gemma | `9053f805932ace8be7258d46d9eadd999d138c52` | First exclusive Metal/model slot: dedicated committed-query controls, unchanged E2B oracle, serial verification and 512/513 boundary controls; then fix/qualify integration and performance |
| Qwen | `e4144be510de3379c7a318977c91cdf2a6b3edee` | Next exclusive slot: ordinary headed MTP0 restoration, corrected image/text lifecycle matrices, short matched MTP0/3 and headed/headless benchmarks, independent head numerics and MoE cache controls |
| GLM | `cc4e5aa2` | Source/CPU work toward a complete bounded-command memory plan and safe execution; review the plan before requesting a full-model slot |

Qwen's candidate builds, 135 worker tests and three CPU prefix tests pass.
Independent source review found no confirmed blocker; it identified a missing
ordinary-stream test, now added for runtime execution. Gemma's fresh build and
two dedicated CPU controls pass; independent source review found no reportable
defect. These are build/component gates, not completed real-model qualification.

Agents must continue through their pending gates and resolve actual failures,
preserving failed measurements and updating their ledgers. Only one model/Metal
run owns the machine at a time. No additional PR is authorized by this resume.

### Resumed Gemma numerical gate

On the sealed `9053f805` candidate, both dedicated committed-KV Metal controls
passed. The same pinned E2B oracle then passed its unchanged gates: feedback,
centroids and masked logits all had zero maximum absolute error and zero relative
L2 error; root, support and first greedy proposal matched exactly. Three actual
proposals were target-verified. Supplied all/zero/partial/rejected acceptance
controls, cancellation/fresh controls and 512-admit/513-reject with ordinary
continuation passed. Runtime was
40.57 seconds, approximately 10.4 GB peak memory and zero swaps.

The actual sampled round accepted zero of three proposals. This qualifies the
tested component's correctness, not acceleration or normal worker integration.
Serving/default loading, complete head-aware cache lifecycle and useful matched
performance remain open. The agent released all model processes and handed the
exclusive hardware slot to Qwen, continuing Gemma integration in source.
The oracle/serial receipt SHA-256 is
`770a6bc0ba68b628a445d27956c027dfdcc0d21f910e95cbf647eff3de6b61a1`;
execution seal `c2a53fd8b6cc97747f8ec4cefd2326521fff9340ed2680c69cfd526d334c2ae4`.

### Resumed Qwen cache gate and resource handoff

The corrected headed BF16 worker `1e592b48` passed its expanded ordinary-stream
lifecycle matrices: 34/34 image-loaded and 28/28 text-only, including seeded MTP0
and switches among depths 0/1/2/3. Both runs had zero paging and no watchdog stop;
peak RSS was 2,156,576,768 and 1,916,878,848 bytes respectively.

| Private receipt | SHA-256 |
| --- | --- |
| `runtime/bf16-08b-head-cache-mtp0-image-01.json` | `4a0ca99e3a1f1954278b6dcd546ba43da9f001aed874e9b3c12961a68063163b` |
| `runtime/bf16-08b-head-cache-mtp0-text-01.json` | `3c36e91dc62ec388f492a8ac877f357c1b85fb7541733e785ae52aa97f1af963` |
| `runtime/qwen-final-toy-01.json` | `6412ef65e68fb418b406be8decb88723ab93f52004096b24b9db300bbb36431f` |

The extended synthetic ordinary/predictor restoration control passed on engine
`daf7b79d7614a61ac990d3d6b71683ddfde5bf5c0bfe2cf87da5acf0ae9b3119`.
It complements the actual-model matrices; it does not replace numerical head
qualification or speed measurements.

A second external worker from the separate resource-budget app briefly paused
timings. After the user reported that task finished, the coordinator verified
no model/compiler processes, 94% memory free and zero swap, and resumed Qwen's
exclusive hardware slot. No external process was killed by these agents.

### Headed MTP0 performance restoration

A quiet comparison completed before another external app load: corrected headed
MTP0 decoded at 294.968 tokens/s versus 299.031 for the headless target, a 1.36%
difference within the strict 5% gate, with zero paging. This closes the observed
inactive-predictor dispatch regression for that short BF16 0.8B comparison.
The private receipt SHA-256 is
`88880038813cdfc7b85eb1cb6fa3ff43910cd89d78d85a221a4c246f8abf580f`.

The subsequent loading-mode attempt was incomplete and excluded when the
installed Stark app loaded Lily again. The coordinator used the normal app's
Stop model control and observed Stopped, GPU memory unavailable and Stop model
disabled, then verified no remaining model/compiler processes, 94% free memory
and zero swap. Qwen resumed with a required hard quiet-screen and continuous
external-worker check. The interrupted receipt remains preserved as invalid.

The repeated loading-mode ABBA comparison then completed with the hard preflight
and continuous external-worker supervisor. Context and sampling were matched;
the 64-token budget ended validly at EOS after 62 tokens. Six requests were
measured per arm/depth. Decode, first-token and request medians all passed the
strict 5% gates; no paging or supervisor stops occurred. Peak RSS was
2,126,823,424 bytes.

| Exact BF16 0.8B headed artifact | Text-only tokens/s | Image-loaded tokens/s | Measured proposals / accepted, each arm |
| --- | ---: | ---: | ---: |
| MTP0 | 299.3765 | 299.8096 | 0 / 0 |
| MTP3 | 149.5862 | 153.8603 | 468 / 222 |

These results establish short loading-mode parity, not MTP acceleration on this
small model. The separate ordinary MTP0 comparison above establishes that the
resident inactive head no longer forces the slow predictor path.
The receipt `runtime/bf16-08b-head-bench-mtp0-02.json` SHA-256 is
`a5154274e9e9d4f9830de4c4693460424b667d8f8a194bd99308c52c8f7d9d7c`.
Independent numerical head replay and the complete affine MoE cache matrix
remain open at this checkpoint.

### Independent head numerical replay remains open

The pinned MLX replay exposed a real failed gate after the speed/cache checks.
Target normalization is bit-exact, all six compared rows agree on the top choice
and logit maximum error is at most 0.15625 with relative RMS below 1%. However,
normalized hidden-feature maximum errors 0.25/0.125/0.140625 exceed the registered
0.0625 limit. The tolerance is unchanged; this is not a numerical parity pass.
The agent is investigating BF16 norm-gain folding and eager operation boundaries.
Normal MTP0 cache/speed evidence remains bound to the tested binary, while any
head precision fix requires new source/artifact-bound numerical controls.
Private receipt `runtime/qwen-final-head-gates-01.json` SHA-256:
`3626be445f0ec47d288fb127fc707fabcb25da62daac34beabc13511c8810dc2`.

### Continued resource coordination and GLM preparation

The first corrected affine MoE matrix attempt was stopped before case zero by
the continuous foreign-worker screen when installed Stark loaded Lily again.
Only the owned 20.72 GB MoE worker was terminated, with zero paging. This is
resource-deferred, not a MoE pass or model failure. Receipt SHA-256:
`65f9afac8e10ea71619f3bb7c31bf4c7074bd2e7bc3efd19bbc04f03f97b204d`.
The coordinator invoked the app's normal Quit control and observed its exit.
A new idle app process subsequently appeared; relaunch ownership is under
read-only investigation. Source/CPU work continues without hardware timing.

GLM's bounded accounting fix was pushed as `b76f560d`. Eight source-extracted
Rust CPU arithmetic controls pass; full native compilation remains required.
The reviewed owner-only watchdog now enforces 32 GiB available/RSS/footprint and
wired-growth limits, less than 1 GiB compressor growth and no pageout/swap growth.
Five actual tiny child-process cleanup controls passed and reaped every child,
including cancellation, termination and constructor/base-close failure. Smoke
receipt SHA-256:
`050a41d9b8cf7b52d6de5a0aed060b60d81ede465c8ebfdeda4a3bb63353527a`.
This prepares safe bounded execution; it is not a full-model execution result.

Qwen explicitly released its model/build slot to continue CPU-only numerical
diagnosis. Gemma owns the next exclusive native build slot for its longer-context
and transactional batched-verification candidate. GLM's native build follows.

The idle app relaunch was subsequently traced to the computer-use service:
the coordinator's post-Quit app-state query reopened the application. It was
closed again using normal Quit and verified through inventory/process checks
without another app-target query. No model/app process remained. This is not
evidence of a Stark self-relaunch mechanism. Future post-Quit verification must
avoid app APIs that implicitly open the selected application.

### Gemma longer-context candidate built

G4AP-03's first compile caught a missing buffer trait import; the corrected
candidate compiled without changing the reviewed rollback control flow.
Test binary SHA-256:
`e225f41a1cd02dd1812c41dc72413b65e35c58d38daea28e5a4090963f11af81`;
source graph `8aa840984a76fd25d9a564d25dd9025a573e394e04cdc6894ac819571d78d397`.
Six tiny controls pass: new preaccept exact-row CPU/Metal checks plus the four
existing committed-query checks. Its actual E2B long chained oracle, batched
verification/repair and prefix controls are running under a continuous
foreign-worker and memory/paging supervisor. Those gates are not yet passed.

The first 512-token long case subsequently failed the unchanged oracle gate.
Step-one feedback/centroids stayed within their tolerances, but masked-logit
maximum absolute error was 55.125 with relative L2 0.01727 and 256 support
differences. Step-three feedback maximum error 0.625 also failed. All three greedy
proposals and root embeddings still matched; that does not pass the oracle.
The test stopped before the batched/cache controls. Peak RSS was 10.74 GB with
zero foreign workers and zero paging, so this was a numerical failure, not a
resource stop. All model/reference processes closed; the exclusive slot passed
to GLM's fresh native build while Gemma investigates source arithmetic.
Failed oracle aggregate SHA-256:
`145a3048826a2752e1e98fc99cd02db2561e5871541ce32b72b4f0d0624a9183`;
process guard `b012759ba2133ab40156dad04d32678096cab5f9c92887b66a2ede150f019f61`;
failed-gate seal `9f3cd5141d088789c9086cf0d8145fdcb29e1e7cce3aae0e2435c1b637129b13`.
The new chained-oracle tool is separately sealed as
`be71c7475d16adaba90c8a0842ebad2ed0653944b2ff53df71a6cf3bd63fe3dd`;
the prior short oracle remains unchanged.

### GLM bounded initial execution

Fresh worker/helper and native engine builds passed at the unchanged
`b76f560d` runtime source. Worker SHA-256:
`c8e528dc017809edb9c717b55c6c2dce6533e34259a4175988e9c8d4aa506f3d`;
helper `7d8f92e9552a9fd4e898113d674dd3ad235f423f7ce8a03d1ed03a2f6380164d`.
The pre-execution checkpoint
`runtime/glm-image-source-2026-11-09/bounded09-pre-execution-checkpoint.json`
SHA-256 is `1ff4591950b3ec059560a9d9f2df0130fdd2c942e1aa44b9429533c6ef7b1ce4`.
It binds the exact command, driver, fixed flags, supervisor and source/artifacts.

The first short control stopped on the unchanged pageout-growth watchdog during
the first text-only request, before any completion or image stage. Entry available
memory was 107,748,147,200 bytes; minimum observed was 91,591,655,424. Peak worker
RSS was 767,393,792 bytes, footprint 737,154,592 and host wired growth
3,830,464,512. Compression growth and swap were zero. The owned worker was
cancelled/terminated and reaped; its supervisor joined without kill escalation.
This is a host-screen/resource-deferred result, not a model/image pass or failure.
Receipt SHA-256:
`062f1ae36caa6e5266f534f85e8a821610a69e7a49d54e7de32edc9c04bfb9f1`.

A subsequent independent ten-second screen found no model/app/compiler
processes, stable pageout counter 5,503, zero swapouts/swap and no compressor
growth. Available memory stayed between 110.278 and 110.334 GB. The coordinator
granted one same-source, same-limits short retry with a new receipt; another
stop requires diagnosis rather than an automatic loop or relaxed limits.

Use actual available-byte measurements for peak admission. The earlier
`memory_pressure -Q` percentage is not interchangeable with psutil's available
bytes and must not be multiplied by physical RAM to infer a budget. Full-resident
GLM and ddalcu remain held pending their complete plans and measured headroom.

The quiet retry also stopped before text completion. Minimum available memory
was 94,429,216,768 bytes, peak RSS 772,243,456, footprint 722,671,136 and host
wired growth 3,682,631,680. Compressor growth and swap remained zero. All owned
processes were cancelled/terminated and reaped; no image stage ran. Receipt:
`e4ebed01ad400f1fc756eeec2eb2dbfc5d2ad883d287d6a05ac6ad6077c47a36`.
The first attempt's sampled interval had seven global pageouts; the retry's
interval had 102. Exact first-breach deltas were not captured, and their cause
has not been attributed. New supervision must record exact timestamped counters.

Independent review of Apple's public XNU source distinguishes global pageouts
from compressor swapouts. The [UPL commit path](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/vm/vm_upl.c)
counts cleaned pageout pages; the [vnode pager](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/vm/vnode_pager.c)
also services real-file pageouts. This public source is in the host's XNU 12377
family, not its exact private build. It establishes counter semantics, not the
cause or harmlessness of the observed activity.

A separately registered functional-only candidate may record global pageouts
without treating that counter alone as exhaustion, while retaining all existing
available/RSS/footprint/wired/compression/swap/foreign-worker limits and unchanged
production admission. Any pageout growth remains ineligible for performance
claims. The two earlier stopped runs stay stopped/incomplete; no result is
retroactively promoted. Source review and a new recipe seal precede execution.

The full-resident GLM refresh requires 127.91 GB for image-loaded text, 127.94 GB
for an image request, or 130.72 GB including the retained source-head payload,
with the declared host margin. The measured available sample was 110.15 GB;
none fits. Receipt `1b78195c3a316432176501d0e0c3e7eb301ae82cccce1af164cea4fc3d59cbc0`.
Bounded-mode parity cannot by itself qualify the previous resident text path's
speed, and must not silently replace it as a completed fast default.

The ddalcu source/header inventory separately finds 75,303,264,888 bytes of
resident allocations, including 67,947,724,800 expert bytes, 1,511,927,296 head
bytes and 897,862,112 tower bytes. Only the 32,000,153,976-byte ngram mapping is
paged; resident experts must not be described as paged. Initial state capacity is
16,384 even for shorter requests. The worst-case ngram/request/driver/host plan
leaves insufficient reviewed margin against actual available bytes. Full loading
stays held. Source-plan receipt:
`dcd593dab1b589a4cf013506324aedb3d042a8b2e79891c1080de11e9e1a1716`.

### Functional-only supervisor approved and GPU coordination established

GLM registration `19ead868` preserves the strict09 failed attempts and adds a
separately versioned functional10 recipe. Independent review of immutable patch
`a498b1d4d3320d0fd77cdfba279581d256cc3830bb7cdcfeb55296e03c3728f8`
approved the source change: hard available/RSS/footprint/wired/compression/swap
and owner-cleanup guards remain; global pageouts are timestamped diagnostics;
all functional10 performance eligibility is explicitly false. Fourteen injected
controls and five real child-process cleanup controls pass.
Supervisor SHA-256:
`9860befa761a1b7fabc563ddb873e6854fcbeb072e52b7fe6755bb83bb096b6e`;
live-smoke receipt `a40ec46f2ec143b92e55e6ffd6e7fb2c868c8263a0c30c0ce02cfbc7a9782983`.
One initial short functional run is conditionally authorized in its queued slot,
with the same runtime source/binaries and bounded flags. No resident/MTP/cache
matrix or performance promotion is implied.

After explicit human authorization, the coordinator messaged the separate
resource-budget chat to finish/drain its current Metal test and hold new GPU
tests, model loads and app reloads. Its compact status acknowledged that its
GPU run finished and it is continuing CPU checks/review. No external process
was killed. This resolves the immediate interference; future starts still
require fresh preflight and an explicit GPU handback at the queue boundary.
