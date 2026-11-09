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
| Qwen BF16 0.8B | Corrected MTP0 dispatch; 48/48 image-loaded and 26/26 text-only lifecycle checks; headed/headless and loading-mode MTP0/3 short timing passes. Source sigmoid/SiLU candidate passes stages 1/2 exactly | Stage 0 hidden maximum remains 0.09375 against unchanged 0.0625 limit. Targeted test-head projection reduction fix is under review; no production arithmetic change |
| Qwen affine 0.8B / 35B MoE | Physical-256 prefill and completion-order fixes built. Actual 35B 1026-token cache 4/4 passes and matches old-worker output; sampled footprint 30.9 to 25.9 GB. Quick text-only/image-loaded comparison 39.25/39.42 tokens/s | Original 1792-token matrix remains stopped, not passed. Short comparison covers this affine artifact only; full BF16 MoE and executable MoE head remain unqualified |
| Gemma | G4AP05 passes 15/15 assistant steps through 1792. Staged worker 136 unit tests, 9/9 smoke checks and 37/37 normal image-loaded lifecycle checks pass with actual draft acceptance/rejection, cancellation, stop and image invalidation | Matching text-only lifecycle retry and quick comparison pending. A shell-to-native identity transition stopped the first text-only launch before model load; direct native spawn fixes the adapter. Default serving MTP remains disabled pending completion |
| GLM | `b76f560d` builds and accounting controls pass; tiny mapping control passes sixty blits; staged owned-buffer Control/Q5 initialization passes without pressure | Full functional run hit a hard compression limit; neither tiny test reproduces it. Audit actual GGUF mapping/Metal registration geometry before another full load. Full-model image/cache/MTP/performance remain held |

The installed app was closed without implicitly reopening it. The coordinator
verified no competing model/compiler/app processes and zero
swap before Gemma's actual E2B gate. Gemma released its processes after the failed
long gate, and GLM completed its builds and two stopped short attempts. Qwen now
owns the next diagnostic build/model window, Gemma's trace is queued afterward,
and GLM prepares a separately registered functional-only supervisor in source.
Repeated external Metal tests from `stark-playground-resource-budget` are caught
by the preflight. The user authorized cross-chat coordination, and that chat
confirmed its current GPU run finished and it is holding new GPU/model runs.
That queue completed its initial attempts and reached a safe boundary. The
coordinator explicitly handed the GPU/native-test window back for the separate
chat's pending validation batch; Qwen may compile its next diagnostic, while
all agents hold model/Metal runs until the next explicit handoff. No PR is
authorized here.

The separate chat subsequently completed its validation batch and opened its
own PR #117. The coordinator explicitly reserved the GPU/native-test window
again. Qwen's causal diagnostics, Gemma's further trace and GLM's tiny control
completed. Gemma's independently reviewed versioned long-oracle replay passed;
Qwen and GLM prepare source changes. The reservation remains explicit, with
other native tests held. Full GLM execution and a MoE retry are not authorized
by this queue entry.

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

### Primitive-level traces and actual GLM hard stop

Qwen's valid trace is bit-exact with its normal head path. Embeddings and both
input norms match the pinned reference exactly; first drift is FC fusion maximum
absolute error 0.001953125, relative RMS 0.00251255. A separately registered,
test-only alternate reference-FC experiment will measure causality before any
production arithmetic change. Registration SHA-256:
`2b29193fbe16524aef64ac060bd24b68eafe78c764fb5f64e70d5b81febb09d9`.
Its source review found no confirmed blocker; the selected diagnostic adds a
bounded four-MiB raw-weight loading overlap. This carries no performance claim.

Gemma's merged-source trace also matches ordinary head execution exactly across
three steps. The first difference is layer-one MLP: one of 256 output elements,
maximum error 0.015625, relative L2 0.000374. Preceding projections, norms,
Q/K, softmax and attention-value products match exactly. The small MLP difference
propagates through chained feedback and changes one centroid cluster, accounting
for 256 support differences; the tie-policy comparison itself matches. On common
selected support, logit maximum error is 0.5 and relative L2 0.00371. This explains
the discontinuous masked-logit error; it does not retroactively pass the failed
strict gate. Next isolate up/gate projection, GELU, product and down projection.
Trace receipt `3cc8f86b810c1236edc2e2e67e69b29479e6646853bf8e4bec198caad3a78704`;
execution seal `bfb1b82679ea03541646092ff286cbe314746294ca42897d004cbc3522eb529f`.

GLM's single approved functional10 run then hit the hard compressor-growth
ceiling during the first text-only request, before any completion or image stage.
The exact breach sample had 62,759,632,896 available bytes, compressor growth
1,399,554,048 bytes, zero swap, worker RSS 762,822,656 and footprint 748,508,824.
During cancellation/draining, compressor growth peaked at 13,427,277,824 bytes
and available memory fell to 57,463,390,208. The owned worker was terminated and
reaped, the supervisor joined, and no kill escalation was required.
Receipt `d4b5fbd295d346ae6c37833345da4e97f70f43b5d54983e9e823cacc33188a35`.

This actual pressure invalidates treating the active-view inventory as a complete
physical ceiling. No text/image/cache/MTP/performance pass exists. Full-model
execution stays held without weaker hard limits. Investigate full-file versus
range mapping and no-copy versus bounded-copy physical lifetimes with a tiny,
separately registered nonmodel fixture before any full-model retry.

An independent ten-second post-exit screen found 105.23–109.40 GB available,
zero swap, stable pageouts at 5,964 and no compressor growth (194,666→194,662
pages, approximately 3.19 GB retained). The prior pressure remains recorded.
Qwen was granted only a CPU diagnostic build while the resource-budget chat
received the GPU/native-test window back; no model may start merely because a
momentary process screen is clear.

### Further Qwen cache and numerical evidence

The expanded production-worker matrix passed 48/48 image-loaded and 26/26
text-only cases on `1e592b48`, with zero paging/foreign workers. It adds shrinking,
growing and divergent requests, final-position 1024/1025/1026 guards and primed
failed/cancelled/real-image recovery in MTP0/3. Receipts:
`aeb8a3034c56bc34ffdcf172be105906e9f8951d87fd50100c2e5fdd06c72012`
and `cadb6fe0a8f910c5ab674f9cce943644431bb092d1efd38ecdc7c969dc24225f`.
Prior matrices and timing evidence remain separate source-bound records.

The test-only reference-FC candidate eliminates FC error exactly against MLX,
but full hidden-feature limits still fail (maxima 0.25/0.1875/0.1875). Local norms
and residual addition are exact; attention and MLP retain drift. No production
head precision policy was adopted. Aggregate:
`a623312e224ff374b7ee7a43e3f551d27da654e5865f69765a4369671dad8ba3`.
An all-head-linear test-only experiment preserves the incoming Legacy/nonlinear
policy, rejects incompatible/conflicting selectors, and declares its scratch
overhead before execution.

Corrected MoE attempt03 stopped on one global pageout (16 KiB), with zero swap
or foreign workers, before case zero completed. Peak RSS was 31,748,620,288 and
minimum available bytes 74,006,904,832; neither hard memory limit was hit.
Receipt `adceb2e976070e6abb9417542888a22a606acc0c15d06c6a28abab247fbc688d`.
A separately registered functional-cache monitor is under review: keep all hard
limits and zero swap, add compressor monitoring, record global pageouts without
performance eligibility. No old stopped case is promoted by a new policy.

### Gemma source GELU isolation and remaining projection drift

The finer trace localized first drift to legacy GELU (18/2048 layer-zero values,
max error 0.0000153). An existing source-matching VisionGelu kernel gives exact
activation, BF16 product and down output on matched native inputs across all
four layers and three steps. Trace receipt:
`f8ec889d92658ef93b84d58514e911d6354ffedd909da24e5039378f8e222794`.

The independently reviewed assistant-only change was built and tested under the
unchanged long gate. It fixes those isolated operations, but the full aggregate
still fails and later batch/cache controls did not execute. Follow-up trace now
localizes first drift to layer-one fused-up projection: one of 4096 elements,
maximum 0.0000305176. Corrected layer-zero operations match exactly.
Full-gate receipt `507d55792f6291ccffb28b0b35165f35580a3d6cb42af28fbd8b1f60f03c1df8`;
trace `39177cfb4fe99019d6cd060ea83b53606355eef8af142b06d935d64aa07128ea`.
Checkpoint `772993fe` preserves the partial fix and failed full gate. Next compare
the existing reference matmul on native up/down inputs in a test-only experiment.
Ordinary target kernels, the >512 draft guard and serving capability stay unchanged.

### GLM tiny mapping result and performance constraint

The reviewed 64-MiB synthetic fixture uses distinct absolute-position records
and rejects a wrong-offset CPU read before sixty GPU blits. All copies match;
45 no-copy deallocators run, no unmaps fail, the fixture is removed, and the
owned child/timer/supervisor drain. Peak wired growth is 368,001,024 bytes, with
zero compression/pageout/swap growth. Receipt:
`9c9daec77cbc6ef58845fd54ca1ea1c590b5fc1550222f6a72d2fbc2604c9994`.

Metal wrapper allocations return to baseline after pool drain, while native task
footprint remains around 135–144 MB after the first completed blit. Full-file
mincore counts the warm 64-MiB backing; dedicated-range mincore counts its own
4,210,688-byte span. These are different measurements, not additive unique
committed allocations. No compression problem is reproduced and no cause or
mapping-policy fix is established. Full-model runs remain held.

A private descriptor-bound owned-upload CPU primitive passes four small tests,
but is not integrated into a backend. Whole-bank uncached uploads would read
99,033,808,896 bytes per target token; that route is reference-only and cannot
be promoted as recovery of normal text performance. Further full-model execution
requires a viable ownership/residency plan and appropriate performance evidence.

### Further source isolation and exclusive-run results

Qwen's test-only all-head reference linears remain rejected: hidden maxima
0.1875/0.25/0.15625 exceed the unchanged limit and step two changes top-1.
Aggregate `d57901432eac1712f820a03f24d644688327c5cf5ea36a9836eebb4190e56e3c`.
A separate metadata derivative materializes only the head's BF16 SiLU sigmoid,
with the existing reference-FC test engine. Local MLP error improves from
0.015625 to 0.00390625, but full hidden maxima 0.25/0.125/0.1875 still fail.
Logit and top-1 gates pass; this does not establish full head qualification.
Aggregate `317613dde312b9da733aa65980a724ac43ea325ace5f6714207fe8688b9c592d`.
Both experiments leave production precision unchanged.

MoE04 uses a separately reviewed functional-only monitor; global pageouts are
diagnostic and performance eligibility is always false. It stops during the
first 1792-token prefill, before any lifecycle case completes, when available
memory reaches 68,421,844,992 bytes below the registered 68,719,476,736-byte
floor. Final minimum is 68,270,899,200, peak RSS 36,751,753,216 and peak task
footprint 37,054,278,344 bytes. There is zero pageout, swap, compression or
foreign-model growth; all owned processes drain and exit. Receipt:
`31b0c7bed7a4362e56669a061b56ba097e54675853ac00bb64b37f3b3ebeab92`.
No retry or relaxed floor is granted. The next source audit examines flat
prefill's per-token recurrent-state histories while preserving FP32 recurrence,
1024-token snapshot boundaries and speculative rollback.

That initial history-allocation hypothesis is subsequently disproved for the
actual frozen worker too. Its source seal binds `4010b86e` to `0d7aea46` plus
listed overrides; DeltaNet, stream, pool and MoE files have no such override
and match the audited source. Ordinary flat prefill already retains only final
FP32 state, while speculative verification retains rank updates for acceptance.
The DeltaNet source hash is
`e95530abf21e5498812a004092df290940dee6c280c2cd14d30407fadd261174`.
A duplicate final-state backend is unwarranted. Scratch/router/pool lifetime
attribution is the next investigation; the stopped run remains inconclusive.

The routed MoE path emits at least six dispatches per active expert and input
row, giving at least 1,966,080 expert dispatches for a 1024-row, forty-layer
chunk. The current stream keeps the 1024+768 prompt chunks in one encoding,
at least 3,440,640 such dispatches for the stopped prompt. Pool scratch handles
return to their size bins as route-local values drop; command encoding memory
remains a distinct hypothesis. A proposed one-layer allocator measurement
precedes a fixed 256-token physical execution cap with submit/wait/reuse.
Logical 1024-token prefix snapshot boundaries must remain unchanged. The
256-row choice preserves the existing chunked GDN route; changed batch shapes
still need new numerical, cache and timing evidence. No implementation or
full-model admission is claimed yet.

Gemma G4AP05 (`21f29608`, engine
`75e5eff55ebcdf9cdbc0a22d1dcae7220e181fa74358c07d58215c4e3431a320`)
uses assistant-only reference up/down projections and source GELU. All three
oracle steps and transactional acceptance/rejection/cancellation controls pass
at 512, 513, 1024 and 1026 tokens, including actual 3/3 accepted drafts for
each input. The 1026-token prefix boundary and changed-tail handoff pass.
Receipt `0c944464ea413ba0e3005375b6b0a8f9fe179c01ef4cec24e98f69ad52f4ebf1`.
At 1792, step one still fails the original raw masked-logit comparison, with
exact feedback and centroids; later batch/cache controls do not execute.

The subsequent diagnostic derives actual native support from the observed
mask, validates whole clusters, sentinel separation and the token permutation,
and compares a canonical readout. At 1792 it finds 29 strictly greater scores
and four exactly tied cutoff scores for three remaining slots. All changed
clusters are cutoff ties, native support matches the independent canonical
selection, and canonical readout error is exactly zero. All fifteen steps
have valid top-k selections and canonical maximum error at most 0.125.
Receipt `b05242f3a2f35ba552a9db6469ad5b944418ac4eeff001110d5ed3f1517be69d`.
This diagnostic deliberately preserves the old raw failure. A new qualifying
oracle must prove actual reference-mask support too, reject non-tied changes,
retain unchanged numerical limits, pass meaningful negative controls and receive
independent review before execution. Serving MTP and the >512 guard remain
unchanged pending qualification and integration.

The new `mathematical-topk-canonical-tie-v1` qualification proves both actual
native and actual reference mask support, uses reference top-k values rather
than unstable tied indices, and rejects every non-tied mismatch. Fourteen CPU
controls include invalid actual reference selection and valid alternate cutoff
tie selection. Helper SHA
`d47525f9d46bb0a5051f3982cf88e3fb697751c141ff009bd994a351f1a2f597`;
wrapper `3fd623cc257a5e188d9a107163cf1a7377f48bc478d7162436a0631a3ae587f8`.
Independent GLM review found no confirmed policy blocker before execution.

The unchanged `75e5eff5` engine then passes all 15 assistant steps under this
versioned contract, all five actual three-draft rounds (3/3 accepted each),
ordinary-target equality, supplied full/partial/reject/zero/cancel/fresh controls
and the 1792-token cache handoff at boundary 1024 with suffix 768. Changed-tail
and 1026-token suffix-two checks pass too. Receipt:
`15a0a1b79b62948bd1dde63bff1b2a64b309e40850b787e2f7611ad32a482ac3`.
Guard `ab79d846a4b5182eb09f69354cbb70266519fde0c0128c0f169f8f15065d08c8`
reports zero paging/foreign workers and peak 12.534 GB below the unchanged
12-GiB cap. All owned native/reference processes exit. The original 1792 raw
comparison remains false and retained. This is component qualification, not
normal-worker loading, serving enablement or a performance measurement.

The coordinator independently reruns the tracked fourteen CPU controls against
the pinned Torch build: all pass in 0.418 seconds, with the same deterministic
receipt hash `7e2a8e3009907edb30bf0d35e956fc586d4db4f044131e13633cbab5b2940a79`.
The reproducible helper, original raw oracle, wrapper and controls are committed
on the Gemma branch at `3c2791ec` under `tools/validation/gemma4_assistant/`.
Normal-worker integration remains source work pending independent review.

A proposed matched-arm control that loads and drops the image tower is rejected
before benchmarking: Metal global-buffer drop retires storage into an allocator
cache and therefore would retain backing allocations. The staged validation-only
text arm must skip tower allocation for the exact admitted artifact, retain the
same decoder/head/policy and report actual Metal allocations at readiness.
The normal loading path retains default image support. No speed claim follows
from the rejected drop-only control or unfinished worker source.

GLM's corrected first-use audit identifies layer-zero `attn_q` as kind-13 Q5_K,
not Q8 merely because Q8 tensors occur elsewhere. `Control::pre` also triggers
the GLM libraries and eager graph pipeline creation before that packed
projection. Audit receipt:
`17f02cba7e625973f8d346b173c1213e0ed76e75a8054c5ddb567467cd396989`.
A proposed tiny staged Control-to-Q5 test measures these phases independently
of full-model mapping and captures compiler-service peaks through cleanup.
It is a source proposal; neither a cause nor a pressure fix is established.

The source-only package is subsequently frozen as registration
`2e4fd0d4dd69aeabcb0f3ecc41e6407be539431f585679132a45d7c265679138`,
with source graph
`39715e1e825f8291d5fa2974b3ccdb94019f8177a30e041ddc5c14bfb5cd05be`.
Root reviews the ignored test, independent projection oracle, phase/start
gates and owner-only supervisor. Twenty-one injected CPU controls pass, including
new compiler-service growth that an exiting baseline service would otherwise
mask. Positive per-PID RSS growth and net growth are recorded separately.
Limits are 1 GiB for owned RSS/footprint, host wired growth and positive service
RSS growth, 256 MiB compressor growth, zero swap and 60 seconds; no model or
mapped file participates. CPU receipt:
`876c72e95c74d02ee78e969f4db5ae0ae1f6f20aab59318172d0c5a489c6cbc7`.
An exclusive build-only slot is granted, reusing only this checkout's test target
after archiving/hashing its old binary. Native execution requires the new
source/binary seal and host screen; full GLM remains held. Persistent shader-cache
state is unknown, so the result cannot establish cold-cache behavior or speed.

The build passes in 89.36 seconds, producing binary
`8e7abaeeca0f052d56ae0e0e9afee04099379cb4b6ec29c00662fc4d65653afa`.
Build seal `2157e54902b5b07cb2f39257510f81bed7936eaf970cef3fcfe6ceb89552abf2`
retains the original source binding. The first runner attempt selects zero tests
because its exact filter omits the `stark_engine::` crate prefix. Its zero-phase
receipt `44cd32a2708f1492f242bd2be1e0be34d60035fa6b789210fc9467a72c7ca0ca`
remains invalid. The corrected execution driver is separately registered as
`5ea8d20344939292848433f516c06ec666a2ee9e22b180a52a25a86ceb93343e`,
with an exact-one list-only preflight and all 1267 original source files verified
unchanged. No new build or relaxed limit is involved.

The corrected run passes all eleven phases and independent Q5 projection checks.
First graph completion is 0.653 seconds after test start; test buffers drain at
0.654 seconds. Metal allocations peak at 1,867,776 bytes and finish at 720,896
with process-owned library/pipeline singletons remaining. Sampled compiler-service
positive RSS growth peaks at 59,310,080 bytes; own RSS/footprint peak at
19,677,184/8,946,216 bytes, host wired growth 475,136. Swap, compression and
pageout growth are zero. Receipt:
`f9706505aec9701d27555ff0dfc7856b809b323a496e669b73acf54633b8b058`.
The supervisor drains/joins and the owned child exits; two live samples and four
post-drain samples cannot rule out brief unsampled compiler peaks. Shader-cache
state and cfg(test) source differences also limit inference. This owned-buffer,
no-model result does not reproduce the full failure or prove a full-model fix.
Next audit the actual GGUF VM-map, logical-tensor and Metal-registered spans
before proposing a bounded real-file contrast. Full GLM remains held.

### Pinned external implementation research

The user authorized consulting other open source engines. These sources guide
bounded experiments; their reported performance is not Stark evidence:

- [MLX-LM gated DeltaNet, revision 56230fa0](https://github.com/ml-explore/mlx-lm/blob/56230fa02b8ff8330daf26dc95c28480423ed5fe/mlx_lm/models/gated_delta.py)
  retains FP32 recurrence and emits the final state separately from token outputs.
- [oMLX routing, revision 2165e75](https://github.com/jundot/omlx/blob/2165e75726ba83d9651dbaced61c97728da77bba/omlx/patches/qwen35_gdn_chunked.py)
  uses optimized prefill only for supported layouts and falls back when
  speculative cache/state steps are requested. Its
  [Metal kernels](https://github.com/jundot/omlx/blob/2165e75726ba83d9651dbaced61c97728da77bba/omlx/custom_kernels/attention/gated_delta.py)
  keep FP32 recurrent state. This supports separating ordinary final-state-only
  execution from rewind-capable execution; it does not prove a Stark patch.
- [PyTorch top-k documentation](https://docs.pytorch.org/docs/stable/generated/torch.topk.html)
  explicitly leaves tied indices unstable. Mathematical cutoff-tie validity
  needs direct mask evidence, independently of any particular heap ordering.

Family agents retain additional pinned MLX-VLM, Transformers, MLX and llama.cpp
source/license records. No borrowed implementation is qualified merely by
matching its design; new validation remains bound to Stark source and artifact.

For MoE command-memory attribution, Apple's
[allocator API](https://developer.apple.com/documentation/metal/mtl4commandallocator/allocatedsize%28%29)
exposes internal command-heap bytes separately from scratch payloads. Its
[reset contract](https://developer.apple.com/documentation/metal/mtl4commandallocator/reset%28%29)
requires completion of all dependent command buffers first. This provides a
direct diagnostic for the command-overhead hypothesis; no cause or fix is
claimed from dispatch count alone.

### Measured MoE command memory and new Qwen source candidates

Two tiny diagnostic setup failures remain excluded: one build used the wrong
backend cfg spelling and omitted the test; the next run selected a full-BF16
profile incompatible with affine weights. Exact-one listing and the corrected
affine profile precede the successful run. Failed run receipt:
`877818ee09ff7b99e2fa4634e4f6adaf3984dec3816a23c1585d460f25be7dd0`.

The corrected one-layer affine4/group64, eight-active-expert operator passes.
At 16/64/256/1024 rows, measured Metal command-allocator bytes are respectively
3,809,856 / 13,787,712 / 53,862,976 / 213,983,808. At 1024, scratch payload is
460,032 bytes and constant pages 1,310,720. Four subsequent 256-row submissions
reuse the same completed/reset allocator at its existing high-water size,
without shrink or further growth. Split outputs are exact, including absolute
row digits and a wrong-offset-248 negative control. Curve receipt:
`f1d4cb775622feca5e5b65df5959b626c0c89fe30a3a8329085a7a803044fb3c`.
Guarded receipt `4b67c785ed66f8cb2b024bc52b41947be44c772fe64b93d9cd5166501ce72524`
reports zero paging/compression/swap/foreign workers and drained/joined owners.
This measures a small operator, not the full model's command bytes or peak.

Two separately reviewed source changes await compilation: publish the completed
allocator cache entry before waking a waiter (`83d5ef1b` patch prefix), and cap
ordinary typed-Qwen-MoE text submissions at 256 (`78cf769f` patch prefix).
Images, image continuations, dense targets and generic speculation keep their
previous schedule. Logical 1024 boundaries remain fixed; all submissions finish
before prefix publication, and partial failures fence state. A synthetic dense
state fixture and CPU partition controls are planned; actual GDN/full-model
numerics, cache lifecycle and timing remain required. A context may retain more
than one large allocator across decode/request cycles, so a full-model estimate
must include multiple cached high-water heaps rather than assuming one heap.

The finer BF16 head diagnostic completes all three guarded execution phases in
2.647 seconds, while the unchanged full hidden-feature gate still fails.
Aggregate `a95181746225450b3916e70026d6f907d90be0ea0a4efc7654b23a9463bf9cca`;
primitive receipt
`05f0f4da8e26dc19c6a6069e62d14da3cf6d92df4232ff2155d1cfca4c1a1541`.
Duplicate attention/MLP and manual/actual hidden outputs are exact. On native
operands, Q/K norm, rotary Q/K/V and SDPA match exactly. Differences remain in
the sigmoid/product boundary, SiLU activation and a few projection elements.
No inference from aggregate attention drift is used to rewrite SDPA.

The coordinator's independently reviewed `d2cdde18` adds an isolated test head
experiment: reference linears plus source BF16 sigmoid and materialized SiLU.
The selector requires a Legacy incoming head, all-reference-head mode and no
FC-only conflict; only a cloned head transformer config changes. Production
selectors remain false. Eight source files, public rounding/stride controls and
pinned MLX MIT/source notices are bound by patch
`658da6f8d4a46bac0e13b2c63a98f7d1c3bc4672530227218fed0348ed292847`.
Formatting and policy checks pass; compilation and actual unchanged-gate replay
remain pending. This is a test candidate, not an adopted precision policy.

### Gemma staged worker compilation and rejected lifecycle supervisor

The independently reviewed normal-worker candidate builds with Rust 1.98.1.
An initial private-type import error is fixed by a public assistant re-export;
the failed build remains recorded. Staged worker
`a356edf19c1652536a2d2feb12c096e57458105bf6500846779f5ca295997358`
and matching helper
`77e45824cb995adc0322f27a466f9cd4cf13f1bda9edb1e8cf2e107b457ea799`
are bound by seal
`5e99a297af787c07a78dd13e26b4c54d4c6e676c549004433485d9c1a0a686c4`.
Worker unit results are 136 passed, two ignored; no model/Metal execution is
claimed from them. The loader's skipped-subtree CPU control is next.

Before any normal-worker model run, independent GLM and coordinator review
reject the inherited supervisor in fixture `60600e1e`: telemetry exceptions can
silently end monitoring, and its final result ignores guard stop/join/sample
status. This is an unexecuted validation-source defect. A separately versioned
fail-closed supervisor must monitor the owned worker before readiness, pin its
dependencies, preflight fresh receipts and require positive/final samples,
drain/join and all hard limits. Stop, two-token-cap and image-zero controls also
need explicit finish/count assertions. No lifecycle or performance pass is
claimed from the rejected fixture.

### GLM bounded real-format mapping proposal

Header/source audit `328d8d183893344185c8ee66eaecb05d4f6fa9e1d89cc31f17227e3aef64a30d`
finds the first Q5 tensor's original shard is 49,294,975,936 bytes. Its logical
payload is 23,068,672 and registered span 23,085,056 bytes, with in-page offset
9408. A dedicated range into that original file still shares the large backing
object; a 23-MB logical window is not a proven physical ceiling. Original-shard
GPU views remain held.

Permitted CPU preparation reads exactly the 23-MB payload with `F_NOCACHE` and
unchanged descriptor identity, then creates private capped 64/256/512-MiB sparse
fixtures with identical tensor bytes/registered geometry. Four pre-I/O invalid
range checks and an actual wrong-offset hash negative pass. Receipt:
`5e112a4b7ab2325a86a3496fef09a804136e1c1359ae76d8f20f6f5fb7a686e3`.
No mapping or Metal execution occurs in that preparation. The bounded contrast
will use fresh child processes per arm to avoid retained-footprint confounding;
native source, build and execution still require review. Full GLM remains held.

### Targeted fixes and actual worker results

The user requested keeping implementation focused. Further calibration variants
and new validation infrastructure are stopped; the remaining runs reuse existing
checks and quick comparisons.

Qwen branch `f249e1dc` includes the reviewed completion-order and physical-256
prefill fixes. Worker SHA-256
`7f41246a4e265ad2ba4060c64a57888c4d0541a758f19ec4e0c9f80511c5700a`
passes all four actual affine 35B 1026-token cold/repeat/changed-tail/fresh cases;
receipt `c6813324f7eb3fdf598a20faed58f73898a07c5c3188bdda7dce332ba7eb48e2`.
Its output matches the old worker at the same length. Old/new sampled footprint
is 30,881,831,136 / 25,884,543,840 bytes, with zero paging, compression or swap.
The old-reference receipt is
`e753f4748357b8004b38a209030c7270a6527b43f8bfd891a6fcd779f2fca999`.
The original 1792-token stopped run is retained and does not become a pass.

Six measured 64-token requests per loading mode give text-only/image-loaded
medians 39.2466 / 39.4158 tokens/s, identical outputs and passing fixed 5% checks.
Receipt `419b331222b50eec9f0e34eb3be965d8b902fbfc292d8e749ca7e9fcdf9ad119`
is a short observed comparison; its functional guardian remains marked
performance-ineligible. Image-loaded startup is approximately 74 seconds versus
2 seconds text-only, separate from steady inference.

The Qwen BF16 source-operator candidate was built and tested. Stages 1/2 match
hidden features exactly, while stage 0 maximum error remains 0.09375 against the
unchanged 0.0625 limit. Full-gate receipt:
`e10a3413289d4b50cd1d5a0f3a0d05ccc763ce788ba839a6eb14840e42e9bd2b`.
A targeted test-head projection reduction adaptation from pinned MLX is under
review; ordinary production arithmetic is unchanged. The fast-exit primitive
attempt lacked process telemetry and remains excluded.

Gemma staged worker
`a8775bb8fe43e0bdcabc3e3433de869e705089baa2f35bb4fc9ecf963aafff14`
passes 9/9 smoke checks and 37/37 complete image-loaded lifecycle checks. Full
receipt `5fc7f363e8aaf9cdba8badc550245a2b48fedebbd5cb3c8c39b632c24af48d7b`
includes actual draft acceptance/rejection, seeded cache agreement, stop/cap,
mid-generation cancellation and image invalidation. All 145 guardian samples
report zero paging, compression or swap. The first matching text-only launch
stopped before model loading because its shell shim changed process identity;
that failed receipt is retained. Independently reviewed direct native spawning
removes the shim while preserving guardian checks. Matching text lifecycle,
then the already authorized quick comparison, are the current exclusive window.

GLM's small mapping contrast remains unbuilt and held. Sparse fixtures were
replaced by fully materialized high-entropy private files before freezing the
minimal owned/dedicated/full contrast. No full model retry or further variants
are authorized by this checkpoint.
