# GLM EXL3 throughput comparison and gap investigation

Date: 2026-11-04. Investigation in progress on
`codex/glm-prefill-throughput`, starting from `d0c54deb`.

The objective is approximately 1,000 prefill tokens/s and 33 decode tokens/s
with validated inference and a bounded memory budget. The current qualified
Stark result is 723.751 prefill and 17.070 decode tokens/s. The historical
mlx-stream cell reports 1,093.5 and 33.15. These are observations from different
configurations, rather than an isolated regression measurement.

The [Stark qualification record](../archive/runs/2026-11-04-glm-exl3-qualification.md)
preserves the original source, artifact, precision, executable and result
bindings. The [upstream record](https://github.com/davidtai/mlx-stream/blob/7d29eaa7217ea8d0e1b50000eb23ad37b06f5d80/docs/benchmarks/glm53-flash/README.md)
identifies the historical cell and its controls. New experiments retain separate
private source-bound receipts; their results must not retroactively qualify an
earlier rejected run.

## Retained working profile

The selected baseline for continued work is the completed 16,384-input /
1,024-output profile at **922.6615 prefill / 20.3003 decode tok/s**. Its recipe
uses the BF16 draft head, depth 3, gate 0.5, original native draw-match
acceptance, transition-weighted expert retention and direct reads. Sampling
is temperature 1 / top-p 0.95 / seed 1, with a 32,768-token context.

The recorded run admitted 129 prompt / 156 decode entries within the existing
memory ceiling and reserves. Those counts depend on the host baseline; normal
admission remains in force. This is a diagnostic result with a failed external
CPU screen and unverified clocks, not a new throughput qualification.

Further acceptance, Q8-head and cache-retention experiments are stopped at the
user's request. The experimental alternatives and operator-pin overrides remain
disabled. The prediction-only scorer remains unmeasured; the earlier request
for an idle benchmark window is superseded by this selection.

## Configuration and evidence

| Area | Stark qualified baseline | Historical mlx-stream cell / current public host | Performance effect established so far |
| --- | --- | --- | --- |
| Throughput | 723.751 prefill / 17.070 decode | 1,093.5 / 33.15 | Observed gaps of about 34% and 49%; causal split unknown |
| Weights | Verified HF revision `50237d5c26ceb98bfd85f779990b0f5e6d7af144`; EXL3 K3 experts, FP8 residents | Same advertised pack format | Exact historical payload identity remains unproven |
| Engine source | Public host `dc92a3e8`, plugin `7d29eaa7`, pinned MLX/MLX-C | Historical host `a8f6c6b8`, plugin `88d2aad` | A screened current-public-source host control measured 835.678 prefill / 16.632 decode at 131/159 rows with MTP off |
| MTP | Disabled in qualified baseline; native commit interface and head-inclusive prefix snapshot implemented in a private candidate | Depth 3, gate 0.5, exact acceptance, BF16 draft head | Same-policy greedy128 output hashes and round counts match the normal host. Observed native-host decode 23.523 / 23.668 tok/s failed the external CPU screen; production throughput remains unqualified |
| Expert rows, prompt/decode | 142 / 169 in the qualified headless baseline; latest head-loaded pair 104 / 153 versus 104 / 130 | 190 / 217 | At fixed 104 prompt rows, 130-to-153 decode rows repeatedly cut settled payload by 16.91%. High Power observations were 21.16–21.38 versus 19.32–19.34 decode tok/s; the clock-matching gate failed, so the gain remains provisional |
| Allocation admission | 112 GiB ceiling, 2 GiB margin, native census and automatic fill | Same ceiling/margin; 190/217 requires a baseline at most 12.69 GB | After the operator closed nonessential apps, an exact metadata admission census measured 35.455 GB, down from 46.766 GB. With an additional 2 GiB selection reserve it admitted 153 decode rows. The upstream 12.69 GB is an admission threshold, not a measured baseline for the 33.15 tok/s run |
| Residency capacity | About 81 GB, bounded by the model bill and remaining budget | Current public host defaults to Metal's recommended maximum; historical policy not independently established | Fixed-row ABBA diagnostic observed under 1% effect and exact readout fingerprints; its external CPU screen failed, so no throughput gain qualifies |
| Prompt-combine prerequisite | Unset wiring environment selects `inplace`, despite bill-bound capacity | Unset/`max` selects `inplace`; `fit`/`off` selects `fused` | The plugin couples this route to residency policy. The capacity experiment keeps the route unchanged in both arms |
| KDA | Baseline did not record installation; new fixed-row control confirms per-core on 40 cores, 80 groups, 104 rows, 2 groups/core | Same expected plan | Stock fallback is ruled out for the new control |
| DSA / EXL3 / fusions | NAX, 2,048-row waves, route cut, sampled ordering, direct reads and launch fusions configured; normal-host startup confirms NAX | Tuned defaults present | Source configuration aligns; full invocation counters remain unchecked |
| Prefill tail | Eager completion inside prefill | Deferred phase/growth and MTP catch-up | The MTP-on chat pair produced identical outputs and observed about 77 seconds total in both cells; both failed the CPU screen. No qualified speedup or decode-gap explanation |
| Prompt residents and startup | Current `keep` default; excluded warmup then repeated reset requests | Historical `forget`; one request per fresh server, no eager warmup | First requests after load reached about 918–936 prefill tok/s in screened serial controls; later requests slowed. These are fresh-server observations with existing OS file cache, not cold-storage measurements |
| Readout and sampler | Baseline uses complete 154,880-value F32 readout and CPU greedy argmax; private candidate uses native chosen-token rounds | Native chosen-token/card path | Actual sampled host/control and 41-case worker checks match native references. A screened first-request sampled host cell measured 884 prefill / 16.6 decode; the adapter is not supported as the dominant decode-gap explanation |
| Input | 16,384 raw-source tokens | 16,381 templated tokens | Contents affect expert routes and MTP acceptance; the three-token difference is not the substantive issue |
| Decode length and sampling | 128 greedy steps | 1,024 sampled steps, temperature 1, top-p 0.95, seed 1 | Different trajectories and fixed-cost sensitivity |
| Context and prefix reuse | 32,768; reuse off for throughput | Same context; reuse off | Aligned. The serial stage has no snapshot cost; prefix validation is separate evidence |
| Hardware conditions | M5 Max, 40 GPU cores, 128 GiB; macOS 26.7.1 (25G241); Automatic initially, operator-enabled High Power for the repeat | Same advertised chip/RAM, maximum fans and clock-matched pairs; historical OS build not established | Workload IOReport capture now works. Automatic-mode decode clock fell from about 1,568 to 1,002 MHz; High Power reduced that collapse but did not meet the paired 0.5% clock gate. Maximum fans and RPM were not established |
| Global GPU wiring setting | Read-only `iogpu.wired_limit_mb` query returns `0`; unchanged | Published serve recipe explicitly sets `114688` | The raw zero does not establish the kernel's default capacity. This is separate from MLX's measured per-process residency capacity; no causal effect measured |

The public pack sets MTP lane construction, depth and gate. The qualified bridge overrides
the lane off and the prompt tail to eager completion. Most advertised kernel
switches are already inherited from the pinned plugin. Enabling their names
again is not an evidenced optimization.

## Same-source normal-host control

The current public server was built from the same host/plugin/dependency source
as the bridge, with private settings paths and no changes to inference. A fresh
server used 131 prompt / 159 decode expert rows, MTP lane construction disabled,
eager tail completion, `keep` prompt residents, maximum recommended residency
capacity, a 32,768-token context, the identical raw 16,384-token input and 128
greedy outputs. One warmup was excluded, followed by two measured requests.

| Measurement | Request 1 | Request 2 | Median |
| --- | ---: | ---: | ---: |
| Prefill tokens/s | 849.813 | 821.543 | 835.678 |
| Decode tokens/s | 16.670 | 16.593 | 16.632 |

Native histogram deltas established exactly one successful request and the
expected input/output counts per measurement. Host screening passed: maximum
external CPU 134.585 percent, thermal state zero, no paging or swapout increases,
compiler/download candidates, monitor failures or sampling gaps. Startup
confirmed the per-core KDA plan and NAX route. The process group was reaped and
the original private configuration restored. The verified model pack and user
global settings were unchanged.

The HTTP endpoint does not expose full readouts, so this result does not
establish readout/token parity. Its serial decode rate is close to the bridge's
fixed-row diagnostic. It provides no evidence of a twofold adapter overhead.
The cold excluded warmup's prompt phase took 17.376 seconds (about 943 tokens/s);
that is a diagnostic observation, not a qualified cold-request speed result.
Tail, resident policy and fresh-server timing remain a separate experiment.

Private result SHA-256:
`053af3675a498f41c4afaba2415a124f60b1dbd390173768d6d918f451240666`.
Controller SHA-256:
`fe4cd9851f56300f200d9ad27d0f450dea9fa9d9b8873181c2343e6924f6917a`.
Server SHA-256:
`c14fa2ecba008b3a59c5a1f6ee4bdd5b16731a081cd3368a5e31d33c05027456`.

## MTP numerical policy investigation

A common-prefix diagnostic found the first greedy disagreement at output 64,
in zero-based verification row 2 of a three-row round. The entire offending round was
accepted, so a rollback in that round cannot explain it. Initial prefill
readouts and logical state matched. At the disagreement, corresponding target
readouts had no non-finite values, maximum absolute difference 0.8125 and RMS
difference 0.141910; their argmax differed. Logical state had already diverged
before that round. These observations locate numerical drift without proving
its cause or its acceptability.

The pinned source explicitly describes fused serial KDA as rounding-class
relative to the operation chain: convolution rounding, normalization epsilon
placement and gate summation differ. Fused verification promises agreement
with the captured operation chain. A separate matched `decode_kda=ops` control,
with fused verification retained, still disagreed after 128 greedy outputs
(44 rounds, 96 verified drafts, 84 accepted). Therefore changing serial KDA
alone does not establish whole-model agreement.

A direct source comparison found no changes to KDA, its fused decode and verify
implementations, FP8 projection or the native sampler between historical plugin
`88d2aad` and current `7d29eaa`. The historical defaults already used fused serial
KDA and fused verification. A later KDA change therefore does not explain this
disagreement. The historical exact-stack fixture result remains evidence for
its recorded configuration and workload; it does not qualify the current one.

MTP remains unqualified in the production bridge. Further evidence must
separate adapter correctness against the identical native policy, verification
and rollback correctness at matching execution shapes, and the quality of any
new rounding-class policy. Different generated text cannot itself establish
either a correctness defect or acceptable quality. The original serial F32
readout evidence remains an unchanged reference.

Private common-prefix diagnostic result SHA-256:
`f34e024eba6bb55a01e51ad8332a4a2b681add8da44466774c5f2251b87eacfb`.
Its native source SHA-256:
`114cf826c1734626c69cbfa2eedebbe4f5d810538536d1be985cfd6b8e6d41c2`.
Private ops-control result SHA-256:
`497e17b74a9f9b6491299fa574f780070a2db565e4a3f428bc5c6104d243cea5`.
Its native source SHA-256:
`2f0bd427d38ff480d116007268e035eb5b788f8d8a18d84d79a084f6bab717c3`.

## Initial experiment order

This was the initial sequence. Results and remaining gates are recorded below;
the list is retained to make the investigation's controls reviewable.

1. Record the actual KDA installation, effective NAX route, admission terms,
   allocator active/cache/peak bytes and process footprint. Observation must read
   the existing installation result without rerunning its probe.
2. Compare bill-bound and maximum recommended residency capacity on one loaded
   serial model at identical rows, using excluded warmups and ABBA order. Keep
   allocation admission and the live memory/paging stop guard unchanged.
3. Compare the normal current-public-source host with the bridge at identical
   artifact, rows, context, tail, residency, input and sampling settings. Disable
   MTP lane construction when measuring the serial path. Isolate host settings
   without changing the user's global configuration or the verified pack.
4. Resolve the MTP control's token agreement and connect the sampler/commit
   interface. Then measure drafting and the complete feature at their admitted
   memory footprints. A loaded no-draft control is not a measurement of an
   unloaded MTP lane.
5. Measure 1,024-output requests, first-token latency, first decode interval,
   steady decode and total request time. Record the achievable expert-row
   frontier rather than forcing an over-budget pin.

Each experiment changes one relevant variable, checks outputs/state as
appropriate, and records actual installed settings. A new kernel project or a
large settings sweep is premature while runtime engagement is unknown.

## Initial MTP diagnostic progress

These paragraphs record the earlier failed pilots. The later native-policy,
worker and quality sections contain their follow-up results.

The first native MTP pilot failed before measured arms, without identifying the
cause. A separately built diagnostic retained the same inference settings and
added bounded integer failure stages. It completed the first depth-three warmup
and identified a generated-token hash mismatch: 128 committed tokens, 44 rounds,
109 gate-admitted verified drafts and 84 accepted drafts. It did not report an
execution error. No MTP timing or token-parity result qualifies from that run.

The unchanged greedy 128-token control also completed native execution and
failed token agreement: 45 rounds, 99 verified drafts and 83 accepted drafts.
The issue therefore also exists without random sampling. The next diagnostic
must find the first disagreement and compare corresponding target readouts
before their input histories diverge; later readouts are not a common-prefix
numerical comparison.

The pinned sampler describes position-keyed draws and pointwise serial/drafted
agreement. The next check separates greedy agreement from sampled behavior and
then examines the earliest disagreement using aggregate position/error/hash
information only. Actual prompts, token vectors, responses, probabilities and
state payloads remain private.

An independent Astra review confirmed the priority of runtime telemetry,
residency and a same-source host control. It also identified the historical
fresh-server and prompt-resident differences. The private audit reports and
experiment sources are retained with their hashes under ignored storage.

## Fixed-row residency diagnostic

A separate source-bound control loaded one serial model and compared bill-bound
and maximum recommended residency capacity, retaining the same in-place combine
route. It used one excluded warmup per arm, then ABBA at 131 prompt / 159 decode
rows. All six arms matched both full F32 readout fingerprints and the earlier
v7b fingerprints. Destruction restored the original capacity.

The runtime KDA observer confirmed per-core recurrence on 40 GPU cores with
80 groups, 104 rows and two groups/core. That observation applies to this new
diagnostic; the earlier baseline did not retain route telemetry.

| Observation | Value |
| --- | ---: |
| Admitted baseline | 41,107,914,752 bytes |
| Pre-admission non-file floor | 26,700,070,912 bytes |
| Pre-admission physical-used reading | 41,576,857,600 bytes |
| Bill-bound / maximum capacity | 76,990,787,684 / 115,448,725,504 bytes |
| Median prefill, bill-bound / maximum | 803.171 / 798.408 tok/s |
| Median decode, bill-bound / maximum | 16.474 / 16.579 tok/s |

This attempt is **unqualified for throughput**: maximum external CPU reached
208.314 percent, failing the control's existing CPU screen. It had no compiler
or download candidates, pageouts, swapouts, sampling gaps or sampling failures;
maximum thermal reading was 1. The observed capacity effect was under one
percent, so the attempt does not support capacity as the main performance cause
on this workload. It also does not prove the broader indirect-buffer residency
invariant.

The plugin selects in-place prompt combine for an unset/`max` wiring environment
and fused combine otherwise. In-place accesses indirect source buffers through
the residency set. The bridge's bill-bound policy therefore needs a separately
established residency invariant or a matching route/policy change even if a
particular performance comparison is flat. A capacity change cannot be
presented as a demonstrated speedup here.

Allocator endpoints were about 75.2 GB active, with prompt cache purged at the
eager-tail transition and about 194 MB decode cache. The peak getter counts
active memory only, so it cannot establish peak active-plus-cache residency.
Source audit found shared phase cache caps of 2 GiB prompt / 256 MiB decode;
the host's omitted 8 GiB startup cap is superseded during those phases.

Private result SHA-256:
`536e20b2acdfcd2015320230f6c29307843459cc14254a27eeb6347446f6c1d4`.
Native source SHA-256:
`07e84c22be2f2e191d33c9ace597a75c478c2b1424de000a69a8efcc571f3381`.
Controller SHA-256:
`06648a4d273eeb9e2c816c2b980a6a614797eb17cef21f065ef24b35576b8b88`.

## Native-policy fidelity and prefix candidate

An instrumented normal host and a separate C control used identical eager-tail,
keep-resident, BF16-head depth-three/gate-0.5 inference at 131/158 rows. Every
normal-host request produced the same aggregate fingerprint and counts as the
C control: 128 committed choices, 45 rounds, 83 accepted drafts and FNV64 U32LE
`882d4378e2910a4d`. The oracle retains token choices only in RAM, suppresses raw
transport text and releases its fingerprint only after successful completion.
The normal host still performs token decoding and filtering. Its rates are
instrumented diagnostics, with production performance qualification false.

The measured host observations were 806.954 / 806.643 prefill and
23.523 / 23.668 decode tokens/s. External CPU reached 272.544 percent and failed
the unchanged screen. This establishes tested native-policy fidelity; it does
not qualify these rates, equivalence to the serial rounding policy or model
quality. The corresponding C diagnostic observed a roughly 1.42-fold MTP decode
ratio, also rejected for external CPU.

The new private bridge connects that native sampler and committed-round
interface without a second sampler. It accounts for the already-published
initial choice in the committed budget, suppresses its committed duplicate and
never publishes the next uncommitted pending choice. It drains request work
before success. The original headless loader and F32 readout remain reference
paths. Native capability bits stay disabled in the ordinary build while the
remaining quality and performance gates are unresolved.

The native candidate also captures the MTP head at the nonfinal, chunk-aligned
16K prefill boundary. On the actual model, restored continuation B and fresh B
matched all 154,880 F32 readout values bit for bit, both before/after logical
state hashes, all eight committed choices and round/acceptance counts. The
parked-head legacy serial path matched the original prefill and final F32
fingerprints. Cancellation, mode change, invalid input and model-owner changes
rejected stale snapshots. The owner replacement check passed with a predeclared
two-second wait after destruction; immediate replacement previously failed
admission, and its cause remains unproven. This result does not qualify immediate
reload, stochastic requests, arbitrary contexts or worker-level lifecycle.

The frozen 30-file source fixture is reconstructed in RAM from the original
`d0c54deb` source archive. Build-file edits therefore cannot silently change the
benchmark input. Its 16,384-token U32LE SHA-256 remains
`06da8fa52d1c92a827ae8022d756fbf26e3de95e706803769b7952db898b57c9`.

Private normal-host oracle result SHA-256:
`bf497382e70747daf69b73a50b9a1af9bb4b64b9b3d347022416933a3c677cd9`.
Private native-state result SHA-256:
`feec93cc6a7144f58d7646832e9b090686aac7aaec9da909e54163eb7263da64`.
Native source bundle SHA-256:
`43379ce0d2acc566dade421a368111edf22950c446297115eeadc8ce6ab10765`.

## Prefill boundaries and remaining checks

Three-request serial controls measured the first request after a fresh model
load and two repeated reset requests. All used the same raw 16K input, greedy128,
131/159 rows, MTP off and unchanged memory/paging screens. They do not establish
an MTP-on tail policy.

| Tail / residents | Prefill tok/s, first / repeats | End-to-end seconds, first / repeats | Status |
| --- | --- | --- | --- |
| Off / keep | 929.626 / 900.654 / 858.457 | 25.807 / 26.021 / 26.886 | All request screens passed |
| Deferred / forget | 917.877 / 914.760 / 894.324 | 26.197 / 25.902 / 26.268 | All request screens passed |
| Deferred / keep | 935.541 / 917.605 / 845.393 | 25.855 / 25.834 / 27.576 | All request screens passed; cleanup proof failed, so the complete attempt remains unqualified |
| Off / forget | 926.638 / 895.049 / unavailable | Not qualified | Second request failed external CPU screening |

With MTP enabled, deferred tail can move head input projection and DSA
latent/index catch-up into the first decode round. It does not defer the full
expert trunk or output head. The configured append chunks and input order remain
the same, but runtime fingerprints must confirm identity. Prefill, TTFT, decode,
their sum and complete request latency must be compared together. A 128-output
decode result is more sensitive to fixed catch-up cost than the historical
1,024-output cell.

Two potential memory explanations have negative evidence. All 24 verified pack
files are directly under the pack root, so nested weight banks cannot explain a
missed pack-cache census here. A read-only residency census of the two owned
Rust build targets found about 1.21 GB resident across 6,396 unique files, much
less than the roughly 8.8 GB needed even for 22 more expert rows. File residency
alone also does not prove immediate reclaimability. No eviction, purge, budget
increase or security/management-process changes were made.

The later sections record MTP-on tail/total latency, 1,024-output acceptance,
aggregate expert I/O, the worker matrix and bounded quality results. Still open
are a qualified production MTP throughput result, the unmet quality gate,
admission-term reconciliation and matched hardware clocks. The historical
target has not been reproduced.

## Longer sampled diagnostic

A separately prepared original C control completed the same raw 16K source
fixture with 1,024 fixed outputs, temperature 1, top-p 0.95 and seed 1. Both arms
used one loaded BF16-head model at 122/149 rows, eager tail and maximum
recommended residency capacity. Each arm had its own excluded warmup, followed
by ABBA. Cross-arm numerical differences were explicitly allowed and recorded;
within-arm warmup/repeat fingerprints were mandatory.

| Arm | Decode observations, tok/s | Median | Committed tokens / rounds / verified / accepted |
| --- | --- | ---: | --- |
| Loaded head, no drafts | 15.304 / 14.115 | 14.710 | 1,024 / 1,024 / 0 / 0 |
| Depth-three MTP | 19.089 / 18.209 | 18.649 | 1,024 / 367 / 889 / 657 |

Both MTP observations matched fingerprint `89c3e863f47bad76`; both no-draft
observations matched `73d0361c363cc56f`. This is a same-policy repeatability
reference for later adapter checks. It does not establish serial/MTP equivalence
or quality. MTP committed about 2.79 choices per round, close to the 2.84 observed
in the greedy128 control. Output length alone did not recover the throughput
target on this footprint.

The rate screen failed: external CPU peaked at 260.821 percent. A broad file
search made by this investigation briefly used about two CPU cores inside the
window and contributed to that failure. The window had zero new pageouts or
swapouts, no compiler/download candidates, monitor failures or sampling gaps,
and maximum thermal reading 1. All rate and quality qualifications remain false.
Later timed windows hold local investigation work as well as builds. The
separate post-run preparation verified the same artifacts, dependencies, pack
metadata and frozen input; it cannot retroactively qualify rejected rates.

Private diagnostic result SHA-256:
`adc7bc4b2ab8038930b44a28a935f5b7e561808befe8f017e026594496061277`.
Separate post-identity preparation SHA-256:
`fa0b259a7ef95152cc908a3b40674b4fe62ead13fcaf3af1f6ef9f94c48b1036`.

The MTP-on tail control's first 131/158-row attempt failed its startup admission
screen: its recorded 39.619 GB baseline and 81.910 GB model bound exceeded the
118.112 GB target by 3.418 GB. It performed no inference and restored the original
private settings. A separately frozen 119/146-row pair was prepared; the later
properly templated 104/130-row control below was executed instead. No
over-budget pin was forced.

## Trunk expert I/O diagnostic

A separate private stage changes only method visibility and adds a read-only
getter for existing aggregate expert counters. It adds no fence, kernel change
or per-expert instrumentation. One raw 16K/card1024 request used 118/144 rows,
the same native MTP policy and no warmup. It matched the 1,024-output fingerprint
and every round/acceptance count. Complete cleanup and source/model/artifact
postguards passed. This is an untimed diagnostic, with performance qualification
false.

| Phase | Cache hits / misses | Settled payload | Closed read-busy intervals | Host read wait |
| --- | ---: | ---: | ---: | ---: |
| Prompt boundary | 2,799 / 9,269 | 87.815 GB | 6.769 s | 0.081 ms |
| Decode and request-end drain | 281,303 / 53,447 | 506.359 GB | 34.120 s | 5.807 ms |
| Whole request | 284,102 / 62,716 | 594.174 GB | 40.889 s | 5.888 ms |

Hits/misses count distinct experts per routing operation. Decode's miss fraction
was about 16.0%, and settled payload averaged about 494.5 MB per output choice.
Bytes include settled read-ahead/adopted payload; they are not a physical SSD
traffic measurement. These counters cover the trunk expert source, excluding
head and weight-load I/O. Pool busy intervals are recorded when they close, so
an interval crossing a snapshot boundary is charged to the later phase.

This makes cache residency an actionable hypothesis. It does not establish
34 seconds of removable latency: busy reads overlap computation, and GPU event
waits need not enter the small host `waitReads` counter. A separately prepared
same-source cache-row comparison held prefill rows at 104 and compared
104 versus 130 decode rows, retaining identical greedy128 output hashes and
counts; its results follow below. Claims about 217 rows or 33 tok/s still
require an actual admitted run.

Private result SHA-256:
`0dea588dbb24301573dd2aef20d86d0b22cfd7de213c09fcdf0412d33643f320`.
I/O stage receipt SHA-256:
`fc7021da55e13488c2230db71559ffb32174fa50a390aa46581b5f0e0f9d4746`.

## Worker and quality status

The private worker now passes a 41-case source-bound matrix, including the
serial/greedy-MTP/sampled-MTP native references, prefix reuse in both modes,
budget/context edges, explicit stops, cancellation and recovery. The
[native mechanism record](../archive/runs/2026-11-04-glm-exl3-native-mtp-prefix.md)
binds its source and executable. This does not qualify production performance.

The initial objective quality harness had a template mismatch. It supplied
`enable_thinking=False`, but the verified GLM template ignores that variable,
defaults `reasoning_effort` to maximum and opens the thinking channel in its
generation prompt. The harness graded the whole raw decoded response as JSON.
Twelve completed requests reached EOS within the original 1,024-token limit
and failed that framing rule; the run was then interrupted with complete
cleanup and matching postguards. That result cannot establish a model defect
or be retroactively regraded, because response content was discarded.

A separately versioned correction preserved all six original task, expected
answer, rendered input and token-input hashes, all 24 arms, maximum reasoning
and the 1,024-output limit. It graded the template-defined final channel
with the unchanged strict JSON grader, failed closed on malformed framing or
truncation, and retained every failed arm. No question replacement or selected
retry was used.

That corrected run completed all 24 requests: 22 passed. Both modes reached
the 1,024-token limit on the same greedy interval-merging case and failed the
framing/truncation gate. Every other paired task passed in both modes. This
does not establish an MTP-specific task regression, and it does not satisfy the
predeclared all-pass gate. The budget was not increased and no failed question
was replaced or selectively retried. Semantic quality qualification remains
false; ordinary native MTP capability bits remain disabled. Cleanup and all
identity guards passed.

Interrupted quality result SHA-256:
`d4f693c0745d16bed7e436b72d2ed82b1ba16bc8a0cf73a777e591e15eb76b29`.
Corrected quality result SHA-256:
`b336fd1e1b97dcda1d9527a1cbf93ce3aac93e4c127245da93fdfcebe4942cd6`.

## Cache capacity sensitivity

A new C client retained the I/O stage, native inference and input, and pinned
104 prefill rows with either 104 or 130 decode rows. Each fresh child ran one
excluded identical warmup and one measured greedy128 request. The frozen order
was 130/104/104/130. Every request matched fingerprint `882d4378e2910a4d`,
45 rounds, 99 verified drafts and 83 accepted drafts. Load, warmup, source
hashing and a guarded quiet wait were excluded from measurement. All four
children passed numerical, counter, cleanup, source and model identity checks.

| Order / decode rows | Prefill tok/s | Decode tok/s | Decode payload / misses | Host screen |
| --- | ---: | ---: | --- | --- |
| 1 / 130 | 884.012 | 19.286 | 61.401 GB / 6,481 | Passed |
| 2 / 104 | 848.526 | 15.726 | 83.154 GB / 8,777 | Passed |
| 3 / 104 | 712.981 | 15.604 | 83.154 GB / 8,777 | Passed |
| 4 / 130 | 693.786 | 18.823 | 61.401 GB / 6,481 | CPU-only rejection |

The additional 26 decode rows reduced settled decode payload by 26.2%, exactly
repeating in both arms. Closed read-busy intervals were about 4.1 seconds at
130 rows and 5.6 seconds at 104. The timing observations support investigating
residency, but the complete ABBA experiment remains **unqualified for
performance**. The fourth arm reached 370.887% foreign CPU during macOS
background maintenance. This GLM screen retains its original 150% measured
external-CPU limit; no threshold was relaxed. All arms had zero new pageouts or
swapouts, no compiler/download candidates, no monitor failures or sampling gaps,
and thermal readings at most 1. Hardware clocks were not measured.

The second controller version continues only after CPU-only rejection when
every other guard and child cleanup/postcheck succeeds. This permits complete
numerical/counter diagnostics while permanently vetoing performance
qualification. The first version stopped after its first arm's CPU rejection;
that source and failed receipt remain intact. These observations do not justify
extrapolating 130 rows to 217 rows or claiming 33 tok/s.

Private complete diagnostic result SHA-256:
`7c8efb3299ddf19df8e274bd38d3ff9cf62837d0285662038811a2cabe830a77`.
Controller SHA-256:
`7cde17c5deec693bfed57447a312cd2086847640e6226c677be75a5c6abbe9cd`.

## First-request sampled and templated host controls

Three cells were frozen before execution: raw/card1024/eager, one natural
engineering-chat/card1024/eager, and the identical chat/card1024/deferred.
Each used a fresh normal server, no warmup or prefix reuse, fixed 104/130 rows,
`forget` residents, maximum process residency capacity, a BF16 depth-three
head, gate 0.5, exact acceptance, temperature 1, top-p 0.95 and seed 1. The
native executable and inference source were unchanged from the earlier
normal-host fingerprint oracle. These are instrumented host observations;
they are not Stark-worker production benchmarks.

The raw fidelity gate matched the native C and worker reference exactly:
1,024 choices, fingerprint `89c3e863f47bad76`, 367 rounds and 657 accepted
drafts. Its measured request passed the existing host screen, with external
CPU at most 149.584%, thermal reading zero, no paging, compiler/download
candidates, sampling failures or gaps. It observed 883.957 prefill and 16.618
decode tokens/s. Different row counts, startup and resident policies prevent
calling this a rate reproduction of the earlier C control. It does show that
the actual server's sampled path reproduces the native policy without
recovering the historical decode rate on this footprint.

The chat workload was constructed once from a fixed natural engineering
question and one prefix of the frozen public-source corpus. The real pack
template used maximum reasoning and its assistant thinking opener. It rendered
to exactly 16,381 tokens, input U32LE SHA-256
`55e027d0663b27edfa24e5b2d91aaa1fdbf0a13d9326e3478fd649451ff631f4`.
Construction was frozen without observing output or rate; no prompt cycling,
padding or search for favorable output was used. Full text and tokens remained
in RAM. The legacy completion endpoint consumed the rendered tokens, so this
is a templated-token completion, not a chat-endpoint benchmark. EOS remained
enabled; both requests naturally reached the 1,024-output limit.

| Cell | Prefill tok/s | Decode tok/s | Prefill / decode / total seconds | Host screen |
| --- | ---: | ---: | --- | --- |
| Raw eager | 883.957 | 16.618 | 18.535 / 61.620 / 80.155 | Passed |
| Chat eager | 885.191 | 17.363 | 18.506 / 58.976 / 77.482 | CPU-only rejection |
| Chat deferred | 915.165 | 17.317 | 17.900 / 59.134 / 77.033 | CPU-only rejection |

The chat cells matched each other exactly: fingerprint `89f856128b48cd6e`,
1,024 choices, 478 rounds and 546 accepted drafts. Their external CPU peaks
were 173.650% and 178.969%, primarily the Codex renderer. All other guards,
identity postchecks, process-group cleanup and byte-identical private-settings
restoration passed. The comparison therefore admits output agreement and
rejects timing comparability. The observed 0.45-second total difference is not
a qualified gain. No raw-to-chat speedup ratio is meaningful because their
inputs, routes and outputs differ. Neither chat formatting nor tail deferral
has supplied evidence of the missing decode recovery.

Private chat eager / deferred result SHA-256:
`5520a296a7a8827768d07c7adad825ce628a6a1d0077f1bf3fe576480814993e` /
`d70ec07e3d4e53bfdb803ea608ac63a4173239f26af883ad234efb4851f1ae50`.
Controller SHA-256:
`7ff1711a0d2689bb6589a44de8854d1dea2b131e221900be2a49a7e2304da69d`.
Private raw fidelity result SHA-256:
`54b7b666fbc45c107763ee2e6471d3e0366a9fd5c587bdaa67ccc70199e975e6`.

The remaining performance lead is expert residency. Its payload effect has
been measured directly; the latest 46–47 GB admission baseline permits far
fewer rows than the historical cell. A bounded source audit must reconcile
that baseline and establish a safely admissible frontier before a higher-row
control. No additional cache eviction, global wiring change, application quit,
budget increase or forced historical pin is implied by these results.

## Admission source reconciliation

The source audit found no current-model double count. Both host and bridge
sample admission before loading resident weights and allocating expert slots.
The plugin uses `max(host_floor, physical_used - pack_cached)`, then adds the
model bill once. Resident and expert readers already use checked `F_NOCACHE`,
aligned reads and disabled OS read-ahead. Pack-owned file-cache invalidation is
already enabled. Another no-cache switch or MLX free-cache clear is therefore
not an evidenced capacity recovery.

| Latest raw host admission term | Bytes |
| --- | ---: |
| Baseline | 46,766,309,376 |
| Process bound at 104/130 rows | 70,955,800,308 |
| Combined bound | 117,722,109,684 |
| Target after margin | 118,111,600,640 |
| Remaining headroom | 389,490,956 |

The [upstream rows ladder](https://github.com/davidtai/mlx-stream/blob/ac3b772b61bce66450862bca697dbb742c35453f/docs/benchmarks/glm53-flash/README.md#rows-ladders)
explicitly defines 12.69 GB as the maximum admitted baseline for 190/217, not
the baseline measured in that throughput cell. The current baseline exceeds
that bar by about 34.08 GB. The separately sampled pre-load non-file floor was
29.62 GB; it is a different metric and sample, not proof that the remaining
17.15 GB is reclaimable. Even hypothetically crediting that entire difference
would not establish 217-row capacity. No pressure allocation or broader
foreign-cache credit is proposed.

The existing startup log emits exact admission components, but the frozen
controller retained only the final baseline. A future bounded parser may
whitelist that four-integer line without a native rebuild or another mutating
census. This would attribute the final branch; it would not itself free memory
or justify a larger pin. At the present footprint, useful engineering work
must reduce misses or exposed read latency within the same admitted budget.

## Direct versus staged expert reads

One separately prepared private pair tested the supported `miss_reads`
policies. Direct reads bypass staging copies and disable preread. Staged reads
permit earlier preread and pay the copy. Both policies allocate and bill the
same worker/speculative staging, metadata and slot geometry. This selector
affects only the trunk stream; the draft head preloads its complete bank and
retains its original math and reads. The existing calibrated 42-layer row
offsets are already enabled, so a uniform-to-per-layer switch is not missing.

Before any new inference, both arms were fixed at 96 prompt / 122 decode rows
to retain headroom after CPU preparation. Each fresh child ran one identical
excluded card1024 warmup, reset, passed the quiet gate and ran one measured
request. All four requests matched `89c3e863f47bad76`, 1,024 choices, 367 rounds,
889 verified drafts and 657 accepted drafts. Installed rows matched; every
model/source/artifact postguard, cleanup and live memory/paging check passed.

The private getter returns 36 existing cumulative trunk counters. One call
at each phase boundary supplies both this snapshot and the original seven-field
projection, checked for exact agreement. It adds no graph evaluation or fence.
The native stage, C client and controller received independent source and
receipt review; all 17 controller tests and model-free preparation passed.

| Observation | Direct | Staged |
| --- | ---: | ---: |
| Prefill tok/s | 851.289 | 653.146 |
| Decode tok/s | 16.486 | 15.564 |
| Complete request seconds | 81.361 | 90.877 |
| Decode cache misses / hits | 63,762 / 270,988 | 63,762 / 270,988 |
| Settled decode payload | 604.084 GB | 604.084 GB |
| Adopted bytes | 144.972 GB | 144.972 GB |
| Speculative issued | 28,108 | 28,108 |
| Source-reported speculative read bytes | 227.913 GB | 225.250 GB |
| Abandoned speculative bytes | 78.883 GB | 75.394 GB |
| Prereads issued / served | 0 / 0 | 96,910 / 44,791 |
| Closed demand/preread busy intervals | 40.203 s | 43.612 s |

Preread engagement is established by its counters and the direct-range change
from 96,920 to zero. Demand payload, misses and adopted bytes remained identical.
The new counters expose speculation separately: settled payload includes adopted
data, so matching payload per miss does not establish zero speculative waste.
Payload and speculative bytes must not simply be added; adoption overlaps them.
These are source-level read statistics, not physical SSD-device telemetry.
Closed busy intervals overlap computation and do not establish removable time.
Their gauge excludes speculative workers and adoption waits/copies; it is not
a measure of all expert I/O or GPU event-wait time. The small host `read_wait_ns`
counter does not establish small GPU-visible waits.

Both measured windows failed the unchanged external-CPU screen, at 171.595%
and 173.540%, largely the Codex renderer. All other screens passed, with no new
pageouts or swapouts, no compiler/download candidates, no monitor failures or
gaps and thermal readings at most 1. The complete pair is numerical/counter
evidence, with timing comparability and production performance false. Its
observations provide no evidence supporting staged-policy promotion. No
production read policy changed.

Source review defines speculative bytes as bytes returned by speculative
`preadv`, and abandoned bytes as partial reads for abandoned records. For the
direct arm, speculative bytes minus adopted bytes is about 82.94 GB. A
read-accounting proxy, `settled - adopted + speculative`, is about 687.03 GB;
the excess is about 12.1% of that proxy. This is not SSD telemetry or a latency
bound. About 95.4% of claimed records were still in flight when claimed, which
can include useful partial lead. Neither percentage establishes that removing
speculation would improve throughput. The evidence does not identify a
same-budget change capable of doubling decode speed.

Private result SHA-256:
`9fd046e5d8b7d6216e4d9e8d8374f964a5628d6feb3825eb2b78f76352d36a98`.
Controller SHA-256:
`bbb741caa8f330ea3765a4eed5155d2573ab5a5cd00bcc33077acd166ed652ca`.
Tooling receipt SHA-256:
`2615153e0c7194dfb6c4418bf690ebc5f3413a67f35e96906739475f00df529a`.
Native source/stage provenance SHA-256:
`704e23f73aec835e7fcf5bd0f35cdaa4f33cbe1216676bbbea8630f651bd9789`.

## Read-only hardware telemetry follow-up

The earlier denied `powermetrics` query did not establish that all frequency
telemetry requires administrator access. A bounded idle probe compiled the
pinned upstream `mimo_gpu_power.inc` unchanged into a standalone helper. Its
no-root IOReport subscription, two samples and delta succeeded, with finite
nonnegative GPU-state residency and available energy/thermal-limit fields.
It reported 16 state entries and a 14-entry static frequency table. No model,
Metal context or tensor workload was created, and no fan, power or kernel
setting changed.

This establishes an available measurement path. It does not establish held
clocks or clock matching in any earlier timed run. State labels and frequency
table entries must be mapped correctly before deriving operating frequency;
their differing counts must not be indexed naively. The sealed direct/staged
pair was not modified to add this collector and retains its unmeasured-clock
disposition. The unchanged upstream helper source SHA-256 is
`5bec7460db07be0b07e292b2d91e48d7333c3b1e11386a4003a413976748f8d9`;
the standalone wrapper source SHA-256 is
`83627b2564785e766c76ef4cde592301430d8cf6531ac6a7019ecd3f1def8472`.

## Lower host baseline and exact cache admission

The operator closed nonessential apps and identified Codex as the only remaining
open app. A new standalone admission helper used the frozen bridge's exact
configuration, normal pack-only page-cache census/invalidation and metadata
bill. It read bounded manifests and safetensors headers, never tensor payloads.
Its executable links only libSystem and creates no MLX device, tensor or GPU
context. This is a GPU-memory admission calculation, not CPU inference.

The original 104/130 process bound matched exactly at 70,955,800,308 bytes.
All 24 pack-file stat guards and source/artifact postguards matched.

| Admission observation | Bytes |
| --- | ---: |
| Physical used | 35,701,030,912 |
| Pack-owned file-cache credit | 246,038,528 |
| Separately sampled non-file floor | 20,730,281,984 |
| Admitted baseline | 35,454,992,384 |
| Ceiling | 120,259,084,288 |
| Existing margin plus snapshot reserve | 2,453,094,400 |
| Additional selection headroom | 2,147,483,648 |
| Protected target | 115,658,506,240 |
| P104/D153 decode-phase total | 115,373,021,432 |
| P104/D154 decode-phase total | 115,771,514,504 |

The helper therefore selected D153 with P104. The normal joint fill was
132/159; the joint fill with the additional reserve was 127/153. These are
snapshot-specific admission results, not promises of future availability or an
optimal performance configuration. No over-budget pin, global cache purge,
memory-pressure allocation or budget relaxation occurred. The difference
between physical and non-file readings is not established reclaimable memory.

Helper source SHA-256:
`e9852a2dd54c6ad5c66c3498adc4c3e4ea8c3e3fd93cca270a55d966b1670a5f`.
CPU build receipt SHA-256:
`3c366216b14f5b9759a8cce94a818570b9f2db2c04401127f966257d36ab5817`.
Actual census result SHA-256:
`fbc28acb972dd69fed4d781e71c3270e7779f4363701213e3fa040749df86551`.

## Cache frontier with workload clock capture

Before inference, a new controller fixed P104 and D130/153/153/130 in ABBA
order. Each fresh child ran an identical excluded greedy128 warmup, reset,
quiet gate and one measured greedy128 request. Every warmup and measurement
matched `882d4378e2910a4d`, 45 rounds, 99 verified drafts and 83 accepted drafts.
All model/source/artifact guards, actual row checks, process-group cleanup and
live memory/paging checks passed. The native engine and inference source were
unchanged. This isolates a decode-row intervention; it does not establish a
prefill improvement.

The C caller now captures the upstream read-only IOReport counters through
symbols already present in the staged library. Three samples surround prefill
and decoding; the middle sample follows native initial-token selection. The
calls add no MLX evaluation or fence, but their boundaries have small documented
offsets from the native timers. It retains labeled state-residency deltas and
the frequency table. P14/P15 have no frequency mapping on this host, so any
positive residency there vetoes clock qualification. All measured intervals
had positive mapped coverage and zero unmapped active residency.

Timing comparability required all four original host screens and a relative
difference of at most 0.5% between the two arms' mean active decode clocks.
Clock correction was not used. The supplied row plan, source bindings and
clock policy were frozen before model execution; no thresholds were relaxed.

| Automatic mode, arm order | Prefill tok/s | Decode tok/s | Active decode MHz | Host screen |
| --- | ---: | ---: | ---: | --- |
| D130 | 890.529 | 19.179 | 1,569.959 | CPU rejection |
| D153 | 852.716 | 21.221 | 1,565.998 | Passed |
| D153 | 621.775 | 18.880 | 1,013.616 | Passed |
| D130 | 579.388 | 17.568 | 990.313 | Passed |

The first arm had 204.581% foreign CPU, including `codex` and `xcodebuild`.
The original candidate filter did not identify `xcodebuild` as a compiler
candidate; the CPU veto already rejected the arm. Its original classification
is preserved. No rate from that arm becomes qualified by ignoring the build.
The clock gate also failed: the arm means differed by 0.753%, with a large
frequency decline between the first and second halves.

The slow phases had sharply increased CLTM aggregates alongside the lower
mapped clock. The counter sums non-`NO_CLTM` residency across matching
subgroup channels without identifying their count. Its aggregate nearly
equaling a phase duration must not be described as an exact fraction of unique
wall time throttled. The observations support thermal/power limiting as a
practical explanation; they do not distinguish temperature from power policy.

## Operator-enabled High Power repeat

The operator enabled High Power on AC. Read-only `pmset` observations confirmed
AC `powermode 2` before and after the repeat, with battery mode unchanged at 0.
The exact same source, binary, row plan, request policy, ABBA order and guards
were used. No fan utility was installed or used; maximum fans and RPM were not
verified. The repeat is a separate result, not a requalification of the first.

| High Power, arm order | Prefill tok/s | Decode tok/s | Active decode MHz | Host screen |
| --- | ---: | ---: | ---: | --- |
| D130 | 905.319 | 19.344 | 1,584.085 | Passed |
| D153 | 886.825 | 21.384 | 1,566.108 | Passed |
| D153 | 785.706 | 21.162 | 1,440.971 | Passed |
| D130 | 787.232 | 19.318 | 1,491.502 | Passed |

All four host screens passed. The clock gate still failed: arm means were
1,537.794 MHz at D130 and 1,503.539 MHz at D153, a 2.253% difference. Observed
decode was about 10% higher with D153 despite its lower mean clock, but the
predeclared paired timing criterion remains unmet. Production performance
qualification remains false. No numerical or artifact failure occurred.

Across both experiments, D130 had 6,481 decode misses and 61,401,305,088 bytes
of settled payload; D153 had 5,385 misses and 51,017,748,480 bytes. These values
repeated exactly in every corresponding arm, a 16.91% reduction. They establish
the cache intervention's payload effect, not a physical SSD or removable-time
estimate. The 1,000 prefill / 33 decode objective remains unmet.

Automatic-mode result SHA-256:
`36dc7c560ed5702e2805dbdf7a66520c58d77400f82ee429882151bc92992261`.
High Power result SHA-256:
`78d927d9037dd11f81e596ac1e77aa131913008262e3e300b89d509d09672dec`.
Controller source SHA-256:
`452c93dfcbd9e173ee5dc7f8b1f1dcf80254f1e7de79321d2e5fb63d6bc2e183`.
Tooling receipt SHA-256:
`bd4c27ff309613797dbd003f031128e6586ad8aef69c33659006689cd422e02a`.
The C client links the unchanged native stage `44cc7794…`; its build receipt
SHA-256 is
`ab4451716c86df816d76bd826438e3345a8e71a08f6de49f89fe89fbd46fa20a`.

## Private GPU p/q sampler speed check

The private GPU p/q sampler passed its tiny three-case, 768-draw fixture after
fixing an unsupported scalar-zero construction. A same-stage exploratory pair
then used the same weights and raw 16,384-token input, temperature 1 / top-p
0.95 / seed 1, 1,024 outputs, one excluded warmup and one measurement per
fresh child, P16/D40 rows, and an equal 128 MiB workspace reserve. Results are
sampler-specific: the trajectories differ, so output fingerprints and draft
counters are not expected to match across arms.

| Sampler / gate | Prefill tok/s | Decode tok/s | Measured rounds / verified / accepted | Host screen |
| --- | ---: | ---: | ---: | --- |
| Legacy / 0.5 | 821.44 | 7.2977 | 367 / 889 / 657 | Passed |
| GPU p/q / 0.5 | 846.13 | 7.55494 | 380 / 848 / 644 | Failed: external CPU 240.8% |
| GPU p/q / 0 | 562.95 | 6.35507 | 357 / 1,070 / 667 | Passed |

Both p/q cells committed 1,024 outputs. The gate-0.5 cell's faster observed
times are not comparable under its failed host screen; the screened gate-0
cell had a lower observed throughput. Clocks were not matched, and these single observations show no
useful speed gain. No paging or swapout increase occurred. The private sampler
remains disabled and unqualified. The 1,000 prefill / 33 decode objective
remains unmet.

The preflight census that selected this P16/D40 footprint measured an admission
baseline of 78.26 GB, about 49 GB above its separately sampled non-file floor
of 29.01 GB. This difference was not established as safely reclaimable capacity.
Admission retained the existing pack-specific
cache invalidation, ceiling, margin and reserve. No global cache purge or
admission-budget relaxation was used. These exploratory timings do not qualify
the 104/130 or historical 190/217 profiles.

Private pair result SHA-256:
`6e5f7096ed3302f958c454e9dcb7de7638797f56e6522489706f3c0116f6b891`.
Gate-0 p/q result SHA-256:
`fcbd482a0857caa3a5f988b235470d91abe6892d96be87a2e632a1141068db4e`.

## Post-purge first-request capacity check

After the authorized OS purge, the exact CPU admission census measured a
34.106 GB baseline, down from 79.560 GB before the purge. With the existing
112 GiB ceiling, margin and snapshot reserve, the census's protected joint fill
was P130/D157. It excluded the private stage's paired 128 MiB workspace reserve;
selecting P129/D156 covered that reserve and kept the extra 2 GiB selection
headroom. This confirms a large cache-sensitive change in the
admission snapshot; it does not establish that the same baseline will recur.

A fresh legacy child then ran the frozen raw 16,384-token, temperature 1 /
top-p 0.95 / seed 1 workload for 1,024 outputs, with no warmup. It matched the
strict fingerprint `89c3e863f47bad76`, 367 rounds, 889 verified drafts and 657
accepted drafts. At P129/D156 it measured 922.6615 prefill and 20.3003 decode
tok/s (17.757324 s prefill, 50.442620 s decode; 68.199944 s total). The host
screen failed at 193.606% external CPU; paging, sampling, compiler and thermal
checks passed. Clocks were not verified, so this is diagnostic evidence, not
qualified throughput.

Settled decode payload was 430.406 GB, compared with 1,689.924 GB in the earlier
P16/D40 legacy run. Both had the same output fingerprint, but one was a fresh
first request and the other followed an excluded warmup. This payload change
does not make their timing ratio comparable.

The source comparison found the historical `88d2aad` and current `7d29eaa`
plugin use the same BF16 bill and baseline formula; the SDK memory-bill source
is byte-identical. The historical loader's bill selection honors explicit pins
and can warn on positive overfill, while the current private bridge rejects
over-budget pins. The upstream benchmark README documents a pre-arm admission
gate and reports 225-row cells that exceeded the bill fill by 1.25–1.79 GB
after baseline drift. That does not show that the historical 190/217 cell
overfilled. The historical host's actual load baseline was not reported, and
12.69 GB is its published modeled maximum baseline for 190/217, not a measured
baseline from the 33.15 tok/s run. The current bridge additionally reserves
305.6 MB for prefix snapshots and, in the private p/q stage, 128 MiB for paired
sampler workspace.

Post-purge first-request result SHA-256:
`01c6a99aea0a29bdd6c377597896b8fbc4e8fe2362c09541171052bb5bad5a2a`.

## Operator-pinned 217-row attempts

With approval to bypass the modeled-fit rejection for fixed operator pins,
the runtime non-file-memory ceiling of 112 GiB and paging guards remained
active. P190/D217 aborted before model load completed: non-file
baseline was 31.7228 GB and the observed maximum reached 124,475,326,464 bytes
(115.927 GiB), above the configured ceiling. The owned process group was
reaped; pageouts rose by 15 and swapouts did not change.

A first P129/D217 attempt loaded the header but timed out at quiet admission,
without a measured request. Review found that the private screen had counted an
installed GUI Stark process as model CPU. The classifier was corrected only
for the exact GUI process name `Stark`; generation-worker CPU, global CPU,
GPU and memory/paging guards
remain. On retry, P129/D217 passed the 10.12-second quiet gate (external CPU
75.57%, model CPU and GPU zero), then the live memory guard stopped execution:
the non-file baseline was 31.9499 GB and its maximum observed floor reached
127,330,680,832 bytes (118.586 GiB), above the retained ceiling. Pageouts rose
by 8, swapouts were unchanged, and the process group was reaped.

No 217-row attempt reached a measured end or produced an output
fingerprint, timing or correctness result. The retry only reached measurement
start before the guard aborted. These observations constrain this host under
the retained 112 GiB policy; they do not establish that 217 rows are
physically impossible on a 128 GiB machine. The runtime limits were not raised.
A subsequent, separate `vm_stat` check after cleanup measured approximately
31.7 GB of non-file memory; the test process groups were absent.

P190/D217 attempt SHA-256:
`06318e9601d9139b357128da5142bf66eb68e29390411d1915eec89e90f9f230`.
P129/D217 retry SHA-256:
`89dce7ffb5e4318496982c6986e634546318a5ace396d38ad9bb43d4dd824b76`.

## Private Q8 draft-head comparison

At the strict 39.060 GB admission baseline, the protected BF16 fill was
P118/D144. Fixed P115/D141 rows for both arms also covered the Q8 copy and
wiring, paired 128 MiB reserve, and retained 2 GiB selection headroom. The
Q8 used the existing 256-candidate draft filter; BF16 used full-vocabulary
draft sampling. Both used full target-row legacy draw matching, gate 0.5,
raw 16K input, and T1/top-p 0.95/seed 1.

Both heads matched the excluded 128-output warmup's full-target F32 readout
hash (`3379492f9916220b`, 154,880 float32 values) and sampled fingerprint (`499c3d4762b75ec4`,
44 rounds, 109 verified, 84 accepted). On the measured 1,024-output request,
BF16 produced `89c3e863f47bad76` (367 / 889 / 657) at 891.4517 prefill and
18.279755 decode tok/s; its host screen failed at 174.56% external CPU. Q8
produced `427cfb361e7da505` (461 / 823 / 563) at 898.5284 /
18.570303 tok/s; its host screen passed at 143.42% external CPU. Paging,
compiler and thermal checks passed for both; clocks were unmatched.

The observed 1.59% decode difference does not demonstrate a useful gain: one
arm failed the CPU screen, clocks were unmatched, and the Q8 head changed the
sampled trajectory and gate inputs through quantized logits and bounded
filtering. This is not evidence of a normalization defect. Quality remains
unqualified; Q8 stays private and disabled, and BF16 remains the retained head.

Private pair result SHA-256:
`b78bf50a0943178f752ab0447ca626809b60892728b4f515eb18fa86c91b260f`.

## Private frequency-first comparison

At P115/D141, the matched BF16 frequency-first run used the same full-target
readout and warmup fingerprints, frozen 1,024-output workload, and strict
`89c3e863f47bad76` output fingerprint with 367 / 889 / 657 counts as the
transition-weighted arm. Weighted ranking measured 900.19 prefill /
18.0908 decode tok/s (152.55% external CPU, host screen failed). Frequency-first
measured 878.65 / 16.6914 tok/s (247.07% external CPU, CPU screen failed;
thermal state reached 1). Paging, sampling and compiler checks passed; clocks
were unmatched.
The frequency-first arm also recorded 61,285 misses and 580,617,031,680 bytes,
versus 54,775 misses and 518,940,979,200 bytes for weighted ranking, an 11.89%
increase in this matched-work comparison. Reject frequency-first; this run does
not support a statistical wall-time claim.

Frequency-first result SHA-256:
`6c9613ee8c2a27552a4aeb84c60e667b5c18ea0b9579d0b200fef6a8af064ff2`.

## Transition-only prediction status

A private transition-only scorer was built with the 1/0/0 scoring values;
existing rank ties, cache-pinned experts and held-slot protections remain.
Strict admission is unchanged. It has no runtime
verdict because no candidate request completed. At the 43.516 GB admission
baseline, the P115/D141 candidate was refused before inference. For the retry at P96/D122,
legacy completed while Cargo activity prevented the candidate run; the next
candidate-only launch was blocked before launch by Cargo process 88619. The
scorer is prediction-only and unmeasured. No source or runtime evidence yet
shows whether it improves cache selection. The original weighted ranking is
retained; the private scorer remains disabled.

## Experiment index and retained conclusion

The selected long profile is the strongest completed result for the frozen
sampled workload: raw 16,384-token input; 1,024 outputs; T1/top-p 0.95/seed 1;
BF16 head, depth 3, gate 0.5, original target-draw-match acceptance, weighted
cache retention, P129/D156, 32,768 context, strict admission, and no warmup.
It produced 922.6615 prefill / 20.3003 decode tok/s and the strict reference
fingerprint `89c3e863f47bad76` (367 rounds / 889 verified / 657 accepted).
The external CPU screen failed at 193.606%; clocks were not verified. Retain it
as a diagnostic working profile, not a production throughput qualification.
This number is not combined with the 21.384 short greedy result or the 23.668
native-host observation; their workloads and controls differ.

| Investigation | Outcome and disposition | Aggregate evidence |
| --- | --- | --- |
| Qualified Stark baseline | 723.751 prefill / 17.070 decode tok/s; qualification remains governed by the archived run record. | [Qualification record](../archive/runs/2026-11-04-glm-exl3-qualification.md) |
| Current-source normal host control | 835.678 / 16.632 at MTP off; quiet-host checks passed, but full output/readout parity was unavailable. | [Normal-host control](#same-source-normal-host-control); `053af3675a498f41c4afaba2415a124f60b1dbd390173768d6d918f451240666` |
| Selected first-request sampled profile | Best completed long sampled run, diagnostic only for the CPU and clock limitations above. | `large-cache-first-request-129-156-v1.json`; `01c6a99aea0a29bdd6c377597896b8fbc4e8fe2362c09541171052bb5bad5a2a` |
| Named retained profile setup | Explicit BF16 native profile; corrected v2 stages pass ABI/protocol checks, and the new worker passes six actual-model defaults/opt-out/prefix cases. Preparation v1 and smoke v1 harness failures remain separate. No new throughput or broad quality qualification. | [2026-11-05 profile record](../archive/runs/2026-11-05-glm-retained-profile.md); smoke `dc37af1bcd4340744e196cc9f5bd3bc0564831c7274a021ab1a53f50d2535697` |
| Numerical/MTP and state investigations | Parity, state and mismatch diagnostics are recorded above; they do not qualify sampled performance. No MTP throughput result cleared the host screen. | Sections [MTP numerical policy](#mtp-numerical-policy-investigation), [longer sampled diagnostic](#longer-sampled-diagnostic), and [worker and quality status](#worker-and-quality-status) |
| Quality smoke | 22/24 across both modes (11/12 each); an all-pass claim is false. No quality clearance follows for the retained profile or experimental heads. | `runtime/glm-prefill-throughput-2026-11-03/mtp-quality-smoke-v2/QUALITY-V9C.json` |
| Residency, read policy and cache-frontier controls | Fixed-row residency effect was under 1% with a failed CPU screen. Higher-row/high-power comparisons failed clock matching; none establishes a causal throughput gain. | Sections [fixed-row residency](#fixed-row-residency-diagnostic), [direct versus staged reads](#direct-versus-staged-expert-reads), [cache frontier](#cache-frontier-with-workload-clock-capture), and [High Power repeat](#operator-enabled-high-power-repeat) |
| P/Q sampler prototype | No useful speed gain demonstrated; remains private and disabled. | [P/Q sampler check](#private-gpu-pq-sampler-speed-check); `pq-speed-pair-v3.json` and `pq-speed-verify-all-v1.json` |
| Operator-pinned P190/D217 and P129/D217 | Both were aborted by the retained 112 GiB live memory guard before a measured completion; no output or throughput result. | [Attempt record](#operator-pinned-217-row-attempts); `06318e9601d9139b357128da5142bf66eb68e29390411d1915eec89e90f9f230`, `89dce7ffb5e4318496982c6986e634546318a5ace396d38ad9bb43d4dd824b76` |
| Q8 draft head | No useful gain shown; BF16 control failed CPU screening, clocks unmatched, Q8 trajectory differed; quality remains unqualified and Q8 stays off. | [Q8 comparison](#private-q8-draft-head-comparison); `b78bf50a0943178f752ab0447ca626809b60892728b4f515eb18fa86c91b260f` |
| Frequency-first retention | Rejected: 11.89% more matched decode reads/bytes; both arms failed CPU screening. | [Frequency-first comparison](#private-frequency-first-comparison); `6c9613ee8c2a27552a4aeb84c60e667b5c18ea0b9579d0b200fef6a8af064ff2` |
| Transition-only cache scorer | Built as prediction-only; no candidate inference completed because external Cargo activity blocked launches. No measured verdict; do not enable. | [Prediction status](#transition-only-prediction-status); `30412ab62c2f9c1b4f8f2c0a0fc6f2fd27a278fe0050682c7d1788409eb139fa`, `f51f7d597370475f577e11a3c23a501ff6964c9412638eec7a328fe02dee4832`, `18789832e50b4b5894120e0d998d86af67c099b058a667aaadce504376715c17` |
| GPU timeline profiler | Source review only; no profile build, capture or run occurred. Route tracing was not enabled. File output without route tracing was not established, and synchronization/fence overhead remains unknown. | Private source review recorded in `runtime/glm-prefill-throughput-2026-11-03/WORK_STATE.md`; no profiler result artifact. |

The detailed sections above preserve the earlier admission, first-request,
templating, read-policy and hardware investigations. Private runtime artifacts
and checkpoint notes remain ignored; this tracked index records aggregate
outcomes and hashes only. No experimental alternative is selected for
production, and the approximately 1,000/33 target remains unmet.
