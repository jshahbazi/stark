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

## Configuration and evidence

| Area | Stark qualified baseline | Historical mlx-stream cell / current public host | Performance effect established so far |
| --- | --- | --- | --- |
| Throughput | 723.751 prefill / 17.070 decode | 1,093.5 / 33.15 | Observed gaps of about 34% and 49%; causal split unknown |
| Weights | Verified HF revision `50237d5c26ceb98bfd85f779990b0f5e6d7af144`; EXL3 K3 experts, FP8 residents | Same advertised pack format | Exact historical payload identity remains unproven |
| Engine source | Public host `dc92a3e8`, plugin `7d29eaa7`, pinned MLX/MLX-C | Historical host `a8f6c6b8`, plugin `88d2aad` | A screened current-public-source host control measured 835.678 prefill / 16.632 decode at 131/159 rows with MTP off |
| MTP | Disabled in qualified baseline; native commit interface and head-inclusive prefix snapshot implemented in a private candidate | Depth 3, gate 0.5, exact acceptance, BF16 draft head | Same-policy greedy128 output hashes and round counts match the normal host. Observed native-host decode 23.523 / 23.668 tok/s failed the external CPU screen; production throughput remains unqualified |
| Expert rows, prompt/decode | 142 / 169 | 190 / 217 | Lower rows increase misses; local penalty unmeasured. The upstream nearby 217-to-225 ladder gained about 3% decode, with flat prefill; extrapolation to 169 is unsupported |
| Allocation admission | 112 GiB ceiling, 2 GiB margin, native census and automatic fill | Same ceiling/margin; standard row pin needs a much smaller baseline | Current admission baselines around 35–41 GB and non-file readings around 27–28 GB constrain rows; historical baseline was about 12.6 GB. The unexplained difference is not established reclaimable capacity |
| Residency capacity | About 81 GB, bounded by the model bill and remaining budget | Current public host defaults to Metal's recommended maximum; historical policy not independently established | Fixed-row ABBA diagnostic observed under 1% effect and exact readout fingerprints; its external CPU screen failed, so no throughput gain qualifies |
| Prompt-combine prerequisite | Unset wiring environment selects `inplace`, despite bill-bound capacity | Unset/`max` selects `inplace`; `fit`/`off` selects `fused` | The plugin couples this route to residency policy. The capacity experiment keeps the route unchanged in both arms |
| KDA | Baseline did not record installation; new fixed-row control confirms per-core on 40 cores, 80 groups, 104 rows, 2 groups/core | Same expected plan | Stock fallback is ruled out for the new control |
| DSA / EXL3 / fusions | NAX, 2,048-row waves, route cut, sampled ordering, direct reads and launch fusions configured; normal-host startup confirms NAX | Tuned defaults present | Source configuration aligns; full invocation counters remain unchecked |
| Prefill tail | Eager completion inside prefill | Deferred phase/growth and MTP catch-up | With MTP off, deferred/forget did not improve total latency over off/keep. MTP-on catch-up has a different cost and is being measured separately |
| Prompt residents and startup | Current `keep` default; excluded warmup then repeated reset requests | Historical `forget`; one request per fresh server, no eager warmup | First requests after load reached about 918–936 prefill tok/s in screened serial controls; later requests slowed. These are fresh-server observations with existing OS file cache, not cold-storage measurements |
| Readout and sampler | Baseline uses complete 154,880-value F32 readout and CPU greedy argmax; private candidate uses native chosen-token rounds | Native chosen-token/card path | Matching serial host control and matching MTP host/control observations provide no evidence of a twofold adapter tax; sampled-policy and worker qualification are separate gates |
| Input | 16,384 raw-source tokens | 16,381 templated tokens | Contents affect expert routes and MTP acceptance; the three-token difference is not the substantive issue |
| Decode length and sampling | 128 greedy steps | 1,024 sampled steps, temperature 1, top-p 0.95, seed 1 | Different trajectories and fixed-cost sensitivity |
| Context and prefix reuse | 32,768; reuse off for throughput | Same context; reuse off | Aligned. The serial stage has no snapshot cost; prefix validation is separate evidence |
| Hardware conditions | M5 Max, 40 GPU cores, 128 GiB; current AC power and low-power mode off; CPU/paging/thermal screen | Same advertised hardware, maximum fans and clock-matched pairs | GPU frequency/fan telemetry remains unavailable: the read-only power sampler requires administrator authentication. Thermal readings do not prove matched GPU frequency |

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

## Experiment order

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

## Diagnostic progress

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
208.314 percent, above the existing 200-percent screen limit. It had no compiler
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
paths. Native capability bits stay disabled in the ordinary build while worker
and sampled-policy qualification proceed.

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

Still open: MTP-on tail/total-latency measurements, 1,024-output acceptance and
throughput, aggregate expert I/O at the admitted cache frontier, source-bound
worker lifecycle and quality checks, admission-term reconciliation, and matched
hardware clocks. These checks are needed before attributing the remaining gap
to a new kernel or declaring the historical target reproduced.

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
private settings. A separately frozen 119/146-row pair is the next control; no
over-budget pin is forced.

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
same-source cache-row comparison will hold prefill rows at 104 and compare
104 versus 130 decode rows, retaining identical greedy128 output hashes and
counts. Claims about 217 rows or 33 tok/s still require an actual admitted run.

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

A separately versioned correction preserves all six original task, expected
answer, rendered input and token-input hashes, all 24 arms, maximum reasoning
and the 1,024-output limit. It will grade the template-defined final channel
with the unchanged strict JSON grader, fail closed on malformed framing or
truncation, and retain every failed arm. No question replacement or selected
retry is allowed. Until that run completes, semantic quality remains
unqualified and ordinary native MTP capability bits remain off.

Interrupted quality result SHA-256:
`d4f693c0745d16bed7e436b72d2ed82b1ba16bc8a0cf73a777e591e15eb76b29`.
