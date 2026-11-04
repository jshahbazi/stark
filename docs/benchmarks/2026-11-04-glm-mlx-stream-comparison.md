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
| Expert rows, prompt/decode | 142 / 169 in the qualified headless baseline; latest head-loaded host control 104 / 130 | 190 / 217 | At fixed 104 prompt rows, 104-to-130 decode rows cut settled decode payload by 26.2%; faster screened observations support the residency hypothesis, but the complete ABBA failed its CPU screen. Extrapolation to 217 is unsupported |
| Allocation admission | 112 GiB ceiling, 2 GiB margin, native census and automatic fill | Same ceiling/margin; 190/217 requires a baseline at most 12.69 GB | Baselines vary by run: about 35–41 GB in earlier controls and 46–47 GB in the latest host cells. The upstream 12.69 GB is an admission threshold, not a measured baseline for the 33.15 tok/s run; the current excess is not established reclaimable capacity |
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
| Hardware conditions | M5 Max, 40 GPU cores, 128 GiB; macOS 26.7.1 (25G241), AC `powermode 0`; CPU/paging/thermal screen | Same advertised hardware, maximum fans and clock-matched pairs; historical OS build not established | GPU frequency/fan telemetry remains unavailable: the read-only power sampler requires administrator authentication. Thermal readings do not prove matched GPU frequency |
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
