# Flash-Next optimization campaign: findings and attempt history

Compiled: 2026-11-08. Covers the complete optimization conversation beginning
with the Isoquant benchmark discussion, the November 7 experiments, the
November 8 cross-backend follow-up, and the CI repair. Conversation statements
were checked against the source-bound reports and aggregate receipts listed
below. This is a retrospective; it does not rerun or strengthen those studies.

Implementation status was checked after [PR 98](https://github.com/jshahbazi/stark/pull/98)
merged on November 8, at main commit `7dc51834`. Its final branch head was
`dfb2bacf1b02684b497b1f2470656849e40dc647`, with CI, Security and CodeQL passing.
The production changes described here are in merged source. Installation of
those changes in a particular app build is outside this record.

Stable attempt IDs below are indexed in the rolling
[model experiment ledger](../model-experiments/2026-11-08-flashnext-optimization.md).
The ledger maps short report labels to globally dated IDs. Original reports, patches,
hashes and rejected observations remain unchanged.

## Outcomes by model and artifact

The useful production outcome was lower latency after idle. The MoE projection
candidates had small synthetic component gains and passed bounded correctness
checks, but established no reliable whole-model benefit and were removed.
The sustained decline from roughly 95 decode tok/s remains unresolved.

| Model / artifact | Work completed in this conversation | Result and final disposition |
| --- | --- | --- |
| Qwen3.8-Flash-Next, Lily Q4 XL | Draft-depth screening, encoder cutoff, MoE projection candidates, cadence/clock diagnostics, generation and scoring wake fixes | Encoder and projection candidates excluded or withdrawn. Generation and scoring queue maintenance merged. Post-idle generation falls from about 1.08 s to 0.30 s; fresh scoring from about 1.08 s to 0.30 s, with its coverage limits below. No sustained 95 tok/s guarantee. |
| Qwen3.8-Flash-Next, official mixed-4/8-bit pack | Generation wake-delay reproduction and matched queue-maintenance validation | Generation fix merged: about 1.12–1.13 s to 0.31–0.32 s after idle. MoE projection and scoring results from the Lily pack do not qualify this artifact. |
| Dense Qwen3.8-27B, native oQ4e `coarse3` conversion | Metal4 generation wake-delay reproduction and matched validation | Shared-event maintenance merged. Separate fully screened 512-input/64-output repeat: about 0.71–0.75 s to 0.62 s after idle. No new dense decode optimization or draft-depth recommendation. |
| GLM-5.3-Flash, GLM EXL3, Flint, other dense models and vision wrappers | Considered in the initial model-family discussion and backend exposure audit | No model-bound speed or wake qualification in this conversation. Their different execution paths need their own assessment. Vision wrappers were excluded from the new native text hook. |

The initial request's phrase “dense Qwen3.8 Flash-Next” was clarified to
**Flash-Next only**. Flash-Next is the MoE architecture; dense Qwen3.8-27B is
a separate model. Dense 27B entered this campaign only in the later wake-delay
follow-up. Earlier dense optimization work is linked under prior history below.

## Artifact and measurement boundaries

| Artifact key | Checkpoint revision and execution policy | Evidence |
| --- | --- | --- |
| FN-LILY | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, `0f94842e3682b49acace4daeb4cf2e65a6f94f83`; 55 weight shards, 106,905,155,192 bytes | [Pilot](2026-11-07-flashnext-optimization-pilot.md), [projection study](2026-11-07-flashnext-moe-projections.md), [cross-backend receipt](data/idle-wake-cross-backend-2026-11-08.json) |
| FN-OFFICIAL | `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`, `788c07a8725642bccf13a0a3b8a21954ab84b4f2`; 101 shards plus n-gram table, 107,303,406,192 weight bytes | [Cross-backend study](2026-11-08-cross-backend-idle-wake.md) |
| D27-COARSE3 | `Jundot/Qwen3.8-27B-oQ4e-mtp`, `7dd276b77fcb27c982ebd8d8a434539345713fea`; converted native weight payload 17,078,038,512 bytes; three-bit draft-only readout | [Cross-backend study](2026-11-08-cross-backend-idle-wake.md), [prior dense qualification](2026-11-06-qwen38-27b-packed-followup.md) |

The Lily scheduling/projection experiments preserve BF16 activations, Q8
attention cache, F32 reductions/recurrent state and the existing F32 readout
policy. No new precision policy or requantization was tested here. The official
pack and dense conversion retain their separately qualified arithmetic.
Weight, configuration, tokenizer, binary and source identities are bound by
the linked receipts; a shared model name does not transfer qualification.

The measured host was an Apple M5 Max, Mac17,7, 40 GPU cores, 128 GiB physical
memory, macOS 26.7.2. Later studies verified AC and the existing High Power
preference. This campaign did not isolate a power-mode effect.

Decode throughput counts advances **after** the first sample. For the common
256-output fixture it is 255 advances divided by native decode time. Time to
first sample and client-observed complete-request wall time are separate
metrics. Scoring latency covers the complete fresh score call. Synthetic
projection GPU microseconds exclude encoding, transport and the rest of the
model. These measurements cannot be combined into one speedup curve.

## Pilot: draft depth, encoder scheduling and profiling

**FN-01 — draft-depth screen.** Extended the paired MTP harness into a bounded
runner with artifact/binary guards, deterministic within-route hashes,
candidate comparisons, sanitized failures and explicit observed/screened modes.
The completed pilot had 18 cells and 144 measured requests. Correctness checks
passed; **84/144 request host screens failed**. Quiet admission had timed out,
so completed grids used an explicitly observation-only protocol. No pilot
timing result was qualified.

With 64 outputs, seed 42, temperature 1, top-k 20 and top-p 0.95, the draft screen
observed:

| Input tokens | Drafts | MTP off decode tok/s | MTP on decode tok/s | Median paired complete-request speedup |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 81.55 | 92.75 | 1.116× |
| 32 | 2 | 72.62 | 78.81 | 1.079× |
| 32 | 3 | 78.75 | 66.42 | 0.856× |
| 512 | 1 | 76.92 | 89.04 | 1.109× |
| 512 | 2 | 76.73 | 90.35 | 1.123× |
| 512 | 3 | 78.01 | 81.15 | 1.070× |

One or two drafts were useful follow-up candidates. Three drafts lost on the
short prefix. This did not qualify a new default or an adaptive controller;
MTP on/off can consume randomness differently, so checks compare repeatability
within each route rather than requiring the two routes' trajectories to match.

**FN-02 — concurrent encoder cutoff 16 to 64.** The one-line candidate applies
to the complete batched graph. It is not an isolated expert-kernel change,
and speculative verification's at-most-four rows already choose the same
encoder in both builds. Same-route outputs matched, including a separate
48-input/128-output corpus and seed. Complete-request baseline/candidate ratios
were 0.977×, 0.918×, 0.993× and 0.966× across the four main conditions.
Prefill-only ratios were 1.004–1.023×. No complete-request benefit was
demonstrated. **The production cutoff remains 16.**

**FN-03 — separate stage profiling.** Two warm diagnostic requests, 62 decode
passes, placed MoE at 35.1% of the recorded GPU command span, GDN at 24.1%,
hyper-connections at 20.8%, QSA at 10.8%, and the head at 8.3%. Expert gate/up
plus shared down accounted for 16.8%, suggesting the next bounded target.
These instrumented shares overlap hierarchical counters and are not speedups
or utilization measurements. They were kept separate from uninstrumented runs.

Evidence: [pilot report](2026-11-07-flashnext-optimization-pilot.md),
[aggregate](data/flashnext-optimization-pilot-2026-11-07.json),
[cutoff patch](../archive/experiments/2026-11-07-flashnext-pilot/concurrent64.patch).

## MoE projection candidates and numerical controls

**FN-04 — fixed-geometry gate/up and shared down.** Specialized addressing for
Q4/group-64 expert gate/up at hidden width 2,560 / expert width 640, plus
Q8/group-64 shared down at 640 to 2,560. Shape guards retain generic fallbacks.
The existing path already fuses gate/up with SwiGLU and overlaps shared down;
this was a specialization of that path. The ordinary decode kernels differ
from MTP's batched target-verification row kernels.

**FN-05 — fixed gate/up with shared down queued first.** Retained the gate
specialization, restored the ordinary shared-down kernel, and queued shared
down first inside the existing concurrent dependency level. No new barrier,
weight format, reduction policy or persistent inference state was introduced.

| Component experiment | Reference / candidate median, µs per layer | Median paired speedup | Decision |
| --- | ---: | ---: | --- |
| FN-04 expert gate/up | 44.82 / 43.45 | 1.0321× | Component evidence only |
| FN-04 combined projections | 37.53 / 37.19 | 1.0074× | Component evidence only |
| FN-05 expert gate/up, separate window | 45.64 / 43.89 | 1.0231× | Component evidence only |
| FN-05 combined, shared first | 39.39 / 38.42 | 1.0221× | About 2.2% paired component gain; 16/16 pair wins |

Paired speedups are medians of pairwise ratios, not ratios of the marginal
medians. Separate windows cannot be subtracted to estimate shared-down cost.

**FN-06 — correctness coverage and repair.** Both candidates passed bounded
controls. The original 12-case test compared 99,840 BF16 values exactly, but
review found its “duplicate routes” case contained only permutations. A
separate test-only supplement added actual repeated routes: 16 cases and
135,680 values matched exactly, with unchanged production code and generator
hash. Offset views, canaries, signed zeros, subnormals and cancellation were
also covered.

The 35,758,080-value serial/verify/rollback audit had unchanged results across
builds. Verification, rollback and host-serial phases had zero unequal values.
The pre-existing draft-head seam retained 248,320 unequal values, maximum
absolute error 0.0533607, and no top-1 mismatches. The experiment preserved
that policy; it did not make every draft-head value exact.

Prefix capture/restoration matched fresh execution. The final candidate's
independent scoring replay matched complete-vocabulary digests for 993,280 F32
values across input lengths 1, 4, 64 and 512. Its native suite passed 817 tests,
with 91 ignored, plus nine integration checks and three doctests. These checks
support the named scheduling changes, not broad answer quality.

**FN-07 — whole-model comparisons.** The first candidate's completed ABBA run
observed 95.37 baseline versus 95.54 candidate decode tok/s with two drafts,
and 82.61 versus 82.92 off. One of 32 arms exceeded the CPU screen at 154.8%
against a 150% limit. The small difference was unqualified.

Repeated attempts for the final candidate were incomplete or blocked by host
activity. A separately identified observation-only ABBA then completed all
32 measured arms on 512 input / 256 output / context 768:

| Route | Baseline decode tok/s | Final candidate decode tok/s | Baseline / candidate median full wall, s |
| --- | ---: | ---: | ---: |
| MTP off | 80.33 | 75.63 | 3.5000 / 3.6358 |
| Two drafts | 85.50 | 81.23 | 3.4038 / 3.4021 |

Four arms failed host screens; no quiet admission was attempted. The observed
candidate rates were 5.86% and 4.99% lower, but this did not isolate a causal
kernel regression. Two-draft whole-request latency was essentially unchanged.
Later same-model crossover controls also failed to establish a reliable gain.
**Both candidates were withdrawn, and original projection dispatch restored.**

Evidence: [projection report](2026-11-07-flashnext-moe-projections.md),
[initial receipts](data/flashnext-moe-projections-2026-11-07.json),
[completed observed ABBA](data/flashnext-moe-matched-observed-2026-11-07.json),
[source patches](../archive/experiments/2026-11-07-flashnext-moe/README.md).

## What happened to 95 tok/s

The conversation initially answered “still roughly 95” from **the first
candidate's 95.54** result while the final candidate's quiet comparison was
incomplete. That answer was too broad. The final candidate's later completed
observed ABBA measured **81.23**. A later cadence control did reproduce
**95.61** on that final candidate, specifically after a ten-second pause.
These are different measurements and must retain their candidate/protocol labels.

**FN-08 — request cadence.** With the same executable, input, seed, outputs and
MTP work counts, the baseline measured 94.60 tok/s after a pause and 85.29 by
the fourth consecutive request. The final candidate measured 95.61 then 89.45.
All eight candidate screens passed; only 8/16 baseline screens passed, and
the probes used different cycle counts. This establishes within-worker cadence
dependence, not superiority of one kernel.

The same baseline also moved from 95.37 to 85.50 between protocols while its
median first-sample time improved from 1.1907 to 0.3005 seconds and full wall
time from 3.8722 to 3.4038 seconds. Lower decode tok/s can coexist with a faster
complete request when the idle wake cost disappears.

**FN-09 — display/lock-screen hypothesis.** The console was locked during
early failed quiet admissions, and `loginwindow` had the largest observed
graphics-client counter increase. A separate display-sleep control could not
confirm actual display power state; all eight arms failed host screens,
with another model worker and 6,508 swapouts. **No causal lock-screen/display
conclusion was established.** Activity in that later test cannot explain an
earlier run retrospectively.

**FN-10 — GPU span and clock diagnosis, including gate-only variant.** A
64-request grid and separate initial 16-request clock probe added source-bound
native GPU spans and rootless clock/temperature telemetry. In the unchanged
baseline, the first/eighth requests measured 95.46 / 82.20 tok/s and
1,620 / 1,488 MHz. Verification elapsed grew 0.3893 seconds; GPU command span
grew 0.3808 seconds, approximately **98% of the increase**.

MTP work remained 108 rounds, 214 proposals, 147 accepted tokens and 322
verified rows per request. Reduced acceptance does not explain this case.
GPU spans include stalls/preemption; they do not identify pure arithmetic cost.
Other slow requests had different clock/temperature relationships. Coarse
nominal thermal state did not guarantee a fixed GPU frequency. **The exact
thermal, power, memory or firmware limiter remains unknown.** The gate-only
variant also established no accepted gain.

**FN-11 — one loaded model, three projection routes.** A private crossover
worker compared generic, fixed-gate, and fixed-gate/shared-first routes inside
one loaded model, clearing retained state when switching. Shuffled ABBA/BAAB
groups retained all 32 arms. Geometric-mean changes were −1.13% / −1.64%
with MTP off and +0.62% / +0.85% with MTP on. MTP-on signs reversed between
groups; two MTP-off arms failed CPU screens. No reliable benefit or confirmed
5% code regression followed. Restored-source smoke subsequently observed
81.21 off and 84.99 on, with all eight screens passing but no quiet admission.

The sustained study retains **128 diagnostics, 125 host-screen passes**, plus
separately identified warmups and restored-source validation. The initial
clock-only probe has a recorded harness hash without a retained byte-exact
v2 source snapshot; later v3/v4 sources are archived. Telemetry coverage is
partial and has no privileged `powermetrics` cross-check.

Evidence: [cadence receipts](data/flashnext-moe-cadence-2026-11-07.json),
[sustained diagnosis](2026-11-07-flashnext-sustained-diagnosis.md),
[128-request aggregate](data/flashnext-sustained-diagnosis-2026-11-07.json).

## Wake-delay fixes that were retained

**FN-12 — private causal prototype.** During ten-second pauses, signal/wait
on the existing empty queue every 500 ms. Plain/keepalive/keepalive/plain
control reduced mean first-sample time **1.0035 to 0.3035 seconds** while
prefill GPU span stayed about 0.292 seconds and decode about 95 tok/s.
All 16 screens passed. This implicated queue/residency wake work without
identifying a precise driver function. At this stage it was a private
diagnostic, and the conversation correctly clarified that no production fix
had yet shipped.

**FN-13 — Lily generation production fix.** The owner thread polls while
waiting for input and signals the existing queue every 500 ms between complete
frames. All 16 measured ABBA requests passed host screens and seeded hash
checks. With 512 input / 256 output, after ten seconds idle:

| Lily generation | Baseline first sample, s | Fixed first sample, s | Fixed warm control, s |
| --- | ---: | ---: | ---: |
| MTP off | 1.0747 | 0.2965 | 0.2965 |
| Two drafts | 1.0919 | 0.3053 | 0.2979 |

The benefit is about 0.78 seconds of avoided idle latency. Decode was
effectively unchanged: 82.82 / 82.74 tok/s off and 95.23 / 95.12 on, old/fixed.
Those post-pause observations do not establish sustained throughput. A
separate six-request control restored 4,096 prefix tokens after idle in both
MTP modes and matched fresh outputs.

**FN-14 — official generation. FN-15 — Lily scoring/digest. D27-01 — dense
Metal4 generation.** Extended the shared hooks after reproducing exposure
on these additional paths. The primary 14-session campaign used separate
baseline/candidate/candidate/baseline blocks plus digest controls:

| Endpoint | Baseline after idle, s | Fixed after idle, s | Timing boundary |
| --- | ---: | ---: | --- |
| Official generation, MTP off | 1.1226 | 0.3174 | All four first-response screens pass |
| Official generation, two drafts | 1.1305 | 0.3131 | All four first-response screens pass |
| Lily fresh scoring | 1.0810 | 0.3028 | 3/4 first-response screens pass; one fast candidate call lacks a sample |
| Lily complete-vocabulary digest | 1.1091 | 0.3032 | Both post-idle screens pass; later baseline warm control lacks coverage |
| Dense 27B, MTP off, separate repeat | 0.7057 | 0.6229 | Separate 512-input/64-output/context-576 ABBA |
| Dense 27B, two drafts, separate repeat | 0.7542 | 0.6247 | Same repeat; all 16 request screens pass |

The primary 512-input/256-output/context-768 campaign retained **38/44 passing
host screens**: three CPU-interfered calls and three short calls without
captured host observations remain unqualified. Its first-response counts are
8/8 official, 3/4 scoring and 7/8 dense. The whole primary campaign is not
timing-qualified. The original dense first-sample medians were 0.7267 to
0.6328 seconds off and 0.7438 to 0.6346 on; these observations remain intact.

The dense-only repeat was triggered by failed host screens, not its measured
latency. Its smaller output budget fit the reserved window. It qualifies the
named 512/64 latency fixture and cannot qualify the original 512/256 workload
or a decode-throughput claim. The scoring artifact is FN-LILY; this did not
benchmark fresh scoring on FN-OFFICIAL.

Candidate outputs and score fingerprints match baseline within each artifact
and route. Full-vocabulary digest controls pass. Post-idle cache hits increment
no native-call counter. Official generation restores 4,096 prefix tokens;
dense restores 1,024, each matching capture/reuse/fresh references in both
MTP routes. All 18 primary/repeat sessions exit zero on EOF without fallback.

Official generation and Lily scoring use the Lily queue. Native text generation
uses one lazy private shared event on its existing Metal4 queue, with increasing
signal values and a five-second wait limit. A stalled maintenance wait can delay
an arriving request until that timeout. Successful ticks change no model math,
sampling or cached state. Failure clears retained state and exits for reload.
The ten-minute generation prefix deadline includes maintenance time.

Observed fixed-worker idle CPU was about **0.03–0.035% of one core**, versus
about 0.002% baseline. The original Lily control measured 0.033%. The recorded
GPU activity and rootless GPU-watt estimates increased slightly; short AC
windows do not qualify battery life or long-duration system energy. Maintenance
continues while the worker is loaded. System suspension and stalled partial
request frames remain outside the awake, between-frame guarantee. Dense scoring,
vision wrappers and other hybrid backends retain their prior scheduling.

Evidence: [Lily fix](2026-11-07-flashnext-idle-wake-fix.md),
[Lily receipt](data/flashnext-idle-wake-fix-2026-11-07.json),
[cross-backend study](2026-11-08-cross-backend-idle-wake.md),
[cross-backend receipt](data/idle-wake-cross-backend-2026-11-08.json).

## Tooling failures, repairs and evidence controls

These attempts affected experiment reliability or model admission; they are
tracked independently from model speedups.

| ID | What failed or was learned | Repair / final outcome |
| --- | --- | --- |
| TOOL-01 | Early quiet-window attempts overlapped compilers, other workers, CPU bursts or graphics activity. Reserving the machine did not itself establish quiet admission. | Added/retained explicit gates, per-arm screens and durable rejected receipts. Initial MoE attempts retain 47 measured arms across five attempts. Session-level admission and observation-only protocols are separately identified. Later GPU windows coordinated through one shared lock with the recommended-model benchmark chat, including teardown. |
| TOOL-02 | Initial pilot failure receipt was generic; later cross-backend scoring failed before durable result capture. A harness also expected flat MTP counters while the hybrid worker nests them under `mtp.stats`. | Added sanitized static failure categories, partial stage checkpoints, nested-counter handling, and numeric errno metadata. Very short cache hits are correctness controls; missing samples for fresh calls stay ineligible. Preliminary completed cells remain separate from final runs. |
| TOOL-03 | A saved candidate diff used a local display formatter; duplicate-route test coverage was overstated; the first clock probe lacked a retained source snapshot. | Regenerated an applyable patch, reverse-checked it and preserved both receipt hashes. Expanded route tests without rewriting old evidence. Retained the clock-probe provenance limitation. No retroactive source qualification. |
| TOOL-04 | Initial Lily wake implementation could reuse a prefix when input arrived during maintenance after the expiry deadline. | Added deadline checks after maintenance and when input is ready. Stopped the preliminary run after two sessions, preserved eight measured arms separately, rebuilt and restarted the full ABBA. Final 16-arm result uses corrected source. |
| TOOL-05 | Historical dense `long3` fixture contained unsupported experimental `mtp_config.max_drafts: 7`; strict admission rejected it. | Used the previously qualified `coarse3` conversion with the same unchanged weight payload and separately sealed admitted metadata. Did not loosen the loader or edit the rejected artifact. This was fixture selection, not a measured dense inference regression. |
| TOOL-06 | macOS process-group cleanup raised `EPERM` for an exiting/unreaped worker and masked the original startup rejection. | Reap/poll the leader and handle the short exit race before signaling surviving group members. Real subprocess regression preserves the static startup failure. Cleanup failures for genuinely live groups are not silently ignored. |
| TOOL-07 | Main's newer image work conflicted with idle hooks and changed the source file used as the benchmark corpus. | Preserved image support and hooks, rebuilt in an isolated target, and froze corpus bytes from `753884b8`. Separate integration smoke/prefix controls qualify the merged worker; original timing keeps its original source binding. |
| TOOL-08 | Ubuntu CI returned `unsupported` for `snapshot(0)` while the macOS-tested helper expected `invalid_pid`. | PID validation now precedes platform selection. Mocked Linux/macOS invalid-input and native-failure tests added. Final hosted checks pass: 116 benchmark tests plus 12 subtests; converter 6,604 passed / 96 skipped; portable generation 85 passed. |

Other setup-only failures were a missing CMake entry in shell PATH and a private
dense-repeat launcher missing the scripts import path. Both were corrected
before the relevant successful runs. The cross-backend report preserves the
initial warmup-only attempts, Docker-scan admission timeout, wrong-fixture
baselines and launcher failure separately. They are not performance-based
exclusions. Raw exception dumps are not published.

Worker/source identities, artifact guards, workload hashes and precision
policies remain essential. The later coordinated fingerprint pass freshly
checked official and Lily payloads; current official metadata is sealed
separately from its older snapshot. Dense payload hash reuse relies on the
same unchanged inode and full identity tuple. No whole-snapshot byte identity
is inferred merely from weight identity.

Evidence: [pilot](2026-11-07-flashnext-optimization-pilot.md),
[projection attempts](2026-11-07-flashnext-moe-projections.md),
[Lily wake preliminary/final split](2026-11-07-flashnext-idle-wake-fix.md),
[integration record](../archive/runs/2026-11-07-flashnext-pr98-integration.md),
[cross-backend preliminary attempts](2026-11-08-cross-backend-idle-wake.md),
[CI repair/recheck](../archive/runs/2026-11-08-pr98-ci-recheck.md).

## Source decisions and remaining work

| Change | Source transition | Final state at compilation |
| --- | --- | --- |
| Bounded pilot harness and records | `f5750888`, `72b5aaef` | Retained tooling; concurrent-64 candidate archived |
| MoE candidates | `fa0db655`; later observed run `af9cfba0` | Archived patches, removed from current projection dispatch |
| Cadence diagnostics | `6698d2c1` | Retained tooling and evidence |
| Restore projections / GPU-span diagnostics | `5f974a93` | Original projection dispatch restored; opt-in aggregate telemetry retained |
| Lily generation wake fix | `753884b8` | Retained; separately validated after main integration at `640a6a7b` |
| Official/scoring/Metal4 wake follow-up | `4fe11fad` | Retained scheduling/input changes; no shader or precision change |
| Telemetry CI portability repair | `dfb2bacf` | Retained, hosted checks green |
| Integration into main | PR 98, `7dc51834` | Merged |

The initial [Isoquant discussion](https://isoquant.ai/benchmark) supplied an
experimental structure: bounded tasks, frozen candidates, correctness gates,
independent checks and complete-request measurements. No Isoquant code was
imported, and no NVIDIA result establishes a Metal speedup. A general automatic
optimizer, online draft controller, and measured improvement in optimization
search efficiency were not implemented or demonstrated.

Outstanding work is recorded as follow-up rather than as an implicit success:

- **Sustained Lily throughput:** identify a concrete limiter using measurements
  that distinguish execution stalls, memory behavior and operating-state changes.
  The demonstrated wake fix does not resolve this.
- **Projection search:** original dispatch remains the baseline. Any revised
  candidate needs fresh numerical controls and a repeatable screened
  whole-request benefit. Re-running the withdrawn patch is not a prerequisite
  for retaining the independent wake fix.
- **Scoring coverage:** repeat the fresh-score comparison with phase sampling
  that captures short calls, if a fully covered primary latency claim is needed.
  Never reclassify the historical missing samples.
- **Idle resource cost:** longer-duration energy/battery evidence remains open.
- **Other models/backends:** GLM, EXL3, Flint, vision and other dense artifacts
  need model-specific warm/post-idle checks before claiming exposure or benefit.
- **Dense adaptive verification:** proposed in the opening discussion and
  already marked unimplemented in its earlier ledger; not attempted here.

Existing histories remain the authority for work predating this conversation:
[142-attempt Qwen3.8 history](QWEN38_OPTIMIZATION_HISTORY.md),
[dense 27B approach ledger](2026-11-06-qwen38-27b-approaches.md),
and the [benchmark index](README.md) for GLM, OLMo, Gemma and other engines.
The rolling ledger adds this campaign to those records without relabeling
older results or claiming repository-wide historical completeness.
