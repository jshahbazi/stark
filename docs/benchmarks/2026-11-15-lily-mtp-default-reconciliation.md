# Lily MTP default: comparison reconciliation

Date: 2026-11-15. Attempt: `20261115-LILY-MTP1-03`.

Decision: restore the two-draft recommendation. One versus two has a real but
modest benefit in the motivating top-k case, no benefit in the 512-token greedy
guard, and a consistent long-nucleus regression. Explicit one draft remains available;
changing the artifact-wide default is not justified. All 198 planned measured
requests and 24 fixed setup requests are retained. No app is installed or
released.

## Question and fixed study plan

The original +9.93% result compared one draft against MTP off. The later +2.65%
result compared the candidate's recommended one draft against the prior two
drafts, using separate loaded sessions and synthetic cross-session pairs.
These are different comparisons. The work here must explain the measurement
gap and decide whether the branch's one-draft recommendation is worthwhile;
it must not seek a favorable rerun or claim a universal speedup.

The saved actual-default sessions exhibit large within-session rate changes
at unchanged work counts. An inspection of routing, model resets, scratch
capacity and selectors found no distinct decode path for explicit one draft
versus the recommendation resolving to one. The source patch changes the
recommendation, not inference kernels. Environmental/session effects remain a
hypothesis rather than an identified GPU limiter.

Preregistered before new execution:

- Use the exact pinned Lily artifact, frozen corpora, images/MTP resident,
  unchanged precision and sealed before/candidate binaries from
  [the recommendation study](2026-11-15-flashnext-lily-one-draft-default.md).
  Do not rebuild or alter native code during measurement.
- Compare explicit depths zero, one and two within each loaded worker. Raw
  requests have no cache scope and must report zero reused tokens.
- Six primary-cell blocks on the before binary: each of the six permutations
  of 0/1/2 once. This is a negative control for a build-specific regression.
- Twelve blocks per candidate cell: each permutation twice, with a fixed-seed
  randomized order and no equal depths adjacent across block boundaries.
  Within every block, each depth appears once. All measured requests are kept.
- Five candidate cells: motivating 512-input/256-output top-k20, 512/256 greedy
  (desktop common mode), 4,096/256 top-k20, 4,096/256 nucleus-only, and held-out
  512/128 top-k20 with alternate corpus and seed 7. Other cells use seed 42;
  stochastic temperature is 1 and top-p 0.95. The matrix is 198 measured
  requests, plus fixed excluded warmups and selector probes.
- One `mtp: true` mapping probe and one warmup per explicit depth per loaded
  worker are excluded before measurement. All requests use the same context
  and output budget within their cell. Verify outputs/counters repeat within
  each depth and match the prior reference when the workload matches.
- Record complete worker request time, useful/sample decode rates, prefill,
  request order, per-depth MTP work and stage timings, host observations,
  source/artifact hashes and any available clock observations. Do not wait for
  quiet, filter busy requests, add cooling pauses or retry unfavorable runs.
- The primary endpoint is one-versus-two complete-request latency at 512
  top-k20. Keep one as the artifact-wide recommendation only if the candidate
  improves median within-block latency at least 5%, wins at least 10/12 blocks,
  and the moving-block bootstrap interval for the primary latency change
  excludes zero. Use a predeclared circular block length two and 10,000 draws;
  its interval is conditional/descriptive, not proof of general independence.
- Require the same direction in the held-out cell and no >5% median regression
  in either complete latency or useful decode rate in the greedy/long-context
  guard cells: 512 greedy, 4,096 top-k20 and 4,096 nucleus. For each block,
  latency change is `T1/T2 - 1` (negative favors one) and useful decode change
  is `R1/R2 - 1` (positive favors one); take the median of those block ratios.
  The before-build primary contrast must not contradict the
  candidate direction. If these conditions fail, restore recommendation two
  on the branch while retaining explicit one-draft support and all evidence.

Do not compare the 9.93% off-control number directly with a two-control number,
subtract paired medians to infer another contrast, or replace an unfavorable
paired statistic with a favorable arm-median ratio. Report each estimand and
all observations. No installation, release or PR is part of this study.

## Why 9.93% and 2.65% differ

The 9.93% observation measures the benefit of enabling one-draft MTP over
ordinary decoding. Desktop already enables supported Lily MTP and had a
two-draft recommendation. Its relevant proposed change is two drafts to one.
The 9.93% figure cannot establish the benefit of that change.

The two observations also use different timing designs. The original study
interleaves requests in one loaded worker. The actual-default follow-up loads
four fresh workers in before/candidate/candidate/before order, then pairs
request positions across sessions. Those cross-session pairs are not adjacent
controlled requests. They retain operating-state drift in their contrast.

| Study | Comparison | Design | Median paired useful decode change | Median paired request time change |
| --- | --- | --- | ---: | ---: |
| Original motivating cell | one versus off | 8 pairs, one loaded worker | +9.93% | -8.32% |
| Initial depth follow-up, same workload | one versus two | 8 pairs, one loaded worker | +17.24% | -13.49% |
| Actual-default follow-up | candidate one versus before two | 16 positional pairs across four loaded sessions | +2.65% | -2.16% |

All three rows use 512 input tokens, 256 output tokens, temperature 1,
top-k 20, top-p 0.95, seed 42, identical artifact/corpus and no reused prefix.
Negative request change favors one. These are medians of paired ratios;
subtracting them or dividing displayed arm medians does not recover another
paired contrast. Counts do not imply the studies were contemporaneous.

The saved actual-default sessions show material drift with unchanged code and
unchanged speculative work: the before session's first four requests had
median decode 95.47 tok/s and its last four 80.98 tok/s, a 15.2% drop. The
other before session dropped 5.3%; candidate sessions dropped 8.5% and 11.5%.
The candidate also had a 4.22-second request, versus its overall median 3.18
seconds. All are retained. Pooled useful decode is only 1.42% higher for the
candidate, while the ratio of arm median rates is 8.89% higher. This sensitivity
is evidence that a practical gain was not established by that session design.

The earlier +17.24% cell was particularly busy: its 16 measurements failed the
legacy host screen and external CPU reached roughly 1,794%. The +2.65% study
passed 30/32 screens, but that screen does not establish equal GPU availability
or frequency. No contemporaneous GPU clock record exists for either result.
The data support session/environment sensitivity; they do not identify a
particular power, thermal or competing-GPU cause.

## What was checked for an implementation explanation

Source inspection and the completed selector controls found that `mtp: true`
and an explicit count converge on the same effective generation options.
Reset clears model state and MTP statistics; scratch capacity is preallocated
for the same supported maximum. The recommendation patch changes no decoder
kernel, quantization, sampler, image head, precision or prefix representation.
The same per-depth output hashes and MTP work counts recur across the studies.
There is no evidence of an accidentally different one-draft algorithm in the
candidate.

The older chat-worker harness set a dynamic request-budget environment value;
the raw-worker harness used native automatic budgeting. The exact older budget
is not recorded. That budget bounds transport/input/output admission rather
than reserving GPU work or selecting decoder kernels, and these requests fit
the minimum budget. It remains a reproduction limitation, with no source-backed
mechanism explaining the throughput gap. All new within-worker contrasts use
one common budgeting policy.

One versus two drafts is a real workload tradeoff. On the motivating sequence,
one uses 139 rounds, 139 proposals and 278 verified target rows; two uses 108
rounds, 214 proposals and 322 rows. Each produces 255 useful decode advances.
One reduces speculative work; two reduces round overhead and increases batch
width. Neither the higher acceptance count at two nor fewer rows at one alone
predicts complete-request performance across sampling and context lengths.

Greedy is the ordinary Playground fallback in
[`desktop.py`](../../tools/converter/stark_convert/desktop.py), and Chat
Completions defaults temperature to zero in
[`api_request.py`](../../tools/converter/stark_convert/api_request.py). API
clients can explicitly request stochastic sampling. A top-k-only benefit
therefore does not establish an improvement to default desktop chat.

## Outcome

The primary cell is complete. One versus two has **4.25% higher useful decode
rate and 3.58% lower complete-request time**, with 12/12 latency wins. Its
conditional block-bootstrap request-change interval is -5.28% to -2.58%.
Direction is consistent within this session, but the point estimate fails the
predeclared 5% material-effect requirement. The unchanged before binary also
favors one in all six blocks: +5.23% decode and -4.47% request time.

In the candidate's same balanced blocks, one versus off gives +6.08% useful
decode and -4.96% request time, while two versus off gives +1.27% decode and
-0.91% request time. This directly measures each contrast under one set of
conditions. It does not retroactively replace or invalidate the 9.93% result.

The 512-token greedy control is also complete: -0.36% useful decode and +0.46%
request time for one versus two, only 5/12 latency wins, with an interval
spanning zero. It provides no practical reason to change default desktop chat.

The 4,096-token top-k control gives +2.21% useful decode and -0.54% request time,
only 7/12 latency wins, with its latency interval spanning zero. The 4,096-token
nucleus control is worse at one draft in all 12 blocks: -6.01% useful decode
and +4.36% request time, with a request-change interval of +3.67% to +5.33%.
This repeats the direction of the earlier long-nucleus regression and fails
the registered useful-decode regression guard.

The held-out corpus/seed strongly favors one: +20.03% useful decode and -14.45%
request time, with 12/12 latency wins. This preserves a useful positive finding;
its gain does not cancel the adverse workload or change the primary endpoint.

Every candidate row below has 12 balanced three-arm blocks. Changes compare
one with two within each block, then take the median. The interval is the
predeclared conditional moving-block interval for request-time change. The
held-out row has 128 output tokens and seed 7; other rows have 256 and seed 42.

| Input / sampling | Useful decode change | Request time change | Latency wins | Request-change interval | Median absolute time saved |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 / top-k20, primary | +4.25% | -3.58% | 12/12 | -5.28% to -2.58% | 0.126 s |
| 512 / greedy | -0.36% | +0.46% | 5/12 | -0.74% to +0.88% | -0.016 s |
| 4,096 / top-k20 | +2.21% | -0.54% | 7/12 | -2.98% to +0.08% | 0.029 s |
| 4,096 / nucleus | -6.01% | +4.36% | 0/12 | +3.67% to +5.33% | -0.236 s |
| 512 / held-out top-k20 | +20.03% | -14.45% | 12/12 | -14.95% to -12.53% | 0.314 s |

Primary direction, win count, conditional interval, held-out direction and
before-build direction pass. The primary 5% request-benefit requirement and
long-nucleus 5% useful-decode regression guard fail. Therefore the predeclared
keep rule rejects the artifact-wide one-draft recommendation.

The main benefit is small but repeatable within these blocks; describing it
as merely random noise would be inaccurate. Its practical scope is limited:
about 126 ms saved on this 256-output-token primary workload, no established
gain for the default greedy route, and about 236 ms lost on long nucleus.
The held-out 314 ms saving shows that sequence-dependent gains can be larger.
These observations do not establish a representative traffic-weighted gain.

## Engineering judgment and disposition

| Action | Judgment | Reason |
| --- | --- | --- |
| Change the general enabled recommendation to one | Reject; restore two | No greedy benefit, modest primary request benefit, and consistent long-nucleus loss. Both registered promotion requirements above fail. |
| Keep explicit one-draft support | Worth retaining | Already supported and correct in the tested lifecycle; selected top-k workloads gain 4–20% useful decode here. |
| Explore one draft on representative top-k coding requests | Worth a bounded follow-up if that workload matters | The held-out result is large, but two short fixed sequences do not establish task-mix benefit. Measure actual context/output lengths and request latency before choosing a narrower policy. |
| Add automatic depth adaptation now | Defer | This study does not qualify a reliable selection rule or its switching/cache behavior. More code is unnecessary to resolve the current default decision. |
| Attribute the gap to a specific GPU/power problem | Unsupported | Historical clocks are absent and this run's clock observer had a schema mismatch. Timing drift alone cannot identify the limiter. |

The earlier implementation moved too quickly from an off-control result to a
recommendation change. Desktop already enabled two drafts. The actual change
needed its own product comparison before treating 9.93% as a reason to adopt
one. Functional validation was successful; it did not establish the speed case.

The source-specific recommendation override, extra model field and its helper
test are removed. Final engine, generator and production Python sources match
study base `f9c9720c`; Lily again reports `2.min(max_drafts)`. Explicit counts,
raw omitted-selector behavior, saved-off preferences, image behavior and
existing prefix restoration remain as before. The generic host regression
test for consuming a one-draft capability is retained. Commit `86cb55ca` and
its sealed candidate remain historical experiment identities, not the final
recommendation or an installed build.

## Evidence, validation and limits

The [aggregate data](data/lily-mtp-default-reconciliation-2026-11-15.json)
records all measured timings, work counts, host screens, contrasts, order plans,
source/corpus/artifact/binary hashes and the explicit decision checks.
[Exact as-run tooling](../archive/experiments/2026-11-15-lily-mtp-reconciliation/README.md)
preserves the initial failed observer/aggregation and the continuation separately.
The artifact and both sealed worker identities are those in
[the original recommendation study](2026-11-15-flashnext-lily-one-draft-default.md#identity).

All six cells pass per-depth output-hash and round/proposal/acceptance/verified-row
repeatability. The primary cells match across before/candidate at depths zero,
one and two; their hashes and work also match the earlier primary receipts.
All five candidate workloads additionally match the prior private depth-study
receipts at depths one and two: ten matching output hashes and work tuples,
with identical corpus, input, artifact, sampling and token-budget identities.
Every measured request reports zero prefix reuse and uses the retained images
and MTP head. Each cell's artifact-file and executable guards pass. The
publisher verifies the recovered 18 native records against the original
receipt and counts exactly 198 measurements plus 24 fixed setup requests.

After restoring the source, generator protocol checks pass 98/98 under Rust
1.98.1 with `--no-default-features`; focused desktop MTP checks pass 60/60 with
the sealed template helper on PATH. Repository policy, pinned-nightly Rust
formatting, publication assertions and archived Python compilation pass. The
initial Rust check commands found no `cargo` on the shell PATH and executed
no checks; both pass using the absolute rustup cargo path. The sealed before
worker executed this study's image-loaded selector/depth controls; no new
native decoder path was introduced by restoring its identical source.

The legacy host screen passes 190/198 measurements. There are no missing host
observations; maximum observed external CPU is 262.52%, maximum coarse thermal
state is 1, pageout increments total three and swapout increments zero. All
measurements are retained regardless of eligibility. GPU operating state,
exclusive device use and peak process footprint are unknown.

These are repeated fixed sequences on one machine, with no reused prefix.
There is no new model-quality, image-answer, long-output, near-capacity,
cross-hardware, app/API end-to-end or representative coding-task performance
claim. The source-bound earlier selector/image checks and prefix exactness
results remain separate evidence, without transferring qualification to other
artifacts or precision policies.

## Execution setup failure and continuation

The first run completed all 18 before-build measurements. Its ancillary clock
sampler accepted zero samples: the sealed `macmon` executable emits ordinary
metrics but omits the four source-interval fields required by the existing
clock helper. Summarizing an empty frequency list then raised `StatisticsError`.
Native timing, output hashes and work counters were already saved. This is an
observer/aggregation failure, not a failed generation result.

The original harness and receipts are preserved. A separately hashed
continuation makes absent clock summaries null, reprocesses the same 18
measurements, verifies artifact/binary identity again, and executes only the
five remaining candidate cells. It starts no clock observer. The workload,
order, endpoints and decision rule remain unchanged. No completed request is
rerun or excluded, and no GPU frequency is inferred from ordinary timestamps.
