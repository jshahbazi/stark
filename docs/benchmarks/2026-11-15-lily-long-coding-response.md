# Lily long coding response comparison

Date: 2026-11-15. Attempt: `20261115-LILY-LONG-01`.

## Question and fixed plan

Compare the previous two-draft recommendation with the new sampling-specific
automatic one-draft policy on the user's natural tic-tac-toe Python coding
request. The example is a short input asking for a potentially long program;
this study tests longer completion generation, not long-context prefill.
The request is supplied only in memory. Save its hash and the rendered-input
token count/hash, without storing prompt, response or token arrays.

Preregistered before native execution:

- Reuse the exact pinned Lily Q4-XL artifact, image tower/MTP head loaded,
  unchanged precision and sealed before/candidate workers from the
  [sampling-policy qualification](2026-11-15-lily-sampling-mtp-policy.md).
  Before SHA-256 is
  `61b9c8cc724407573169eb2abad7f908e4943eead7ac3122eeec3ef841a48640`;
  candidate is
  `e1e432fcd5d490c8ab900be23c7cbc3855fc4515d75a38089bac672a85e9be22`.
  Source candidate is `1b0f5ed0`.
- Use the real model chat template, its default system message when present,
  and thinking disabled. Use identical rendered inputs for every request.
  Native EOS is honored; add no task-specific stop strings or protocol stops.
- Context is 8,192, response ceiling 2,048, seed 42; stochastic temperature 1,
  top-k 20, top-p 0.95 and no min-p filter. Both workers receive `mtp: true`;
  verify actual selected depths two before and one after. Preserve the mixed
  Q4/Q8 weights, BF16 activations, Q8 KV and F32 reductions/recurrent/readouts.
- Four fresh workers run sequentially in before/candidate/candidate/before
  order. Each has one excluded 64-token warmup and two measured full responses:
  eight measurements, four warmups and four positional before/candidate pairs.
  Prefix reuse stays disabled and must report zero. All observations are kept
  under concurrent host work; no quiet admission, cooling, retries or timing
  exclusions.
- Record actual emitted/sampled tokens, natural EOS or length termination,
  complete request time, prefill/decode time, useful decode throughput, MTP work,
  host observations, output hashes and source/artifact/tool identities.
  Useful decode excludes the first emitted token. Preserve capped, empty or
  invalid-code outcomes without treating them as completed coding tasks.
- Different draft depths use different valid stochastic draw schedules and
  can generate different programs/lengths. Compare both useful token rate and
  complete-task latency, report paired ratios and both session-half medians,
  and do not attribute a shorter answer solely to faster inference. Repeated
  fixed-seed outputs should match within each depth; cross-depth identity is
  not required. This single request does not qualify general coding quality.
- Keep generated programs only in memory. After all generation timings and
  workers finish, check each unique response once: code extraction, Python
  syntax, and isolated execution checks where the host sandbox is verified.
  Never run generated code without isolation. Report unsupported GUI/dependency
  or unavailable runtime validation separately from a proven game failure.
  No generated code, program stdout or exception dumps enter saved artifacts.

No native code, installed app or sampling defaults are changed by this study.
This is a new source-bound test of the existing implementation; it does not
replace the earlier 128/256-token workload results.

## State and outcome

All eight measured responses and four fixed warmups completed. Before selects
two drafts; candidate selects one. Every measured response ends naturally at
EOS, passes count/timing/depth checks and reports zero prefix reuse. The exact
request renders to 24 input tokens; this is a long-response test.

**This example favors two drafts.** One draft has lower useful decode rate
and longer response time in all four positional comparisons. The candidate
also emits 22.72% more tokens, so its extra total time is not a comparison of
equal amounts of generated work. Both programs pass Python syntax checks;
gameplay execution and correctness are untested.

## Observed responses

Each arm has four measured responses across two worker sessions. Repeated
outputs, emitted lengths and MTP work are identical within an arm. The two
arms produce different sampled programs. Displayed throughput and time are
arm medians; they are not the paired-change statistic.

| Metric | Before: two drafts | Candidate: one draft |
| --- | ---: | ---: |
| Emitted tokens | 1,109 | 1,361 |
| Sampled tokens, including EOS | 1,110 | 1,362 |
| Useful decode tokens/s, median | 101.17 | 91.47 |
| Useful decode tokens/s, observed range | 93.90–107.34 | 83.02–94.23 |
| Complete request seconds, median | 11.067 | 14.994 |
| Complete request seconds, observed range | 10.437–11.920 | 14.547–16.497 |
| Prefill seconds, median | 0.11183 | 0.11240 |
| Natural EOS finishes | 4/4 | 4/4 |
| Python syntax-valid unique programs | 1/1 | 1/1 |

The median paired useful-decode change is **-11.35%** and paired request-time
change is **+38.06%**, with zero candidate wins in four pairs. Both session
halves favor two: median paired decode changes -11.66% and -9.84%; request
changes +38.55% and +35.81%. These four positional pairs are descriptive,
not independent task-quality observations.

Neither arm hits the 2,048-token response ceiling. Native generation and
complete-request times are nearly identical at the displayed precision; rendering,
model loading, warmup, separate input tokenization and quality checks are
outside the request timer. The `generate` request uses the identical rendered
prompt and includes the worker's ordinary prompt processing.

## Work and interpretation

Before has 394 MTP rounds, 788 proposals, 717 accepted drafts and 1,182 verified
rows per response. Candidate has 696 rounds, 696 proposals, 666 accepted drafts
and 1,392 verified rows. The response lengths differ, so raw work totals cannot
be treated as equal-output efficiency measures.

Normalized by emitted tokens after the first, two uses about 0.356 rounds and
1.067 verified rows per useful token; one uses 0.512 rounds and 1.024 rows.
One saves roughly 4% of verified rows per useful token while requiring roughly
44% more rounds. The two-draft proposal acceptance rate is about 91%, much
higher than the earlier favorable held-out case's 51%. Those counters are
consistent with two drafts being useful on this predictable coding sequence.
Different sampled programs and unknown GPU operating state prevent attributing
the entire measured rate change to that mechanism alone.

The previously confirmed 20.47% gain remains valid for its selected workload.
This natural coding request is a material counterexample to using the sampler
profile alone as a generally optimal draft-depth selector. For this workload,
two is the better measured setting. The existing one-draft branch policy needs
broader long-response qualification or a validated narrower/adaptive rule
before claiming a general top-k improvement. No policy implementation change
is made by this test.

## Quality and setup limits

Both unique responses contain syntax-valid Python and no detected GUI import.
The checker performs no generated-code execution. Sandbox setup probes could
not qualify the intended restrictive runtime policy; runtime availability,
normal exit, win/draw handling and gameplay correctness remain unknown. An
unavailable execution check is not a failed game test or evidence that the
programs provide equivalent features.

The first harness invocation encountered closed plain-pipe stdin before a
prompt was supplied, an output directory was created or any worker started.
Its source and a zero-native-request preflight record are retained. The actual
run uses a tool PTY with echo disabled before reading the JSON prompt, keeping
the instruction out of terminal output and process arguments. There are no
native retries or excluded measurements.

The same artifact, 63-file payload/preparation guard plus nine pinned ancillary
entries, worker binaries, config, codec/template/helper identities and precision
as the preceding study are retained. Prompts, generated programs and token
arrays exist only in memory. Public artifacts contain counts, timing, hashes,
host observations and syntax-check flags.

The [aggregate record](data/lily-long-coding-response-2026-11-15.json) contains
all eight measured responses and four warmups as content-free observations,
the four comparisons, per-arm summaries, host activity and 72-file postrun
guard. All eight observations pass the legacy host screen; none is filtered.
Observed pageouts and swapouts are zero, coarse thermal state is zero, and
external CPU peaks at 106.9%. Source clocks and GPU exclusivity are unknown.
The [as-run sources](../archive/experiments/2026-11-15-lily-long-responses/README.md)
preserve the successful runner, initial stdin preflight, syntax checker and
publication tool. This is documentation/evidence on the existing branch;
production code, installed app and release state are unchanged.
