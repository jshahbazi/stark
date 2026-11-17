# Lily long business-planning response comparison

Date: 2026-11-15. Attempt: `20261115-LILY-SLACK-LONG-01`.

## Question and preregistered plan

Compare automatic two-draft and one-draft decoding on the user's Slack
competitor business-planning request. This tests a potentially long response
from a short input; it does not test long-context prefill. The request text is
kept only in harness memory. The private receipt may retain its hash and the
rendered input token count/hash; this document contains no prompt or generated
response text.

The plan fixes these conditions before native execution:

- Use the pinned `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` artifact at
  revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, with its retained image
  tower and trained MTP head loaded. Keep the mixed Q4/Q8 weights, BF16
  activations, Q8 KV and F32 reductions, recurrent state and readouts. Use the
  same source, config, precision and image-loaded worker modes for both builds
  as the [tic-tac-toe long-response comparison](2026-11-15-lily-long-coding-response.md)
  and the [sampling-specific MTP qualification](2026-11-15-lily-sampling-mtp-policy.md).
- Compare the sealed production-source baseline with the policy candidate
  built from source `1b0f5ed0`. The candidate changes automatic draft
  selection only; production source `1b0f5ed0` remains unchanged. It does not
  change sampling, verification arithmetic, weights, precision or image
  support. Record exact executable, source, artifact and preparation
  identities in the receipt.
- Render the real user request with the model's chat template and the same
  default system-message and thinking settings in every request. Use context
  8,192 and `max_new_tokens=4096`. Use stochastic temperature 1, top-k 20,
  top-p 0.95, no min-p filter, and seed 42. Honor native EOS and add no
  task-specific stop strings.
- Send `mtp: true` with automatic selection to both builds. Require actual
  draft depth 2 for the before build and 1 for the candidate. Verify the
  selected depth and token accounting for every warmup and measured request.
- Run four fresh workers in before/candidate/candidate/before order. Each
  worker performs one excluded 64-token warmup and two measured full responses:
  eight measured requests, four warmups and four positional before/candidate
  pairs. Use fresh workers sequentially. Disable prefix reuse and require zero
  reused tokens.
- Retain all requests. Do not impose quiet-host admission, cooling pauses,
  retries, timing exclusions or result filtering. Record available host
  observations descriptively, with monitor gaps or sampling failures visible.
- Record actual emitted and sampled token counts, EOS versus token-ceiling
  termination, complete request time, prefill and decode time, useful decode
  throughput, draft counters, output hashes and host observations. Compare
  throughput as well as complete-request latency. Report four positional
  ratios and both two-pair session-half medians; do not substitute a ratio of
  arm medians for paired changes.
- Stochastic runs at different draft depths can produce different response
  lengths. Require repeated fixed-seed outputs and counts to be stable within
  each arm; cross-arm output identity is not required. Report response lengths
  alongside latency and do not interpret a shorter response as a causal
  compression benefit. The 4,096-token ceiling can still cap a response; keep
  a capped result visibly capped and do not describe it as a completed task.
- Do not grade business-plan truth, market claims, competitor research or
  source quality. Optional structural counts (such as word and heading counts)
  may describe the emitted responses, but cannot establish quality or task
  completion. No generated response, prompt text or token arrays are written to
  public artifacts.

## Interpretation limits

This is one fixed user request and one pinned artifact/configuration. The two
draft depths can generate different stochastic continuations, so end-to-end
request time includes both inference speed and the amount of sampled work.
Useful decode rate provides a length-aware view, but neither metric alone
isolates a universal causal effect of draft depth. The test does not qualify
general business-planning quality, truthfulness, research capability,
representative user traffic, other sampling profiles or other artifacts.

The earlier tic-tac-toe long-response result is a separate task and must remain
separately reported. The broader sampling-specific qualification remains bound
to its own selected workloads and does not predetermine this request's outcome.

## Observed result

The eight-request run completed with the preregistered automatic depths:
before selected two drafts and candidate selected one. All count, timing and
depth checks pass. All eight responses ended at natural EOS; none reached the
4,096-token ceiling. The identical rendered input was 20 tokens. Prefix reuse
was disabled and measured reuse was zero. Repeated outputs and MTP work counts
match within each arm; the two arms produced distinct stochastic responses.

| Metric | Before: two drafts | Candidate: one draft |
| --- | ---: | ---: |
| Emitted tokens, median [min, max] | 1,958 [1,958, 1,958] | 1,902 [1,902, 1,902] |
| Sampled tokens including EOS | 1,959 | 1,903 |
| Useful decode tokens/s, median [min, max] | 81.91 [79.26, 84.01] | 82.61 [74.95, 84.90] |
| Prefill seconds, median [min, max] | 0.11334 [0.11043, 0.11459] | 0.11259 [0.11015, 0.12559] |
| Complete request seconds, median [min, max] | 24.021 [23.409, 24.811] | 23.138 [22.505, 25.495] |
| Natural EOS / length cap | 4 / 0 | 4 / 0 |
| Words / Markdown headings, per repeated response | 1,049 / 14 | 996 / 17 |

The candidate emitted 2.86% fewer tokens. Paired complete-request time changed
by -3.443%, +3.867%, -4.219% and -4.585% for the four positional pairs. The
median paired change is -3.831%, with 3/4 candidate latency wins. Useful decode
rate changed by +0.617%, -6.446%, +1.423% and +1.825%; its median paired change
is +1.020%, also with 3/4 wins. Session-half medians of paired changes are:

| Session half | Request-time change | Useful decode-rate change |
| --- | ---: | ---: |
| First two pairs | +0.212% | -2.915% |
| Last two pairs | -4.402% | +1.624% |

The token-rate difference is small relative to the observed range. The wall
time result is also a comparison of different stochastic completion lengths,
not a fixed-work latency measurement or evidence that the candidate compressed
the answer. These numbers do not demonstrate a 20% speedup.

The private aggregate records only response hashes and structural counts; the
business-plan content, truthfulness, competitor research, and task completion
were not assessed. Candidate and before word and heading counts describe
structure only.

The 72-file model identity guard passes (63 files checked against the earlier
manifest and nine ancillary files checked separately). All observations are
retained with no quiet-host filter, retry or timing exclusion. There were no
monitor failures, pageouts or swapouts; the coarse thermal maximum was 1.
External CPU peaked at 447.4% during a before request and 168.7% during another
before request; those two host screens were marked ineligible. Candidate
request peaks were at most 75.2%, and all candidate screens were eligible.
This host imbalance can affect the small latency result, and the unfiltered
observations do not isolate a causal draft-depth effect.

This planning response is a useful counterexample to a broad claim that one
draft is preferable for every long top-k response, but it does not establish
that two drafts always win. The [tic-tac-toe coding response](2026-11-15-lily-long-coding-response.md)
measured a materially slower one-draft completion with substantially more
emitted tokens. The separate [sampling-policy qualification](2026-11-15-lily-sampling-mtp-policy.md)
retains a positive result for its specific held-out workload. Together these
are task-specific observations, not a general selector rule or quality
comparison.

The [published aggregate](data/lily-long-planning-response-2026-11-15.json)
contains content-free measurements and hashes. The [as-run source archive](../archive/experiments/2026-11-15-lily-long-responses/README.md)
records the runner and private-artifact handling. Production engine/host code
matches `1b0f5ed0`; this study changes no production code, installed build or release.
