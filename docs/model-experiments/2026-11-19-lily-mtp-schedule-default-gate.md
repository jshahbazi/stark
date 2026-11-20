# 20261119-LILYADAPT-01: Lily MTP schedule, matched-window gate for a default

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | `20261119-LILYADAPT-01` / 2026-11-19 / 2026-11-19 / orchestrating session for the repository owner |
| Kind | Performance and sampling (schedule selection); exactness controls |
| Attempt state | Completed |
| Result and acceptance criterion | Passed: under the decision rule below, written before any result of this attempt was analyzed, the adaptive policy with prompt lookup replaces two fixed drafts |
| Evidence scope | End-to-end timing and output hashes on the full model |
| Promotion decision | Promote the adaptive policy with lookup as the default schedule of MTP-enabled text requests on the pinned Lily Q4-XL artifact only |
| Disposition | Measurement only; the change is `20261120-LILYSCHED-01` |
| Predecessor / related attempts | [20261119-LILYDEPTH-01](2026-11-19-lily-mtp-depth-policy.md), [20261119-LILYMERGE-01](2026-11-19-lily-mtp-merged-quiet-timing.md); prefix reuse with the selectors is `20261119-LILYPREFIX-01` |

## Hypothesis and change

- Problem: the desktop app enables MTP for this model at two fixed drafts.
  LILYMERGE-01 measured the opt-in adaptive policy and prompt lookup almost
  only with greedy decoding at 4,096 input tokens, with repeats that reused one
  input and one seed. Sampled decoding, prose, longer contexts and the two
  selectors together were not established, so no schedule could be called
  best.
- Question: which schedule should an MTP-enabled text request on the pinned
  Lily Q4-XL artifact use when it names none: one, two or three fixed drafts,
  the adaptive policy with a limit of three, or the adaptive policy with
  prompt lookup.
- No engine change in this attempt. The candidate worker is the committed
  build of `a246039f`.

### Decision rule (registered before analysis)

Written after the first three measured requests of the first session had been
glanced at to confirm the harness ran (one prose window, greedy off, one and
two drafts), and before any other result existed.

A cell is one corpus, one sampling mode and one input length: prose or code,
greedy or sampled, 4,096, 16,384 or 32,768 tokens, twelve cells. Within a cell
every arm sees the same windows and seeds. An arm's cell statistic is the
median over windows of its decode rate divided by the MTP-off arm's rate in
the same mode on the same window.

Candidates: one fixed draft, two fixed drafts (the app's present default), the
adaptive policy with a limit of three, and the adaptive policy with a limit of
three and prompt lookup. Three fixed drafts are measured greedy only and are
not a candidate: LILYMERGE-01 already measured that depth below MTP off with
sampled decoding.

Eligibility, all required:

1. Every session completes with no request error.
2. In every window the greedy arms share one output hash, so greedy output
   does not depend on depth, policy or lookup.
3. Every repeated request reproduces the output hash of its first run.
4. For a candidate that uses a selector, `20261119-LILYPREFIX-01` passes.

Choice among eligible candidates:

1. Score is the geometric mean of the twelve cell statistics, equally
   weighted.
2. No-harm guard: a candidate with any cell statistic below 0.97 is dropped.
3. The highest score wins. The default moves away from two fixed drafts only
   if the winner's score is at least 2 percent above that of two fixed drafts
   and no cell of the winner is more than 5 percent below the same cell of two
   fixed drafts. Otherwise two fixed drafts stay.
4. Prompt lookup: these corpora seldom repeat their own input, so they measure
   lookup's overhead rather than its benefit. If the adaptive policy wins,
   lookup is on when its score is within 1 percent of the adaptive policy
   alone, because LILYMERGE-01 measured 7 to 10 percent on copied input.
5. If no candidate passes, nothing changes.

### Amendments after results began to arrive

Both were written before the results they affect were seen. Neither changes
the rule above.

1. Window validity, after the 4,096-token sessions. The last window of each
   corpus ended where the text ends and the model stopped at once, and some
   sampled requests stop early elsewhere. A window counts for a sampling mode
   only when every arm of that mode finished by length in it; otherwise the
   window is dropped for every arm of that mode. Later sessions leave 4,096
   corpus tokens unused after the last window.
2. Supplement, after the first 16,384 and 32,768-token sessions and before
   its own results. The sampled long-context cells had three valid windows or
   fewer, and sampled prose at 32,768 tokens had one, because sampled requests
   stopped early in the other two. The supplement adds six windows to each of
   the four sampled long-context cells (eight for sampled prose at 32,768),
   sampled arms only, seeds from 142, new window positions. Every sampled
   long-context cell is supplemented alike, not only the unfavourable ones. A
   cell's statistic is then the median over every valid window of both
   sessions. Standing before the supplement: scores 1.2185 adaptive, 1.2184
   adaptive with lookup, 1.2108 one fixed draft, 1.1815 two fixed drafts; two
   fixed drafts fail the no-harm guard on prose at 4,096 tokens (0.956 greedy,
   0.952 sampled); adaptive with lookup is more than 5 percent below two fixed
   drafts in sampled code at 16,384 (1.327 against 1.410, three windows) and
   sampled prose at 32,768 (1.086 against 1.273, one window), so under the
   rule the default would not move.
3. Limit check, after the pooled rule selected the adaptive policy with
   lookup and before its own results. The candidates above ran with a draft
   limit of three. The desktop app and its local API request the worker's
   recommended two drafts, and the app's paged-residency memory profile for
   this artifact is bounded at two drafts, so the app's default will be the
   adaptive policy with lookup at a limit of two. A further set of sessions
   measures that arm against MTP off, two fixed drafts and the limit of
   three, greedy and sampled, both corpora, at 4,096 (six windows) and 16,384
   tokens (four windows), seeds from 242. The same rule is applied to the
   limit-two arm over those eight cells: no cell below 0.97, no cell more
   than 5 percent below two fixed drafts, score at least 2 percent above two
   fixed drafts. If it fails, the default does not ship at a limit of two.

## Artifact, source and environment

| Field | Value |
| --- | --- |
| Model | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, installed copy; payload not rehashed in this attempt |
| Worker | `stark-generate` SHA-256 `5abca56f752489dde82d60b42e29ddaf36542962a83770bfba27a52633a3e26b`, built from the tree committed as `a246039f`, Rust 1.98.1, release, image tower loaded |
| Hardware | Apple M5 Max, 128 GiB |
| Host state | Not screened or quiet-admitted. One model process at a time under a local lock; load averages recorded at the start and end of each session |

## Workload and controls

- Corpora: a public-domain English novel (text SHA-256
  `0a0ead8d7e5be2a1ffb4a929528ce245412af22d7848c723b43b2d91105e63f0`, 174,832
  tokens) and the frozen public code corpus of SUSHI-02 (text SHA-256 prefix
  `f23f98fc`). Raw continuation of token windows, no chat template.
- Each repeat takes its own evenly spaced window and its own seed (42 plus the
  repeat index). Every arm of a repeat gets that input and seed. Arm order
  alternates forward and reverse between repeats.
- Greedy arms decode 512 tokens; sampled arms 256 at temperature 1, top-k 20,
  top-p 0.95.
- Arms: MTP off, 1, 2 and 3 fixed drafts, adaptive with limit 3, adaptive
  with limit 3 and lookup (greedy); off, 1 and 2 fixed drafts, adaptive,
  adaptive with lookup (sampled).
- 4,096 tokens: six windows per corpus. 16,384 and 32,768: three windows.
- Fresh requests: no prefix cache, zero reused input tokens expected.
- After the main loop every arm of the first window runs once more; only its
  output hash is compared.
- Decode rate is sampled tokens less one over decode seconds; prefill and load
  are excluded.

## Attempt history and failures

| Trial | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| a1, 4,096 tokens | Eleven arms, six windows, prose and code, 79 requests each | No request error. The sixth window of each corpus ended at the end of the text: every prose arm stopped at the first token, and three sampled code arms stopped early | Valid. Sixth window dropped for prose (both modes) and for sampled code under amendment 1 |
| First long job | 16,384 and 32,768 tokens with the a1 window placement | Stopped by the operator one minute into its first session to move the last window away from the end of the text | Aborted, no result, files removed |
| a2, 16,384 and 32,768 tokens | Eleven arms, three windows, 46 requests per session | No request error. Sampled prose at 32,768 lost two of three windows to four early stops (4 to 83 tokens) | Valid. Sampled long-context cells too thin; supplemented under amendment 2 |
| a3, supplement | Sampled arms only, six windows per cell (eight for prose at 32,768), seeds from 142 | No request error. One sampled code window at 32,768 dropped (all five arms stopped early) | Valid, pooled with a2 |
| a4, limit check | Off, two fixed drafts, adaptive with lookup at limits of two and three; six windows at 4,096 and four at 16,384, seeds from 242 | No request error, no early stop, no window dropped | Valid; evaluated on its own eight cells under amendment 3 |

648 requests in fourteen sessions, no error, 23 early stops.

## Correctness and lifecycle

- Greedy output is independent of the schedule: in every window of the six
  sessions with greedy arms, off, one, two and three fixed drafts, adaptive
  and adaptive with lookup share one output hash.
- Run-to-run reproduction: 66 of 66 repeated requests (every arm of the first
  window of the a1 and a2 sessions, greedy and sampled) reproduced the output
  hash of their first run.
- Prefix reuse with the selectors: passed in `20261119-LILYPREFIX-01`, at a
  limit of two drafts, 8,192 context tokens and 16-token outputs.
- Every request was fresh: zero reused input tokens in every session.
- Not tested here: image requests, cancellation, failure recovery, memory.

## Performance and memory

Decode rate relative to MTP off, median over valid windows, pooled over
sessions. MTP off decoded at 72.3 to 75.9 tokens per second.

| Corpus, mode, input tokens | Valid windows (dropped) | 1 fixed | 2 fixed | Adaptive | Adaptive with lookup | 3 fixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Code, greedy, 4,096 | 6 (0) | 1.288 | 1.419 | 1.421 | 1.415 | 1.421 |
| Code, greedy, 16,384 | 3 (0) | 1.303 | 1.406 | 1.435 | 1.379 | 1.403 |
| Code, greedy, 32,768 | 3 (0) | 1.293 | 1.390 | 1.378 | 1.376 | 1.386 |
| Code, sampled, 4,096 | 5 (1) | 1.219 | 1.261 | 1.212 | 1.234 | not run |
| Code, sampled, 16,384 | 9 (0) | 1.280 | 1.343 | 1.327 | 1.327 | not run |
| Code, sampled, 32,768 | 8 (1) | 1.267 | 1.340 | 1.341 | 1.396 | not run |
| Prose, greedy, 4,096 | 5 (1) | 1.070 | 0.956 | 1.061 | 1.061 | 0.827 |
| Prose, greedy, 16,384 | 3 (0) | 1.103 | 0.999 | 1.099 | 1.096 | 0.863 |
| Prose, greedy, 32,768 | 3 (0) | 1.106 | 0.984 | 1.093 | 1.097 | 0.818 |
| Prose, sampled, 4,096 | 5 (1) | 1.138 | 0.952 | 1.060 | 1.084 | not run |
| Prose, sampled, 16,384 | 9 (0) | 1.188 | 1.011 | 1.128 | 1.136 | not run |
| Prose, sampled, 32,768 | 9 (2) | 1.095 | 0.988 | 1.120 | 1.104 | not run |

| Candidate | Score | Lowest cell against off | No-harm guard | Lowest cell against 2 fixed | Cells more than 5 percent below 2 fixed |
| --- | ---: | ---: | --- | ---: | ---: |
| 1 fixed draft | 1.1928 | 1.070 | passes | 0.907 | 4 |
| 2 fixed drafts | 1.1549 | 0.952 | fails | 1.000 | 0 |
| Adaptive, limit 3 | 1.2149 | 1.060 | passes | 0.961 | 0 |
| Adaptive with lookup, limit 3 | 1.2180 | 1.061 | passes | 0.978 | 0 |

- Input length changed little: the cost table measured at 4,096 tokens chose
  the same depths at 16,384 and 32,768.
- On prose the planner ran one draft in 72 to 89 percent of head rounds, two
  drafts in 7 to 26 percent and zero in 1 to 13 percent; it never chose three.
  On code it ran two or three drafts in 82 to 99 percent of head rounds,
  except sampled code at 4,096 tokens, where one draft took 46 percent.
- One fixed draft beat the adaptive policy on sampled prose at 4,096 and
  16,384 tokens (1.138 and 1.188 against 1.060 and 1.128) and lost to it on
  every code cell but sampled code at 4,096.
- Lookup ran in 0 to 15 percent of rounds on these corpora; its score is 0.3
  percent above the adaptive policy alone.
- Memory, first-token time and prefill were not compared between arms.

### Limit check (amendment 3)

Same statistic, a4 sessions only.

| Corpus, mode, input tokens | Valid windows | 2 fixed | Adaptive with lookup, limit 2 | Adaptive with lookup, limit 3 |
| --- | ---: | ---: | ---: | ---: |
| Code, greedy, 4,096 | 6 | 1.391 | 1.383 | 1.380 |
| Code, greedy, 16,384 | 4 | 1.348 | 1.346 | 1.331 |
| Code, sampled, 4,096 | 6 | 1.269 | 1.216 | 1.315 |
| Code, sampled, 16,384 | 4 | 1.316 | 1.306 | 1.346 |
| Prose, greedy, 4,096 | 6 | 0.942 | 1.060 | 1.049 |
| Prose, greedy, 16,384 | 4 | 0.996 | 1.087 | 1.087 |
| Prose, sampled, 4,096 | 6 | 0.954 | 1.086 | 1.087 |
| Prose, sampled, 16,384 | 4 | 0.971 | 1.109 | 1.108 |
| Score | | 1.1333 | 1.1929 | 1.2056 |

- The limit of two passes the rule: lowest cell 1.060 against MTP off, at
  most 4.2 percent below two fixed drafts (sampled code at 4,096), score 5.3
  percent above two fixed drafts.
- The limit of three scores 1.1 percent above the limit of two; the
  difference is in sampled code (1.315 against 1.216, and 1.346 against
  1.306). On prose the two limits made the same planner choices.
- Greedy arms shared one output hash in every window of the four sessions.

## Evidence, decision and limits

- Public aggregate:
  [`lily-mtp-schedule-default-gate-2026-11-19.json`](../benchmarks/data/lily-mtp-schedule-default-gate-2026-11-19.json),
  with the SHA-256 of every private session receipt.
- Eligibility: all four conditions hold for every candidate.
- Rule outcome: the adaptive policy with lookup has the highest score, is 5.5
  percent above two fixed drafts, passes the no-harm guard, and is at most 2.2
  percent below two fixed drafts in any cell. The default moves to the
  adaptive policy with prompt lookup. Two fixed drafts, the app's default
  until now, fail the no-harm guard.
- Limit: the app requests two drafts and its paged-residency memory profile
  for this artifact is bounded at two, so the app's default is the adaptive
  policy with lookup at a limit of two, which passes the same rule. The limit
  of three is about 1 percent better overall and remains what a raw
  `"mtp": true` request gets. Raising the app's bound needs a memory
  qualification and was not attempted.
- The supplement changed the outcome. Before it, with one valid window for
  sampled prose at 32,768 and three for sampled code at 16,384, the same
  candidate was more than 5 percent below two fixed drafts in those two cells
  and the default would not have moved. With nine valid windows each, both
  cells are within the bound (1.104 against 0.988, and 1.327 against 1.343).
- Scope: pinned Lily Q4-XL `0f94842e`, image tower loaded, this host, raw
  continuation of two public corpora, greedy and one sampled setting
  (temperature 1, top-k 20, top-p 0.95). Not established: chat-template
  prompts, other sampled settings, other artifacts, other hosts.
- The implementation and its own gates are `20261120-LILYSCHED-01`.
- [Performance playbook](../PERFORMANCE_PLAYBOOK.md): updated under draft
  depth, adaptive depth, prompt lookup, measuring and the open ideas.

Keep prompts, completions, token vectors, probabilities, credentials, exception
dumps and model/private evaluation files out of this document.

## Follow-up history

None yet.
