# 20261119-LILYPREFIX-01: Lily MTP scheduling selectors under generation prefix reuse

Draft record. Registered and executed 2026-11-19. Owner: prefix-reuse validation
workstream (agent). State: completed. Kind: prefix reuse, correctness. Not
committed or installed.

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | 20261119-LILYPREFIX-01 / 2026-11-19 / 2026-11-19 / prefix-reuse validation workstream |
| Kind | Prefix reuse, correctness |
| Attempt state | Completed (one exactness run) |
| Result and acceptance criterion | Passed: 67 of 67 checks and 6 of 6 invalidation paths in run `prefix-1`. Criterion fixed before the run: every selector case identical to the same candidate without prefix reuse with the same options (token IDs, text, parsed tool calls), the reply acknowledges the selector and its integer schedule counters equal the uncached run; greedy selector cases also equal the fixed-schedule greedy output of the same turn on the candidate and on the baseline worker; every reuse count matches the documented rule, whose key ignores the selectors; all pre-existing checks pass unchanged |
| Evidence scope | Full-model numerical (output hashes) and lifecycle, one artifact, one synthetic session, 16-token outputs |
| Promotion decision | Qualifies the documented "prefix-reuse key is unchanged" statement for the tested scope only; no code promotion |
| Disposition | Validator extension uncommitted on the working tree; no engine change |
| Predecessor / related attempts / governing issue / PR | 20261119-LILYDEPTH-01 (selectors), 20261119-LILYMERGE-01 (merged build); prior Lily validator recipe in [the one-draft study](../benchmarks/2026-11-15-flashnext-lily-one-draft-default.md); no PR |

## Hypothesis and change

- Problem: [Generation](../GENERATION.md#mtp-round-scheduling) states that the
  prefix-reuse key is unchanged by `mtp_policy` and `mtp_lookup` because the
  retained snapshot is captured during prompt prefill. That was not validated on
  the real model.
- Implementation check (source reading, not a run): the hybrid worker key is
  `model-instance+context-tokens+cache-scope+mtp-enabled`; restore compares only
  MTP on/off. The lookup index is built from the full input, not only the
  suffix. So a fixed-to-selector switch at the same draft count is expected to
  reuse, and a plain-to-selector switch is not. The documentation and the
  implementation agree.
- Change: `scripts/validate_generation_prefix_cache.py` gains opt-in
  `--mtp-policy adaptive` and `--mtp-lookup`. With both, the selector block runs
  once each for adaptive, lookup and adaptive with lookup. Each block: after a
  plain request, warm repeat, seeded stochastic repeat (seed 17), fixed schedule
  after the selector, selector after the fixed schedule, a shorter prompt, regrow,
  stochastic regrow (seed 23), plain after, plain warm. Without the flags the
  plan, requests and record schema are unchanged.
- Baseline handling: the baseline worker predates the selectors and is never
  sent a selector case. A greedy selector case is compared with the baseline's
  fixed-schedule request of the same turn and draft count. A stochastic selector
  case has no baseline counterpart and rests on the cached against uncached
  candidate comparison. One probe per combination records only the baseline's
  refusal code.

## Artifact, source and environment

| Field | Baseline | Candidate |
| --- | --- | --- |
| Model family, exact checkpoint and upstream revision | Same: retained-vision `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at `0f94842e` | Same |
| Format/quantization and config/weight/tokenizer/processor/template hashes | Weights manifest `8d7cf6d215df4ece54a9d8ebd2d5a756df7b0f7136ae73980b5eca0143fc5324`; config `2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899`; tokenizer `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`; tokenizer config `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27`; chat template `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`; native source `7f5c685f35fc4c5d1509f6379feb1a965cb625f7a1ce76a0b5fe69983561dd30`; preprocessor `022f906548d5a89d3a18c61c0a87de760db8c81d618437501baa3dcdc14d4ecd` | Same |
| Backend, vendor revision and local patch provenance | Native hybrid Lily route | Same |
| Stark commit, dirty patch/runtime manifest and executable hash | `stark-generate` built from `4d09b550`, SHA-256 `5dfb90b61a459560702590703117b5c705e2ae425b7d5da2d5a709bcc766298d` | `target/release/stark-generate` at HEAD `a246039f` working tree, SHA-256 `5abca56f752489dde82d60b42e29ddaf36542962a83770bfba27a52633a3e26b` (verified before use, not rebuilt) |
| Toolchains, dependency lock hashes and build flags | Not rebuilt here | Not rebuilt here |
| Weight, activation, accumulation, recurrent-state and KV precision | Artifact defaults | Same |
| Selected kernel and materialization policies | Defaults; midpoint prefix snapshots on (advertised chunk 2,048) | Same |
| Hardware, physical RAM, OS/driver/Metal toolchain | Apple M5 Max, 137,438,953,472 bytes, macOS 26.7.2 | Same |
| Available headroom, observed peak memory and measurement method | Not measured | Not measured |

Validation tooling: validator SHA-256
`6a04b919751afdfc5a47cb9dd8e1300c5fc1fbbf4f0e35cd0a9eb12983251043`, tests
`e6ffc1ec2f571a6928cb4ca322748242fa4fd95f69d251c5c086522ff62961e3`. The
checkout has no `stark-template` build and none was built; the installed app's
helper (SHA-256 `a1b4d09d8fdbc6cec1201baa0fa73755d09bb2538c945d4d373c7dc744545713`)
was put on `PATH`. It only renders the synthetic prompt; all three workers
receive the same token IDs.

## Workload and controls

- Image-loaded workers (`--vision`); the image invalidation completes a synthetic
  image request. Selector requests are text only.
- Session: seed 11, one tool, two instruction paragraphs, follow-ups 4 4 8;
  turns of 4,276, 4,389, 4,490 and 4,649 tokens; context 8,192; 16 new tokens.
- MTP: two drafts (the artifact recommendation). Stochastic: temperature 0.7,
  top-k 20, top-p 0.95.
- Order: cached candidate (all cases and actions in sequence), uncached
  candidate, baseline, restarted cached candidate; one model load at a time
  inside the model slot.
- Command, wrapped once in `runtime/slot/with_model_slot.sh prefix-validate`,
  with the helper directory prepended to `PATH`, `HF_HUB_OFFLINE=1` and
  `TOKENIZERS_PARALLELISM=false`:

```text
uv run --project tools/converter --locked --python 3.13.13 python \
  scripts/validate_generation_prefix_cache.py --model MODEL \
  --baseline runtime/baseline-worker/stark-generate \
  --candidate target/release/stark-generate \
  --vision --reuse boundary --context 8192 --first-tokens 4600 \
  --tools 1 --instruction-paragraphs 2 --follow-up-sentences 4 4 8 \
  --max-new 16 --mode exactness --mtp-drafts 2 \
  --mtp-policy adaptive --mtp-lookup --progress \
  --output runtime/adaptive-gate/prefix-1.json
```

This is the 2026-11-15 Lily recipe except `--mtp-drafts 2` (was 1) and the two
new flags; two drafts give the adaptive planner a choice of 0, 1 or 2.

## Attempt history and failures

| Trial / source | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| Template preflight | Model-free render check before the run | Failed: no native template helper in the checkout | Setup only; resolved with the installed helper on `PATH`, no build |
| `prefix-1` | Exactness, all three selector combinations | Passed, 260.5 s elapsed (cached 91.2 s, fresh 69.0 s, baseline 43.2 s) | Valid |

## Correctness and lifecycle

- All 67 checks passed: 21 selector rows (15 greedy, 6 stochastic), 3
  fixed-after-selector and 6 plain-after-selector rows, and 37 pre-existing rows
  including the worker
  restart (zero reuse).
- Reuse per selector combination, all matching the rule: after plain 0; warm,
  stochastic repeat, fixed after selector, selector after fixed, shorter (4,490
  tokens), regrow and stochastic regrow 4,096 each; plain after selector 0;
  plain warm 4,096. The selectors did not change any reuse count.
- Cached against uncached with the same options: identical generated token
  hashes, text hashes and parsed tool-call hashes for all 21 selector rows,
  including all 6 stochastic rows. Every reply acknowledged its selector, and
  the integer schedule counters equalled the uncached run's in all 21.
- Greedy selector independence: all 15 greedy selector rows equal the
  fixed-schedule greedy output of the same turn on the candidate and on the
  baseline worker.
- Schedules ran: adaptive used 0-, 1- and 2-draft head rounds; lookup ran 5
  lookup rounds per request, 8 to 10 of 10 lookup drafts accepted.
- Baseline refusal: `invalid_request` for all three combinations.
- Invalidations as documented: error reply (`context_overflow`), unscoped
  request, reset, completed image request and cancellation
  (`generation_cancelled`) each cleared a warm 4,096-token entry;
  tokenize/decode kept it.
- Model-free tests: `uv run --project tools/converter --locked --python 3.13.13
  pytest scripts/test_validate_generation_prefix_cache.py -q` gave 27 passed, 1
  skipped (the pre-existing raw streaming template test, which needs the helper
  in the checkout), 12 subtests passed. New tests cover selector combinations,
  plan invariance without flags, the selector block, readiness, schedule record
  privacy, a synthetic end-to-end run (baseline receives only the refusal
  probes), the default run's unchanged schema, and three injected defects (output
  drift on a hit, selector in the key, ignored selector), each of which fails.

## Performance and memory

| Metric and measurement boundary | Baseline | Candidate | Repeats/dispersion | Eligibility and conclusion |
| --- | --- | --- | --- | --- |
| Tokenization/rendering/planning | not measured | not measured | | |
| First token / prefill | not measured | not measured | | Timing mode not run |
| Useful decode / end-to-end rate | not measured | not measured | | |
| Observed peak memory / retained payload | not measured | not measured | | |

## Evidence, decision and limits

- Private receipts: `runtime/adaptive-gate/prefix-1.json` SHA-256
  `b753f8d8f5001fd91f5c06d5041680a3c82fadf2e6f7a70119550a7e43a3e17b`, log
  `prefix-1.log` SHA-256
  `a4563c2664c811de6a86693ce52569d588c68f4d65e66d32c1889f6853e722f6`.
- Covers this artifact, the two worker hashes above, 4.3k to 4.6k-token
  prompts, reuse at the 4,096 physical boundary, two drafts and 16 new tokens.
- Not shown: in all three stochastic repeats (seed 17) the selector output
  equalled the fixed-schedule output, although the schedules differed. With 16
  tokens this run does not show that a selector changes a sampled trajectory, so
  the stochastic rows confirm cached against uncached equality but not
  schedule-dependent sampling.
- Not tested: interior midpoint restores (prompts past 6,144 tokens), longer
  outputs, one or three drafts, `"mtp": true` with adaptive (limit from
  readiness), lookup on copy-heavy prompts, other artifacts, timing.
- Next: a longer-output or longer-prompt run if midpoint restores or sampled
  divergence under selectors need coverage.

Keep prompts, completions, token vectors, probabilities, credentials, exception
dumps and model/private evaluation files out of this document.

## Follow-up history

### 2026-11-20: non-fixed model default (adaptive with lookup)

Validator extension (SHA-256
`9bee9a1f50e0e975b1bea75124a3d4687653f0a587755fac32f5e7436d8199f2`, tests
`722bec3e9c5a42074a456f255cd4b9453a4d8a976c6451fc51e33a7fc27b2e1f`; 33 passed,
1 skipped, 16 subtests passed). Opt-in by readiness: it applies only when
`mtp.default_policy` and `mtp.default_lookup` are not fixed with lookup off. A
fixed-default candidate keeps the plan, requests and record of the original
attempt.

- Default MTP cases (no selector fields): the reply must name the readiness
  default in `policy`, `lookup` and `stats.schedule` on the cached and uncached
  candidate, with identical integer schedule counters. Greedy cases must equal
  the baseline. Stochastic cases are not compared with the fixed-schedule
  baseline; a recorded comparison fails the row.
- Ten opt-out cases send `"mtp_policy": "fixed", "mtp_lookup": false`, greedy
  and stochastic, cold and warm. They must reuse per the rule, keep the plain
  acknowledgement (no `policy`, `lookup` or `stats.schedule`) and equal the
  baseline's request without the two fields, stochastic included. The selector
  block's fixed step becomes this opt-out.
- Injected defects that fail: default not applied, opt-out not honoured, a
  stochastic default row compared with the baseline.

Run `prefix-2`, the same recipe and flags as `prefix-1` with candidate
`runtime/candidate-02d8247c/stark-generate`, SHA-256
`02d8247c1341b5b8865dbc5fa0a749ceae57ca05950a9f7c7a3e9b5c3cfc074a` (verified),
baseline unchanged. Receipts: `prefix-2.json`
`93f089dcf18abb8e4ec5b1f40e3e6227baef257cca929e08e317654c737196ae`,
`prefix-2.log`
`cbdb0d8a2b59a45f27bb604644a002c4631656260ff673c242c2da9a3ac6fb47`.

Passed: 77 of 77 checks and 6 of 6 invalidation paths, 281.9 s (cached 103.4,
fresh 78.7, baseline 43.0).

- Readiness reported default adaptive with lookup. Every default and selector
  request acknowledged (adaptive, lookup on) with limit 2 and 5 lookup rounds.
  A lone `mtp_policy: "adaptive"` or `mtp_lookup: true` inherits the other
  field from the default, so all three selector blocks ran the default schedule.
- Default cases: reuse 0 after plain, 4,096 warm, stochastic and after the
  opt-out; greedy equal to the baseline.
- Opt-out cases: reuse 0 cold (greedy and stochastic), 4,096 warm; all 9 opt-out
  rows (6 in the opt-out block, 3 selector-block fixed steps) equal to the baseline with a
  plain acknowledgement.
- Selector blocks: 21 rows passed with the same reuse pattern as `prefix-1`.
- Invalidations: error reply, unscoped request, reset, completed image request
  and cancellation cleared the entry; tokenize/decode kept it; restart reused 0.
- Not shown: the seed-17 stochastic default output equalled the stochastic
  opt-out output, so schedule-dependent sampling is again not demonstrated at
  16 tokens. Not verified: the `cost_table` value (not recorded), `"mtp": true`
  resolving to three drafts, the `STARK_LILY_MTP_ADAPTIVE=0` kill switch,
  `default_lookup: false` on fixed-default backends, timing.
