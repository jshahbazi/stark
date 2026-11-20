# 20261119-LILYMERGE-01: merged Lily MTP changes, quiet before/after timing

Date: 2026-11-19. [Ledger](README.md). Predecessors:
[20261119-SUSHI-02](2026-11-19-lily-mtp-round-profile.md) and the six
workstream attempts it opened.

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered / executed | `20261119-LILYMERGE-01`, registered and executed 2026-11-19 |
| Kind | Performance, end-to-end timing of a merged candidate |
| Attempt state | Completed |
| Result and acceptance criterion | Passed the fixed checks: every default arm reproduces the base output hash, MTP-off arms unchanged within the observed spread, every MTP arm faster with non-overlapping ranges |
| Evidence scope | End-to-end timing and output hashes on the full model |
| Promotion decision | Awaiting the owner's decision. Nothing is promoted by this record |
| Disposition | Adopted on one uncommitted working tree; not committed, not installed, defaults for `mtp` unchanged |
| Related attempts | Merges `20261119-LILYROWS-02`, `20261119-LILYHOST-01`, `20261119-LILYDEPTH-01`, `20261119-LILYBENCH-01`. `20261119-LILYROWS-01` and `-03` are not in the candidate |

## Hypothesis and change

The four adopted workstreams were each screened on a host shared with other
work. This attempt measures them together, against the unmodified base, with
no other project process running.

Candidate, relative to base: GDN verify writes only the final recurrent state
and rolls back by replay; greedy draft steps chain on the device in one
submission; draft and verify passes are encoded ahead and waited on by
polling; verify rows are read in place and rows after the first rejection are
not read; the greedy host sampler is vectorized; opt-in adaptive depth and
prompt-lookup drafts exist behind `mtp_policy` and `mtp_lookup`. With no new
request option the candidate is meant to behave as the base does, faster.

Fixed before the run: all eight default arms must reproduce the base output
hashes; a new option on a greedy request must reproduce the greedy base hash;
a sampled request with a new option must give one hash across repeats.

## Artifact, source and environment

| Field | Base | Candidate |
| --- | --- | --- |
| Model | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, installed artifact read-only, payload not rehashed | same |
| Stark source | `4d09b55045f24eab81da0a9b5b024b9c02318459` | the same commit plus the uncommitted integration working tree |
| Worker SHA-256 | `5dfb90b61a459560702590703117b5c705e2ae425b7d5da2d5a709bcc766298d` | `d4182a6769e7c86c6865d0660ccc0d82d4d6442c774eb0f0bdbe83fd6c3041af` |
| Toolchain | Rust 1.98.1, locked release build | same |
| Precision | Source mixed Q4/Q8 weights, BF16 activations, F32 reductions, logits and GDN state, Q8 KV | unchanged |
| Images and draft head | Image tower and MTP head loaded | same |
| Hardware | Apple M5 Max, 128 GiB | same |
| Memory | Not measured | Not measured |

The base worker is the binary built for `20261119-SUSHI-02`, preserved before
the merge. The candidate passed the integration gates recorded in the
workstream records: unit suites, the real-model verify and rollback control
with zero unequal values in the verify, rollback and host-serial phases, and
the eight-arm hash gate.

## Workload and controls

- Frozen corpus, as in `20261119-SUSHI-02`: 4,096 input tokens of public
  source, sampled arms at temperature 1, top-k 20, top-p 0.95, seed 42 for 256
  tokens, greedy arms for 512 tokens, `mtp_drafts` 0 to 3. Sessions in the
  order base, candidate, candidate, base. Each session: two warmups, then the
  eight arms twice, the second pass reversed. Four measured requests per arm
  per side.
- Candidate-only option session on the same input, two requests per arm.
- Candidate-only copy-heavy session: the first 1,500 corpus tokens repeated to
  4,096, so the continuation copies its context; two requests per arm.
- Fixed-corpus benchmark (`scripts/benchmark_lily_fixed_corpus.py`,
  `20261119-LILYBENCH-01`): 4k rung, greedy, 512 decoded tokens on the
  generated code corpus and 128 on the prose text, three samples per cell,
  base at fixed depths, candidate at fixed depths, adaptive, and adaptive with
  lookup. No sample stopped early.
- No other process of this project was running. The host was not otherwise
  screened; one-minute load averages were 1.77 before and 0.74 after the
  frozen-corpus sessions, and 0.87 before and 0.59 after the benchmark.
- Every request reported zero reused input tokens on the frozen corpus. The
  benchmark's second and third samples reuse 2,048 tokens by design.
- Rates are `(sampled_tokens - 1) / decode_seconds`.

## Results

Frozen corpus, medians of four requests with the full range:

| Arm | Base tok/s | Candidate tok/s | Change | Output hash equal |
| --- | ---: | ---: | ---: | --- |
| Greedy, off | 75.55 (70.4–76.2) | 75.52 (75.2–76.1) | 0.0% | yes |
| Greedy, 1 draft | 81.75 (78.8–83.1) | 96.72 (96.4–97.3) | +18.3% | yes |
| Greedy, 2 drafts | 90.29 (87.0–91.3) | 105.86 (105.6–107.2) | +17.2% | yes |
| Greedy, 3 drafts | 88.51 (87.5–88.7) | 104.66 (104.0–105.6) | +18.2% | yes |
| Sampled, off | 76.05 (74.3–79.2) | 76.42 (75.7–77.7) | +0.5% | yes |
| Sampled, 1 draft | 77.92 (75.4–81.5) | 88.40 (84.9–90.4) | +13.4% | yes |
| Sampled, 2 drafts | 77.90 (75.6–79.6) | 87.38 (86.7–88.1) | +12.2% | yes |
| Sampled, 3 drafts | 67.12 (64.5–68.3) | 74.71 (73.6–75.6) | +11.3% | yes |

Output hashes are equal between base, candidate and the frozen
`20261119-SUSHI-02` baseline on every arm, and MTP work counters are
unchanged.

Per-round time, milliseconds, as round / verify / draft / finish:

| Arm | Base | Candidate |
| --- | --- | --- |
| Greedy, 1 draft | 23.71 / 20.55 / 1.74 / 0.73 | 19.92 / 17.83 / 1.27 / 0.75 |
| Greedy, 2 drafts | 30.96 / 25.58 / 3.47 / 0.89 | 26.17 / 22.86 / 2.29 / 0.92 |
| Greedy, 3 drafts | 39.10 / 31.02 / 5.20 / 1.64 | 32.96 / 28.44 / 3.31 / 1.09 |
| Sampled, 1 draft | 22.21 / 19.88 / 1.45 / 0.73 | 19.72 / 17.44 / 1.36 / 0.74 |
| Sampled, 2 drafts | 29.30 / 25.31 / 2.90 / 0.88 | 26.05 / 22.22 / 2.67 / 0.93 |
| Sampled, 3 drafts | 37.10 / 31.39 / 4.40 / 1.10 | 33.17 / 27.70 / 4.00 / 1.21 |

Verify wall time falls 2.4 to 3.7 ms per round on every arm. Which share comes
from GDN replay and which from polled waits is not separated here; the
host-roundtrip record notes that the polled-wait gain is larger than the
documented wake latency and its cause is not established.

Opt-in options on the candidate, frozen corpus, two requests per arm:

| Arm | tok/s | Note |
| --- | ---: | --- |
| Greedy, 2 drafts fixed | 104.94 | |
| Greedy, 3 drafts fixed | 103.10 | |
| Greedy, adaptive, limit 3 | 105.37 | Per request, 183 two-draft rounds and one zero-draft round; greedy base hash |
| Sampled, 2 drafts fixed | 86.81 | |
| Sampled, adaptive, limit 3 | 86.56 | Per request, 68 one-draft, 61 two-draft and one zero-draft round; one hash across repeats |

Copy-heavy input on the candidate, two requests per arm:

| Arm | tok/s | With prompt lookup | Change |
| --- | ---: | ---: | ---: |
| Greedy, off | 75.19 | | |
| Greedy, 2 drafts | 110.80 | 119.29 | +7.7% |
| Greedy, 3 drafts | 115.96 | 128.04 | +10.4% |
| Greedy, adaptive, limit 3 | | 127.82 | |
| Sampled, 2 drafts | 112.36 | 121.52 | +8.2% |

Greedy lookup arms reproduce the greedy output hash of that input. Each
sampled arm gives one hash across its two requests; sampled output with
lookup differs from sampled output without it, as permitted.

Fixed-corpus benchmark, 4k rung, greedy, medians of three:

| Cell | Base | Candidate |
| --- | ---: | ---: |
| Code, off | 77.9 | 78.1 |
| Code, 1 draft | 85.0 | 100.6 |
| Code, 2 drafts | 94.5 | 111.9 |
| Code, 3 drafts | 97.1 | 115.9 |
| Code, adaptive, limit 3 | | 112.6 |
| Code, adaptive with lookup, limit 3 | | 122.6 |
| Prose, off | 75.9 | 77.3 |
| Prose, 1 draft | 72.8 | 87.1 |
| Prose, 2 drafts | 66.7 | 79.9 |
| Prose, 3 drafts | 54.6 | 64.4 |
| Prose, adaptive, limit 3 | | 77.4 |
| Prose, adaptive with lookup, limit 3 | | 77.3 |

Greedy output hashes are identical across every depth and option within each
corpus and between base and candidate.

## Reading

- With defaults the candidate is 11 to 18 percent faster on every MTP arm and
  unchanged with MTP off. The candidate's range does not overlap the base's
  on any MTP arm.
- One draft on prose changes from a loss on the base (72.8 against 75.9 off)
  to a gain on the candidate (87.1 against 77.3 off).
- The adaptive planner is behind the best fixed depth on the benchmark: 112.6
  against 115.9 on code and 77.4 against 87.1 on prose. Its static cost table
  was measured on the base build and is now stale: it prices a one-draft
  round as not worth taking on prose. Retuning it is a separate follow-up.
- Prompt lookup adds 8 to 10 percent on copy-heavy input and 9 percent on the
  benchmark's code corpus over the adaptive arm, and nothing on prose.
- Beside Sushi's published 4k figures for its own pack (146.2 code, 79.5
  prose), the candidate's best cells are 122.6 and 87.1. That is not a
  matched comparison; the differences are listed in
  [the benchmark record](2026-11-19-lily-fixed-corpus-benchmark.md).

## Follow-up: retuned adaptive planner

The same day the planner's cost table was replaced with merged-build values
and its settling behaviour changed, as recorded in the
[depth-policy record](2026-11-19-lily-mtp-depth-policy.md). Only
`mtp_policy.rs`, a constant name and a new test changed. Candidate worker
SHA-256 `5abca56f752489dde82d60b42e29ddaf36542962a83770bfba27a52633a3e26b`.
The whole protocol above was repeated against the same base worker.

Defaults, frozen corpus, four requests per arm per side: every output hash
again equals the base; greedy MTP arms are +17.5 to +17.8 percent, sampled
MTP arms +9.2 to +14.9 percent, MTP-off +1.0 and +0.1 percent. The planner
change does not affect fixed-depth requests; the retune author's slower
session was session variation.

Fixed-corpus benchmark, 4k rung, candidate only, in the order fixed, adaptive,
adaptive with lookup, fixed; tok/s medians of three:

| Cell | Fixed, first session | Fixed, last session | Adaptive, limit 3 | Adaptive with lookup |
| --- | ---: | ---: | ---: | ---: |
| Code, off | 75.2 | 77.0 | | |
| Code, 1 draft | 98.8 | 101.6 | | |
| Code, 2 drafts | 109.0 | 112.9 | | |
| Code, 3 drafts | 113.3 | 116.9 | 113.2 | 119.1 |
| Prose, off | 76.2 | 76.7 | | |
| Prose, 1 draft | 86.2 | 87.3 | 83.9 | 84.8 |
| Prose, 2 drafts | 77.3 | 79.9 | | |
| Prose, 3 drafts | 62.9 | 64.7 | | |

The adaptive cells are shown on the row of the best fixed depth. The two
fixed sessions differ by up to 3.6 percent, which bounds what this layout
can resolve.

- Prose adaptive rises from 77.4 to 83.9 tok/s. It is now 9 to 10 percent
  above MTP off and 3 to 4 percent below the best fixed depth.
- Code adaptive is 113.2 against 113.3 and 116.9 for three fixed drafts.
- Code adaptive with lookup is 119.1, against 122.6 before the retune. The
  difference is inside the session drift and is not resolved.
- Frozen corpus, two requests per arm: greedy adaptive 104.90 against 103.35
  at two fixed drafts, reproducing the greedy base hash. Sampled adaptive
  82.09 against 85.98 at two fixed drafts, one hash across repeats; sampled
  arms generate different text from each other, so 256-token samples cannot
  rank them and sampled adaptive is not established as a gain or a loss.
- Copy-heavy input: greedy three drafts 115.56, with lookup 126.18, adaptive
  with lookup 127.21; sampled two drafts 112.38, with lookup 120.29.

[Public aggregate for the follow-up](../benchmarks/data/lily-mtp-merged-quiet-timing-retuned-2026-11-19.json).

## Evidence, limits and privacy

- [Public aggregate](../benchmarks/data/lily-mtp-merged-quiet-timing-2026-11-19.json):
  rates, per-round times, MTP work, schedule counters, hashes and private
  receipt hashes. Private run reports stay under ignored `runtime/`.
- The host was free of this project's processes but was not quiet-admitted or
  screened, and host observations beyond load averages were not recorded.
- Small sample: four requests per arm per side on the frozen corpus, two per
  arm for options, three per benchmark cell, one input length.
- Prefill, memory, image execution, prefix restoration and answer quality are
  not measured. The full engine test suite was not run on the candidate; the
  hybrid module, MTP, Lily core and worker suites were.
- The candidate is an uncommitted working tree. Its worker hash identifies
  that tree at the time of the run only.
- No prompt text, token vector, output, probability or exception dump was
  saved.
