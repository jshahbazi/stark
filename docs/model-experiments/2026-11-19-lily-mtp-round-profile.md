# 20261119-SUSHI-02: Lily MTP round profile on Qwen3.8-Flash-Next

Date: 2026-11-19. [Ledger](README.md). Predecessor:
[20261119-SUSHI-01](../benchmarks/2026-11-19-sushi-flashnext-lily-assessment.md).

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered / executed | `20261119-SUSHI-02`, registered and executed 2026-11-19 |
| Kind | Performance diagnosis |
| Attempt state | Completed |
| Result and acceptance criterion | Passed as a diagnosis: the split of one MTP round into draft, verify by stage, finish and host work is measured. No speedup candidate is tested and no gate applies |
| Evidence scope | End-to-end timing on the full model plus instrumented stage attribution; one session each |
| Promotion decision | Diagnostic only |
| Disposition | Experiment only. No production code, default or installed build changes |
| Related attempts | `20261119-LILYROWS-01/02/03`, `20261119-LILYHOST-01`, `20261119-LILYDEPTH-01`, `20261119-LILYBENCH-01` were opened from this result and are recorded separately |

## Hypothesis and change

The source reading in `20261119-SUSHI-01` derived that each extra speculative
position costs Stark roughly 9 ms, against about 3 ms in Sushi, but could not
say how that divides between GPU row cost, draft steps and host work. This
attempt measures the division. There is no candidate; baseline only.

## Artifact, source and environment

| Field | Value |
| --- | --- |
| Model | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83`; the installed artifact directory, read-only. Payload not rehashed for this run |
| Backend | `Forward::Lily`, vendored Lily core as at the source commit; `sampling: exact-rejection-v1`, `verification: batched-decode-exact-v1` |
| Stark commit and executable | `4d09b55045f24eab81da0a9b5b024b9c02318459`, clean tree for `crates/` and `vendor/`; worker SHA-256 `5dfb90b61a459560702590703117b5c705e2ae425b7d5da2d5a709bcc766298d` |
| Toolchain | Rust 1.98.1, locked release build of `stark-generate` |
| Precision | Source mixed Q4/Q8 weights, BF16 activations, F32 reductions, logits and GDN state, Q8 KV; unchanged |
| Images and draft head | Image tower loaded (`qwen38-lily-retained-bf16-vision-v1` readiness) and MTP head loaded |
| Hardware | Apple M5 Max, 128 GiB |
| Memory | Not measured |

## Workload and controls

- Input: 4,096 tokens taken from a public source file of this repository at
  the source commit (`vendor/lily-core/src/qwen4exp/model.rs`), corpus SHA-256
  `f23f98fc7c8bbb224b67faba436a4643dff252d8c6b26d37e4e91f05d66b1257`, supplied
  as token ids. Input token hash is in the public aggregate.
- Sampled arms: temperature 1, top-k 20, top-p 0.95, seed 42, 256 tokens.
  Greedy arms: 512 tokens. `mtp_drafts` 0, 1, 2 and 3 in each. Context is input
  plus output budget. Every request finished by length.
- Prefix cache flag on; every request reported zero reused input tokens.
- Session one, uninstrumented: two 64-token warmups, then each of the eight
  arms twice, second pass in reverse order. Session two, with
  `STARK_QWEN38_PROFILE_OUTPUT` set: the same warmups, then sampled 0 to 3
  drafts and greedy 2 drafts once each. Throughput is quoted only from session
  one.
- The worker was launched image-loaded without the Python client's additional
  image-request check, as `scripts/benchmark_lily_release_baseline.py` does.
- No quiet admission, host screening or host observation. The runner refused
  to start while another `stark-generate` process existed; the owner stopped a
  model loaded in the desktop app before the run.
- Useful decode rate is `(sampled_tokens - 1) / decode_seconds`. Per-round
  phase times are the worker's own `draft_seconds`, `verify_seconds` and
  `rollback_seconds` divided by rounds; "other" is decode time not in those
  three.

## Attempt history and failures

| Trial | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| Run 1 | Launch through the Python `Worker` class | The client's automatic image-request check failed at readiness and closed the worker. No generation request was sent | Invalid setup; no timing. Cause of the check failure not investigated |
| Run 2 | Same runner with the text-baseline launch | 18 uninstrumented and 7 instrumented requests completed | Valid; all results below |

## Results

Uninstrumented timing, medians of two requests per arm:

| Arm | tok/s | ms/token | tokens/round | Round ms | Draft | Verify | Finish | Other host | Accepted/proposed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Greedy, off | 71.87 | 13.93 | | | | | | | |
| Greedy, 1 draft | 79.11 | 12.65 | 1.928 | 24.40 | 1.80 | 21.12 | 0.76 | 0.72 | 492/528 |
| Greedy, 2 drafts | 88.70 | 11.28 | 2.777 | 31.32 | 3.54 | 25.84 | 0.92 | 1.02 | 654/732 |
| Greedy, 3 drafts | 87.30 | 11.45 | 3.453 | 39.55 | 5.26 | 31.37 | 1.67 | 1.26 | 726/882 |
| Sampled, off | 75.45 | 13.26 | | | | | | | |
| Sampled, 1 draft | 77.87 | 12.84 | 1.735 | 22.28 | 1.45 | 19.95 | 0.72 | 0.16 | 216/292 |
| Sampled, 2 drafts | 77.52 | 12.91 | 2.277 | 29.39 | 2.91 | 25.39 | 0.88 | 0.20 | 286/442 |
| Sampled, 3 drafts | 64.80 | 15.45 | 2.476 | 38.24 | 4.43 | 32.48 | 1.11 | 0.22 | 304/612 |

The two requests of an arm differ by 0.4 to 6.8 percent. Greedy output hashes
are identical at 0, 1, 2 and 3 drafts; each sampled arm reproduces its own
hash on both requests.

GPU stage time per serial step and per verify pass, from the instrumented
session (attribution only):

| Stage | 1 row (serial) | 2 rows | 3 rows | 4 rows | Per extra row |
| --- | ---: | ---: | ---: | ---: | --- |
| GPU command total | 14.56 | 19.52 | 25.29 | 30.96 | about 5.7 |
| MoE total (both phases) | 4.89 | 7.38 | 10.19 | 12.71 | about 2.6 |
| MoE routed gate/up with shared down | 2.33 | 4.02 | 5.51 | 6.87 | about 1.4 |
| MoE expert down/combine | 1.30 | 2.28 | 3.36 | 4.41 | about 1.05 |
| Hyper-connections | 2.91 | 4.26 | 5.39 | 6.37 | about 1.15 |
| GDN, all sub-stages | 3.36 | 3.91 | 5.15 | 6.80 | 0.55, 1.24, 1.65 |
| QSA attention | 2.16 | 2.71 | 3.29 | 3.78 | about 0.54 |
| Vocabulary head | 1.13 | 1.14 | 1.15 | 1.17 | flat |
| PLE | 0.10 | 0.11 | 0.12 | 0.13 | flat |

The MoE sub-rows are separate categories and do not partition the MoE total.
The verify profile did not emit GDN sub-stages.

Host and synchronization, instrumented session:

| Item | Sampled | Greedy (2 drafts) |
| --- | ---: | ---: |
| Draft step, wall / GPU busy | 1.41 to 1.45 / about 1.0 ms | 1.72 / 1.03 ms |
| Verify pass, wall minus GPU command | 1.0 to 1.2 ms | 1.1 ms |
| Finish pass, wall / GPU busy | 0.70 to 1.10 / 0.40 to 0.75 ms | 0.87 / 0.55 ms |
| Host target sampling per round | 0.15 to 0.21 ms | 1.00 ms |
| One sampler call | 0.086 ms | 0.357 ms |
| Per row: readback, finite check, bounded top-k select | 0.015, 0.014, 0.070 ms | |

## Reading

- Verify dominates a round: 86 percent of a sampled two-draft round. The
  first row costs about what a serial step costs and each further row about
  5.7 ms of GPU time, 40 percent of a serial step.
- The earlier derived figure of about 9 ms per extra position was the sum of
  that 5.7 ms, one draft step of 1.45 to 1.75 ms, and shares of finish and
  host work.
- Host work is small: about 2.5 ms of a sampled two-draft round and 3.8 ms of
  a greedy one. Reading and scanning complete vocabulary rows costs about
  0.1 ms per row. The larger host items are graph encoding that does not
  overlap GPU work, the synchronous wait around each draft step, and a slow
  greedy sampler call.
- The routed experts are the largest part of the row slope and are close to
  memory-bandwidth-bound: about 1.18 GB of Q4 expert weights per row per pass
  (estimated from tensor shapes, not measured).
- Hyper-connections and GDN together add about 2.3 ms per extra row. Both
  read weights or write state once per row today, and the engine compared in
  `20261119-SUSHI-01` records them as nearly flat to four rows.
- Draft acceptance is high on greedy code (93, 89 and 82 percent of proposals
  at one, two and three drafts). Three drafts does not beat two because the
  third position costs about 8.2 ms for 0.68 more tokens. At temperature 1 the
  third position adds 0.2 tokens and three drafts is 14 percent slower than
  off.

Under conditions close to Sushi's code arm (greedy, 512 tokens, 4k of code
context, though a different corpus), Stark's best fixed depth records 88.7
tok/s. Sushi publishes 146.2 tok/s for its own corpus and pack. This is not a
matched comparison; `20261119-LILYBENCH-01` builds one.

## Decision

Worth pursuing. Arithmetic on the table above: removing 2.1 ms from the row
slope (hyper-connections and GDN) would bring a sampled two-draft round from
29.4 to about 25.2 ms and a greedy three-draft round from 39.6 to about 33.3
ms, roughly 15 to 19 percent more throughput, before any change to host work
or depth. These are estimates and not results.

Workstreams opened, each with its own record and exactness gates:

| Attempt | Scope |
| --- | --- |
| `20261119-LILYROWS-01` | Hyper-connection reads share weights across verify rows, byte-identical to the one-row bodies |
| `20261119-LILYROWS-02` | GDN verify without per-row state capture; rollback by replay |
| `20261119-LILYROWS-03` | Routed experts: measure cross-row expert overlap first; build exact sharing only if at least 20 percent of selections repeat |
| `20261119-LILYHOST-01` | Host round trips: greedy sampler path, encode overlapped with GPU work, GPU-chained greedy drafts, redundant row passes |
| `20261119-LILYDEPTH-01` | Opt-in deterministic adaptive depth and prompt-lookup drafts within the current limit; plan for a higher limit |
| `20261119-LILYBENCH-01` | Fixed-corpus greedy benchmark under Sushi's published protocol |

Raising the draft limit above three is deferred: it needs the GDN rollback
change first, and at today's row cost a fourth draft does not pay.

## Evidence, limits and privacy

- [Public aggregate](../benchmarks/data/lily-mtp-round-profile-2026-11-19.json):
  timings, counts, MTP work, stage milliseconds per call, hashes and private
  receipt hashes. Private run reports stay under ignored `runtime/`.
- One session per mode, two requests per arm, one input length, one corpus,
  no host screening. Differences of a few percent between arms are inside the
  observed spread.
- Stage timings come from an instrumented session and perturb scheduling.
- Prefill, memory, image execution, prefix restoration and quality are not
  measured here.
- No prompt text, token vector, output, probability or exception dump was
  saved.
