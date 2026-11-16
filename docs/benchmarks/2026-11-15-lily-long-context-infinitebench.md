# Lily projection reuse with a very long public benchmark input

Date: 2026-11-15, local time; execution receipts use UTC.
Attempt: `20261115-LILY-LONGCTX-01`.

## Question and state

Run an additional matched baseline/projection-reuse comparison with a genuinely
long input. The requested half-window size was illustrative, not a requirement.
The selected unmodified public benchmark case contains **88,398 native input
tokens**, with a declared context of **98,304** and a **4,096-token output
ceiling**. Two drafts are fixed in both arms, with images and the draft head
loaded.

**The candidate improves median paired useful decode throughput by 5.40%,
winning all eight pairs. Complete request time improves 2.39%, with seven
of eight wins.** Every measured response is the same 1,339-token summary,
with identical MTP work and natural EOS. The two session halves improve
decode rate by 6.70% and 4.24%.

This adds evidence for a modest, consistent observed decode benefit on a very
long input. Prefill dominates the complete request. Production source, the
two-draft recommendation and the installed app are unchanged. The earlier
[short-input, long-response results](2026-11-15-lily-small-row-verifier.md)
retain their original source binding and promotion decision.

## Internet benchmark and fixture identity

Use [InfiniteBench's English whole-book summarization task](https://github.com/OpenBMB/InfiniteBench),
`longbook_sum_eng`, with the upstream `gpt4_templates` instruction and the
complete supplied book context. This is one **benchmark-derived performance
case**, not a full InfiniteBench evaluation or official quality score.

| Field | Frozen identity |
| --- | --- |
| Prompt source | [OpenBMB/InfiniteBench `src/prompt.py`](https://github.com/OpenBMB/InfiniteBench/blob/51d9b37b0f1790ead936df2243abbf7f0420e439/src/prompt.py), revision `51d9b37b0f1790ead936df2243abbf7f0420e439` |
| Prompt-source SHA-256 | `d49a6891d6bc61522bb428d66a07b092a6bb6bc195aa0fe103eddd6bb39c7f10` |
| Dataset | [Author's `longbook_sum_eng.jsonl`](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/blob/90f0394333616266d9fe85824ceaf505093cbaa5/longbook_sum_eng.jsonl), revision `90f0394333616266d9fe85824ceaf505093cbaa5` |
| Complete dataset file | 78,601,868 bytes; SHA-256 `4e4279cc06c07778f646cda1335755cf5ccdcc38fe9c8a62f1631642fc031027`, verified against the source LFS identity |
| Selected record | Original zero-based row 1, dataset ID 1; record SHA-256 `7b3d2126b81368682234078fef1b4aae8a8ef36a6b3b16e6ee81aff8d9c89f10` |
| Selection policy | First original-order complete record with 65,536–131,072 rendered native tokens; selected by length before any model timing |
| Excluded earlier record | Row 0 renders to 402,018 tokens, outside the selection range and the configured model limit; it was not truncated |
| Book context | Unmodified, 384,908 UTF-8 bytes; SHA-256 `1856d77efc6efe181cc3f78ae164c11183b832255b6586db5ff9e01086bf6c72` |
| Model-rendered request | 88,398 tokens, 385,007 UTF-8 bytes; rendered SHA-256 `15a2484f02c29605124d14d5a7a79535c0386f0eecc2df7bb43b64a26501f374` |
| Input token digest | `40e461e14c18b8da56cc0d4ab5ce0fb63ec47b5246a090603f9354d82fe58ede`; native worker tokenization must exactly match preparation |
| Source license declarations | Repository code: MIT; dataset card: Apache-2.0. Source links and license/card hashes are retained in the aggregate record; no dataset text is redistributed |

The dataset is fetched, integrity-checked and selected entirely in memory.
The upstream template is extracted as literal data without executing remote
Python. The same selected book is fetched and rechecked for the run; only its
identity, byte/token counts and hashes are written. There is no padding,
truncation or performance-based example selection.

The original benchmark's evaluation utility declares a 1,200-token summary
budget. This performance case allows 4,096 with natural EOS to avoid
artificially ending a longer response. It uses Stark's real model chat
template, default system message when present and thinking disabled. The
benchmark instruction and book are unchanged; these inference settings are
an explicitly adapted performance protocol.

## Model, builds and unchanged execution policy

| Field | Bound identity |
| --- | --- |
| Artifact | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83` |
| Payload manifest | `97c11dd6f7ee5a69e23759b9774acb21a786c48552ca2ede9dac8b6be4b5bc98`; existing verified receipt, not a new full payload rehash |
| Model integrity checks | Once before the run and after each worker session: 63 files by size/mtime/inode, nine ancillary files by size/SHA |
| Source checkout | `a4a1cc3021281a329cd42f0f7bf350a14a25bee1`; engine/vendor source is unchanged from the earlier frozen worker base |
| Baseline source/build | Source `cc8e47a824f802e24d442183ebbdd29fd25ecaaa`; worker SHA `61b9c8cc724407573169eb2abad7f908e4943eead7ac3122eeec3ef841a48640` |
| Candidate base/patch | Base `2ac1b713d893f85b97111c1279465cfaa12d7bbe`; patch SHA `9f406ddd07961d1d529622d13699cb119aacdbf9dbeba33fef0d6e4e5e4b0768` |
| Candidate worker | SHA `096cceb6b370fea71b49523c928c68c1ac172ca77b63a968acf7ceb0a26dbe2e` |
| Native template helper | SHA `a3697940cf58166e31a022b1788c5c48ed2605792e97f686ce4626dc05a27ca5` |
| Build provenance | Reuse the sealed Rust 1.98.1 locked release binaries from the [verifier study](2026-11-15-lily-small-row-verifier.md); no rebuild or new kernel source |
| Host | Apple M5 Max, 128 GiB, macOS 26.7.2 / `25G332`; existing concurrent work is retained |
| Precision/backend | `Forward::Lily`; source mixed Q4/Q8 weights, BF16 activations, F32 accumulation/recurrent/readouts, Q8 KV; unchanged |
| Image/MTP capability | Image tower and supported draft head retained; require `qwen38-lily-retained-bf16-vision-v1` and recommended drafts two |
| Configured/native maximum | 262,144 tokens; actual requested capacity is 98,304, with memory admission still enforced |

The candidate selector is enabled only in candidate worker processes. It shares
packed weight loads across two to four rows in remaining exact generic
projections; its arithmetic, scope and historical 60-case component control
are documented in the earlier study. That component result is not rerun or
broadened here. The production checkout does not contain this experimental
dispatch.

## Fixed run protocol

- Four fresh workers run sequentially in before/candidate/candidate/before
  order. Each performs one excluded same-book, full-context warmup with 64
  output tokens and four measured responses: **16 measured requests, four
  warmups, eight positional pairs**.
- Every request uses the identical complete 88,398-token input, context 98,304,
  temperature 1, top-k 20, top-p 0.95, no min-p, seed 42, explicit
  `mtp_drafts: 2` and natural model EOS. The measured output ceiling is 4,096.
- Prefix reuse is disabled and every response must report zero reused tokens.
  Each generation starts fresh logical state. Loaded buffers/capacity may be
  retained within a worker, equally in both arms.
- Streaming is enabled in both arms for count/timing progress during long
  prefills. Text deltas are consumed and discarded; the terminal response
  remains only in memory. Streaming overhead is included here; these timings
  are not pooled with the earlier non-streaming natural-response study.
- Loading, dataset fetching, rendering and the preparation/native-equality
  tokenization probes are outside request timings. Each timed request sends
  the rendered prompt, so complete wall time includes JSON/transport and the
  worker's repeated native tokenization. Backend-native prefill and decode
  fields exclude that tokenization. Preserve these boundaries separately,
  along with emitted/sampled counts, EOS, MTP work and hashes. Useful decode
  is `(emitted_tokens - 1) / decode_seconds`.
- Pair session 1 with 2 and session 4 with 3 by repetition. Report median
  individual paired percentage changes, wins, both session halves, absolute
  savings, arm ranges/IQR/MAD and unchanged-baseline variation. Prefill, decode
  and whole-request changes answer different questions.
- Preserve all host observations under current concurrent work. No quiet
  admission, cooling, host setting changes, retries or slow-run exclusions.
  A fixed absolute 1,800-second deadline and 900-second inactivity timeout
  apply. Only this study's owned workers can be stopped.
- Require exact matching native input vectors in RAM and repeated response
  hashes, lengths and MTP work for scheduling attribution. Warmup equality
  uses its separate 64-token budget. Hash agreement is not a semantic quality
  grade or a full-vocabulary distribution proof.

The preflight planner records only whether its probe returned an error; it
does not retain the planner's potentially reduced returned capacity. Actual
successful generation with the full input/context/output budget establishes
admission for those requests. A resource failure remains a separate outcome
and does not authorize disabling images, MTP or memory guards.

## Results

All four sessions complete: **16 measured responses and four fixed warmups**.
Every request passes native input equality, explicit two-draft/exact-verifier,
count/timing, zero-prefix-reuse and stable output/work checks. All four
post-session model guards pass. There are no failed or retried generations.
The measured responses end at EOS with 1,339 emitted tokens and 1,340 samples
including EOS; the four warmups end at their separate 64-token ceiling.

Absolute columns below are medians of eight requests per arm. Changes are
medians of eight individual paired ratios, rather than ratios of the displayed
medians. Time decreases favor the candidate; rate increases favor it.

| Metric | Before | Candidate | Median paired change | Candidate wins |
| --- | ---: | ---: | ---: | ---: |
| Native prefill seconds | 43.998 | 43.660 | −1.33% | 6/8 |
| Native decode seconds | 18.347 | 17.385 | −5.12% | 8/8 |
| Useful decode tokens/s | 72.93 | 76.96 | **+5.40%** | 8/8 |
| Complete request seconds | 62.332 | 61.107 | **−2.39%** | 7/8 |

Median paired absolute savings are **0.944 seconds in decode** and **1.495
seconds for the complete request**. Prefill saves an observed median paired
0.588 seconds, but this is inside the baseline's 5.78% full relative prefill
range. This candidate changes exact small-row verification projections; it
does not establish a faster long-prefill algorithm.

| Dispersion | Before | Candidate |
| --- | ---: | ---: |
| Useful decode tokens/s, min–max | 66.68–74.09 | 76.55–78.31 |
| Useful decode tokens/s, IQR / MAD | 1.026 / 0.643 | 0.754 / 0.348 |
| Complete request seconds, min–max | 61.671–66.152 | 60.287–63.098 |
| Complete request seconds, IQR / MAD | 0.691 / 0.496 | 1.177 / 0.618 |

The first four-pair half improves median paired decode rate 6.70% and wall
time 1.74%; the reversed second half improves them 4.24% and 2.78%. All eight
decode pairs favor the candidate, with pair gains from 3.44% to 16.78%.
The first baseline response has the slowest decode time and remains included.
The full relative baseline rate range is 10.16%, larger than the median gain;
its IQR is 1.03 tokens/s. These descriptive measures and repeated paired wins
should be assessed together. This is additional evidence, not a statistical
confidence claim or a new automatic promotion gate.

Each complete response performs 571 rounds, 1,142 proposals, 769 accepted
proposals and 1,713 verified rows, identically across arms. The outcome is not
explained by a shorter answer, changed draft count or reduced speculative work.

All 16 measured requests retain host observations; none meet the historical
quiet screen, and none are excluded. Before/candidate peak external CPU
medians are 282.98%/279.46%, with maxima 392.65%/462.07%. Measured pageout
counter increments total 3,904/5,576, with zero swapout increments. All measured
observations have no sampling gaps, observer failures or compiler flags;
maximum thermal state is one. Fixed warmups separately retain 2,162 pageout
and 376 swapout increments in total, including one sampling gap during a
candidate warmup. The slower 56.33-second candidate warmup is retained as
the preregistered excluded warmup, without any extra retry.

These are measurements under accepted ongoing system work. GPU clocks, power
and exclusive GPU use are unmeasured. Prefill accounts for most request time,
so a roughly five-percent decode gain translates to a much smaller complete
request change. Task content and streaming also differ from the preceding
short-input studies; comparing their gains does not isolate context length
as the sole cause.

## Evidence boundaries and disposition

This is one task, record, sampler and seed on the same pinned quantization.
It tests a much longer prefill and attention history than the 20/24-token
inputs in the preceding study. It does not measure a complete benchmark
suite, grade summary accuracy or qualify other models/settings.

No new inference state, cache option or precision path is introduced. Prefix
restoration, image execution/transitions, cancellation recovery and complete
target-distribution qualification remain outside this fresh-request case.
The archived candidate is not enabled in production or installed. Modest
positive results, regressions and host variation will be preserved without
automatically treating this single case as a promotion decision.

The [public aggregate](data/lily-long-context-infinitebench-2026-11-15.json)
retains individual request counts/timings, source/build identities, host
observations, pairing and hashes. The [archived original harness and analysis](../archive/experiments/2026-11-15-lily-long-context/README.md)
bind four source files and nine private aggregate receipts. Internet
corpus, rendered prompts, responses, token arrays, probabilities and exception
dumps are never written. Earlier benchmark data and archived sources remain
unchanged.

Repository policy, whitespace and archived Python syntax checks pass. Four
archived source hashes and nine private aggregate receipt hashes verify.
All 226 existing tracked benchmark-data/archive files match their `a4a1cc30`
Git blobs byte-for-byte, and production source and dependency locks are
unchanged. The earlier report's tokenization wording is clarified without
changing any timing values: complete request wall time includes native
per-request tokenization, while native prefill/decode excludes it.
