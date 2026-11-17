# Flash Next single midpoint snapshot

This prototype adds a default-off `STARK_LILY_PREFIX_MIDPOINT=1` selector for the retained-vision Lily Q4-XL artifact pinned at `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl@0f94842e3682b49acace4daeb4cf2e65a6f94f83`, with artifact digest `18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4`. The baseline and candidate keep 4,096-token physical chunks. The candidate captures one selected boundary on the 2,048 grid. It retains only the immediately preceding matching scoped text request, with the existing token, context and MTP-enabled checks. At an interior boundary it writes one F32 GDN state per recurrent layer, the convolution windows, PLE history and draft hidden. After midpoint restore, the first suffix pass bridges to the next absolute 4,096 boundary. [Prior trace and design](2026-11-17-flashnext-kernel-and-prefix-follow-up.md)

Interior reuse requires at least 128 suffix rows. A smaller suffix can select the serial GDN recurrence or small-row projections where the containing pass used batched arithmetic. Those requests follow fresh prefill. The existing restriction for full-chunk boundaries stays at two suffix tokens.

| Attempt | Control | Decision criterion |
| --- | --- | --- |
| 20261117-FLASHSPEED-11 | Model-free and physical Metal GDN selected-row tests, then full-model capture/no-capture and fresh/resume controls with MTP enabled. | Require bit-identical complete-vocabulary first logits and draft and target readouts consumed by the sampler, plus equal sampled tokens and all four MTP work counts. Cover initial 2,048 midpoint with append and bridge to a full physical boundary, plus capture and repeat at 8K, 12K and 32K. |
| 20261117-FLASHSPEED-12 | Fixed 4,096 physical fresh-prompt A/B at 8K, 12K and 32K, with a unique native cache scope per scored request. | Test the actual capture cost with zero reused tokens. Require matching prompt hashes, emitted output, samples, finish and MTP work. A median first-token regression above 5% at any tested length blocks promotion. |
| 20261117-FLASHSPEED-13 | Fixed 4,096 physical Pi follow-up A/B with the previous scoped coding sessions and independent grader. | Require identical paired trajectory and check counts with exact MTP2 work, demonstrate incremental hits and report first-token and turn time in both launch orders. Reuse the earlier acceptance criteria. |

Full-model timing requires the numerical gate. Run one model at a time, hold startup, warmups, context, seeds and sampling fixed, and use the existing paired launch sequence in both orders. Keep the same checkpoint, vision tower and MTP head resident on the M5 Max. Stop a timing arm on worker or guard failure. Model artifacts, fixtures and raw telemetry remain private. Restore the installed Lily worker and local API after the campaign.

## Exactness

The physical Metal GDN fixture matched 155,762,688 bytes across 2,049, 2,113 and 4,096-row geometries. It compared the ordinary scan with the checkpointing pass, the saved F32 matrix with a standalone 2,048-row prefix, and the resumed GDN scan with the same full-block and serial-tail partition. Offline Metal 4 compilation passed.

The final full-model test passed 12 paired checks, six capture-neutrality and six restore comparisons. It matched 2,979,840 complete-vocabulary first-logit values, 215 full draft readout hashes, 288 full target readout hashes, all generated tokens and all four work counts. All comparisons used the model's native two-draft verifier with a deterministic argmax sampler. The fixture covered the 2,048 midpoint from a 2,205-token prefix, a 128-row suffix at the minimum, a 293-row suffix and a continuation crossing the next 4,096 boundary. It also compared cold and restored 8K, 12K and 32K requests at positions 6,144, 10,240 and 30,720. These are bounded checks on the pinned artifact. Restore and replacement of later interior checkpoints in the same request, cancellation during an in-pass capture and real image transitions need wider coverage before promotion.

The production-shape release worker built with Rust 1.98.1 from `8168ab30` has SHA-256 `5b495c9622b92c4fbf1ae409de2622a607576dbb0c229cd8ed5297471e9b4fc3`. The full-model Rust test used the repository's pinned nightly `2026-10-01` with CPU and Metal enabled. Its executable SHA-256 is `5b76fe33ec2c6c77001f38719e63ab4fb77ff86b1941877435d24231cea68fd8`, fixture bundle receipt `a0c9885ff40851d0999c28e8072a5bc7a619d0acad9f520bbef110cdf99d617c`, and final test receipt `9488b74c30f7ce20d935e101075c31cd53123b7df2243200ef0c397bda1d1807`. The initial unit-test command used stable Rust and omitted CPU, which the test harness requires. The corrected command compiled and both the initial prototype and guarded variant passed their separate physical full-model checks.

The release worker's scoped repeat probes passed at all three lengths, with the physical grid set to 4,096 in each arm. The baseline reused 4,096, 8,192 and 28,672 tokens respectively. The midpoint worker reused 6,144, 10,240 and 30,720. Each of the six repeats held exact emitted output and MTP work relative to its cold request. The probe receipt is `ce3b3207ab992fc34928c67252237994a8d2a68e4977ca3fd73add0952ca0b28`. These probes establish live capture and restore at the measured boundaries. Their timings are excluded from the A/B.

## Fresh coding prompts

The four worker launches followed baseline/midpoint/midpoint/baseline order. Three repository-source prompts at each of 8,192, 12,288 and 32,768 tokens were fed with seed 42, temperature 0.6, top-k 20, top-p 0.95, 40,960 context, 96 output samples and two exact-rejection drafts. Each launch excluded one 32K warmup. Separate scopes trigger capture in fresh workers, and every measured request reported zero reused tokens. Both arms keep the loaded vision tower and two-draft verifier. The worker completed all 36 measured requests and all 18 pairs matched emitted tokens, text, sample counts and MTP work. The runner reported stable file identities and no invalid rows.

| Fresh prompt | Baseline median first token | Midpoint median first token | Median paired change | Midpoint wins |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 4.267 s | 4.300 s | +2.63% | 2 of 6 |
| 12,288 | 6.324 s | 6.450 s | +5.51% | 2 of 6 |
| 32,768 | 17.193 s | 17.796 s | +4.98% | 1 of 6 |

The baseline-first and candidate-first order medians were −1.02% and +3.85% at 8K, −1.41% and +5.90% at 12K, and +0.11% and +9.23% at 32K. At 12K the paired median misses the 5% limit. Across lengths, paired decode and full request wall medians change by +0.33% and +4.18%. Measured windows saw 327 system pageouts on baseline and 963 on the candidate, with zero swapouts in either arm. These measurements include the complete capture path in each arm, including staging. Repeated fixtures share two pairs of launches, with host variation visible across the order controls. The timing receipt is `faf20b3a42c55b543f7592c52c04619b2b7dcd71b9fef8014aab644560f00570`.

## Pi coding-agent follow-ups

Pi 1.5.2 ran the same four bounded three-turn coding tasks as the [physical-grid study](2026-11-17-flashnext-coding-agent-grid.md) through Stark Desktop HTTP and the Rust/Metal release worker. Each task ran twice per arm, reversing launch order. The real Pi client used its read, edit and write tools in private disposable workspaces. The run retained the independent grader, the 2,048-token completion cap, deterministic sampling controls and two exact-rejection drafts. All 16 sessions started with zero reuse. Across 94 ordered native pairs, every input, emitted token sequence, output budget, finish and MTP work count matched. All 24 paired turn grades, client finishes and turn acceptances agreed. There were zero API, native and tool errors. The ledger task hit the intentional Pi tool cap on its last turn in each arm, and ledger and schedule missed the same late grading checks.

| Measure | Baseline | Midpoint |
| --- | ---: | ---: |
| Complete accepted sessions | 4 of 8 | 4 of 8 |
| Accepted user turns | 20 of 24 | 20 of 24 |
| Cumulative checks passed | 244 of 252 | 244 of 252 |
| Native requests with cache hits | 14 of 94 | 60 of 94 |
| Input tokens reused | 57,344 | 159,744 |
| Sum of native first-token times | 124.77 s | 80.25 s |
| Sum of the three Pi turns across all sessions | 311.84 s | 267.01 s |

The 50 requests that gained reuse each skipped another 2,048 tokens and all reached the first token sooner. The median paired first-token reduction was **55.32%** on these requests and **52.82%** when summing them within each session. The median paired reduction in total Pi turn time across the eight session pairs was **15.34%**, with eight wins. Medians were 14.95% lower in baseline-first runs and 15.40% lower in midpoint-first runs. The 20 matched accepted turns were **13.90%** faster at the median, with 15 wins. Every matched accepted follow-up was faster. Among the 34 requests fresh in both arms, median first-token time was 2.50% higher. System-wide pageouts were 860 and 981 pages, with zero swapouts in either arm. The Pi receipt is `03280fb92b550710e22a1aa0524216071ef3dc4d7700b05473ed42b9d32b1575`.

## Decision

Keep the midpoint selector opt-in. It passes the bounded exactness checks and produces a repeatable coding-session gain without changing the physical grid. The 12K fresh median exceeds the 5% promotion limit and the four fully accepted Pi pairs fall below the six-pair minimum. Qualification also remains specific to the pinned retained-vision Lily artifact. The installed production worker returned to Ready after the campaign. Authenticated health, model listing and an MTP-enabled local completion all succeeded.

[Fresh runner](../archive/experiments/2026-11-17-flashnext-speed/fresh_coding.py) · [Pi runner](../archive/experiments/2026-11-17-flashnext-agent-ab/benchmark.py) · [Aggregate results](../benchmarks/data/flashnext-midpoint-2026-11-17.json)
