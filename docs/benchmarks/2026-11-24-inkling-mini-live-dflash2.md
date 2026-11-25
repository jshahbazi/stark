# Inkling 2 Mini with live DFlash 2 on M5 Max

Attempt `20261124-INKDRAFT-02`. The previous [matched trace experiment](../model-experiments/2026-11-24-inkling-mini-dflash2-traces.md) measured proposal acceptance and isolated target verification. This run measures a complete request through the image-loaded Stark worker, including DFlash 2 execution, target verification, streaming and the client.

DFlash 2 led both comparison arms on novel prose in all three blocks. Its paired median was 13.2% above DSpark and 7.5% above MLX-Serve there. DSpark led DFlash 2 at 512, 4K and 8K in every block. The initial and sustained coding comparisons between the two Stark arms were unresolved. DFlash 2 is available through the explicit worker profile. The results support retaining DSpark as the ordinary default.

## Registered comparison

Use the original `thinkingmachines/Inkling-2-Mini` target at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb` and the BF16 `incoai/Inkling-2-Mini-DFlash2` checkpoint at `4f9088ea66c21f2b99ed657d1941c306b03e84d2`. The live DFlash arm couples the Stark Rust/Metal target with a persistent MLX GPU draft process. The comparison arms are the same Stark worker and target with DSpark and the original checkpoint under MLX-Serve. DFlash proposal generation, transfer and verifier commits are inside the timed decode path.

Repeat the [November 23 standard benchmark](2026-11-23-inkling-mini-standard-benchmark.md) on the same 40-GPU-core M5 Max with 128 GiB. Use its original benchmark tool and Inkling profile, pinned llmprobe bundle, image-loaded text serving, one stream, greedy sampling, 192-token decode cap, three blocks with reversed arm order and the probe and prefill sessions. Its 16-user burst was omitted in this comparison as on November 23. Give each arm the same tagged inputs, keep the standard exclusions, attach the target quality probe and report cold prefix reuse, the novel-prose and long-context cells individually. Compare contemporaneous block ratios for a standing. Preserve the November 23 rows as dated measurements.

Before timing, verify the DFlash identity in worker readiness, positive live proposal counters, and greedy outputs against the archived hashes. Check target sampling with drafting on/off, text reuse and image invalidation using the actual worker. Select a fixed width before the benchmark, bind the launch profile and binaries, and keep all request material in ignored private files.

## Standard benchmark results

The benchmark ran from 19:17:53 to 19:45:54 EST on November 24. All 18 sessions completed, with AC power and fair process thermal state at every session check. The adapter negotiated 94 W of its rated 140 W. Three samples contributed to each context rung and each exact-prefill clock, and nine to each other cell. No samples were excluded.

The table reports median tokens/s from three blocks. The ratios are medians of the three within-block ratios, followed by their range and the standard's standing computed on unrounded ratios. The two exact-prefill entries time the same workload through different clocks. [Aggregate measurements and build identities](data/2026-11-24-inkling-mini-live-dflash2.json)

| Scenario | DFlash 2, tok/s | DSpark, tok/s | MLX-Serve, tok/s | DFlash 2 / DSpark | DFlash 2 / MLX-Serve |
| --- | ---: | ---: | ---: | --- | --- |
| Initial coding | 84.9 | 86.8 | 87.9 | 0.95 (0.95–1.01), unresolved | 0.99 (0.94–1.00), behind |
| Sustained coding | 83.0 | 86.3 | 82.2 | 0.96 (0.92–1.06), unresolved | 1.01 (1.00–1.08), ahead |
| Novel prose | 79.3 | 69.2 | 73.6 | 1.13 (1.13–1.20), ahead | 1.08 (1.06–1.08), ahead |
| 512 | 78.3 | 82.2 | 80.2 | 0.97 (0.89–0.98), behind | 0.98 (0.91–1.05), unresolved |
| 4K | 97.9 | 107.5 | 88.7 | 0.91 (0.90–0.92), behind | 1.09 (1.07–1.17), ahead |
| 8K | 93.7 | 103.3 | 93.8 | 0.90 (0.84–0.99), behind | 1.00 (0.87–1.04), unresolved |
| 16K | 99.7 | 101.0 | 88.8 | 0.99 (0.95–1.04), unresolved | 1.08 (1.04–1.14), ahead |
| Repeated-sentence prefill | 2227.1 | 2231.0 | 2210.8 | 1.00 (0.99–1.00), unresolved | 1.01 (0.98–1.02), unresolved |
| Exact 8,800-ID prefill, client | 2354.9 | 2325.0 | 2414.1 | 1.02 (0.99–1.04), unresolved | 0.98 (0.96–1.02), unresolved |
| Exact 8,800-ID prefill, engine phase | 2356.4 | 2326.5 | 2415.2 | 1.02 (0.99–1.04), unresolved | 0.99 (0.96–1.02), unresolved |

On the novel-prose requests, both Stark arms started with zero cached input tokens. MLX-Serve reused nine input tokens, as in the November 23 comparison. The effect on its client decode rate was not isolated.

Stark's two arms produced the same output length and token hash on 183 matched native requests (183 DSpark and 183 DFlash 2 receipts). Across the full suite, including warmups and supplementary prompts, each arm emitted 20,825 tokens. DFlash 2 supplied 15,288 accepted proposals and DSpark supplied 12,067. All six DFlash worker sessions reported images loaded and DFlash verification. The ten-input, 2,580-row teacher-forced quality check passed for the shared target. Reference accuracy was 73.64% and candidate accuracy was 73.57%. The worker hash checks cover the live drafter trajectory. The exact 8,800-token prefill input hashes matched across engines.

## Worker and width selection

The worker implementation is `04e448991f88a079957c411127167cf14ab30416`. Both Stark arms ran the executable with SHA-256 `7fa947a0189585c966d6217a381dbf61b555bd77cac16028a3cb60c921a52897`. The DFlash arm used an exec-only launcher to append the operator profile after the standard worker arguments. Its identity binds the executable, companion, config and reference source digests. The standard harness recorded the launcher digest for that arm. Default DSpark ran directly from the same executable. The timing harness and render helper came from the archived November 23 checkout. The model artifact, MLX-Serve and client were supplied from the original run directories. [Worker configuration and pinned dependencies](../INKLING_DFLASH2.md)

First, 52 live companion blocks matched all 364 proposed IDs against the frozen MLX replay, including full-window initialization and incremental updates. Median warm draft-plus-IPC time across 13 anchors was 8.71 ms on prose and 9.39 ms at 4K. The shared target then generated all four archived 192-token outcomes exactly through the streaming worker, with images loaded, at both widths. A fixed-seed sampled control matched drafting off on initial coding and 4K coding.

| Trace | Mean worker decode seconds, three proposals | Mean worker decode seconds, seven proposals |
| --- | ---: | ---: |
| Initial coding | 2.209 | 2.135 |
| Sustained coding prompt | 2.396 | 2.395 |
| Novel prose | 2.527 | 2.437 |
| 4K coding | 2.148 | 2.077 |

Each cell contains two worker requests on the same prompt, with arm order alternating between traces. Across all eight requests per width the decode times summed to 18.560 s and 18.088 s. We selected seven proposals for this benchmark, a 2.6% throughput gain in the width screen. The general operator-profile default remains three. Image-loaded text with drafting on/off also matched in the lifecycle smoke. The next scoped text request reused 2,048 input tokens, followed by a generated PNG sent through the image worker with drafting off and a text request with zero reused tokens and the expected output.

The [isolated verifier replay](../model-experiments/2026-11-24-inkling-mini-dflash2-traces.md#verifying-actual-proposals) counted native rounds on four fixed continuations. At seven proposals, DFlash 2 emitted 3.51 tokens per round and DSpark emitted 2.67. Stark spent 31.09 ms per DFlash round and 28.65 ms per DSpark round on target verification. Per emitted token, verification cost fell to 8.87 ms with DFlash from 10.74 ms with DSpark. The server cells above also include draft generation, feature transfer and commit, and use DSpark's ordinary adaptive policy. The server receipt's `tokens_per_step_median` comes from llmprobe's estimate of steps from the arrival times of streaming frames. The native worker receipts also retain a histogram of round starts by the draft depth the worker chose. Across all 183 requests, including warmups and supplementary prompts, the worker started 5,368 rounds with DFlash 2 and 8,588 with DSpark, or 37.5% fewer with DFlash 2. These totals include zero-depth steps (44 and 66 respectively). The initial draw from prefill precedes the first counted round. A second verifier call in the same round does not increment the histogram. The benchmark adapter retained round counts but omitted the worker’s per-stage clocks.

## Correctness and scope

The image-loaded prefix-cache suite passed all 45 cases on the benchmarked worker. Each cached output matched a fresh DFlash run and the DSpark baseline, and every case reused the expected number of input tokens. Error, unscoped request, reset, image and cancellation paths all cleared the prefix. Tokenization preserved it. The suite exercised the seven-proposal greedy profile, the two-proposal sampled default and a common explicit depth of three. An initial invocation copied the greedy recommendation of seven into explicit stochastic and baseline controls. Those configurations exceed the corresponding worker limits, so the suite was rerun with the supported shared depth. The final worker library test run passed all 179 active tests, with six ignored.

During integration, the first launcher placed the draft option before the required model path and the worker rejected the arguments. A subsequent build incorrectly selected the DSpark-only overlapped verifier for DFlash 2 during the converter's local image startup check. The final executable guards that path on a loaded DSpark drafter and passed the previously failing startup check. An initial quality invocation supplied a file where the diagnostic expected its directory and produced no quality measurement.

Both Stark arms retained Inkling’s vision tower while serving text. The session constructs the tower during model load, including when no staging directory is supplied. A staging-off comparison therefore cannot measure the cost of removing the tower. A genuine tower-absent timing comparison is outside this result.
