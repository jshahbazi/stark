# Inkling 2 Mini first-look engine matrix against MLX-Serve

Attempt `20261122-INKENGINE-01`. One llmprobe invocation per arm on 2026-11-22 compared two uncommitted Stark builds with MLX-Serve on the original Inkling 2 Mini checkpoint. On matched rendered inputs both Stark arms were ahead of MLX-Serve on all four context-ladder rungs and on three of four counting-ceiling rungs, and behind on sustained coding, novel prose and both prefill scenarios. Initial coding overlapped. This is a first look: one invocation per arm, with n counting samples inside that invocation. It does not qualify a speed change. Record in the [experiment ledger](../model-experiments/README.md).

## Identity and conditions

| Field | Value |
| --- | --- |
| Attempt ID / date | `20261122-INKENGINE-01` / 2026-11-22 |
| Kind | Performance, engine comparison |
| Result | Inconclusive as a speed qualification; observations recorded per cell |
| Model | Original fixed-three-pass `thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, hashes as in the [default-router record](../model-experiments/2026-11-21-inkling-mini-decode-default.md#artifact-and-run-controls). MLX-Serve served its own load of the same original checkpoint. |
| Stark current arm | Defaults plus the first launch-fusion pass ([20261122-INKFUSE-01](../model-experiments/2026-11-22-inkling-mini-launch-fusion.md)), before the long-position fix ([20261122-INKLONGPOS-01](../model-experiments/2026-11-22-inkling-mini-long-position-gain.md)). `stark-generate` SHA-256 beginning `133f76c237df`. |
| Stark baseline arm | Defaults only ([20261122-INKDEFAULT-01](../model-experiments/2026-11-22-inkling-mini-default-fast-paths.md)), `stark-generate` beginning `2da8222d57cf` |
| Stark source | Uncommitted working tree on top of `f1666090`. Full binary hashes and patch hashes are not recorded here. Not installed. |
| MLX-Serve | `v26.11.4-pre-release.1`, binary SHA-256 beginning `05983e1278d6` |
| Client | llmprobe 0.6.18 with the aggregate receipt patch |
| Harness | [`scripts/benchmark_inkling_engine_matrix.py`](../../scripts/benchmark_inkling_engine_matrix.py), new today. Subcommands `stark`, `mlx-serve`, `both` and `summarize`; arms are binaries. It refuses to time on battery, in Low Power Mode or on a busy host, and writes aggregate-only receipts. Model-free tests in [`scripts/test_benchmark_inkling_engine_matrix.py`](../../scripts/test_benchmark_inkling_engine_matrix.py). |
| Command | `both --repeats 1 --sessions probe`, arms alternated, one llmprobe invocation per arm on a fresh server |
| Reasoning | `off` on both engines. On this model the template then renders effort level 0. llmprobe reported that both engines still produced thinking. |
| Input matching | Every arm received identical rendered input token counts: 71, 72, 40 and 498 on the short scenarios, and the counts below on the longer ones |
| Drafting | Each engine's own default: Stark's worker default of three proposals for these greedy requests, MLX-Serve's default on |
| Host | 128 GiB Apple M5 Max (Mac17,7), macOS 26.7.2 (25G332), AC power, `pmset` powermode 2 |
| Receipts | Aggregate receipts are kept in an ignored private directory. Their SHA-256 values are not recorded in this report. |

## Results

Median tok/s with the range where n > 1. Decode scenarios report client decode rate; prefill scenarios report input tokens over client time to first token. Stark current is the fused build, Stark baseline the defaults-only build.

| Scenario | Input tokens | n | Stark current | Stark baseline | MLX-Serve original |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial coding | 71 | 3 | 78.0 (76.6–82.4) | 76.3 (75.8–77.0) | 80.5 (75.4–81.1) |
| Sustained coding | 72 | 3 | 77.0 (75.0–78.9) | 77.1 (75.0–78.8) | 84.2 (81.0–85.4) |
| Novel prose | 40 | 3 | 58.6 (58.5–59.9) | 65.3 (64.2–66.0) | 71.8 (70.9–72.4) |
| Context about 512 | 498 | 1 | 84.0 | 82.7 | 80.6 |
| Context about 4K | 4,019 | 1 | 106.5 | 104.1 | 84.2 |
| Context about 8K | 8,252 | 1 | 105.4 | 103.4 | 93.7 |
| Context about 16K | 16,335 | 1 | 93.1 | 91.1 | 86.8 |
| Counting ceiling, 512 rung | not recorded | 1 | 77.8 | 76.9 | 72.1 |
| Counting ceiling, 4K rung | not recorded | 1 | 102.9 | 101.5 | 84.8 |
| Counting ceiling, 8K rung | not recorded | 1 | 96.1 | 94.4 | 97.6 |
| Counting ceiling, 16K rung | not recorded | 1 | 88.3 | 86.5 | 83.5 |
| Repeated-sentence prefill | 8,768 | 3 | 837.4 (832.3–838.6) | 830.0 (817.4–830.2) | 2,065.5 (2,059.7–2,088.1) |
| Unseen-text prefill | 7,514–7,586 | 3 | 814.6 (811.8–824.2) | 839.3 (827.7–841.2) | 2,135.7 (2,072.7–2,142.0) |

Tokens per step on the short scenarios, current / baseline / MLX-Serve: initial coding 2.37 / 2.37 / 1.68, sustained coding 2.23 / 2.23 / 1.70, novel prose 2.00 / 2.00 / 1.19. Stark reports more tokens per step on all three and is still slower on sustained coding and novel prose. On these figures its time per step is higher there; whether the two engines count steps the same way was not checked.

Stark reused 2,048 cached tokens at the 4K counting-ceiling rung and 14,336 at the 16K rung, by the scenario's design. Those two Stark cells are not cold-prefix measurements. Reuse on the MLX-Serve side is not recorded.

## Not run

- The 8,800-ID unseen-text prefill control with matching input-ID hashes and the 16-user burst were not run.
- The author 8-bit pack arm did not run. Its four weight shards had been deleted from disk by another session, and MLX-Serve exited 1 at startup. The published MLX-Serve headline of 116 tok/s on M4 Max is the sustained-coding scenario on that 8-bit pack, so this run could not measure it.
- No other llmprobe scenario results are recorded in this report.

## Reading the results

- One invocation per arm. The differences between the two Stark arms are within what a single invocation can resolve on most rows, and the novel-prose drop for the current arm (58.6 against 65.3) is unexplained. This run does not qualify the launch-fusion pass in either direction.
- "Sustained coding" is the initial-coding request repeated three times with a cold prefix after the rest of the suite. It is not a multi-turn or long-context test.
- Both Stark arms predate the long-position fix, under which drafted decode committed state that differed from serial decode above about 13,000 positions while emitting the same tokens. The 16K rungs ran on those builds. The effect of the fix on these timings was not measured.
- Prefill is the largest gap: MLX-Serve measured about 2.5 to 2.6 times Stark's rate on both prefill scenarios at about 7,500 to 8,800 tokens.

## Earlier comparison, not comparable

An earlier same-machine comparison on the morning of 2026-11-22, recorded in another worktree and not on this branch, had Stark at 34.76 tok/s against 97.58 on short coding and 32.92 against 68.39 at about 8K. At that time Stark's fast paths were not admitted outside 512 to 4,352 tokens. That comparison also used mismatched reasoning modes, because the client's `reasoning_effort: "medium"` was rejected by Stark and accepted by MLX-Serve, and MLX-Serve was forced to full length while Stark stopped at end of sequence. Its absolute numbers are not comparable with this run.

## Next

Repeat with several invocations per arm, run the 8,800-ID prefill control and the burst, restore the author 8-bit pack, and pair the two Stark builds directly before attributing the prose difference.

## Follow-up history

### 20261122-INKENGINE-02 · Second run with the author 8-bit pack

A single-repeat run: one llmprobe invocation per arm on a fresh server, with n counting samples inside that invocation. It is a second first look, not a speed qualification, and it does not replace the first run's numbers.

**Why the first run lost its 8-bit arm.** The first run (private directory `20261122-161108-both`, manifest status `failed`) could not start MLX-Serve on the author 8-bit pack. Another session's disk-cleanup script had deleted the pack's four weight shards after an earlier 8-bit benchmark, leaving only the index, configuration, tokenizer and `mtp.safetensors`. A bounded reproduction showed MLX-Serve loading the 54 draft-head weights only and exiting 1 with `Inkling2PackLayoutUnknown`. The harness's own launch arguments were not the cause. The shards were later restored from the pinned Hugging Face revision `cb7fa052a9bbafbbeb91750edc67046e7bacd4f1` of the pack, and the restore verified all four shards before this run.

**Harness changes since the first run** ([`scripts/benchmark_inkling_engine_matrix.py`](../../scripts/benchmark_inkling_engine_matrix.py), tests in [`scripts/test_benchmark_inkling_engine_matrix.py`](../../scripts/test_benchmark_inkling_engine_matrix.py), 45 passing at the time of the change):

- Power-mode parsing: `pmset`'s `lowpowermode` is read as the Low Power flag, and `powermode` as an enumeration in which only 1 is Low Power, so High Power mode (2) is no longer mistaken for Low Power Mode. How the earlier parse behaved on this host is not recorded in the session reports.
- Weight-shard preflight: every shard named in `model.safetensors.index.json` must exist and be non-empty (without an index, at least one shard other than `mtp.safetensors`). A pack with missing shards is refused before anything starts.
- A failed arm no longer aborts the run: it is recorded under `arm_failures`, skipped for the rest of the run, the status becomes `partial` and the command exits 3. Losing AC power still aborts everything. `summarize` lists arms with no data as failed and merges a later run of the same arm under its normal name.
- Readable start failures: up to 60 lines of MLX-Serve's own pre-ready output are kept in the private run directory on a failed start; collection stops once the server answers, before any benchmark request. For Stark only the fixed failure code is recorded.
- `--arms NAME[,NAME]` runs only the named arms.

**Conditions.** Run directory `20261122-175344-both` (manifest SHA-256 `db235fb99a003d8811813dbebceeeae3b315b71ccbc8cc146fdc8fd66e2e905c`, status `complete`, no arm failures), 2026-11-22 22:53 to 23:04 UTC. Harness SHA-256 `b81feb7dd16948c5792c4fe5740d93f5e03d77a1ec37f9a9f4d8f3884117acc9`; llmprobe 0.6.18 with the receipt patch; repository `f1666090` with 44 dirty paths. AC power, Low Power Mode off, no sleep events; the process thermal state was 0 before the first session and 1 (Fair) from the second session on. Reasoning `off` (effort level 0, same render on both engines; llmprobe still reported thinking for every arm). Each engine's default drafting; MLX-Serve's draft not forced. Sessions: probe and prefill. The 16-user burst was not run.

| Arm | Binary SHA-256 | Model |
| --- | --- | --- |
| stark-current | `stark-generate` `73a0fa2e1a5191e1087f2a7152d72b5e280466e2d8e06a5802c8adc80672e76b`: the exact fast-prefill build of [INKFASTPRE-01](../model-experiments/2026-11-22-inkling-mini-exact-fast-prefill.md), with the first fusion pass and the long-position fix, before the unguarded MXFP8 default and the INKFUSE-02 single-row fusions | Prepared original checkpoint, single shard plus `mtp.safetensors` |
| mlx-original | MLX-Serve `05983e1278d6253d3648ddaa2514ec2a2f73ff3f617d41412d18334759991392` | Original checkpoint (`config.json` `8216dd43…`) |
| mlx-author-8bit | Same MLX-Serve binary | Author 8-bit pack, four shards, `config.json` `2b328150…`, index `6280a783…` |

The 8-bit pack is a different artifact: 8-bit affine attention, embedding and head, original expert bytes. Its figures do not transfer to the original checkpoint. MLX-Serve forces decode length to 192 tokens; Stark follows the model's own stop, so decode work can differ between engines.

**Results.** Median tok/s with ranges; tokens per step in brackets for decode rows.

| Scenario | Input tokens | n | Stark | MLX-Serve original | MLX-Serve 8-bit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial coding | 71 | 3 | 79.7 (77.7–81.7) [2.37] | 79.3 (78.7–83.0) [1.73] | 93.3 (92.7–95.4) [1.37] |
| Sustained coding | 72 | 3 | 78.6 (76.5–80.4) [2.23] | 82.8 (82.7–88.2) [1.47] | 101.3 (99.3–104.9) [1.57] |
| Novel prose | 40 | 3 | 62.0 (60.9–62.8) [2.00] | 71.7 (71.0–71.8) [1.18] | 89.0 (87.9–89.3) [1.19] |
| Context about 512 | 498 | 1 | 77.6 [2.59] | 78.2 [1.64] | 91.6 [1.39] |
| Counting ceiling, 512 | 451 | 1 | 70.9 [2.43] | 71.8 [1.34] | 88.3 [1.81] |
| Context about 4K | 4,019 | 1 | 98.2 [3.20] | 85.1 [2.37] | 113.1 [3.43] |
| Counting ceiling, 4K | 3,972 | 1 | 95.0 [3.10] | 86.2 [2.63] | 87.0 [1.51] |
| Context about 8K | 8,252 | 1 | 99.0 [3.31] | 97.2 [3.15] | 98.7 [2.49] |
| Counting ceiling, 8K | 8,205 | 1 | 91.3 [2.95] | 98.4 [3.05] | 95.5 [1.85] |
| Context about 16K | 16,335 | 1 | 87.0 [3.10] | 87.4 [3.10] | 93.5 [2.43] |
| Counting ceiling, 16K | 16,288 | 1 | 82.6 [2.91] | 83.3 [2.40] | 91.4 [1.75] |
| Repeated-sentence prefill | 8,768 | 3 | 1,224.4 (1,223.4–1,239.1) | 2,051.0 (2,051.0–2,053.9) | 2,079.7 (2,069.4–2,087.1) |
| Unseen-text prefill | 7,514–7,586 | 3 | 1,237.5 (1,226.3–1,239.7) | 2,090.7 (2,080.6–2,094.0) | 2,124.9 (2,103.3–2,129.2) |
| Exact 8,800-ID prefill, engine phase | 8,800 | 1 | 1,194.4 | 2,235.7 | 2,225.0 |
| Exact 8,800-ID prefill, client | 8,800 | 1 | 1,194.1 | 2,234.7 | 2,224.0 |

The 8,800-ID control's input IDs were identical across arms. Stark reused 2,048 cached tokens at the 4K counting-ceiling rung and 14,336 at the 16K rung by the scenario's design; MLX-Serve reported 9 cached tokens on novel prose. Stark's acceptance per proposal ranged from 0.33 (novel prose) to 0.76 (about 8K).

**Reading the results.**

- Against MLX-Serve on the same original checkpoint, Stark is level on initial coding and at the 512, 8K and 16K context rungs, ahead at about 4K (98.2 against 85.1), and behind on sustained coding, novel prose, the 8K counting-ceiling rung and every prefill scenario (about 0.53 to 0.60 of MLX-Serve's rate).
- The author 8-bit pack is the fastest arm on most decode rows, up to 1.33 times MLX-Serve original at about 4K. Its prefill is level with MLX-Serve original. The published 116 tok/s headline on M4 Max was not reproduced on this M5 Max; the best 8-bit cell here is 113.1 at about 4K, and its sustained coding is 101.3.
- Stark's prefill moved from 837 to 1,224 tok/s on the repeated-sentence scenario between the two runs, with a different Stark build; the runs are separate sessions, so that change is indicative only. The paired prefill evidence is in the INKFASTPRE records.
- The worker's 1,194 tok/s at 8,800 tokens sits well below the same tree's engine diagnostic; the [prefill attribution](2026-11-22-inkling-mini-prefill-attribution.md) explains most of that gap.
