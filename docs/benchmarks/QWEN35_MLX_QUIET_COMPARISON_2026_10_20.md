# Qwen3.5-2B: Stark versus MLX-LM under observed quiet conditions

Measured on 2026-10-20 local time (2026-10-21 UTC). Internal follow-up to the [MLX optimization experiment](MLX_GENERATION_OPTIMIZATION_2026_10_20.md). No engine, model or worker implementation changed for this follow-up.

## Result

Optimized Stark had **7.36% higher end-to-end generation throughput** than pinned, unmodified MLX-LM across four matched model-load pairs that passed the frozen host guards. Each performance cell contains 20 requests: five repeats in each of four independent model loads. Stark led on all six fixtures in all four pairs. The six-fixture overall lead ranged from **6.09% to 7.95%** across pairs.

The planned campaign contained six pairs. Pair 0 lost its Stark load to concurrent `pnpm` CPU activity; pair 1 lost its MLX load to system pageouts. Only pairs **2, 3, 4 and 5** enter the comparison below. Their order remains balanced: Stark first twice and MLX first twice. Whole-campaign `timing_accepted` is **false**. The four accepted pairs support this bounded internal result; the campaign does not qualify as wholly isolated.

| Fixture | Input / output tokens | Stark tok/s | MLX-LM tok/s | Stark advantage |
| --- | ---: | ---: | ---: | ---: |
| Fibonacci module | 42 / 128 | 358.2 | 321.5 | 11.40% |
| GPU explanation | 40 / 128 | 358.7 | 322.1 | 11.34% |
| Copy edit | 383 / 128 | 330.4 | 305.3 | 8.23% |
| Medium context | 522 / 128 | 315.0 | 296.7 | 6.13% |
| Long context | 2118 / 128 | 238.8 | 230.6 | 3.53% |
| Long output | 50 / 512 | 344.2 | 331.6 | 3.83% |

Rates in this table are the geometric means of the four within-load median rates. Each rate is actual generated output tokens divided by complete client request time, including prefill, decode, IPC/readout and request cleanup. Loading, warmup and quiet admission are outside timing. The overall figure is the geometric mean of all 24 paired fixture/load rate ratios, with equal fixture weight. It is a throughput advantage; MLX's percentage deficit uses a different denominator.

The longest-context fixture approached parity in one pair: Stark's per-load advantage ranged from 0.20% to 5.45%. The long-output advantage ranged from 1.59% to 5.98%. These ranges describe the observed four pairs, not confidence intervals or a broad engine ranking.

## Interference identified and actions taken

The earlier monitor retained maximum CPU values without process names or phase timestamps. Its old exclusions cannot be attributed retrospectively. The new monitor reproduced and attributed interference directly:

| Contributor | Observed evidence | Action / disposition |
| --- | --- | --- |
| Three stale repository diagnostic loops | About 15 hours old, repeatedly launching `ps`; their intended 30-second deadline reused a timestamp reset every iteration | Verified command hashes, repository working directory and diagnostic code before stopping those three processes. All exited. |
| Apple malware scanning | `XProtectRemediatorColdSnap` and `XProtectRemediatorSheepSwap` peaked at 99.1% and 92.5% CPU in initial idle screening | Waited; left security services running. Initial admission failed. |
| Codex renderer | Reproduced 660.2% CPU in a timed Stark request in the first attributed attempt | Preserved the interrupted attempt. Hid application windows through macOS System Events and verified visibility was false before subsequent attempts. Restored visibility after measurement. |
| Concurrent Rust compilation | `rustc` peaks of 138–161% CPU contaminated multiple hidden-window loads | Preserved the interrupted attempt. A separate 40-second investigation did not find a surviving compiler to identify its owner. A later continuous 60.87-second quiet preflight passed. |
| Concurrent JavaScript tooling | `pnpm` reached 167.2% CPU, aggregate external CPU 317.6%, in final Stark load 0 | Excluded that load and its pair. Its owner was not established. |
| System paging | Final MLX load 1 recorded 40 pageouts and zero swapouts, with clean CPU/thermal guards | Excluded that load and its pair under the predeclared zero-pageout rule. The counters are system-wide; no worker-level paging attribution is claimed. |

CPU percentages use 100% for one core. The stale diagnostics were an avoidable task-owned cleanup failure. The replacement monitor has an explicit stop event and joins its thread; the driver handles SIGINT/SIGTERM through owned worker and sleep-assertion cleanup. Regression tests cover stale quiet samples, PID reuse/new processes, descendant ownership, phase-boundary guards and quiet-streak resets.

The final campaign ran for 599.39 seconds. In its eight paired accepted loads, 249 measurement intervals recorded a maximum single external process load of 83.58%, maximum aggregate external CPU of 170.60%, and maximum sampling interval of 0.635 seconds. Thermal state remained nominal; pageout and swapout deltas were zero. Whole-window sleep screening succeeded with zero sleep events. All 360 performance requests across the complete campaign reached their requested token limits; no early-stop work was used for the speed table.

## Artifacts and controls

- Host: Apple M5 Max, 40 GPU cores, 128 GiB, macOS 26.7.1.
- Stark: ordinary Rust 1.98.1 locked release build from `045d0ca6cca8eafae89f32a303e5123f001eaf43`, including completion-gated scratch-pool reuse and the narrow GEMV specialization. Executable SHA-256: `66c0aae91af87dadcfb461de536b27479822ea446ffddc78897e5bd98301ba89`.
- MLX-LM: unmodified upstream `72be8c955d9a0d3099bd01865af288e0e947c3fa`, version `0.32.3.dev6+g72be8c955`, MLX/MLX-Metal 0.32.4, Python 3.13.13. Existing task-owned locked environment; the literal venv interpreter path was retained.
- Model: `mlx-community/Qwen3.5-2B-MLX-4bit@93760be4f1f69842a46bc13dbdc0f19e291392a3`, affine W4/G64. Source weights SHA-256 `713fe7e5d3c3965f7106b0d0ee17615f7869c23c8d327996df8c1196fbcf07d5`; unchanged converted Stark weights `a9e01fbf845a1d2af4727f4394b244815abd26b25d186bc49730249fb16c840d`.
- Context 8192, identical source-tokenizer input IDs, stock chat template with thinking disabled, greedy sampling, seed 0, fresh request state, one request in flight and one task-owned model worker loaded at a time. Existing unrelated idle model services were left running.
- Warm all ten fixtures once per model load, then require ten seconds of fresh quiet samples before timing. Repeat the quiet check before loading. Timed guards reject external process CPU above 100%, aggregate external CPU above 200%, recognized model activity above 10%, non-nominal thermal state, monitor failure/gaps, or any system pageout/swapout increase.

The [frozen protocol](../archive/experiments/clean-qwen-2026-10-20/PROTOCOL.md) describes these guards and the change from the older load-wide screen. The final driver was frozen at `d29c30f7`; [provenance](../archive/experiments/clean-qwen-2026-10-20/PROVENANCE.json) binds its exact source hash, workers, locks, executables and model artifacts.

## Correctness and interpretation

All ten fixtures had stable output-token and decoded-text hashes within each engine across all six loads, including rejected timings. Between engines, token hashes matched on 6/10 fixtures and decoded-text hashes on 5/10. Among performance fixtures, Fibonacci and copy edit had identical token and text hashes in both engines; the other four differed. Fixed token counts establish comparable amounts of generation work, not answer-quality equivalence.

Both engines passed only sorting among the four strict sanity cases. Arithmetic, prime and Python failed the exact-reply checks in every load. Their outputs were short (3, 1 and 2 tokens respectively), so these failures were not caused by hitting the 64-token cap. The aggregate receipts do not establish whether the failures were semantic or formatting failures. No response payloads were inspected or retained for this follow-up, and no general quality claim follows from the speed measurements.

This comparison establishes where the **optimized Stark build** stood against the **pinned MLX build** on these six requests under the observed accepted conditions. It does not measure the optimization's isolated contribution: original Stark was not rerun here. Earlier before/after results and historical MLX timings retain their separate sources and exclusions. The changed warmup/quiet procedure and improved host conditions prevent attributing the difference from those older tables solely to code changes.

Device GPU utilization is sampled for idle admission and diagnostics. It cannot attribute competing GPU work during requests. Per-process GPU/power monitoring required unavailable privileges, and nominal thermal state does not establish equal clock frequencies. Sampling can miss very short processes; basename matching can miss Python-hosted model work. These limits, four accepted pairs and the narrow workload set leave publication qualification open. JevK5, longer contexts, other GPUs, concurrency, isolated decode rate and general quality remain outside this result.

## Evidence and replay

The [aggregate packet](../archive/experiments/clean-qwen-2026-10-20) preserves every attempt separately, [attempt dispositions](../archive/experiments/clean-qwen-2026-10-20/attempts.json), the [completed receipt](../archive/experiments/clean-qwen-2026-10-20/final-clean-results.json), [derived summary](../archive/experiments/clean-qwen-2026-10-20/summary.json), configuration template and checksums. Only scalar measurements, process basenames, counts and hashes are retained. No prompts, responses, token vectors, logits, model weights or full command arguments are committed.

Resolve the placeholders in the private configuration template using the pinned artifacts above, then run through the existing locked stock-benchmark Python environment:

```sh
python scripts/benchmark_clean_qwen.py --config PRIVATE_CONFIG --output NEW_RECEIPT_JSON
python docs/experiments/clean-qwen-2026-10-20/summarize.py --receipt NEW_RECEIPT_JSON --output NEW_SUMMARY_JSON
```

Use fresh output paths for each replay. The summarizer reuses the source-hashed paired analysis from the earlier experiment and refuses to overwrite an existing summary. To verify the saved packet, pass its `final-clean-results.json` as `--receipt` and a fresh temporary output path, then compare the resulting summary with the saved one. No article was published.
