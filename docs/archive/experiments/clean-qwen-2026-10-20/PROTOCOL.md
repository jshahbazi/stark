# Quiet-window Qwen3.5-2B comparison

This protocol is frozen before the measurement campaign. It follows the [MLX optimization experiment](../../MLX_GENERATION_OPTIMIZATION_2026_10_20.md). The earlier campaigns remain unchanged, including their timing exclusions.

## Engines and work

- Compare the optimized default Stark release binary built from `045d0ca6cca8eafae89f32a303e5123f001eaf43` with unmodified MLX-LM `72be8c955d9a0d3099bd01865af288e0e947c3fa`, MLX 0.32.4. Reuse the existing locked runtimes and binaries; no engine or model changes.
- Use the same Qwen3.5-2B affine W4/G64 checkpoint revision `93760be4f1f69842a46bc13dbdc0f19e291392a3` and its unchanged Stark conversion. Record executable, model, worker, harness and dependency-lock hashes.
- Six independent model loads per engine, five measured repeats per performance fixture per load. Rotate/reverse the two-engine order using the original algorithm; first position is balanced three times each. Preserve every load without replacement or selective reruns.
- Retain the original six performance fixtures and four sanity fixtures, exact source-tokenizer inputs, context 8192, greedy decoding, seed 0 and fresh request state. One request in flight and one task-owned model worker loaded at a time. Warm all ten fixtures once after loading; sanity fixtures have one measured repeat.
- Primary rate is actual output tokens divided by complete client request time, including prefill, decode, IPC/readout and request cleanup. Model loading, warmup and quiet admission are excluded. Fixed work requires the specified output limit and length termination. Sanity fixtures never enter the speed aggregate.
- Report paired-load median-rate ratios and their geometric mean across all six performance fixtures. Also show each fixture, per-load ratios, ranges and output-hash consistency. Differences between engines' greedy outputs remain explicit.

## Admission and rejection

The new driver separates loading, warmup, quiet admission, measurement and teardown. This changes the measurement protocol: the earlier driver rejected an entire load for an external CPU spike in any phase. It cannot retrospectively attribute those spikes. A load/warmup compiler peak in this new campaign is retained and is not by itself a rejection of subsequent request measurements.

- Before loading and again after full warmup, require ten seconds of newly observed, contiguous quiet samples. Poll approximately every 0.5 seconds. No stale samples from a prior quiet window may qualify. Fail admission after 120 seconds; retain the failure.
- Quiet samples require each external process at or below 80% CPU, aggregate external CPU at or below 150%, recognized model processes at or below 10%, device GPU utilization at or below 5%, and nominal `NSProcessInfo.thermalState` (0). A CPU percentage of 100 means one core.
- During measurement reject a load if an external process exceeds 100% CPU, aggregate external CPU exceeds 200%, a recognized model process exceeds 10%, thermal state leaves nominal, monitoring fails, a sample interval exceeds two seconds, or system pageout/swapout counters increase. These CPU limits are stricter than the earlier 200% per-process guard. Intervals crossing measurement boundaries count conservatively against measurement.
- Record external process basenames, ephemeral PIDs, CPU deltas, interval timestamps and phases. Check process start identity to avoid PID reuse artifacts; include CPU consumed by newly observed processes. Do not persist arguments, full paths, prompts, responses, token vectors, logits or exception dumps.
- Keep the owned idle-sleep assertion, whole-campaign sleep-event check and wall/monotonic discrepancy check. Whole-campaign timing acceptance requires every planned load to complete and pass the guards. Rejected subsets stay labeled; do not call a partially accepted campaign clean.
- Finish all builds, tests and agent investigations before timing. Leave unrelated processes running; wait for background scans to finish. Do not disable security services or terminate user applications.

## Limits

The host remains an interactive desktop. Device GPU utilization is a coarse idle-admission check and a diagnostic during requests; it cannot separate benchmark work from other GPU clients. Per-process GPU/power collection is unavailable without additional privileges. CPU sampling can miss processes that start and exit entirely between polls, and basename matching does not reliably identify Python-hosted models. Thermal state does not establish identical GPU frequencies. Resident idle model servers may retain memory; paging deltas are checked rather than treating historical swap allocation as current swapping. These controls support a bounded internal comparison, not a publication-ready hardware-isolation claim.

## Hidden-window campaign addendum

The first attributed attempt is retained as an interrupted diagnostic campaign. It reproduced a 660% CPU spike in `Codex (Renderer)` during Stark measurement and a separate five-page system pageout increase during MLX measurement. It is not the final comparison. Before the fresh six-pair campaign, hide the Codex application windows through macOS System Events, verify visibility is false, and restore their prior visibility afterward. An initial `NSRunningApplication.hide` call returned false; its setup attempt was immediately interrupted and retained separately. Preserve the same engines, fixtures, repeat counts and strict guards. Keep earlier attempts' aggregate receipts and dispositions separate; never combine their favorable cells with the fresh campaign. The harness also gains bounded SIGINT/SIGTERM handling so owned worker, monitor and sleep-assertion cleanup runs on interruption.
