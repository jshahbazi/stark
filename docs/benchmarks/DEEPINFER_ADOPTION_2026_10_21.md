# DeepInfer adoption study: no runtime change retained

Two portable ideas from DeepInfer were implemented in Stark, checked for correctness, and measured on the existing Qwen3.5-2B workload. Neither demonstrated a useful throughput gain. Both changes were reverted, and the rebuilt generation and scoring-digest executables exactly match the original reference hashes.

| Experiment | Accepted matched load pairs | Samples per performance fixture per engine | Geometric mean throughput change |
| --- | --- | --- | --- |
| Both changes together | 0, 2 | 10 | −1.24% |
| Command metadata reuse bundle | 2, 3 | 6 | +0.02% |
| Direct device token handoff | 0, 1, 2, 3 | 12 | −0.12% |

These are separate campaigns and accepted subsets under the predeclared host guards. Both whole-campaign receipts retain `timing_accepted: false`. The small isolated differences and mixed pair-level results establish no speed improvement; the combined result does not establish a stable regression magnitude. No new comparison against MLX-LM was run.

## Source and architectural fit

The study pins [DeepInfer `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9). Its root license is [Apache-2.0](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/LICENSE). No upstream code was executed or copied into Stark. The rejected Rust patches were independently written adaptations; existing Stark/Uzu notices remain intact.

At this revision, the [upstream README](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md) documents Linux execution for DeepSeek-V4.1-Flash on NVIDIA SM100 and Huawei Ascend 950. It supplies no Metal backend. DeepGEMM, FlashMLA, distributed MoE communication and hardware-specific low-precision kernels require a different device and model execution path from Stark's current Apple-silicon Qwen/Jev workloads.

| Upstream mechanism | Stark assessment | Disposition |
| --- | --- | --- |
| Reuse sampled device tensors as next-step inputs: [worker, lines 236–243](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/worker.py#L236-L243), [input preparation, lines 210–243](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/input_metadata/prepare_tensor_input_zos.py#L210-L243) | Stark retained the source allocation while still copying the ordinary next-token input into four bytes of scratch storage. | Implemented and measured; reverted. |
| Stable execution metadata: [graph runner, lines 93–110 and 139–154](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/cuda_graph_runner.py#L93-L154) | Stark already reused completed Metal command allocators/buffers. An argument table was still allocated for each encoding. | Implemented table recycling with completion handling; measured and reverted. |
| Submit current work before postprocessing previous work: [worker, lines 89–187](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/worker.py#L89-L187) | Stark already submits the next step before resolving the preceding one. | Already present. |
| Fuse common activation and normalization work | Stark already fuses residual/copy operations with RMSNorm and gate activation with multiplication. | No new adoption claim. |
| Confidence heads, YOCO/SWA replay and distributed expert scheduling | These depend on model structure or serving architecture absent from the measured Stark path. | Outside this experiment. |

The metadata adaptation used Metal's own resource model. Apple's [Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api) and [argument table](https://developer.apple.com/documentation/metal/mtl4argumenttable) documentation support reusable execution objects and resource bindings. This motivated an experiment; it predicted no measurable benefit.

## What was implemented and rejected

Reference commit: `8be4b52d770a526ebce8d05ddeb24d9c901403c9`, with engine code from `045d0ca6`. Combined candidate: `f2a5a9885fd8c5b8c2ad01de1cdad558e1a8acaf`. Restoration: `42ffa85d`.

The **direct token handoff** candidate passed the preceding sampled buffer directly to the next ordinary decode pass through `BufferRef`, retaining its `Arc<GlobalBuffer>` through consumer completion. It removed a four-byte scratch allocation/copy and related hazard bookkeeping. Grammar and speculative decoding kept the original copy path. Review checked producer/consumer queue ordering, buffer ownership and stream teardown.

The **command metadata bundle** retained a Metal argument table alongside each command allocator/buffer through encoding, execution and completion. Outstanding encodings owned distinct tables. Successful completion returned resources to the cache before notifying the waiter; failed submissions were excluded. Active buffer and constant bindings were overwritten on each encoding. This bundle includes callback ordering and reference-count operations, so its timing cannot isolate argument-table allocation alone.

The two ablations used exact source-file compositions of the reference and combined candidate. Their [provenance records and rejected patches](../archive/experiments/deepinfer-adoption-2026-10-21/PROVENANCE.json) bind the file maps, patch hashes and executable hashes. The combined patch also retains the two candidate-specific Metal tests. No model, quantization, arithmetic kernel, sampling algorithm or locked dependency was changed.

The original runtime `.patch` artifacts used the machine's external display formatter. Their bytes and historical source fingerprints remain preserved. The separately named `*.unified.patch` files are standard applicable patches, checked in temporary Git indexes against the reference and verified to recreate the exact candidate source bytes. Use these unified patches for replay; their mappings and hashes are recorded in provenance.

## Correctness and build evidence

- Rust 1.98.1 locked release builds of generation, scoring and scoring-digest tools passed. Production builds used no bootstrap. The default engine check, including grammar, passed.
- Both new Metal tests passed: distinct tables for outstanding encodings, reuse after completion, and rebinding across changing buffers, constant offsets and optional-input specializations. Actual GPU failure was not injected; failed-completion cache exclusion was reviewed in source.
- One existing sampler property test passed with `PROPTEST_CASES=16`. The randomized seed was not captured.
- The combined scoring candidate matched **23/23 complete-vocabulary F32-bit digests**, covering ten fixture inputs and thirteen boundary-prefix lengths with scoring caches disabled. This receipt covers the combined scoring candidate; it does not independently validate each ablation or exercise generation token handoff.
- All ten generation fixtures had identical output-token and decoded-text hashes across the reference, combined candidate and isolated builds within their respective campaigns, including timing-excluded loads. Each engine's hashes were stable across all loads.
- Formatting and whitespace checks passed. An initial candidate-test compilation error involving pointer type inference was corrected before the successful test run.

The [numerical receipt](../archive/experiments/deepinfer-adoption-2026-10-21/numerics.json) and [validation record](../archive/experiments/deepinfer-adoption-2026-10-21/validation.json) preserve the scope and commands. Hash parity establishes preservation of the reference's behavior on these cases. It provides no broader answer-quality qualification; the earlier stock sanity limitations remain in the [Qwen comparison report](QWEN35_MLX_QUIET_COMPARISON_2026_10_20.md).

## Measurement and exclusions

Host: Apple M5 Max, 40 GPU cores, 128 GiB, macOS 26.7.1. All runs used the same `mlx-community/Qwen3.5-2B-MLX-4bit` revision `93760be4f1f69842a46bc13dbdc0f19e291392a3` and unchanged affine W4/G64 Stark conversion. Context was 8192, greedy sampling with seed 0, fresh request state, one loaded worker and one in-flight request. Builds, tests and agent investigations finished before measurement. Codex windows were hidden for each campaign and restored afterward; unrelated services were left running.

The unchanged clean-comparison harness warmed all ten fixtures and required ten seconds of fresh quiet admission before loading and after warmup. Timed guards rejected external process CPU over 100%, aggregate external CPU over 200%, recognized background model CPU over 10%, non-nominal thermal state, monitoring failure or gaps over two seconds, any system pageout/swapout increase, and unequal performance output work. The [frozen protocol](../archive/experiments/deepinfer-adoption-2026-10-21/PROTOCOL.md) and [separately frozen ablation protocol](../archive/experiments/deepinfer-adoption-2026-10-21/ABLATION_PROTOCOL.md) define the full rules.

Throughput is actual generated tokens divided by complete client request seconds, including prefill, decode, IPC/readout and request cleanup. Loading, warmup and admission are excluded. The calculation takes a median within each load/fixture, divides candidate by reference for matched accepted loads, and takes the geometric mean across the six performance fixtures and accepted pairs. Four sanity fixtures are excluded from the throughput aggregate. Engine order was pairwise balanced, including the retained matched subsets.

The combined campaign ran six loads per engine, five repeats per performance fixture and one per sanity fixture: **408 measured requests**, including 360 performance requests, in 674.07 seconds. Only matched pairs 0 and 2 passed both engines' guards.

| Excluded combined load | Observed cause |
| --- | --- |
| Candidate 1 | 39 system pageouts |
| Reference 3 | Codex renderer CPU 268.47% |
| Candidate 3 | 63 system pageouts |
| Reference 4 | Codex renderer CPU 271.88% |
| Candidate 5 | Non-nominal thermal state |

The isolated campaign ran four loads per engine, three repeats per performance fixture and one per sanity fixture: **264 measured requests**, including 216 performance requests, in 456.37 seconds. All reference and token-handoff loads passed. Metadata loads 0 and 1 were excluded for one and seventeen system pageouts respectively; the other host-screen flags were clear. No excluded load is silently removed from the receipts.

| Matched block | Combined candidate change | Metadata bundle change | Token handoff change |
| --- | --- | --- | --- |
| 0 | −2.4404% | Excluded | −0.5213% |
| 1 | Excluded | Excluded | +0.1446% |
| 2 | −0.0192% | −0.0370% | +0.0753% |
| 3 | Excluded | +0.0821% | −0.1572% |

The combined and isolated columns come from separate campaigns. The [derived analysis](../archive/experiments/deepinfer-adoption-2026-10-21/analysis.json) includes per-fixture results, retained sample counts and all hash comparisons. Both campaigns had zero observed sleep events; wall/monotonic clock discrepancies were −0.0202 and −0.00042 seconds respectively.

These guards describe observed quiet conditions. Device-wide GPU utilization cannot attribute competing GPU clients during a request, per-process GPU/power monitoring was unavailable, and nominal thermal state does not prove equal clock frequencies. Small accepted-subset changes cannot qualify a generally faster engine. The earlier Stark-versus-MLX result remains separate.

## Remaining ideas worth a bounded investigation

Two source-backed candidates remain proposals:

1. **Bounded stochastic filtering.** DeepInfer's [CUDA sampling implementation](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cuda/fused_sample.py#L35-L150) uses radix threshold selection. Stark's [general stochastic sampler](../../crates/stark-engine/src/backends/metal/kernel/sampling/unified_sampling.metal) can iterate candidate rejection up to 64 times. A Metal adaptation would need to preserve Stark's Philox/Gumbel draws, top-k/top-p semantics and lower-token-ID tie behavior, then demonstrate improvement on difficult stochastic distributions. The RNG implementations differ. No greedy generation or scoring gain follows from this observation; compiler specialization may already remove unused greedy work.
2. **Retain normalization inputs in local storage.** DeepInfer's [normalization kernel](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cuda/norm.py#L187-L269) keeps input fragments locally. Stark's [RMSNorm source](../../crates/stark-engine/src/backends/metal/kernel/normalization/normalization.metal) reads inputs around its reduction. Inspect generated Metal code first: the compiler may already retain these values, and the input is small/cache-resident. Any specialization must preserve reduction order and BF16 rounding, pass bitwise checks, and earn retention through measurements.

Neither proposal is implemented or a demonstrated improvement. The tested resource changes offer no evidence-based reason to add runtime complexity to Stark.

## Evidence and replay

The [evidence packet](../archive/experiments/deepinfer-adoption-2026-10-21/PROVENANCE.json) pins source, model, binary, patch and tool hashes. [SHA256SUMS.json](../archive/experiments/deepinfer-adoption-2026-10-21/SHA256SUMS.json) covers the packet. Receipts contain aggregate measurements, counts, hashes and process basenames. Prompts, generated text, token vectors, logits, binaries and models remain outside Git. Configuration templates replace local paths with explicit placeholders.

To recompute the published tables from the checked-in aggregate receipts, use the existing locked benchmark environment:

```sh
tools/stock_benchmark/mlx/.venv/bin/python \
  docs/experiments/mlx-optimization-2026-10-20/analyze.py \
  docs/experiments/deepinfer-adoption-2026-10-21/combined-results.json \
  docs/experiments/deepinfer-adoption-2026-10-21/ablation-results.json \
  --output runtime/deepinfer-reanalysis.json
```

For a new measurement, materialize the [combined](../archive/experiments/deepinfer-adoption-2026-10-21/combined-config.template.json) or [ablation](../archive/experiments/deepinfer-adoption-2026-10-21/ablation-config.template.json) template with fresh ignored runtime paths and the recorded model/builds, then run `scripts/benchmark_clean_qwen.py --config <configuration.json> --output <receipt.json>` under the frozen protocol. Rebuilding a rejected candidate requires its recorded reference plus patch in an isolated checkout. The successful production build command was:

```sh
cargo +1.98.1 build --release --locked -p stark-generate -p stark-score \
  --bin stark-generate --bin stark-score --bin stark-score-digest
```

Production engine, generation and scoring source trees and `Cargo.lock` match the original reference exactly. The restored generation executable SHA-256 is `66c0aae91af87dadcfb461de536b27479822ea446ffddc78897e5bd98301ba89`; the restored scoring-digest SHA-256 is `c8d40289fcbb958fd6c385e43fd94804254fcea337bf38ce4d158f786a7ce460`. Task workers exited and prior window visibility was restored, as recorded in the [cleanup receipt](../archive/experiments/deepinfer-adoption-2026-10-21/process-cleanup.json).
