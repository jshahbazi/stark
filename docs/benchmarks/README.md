# Benchmarks

Measured results, each bound to the source revision, build and host named inside it. This index says which result is current and which has been superseded. Earlier archived comparison artifacts are under [archive/experiments](../archive/experiments/). Each report states where its source-bound aggregates are retained.

## Current result

**Qwen3.8, identical checkpoint, maximum sampled throughput (2026-10-30).**
Stark peaks at **66.81 tok/s** versus **61.17** for stock MLX-Serve 26.10.3,
**9.21% higher**, with the same verified files, quantization and no speculation.
Ten measured 4K/256 requests per engine ran while the user reserved the machine
and confirmed other apps were closed. Exact component controls and retained
Lily output parity pass. Nine aligned next-token choices match; longer greedy
continuations still differ, so whole-model numerical and quality equivalence
remain unestablished. The [optimization record](QWEN38_MATCHED_OPTIMIZATION_2026_10_30.md)
records the 389-check aggregate audit, timing conventions and machine-readable
results. The earlier [58.13/60.95 baseline](QWEN38_SAME_CHECKPOINT_STARK_MLX_SERVE_2026_10_30.md)
is retained unchanged.

**Earlier Qwen3.8 comparison, different quantization profiles (2026-10-30).**
The Lily-format Stark checkpoint peaked at 79.52 tok/s against 61.00 for the
official MLX pack. Its [source-bound report](QWEN38_STARK_MLX_SERVE_2026_10_30.md)
retains pooled rates and block drift. Its two quantization profiles and
measurement window are separate from the current fixed-checkpoint comparison.

**Generation throughput, Qwen3.5-2B 4-bit, Apple M5 Max, one request at a time (2026-10-22).** Stark `97f15f38` measured 245.5 tokens/s aggregate against 218.8 for MLX-LM, 174.7 for vLLM Metal, 147.9 for llama.cpp and 125.6 for TensorFold, and led all six workloads. Four independent loads per configuration, three repeats per workload, 432 requests. Rates include input processing, generation, transport and cleanup; loading and warmup are excluded. [Full report](QWEN35_ENGINE_COMPARISON_2026_10_22.md), [machine-readable results](../../website/public/benchmark-results.json).

**Conversion time and memory against Lalamo.** Collected across converter milestones; see [conversion benchmarks](CONVERSION_BENCHMARKS.md).

**Native GLM-5.3-Flash and Qwen3.8-Flash-Next integration (2026-10-28).**
Qwen's initial mixed-precision Lily-format path measures 77.18 tok/s warm decode at 4K
input and 256 outputs, with speculation disabled. The original Pipe checkpoint
improves from 23.16 to 43.70 tok/s on its matched short F32 workload. See the
[mixed-precision study](QWEN38_MIXED_PRECISION_2026_10_28.md) for distinct artifacts,
precision policies and source validation. GLM's current 9.04 versus 8.13 tok/s
result remains in the [decode graph study](GLM_QWEN_DECODE_GRAPH_2026_10_28.md).
Earlier Qwen throughput and kernel measurements remain historical records for
their named builds.

**Qwen3.8 sampled generation at fixed quantization (2026-10-28).** Repeated
4K-input/256-output runs measure top-k 20 sampling at 75.08 tok/s versus 50.62
for the preserved worker, and nucleus-only sampling at 65.57 versus 49.73.
Both use temperature 1, top-p 0.95, seed 42 and the same Lily q4-xl checkpoint,
without speculation. All output hashes match within each sampling mode. The
[sampled BF16 study](QWEN38_SAMPLED_BF16_2026_10_28.md) retains both measurement
windows, profiles GPU/CPU stages and records exact sampler validation.

**Qwen3.8 HC and GDN kernels (2026-10-29).** Fixed-quantization paired component
controls reduce HC time by 5.40%, GDN input projection/convolution by 5.62%, and
the recurrent output wrapper by 1.15%. Retained routes match full-logit and
seeded sampled references. MoE trials are rejected. The [kernel study](QWEN38_KERNELS_2026_10_29.md)
records every sampled stage and separates component gains from unstable
between-load throughput measurements. A private same-loaded-model control
observes +1.93% pooled top-k 20 decode and effectively unchanged nucleus
throughput; earlier source-bound records are retained.

**Qwen3.8 sampled throughput during desktop activity (2026-10-29).** Real-readout
paired controls reduce top-20 selector time by 73.5% and full-vocabulary ordering
by 43.2%, preserving complete candidate orders and seeded outputs. The
[stability study](QWEN38_STABILITY_2026_10_29.md) records same-loaded-model
activity/QoS/feedback-queue controls and correlates throughput swings with
concurrent GPU work. These controls do not establish consistent 77–80 tok/s
during normal desktop multitasking. Earlier kernel and throughput records
remain source-bound historical evidence.

**Qwen3.8 AC power-mode control (2026-10-29).** The same worker measures 72.88,
66.79 and 65.41 pooled top-k 20 tok/s across sequential Automatic, High Power
selected, and restored Automatic phases. Desktop GPU load and clocks vary
materially, so the [power-mode study](QWEN38_POWER_MODES_2026_10_29.md) does not
isolate a causal mode effect. All 60 outputs match their references, the
77–80 tok/s target remains unmet, and the original preferences are restored.

## Engine comparison history

| Study | Date | Status |
| --- | --- | --- |
| [Qwen3.8 matched-checkpoint optimization](QWEN38_MATCHED_OPTIMIZATION_2026_10_30.md) | 2026-10-30 | Current: same files and quantization, no speculation; maximum 66.81 vs 61.17 tok/s (+9.21%); numerical equivalence unestablished |
| [Qwen3.8 same-checkpoint maximum throughput baseline](QWEN38_SAME_CHECKPOINT_STARK_MLX_SERVE_2026_10_30.md) | 2026-10-30 | Historical baseline before the matched-checkpoint optimization: maximum 58.13 vs 60.95 tok/s; source-bound evidence retained |
| [Earlier Qwen3.8 Stark / MLX-Serve comparison](QWEN38_STARK_MLX_SERVE_2026_10_30.md) | 2026-10-30 | Historical different-pack comparison: maximum 79.52 vs 61.00 tok/s; does not isolate engine efficiency |
| [Qwen3.5 engine comparison](QWEN35_ENGINE_COMPARISON_2026_10_22.md) | 2026-10-22 | Current: five engines, six workloads, 12.2% over MLX-LM |
| [TensorFold stock Qwen3.5 comparison](TENSORFOLD_STOCK_QWEN35_COMPARISON.md) | 2026-10-16 | Superseded by the 2026-10-22 comparison, which includes TensorFold |
| [TensorFold JevK5 scoring comparison](TENSORFOLD_JEVK5_COMPARISON.md) | 2026-10-16 | Scoring-path comparison; not repeated since |
| [Stock engine benchmark pilot](STOCK_ENGINE_BENCHMARK_2026_10_19.md) | 2026-10-19 | Pilot run; superseded by the 2026-10-22 comparison |
| [Qwen3.5 MLX quiet-window comparison](QWEN35_MLX_QUIET_COMPARISON_2026_10_20.md) | 2026-10-20 | Reported 7.36% over MLX-LM; superseded by the 2026-10-22 comparison |
| [Benchmark brief](STARK_BENCHMARK_BRIEF_2026_10_21.md) | 2026-10-21 | Summary of the 7.36% result; superseded |
| [Inference comparison evidence inventory](INFERENCE_COMPARISON_EVIDENCE.md) | rolling | Inventory of every recorded engine, reference and conversion result up to 2026-10-21 |

## Optimization records

The [Qwen3.8 optimization history](QWEN38_OPTIMIZATION_HISTORY.md) catalogs 142
attempts and controls across the original F32 path, mixed precision, sampling,
Metal kernels, Uzu execution and matched-checkpoint work. It records retained,
rejected, inconclusive and deferred outcomes with source and evidence links.

| Study | Outcome |
| --- | --- |
| [Qwen3.8 matched-checkpoint Metal optimization](QWEN38_MATCHED_OPTIMIZATION_2026_10_30.md) | Same-checkpoint maximum 66.81 vs 61.17 tok/s (+9.21%), no speculation; exact stock component controls and retained Lily parity; 77–80 tok/s and whole-model quality equivalence remain unestablished |
| [Qwen3.8 converted-format speed](QWEN38_UZU_SPEED_2026_10_30.md) | NEON affine restoration reduces median process-to-ready time from 20.54 to 18.09 seconds; exact nucleus-prefix work and sampled controls retain quantization; consistent 77–80 tok/s remains unqualified |
| [Qwen3.8 Uzu conversion and generation](QWEN38_UZU_FORMAT_2026_10_29.md) | Full quantization-preserving conversion, self-contained load and exact readout/sample parity; paired timing series has substantial throughput drift |
| [Qwen Metal hotpaths and finite readout checks](QWEN38_METAL_HOTPATHS_2026_10_29.md) | Quiet baseline 73.62 top-k / 69.85 nucleus tok/s; MoE, HC and GDN trials rejected or inconclusive; retained exact SIMD finite checks reduce guard elapsed time, with no reliable end-to-end gain or desktop target acceptance |
| [Qwen residual fusion and exact CPU sampling](QWEN38_FUSION_2026_10_29.md) | MoE down/injection component 5.55% faster; exact probability mapping 58.7% faster; nucleus same-loaded observation +1.3%, with broad overlap and an unmet 77–80 tok/s target |
| [Qwen iQ-MLX 4.7 bpw investigation](QWEN38_IQ47_INVESTIGATION_2026_10_29.md) | Exact pack unmeasured: storage and loader support pending; publisher's M5 Ultra result uses MTP; fresh existing-checkpoint Stark control is 72.08 greedy / 64.92 sampled warm tok/s |
| [Qwen AC power-mode control](QWEN38_POWER_MODES_2026_10_29.md) | Same inference build; 60 exact requests; selected High Power does not recover the target in this window; original settings restored |
| [Qwen desktop throughput stability](QWEN38_STABILITY_2026_10_29.md) | Current sampler and host diagnostics; exact selector gains; normal-desktop 77–80 tok/s acceptance not established |
| [Qwen HC and GDN kernels](QWEN38_KERNELS_2026_10_29.md) | Current kernels; exact references and component gains; same-loaded-model top-k observation +1.93%, nucleus unchanged; between-load throughput remains unstable |
| [Qwen sampled generation and BF16 profiling](QWEN38_SAMPLED_BF16_2026_10_28.md) | Sampler/scheduling baseline at fixed quantization: repeat-window warm gains of 48.33% with top-k 20 and 31.84% without top-k; exact seeded outputs |
| [Qwen mixed-precision native inference](QWEN38_MIXED_PRECISION_2026_10_28.md) | Initial Lily integration: 77.18 tok/s at 4K without speculation; original Pipe/F32 checkpoint 43.70 tok/s on the matched short workload |
| [Native GLM and Qwen decode graphs](GLM_QWEN_DECODE_GRAPH_2026_10_28.md) | Current GLM result; earlier Qwen throughput superseded, source-bound validation retained |
| [Native GLM and Qwen graph and kernels](GLM_QWEN_NATIVE_GRAPH_2026_10_28.md) | Earlier stage-chaining study; superseded throughput, source-bound validation retained |
| [Native GLM and Qwen optimization](GLM_QWEN_NATIVE_OPTIMIZATION_2026_10_28.md) | Initial no-copy and batching study; build identities and validation remain source-bound historical evidence |
| [MLX-informed generation optimization](MLX_GENERATION_OPTIMIZATION_2026_10_20.md) | Two decode changes accepted after bitwise validation; the throughput they produced is included in the current result |
| [DeepInfer resource reuse](DEEPINFER_ADOPTION_2026_10_21.md) | Two experiments measured and reverted; patches retained under the archive |
| [Qwen3 GGUF import experiment](QWEN3_GGUF_IMPORT_EXPERIMENT.md) | Experimental Q8_0 import route; see the [converter guide](../../tools/converter/README.md#experimental-qwen3-gguf-import) |
| [Qwen3 GGUF prefill optimization](QWEN3_GGUF_PREFILL_OPTIMIZATION.md) | About 15% higher prefill throughput for the imported model at 512 and 1,024 tokens, exact outputs on the tested cases |

## How to read these studies

Every study names its Stark commit, engine versions, model, quantization and host. Results apply to that build; they are not regenerated when the code changes. A new comparison should be added here with its date and should state which earlier rows it supersedes. Throughput studies do not measure answer quality; the 2026-10-22 report includes separate answer-format checks.
