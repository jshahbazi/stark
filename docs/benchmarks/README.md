# Benchmarks

Measured results, each bound to the source revision, build and host named inside it. This index says which result is current and which has been superseded. Raw protocols, result JSON and patches for the engine comparisons live under [archive/experiments](../archive/experiments/).

## Current result

**Generation throughput, Qwen3.5-2B 4-bit, Apple M5 Max, one request at a time (2026-10-22).** Stark `97f15f38` measured 245.5 tokens/s aggregate against 218.8 for MLX-LM, 174.7 for vLLM Metal, 147.9 for llama.cpp and 125.6 for TensorFold, and led all six workloads. Four independent loads per configuration, three repeats per workload, 432 requests. Rates include input processing, generation, transport and cleanup; loading and warmup are excluded. [Full report](QWEN35_ENGINE_COMPARISON_2026_10_22.md), [machine-readable results](../../website/public/benchmark-results.json).

**Conversion time and memory against Lalamo.** Collected across converter milestones; see [conversion benchmarks](CONVERSION_BENCHMARKS.md).

**Source-qualified OLMo 3 and Gemma 4 Q4 text (2026-10-29).** On Apple M5 Max, repaired Stark has a median paired combined-work rate 1.55% above MLX for OLMo and 2.15% below for Gemma. Relative to the numerically repaired C2 native baseline, paired request latency falls 19.46% and 3.07%. Four loads per engine, three measured requests per load, and matching 128-token hashes accompany the numerical qualification. Rates cover the complete 512-input/128-return request. See the [repaired comparison](OLMO3_GEMMA4_Q4_REPAIRED_2026_10_29.md) and [aggregate results](olmo3-gemma4-q4-repaired-2026-10-29.json).

**Native GLM-5.3-Flash and Qwen3.8-Flash-Next optimization (2026-10-28).**
GPU decode graphs measured Qwen at 22.94 tok/s versus 11.21 staged, and GLM at
9.04 versus 8.13, on Apple M5 Max. Five fresh requests per mode use the same
binary and retain matching completion hashes. See the
[decode graph study](GLM_QWEN_DECODE_GRAPH_2026_10_28.md). Earlier stage-chaining
and kernel measurements remain historical records for their named builds.

## Engine comparison history

| Study | Date | Status |
| --- | --- | --- |
| [Qwen3.5 engine comparison](QWEN35_ENGINE_COMPARISON_2026_10_22.md) | 2026-10-22 | Current: five engines, six workloads, 12.2% over MLX-LM |
| [TensorFold stock Qwen3.5 comparison](TENSORFOLD_STOCK_QWEN35_COMPARISON.md) | 2026-10-16 | Superseded by the 2026-10-22 comparison, which includes TensorFold |
| [TensorFold JevK5 scoring comparison](TENSORFOLD_JEVK5_COMPARISON.md) | 2026-10-16 | Scoring-path comparison; not repeated since |
| [Stock engine benchmark pilot](STOCK_ENGINE_BENCHMARK_2026_10_19.md) | 2026-10-19 | Pilot run; superseded by the 2026-10-22 comparison |
| [Qwen3.5 MLX quiet-window comparison](QWEN35_MLX_QUIET_COMPARISON_2026_10_20.md) | 2026-10-20 | Reported 7.36% over MLX-LM; superseded by the 2026-10-22 comparison |
| [Benchmark brief](STARK_BENCHMARK_BRIEF_2026_10_21.md) | 2026-10-21 | Summary of the 7.36% result; superseded |
| [Inference comparison evidence inventory](INFERENCE_COMPARISON_EVIDENCE.md) | rolling | Inventory of every recorded engine, reference and conversion result up to 2026-10-21 |
| [OLMo 3 and Gemma 4 Mac engine evidence](OLMO3_GEMMA4_MAC_ENGINES.md) | 2026-10-28 | Published benchmark evidence, comparison limits, source licensing and native implementation techniques |
| [Repaired OLMo 3 and Gemma 4 Q4 comparison](OLMO3_GEMMA4_Q4_REPAIRED_2026_10_29.md) | 2026-10-29 | Current for these pinned Q4 sources: all numerical gates pass; paired Stark/MLX rates 1.0155 and 0.9785; 128-token hashes agree |
| [Matched native OLMo 3 and Gemma 4 comparison](OLMO3_GEMMA4_NATIVE_MATCHED.md) | 2026-10-28 | Historical diagnostic: original failed numerical qualification and timings preserved; superseded for these checkpoints by the repaired 2026-10-29 comparison |

## Optimization records

| Study | Outcome |
| --- | --- |
| [Native GLM and Qwen decode graphs](GLM_QWEN_DECODE_GRAPH_2026_10_28.md) | Current original-format checkpoint results: Qwen 2.05x, GLM 11.1% faster warm decode on the short matched workload |
| [Native GLM and Qwen graph and kernels](GLM_QWEN_NATIVE_GRAPH_2026_10_28.md) | Earlier stage-chaining study; superseded throughput, source-bound validation retained |
| [Native GLM and Qwen optimization](GLM_QWEN_NATIVE_OPTIMIZATION_2026_10_28.md) | Initial no-copy and batching study; build identities and validation remain source-bound historical evidence |
| [MLX-informed generation optimization](MLX_GENERATION_OPTIMIZATION_2026_10_20.md) | Two decode changes accepted after bitwise validation; the throughput they produced is included in the current result |
| [DeepInfer resource reuse](DEEPINFER_ADOPTION_2026_10_21.md) | Two experiments measured and reverted; patches retained under the archive |
| [Qwen3 GGUF import experiment](QWEN3_GGUF_IMPORT_EXPERIMENT.md) | Experimental Q8_0 import route; see the [converter guide](../../tools/converter/README.md#experimental-qwen3-gguf-import) |
| [Qwen3 GGUF prefill optimization](QWEN3_GGUF_PREFILL_OPTIMIZATION.md) | About 15% higher prefill throughput for the imported model at 512 and 1,024 tokens, exact outputs on the tested cases |

## How to read these studies

Every study names its Stark commit, engine versions, model, quantization and host. Results apply to that build; they are not regenerated when the code changes. A new comparison should be added here with its date and should state which earlier rows it supersedes. Throughput studies do not measure answer quality; the 2026-10-22 report includes separate answer-format checks.
