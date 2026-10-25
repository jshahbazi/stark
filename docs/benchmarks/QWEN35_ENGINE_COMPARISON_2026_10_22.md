# Qwen3.5-2B: Stark and four other Mac inference engines

Measured 2026-10-22 on an Apple M5 Max (40 GPU cores, 128 GiB), macOS 26.7.1, one request at a time.

Stark recorded 245.5 end-to-end tokens/s against MLX-LM's 218.8, a 12.2% higher observed aggregate. Stark recorded the highest rate in each of the six workloads in this run.

## Results

| Engine / configuration | Observed tokens/s |
|---|---:|
| Stark | 245.5 |
| MLX-LM | 218.8 |
| vLLM Metal | 174.7 |
| llama.cpp | 147.9 |
| TensorFold (drafting on) | 125.6 |
| TensorFold (drafting off) | 118.5 |

These are local desktop measurements. Background activity and thermal state varied during acquisition. The user requested all acquired measurements be included; no clean-host or hardware-isolation claim is made.

## Method

Each configuration was loaded independently four times. Every load warmed all ten fixtures, then measured three repeats of each of six performance fixtures and one repeat of four separate sanity fixtures. All 432 performance samples met the fixed-work input/output-count and length-termination requirements. All 24 loads and 528 measured requests are retained in this campaign; no older timings were substituted.

The rate is generated tokens divided by complete client request time, including input processing, generation, transport/readout and request cleanup. Model loading and warmup are excluded. Each workload uses the geometric mean of four within-load medians; the aggregate is the geometric mean of six workload rates. Five fixtures generated 128 tokens and the long-output fixture generated 512. Context was 8192, with greedy sampling and seed zero where supported.

Four rotated/reversed blocks reduce order effects but do not fully balance six configuration positions. This bounded test supplies no confidence interval or universal speed ranking. HTTP versus private-worker transport, upstream dependencies and stock cache controls remain part of each stack. Cache behavior is not identically observable across engines.

## Models and versions

Stark, MLX-LM, TensorFold and vLLM Metal use the MLX affine W4/G64 checkpoint mlx-community/Qwen3.5-2B-MLX-4bit at revision 93760be4f1f69842a46bc13dbdc0f19e291392a3. Stark uses an unchanged repack. llama.cpp uses a separate Unsloth Q4_K_M GGUF: the quantizer differs and source-weight equivalence is unverified.

Stark's verified build is 97f15f38201cc0b465a81c45e26a4b5b135d903c; native build inputs remain unchanged in main bba01710d7996c10713c4968a758288966ca50f6. Peers are MLX-LM 0.32.3.dev6+g72be8c955 (MLX 0.32.4), llama.cpp b11747, TensorFold 1.1.1 and vLLM Metal 0.32.0. No engine or model implementation was modified for this comparison.

Four strict answer-format smoke prompts were measured separately from speed. Stark, MLX-LM, both TensorFold modes and vLLM Metal matched 4/16 replies; llama.cpp matched 12/16. Each set repeats four unique prompts across four loads. These checks are too narrow for a general quality comparison, and the retained exact-match counts do not distinguish formatting failures from semantic failures.

MetalRT is absent because stock Qwen3.5-2B support on this Mac remains unverified. No zero-throughput value is assigned.

## Evidence

- [Full acquisition report](../archive/experiments/current-engines-2026-10-21/resume-1/RESULTS.md)
- [Observed rates and per-load values](../archive/experiments/current-engines-2026-10-21/resume-1/campaign-4-observed-summary.json)
- [Aggregate acquisition receipt](../archive/experiments/current-engines-2026-10-21/resume-1/campaign-4.json)
- [Run policy and instrumentation hashes](../archive/experiments/current-engines-2026-10-21/resume-1/RUN_4_PROTOCOL.md)
