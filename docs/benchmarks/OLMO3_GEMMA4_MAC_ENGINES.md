# OLMo 3 and Gemma 4 Mac engine evidence

Research date: 2026-10-28. This report records published measurements and inspected source revisions. It does not establish Stark performance or replace model-bound native validation.

## Published benchmark evidence

No source found establishes a globally fastest Mac engine for either whole family. Compare the same checkpoint, quantization, chip, context, generation length and timing boundary before ranking engines. Prefill, ordinary decode and speculative decode are separate measurements.

| Model and machine | Published result | Reproducibility and limits |
| --- | --- | --- |
| OLMo 3 7B Instruct, M4 Max 128 GB | MLX: 41.2 generated tokens/s at 4-bit; 14 at BF16 | [First-person report](https://www.linkedin.com/posts/michaelbaeriswyl_machinelearning-mlx-apple-activity-7412507946422558720-M1dT). Prompt length, output length and source revision are absent. It cannot establish an engine winner. |
| OLMo 3.1 32B Think 4-bit, M5 Max 128 GB | mlxcel: 28.73 decode tokens/s; 798.39 prefill tokens/s | [Recorded CSV](https://github.com/lablup/mlxcel/blob/98a692ce85cf2c4c223e2de9164dd96786e1aad0/benchmarks/metal_m5max_2026-09-06.csv): pp512/tg128, source `a50ff440`, MLX `9a795735`. Its MLX-LM comparison uses a different checkpoint basename; verify weight identity before comparing. |
| Gemma 4 E2B Q4, M5 Pro 48 GB | BaseRT 149.8, MLX-LM 146.9, llama.cpp 103.3 decode tokens/s | [Publisher paper](https://arxiv.org/html/2607.19438v1), Table 1: BaseRT v0.1.6, MLX-LM 0.31.3 with MLX 0.32.0, llama.cpp b9960; tg128, five repetitions on the same device. BaseRT is the fastest of these measured ordinary-decode configurations. Quantization formats are matched by nominal bit width rather than identical quantized bytes. |
| Same Gemma 4 experiment, pp512 | BaseRT 13,444, MLX 9,203, llama.cpp 2,742 prefill tokens/s | Tables 2–3 of the same paper. The BaseRT advantage here uses M5 Metal 4 matrix operations; it does not predict an earlier Mac's performance. |
| Gemma 4 E2B, E4B and 12B 4-bit, M5 Max 128 GB | mlxcel: 218.38, 135.79 and 45.19 decode tokens/s | Same [recorded CSV](https://github.com/lablup/mlxcel/blob/98a692ce85cf2c4c223e2de9164dd96786e1aad0/benchmarks/metal_m5max_2026-09-06.csv), pp512/tg128. These are faster-hardware observations, not comparisons with the M5 Pro paper. |

The mlxcel measurements use a deterministic synthetic 512-token input and force 128 output tokens. Its [methodology](https://github.com/lablup/mlxcel/blob/98a692ce85cf2c4c223e2de9164dd96786e1aad0/docs/benchmark_results/model_tests_m5max.md) documents a changed measurement interval; earlier natural-prompt results must remain separate.

[Ollama's Gemma 4 MTP report](https://ollama.com/blog/faster-gemma-4-mlx-mtp) measures a nearly 90% generation improvement for 12B NVFP4 on an M5 Max using Aider tasks. Draft acceptance varies with the workload. Treat MTP as a separate deployment profile requiring an assistant model, target verification and correct cache rollback.

## Inspected source and reuse boundaries

| Project | Inspected commit | License and usable source |
| --- | --- | --- |
| MLX-LM | `1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7` | MIT, Apple copyright. [OLMo 3](https://github.com/ml-explore/mlx-lm/blob/1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7/mlx_lm/models/olmo3.py), [Gemma 4 text](https://github.com/ml-explore/mlx-lm/blob/1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7/mlx_lm/models/gemma4_text.py), wrapper, cache and RoPE implementations are available. |
| MLX-VLM | `53ebd1647cb83549b482a031ea70ba9d9ea60a04` | MIT. [Gemma 4 Unified](https://github.com/Blaizzy/mlx-vlm/tree/53ebd1647cb83549b482a031ea70ba9d9ea60a04/mlx_vlm/models/gemma4_unified) supplies a multimodal reference. |
| mlxcel | `98a692ce85cf2c4c223e2de9164dd96786e1aad0` | Apache-2.0, Lablup copyright. [OLMo 3 Rust](https://github.com/lablup/mlxcel/blob/98a692ce85cf2c4c223e2de9164dd96786e1aad0/src/models/olmo3.rs), [Gemma 4 Rust](https://github.com/lablup/mlxcel/blob/98a692ce85cf2c4c223e2de9164dd96786e1aad0/src/models/gemma4.rs) and fused-operation paths are inspectable. |
| BaseRT | `a869c1cd86b6b34f8fd208d2c4d4ac3b2188aa08`; paper release v0.1.6 `67c2dac2d9ab5bdbbf7664d8f0f57050271dce62` | Apache-2.0 applies to CLI, converter, format, headers and bindings. The runtime and Metal kernels are proprietary binaries. Both inspected trees lack runtime source. [License boundary](https://github.com/basecompute/baseRT/blob/a869c1cd86b6b34f8fd208d2c4d4ac3b2188aa08/docs/reference/engine-releases.md). |

Retain copyright notices and license texts for adapted source, identify modified files, and record which implementation and revision informed each patch. BaseRT's public design descriptions can inform an independent implementation; its published benchmark does not grant access to its engine kernels.

## Concrete Stark implementation and optimization order

1. Use pinned MLX-LM as the numerical reference. OLMo 3 applies Q/K RMSNorm across the complete projected vector before reshaping into heads, then post-attention and post-FFN norms before residual addition. Its full-attention layers apply YaRN while sliding layers use ordinary RoPE. The inspected mlxcel OLMo forward path does not visibly consume its parsed `rope_scaling`; validate long positions before adopting it.
2. Preserve Gemma 4 attention scale 1.0, local/global head dimensions, proportional partial RoPE, configured global K=V, shared-KV layer ownership, per-layer embeddings and gating, doubled FFN widths, layer scalars and final softcap. Gemma 4 RMSNorm weights use no Gemma 3 unit offset. Validate each enabled variant against its own configuration.
3. Allocate KV only for producing layers and reference its buffers from shared layers. Bound local-attention storage with a ring cache. Validate prefill crossing the window, subsequent decode, resets and chunk boundaries before measuring speed.
4. Reuse scratch allocations and GPU command structure across tokens. Keep quantized weights packed for decode; fuse compatible normalization, rotary, activation and residual operations. Use separate decode GEMV and chunked prefill GEMM dispatch. These are practical adaptations of the inspected engines' approaches without adding MLX to Stark's runtime.
5. Measure a non-speculative pp512/tg128 baseline with source revision, chip, memory, weight hashes, quantization and warm-up recorded. Run correctness first, serialize GPU measurements, and publish aggregate metrics only. Investigate Metal 4 prefill and MTP after profiling identifies their value on the actual Mac.

The isolated MLX reference environment, its dependency lock, downloaded weights and private numerical artifacts belong under ignored `runtime/olmo3-gemma4/`. Reference dependency changes require new source-bound validation records.
