# MLX-informed generation optimization: final comparison protocol

This is an internal follow-up to the [stock-engine comparison](../../../benchmarks/STOCK_ENGINE_BENCHMARK_2026_10_19.md). Exploratory allocator, DeltaNet and GEMV experiments select the implementation before the final comparison. These are separate campaigns; earlier timing exclusions and changed outputs remain in their original receipts.

## Final comparison

- Host: Apple M5 Max, 40 GPU cores, 128 GiB, macOS 26.7.1.
- Engines: original default Stark, modified default Stark, and unmodified MLX-LM at `72be8c955d9a0d3099bd01865af288e0e947c3fa` with MLX 0.32.4.
- Model: the existing `mlx-community/Qwen3.5-2B-MLX-4bit` revision `93760be4f1f69842a46bc13dbdc0f19e291392a3`. Stark uses its previously converted unchanged affine W4/G64 representation. Model hashes are recorded separately.
- Rust builds use 1.98.1, the locked dependencies and the ordinary release configuration. No experimental scoring features are enabled.
- Use the unchanged `scripts/benchmark_stock_engines.py` and `scripts/stock_compare_mlx.py`; record their SHA-256 hashes. Integer inputs come from the same source tokenizer and chat template, thinking disabled.
- Context 8192, greedy decoding, seed 0 where applicable, one request in flight, one model loaded at a time, fresh request state. Model loading is excluded from request timing.
- Four independent model loads per engine. Rotate/reverse engine order using the existing driver's algorithm. Each load warms each performance fixture once and measures it three times. Run each of the four short sanity fixtures once per load.
- Six performance fixtures: input lengths 42, 40, 383, 522, 2118 and 50; output limits 128 for the first five and 512 for the last. Sanity fixtures are correctness checks and are excluded from performance aggregates.
- Primary rate is actual generated output tokens divided by complete client request time, including prefill, decode, transport, readout and cleanup. It is not an isolated decode rate.
- Fixed work requires the exact output limit and a length termination. Preserve all early stops and failures.
- Keep the existing sampled host guard: reject a load when any external process exceeds 200% CPU, a recognized model process exceeds 10%, or monitoring fails. Retain whole-window sleep checks. There is no continuous GPU, thermal or power isolation.
- Preserve every load. Report accepted subsets and paired load intersections explicitly if any load is excluded. Never turn whole-campaign rejection into whole-campaign acceptance.
- Persist scalar measurements, counts and SHA-256 hashes only. Request text, responses, token vectors and logits remain in memory/private pipes.

## Correctness

Compare original and modified Stark output-token and decoded-text hashes on all ten fixtures, including excluded timing loads. Compare full-vocabulary F32 bit digests on the ten original inputs plus thirteen prefix lengths spanning single-token, SIMD/batch boundaries and 1024-token chunk boundaries. Scoring caches stay disabled. This establishes bounded numerical equivalence, not general model quality.

The targeted GEMV test also compares every BF16 output bit in memory for both tuned projection shapes and checks device, shape, format and dtype fallback guards. Scratch allocator tests cover overlapping submissions, completion failures and live allocation handles. Existing worker tests cover stop limits, stream cleanup and error fencing.

## Scope

The default changes are completion-gated reuse of two decode scratch pools and a narrowly admitted GEMV tile adjustment. The rejected DeltaNet rewrite is retained as a reproducible patch against `54d113a7034fc0d9bacb4c0653a7713500fb5bbe`; it is absent from the shipped implementation. This experiment does not qualify speculative decoding, grammar performance, other GPUs or JevK5 performance. No article or external benchmark publication is authorized by this work.
