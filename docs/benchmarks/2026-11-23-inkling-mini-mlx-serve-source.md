# MLX-Serve and Stark on Inkling 2 Mini: source comparison

Attempt `20261123-INKMLXSRC-01`, 2026-11-23. A read-only comparison of MLX-Serve's Inkling 2 Mini implementation with Stark's, from source, with no builds or GPU runs. It explains the measured gaps in the [engine matrix](2026-11-22-inkling-mini-engine-matrix.md) and the [prefill attribution](2026-11-22-inkling-mini-prefill-attribution.md); it measures nothing itself. Record in the [experiment ledger](../model-experiments/README.md). Each difference and its current Stark status is tracked on the [differences page](../model-experiments/inkling-mlx-serve-differences.md).

## Source and limits

| Field | Value |
| --- | --- |
| Source | `ddalcu/mlx-serve` at `2fe9be834a601c70ef1a597563026340f0e722fe`, archive SHA-256 `266d0cc2d910779350f5fed3b40a68113988561c8f8693ebab30992b1fa2745b`, downloaded with the owner's approval to an ignored directory (`runtime/third-party/mlx-serve-2fe9be83/`). MIT, with Apache-2.0 parts. Nothing was copied into Stark beyond short cited fragments in private notes. |
| Inkling 2 Mini code | `src/src/plugins/inkling2/`. `plugins/inkling/` is Inkling Small, which runs serially with speculation off. An earlier web-based reading (2026-11-22, through a summarizing fetch tool with approximate line numbers) confused the two; this reading corrects it. |
| Not readable | The MLX library itself (`lib/mlx-src` is a submodule not in the download). Every statement about MLX's own `quantized_matmul`, `gather_qmm`, steel GEMM and `scaled_dot_product_attention` internals is inferred. |
| Stark side | The uncommitted working tree of 2026-11-23 on top of `f1666090` |
| Private notes | `NOTES-stark-comparison.md` beside the source, with file and line citations for every statement |

## Findings

**Decode.**

- Launches: about 12 to 14 per first-pass visit and 17 to 19 per later-pass visit, about 0.75k per token (inferred counts), against Stark's 991 after fusion. The difference is about 25 percent, not 2 times. What matters more is what MLX-Serve does not run: no NVFP4 activation quantization, SwiGLU inside the up kernel, sum, conv and residual inside the down kernel, the output gate inside the output projection, and one input kernel per visit (`kernels.zig:985-1002`, `ups.metal`, `downs.metal:36-52`, `oproj.metal`, `pre.metal`).
- Speculation has two layers: a per-round confidence cut (`plugin.zig:290-342, 425-435`), `min_sure` = clamp(1.34·√(verify-row cost / plain-step cost), 0.3, 0.8); and an engine backoff that runs pipelined plain steps for a doubling period up to 32 rounds when drafted rounds do not pay (`engines/mlx/spec/dflash/policy.zig:169-224`, `engines/mlx/spec/dspark.zig:92-93, 208-215`). Tokens per step is therefore a policy outcome. The draft blocks on an eval before the verify is built (`plugin.zig:395`); there is no overlap and no rollback.
- Plain steps are pipelined: the next forward is evaluated asynchronously and its token resolved at the next tick (`engines/mlx/generate/pipeline.zig:75-90`).
- One-row router: MLX F32 matmul on first-pass visits, fused into the `norm` kernel on later passes (`kernels.zig:827-835`).
- One-row dense and head: MLX affine qmv on the 8-bit pack; the original's BF16 `x @ W.T` was chosen over a pre-transposed one-row `gemv_t` (about 43 against 9 µs, `docs/gotchas/engine-mlx.md:1596-1603` in the source tree). Stark is already at 80 to 90 percent of bandwidth here; no gap expected (inferred).

**Expert inputs.** No quantization of the NVFP4 or MXFP8 expert input on any runtime path: decode reads BF16 inputs into F32 registers (`ups.metal:26`, `downs.metal:31`), prefill NVFP4 loads BF16 activation fragments from device (`gather_qmm_nax_header.metal:616-623`), and there is no `mlx_quantize` call at run time. The canary that arms the custom prefill kernel requires bit equality with stock MLX `gather_qmm` (`gather_qmm_nax.zig:282-288`), which implies stock MLX does not quantize either (inferred). This differs from the publisher's four-over-six activation quantization; MLX-Serve has no reference check of its own. Stark's later measurement of this choice is [INKROUND-07](../model-experiments/2026-11-23-inkling-mini-decode-round.md#20261123-inkround-07--unquantized-nvfp4-expert-inputs).

**Prefill.**

- Chunking: 8,192 rows by default (`engines/mlx/generate.zig:106`), tails under 512 merged, so 8,800 tokens run as 8,192 + 608.
- Experts: two gathered matmuls per visit (gate and up over the interleaved bank, then down on the already-sorted SwiGLU output), one materialized sorted copy of the input (about 300 MB per visit at 8,192 rows), un-sort fused into `sum.metal`. The planner keys on mean rows per expert over the call: at 8,192 rows (mean 512) both products run the double-buffered schedule with 96 × 64 tiles and BK 64; the 608-row tail runs 64 × 64 (`gather_qmm_nax.zig:28-39`). Weight tiles are dequantized into threadgroup memory with the next K step's packed words prefetched into registers; activations come from device as `vec4` register fragments; decode and tensor work share the same SIMD groups (`gather_qmm_nax_header.metal:543-651`). `relaxed_precision` is set, as in Stark's kernels.
- Dense: q, gate, r and sink joined, k and v joined on sliding first-pass visits; on the 8-bit pack at 2,048 rows and above each weight is dequantized to a BF16 transient and run through MLX's BF16 GEMM (`prefill_dq_gemm.zig:15-31, 68-107`).
- Attention: chunks over 32 rows use `relattn.metal`, a half-precision flash kernel (half q, K, V and probabilities; F32 scores and accumulator), stopping at the causal diagonal and starting at the window, with no mask tensor (`kernels.zig:420-433`, `relattn.metal:32-129`).
- Cheaper precision: the first-pass router and message-gate projections in prefill run as MLX F32 matmuls over more than 32 rows, which MLX-Serve's own notes call TF32 on M5 at about 1e-3 relative (`docs/gotchas/engine-mlx.md:1605-1611` in the source tree).

**Published numbers.** `benchmarks.md` publishes decode only (llmprobe `--bench-only`, warmup discarded, median of 3, its own code prompt) for M4 Max, M5 Ultra and an RTX box; no M5 Max table, no Inkling 2 Mini row and no prefill figure. Every MLX-Serve number in Stark's records is Stark's own measurement.

## Ranked gaps (estimates by the comparison's author, not measured)

Decode, against the run-4 gap (101 against 86.9 tok/s on code, 91.6 against 67.8 on prose for the 8-bit artifacts): (1) speculation policy, high confidence; (2) expert-input quantization, about 1.05 ms per token, a semantic change; (3) expert tail and controller fusion, 0.5 to 1.5 ms; (4) one-row router, 0.5 to 0.8 ms; (5) pipelined plain steps, 0.3 to 0.8 ms; (6) dense, head and attention at one row, low confidence of any gap.

Prefill, against 3.42 s for MLX-Serve and 4.68 s for the Stark server at 8,800 tokens: (1) half-precision attention with causal and window skipping, 250 to 400 ms of 533; (2) dense projections, 300 to 450 ms of 687; (3) worker overhead outside the GPU, 0.3 to 0.5 s; (4) routing and expert-input passes, 200 to 400 ms, precision trade-offs; (5) elementwise and controller passes, 100 to 250 ms; (6) the NVFP4 gathered product itself, low confidence. Items 1 to 5 add up to about 1.2 to 1.9 s (inferred).

## What happened to the ranked items

Measured afterwards (details on the [differences page](../model-experiments/inkling-mlx-serve-differences.md)): the speculation rules were simulated on Stark and lost to Stark's own history planner, because Stark's cost ratios differ ([INKROUND-06](../model-experiments/2026-11-23-inkling-mini-decode-round.md#20261123-inkround-06--depth-policy-history-planner-confidence-cut-and-backoff)); unquantized expert inputs saved about 0.78 ms per step and passed the quality bar on the original artifact ([INKROUND-07](../model-experiments/2026-11-23-inkling-mini-decode-round.md#20261123-inkround-07--unquantized-nvfp4-expert-inputs)); a one-row router rewrite saved 0.06 ms and was rejected ([INKROUND-04](../model-experiments/2026-11-23-inkling-mini-decode-round.md#20261123-inkround-04--decode-router-rewrite)). The attention, dense-join and elementwise items were in progress or queued.
