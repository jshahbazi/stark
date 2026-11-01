# GLM-5.3-Flash baseline and optimization trial, 2026-11-01

MLX-Serve 26.10.3 measured **35.44 tok/s** on the existing GLM GGUF in four
screened warm requests. Excluding the first token from the numerator gives
**34.33 tok/s**. The packed finite-check fusion trial passed numerical checks
and was not admitted to production: the native timing campaigns did not
establish a performance benefit. Stark's production kernels remain unchanged.

Work began October 31 and finished November 1, America/New_York. The host was
a MacBook Pro with Apple M5 Max, 128 GB unified memory and macOS 26.7.1.
The [aggregate JSON](data/glm53-mlx-baseline-2026-11-01.json) retains scalar
observations, host screens, source identities and private receipt digests.

## Exact checkpoint and executors

Both engines loaded the same physical four-shard
`unsloth/GLM-5.3-Flash-GGUF` `UD-Q2_K_XL` artifact at revision
`a38483c8cd5df544f53d70fb281afe97369d5ab6`, totaling 108,720,071,552 bytes.
Configuration and tokenizer came from
`zai-org/GLM-5.3-Flash@04c4e9e95c5da8862dced7e5056455116f83a7e0`.
After timing, all four payload SHA-256 values were freshly recomputed and
matched the pinned validation manifest. No conversion or requantization occurred.

| Executor | Identity |
| --- | --- |
| MLX-Serve | Unmodified 26.10.3, source `046b647686419df06c6c7132fb1d15d76185c3e5` |
| MLX-Serve executable SHA-256 | `908e87168f7fbced9a0a40fdd0469e18b510594ca8f76b72c32c69392be918c8` |
| MLX-Serve backend | Embedded llama.cpp, selected explicitly with `--engine llama` for the split GGUF |
| Stark baseline | Source `3eca4590`, existing native F32 execution policy |
| Baseline generator SHA-256 | `84424f4ee110590264d195eee1919e49766a6477a7fcec2458381ce712e8be32` |
| Trial generator SHA-256 | `b9893aec85c1bef0297e2c23a2d98f2eede7be50d0bc04ff4e6ff14e3aa79f73` |

This baseline measures MLX-Serve's GGUF backend. A native MLX-format GLM pack
was not evaluated. The checkpoint declares 45 trunk layers; MLX-Serve's GGUF
catalog returns a generic 48-layer stub, which is not architectural evidence.

## Screened MLX-Serve baseline

Each request used the identical 14 input token IDs, context 512 and a
32-output greedy limit. Speculation, vision, prefix reuse and request logging
were disabled. The embedded llama MTP switch was explicitly set to
`--llama-mtp-drafts 0`. Every request reported zero cached tokens, 14 inputs,
32 completions and a length finish. MLX-Serve reported KV quantization off.

| Measured request | Server numerator, 32 outputs | Useful numerator, 31 outputs |
| --- | ---: | ---: |
| 1 | 35.07 tok/s | 33.97 tok/s |
| 2 | 35.39 tok/s | 34.29 tok/s |
| 3 | 35.52 tok/s | 34.41 tok/s |
| 4 | 35.49 tok/s | 34.38 tok/s |
| Median | **35.44 tok/s** | **34.33 tok/s** |

One warmup request was excluded. Each measured request followed at least ten
seconds of quiet admission. Admission required external CPU at most 150%
(one core is 100%), any one process at most 80%, external model CPU at most
10%, GPU utilization at most 5%, and thermal state no higher than fair.
Request screens required no active compiler, paging, monitoring failures or
sampling gaps, external CPU at most 150%, external model CPU at most 10%,
and thermal state no higher than fair. All four measured requests passed.
Their maximum external CPU observations were 114.91%, 50.06%, 14.99% and
35.94%. All had fair thermal state, zero external model CPU and no paging.
This is a screened desktop measurement; nominal thermal state and long-run
thermal stability are not claimed. Admission and request summaries are retained
in the aggregate JSON.

Phase times come from per-request server histogram deltas. Success and phase
histogram counts must each advance exactly once. Both numerators use the same
observed decode interval. These are server phase timings, with engine-specific
boundaries; they do not measure isolated GPU time.

The initial unscreened pilot observed 35.37 tok/s using an earlier unsealed
harness. A later unscreened refresh fell to 24.59 tok/s while external CPU
peaked above 1,800% and pageouts increased. Those windows remain separate
observations and do not determine the screened baseline.

## Numerical controls and trial disposition

The input compact-JSON SHA-256 was
`3509a541495b39640d71f426f454b6cb2a1647f22b37535bd9705764e0e3c96d`.
Every screened MLX completion had decoded-text SHA-256
`386d79a13ffaa7e65b514edc6a58fbf2cccf934202c5c08e95022d0799e75f36`.
All native baseline and trial replays matched that decoded text and native
token hash `5377eeb8513c16603090026f7a432327b456842a96d54565044e24bdb5986e1c`.

The trial transferred Qwen's finite-check fusion idea to packed GGML matrix
products, preserving arithmetic order and sticky error checks before clipping.
The [archived patch and assessment](../archive/experiments/glm53-packed-finite-2026-11-01/README.md)
retain its source and executable identities. The complete A/B/B/A campaign
failed its host screens. In a subsequent A/B/A campaign, only the candidate
block passed the prospective desktop screen. There is no accepted matched
pair from which to quantify an optimization effect. The trial remains archived.

Validation below applies to the rejected candidate:

- Physical CPU/Metal suite: 675 passed, 48 ignored; seven CXX and three
  documentation tests passed. New packed checks covered all 13 admitted
  formats, finite output bits, row tails, multiple matrices, nonfinite input
  and overflow concealed by later clipping.
- Saved GLM pool-2 and pool-4 references: 94 positions each, with unchanged
  final-readout absolute and relative-RMS limits of `1e-4`. Pool 2 maximum
  absolute / relative RMS were `3.3378601074440795e-5` /
  `2.1092784081480096e-6`; pool 4 values were `3.910064697265625e-5` /
  `2.977030357176713e-6`.
- Worker integration: four greedy reference matches, tokenization/decode,
  EOS handling, isolation and invalid-request recovery passed. The optimized
  scorer passed reference logits with maximum absolute error
  `2.2411346435546875e-5`, normalized candidates and consecutive-cache
  invalidation checks.
- Rust 1.98.1 release builds and generator/optimized-scorer Clippy with
  `--all-targets -- -D warnings` passed. Nightly formatting and repository
  policy checks passed.

A separate baseline diagnostic recorded 172 submissions and waits for four
decode steps: 43 per token. Reducing those routing boundaries through bounded
execution groups remains a distinct design task under the existing memory
admission limits.

## Reproduction

With the pinned model, prior weight manifest and unmodified MLX-Serve installed:

```sh
uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_glm_mlx_serve.py --model runtime/native-hybrid/glm --tokenizer runtime/native-hybrid/glm --weights-manifest runtime/native-hybrid/glm53-q2-full-validation.weights.json --binary /absolute/path/to/mlx-serve --engine llama --tokens 32 --replays 5 --screened --output runtime/glm53-baseline-new.json
uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_hybrid.py --model runtime/native-hybrid/glm --worker /absolute/path/to/stark-generate --tokens 32 --warmups 4 --replays 8 --output runtime/glm53-native-new.json
```

The first command screens individual requests and preserves excluded attempts.
It refuses to overwrite a receipt. The second command provides explicit
warmups and output hashes; its timings need an external host screen before
making comparative claims. Models and runtime captures remain ignored;
published artifacts contain aggregates and hashes.

This short, single-prompt greedy control does not qualify answer quality,
long contexts, multimodal inference, sustained thermals or whole-model
numerical equivalence between executors. The earlier
[GLM graph study](GLM_QWEN_DECODE_GRAPH_2026_10_28.md) remains source-bound
historical evidence for its own build and timing window.
