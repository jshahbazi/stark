# Qwen3.8 same-checkpoint maximum throughput, 2026-10-30

Stark peaked at **58.13 tok/s**, against **60.95 tok/s** for stock MLX-Serve
26.10.3, with both engines loading the exact same checkpoint files. Stark's
maximum was **4.62% lower** in this run. The earlier approximately 79 tok/s
Stark result used the different Lily quantization profile and does not apply
to this checkpoint.

This is a performance measurement with numerical limits. Both engines passed
loading and sampled-generation smoke checks. Nine aligned next-token choices
matched, but two of three longer greedy continuations diverged, and candidate
scores differed measurably. Numerical and quality equivalence remain
unestablished; the numerical controls below are retained alongside the rates.

The host was an Apple M5 Max with 40 GPU cores, 128 GiB unified memory and
macOS 26.7.1 (25G241), on AC power in Automatic energy mode. The user confirmed
other applications were closed immediately before the correctness checks and
comparison. Downloads, checksum verification and builds had finished before
timing. Model processes and requests ran serially.

## Maximum sampled decode throughput

Maximum means the fastest complete 256-output request among ten measured
requests per engine. Each request had 4,096 input tokens.

| Engine | Maximum decode tok/s | Measured range | Pooled decode tok/s |
| --- | ---: | ---: | ---: |
| Stark, direct official-checkpoint adapter | **58.13** | 54.00–58.13 | 55.63 |
| MLX-Serve 26.10.3 | **60.95** | 56.15–60.95 | 58.30 |

The primary numerator is **255 useful outputs after the first token**, which
comes from prefill. It matches Stark's own worker rate. The previous report's
256-output/internal-decode-time convention gives **58.36** and **61.19 tok/s**
respectively. Changing both numerators leaves the engine ratio unchanged.

The timers have different boundaries. Stark measures the interval from its
first sample to its last, covering 255 subsequent forwards. MLX-Serve performs
an additional final forward producing an unused next token; its terminal
null/cleanup tick is excluded from the decode histogram. That actual engine
work remains included. No estimated time was subtracted. These are engine
internal phase rates, not GPU-counter measurements or end-to-end transport
rates.

| Fresh-load block | Engine | Maximum decode tok/s | Pooled decode tok/s |
| --- | --- | ---: | ---: |
| 0 | Stark | 58.13 | 57.24 |
| 1 | MLX-Serve | 60.75 | 57.94 |
| 2 | MLX-Serve | 60.95 | 58.66 |
| 3 | Stark | 54.20 | 54.10 |

Stark's pooled rate declined 5.50% between its two blocks. The cause is
unestablished. This run does not establish consistent 77–80 tok/s or throughput
during desktop multitasking.

## Identical checkpoint and workload

Both engines directly loaded
[`ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit/tree/788c07a8725642bccf13a0a3b8a21954ab84b4f2)
at revision `788c07a8725642bccf13a0a3b8a21954ab84b4f2`, from one shared local
directory. All **113 files totaling 107,326,639,344 bytes** were independently
checksum-verified. All 103 LFS payload hashes matched the pinned manifest.
The completed receipt's file sizes, modification times and inodes matched
before and after execution. No conversion or requantization was performed.

The shared pack has Q4/group-64 routed experts and token embeddings,
Q8/group-64 GDN projections and most other quantized projections, BF16 HC
injection/shared-expert gates, and a Q4/group-32 external n-gram table. Stark
uses BF16 activations, F32 reductions/logits/GDN state and Q8 attention KV.
Stored weights and quantization are identical; executor arithmetic differs.

- Source/build: Stark `434beaf6aa679ba3af55fcaa9aac625d494d222b`, Rust 1.98.1
  release; unmodified [MLX-Serve 26.10.3](https://github.com/ddalcu/mlx-serve/releases/tag/v26.10.3).
- Input: the identical 4,096 corpus-derived token IDs, passed directly without
  a chat template. Their compact-JSON SHA256 is
  `fd1222c74a823b3a6cf5ed3ffd3dd22cb627afde17c21cef8b2150ac9e14b322`.
- Sampling: 256 outputs, temperature 1, top-k 20, top-p 0.95, min-p 0, seed 42;
  repetition, frequency and presence penalties off. Context capacity 4,352.
- Four fresh loads in Stark/MLX/MLX/Stark order, six requests per load. The
  first request of each load was excluded as warmup, leaving ten measured
  requests per engine.
- Speculation disabled: native no-MTP loader; MLX `--no-mtp --no-drafter
  --no-pld` and matching false request flags. Both engines used Q8 KV. MLX
  prefix entries, disk prefix reuse and tokenize caching were disabled.
- All twelve calls per engine repeated that engine's output hash exactly.
  Stark hashes token IDs; MLX hashes decoded text. The hashes are not a
  cross-engine sampled-output parity claim.
- MLX's default 32,000,153,976-byte table warm completed before any request
  in both loads. `/props` completion was checked explicitly. Cache pages
  remain reclaimable. Stark uses demand-driven row prefetch; its available
  whole-table preload is not wired into the active adapter.
- Every MLX request was bracketed by metrics snapshots: histogram counts and
  successful-request counts advanced by exactly one. Executable, library,
  settings and checkpoint identities were checked around execution.

Timer and warming sources are pinned to MLX revision
`046b647686419df06c6c7132fb1d15d76185c3e5`:
[generator](https://github.com/ddalcu/mlx-serve/blob/046b647686419df06c6c7132fb1d15d76185c3e5/src/generate.zig),
[scheduler](https://github.com/ddalcu/mlx-serve/blob/046b647686419df06c6c7132fb1d15d76185c3e5/src/scheduler.zig),
[table warm](https://github.com/ddalcu/mlx-serve/blob/046b647686419df06c6c7132fb1d15d76185c3e5/src/qwen4_exp.zig#L462-L516).

## Correctness controls and remaining limits

The earlier [adapter component controls](../archive/runs/QWEN38_EXTERNAL_MLX_ADAPTER_CONTROLS_2026_10_30.md)
remain bound to their original build. This follow-up adds complete-file
verification, real model admission, sampled smoke checks and these controls:

1. Three greedy fixtures requested sixteen outputs at 32, 128 and 4,096 input
   tokens. The 32-token fixture matched decoded output exactly. The other two
   diverged; counts and normalized finish reasons matched in all three.
   The original failed exact-match result is preserved.
2. Nine fixed corpus prefixes used lengths 32/33/36, 128/129/132 and
   4096/4097/4100. Stark GPU greedy, Stark sampled top-k-1 and MLX-Serve each
   selected the same next token at all nine. Scorer readouts used the same
   native loader. The model has 248,320 head rows and 248,077 tokenizer IDs;
   243 padded rows were not claimed as decodable tokens.
3. At each identical prefix, up to sixteen unambiguously identified
   candidates were scored. Comparisons used full-vocabulary-normalized log
   probabilities, not probabilities normalized over the candidate subset.
   Across 144 candidate scores, log-probability error RMSE was **0.47715** and
   maximum absolute error **1.49102**. The approximate reported-score
   coarsening had mean TV **0.03486**, maximum **0.11471**. It combines the
   common candidates and a residual tail; MLX F32 normalization and wire
   rounding prevent a formal full-vocabulary bound or quality-pass claim.
4. Source review found different BF16 rounding boundaries in grouped norms,
   PLE gating/residuals and fused HC projections. It found no confirmed
   loader or missing folded-gain call-site defect. This does not prove all
   observed drift is harmless or attributable solely to rounding.

The controls establish loading, request integrity, repeatability and bounded
readout agreement. They do not qualify complete numerical or answer-quality
equivalence. No acceptance tolerance was retroactively assigned to these
diagnostics.

## Optimization follow-up

Static dispatch review identified two concrete targets for the official pack:
Q8 GDN misses the existing Q4 projection/convolution fusion, and BF16 HC
injection takes a separate projection dispatch from the fused Q8 down path.
The retained Lily profile is eligible for different fusions. Folded norms
retain fused Q/K and QSA paths, and expert gate/up and down/combine fusions
remain active. These are source-level eligibility differences, not measured
attributions of the throughput gap. Profiling should precede changes.

## Evidence

[Machine-readable aggregates](data/qwen38-same-checkpoint-2026-10-30.json)
include maxima, ranges, pooled results, block data, numerical limits and
evidence hashes. Raw outputs, token vectors, logits and probabilities were
kept in process memory and were not persisted.

| Artifact | SHA256 |
| --- | --- |
| Stark generation worker | `0f95ff0d43cebedcc8f622b16d7b3e15eafb80b08abb4f03f8e8202241169fe0` |
| MLX-Serve executable | `908e87168f7fbced9a0a40fdd0469e18b510594ca8f76b72c32c69392be918c8` |
| Complete checkpoint hash receipt | `e112599b932abf197011afe5eb91bc1f8a2bb55716eda470b6ee4ef0fcb275e8` |
| Sampled ABBA controller v3 | `920f29cf7c399518c810e75220b9b3ae4f8cefb17d4bca67a3aa47cdeb2a877b` |
| Sampled ABBA report | `fa5ea052ac478aac36fd3373638e74d2fbe7ec360e2ad9ec09f0a66a3f529bfe` |
| Original greedy validation | `69ac6aa13916b796bad6450e325ed597f9ecb98e40e34f1de601918a1c543179` |
| Aligned numerical diagnostics | `21eb546f62c9f6648a81e0b8076d90dda1ccdf5e774761fd77ea47af7e92e7f2` |
| Independent aggregate audit | `244ee326aba99f386476a0741627aa72b0d4158918503faf957d84b95efdca45` |

The independent aggregate audit passed 185 checks covering counts, timings,
maxima, repeats, guards, warm completion and privacy. Its evidence bundle also
binds the greedy, numerical and host-control records. It does not add an
independent observation of other GPU processes.

Private aggregates, scripts and source receipts remain in the ignored runtime
directory. The single shared checkpoint is retained for follow-up validation
and optimization.
