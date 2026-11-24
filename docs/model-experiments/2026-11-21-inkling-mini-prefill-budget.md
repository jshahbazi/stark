# 20261121-INKPREFILL-05 · Inkling 2 Mini 4K prefill budget

| Field | Recorded value |
| --- | --- |
| Artifact | Original vision-language target `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, target shard SHA-256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1` |
| Host | 128 GiB M5 Max, macOS 26.7.2, locked Rust 1.98.1 release |
| Incumbent | `29dfe1a948286ddea6692e2ac97d75cc5ed4d748` with 128-row commands, grouped experts, batched controller expansion and four-/eight-row dense weight sharing |
| Model control | 4,096-token pinned workload at context 4,100, private bundle SHA-256 `74d6457708d31fe3ba6aeeab41b836d6ac7cd825cb3837fc3ddacc64e0c19719` |

## Time budget

The incumbent took 23.637 s for one uninstrumented native 4K ingestion. The separate paired Metal counters measured 22.504 s of command GPU time across 32 commands and 5,131,776 dispatches. All 200,058 continuation logits matched between instrumented and uninstrumented runs.

| Phase | GPU seconds |
| --- | ---: |
| NVFP4 experts | 5.720 |
| MXFP8 experts | 1.930 |
| Attention input preparation | 4.237 |
| Expert input preparation | 2.683 |
| Expert finish | 0.550 |
| Dense attention projections | 3.004 |
| Relative attention and cache | 4.201 |
| Embedding and gaps | 0.179 |

The 589.58–626.15 tok/s floors allow 6.95–6.54 s for 4K. Even zeroing the whole 7.65 s expert band gives a ceiling near 276 tok/s under the measured budget. Allocation layout matters across the graph. The pinned Ollama server allows up to 2,048 input rows per prefill step and sorts expert routes across each step. Its 512-row trace was a bounded validation capture. The released path uses sorted tensor products and query-tiled relative attention. The compact NVFP4 kernel that inspired Stark's exact guard is a separate compatibility path. [Ollama prefill pipeline](https://github.com/ollama/ollama/blob/6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8/mlxrunner/pipeline.go) · [Original benchmark](../benchmarks/2026-11-20-inkling-mini-m5-max.md) · [Pinned trace](2026-11-21-inkling-mini-matched-prefill.md)

Incumbent profile receipt SHA-256 is `b3b0190d85995683bf6327ac80775b78a8b585c543a89c9a26ffb701d37dca33`. Host thermal and performance warnings were absent.

## Paired screening

The eight pairs per cell alternated launch order in one loaded checkpoint. They used distinct prompt windows with one warmup per arm. The numeric check compared live cache and convolution state, the final BF16 taps, four F32 continuation logits and snapshot replay. The candidate is exact in the table if every pair passed those comparisons.

| Change | Rows per window | Median paired wall ratio | Wins | Parity and disposition |
| --- | ---: | ---: | ---: | --- |
| 256-row commands and expert grouping | 512 | 0.944 | 8/8 | Exact. Required 178.5 MiB. Archived. |
| 512-row commands and expert grouping | 512 | 0.916 | 8/8 | Exact. Required 355.5 MiB. Archived. |
| Source NAX dense with 16-row pack/scatter copies | 512 | 0.909 | 8/8 | Cache, taps and continuation changed. Production precision gate failed. |
| Batched controller RMS and staging | 512 | 0.888 | 8/8 | Exact. Reuses existing Metal buffers. |
| Direct expert input and output bindings | 256 | 0.980 | 7/8 | Exact. Uses the existing expert slabs for ingress and the MLP convolution. |
| Batched controller RMS and direct bindings | 256 | 0.869 | 8/8 | Exact. Selected as a group for the worker follow-up. |
| Compact 64-K NVFP4 matrix path | 256 | 0.934 | 8/8 | Exact on the prompt controls. Removes the certified reduction replay and lacks the required extremal numerical qualification. Archived. |
| Cache normalized residual-block and prior-pass rows | 256 | 0.990 | 7/8 | Exact. Marginal gain with added workspace. Archived. |
| Batched RMS, cached residuals and direct bindings | 256 | 0.860 | 8/8 | Exact. Residual caching made little marginal difference. |
| Batch history convolutions across four rows | 256 | 0.956 | 6/8 | Exact, compared with the selected RMS and binding path. Failed the win-count gate. |

In the diagnostic that combined batched RMS, cached residuals, direct bindings and compact NVFP4, the complete 4K pair ran in 20.430 s against 23.052 s and matched the native state and continuation. This was one sequential pair, with compact NVFP4 still outside extremal qualification. The final source discards the unselected probes. During that cleanup a Rust build caught two missing struct initializers. The fix restored the explicit defaults before rerunning.

The four screening executables have SHA-256 `a14a0afff6dd3621adfa2f7f202bb420efb44b77ac2462de5490064f318c73bc`, `f8e2fedd794b2b370c47a1b8c24234d9bc080951d4d8d0dbd4aee3b963a39b85`, `b1daecce9898c063e26cdd5903a8a46aa91f78371363484a54c52dd4ab30287f` and `1cc5f0cd5e28abfa0e2b78cd36e6813f0dbd4c71708e92cc0acfde28c474198d`. The combined diagnostic 4K receipt SHA-256 is `2df63a26a54e7acb644b42c7c33236b41c709ae5812a98c73f863feab1c5266c`. Complete screened source changes and private data remain in the owner-private experiment directory. The tracked portion of the archived patch has SHA-256 `8d7dbbd788c802b5f7780849ed89de72a533973830debc74e5a3d2ced0b851fb`.

The profile supports a packed layer-major graph for the remaining gap. Normalization, residual mixing, projection, convolution and router stages need contiguous row buffers that the next stage can consume directly. The attention redesign needs query-by-key matrix tiles for both local and global visits. Wider expert groups are useful when they operate on those packed activations within a bounded working set. Each component needs a matched numerical control before measuring the integrated graph.

## Selected graph

The shipped candidate fuses controller input and message RMS with staging, writes weighted RMS to the packed expert input and reads the packed expert result directly in the MLP convolution. It preserves the 128-row command bound and uses the incumbent's 89.8 MiB of additional workspace. Trace mode uses the individual expert buffers for boundary capture. Tiles that do not use grouped experts still use row-local scratch.

The final eight-pair 159-token holdout had a median wall ratio of 0.877 and won all eight pairs. The complete 4K control took 21.148 s against 23.393 s in one same-session pair. Every holdout and full-model case reproduced the state, taps, all four continuation logits and prefix snapshot replay. These are native controls. [Source-runtime parity status](2026-11-21-inkling-mini-matched-prefill.md)

```sh
inkling_slice --prefill-speed "$ORIGINAL_SHARD" "$FULL_ORACLE_4096" 4096 \
    wide+normbatch+direct+controllers+mx+small+widegemv:wide+controllers+mx+small+widegemv
```

The final example SHA-256 is `2734316ea4b739ee79ab7fd12a07a75041b2a83357581acfc4057921bc34b49b`. The tail and 4K receipts are `b10c3f1d0f35ceae9a0ef2692e50cfdd77b1b67e2a37bebc634c8c76da465423` and `dcdaa2f22a8c3bbc63637f848223ac810be59c6f47f70177474cb7b060f6b0a9`.

The first worker comparison showed a sampled decode outlier, so the incumbent and candidate each repeated the seeded 4K sampled request twice, with opposite launch orders. The earlier observation stays in the record. Before the repeat, the 32-step seeded-cache target control measured 28.025 ms median for the incumbent and 28.055 ms for the candidate. Both encoded 1,883 dispatches and used one completion per step.

The worker reached 199.98 input tok/s in the greedy pair and 197.49 tok/s at the sampled median across all three pairs. Greedy decode was 35.56 to 35.54 tok/s and sampled median decode was 33.38 to 33.14 tok/s. Full outputs matched in all pairs. [Worker results and repeat controls](../benchmarks/2026-11-21-inkling-mini-native-worker.md#batched-rms-and-direct-expert-bindings)

The live prefix test restored 1,024 of 1,031 input tokens with matching output, lowering prefill from 5.026 s to 0.142 s for that request. The request after cancellation reused zero tokens and produced the same output. Vision readiness, a visible image color answer, greedy and sampled draft-on/off equivalence, text streaming and a tool-result round trip passed. The two worker receipts have SHA-256 `107332a4e004647ace2a78ece87fd3bcf839718f59cdacc9bcdd01480ffb8288` and `1f9a8bf0291a1b487b9cbfcbe49b75e16a18ea29b4c5d452ab1b645735f8c609`. Short and 4K slice parity reproduced both original aggregate hashes.

Locked `stark-generate` tests passed 157 cases, with five private-artifact tests ignored. Rust formatting and repository policy checks passed.
