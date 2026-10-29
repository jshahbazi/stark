# Qwen3.8 iQ-MLX 4.7 bpw investigation

Date: 2026-10-29. Status: checkpoint metadata inspected; exact-checkpoint Stark
throughput remains unmeasured. The fresh generation measurements below use the
existing Lily checkpoint. No inference implementation changed in this study.

## Published speed and checkpoint identity

The requested checkpoint is
[`ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw`](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw),
pinned to `a0e89db090a3e4b2e8ea2118fe9d0324d3854aca`.
Its [model card at that revision](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw/blob/a0e89db090a3e4b2e8ea2118fe9d0324d3854aca/README.md)
reports **157 tok/s median, range 149–162**, using mlx-serve 26.10.1 on an
**M5 Ultra with MTP enabled**, across six alternating runs. Prefill is reported
as 5,411 tok/s. It supplies no corresponding M5 Max result with speculation
disabled. This evidence does not verify the separately reported 100 tok/s claim
under Stark's existing no-speculation conditions.

The pack uses importance-calibrated affine quantization. Its repository contains
107,326,639,758 bytes in total: 75,303,252,216 bytes of Safetensors weights,
including the vision tower, and a separate 32,000,153,976-byte n-gram table.
The model and MTP weights occupy 100 shards, plus the vision shard.
Seven remotely fetched container headers confirm
Q4 token embeddings, Q8 readout, Q8 GDN projections, Q8 HC down/up projections,
and dense BF16 HC inject/shared-expert gates. The inspected router is Q8 despite
the card's BF16-router description. Experts use Q4 group 64; the merged n-gram
table uses Q4 group 32. These storage policies differ from Lily q4-xl, which
uses Q4 GDN projections and Q8 token embeddings.

## Current Stark compatibility

The exact export cannot load unchanged through either existing Qwen backend.
This is a source/header compatibility finding; a complete-model load has not
been attempted because the checkpoint is not downloaded.

- The fast BF16 backend requires the Lily format marker and Lily tensor names.
  MLX-Serve uses different names and stores the n-gram table separately in
  `ngram_table.bin`. Adding a Lily marker alone would be incorrect.
- Its HC inject and shared-expert gate matrices are dense BF16. The current
  fast backend requires affine triples for those projections. They need a
  dense projection route that preserves their stored weights.
- MLX-Serve folds the normalization weight's additive one into the checkpoint.
  The fast backend must carry that convention through ordinary and fused norm
  kernels. Subtracting one and rounding to BF16 has not been established as a
  lossless conversion.
- The F32 backend already follows several MLX conventions, but requires explicit
  per-tensor quantization widths and support for the external merged n-gram
  table. Its global Q4 policy and expected 128 n-gram shards do not describe
  this export.

Existing packed Q4/Q8 kernels, HC down/up kernels, convolution transposition,
and the Q8 GDN path can be reused. A storage adapter can preserve the original
packed weights; a new quantizer is unnecessary. Matching-reference numerical
validation and full generation on these exact weights are required before
assigning this checkpoint a Stark throughput result.

## Fresh Stark control on the existing checkpoint

The control ran on the local M5 Max with 128 GiB memory, macOS 26.7.1
(`25G241`), AC power, and Automatic power preferences. Other desktop applications
remained active. No system preferences changed. This short measurement window
does not establish sustained throughput or isolate desktop contention.

Source was `565a9a5f`, whose tracked tree is identical to `6f953e8c` after moving
the optimization series onto a branch based on current main. The unchanged
generator SHA-256 is
`ff4df5d10ccb95dd3ab3f9453be1bcbce830bbc6f99b27c079deced20d5e354e`.
No rebuild was needed for the branch move.

The checkpoint remains `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`: 55 shards,
106,905,155,192 bytes. All 55 current device/inode/size/modification-time
identities match the earlier complete SHA-256 verification. This check did not
rehash the complete weights again.

Each mode used a separate loaded Stark worker, 4,096 corpus-derived input tokens,
256 requested output tokens, and five sequential requests with fresh logical
state. The first request per mode is warmup; the four remaining requests provide
the table below. Decode rates count the 255 advances after the first prefill
draw. Sampling uses temperature 1, top-k 20, top-p 0.95 and seed 42. MTP, prompt
lookup and other speculation are disabled.

| Mode | Warm median tok/s | Warm range | Pooled warm tok/s |
| --- | ---: | ---: | ---: |
| Greedy | 72.08 | 69.99–72.77 | 71.72 |
| Sampled, top-k 20 | 64.92 | 60.30–66.88 | 64.15 |

All ten requests produced 256 outputs and matched exactly within their mode.
The sampled hash also matches the preserved Lily reference. These checks show
repeatability on the existing checkpoint; they do not qualify the iQ-MLX pack
or establish answer quality.

## Evidence and remaining work

Private metadata and aggregate generation reports are retained under
`runtime/native-hybrid/qwen38-iq47-20261029/`. No prompt, output text, token vector,
logit or probability array is published.

| Artifact | SHA-256 |
| --- | --- |
| Pinned model card | `20aacd06347ecb906516aa783b568708186506f8a8e4647b601c0c9951a2a57f` |
| Pinned source config | `9a2ab86be9f4860a1ac78c4d908aa78a4ad1522d0358e68bbc837af20161fe7a` |
| Pinned source index | `095d52d076b5ecdc50d3206ef645cc9ca9a571d969cef7685a75f5018a4ce068` |
| Compatibility metadata and seven headers | `e3336dbe7af3c12c1c7ac59ca7865e3e59bd18cbc3230b70205740162d178f14` |
| Fresh greedy Stark control | `21aeca0f27ee59c2f79b1ea1ffea2d3771bf300d8e35d7f2159ec4259e2ef22b` |
| Fresh sampled Stark control | `50a75a4eaf2f568e60e04371f7d6d61dc78760af5eacbbf1cdff26b7069ba05f` |

The corpus is the pinned Lily `src/qwen4exp/model.rs` used by the earlier
studies, SHA-256
`daecdf178c6429a721f9e1b5ae379c664adb8f8af55b381535355636ec469955`.
Input hash is
`fd1222c74a823b3a6cf5ed3ffd3dd22cb627afde17c21cef8b2150ac9e14b322`.
Greedy output hash is
`3a94579eb57b2d943bfb8d6e1b9dc65bbaa4588b98fef7f96895b28bb2073e5a`;
sampled output hash is
`621ca90417af420aaf9defdc957f7cc1f57f318c08d908be764834abb382b0d0`.

The disk had approximately 51 GB free during the investigation. Downloading the
107 GB source checkpoint requires another storage location or sufficient space
to be freed. No existing checkpoint has been deleted. Once storage is available,
the remaining work is to preserve and verify the source weights, implement and
qualify the storage adapter, and benchmark repeated sampled generation with
speculation disabled. No iQ-MLX performance comparison is established here.
