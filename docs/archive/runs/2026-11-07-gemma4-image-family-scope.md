# Gemma 4 image family architecture assessment

Date: 2026-11-07

The original eight native OLMo 3/Gemma 4 commits, ending at `294b228663c3bf47c6ee250aa4083cb42c060aa3`, were merged onto current main `8198f748` as `3bc9e8cd`. This retains the original E2B BF16 image implementation and all canonical and affine-source precision repairs. The prior E2B validation remains bound to its original artifacts and builds.

## Source-bound scope

The following official configurations were inspected at their immutable Hugging Face revisions. All sizes accept images upstream. Their native requirements differ.

| Official checkpoint | Revision | Native architecture requirements |
| --- | --- | --- |
| [E2B-it](https://huggingface.co/google/gemma-4-E2B-it/blob/3e22461f65e89153144f8adb70e3b8c2cc9845a7/config.json) | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | Existing 35-layer PLE decoder, width 1,536, 20 shared KV layers; 16-layer vision tower, width 768/head 64, clipped linears. |
| [E4B-it](https://huggingface.co/google/gemma-4-E4B-it/blob/4ecc2ba63bcac23ba4a767fc87025b1450f84c9c/config.json) | `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c` | 42-layer PLE decoder, width 2,560, two KV heads, 18 shared KV layers, ordinary-width MLP. Same 16-layer vision geometry and processor as E2B, with a 2,560-wide projector and distinct tokenizer bytes. Implemented as its own schema-4 profile; exact checkpoint qualification is recorded separately. |
| [12B-it](https://huggingface.co/google/gemma-4-12B-it/blob/43c9d014b231474fa7212dbb48a762566c2b255d/config.json) | `43c9d014b231474fa7212dbb48a762566c2b255d` | Unified encoder-free image embedder, width 3,840; 48 layers, no PLE, global K=V attention with one global KV head; causal global attention and window-bounded same-image visibility in local layers. |
| [26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/4d7ae4984b7db7de8f8457170b3f1a419ee76d52/config.json) | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | 30 layers, width 2,816, 128 experts/top 8, shared dense plus expert residual branches; no PLE, global K=V/two global KV heads; 27-layer width-1,152/head-72 vision tower with learned output standardization and ordinary linears. |
| [31B-it](https://huggingface.co/google/gemma-4-31B-it/blob/842da3794eaa0b77d5f08bae87a17459d91ff475/config.json) | `842da3794eaa0b77d5f08bae87a17459d91ff475` | 60-layer dense decoder, width 5,376, no PLE, global K=V/four global KV heads; causal global attention and window-bounded same-image visibility in local layers; same larger vision geometry as 26B. |

## Larger-model implementation and remaining blockers

The efficient E2B/E4B path keeps its PLE and causal-image policy. Each larger family now has an explicit source-bound image profile and rejects geometry from other families before allocating model tensors. Implementation and qualification remain distinct.

- **31B:** the no-PLE planner, per-layer KV counts, raw global K=V duplication, per-layer image masking and standardized 27-layer/head-72 image tower are implemented. CPU canonical vision references are complete. The corrected extracted component executes on sealed source `af356919`, but all three image-feature cases fail the frozen gates (relative RMS 0.05464–0.09914); stage diagnosis is pending. Full checkpoint generation is unqualified. Disk cannot retain both full source and converted weights with the reserve.
- **26B:** the no-PLE K=V decoder, larger standardized image tower and combined dense/expert MLP are implemented. The router preserves RMS/channel/root-width scaling, F32 softmax and top-eight renormalization followed by learned expert gains. Dense and expert outputs have separate norms before their BF16 sum/final norm. Controlled six-layer MoE fixtures pass CPU and Metal comparisons. The shared larger-tower failure also blocks vision qualification here. Released-checkpoint expert and end-to-end generation qualification remain pending.
- **12B:** the separate unified converter/runtime is implemented: raw row-major 48px patches, LayerNorm, biased projection, LayerNorm, learned factorized positions/LayerNorm, then weightless RMS/projection. The pinned source namespace is mapped explicitly and loaded strictly. Its extracted ten-tensor image embedder passes three canonical CPU-reference/Metal feature comparisons with exact patches and feature relative RMS at most 0.00007343. Full decoder/image generation is unqualified.

All no-PLE paths explicitly select the materialized BF16 final residual/scalar and rotary boundaries. A controlled six-layer residual fixture passes; the original six-layer unit-Q/K stress fixture still fails and remains a qualification blocker. One- and two-layer unit-Q/K controls pass. No controlled/component result substitutes for a full exact-checkpoint replay. Audio/video and speculative drafts remain outside this image task.

The larger official vision checkpoints retain `ClippableLinear` source wrapper
names (`*.linear.weight`) even with clipping disabled. The converter validates
those exact source names and maps them to plain native `*.weight` destinations.
This was checked against all 356 tensors in the pinned extracted 31B header;
header-only 26B/31B regression fixtures preserve that distinction.

## Upstream provenance

Equations were checked against [Transformers v5.15.1 Gemma 4](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/gemma4/modeling_gemma4.py), SHA-256 `4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3`, and [unified Gemma 4](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/gemma4_unified/modeling_gemma4_unified.py), SHA-256 `c52d9b27fc83b6ced166aee816b235e1db057fde10df6c1eff64bbc91fb16f39`. Both are Apache-2.0, Copyright 2026 the HuggingFace Team. Existing native image equations retain their independently implemented Google and MLX-VLM provenance in [UPSTREAM.md](../../../provenance/UPSTREAM.md).
