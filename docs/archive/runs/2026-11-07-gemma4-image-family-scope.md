# Gemma 4 image family architecture assessment

Date: 2026-11-07

The original eight native OLMo 3/Gemma 4 commits, ending at `294b228663c3bf47c6ee250aa4083cb42c060aa3`, were merged onto current main `8198f748` as `3bc9e8cd`. This retains the original E2B BF16 image implementation and all canonical and affine-source precision repairs. The prior E2B validation remains bound to its original artifacts and builds.

## Source-bound scope

The following official configurations were inspected at their immutable Hugging Face revisions. All sizes accept images upstream. Their native requirements differ.

| Official checkpoint | Revision | Native architecture requirements |
| --- | --- | --- |
| [E2B-it](https://huggingface.co/google/gemma-4-E2B-it/blob/3e22461f65e89153144f8adb70e3b8c2cc9845a7/config.json) | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | Existing 35-layer PLE decoder, width 1,536, 20 shared KV layers; 16-layer vision tower, width 768/head 64, clipped linears. |
| [E4B-it](https://huggingface.co/google/gemma-4-E4B-it/blob/4ecc2ba63bcac23ba4a767fc87025b1450f84c9c/config.json) | `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c` | 42-layer PLE decoder, width 2,560, two KV heads, 18 shared KV layers, ordinary-width MLP. Same 16-layer vision geometry and processor as E2B, with a 2,560-wide projector and distinct tokenizer bytes. Implemented as its own schema-4 profile; exact checkpoint qualification is recorded separately. |
| [12B-it](https://huggingface.co/google/gemma-4-12B-it/blob/43c9d014b231474fa7212dbb48a762566c2b255d/config.json) | `43c9d014b231474fa7212dbb48a762566c2b255d` | Unified encoder-free image embedder, width 3,840; 48 layers, no PLE, global K=V attention with one global KV head; image-block bidirectional masks. |
| [26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/4d7ae4984b7db7de8f8457170b3f1a419ee76d52/config.json) | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | 30 layers, width 2,816, 128 experts/top 8, shared dense plus expert residual branches; no PLE, global K=V/two global KV heads; 27-layer width-1,152/head-72 vision tower with learned output standardization and ordinary linears. |
| [31B-it](https://huggingface.co/google/gemma-4-31B-it/blob/842da3794eaa0b77d5f08bae87a17459d91ff475/config.json) | `842da3794eaa0b77d5f08bae87a17459d91ff475` | 60-layer dense decoder, width 5,376, no PLE, global K=V/four global KV heads and image-block bidirectional masks; same larger vision geometry as 26B. |

## Larger-model implementation and remaining blockers

The original E2B/E4B implementation required PLE tables, separate K and V projections and causal image attention. This work adds an explicit larger dense 31B path with its own geometry and source profile. Each path rejects other family/source profiles before allocating model tensors.

- **31B:** implemented the no-PLE decoder planner and global K=V projection semantics, four global KV heads, image-block bidirectional attention, width-1,152/head-72/depth-27 vision tower and ordinary linear namespaces. New padding/unpadding kernels reuse the online attention primitive without truncating 72-wide heads. A dedicated pool kernel preserves F32 square-root scaling and learned `std_bias`/`std_scale` standardization before BF16 storage. Native production compilation and 17 focused CPU tests pass. Full-checkpoint generation is unqualified; the available disk cannot retain both a 62GB source and a converted artifact with the required reserve. Vision-only extraction and reference checks are separate.
- **26B:** reuse the larger vision and K=V paths and implement Gemma-specific routing and expert composition. The upstream router applies an RMS normalization, learned input scaling, F32 softmax, top-8 probability renormalization, then learned per-expert scales. The decoder uses separately normalized dense and expert branches and a final combined norm. Existing generic MoE conversion does not represent this complete Gemma branch topology.
- **12B:** implement a separate `gemma4_unified` converter/runtime path. Its image embedder groups patches to model patch size 48, applies patch LayerNorm, learned projection with bias, a second LayerNorm, factorized learned positions and positional LayerNorm, then a multimodal RMS/projection. It has no 16-layer image encoder. Reusing the E2B tower would run a different model. The decoder also needs no-PLE/K=V and image-mask support.

These are implementation blockers. Model size alone is not the reason for rejection. Audio/video and speculative draft models remain outside this image task.

## Upstream provenance

Equations were checked against [Transformers v5.15.1 Gemma 4](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/gemma4/modeling_gemma4.py), SHA-256 `4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3`, and [unified Gemma 4](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/gemma4_unified/modeling_gemma4_unified.py), SHA-256 `c52d9b27fc83b6ced166aee816b235e1db057fde10df6c1eff64bbc91fb16f39`. Both are Apache-2.0, Copyright 2026 the HuggingFace Team. Existing native image equations retain their independently implemented Google and MLX-VLM provenance in [UPSTREAM.md](../../../provenance/UPSTREAM.md).
