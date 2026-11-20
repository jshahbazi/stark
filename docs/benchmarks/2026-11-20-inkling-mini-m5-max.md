# Inkling 2 Mini on the M5 Max

## Measurement setup

The host was an 18-core Apple M5 Max with 128 GiB unified memory, macOS 26.7.2 (25G332), on AC power with power mode 2. Only one experiment used the GPU at a time. An existing Stark worker remained resident and the desktop stayed active. Each server had one sequence and an 8,192-token context. Text ran with vision loaded and drafting disabled. The main checkpoint, draft weights, GGUF, projector and runtime versions are bound to independent digests. [Registered protocol](../model-experiments/2026-11-20-inkling-mini-milestone-1.md) · [Port plan and source revisions](../INKLING_2_MINI_PLAN.md)

Each input had exactly 512 or 4,096 model tokens. The official tokenizer and pinned GGUF vocabulary produced identical sequences for all 48 prepared requests. Their combined token-ID SHA-256 was `f4763104df51a82539261dd30a2d001d9688275812d12709ed3ce9f560276187`. The private fixture file SHA-256 was `fade74992579dc76ff0f5319eec5b70c9e07be7c54338ebae7f43c1097fe98bd`. Both engines received identical raw template-rendered requests, alternately drawn from generated code and prose. The output cap was 256 tokens with natural stopping. Sampling used temperature 0 or 1, top-p 1, disabled top-k and min-p, unit repetition penalty, and a matched seed per pair. Generation trajectories may differ across artifacts and samplers. [Fixture generator](../archive/experiments/2026-11-20-inkling-mini/workloads.py)

## Original-weight component budget

The connected test exercised the layer 0 router, all 128 packed experts, four-over-six quantization, indexed gate/up, BF16 projection rounding, soft-capped SwiGLU with routing weights, down quantization and projection, and the F32 sum rounded to BF16. Thirty-two token positions from the original checkpoint's reconstructed first-layer inputs selected 79 distinct experts. Singleton and tile executions matched every CPU top-eight set. On three positions they matched all 144 checked final BF16 values, including the two nonlinearities and eight-way reduction. The remaining positions matched across GPU execution shapes over all output channels. Full-model logits require a separate check. [Connected benchmark](../archive/experiments/2026-11-20-inkling-mini/connected_moe.py) · [Original-layer input reconstruction](../archive/experiments/2026-11-20-inkling-mini/original_layer0.py)

The following calculation uses original weights on Apple MLX 0.31.2. Each row times a different part of the model. The ordered control executes eight singleton graphs with explicit completion dependencies, rotating inputs and one terminal host wait. The eight executions reuse the layer's resident weights. The four MXFP8 visits use separate gate/up and down measurements with eight route-specific down inputs. Attention uses fixed, synthetic chronological KV and convolution history. For the local stage the latest singleton attends over 1,024 positions, and for the global stage over 4,096. [Auxiliary benchmark](../archive/experiments/2026-11-20-inkling-mini/auxiliary.py)

| Component | Visits/token | Ordered ms/visit | Weighted ms/token | Isolated 32-row tile, ms |
| --- | ---: | ---: | ---: | ---: |
| NVFP4 routed sublayer, layer 0 | 44 | 0.155 | 6.83 | 2.166 |
| MXFP8 projections, layer 20 | 4 | 0.260 | 1.04 | 3.118 |
| Four learned message reads and gates, layer 6 | 24 | 0.138 | 3.31 | 0.626 |
| Local attention sublayer, layer 6 | 40 | 0.294 | 11.76 | 0.943 |
| Global attention sublayer, layer 11 | 8 | 0.464 | 3.71 | 1.785 |
| **Measured-stage estimate** | | | **26.65** | |

The same weighting with a separate host wait per call gives **67.36 ms**. For the connected NVFP4 sublayer alone, isolated singleton medians ranged from 0.607 to 0.685 ms across three sessions. The eight ordered sublayers took 1.242 ms together. The gap shows how a prototype with per-stage synchronization can misstate the budget. The 26.65 ms estimate also has limits. Cross-layer working sets differ, the attention probe constructs and widens a chronological cache view, and the rest of the decoder still has to run. Omitted costs include block residual reads and updates, post-sublayer convolutions, persistent cache and ring writes, the MXFP8 router and pointwise combination, head, sampling and worker overhead. Timing one complete native token is the test of the budget.

The layer 20 MXFP8 check used a direct E4M3 weight decode, an independent activation codebook and FP64 dot products. Each of the four projection/shape cases checked 256 output values and had zero unequal reconstructed activation values. Relative L2 was at most `4.37e-8`. Local attention, global attention and the cross-pass gates produced finite outputs with learned weights. Their tests establish cost on these shapes. Matching the original's layer outputs, cache contents and logits is a later numerical gate.

The earliest MXFP8 screen shared one down input across experts. The reported budget uses independent rows for the eight down routes and the upstream exponent-bit scale rule. Earlier outputs remain separate from the final geometry. The complete routed NVFP4 screen also retains the failed early-scale arm and the corrected delayed-scale arm. [Early kernel receipt](../archive/experiments/2026-11-20-inkling-mini/nvfp4-block6-screen.json) · [Kernel reproduction and timing boundaries](../archive/experiments/2026-11-20-inkling-mini/NVFP4_PROBE.md)
