# Inkling 2 Mini on the M5 Max

**Proceed with a bounded original-checkpoint Stark port.** The packed expert operations cleared the precision gates, and the measured components leave room in the original model's serial budget. The next investment is one executable sequence of local and global visits in Stark, with actual cache writes, convolutions, cross-pass messages and residual updates on the same command stream. Ship decisions come after that sequence and the whole worker are measured.

At 4,096 input tokens the original took the equivalent of **38.28 ms per useful greedy token** and **39.52 ms sampled**. The 26.65 ms component estimate leaves **11.63–12.87 ms** to match those reference medians. The proposed 90% throughput floors allow 42.53 and 43.91 ms respectively. Use these targets for the full native reference-matched run.

| 4,096-token native acceptance floor | Greedy | Temperature 1 |
| --- | ---: | ---: |
| Cache-cold prefill, input tokens/s | 626.15 | 589.58 |
| Useful serial decode, output tokens/s | 23.51 | 22.77 |

Use one terminal wait, persistent per-visit KV/ring/conv state, and the same scale ordering for singletons and grouped prefill. Compare teacher-forced route sets, visit outputs and logits against the original. Profile local and global attention separately. If the visits plus a measured allowance for omitted work exhaust the budget, optimize the two largest stages before expanding the graph. The complete worker also needs image-conditioned execution, exact prefix restoration and independent DSpark qualification. Keep Q4_K_XL as the user-visible speed and latency comparison as those stages mature.

## Matched server results

Each row reports medians from eight requests on each engine. All 64 measured requests admitted the exact input, reused zero prompt tokens and generated 256 output tokens. Every pair passed the host gate. Four pairs per cell ran Unsloth first and four ran the original first. The four predeclared spare windows per cell stayed unused. [Aggregate distributions, per-order ratios and source hashes](data/inkling-mini-m5-max-2026-11-20.json)

| Input and sampling | Engine | Prefill tok/s | Useful decode tok/s | First visible text, s | Request wall, s |
| --- | --- | ---: | ---: | ---: | ---: |
| 512 greedy | Original | 714.34 | 23.46 | 0.831 | 11.989 |
| | Unsloth Q4_K_XL | 777.72 | 36.81 | 0.735 | 7.619 |
| 512 temperature 1 | Original | 736.38 | 23.74 | 0.819 | 11.544 |
| | Unsloth Q4_K_XL | 849.58 | 37.00 | 0.691 | 7.529 |
| 4,096 greedy | Original | 695.72 | 26.13 | 5.983 | 15.661 |
| | Unsloth Q4_K_XL | 660.39 | 36.30 | 6.294 | 13.287 |
| 4,096 temperature 1 | Original | 655.09 | 25.30 | 6.418 | 16.463 |
| | Unsloth Q4_K_XL | 641.41 | 35.91 | 6.503 | 13.726 |

At 4,096 tokens the median paired Unsloth/original useful-decode ratio was **1.403× greedy** (range 1.250–1.618×) and **1.384× sampled** (1.287–1.565×). The corresponding full-request wall-time reductions were **14.1% and 16.1%**. The original delivered first text sooner at the median in both 4K cells. Using the same client-side interval from first visible text to request end gave 4K paired rate ratios of 1.410× and 1.390×. The greedy engine-timed ratio varied from 1.319× to 1.503× between launch orders. These results measure the two released artifact/runtime combinations.

Fresh original processes reached model-ready in 10.35 and 5.90 s. Both reported 19.93 GiB of model allocation. GGUF reached ready in 5.87 and 7.14 s, with peak observed RSS of 20.86 and 24.35 GiB. Its respective Darwin lifetime peak physical footprints were 8.09 and 8.25 GiB. Model allocation, process RSS and physical footprint use different accounting. The OS file cache was warm from artifact checks and previous launches. Importer full wall time and peak import RSS were unavailable. In the original import, all 368 sampled byte windows matched across 176 expert arrays. [Aggregate data](data/inkling-mini-m5-max-2026-11-20.json)

Thermal state stayed nominal or fair and every memory-pressure reading was normal. At the roughly two-second sampling cadence, the original process accounted for 89.6% of recorded Metal counter increments during its timed requests and llama-server for 94.3% during its requests. Other graphics work remained active. Increment intervals can start just before a request. The system pageout counter increased across the experiment and the swapout counter had zero change across the timed samples for each arm. Absolute rates apply to this host condition.

Eight separate 480-token prefixes ended inside ordinary text. Both engines generated a single nonempty, valid UTF-8 piece on every prefix and agreed on **six of eight**. Use the official checkpoint for native numerical checks. Both image-loaded servers correctly identified the same red and blue 120×120 PNGs in separate chat requests. Each used 35 input tokens, zero cached tokens, and stopped naturally. The two native chat APIs apply their own templates and handle reasoning text differently. These are two bounded image-answer checks. The 80×80 GGUF transport control separately increased the input count from 25 to 32 with the image attached. [Aggregate controls](data/inkling-mini-m5-max-2026-11-20.json)

## Measurement setup

The host was an 18-core Apple M5 Max with 128 GiB unified memory, macOS 26.7.2 (25G332), on AC power with power mode 2. Only one experiment used the GPU at a time. An existing Stark worker remained resident and the desktop stayed active. Each server had one sequence and an 8,192-token context. Text ran with vision loaded and drafting disabled. The main checkpoint, draft weights, GGUF, projector and runtime versions are bound to independent digests. [Registered protocol](../model-experiments/2026-11-20-inkling-mini-milestone-1.md) · [Port plan and source revisions](../INKLING_2_MINI_PLAN.md)

Each input had exactly 512 or 4,096 model tokens. The official tokenizer and pinned GGUF vocabulary produced identical sequences for all 48 prepared requests. Their combined token-ID SHA-256 was `f4763104df51a82539261dd30a2d001d9688275812d12709ed3ce9f560276187`. The private fixture file SHA-256 was `fade74992579dc76ff0f5319eec5b70c9e07be7c54338ebae7f43c1097fe98bd`. Both engines received identical raw template-rendered requests, alternately drawn from generated code and prose. The output cap was 256 tokens with natural stopping. Sampling used temperature 0 or 1, top-p 1, disabled top-k and min-p, unit repetition penalty, and a matched seed per pair. Generation trajectories may differ across artifacts and samplers. [Fixture generator](../archive/experiments/2026-11-20-inkling-mini/workloads.py)

| Arm | Artifact and runtime |
| --- | --- |
| Original | Publisher safetensors, 19,526,019,168 bytes, SHA-256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`. Isolated Ollama `v0.40.17-rc3` executable `1abf07524333cd1f4e4c1b81362523dca213e217af0bb146f7d392dc29742325`, with its Metal v4 MLX library mapped at runtime. Import produced manifest `bd9c92c3d4c4c1a6c6904dea6a51e8503cceb0f77f8073c2c9e5c63976e0aa4d`, including 422 vision tensors. The separate publisher DSpark file was SHA-verified and kept outside the serial import. [Original source](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/6e7c8248f75265bdd4fbab5273304243b5d8c2cb) · [Ollama release](https://github.com/ollama/ollama/releases/tag/v0.40.17-rc3) |
| Unsloth | `UD-Q4_K_XL`, 17,629,759,456 bytes, SHA-256 `e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15`, with the 639,348,160-byte F16 projector. The pinned Unsloth release server was `9c99f4b9182aa90830c9bd15d2635707afeab55e7e64dbe3f37de97eaf4115a8`. A separate startup trace with identical model options reported Metal, 49/49 layers offloaded and the Inkling projector. The GGUF retains NVFP4 in 40 expert-bank tensors and uses Q8_0 for the final eight. [Unsloth files](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/a98a3457c896cd784094c733e38ace2ac129ab74) · [Unsloth runtime](https://github.com/unslothai/llama.cpp/releases/tag/b12572-mix-8eebad4) |

Both clients report useful decode as `(256 − 1) / engine decode seconds`. The GGUF server closes prefill after synchronizing its first sampled token and records the last sample before text handling. The MLX runner synchronously prefills through the penultimate input token, queues the last forward and first sample, then starts its decode timer. That interval can include completion of the first sample, a pending lookahead sample and cache cleanup. Hence the per-stage rates retain their engine timing boundaries. Client time to first nonempty streamed text and complete request wall time share boundaries across arms. [Ollama scheduler](https://github.com/ollama/ollama/blob/6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8/mlxrunner/scheduler.go#L529-L562) · [Ollama pipeline](https://github.com/ollama/ollama/blob/6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8/mlxrunner/pipeline.go#L292-L334) · [GGUF timing implementation in the released source archive](https://github.com/unslothai/llama.cpp/releases/tag/b12572-mix-8eebad4)

For reproduction, import the complete original directory with the pinned `ollama create`, an isolated model store, `RENDERER inkling2`, `PARSER inkling2` and `PARAMETER draft_num_predict 0` in its Modelfile. Use one worker and one parallel slot. The benchmark sends raw prompts with the explicit draft limit. The GGUF harness launches the pinned server with the F16 projector, one slot, batch 2,048, microbatch 512, `--n-gpu-layers 99` and prompt caching disabled. Record host samples concurrently, run each server separately, then pair receipt files with the comparison script. [Original harness](../archive/experiments/2026-11-20-inkling-mini/ollama_bench.py) · [GGUF harness](../archive/experiments/2026-11-20-inkling-mini/gguf_bench.py) · [Host sampler](../archive/experiments/2026-11-20-inkling-mini/host.py) · [Comparison](../archive/experiments/2026-11-20-inkling-mini/compare.py)

## NVFP4 kernel on real routes

The probe linked the released Unsloth Metal backend. It captured one 32-row prefill and the next teacher-forced singleton for both projections on each of three visits to physical layers 6 and 11. These captures came from the live GGUF graph. Two additional gate/up cases used CPU-reconstructed layer 0 inputs from the original checkpoint. The CPU reference decoded the original E2M1 weight codes and E4M3 scales independently, checked the GGUF row permutation and per-expert scales, and accumulated sampled rows in FP64. Each core layer exercised 71 distinct routed experts. [Implementation and reproduction](../archive/experiments/2026-11-20-inkling-mini/NVFP4_PROBE.md)

| Candidate or control | Finding |
| --- | --- |
| Restore the activation and expert outer scales after the indexed dot product | Passed all 26 cells. Worst relative L2 was `6.00e-8` and worst max error normalized by reference peak was `1.05e-7` over 155,520 projection samples. All complete GPU outputs were finite. |
| Restore activation outer scale before the dot | Failed the numerical gate in 11 of the 12 live 32-row grouped cells. The initial eight-expert layer 6 screen had already exposed this failure. |
| Skip the released four-over-six activation quantization | Changed reference projection outputs by 5.96–9.06% relative L2 on the core captures. The original-policy Metal activation pass matched all 4,529,664 checked values, including repeated execution shapes. |
| Use grouped products at 32 rows | Median gate/up and down took 0.753 and 0.458 ms. Replaying the same inputs as singleton products took 1.819 and 1.441 ms. The activation dispatch and staging between libraries sit outside these projection timings. |

The two runs agreed on all 24 captured input and routing hashes and the three last-position logit hashes. Each readout contained 200,058 finite logits. The callbacks that timed the GGUF model initially matched view nodes more than once. Requiring the exact node names yielded 96 alternating boundaries for 48 visits. On one instrumented decode, the intervals through route selection totalled 24.11 ms and the intervals from routing through layer end totalled 15.49 ms. Synchronizing those boundaries perturbs timing. They identify where to instrument the port, while the uninstrumented server requests supply throughput.

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
