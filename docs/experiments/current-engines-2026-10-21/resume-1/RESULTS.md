# Fresh Qwen3.5-2B observed comparison

2026-10-22. All 24 independent model loads completed, yielding 432 fixed-work performance measurements and 96 separate sanity measurements. Every engine was measured afresh in this campaign; no earlier timings were reused.

Stark recorded 245.5 tokens/s against MLX-LM's 218.8 tokens/s, a 12.2% difference in this observed run. Background activity and thermal flags affected the run. The user directed acquisition to proceed after quiet admission failed. All 24 loads carry interference exclusions under the original clean protocol; this report includes their acquired samples explicitly as observed data. It is not a clean-host publication result.

## Observed results

The aggregate is a geometric mean of the six workload rates. Each workload rate is the geometric mean of four per-load medians, each based on three repeats. Rates include prefill, generation, transport/readout and request cleanup; model loading and warmup are excluded.

| Engine | Observed tokens/s | Strict sanity matches |
|---|---:|---:|---:|
| Stark | 245.5 | 4/16 |
| MLX-LM | 218.8 | 4/16 |
| llama.cpp | 147.9 | 12/16 |
| TensorFold (drafting) | 125.6 | 4/16 |
| TensorFold (no drafting) | 118.5 | 4/16 |
| vLLM Metal | 174.7 | 4/16 |

Four unique sanity prompts were repeated once per load. These strict-match smoke checks are too small for an overall quality claim. The differing outcomes, especially with the separate GGUF representation, need investigation before publication.

## Workload rates

Observed client tokens/s, aggregated across all four loads, including host-flagged loads.

| Workload | Stark | MLX-LM | llama.cpp | TF draft | TF plain | vLLM Metal |
|---|---:|---:|---:|---:|---:|---:|
| fibonacci-module | 276.3 | 247.7 | 163.6 | 132.8 | 142.5 | 211.3 |
| gpu-explanation | 267.1 | 240.9 | 159.8 | 131.8 | 140.9 | 210.9 |
| copy-edit | 252.6 | 219.3 | 151.3 | 192.5 | 121.8 | 167.3 |
| context-medium | 232.6 | 216.6 | 146.0 | 114.1 | 113.3 | 148.8 |
| context-long | 179.0 | 166.1 | 113.0 | 70.5 | 70.4 | 135.9 |
| long-output | 281.9 | 233.1 | 160.1 | 144.8 | 142.0 | 188.5 |

## Method and limits

Host: Apple M5 Max, 128 GiB unified memory, macOS 26.7.1. Concurrency one, context 8192, greedy sampling, seed zero where supported. Each load warmed all ten fixtures. Five performance fixtures generated exactly 128 tokens and long-output generated exactly 512. All 432 performance samples passed the token-count, input-count and length-termination checks.

The fixed rotate/reverse schedule used four independent loads per configuration. Four blocks do not fully balance all six order positions. Per-load values and ranges remain in the observed summary; this design supplies no confidence interval or universal ranking. Prefix/retained-cache controls were requested where exposed; identical cache behavior is unverified. Pipe and HTTP overhead and upstream dependencies are part of each configured stack.

The shared checkpoint is mlx-community/Qwen3.5-2B-MLX-4bit revision 93760be4f1f69842a46bc13dbdc0f19e291392a3, affine W4/G64, with an unchanged Stark repack. llama.cpp uses the separately labelled Unsloth Q4_K_M GGUF; its quantizer differs and original source-weight equivalence is unverified. This compares engine-and-representation stacks.

Measurement host-screen peaks are retained below. Thermal state zero is nominal. CPU percentages may exceed 100% across cores; the GPU is shared. No unrelated workloads were stopped.

| Engine | Peak external CPU | Peak thermal state | Pageouts / swapouts |
|---|---:|---:|---:|
| Stark | 606.6% | 1 | 48 / 0 |
| MLX-LM | 691.8% | 1 | 61 / 0 |
| llama.cpp | 697.8% | 1 | 323 / 0 |
| TensorFold (drafting) | 610.5% | 1 | 2 / 0 |
| TensorFold (no drafting) | 554.3% | 1 | 580 / 0 |
| vLLM Metal | 556.6% | 1 | 389 / 0 |

The power log check found zero sleep events. Wall/monotonic discrepancy was 0.002000 seconds. Acquisition took 13.0 minutes. Quiet-window admission was disabled and timing_accepted is false by construction for this user-directed mode.

## Builds and reproducibility

Stark uses the verified main-build bundle from 97f15f38201cc0b465a81c45e26a4b5b135d903c, with native inputs unchanged in main bba01710d7996c10713c4968a758288966ca50f6. MLX-LM is 0.32.3.dev6+g72be8c955 with MLX 0.32.4; llama.cpp is b11747; TensorFold is 1.1.1; vLLM Metal is 0.32.0 with its locked dependencies. No engine or model implementation was patched for this comparison.

- [Run policy and hashes](RUN_4_PROTOCOL.md)
- [Aggregate acquisition receipt](campaign-4.json)
- [Observed summary, per-load rates and identities](campaign-4-observed-summary.json)
- [Observed summarizer](summarize_observed.py)
- [Original clean-gate summary](campaign-4-summary.json)
- [Original artifact identities](ARTIFACTS.json)
- [MetalRT exact-model support gap](METALRT_ADMISSION.md)

The clean-gate summary retains allow_public_table: false. The public site is unchanged. All task-owned workers, temporary server sessions and sleep assertions have stopped. Runtime environments remain ignored and available for a future explicitly requested run; no automatic retry is scheduled.
