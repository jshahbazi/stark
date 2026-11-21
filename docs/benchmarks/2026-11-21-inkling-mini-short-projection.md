# Inkling 2 Mini short-prefix projection and 4K budget

The original checkpoint passes the connected two-visit gate at both prefixes on the M5 Max. The source-order BF16 projection corrected the first layer 11 divergence. The 44-visit paired 4K budget stayed inside the 5% regression bound. [Predecessor](2026-11-21-inkling-mini-pinned-oracle.md) · [Attempt 20261121-INKLING-07](../model-experiments/2026-11-21-inkling-mini-short-projection.md)

## Connected parity

The live fixture uses Ollama `6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8` and its released MLX libraries. At physical layer 11, position 4, the attention gate projection differed in one BF16 word on the same normalized input. Replacing that gate with the captured result alone corrected 2,148 differing expert words and all three failing connected stage-rows. Replacing the query projection or first controller rank left those failures intact. F64 CPU accumulation and padded Thin NAX replay still differed at the gate.

The released MLX matmul selected `gemv_wide_bfloat16_nv4_kl16_nc0_axpby0` for wide outputs and the `kl32` variant for narrow outputs. Its eight-row padded calls matched all 18 isolated projected tensors across four input rows. Stark now groups eight FP32 four-element dot products and follows the pinned GemvWide reduction order on the M5 Max. Dispatch count and BF16 storage stay the same. [Capture protocol](../archive/experiments/2026-11-21-inkling-oracle/README.md)

| Gate | Short positions 4–7 | 4K positions 4,096–4,099 |
| --- | ---: | ---: |
| Failing connected stage-rows, baseline → candidate | 3 → 0 | 0 → 0 |
| Mismatching connected BF16 expert words | 0 of 18,432 | 0 of 18,432 |
| Route set and cache mismatches | 0 | 0 |
| Mismatching isolated BF16 projection words | 0 | 0 |

## Paired 4K slice budget

Each process starts from the same 4,096-token seed, then runs eight warmups and 32 timed singleton steps through 44 original NVFP4 visits. The eight pairs alternate run order. Timing includes encoding, one command buffer and one completion, with 1,756 dispatches per step in both arms. Trace reads and isolated replays stay outside the timed path. The host ran on AC in high-power mode with nominal thermal state and normal memory pressure across 123 samples. It recorded zero pageouts and swap deltas. The two arms accounted for 99.75% of observed Metal counter increments. [Budget and reference ceilings](2026-11-20-inkling-mini-m5-max.md)

| Measurement | Baseline | Source-order projection |
| --- | ---: | ---: |
| Wall median of process medians | 27.190 ms | 26.430 ms |
| GPU median of process medians | 26.034 ms | 25.183 ms |
| Wall median range across processes | 25.269–27.689 ms | 24.233–26.826 ms |
| Remaining wall budget, greedy / sampled | 15.34 / 16.72 ms | 16.10 / 17.48 ms |

The median paired wall ratio was 0.965 with lower wall time in all eight pairs. The median paired GPU ratio was 0.962. This clears the no-regression criterion and falls below the playbook's 10% practical speedup threshold. The historical 24.84 ms baseline came from a separate host session. The regression decision uses the 27.190 ms baseline measured in the current paired session.

The timing excludes four MXFP8 visits, embedding, vocabulary readout, sampling, worker dispatch, vision residency and prefill.
