# 20261121-INKMATCH-01 · Inkling 2 Mini matched 512-row prefill

| Field | Recorded value |
| --- | --- |
| Executed | 2026-11-21 on the 128 GiB Apple M5 Max, macOS 26.7.2 build 25G332 |
| Checkpoint | Original vision-language target revision `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, shard SHA-256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1` |
| Reference | Ollama 0.40.17-rc3 source `6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8`, release Metal v4 MLX `0.32.4-82-gfc344c9`, Go 1.26.10. The capture extension leaves the target graph and cache implementation in place. |
| Candidate | Stark based on `4962169e888f1b7e8558a42826f7331976bb0ab8`, release Rust 1.98.1, locked, Metal 4.0, source probes restricted to the diagnostic |
| Workload | One pinned 4,096-token greedy input, eight 512-row prompt calls at context 4,352 and four singleton teacher rows. The shared private token bundle SHA-256 was `74d6457708d31fe3ba6aeeab41b836d6ac7cd825cb3837fc3ddacc64e0c19719`. |
| Status | First dense mismatch repaired in the matched diagnostic. Local attention and router logits qualified at visit 0 with the captured prompt. Full-model reference parity remains open. |

The earlier trace began after the 4K prompt. This run captured the first complete 512-row Ollama forward and collected the partial residuals for all 48 visits. Visit 0 also includes the dense input and outputs, K/V writes, relative table, attention and expert stages. The new capture reproduced every F32 teacher logit and BF16 final-hidden word from the archived 4K capture on all four continuation rows. An initial launch failed to locate the MLX libraries because the runner's relative symlink pointed outside the retained runtime. The corrected launch used the original released Metal v4 files.

The matched native baseline encoded eight 512-row layer-major commands. It retained the worker's four-row shared dense products, per-query attention and internal 64-row grouped NVFP4 tiles. It reproduced the previous B64 run's four cross-runtime logit error measurements to the reported precision. Matching the scheduling boundary alone did not change those results. [Capture and replay commands](../archive/experiments/2026-11-21-inkling-oracle/README.md#matched-512-row-prompt-trace)

## First mismatch and repair

Before the first attention projection, all 512 embedding residuals and the three BF16 normalized inputs matched. The first unequal word appears in the visit 0 attention gate at row 0. K and V first differ at row 2 and Q at row 3. The earliest connected visit 0 output differs at row 1. Ollama's large-forward BF16 linear projections accumulate in NAX matrix fragments. The native B64 implementation shares a singleton-style SIMD dot across four rows.

The source-shaped 16×32×16 BF16 tensor product uses the pinned fragment order. It retains the K=2,048 split for applicable contracting shapes and the BF16 output boundary. On this trace, Q, K, V, R, gate and sink projections all matched, as did the clipped Q and K/V writes. The sink remained on the existing narrow projection and matched in this control. The dense repair cleared all 151 differing gate words out of 1,572,864. One Metal shader build failed because the Apple SDK's cooperative tensor copy constructor rejected a const pointer. An explicit FP32 accumulator array resolved the compile failure.

An independent difference remained in the relative table with exact relative ingress. The matched local arm uses the source's BF16 K=16 NAX product for that table. It reads a chronological window assembled from the existing persistent ring and evaluates 16 queries against reversed 128-key blocks. It rounds unnormalized probabilities to BF16 before the value product and retains the source's F32 denominator and output gate. On the first chunk, all 12,582,912 relative-table words and all 1,572,864 gated-attention words then matched. The output projection, post-attention convolution and BF16 MLP ingress also matched.

The routed arm uses the pinned TF32 NAX router with exactly widened BF16 inputs and the source's K split. It matched all 65,536 F32 logits and 4,096 expert indices at visit 0. It reduced route-weight mismatches to 250 of 4,096. The first remaining BF16 visit 0 output difference is at row 52, where the expert sum already differs. This capture does not independently attribute the remaining expert error to route-weight rounding or packed expert arithmetic.

| Arm | Gate unequal | Relative table unequal | Gated attention unequal | Router logits unequal | Visit 0 BF16 output unequal | First unequal visit 0 output row |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched width | 151 | 2,350 | 291,453 | 65,517 | 1,098,646 | 1 |
| Source dense | 0 | 108 | 290,938 | 65,515 | 1,098,401 | 1 |
| Source dense and local attention | 0 | 0 | 0 | 57,383 | 8,045 | 52 |
| Source dense, local attention and router | 0 | 0 | 0 | 0 | 7,221 | 52 |

At 4K the final routed arm agreed on one of four greedy teacher decisions. The four relative L2 logit errors were 14.96%, 20.81%, 13.93% and 3.92%. The local-attention arm agreed on two decisions. These four rows support diagnosis and keep full cross-runtime parity open. The eight global attention visits still use the native per-query reduction. The production worker retains B64 scheduling and the previously qualified native kernels.

The follow-up worker screening `20261121-INKPREFILL-03` ran both original 4K workloads on the M5 Max. Each arm produced the full 256-token output and matched hashes within its pair with zero reused tokens. Prefill remained around 33–34 seconds on the new worker, within the observed run-to-run range. [Worker benchmark rerun](../benchmarks/2026-11-21-inkling-mini-native-worker.md#4k-worker-rerun-after-the-matched-trace)

## Private evidence identities

| Artifact | SHA-256 |
| --- | --- |
| Additional capture script | `a016674f3b2adfe492d0909bd0ac7d55719cbfab1e6a5cbea9f02899baafb88c` |
| Capture runner | `5d2d6c5e204e00b0cc2fcf670099cd66522fc5fd6ab63f10aecc9ae1fd17b293` |
| First-chunk trace | `c78a1392091e3d7d513e9c823476f67a379b49e054d1145891e9d158171adfc2` |
| Matched-width binary and aggregate | `1b5fa50ae39aec03a32d8ae7daa191050547aba58b2149dc401a49886290c2ce`, `41b933627803711b30401d44bb18c182989a092b349298af4478f2fbc8adc7af` |
| Dense binary and aggregate | `cd635f12e1e0e935f4de7e522b2d99c4ae8fc43dd5021d33b3d0084daae47a88`, `0483d96cb4c269afbc6b654591ac5a989897c13b3833c89345f6cf8ed1a002e0` |
| Local binary and aggregate | `d4dc7e0930b589091e1d9a768fdbf2af17d757d7ef1bd5e59137681034e0a524`, `b7c68c097b7e6843e60d3e245d0bee3b4c82d9f18b824e95f5bf5af63fdc9ceb` |
| Routed binary and aggregate | `aa73702587f4b586826bd46c9350b1f29a7677631fcbf9321bb777bb9f9480db`, `d2b9f293ef442d4e139a25cfef0f536be9b2bd94fd87da5de4325210c64f7c36` |
