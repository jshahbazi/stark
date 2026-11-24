# Inkling 2 Mini pinned-runtime parity and 4K budget

The native slice passed connected parity at positions 4,096–4,099 against instrumented Ollama. The short capture missed the gate at one global expert row. The diagnostic stays opt-in while short-prefix parity and the rest of the target graph are qualified. [Original slice](2026-11-21-inkling-mini-native-slice.md) · [Capture and reproduction](../archive/experiments/2026-11-21-inkling-oracle/README.md) · [Aggregate receipt](data/inkling-mini-pinned-oracle-2026-11-21.json)

## Live reference and first mismatch

The reference uses Ollama `6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8` with its shipped Metal v4 libraries and actual original checkpoint cache state. The short run prefilled four tokens. The 4K run prefilled eight chunks of 512 tokens. Each captured four following singleton tokens through physical layers 10 and 11 on the second core pass. Each native run seeds the same residual blocks, cross-pass state, and visit caches from the runtime. The native local output feeds the global visit in the same command. [Pinned build, checkpoint, fixture hashes and cache export](../archive/experiments/2026-11-21-inkling-oracle/README.md)

The first differing 4K BF16 boundary was the normalized input pair to the cross-pass controller. At layer 10, four BF16 words differed across its three initial Q/K/V controller pairs. The old mixer factored the RMS scale out of the score and combined it with the softmax weight. The changed mixer rounds the normalized values, scaled query, dot products and weighted sum at the source's FP32 boundaries. The RMS reduction uses the source's four contiguous inputs per lane, two SIMD reductions and precise reciprocal square root. This makes all **110,592** initial controller-input BF16 words exact across the short and long captures.

Identical-ingress projections and weighted norms isolated the early difference. The padded matrix projection candidate produced more mismatches than the existing BF16 matvec on identical short ingress, so the native dot is retained. On the M5 Max, the controller expansion now accumulates ordered FP32 matrix fragments before the gate. Decode attention uses the source's 32-shard short and 64-shard long reductions over Stark's persistent BF16 cache layout. No extra command submissions or waits were added.

## Connected controls

The original numerical thresholds remain in force. Check the top-eight route set exactly, cross-pass messages within `2e-5` and BF16 state/attention within `0.008`, and connected expert output within `0.015` for both row relative L2 and maximum absolute error normalized by the reference peak. Reference expert ingress replay and global-visit ingress isolation are separate controls.

| Check | Short, positions 4–7 | 4K, positions 4,096–4,099 |
| --- | ---: | ---: |
| Connected stage-row failures, previous → changed | 37 → 3 | 37 → 0 |
| Connected route sets exact | 8 of 8 | 8 of 8 |
| Layer 10 expert worst row relative L2 | 0 | 0 |
| Layer 11 expert worst row relative L2 | 2.69% | 0 |
| Layer 11 post-attention worst row relative L2 | 0.00277% | 0.00518% |
| Full connected gate | Fail | Pass |

The short failure is one position in layer 11, observed in the expert result, the convolution write of that result and the final residual. The preceding native attention gate differs at one BF16 value across the four rows. Native layer 10 partials are bit-exact at both depths. Both depths pass expert and global-attention replay with exact reference ingress. At 4K all 18,432 connected BF16 expert outputs match bit-for-bit. The validation compares two visits. The 44-visit schedule is timed separately.

## 4K decode budget

The final build and prior `f93b4fb3` executable used the same original-weight 44-visit workload and the same controlled 4K seed fixture as the previous budget. Each process ran eight warmups and 32 timed singleton steps. Eight pairs alternated which build ran first. This comparison holds input shape, state format and cadence fixed. Exact routes can change as the connected numerical implementation changes. The timing covers encoding, one Metal command buffer and one completion wait. Trace copies and isolated replays stay outside the timed kernel path. [Performance playbook](../PERFORMANCE_PLAYBOOK.md#measuring)

| Measurement | Result |
| --- | ---: |
| Previous / changed wall median of process medians | 27.23 / 24.84 ms |
| Median paired wall ratio, changed ÷ previous | 0.912, lower wall time in 8 of 8 pairs |
| Changed process wall p10–p90 envelope | 24.72–24.96 ms |
| Dispatches / command buffers / completion waits per step | 1,756 / 1 / 1 |
| Remaining wall budget at 90% reference floor, greedy / sampled | 17.69 / 19.07 ms |

The same 18-core, 40-GPU-core M5 Max with 128 GiB ran macOS 26.7.2 (25G332) on AC in high-power mode. All 98 samples had nominal thermal state and normal memory pressure. The two measured processes accounted for 99.6% of observed Metal counter increments. The sampled windows recorded no pageouts or swapouts. The source full-worker ceilings remain 42.53 ms greedy and 43.91 ms sampled per useful token. This partial graph leaves out four MXFP8 visits, embedding, vocabulary readout, sampling, worker dispatch, vision residency and 4K prefill. [Original full-worker reference](2026-11-20-inkling-mini-m5-max.md)

## Decision

Retain the source-order controller and attention work in the opt-in diagnostic. Use the live fixture as the first test for the next serial changes. Isolate the remaining short global-visitation projection differences with identical ingress before extending a worker path. Qualify full-model logits, prefill, prefix reuse, vision and DSpark on the original artifact.
