# Inkling 2 Mini native worker on M5 Max

The original checkpoint's grouped Metal worker cleared the first 4K decode budget for greedy and sampled generation. Grouped prefill improved the two worker controls by 3.47× and 3.34×. It still falls short of the cache-cold prefill floors. [Attempt 20261121-INKLING-13](../model-experiments/2026-11-21-inkling-mini-worker.md)

Both native workers loaded the original `6e7c8248f75265bdd4fbab5273304243b5d8c2cb` target, vision tower and DSpark head on the 128 GiB M5 Max. The comparison uses one existing 4,096-token fixture per mode from the pinned reference corpus. Drafting and text prefix reuse were off. Greedy and temperature-1 sampling produced 256 tokens per arm. The worker times prefill through the first sample and decode from the first to last sample, with 255 useful decode intervals. Binary hashes are in the attempt record. [Source corpus and throughput floors](2026-11-20-inkling-mini-m5-max.md)

| Worker | Mode | Prefill seconds | Input tokens/s | Decode seconds | Useful decode tokens/s |
|---|---|---:|---:|---:|---:|
| Serial | Greedy | 114.52 | 35.77 | 8.10 | 31.47 |
| Serial | Temperature 1 | 114.63 | 35.73 | 8.31 | 30.67 |
| Grouped | Greedy | 32.98 | 124.19 | 7.39 | 34.52 |
| Grouped | Temperature 1 | 34.28 | 119.50 | 8.45 | 30.17 |

Grouped-worker steps averaged 28.97 ms greedy and 33.15 ms sampled, inside the 42.53 and 43.91 ms ceilings. Prefill targets are 626.15 and 589.58 input tokens/s. These are separate worker screening sessions with one request per cell. The eight paired native graph windows on a common build measured a 0.294 median tile-time ratio. A complete 4K comparison on one native build produced identical cache state, target taps and continuation logits with both schedules. Whole-model logits against the pinned Ollama prefill schedule remain outside parity. [Native graph controls and reference trace](../model-experiments/2026-11-21-inkling-mini-full-target.md)

## Final slice regression checks

The integrated native example reran the original pinned 4-token and 4,096-token slice fixtures. In each fixture all 18,432 expert-output BF16 words and all route sets matched, no BF16 stage row failed, and no unrelated cache writes occurred across 10,993,664 checked words. The earlier eight paired measurements established the slice's no-regression result. [Pinned slice and paired budget](2026-11-21-inkling-mini-short-projection.md)

The same binary completed 32 timed full-target 4K steps on a seeded cache. Median wall time was 27.93 ms, with 27.86 and 27.96 ms p10 and p90. It used one Metal command buffer, one completion, 1,883 dispatches and 800,232 bytes of F32 logits readback per step. This diagnostic excludes sampling and worker overhead.

| Final diagnostic artifact | SHA-256 |
|---|---|
| Native example | `bd3dcf1b90cac098b3747e60729f1420bfb279ca155bf48b8a125fb236334315` |
| Short slice aggregate | `3ec9bea159f2df8ab1c508329cd2b102cb53e672c740ae3fbddc9ba677cbfa63` |
| 4K slice aggregate | `c4a5cefe72c1633b5a56a7092e2e47479698d79936933e66910289b867c40652` |
| Full-target budget aggregate | `199ff0349ae71058d05c9348b22ab4d2a050f63e681e7183cbc7ea29389792ac` |
