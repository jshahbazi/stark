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

## 4K worker rerun after the matched trace

The original-checkpoint worker retained its production 64-row scheduler during the 512-row diagnostic work. This rerun alternated the previously measured grouped binary and the new worker in fresh sessions in AB/BA order. Greedy ran baseline first. The temperature-1 case ran the new worker first. Each session completed local image checks with vision and draft weights resident, then waited ten seconds before one timed request. Both builds ingested the same 4,096-token input and used the same seed within each mode, with draft execution off, zero reused input and 256 output tokens. The output token hash matched within each pair.

| Mode | Baseline prefill | Rerun prefill | Rerun input tokens/s | Rerun useful decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 33.47 s | 33.05 s | 123.93 | 34.89 |
| Temperature 1 | 36.06 s | 33.52 s | 122.20 | 31.20 |

Prefill wall ratios were 0.988 and 0.930. One pair per mode screens for regression and leaves the speedup unqualified. Useful decode averaged 28.66 ms and 32.05 ms per interval, within the existing 42.53 and 43.91 ms budgets. The cold prefill floors remain 626.15 and 589.58 input tokens/s. The host reported no thermal or performance warnings before or after the run.

The initial setup lacked the native template helper on the executable search path. Image readiness rejected that launch before a benchmark request. The rerun used the checkout's built helper on `PATH` and completed the local image checks for both arms. Its aggregate is SHA-256 `e3cf008530d300ce88005aa34863a2762675bb684475a2cfc21688fed641cb72`. Baseline worker SHA-256 is `2579436a9389bd97c5e19287a92bf745aa14105932f0f416770e18f7b4367f37`. Rerun worker SHA-256 is `64928c90e87754db290f198bed0e7a414c82290d27623fb5e6890b3784363035`. The benchmark used the production grouped path. [Matched prefill attempt](../model-experiments/2026-11-21-inkling-mini-matched-prefill.md) · [Stage profile](../model-experiments/2026-11-21-inkling-mini-prefill-profile.md)

The final diagnostic reproduced the short and 4K slice aggregate hashes above exactly. Its 32-step synthetic-cache full-target control measured 28.071 ms median wall, with 28.030 ms p10 and 28.282 ms p90. It used the same 1,883 dispatches and one completion per step. The new diagnostic SHA-256 is `2794f11765305a6607534e4ce3d4f136cb1dbba16823239af770681a7930968a` and the step-budget aggregate is `c637557451ca5241bfc6a169c0b8016f2eef11d593dd372e2d29e3e878662df0`.
