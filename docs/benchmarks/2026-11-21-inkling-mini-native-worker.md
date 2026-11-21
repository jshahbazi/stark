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

## 128-row worker prefill

The M5 Max worker now uses 128-row commands, batched cross-pass controllers, grouped MXFP8 and NVFP4, and eight-row shared BF16 projections. It also groups expert tiles from 16 rows. In the same-session model control, 4K ingestion took 23.28 s versus 32.03 s and produced identical live state, target taps and four continuation logits. The new workspace uses 89.8 MiB of the existing 128 MiB reserve. [Model control and screening](../model-experiments/2026-11-21-inkling-mini-faster-prefill.md)

The original image-loaded and draft-resident worker benchmarks used one matched 4,096-input, 256-output request per mode and build. The worker ran the builds in AB/BA order with a ten-second wait after local image checks. Sampling controls, seeds and input tokens matched within each pair, and all requests reported zero reused inputs and draft execution off. Output hashes matched in both modes.

| Mode | B64 prefill | B128 prefill | B128 input tokens/s | B64 useful decode tokens/s | B128 useful decode tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 32.87 s | 23.41 s | 174.95 | 35.46 | 33.93 |
| Temperature 1 | 32.86 s | 23.28 s | 175.96 | 33.58 | 33.69 |

Prefill wall time fell by 28.8% greedy and 29.2% sampled. Decode met the 42.53 and 43.91 ms ceilings, and the prefill floors remain open. The repeated model-window controls and the same-session 4K pair support the performance change. The worker comparison is one independent pair per mode. The host reported no thermal or performance warnings before or after the run.

Baseline worker SHA-256 is `64928c90e87754db290f198bed0e7a414c82290d27623fb5e6890b3784363035`. B128 worker SHA-256 is `b239258f6d1bb4280313517389030b07ef0eed32e08560680af54ae67bed4da5`. The aggregate SHA-256 is `1f46160e14bd2ef0ffd2ae76c963b3361f7fd4f8f8cefbedbcff36bd7fff3dc8`.

## Batched RMS and direct expert bindings

The M5 Max worker batches controller RMS and stages activations directly into and out of the grouped expert buffers. The incumbent and candidate used the same original checkpoint, vision and draft weights resident, draft execution off, zero reused input tokens, 4,096 input and 256 output tokens. The request followed a ten-second wait after readiness. All output hashes matched.

| Mode | Pairs | Incumbent prefill | New prefill | Incumbent input tok/s | New input tok/s | Incumbent / new useful decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Greedy | 1 | 23.27 s | 20.48 s | 176.03 | 199.98 | 35.56 / 35.54 |
| Temperature 1 | 3 | 23.30 s | 20.74 s | 175.80 | 197.49 | 33.38 / 33.14 |

The sampled medians include the initial slower observation. That candidate request measured 182.53 input tok/s and 28.41 useful decode tok/s. The two repeats reversed order on the same seeded workload and measured 197.49–200.25 input tok/s and 33.14–33.54 useful decode tok/s. The requests accounted for at least 99.95% of observed GPU activity during sampled intervals in the repeat, with no pageouts or swapouts. One incumbent repetition reached fair thermal state. The synthetic singleton decode check measured 28.025 to 28.055 ms median. The cold prefill floors remain 589.58–626.15 tok/s. [Numerical and timing controls](../model-experiments/2026-11-21-inkling-mini-prefill-budget.md)

Candidate worker SHA-256 is `aae3755e67daf4142eac0bed33acaed8fe6c393cf824e85d41b9374f53f7b00a`. The incumbent is the B128 worker above. The first aggregate, sampled repeat and repeat host summary have SHA-256 `6a7944a8567489ef1a6bad7b332e404cd92b3bd267ba0121996208fc4f3df5ed`, `bbc5cd926b5a6dad47abe0b04d2b2ed0ea6d88972f5bed8dfaf7354dce615bd5` and `4cc48e800bacec932891247a4c21be494b6f0268efe313922f407cde460b8c16`.

## Packed 512-row prefill

The M5 Max worker processes up to 512 text rows in one command with packed activations and eight-route MXFP8. Image embeddings and draft-enabled text use the existing 128-row path. The 384 MiB reserve covers the 252.3 MiB packed allocation and the previously measured 89.8 MiB per-row workspace. The benchmark uses the original checkpoint with vision and draft weights loaded, drafting off, 4,096 input tokens, 4,352 context slots, 256 outputs and zero reused tokens. Each fresh worker completed image readiness and waited ten seconds before the timed request.

| Mode | Pairs | Incumbent prefill | Packed prefill | Incumbent prefill tok/s | Packed prefill tok/s | Incumbent / packed useful decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Greedy | 1 | 20.730 s | 9.056 s | 197.59 | 452.30 | 34.12 / 33.95 |
| Seeded temperature 1 | 3 | 22.678 s | 11.277 s | 180.62 | 363.22 | 30.44 / 31.76 |

Prefill speed improved 2.29× in the greedy pair and 2.12× at the median of the three paired sampled ratios. Every 256-token output hash matched. The sampled packed observations spanned 343.82–382.40 tok/s and the incumbent spanned 155.42–186.24 tok/s. During the two monitored repeat pairs the worker accounted for at least 99.89% of observed GPU activity, with fair thermal state and zero measured pageouts or swapouts. The earlier packed build reached 473.05 tok/s greedy and 406.67 tok/s sampled in its one pair per mode. Both builds used the same packed precision policy. The 500 tok/s target corresponds to 8.192 s on a cold 4K request and remains open. [Packed graph and numerical controls](../model-experiments/2026-11-21-inkling-mini-packed-prefill.md)

The incumbent and packed worker executables have SHA-256 `aae3755e67daf4142eac0bed33acaed8fe6c393cf824e85d41b9374f53f7b00a` and `6bbafc99e1d880fb90ccbef9ea461e22f1204cdc689a3a4c205be176894a700d`. The primary and sampled repeat aggregates have SHA-256 `23dfc768e43b7765224fce6de08aa446a6ed597fde960e76238277928b979e83` and `13f3674757927dca5f60d59e60bf03c49268dc14c06b401c72efd5e521dd2839`. The private host receipt is `d5e479ac908cb01911d8a2e171d0ee4101de1a89078df98b8c0940febffdaa36`.
