# GLM-5.3 Batched Prefill Experiment (2026-11-02)

**Status: bounded experimental recipe qualified.** The screened T4 and T8 comparisons support selecting v48 T8 for the opt-in `scripts/with_glm53_optimized.sh` recipe (`STARK_GLM53_PREFILL_MMA=1`, `STARK_GLM53_PREFILL_CHUNK=8`). Engine-wide defaults remain off. Qualification is limited to the measured workloads and numerical checks below; it does not establish general engine performance or a universal default.

## Result

The experiment adds an opt-in `STARK_GLM53_PREFILL_MMA=1` route with chunks of four known prompt tokens. It schedules layers across those tokens while retaining F32 activations and accumulation. MMA admission is restricted to Apple M5 Max and the measured shapes; the loaded model uses it for 68 Q6_K attention key/value matrices of shape 8192×4096. The kernel derives from [llama.cpp revision dde879bc4f954b809892fd0d660515f42536394a](https://github.com/ggml-org/llama.cpp/blob/dde879bc4f954b809892fd0d660515f42536394a/ggml/src/ggml-metal/kernels/mul_mv_mma.metal), with its MIT license retained in `LICENSE.ggml-hybrid`. The candidate source was staged and uncommitted in the bound build; its archived patch is `runtime/glm53-resume-20261102/candidate-v47.patch`.

| Screened metric | Merged control v46 | MMA candidate v47 | Change |
| --- | ---: | ---: | ---: |
| Median prefill throughput | 30.935 tok/s | 35.332 tok/s | +14.2% |
| Median useful decode throughput | 34.919 tok/s | 34.993 tok/s | +0.2% |
| Median measured request wall time | 1.356 s | 1.296 s | -4.4% |
| Accepted measured requests | 4 | 4 | — |

Decode movement is within noise and is not claimed as an improvement. The short request wall-time reduction reflects prefill as well as decode and applies only to this workload.

Both configurations used the same GLM-5.3-Flash UD-Q2_K_XL GGUF files, tokenizer, prompt template, 14-token input, and 32-token greedy output. Input, output, tokenizer, and prompt-template hashes match across the screened receipts. Weight files were verified using their prior SHA-256 values with size, modification-time, and inode guards before and after execution. Both used the same Mac17,7 host and 512-token context.

The control used four warmups followed by four accepted measured requests. The candidate had four warmups and four accepted requests; three additional attempts were excluded by host screening. The control had eight excluded attempts. All accepted requests had zero pageouts and swapouts. The screened receipts preserve excluded attempts and their paging deltas. Rates and wall times above are medians of accepted measured rows; useful decode rate counts generated tokens after the first prefill-produced token.

## Separate T8 chunk experiment (v48)

The v48 candidate is a distinct opt-in build using eight-token prefill chunks. Its Q5+Q6 MMA route remained default-off. It preserves the same F32 activation and accumulation policy as the v47 experiment. Do not interpret these measurements as changing the v47 T4 results.

On the screened 512-context, 14-input, 32-output short workload, T8 measured 35.339 tok/s prefill and 34.969 tok/s useful decode, close to v47 T4's 35.332 and 34.993 tok/s. The native diagnostic probe measured 40.0 tok/s prefill and 34.877 tok/s decode, but it is unscreened and diagnostic only, so it is not used for comparison claims. Its request scratch high-water was 337,641,472 bytes (322 MiB); phase counters report the same request-wide maximum for prefill and decode. The decode phase's `request_max_bytes_at_end` is request-wide, not a phase-local allocation.

The broader scorer comparison passed 192 points over 12 contexts (22–512 input tokens): maximum absolute log-probability delta `1.3879155950746735e-5`, RMS delta `4.18813704e-6`. A separate eight-context check passed 128 points. These are next-token normalized log-probability comparisons and do not establish whole-model output equivalence or answer quality.

For the screened 96-input-token, eight-output-token workload, the v46c control measured 34.895 tok/s prefill and 2.968 s request wall time; v47 T4 measured 38.881 tok/s and 2.684 s; v48 T8 measured 40.692 tok/s and 2.577 s. Relative to the same v46c control, T4 improved prefill throughput by 11.4% and reduced request wall time by 9.5%; T8 improved prefill by 16.6% and reduced request wall time by 13.2%. Relative to T4, T8 was 4.7% faster on prefill and reduced request wall time by 4.0%. The T4 receipt (`runtime/glm53-resume-20261102/context96-screened-mma-v47.json`, SHA-256 `f93d90dd7a0301f7a80b26985531729b413a3c79fe5388be2d26ed2b898baf0c`) records one warmup, four accepted requests, two exclusions, seven total attempts, and exact output hashes across attempts. All three receipts use the same input hash `da577625b599110274643266397b057f833510568c471efca58f36eb100734cd` and all report output hash `8669bd40bd02bcc35680c255fc660be01a22a86ff01201463acfa7c457d64cad`. These screened results support the bounded opt-in T8 recipe; they do not establish general performance.

## Refreshed MLX-Serve comparison

A fresh MLX-Serve 26.10.3 embedded llama.cpp control completed screened measurement. It used the same 512-token context, 14-token input, 32-token output, and greedy sampling. The MLX input-token hash equals the native input hash, the decoded response-text hash equals the native output-text hash, and the four GGUF payloads have matching prior SHA-256 values and byte lengths. The MLX manifest (`runtime/glm53-optimization-20261101/mlx-fresh-weights-manifest.json`, SHA-256 `f8b0afb2d832a79506fa49ad9e32f8e6d5af523911e8e05d4ac552f47e6030ba`) has a different wrapper format from native `fresh-weight-verification.json`; matching payload entries establish checkpoint identity. MLX settings were version 26.10.3, engine `llama`, KV quantization off, automatic KV attention mode, decode attention quantization off, no prefill chunking or drafter, one concurrent request, and prefix cache disabled. The executable SHA-256 is recorded in the aggregate JSON.

| Screened metric | MLX-Serve 26.10.3 | Native v47 T4 |
| --- | ---: | ---: |
| Median prefill throughput | 88.523 tok/s | 35.332 tok/s |
| Median useful decode throughput | 34.062 tok/s | 34.993 tok/s |
| Median all-sampled decode throughput | 35.161 tok/s | 36.121 tok/s |
| Median prefill time | 0.1582 s | 0.3962 s |
| Median decode time | 0.9102 s | 0.8859 s |
| Median request wall time | 1.0694 s | 1.2955 s |
| Accepted / excluded measured attempts | 4 / 1 | 4 / 3 |

This comparison uses the embedded llama.cpp backend in MLX-Serve, not native MLX inference. Native prefill remains substantially slower, and native end-to-end request time remains longer for this short prompt. Native useful decode is slightly higher in this sample. The earlier MLX refresh attempt failed preflight with a wrong manifest and produced no GPU measurement; it is not included. No output-token-ID hash comparison is available from the MLX receipt, so matching decoded text does not establish token-ID parity.

## Numerical and validation evidence

The narrow scorer comparison passed 128 next-token log-probability points over eight contexts. Maximum absolute delta was `1.14266953e-5`; RMS delta was `4.00372032e-6`. The scorer-side MMA probe recorded 3,400 encoded dispatches across eight successful requests. This is a software dispatch count, not a GPU utilization measurement.

The candidate uses F32 activations and accumulation with reduction reassociation. Private validation recorded three tests across three valid configurations, all nine passing. An older HC4 quantized fixture failed before model admission because its GGUF contains overlapping tensor spans; it is recorded separately and admission checks remain enforced.

Validation totals for the candidate: hybrid suite 195 passed / 73 ignored; score suite 25 passed; generator suite 96 passed / 2 ignored; both Clippy runs passed. These are targeted checks, not a whole-engine suite result.

### v48 validation totals

Keep these v48 results separate from the v47 totals above. The retained v48 hybrid suite log reports 197 passed and 74 ignored; the v48 scorer unit log reports 25 passed. Both v48 worker Clippy logs completed successfully. The persistent pure-tool receipt reports 102 benchmark, compare, context, and host-metrics tests passed; formatting and policy checks passed, and the opt-in wrapper shell check passed. The v48 hybrid suite also includes a passing MMA runtime shape/source/F64/failure-gates test. Focused chunk8 logs in the separate `glm53-prefill-mma` worktree record five kernel/source/F64/failure/scratch gates passed, one policy gate passed, and 12 fixture invocations passed across three configurations (four tests each). Those logs were generated from an earlier base commit; the relevant `glm_prefill_batch.rs`, `glm_prefill_batch_tests.rs`, `weight_metal_kquant_mma_tests.rs`, and `weight_metal_kquant_mma_components.rs` files have identical SHA-256 values in that worktree and the v48 worktree. This supports those focused source-level counts without treating them as a whole-v48 build suite. These remain targeted checks, not a whole-engine suite result.

## Limits

The first cold-prefill observations were 23.7466 seconds for the baseline and 21.8563 seconds for the candidate, but they were not a matched cold experiment and do not support a cold-start gain claim. The unscreened native MMA probe and phase trace are diagnostic only and are not mixed into the headline results. The November 1 MLX-Serve control remains historical; the comparison above uses the November 2 refresh of version 26.10.3.

An additional unfiltered comparison used 96 input tokens, eight output tokens, and five replays (the first replay is excluded from the reported warm medians). Exact replay output hashes match between v46 and v47. Its warm medians were:

| Unfiltered 96-token observation | v46 control | v47 candidate |
| --- | ---: | ---: |
| Prefill throughput | 34.69130 tok/s | 29.70173 tok/s |
| Decode throughput | 33.04145 tok/s | 25.71305 tok/s |
| Request wall time | 2.98112 s | 3.50743 s |

These observations are not performance-qualified and do not establish a regression caused by the candidate. The candidate run coincided with substantial unrelated load: other-process CPU averaged roughly 470–568%, with WindowServer and other-process GPU activity each around 8%. The control's other-process CPU was roughly 73–88% and external GPU activity was near zero. Both runs had paging. The large host-load difference makes this an uncontrolled comparison. The v46 and v47 source/build identities remain separate as documented above; these measurements do not include the separate chunk-size-8 experiment.

The screened v46c control, v47 T4 candidate, and v48 T8 candidate now provide a qualified 96-token comparison. The opt-in wrapper recipe selects v48 T8; engine-wide defaults remain off. The [rejected Q5 MMA study](../archive/experiments/glm53-q5-mma-20261102/README.md) remains a separate prototype history; v48's Q5+Q6 route is measured only within the bounded experimental recipe. No whole-model equivalence, answer-quality result, or general throughput floor is claimed.

[MTP feasibility and head-cost profiling](../archive/experiments/glm53-mtp-feasibility-20261102/README.md) are recorded separately. Those test-only component probes establish no speculative throughput improvement.

## Source and aggregate records

The v46 and v47 builds start from GLM branch merge `13de5d4`, which incorporates main `81f8b264`. Its build passed 96 generator-library tests. The v47 candidate has tracked diff SHA-256 `0f6105eea93f594c4181bdae7f7932554ebca491af9cdc3b64009f09fa9f327d` and hybrid source bundle SHA-256 `d082a80c8d935d405703300f9af0567c72531c79b233b0a1e0d32f32e6f4420c`. Executable builds are bound separately by the receipts below.

The [aggregate JSON](data/glm53-batched-prefill-2026-11-02.json) contains public-safe metrics and hashes only. Source-bound private receipts and logs are retained under `runtime/glm53-resume-20261102/`; they include the v46/v47/v48 build receipts, screened controls, scorer parity receipts, native diagnostic probe and phase trace, and archived candidate patches. Prompts, responses, token IDs, logits, probability arrays, exception dumps, and raw logs are not published.

| Build | Generator SHA-256 | Scorer SHA-256 | Build receipt |
| --- | --- | --- | --- |
| Control v46 | `2254405fd2653b69519e974d72b2e21778a71a61522389b6ecb56aa2e953805a` | `52150ee9fabd300fc5b6bc70d13f7680b31158f3344055aac58c24ba6b8c4ea7` | `runtime/glm53-resume-20261102/build-receipt-v46.json` |
| Candidate v47 | `078a0cdd934722d904e372c0a64564a280639164d1daf2ef2a156f8f5f279ae4` | `88a4bfc30385bba9ee6f8f88c25d32ec4cc4025f027503dd606b8f575ea8c62f` | `runtime/glm53-resume-20261102/build-receipt-v47.json` |
| Candidate v48 T8 | `41fd421767f14cf9fa5a6dd441eec73b5a2c0e4ec1eeba10e11951b6cc11cedf` | `49045d592af366a9af405cc3eddead7eb8596a107136954bcc1e5a55148e61fd` | `runtime/glm53-resume-20261102/build-receipt-v48.json` |

The v48 source binding is HEAD `379e5c79d1cfae249e947740732b783b1a6fec82`, tracked diff SHA-256 `3e8a7f2ed4c63c159657209f732398bcce9706180c0b5736a09f5e6633c15c0c`, and hybrid source bundle SHA-256 `0ba85f6a4aa181a39651c2fb83491bdef4d1f2c399a2cf818f1e7df5313eccb2`; the build receipt records the same identity before and after. Its archived patch is `runtime/glm53-resume-20261102/candidate-v48.patch`.

The benchmark index links this report as a bounded experimental qualification. It does not replace or alter the November 1 reports.
