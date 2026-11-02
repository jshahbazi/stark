# GLM-5.3 Batched Prefill Experiment (2026-11-02)

**Status: qualification pending longer-context runs.** The candidate improves the measured short-input prefill rate in a screened 512-token-context comparison. It has not been promoted to a default or final wrapper, and the result does not establish performance at longer contexts.

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

## Numerical and validation evidence

The narrow scorer comparison passed 128 next-token log-probability points over eight contexts. Maximum absolute delta was `1.14266953e-5`; RMS delta was `4.00372032e-6`. The scorer-side MMA probe recorded 3,400 encoded dispatches across eight successful requests. This is a software dispatch count, not a GPU utilization measurement.

The candidate uses F32 activations and accumulation with reduction reassociation. Private validation recorded three tests across three valid configurations, all nine passing. An older HC4 quantized fixture failed before model admission because its GGUF contains overlapping tensor spans; it is recorded separately and admission checks remain enforced.

Validation totals for the candidate: hybrid suite 195 passed / 73 ignored; score suite 25 passed; generator suite 96 passed / 2 ignored; both Clippy runs passed. These are targeted checks, not a whole-engine suite result.

## Limits and pending work

The first cold-prefill observations were 23.7466 seconds for the baseline and 21.8563 seconds for the candidate, but they were not a matched cold experiment and do not support a cold-start gain claim. The unscreened native MMA probe and phase trace are diagnostic only and are not mixed into the headline results. The embedded llama.cpp MLX-Serve 26.10.3 control from November 1 remains historical until refreshed by the main task.

An additional unfiltered comparison used 96 input tokens, eight output tokens, and five replays (the first replay is excluded from the reported warm medians). Exact replay output hashes match between v46 and v47. Its warm medians were:

| Unfiltered 96-token observation | v46 control | v47 candidate |
| --- | ---: | ---: |
| Prefill throughput | 34.69130 tok/s | 29.70173 tok/s |
| Decode throughput | 33.04145 tok/s | 25.71305 tok/s |
| Request wall time | 2.98112 s | 3.50743 s |

These observations are not performance-qualified and do not establish a regression caused by the candidate. The candidate run coincided with substantial unrelated load: other-process CPU averaged roughly 470–568%, with WindowServer and other-process GPU activity each around 8%. The control's other-process CPU was roughly 73–88% and external GPU activity was near zero. Both runs had paging. The large host-load difference makes this an uncontrolled comparison. The v46 and v47 source/build identities remain separate as documented above; these measurements do not include the separate chunk-size-8 experiment.

The 96-token harness is being updated to prospectively apply the same quiet-host screening used by the 512-context headline comparison. Longer-context qualification, including screened reruns, remains pending. An optional eight-token chunk experiment is separate work and is not part of the v47 candidate. The [rejected Q5 MMA study](../archive/experiments/glm53-q5-mma-20261102/README.md) records a prototype rejection; it did not admit or run a Q5 model route. The final wrapper and default remain unpromoted. No whole-model equivalence, answer-quality result, or general throughput floor is claimed.

## Source and aggregate records

Both builds start from GLM branch merge `13de5d4`, which incorporates main `81f8b264`. Its build passed 96 generator-library tests. The candidate has tracked diff SHA-256 `0f6105eea93f594c4181bdae7f7932554ebca491af9cdc3b64009f09fa9f327d` and hybrid source bundle SHA-256 `d082a80c8d935d405703300f9af0567c72531c79b233b0a1e0d32f32e6f4420c`. Executable builds are bound separately by the receipts below.

The [aggregate JSON](data/glm53-batched-prefill-2026-11-02.json) contains public-safe metrics and hashes only. Source-bound private receipts and logs are retained under `runtime/glm53-resume-20261102/`; they include the v46/v47 build receipts, screened controls, score parity receipt, native diagnostic probe and phase trace, and archived candidate patch. Prompts, responses, token IDs, logits, probability arrays, exception dumps, and raw logs are not published.

| Build | Generator SHA-256 | Scorer SHA-256 | Build receipt |
| --- | --- | --- | --- |
| Control v46 | `2254405fd2653b69519e974d72b2e21778a71a61522389b6ecb56aa2e953805a` | `52150ee9fabd300fc5b6bc70d13f7680b31158f3344055aac58c24ba6b8c4ea7` | `runtime/glm53-resume-20261102/build-receipt-v46.json` |
| Candidate v47 | `078a0cdd934722d904e372c0a64564a280639164d1daf2ef2a156f8f5f279ae4` | `88a4bfc30385bba9ee6f8f88c25d32ec4cc4025f027503dd606b8f575ea8c62f` | `runtime/glm53-resume-20261102/build-receipt-v47.json` |

The benchmark index links this report as a qualification-pending result. It does not replace or alter the November 1 reports.
