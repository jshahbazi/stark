# 20261125-INKREP-01 · Inkling 2 Mini repeated reasoning

## Question

Can a two-turn code-review exchange elicit repeated reasoning on the original Inkling 2 Mini weights? A reproducible failure would give us an input for investigating a targeted weight edit. Thinking Machines reports that Inkling sometimes revisits the same checks while debugging. A separate experiment on Qwen3.5-0.8B reduced repeated generation by editing feed-forward weights in two layers. Inkling routes tokens among experts and shares the middle layers across three passes, so it needs its own evidence for where to intervene. [Inkling model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) · [Weight-space programming, experiment 8](https://rikiyat.github.io/blog/weight-space-programming/)

The [earlier Stark budget study](2026-11-24-inkling-thinking-budget.md) tested a decoding control on an 8-bit dense artifact. Its 512-token review exhausted the output allowance before producing a visible answer. A length limit alone gives no measure of how often the model repeated itself.

## Baseline

The original target and template come from `thinkingmachines/Inkling-2-Mini` revision `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`. Its target SHA-256 is `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1` and its template SHA-256 is `53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc`. The host is an Apple M5 Max with 128 GiB of unified memory running macOS 26.7.2.

The original-weight Stark Metal worker has SHA-256 `54b9b23e4be18dbdeffdeee8a46a5e699aa303d4e80d0570cd8057fe839d4f3f`. The text requests used the checkpoint's chat template at medium reasoning effort, temperature 1.0, top-p 1.0, a 4,352-token context, a 1,024-token output limit and zero draft tokens. The initial template setup failed before generation. The completed run used Stark's template preparation helper.

## Original-weight replay

The two user turns were supplied by the reporter. The first asked about a local project's directory structure and the second asked for a code review. Two follow-ups received a reconstructed first answer based on the tracked directory layout. A third received the model's own first answer in the conversation history. The model had no file tools in these requests. The original assistant reply, tool traffic and sampling configuration were unavailable.

| History for the follow-up | Seed | Generated tokens | Reasoning tokens after parsing | Visible tokens after parsing | Finish |
| --- | ---: | ---: | ---: | ---: | --- |
| Directory-grounded first answer | 311 | 342 | 66 | 270 | EOS |
| Directory-grounded first answer | 312 | 291 | 22 | 263 | EOS |
| Model-generated first answer | 314 | 490 | 46 | 438 | EOS |

All three follow-ups emitted a visible answer. The model-generated first turn also reached EOS in 399 tokens at seed 313. Four synthetic arithmetic and debugging controls reached EOS and passed a required-answer substring check at seeds 400–403. The code-review replies were not scored for accuracy. The initial screen measured repeated spans only in the visible answer. Its aggregate receipt has SHA-256 `c97b3806af816b281d298cdc7c8f7d16b374fdc91ad1cb64212bd63dc2f81c59`.

The same seeds and histories were replayed to check the reasoning and visible answer separately. Each part was tokenized again after parsing. The generated total includes message framing. The screen counted eight-token sequences that appeared at least three times without overlapping and contained at least four distinct tokens. It also counted normalized sentences of ten or more words that appeared at least three times. Both counts were zero in both parts for each follow-up and for the regenerated first turn. Paraphrased rechecking can escape both detectors. Replay aggregate SHA-256 `3f2c106a5864df35ef24bc301ea3fdfcc6ff298647a660efc443790937fee32d`.

## Installed artifact

A separate process loaded the installed affine 8-bit dense artifact on the bundled Stark worker. The artifact retains the published low-precision expert tensors. Its target SHA-256 is `c9e14848ad017467dc0deb91c6aa67022e181dbcdfe02a1b04024434bf9acddd` and the worker SHA-256 is `ce5a983645f5d87ee1127b2c6ce4864919c37dc8895984e03dc4db76b94dadfa`. This executable predates the explicit thinking-budget control. The first harness attempt failed before returning a result and is excluded.

With the directory-grounded history, greedy token selection, seed 0, medium effort, zero drafts, a 4,352-token context and a 1,024-token output limit, the model reached EOS at 742 tokens. Both channels contained text. Neither reached the three-occurrence threshold for repeated spans or sentences. Sampling, artifact and executable differed from the original-weight run, so these counts cannot isolate an effect of quantization.

The installed worker also ran the follow-up at minimum reasoning effort, greedy seed 0, zero drafts, an 8,192-token context and a 4,096-token output limit. It reached EOS at 468 tokens. The parsed reasoning contained 784 characters and the answer contained 1,198. Neither channel had an eight-token sequence or normalized sentence of ten or more words repeated three times under the detector. Both installed runs omitted a thinking budget. The installed aggregate receipt SHA-256 is `bad1e6d85f7edab7c4e5ae743cd897999259fdaf3b7a54f6c92c23ff4504e96c`. The manually transcribed target digest in its medium-effort row omits two characters. Both runs used the installed artifact directory identified above.

## Decision

`20261125-INKREP-01` completed as a diagnostic on November 25, 2026. The reported exchange produced visible answers and stopped on all five follow-ups. The test did not reproduce sustained repetitive generation. The number of histories and seeds is small, and the exact-span checks can miss repeated ideas expressed in different words. The result applies to text-only conversations without repository tools or file contents.

Both full target shards were hashed after the runs and matched the respective digests above. A weight-edit trial is deferred until a failure can be replayed. A candidate would need to reduce repetition and preserve answer correctness on separate debugging cases at matched sampling settings. No weights, runtime defaults or installed files were changed. Performance and peak memory were not measured. [Model experiment ledger](../../model-experiments/README.md)
