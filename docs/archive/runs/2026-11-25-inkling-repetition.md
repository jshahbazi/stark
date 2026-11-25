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

## 20261125-INKREP-02 · OpenCode workspace replay

The reporter identified OpenCode as the original client and specified two exact user turns. The replay used the OpenCode 2.4.1 engine bundled with the desktop app, launched in the named repository with project instruction discovery enabled. Both prompts used one fresh session backed by an in-memory database. The tool policy allowed workspace `glob`, `grep` and `read`. Repository contents, model text, tool arguments and tool results stayed out of the receipts.

An isolated local Stark API process served the installed affine 8-bit target and native executable identified above, with a 32,768-token context. It advertised streaming, tool calling and default DSpark depth 3. The replay did not change the existing desktop connection. OpenCode performed the file operations and sent their results back to the model. The observer counted tool calls, exact duplicate arguments, repeated 12-word or punctuation sequences in each output channel, finish reasons and OpenCode's reported token usage. Counts across model calls can exceed the context window because the client resubmits conversation history.

| Turn | Observed workspace tool calls | Terminal outcome |
| --- | --- | --- |
| Directory question | 16 searches, three globs, two reads. Twenty calls completed and one returned a tool error. | Produced an answer and stopped. The most frequent exact call occurred twice. |
| Review follow-up | Seven distinct reads completed. | OpenCode returned `provider.invalid-request` with a context-related message. There was no final answer. |

The repeated-span detector found no 12-word sequence occurring twice inside one reasoning or text part on either turn. The working tree was checked before and after. The narrow tool policy and unavailable historical runtime settings limit comparison with the reported session.

### Larger-context replay

A second fresh OpenCode session used the same two user messages, default model variant and affine 8-bit weights. The `build` agent could also request a limited set of read-only shell operations. The isolated API used the [request-sized context build](https://github.com/jshahbazi/stark/blob/b92055b983b3c5d6887abf63422d04849544af27/docs/model-experiments/2026-11-24-inkling-mini-request-context.md) with worker SHA-256 `594c68d74304fe29847eeb1ba6d4ee9d524167f46acfc9e30651116d03d54f60` and advertised 131,072 tokens. Context size, worker and tool policy changed together. Neither fresh session enabled the historical review skill.

The first turn called tools 13 times and gave a final answer identifying both application directories. The follow-up called tools 47 times across 43 assistant steps. Forty-one calls completed and six returned tool errors. The most frequent exact call occurred four times. OpenCode ended the follow-up after 623 seconds with `provider.rate-limit` and no final answer. On a request with 53,373 input tokens, the worker's last progress event counted 15,916 generated tokens. That event contains no text and cannot determine whether those tokens repeated. OpenCode's terminal error records the client classification. The harness did not interrupt either live replay.

Aggregate receipt SHA-256 values are `e4093ec8dc76417cd760901efffb7da39b9356ebf542011e6d25c4d8f186b559` for the 32K run and `1189a7503cac46a8636da0015dedf489694dc56b442b38e264239c6152ef5738` for the 128K run.

### Historical OpenCode evidence

Two existing OpenCode 2.4.1 sessions matched the directory question and review request in the named workspace. Both used the `build` agent, Stark provider and Inkling 2 Mini. In each session the directory turn finished after three successful shell calls. The review turn loaded a local review skill and reached a normal model stop. Neither turn issued identical tool calls repeatedly. Both sessions also contained one later user turn.

In both sessions the later turn was a short affirmative reply. In one, OpenCode recorded 60 completed shell calls with 15 distinct argument sets. The model requested the same shell command containing three read-only content searches **45 times consecutively**. All 45 executions returned the same result. Across assistant reasoning in that turn, a separate detector counted 75 recurring eight-word spans and four normalized sentences repeated three or more times. That turn emitted no visible answer and ended interrupted. In the other session the later turn ended interrupted after seven successful shell calls, all distinct. Neither stored original indicated a context error for the matched workflow. These records identify a tool-use failure on a later turn that the initial text-only baseline could not observe.

### 20261125-INKREP-03 · Next action after a recorded result

Two model-only requests tested whether a short reconstruction could reproduce the next identical call. Both used the 128K worker, greedy sampling, top-p 1, a 1,536-token output limit and automatic tool choice. The model saw the two user turns, visible replies, later acknowledgment, a preceding different tool call and its result, and either one or five copies of the repeated call and its actual result. All tools were inert. The reconstruction used a minimal shell schema and omitted the original system instructions, intervening steps and reasoning history.

| Copies of the completed search in history | Input tokens | Output tokens | Next action |
| ---: | ---: | ---: | --- |
| 1 | 2,682 | 94 | Requested a different shell command. |
| 5 | 8,210 | 694 | Returned text and stopped. |

The historical loop is confirmed. Neither abbreviated history regenerated its next identical call. A weight-edit validation needs a baseline that reproduces the call and controls on which repeated searches are appropriate. No candidate weight edit was tested. A [separate synthetic cache experiment](https://github.com/jshahbazi/stark/blob/c5113a38c0faf415cec16d78e701e28e94383ef9/docs/model-experiments/2026-11-24-inkling-mini-tool-loop-cache.md) elicited a repeated call both with and without prefix reuse on its own fixture. Frozen-history aggregate receipt SHA-256 `adea741fbaa4e27b324c9d6aa455e2d999fc449a4d95f54bac4679e70d9e6b20`.

The [follow-up weight-edit trial](2026-11-25-inkling-weight-edit.md) tested changes to the vocabulary projection on separate copies of the installed checkpoint.
