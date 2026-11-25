# Inkling 2 Mini weight edits for repeated OpenCode searches

## Incident and acceptance criteria

In the recorded OpenCode session, the directory question and the first code-review request both finished. After the next user acknowledgment, Inkling issued one successful search command 45 times consecutively and received the same result each time. The model produced no visible answer before interruption. The earlier report describes both matching sessions and the differences between the original and reconstructed histories. [OpenCode incident and baseline replays](2026-11-25-inkling-repetition.md)

The desired behavior is to use the information in the completed search and move the task forward. An edit earns credit on a replay only if the parent repeats a completed search on the same input and the candidate takes an appropriate next action. A different rendering of the same command is still a repeat. For the synthetic fixture, the result contains the answer, so success requires a correct visible answer. A necessary fresh search must remain available on controls. One-step historical replays test the next action. They cannot establish recovery over a complete OpenCode conversation.

## Checkpoints and worker

Both the parent and the experimental copies use the native affine 8-bit dense checkpoint from `thinkingmachines/Inkling-2-Mini` at revision `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`. The parent target SHA-256 is `c9e14848ad017467dc0deb91c6aa67022e181dbcdfe02a1b04024434bf9acddd`. The DSpark shard SHA-256 is `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`. The target shard occupies 18,052,013,648 bytes. Its output projection stores 201,024 rows of 2,304 weights. Each group of 64 weights uses unsigned 8-bit codes and BF16 scale and bias coefficients. The Metal kernel rounds the reconstructed weights to BF16.

A private generation worker built from Stark `c5113a38c0faf415cec16d78e701e28e94383ef9` accepts an explicitly pinned target digest for this experiment. It verifies the target and draft shard bytes at admission. The released worker pins its supported artifacts. The private build used Rust 1.98.1 in release mode. Its SHA-256 is `8f84e78381f1c8b6985f82bf1beb9d548c0405ec97d6235f2e3efcf8fae1241c`. A relocated Metal build cache initially failed to compile. Rebuilding its generated output resolved that setup failure.

## 20261125-INKREP-04 · Scale one output row

The tokenizer encodes the tool name `shell` as one token, ID 4198, between the assistant message marker and the tool-arguments marker. Candidate A multiplies the 36 scale and 36 bias coefficients for its output row by 0.94, rounding each to BF16. It changes 73 stored bytes across two 72-byte spans. Comparing the full shards confirmed that all other bytes, including the entire code plane, matched. The candidate target SHA-256 is `e20508d2fb961aeffc0346b5bdbc00ac956e7f093e9911c66ba95276a9d311b1`.

The edit is active whenever the model predicts a token. It can also affect appropriate shell calls and occurrences of the word in text. The article that motivated this experiment reported an edit to internal feed-forward layers on Qwen3.5-0.8B. This trial changes Inkling's vocabulary projection. [Weight-space programming, experiment 8](https://rikiyat.github.io/blog/weight-space-programming/)

## 20261125-INKREP-05 · Fit a change to the same output row using the incident

Candidate B uses the parent's normalized representation at the instant the model must select a tool name or a response channel. Two examples come from just before the second and fourth identical searches. Two protected examples come from distinct successful shell commands before the streak. All four prefixes end at the assistant marker immediately before the recorded tool name. The checkpoint's template renders the actual recorded reasoning for that step. Recorded tool results remain inert. A neutral coding-assistant instruction and approximated tool descriptions stand in for the original client instructions. The historical review skill body is omitted. The prefixes contain 27,194, 30,156, 336 and 1,921 tokens respectively. Neither later loop positions nor independent controls enter the fit.

For a final hidden vector `h`, changing only the shell row by `d` changes its raw score by their dot product. A least-squares solve finds the minimum-length change that would put the shell score at least one raw logit unit below the text channel on both repeat examples, lower it by at least three units there, and leave it fixed on both protected examples. The prescribed maximum change is 35% of the original row's Euclidean norm. The row is then requantized in groups of 64 and read back using the native effective-weight rule. The goal is a contextual change that keeps legitimate tool choices available. Crossing this score threshold is a measurement of the fit. Free generation tests which action the model actually chooses.

The private feature extractor reads 2,304 normalized BF16 values per prefix before the vocabulary projection with MTP off, 2,048-row source-text prefill and fresh state per input. A short reset check returned identical features on an identical prefix and distinct features on a changed prefix. Its binary SHA-256 is `ba5a5f74ac275d3e9434041696cbbc974b8f096b315e504d84022e7ec2078e9b`. The fitter and extractor exchange features in memory. Only aggregate results and hashes are recorded.

The 35% limit was active. Candidate B changed 2,387 stored bytes. After quantization, it still scored the shell token above the text channel on both discovery examples. The maximum absolute change on the two protected examples was 0.041 raw logit units. The candidate target SHA-256 is `af02a96f287fed6e6ddce398ae58491115b69b36c9eff5252d216bef53377593`.

## 20261125-INKREP-06 · Test the full fitted change

Before scoring held-out behavior, candidate C used the same four examples and the full minimum-length solution. It changed 2,398 stored bytes. Its effective change measured 45.56% of the norm of this one vocabulary row. After quantization, both discovery examples scored the text channel above the shell token. The maximum absolute shift on the two protected examples was 0.043 raw logit units. The full shard comparison for each candidate confirmed that only the selected row's code, scale and bias spans could differ. The candidate C target SHA-256 is `561bea5910c62367839cef72b7db27b3352b9a1dd07470815c2a1e59b2a88430`.

## Replay design

The primary historical comparisons use the same private generation binary, greedy decoding, seed 0, MTP off, a 131,072-token maximum context and a 1,536-token output allowance. At the early position the Stark converter selected reasoning effort `none` for the default settings used by the harness. At position 44 and during fitting the renderer selected `high`. Each comparison holds effort and the rendered input constant across checkpoints. The saved session does not establish the original effort or sampler settings. The free replay starts just after the preceding tool result and lets the model generate the reasoning for its next action. A second test supplies the recorded reasoning up to the tool-name decision. All checks run locally on an M5 Max with 128 GiB of unified memory and AC power.

Five independent, short controls check ordinary generation and appropriate tool use. They require a specific word, an arithmetic answer, an initial file read, a correct answer using a supplied file result, and repetition of the original file read after the user explicitly changes the file. The last two cases distinguish a redundant read from one needed to get up-to-date evidence.

All four checkpoints passed all five controls. Each used the same number of sampled tokens per control (65, 47, 63, 61 and 56, in that order). The first two controls used no reasoning effort, a 2,048-token context and a 256-token output limit. The three file controls used high effort, a 4,096-token context and a 768-token output limit. Two early control-harness setups failed at readiness or parsing before a valid comparison. Final runs explicitly enabled native vision staging on each arm and checked the admitted target digest. The generation requests used text only.

## Decisions earlier in the recorded incident

Starting just after the first completed search, the parent independently regenerated the exact next shell command and arguments. Candidate A repeated it too. The input had 27,093 tokens. C emitted 146 visible characters and requested a different shell action. A fresh run matched its complete decoded response hash, `bf205b54df3e169c98493dbec98ae0c804296cabd0d3c0e15992d100c55c4200`. The command contained none of the three original search components recognized by the parser and did not match any previously completed command. The inert replay ends at that request. This is the same incident position used for fitting, with freely generated reasoning and different effort.

The same early input was replayed with DSpark enabled at depth three on the parent and C. Each worker reported the requested depth. Each produced the same complete decoded response hash as its own MTP-off run. This check covers one early action on two checkpoints.

At the held-out 24th position, the 59,674-token free replay also produced the exact historical shell command and arguments on the parent. C emitted 86 visible characters and requested a different shell command. Correctness of the new command at this position was not scored. The three free positions in this report are correlated observations from one incident.

## Held-out decision before the 44th search

The free replay had 89,296 input tokens. Supplying the recorded reasoning added 100 tokens. The rendered input hash matched across all four checkpoints within each mode, and all eight requests reported zero prefix tokens reused.

| Checkpoint | Free generation after the 43rd result | With the recorded reasoning supplied |
| --- | --- | --- |
| Parent | Repeated the original command. | Repeated the original command. |
| A, scaled output row | Identical decoded output to the parent. | Identical decoded output to the parent. |
| B, capped directional change | Identical decoded output to the parent. | Identical decoded output to the parent. |
| C, full directional change | Emitted 99 visible characters and requested a different shell command. | Emitted 99 visible characters and requested a different shell command. |

The native EOS in a tool-call response returns control to OpenCode, which can then execute the tool and call the model again. Each of these eight responses requested a tool. A different command requires a separate check for equivalent work.

A fresh replay of C reproduced its free-generation response hash, `4ad841a7419abe46faf108733a14852432c04141dab185883e7db9f64b3831fb`. Both its new command and the original contained three search components. Two components matched exactly and also matched when comparing their query and path while ignoring flag order. The third had a different query signature. The complete command had no matching completed request earlier in the recorded history. Its result and any subsequent assistant action therefore remain unknown. The changed command still repeats two searches that already succeeded.

The position-44 aggregate receipt SHA-256 values are `8953b9f03e28294665016d09db7c5ad86e95b8354891ab6e9cb52274c78c4904` (parent), `95a3e3a897fb8b43269ed6be28db8095dd7a58f07fc65e3b10f7a1c6f53ebd6e` (A), `501d9734bac0b762fde5e55998b70a1b3688485a0dfa3918dc95b88afa0e2047` (B) and `289e78f5e43cf4590fba717a860ac12de78dbe777878e4df814884d27f71a98e` (C).

## Independent case with a known answer

An independent fixture asks which gate is closed in a fictional ferry archive. The parent first requested the specified file. The follow-up supplies five successful copies of that same call and the same result, which explicitly identifies the closed gate. Each arm receives the same 58,398-token input, including the parent's generated reasoning and tool arguments. The target test uses greedy seed 0, MTP off, a 131,072-token context, a 1,024-token output limit and a cold cache. This is a new run of the [synthetic tool-loop fixture](https://github.com/jshahbazi/stark/blob/c5113a38c0faf415cec16d78e701e28e94383ef9/docs/model-experiments/2026-11-24-inkling-mini-tool-loop-cache.md).

All four checkpoints requested the same file again and supplied no visible answer. The three edits fail the independent test. The aggregate receipt SHA-256 is `97eafa6f22e7d84ad2bee6e54782a6ef7b592eb4947f3a6ff67e460e3c8beae0`. Managed-worker setup, incorrect tool-envelope handling, and a missing candidate digest pin each invalidated an earlier attempt. The scored run verified each loaded target digest and used the Stark tool parser and the same native worker for all arms.

## Decision

All three experiments completed on November 25, 2026. Candidate C changed the exact next command at all three sampled points in one recorded incident, including both held-out positions. Every position still led to a shell request, and the independent answerable task still elicited a redundant call. The five short controls retained their expected behavior. These results support a local change to action selection and fail the acceptance criterion for a general loop fix.

The edited checkpoints remain local research artifacts. The installed model and OpenCode configuration use the original checkpoint. End-to-end task completion, broader reasoning quality, image behavior, throughput and peak memory were not qualified for these edits. [Model experiment ledger](../../model-experiments/README.md)
