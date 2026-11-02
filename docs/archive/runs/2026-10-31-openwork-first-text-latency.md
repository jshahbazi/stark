# OpenWork time to first text

Date: 2026-10-31. Follow-up to the report that OpenWork waits before showing
any text. This measures the normal installed client/backend path after the
[text-streaming fix](2026-10-31-window-and-agent-streaming-followup.md).

## Environment

- Official OpenWork 0.18.63, bundled OpenCode 2.0.26, existing normal local
  workspace and managed Stark connection.
- Installed Stark 0.11.14, source
  `69c9cffc491e1292d4118b9ee6b4ae20232a6bcb`; host SHA-256
  `3ff648d6f063a4d9c48aa3100cde45db76b2c680019da6edd9292b31198df88a`.
- Model `mlx-community/Qwen3-4B-4bit`, local ID `qwen3-4b-4bit`, MLX affine
  4-bit/group-64, source revision
  `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`. Artifact identities are in the
  [managed connection record](2026-10-31-openwork-managed-connection.md).

## Method

A bounded read-only metadata query found a recent normal Stark-provider turn
with 18,648 input tokens, 9 output tokens and 5.805 seconds between persisted
assistant creation/completion. It had no error or tool part. Its text part had
no first-text timestamp, so saved history alone does not establish latency to
first text.

The live diagnostic created a disposable session through OpenWork's native
OpenCode V2 API in the existing workspace, bound it to Stark's running model,
subscribed to the native event stream before submitting a short synthetic
prompt, and timestamped the first nonempty text delta with a monotonic clock.
It used the stock client engine, its instruction/tool context and its managed
Stark provider. It did not construct a direct Chat Completions request or
provide Stark credentials. This captures the same native text event consumed
by OpenWork's frontend adapter; frontend paint timing was not measured. A second
independent disposable session requested a longer response to check incremental
delivery while native generation continued.

## Results

| Measurement | Short reply | Longer reply |
| --- | ---: | ---: |
| Prompt POST acknowledgement | 0.011 s | 0.008 s |
| Submit to native execution-start event | 0.034 s | 0.025 s |
| Submit to first nonempty text event | 5.321 s | 5.399 s |
| Persisted assistant duration | 5.288 s | 10.683 s |
| Completion observed by polling | 5.402 s | 10.784 s |
| Input tokens | 18,544 | 18,557 |
| Output tokens / text deltas | 1 / 1 | 408 / 51 |
| Tool events / errors | 0 / 0 | 0 / 0 |

Immediately afterward, Stark's public counters showed input throughput
3,563.9 tokens/s and output throughput 71.1 tokens/s. Dividing this request's
18,544 input tokens by the rounded input rate gives an approximate **5.203 s
native first-token stage**. This stage includes native setup, prefill and the
first sampled token; it is not pure GPU compute time. Approximately **0.118 s**
of the measured first-text interval falls outside that stage. This difference
includes other generation, serialization, client and transport work and must
not be labeled solely as connection overhead. The longer reply's public input
rate was 3,555.7 tokens/s, yielding an approximate 5.219 s native first-token
stage and 0.180 s outside it.

The remaining several-second wait is dominated by Stark's native first-token
stage for the client's large input. The longer reply delivered its first native
text event over five seconds before generation finished, then delivered 51 text
deltas. The short reply alone does not establish incremental delivery because
its one output token arrived near completion. Live streaming does not remove
the instruction/tool context or accelerate native long-prompt processing. These
two observations do not establish general throughput or agent competence, and
did not use exclusive-host performance controls or tool execution.

## Privacy, cleanup and window behavior

Only timings, counts, fixed outcomes and public version/model identities were
retained. History/event bodies stayed in RAM; no prompts, responses, token
vectors, titles, workspace paths or raw error dumps were saved in the reports.
Synthetic request definitions remain in the ignored diagnostic source.

OpenWork's existing local client token was read into RAM and sent only to the
fixed loopback OpenWork server, with proxies and redirects unused. The
diagnostic never accessed Stark's master key or supplied a credential to
OpenWork's provider. The ordinary signed-sidecar broker remained responsible
for Stark authorization. No key was copied, displayed, placed in arguments or
environment variables, or saved by the diagnostic. Evidence files were created
exclusively with owner-only permissions and symlink following disabled.

Each diagnostic deleted only its own synthetic session; existing sessions and
connection settings were left unchanged. After both actual connected-client
requests, accessibility checks found both Playground and Connection Details
closed. The original external trigger for the reported spontaneous windows
remains uncaptured.

Short-reply diagnostic SHA-256:
`a73f8ddfb05e8273a7e3d956f4f87b5d5d0d871cc8b10caa68d100715ff00287`.

Longer-reply diagnostic SHA-256:
`cdcbd502e1c24e303cf171a19966150857578840fbc9a6183bfd6e85ba213d2b`.
