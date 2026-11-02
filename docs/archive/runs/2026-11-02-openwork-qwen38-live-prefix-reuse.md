# Live OpenWork session with Qwen3.8 prefix reuse

Date: 2026-11-02. Follow-up to the
[Qwen3.8 prefix reuse run](2026-11-02-hybrid-prefix-reuse.md), which used
synthetic sessions. This records one real OpenWork agent session driven by the
user against a signed development build with prefix reuse. Aggregates only;
no prompt, response, title, path or token vector was read or saved.

## Environment

- Host: Apple M5 Max, 40 GPU cores, 128 GiB, macOS 26.7.1.
- Client: OpenWork 0.18.63 with its bundled OpenCode v2 sidecar and the
  existing managed Stark connection. No Reconnect was needed.
- Stark: development build `20261102135442758115`.
  - Source `fb3e5e3a`: this branch merged with `main` at `81f8b264`, which
    adds the Lily XML tool-call parser.
  - Developer ID Team `S5BT2PRN5B`, hardened runtime and secure timestamp;
    not notarized.
  - Run from its staging folder; the installed `/Applications/Stark.app` was
    left unchanged.
  - Connector `a5534db0526e08c9b9b4df12825c9da7117cae86`.
  - The worker ran with `--prefix-cache`.
- Model: `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision
  `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, local ID
  `qwen38-flash-next-lily-q4-xl-0f94842e`.

| Artifact | SHA-256 |
| --- | --- |
| Signed generation worker | `528a99a77763ec3fb55e47ac79cf5b6c9e0472a05e9ac491f25058826441b230` |
| Signed app executable | `bae39a6aaf1ef25c8ef2cddee1c55786f854f0bdb1671ae26f963723aa12b42b` |

The unsigned generation worker of this build is the one validated in the
synthetic run (`088f4cd4...`); `main` changed no Rust code.

## Method

The user ran Qwen3.8 in Stark and gave OpenWork a task needing several tool
calls in a new session, followed by more steps. Afterwards, a read-only query
of OpenWork's OpenCode v2 database read only these fields of each assistant
message created after the test started:

- provider and model ID
- `time.created`, `time.streamed` and `time.completed`
- `tokens.input`, `tokens.cache.read` and `tokens.output`

OpenCode reports `tokens.input` without the cached part, so the prompt length
is `input + cache.read`. Stark reports `cache.read` from
`usage.prompt_tokens_details.cached_tokens`.

**First text.** This is `time.streamed - time.created`, the client's wait until
the first visible text. For a reply that contains a tool call, Stark withholds
call text until the call is complete. Its first text therefore also includes
decoding the call, so it is an upper bound on the native first-token time.

## Results

One session, 18 Stark replies, no errors:

| Reply | Prompt tokens | Reused | Processed | First text |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 19,064 | 0 | 19,064 | 11.30 s |
| 2 | 19,592 | 16,384 | 3,208 | 3.05 s |
| 3 | 19,735 | 16,384 | 3,351 | 3.02 s |
| 4 | 19,866 | 16,384 | 3,482 | 2.26 s |
| 5 | 19,920 | 16,384 | 3,536 | 2.88 s |
| 6 | 20,201 | 16,384 | 3,817 | 3.61 s |
| 7 | 20,378 | 16,384 | 3,994 | 2.97 s |
| 8 | 20,473 | 16,384 | 4,089 | 1.87 s |
| 9 | 26,153 | 16,384 | 9,769 | 5.47 s |
| 10 | 26,663 | 24,576 | 2,087 | 2.31 s |
| 11 | 27,257 | 24,576 | 2,681 | 3.01 s |
| 12 | 28,418 | 24,576 | 3,842 | 3.70 s |
| 13 | 32,325 | 24,576 | 7,749 | 5.85 s |
| 14 | 34,085 | 28,672 | 5,413 | 4.16 s |
| 15 | 40,757 | 32,768 | 7,989 | 5.71 s |
| 16 | 45,917 | 36,864 | 9,053 | 6.29 s |
| 17 | 59,480 | 45,056 | 14,424 | 9.11 s |
| 18 | 60,720 | 57,344 | 3,376 | 22.14 s |

**Reuse.**
- Every reply after the first reused exactly the previous prompt's last
  4,096-token boundary, so there were no misses. OpenWork kept the start of
  its prompt unchanged between steps and sent no other request to Stark in
  between.
- In total, 430,080 prompt tokens were reused instead of being processed
  again, out of 541,004 (79%).

**First text.**
- Replies 2 to 8 (19,600 to 20,500 tokens) took 1.87 to 3.61 s, against
  11.30 s for the cold first reply.
- In the synthetic run, follow-ups of this length without reuse took 9.9 to
  10.9 s to the first native token alone.
- The time grew where a large tool result had to be processed (replies 9, 13
  and 15 to 17).
- Reply 18 produced 1,238 tokens, so its 22.14 s is mostly decoding: its text
  was released only at completion.

There is no same-session measurement without reuse at these lengths. A prompt
of 26,000 to 60,000 tokens processed in full would take longer than the
11.3 s cold first reply at 19,000 tokens.

## Limits

- One session and one task. Not a benchmark, and not controlled for GPU
  load from other applications.
- No baseline was measured in the same session.
- The first-text figures include tool-call decoding where a reply made a
  call.
- MTP selection was not recorded in the metadata read.
- The build ran from its staging folder. Installing it over the signed
  `/Applications` copy was not done.

## Privacy

Only the numeric fields listed above were read, from a read-only database
connection; message text, titles, paths and every other table were not
accessed. Nothing from the session was saved apart from this table. No
credential was read, copied or used, and the OpenWork connection settings
were not changed.
