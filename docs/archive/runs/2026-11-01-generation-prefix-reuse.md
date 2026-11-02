# Generation prefix reuse and long-prompt prefill profile

Date: 2026-11-01. Follow-up to the
[OpenWork first-text measurement](2026-10-31-openwork-first-text-latency.md),
which attributed about 5.2 seconds of each agent step to native processing of
an 18,500-token prompt. This record covers two pieces of work:

- prefix reuse between consecutive generation requests, implemented in
  `cc942e25e87fc876bf7dbecc08c7874ab7989d3d`
- a profile of where long-prompt prefill time goes

Aggregates and hashes only; prompts, outputs and token vectors stayed in
memory.

## Environment

- Host: Apple M5 Max, 40 GPU cores, 128 GiB, macOS 26.7.1 (25G241), on AC
  power.
- WindowServer GPU load, measured from per-process `accumulatedGPUTime`:
  - About 65% during the first profiling runs.
  - 18–24% after the user hid all windows.
  - About 0% for the final validation and timing run.
- Tools:
  - Workers: Rust 1.98.1, locked dependencies, release builds.
  - Engine tests: `nightly-2026-10-01`.
  - Scripts: uv with Python 3.13.13.
- Model: `mlx-community/Qwen3-4B-4bit`, revision
  `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`, MLX affine 4-bit/group-64.

| Artifact | SHA-256 |
| --- | --- |
| Weights | `7c2af9a45e5b0d196f06a5cbfd6fc5cc95adf29df79885d0540f2206fb3d40d8` |
| Configuration | `5b505594bf00ef9682b373c6a292b88f94f1e1ceab449784c12accb835806c7f` |
| Tokenizer | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| Baseline worker (`6c396202`, fresh state per request) | `4136fb25a2a367b4ce0736d012e3877812df8cd3beee8f44126950f5ece0542d` |
| Candidate worker (`cc942e25`) | `7ed447fe3f0bb98734de42270fd84b693a1cd69c805680a411093a9495be2f3c` |
| Validation record (private, mode 0600) | `c7abf0b649a627b66e0dbfb8051a6ee705dc5f306f5849d295beb65710df9343` |

## Prefix reuse

Design and rules are in [generation](../../GENERATION.md#prefix-reuse) and the
[local API](../../LOCAL_API.md#prefix-reuse).

**Exact by construction.** Reuse stops at a 1,024-token prefill chunk boundary
below the new prompt's final chunk. Only rows from the previous prompt's full,
non-final prefill chunks are kept, so the remaining chunks run with the same
tokens, offsets, shapes and kernels as on a fresh state.

### Method

[`scripts/validate_generation_prefix_cache.py`](../../../scripts/validate_generation_prefix_cache.py)
builds deterministic synthetic agent sessions and renders them with the
pinned model's own chat template. Each session has:

- 40 paragraphs of instructions
- 24 tool definitions
- tool-call/tool-result history

A session is a first turn of about 18,200 tokens and three follow-ups. Each
follow-up adds a tool call and result, or an assistant reply and a user
message. Context capacity is 40,960 tokens, the API default for this model.

**Exactness.** Three workers each run every case:

- the candidate started with `--prefix-cache` and a scope
- the candidate without the flag
- the baseline build

Each case is greedy with up to 128 new tokens, except one seeded stochastic
case. Every output is compared by SHA-256 of its generated token IDs, its text
and its tool calls as parsed by the API's own `qwen3-hermes-v1` parser.

**Timing.** The baseline and the candidate are both loaded. Each runs the
four-turn session in each of four rounds, alternating which goes first. Each
round uses a fresh random scope, so its first turn is cold. Timing requests
generate 8 tokens; the timed figure is the worker's `prefill_seconds`, its
wall-clock first-token stage.

### Exactness results: passed

All 18 cases produced identical generated token IDs, text, finish reasons and
parsed tool calls (one call each) across all three workers.

| Case | Input tokens | Reused tokens |
| --- | ---: | ---: |
| First turn, cold | 18,239 | 0 |
| Follow-up 1 | 18,916 | 17,408 |
| Follow-up 2 | 19,339 | 18,432 |
| Follow-up 3 | 20,669 | 18,432 |
| Repeat, seeded stochastic (temperature 0.7, top-k 20, top-p 0.95) | 20,669 | 20,480 |
| Shorter conversation (history truncated) | 18,916 | 18,432 |
| Divergence at token 3,172 | 20,669 | 3,072 |
| Restore after divergence | 20,669 | 3,072 |
| Context size changed, then restored | 20,669 | 0, 0 |
| Scope changed, then restored | 20,669 | 0, 0 |
| Warm again, and before each invalidation below (6 cases) | 20,669 | 20,480 |

**Invalidation.** Each path was preceded by a warm hit, and the next request
reused 0 tokens after:

- an error reply (`context_overflow`)
- a request without a scope
- `reset_cache`
- an image operation
- `generation_cancelled` raised through the cancellation pipe

`tokenize` and `decode` between turns kept the entry.

### First-token timing (median of 4 samples per worker and turn)

| Turn | Input tokens | Baseline | Prefix reuse | Reused | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 (first turn, cold) | 18,210 | 7.217 s | 7.285 s | 0 | 0.99× |
| 2 (tool result, +654 tokens) | 18,864 | 7.567 s | 0.848 s | 17,408 | 8.9× |
| 3 (tool result, +428 tokens) | 19,292 | 7.838 s | 0.593 s | 18,432 | 13.2× |
| 4 (user turn, +1,251 tokens) | 20,543 | 8.569 s | 1.350 s | 18,432 | 6.3× |

Min/max ranges did not overlap for turns 2–4:

| Turn | Baseline | Prefix reuse |
| --- | --- | --- |
| 2 | 7.51–7.66 s | 0.83–0.86 s |
| 3 | 7.80–7.97 s | 0.59–0.60 s |
| 4 | 8.47–8.65 s | 1.34–1.37 s |

All outputs matched within each turn. A follow-up still prefills its new
tokens plus up to 1,023 tokens left over from the previous final chunk, at
long context.

The cold turn measured 68 ms (0.9%) slower than baseline (ranges 7.12–7.34 s
and 7.21–7.41 s). That turn first releases the previous round's retained
state. This is consistent with the cost of freeing about 3 GiB of KV buffers
on a miss, but the difference was not isolated.

Absolute baseline times here (7.2 s for 18,210 tokens) are higher than the
5.2-second stage in the OpenWork record and the 6.07 seconds recorded on the
separate prefill branch. The cause was not established; comparisons above are
interleaved within one window.

### Tests

| Check | Result |
| --- | --- |
| Generation worker, release | 105 passed, 2 ignored |
| Generation worker, no default features | 69 passed |
| Worker process harness | 6 cases passed |
| Native Metal engine suite | 676 passed, 48 ignored, including the KV rewind tests |
| CPU/grammar engine suite | 433 passed, 20 ignored |
| Scoring worker, optimized | Builds; Clippy clean; tests pass |
| Converter, chat, API and app backend | 5,307 passed, 88 skipped, including 8 new prefix-reuse tests |
| Formatting, Clippy (`-D warnings`), Ruff and repository policy | Pass |

New model-free tests cover:

- chunk alignment
- scope and context keys
- publication only after the reply is flushed
- every invalidation path, including idle expiry and oversized lines
- readiness and rejection when disabled
- per-principal API scopes and revocation resets
- `cached_tokens` usage
- progress validation and first-token estimates from processed tokens

### Limits

- **Synthetic prompts.** No live OpenWork session was measured. A client that
  changes text near the start of its prompt on every step reuses less.
- **One entry.** A request with another scope or an unrelated prompt replaces
  the entry. An example is a client's side request, such as title
  generation, between agent steps. The next step is then processed in full.
- **Not installed or signed.** This build has not been installed into the
  app or signed.
- **Memory.** The retained state keeps up to `context_tokens + 1,024` KV rows
  resident while idle, for at most 600 seconds.
- **Ineligible models.** Native hybrid, recurrent, sliding-window and image
  models are unaffected and keep fresh state.
- **Policy amendment not applied.** The `AGENTS.md` amendment permitting this
  cache was approved by the user, but the automated permission policy blocked
  the agent from editing `AGENTS.md`. It must be applied separately.

## Where long-prompt prefill time goes

### Method

A temporary instrumented build split prefill into one command buffer per
1,024-token chunk and recorded GPU time per chunk; it was reverted. An
18,249-token prompt was used, on the quieter host.

### Per-chunk profile

The linear fit is about 0.221 s per chunk plus 16.75 ms per 1,024 prefix
tokens:

- **Weight matrix products and other per-chunk work:** 55–60%.
- **Attention over the earlier prompt:** about 40%, at about 37 TFLOPS.
- **CPU encoding of all 18 chunks:** 21–26 ms.
- **Sparse KV mapping:** under 1 ms.
- **Pipelining chunk submission:** cannot win meaningfully.

### Matrix products

A temporary Criterion bench measured the Qwen3-4B shapes at 1,024 rows:

| Projection | 4-bit/group-64 | Dense BF16 MXU |
| --- | ---: | ---: |
| QKV | 41.3 TFLOPS | 47.2 TFLOPS |
| O | 39.1 TFLOPS | 49.2 TFLOPS |
| Gate/up | 41.9 TFLOPS | 47.7 TFLOPS |
| Down | 36.5 TFLOPS | 42.1 TFLOPS |

The 4-bit routes already run at about 85% of dense throughput.

### Candidates

**Grouped attention for head-dim-128 prefill** (already used for head-dim
256): measured, not shipped. In an interleaved four-round comparison the median
prefill fell from 6.56 s to 6.27–6.30 s (3–4%). It gained only below about 12K
KV positions. The scorer shares this kernel, so shipping it would break the
recorded bit-exact Qwen3 scoring parity unless gated to generation and
re-qualified numerically.

**Dequantized-weight staging:** not repeated. The separate branch
`codex/openwork-long-prompt-prefill` (`0c49a4db`) already tried it and recorded
bit-exact logits but no speedup. Its records also list grouped-attention trials
as exploratory.

No prefill change was shipped. Prefix reuse is the effective lever for agent
follow-up turns. First turns still depend on prefill throughput, and the
remaining headroom there requires kernel work.

## Privacy

- **Retained in records:** timings, counts, booleans, sanitized error codes
  and SHA-256 hashes of token IDs, text and parsed tool output.
- **Kept in memory only:** synthetic session text and model outputs. The
  harness creates its record exclusively with mode 0600 under the ignored
  `runtime/` directory.
- **Content handling:** no prompt, response, token vector, credential or
  scope value was logged or saved.
- **Unchanged:** the installed app, its Keychain items and the OpenWork
  connection.
