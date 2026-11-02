# Qwen3.8 prefix reuse with recurrent-state snapshots

Date: 2026-11-02. Follow-up to the
[generation prefix reuse run](2026-11-01-generation-prefix-reuse.md), which
covered converted full-attention models only. The installed app was serving
the Lily-format Qwen3.8 checkpoint, a hybrid model with 36 Gated DeltaNet
layers, 12 sparse-attention layers and an MTP head. Its recurrent layers cannot
be rewound, so the converted-model rewind did not apply and every agent step
was processed in full.

This record covers prefix reuse for the native mixed-precision Qwen3.8 backend.
The worker copies the recurrent state at a prefill pass boundary and resumes
the next request there. Aggregates and hashes only; prompts, outputs and token
vectors stayed in memory.

## Design

Lily's prefill runs in 4,096-token passes counted from the position it starts
at, one command buffer per pass.

- **Capture.** A scoped request computes the start of its final pass,
  `floor((N - 1) / 4096) * 4096`. Its prefill is split there, so both parts run
  the same passes, tokens, offsets and final draw as one call. Between the
  parts, the vendored core's existing `DecodeStateApi::snapshot` copies:
  - the GDN states and convolution windows
  - the n-gram history and window
  - the draft head's last hidden
- **Retention.** The attention, indexer and draft-head caches below the
  boundary stay in the loaded model's own state. That state was already
  allocated between requests, so the entry adds only the snapshot, about
  110 MiB.
- **Resume.** The next request reuses the snapshot only when:
  - its prompt begins with the whole snapshotted prefix
  - it leaves at least two tokens to prefill, so its final token takes the
    fresh-state route
  - scope, `context_tokens` and MTP selection match

  The worker then resets the model, calls `restore` and prefills the rest.
  Anything else reuses nothing.
- **Unchanged core.** The vendored Lily core is not modified. The split,
  capture and admission checks are in Stark's wrapper.

Readiness reports `chunk_tokens: 4096` and the key
`model-instance+context-tokens+cache-scope+mtp-selection`. Lifetime,
invalidation and privacy rules are those of the converted-model cache.

## Environment

- Host: Apple M5 Max, 40 GPU cores, 128 GiB, macOS 26.7.1, on AC power. The
  app's own model was stopped for the run, so one worker held the model at a
  time. WindowServer GPU load was not measured.
- Tools:
  - Workers: Rust 1.98.1, locked dependencies, release builds.
  - Scripts: uv with Python 3.13.13.
- Model: `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision
  `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, `lily.format=qwen4_exp-affine-v1`,
  55 shards, 99.56 GiB. Q8 attention K/V, F32 GDN state, BF16 activations.

| Artifact | SHA-256 |
| --- | --- |
| Weights (digest of every shard's name and SHA-256) | `8d7cf6d215df4ece54a9d8ebd2d5a756df7b0f7136ae73980b5eca0143fc5324` |
| Configuration | `2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899` |
| Tokenizer | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| Baseline worker (`6c396202`, fresh state per request) | `d591ea1346e50aaa83a225ed115299dbfe0f37e7bbfbe2f18b2008d1044962ee` |
| Candidate worker (`1f9f5fa8`) | `088f4cd42c495dd8c7ed83348e2c6d01d82de011f23e493d3d9d6aad36edd262` |
| Validation record (private, mode 0600) | `69befec8c6267d88c3212e30629bde68e7fa3919dc51ff766273f7220381fb1c` |

The candidate ran from the working tree before commit. A rebuild from
`1f9f5fa8` reproduces its hash.

## Method

[`scripts/validate_generation_prefix_cache.py`](../../../scripts/validate_generation_prefix_cache.py)
was run with `--reuse boundary`, the default 40,960-token context and up to
128 new tokens. It uses the same deterministic synthetic agent sessions as the
Qwen3-4B run, rendered by the checkpoint's own template:

- 40 instruction paragraphs and 24 tool definitions
- tool-call history
- a first turn of about 18,300 tokens and three follow-ups

The harness now runs workers one at a time, because each Qwen3.8 worker holds
about 72 GB:

- the candidate with `--prefix-cache` runs the full sequence of cases and
  invalidating actions
- the candidate without the flag runs each distinct request once
- the `main` baseline does the same

It compares SHA-256 hashes of generated token IDs and text. It also checks
every reported reuse count against a simulation of the documented rule.

The harness first reproduced the Qwen3-4B result in this sequential form:
28 cases identical, all reuse counts as documented, all invalidations
cleared.

## Exactness: passed

All 32 Qwen3.8 cases produced identical generated token IDs, text and finish
reasons across all three workers. Every reuse count matched the rule.

| Case | Input tokens | Reused tokens |
| --- | ---: | ---: |
| First turn, cold | 18,313 | 0 |
| Follow-ups 1 to 3 | 18,947 to 20,600 | 16,384 each |
| Repeat, seeded stochastic (temperature 0.7, top-k 20, top-p 0.95) | 20,600 | 20,480 |
| Shorter conversation (no rewind below the snapshot) | 18,947 | 0 |
| Regrow after shrinking | 20,600 | 16,384 |
| Divergence at token 12,388, then restore | 20,600 | 0, 0 |
| Change in the final pass, then restore | 20,600 | 20,480, 20,480 |
| Context size changed, then restored | 20,600 | 0, 0 |
| Scope changed, then restored | 20,600 | 0, 0 |
| MTP (2 drafts) after a plain request, then repeated | 20,600 | 0, 20,480 |
| Plain after an MTP request, then repeated | 20,600 | 0, 20,480 |
| Warm, and before and after each action below (12 cases) | 20,600 | 20,480 before; see below |

Six actions were each preceded by a warm hit:

- an error reply (`context_overflow`)
- a request without a scope
- `reset_cache`
- an image operation
- `generation_cancelled` through the cancellation pipe
- `tokenize` and `decode`

The next request reused nothing after each of the first five. It still reused
20,480 tokens after `tokenize` and `decode`.

This branch has no qualified tool parser for the Lily template. The XML
tool-call parser is on a separate branch, so parsed tool calls were not
compared; token IDs and text were.

## First-token timing (median of 4 samples per worker and turn)

Two blocks of two rounds per worker, alternating which worker went first, with
8 new tokens. Each round of the prefix-cache worker used a fresh scope, so its
first turn was cold. The timed figure is the worker's `prefill_seconds`, its
wall-clock first-token stage.

| Turn | Input tokens | Baseline | Prefix reuse | Reused | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 (first turn, cold) | 18,221 | 9.717 s | 9.781 s | 0 | 0.99× |
| 2 (tool result, +660 tokens) | 18,881 | 9.875 s | 1.436 s | 16,384 | 6.9× |
| 3 (tool result, +442 tokens) | 19,323 | 10.127 s | 1.644 s | 16,384 | 6.2× |
| 4 (user turn, +1,227 tokens) | 20,550 | 10.853 s | 2.273 s | 16,384 | 4.8× |

Min/max ranges did not overlap for turns 2 to 4. Outputs matched within each
turn.

| Turn | Baseline | Prefix reuse |
| --- | --- | --- |
| 2 | 9.73–10.05 s | 1.41–1.46 s |
| 3 | 9.92–10.29 s | 1.63–1.68 s |
| 4 | 10.55–11.01 s | 2.26–2.33 s |

**Cold turns.** The cold turn's median was 64 ms (0.7%) slower and the ranges
overlapped (9.38–10.28 s and 9.48–10.25 s). That turn also copies the snapshot,
which upstream measured at about 7 ms.

**Follow-up cost.** A follow-up reprocesses everything after the last 4,096
boundary, here 2,500 to 4,200 tokens. This is coarser than the converted
models' 1,024-token grid.

## Tests

| Check | Result |
| --- | --- |
| Generation worker, release | 108 passed, 2 ignored, including chunk and misalignment fencing |
| Generation worker, no default features | 72 passed |
| Worker process harness | 6 cases passed |
| Engine test suite | Compiles with the new ignored private-checkpoint test |
| Converter, chat, API and app backend | 5,308 passed, 88 skipped |
| Formatting, Clippy (`-D warnings`, CI configuration) and Ruff | Pass |

New model-free tests cover:

- whole-prefix snapshot matching and the two-token remainder
- backend reuse grids in readiness
- fencing when a backend reports reuse off its grid
- Python acceptance of 4,096-token grids

The ignored engine test
`private_checkpoint_prefix_capture_and_restore_match_fresh_prefill` was not
run for this record. It checks that capture leaves the logits unchanged and
that a resumed prefix matches a fresh prefill bit for bit. The worker-level
comparison above covers the same property end to end.

## Limits

- **Synthetic prompts.** No live OpenWork session was measured, and the app
  was not rebuilt with this worker.
- **One snapshot.** Any change before the last 4,096 boundary of the previous
  prompt reuses nothing, as does a shorter prompt. A side request with
  another prompt between agent steps also replaces the entry.
- **One checkpoint.** Only the Lily-format checkpoint was run. The official
  mixed-affine and Uzu-converted Qwen3.8 artifacts take the same code path,
  including the official holdout, but were not measured.
- **Policy wording.** The `AGENTS.md` amendment speaks of reusing attention
  KV state. Snapshot reuse of recurrent state needs its wording extended.

## Privacy

- **Retained in records:** timings, counts, booleans, sanitized error codes
  and SHA-256 hashes.
- **Kept in memory only:** session text, outputs and token vectors.
- **Record handling:** the record file was created exclusively with mode 0600
  under the ignored `runtime/` directory.
- **Content handling:** no prompt, response, token vector, credential or
  scope value was logged or saved.
