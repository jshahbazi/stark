# 20261119-LILYBENCH-01: Lily fixed-corpus benchmark under Sushi's bench conditions

Registered and executed 2026-11-19. Owner: Codex (bench workstream of the Lily
MTP speed project). State: completed (tool) / screening run recorded.
Kind: performance measurement tooling. Disposition: the script and tests are
merged into one uncommitted integration working tree (see
[Integration](#integration)); not committed.

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | `20261119-LILYBENCH-01` / 2026-11-19 / 2026-11-19 / Codex |
| Kind | Performance measurement tooling plus one screening run |
| Attempt state | Completed |
| Result and acceptance criterion | Tool: passed its model-free tests (29). Screening run: 48 of 48 decode samples valid (all finished by `length`, no early EOS); greedy output hashes identical across MTP off and 1 to 3 drafts at each arm and rung |
| Evidence scope | End-to-end timing on the actual artifact, screening only (shared GPU) |
| Promotion decision | Diagnostic only. No number here is a matched comparison with Sushi |
| Disposition | Tool merged with 20261119-LILYROWS-02, 20261119-LILYHOST-01 and 20261119-LILYDEPTH-01 into one uncommitted working tree; not committed, installed or released. Its screening numbers describe the base worker only |
| Predecessor / related attempts / governing issue / PR | Baseline profile `20261119-SUSHI-02` (shared brief); no PR |

## Hypothesis and change

- Problem: Sushi publishes 146.2 tok/s (code) and 79.5 tok/s (prose) decode
  medians for Qwen3.8-Flash-Next at its 4k rung. Stark had no measurement under
  the same workload, so the gap could not be stated.
- Change: `scripts/benchmark_lily_fixed_corpus.py` and
  `scripts/test_benchmark_lily_fixed_corpus.py`, `LICENSE.sushi-reference`,
  and a provenance paragraph in `provenance/UPSTREAM.md`. No engine or worker
  code changed.
- Fixed before the run: a sample is valid only when it finishes by `length`
  with exactly the requested sampled tokens and the MTP acknowledgement matches
  the requested depth; medians use valid samples only.

## Sushi protocol as read from source

Source: Sushi `src/bench.zig` at commit
`297d1e2cf360984340298cfcf86261abbc2eb223` (SHA-256
`47de56fc083b72628861664b1412ac2cbb7fc8e729d5c04cf5b99f0448616ced`), read
only; nothing from Sushi was built or run. Verified against the prompt's summary:

- Requests: `/v1/completions` token ids, `temperature: 0`, `ignore_eos: true`,
  `stream: false` (`promptBody`).
- Code corpus: SplitMix64 seeded with `0x5a5ac0de`; whole TypeScript modules
  (import line, interface with 3 to 6 fields, a service class with 3 to 5
  async retry/cache methods, a key helper) from 20 nouns, 15 verbs and 8 types,
  appended until a chunk reaches at least 24 KiB. Each chunk is tokenized
  separately and the ids concatenated until the corpus holds the largest rung
  plus 928 tokens.
- Prose corpus: embedded Pride and Prejudice (Project Gutenberg #1342, header
  and footer removed), cut into pieces of at most 24 KiB ending after the last
  blank line, each tokenized separately.
- Ladder: a cold 4,096-token request, then 8,192-token windows up to the
  largest rung, each restoring the previous request's prompt end. A listed rung
  decodes and is followed at once by two repeats (three samples, the ladder
  request counts); other windows decode one token. Default rungs 4k, 8k, 16k,
  32k, 64k, 128k.
- Decode tokens: 256 by default; the prose arm decodes at most 128.
- Decode rate: `completion_tokens / decode_ns`, where the decode time starts
  after the first token (`e2e = first_token_ns + decode_ns` in
  `src/scheduler.zig`). Stark's documented rate is
  `(sampled_tokens - 1) / decode_seconds`; the report carries both.
- Sushi's server keeps its default speculative mode (adaptive MTP) on.

The published figures come from `docs/bench/v1.6.0/Qwen3.8-Flash-Next-Sushi-4bpw.md`
at that commit: Sushi 1.6.0, commit `ac3da7aa`, 2026-11-10, M5 Max 128 GB,
fans at max with idle under 50 C, 512 decoded tokens (128 prose), default
speculative mode. 4k code samples 139.8, 146.2, 153.3 (median 146.2); 4k prose
71.4, 80.7, 79.5 (median 79.5); 8k code median 159.8, 8k prose median 98.5.

## Generator port and confidence

The Python port reproduces `Rng`, `Corpus.writeModule`, `Corpus.nextChunk` and
`ProseChunks.next`, including the draw order. Its first three code chunks for
seed `0x5a5ac0de` are 24,901, 27,232 and 25,495 bytes with SHA-256
`a730d434...c62d529`, `d94a9a17...28fe367` and `36a0dc2a...87dfd05` (pinned in
the tests). The SplitMix64 core matches Vigna's published seed-0 outputs.
Prose chunking reassembles Sushi's fixture byte for byte (31 pieces).

Confidence that the code text is byte-identical: high but unconfirmed. The
remaining assumption is that Zig evaluates the arguments of each `print` tuple
left to right, which the port relies on for draw order. Token-id identity is
weaker: Stark tokenizes with the artifact's `tokenizer.json`
(`add_special_tokens: false`), Sushi with its own tokenizer implementation of
its pack. Confirmation would be the SHA-256 of Sushi's first chunk for the same
seed from its own `Corpus.nextChunk` test, and the corpus id hash from Sushi's
`/tokenize`, compared with `chunk_text_sha256` and `corpus_ids_sha256` in this
report. Neither was run here.

## EOS handling

The worker cannot ignore EOS. `docs/GENERATION.md` states model EOS ids are
always honoured and a caller cannot remove them, `crates/stark-generate`
checks `options.eos_token_ids` on every sampled token (`generation.rs` near
line 474 and the native first token in `hybrid.rs` near line 532), and the
request schema rejects unknown fields. The script counts any sample that stops
before the requested length as invalid (`early_stop`) and reports the count.
The minimal worker option needed for a matched protocol would be a
benchmark-only `ignore_eos: true` generate field that skips the model-EOS stop
check (keeping caller stops), with the reply's `finish_reason` still `length`.

## Workload and controls

- Worker launched through `stark_convert.chat.Worker` with
  `--vision-staging-root` and `--prefix-cache`; the Python media handle is
  hidden only during initial readiness (as in
  `scripts/benchmark_lily_release_baseline.py`). Readiness must report the
  pinned Lily image profile.
- Environment names with prefixes `STARK_`, `UZU_`, `LILY_`, `GGML_` removed
  before launch (names recorded).
- Prefix handling (`--prefix scoped`, default): one random `cache_scope` per
  ladder, `reset_cache` before each ladder, fixed `context_tokens` = largest
  rung + arm decode tokens. Stark restores a midpoint snapshot on its 2,048-token
  grid, not Sushi's prompt-end checkpoint; reused counts are recorded per sample.
- One unmeasured 32-token warmup request on a 512-token prefix per depth, with
  the cache reset afterwards (Sushi has no warmup but flags in-request kernel
  compiles as invalid).
- Depths: MTP off and fixed drafts 1, 2, 3 run as separate ladders in that order.
  `--request-option KEY=JSON` passes extra generate fields (for example a future
  adaptive selector) and `--depths default` omits the draft field.
- Refuses to start while another `stark-generate` runs; ran under the model slot.
- Privacy: report holds counts, timings, MTP counters and hashes. A test
  confirms no text, token vector or output reaches it.

## Artifact, source and environment

| Field | Value |
| --- | --- |
| Model | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83` (Lily Q4-XL), image tower loaded, readiness profile `qwen38-lily-retained-bf16-vision-v1` |
| Stark source | `4d09b55045f24eab81da0a9b5b024b9c02318459`, no engine or worker change (tree dirty only with this script, tests, license and docs) |
| Worker | `cargo +1.98.1 build --release --locked -p stark-generate --bin stark-generate` in the bench worktree, SHA-256 `12263dfe03392bcb6a97414e501665c30f2d393e10c8ea31d9f937e666e6c306`. This differs from the shared baseline worker hash `5dfb90b6...` although built from the same commit; the cause (build path or environment) was not investigated |
| Hardware | Apple M5 Max, 128 GB |
| Run | 2026-11-19, under the model slot, other agents active on the host (builds, CPU). Host observation: the worker held essentially all observed GPU client time during every sample (no other model process on the GPU); CPU load from other agents was not screened. Load 12.1 s |
| Report | `runtime/agent-runs/bench-1.json` (private, ignored), SHA-256 `1763c5b6c5b54c369e5ba7861c4cf3724fe888bcd5a54207177770a9640304bd` |
| Corpus hashes | Code: 2 chunks, chunk-text SHA-256 `6693214e...85ce1d`, 15,612 tokens, ids SHA-256 `a5d81d73...df6a04`. Prose: Sushi fixture (SHA-256 `0a0ead8d...5e63f0`, matches), 2 chunks, 11,052 tokens, ids SHA-256 `9961744e...7fa368` |

The prose text was read at run time from the Sushi clone's embedded fixture
`src/fixtures/prose_pride_and_prejudice.txt` (public domain). Nothing was
copied into this repository. To supply it elsewhere, pass any UTF-8 copy of
Project Gutenberg #1342 with the Gutenberg header and footer removed to
`--prose-corpus`; the report's `matches_sushi_fixture` says whether its
SHA-256 equals Sushi's fixture.

Command (from the worktree root, under the slot):

```sh
uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_lily_fixed_corpus.py --model MODEL_DIR --worker target/release/stark-generate --output runtime/agent-runs/bench-1.json --prose-corpus PROSE_TXT --rungs 4k,8k --tokens 512 --depths off,1,2,3 --observe-host --progress
```

## Results (screening)

Decode tok/s by Stark's definition `(sampled - 1) / decode_seconds`; medians
of three valid samples. Sushi's formula (`sampled / decode_seconds`) is
0.1 to 0.5 tok/s higher on every row and is in the report.

| Arm, rung, tokens | Off | 1 draft | 2 drafts | 3 drafts | Samples (off / 1 / 2 / 3) |
| --- | ---: | ---: | ---: | ---: | --- |
| Code 4k, 512 | 74.1 | 85.4 | 81.5 | 92.2 | 74.1 74.1 68.0 / 87.1 85.4 85.1 / 90.8 81.5 72.3 / 92.4 92.2 90.3 |
| Code 8k, 512 | 74.3 | 81.7 | 87.7 | 74.7 | 75.1 74.3 74.2 / 70.1 82.2 81.7 / 67.1 87.9 87.7 / 62.9 74.7 78.7 |
| Prose 4k, 128 | 61.5 | 62.1 | 49.9 | 44.9 | 67.8 60.4 61.5 / 62.1 63.0 57.8 / 60.1 49.9 42.2 / 44.9 45.9 42.2 |
| Prose 8k, 128 | 66.9 | 59.6 | 55.7 | 51.0 | 55.9 66.9 67.9 / 57.3 59.6 64.4 / 55.7 57.0 54.8 / 51.0 47.1 51.5 |

MTP work per rung (three samples pooled; ms per round):

| Arm, rung, drafts | Rounds | Accepted / proposed | Tokens per round | Draft ms | Verify ms | Rollback ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Code 4k, 1 | 777 | 756 / 774 | 1.97 | 1.74 | 19.75 | 0.73 |
| Code 4k, 2 | 528 | 1005 / 1050 | 2.90 | 5.22 | 27.00 | 2.08 |
| Code 4k, 3 | 402 | 1131 / 1203 | 3.81 | 5.40 | 33.00 | 1.70 |
| Code 8k, 1 | 792 | 741 / 792 | 1.94 | 2.12 | 20.86 | 1.11 |
| Code 8k, 2 | 540 | 993 / 1080 | 2.84 | 5.16 | 27.32 | 1.78 |
| Code 8k, 3 | 414 | 1119 / 1242 | 3.70 | 10.49 | 35.11 | 3.47 |
| Prose 4k, 1 | 228 | 153 / 228 | 1.67 | 2.59 | 22.28 | 1.63 |
| Prose 4k, 2 | 189 | 192 / 375 | 2.02 | 7.57 | 28.77 | 2.92 |
| Prose 4k, 3 | 186 | 195 / 549 | 2.05 | 8.10 | 34.29 | 2.50 |
| Prose 8k, 1 | 195 | 186 / 195 | 1.95 | 4.03 | 24.37 | 2.96 |
| Prose 8k, 2 | 159 | 222 / 312 | 2.40 | 7.68 | 29.81 | 3.94 |
| Prose 8k, 3 | 147 | 234 / 429 | 2.59 | 11.28 | 35.20 | 3.79 |

Reused prompt tokens per rung: 4k samples 0, 2,048, 2,048; 8k samples 2,048,
6,144, 6,144 (the midpoint grid). Several first-of-rung or repeat samples are
outliers (for example code 8k at 2 and 3 drafts), consistent with shared-host
noise; three samples cannot separate that from a real effect.

## Beside Sushi's published numbers (not a matched comparison)

| Rung, arm | Sushi 1.6.0 median (default adaptive MTP) | Stark best fixed depth (median) | Stark MTP off |
| --- | ---: | ---: | ---: |
| 4k code, 512 tokens | 146.2 | 92.2 (3 drafts) | 74.1 |
| 4k prose, 128 tokens | 79.5 | 62.1 (1 draft) | 61.5 |
| 8k code, 512 tokens | 159.8 | 87.7 (2 drafts) | 74.3 |
| 8k prose, 128 tokens | 98.5 | 66.9 (off) | 66.9 |

On these screening numbers Stark's best fixed depth reaches about 63% of
Sushi's 4k code median and 78% of its 4k prose median. Remaining differences
in conditions, each of which can move the numbers:

- Pack and quantization: Sushi measured its own `Qwen3.8-Flash-Next-Sushi-4bpw`
  pack; Stark ran the Lily Q4-XL artifact with the image tower loaded. Weight
  formats, bytes per token and the MTP head differ.
- Sampling path and EOS: both greedy, but Sushi sends `ignore_eos`; Stark
  honours model EOS. No Stark sample stopped early, so this did not change
  these samples, but the outputs being decoded are not known to be the same
  token sequence.
- MTP policy: Sushi's default mode adapts depth per round (expected-value
  planner, plus prompt-lookup drafts); Stark used one fixed depth per ladder.
  "Best fixed depth" is chosen after the fact, which favours Stark slightly.
- Tokens per sample: both 512 code and 128 prose. Rate formula: Sushi divides
  all completion tokens by post-first-token time; Stark's figure counts one
  fewer token (Sushi's formula is in the report).
- Prompt tokens: same corpus text by construction (port unconfirmed), but
  token ids come from different tokenizer implementations and are unverified.
- Prefix handling: Sushi restores its prompt-end checkpoint (restored counts
  4,065 and 8,161 at 8k); Stark restores 2,048-token-grid midpoints and
  prefills the rest. This affects prefill, not the decode timing window; Stark's
  restore is documented as exact, so decode starts from the same state as a
  fresh prefill.
- Host state: Sushi ran fans at max, idle under 50 C, taskpolicy, a GPU lock
  and clock monitoring; Stark ran on a host shared with other agents without
  thermal or clock control.
- Date and build: Sushi 2026-11-10 (commit `ac3da7aa`); Stark 2026-11-19
  (`4d09b550`). Different software on both sides.
- Warmup: Stark ran one unmeasured 32-token request per depth; Sushi has none.

## Attempt history and failures

| Trial / source | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| bench-1 | 4k and 8k rungs, both arms, off and 1 to 3 drafts, 3 samples each | 48/48 valid, completed in one slot hold | Screening only |

No failed or discarded runs.

## Correctness and lifecycle

- Model-free tests: `uv run --project tools/converter --locked --python 3.13.13 pytest scripts/test_benchmark_lily_fixed_corpus.py -q`, 29 passed; `ruff check` clean with the converter configuration.
- The run checks MTP acknowledgements per sample; greedy output hashes were
  identical across all four depths and all three samples at each arm and rung
  (one hash per arm and rung), including restored-prefix samples.
- Prefix reuse is exercised (scoped) but this is not a reuse qualification.

## Evidence, decision and limits

- Use: every later optimization in the project can be screened with the same
  command; `--request-option` and `--depths default` cover future selectors.
- Not run: 16k to 128k rungs, quiet-host conditions, Sushi itself, the
  tokenizer and byte-identity confirmations above.
- Next: the orchestrator's quiet run; a worker `ignore_eos` benchmark option
  if a sample ever stops early; confirming the corpus hashes against Sushi.

## Follow-up history

None yet.

## Integration

2026-11-19. The adopted attempts were merged into one uncommitted working tree
(the `claude/sushi-stark-performance-598713` checkout, base `4d09b550`), in
this order: `bench.patch` (SHA-256 `95e3ca04...`, 20261119-LILYBENCH-01),
`gdn-replay-v1.patch` (`542bc1e5...`, 20261119-LILYROWS-02, without the later
projection variants), `host-roundtrip.patch` (`ff797c62...`,
20261119-LILYHOST-01) and `depth-policy.patch` (`683cb04f...`,
20261119-LILYDEPTH-01). Nothing is committed, installed or released; the final
quiet before/after timing is pending.

Merge resolutions:

- `MtpSampler::draft_point_mass` exists once. The kept implementation refuses
  a call after verification began and an id at or past a known vocabulary
  size, and, like `draft`, accepts a sampler whose vocabulary size is not yet
  known; the verify pass refuses ids outside the model vocabulary and
  `target` checks and sizes the readout before any proposal is used. In the
  worker, MTP requests always draw the first token through `sample_initial`,
  so the sampler is sized before the first round on both the device chain and
  lookup paths. Both authors' tests are kept (the host test's unsized case now
  expects acceptance) and `first_round_point_masses_before_the_sampler_is_sized`
  covers the first round with and without sizing.
- `MtpDecode::round`: a head plan of d > 0 drafts (fixed or adaptive) goes
  through `draft_host_encoding_verify`, so greedy requests use the device
  draft chain and every head round encodes its verify graph ahead; a lookup
  plan records point masses, calls `advance_head_host` before the last round
  and verifies with `verify_host_batched`; a scheduled zero-draft plan returns
  through `serial_round` before anything is encoded.
- `verify_host_encoded` takes the scratch mutably for the GDN shadow swap, and
  an encoded-ahead verify also checks that the GDN recurrent-state buffers it
  was encoded against are still current (the replay rollback swaps them).
- The depth planner's static cost table was measured before the host
  round-trip change and is unchanged; retuning is a separate decision.

Gates on the merged worker (Rust 1.98.1 release build, SHA-256
`d4182a6769e7c86c6865d0660ccc0d82d4d6442c774eb0f0bdbe83fd6c3041af`). All
timings are unscreened shared-host screening and are not quoted here.

| Gate | Result |
| --- | --- |
| Release build | Clean |
| stark-generate library tests | 153 passed, 5 ignored |
| stark-engine `hybrid` filter / `mtp_` filter | 285 passed, 73 ignored / 30 passed, 8 ignored |
| lily-core library tests | 98 passed, 58 ignored; plus 12 ignored physical-Metal byte tests run and passed (GDN verify, value-key verify, projection rows, chunk capture, attention verify, chained argmax and finite readout) |
| Benchmark script tests | 29 passed |
| `cargo fmt --check` (workspace crates) | Clean |
| `private_checkpoint_mtp_verify_and_rollback_match_serial`, real model | Passed, including the device-versus-host greedy section: verify 86, rollback 39 and host-serial 30 readouts with 0 unequal values; draft-head phase differs as allowed (0 top-1 mismatches) |
| Eight-arm output-hash gate (`runtime/agent-runs/integration-1.json`) | All eight output hashes equal the 20261119-SUSHI-02 baseline |
| New options, frozen corpus, two repeats (`integration-options-1.json`) | Greedy adaptive, lookup and adaptive plus lookup equal the baseline greedy hash; sampled adaptive, lookup and adaptive plus lookup one hash each across repeats. Sampled lookup had no lookup rounds on this input |
| New options, copy-heavy input, two repeats (`integration-options-2-copy.json`) | Greedy adaptive, lookup and adaptive plus lookup equal fixed two-draft greedy; sampled lookup (2 lookup rounds) and adaptive plus lookup (6) one hash each across repeats |

Not covered end to end: a scheduled zero-draft round in the middle of a
stream (only final-round zero-draft rounds occurred in these runs).
