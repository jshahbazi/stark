# Guarded Pi live qualification, 2026-11-13

Installed Pi 1.5.2 passed **21 live checks** with the real Qwen3-4B 4-bit
artifact through the installed Stark 0.20.5 desktop backend. **43 model-free
checks passed.** The required manual guard fixes the missing-usage and missing
`[DONE]` client gaps before successful completion or tool execution.

The implementation and full aggregate report are in the separate integration
repository at
[`d05490c6af5eccf7f8ca068f4a5840a8dd3fb76f`](https://github.com/jshahbazi/stark-agent-clients/tree/d05490c6af5eccf7f8ca068f4a5840a8dd3fb76f).
See its [dated record](https://github.com/jshahbazi/stark-agent-clients/blob/d05490c6af5eccf7f8ca068f4a5840a8dd3fb76f/docs/runs/2026-11-13-pi-guard-live.md)
and [aggregate JSON](https://github.com/jshahbazi/stark-agent-clients/blob/d05490c6af5eccf7f8ca068f4a5840a8dd3fb76f/docs/runs/data/2026-11-13-pi-live-summary.json).
Stark runtime code was not changed by this work.

## Bound identities

| Component | Identity |
| --- | --- |
| Installed Stark | 0.20.5, source `9e33ab729581994269f56c64685e27a0bdd94c27`, build `20261112205226880595` |
| Worker SHA-256 | `da8a9e5384300e488da486741f617981c36ff8e83c7d7e1f66925451ef835ee7` |
| Template worker SHA-256 | `21bd37828286014b6e96c0d5d1877fd068ec0cd32991ed6fc9cccd841e2fe9f6` |
| Pi | Installed 1.5.2; bundled CLI SHA-256 `e79626f2dd6f94aa45d30f3fa63cd84319a6eefcd150b353cfaf274366926774` |
| Primary checkpoint | `mlx-community/Qwen3-4B-4bit`, revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` |
| Primary weights SHA-256 | `7c2af9a45e5b0d196f06a5cbfd6fc5cc95adf29df79885d0540f2206fb3d40d8` |
| Primary config SHA-256 | `5b505594bf00ef9682b373c6a292b88f94f1e1ceab449784c12accb835806c7f` |
| Primary tokenizer SHA-256 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| Primary template / parser | `87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5` / `qwen3-hermes-v1` |
| Source/test manifest SHA-256 | `e7a61f9f3fa3ad08c8458d0c95a0d966d3206d3802eb8ec99855246db21b45f4` |
| Executed harness SHA-256 | `852da1c26c699195aacaee1155d955d66137f1cca93e3b0c69549520fc7b3e36` |
| Live aggregate SHA-256 | `a2a945d09107100aa58b345108b612413a5845239cbf47dc0846b0a99f4bfe40` |

The packaged desktop/API Python files matched manifest and Git revision hashes.
Host: Node.js 26.13.0, macOS 26.7.2 (25G332), arm64 Mac17,7, 128 GiB RAM.
The model used unchanged MLX affine 4-bit/group-64 weights through Rust/Metal;
this is client/lifecycle qualification, not new numerical precision evidence.
Context was 40,960; API output ceiling 40,959; Pi output budget 1,024, with a
separate eight-token truncation profile. MTP/reasoning controls were off. Both
named Qwen3 artifacts are text-only; image/MTP checks are not applicable.

## Results and independent review

The generated API marker `stark-guarded-chat-completions` is implemented only by
the required manual extension. It uses Pi's public Chat Completions delegate,
bounded event validation and direct loopback HTTP. Missing extension fails
before inference. The wire API remains ordinary Stark Chat Completions.

| Live workflow | Result |
| --- | --- |
| Setup, registration, real model load, authenticated discovery, installed Pi route | Passed, five checks |
| Text/follow-up and real read/edit/test/diff | Passed; four exact tool rounds, test file unchanged |
| Failed command followed by the real test command | Passed |
| Native prefill/decode cancellation and recovery | Passed using worker phase/sample counters, not admission or text arrival alone |
| Context rejection, compaction and output truncation | Passed; native context error verified; historical marker survived compaction without being supplied in summary instructions |
| Unavailable/stale model, model switch, port change, backend restart | Passed with expected error classes and explicit recovery |
| Interleaved sessions and killed-client recovery | Passed; shared manual-principal cache scope retained |
| Prefix versus fresh control | Passed; 3,072 tokens reused; output/finish/tool structure equal after fresh API-session reset |
| Matched raw/guarded controls | Passed in ABBA order, with separate cold/reused arms and equal output |

Manual compaction reduced estimated context from 3,307 to 1,488 tokens. First
visible client text was 336 ms on the initial request and 110 ms on follow-up.
Short matched controls had 3,527 input and two output tokens. Raw fresh first
text was 701–708 ms, guarded fresh 679–707 ms; reused raw 126–150 ms and guarded
133–141 ms. Two observations per arm support only these bounded observations;
no speedup or full-context throughput claim is made.

An independent reviewer checked the guard and identified weak initial harness
predicates. Final predicates verify native cancellation phases, exact requested
commands/order, the unchanged test file, answer-free compaction retention and
expected context/model errors. The reviewer verified the final harness hash,
21/21 aggregate result and cleanup, and found no remaining blocker in this scope.

## Prior attempts, limits and cleanup

Attempts `20261113-PI-01` through `20261113-PI-08` are recorded in the
[model ledger](../../model-experiments/2026-11-13-pi-manual-qualification.md).
Preparation failures, an earlier literal-word recall miss, ineligible
compaction, a stopped-API harness mistake, weak predicates and a progress-field
timeout remain recorded. Earlier passing recipes with weak predicates are
superseded. The original [foundation](2026-11-12-pi-manual-foundation.md) and
[recheck](2026-11-12-pi-manual-recheck.md) records remain unchanged.

The actual client surface was official Pi CLI RPC. The harness ran the packaged
Stark desktop backend/native worker in an isolated library using an ephemeral
manual key through the private host pipe. It did not retrieve or rotate the
production Keychain key, modify the user's Pi profile/model registry, or run
the Swift credential/setup UI. Owned processes and temporary directories were
removed; only aggregates and hashes were published.

The alternate Qwen3-0.6B BF16 artifact was used for routing and short recovery,
not the primary tool matrix. Broad coding quality, other artifacts/versions,
managed grants, terminal keybindings/setup UX, numerical parity and full
UI-consumer backpressure remain outside this result. Pi's own event queue has
separate buffering from the bounded guard.
