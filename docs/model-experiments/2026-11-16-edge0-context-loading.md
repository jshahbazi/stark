# 20261116-EDGE0-20: context length and normal local loading

## Identity and state

| Field | Value |
| --- | --- |
| Registered / owner | 2026-11-16 / Codex, at the owner's request |
| Kind | Context correctness, loading compatibility, lifecycle and observed performance |
| State | Bounded local context/loading validation passed |
| Baseline | `da003d3c`, [singleton attempt 19](2026-11-13-edge0-singleton-speed.md) |
| Scope | Pinned Edge0-145B Q3_K, local non-commercial evaluation only, MTP off |
| Promotion | Adopted on the evaluation branch behind one explicit gate; no catalogue, product, merge, release or installed-build claim |

## Hypothesis and fixed acceptance criteria

The native Metal graph already generates at short context. Its 512-token cap
and diagnostic loading selectors prevent useful ordinary local requests. This
attempt fixes the known indexer finite-check concern before increasing the
evaluation limit, then selects the native backend from the prepared folder
through one explicit local-evaluation gate. The immutable prepared configuration,
payloads and original preparation receipt remain unchanged.

Before actual model execution, require:

- Native indexer controls for sparse selection and finite-error propagation,
  including nonfinite query/key/intermediate values which must not become
  apparently valid through clamping or masking. Existing finite checks remain.
- The ordinary loader must reject Edge0 without an explicit evaluation gate,
  preserve the pinned source/geometry checks, select the Metal backend with the
  gate, and keep MTP off. Historical diagnostic routes retain their identity.
- Short greedy generation must match the retained 128-input/16-output reference
  hash exactly. Long requests must complete with finite timings, valid output
  counts, zero MTP activity and zero prefix reuse. Repeat a long request after
  intervening work to check deterministic reset behavior.
- Exercise the indexer transition on the actual artifact, record the tested
  prompt/context lengths and reject requests above the declared runtime limit.
  Component checks do not establish full-model long-context quality.
- Serialize native tests, builds and model sessions. Use Rust 1.98.1 for the
  release worker and locked Python dependencies. Reap owned workers and verify
  source/build identities. Keep content and token vectors private and unlogged.

The adopted local cap is 8,192 tokens. Failed trials and deferred scopes are
recorded separately below. Timings are
observational, with no host-isolation claim and no change to historical strict
performance screens. Report prefill as input tokens divided by worker prefill
seconds and decode as emitted tokens after the first divided by decode seconds.

## Artifact, precision and capability boundaries

The artifact is `Edge0/Edge0-145B-q3k` at
`d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`, prepared under the
`edge0-native-v2` tokenizer policy. Original Q3_K/IQ4_NL expert weights, Q8_0
projections and the FP8 table remain unchanged. Activations, accumulation,
recurrent state and attention caches remain F32. The prior payload receipt is
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`;
new runs must state whether payloads were rehashed or assumed immutable.

Images are N/A for this exact text-only source; the source inventory is in the
[plan](../EDGE0_PLAN.md#what-the-release-contains). The compatible MTP head is
preserved on disk; this attempt does not enable it. Tool calling is unqualified.
Prefix reuse stays disabled. The updated [Phase 6 inventory](../EDGE0_PLAN.md#phase-6-prefix-reuse)
identifies private checkpoints limited to 512-token single-step staging,
missing Qwen worker prefix publication and snapshot memory admission, and
unqualified longer-context batched restoration. Licensing scope is
the owner's [local non-commercial decision](../archive/runs/2026-11-12-edge0-license-intake.md#owner-decision-2026-11-12).

## Results

Native component checks passed: 56 Edge0 tests with 11 ignored, and 14 Qwen
graph tests with one ignored. The sparse-selector comparison covers token
counts 2,048, 2,049, 4,095, 4,096, 4,097 and 8,192, including exact scalar/parallel
score and visibility agreement. New poison controls check sticky failure before
normalization, clamping and valid overwrites. These component tests do not load
the full artifact. The Python chat/preparation tests pass 154 cases, including
nine new evaluation-gate and template-integrity cases. Formatting, targeted Ruff
and repository policy checks pass.

The first native test compilation found an incorrect module path in a new
inventory assertion; it was corrected before the passing runs and release
build. The first formatting check also identified a function-signature layout,
which was fixed. Neither failed check supplied model or performance evidence.

The fresh release worker is built with:

```sh
CARGO_TARGET_DIR=target/edge0-context-20261116 cargo +1.98.1 build --release --locked -p stark-generate
```

Actual-artifact validation passed with context 8,192, greedy seed 42 and
MTP off: 128, 2,048 and 8,176 input tokens, maximum 16 output tokens, a repeated
2,048-token request after the longest request, a rendered-chat smoke, short
reset replay, and overflow/MTP rejection followed by recovery. The 8,176-token
input leaves 16 tokens to reach the declared limit exactly. Natural EOS is
permitted on new long inputs; emitted and sampled counts are recorded separately.
All seven generation requests completed. The 128-token control exactly matches
the historical 16-output hash. The 2,048-token replay after the longest request,
short reset replay and post-rejection recovery all match their original output
hashes/counts/stopping conditions. The normal prepared template rendered an
18-token chat input; generation emitted two tokens and sampled an EOS as the
third. Requests for context 8,193 and one MTP draft were rejected with
`context_overflow` and `mtp_not_supported`; the following ordinary request
matched the reference. The worker exited zero through stdin EOF and was reaped.

The gate-negative startup also rejects the unchanged prepared folder with
`hybrid_model_load_failed` and exit 1. Its first private assertion incorrectly
expected the generic `model_load_failed` code; that failed assertion is retained
separately, and the corrected check passes. This was an observer expectation
error, not successful ungated loading.

## Observed performance and identity

Hardware: Apple M5 Max, 128 GiB RAM, macOS 26.7.2, arm64. Loading took 1.565 s.
These are individual ordinary-host observations, without quiet admission or a
new strict performance qualification. Original stored weights and F32 arithmetic
are retained; no model payload was rehashed for this run, under the recorded
immutable-payload assumption.

| Request | Input / emitted output | Prefill tok/s | Useful decode tok/s |
| --- | ---: | ---: | ---: |
| Initial reference | 128 / 16 | 8.03 | 9.76 |
| Sparse transition | 2,048 / 16 | 13.61 | 11.22 |
| Near capacity | 8,176 / 16 | 13.74 | 10.87 |
| Long reset replay | 2,048 / 16 | 14.10 | 11.14 |
| Short reset replay | 128 / 16 | 14.54 | 11.71 |
| Recovery after rejections | 128 / 16 | 11.28 | 11.03 |

The near-capacity request took 594.871 s to prefill and 596.275 s overall.
The rendered-chat case has only two emitted tokens and an EOS; its useful decode
rate is 4.89 tok/s versus sampled decode 9.78 tok/s. It is a functional smoke,
not a comparable 16-output performance request. Sampled RSS peaked at
1,030,078,464 bytes; RSS does not count the complete mapped/Metal weight footprint
and does not establish minimum host memory.

The private `normal-context8192-1.context-report.json` and independent offline
audit live under ignored `runtime/edge0-off-speed-2026-11-13/`. The audit verifies
current source inventory, build/helper/corpus/receipt bindings, request counts,
reference/replay hashes, rate formulas, policy and owned-worker cleanup.

| Identity | SHA-256 |
| --- | --- |
| Source base | `da003d3c` plus the bound source manifest below |
| Source manifest | `25c61850b981b1cf383312327b66f1f8cda33fafcca2b9014c9b81de76e4273d` |
| Release worker | `e66b30c4903990c15b3a5022296f6f7313711e542848a0bb0ae8a9880ec5a335` |
| Runner | `052b6231181ff5f7de24e4afbf933339b039ae2fcd9f90af7d0b013180c85a55` |
| Context report | `169c27aea5bf905edb8173bcc474f5c3e2afb8f19a490f8832d5e8cebffcc1e9` |
| Independent audit | `09af54d122b52a364380679c4dd55e325267dae966ddc064df26f4dd9b57d8de` |
| Corrected gate-negative report | `b0692544bfc403e114fc8d13a3f1f79cccc998a6efd76ac7996b744a8e3edc7e` |

Adopt this bounded context/loading path behind
`STARK_EDGE0_LOCAL_EVALUATION=1`; the [native guide](../NATIVE_HYBRID_MODELS.md#checkpoints-and-preparation)
gives the exact source-checkout command. Historical diagnostic routes keep
their 512-token limits. Prefix reuse, tool calling, long-context retrieval and
broader quality, lower-memory hosts, strict timing qualification, catalogue
admission and commercial availability remain unqualified or outside scope.
