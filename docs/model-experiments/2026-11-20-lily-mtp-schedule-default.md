# 20261120-LILYSCHED-01: Lily Q4-XL default MTP schedule (adaptive with prompt lookup)

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | `20261120-LILYSCHED-01` / 2026-11-20 / 2026-11-20 / implementing agent and orchestrating session for the repository owner |
| Kind | Sampling (default schedule selection), lifecycle and protocol |
| Attempt state | Completed |
| Result and acceptance criterion | Passed. Criteria fixed before the real-model runs are listed under "Hypothesis and change" |
| Evidence scope | Structural and protocol (unit tests with a mock backend, engine predicate tests); full-model output hashes; prefix-reuse lifecycle on the full model; end-to-end timing at 4,096 input tokens |
| Promotion decision | Promoted within named scope: the default schedule of MTP-enabled text requests on the pinned image-loaded Lily Q4-XL artifact. Every other model, layout and backend unchanged |
| Disposition | Adopted on branch, uncommitted at registration; not installed |
| Predecessor / related attempts / governing issue / PR | Decision: [20261119-LILYADAPT-01](2026-11-19-lily-mtp-schedule-default-gate.md). Prefix reuse with the selectors and with the default: [20261119-LILYPREFIX-01](2026-11-19-lily-mtp-selector-prefix-reuse.md). Planner and cost table: [20261119-LILYDEPTH-01](2026-11-19-lily-mtp-depth-policy.md), [20261119-LILYMERGE-01](2026-11-19-lily-mtp-merged-quiet-timing.md). PR: not opened |

## Hypothesis and change

- Problem: LILYADAPT-01 selected the adaptive policy with prompt lookup as the
  schedule an MTP-enabled text request on the pinned Lily Q4-XL artifact
  should get when it names none. Until now both selectors were opt-in, so
  every request without them ran the fixed schedule.
- Change: a model-level default schedule.
  - Engine (`crates/stark-engine/src/engine/hybrid/mod.rs`): the pinned
    artifact identity (source, revision, artifact SHA-256) is one predicate,
    `pinned_lily_q4_xl`, shared by the midpoint-boundary default and the new
    schedule default. The image-loaded pinned artifact with its MTP head
    loaded and `supports_mtp_scheduling()` gets `MtpScheduleDefault`
    (adaptive, lookup on); every other model, layout and backend keeps
    (fixed, lookup off). `STARK_LILY_MTP_ADAPTIVE=0`, read once at load,
    keeps the fixed schedule with lookup off.
  - Planner (`mtp_policy.rs`): `CostTable` carries a stable `id`
    (`"lily-q4-xl-lilymerge-01"` for `LILY_Q4_XL_LILYMERGE_01`), reported as
    `stats.schedule.cost_table` whenever the adaptive policy ran.
  - Worker (`crates/stark-generate`): `GenerationBackend` gains
    `default_mtp_scheduling()` (default fixed, no lookup); the hybrid backend
    forwards the engine's. The worker captures it at construction beside
    `mtp` and `default_mtp_drafts`, rejects a non-default schedule without
    scheduling support as `invalid_model_config`, and fences a later change as
    `invalid_generation_output`. With MTP on, an omitted `mtp_policy` or
    `mtp_lookup` takes the model default; explicit values win. With MTP off no
    default applies. The existing limit rule (explicit `mtp_drafts` is the
    limit; `"mtp": true` under the adaptive policy plans to `max_drafts`)
    applies whether the adaptive policy was requested or defaulted. The
    worker's `default_drafts` stays 0. Readiness reports `default_policy` from
    the model and a new `default_lookup`. The acknowledgement adds `policy`,
    `lookup` and `stats.schedule` whenever the effective schedule is not
    (fixed, lookup off).
  - Unchanged: prefix-reuse key and snapshot, image requests (MTP off, fixed
    schedule), `recommended_drafts`, the Python host and the app's
    paged-residency memory profile.
- Expected behaviour: greedy output is unchanged on every request. Seeded
  sampled output of MTP-enabled requests on the pinned artifact without
  selectors differs from the previous default. The desktop app sends an
  explicit `mtp_drafts` equal to `recommended_drafts` (2), so in the app the
  default becomes adaptive with lookup at a limit of two.
- Real-model criteria, fixed before the runs. On the windows and seeds of the
  LILYADAPT-01 limit sessions (a4), which the previous worker ran with
  explicit selectors:
  1. A request with MTP on and no selector on the new worker gives the output
     hash the previous worker gave with `"mtp_policy": "adaptive"` and
     `"mtp_lookup": true`, at limits of two and three, greedy and sampled.
  2. A request with the explicit opt-out gives the hash of the previous
     worker's fixed schedule.
  3. With the kill switch set, the same holds with no selector, and the frozen
     eight-arm hash gate of SUSHI-02 reproduces the base worker's hashes.
  4. MTP-off output is unchanged.
  5. The prefix-reuse validator passes on the new build, with opt-out cases
     equal to the base worker and an image request between text requests.

## Artifact, source and environment

| Field | Baseline | Candidate |
| --- | --- | --- |
| Model family, exact checkpoint and upstream revision | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, artifact SHA-256 `18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4` (identity constants in the engine; the installed copy was loaded, payload not rehashed) | Same |
| Format/quantization and config/weight/tokenizer/processor/template hashes | Not rehashed in this attempt; the prefix validator records config, tokenizer and weight-index hashes in its private receipt | Same |
| Backend, vendor revision and local patch provenance | Native hybrid Lily backend; `vendor/lily-core` unchanged | Same; no vendor change |
| Stark commit, dirty patch/runtime manifest and executable hash | Previous worker: `a246039f`, SHA-256 `5abca56f752489dde82d60b42e29ddaf36542962a83770bfba27a52633a3e26b`. Base worker: `4d09b550`, SHA-256 `5dfb90b61a459560702590703117b5c705e2ae425b7d5da2d5a709bcc766298d` | `a246039f` plus this uncommitted patch; `stark-generate` SHA-256 `02d8247c1341b5b8865dbc5fa0a749ceae57ca05950a9f7c7a3e9b5c3cfc074a` |
| Toolchains, dependency lock hashes and build flags | Rust 1.98.1 release; nightly-2026-10-01 for format and engine tests; `--locked` | Same; no dependency change |
| Weight, activation, accumulation, recurrent-state and KV precision | Unchanged | Unchanged |
| Selected kernel and materialization policies | Unchanged | Unchanged; only the round schedule selection changes |
| Hardware, physical RAM, OS/driver/Metal toolchain | Apple M5 Max, 128 GiB | Same |
| Available headroom, observed peak memory and measurement method | Not measured | Not measured |

## Workload and controls

- Load mode: the installed artifact with its image tower loaded, which is the
  condition for the default.
- Pairwise sessions (v1): the a4 inputs of LILYADAPT-01 again on the new
  worker: 4,096 input tokens, six windows each of the prose and code corpora,
  seeds from 242, greedy 512 tokens and sampled 256 (temperature 1, top-k 20,
  top-p 0.95). Arms: MTP off; two and three drafts with no selector; two
  drafts with the explicit opt-out. Fresh requests, zero reused tokens.
- Kill-switch sessions (v1k): the code session again with
  `STARK_LILY_MTP_ADAPTIVE=0` and no selector, and the frozen eight-arm
  session of SUSHI-02 (off and one to three drafts, greedy and sampled, two
  requests per arm) compared with the base worker's recorded hashes.
- Prefix reuse: the model-bound validator, 8,192 context tokens, 16-token
  outputs, two drafts, boundary reuse, image-loaded, with a completed image
  request and a cancellation among the invalidation paths. Details in
  LILYPREFIX-01's follow-up.
- Host not screened or quiet-admitted; one model process at a time under a
  local lock. Load averages at the start and end of the v1 sessions were 0.5
  to 1.9.
- Matched text-only versus image-loaded arms: not run; the default exists
  only on the image-loaded artifact.

## Attempt history and failures

| Trial / source | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| Implementation | Engine default, worker resolution, readiness, acknowledgement, cost-table identifier, documentation | Model-free checks below pass | Adopted on branch |
| Converter suite, first run | Run before `target/release/stark-template` was built | 369 failures and 140 errors in template tests | Invalid setup; rerun after building the helper passed |
| v1 pairwise sessions | Criteria 1, 2 and 4 | 100 requests, no error; every pair equal | Passed |
| v1k sessions | Criterion 3 | 32 requests, no error; every pair equal; eight of eight frozen arms equal | Passed |
| `prefix-2` | Criterion 5 | 77 of 77 checks, 6 of 6 invalidation paths | Passed |
| Engine real-model tests | Four ignored tests selected by `private_checkpoint_mtp_` | Three passed; one stopped on its own precondition (it needs an official MLX-rounding checkpoint) | Three valid; the fourth is an invalid setup for this artifact, unchanged from the base commit |

## Correctness and lifecycle

- Readiness on the installed artifact: `default_policy: "adaptive"`,
  `default_lookup: true`, `default_drafts: 0`, `recommended_drafts: 2`,
  `max_drafts: 3`. With the kill switch: `default_policy: "fixed"`,
  `default_lookup: false`.
- Default path (criterion 1): in all six windows of both corpora, a request
  with two or three drafts and no selector gave the output hash, and the same
  round, proposal, acceptance, depth and lookup counters, as the previous
  worker with explicit adaptive and lookup selectors; greedy and sampled.
  Each reply acknowledged `policy: "adaptive"`, `lookup: true`, the requested
  limit and `cost_table: "lily-q4-xl-lilymerge-01"`.
- Opt-out (criterion 2): two drafts with `"mtp_policy": "fixed"` and
  `"mtp_lookup": false` gave the previous worker's fixed-schedule hash in all
  twelve windows, greedy and sampled, with the plain acknowledgement.
- Kill switch (criterion 3): two drafts with no selector gave the previous
  worker's fixed-schedule hash in all six code windows, greedy and sampled,
  with the plain acknowledgement. All eight frozen arms reproduced the base
  worker's hashes and work counters.
- MTP off (criterion 4): equal in all twelve windows, greedy and sampled.
- Prefix reuse (criterion 5): every case identical to the same build without
  the prefix cache; reuse counts follow the existing rule (4,096 tokens on
  warm cases, zero where the rule says zero); switching between the default
  and the opt-out on consecutive requests keeps the entry. All 16 opt-out and
  fixed cases equal the base worker, stochastic included. Seven stochastic
  cases under the adaptive schedule have no base-worker counterpart and were
  compared only with the uncached build. A completed image request, a
  cancellation, an error reply, an unscoped request and a reset each cleared
  the entry; tokenize and decode kept it.
- Engine real-model tests on the final tree
  (`cargo +nightly-2026-10-01 test --locked -p stark-engine --features cpu,metal -- --ignored private_checkpoint_mtp_ --test-threads=1`
  with `STARK_TEST_MODEL_PATH` at the installed artifact): three passed.
  `private_checkpoint_mtp_verify_and_rollback_match_serial`: verify (86
  readouts), rollback (39) and host-serial (30) phases have 0 unequal values;
  the draft-head phase has 248,320 unequal values with 0 top-1 mismatches,
  the known allowed difference.
  `private_checkpoint_mtp_zero_draft_round_mid_stream_matches_serial`: 0
  unequal continuation values. `private_checkpoint_mtp_batched_stage_differential`:
  554 stages, no divergence. The filter also selected
  `private_checkpoint_mtp_first_drafts_are_independent_of_previous_request`,
  which asserts an official MLX-rounding checkpoint as its first step and so
  stops on this artifact; the assertion is the same at `a246039f` and at
  `4d09b550`, and the test is not evidence for or against this change.
- Worker unit tests added (`crates/stark-generate/src/generation.rs`):
  `model_default_schedule_applies_only_with_mtp_on_and_explicit_selectors_win`,
  `model_default_schedule_requires_scheduling_support_and_is_fenced`,
  `fixed_default_backend_reports_and_acknowledges_as_before`,
  `model_default_schedule_never_reaches_image_requests`. Existing tests are
  unchanged.
- Engine unit tests added: `lily_mtp_adaptive_defaults_on_with_zero_as_kill_switch`,
  `pinned_lily_q4_xl_identity_requires_source_revision_and_artifact_hash`
  (`hybrid/mod.rs`), `adaptive_schedule_reports_its_cost_table_and_fixed_does_not`
  (`hybrid/mtp_policy.rs`).
- Commands and counts by the implementing agent, after the timing quiet
  period ended:
  - `cargo +nightly-2026-10-01 fmt -p stark-engine -p stark-engine-macros -p stark-score -p stark-generate -p stark-template -p stark-resource-budget --check`: clean.
  - `cargo +1.98.1 test --locked -p stark-generate`: 157 passed, 0 failed, 5 ignored.
  - `cargo +nightly-2026-10-01 test --locked -p stark-engine --features cpu,metal -- --test-threads=1 hybrid`: 289 passed, 0 failed, 74 ignored.
  - Same with the filter `mtp_`: 33 passed, 0 failed, 9 ignored. The new
    identity-predicate test, outside both filters, passed when run by name.
  - `cargo +1.98.1 build --release --locked -p stark-generate --bin stark-generate`: built, SHA-256 above.
  - `HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false uv run --project tools/converter --locked --python 3.13.13 pytest tools/converter/tests -q`:
    7,960 passed, 96 skipped.
- Repeated by the orchestrating session on the final tree: the format check
  (clean), `scripts/ci/check_repository.py` (passed), the `stark-generate`
  tests (157 passed, 5 ignored) and the validator and fixed-corpus script
  tests (63 passed). The engine hybrid suite and the converter suite were not
  repeated.
- Python host: no change. `mtp_capabilities` ignores unknown readiness fields
  and `validate_mtp_reply` checks only `enabled`, `drafts`, `sampling` and
  `verification`, so `default_lookup`, `policy`, `lookup` and
  `stats.schedule` are accepted. The harness used here drives the worker
  through the same `stark_convert.chat.Worker` class as the app.
- Not tested: the desktop app end to end, `"mtp": true` resolving to three
  drafts on the real model (unit-tested only), failure recovery other than
  the validator's error reply, memory.

## Performance and memory

Decode rate relative to MTP off on the new worker, median of six windows,
4,096 input tokens. LILYADAPT-01 holds the measurements the decision rests on;
these confirm the shipped paths behave the same.

| Metric and measurement boundary | Fixed opt-out, 2 drafts | Default, limit 2 | Default, limit 3 | Repeats/dispersion | Eligibility and conclusion |
| --- | ---: | ---: | ---: | --- | --- |
| Prose, greedy | 0.938 | 1.060 | 1.044 | six windows | Unscreened host; consistent with a4 (0.942, 1.060, 1.049) |
| Prose, sampled | 0.952 | 1.053 | 1.062 | six windows | Consistent with a4 (0.954, 1.086, 1.087) |
| Code, greedy | 1.372 | 1.376 | 1.362 | six windows | Consistent with a4 (1.391, 1.383, 1.380) |
| Code, sampled | 1.261 | 1.224 | 1.315 | six windows | Consistent with a4 (1.269, 1.216, 1.315) |
| Tokenization/rendering/planning, first token, prefill | not measured | not measured | not measured | | |
| Observed peak memory / retained payload | not measured | not measured | not measured | | |

## Evidence, decision and limits

- Public aggregate:
  [`lily-mtp-schedule-default-validation-2026-11-20.json`](../benchmarks/data/lily-mtp-schedule-default-validation-2026-11-20.json),
  with the SHA-256 of every private receipt.
- What it establishes: on the pinned image-loaded artifact the default path
  is exactly the previously opt-in adaptive schedule with lookup, the opt-out
  and the kill switch are exactly the previous fixed schedule, MTP-off output
  is unchanged, and prefix reuse is exact under the default.
- Consequence for callers: seeded sampled output of MTP-enabled requests
  without selectors on this artifact differs from the previous default.
  Greedy output does not. A later change to the planner or its cost table
  changes such sampled output again and must change `cost_table`.
- Adoption: on branch, uncommitted; not installed.
- [Performance playbook](../PERFORMANCE_PLAYBOOK.md): updated under adaptive
  depth, prefix reuse and the correctness gate for a change of default.
- Untested: other artifacts (they keep the fixed default), other sampled
  settings, the app end to end, memory, and a draft limit above the app's two
  in its paged-residency profile. Natural responses through the chat template
  were measured afterwards; see "Follow-up history".

Keep prompts, completions, token vectors, probabilities, credentials, exception
dumps and model/private evaluation files out of this document.

## Follow-up history

### 2026-11-20: natural long responses through the chat template

Registered before any request of this follow-up ran.

- Question: both earlier changes of the default depth were reverted after
  losing on a natural long response, and LILYADAPT-01 measured raw corpus
  continuation only. Does the new default hold against two fixed drafts and
  against MTP off on natural responses rendered through the chat template?
- Workload: 24 requests written for this check, eight each of coding,
  planning and explanatory or narrative prose. Each is rendered through the
  artifact's chat template with its default system message and thinking
  disabled. Context 8,192 tokens, response ceiling 1,024, native end of
  sequence honoured, no stop strings. Greedy, and sampled at temperature 1,
  top-k 20, top-p 0.95 with seed 42 plus the request index.
- Arms, each on the same rendered input and seed: MTP off; two fixed drafts
  through the explicit opt-out; the default as the desktop app requests it,
  `mtp_drafts: 2` with no selector. Worker `02d8247c`, one loaded process,
  fresh requests, arm order alternating by request.
- Validity: a request counts for a sampling mode only when every arm of that
  mode generated at least 384 tokens. A cell is one category and one mode and
  needs at least six valid requests. If any cell has fewer, the same number of
  new requests is added to all three categories and the pool is evaluated.
- Statistic: per request, decode rate over the MTP-off arm's rate; the cell
  statistic is the median.
- Pass: in every cell the default is at least 0.97 and no more than 5 percent
  below two fixed drafts, the thresholds of LILYADAPT-01. Otherwise fail.
- Reported without gating: the geometric mean over cells, request wall time
  for greedy arms (which write the same text), response lengths and how each
  ended, and whether greedy arms share one output hash per request.
- A failure is reported to the owner with its cells. The default is not
  changed again without the owner's decision.

Result: passed in all six cells. 146 requests in one session, no error; all
24 requests valid in both modes; none dropped.

| Category, mode | Valid requests | 2 fixed drafts | Default, limit 2 | Default over 2 fixed |
| --- | ---: | ---: | ---: | ---: |
| Coding, greedy | 8 | 1.363 | 1.385 | 1.017 |
| Coding, sampled | 8 | 1.382 | 1.367 | 0.989 |
| Planning, greedy | 8 | 1.143 | 1.149 | 1.005 |
| Planning, sampled | 8 | 1.152 | 1.152 | 1.000 |
| Prose, greedy | 8 | 1.203 | 1.206 | 1.003 |
| Prose, sampled | 8 | 1.196 | 1.203 | 1.006 |
| Geometric mean | | 1.236 | 1.240 | |

- The default is level with two fixed drafts on natural responses, within 1.7
  percent in every cell, and 15 to 38 percent above MTP off. It does not lose
  where the two earlier default changes lost.
- It is also no faster there. These responses are more predictable than the
  corpus continuation of LILYADAPT-01: two fixed drafts were accepted at 90
  percent on coding and 71 to 72 percent on planning and prose, and ran 14 to
  38 percent above MTP off in every cell, where on the novel they ran 0.95 to
  1.01. The planner chose two drafts in 91 to 98 percent of head rounds on
  coding and 59 to 77 percent on planning and prose. The gain LILYADAPT-01 measured
  on prose belongs to text the model finds hard to predict, not to the
  model's own prose in a chat.
- MTP off decoded at 67 to 76 tokens per second; the default at 98 to 100 on
  coding, 80 to 81 on planning and 89 to 92 on prose. Request wall time
  against MTP off was 0.72 to 0.88 for both schedules.
- All 24 greedy requests gave one output hash across MTP off, two fixed drafts
  and the default. Default requests acknowledged the adaptive policy, lookup
  and `lily-q4-xl-lilymerge-01`; opt-out requests kept the plain
  acknowledgement. Zero reused input tokens.
- Every response ran to the 1,024-token ceiling, so no arm ended on its own
  and response length did not differ between arms.
- Limits: rendered inputs of 39 to 60 tokens with no system message (the
  template has no default one); single-turn requests without tools; requests
  written for this check, not the owner's earlier private request; host load
  averages of 4.8 at the start and 1.2 at the end, with arms interleaved
  inside each request.
- Public aggregate:
  [`lily-mtp-natural-responses-2026-11-20.json`](../benchmarks/data/lily-mtp-natural-responses-2026-11-20.json).
