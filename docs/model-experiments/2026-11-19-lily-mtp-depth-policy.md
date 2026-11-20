# 20261119-LILYDEPTH-01: Lily MTP adaptive depth and prompt-lookup drafts

Registered and executed 2026-11-19. Owner: depth-policy workstream (agent).
State: attempted. Kind: sampling scheduling and performance. Disposition:
default-off; merged into one uncommitted integration working tree (see
[Integration](#integration)); not committed or installed; final quiet timing
pending.

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | 20261119-LILYDEPTH-01 / 2026-11-19 / 2026-11-19 / depth-policy workstream |
| Kind | Sampling scheduling, performance |
| Attempt state | Completed (screening) |
| Result and acceptance criterion | Exactness gates passed; performance mixed (see [Results](#results)). Gates: default arms reproduce all eight baseline output hashes; adaptive greedy equals the baseline greedy hash; adaptive and lookup sampled output identical across repeats; exactness unit tests pass |
| Evidence scope | Operator/component (planner, index, sampler), end-to-end timing screening on a shared host |
| Promotion decision | Adopted default-off on the integration working tree (opt-in request options only). The planner's cost table predates the host round-trip change and was not retuned; retuning is a separate decision. Production promotion awaits the final quiet comparison |
| Disposition | Merged with 20261119-LILYROWS-02, 20261119-LILYHOST-01 and 20261119-LILYBENCH-01 into one uncommitted working tree; not committed, not installed, final quiet timing pending |
| Predecessor / related attempts / governing issue / PR | Baseline 20261119-SUSHI-02; benchmark harness 20261119-LILYBENCH-01; no PR |

## Hypothesis and change

- Problem: Stark fixes the MTP draft depth per request. Baseline 20261119-SUSHI-02
  shows the best depth depends on the regime (greedy code best at 2 to 3
  drafts, sampled flat at 1 to 2 and much worse at 3), and the benchmark
  workstream observed fixed depths 17 to 27 percent slower than MTP off on
  prose. Every draft also costs a 1.45 to 1.75 ms head step.
- Candidate: two opt-in request features, both written against the model's
  draft limit (`MAX_DRAFTS`, read at run time through `max_drafts()`), not the
  literal 3:
  1. Deterministic adaptive depth (`"mtp_policy": "adaptive"`).
  2. Prompt-lookup drafts (`"mtp_lookup": true`).
- Fixed before the run: default and fixed-policy requests must reproduce all
  eight SUSHI-02 output hashes; adaptive greedy must equal the greedy baseline
  hash (exact verification makes greedy output independent of drafts);
  adaptive and lookup sampled output must repeat bit for bit; the point-mass
  rejection must recover the target distribution in unit tests.

### Adaptive depth

`crates/stark-engine/src/engine/hybrid/mtp_policy.rs`.

- Expected tokens of a round with m drafts: E(m) = 1 + sum over k of the
  product a[0..k], where a[i] is the conditional acceptance of position i.
- Per round the planner picks m in 0..=cap (cap = min(limit, remaining - 1))
  maximizing E(m) / cost(m). Ties go to the smaller m.
- Estimates: exponential average with weight 1/8 (about the last eight
  observations of a position, roughly 20 output tokens at 2.3 to 2.8 tokens per
  round; exact in binary). Prior 0.85 for every position: with the cost table
  it opens at two drafts in both modes, the recommended fixed depth. Only
  positions that were actually verified are updated (positions after a
  rejection are unobserved). Every unobserved position is pulled toward the
  prior by 1/64 per round, so a depth the planner abandoned is re-tried after
  its estimate recovers instead of freezing (this also makes m = 0 recover
  when acceptance returns).
- Hysteresis: the incumbent depth is kept unless a challenger promises more
  than 5 percent more tokens per millisecond. A round capped by the remaining
  budget does not replace the incumbent.
- Cost table: one `CostTable` constant, `LILY_Q4_XL_SUSHI_02`, naming attempt
  20261119-SUSHI-02 as its source. Head round cost = m draft steps + verify(m+1
  rows) + finish(m) + host(m); it reproduces the measured 1/2/3-draft rounds
  within 0.1 ms. The zero-draft cost `serial_ms` (15.10 ms greedy, 14.20 ms
  sampled: the MTP-off step plus the lost encode overlap, readout and host
  draw) and `head_advance_ms` (1.4 / 1.2 ms, about 0.8 of a draft step,
  from this attempt's screening) are estimates. Rows past the table extrapolate by the last measured
  slope, so a later limit raise needs no policy change, only new constants.
- Determinism: the plan is a pure function of acceptance history, remaining
  budget, sampling mode and the static table. Measured round times are
  reported (`head_round_seconds_by_depth`) but never read by the planner.
- m = 0 inside an MTP request (floor 0): one ordinary decode step of the
  pending token (`MtpDecode::serial_round`), a complete F32 readout, and an
  exact target draw with no proposal. With MTP enabled the ordinary decode
  graph already runs the head's cache-only catch-up for its row and leaves the
  head hidden at that row, which is exactly the head state after a verify
  round that accepted nothing; no verify captures, finish pass or head
  advance are needed. Compared with plain MTP-off decode it loses only the
  overlap of host encoding with the previous GPU pass (about 0.66 ms) and adds
  the readout and host draw (about 0.2 ms sampled, 0.5 ms greedy, where
  MTP-off greedy draws on the GPU). The first version used a one-row verify
  plus a head advance plus the finish pass instead; it was replaced before any
  timing because that path costs about 2 to 3 ms more per token (see the
  attempt history).
- After benchmark-workstream evidence that fixed depths lose 17 to 27 percent
  against MTP off on prose, a table-driven test pins the settling behaviour:
  with per-position acceptance 0.3 the plan reaches m = 0 within 12 rounds and
  spends at least 110 of the next 130 rounds there (the rest are probes), and
  after acceptance returns to 0.95 it drafts two or more for at least 140 of
  the last 150 rounds.

### Prompt lookup

`crates/stark-engine/src/engine/hybrid/mtp_lookup.rs`.

- Index: hash map from every 4-gram of prompt plus committed output to the
  position after its latest earlier occurrence; updated with each round's
  committed tokens. The current tail never matches itself.
- Gate: at least 8 tokens of backward agreement when the agreeing span
  contains a line break, otherwise 32; proposes the tokens that followed the
  match, up to the round cap. Proposals may run into the copy that follows.
- Line breaks: the worker decodes each vocabulary id alone through the loaded
  tokenizer once per loaded model (on the first lookup request) and keeps a
  bitset of ids whose text contains a line feed. Tokenizer metadata only; no
  request data, nothing logged.
- Privacy and lifetime: the token ids live only in the index owned by the
  decode stream and are dropped with it on completion, failure, cancellation
  and early stop. The type does not implement `Debug`.
- Sampling: a lookup proposal is a point mass (q = 1 at x). New additive trait
  method `MtpSampler::draft_point_mass` (default refuses; same name,
  signature and default error as the host-roundtrip workstream's method for
  device greedy drafts) and its implementation in `ExactMtpSampler` push a
  point-mass proposal; the unchanged `target` path
  then accepts with min(1, p(x)) and draws a rejection from max(p - q, 0),
  which is p with x removed and renormalized. Greedy reduces to an argmax
  compare. No draft randomness is consumed.
- Choice of round: under the fixed policy a lookup round runs whenever the
  gate yields a proposal (up to the request's draft count). Under the
  adaptive policy the lookup round runs when its best depth's expected tokens
  per millisecond (separate lookup acceptance estimates, same prior, weight
  and decay; cost = head round without draft steps plus one head advance) is
  at least the head plan's.

### Head state after rounds without head drafts

`draft_host`'s first step is the only writer of the draft head's cache row for
the pending token (sequence index pos - 1); `finish_host_verify` writes the
committed rows after it. A round that skips `draft_host` would leave that row
holding whatever an earlier round or an earlier request wrote, which would
make later head drafts, and so sampled output, depend on previous requests.
A new vendored function, `Qwen4ExpModel::advance_head_host`
(`vendor/lily-core/src/qwen4exp/host_spec.rs`), writes that row with
`mtp_block_caches` (the cache-only head block prefill already uses for the
same purpose; made `pub(super)` in `model.rs`), without attention read,
vocabulary readout or sampling. Lookup rounds before the last round call it
before verification; afterwards the head's caches are
as after a head round with the same accepted count. `draft_host`,
`finish_host_verify` and the verify graph are unchanged.

### Interface (proposed protocol text, not yet in the reference pages)

- `generate` accepts `"mtp_policy": "fixed" | "adaptive"` (default `"fixed"`)
  and `"mtp_lookup": true | false` (default false). Explicit null is invalid,
  as for the other MTP selectors.
- Either non-default selector needs MTP enabled (`mtp: true`, `mtp_drafts > 0`
  or a non-zero model default) and a backend that supports scheduling; otherwise
  `invalid_generation_request` (MTP off) or `mtp_not_supported`. An unknown
  policy is `invalid_generation_request`.
- `mtp_drafts` is the per-round limit under the adaptive policy. With
  `"mtp": true` (or the model default) and the adaptive policy the limit is
  the model's `max_drafts`.
- Readiness: on a backend that supports scheduling, the `mtp` object gains
  `"policies": ["fixed", "adaptive"]`, `"default_policy": "fixed"` and
  `"lookup": true`. Other backends are unchanged.
- Acknowledgement: only for a request that selected a non-default option, the
  `mtp` object gains `"policy"` and `"lookup"`, and `stats.schedule` reports
  `policy`, `lookup`, `limit`, `head_rounds_by_depth` and
  `head_round_seconds_by_depth` (index = drafts, length limit + 1),
  `lookup_rounds`, `lookup_proposed`, `lookup_accepted`, `lookup_round_seconds`,
  `head_advances` and `head_advance_seconds`. `rounds`, `proposed`, `accepted`
  and `verified_rows` include lookup rounds. A default request's
  acknowledgement is byte-for-byte the existing schema.
- Image requests keep MTP off; the selectors are not accepted there.
- Prefix cache: the key stays `model-instance+context-tokens+cache-scope+mtp-enabled`.
  The retained snapshot is captured during prompt prefill, before any round
  runs, so neither selector changes it. The committed trunk state is
  bit-identical to serial decode for any proposal source, and the head cache
  rows for committed positions are written by the same kernels whatever the
  round kind, so the selectors need no key component. As with `mtp_drafts`,
  the head's batching (and so head numerics) can differ between scheduling
  choices; this affects only proposals, never exactness.

## Artifact, source and environment

| Field | Baseline | Candidate |
| --- | --- | --- |
| Model | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, installed copy, image tower loaded | Same |
| Stark source | `4d09b55045f24eab81da0a9b5b024b9c02318459` | Same base plus this workstream's uncommitted diff (checkpoint `runtime/patches/depth-policy.patch`) |
| Worker SHA-256 | `5dfb90b61a459560702590703117b5c705e2ae425b7d5da2d5a709bcc766298d` (SUSHI-02) | Final `b570b6db2e3e00038e308e7227559c0b025a787930194de1855e665ca1bd2532` (v4); earlier builds in the attempt history |
| Toolchains | Rust 1.98.1 release, locked | Rust 1.98.1 release, locked; unit tests nightly-2026-10-01 for the engine |
| Precision | Unchanged | Unchanged |
| Hardware | Apple M5 Max, 128 GB | Same, shared with other workstreams (screening only) |

Vendored-code patches (Stark-local, for provenance integration):

- `vendor/lily-core/src/qwen4exp/host_spec.rs`: new `advance_head_host`.
- `vendor/lily-core/src/qwen4exp/model.rs`: `mtp_block_caches` visibility
  `fn` to `pub(super)`.

No Sushi code was copied. The planner and lookup gate reimplement the ideas
described in the shared brief (expected-value planner over per-position
acceptance; 4-gram index with 8/32-token agreement); the constants were chosen
and justified here.

## Results

All timing is screening on a host shared with other workstreams, one
process at a time through the model slot; the orchestrator owns the quiet
comparison. The cost table's measured round costs predate the host-roundtrip
workstream's draft/verify sequencing change (+8.6 percent sampled, +12.9
percent greedy at two drafts in its screening); the table was not updated
for it.

Run files are under `runtime/agent-runs/` of the coordinating worktree
(SHA-256 prefixes in brackets). Runners: `depth_policy_profile.py`
(adapted from SUSHI-02's `run_profile.py`, arm grammar `<g|s><drafts>[a][l]`,
`--copy-tokens`), `depth_policy_fixed_corpus.py` (a copy of the
LILYBENCH-01 harness, SHA-256 prefix `25bf5c03` before the edit, with one
change: it keeps the `schedule` counters, `policy` and `lookup` from the
acknowledgement), summaries `depth_policy_summary.py` and
`fixed_corpus_summary.py`. Prose is the Sushi fixture
`prose_pride_and_prejudice.txt` (SHA-256 `0a0ead8d...05e63f0`).

### Attempt history

| Trial | What was attempted | Observation | Disposition |
| --- | --- | --- | --- |
| v1 build `2b65dcf9`, run 1 [`9b272db4`] | Default arms gate | 8/8 hashes and work counters equal | Pass; timing invalid (host contention, s0 -16 percent with no code change) |
| v1, run 2 [`0446a1eb`] | Adaptive, lookup, frozen corpus | Greedy adaptive hash equal; sampled adaptive one hash per arm; head advance 1.9 ms on a loaded host | Superseded |
| v1 design | m = 0 as one-row verify plus head advance plus finish | Estimated 2 to 3 ms per token above serial | Replaced by `serial_round` before any m = 0 timing |
| v2 build `bcda0a0d`, runs 6 to 9 | Same arms after the serial change | Killed with the agent during the owner's pause; no output kept | Invalid, rerun as 10 and 15 to 18 |
| v2, run 10 [`b7f62487`] | Gate plus adaptive and lookup arms, quiet slot | 8/8 hashes equal; head advance 0.36 to 0.38 ms; sampled adaptive fell to serial for 39 percent of rounds at 15.2 to 15.9 ms each and ran 70.6 tok/s against 78.1 to 78.7 fixed | Planner rejected: serial cost too optimistic, recovery too slow, probes jumped to depth 2 |
| v3 build `90002a3c` | Measured `serial_ms`, `head_advance_ms`; decay 1/48; depth rises at most one step per round; polled wait in the serial step | Unit tests updated | Measured in runs 15 to 18 |
| run 11 to 14 | Chain script word-splitting error | No model run | Invalid, rerun as 15 to 18 |
| v3, runs 15 to 18 [`d5618d33`, `6b84160a`, `4254ac04`, `959161cd`] | Frozen corpus, copy-heavy, harness fixed and adaptive | See below; greedy serial rounds measured 13.7 ms | Greedy `serial_ms` lowered to 14.0 |
| v4 build `b570b6db` (final), runs 19 to 21 [`f0fe34aa`, `7f1ff201`, `fc1b2a0b`] | Final gate, harness adaptive and paired fixed | See below | Current |

v4 also renamed the sampler method to `draft_point_mass` (no behaviour
change; run 19 reproduces v3's sampled adaptive hash).

### Exactness

- Default and fixed arms, frozen corpus, final build (run 19): all eight
  SUSHI-02 output hashes and work counters equal. Also equal in runs 1 and 10.
- Adaptive greedy (g3a, runs 2, 10, 15, 19): hash `cdb7b6df...`, the baseline
  greedy hash. Greedy adaptive plus lookup (g3al): same hash.
- Fixed-corpus harness, greedy, final build: adaptive hashes equal the fixed
  and off hashes in all four cells (code/prose x 4k/8k), including 4k prose
  where 246 of 309 rounds were serial steps (run 20 against run 21).
- Copy-heavy greedy (run 16, v3): g0, g2, g3l, g3al all hash `4e2c14e8...`
  with 214 to 228 lookup rounds per pair of requests.
- Sampled: each adaptive or lookup arm produced one hash across its repeats
  in every run (s3a `db93e82f...` identical in v3 run 15 and v4 run 19).
  Sampled MTP output legitimately differs from fixed depths (the draft
  distributions and random draws differ), as for any depth change.
- Unit tests: stark-generate 150 passed (including 4 new point-mass tests and
  the protocol test); stark-engine `mtp_` filter 30 passed (10 planner, 6
  index tests new). `cargo fmt --check` clean. The real-model test
  `private_checkpoint_mtp_verify_and_rollback_match_serial` was not run: this
  workstream changed no verify, rollback or kernel path.

### Performance (greedy unless noted, tok/s, medians)

Frozen corpus, 4,096 input tokens, final build run 19 (2 repeats; the fixed
arms moved -11 to +8 percent against SUSHI-02 on this shared host):

| Arm | tok/s | Rounds by depth (0/1/2/3, two requests) |
| --- | ---: | --- |
| g2 fixed | 85.7 | |
| g3 fixed | 92.9 | |
| g3a adaptive | 95.4 | 2 / 0 / 366 / 0 (stays at 2; 0 is the final token) |
| s1 / s2 / s0 fixed | 69.3 / 68.3 / 77.1 | |
| s3a adaptive | 81.4 | 2 / 136 / 122 / 0 |
| s3al adaptive + lookup | 81.9 | same plan, no lookup rounds fired |

Run 15 (v3, quieter): g2 92.2, g3a 91.5; s0 77.6, s1 78.9, s2 78.9, s3a 75.8.
Sampled adaptive is within noise of the best fixed depth, between -4 and +17
percent depending on the run; greedy adaptive equals two drafts.

Fixed-corpus harness (LILYBENCH-01 ladder, 3 samples per cell, code 256 and
prose 128 decoded tokens), final build, run 20 adaptive (limit 3) against
paired run 21 fixed, with run 17 (v3, fixed depth 3) for depth 3:

| Cell | off | 1 | 2 | 3 (run 17) | adaptive | Adaptive plan (0/1/2/3) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| code 4k | 79.5 | 84.1 | 94.3 | 93.5 | 101.9 | 0 / 3 / 264 / 0 |
| code 8k | 76.7 | 82.5 | 91.7 | 92.6 | 91.9 | 0 / 3 / 270 / 0 |
| prose 4k | 77.0 | 71.6 | 66.9 | 54.2 | 71.1 | 246 / 0 / 63 / 0 |
| prose 8k | 76.1 | 82.5 | 79.9 | 68.0 | 78.0 | 0 / 42 / 123 / 0 |

- Code: adaptive matches the best fixed depth (the 4k figure is above fixed
  two drafts with the same plan, which is noise or ordering, not a gain).
- Prose 4k: adaptive avoids the fixed-depth losses (+6 percent over two
  drafts, +31 percent over three) and settles at serial steps (13.96 ms per
  round), but stays 8 percent below MTP off. Causes: every request opens at
  two drafts from the prior and spends about 21 rounds there before
  settling (63 of 309 rounds), and a serial step inside MTP costs about
  1 ms more than MTP-off decode (no encode overlap; the MTP decode graph also
  runs the head catch-up).
- Prose 8k: acceptance is high on this window (0.95 at one draft); adaptive
  mixes 1 and 2 drafts and lands between them (78.0, one slow sample at 66.8).
- Head advance (lookup rounds): 0.36 to 0.38 ms quiet (run 10), 0.66 to
  0.69 ms loaded (run 16).

Prompt lookup, copy-heavy input (first 512 frozen-corpus tokens repeated to
4,096), v3 run 16 (loaded host, wide spread):

| Arm | tok/s | Lookup rounds | Lookup proposed / accepted |
| --- | ---: | ---: | --- |
| g0 off | 75.3 | | |
| g2 fixed head | 92.7 | | |
| g3l fixed + lookup | 107.4 | 214 | 642 / 638 |
| g3al adaptive + lookup | 106.0 | 228 | 680 / 668 |
| s0 / s2 | 68.3 / 80.4 | | |
| s3l / s3al | 71.0 / 67.4 | 2 / 6 | 6/4, 12/2 |

Greedy copying is where lookup pays: 98 percent of lookup drafts accepted,
about 80 percent of rounds were lookup rounds, +15 percent over two head
drafts. At temperature 1 the sampled continuation leaves the copy almost
immediately and the gate rarely fires; sampled lookup is neutral to
negative in this screening. On the frozen corpus (no exact copies) lookup
fired 8 to 10 times per two requests and changed nothing measurable.

## Plan for a draft limit of 8

Prerequisites:

- GDN rollback that no longer captures a full F32 state per verified row
  (113,246,208 bytes per extra row across 36 layers at the base revision, so
  8 drafts would hold about 0.8 GB of mid states). The GDN workstream's shadow
  state plus replay removes this blocker in its worktree.
- The fused GDN input projection kernel exists only for 1 to 4 rows and its
  cost climbs about 0.8 ms per row at rows 3 and 4 (GDN workstream). Rows 5
  to 9 need new variants or a grouped dispatch with byte tests at every m;
  this is now the main blocker.
- Expected payoff is small under the current exactness contract: the kernel
  workstreams found extra verify rows cannot be made much cheaper (expert
  sharing byte-exact but not faster, hyper-connection change tiny, GDN replay
  2 to 3 percent). At the baseline row slope (5.5 to 7 ms per row) a fourth
  head draft already does not pay; deeper limits mainly help prompt-lookup
  rounds on long copied spans, which pay no draft steps.

Constants, buffers and guards that change:

- `vendor/lily-core/src/qwen4exp/model.rs`: `MAX_DRAFTS` 3 to 8;
  `GATED_PREFILL_ROWS = MAX_DRAFTS + 1` (9 rows of gated expert reads);
  the `MAX_DRAFTS < HC_FUSED_MAX_ROWS` assertion (`kernels/hc.rs`,
  `HC_FUSED_MAX_ROWS = 4`) must become a grouped loop over rows of at most 4,
  or the HC fused kernels gain m up to 9 variants; `SpecScratch`: `mid`
  (removed or resized by the rollback rework), `conv_in`, `ple_conv_in`,
  `logits` (`[MAX_DRAFTS + 1, vocab]` F32), `verify_tokens`, `draft_tokens`,
  `dists`, `ctrl` (`ctrl_words(MAX_DRAFTS)`), `mtp_ids`.
- Row guards: `model.rs` (`rows <= MAX_DRAFTS + 1` near 2041 and the verify
  batch check near 2482), `spec.rs` (`(1..=MAX_DRAFTS + 1).contains(&m)` and
  `rows <= MAX_DRAFTS + 1`), `host_spec.rs` (`drafts.len() <= MAX_DRAFTS`,
  `(1..=MAX_DRAFTS).contains(&drafts)`).
- Kernels: the projection reuse kernels exist for m = 2, 3, 4 only
  (`stark_projection_reuse_q4_bf16_m{2,3,4}`, `..._q8_f32_m*`); rows 5 to 9
  need new variants or a grouping pass with the same per-row lane order. HC
  `hc_read_down_q8_m*` and `hc_read_up_mix_q8_g*_m*` likewise stop at m = 4.
  The GDN verify kernel's sequential rows and attention verify rows need the
  new upper bound in their dispatch checks and byte tests at every new m.
- `request_memory.rs` accounting (the `MAX_DRAFTS` and `MAX_DRAFTS + 1` terms)
  follows the constant automatically; re-qualify the planned context limits.
- `lily.rs` `mtp_capabilities` advertises the new limit; keep
  `recommended_drafts` at 2 for the fixed policy.

Policy changes:

- None structurally: the planner, lookup and counters already take the limit
  at run time; `DEPTH_SLOTS = 9` covers depth 8.
- Replace `CostTable::LILY_Q4_XL_SUSHI_02` with a table measured at 1 to 9
  verify rows (the extrapolation by the last slope is only a placeholder).
- Deep positions: with prior 0.85 at every position, deep depths only win when
  observed acceptance is high; consider a prior that decreases with position
  and a smaller hysteresis band once rows are cheap.
- Lookup benefits most: lookup rounds pay no draft steps, so long copied runs
  can verify up to 8 proposals for one head advance.
- Draft steps chained on the GPU without host reads would cut the per-step
  cost that dominates deep head rounds.

## Evidence, decision and limits

- Worked: both features are exact, deterministic and default-off; default
  and fixed requests reproduce SUSHI-02 exactly. Adaptive tracks the best
  fixed depth on code and avoids the large fixed-depth losses on prose.
  Greedy prompt lookup gains about 15 percent on copied text.
- Did not work yet: adaptive does not beat MTP off on 4k prose (71.1 against
  77.0); sampled lookup is not useful at temperature 1.
- Scope: Lily Q4-XL `0f94842e`, image tower loaded, the inputs above, greedy
  and temperature 1 / top-k 20 / top-p 0.95. Not qualified for other
  artifacts, image requests (MTP stays off there) or prefix-cache hits.
- Decision: keep default-off, awaiting the orchestrator's quiet comparison.
  Do not change the default policy on this evidence.
- Next steps: (1) open each request at a lower depth (prior below 0.85, or a
  first-round plan of one draft) so prose settles sooner; the one-step rise
  limit then costs code only a few rounds; (2) make the serial step overlap
  host encoding like `SampledDecode` and `GreedyDecode` (would close most of
  the 1 ms gap to MTP off); (3) replace the cost table after the
  host-roundtrip sequencing change lands; (4) a lookup round could append
  head drafts after a short lookup proposal.

Prefix reuse: unaffected (see Interface). Lifecycle: the scheduler, its
lookup tokens and counters live in the decode stream and drop with it on
completion, error, cancellation and early stop; `reset` clears the counters.
Nothing is logged; counters only.

Keep prompts, completions, token vectors, probabilities, credentials, exception
dumps and model/private evaluation files out of this document.

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

## Follow-up 2026-11-19: merged-build cost table and planner tuning (20261119-LILYMERGE-01)

Scope: `crates/stark-engine/src/engine/hybrid/mtp_policy.rs` and two edits in
`lily.rs` (the cost-table constant name in `mtp_decode_with`, and one new
ignored real-model test). `MtpDecode` behaviour is unchanged. Checkpoint:
`runtime/patches/merge-costs.patch` (only this change, against the integrated
tree). Final worker SHA-256
`5abca56f752489dde82d60b42e29ddaf36542962a83770bfba27a52633a3e26b`
(Rust 1.98.1 release build, uncommitted tree).

### Cost table

`CostTable::LILY_Q4_XL_SUSHI_02` is replaced by
`CostTable::LILY_Q4_XL_LILYMERGE_01`. The draft phase is now a table by drafts
(`draft_ms`), because greedy drafts chain on the device and each extra greedy
draft costs about 1.02 ms against 1.33 ms sampled; a lookup round also pays
`lookup_verify_extra_ms` because its verify is not encoded ahead. The table
stays keyed on the request's sampling mode only.

| Field (greedy / sampled) | Value | Source |
| --- | --- | --- |
| Round at 1/2/3 drafts | 19.92/26.17/32.96; 19.72/26.05/33.17 (reproduced within 0.02 ms) | `runtime/quiet-ab/q1-summary.json`, quiet host, worker `d4182a67` |
| `verify_ms` (2/3/4 rows) | 17.83/22.86/28.44; 17.44/22.22/27.70 | same |
| `draft_ms` (1/2/3 drafts) | 1.27/2.29/3.31; 1.36/2.67/4.00 | same |
| `finish_ms` (1/2/3) | 0.75/0.92/1.09; 0.74/0.93/1.21 | same |
| `host_ms` (round less verify, draft, finish) | 0.07/0.10/0.13; 0.18/0.24/0.26 | same |
| `serial_ms` | 13.75; 13.61 | measured for this attempt: `runtime/agent-runs/merge-costs-m1-prose.json`, 896 greedy and 104 sampled zero-draft rounds in the middle of streams on 4k prose (MTP off: 13.23; 13.06) |
| `head_advance_ms` | 0.40 | `runtime/quiet-ab/q1-6-cand-copy.json` (0.404 to 0.406 ms) |
| `lookup_verify_extra_ms` | 0.44; 0.75 | same file: greedy lookup rounds of 2 and 3 proposals 24.71 and 30.50 ms, sampled 24.54 ms at 2 |

The previous values (attempt 20261119-SUSHI-02 base build, plus serial and
head-advance estimates from this record) were, greedy / sampled: serial 14.00
/ 15.60; verify 16.16, 21.12, 25.84, 31.37 / 14.99, 19.95, 25.39, 32.48 (extra
row 5.53 / 7.09); draft step 1.77 / 1.46; finish 0.60, 0.76, 0.92, 1.67 /
0.55, 0.72, 0.88, 1.11 (extra 0.75 / 0.23); host 0.40, 0.72, 1.02, 1.26 /
0.12, 0.16, 0.20, 0.22 (extra 0.24 / 0.02); head advance 0.38 / 0.38.

### Planner changes beyond constants

Still a pure function of acceptance history, position, remaining budget,
sampling mode and static constants; no wall-clock input.

- Hysteresis 5 to 2 percent: with the new costs three drafts beat two on code
  by about 4 percent, inside the old band, so adaptive stayed at two.
- Opening (weakness 1): each position's n-th observation now has weight
  `max(1/(n + 2), 1/16)` (the prior counts as two observations), so low
  acceptance moves the estimate after a few rounds. A unit test pins the
  prose profile (0.67, 0.51, 0.09) leaving the opening two drafts by round 6
  (simulated: round 3) instead of about 11 to 21.
- Steady weight 1/8 to 1/16, which keeps rejection runs on prose from
  dropping to serial steps; the first position's pull toward the prior while
  unobserved is 1/48 to 1/96, so probes out of serial steps are rarer.
- Later unobserved positions now drift (1/16 per round) toward the position
  before them, not toward the 0.85 prior, so a deeper plan is tried once the
  shallower positions show it would pay.
- Prompt-lookup estimates start at 0.95 (`LOOKUP_PRIOR`) instead of 0.85. At
  0.85 a lookup round could never beat a three-draft head plan on code, so it
  was never tried and never learned; in the screening build without this
  change, adaptive plus lookup was only 1 percent above adaptive on code.
- Serial step (weakness 2): not fixed in `MtpDecode`. Overlapping its host
  encoding needs the next round's plan before the current pass finishes, an
  encoded pass kept across rounds, and drop handling; not a small, contained
  change. The measured cost inside MTP is now 0.5 ms above MTP-off decode
  (not about 1 ms), and the table charges it as measured.
- Test-only `RoundScheduler::scripted` (cfg(test)) runs given head depths
  before the policy.

Candidates screened with a deterministic Bernoulli simulator and rejected:
opening at one draft (no gain), weight 1/16 with the 1/48 pull (more probes
on low acceptance), uniform 1/96 pull for every position (recovery after
prose took about 90 rounds; unit tests failed).

### Gates

| Gate | Result |
| --- | --- |
| stark-engine `mtp_` filter (nightly-2026-10-01) | 31 passed, 9 ignored |
| `cargo fmt --check`, workspace crates | Clean |
| Eight default arms (`runtime/agent-runs/merge-g4-default.json`) | All eight output hashes equal 20261119-SUSHI-02 |
| Greedy adaptive and adaptive plus lookup, frozen corpus, two repeats (`merge-g4-options.json`) | Equal the baseline greedy hash |
| Sampled adaptive and adaptive plus lookup, two repeats | One hash each across repeats |
| Fixed-corpus greedy outputs, all arms | Code and prose hashes equal across off, fixed and adaptive |
| `private_checkpoint_mtp_zero_draft_round_mid_stream_matches_serial` (new, real model, under the slot) | Passed: 17 rounds, depths 0/1/2/3 = 6/5/3/3, 20 proposed, 7 accepted; 24 outputs equal ordinary greedy decode; continuation logits 248,320 values, 0 unequal |
| `private_checkpoint_mtp_verify_and_rollback_match_serial` | Not run: `MtpDecode` unchanged |

The new test uses a scripted schedule (2, 0, 3, 0, 0, 2, 1, 0, 3, 0, 2, 0, 3
drafts) and a host draft sampler that proposes a wrong token at chosen
positions, so draft rounds right after serial rounds verify and roll back
partially.

### Fixed-corpus benchmark, 4k rung, greedy, tok/s medians of 3

Final worker, same session (`runtime/agent-runs/merge-g4-bench-*.json`; the
host had only this chain running, but the session ran about 6 percent below
the `b1` session on every arm):

| Arm | off | 1 | 2 | 3 | adaptive | adaptive + lookup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code (512 tokens) | 77.1 | 97.3 | 105.7 | 109.1 | 108.4 (-0.6 percent) | 113.5 (+4.0 percent) |
| prose (128 tokens) | 74.7 | 82.5 | 74.4 | 58.9 | 80.6 (-2.3 percent) | 83.0 (+0.6 percent) |

Percentages are against the best fixed depth in the same session. Earlier
screening sessions of intermediate builds: `merge-g2` (weight 1/8, lookup
prior 0.85): code 118.7 against 116.5 fixed, prose 84.4 against 86.9, lookup
code 124.6; `merge-g3` (this planner without the lookup prior): code 109.2
against 111.2, prose 82.2 against 82.9, lookup code 110.3. Prose adaptive runs
80 rounds with 73 proposals against 76 rounds at fixed one draft: it opens at
two drafts for a few rounds and takes a few serial steps. Sample-to-sample
spread is 1 to 3 percent.

Decision: adopt the table and planner changes for the opt-in policy. Targets:
code within 2 percent (met in the final session); prose within 2 percent
(missed by 0.3 points in the final session, met in `merge-g3`); adaptive plus
lookup on code at or above 122.6 (not met in absolute terms in a slower
session; 4.0 percent above the best fixed depth, against 5.8 percent in `b1`).
The orchestrator's quiet comparison should settle prose and lookup. Defaults
are unchanged.
