# Lily small-row verification on natural long responses

Date: 2026-11-15, local time; execution receipts use UTC on 2026-11-16.
Attempts: `20261115-LILY-VERIFY-01` through `-04`.

## Decision

**The projection-reuse prototype does not meet the practical performance
gate.** Median paired useful decode improves 1.84% on coding and 4.54% on
planning; request latency falls 1.80% and 4.31%. Coding wins six of eight pairs,
with a gain inside the unchanged baseline's observed range. Planning wins all
eight pairs and both session halves, a modest positive result below the fixed
10% minimum. It should not be described as a 10% improvement or dismissed as
entirely noise.

The candidate is rejected for production promotion under this study's goal.
Its source and results are preserved. Two drafts remain the enabled Lily
recommendation; production code and the installed app are unchanged.

This follows the first-ranked investigation in the
[llama.cpp assessment](2026-11-15-llama-v1-flashnext-lily-assessment.md).
The experiment measures the same natural coding and product-planning requests
as the earlier long-response studies, with **two drafts in both arms**. Their
rendered inputs contain 24 and 20 tokens respectively; this tests long answers,
not long-context prefill. The previous draft-depth results remain unchanged.

## Identity and scope

| Field | Bound identity or policy |
| --- | --- |
| Base source | `2ac1b713d893f85b97111c1279465cfaa12d7bbe` |
| Baseline source relationship | `crates/`, `vendor/`, `Cargo.toml` and `Cargo.lock` match baseline worker source `cc8e47a824f802e24d442183ebbdd29fd25ecaaa` byte-for-byte |
| Artifact | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83` |
| Payload identity | Prior verified manifest `97c11dd6f7ee5a69e23759b9774acb21a786c48552ca2ede9dac8b6be4b5bc98`; 55 weight shards, 106,905,155,192 bytes |
| Identity recheck | Existing 63-file size/mtime/inode guard plus size/SHA checks for nine ancillary files, once before each run and after each session; no new full payload rehash |
| Backend | Actual generator `Forward::Lily` host exact-rejection MTP bridge; `batched-decode-exact-v1` |
| Precision | Source mixed Q4/Q8 weights, BF16 activations, F32 accumulation/recurrent state/readouts, Q8 KV; unchanged |
| Images and draft head | Image tower and supported MTP head loaded; require `qwen38-lily-retained-bf16-vision-v1` readiness |
| Host | Apple M5 Max, 128 GiB, macOS 26.7.2 / `25G332` |
| Build | Rust 1.98.1, locked release generator builds; separate detached worktrees and build targets |
| Dependency locks | Root Cargo SHA `f836a99ba1c6bd017fb829cd45f108fd97f9f1fb40c1497564898b3eac625f5f`; vendored Cargo SHA `34c73bedd511e540c8d0f38ab5c22a5704cf55acb8e14fc9dc2e2aa4d1aa813f` |

The baseline worker SHA-256 is
`61b9c8cc724407573169eb2abad7f908e4943eead7ac3122eeec3ef841a48640`.
The finer diagnostic worker is
`8682ec529e0d0be43c613e3fa11de48003ff014b283c1e35230be5d17b067dd1`,
with timestamp-only patch
`3b8214dd51c46b08588fdadaccc54a5c93249f99ae2e890d153f33197de26f4c`.
The uninstrumented projection candidate worker is
`096cceb6b370fea71b49523c928c68c1ac172ca77b63a968acf7ceb0a26dbe2e`,
with source patch
`9f406ddd07961d1d529622d13699cb119aacdbf9dbeba33fef0d6e4e5e4b0768`.

## Reached execution and diagnosed opportunity

The generation iterator calls `draft_host`, `verify_host_batched`, host target
sampling and `finish_host_verify`. The separate GPU-sampled speculative API
is not this request path. Verification batches the pending token with two
proposals, normally three rows, retaining ordinary target arithmetic.

The first source review distinguished output-channel packing from sharing
weights across input rows. Generic exact projections launch a row axis with
adjacent groups reading the same packed weights for different input rows.
Q4's existing two-output-channel specialization preserves a 16-lane reduction;
it does not by itself share packed weights between token rows. The fused GDN
input/convolution path already performs cross-input weight reuse and is
excluded from the new candidate. HC has a separate exact implementation and
also remains outside this change.

The first profile rejected a standalone vocabulary-head project: that stage
occupies only about 5–6% of measured decode time on these prompts. Eliminating
it cannot reach the 10% whole-request objective. A prepared head-only patch
is retained as unbuilt source; it has no correctness or performance result.

A second, timestamp-only diagnostic separates GDN input/convolution,
recurrence/norm and output projection, plus MoE input projections,
routing/shared activation, expert gate/up with shared down, and down/combine.
It reuses fixed numeric phase categories and preserves the original barriers,
command buffers, arithmetic and state operations. Inclusive parent categories
are separated from child categories in analysis to avoid double counting.
These timestamps perturb scheduling; uninstrumented runs decide performance.

The following values are medians of the two per-request phase/decode fractions
in the finer diagnostic, expressed as percentages. They are attribution
evidence under observed host load, not isolated kernel timings.

| Exclusive diagnostic stage | Coding decode share | Planning decode share |
| --- | ---: | ---: |
| GDN input projection/convolution, already sharing weights | 8.67% | 8.68% |
| GDN output projection | 5.98% | 5.80% |
| GDN recurrence/norm | 4.09% | 3.99% |
| HC | 16.29% | 15.76% |
| MoE input projections | 2.56% | 2.53% |
| MoE routing/shared activation | 2.69% | 2.59% |
| Expert gate/up with overlapping shared down | 16.45% | 16.47% |
| Expert down/combine | 11.07% | 11.05% |
| Attention including projections | 8.60% | 8.93% |
| Vocabulary head | 5.22% | 5.24% |

Residual parent intervals, embedding and PLE remain in the aggregate data;
this table is not a complete partition of request time. Host encode, sampling,
wait and transfer spans are not independently additive to these GPU spans.

The remaining generic projections include GDN output, attention QKV/indexer/
output, shared-expert projections, routing and the vocabulary head. Their
combined potential justified one bounded prototype. Some attention work is not
projection work, and shared down overlaps expert gate/up: neither entire span
can be credited as a guaranteed optimization budget.

## Candidate and numerical controls

The candidate shares packed codes and affine coefficients across two to four
input rows, retaining independent accumulation for each row:

- Q4/BF16 with even output width uses the existing packed 16-lane helper and
  the same BF16 output boundary.
- Q8/BF16 and Q8/F32 retain the ordinary dot expression, loop order, SIMD32
  reduction and requested output precision.
- Single rows, odd-width Q4 and Q4/F32 retain the existing fallback.

The private selector `STARK_LILY_VERIFY_PROJECTION_REUSE=1` is frozen once per
worker through `OnceLock`. It is enabled only for candidate workers. This
selector and its code exist only in the isolated experimental patch. The
production checkout contains neither the selector nor the new dispatch.
The implementation extends Stark's existing exact helpers; no CUDA code or
upstream speed percentage is transferred into a Stark performance claim.

One physical Metal component test passed **60 cases and 9,375,336 compared
bytes, with zero unequal bytes**, through the actual experimental public
dispatch against ordinary per-row GEMV. It covers widths two through four,
offset views and output canaries, stress patterns at small shapes, and random
inputs at actual GDN output, shared-expert, attention, indexer, router and
vocabulary dimensions. Small-shape patterns include zeros, subnormals and
cancellation. The test executable SHA is
`6d48aa64d3f5c15947521fc31aae2b2758476ff4ba9bd979e99176e70548191d`.
Other 151 library tests were filtered out; this is a finite component control,
not full-model distribution or lifecycle qualification.

## Fixed measurement plan

- Exact previous natural requests, supplied through private stdin; actual
  model template, default system message when present, thinking disabled.
  Prompt/rendered/token hashes must match across all sessions and the earlier
  natural-response records. Prompt and answer text never enter files.
- Context 8,192, coding ceiling 2,048, planning ceiling 4,096; native EOS,
  no added stop strings. Temperature 1, top-k 20, top-p 0.95, no min-p,
  seed 42; explicit `mtp_drafts: 2` in both arms.
- Image-loaded workers, fresh logical requests, prefix cache disabled and
  zero reported reused tokens. Tokenization, model loading and fixed
  64-token warmups are outside measured request timings.
- Coarse profile: four sessions off/stage/stage/off, one measured response
  and one warmup per prompt per session: eight measured and eight warmups.
- Detailed profile: two stage sessions with reversed prompt order, one
  measured response and one warmup per prompt: four measured and four warmups.
- Candidate comparison: four fresh sessions before/candidate/candidate/before,
  four measured responses and one warmup per prompt per session: **32 measured
  responses and eight warmups**, giving eight positional pairs per prompt.
  Prompt order reverses on alternate sessions. Pair sessions 1/2 and 4/3 by
  prompt and repetition; preserve every observation.
- No quiet admission, host changes, cooling, retries or timing exclusions.
  Host screens are descriptive only. Native workers execute serially; only
  this study's processes may be stopped. Per-request timeout is 900 seconds.
- Record complete request wall time, native prefill/decode, useful decode
  rate `(emitted_tokens - 1) / decode_seconds`, actual emitted/sampled counts,
  EOS, MTP work, hashes and sanitized host observations.
- A scheduling change must preserve seeded outputs, lengths and MTP work.
  The practical hurdle is the larger of **10%** and the unchanged baseline's
  full relative range for the metric. Require at least seven of eight paired
  wins per prompt, confirmation on both prompts and no greater than 5%
  regression. These are engineering screening rules, not confidence intervals.

## Matched performance results

All 32 measured requests and eight warmups complete and pass the fixed route,
identity, count, timing and zero-reuse checks. All measured responses finish
naturally at EOS. Coding emits 1,109 tokens plus the sampled EOS; planning
emits 1,958 plus EOS. Across all sessions, both arms have identical input,
output-token and decoded-text hashes, lengths and MTP work for each prompt.
All warmups also match within their separate 64-token budget.

Displayed absolute times/rates are medians of eight requests per arm and
prompt. Percent changes are medians of eight positional pair ratios, not
ratios of the displayed arm medians. Positive throughput change and negative
latency change favor the candidate.

| Metric | Coding before | Coding candidate | Planning before | Planning candidate |
| --- | ---: | ---: | ---: | ---: |
| Useful decode tokens/s, median | 95.69 | 98.10 | 79.32 | 82.61 |
| Useful decode tokens/s, range | 94.55–105.68 | 97.36–99.18 | 78.34–79.65 | 82.16–88.76 |
| Complete request seconds, median | 11.696 | 11.411 | 24.787 | 23.806 |
| Complete request seconds, range | 10.597–11.834 | 11.287–11.493 | 24.683–25.095 | 22.163–23.938 |
| Median paired useful-decode change | — | **+1.84%** | — | **+4.54%** |
| Median paired request-time change | — | **−1.80%** | — | **−4.31%** |
| Candidate faster pairs | — | 6/8 | — | 8/8 |
| Median paired seconds saved | — | 0.211 | — | 1.077 |

The two four-pair session halves show coding throughput changes of +0.60% and
+3.63%, and request-time reductions of 0.60% and 3.48%. Planning throughput
changes are +6.57% and +4.04%, with request-time reductions of 6.13% and 3.85%.
The baseline's full relative range is 11.64% for coding throughput and 10.57%
for coding wall time; the coding result fails both this hurdle and the required
seven-of-eight win count. Planning baseline ranges are 1.66% for rate and wall
time, so its screening hurdle stays at 10%. Its observed gain exceeds that
baseline range but misses the practical minimum in both session halves.

MTP work per complete response is identical across arms:

| Request | Rounds | Proposed | Accepted | Verified rows |
| --- | ---: | ---: | ---: | ---: |
| Coding | 394 | 788 | 717 | 1,182 |
| Planning | 821 | 1,642 | 1,138 | 2,463 |

All 32 measured requests have host observations. Descriptive host eligibility
is 13/16 before and 15/16 candidate; all four ineligible observations remain
in every statistic. External CPU maxima are 167.60% before and 195.73%
candidate; the per-request peak CPU medians are 113.92% and 88.12%. Maximum
thermal state is one, with zero measured pageout/swapout increments, compiler
flags or observer failures. GPU clocks, power and exclusive GPU use are not
established. The observations support stopping this candidate under the fixed
goal; they do not isolate a universal 4.54% kernel improvement.

## Attempt history and limits

| Attempt | Scope | Recorded state |
| --- | --- | --- |
| `VERIFY-01`, initial harness | Numeric `mtp` submitted where protocol expects a boolean | Worker rejected the request before generation; zero generated tokens. Source and sanitized failed receipt retained. Corrected to explicit `mtp_drafts: 2` in a separate run directory. |
| `VERIFY-01`, corrected screen | Coarse full-response attribution and uninstrumented controls | Completed 16 generation requests, including eight warmups. All eight paired checks, including warmups, match output/text hashes, lengths, verifier ID and MTP work. |
| Head-only prepared source | Share Q8 vocabulary-head loads | Rejected before build because measured ceiling is below the whole-request hurdle; no numerical or timing result. |
| `VERIFY-02` | Finer GDN/MoE timestamp attribution | Completed eight generation requests, including four warmups; source-bound aggregate profiles retained privately. |
| Projection candidate v1 | Initial selector used repeated environment lookups | Prepared but never built or run. Preserved separately; v2 freezes the selector once per worker. |
| `VERIFY-03` | v2 component exactness | One test passed: 60 cases, 9,375,336 bytes, zero differences. |
| `VERIFY-04` | v2 uninstrumented long-response comparison | Completed 32 measured responses plus eight warmups; exact repeated answers/work, +1.84%/+4.54% paired decode. Failed the practical performance gate; rejected for promotion. |

An initial diagnostic patch export used the configured visual diff renderer
and was not applyable. It was retained privately and replaced with a standard
unified patch before the diagnostic build. The valid patch reverse-checks
against its source worktree. No measurement is bound to the invalid export.

Final review also corrected the coarse screen analyzer's second-half labels:
session four is off and session three is stage. The initial analyzer had
reversed those labels in its symmetric hash/work equality checks. The
corrected analysis uses pairs 1/2 and 4/3; all eight checks still pass. Native
measurements, phase fractions and the separate performance comparison are
unchanged. The original analyzer, analysis and initial publication are
preserved with hashes; the public data records this label-only correction.

The candidate adds no inference buffers or retained model state. Prefix
snapshot layout, restore keys, rollback and image invalidation are unchanged
by source inspection. This study deliberately measures fresh requests. It
does not qualify prefix restoration, full-vocabulary target distribution,
image execution/transitions, cancellation/recovery, other artifacts or other
sampling/context settings. Further promotion controls were not run because
the candidate failed the performance gate. The archived implementation carries
no qualification for an additional state path. Answer quality and game
execution are not graded here; generated programs are never executed.

## Final verification

The three valid measurement runs contain 64 successful generation requests:
44 measured responses and 20 fixed warmups. The initial rejected protocol
request is separate. All ten valid worker sessions pass their post-run model
identity guards. Every worker has closed.

The repository policy and whitespace checks pass. Both executed source
patches apply cleanly to the recorded base. Archived Python sources parse,
all 16 archived source hashes and 23 private receipt hashes verify, and the
public JSON contains no raw request/output/vector/probability fields. All
208 previously tracked files under benchmark data and archived experiments
match their `2ac1b713` Git blobs byte-for-byte. Production Rust/vendor/host
sources also match that base; the new commit contains documentation, aggregate
evidence, archived experimental source and a scoped archive whitespace
attribute. The attribute preserves unified-patch blank context lines without
rewriting the sealed patch bytes.

The actual component test is the one finite native test described above.
This study does not claim a newly executed full engine suite or additional
model lifecycle qualification.

## Evidence and reproducibility

[Public aggregates](data/lily-small-row-verifier-2026-11-15.json) retain the
timings, counts, hashes, host observations, dispersion, pairing, scope and
private-receipt hashes. [Archived experimental sources](../archive/experiments/2026-11-15-lily-verifier/README.md)
retain the timestamp patch, prepared/unbuilt variants, tested candidate,
as-run harnesses and analysis/publication sources. Binaries and raw aggregate
native profiles stay under ignored `runtime/lily-verifier-study/`.

No prompt, response, token vector, probability or exception dump is saved.
No installation, release or PR is part of this study. The original benchmark
JSON and archived experiment sources remain byte-for-byte unchanged.
