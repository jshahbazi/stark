# Flash Next Lily one-draft recommendation study

Date: 2026-11-15. Attempts: `20261115-LILY-MTP1-01` and `-02`.

## Current reference update

This report remains the historical record of its as-run candidate and
one-versus-two reconciliation. The general scalar recommendation remains two.
A separate follow-up implements one automatic draft only for the exact pinned
Lily receipt with stochastic temperature 1, top-k 20, top-p 0.95 and absent/null
min-p. Its branch qualification now passes: the held-out top-k endpoint
reduced median request latency by 14.66% with 8/8 paired blocks favoring one,
and seven image-loaded prefix checks matched fresh runs across one-to-two and
two-to-one transitions. This narrower selector does not change the recorded
historical results or qualify other samplers. No installation or release is
claimed. See the
[sampling-specific follow-up](2026-11-15-lily-sampling-mtp-policy.md) and
[aggregates](data/lily-sampling-mtp-policy-2026-11-15.json).

## Result

This report preserves the as-run one-draft candidate and its measurements. The
candidate changed the enabled recommendation from two drafts to one for the
pinned Lily Q4-XL source; it was later reverted after the preregistered
three-depth reconciliation failed its performance gate. Two drafts remain
recommended. Explicit one-draft requests, off/count controls, the raw omitted
default and image behavior are unchanged. Functional and prefix checks pass.
No app is installed or released by this work.

The new within-worker reconciliation shows a workload-specific benefit and a
failure to qualify one draft as the artifact-wide recommendation. In the
primary 512-token top-k case, one versus two improved decode rate by 4.25% and
reduced median request latency by 3.58%, with 12/12 latency wins; the latency
gain missed the preregistered 5% threshold. The held-out top-k cell favored one
by 20.03% decode and 14.45% request latency, also with 12/12 wins. However,
the 4,096-token nucleus guard regressed decode rate by 6.01%, so the artifact-wide
gate failed. See the
[three-depth reconciliation](2026-11-15-lily-mtp-default-reconciliation.md)
and [its aggregate data](data/lily-mtp-default-reconciliation-2026-11-15.json).
The earlier same-worker depth study and actual before/candidate ABBA results
remain distinct historical comparisons; none is a universal speed claim.

The implementation was a small, reversible recommendation change. Its
functional checks passed, but the later reconciliation did not meet the
preregistered performance threshold. The candidate was reverted; measurements
and validation records remain available for review. See the
[three-depth reconciliation](2026-11-15-lily-mtp-default-reconciliation.md).

## Initial problem and proposal (superseded)

The initial decision treated the measured 9.93% one-draft gain over MTP off as
worth implementing and superseded the earlier 10% screening cutoff. The later
one-versus-two reconciliation tested the Desktop default comparison directly
and did not pass its separate preregistered criteria. The 9.93% off-control
result remains valid for its original comparison but does not establish a gain
over two drafts.

The source audit also found an important product distinction: the raw worker
defaults to MTP off, but Desktop already enables MTP for supported Lily models
unless the model's saved preference disables it. Enabled Desktop/API requests
select the loaded worker's recommended depth, currently two. Therefore the
actual product comparison is one draft versus the current recommendation of
two. The earlier off/on measurements alone do not establish that product gain.

Proposed smallest change: recommend one draft for the exact pinned Lily Q4-XL
artifact. Keep explicit off, explicit native draft counts, other artifacts and
raw-worker omitted-option behavior. Image requests continue with zero drafts.
Do not add a new adaptive sampler or kernel implementation.

## Preregistered comparison and controls

- Exact artifact and payload identity: same as the
  [release assessment baseline](2026-11-15-llama-v1-flashnext-lily-assessment.md),
  `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` revision
  `0f94842e3682b49acace4daeb4cf2e65a6f94f83`; retain image tower and MTP head.
- Before engine: `cc8e47a824f802e24d442183ebbdd29fd25ecaaa`, sealed executable
  SHA-256 `61b9c8cc724407573169eb2abad7f908e4943eead7ac3122eeec3ef841a48640`.
  Follow-up branch starts at study commit `f9c9720c`; engine code is identical.
- Measure existing explicit depths one and two within the same loaded before
  worker, in eight ABBA pairs per cell, one excluded warmup per depth.
- Six main cells: 512/4,096 input, 256 output; greedy, temperature-1/top-k-20/
  top-p-0.95, and temperature-1/nucleus/top-p-0.95, seed 42.
- Additional short interactive control: 32 input/64 output, greedy; held-out
  control: 512 input/128 output, top-k-20, seed 7 and a different source corpus.
- Freeze both corpora and retain their hashes. Record actual input/output work,
  image-loaded readiness, draft activity, repeatability, MTP timings and zero
  prefix reuse. Keep all observed host conditions; do not wait for quiet or
  retry unfavorable measurements. Missing monitor observations are unknown.
- A model-wide depth recommendation must not have a >5% median paired request
  latency regression versus two drafts in a declared cell. The motivating
  top-k cell should favor one draft on at least six of eight pairs. If another
  sampler materially regresses, narrow the implementation to the demonstrated
  setting or retain the old recommendation; do not promote blindly.
- For the candidate, check enabled-without-count resolves to one and matches
  explicit one-draft reference output/counters, explicit zero preserves normal
  generation, explicit two remains two, image operations use zero, and text
  recovers its chosen policy. Qualify exact prefix restoration on the
  image-loaded artifact, including mode changes and fresh references.

The setting does not change kernels, weights, precision, exact rejection
sampling, supported depths or snapshot representation. No numerical reference
or performance result is transferred to another artifact. The paired timing
screen and functional/lifecycle checks have separate outcomes.

## State and outcome

The one-draft candidate, its measurements and focused validation are complete
as historical branch work. The subsequent three-depth reconciliation failed
the preregistered gate, so the candidate recommendation was reverted to two.
The held-out top-k cell provides positive evidence for one draft in that
workload; the long-nucleus guard and primary material threshold keep that from
qualifying an artifact-wide default. Explicit one-draft selection remains
supported. Main, release and installed build status are unchanged.

## Completed candidate confirmation: 20261115-LILY-MTP1-02 (historical)

At the time of this completed confirmation, the candidate changed the loaded
recommendation only for the exact native-source repository/revision above.
Other receipts and Uzu loading retained the previous two-draft recommendation.
All raw omitted requests still defaulted to zero. This candidate behavior is
not the current recommendation.

The following historical checklist was preregistered for the completed
`20261115-LILY-MTP1-02` confirmation; its outcomes are recorded below.

Before installation or promotion, confirm actual `mtp: true` behavior across
the sealed before worker and the fresh candidate: image-loaded worker sessions
in before/candidate/candidate/before order, eight measured requests and one
excluded warmup per session, 512 input/256 output, top-k 20/top-p 0.95,
temperature 1, seed 42 and the same frozen corpus. This is 32 actual-default
requests; keep all host conditions. This supplements the same-binary depth
comparison rather than substituting a cross-time rate comparison.

Probe text-only and image-loaded readiness and explicit selectors against the
before worker's explicit zero/one/two reference hashes/counters. Run the
existing image-loaded prefix exactness validator with explicit depth one on
both binaries, so the comparison does not accidentally use two on the before
worker and one on the candidate. Record a real image transaction and text
recovery; images retain zero drafts and zero reuse. Run native identity/resolver
and host recommendation/persistence regression checks. No app installation is
part of these checks.

## Why it was not already the one-draft recommendation

MTP itself was implemented earlier. The initial
[exact serial verifier](QWEN38_EXACT_MTP_2026_10_30.md) reduced throughput in its
tested workloads. The qualified [batched verifier](QWEN38_BATCHED_MTP_2026_10_30.md)
then favored two drafts: its initial top-k comparison measured +12.44% for two
versus +1.53% for one relative to ordinary decoding. A later
[longer combination run](QWEN38_MTP_COMBINATION_2026_10_31.md) measured a 13.54%
two-draft top-k gain, while nucleus sampling had little whole-request benefit.
These source-bound records explain the two-draft recommendation; they do not
establish that two remains optimal after other source/workload changes.

The desktop and raw protocol also have different defaults. Desktop's model
preference normally enables supported Lily MTP and uses the recommendation.
Raw native requests that omit selectors use zero. The earlier claim that MTP
was simply off by default was incomplete for the desktop path. This follow-up
changes the selected depth when enabled, rather than adding a new MTP algorithm
or importing llama.cpp code.

## Exact candidate implementation scope (historical)

In candidate commit `86cb55ca`, `LilyModel::load_with_options` read the existing `native-source.json` through
the bounded JSON loader (1 MiB maximum). Exact `weights_source.repo` and
`weights_source.revision` matches select recommendation one. Missing, malformed,
oversized or different receipts retain the prior recommendation of two.
`from_model` initializes two, so the separate Uzu path retains its previous
behavior. Both ordinary and image-loaded native-source paths use the same
selection. The supported maximum remains three.

The candidate recommendation was bounded by the admitted head's maximum. It did not
enable MTP in engine state or change `default_drafts=0` for raw requests.
`mtp: true`, enabled Desktop/API preference and terminal `--mtp` use the loaded
recommendation; explicit native `mtp_drafts: 0`, `1`, `2` or `3` remain exact
choices. A saved Desktop off preference is preserved. Image requests use zero,
clear the text prefix, and following text resumes its chosen selection.

The prefix snapshot key continued to depend on MTP being enabled, rather than the
draft count. The captured boundary is before speculative decoding; the new
recommendation does not change the snapshot representation or restoration
algorithm. This study qualifies the image-loaded depth-one cache path and
plain/on switching, not every possible positive-depth transition.

## Identity

| Field | Value |
| --- | --- |
| Before source | Engine `cc8e47a824f802e24d442183ebbdd29fd25ecaaa`; study base `f9c9720c23ee848fcd64c002450f04b727e32e35` has identical engine source |
| Before executable SHA-256 | `61b9c8cc724407573169eb2abad7f908e4943eead7ac3122eeec3ef841a48640` |
| Candidate | Study base plus the engine patch below; no kernel, quantization, sampler or precision change |
| Candidate executable SHA-256 | `5618180b1821f1883778cc8099b4fa408699ff6a5cb0d71b33907f9946d441a4` |
| Engine patch SHA-256 | `9a444df3b706136f75a66188b3be823891f90cfff028e0de2b62feb91bf49070` |
| Artifact manifest SHA-256 | `97c11dd6f7ee5a69e23759b9774acb21a786c48552ca2ede9dac8b6be4b5bc98` |
| Source weights | Same 55 source-verified shards and tokenizer as the prior baseline; file identity guards unchanged |
| Precision | Preserved source mixed Q4/Q8 weights; BF16 activations, Q8 KV, F32 reductions/recurrent state/readouts |
| Build | Rust 1.98.1, release/locked generator, dedicated `runtime/build-candidate`; template helper built separately; no scorer feature unification |
| Hardware/OS | Apple M5 Max, 40 GPU cores, 128 GiB; macOS 26.7.2, observed AC power; no power-mode changes |
| Python | uv 0.12.24, locked converter environment, Python 3.13.13 |

All source/input/output/corpus/tool hashes and observed metrics are in the
[aggregate record](data/flashnext-lily-one-draft-2026-11-15.json).
As-run experiment tooling is [archived](../archive/experiments/2026-11-15-lily-mtp1/README.md).
Full receipts, binaries and frozen workload material remain ignored/private.

## Same-worker one-versus-two depth study

Every row has eight paired requests per depth, plus one excluded warmup per
depth. Displayed rates are arm medians; changes are medians of within-pair
ratios, so their ratios need not equal the displayed rate ratio. Negative
request change favors one draft. All 128 measured requests are retained.

| Input | Sampling | Outputs | Seed | Two-draft useful tok/s | One-draft useful tok/s | Paired decode change | Paired request change | One-draft latency wins |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | top-k 20 | 256 | 42 | 64.06 | 75.88 | +17.24% | -13.49% | 7/8 |
| 4,096 | greedy | 256 | 42 | 70.19 | 75.86 | +8.04% | -5.76% | 8/8 |
| 512 | nucleus | 256 | 42 | 69.47 | 81.62 | +16.54% | -12.99% | 8/8 |
| 4,096 | top-k 20 | 256 | 42 | 78.82 | 80.12 | +1.65% | -0.42% | 5/8 |
| 512 | greedy | 256 | 42 | 84.08 | 84.45 | +0.41% | -0.54% | 5/8 |
| 4,096 | nucleus | 256 | 42 | 76.96 | 71.95 | -6.65% | +4.39% | 0/8 |
| 32 | greedy | 64 | 42 | 77.47 | 82.60 | +6.81% | -6.03% | 8/8 |
| 512 | top-k 20, held-out corpus | 128 | 7 | 73.47 | 89.70 | +22.56% | -15.58% | 8/8 |

The largest measured request regression is the long nucleus cell at 4.39%,
within the predeclared 5% tolerance. Its decode rate is 6.65% worse; the
latency criterion must not be misrepresented as no throughput regression.
The primary cell and held-out confirmation favor one, but this is a mixed
workload tradeoff rather than a universal improvement.

All repeated hashes match within each depth; greedy outputs also match across
depths. Sampled paths need not have identical sequences across depths because
the random-draw schedule differs. All requests report zero prefix reuse and
real MTP proposals. The complete-worker-request timer includes native generation
and pipe transport, excluding loading, warmup, tokenization, template rendering
and app/API transport. Useful decode excludes the first emitted token.

Host conditions were busy: 75/128 requests pass the legacy host screen;
maximum external CPU was 1,794.09%, coarse thermal state reached numeric 1,
and the helper observed seven pageout increments and zero swapout increments.
No observation is excluded for those conditions. GPU clocks/exclusivity and
peak process footprint were not measured.

## Actual before/candidate default comparison

Both binaries received `mtp: true`. The before worker selected two and the
candidate one; actual output hashes and round/proposal/acceptance/verified-row
counts match the earlier explicit-depth references. All workers loaded images.
Order was before/candidate/candidate/before, one warmup and eight measured
requests per session: 32 measurements, 16 per arm, no exclusions or retries.

| Metric | Before | Candidate |
| --- | ---: | ---: |
| Median useful decode tok/s | 81.95 | 89.24 |
| Median complete request seconds | 3.415 | 3.182 |
| Observed request range, seconds | 2.965–3.487 | 2.894–4.220 |

The predeclared cross-session pairing yields **+2.65% median paired decode**
and **-2.16% median paired request latency**, with **10/16 latency wins**.
The ratio of pooled arm medians looks more favorable but is a different
statistic; do not substitute it for the paired result. Variation is too large
to establish the requested noticeable binary-to-binary gain. This result is
retained without a favorable rerun.

Thirty of 32 requests pass the legacy screen; maximum external CPU is 156.07%,
thermal state is numeric 0, pageout increments total nine and swapouts zero.
Passing that screen does not prove equal GPU operating state. These data cannot
identify the cause of the difference between the same-worker and cross-worker
observations. The actual-default performance result remains inconclusive.

## Functional, cache and image validation

- Twenty-six real-model checks pass across before/candidate and text/image
  loading: enabled selection, explicit one/two/zero/off, omitted selectors,
  matching per-depth reference hashes/counters, real image generation at zero
  drafts/reuse, and text recovery at one draft.
- The existing prefix validator passes 37 exactness comparisons. Cached outputs
  match fresh candidate and explicit-depth-one before-worker references;
  eligible follow-ups reuse 4,096 input tokens. It covers shrinking/divergence,
  chunk edges, context/scope changes, plain/MTP switching, errors, unscoped
  requests, reset, real image invalidation, cancellation and restart. The
  tokenize/decode control preserves reuse as documented.
- Full payload hashing in that validator is retained separately; its manifest
  algorithm differs from the earlier complete-artifact JSON manifest.
- Image checks establish lifecycle and image-loaded text behavior, not a new
  image-answer-quality certificate or broad model-quality result.
- The source-pinned identity helper passes its exact/wrong/missing/malformed
  receipt test. Generator protocol tests: 98 passed. Focused Python MTP tests:
  112 passed. Menubar controller checks: 156 passed. Repository policy and Rust
  formatting checks pass.

The first Python test run occurred before the fresh checkout's native template
helper was built: 49 passed, 36 failed and 27 errors. The identical selection
passes after building the helper and putting it on PATH. This is retained as
an invalid test setup, not attributed to the recommendation change. No native
model exception dumps or generated content are published.

## Reproduction and disposition

The historical candidate's source-pinned recommendation was implemented in
[`lily.rs`](../../crates/stark-engine/src/engine/hybrid/lily.rs) at `86cb55ca`.
The current source restores the original two-draft recommendation.
The regular host consumes the live recommendation; it does not need a new
per-model preference migration. The retained host regression test checks a one-draft
capability, saved-off behavior and zero image MTP.

The exact prefix command uses the candidate `stark-template` on PATH and the
sealed before executable, with:

```text
--vision --reuse boundary --context 8192 --first-tokens 4600
--tools 1 --instruction-paragraphs 2 --follow-up-sentences 4 4 8
--max-new 16 --mode exactness --mtp-drafts 1
```

Use explicit depth one on both reference/candidate for numerical/cache equality.
Using `mtp: true` on both instead would compare different draw schedules and
is the separate actual-default timing experiment above.

Disposition: the one-draft candidate was implemented and tested, then reverted
after the preregistered reconciliation gate failed. The held-out top-k cell
measured a large scoped benefit (+20.03% decode, -14.45% request latency), but
the primary 512-token top-k latency gain was below the material threshold and
the long-nucleus decode guard regressed. Two drafts remain the artifact-wide
recommendation; explicit one-draft selection remains supported. No kernel or
sampler work, installed app update, release or PR occurred. The
[reconciliation](2026-11-15-lily-mtp-default-reconciliation.md) and
[aggregate data](data/lily-mtp-default-reconciliation-2026-11-15.json) retain
the full three-depth results. Other artifacts, longer contexts, other sampling
filters, GPU operating states, long-form task quality and every positive-depth
cache transition remain outside this bounded qualification. Preserve explicit
controls and these limits when considering future default changes.
