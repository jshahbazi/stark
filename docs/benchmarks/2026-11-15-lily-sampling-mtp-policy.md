# Lily sampling-specific MTP recommendation

Date: 2026-11-15. Attempts: `20261115-LILY-MTP1-04` / `-04b` (functional
controls and image continuation), `-05` (metadata-only preflight),
`-05-CONT-02` (matched performance confirmation), and `-06` (image-loaded
prefix transitions).

## Subsequent long-response evidence

The later natural [coding request](2026-11-15-lily-long-coding-response.md)
favors two drafts: candidate one-draft useful decode is 11.35% lower in the
paired comparison, with all four pairs worse. Its response is also 22.72%
longer. The [planning request](2026-11-15-lily-long-planning-response.md)
shows only +1.02% paired decode and mixed session halves, inside observed noise;
its small time saving includes a 2.86% shorter response and unequal host load.

The selected short-case gain below remains source-bound evidence. These long
responses show that the sampler profile alone does not establish a generally
better draft depth. The automatic one-draft implementation was later reverted;
the current Lily scalar recommendation is two. This record preserves the
historical implementation and confirmation without replacing their results;
see the [restoration record](../archive/runs/2026-11-15-lily-mtp-two-draft-default-restored.md).

## Historical candidate behavior and scope

Implement one automatic draft for the pinned
`fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` revision
`0f94842e3682b49acace4daeb4cf2e65a6f94f83` only when stochastic sampling uses
temperature 1, top-k 20, top-p 0.95 and absent/null min-p. Match the existing
native F32 representation of sampling floats. Min-p zero is a distinct,
unqualified filter setting and retains two. Greedy, nucleus and other filter
settings retain the scalar recommendation of two. Other artifacts retain
their existing recommendations.

The actual loaded worker grants the optional capability
`sampling_recommendation_policy: "lily-topk20-one-draft-v1"`. The repository
and immutable revision must both match its bounded native-source receipt.
This runtime predicate compares receipt metadata; the qualification below
separately binds the actual measured payload and preparation hashes.
The general `recommended_drafts` remains two. An old host ignores the optional
capability and retains two; a new host with an older worker also retains two.
Unknown optional policy values use the scalar recommendation. No model-name
heuristic in the host grants this policy.

Native `mtp: true` and enabled Desktop/API preferences apply the policy to each
request. Explicit native `mtp_drafts` remains the caller's exact count. Raw
omitted selectors and MTP Off remain zero. Image requests use zero and clear
text prefixes; following text resumes its selected policy. No sampler,
verification arithmetic, weights, precision or image head is changed.

This is a narrower follow-up to the
[rejected general one-draft recommendation](2026-11-15-lily-mtp-default-reconciliation.md).
That study measured +4.25%, +2.21% and +20.03% useful decode for its three
top-k cases, while long nucleus lost 6.01% and greedy was flat. The held-out
case won all 12 blocks and reduced request time 14.45%; the earlier study of
that workload measured +22.56% decode. Its one-draft path verifies 141 target
rows versus 188 at two drafts, for the same 127 useful advances, while two
drafts saves only eight rounds. These source-bound results motivate the
sampling-specific policy; they do not establish a 20% benefit for all top-k
requests or other artifacts.

## Fixed confirmation plan

Recorded before executing the new candidate:

- Reuse the exact artifact, frozen corpora and sealed before worker from the
  preceding study. Before executable SHA-256 is
  `61b9c8cc724407573169eb2abad7f908e4943eead7ac3122eeec3ef841a48640`;
  final production engine source at `f331c648` matches that before engine.
  Build a fresh candidate with locked Rust 1.98.1 in its own target directory.
- Keep the retained images and trained MTP head loaded. Preserve source mixed
  Q4/Q8 weights, BF16 activations, Q8 KV and F32 reductions/recurrent/readouts.
  Keep every observation under concurrent host activity, with no quiet gate,
  cooling pauses, unfavorable-result retries or performance filtering.
- Use four fresh sessions in before/candidate/candidate/before order. Each
  session executes four frozen cells: original 512/256 top-k seed 42, held-out
  512/128 top-k seed 7 with its alternate corpus, 512/256 greedy seed 42, and
  4,096/256 nucleus seed 42. Counterbalance cell order. Use the same explicit
  16 MiB transport budget in every worker. Fixed context-growth setup and one
  warmup per cell precede four measurements per cell/session: 64 measured
  requests and 16 cell warmups, with four 4,096-input/2-output/context-4,352
  setup requests counted separately.
- Both builds receive actual automatic `mtp: true`. Before must acknowledge
  two in every cell. Candidate must acknowledge one only in the two qualified
  top-k cells and two in both guards. Keep request time and useful decode
  separate. Preserve hashes, counters, order, timing and host observations;
  keep generated content and token arrays out of saved artifacts.
- This confirms an implementation selected using the prior 198-request study.
  The targeted held-out case is the performance confirmation endpoint: seek
  at least 10% lower median paired request time and at least 6/8 positional
  wins. The original short top-k case should retain its positive direction.
  Cross-session positional pairs retain drift and are descriptive; preserve
  an inconclusive outcome without substituting a favorable arm-median ratio.
  Require both four-pair session-half latency medians to improve and differ
  by no more than ten percentage points before calling that endpoint confirmed.
  The rejected global recommendation's 5% original-cell gate is not reused to
  reject a deliberately narrower policy whose larger held-out gain was
  explicitly selected for implementation.
- Guard workloads must retain matching depth-two outputs and work counts.
  Report observed guard timings without attributing session noise to code that
  leaves their selected algorithm unchanged. No general traffic-weighted gain
  or delivery claim follows from fixed synthetic sequences.
- Validate native and host policy boundaries, nearby native F32 values,
  explicit zero/one/two/three/off/omitted controls, both planning and generation,
  legacy/unknown capability fallback, streaming and saved preferences.
- On the image-loaded artifact, qualify automatic one-to-two and two-to-one
  transitions by changing sampling while preserving an eligible chunk-aligned
  prefix. Compare each restored result against fresh executions at its exact
  sampler/depth. Exercise a real image and verify zero MTP/reuse, fresh text
  recovery and subsequent reuse. The existing enabled-state snapshot boundary
  must be exact before keeping cross-depth reuse.

The existing clock executable's source-interval schema is incompatible with
the clock helper. Do not repeat the observer failure or infer missing clocks;
GPU operating state remains unknown. This work does not install an app,
publish a release or open a PR.

## Candidate state and outcome at `1b0f5ed0`

At that revision, the implementation and bounded qualification were complete
and adopted on the branch. Native and host model-free controls pass; 47 live functional checks
and seven image-loaded prefix comparisons pass. The fixed 64-measurement
before/candidate comparison confirms the selected held-out case: **+20.47%
useful decode and -14.66% complete-request time, with 8/8 latency wins**.
The original short case's small, noisy gain remains inconclusive. No app is
installed or released. The older source-bound records and the `f331c648`
general-policy rejection remain unchanged.

## Matched automatic-policy performance confirmation

The first proper native timing run completes the fixed 64 measurements, 16
cell warmups and four context-setup requests, with no exclusions or repeats.
Both builds receive `mtp: true`. Before selects two in all cells; candidate
selects one in the qualified top-k cells and two in greedy/nucleus. Every
output hash and work tuple matches the prior reference for its actual depth.
Different stochastic depths need not have identical output sequences.
All timing workers load the retained image tower. Text-only loading is covered
by functional controls; this follow-up does not time a separate text-only
configuration or reused-prefix throughput.

Each row contains eight positional pairs across B/C/C/B worker sessions.
Changes are medians of pair ratios, not ratios of arm medians. The last column
shows each four-pair half's median request change and exposes session drift.

| Cell | Useful decode change | Request time change | Latency wins | Session-half request changes |
| --- | ---: | ---: | ---: | ---: |
| Original 512/top-k20/256-output case | +2.01% | -1.28% | 4/8 | +7.98%, -7.40% |
| Known held-out 512/top-k20/128-output case | +20.47% | -14.66% | 8/8 | -19.76%, -14.09% |
| 512/greedy/256-output guard | +3.59% | -4.49% | 6/8 | -0.61%, -7.89% |
| 4,096/nucleus/256-output guard | +0.99% | -1.55% | 7/8 | -0.71%, -2.08% |

The targeted held-out endpoint passes its >=10% request-benefit, >=6/8-win and
session-consistency checks. Its median absolute time saving is 0.3455 seconds.
The original short case meets only its declared overall direction check; its
opposing session halves do not establish a noticeable gain. This confirmation
reuses the previously selected held-out workload and is not a newly unseen
generalization test.

Both guards retain identical depth-two output hashes and work counts across
builds. Their timing changes are observations under different sessions; no
greedy or nucleus compute optimization was implemented. The broad one-draft
nucleus regression is avoided by selecting two there.

Host observations remain descriptive: maximum external CPU is 181.80%, maximum
coarse thermal state is 1, pageout increments total 169 and swapouts zero.
There are no missing per-request CPU/thermal samples or monitor failures.
Available/used memory, memory pressure, source clocks, peak footprint and GPU
exclusivity are not measured, so the paging observations do not identify a
memory or GPU cause. All requests are retained.

The result supports this source-bound configuration choice and the observed
held-out gain. It does not establish a universal 20% top-k benefit, a
representative coding-traffic gain or app/API end-to-end latency. The policy
applies to requests already using the qualified sampling profile.

## Functional and prefix evidence

- The source-identity helper passes under pinned nightly, including exact and
  wrong/missing/malformed/oversized receipts. All 99 generator protocol tests
  pass under Rust 1.98.1 without default features. The 151 focused Python MTP
  tests pass across capability admission, Desktop, API parsing and terminal
  chat; a separate 31-case run covers the new filter/F32 and API policy cases.
- Forty-four saved live checks pass on before/candidate and text-only/image
  loading: native explicit 0/1/2/3, automatic selection, off, omitted selectors,
  same-depth output/work identity, nearby filter fallbacks and planning.
  The image step then stops because its native template helper is absent
  from the test command's PATH. That original failed summary is preserved.
- A separately hashed continuation adds the existing template helper to PATH
  and passes three checks: top-k text at one, real image at zero, text at one
  afterward with the sealed before worker's explicit-one hash and work.
  The original 44 checks are not rerun. Image scope is lifecycle, not answer
  quality.
- The prefix run uses 4,615 input tokens, context 8,192 and chunk 4,096. Native
  automatic selections are `[1, 1, 2, 2, 1, 1, 1]`; reuse counts are
  `[0, 4096, 4096, 4096, 4096, 0, 4096]`. All seven outputs match both fresh
  candidate and fresh before-worker executions at explicit matching depths.
  The real image between the fifth and sixth text requests reports zero MTP
  and reuse, then text recovers cold and reuses again. Workers run sequentially.

The Lily prefix key and snapshot store MTP enabled state. `HybridBackend`
sets that state before prefill; prefill receives no numeric draft depth.
Depth is first used during decode. The unchanged snapshot therefore covers
the same prefill computation at one and two drafts. The measured exactness
checks above establish this transition on the pinned artifact without
discarding an eligible positive-depth prefix.

## Identity and setup records

| Field | Value |
| --- | --- |
| Source base | `f331c648` plus the separately sealed native and Python patches |
| Before executable SHA-256 | `61b9c8cc724407573169eb2abad7f908e4943eead7ac3122eeec3ef841a48640` |
| Candidate executable SHA-256 | `e1e432fcd5d490c8ab900be23c7cbc3855fc4515d75a38089bac672a85e9be22` |
| Native patch SHA-256 | `24bc87a033386e9150a7d264afb546e83cf5dd86b1aee9e9f0bce8d3aa3797c0` |
| Python production patch SHA-256 | `e947c9919583e68d09678c684b7de80ed400a472ff8873150465d7efc6efd307` |
| Prior payload/preparation manifest SHA-256 | `97c11dd6f7ee5a69e23759b9774acb21a786c48552ca2ede9dac8b6be4b5bc98` |
| Build | Rust 1.98.1, release/locked generator, isolated `runtime/build-topk-candidate`; no scorer feature unification |

The initial engine helper-test command used stable, which cannot run the
project's custom test framework; the corrected pinned-nightly command passes.
The prototype capability field on the engine's public Rust struct was replaced
by an additive `Model` getter before sealing the candidate, preserving existing
downstream struct literals. That prototype release build was not used for
native qualification.

The first performance invocation fails a metadata precondition before loading
a worker: the earlier payload receipt records 63 files, while the model folder
also contains nine ancillary files. All original 63 file identities match.
The continuation retains that guard and separately pins the nine ancillary
file identities; no native measurement is repeated. These include source
documentation/license, download metadata, tokenizer side files and image-check
metadata. Their contents are not published.

## Reproduction and delivery

The [combined aggregate](data/lily-sampling-mtp-policy-2026-11-15.json) preserves
the initial functional failure and continuation separately, all seven prefix
comparisons, all 64 measured timings plus fixed setup observations, actual MTP
depths, host metrics, before/candidate identities and sealed source hashes.
The [as-run sources](../archive/experiments/2026-11-15-lily-sampling-mtp-policy/README.md)
include both failed setup snapshots and their continuations. Native contents,
weights, token arrays and frozen workload copies stay ignored/private.

Repository policy, pinned-nightly Rust formatting and archived Python syntax
checks passed for the historical candidate. That candidate kept the public
Rust capability struct's fields and used an additive engine getter and optional
wire metadata. The decoder's numerical policy was unchanged. The experiment
was limited to this pinned artifact; its selected-workload qualification did
not generalize to longer natural responses.

## Superseded branch policy

After the coding comparison found lower useful decode rate for one draft in all
four pairs, the automatic sampling-specific one-draft selector was removed from
the current branch. Lily `recommended_drafts` is again two for every sampling
profile; explicit supported draft counts remain available, and requests with no
MTP selector still use `default_drafts` zero. The planning comparison was near
noise and does not reverse that decision. The held-out short-request timing and
seven prefix checks above remain historical, source-bound evidence for the
candidate implementation; they do not describe current branch behavior. The
[restoration record](../archive/runs/2026-11-15-lily-mtp-two-draft-default-restored.md)
documents the revert. No installation, release or PR is claimed.
