# Lily automatic MTP recommendation restored to two

Date: 2026-11-15. Attempt: `20261115-LILY-MTP1-07`.

## Decision and behavior

The user requested removal of the automatic one-draft policy while preserving
the benchmark results. Enabled Lily MTP again recommends two drafts for every
sampling profile, bounded by the admitted head's supported maximum. Desktop,
API and terminal callers consume that scalar recommendation. Raw native
`mtp: true` selects the same two-draft recommendation.

Explicit supported native draft counts remain available, including one. Saved
Off preferences, explicit MTP Off and raw omitted selectors retain their prior
behavior; omitted native selectors use zero. Image requests remain MTP-off.
Other model families keep their existing recommendations.

## Implementation

The five changed Rust files and Python production modules `mtp.py` and
`desktop.py` are restored to their exact pre-policy contents at `f331c648`.
The complete `crates/` and `tools/converter/stark_convert/` trees match that
revision. This removes the receipt-specific policy field/getter, optional
policy advertisement, sampler-aware native resolution and Python matcher.
Planning and generation again use the scalar recommendation.

Policy-specific tests are removed. The prior generic one-draft-capability test
is retained because explicit counts and other loaded-model recommendations
remain supported. A new API regression check supplies the retired policy
metadata and verifies that it is ignored: a top-k request still sends two
drafts. No API parser, Swift or CLI entrypoint changes are needed.

## Preserved evidence

The selected [short-workload qualification](../../benchmarks/2026-11-15-lily-sampling-mtp-policy.md)
and its 20.47% decode / 14.66% request-time gain remain historical results for
candidate `1b0f5ed0` and executable SHA-256
`e1e432fcd5d490c8ab900be23c7cbc3855fc4515d75a38089bac672a85e9be22`.
The later [coding response](../../benchmarks/2026-11-15-lily-long-coding-response.md)
loses 11.35% useful decode at one draft in all four pairs. The
[planning response](../../benchmarks/2026-11-15-lily-long-planning-response.md)
gains only 1.02%, within observed noise. Response-length and host-load limits
remain attached to those comparisons.

Every tracked benchmark JSON and archived experiment source remains
byte-identical to `3421d2f6`; only current documentation and status notes change.
The unchanged Git tree IDs are:

| Directory | Git tree ID |
| --- | --- |
| `docs/benchmarks/data` | `c194951d05501070ccacaffd45eac09d50f85403` |
| `docs/archive/experiments` | `dddf2dc0d8be3f189516e3ad520c648ffaa83193` |

The original before/candidate receipts, setup failures, source identities,
timings, counters, output hashes and ignored binaries are retained. Historical
helpers require their recorded source/binary identities; the restored host no
longer applies the retired policy metadata. No historical timing is relabeled
as a new validation or overwritten by this revert.

## Verification and delivery

- Full generator protocol suite: 98 passed under Rust 1.98.1 without default
  features.
- Focused Python MTP checks: 120 passed across capability, Desktop, API parsing
  and terminal tests; the new retired-policy API regression also passes.
- Pinned-nightly Rust formatting, repository policy and whitespace checks pass.
- Production source equality to `f331c648` and unchanged benchmark/archive trees
  are verified directly. This is an exact source restoration of the previously
  exercised two-draft path, with no new inference or precision implementation.

The initial Rust test command could not find cargo on the shell PATH; the
absolute pinned-toolchain command passes. No new performance run or numerical
claim is made. The revert is committed and pushed to the existing experiment
branch; no app installation, release or PR is performed.
