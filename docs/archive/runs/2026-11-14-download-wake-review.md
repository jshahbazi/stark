# Download wake recovery corner-case review

Date: 2026-11-14. Reviewed implementation: `94795fde`, based on `4bcaba76`.
The follow-up changes and this record are committed together on
`codex/resume-model-downloads`.

The initial evidence remains in
[the implementation run](2026-11-14-download-wake-recovery.md).

## Findings and fixes

- The remote installation's initial support-check child ran before download
  retry ownership was established. Sleep could interrupt that metadata request
  and terminate the installation. Scope sleep ownership around remote install
  preflight too, reap its interrupted child, and retry after wake. Complete
  conditional, unsupported and inconclusive reports remain authoritative;
  ordinary preflight failures, discovery and local checks are not retried.
- A restarted child resets its transferred-byte counter. Merging counters with
  `max` undercounted actual traffic across attempts. Offset each child's counter
  by previous attempts' total, and accept its early cache-discovery frames while
  preserving monotonically increasing completed model bytes. A three-attempt
  test verifies 100 + 60 + 25 = 185 network bytes with 100 completed model bytes.
- Classifying only the outer exception's disk/permission/configuration category
  could follow a wrapped permanent failure into an older network-error context
  and retry it. Honor terminal categories across the full inspected cause chain
  before deciding to retry, including transport errors wrapping local permission
  failures. Cyclic
  exception chains remain bounded, and raw exception contents remain private.

The independent Swift review found no additional concrete issue in observer
lifecycle, request ownership, model naming or guided selection rollback.

## Validation

- 149 focused tests passed across progress, native preparation, model browsing,
  remote install support and desktop retry/lifecycle cases. Coverage includes a
  real synthetic preflight child interrupted/reaped on sleep, resumed after wake,
  cancellation while waiting, completed terminal support reports, and transfer
  accounting across three restarted children.
- Menu bar controller harness: 154 checks passed, no failures or skips.
- Ruff for changed converter modules/tests, repository policy check and
  `git diff --check`: passed.

This follow-up does not supersede the initial record's broad-suite fixture
failures and successful isolated reruns. Physical lid-close recovery and a real
large-model transfer remain unqualified. No production app was installed and no
release was published.
