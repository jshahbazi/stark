# MTP pull request review, 2026-10-31

Reviewed source: `a401ecb8d060003fdf1a7fc06ca274855c56fc15`.
This synchronizes current `main` (`c3c83fcd`) without changing the preceding
branch's file tree. It includes the exact MTP implementation, official batched
verifier, host sampling optimizations and model-aware toggle backend.

## Fresh validation

- Full converter suite: **5,094 passed, 88 skipped**, 147.94 seconds.
- Generator release tests: **85 passed, two ignored**; release Clippy with
  warnings denied passed.
- Vendor default suite: **75 passed, 51 ignored** on the physical Apple silicon
  host, including the ordinary-versus-batched Q8 head control.
- Model-free MTP benchmark harness: **23 passed**.
- Repository policy, Rust formatting, full converter Ruff lint/format,
  locked converter dependency resolution and whitespace checks passed.
- A release rebuild reproduced the toggle worker SHA256
  `97e0b0ff5d0e873eb3913ade8831eaeb0b4523a4071ea32e16da66b0d8b563c7`.
- The real official mixed-4/8-bit checkpoint passed the nine-request
  [toggle equivalence and recovery smoke test](2026-10-31-mtp-toggle-backend.md#real-checkpoint-smoke-test)
  again, with the same per-route output hashes. The existing validator ran
  unchanged and wrote a separate private receipt.

The first full converter run had two synthetic worker startup failures:
`test_advanced_options_invalid_changes_are_atomic[overrides12]` and
`[overrides13]`. Both failed before the options under test were applied.
All 16 cases passed in isolation, followed by the complete passing run above
after this review's compiler jobs finished. These fixtures and their two-second
startup timeout are unchanged from `main`. The exact failure cause was not
captured; readiness timeout under concurrent load is plausible but unconfirmed.
No product or timeout change was made on that assumption.

## Independent review

Separate read-only reviews found no actionable defect in:

- Acceptance/residual sampling, RNG ownership, verifier readout ownership,
  accepted-prefix accounting, rollback, cancellation, partial delivery and
  failure fencing.
- Model-specific native selectors and limits, capability admission, saved
  per-model preferences, API overrides and queue snapshots, image handling,
  lifecycle clearing and acknowledgement checks.
- The Q8 head's dot expressions, addition order, SIMD reduction, compiler
  policy, BF16-value rounding, row validation and buffer bounds.

The committed official benchmark Markdown and JSON match their later
publication audit and final seal hashes. The earlier private aggregate audit's
`public_json_audited: false` predates that later publication check.

No inference algorithm or quantization changed during this review. No new
throughput benchmark was run. Existing benchmark, GPU component, target-readout
and scorer records retain their original source identities and limitations;
see the [official MTP report](../../benchmarks/QWEN38_OFFICIAL_BATCHED_MTP_2026_10_31.md).
Private receipts and scripts remain ignored; only aggregate results and hashes
are published here.
