# Beginner recommendation review and corrections

Date: 2026-11-05. Branch: `codex/hardware-model-recommendations`.
Source: the commit containing this record, reviewing and correcting
`b1dcc636c14a4ab94b6a0f50f26b906dc905d114`.

## Review findings addressed

1. Installed reuse previously checked source identity and receipt/config
   binding without distinguishing F16/F32 reference conversions from the
   BF16 route used by the guide's memory plan. Reuse now also checks conversion
   capability, actual tensor-header precision, quantization, file size and the
   receipt's output size. Required F32 normalization scales remain valid;
   floating matrices and quantization parameters must match the BF16 route.
   The inspection reads bounded headers and never tensor payloads. Existing
   reference artifacts and their validation records are preserved.
2. Fixed recipes omitted attention scratch memory included for catalog
   candidates. Both now use the same query-head-based formula, including the
   output-header allowance. The current 4B planning estimate is 11 GiB; the
   0.6B estimate remains 9 GiB. The earlier run's 10 GiB 4B estimate remains
   recorded in its original evidence and is superseded as current policy.
3. Installed-model template/header inspection ran while holding the controller
   lock used by generation and Stop. It now works on copied model metadata
   outside that lock. Publication repeats the generation fence, and install/run
   continue to recheck artifact identity and resources.

Regression coverage includes same-source reference-precision copies, blocked
or incomplete receipts, quantization-parameter precision, oversized artifacts,
header-only reads, static/catalog memory-plan parity, GPU-budget boundaries,
and cancellation during a slow installed-model inspection.

## Native reuse and startup checks

The installed Qwen3 4B 4-bit and Qwen3 0.6B BF16 artifacts both passed the new
reuse checks. The source-bound catalog recipe for `Qwen/Qwen3-0.6B` at revision
`c1899de289a04d12100db370d81485cdf75e47ca` was then bound to an isolated
controller and started successfully. Playground and API budgets were both
32,768 tokens. This used an existing converted model and a separate generation
process; it did not edit the installed app registry, download weights, generate
responses, or execute OpenWork tools.

The generator was the pre-existing local model-catalog app bundle's engine;
this change does not modify engine code. Its measured SHA-256 was
`cbf90a888632d6484bd455abd617415e94a7e0cfafac84182c23015c73f29117`.
This confirms the real receipt/header and native startup integration for one
dynamic recipe. It does not qualify every catalog variant or measure speed,
answer quality, long-context generation or agent competence.

## Validation

- Guided recommendation and discovery regressions: 98 passed.
- Native controller checks: 128 passed.
- Native Beginner state/view checks: 9 passed.
- Full offline converter suite: 6,107 passed, 89 skipped in 230.49 seconds.
- Ruff lint/format and whitespace checks passed.

The branch is updated to include the catalog changes merged into `main` in
pull request #82. The recommendation PR retains its own scoped diff.
