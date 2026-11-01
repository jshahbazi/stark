# Model-aware MTP toggle backend, 2026-10-31

Implementation commit: `00cc5d1ebf114750bda10cea5d622c463fe6feea`, based on
`688975102703563992c67065d9506c06c104e794`.

## Behaviour

The loaded backend advertises MTP support, recommended and maximum draft
counts, supported operations and exact-sampling policy. Unsupported workers
expose an unavailable capability. The common protocol no longer assumes every
model has the same draft limit; current Lily and official mixed-affine Qwen3.8
backends recommend two drafts and permit up to three.

The desktop backend stores a separate boolean preference for each registered
model in its private registry. It starts off and survives model switches,
stops and restarts. Unsupported replacement workers make that preference
inactive without deleting it. Changing the preference requires the matching
loaded model in idle Chat mode and preserves the loaded worker and conversation.

HTTP completion requests can override the saved preference with `mtp: true`
or `false` for one request. The CLI provides session-local `--mtp` and
`--no-mtp`. Native requests accept either the boolean or the existing explicit
`mtp_drafts` count. Unsupported enable requests, conflicting native selectors,
null values and invalid types are rejected. Image requests default off and
reject explicit enable requests while retaining the text preference.

Capabilities and completion acknowledgements are checked against the selected
mode. A contradictory acknowledgement fences the worker. Live capabilities
clear on failure, shutdown or mode changes. See the current
[generation protocol](../../GENERATION.md#multi-token-prediction) and
[API/configuration contract](../../LOCAL_API.md#mtp-capabilities-and-per-model-preferences).

No UI control, additional model algorithm or kernel change is included.
Quantization, precision policies and exact rejection sampling are unchanged.
Earlier MTP workers and benchmark records remain preserved. This record adds
no throughput measurement; the preceding
[official-checkpoint benchmark](../../benchmarks/QWEN38_OFFICIAL_BATCHED_MTP_2026_10_31.md)
retains its original scope and source identity.

## Validation

- Full converter suite: 5,093 passed, 88 skipped. A later test-only addition
  for image requests and queued configuration was covered by the final
  desktop MTP selection: 14 passed, 114 deselected. Runtime code did not
  change after the full-suite run.
- Generator unit suite: 85 passed, two ignored. Coverage includes strict
  selectors, model-specific draft limits above three, unsupported models,
  capability changes, optional statistics and contradictory acknowledgements.
- Optimized scorer suite: 41 test executions passed across its three targets.
- Generator and optimized scorer Clippy checks passed with warnings denied.
  Rust formatting, changed-file Ruff lint/format, repository policy and
  whitespace checks passed.
- Independent source review found no actionable runtime issue. A separate
  documentation pass corrected wording about the pending UI and optional
  backend statistics.

Checks ran on the working tree subsequently committed as the implementation
above. The private runner receipts retain the then-current parent HEAD
`688975102703563992c67065d9506c06c104e794`; that label alone does not identify
the new implementation. Final formatting and unused-variable cleanup affected
only test source after execution.

## Real-checkpoint smoke test

The release generator built with Rust 1.98.1 was tested against
`ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`, revision
`788c07a8725642bccf13a0a3b8a21954ab84b4f2`, using unchanged weights. Its SHA256 is
`97e0b0ff5d0e873eb3913ade8831eaeb0b4523a4071ea32e16da66b0d8b563c7`.

Nine generation requests used 128 input tokens, a 32-token output budget,
temperature 1, top-k 20, top-p 0.95, seed 42 and context capacity 192. The live
descriptor reported maximum three, recommendation two, default zero,
`exact-rejection-v1` sampling and `batched-affine-decode-exact-v1` verification.
Workload settings are bound to the private `smoke.py` validator, SHA256
`1f8238c2dffdd320db4629ddbe7ae606f7ea92d51433cd6dc5cc4ee67c57a0d9`;
the aggregate receipt records the results and hashes.

- Boolean on reproduced the explicit two-draft output hash.
- Boolean off, explicit zero and omitted selectors reproduced the same
  autoregressive output hash.
- On/off/on requests did not leak mode or generation state.
- Conflicting selectors and explicit nulls produced recoverable errors;
  the subsequent off request retained its expected output hash.

| Control | SHA256 |
| --- | --- |
| Input token sequence | `9906f12d9971123be284ddad10599f2440f9d2e8c2229414cf31a99de3e6f0c0` |
| Enabled output token sequence | `0c0f1994b95d09d0c7b5f5062d71ed884b851e28807b275bcd7a8919bb5c55c4` |
| Disabled output token sequence | `3e0b3dea963e4634456b76ce11ba87e295d36514ab4a23180cf46740bc338336` |

The hashes compare equivalent selectors within each route. Exact sampling
preserves the target distribution; on and off need not emit the same sequence
at a given seed. The smoke test validates wiring and recovery rather than
performance or the sampling algorithm independently. Private scripts, workers
and aggregate receipts remain ignored; no prompt, response, token vector or
decision probability is published.
