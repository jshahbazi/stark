# Model context default validation

Date: 2026-10-31. Source: base `f11532c12786aae2740b434c08212b6e1c44c3cc`
plus the default-context patch committed with this record. The earlier
[capacity validation](2026-10-31-model-context-capacity.md) records the previous
8,192-token default and remains unchanged.

Desktop and terminal chat default to the loaded model's configured maximum.
The desktop default follows model changes; custom context budgets survive model
clamps and restart within the app session. Applying the current model maximum,
including through Restore Defaults, returns to following each model's maximum.
Output defaults and limits are unchanged.

Validation:

- Shared-worktree converter suite: 5,104 passed, 88 skipped. Concurrent edits
  outside this patch were present; they were excluded from the commit.
- An export of the exact staged source passed all 170 desktop, terminal and
  image-chat tests, all 111 Swift controller checks, and full production Swift
  app type-checking. Cases cover 2-token, 512-token, 32K, 262K and 1M model
  limits, model switching, custom budgets, Restore Defaults and emitted requests.
- Native smoke checks against that staged source confirmed both desktop and
  terminal generation used the loaded dense model's 40,960-token maximum with
  no explicit context override. A one-token output budget kept the checks bounded.
- Repository policy, Python lint/format and staged diff whitespace checks passed.

The staged desktop source SHA-256 was
`751f5c2d242de27e94cfb8b818f647d120ab99512e9ad28559583fd376a89107`.
The native workers were unchanged from the capacity validation. These checks
establish default selection and dispatch; they do not qualify full-length model
quality, memory requirements or throughput. No installed app was replaced.
