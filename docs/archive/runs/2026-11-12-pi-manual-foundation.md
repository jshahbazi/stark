# Pi manual foundation, 2026-11-12

This record starts the [agent client integration plan](../../AGENT_CLIENT_INTEGRATION_PLAN.md)
with configuration tooling and model-free Pi checks. The implementation is in
the separate, private `stark-agent-clients` repository, preserving Stark's host
integration boundary. Stark has documentation changes only.

## Source and test identities

| Component | Identity |
| --- | --- |
| Stark API contract/base | `0bdcf4b3b4cc316fd7fd4f4ec2f6bc3a18268420` |
| Integration source | [`f319ac5021e9edd99a8d033d926c439e2bfa19e0`](https://github.com/jshahbazi/stark-agent-clients/tree/f319ac5021e9edd99a8d033d926c439e2bfa19e0) |
| Pi CLI / Pi AI | npm packages `@earendil-works/pi-coding-agent@1.5.2` / `@earendil-works/pi-ai@1.5.2` |
| Pi release source | `earendil-works/pi` tag `v1.5.2`, commit `b4afdddf44e059c67ccf5097e5317d913521b470` |
| Environment | Node.js `26.13.0`, macOS `26.7.2` (`25G332`), `darwin/arm64` |
| Authentication | Synthetic fixture credentials; no Keychain access |
| Native artifact/build/backend | None loaded or qualified |

The integration's [dated record](https://github.com/jshahbazi/stark-agent-clients/blob/f319ac5021e9edd99a8d033d926c439e2bfa19e0/docs/runs/2026-11-12-pi-foundation.md)
contains exact package-lock, parser, CLI, schema, settings and source/test
manifest SHA-256 hashes. Installed npm artifacts are bound by lockfile integrity
and file hashes; the upstream tag is a source reference, not a reproduced npm
build claim. Upstream packages retain their MIT notices and are not vendored.

## Implemented behavior

- Discover authenticated public `/v1/models` and `/v1/capabilities` using bounded
  loopback HTTP reads without redirects, proxies or automatic retries.
- Require the caller's exact public model ID, known finite API limits, streaming,
  cancellation and supported function-tool metadata. Recheck observable model
  metadata around discovery. Same-ID artifact replacements require separate
  artifact hashes; public metadata cannot identify them.
- Create a new `0600` Pi configuration containing `${STARK_API_KEY}` instead of
  a key. Optional JSON merging preserves the original input and rejects existing
  provider/output conflicts. No default Pi profile is installed or modified.
- Pin conservative text/function-tool settings, explicit terminal-reason
  enforcement, and an isolated pilot profile disabling agent/provider retries
  and automatic compaction.

No Stark API extension, managed grant, broker allowlist change, app integration,
prefix-cache modification, protocol adapter or distribution mechanism was added.
Manual clients continue to share the master key's upstream cache scope.

## Validation results

`npm ci --ignore-scripts` succeeded with exact locked dependencies.
`npm test`: **22 passed**, 0 failed, 0 skipped:

- 10 configuration/discovery checks.
- 2 actual bundled Pi CLI checks. Generated configuration loaded through Pi's
  real loader; disposable workspace read/edit/failed-command/successful-test
  rounds completed. HTTP and partial-output failures were not retried with the
  pilot profile.
- 10 Pi AI protocol checks. Split Unicode and tool arguments, parallel call
  correlation, whole input/cache accounting, terminal/length/error semantics,
  abort-driven HTTP teardown and recovery, and route stability were exercised.

The missing-usage check characterizes an **open gap**: Pi 1.5.2 reports success
with zero counters if upstream omits usage. This does not pass the plan's
missing-accounting requirement. Do not treat zero as measured usage; a fail-closed
client change or qualified guard is still required.

An independent read-only review reran all 22 checks and verified recorded
hashes. It found no blocking code issue. Its missing setup/recovery guide and
source-binding/flag-explanation notes were resolved before publication.
Stark's repository policy check passed. No native build or model suite was
needed for this documentation-only Stark change.

## Remaining qualification

No authenticated running Stark API was available to this session. No model was
started, key obtained or performance claim made. The native text/follow-up,
tool-quality, actual CLI/native cancellation, context/compaction, model/port
changes, restart, interleaved sessions and prefix-reuse matrix remain pending.
Fixture HTTP cancellation does not qualify the native worker. The first live
run must bind its own exact Stark build and model artifact/precision/template,
limits, MTP/thinking options, auth mode and aggregate outcomes.

Milestone 0 has a pinned Pi manual contract inventory. Milestone 1 has initial
tooling and model-free evidence; its exit condition remains incomplete. Other
clients and managed connections retain their original planned status.
