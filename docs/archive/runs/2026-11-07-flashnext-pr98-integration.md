# Flash-Next PR 98 integration with main

Date: 2026-11-07. [PR 98](https://github.com/jshahbazi/stark/pull/98)
integrates branch parent `753884b83454cad069ce5b51c1711437b4417ed1` with
main parent `b9a91a94ea07313cc0558d556f26a300c6603ff1`.

The two conflicts were resolved by retaining main's native image request type
and full image protocol implementation alongside the branch's idle queue
maintenance hooks. Lily provenance retains both the newer vision records and
the Flash-Next experiment/idle-wake records. Vendored runtime sources match
main; the PR adds no shader or precision change.

The [integration receipt](../../benchmarks/data/flashnext-pr98-integration-2026-11-07.json)
binds the merged runtime files, lockfiles, executable and private validation
logs by SHA-256. The separately built Rust 1.98.1 release generator has SHA-256
`37b6f2154a3293d27041f35c45e58993e91b42f78ca2551371dc1a86b7f63ede`.

## Validation

- Native generation worker: 127 passed, two private benchmarks ignored.
- Locked, offline converter suite: 6,604 passed, 96 skipped.
- Python benchmark/telemetry checks: 77 passed.
- Real generator startup boundary: six cases passed.
- Repository policy, pinned Rust formatting and the PR whitespace diff against
  main pass. No unresolved conflicts remain.

The original 16-request ABBA measurements remain bound to `753884b8`. These
additional controls qualify the integrated text path separately, without
relabeling the earlier timing or numerical evidence.

Main changed `vendor/lily-core/src/qwen4exp/model.rs`, which was also the
benchmark corpus. Both new controls therefore use a private copy of that file
from `753884b8`, SHA-256
`33dcb30e762d63db4f16aaef21034fbaab93dd56dfd9b555e6da38f5db3238e3`.
This preserves the original input token hashes, sampling and seeded references.
The same pinned Lily Q4 XL artifact and prior weight receipt are used; shard
size/mtime/inode guards pass before and after the runs. Payloads are not
rehashed again.

After quiet admission and a fixed ten-second idle pause, all four measured
512-input/256-output requests pass host screens and match the original
same-route golden output hashes. Temperature 1, top-k 20, top-p 0.95, seed 42
and zero/two MTP drafts are unchanged. First-sample latency after idle is:

| MTP | Seconds |
| --- | ---: |
| Off | 0.2975 |
| Two drafts | 0.3042 |

The two MTP-on measured requests retain 108 rounds, 214 proposals, 147 accepted
tokens and 322 verified rows each. This is an integration smoke with all arms
retained, not a new sustained-decode comparison. Profiling is disabled. No
compiler work or overlapping model job runs during the model controls.

A separate six-request prefix control uses 4,608 input tokens and 64 outputs.
It reuses 4,096 prefix tokens after ten seconds idle in both MTP modes. Capture,
reuse and fresh-reference output hashes match each other and the original
pre-merge control. EOF shutdown exits with code zero without termination
fallback. These bounded checks preserve existing prefix restoration; they
do not establish general model quality or whole-model numerical equivalence.

Native artifacts stay under ignored `runtime/flashnext-moe-2026-11-07/`.
Published records contain aggregate measurements and hashes only.
