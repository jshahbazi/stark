# OpenWork branch integration with MTP main, 2026-10-31

This integrates PR #59 head `3fe2507431784ee3ce9d7d031d677466c949d4de`
with main `3eca459071e18f275a5e53fb482323a2dc2fcfd0`.

The hybrid backend retains main's MTP iterator, capability metadata and cleanup,
and passes the branch's generation-progress callback through the shared token
collector. Two progress-test request constructors now supply the MTP-off field.
Desktop timing profiles distinguish effective MTP draft counts, including
per-request API overrides and playground toggles. New regression cases cover
profile isolation and the inference fallback path.

The GitHub-applied HMAC change at the PR head updated capability grants only.
HTTP authorization now computes the same session-keyed HMAC, and grant hashing
occurs under the session lock after session validation. The original remote
commit remains in history. Incoming upstream provenance and validation records
are preserved.

## Validation

- Full locked offline converter suite: **5,299 passed, 88 skipped**.
- Focused timing/desktop/chat suite: **241 passed**; focused API/client suite:
  **64 passed**.
- Rust 1.98.1 locked generator release build, default and no-default-feature
  library suites, and all-target Clippy with warnings denied passed.
- Generator process-boundary harness: **6 cases passed**.
- Builder, release-version, signing and updater suites: **126 tests passed**;
  MTP benchmark harness: **23 tests passed**.
- Full converter Ruff lint/format, pinned Rust formatting, repository policy
  and whitespace checks passed.
- A bounded native smoke using an existing text model produced eight sampled
  tokens and nine ordered progress frames. No request or response content was
  retained. This checks the native progress path and is not new MTP numerical
  or performance qualification.

An initial full converter run was stopped after the incomplete remote HMAC
change was identified. The completed full run above uses matching grant and
authorization hashing. The smoke driver's initial bridge-event filter was
corrected to the native `progress` event before its passing run.

| Tested artifact or runtime source | SHA-256 |
| --- | --- |
| Generator executable | `0d3b498e4672c424b2b8717ac18921f9aabcdd71001e371908de88c0024268cb` |
| Hybrid generation source | `b4c0e8fc122f5a107612e19444e15fe247047338d20552eca4a07e54f2e5dc55` |
| Generation protocol source | `5fdb9e83509c1628361f417a45dc8f429846ec5f27bc75934b245dc136fd4f71` |
| Desktop backend source | `6b7d9593f29dbfce4edcd78e8174b3589921d600ac1c5b954a12a33bc947d08d` |
| Local API source | `68a89274be0a186a52c5c24e4346745274d8027d1f24538b13ea7335bbfbb99e` |
| Private smoke driver | `e25f687bc844f80f1b07a4d66889b3251bd6b2cbb5c2abe3aca1639f8235efd4` |
