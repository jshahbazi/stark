# Pi pre-PR review and clean-build validation

Date: 2026-11-13. Owner: Codex. Verdict: **conforms for the reviewed manual Pi
Connections scope** after these corrections. Baseline: refreshed main
`80a04d7b056a9f9a2a6926a904e88b1c76e0c43c`, user-required Advanced Connections
placement, [integration plan](../../AGENT_CLIENT_INTEGRATION_PLAN.md), repository
instructions and [earlier installed validation](2026-11-13-pi-connections.md).
The user authorized review and PR creation, without merge or release.

## Independent findings and evidence

| Requirement | Defect | Correction / evidence | Verdict |
| --- | --- | --- | --- |
| Existing clean distribution builds work | Menu-bar packaging required a missing private checkout; staging helper was absent from recipe provenance | Immutable seven-file public snapshot with strict lock, optional hash-equal override, no ambient sibling lookup. Actual snapshot staging and relocated minimal recipe import/staging subprocess pass. Helper is in recipe/source hashes; CI covers it | Conforms |
| Check Installation repairs moved runtimes | Cached Node/Pi paths survived forced refresh; saved profiles could reference removed executables | Forced discovery and both saved executable availability checks; finder and missing/non-executable runtime regressions pass | Conforms |
| Manual key copy has bounded normal lifetime and preserves later copies | Timer never ran after normal app quit | Metadata-only clipboard lease clears its unchanged item on expiry/normal termination. Private pasteboard tests cover replacement and old-timer/new-copy separation; documentation qualifies crashes and force quits | Conforms for normal process lifetime |
| Every accepted API port remains usable | URL normalization dropped explicit port 80, breaking revalidation and guard matching | Explicit public port retained; canonical guard comparison; IPv4/IPv6 default-port cases pass and another port is rejected before transport | Conforms |

Client logic stays in the separate integration repository. Stark's snapshot is
mechanically copied from immutable Git blobs, license preserved, and never edited
as a separate client implementation. Default Pi settings, beginner setup and
existing OpenWork/OpenCode managed grants retain their boundaries. Version probes
in tests and the live harness use isolated offline profiles too.

## Local validation

- **54 client tests passed, none skipped**, including actual pinned Pi tools,
  streaming/failure cases, hidden Terminal PTY input, companion-fd lookup and the
  new runtime/default-port cases.
- Six-suite Stark run: **249 passed**. Final distribution/packaging rerun:
  **38 passed**, including one subsequently added clean-consumer case; 37 cases
  overlap and counts are not summed.
- A fresh independent reviewer verified all seven snapshot files against their
  lock and Git blobs, ran 39 package/distribution/native cases and three focused
  runtime/URL cases, and found no remaining blocker.
- Ruff check/format, repository policy and whitespace checks passed. Existing CI
  now covers snapshot checks on Linux and native Pi/clipboard checks on macOS.

Hosted Build `43437514182` failed at the menu-bar subprocess with sanitized
`menu_bar_build: command_failed (exit 1)`. The nested diagnostic was not emitted.
Missing Pi source is supported by its exact source/checkout inputs. The corrected
hosted run was pending during PR preparation; local success is recorded separately.

## Signed candidate and fresh artifact check

| Item | Identity |
| --- | --- |
| Code / build | Stark `e00401f24f367094dd4c1f01fbd8f996ede14708`, clean source; development build `20261113143108361356` |
| Manifest SHA-256 | `8aff86ec6d7f8e607c9898528d5e392a18a61ad50ce05a3cf46d97b5cf843c07` |
| Client pin / lock SHA-256 | `11e91d0a23c8a21dee5e6e0ba9b9833bacd17458` / `0cad936209775eec6ed66c911602e772171e2ac63fb91859b631344639aa9a4a` |
| Generation / template worker SHA-256 | `294f1be7e4ef5805737f726ed802c4cdf5dfe733c32380ef6f9905bdece7c95b` / `86c258a0c3218aa9054b9ffbc1da2d47afb6a2a73cc3c7b1cf8f1d74ac1a3114` |
| Harness SHA-256 | `1999be96cdbbd13b8b56db480547317c87b511f7c6cde60390809361d5974d01` |
| Aggregate SHA-256 | `3d223df1afed46ac2ecc50c956c87aba0ce09fed79de2e47e98242479dce8af7`; integration `docs/runs/data/2026-11-13-pi-pr-reviewed-summary.json` |

The Developer ID candidate built without `--pi-client-source`. All 21 Swift
source and 18 public resource hashes matched its manifest; deep/strict signature
verification passed. Native code inputs were reused and re-signed. This review
preserved the user's running installed build `20261113134611603911`.

Attempt `20261113-PIGUI-04`: **8/8 fresh real-model stages passed**, using the
candidate's bundled setup/launch entry points, installed Pi 1.5.2 and real
Rust/Metal worker. Startup, artifact registration/load, discovery, exact route,
text/history, read/edit/test/diff and failed-tool recovery passed. Private fixtures
were removed; `cleaned_up: true`.

Exact artifact: `mlx-community/Qwen3-4B-4bit` revision
`4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`, weights
`7c2af9a45e5b0d196f06a5cbfd6fc5cc95adf29df79885d0540f2206fb3d40d8`,
MLX affine Q4/group 64; parser `qwen3-hermes-v1`; context 40,960, Pi replies 1,024,
MTP/thinking off. Text-only: images N/A. No native engine/precision change,
numerical parity, broad coding score or speedup is claimed. Earlier lifecycle and
human TUI evidence keeps its original source identity. No private payloads or
credentials are published.

The separate integration repository has one branch, also its default. The useful
PR targets Stark main and includes the pinned reviewable distribution snapshot;
no integration history/default branch is rewritten to manufacture a PR base.
