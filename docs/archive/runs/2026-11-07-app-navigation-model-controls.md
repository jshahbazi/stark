# App navigation, setup and catalog validation

Date: 2026-11-07. Branch: `codex/app-navigation-model-controls`.
Base commit: `0f3b5165`. Host: Apple silicon, macOS 26.7.2,
Apple Swift 6.3.1. Python checks use the locked converter environment and
Python 3.13.13; repository policy uses its locked Python 3.12.13 script environment.

## Changes checked

- Parent-window navigation, including returning from a child, reopening an
  ancestor without cycles, and closing a parent before its child.
- Menu Start model and Stop model controls, exact stopped-model/mode restart,
  busy-operation guards, and removal of a remembered model.
- Centered interface controls, revisitable guided setup steps with prerequisites,
  and asynchronous OpenWork identity checks with coalesced refreshes and stale-result rejection.
- One-click continuation through remaining catalog pages, partial-result retry,
  visible scroll boundaries, unsupported-source explanations, and explicit
  conversion/chat/tool compatibility guidance.
- Immutable Hugging Face metadata cache reuse, retaining revision, repository,
  size, file stability and path-authority checks.

## Results

| Check | Result |
| --- | --- |
| Repository policy and whitespace | Passed |
| Native menu-bar controller harness | 131 checks passed |
| Native beginner setup harness | 10 tests passed |
| Native window/navigation/scan harness | 7 tests passed |
| Native desktop connection harness | Passed |
| Menu-bar builder suite | 46 tests passed |
| Catalog, support and desktop backend suites | 457 passed, 1 skipped |
| Catalog subset, included above | 111 tests passed |
| Ruff check and format check for changed converter files | Passed |
| Ad-hoc app build and embedded runtime verification | Passed |

The skipped backend parameter covers an inapplicable JSON-type coercion case
for the Hermes parser. Native UI snapshots used synthetic models and isolated
preferences; the menu layouts were visually inspected in running and stopped states.
No installed app was replaced, no model was converted or loaded, and no live
OpenWork session was exercised. These checks do not qualify a new model or parser.

## Metadata-only continuation timing

A warmed Qwen3 continuation at candidate offset 30 inspected a 30-candidate
window and returned seven entries. A control disabled the new disk-cache lookup
to retain the previous per-file metadata HEAD behavior. Each measurement ran in
a separate Python process. Both returned the same sorted entry-identity digest:
`399bc0e26ad3a07c6da735107a8e0ce1a13e6aa33ce5f0e6d52c72b45302797d`.

| Path | Elapsed seconds |
| --- | ---: |
| Previous HEAD behavior, warm cache | 1.328 |
| Immutable disk-cache reuse, warm cache | 0.939 |

These are single local observations after a warm-up. They do not establish
cold-cache speed, full-family completion time or a general latency percentage.
Fresh repository discovery and uncached metadata still require network requests.
Browsing requested metadata only, with no weight downloads.

## Source identities

Git blob identities for the runtime sources tested above:

| Source | Git blob |
| --- | --- |
| `apps/menu-bar/Sources/StarkApp.swift` | `df9156136ac89e7d15c3a625a304df0ee2916b23` |
| `apps/menu-bar/Sources/BeginnerSetupHost.swift` | `e5e7e700920014bd167a62079554beb1319b393a` |
| `apps/menu-bar/Sources/BeginnerSetupState.swift` | `82bd964ad82a8ae99333451a53b5309285276a19` |
| `apps/menu-bar/Sources/BeginnerSetupView.swift` | `1203f720749f1e726480553adccf936ffbde7ed2` |
| `apps/menu-bar/Sources/DesktopClientPeer.swift` | `076118a301408e6b347b8795b219d2b53922e642` |
| `tools/converter/stark_convert/model_catalog.py` | `82a6d975c98f34730fdfd089701b6ad4bdb5e38d` |
