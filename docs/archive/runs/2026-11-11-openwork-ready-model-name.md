# Ready-screen model identity and merge validation

Date: 2026-11-11. Source: merge `8894e98289d990fcadb8e2a153789fa2bb68cae8`,
integrating the beginner setup branch with main `8aefb539`.

## Behavior checked

The guide now obtains the running model's readable name from installed metadata,
including when no recommendation report exists. Its precision/version appears
under the name in the Ready screen and running-model sidebar. Runtime model IDs,
API IDs and source links retain their existing identities. The merge reuses
main's readable-name formatter and preserves its chat-footer geometry,
recommendation benchmark policy and current-memory admission checks.

The installed `/Applications/Stark.app` was reopened, the previously selected
model was started, and the guide advanced automatically to Ready. The native UI
and screenshot showed **Qwen3.8 Flash Next**, followed on the next line by
**Lily q4-xl (Q4 GDN; Q8 attn/shared/head/embed)**. The sidebar showed the same
name and version. No workspace/model confirmation was required. No prompt was
submitted or OpenWork tool task executed. The model was left running.

## Validation

- Native beginner setup: 18 tests passed.
- Native menu-bar controller: 147 tests passed.
- Native windows, handoff, desktop connections and connector installer: 15 tests passed.
- Focused guided/catalog/installed-model suite: 315 tests passed.
- Broader hybrid/memory/benchmark/desktop/browser suite: 464 passed, 1 skipped.
- Repository policy checks and whitespace checks on changed implementation paths passed.
- Generation, scoring and template workers built separately with Rust 1.98.1
  from current main in a distinct target directory. The signed app package
  passed its embedded imports, template render, CLI and signature checks.

An initial package check rejected an older template helper whose request-budget
contract was limited to 64 MiB. Rebuilding the matching helpers resolved that
packaging failure. This run checks configuration, startup and UI behavior; it
does not establish numerical parity or task quality for newly built workers.

Installed development build: `20261111174055904193`.
Installed Stark executable SHA-256:
`2153e2e2611b92df3a0b6088944b5875e4f20b70a6b5e6065ed79be31b41fbe1`.
