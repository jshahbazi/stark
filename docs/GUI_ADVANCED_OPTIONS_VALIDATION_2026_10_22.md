# GUI advanced options validation — 2026-10-22

This change adds a separate Advanced Options panel for system prompt, context size, output token limit and thinking mode. Apply and New Chat resets history with the current worker; Cancel discards edits. Settings remain in memory for the app session.

## Integration

The feature branch starts from the updated `origin/codex/install-progress` branch and includes `origin/main` at `695f1eb2`. A fresh fetch and merge confirmed main was already an ancestor. Uncommitted installation-progress edits in the original checkout were preserved by using a separate worktree.

Packaging needed a compatibility repair: its standalone loader executed runtime provenance as a script after that module gained package-relative imports. Shared manifest construction now lives in dependency-free `_build_metadata.py`; runtime callers still supply the existing checked file hash function. The builder and installed package verify matching provenance. Prior validation receipts remain unchanged.

## Checks

- Converter suite: 3,950 passed, 88 skipped. Targeted desktop suite: 66 passed, including option rendering, strict type/range checks, atomic rejection, model-limit clamping, history reset, worker reuse and settings privacy.
- Menu bar packaging tests: 15 passed. Provenance tests: 20 passed. Swift type checking, repository policy, Ruff and diff checks passed.
- Full local app build passed, including embedded package provenance equality and signature verification.
- A headless harness compiled the current Swift source, replaced only the application bootstrap and library location, and drove the actual EngineController against the bundled backend and local Qwen3-0.6B model. All 17 checks passed: readiness and worker limits, option acknowledgement, four-option application, native reply and output budget, invalid-setting recovery without losing history/options, New Chat retaining options, settings application clearing history, and Configure/Stop/Run ordering. No stderr or transcript content was retained. Aggregate evidence stays under ignored `artifacts/advanced-options-validation/`.

Swift source SHA-256: `3cd3ae758df99bbe0d0753ef5fd4c3113b840e895369f4099067c919a295ba93`. Existing reused generation worker SHA-256: `f12fd9849d81610170c2b893e98360b25ab553310cd5276e266f6d5e583e320f`. The worker was reused from the installed app; this run does not establish a fresh engine build, answer quality or performance qualification.
