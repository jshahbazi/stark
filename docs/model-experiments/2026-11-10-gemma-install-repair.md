# 20261110-G4INSTALL-01: Gemma installation repair

Registered and executed 2026-11-10; owner Codex. Compatibility/end-to-end attempt, completed for the installation repair and adopted within that scope. Predecessor: [installed-app test](../archive/runs/2026-11-10-installed-app-user-functional-test.md).

## Scope and acceptance

Reproduce the Google Gemma 4 E4B install failure from the complete cached source at revision `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c`. Repair conversion without weakening geometry, format or processor validation. Require successful conversion, app installation and image smoke; retain separate outcomes for each. No inference arithmetic or precision path change is planned. Text/cache/MTP qualification is inherited only for unchanged code paths, not newly claimed from this installation check. No new speed claim; matched performance is not measured in this attempt.

## Identity and controls

Baseline installed build source: `cb4cea9f18e59960b8f29bf0331d8a75ac86fee0`. Candidate branch: `codex/user-functional-test-2026-11-10`; final package source identity and validation receipt are recorded under Promotion boundary below. BF16 source weights; existing runtime precision policy is unchanged. Apple M5 Max, 128 GB shared memory. Cached source is used offline; generated weights remain ignored under runtime. Only owned converter workers may be stopped. Exception dumps and inference contents are not published.

## Trials

- Baseline E4B conversion failed before tensor work: the pinned config hash check opened the HF cache symlink with a regular-file-only reader (ELOOP).
- Candidate authorizes the canonical config target using the existing bounded source-root/cache-layout rules before the strict byte read. Outside-cache links still fail; the pinned configuration hash remains required.
- Full offline E4B conversion completed. The signed development app then installed the same repository through Add from Source with images enabled; Installed increased to six. Gemma loaded, identified both shapes/colors in the predecessor synthetic PNG, and passed the subsequent brief arithmetic check. Tools remained unavailable; no OpenWork tool capability was invented.
- Move to Trash removed the temporary model and restored the original five. Owned CLI reproduction output was deleted; shared source caches were preserved.
- Gemma and streaming regression tests: **264 passed**, including all five named image profiles with authorized cache links and rejection of outside-cache config links.
- Fixed diagnostic categories now travel over the private progress pipe: disk full, permissions, unsupported format/config, invalid checkpoint, generic conversion failure. No exception text or paths are sent. Failure state is invocation-local; cancellation precedes reporting.
- Desktop, vision installation, support-check, CLI and progress tests: **472 passed, 1 skipped**, with the installed native template helper in PATH. The first attempt lacked that prerequisite and failed fixture setup; it is excluded. Focused diagnostic checks: **12 passed**. Repository policy and Ruff passed.
- Pre-PR review found five stale test assertions for the changed converter return/category contract in native preparation, vision preflight and mode preflight. After updating those assertions, the complete locked offline converter suite passed: **6,954 passed, 94 skipped**. Skips comprise 93 opt-in external-reference metadata cases without their private artifacts and one inapplicable Hermes argument-schema case. Ruff, repository policy and whitespace checks passed again. No additional runtime change was required by this review.

## OpenWork follow-up and limits

The earlier refresh and long-task observations did not establish connector defects. The connector polls every five seconds. After twelve seconds following a same-model Decision-to-Chat transition, Qwen3 4B appeared as the current model without Refresh. A saved selection for the deleted temporary model required choosing an existing model. No connector code changed.

The previous compact-source session later displayed an invalid/incomplete tool-call message. That confirms unsuccessful completion, but does not establish whether model output or parsing caused it. A fresh Qwen3 4B task on the candidate created its exact disposable file; final task status is recorded separately below. Active streams have a 300-second inactivity deadline; stopping a two-minute run does not prove a timeout defect. No parser relaxation, output cap or model-quality repair is claimed.

## Promotion boundary

Final installed development build: `20261110182355209171`; manifest SHA-256 `37aaa569f86b18d2eac9b56a4142bf4da7779e5dbb4eb43739f61094170905c6`. The manifest binds base `fdfe346b43ccbe092e3a41a3e43437d933ffe26b` plus the dirty source hashes. The earlier UI-qualified build was `20261110181548823642` (same adapter fix, before diagnostics). Final package launch, original-five inventory, Dense start/stop, and absence of Gemma were checked separately. Final state: Advanced, Dense selected and stopped; OpenWork idle.

The isolated candidate OpenWork Qwen3 4B retest created the exact file but remained active through the bounded observation and was manually stopped. Full read-back completion remains unqualified. Model answer quality and general tool reliability remain unresolved; these are not represented as fixed by this converter patch.

Adopted on branch for source handling and bounded diagnostics. Model math, sampling, tool parsing, cache restoration, MTP, image geometry and kernels are unchanged. Existing source-bound runtime qualifications remain references; this smoke establishes no new speed/cache/MTP qualification. Both development packages reuse the unchanged installed engine workers; their manifests record source/worker identities and dirty patch provenance. No PR or release was published.
