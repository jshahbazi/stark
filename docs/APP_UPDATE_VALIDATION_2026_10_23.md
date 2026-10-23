# Integrated app update validation — 2026-10-23

This run qualifies the menu bar update implementation at `7e78f2092294af3746daacd1dbe04d5854ca873f`. It does not extend engine/model qualification or publish a release. The implementation was built on an Apple silicon Mac with the pinned Python 3.13.13 and locked converter environment.

## Automated checks

- 114 model-free checks passed across updater, app installation, shell installation, menu bar packaging, distribution, DMG and signing tests during implementation.
- The final updater suite passed all 10 checks after adding the final staged-payload recheck. It covers stable numeric version ordering, malformed/prerelease/foreign asset rejection, Mach-O architecture parsing, complete Swift app compilation, app replacement, ownership synchronization, legacy ownership compatibility, concurrent changes, waiting for shutdown and rollback of app/records after launch or record-write failure.
- Ruff lint/format checks and Git whitespace checks passed for changed Python files.
- A clean build from the named implementation commit succeeded, including embedded Python import checks, both workers' model-free invalid-argument probes, provenance recording and deep signature verification. The build reused the published v0.3.0 native workers; no model was loaded and no inference was run.

## Receiving the real release

The production Swift preparation code processed the published v0.3.0 DMG through its SHA-256/size check, pinned Developer ID team check, Gatekeeper disk-image assessment, image verification, read-only mount, app signature/Gatekeeper checks, bundle metadata/architecture checks, private copy and staged signature verification. The image was detached afterward.

The current helper then ran under that candidate's bundled Python against a disposable app destination. It staged and reverified the signed candidate, reported readiness, waited for a simulated parent process to exit, performed the real Darwin atomic exchange, and requested relaunch. LaunchServices invocation was intercepted for this disposable test; the test did not launch a second production app or access model data.

| Artifact | SHA-256 |
| --- | --- |
| Published v0.3.0 DMG | `185a43a2cbec4b4f21dd8c8feb5e1b624b2f6f9e4a080c3129f6a59ffa3fcd21` |
| Published v0.3.0 combined runtime archive | `42b9485d410b3c0853170cd27109ca9fb704585fa5ddc98095bb8a356ac951cb` |
| Clean local build's external app manifest | `209e61361c2303ab595814101817ccba04c695b8c811d7f11501a2c15f010636` |
| Clean local build's embedded app manifest | `292b042810464bbaf7ef14b336a0f1e0e1bea97d4fbaf8b6b32730cda0f2a5c1` |
| Clean local build's signed executable | `e4525a11c40ba340940b06c37c31b86aa0920d6f8a18975f53bd67f370c2823c` |

The local app is recorded as version 0.3.0 with a clean source identity and ad-hoc signature. It is a development artifact, not a replacement for the immutable published v0.3.0 release. Its private bundle and full manifest remain in ignored `artifacts/menu-bar/`.

## Scope and delivery limits

- The existing signed/notarized DMG release flow packages the updater as part of the normal complete app. No hosted workflow was enabled or dispatched.
- No release was published, no installed user app was replaced and no production application relaunch was exercised. The next signed app release must include this implementation; older installed apps need one normal installation to acquire the feature.
- App updates preserve models and leave the separate CLI runtime at its existing version. New CLI ownership records include the exact state path; older default-prefix records are recognized by equality of both ownership records. Legacy custom-prefix records and relocated CLI records require an installer refresh before app-only updates.
- Team identity is pinned to production team `S5BT2PRN5B`, allowing certificate renewal within that team. The updater requires GitHub's exact release asset digest and macOS signature/Gatekeeper checks. It does not use the optional release verification receipt as an update feed.
- Tests cover detected failures and rollback. They do not establish recovery from power loss at every filesystem operation or health of a newly launched model session.

## Pre-merge review follow-up

The requested second review found a failure-report ordering race: rollback could reopen the old app before saving the error that its dropdown should display. Recovery now writes the fixed diagnostic before reopening, with a regression test. A failed diagnostic write still permits the recovery launch.

The original automatic-quit installation change was also reconciled into this release-based branch. Both the combined CLI/app installer and source app builder stage and verify the replacement before quitting the exact installed executable. Normal AppKit termination is followed by bounded TERM/KILL escalation only if necessary. Ownership and app contents are checked before shutdown and again before exchange. Failed shutdown preserves the old installation.

All 120 model-free checks passed across the same seven suites on the reviewed follow-up, including an actual compiled AppKit fixture that exited normally with status 0 through the shared quit helper. Ruff checks passed. The previously recorded clean-build hashes continue to identify the earlier implementation commit; this follow-up is qualified by its new test run and does not overwrite that provenance. Full model startup health and power-loss recovery remain outside this validation.
