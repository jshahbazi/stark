# Background app upgrade verification — 2026-11-12

## Scope and build

Verify automatic stable-release discovery and preparation in the installed menu
bar app without opening its menu, then observe a real recurring macOS background
activity. Installation of the downloaded release is outside this check; the
updated local app must remain installed.

- App source: `94c121f25425ab2f52872026bd74bf7ab42291a4`, clean at build time.
- Installed destination: `/Applications/Stark.app`.
- Development build number: `20261112195832756812`.
- Signed app executable SHA-256:
  `fe4472cf260ff46bab9853193087c1e37495bc33c5bda058bcd2123fa25e2022`.
- macOS: 26.7.2, Apple silicon.
- Signing: Stark Developer ID, hardened runtime and secure timestamp. Deep,
  strict signature verification passed. This is a local development build,
  not a newly notarized or published release.
- Build inputs: the previously installed 0.20.0 runtime and its generation,
  scoring and template workers, supplied explicitly to `build_menubar.py`.
  The builder refreshed the application and converter overlays and signed the
  staged bundle before replacing the installed app.

## Changes checked

The updater now uses `NSBackgroundActivityScheduler` with a 900-second interval,
60-second tolerance and utility quality of service. Its completion handler waits
through asynchronous download and verification. A generation identifier rejects
callbacks queued before stopping or restarting the scheduler.

Review found that applying the menu's strict 900-second cooldown to scheduled
callbacks could suppress a check delivered within the early tolerance window.
Scheduled callbacks now check directly; the existing busy guard still prevents
overlapping requests. Menu clicks retain their cooldown.

Only constant updater lifecycle events are logged. These events contain no
prompts, responses, token data, credentials or exception details.

## Automated checks

- Repository policy check passed.
- `scripts.test_app_update`, `scripts.test_install_app` and
  `scripts.test_build_menubar`: 105 tests passed.
- After the early-callback correction, the production controller regression
  test passed again. It delivers a scheduler callback at 840 seconds and asserts
  that metadata fetching and preparation occur before scheduler completion.
- The production adapter test covers scheduler configuration, waiting for
  preparation, and invalidation. Other controller checks cover stale callbacks,
  stop/restart, failure recovery, preparation reuse and button state transitions.
- Direct optimized compilation and the final signed app build passed.

## Installed app run

Times below are Eastern Standard Time. Earlier UI inspection runs were discarded.
The final run began after confirming that Stark had quit, then opening the app
once from Finder. No Stark menu action was performed during launch, metadata
fetching, downloading or verification. After readiness, the user opened the menu
and reported seeing **Upgrade**. A subsequent UI inspection confirmed that button
and the 0.20.4 ready message; the menu was closed again before the scheduled
check. Opening the menu at that point was inside the cooldown and did not start
another metadata request. The process remained the same throughout the run.

| Time | Observed updater event |
| --- | --- |
| 15:01:07.886 | Background metadata check started at launch. |
| 15:01:08.141 | Metadata check completed; download and verification started. |
| 15:01:43.952 | Download and verification completed; ready to upgrade. |
| 15:14:33.914 | Real scheduled background activity started; a new metadata check began. |
| 15:14:34.155 | Scheduled metadata check completed successfully. |
| 15:14:34.171 | Scheduled background activity finished. |
| 15:14:34.174 | macOS registered the next check for the 15:29:02–15:31:02 delivery window. |

The prepared candidate identifies itself as stable release 0.20.4. Its executable
SHA-256 is
`5f23ce259093bb8f32e76401e17f4841a532b52d8e317e67a0a80ef4ffa00504`.
Reaching the ready state exercised the live DMG checksum, signature, Gatekeeper
and app metadata verification path.

The system scheduler's independent `dasd` records identify the installed app
process and corroborate activity start, completion and resubmission. The menu
remained closed during this scheduled check. The metadata matched the prepared
candidate, so the updater reused it without another download. This verifies both
unattended launch downloading and recurring background checks; it does not claim
that the scheduled check downloaded a second copy.

The final UI observation showed an enabled **Upgrade** button and
**Stark 0.20.4 is ready to upgrade.** The menu was then closed. No upgrade action
was clicked, preserving the newly installed local build. Raw local evidence
stays under ignored `artifacts/updater-verification/`.

## Limits

This run qualifies this installed build on this Mac while awake. macOS may defer
background work while asleep or under system pressure. It does not establish
sleep/wake behavior or behavior on every supported macOS version. The downloaded
release was not installed during this verification.
