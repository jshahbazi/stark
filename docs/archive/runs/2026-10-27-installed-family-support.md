# Installed family-support UI qualification, 2026-10-27

This run starts from `e8f6e2fe512692b14cb3b61be3f528a3af0e2841` on
`codex/qwen35-vision-artifact-contract`. The user requested installation of the
tested build and verification of the actual family list and download warnings.
The destination was the running `~/Applications/Stark.app`.

## Installation and correction

The exact previously qualified candidate was installed, using the existing
local-development builder's staging, strict signature verification and atomic
exchange. A private verified backup was retained before each replacement.
Candidate identity and Developer ID team `S5BT2PRN5B` were checked separately.
The installed bundle was matched to the candidate by its complete tree digest,
executable and application manifest. The updated app launched successfully.
The release updater and its notarization requirement were unchanged. This is a
Developer ID local build with hardened runtime and a secure timestamp; it is
not notarized.

Actual GUI testing exposed one presentation defect: changing the requested
source cleared the structured support report but left the old generic error in
red. The controller now tracks ownership of installation errors and clears only
that error when the selection resets. Assignment of a later runtime or image
attachment error revokes the old ownership, even when its text is identical.
An active installation cannot be reset through this path.

The corrected candidate was rebuilt, signed, verified, installed and launched.
All three warning cases and selection resets were then repeated in that final
installed build. Native worker code and all 55 converter source files are unchanged
from the previous family-support qualification.

## Actual installed-app checks

The app was operated through its real macOS controls using computer use. No
chat message was submitted and no generated model content was logged.

| Check | Result |
| --- | --- |
| Supported-family sheet | All 13 architecture entries present with Text/Decision/Image columns; scrolling and disclosure details work. |
| Qwen3.5 row | Conditional image support and BF16, reviewed-processing and local-check requirements visible. |
| ModernBERT row | All app modes unavailable; expanded detail identifies classifier conversion only. |
| Unsupported family | Public `Qwen/Qwen3-30B-A3B` stops with no implemented converter; no false alternative mode is offered. |
| Unsupported image mode | Public `Qwen/Qwen3-0.6B` with images stops with an image-support explanation and Text/Decision alternatives. |
| Inconclusive metadata | A private local checkpoint fixture without architecture metadata stops with a distinct metadata explanation. |
| Old warning reset | Changing the repository, source type or image option removes the prior structured report and generic red error. |
| Running-model retention | An existing Qwen3 model retained the same worker PID through all three rejected installs. |
| Library and downloads | The two registered models remained byte-identical; managed model inventory was unchanged; no new weight or partial-weight files appeared. The new MoE cache contains only 1,926 bytes counting metadata blob and link. |
| Cleanup | Qualification worker stopped, original selected model/mode restored, updated app left running with the family sheet open. |

Final exact-source controller checks: **46 passed**, including three new
regressions for installation-error reset, retention of later unrelated errors
and the active-install guard. All Swift sources typechecked and the candidate
compiled successfully. Builder checks: **23 passed and 12 subtests passed**.
Scoped Ruff, formatting and repository policy checks passed. The previous
4,803-test Python result remains source-bound to the unchanged converter; the
full converter/native suites were not repeated for this Swift-only correction.

Private backup bundles, aggregate receipts and fixtures remain under ignored
`runtime/installed-family-support-validation-2026-10-27/`. Earlier qualification
records were preserved. These checks establish installed UI behavior and local
installation identity; they do not establish new model quality or a notarized
release.

## Final identities

| Item | SHA-256 |
| --- | --- |
| Installed application tree | `1ca7b9d0ca5731ced6a7b3e0675ab9b9cc1673e99fdcd726e936e76dda6e8c90` |
| Signed Swift executable | `c1fb9013d680fbf6469397bca62e9359444a755eb64a78bfae54296a25d2f985` |
| Application manifest | `4fd62f4a32db70f936cfe542f79f2c39a4e16488a72d2c0d34dad09bd8486a17` |
| Signed generation worker | `623c741d6ab83a3c5c7146bafe15b34be82d6acaf0da109e1aae6696058ebf54` |
| Candidate signing report | `ad837110305a9483b9c610d3806bc06662e0525be5f36ca1b6750764acb5b161` |
| Installation identity/retention report | `2385b7f47d9ef8b7a68a9b22896b2f41956ca57815858ed5f097dd32dbe955b1` |
| Final installed-state and GUI aggregate | `bf417b9ead4eabcdb195f608b87853f5b6a60c240a262ca02cc879a431f1696d` |
| Final controller log | `7463e385e1a9b0d3bb54c9a8a1c2a56325ea1c1f2df7c948d742ac968cab10e7` |
| Builder-test log | `fd9b6d10bcb848ee0fe595fcea7ebebe2466ca76672d7f77410a4a3dd72167f6` |
| Final unsupported-family retention check | `ff2605c2bac5b5f0c3e38539dd4f9dd03377636d7595f2c672c74493f656cfb0` |
| Final unsupported-image retention check | `6d0013a35194c0c91e8c4db0af7fa24909b42cb75084732197ab40faece70aad` |
| Final inconclusive-metadata retention check | `a0d829a227661512ebe0277e2273219ea42e69a7a37e745277ac9055ed7d020d` |
