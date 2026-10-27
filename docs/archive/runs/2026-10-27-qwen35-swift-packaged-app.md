# Qwen3.5 Swift attachments and local signed-package qualification

Swift attachments and local signed-package image execution conform to the
requested stage-5 milestone. The delivered GUI was also launched and exercised
against a fresh application library. Public notarized distribution remains
pending; this candidate does not pass Gatekeeper assessment.

## Reviewed source and acceptance boundary

The clean implementation revision is
`8fe2b2eb941151ea10230f43b09ddfba854efc8e`. This record is a documentation
follow-up. The original native multimodal plan's stage-5 acceptance requires
attachment selection/drop/paste, preview/removal, capability-aware validation,
progress/cancellation/errors, the common media pipeline, and a packaged full
artifact completing an image request from a clean installation.

The exact model is the pinned Qwen3.5-0.8B BF16 artifact from the
[signed-worker execution record](2026-10-27-qwen35-signed-worker.md), running on
the same 40-core Apple M5 Max, Metal family 1010 with MXU, macOS 26.7.1.
Native computation is unchanged. This run adds application and packaging
evidence; it does not replace the earlier source-bound numerical results.

| Requirement | Implementation and evidence | Result |
| --- | --- | --- |
| Attach, drop, paste, preview and remove one PNG/JPEG | Swift composer controls; actual file selection/preview/removal in the delivered GUI; native PNG/file-URL provider and cancellation tests | Conforms |
| Enable images only for a qualified running model | Negotiated `private-file-v1`, five artifact hashes, executable/runtime catalog; normal packaged execution reports images available | Conforms |
| Shared bounded media path | Swift opaque references; descriptor-pinned private claims; unchanged 1 MiB JSON limit and 4 MiB image limit; native preprocessing | Conforms |
| Progress, recoverable errors and cancellation | Controller and delivered-GUI invalid-image → valid B, Cancel → Run → valid image; image reset and staging cleanup | Conforms |
| Fresh packaged execution | Isolated copied signed app, empty registry, embedded interpreter/worker and ordinary catalog; separate delivered-GUI launch with fresh application data | Conforms locally |
| Preserve notices and source provenance | Current locked Cargo/Rust notices and eight unique adaptation notices delivered and independently hash-checked | Conforms |
| Public notarized installation | Developer ID signatures verify; Gatekeeper and stapled-ticket checks fail; release workflow has not run for this source | Pending |

Physical drag and image-clipboard gestures were not automated in the delivered
GUI. Native provider tests exercise their production callbacks, and the SwiftUI
drop/paste registrations compile in the delivered executable. No claim of a
physical gesture replay is made.

## Delivered identities

The existing signing helper produced Developer ID signatures for team
`S5BT2PRN5B`, hardened runtime and secure timestamps. Only the outer application
was resealed after updating its Python package, Swift executable and notices;
the qualified nested worker signature was preserved.

| Identity | SHA-256 |
| --- | --- |
| Delivered generation worker | `741a3ea7b39a80fcd9c9a6fe9deb0cee1d1b383bd40618479ec49a8a58d91ede` |
| Delivered Swift executable | `2a418618b83f139e4e725dcdccefc7a1e694684508b724d715330c220ae2ce73` |
| Canonical inventory, 3,341 files | `6e4edb509e91135876859d813eaf2b03d57ef750785603ad1f08e92c0819bad2` |
| Image qualification catalog | `2222b577fe5a88407ad0377e44cfbdbaeb4af7e0a51c03768a0d8f679bb5316d` |
| Menu-bar manifest | `1bdfba211edc81b1806643f020ea68a58651aeac26f104ec23801a8847b6a8f2` |
| Signed runtime manifest | `b6a01fb57fa44e01500d6db85f21dea6e91b7ea790c0f91aaccf88a8d4bfb703` |
| Swift controller source | `d1db47df16a993a80bca5aae90c359824a582dba2ec8a96c514f9d2e88e18ada` |

All 1,316 source files recorded in the delivered manifest matched the clean
implementation checkout. Models, generated applications, fixtures, and native
execution artifacts remain ignored. Reports contain aggregate predicates and
build hashes, with no prompts, answers, image data or credentials.

## Execution and independent verification

`scripts/qualify_image_app.py` copies the candidate into a fresh private
installation whose path contains spaces. It verifies the full inventory and
signature, imports only the copied package using its embedded isolated Python,
and runs two successful image conversation turns through the production desktop
pipe. The checked process has no Rust, uv, source checkout or system Python in
its executable path. Native image usage is present, staged files are removed,
current notices match their hashes, and the bundle remains unchanged. The
ordinary catalog gate is used without a qualification override.

The production Swift controller was additionally compiled from its exact
packaged source and given the copied application's resources and a private
model-data directory. It passes preview, first image, same-image follow-up,
New Chat, invalid-image recovery, Stop/Run recovery, token metrics and zero-file
cleanup. An independent replay produced the same aggregate report. This harness
injects the data directory explicitly; changing `HOME` alone is insufficient to
redirect Foundation's home-directory lookup on this Mac.

After explicit user authorization to close the running app, the delivered
`Contents/MacOS/Stark` executable itself was launched through the native UI.
The original application-data folder was preserved by renaming it, and a fresh
empty folder was used for the test. The actual GUI registered the converted
artifact, enabled Attach Image after Run, previewed and removed an attachment,
completed an image turn and follow-up, cleared the image with New Chat, recovered
from a malformed image, and completed another image request after Cancel and
Run. Four successful image replies were observed as predicates only. Both owned
staging directories contained zero files before quit and after cleanup. The
signed bundle inventory remained unchanged.

The original library folder and preferences were restored, and the installed
application was reopened with its original selection. Candidate test data was
retained separately. Transient accessibility/capture failures required automation
reconnections; successful outcomes were verified from fresh UI state.

| Private aggregate evidence | SHA-256 |
| --- | --- |
| Packaged backend replay, independently repeated | `3455fba79eed1800d3ecc477deda6ba940d24907ebf58eea776b307bb0123938` |
| Exact-source Swift controller replay, independently repeated | `6890db8512d7052643ace9b7d73b8318d6200667c5a3c839ed371a1a744a7d9d` |
| Delivered GUI checks | `52034248f7f38cf4649562da190d6bc2e45ef6ef583e29a4f424ee57c0481fac` |
| User-state restoration checks | `7f480ee037cd4443c16a6d795d5b1f5f28f94b5ec4c8ea677f0e1275091c4ffd` |
| Independent delivered-notice/source audit | `e4e146ba18172044337a8dde1952c23a2edb2877119106f82c162cd4add2857f` |
| Independent signature/code comparison | `ed42c25faec3ea673d53d07172d0dbda2607e815552acd2fc4f31e1f493f47cb` |
| Signature/Gatekeeper/stapler policy status | `cb1dad1eef87979bb51c3eb5decb686cb01a97195f470c0710a9785d29a2b367` |

## Corrections found during double-checking

- Slow drop/paste callbacks could overwrite newer attachment selections. Tokens
  now invalidate older callbacks; loading disables Send and supports cancellation.
- Private-file cleanup could leak an exception from shutdown. Cleanup now uses
  fixed errors, always attempts descriptor closure, and attempts each remaining
  resource even when another cleanup fails.
- Ready could be published before its operation released admission. A
  deterministic test reproduced immediate request B being rejected by the old
  implementation. Completion events now retain their epoch and are published
  after job/install ownership is released, under the same lock. Twelve new tests
  check ordering, immediate B, cleanup, API wakeups and stale-event suppression.
- An older base CLI supplied incomplete notices for the overlaid vision workers.
  The builder now collects current Rust 1.98.1/Cargo.lock notices and adaptations.
  Independent inspection verified 296 notice files, 136 dependency records, eight
  unique adaptation notices, the four locked image-decoder dependencies, 13 Rust
  standard-library notices, and converter-wheel notices.

An early final-controller evidence run failed in the private runner's replay-tag
validation after the Swift checks had completed. Its failure logs were preserved;
the runner was corrected and the same delivered bundle passed fresh replays.
No runtime result was inferred from the failed invocation.

## Final checks and release boundary

- Converter/Python: **4,404 passed, 88 skipped**.
- Swift controller: **24 passed**, including actual native provider callbacks.
- Packaging, installer and updater checks: **101 passed**.
- Focused bridge/desktop checks: **193 passed**.
- Direct connection JSON/curl and private-pasteboard checks passed.
- Ruff, repository policy and Git whitespace checks passed.

No notarization credential profile was found locally under `stark-notary` or
under the supplied Developer ID identity label. The profile name comes from
the existing GitHub release workflow, which creates it in a temporary CI
Keychain. The supplied Developer ID certificate successfully signed this app.

The existing release workflow is restricted to `main` and builds/signs its own
workers. Its resulting executable hash requires its own applicable image
execution qualification before image capabilities can be enabled. A notarized
release cannot be inferred from this local candidate, and a later re-signing
cannot silently inherit its catalog record. No release was dispatched or
published, and the installed app's bytes were not replaced.
