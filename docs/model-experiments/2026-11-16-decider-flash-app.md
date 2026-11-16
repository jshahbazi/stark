# Decider Flash app integration

Attempt ID: `20261116-DF13APP-01`. Registered 2026-11-16 before app/model runs.
Branch: `codex/decider-flash-v13`; baseline `d2606186`.

## Objective and acceptance

Expose the qualified native Decider Flash v1.3 BF16 artifact in the Stark app's
model setup and Decision workflow, including one image through the existing
private attachment transport. Preserve the standalone native result and its
[failed attempts and qualifications](2026-11-16-decider-flash-v13.md).
No PR, release publication or unrelated integration change is authorized.

Hypothesis: the existing app decision controls, private media leases and worker
lifecycle can route to the dedicated native Decider worker without changing its
inference arithmetic. App answer labels must retain order and use the saved
checkpoint temperature; ordinary language-model Decision scoring stays separate.

Fixed acceptance gates:

- Exact prepared artifact import and a pinned-source installation/catalog route
  retain images by default and expose decision-only capability. Refuse Chat,
  generation, tool and chat-API use of Decider with useful guidance.
- App Decision accepts text/question/answers and an optional single PNG/JPEG.
  Image-only state is allowed for Decider. Inputs remain bounded by 4,096
  expanded positions and existing source-image/request memory limits.
- Match app/backend results against the same official source/reference on
  bounded text and image controls: 100% top-choice agreement and maximum
  probability error <=0.02. Distinguish native/reference qualification from
  frontend mapping, packaging, installed status and full UI execution.
- Reset, Stop, mode/model changes, failure and cancellation clear staged media
  and terminate/reap owned workers as appropriate. No stale result publication,
  image path injection, silent truncation, prompt/response logging or cache reuse.
- Model-free tests cover capability negotiation, dispatch and calibration,
  invalid media/inputs, decision-only modes, registration/install, private file
  cleanup, and existing generic Decision/Chat behavior. Build the app with the
  dedicated default-feature Decider binary, separately from optimized scoring.
- Use the app's native GPU telemetry transport when available. Keep Decider's
  readout/attention/recurrent precision and one-image limits unchanged.

## State, precision and resource policy

Use source `perplexity-ai/pplx-decider-flash-v1.3` revision
`d06fb3f2d97e0a52325a0d463c450cf0338f44fe`, existing private `native-v2`
configuration `d859a249a3089252f164c755d2a584e41c1041ab91dc32751a742ddad29e8337`
and weights `c3ab2961a63c3342ff3390b41036c6029d62095414db4cffbd59a8944c754c97`.
Bind new worker/bundle hashes and any changed artifact separately.

The native model has no cross-request KV, convolution, recurrent or result
cache. Bidirectional attention blocks causal prefix reuse. It has no supported
generation, MTP, function-tool or Locate path. Existing image-loaded text timing
was inconclusive; app integration does not turn it into a performance pass.
Performance optimization is outside this change.

Machine: Apple M5 Max, 128 GiB RAM; verify current OS, free space and build
identities for new runs. Models/GPU checks run sequentially. Initial free disk is
approximately 18 GiB; use existing source/prepared artifacts, avoid duplicate
model acquisition and reserve space before app staging/builds. Do not alter
another worktree or app's active requests. Preserve source/failed-run receipts.

All fixtures, images, transient attachments and runtime receipts remain private
and ignored. Publish aggregate outcomes and hashes only. Reference responses
stay in memory/private pipes. Never log exceptions or request/response content.

## Attempt history, results and disposition

Implementation and bounded packaged-backend qualification passed. Canonical
results and build identities are in the
[app run record](../archive/runs/2026-11-16-decider-flash-app.md).

- Added the dedicated app adapter, Decision-only registry/catalog/source setup,
  checkpoint-temperature readout, private one-image attachments, native GPU
  telemetry and separate default-feature worker packaging. The Swift controls
  preserve answer order and reject stale replies; generic scoring stays separate.
- Review caught and fixed the desktop catalogue's language-only filter, which
  would have hidden the new Decision family. Fixed its image-progress phase to
  match the Swift controller. Source preparation now safely reads metadata from
  standard Hugging Face snapshot/blob links without another weight copy.
- Local raw-source preflight was resource-deferred by the 15 GiB free-space
  margin plus planned output. No duplicate weights were downloaded/converted.
  Prepared-artifact import was exercised through the actual packaged backend.
- App control run 01 was invalid: a relative model path was resolved from the
  relocated backend's private directory, so import failed. Run 02 imported the
  model but was invalid because the harness used `id` instead of the existing
  `model_id` Run field. Both stopped before native model inference. The harness
  fixes are separate from the unchanged packaged app implementation.
- App control run 03 passed all five reference choices (two text, three image,
  including an image-only state), maximum probability error 0.000790242319442.
  Repetition after images and reset was exact. Attachment consumption,
  reset/stop/shutdown cleanup, Chat refusal and 13 GPU memory events passed.
- Initial complete converter check: 7,875 passed, 95 skipped, one failure in an
  input-budget test double missing the new shared request-ID validator. Updated
  the test double; no production fix was needed. Focused recovery/Decider tests:
  23 passed. Final complete rerun: 7,878 passed, 95 skipped in 338.26 seconds.
- Separate packaging/signing checks: 99 passed. Swift Decider controller checks:
  10 passed; existing controller regressions and full Swift typecheck also passed.
  Repository policy, Rust formatting and Python lint checks passed.

Disposition: adopt on this development branch with the image tower retained.
The signed bundle was tested after relocation with an isolated model library;
the user's installed app and library were not replaced. Full manual GUI use,
fresh multi-gigabyte download/conversion, release/notarization and performance
promotion are not claimed. Native `DF13-03` evidence remains unchanged.

## Installed app image check: 20261116-DF13APP-02

Registered before installation and GUI model execution on 2026-11-16, following
the user's explicit request to install the signed build and test an image in the
actual app. Baseline branch head: `f1071817`; use the unchanged signed build
`20261116192037701994` qualified in `DF13APP-01`.

Acceptance: verify the installed bundle matches the tested build, import the
qualified prepared artifact using visible app controls, load Decision mode,
attach one private synthetic image, submit an image decision and observe a
valid matching top choice in the visible result. Check image removal/reset and
Stop through the app. Keep fixture contents, probabilities and screenshots
private; publish aggregate observations and identity hashes only. No timing,
fresh-download, broad task-quality or release claim. Push the record on the same
branch; do not open a PR. Current state: passed with the import-default fix below.

The original signed bundle installed successfully and its full 4,097-file
inventory matched `DF13APP-01`. Initial CUA attachment failed because the app
starts with a menu-bar popup; the user opened Manage Models to expose a normal
window. Actual Add from Source import then revealed a gap in the earlier
backend recipe: the visible form omits `artifact_mode` and inherits Chat's
`support_mode=text`. The typed Decider artifact was detected but refused that
inherited purpose. No GUI model inference had run.

The default-purpose route now selects Decision for an auto-detected typed or
pinned Decider source when no explicit artifact mode was chosen. Explicit
conflicting modes still fail. Added GUI-shaped import regressions and changed
the packaged validation recipe to exercise that same default payload.

Implementation `9bef02aed7354e5f8b79597b91e32b452eb0e59c` was rebuilt from a
clean source tree, Developer ID signed and installed at
`~/Applications/Stark.app`. The complete installed inventory and signature
matched that newly tested bundle. The revised packaged reference run again
passed 5/5 choices, including three images, with maximum probability error
0.000790242319442; exact repeat/reset and cleanup passed, with 14 GPU events.

The actual installed GUI passed Add from Source import, automatic Decision
selection, loading and a single image-only colour decision. The visible result
selected the expected answer and rendered three probability rows and the saved
calibration caption. Remove Image cleared the preview/result; Stop cleared the
worker and GPU display. Private staging was empty afterward. Synthetic inputs
were cleared and the model remains imported as `Native V2`, stopped. There is
no separate Decision Reset button; native reset is covered by the packaged
backend run, not claimed as a clicked GUI control.

Focused regression tests: 59 passed. Common desktop regressions: 363 passed,
one skipped. Repository policy and lint/diff checks passed. Earlier full-suite
results remain bound to the earlier source. Exact new bundle/receipt hashes and
GUI scope are recorded in the [installed follow-up](../archive/runs/2026-11-16-decider-flash-app.md#installed-gui-follow-up-20261116-df13app-02).
No PR, notarization or release was created.

## Naming follow-up

Prepared Decider imports now use `Decider Flash v1.3` rather than the source
folder's basename; the focused desktop suite passed 24 tests. At the user's
request, the existing saved `native-v2` entry was renamed as well. After an
authorized app restart, Manage Models visibly showed the canonical name and
five retained entries. Only the target entry's `name` field changed; its ID,
path, source, artifact and other saved settings were preserved.

The installed app had independently been replaced by another branch's newer
build. That build was retained for the naming check; this follow-up does not
transfer Decider qualification to it or claim its catalogue contains Decider.
The Decider capability and catalogue results above remain bound to the recorded
Decider builds. PR #169 now contains this branch's implementation and evidence;
main/release adoption remains separate.

## Catalogue/link follow-up: 20261116-DF13APP-03

Registered before the new build and UI checks. The user reported the missing
Hugging Face link and catalogue entry in the independently installed app from
another branch. Restore this branch's signed app and verify the actual row and
catalogue, rather than transferring UI claims to the other build.

The prepared artifact has an older receipt that intentionally does not satisfy
the newer display-provenance binding. Add a separate optional upstream-repository
link for the typed Decider family; keep its unbound revision/group identity and
qualification unchanged. Do not modify the model or its historical receipt.
Acceptance: the installed row visibly links to the official Hugging Face repo,
Model Catalog visibly includes the separate Decider family and its pinned BF16
entry, and the installed bundle matches the newly signed build. This is a
display/installation check; no numerical or performance promotion. Pending.
