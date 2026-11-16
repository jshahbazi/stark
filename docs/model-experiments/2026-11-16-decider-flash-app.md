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
