# 20261109-LAUNCH-RECHECK-01: image-resident Lily text controls

## Identity and state

| Field | Value |
| --- | --- |
| Registered date / owner | 2026-11-09 / Codex; registered before execution |
| Kind / scope | Correctness and lifecycle; same installed payload with explicitly prepared image metadata, bounded worker controls |
| Baseline / candidate | Text-only controls in [launch correctness record](../archive/runs/2026-11-09-model-launch-correctness.md); same final worker with retained image tower loaded |
| Worker | Source `1a0517d6`; release SHA-256 `388df7f9d09bf5bbd719cb5998f018bfb5df480f07d2b4a196e781cfcbf56d13` |
| Artifact | Installed ddalcu mixed-affine Qwen3.8-Flash-Next, revision `788c07a8725642bccf13a0a3b8a21954ab84b4f2`; config SHA-256 `84a69b2bcb20a57b033c3827bc98fa21a5db4d18b55e0640dd3687c99a1b765f` |
| Attempt state / result | Completed; three bounded correctness/lifecycle control groups passed after a rejected initial setup |
| Promotion / disposition | Correctness evidence for the explicitly prepared image clone; no installation or default change; performance unqualified |

## Hypothesis and fixed controls

The ordinary-decode MTP head catch-up also runs for text requests with the
image tower resident. Earlier model and worker controls explicitly loaded the
text-only route. Use the unchanged release worker with an owner-private
`--vision-staging-root`; require both image capability and MTP in readiness.

1. Repeat the existing 512-context, 16-output-token, seed-42 stochastic
   history-independence request with three effective drafts, fresh and after
   an unrelated request. Require exact output-ID hashes, finish reason and
   effective draft counts to match the earlier text-only control.
2. Repeat the existing 8,192-context seeded three-draft prefix request, requiring
   zero reused rows on its first request, 4,096 on its repeat and zero after an
   unrelated request. Require exact output-ID hashes against the earlier
   text-only control.
3. When supported by the existing validation helpers, execute an actual image
   request with zero drafts and zero prefix reuse, then replay the seeded text
   request with three drafts. Require exact pre-image/post-image text output
   hashes and a cache miss immediately after the image transaction.

Sampling stays temperature `0.7`, top-k `20`, top-p `0.95`, seed `42` for the
text controls. Preserve the existing private request inputs and budgets;
publish only aggregate counts/hashes. Stop on a native error, mismatched
readiness, output mismatch or memory admission rejection. Do not widen gates.
Run one worker at a time and release it before a fresh-process comparison.

## Environment, precision and limits

Host is the same Mac17,7 with 128 GiB unified memory and macOS 26.7.2. Existing
locked Rust/Python dependencies and the existing binary are used. No weights
or precision policies change: affine weights remain unchanged, activations
BF16, recurrence/reductions F32, K/V Q8Affine64. The image tower stays loaded
for every candidate text request. The earlier text-only checks did not rehash
weight payloads. This corrected image setup fully rehashed all 102 payload
files, totaling 107,303,406,192 bytes, before writing its schema-6 descriptor.
The config and executable identities remained unchanged.

This is a bounded correctness/lifecycle run. Elapsed observations do not
establish performance parity or a speed improvement. Record pageout/swap
counter deltas and observed runtime separately. No full-window or model-quality
claim follows. Keep original text-only receipts immutable.

## Execution follow-up

Initial setup was rejected before loading a model: the installed ddalcu folder
does not contain the required `stark-image-artifact.json`. Its declared vision
payload is not an admitted image artifact. The failed setup receipt remains
`runtime/launch-image-resident-invalid-setup.json` (no inference, 0.047 seconds).

The corrected setup used an ignored APFS clone of that exact installed
folder and the existing `prepare_image_artifact` routine. This adds the
schema-6 image manifest/processor sidecars and fully hashes the existing
weights without changing their bytes or the installed folder. Config identity
remained unchanged; the resulting manifest and payload aggregate were bound
before execution. This is a setup correction within the same fixed correctness recipe;
it does not promote the installed text-only metadata as image-capable.

Both tower-resident workers reported image input, MTP and 4,096-token prefix
support. The seeded history controls matched the earlier text-only output hash
exactly. All four prefix controls matched their earlier output hashes and
reuse counts. A real one-token image request used zero drafts and reused zero
rows; following text recovered three drafts with a cache miss, then a
4,096-token hit on its repeat, preserving the earlier exact output hash.

The run took 41.376 seconds. Host counters increased by 596 pageouts and
536 swapouts, with compression activity. These functional results provide no
screened timing or speed-parity evidence. The native worker did not change,
and the original installed folder still lacks an image descriptor.

Canonical execution evidence is recorded in
[the separate run record](../archive/runs/2026-11-09-model-launch-double-check.md).
