# Gemma 4 default-image text parity

## 20261109-G4IP-01: exact attention-ring prefix snapshots

Status: implemented on branch; bounded E2B/E4B cache correctness and short
image-loaded text speed controls pass. Cached-request performance and Gemma
assistant MTP remain open. Source baseline `5f702450` includes merged vision
text-prefix infrastructure from PR #110. This attempt addresses IMGPAR-07
first. IMGPAR-10/11/12 remain separate larger checkpoint qualifications.

Observed problem: the vision wrapper rejects text prefix reuse for any decoder
with sliding attention because its live ring overwrites older K/V rows. Merely
rewinding scalar lengths cannot restore that state. This also affects Gemma's
shared-KV consumers of each owning ring layer.

Hypothesis and change: retain an independent copy of committed physical K/V
rows, plus ring offset, length, capacity and source traversal origin. Restore all
of them before a new text suffix. Full attention continues to retain its existing
prefix in place. Shared-KV layers use the owning layer's snapshot once. PLE
embedding tables and decoder weights are immutable, not request snapshots.
Account the additional ring payload in checked request admission.

Fixed correctness gates: exact synthetic payload/scalar restoration before and
after wrap; repeated restoration without snapshot aliasing; reject incompatible
states; exact fresh/restored decoder logits and greedy/seeded continuations;
actual E2B/E4B worker cache hits at the 1,024-token grid; images invalidate text
state and use zero MTP. Publish only aggregate counts/hashes.

For matched text speed, use the same retained artifact, precision and request
settings with image loading off/on, separately for cold and cached requests.
Record a baseline variability bound before measuring the image-loaded arm.
Loading cost is separate. The shared coordinator serializes builds and
GPU/model runs. Results below apply to the exact completed runs.

The bounded worker recipe uses 1,792-token initial and 1,824-token changed-tail
inputs, a 1,024-token saved boundary and eight generated tokens. This advances
beyond the small profiles' 512-row local window after capture. Fresh greedy and
seeded controls are compared with restored outputs. The short timing recipe is
one warmup and four measured requests with 64-token output budget at context
4,096, MTP off. Its regression bound is fixed after candidate text-only baseline
calibration and before the image-loaded arm: the larger of 5% and twice measured
relative range; a bound over 10% makes timing inconclusive.

Named retained artifacts are E2B revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`, weights
`5c26289c1536f6afda47e62616c02ae63df489dc5c37d6034624413b2fd75993`,
and E4B revision `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c`, weights
`58e1ed2e61286ccc58223bc46a2efe909f11b0b8eca6c4ccb03f00cdb435ea5a`.
Verify metadata and executable identities for each new run. Existing image
qualification is recorded in the [Gemma ledger](2026-11-08-gemma-images.md);
it does not establish this attempt's cache or performance result.

Source inspection of those retained native configs establishes the payload
inventory below, for contexts at least 512 rows. This is checked allocation
accounting, not measured peak RSS.

| Artifact | Owned / shared-KV layers | Owned 512-row rings | Ring snapshot K/V payload | PLE layers |
| --- | --- | --- | --- | --- |
| E2B | 15 / 20 | 12 | 6,291,456 bytes (6MiB) | 35 |
| E4B | 24 / 18 | 20 | 20,971,520 bytes (20MiB) | 42 |

Only owners add snapshot payload. A new boundary drops the previous snapshot
before allocating its replacement; repeated restoration of the same boundary
retains the existing independent buffers. Image/unscoped requests do not retain
this payload. The worker also drops unusable preceding states before allocating
a fresh request. Existing conservative scratch/admission margins still apply.

MTP inventory: the pinned E2B/E4B base configurations contain no embedded draft
head declaration. Google's [official MTP documentation](https://ai.google.dev/gemma/docs/mtp/overview)
and [usage guide](https://ai.google.dev/gemma/docs/mtp/mtp) identify separate
`google/gemma-4-{size}-it-assistant` checkpoints for all five Gemma 4 target sizes.
They use shared target embeddings/last-layer activations and target K/V access;
E2B/E4B use clustered vocabulary readout. The current converter accepts Gemma
target configs, and the native worker has no Gemma assistant loader or execution
path. These compatible assistants require dedicated conversion, exact target
pairing, scratch/state admission, verification/rollback and prefix snapshots.
Therefore source MTP is **not N/A**. No Gemma MTP support or performance is
claimed by this ring-cache increment; assistant implementation remains open.

Initial disposition: awaiting implementation validation and exact-artifact
qualification. The completed bounded disposition is recorded below.

First focused native run: 28 passed, two failed, eight ignored. Both failures
were invalid synthetic shared-KV fixture setup: the consumer retained scalar
RoPE while its owner used M-RoPE, and strict decoder admission rejected the
positional mismatch before inference. Correct the consumer to the owner's
identical positional contract and rerun; no admission relaxation is justified.
The direct CPU/Metal ring payload copy/restore controls passed in this run.

After correcting the fixture, both CPU and Metal overwritten-ring/shared-KV
decoder controls pass exact logits, greedy/seeded continuations and repeated/
replacement snapshot restoration. Final focused prefix suite: **30 passed,
eight ignored, zero failed**; the changed attention precision/ring/shared-cache
allocation test separately passes. These counts overlap the initial run and
must not be summed. The ignored tests require unrelated private checkpoints.
Sealed nightly test executable SHA-256:
`cfe24f3be7bf3367ef258d4093eac7cae2a16ab66699c7ec7bd7e202d99b358e`.
The tests use `nightly-2026-10-01`, CPU/Metal and the repository's optimized test
profile. The temporary test build directory was retired after preserving this
executable to leave disk reserve for the separate Rust 1.98.1 release worker.

First E2B private worker recipe was **invalid setup** at its first scoped long
request: it supplied 64 hexadecimal scope characters instead of the protocol's
32. The worker correctly returned `invalid_generation_request` before inference.
The preceding 73 aggregate checks included matching old-text/candidate-text/
image-loaded short outputs and baseline long fresh outputs. Preserve receipt
`runtime/gemma4-e2b-image-text-parity-v1.json`; it is not cache qualification.
Correct the private recipe's scope length, keep the error code in aggregate
failed receipts, and rerun as a separate receipt.

## Completed bounded result and remaining work

The corrected E2B and E4B actual-worker recipes each pass 149 aggregate checks.
They match fresh old/candidate text outputs with restored image-loaded greedy
and seeded trajectories after full ring overwrite. Real image→fresh text→cached
text transitions and scope/context/reset/error/cancellation invalidation pass.
Cancellation is requested at the initial progress frame. Mid-GPU cancellation
and injected device failures remain outside this actual-model run.

The Rust 1.98.1 release generation worker passes 134 unit tests, two ignored,
zero failed. Worker SHA-256:
`71280ab529af3994cc3920ced6b18f15381b7129a1250eda035135166e0b2e82`.
An independent read-only review found the ring payload/scalar restoration,
owning-layer accounting and guarded wrapper/worker plumbing consistent. The
shared-KV fixture's positional mismatch was corrected without weakening admission.

The [matched benchmark](../benchmarks/2026-11-09-gemma4-image-text-parity.md)
records per-artifact identities, byte-identical decoder controls, all arm
medians, hashes and the preserved failed recipe. Useful decode with images
loaded changes -0.287% for E2B and +0.024% for E4B against matching text loading;
first-token and end-to-end medians also remain inside the fixed 5% short-request
bound. That workload has 28 input/64 output tokens. Long correctness trace
observations include slower cached first-token times after warm fresh requests;
they require a separate warmed, alternating timing investigation. No cached
speedup or full IMGPAR-07 performance completion is claimed.

Disposition: adopt the exact ring snapshot implementation on this branch, with
bounded retained E2B/E4B correctness and short loading-mode performance evidence.
Keep cached-workload timing and Gemma assistant conversion/execution open.
Larger 12B/26B/31B full generation, resource and parity gates remain unqualified.
The implementation has not been merged, released or installed by this attempt.
