# Gemma 4 E2B/E4B: text with image support loaded

Date: 2026-11-09. Attempt: `20261109-G4IP-01`. Apple M5 Max, 40 GPU cores.
The retained E2B and E4B artifacts pass exact text prefix/lifecycle checks on
this branch. Their short text inference measurements with the image tower
resident fall within the predeclared 5% bound. Larger workloads, cached-request
performance and Gemma assistant MTP remain open qualification/implementation
work. This record makes no release or installed-app claim.

## Matched controls and identities

The previous release worker SHA-256 is
`b3a4f851515a44d42975e120599d084a647fd36a8aa7010b915b4829bf9bc2fa`,
retained from the prior prefix/MTP validation. The candidate Rust 1.98.1 release
worker SHA-256 is
`71280ab529af3994cc3920ced6b18f15381b7129a1250eda035135166e0b2e82`.
Its source baseline is `5f702450`; the implementation is the commit containing
this record. The aggregate runtime-source/Cargo.lock seal is
`f59f631a447fe3ba440713a5d4074489b040f15e7a25f394487cfeb660607cce`.

| Artifact | Source revision | Image artifact weights SHA-256 | Native config SHA-256 |
| --- | --- | --- | --- |
| `google/gemma-4-E2B-it` | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | `5c26289c1536f6afda47e62616c02ae63df489dc5c37d6034624413b2fd75993` | `9acc540130bd576271650c3805467141136570ffc4111c186e317442cc3dc762` |
| `google/gemma-4-E4B-it` | `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c` | `58e1ed2e61286ccc58223bc46a2efe909f11b0b8eca6c4ccb03f00cdb435ea5a` | `977693a5a4bb1378e26db4e4eaff457eef788240c57c5f0c94a369d57b3f302e` |

The strict text loader requires a text wrapper and consumes every declared
tensor. Private APFS-cloned controls therefore select only decoder descriptors
from the same native artifact. They preserve the complete decoder config,
numerical selectors, tensor shapes/dtypes/offsets and every decoder payload byte.
The source image artifacts remain unchanged. No product image-off option was
added. Source-versus-control decoder payload hashes match independently:

| Control | Decoder tensors | Decoder payload SHA-256 |
| --- | --- | --- |
| E2B | 475 | `c0024204602bce6f9073f8d0dc09dd8644ed179a2b9c1f62d0996d3d234f5982` |
| E4B | 575 | `20db70e3560fd0359d31eaa7eacc263ca2993a2d9ec967a9677bab90fc63ca81` |

## Short text inference

For each artifact, four independently loaded arms run serially: previous text
worker, candidate text worker, candidate image-loaded worker, candidate text
worker again. Each arm has one warmup and four measured requests. The model's
own native template produces 28 input tokens; greedy output is 64 tokens,
context 4,096, MTP off. Every output hash matches across all arms.

Before the image-loaded arm, the candidate text-only observations fix each
metric's bound to the larger of 5% and twice its measured relative range; bounds
over 10% make timing inconclusive. All resulting bounds here are 5%. First-token
timing uses the worker's wall-clock proxy, including native setup/lookahead;
useful decode excludes its first sample. Loading/hashing time is excluded.

| Artifact / loaded arm | First-token median (ms) | Useful decode (tokens/s) | End-to-end median (ms) |
| --- | --- | --- | --- |
| E2B previous text | 15.333 | 103.937 | 631.454 |
| E2B candidate text | 15.278 | 103.968 | 631.275 |
| E2B candidate images | 15.584 | 103.669 | 634.301 |
| E2B candidate text return | 15.372 | 103.772 | 632.511 |
| E4B previous text | 34.091 | 55.037 | 1,198.086 |
| E4B candidate text | 33.608 | 54.972 | 1,198.359 |
| E4B candidate images | 33.487 | 54.985 | 1,197.922 |
| E4B candidate text return | 33.541 | 55.127 | 1,195.108 |

Relative to matching candidate text-only loading, image-loaded E2B useful decode
is -0.287%, first-token time +2.005%, end-to-end +0.479%. E4B useful decode is
+0.024%, first-token time -0.361%, end-to-end -0.036%. All are within the fixed
short-workload bound. Relative to the previous worker, image-loaded useful
decode is -0.258% for E2B and -0.096% for E4B.

GPU/build ownership was serialized. A coarse host screen across preparation,
loads and runs observed 306 additional pageouts, 12 swapins and zero swapouts.
No thermal/performance warning level was recorded before the runs. Peak RSS,
thermal temperatures and a full background-load screen were not collected.
These are bounded quick observations; broader performance qualification remains.

## Cache correctness and remaining timing work

Each artifact passes 149 aggregate checks. Fresh old/candidate text outputs
match image-loaded restored greedy and seeded output hashes at 1,792 input
tokens, plus a changed 1,824-token tail. The previous 1,024-token boundary is
restored after the complete 512-row local ring has been overwritten. E2B has
20 shared-KV consumers, E4B 18; each consumes its owner's restored state.
The saved ring payload is 6MiB and 20MiB respectively, counted by request
admission. PLE tables remain immutable.

A real synthetic PNG request produces one token with zero text reuse and zero
drafts. Following text first reuses zero rows, matches fresh output, then reuses
1,024 rows on repetition. Scope/context changes, reset, invalid-context errors
and text cancellation clear the retained entry and preserve fresh text recovery.
Cancellation is requested on the initial streaming progress frame; this run
does not establish cancellation during every GPU stage. Image answer quality,
new image numerical calibration and injected device failures were not tested.

The long checks establish exact state restoration. Their single first-token
observations do **not** establish cached speedup. For example, post-image fresh
versus repeated cached E2B is 86.301ms versus 91.831ms, and E4B is 228.533ms versus
237.639ms, despite 1,024 correctly reused rows. Kernel warmup and trace ordering
were not controlled for a paired cached timing gate. Preserve these observations
and investigate warmed, alternating cache-off/on workloads before declaring
cached performance parity or a benefit. IMGPAR-07 therefore remains open beyond
the bounded correctness and short text-loading results here.

The upstream Gemma 4 assistants are separately released MTP models. Stark has no
Gemma assistant conversion/execution path yet; source MTP is not N/A. This run
keeps MTP off and makes no assistant/MTP performance claim. The [attempt
record](../model-experiments/2026-11-09-gemma4-image-text-parity.md) tracks that gap,
model-free tests, invalid setups and larger-model qualification boundaries.

## Private receipts

Only aggregate counts, timings, booleans and hashes are retained in receipts.
Prompts, responses, token vectors, probabilities and image pixels are excluded.
The private recipe SHA-256 is
`43a978e837013b83f9a90291dd77b41565077003a9ad0d26d0978d69dc8a4c90`.

| Receipt | SHA-256 |
| --- | --- |
| E2B corrected `runtime/gemma4-e2b-image-text-parity-v2.json` | `437bbad8a65d67b255af02d23a0f07e02fe614498003fb39896a4753af414096` |
| E4B `runtime/gemma4-e4b-image-text-parity-v2.json` | `024adfe8ceb6d403026bc62079f98d1b4e9e68a746809427c7875836530c03ff` |
| E2B invalid-scope recipe `runtime/gemma4-e2b-image-text-parity-v1.json` | `2e2edaf6cc84f80e7e08eed956f627b43a1fefabf751d8c216d7192d462e1320` |
