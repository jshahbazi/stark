# 20261112-EDGE0-02: pinned Edge0 source preparation

## Identity and state

| Field | Value |
| --- | --- |
| Registered date / owner | 2026-11-12 / Codex implementation; repository owner authorizes local non-commercial evaluation only |
| Kind / scope | Compatibility, source preparation and admission; no inference or performance experiment |
| Attempt state | In progress; actual tokenizer metadata mismatch identified, final checks pending |
| Acceptance | All exact structural and tokenizer checks below must pass before atomic publication. Any mismatch rejects preparation. |
| Promotion decision | Evaluation-only preparation tooling; no product catalogue entry or native load capability |
| Baseline | Stark `2c966a73`, source record based on `a171326271f6c5eaf31f4ea49bc21905f7e7a813` |
| Related | [Phase 0](2026-11-12-edge0-compatibility.md), [governing plan](../EDGE0_PLAN.md), [owner decision](../archive/runs/2026-11-12-edge0-license-intake.md#owner-decision-2026-11-12) |

The owner instructed continued local evaluation on the branch before the
Phase 0 record is merged. Merge remains pending. No commercial permissions,
product admission or model execution are established by that instruction.

## Hypothesis and change

The proposed pinned preparation route can retain the released binary payloads,
validate their layout, and publish a folder only when the original Qwen
tokenizer can be proven equivalent to the shipped Edge0 metadata. The latter
is a testable prerequisite, not an assumption about the shared model family.

Add an explicit `edge0-145b-q3k` CLI preparation choice, bounded metadata
preflight, exact file identity checks, source geometry/manifest validation,
derived metadata receipts and an explicit storage discriminator. Reuse owned
resumable staging and atomic publication. Keep normal app admission and
`prepared_identity` from treating an evaluation folder as loadable.

## Artifact, sources and environment

- Sole checkpoint target: `Edge0/Edge0-145B-q3k` revision
  `d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`. Eight binary payload identities
  are bound in the [Phase 0 record](2026-11-12-edge0-compatibility.md#artifact-source-and-environment).
  Their subtotal is 103,374,417,600 bytes; small metadata is additional.
- Format/source authority: `Edge0-AI/Edge0` at
  `b7900cbb322926ccebdcd0c0c5a1280497ecff22`, with component origin and
  licenses in [upstream provenance](../../provenance/UPSTREAM.md).
- Tokenizer comparison source: `Qwen/Qwen3.8-Flash-Next` at immutable revision
  `812d96e10a0dd82b69d749e68693698cf6c36daa`; tokenizer files and their hashes
  must be bound separately from that repository's license identity.
- Python checks use the locked converter environment and Python 3.13.13 via
  `uv`. Repository policy uses its locked Python 3.12.13 script environment.
  No dependencies or lockfiles are changed. Build flags, executable hashes,
  GPU/Metal toolchain and inference precision are N/A: this attempt builds no
  worker and performs no inference.
- The initial disk screen reported about 299 GiB free. This is preparation
  capacity only, not loaded-model memory admission or a supported-host claim.
  Recheck before any full download; preserve at least 16 GiB free for this
  local evaluation. No unrelated model/process cleanup is authorized here.

## Workload, controls and fixed gates

1. Fetch only pinned small metadata and bounded GGUF headers for preflight.
   No upstream Python, scripts or foreign runtime is executed. Fix the exact
   expected file paths, sizes and SHA-256 before a full preparation attempt.
2. Prove token IDs, ordered merges, token classifications, special IDs and
   pre-tokenization/normalization semantics agree. A difference fails this
   route. Do not silently alter the upstream tokenizer or accept a weaker
   semantic comparison.
3. Only after preflight passes may full payload acquisition start. Verify all
   expected hashes, geometry, manifest ranges, labels, projection/table size
   formulas and duplicated tensor byte equality before publication. Reject
   symlinks, unsafe ranges/paths, missing or changed required files and
   inconsistent metadata.
4. Bind source files and derived config/tokenizer metadata in
   `native-source.json`; report native loading and execution as unverified,
   and scope as local non-commercial evaluation only.
5. On failure retain only owned resumable staging. No published output folder,
   success receipt, product catalogue/install admission or runtime capability
   may result. Use synthetic successful and mutation controls to exercise the
   publication logic without downloading model payloads.

The model is text-only in the pinned inventory; image comparisons are N/A for
this preparation attempt. MTP, prefix state, sampling, context/output budgets,
request lifecycle and runtime precision are untested and remain separate
later gates. No prompts, generated responses, inference token vectors,
probabilities or exception dumps may be persisted.

## Attempt history and failures

| Trial | Observation | Disposition |
| --- | --- | --- |
| Immutable metadata intake | Ordinary vocabulary IDs and merge order match; six added-token classifications differ between the GGUF and the pinned HF tokenizer. Padding requires separate classification. | Strict tokenizer gate fails; no root tokenizer may be derived from this comparison. |
| Semantic controls | Pre-tokenizer/normalizer differences identified; source hashes and aggregate controls being finalized. | Additional compatibility evidence; never substitute token-count equality for semantic proof. |
| Full payload preparation | Not run; held before 103.4 GB payload download because metadata gate fails. | No prepared folder and no Phase 1 pass. |

## Correctness and lifecycle

Verification is in progress. Required checks include synthetic positive
publication, malformed/mutated source files, failed preflight before payload
requests, resumable staging, honest receipt status, rejection by native app
admission and license/notice packaging. Record exact results and the live
metadata-only command in the follow-up below after execution.

No model-free inference operators, full-model numerical controls, prefix
reuse, MTP or request lifecycle have been exercised by this attempt. A source
preparation test cannot qualify those paths.

## Performance and memory

Not measured: inference speed, first-token cost, prefill/decode, model RSS,
resident weights, retained state or cache benefit. No timing claim is made.
Bounded source/header reads and synthetic fixture sizes are validation
controls, not runtime memory measurements.

## Evidence, decision and limits

The exact-equality tokenizer hypothesis currently fails against the pinned
release. Preserve the rejected comparison and finish the fail-closed
preparation tooling. Phase 2 cannot begin under the existing gate until an
artifact has a valid preparation/tokenizer disposition. Any alternate
tokenizer policy needs its own source-bound correctness design and attempt;
it cannot relabel this comparison as passing.

## Follow-up history

### Strict borrowed-reference route: rejected

The live CLI command below and its `--offline` retry both exited 1 with category
`tokenizer_equivalence_rejected`. They retained owned staging, requested no
full payload files and published no output folder:

```sh
HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false uv run --project tools/converter --locked --python 3.13.13 python -m stark_convert.hybrid_prepare edge0-145b-q3k --output runtime/edge0-local-evaluation --preflight-only
```

The 61 small source files total 10,332,477 bytes; with the eight binary payloads,
the complete pinned set has 69 files and 103,384,750,077 bytes. Only small
metadata and the bounded skeleton header were acquired for this command.
The aggregate rejection receipt has SHA-256
`a499b9847bf2651bbaa7c6158890b3289fc0b648bdc985c0fe88d087b2c4d2cd`.
No full-payload SHA or duplicate-tensor payload proof is claimed.

| Metadata check | Observation |
| --- | --- |
| Ordinary BPE IDs and ordered merge ranks | 248,044 ordinary IDs and 247,587 merges agree |
| Complete vocabulary | Edge0 has 248,320 slots; HF with added tokens has 248,077. The 243 extra Edge0 slots are UNUSED. |
| Token types | Six GGUF CONTROL IDs differ from HF USER_DEFINED classifications |
| Special IDs | BOS metadata differs; Edge0's insertion flag remains false |
| Normalization | HF declares NFC; the pinned source BPE path has no NFC pass |
| Pre-tokenizer | A fixed Unicode case-fold witness splits into two HF pieces and one Edge0 piece; witness SHA-256 `826e174be5effb10384a99e534fbde86afab0af48f2e242f7cb9b6261cddc9c4` |

Pinned HF tokenizer SHA-256:
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`;
tokenizer config SHA-256:
`207c03c41a06a9c3b0cc527d8a56115159c1eac6b59a4ffae85b646e773d074e`.
This reference revision is an immutable original-Qwen comparison target; it
does not establish Edge0's unverified conversion/training base revision.

The initial targeted integration run passed 135 tests. The broader locked,
offline converter run passed 7,620 tests with 94 skipped in 367.64 seconds;
its collection preceded source-native tokenizer tests and the final extra
publication counterexamples, which receive separate focused checks. The
distribution notice suite passed 23 tests. A first converter test setup lacked
the required native template helper; the helper was then built with pinned Rust
1.98.1 and tests rerun. This setup failure supplies no model evidence.

The borrowed-reference route stays rejected. The source-native successor is
tracked in [attempt `20261112-EDGE0-03`](2026-11-12-edge0-native-tokenizer.md),
with its own failed v1 and preregistered Unicode-classification follow-up.

### 20261112-EDGE0-02B: opt-in source-native preparation

Registered before any source-native preparation run. Keep `borrowed-qwen` as
the strict default with its rejected result. An explicit `edge0-native-v2`
policy may be used only after the separately registered native-tokenizer
attempt passes every required gate and an accepted static profile binds its
source, Unicode asset, tokenizer program, validator and passing receipt hashes.
An arbitrary caller-supplied pass flag is insufficient.

Require the same 69 pinned source files and full hash/structural checks as
before. Derive tokenizer/config metadata from the released source semantics,
without copying unqualified HF normalization or token types. Bind that
derivation, policy and proof profile independently in `native-source.json`;
do not claim the rejected HF comparison became equivalent. Keep normal
native-model identity/import disabled. Use a distinct owned output/staging
identity for the policy, and reject changed profile/source/derived metadata.

Before full acquisition, repeat the current disk screen and the accepted
metadata-only preflight. Require a new passing preflight receipt, then run the
full preparation only if it stays within the existing local non-commercial
scope and disk guard. No model inference is part of this attempt. A complete
prepared folder with every payload hash is still the Phase 1 gate; a passing
tokenizer profile alone cannot close it.

The environment-bound source-native profile is now accepted for Python
3.13.13 / tokenizers 0.23.3 with proof SHA-256
`d74501849a329954290a58bb376bc3d59b513cc5af01a04b1da40e3e94052ca0`.
The native-policy metadata-only CLI passed; receipt SHA-256
`25818ada66e28216756f3322c953e8600d887e3cad8c7cd63e98706498b1b8f8`.
All 61 small files matched their pins, with no payload or derived JSON files
present at that preflight boundary. The profile also binds source-native
tokenizer JSON SHA-256
`f1d23198102cc49d8cb199600be2f5ec97d35b7540830b284010a425147066c4`
and tokenizer config SHA-256
`5a095bafcd6923c00e6d7041dbe493f7a7995c1331c75a3c3e6625ec91cce6ae`.

The disk screen before full acquisition reported about 259 GiB free. The full
run uses a distinct policy-bound owned stage:

```sh
HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false uv run --project tools/converter --locked --python 3.13.13 python -m stark_convert.hybrid_prepare edge0-145b-q3k --output runtime/edge0-native-v2-evaluation --tokenizer-policy edge0-native-v2 --progress-fd 3 3> runtime/edge0-native-v2-download-progress.jsonl > runtime/edge0-native-v2-prepare-result.json 2> runtime/edge0-native-v2-download-errors.log
```

An initial invocation supplied progress descriptor 2, which the private-pipe
contract rejects before preparation; it was corrected to descriptor 3. This
setup failure is separate from compatibility results. Progress output contains
only fixed phases and aggregate byte counts. Staged transport bytes do not
establish the final pinned hashes, layout checks or successful publication.
Full acquisition and final validation are in progress; no Phase 1 pass is
claimed yet.

### Final implementation checks before artifact publication

The final locked offline converter suite passed **7,662 tests**, with **94
skipped**, in 353.84 seconds. This includes the source-native tokenizer and
accepted-profile integration added after the earlier suite. Focused preparation
coverage passed 54 tests; source-native tokenizer coverage passed 22. Ruff,
repository policy and whitespace checks passed. The distribution notice suite
passed 23 tests. A rebuilt wheel contains the exact source-native admission
module and Unicode asset bytes. No new dependencies or lockfile changes were
required.

These are implementation and packaging results. The ongoing full payload run
must still establish every expected SHA-256, source layout, duplicate-tensor
byte equality and atomic publication before Phase 1 can pass. No worker or
model execution is represented by the test counts.
