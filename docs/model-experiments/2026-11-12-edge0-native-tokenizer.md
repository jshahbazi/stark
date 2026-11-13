# 20261112-EDGE0-03: source-native Edge0 tokenizer

## Identity and state

| Field | Value |
| --- | --- |
| Registered / owner | 2026-11-12 / Codex, within the owner's local non-commercial evaluation scope |
| Kind / scope | Tokenizer compatibility and correctness, Phase 1 only; no model inference |
| State | CPU oracle established; v1 rejected; successor `20261112-EDGE0-03B.1` passed the declared Python preparation scope |
| Result / promotion | Accepted opt-in local preparation profile only; Rust loading, generation streaming and inference unqualified |
| Predecessor | [20261112-EDGE0-02](2026-11-12-edge0-source-preparation.md), rejected borrowed-HF equivalence route |
| Governing plan | [EDGE0_PLAN.md](../EDGE0_PLAN.md#phase-1-source-preparation-and-admission) |

## Hypothesis and change

The pinned Edge0 release has tokenizer semantics that differ from the official
Qwen HF tokenizer. A separate representation of the actual Edge0 tokenizer
may preserve its behavior without claiming equivalence to that HF program.
This successor keeps the failed comparison and replaces its derivation policy
only after independent source-bound proof.

Candidate design: retain exact source vocabulary/merges and special IDs;
implement the pinned `qwen35` ASCII-expanded contraction splitting and absence
of NFC normalization, with explicit CONTROL/USER_DEFINED and unused-padding
behavior. Do not assume the HF tokenizers representation can express every
source mode. A concrete API/semantic mismatch blocks this candidate.

## Artifact, source and environment

- Edge0 checkpoint:
  `Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`.
  Vocabulary, merges, descriptor, token types, chat template and GGUF header
  must be hash-verified as in the preparation attempt.
- Authoritative proposed oracle: llama.cpp
  `3cf03257f219afbe7334045ff7c6a06ac68c627d`, pinned by Edge0 Strata at
  `b7900cbb322926ccebdcd0c0c5a1280497ecff22`. CPU-only build, vocabulary-only
  load; no weight payload or GPU execution. Record exact source paths/hashes,
  build flags, tool identities and oracle executable hash when built.
- A metadata-only GGUF may be constructed from the sealed original header for
  `vocab_only=true`, with its own derivation and output hash. First verify that
  the pinned loader accepts this; do not present an unbuilt oracle as evidence.
- Python tools use locked converter dependencies with Python 3.13.13 via `uv`.
  New dependency versions are outside scope.
- Alternative Edge0 frontend tokenizers are investigation aids only unless
  independently proven to be the authoritative 145B path.

## Workload, controls and fixed gates

Preregister the fixture corpus and mode matrix before comparing a candidate:

1. Exact vocabulary ID, merge order and token-class inventory from shipped
   metadata and GGUF. Classify unused padded slots explicitly; never turn them
   into recognized literal specials just to make vocabulary counts match.
2. Exact token IDs from source and candidate with no BOS insertion, under each
   `parse_special` mode that the worker uses. Include normal text, whitespace,
   newlines, byte-level characters, multilingual text, combining marks and
   Unicode case-fold contraction boundaries. Source/candidate ID sequences must
   match, with zero mismatches across the fixed corpus.
3. Distinguish the raw-piece and text interfaces. Require exact raw token-piece
   bytes for all vocabulary slots under both source render modes, using an
   independent byte mapping. Stark's decode RPC and generation return UTF-8
   strings; compare those separately with the declared boundary policy of
   concatenating source pieces before UTF-8 replacement decoding. Valid UTF-8
   text must be byte-exact. Invalid UTF-8 behavior remains unqualified until
   the owning worker/output policy is verified; do not compare a raw `0xFF`
   piece directly to a String API and call that an encoding mismatch. Compare
   EOS/EOG decisions and effective loaded token classes explicitly. A required
   worker mode whose behavior cannot be represented remains a blocker.
4. Negative controls must detect the old HF normalization, regex and token-type
   differences, plus a changed vocabulary ID and merge order. No candidate
   promotion if those controls accidentally pass.
5. Record source, oracle, candidate program, metadata and corpus hashes, mode
   counts, mismatch aggregates and exactness result. Keep text/ID vectors only
   in memory; do not persist prompts, token vectors or raw process dumps.

Builds and comparisons are serialized per oracle instance, CPU-only, with a
bounded corpus and a two-minute comparison timeout after build. Fix the corpus
hash and counts before the first comparison. A build failure or unsupported
vocabulary-only load is recorded separately from a semantic mismatch. Source
preparation remains blocked until this attempt passes; Phase 2 does not start.

### Fixed corpus and mode matrix, before comparison

Registration: `edge0-native-tokenizer-corpus-v1`, SHA-256
`bea133e4fee577f39d8f23750d460e7b788aa4e2f748337185c0df50d0fd3d7d`.
The registration contains hashes/counts only and is retained privately with the
oracle receipts. Before the first candidate comparison, fix 1,942 encode cases
with HF `add_special_tokens` false/true against native no-BOS,
`parse_special=true`; 248,327 decode cases with skip-special false/true; all
248,320 raw token pieces with render-special false/true; and the full effective
token-class, EOG and special-ID inventory. Source `parse_special=false` is a
separate control rather than a mode exposed by Stark's current encode API.

This policy clarification precedes candidate comparison. It distinguishes the
raw-byte oracle from the worker's String contract and does not convert any
failed borrowed-reference result into a pass.

The executable candidate comparisons use the locked Python converter binding,
`tokenizers` **0.23.3**, on Python 3.13.13. The Rust worker separately pins
`tokenizers` **0.22.2**. Inspection of the Rust decoder establishes its declared
UTF-8 String boundary; these Python comparisons do not execute or qualify the
Rust tokenizer loader, generation stream or native inference. Those remain
later runtime gates.

## Attempt history and failures

### CPU oracle feasibility

The pinned llama.cpp CPU target built and loaded a vocabulary-only GGUF with
zero weight payload. The 57 original metadata entries were retained byte for
byte, tensor count replaced with zero and tensor descriptors/payload omitted.
Derived GGUF: 10,945,120 bytes, SHA-256
`f84c1aeca75a7c8672d00dc6110b51a59af4ae7edc1ba8ed31bb615978494e37`.
The original header is 11,012,448 bytes, SHA-256
`1cd961c7675aae8dcb9efc219ff58813f37441c33a567073bd521c72673b6685`.

Source archive SHA-256:
`c076d7534afa0e5d0ec2a0d425b11e791c16f3de0d727221aea071cef156a280`.
All 3,608 extracted source files matched it; source-tree manifest SHA-256
`2cfb4bf2251a0ca61e56f6e906af85ed655173ad0b67717e9fbe7d3dc780cca2`.
CPU-only build used CMake 4.4.3, Apple Clang 21.0.0 on macOS 26.7.2
(`25G332`), Release/shared libraries, four jobs, and disabled Metal,
Accelerate, BLAS, CUDA, Vulkan, OpenMP, native dispatch and upstream
tests/tools/server/app/common. No inference context was created. Upstream
CMake displayed an enclosing Stark Git revision; archive/tree hashes identify
the source actually built.

Native probe SHA-256:
`670408bfd012be4c15277089afe0aabd1d087d6d6ba8a70876d761721ecd01db`.
Original bridge source SHA-256:
`6bcce07c37928babf8b7b7ec61af24609b0c95a0d10ea6bc2fb1d203e899db11`.
The bridge silences the native logger before loading. The feasibility receipt
SHA-256 is `4a39336c279e919731acbb04b10828e408ca928e07fddd671318956dcae93c35`.
These ignored receipts record dependency hashes and compile flags; they contain
no corpus text or token vectors.

### v1 comparison: rejected

Candidate revision: `edge0-native-tokenizer-v1`, program JSON SHA-256
`69a7a3e57e2c886ae2d6a6ed6cfb86c0f8416b4df9fdeb1ce68a9dd26989af69`.
Validator source SHA-256:
`801744cb412ea81b639ce6254fe5647d0295d414848fcd73ad63678fe74fc3b1`.
The first-run source was reconstructed by reversing a later reporting-only
change, and its complete bytes match that original hash. It is labeled as a
hash-matching reconstruction, not an original retained snapshot. A separate
full-special-check v1 source snapshot was retained before v2 edits.
The fixed comparison completed in 4.674 seconds, below the 120-second limit:

| Control | Result |
| --- | --- |
| Encode, 1,942 cases in two add-special modes | 8 mismatches, covering four unique cases; rejected |
| Raw pieces, 248,320 slots in two render modes | 496,640 exact comparisons |
| Worker text decode, 248,327 cases in two skip modes | 496,654 exact comparisons: 494,760 valid UTF-8 and 1,894 replacement-policy cases |
| Loaded classes / EOG inventory | Exact; five EOG IDs |
| BOS/EOS/PAD | Exact; the initial receipt does not establish the later full 13-accessor check |
| Five negative controls | All initially reported nonzero; the case-fold result was later found to inherit baseline errors and did not establish isolated mutation detection |

The four encoding differences are fixed corpus indices 351–354 and involve
Unicode 16 Egyptian Hieroglyph Extended-A scalars. The source and locked HF
regex engines classify those scalars differently. Copying the regex string
therefore fails to establish source semantics. Patching only those four cases
is not an accepted correction.

Original rejected receipt SHA-256:
`c257815a69f3d66939b2f1e175a1d67481f1cc044db2107e13a4fd5e2c9ef615`.
It is preserved privately as `native-comparison-v1-failed.json`. No derived
tokenizer was emitted and no preparation hook was enabled.

The worker's UTF-8 replacement policy is independently visible in locked Rust
`tokenizers` 0.22.2, `src/pre_tokenizers/byte_level.rs:155`, inspected source
SHA-256 `f9b5a747a0ad936f0c49cff8f32bc8f166126704ab02d5af0505ad032fe6d99a`;
package checksum `b238e22d44a15349529690fb07bd645cf58149a1b1e44d6cb5bd1641ff1a6223`.
This establishes the current worker String boundary, not every upstream
frontend's output behavior. Raw source pieces are qualified separately above.

## Correctness and lifecycle

No inference operator, model load, MTP, image or prefix-cache capability is
qualified. Tokenizer tests must not grant native readiness or product admission.
The pinned release has no image tower; runtime image transition checks are N/A
to this tokenizer-only attempt.

## Performance and memory

Not measured. CPU build/comparison duration is setup evidence only. No inference
speed, working-set, resource-admission or supported-host claim follows.

## Evidence, decision and limits

The next action is to establish the CPU oracle and report whether the proposed
representation can satisfy the worker's required source semantics. Preserve
failed builds and semantic counterexamples. Bind any successful tokenizer
derivation separately from the rejected borrowed-reference result.

## Follow-up history

### 20261112-EDGE0-03B: exact source Unicode classifications

Registered before successor comparison. Preserve the failed v1 result and
introduce `edge0-native-tokenizer-v2` only if the exact source Unicode
classification behavior can be represented. Derive category/whitespace ranges
from the pinned source with its own hashes and attribution, and expand every
relevant property use in the regex with correct character-class context.
No changes to vocabulary, merge order or token types are authorized as a fix
for classification differences.

Before tokenizer comparison, check every Unicode scalar outside the surrogate
range (1,112,064 values) against the exact source for each property used by the
tokenizer, with zero classification mismatches. Fix the property inventory,
derivation/program hashes and proof recipe before that check. Then repeat the
unchanged corpus/modes above, including all negative controls and the full 13
named special accessors, requiring zero mismatches. The candidate may advance
only on its own passing receipt; v1 remains rejected. A source-native
preparation route still reports loading/execution unverified and remains
evaluation-only even if tokenizer qualification passes.

The property inventory fixed before the check is source letter (mask 4), mark
(mask 16), number (mask 2) and whitespace (mask 256), including the complement
for `\S`. The source flag stream is little-endian unsigned 16-bit values;
its hash and the derived range asset identify the actual table. Compile the
explicit range regexes with the locked tokenizers backend and compare membership
for every valid scalar against that source stream before tokenizer comparison.
Add a sixth negative control restoring implicit Unicode properties (v1 split
semantics); it must detect the original encoding counterexamples in addition to
the five existing mutations. This control is fixed before the v2 comparison.

### 03B first v2 comparison: primary checks pass, mutation gate fails

The explicit-range asset has 16,847 bytes and SHA-256
`a8d7b7bea4f53942453a535e71b52a91098328eed16dcd466af69ed7a8225820`;
source flag-stream SHA-256
`ac10d2cb7eea9e4686334ca6cf7e4d37fa586504ed793702ac81f49b55d7fdd4`.
The source-derived asset preserves the full MIT text and source attribution.
Exhaustive membership checked 4,448,256 property/scalar pairs with zero
mismatches in 1.195 seconds. Property counts were 136,726 letters, 2,450 marks,
1,831 numbers and 25 whitespace scalars. Membership receipt SHA-256:
`bdb393f7a42ebd46ef287d37b4b672fe9cb1282b9fc9beae3b09f333adb19475`.

Candidate revision `edge0-native-tokenizer-v2-unicode`, JSON SHA-256
`f1d23198102cc49d8cb199600be2f5ec97d35b7540830b284010a425147066c4`,
had zero encoding, raw-piece, decode, class, EOG or full 13 named-special
mismatches in 4.94 seconds. Overall acceptance still **failed**: the case-fold
regex mutation changed split boundaries but produced identical BPE IDs in the
declared corpus, so the mutation test did not isolate the semantic difference.
The other five mutations were detected, including the implicit-Unicode
mutation reproducing four encoding failures.

This also corrects the v1 negative-control interpretation: its four case-fold
mutant encoding mismatches were inherited from the failing v1 baseline. They
did not prove the case-fold mutation itself was detected. Preserve that count
and the original receipt; do not use it to claim a sound mutation gate.

First v2 validator source SHA-256:
`3cfb78bea5346042a35f87792f25b39623f463b936a8a619c3470cc0bb95d326`;
receipt SHA-256:
`fa230b7448e466ca1afb79607801e23aff1ba51fc7891a13cb8ef4f7731f5971`.
No tokenizer was published or admitted.

### 20261112-EDGE0-03B.1: direct pre-tokenizer boundary oracle

Preregistered before the next comparison. Extend the pinned CPU oracle to
expose its actual `unicode_regex_split` behavior with raw text pieces, without
changing the source implementation. Preserve prior bridge/library/receipt
hashes and bind the extended oracle separately. Compare source and candidate
split boundaries exactly over the same 1,942 fixed encode cases and the
previously hashed case-fold witness (1,943 cases). Require zero positive
boundary mismatches and require the case-fold negative control to produce a
boundary mismatch against this independent source. BPE ID equality alone
does not satisfy that negative control. Preserve the other five mutations and
all preceding correctness gates. Text and split/token vectors stay in memory.

### Final 03B.1 result and environment-bound acceptance

The final helper is 25,075 bytes, SHA-256
`29097cf0ae44d7b2858b709fda6a484a75e0ccfd4eabd61ff0c42ad7bc3ec5db`.
The candidate JSON remains
`f1d23198102cc49d8cb199600be2f5ec97d35b7540830b284010a425147066c4`.
The fixed boundary corpus SHA-256 is
`bf94af69b727318542ee1933d9553e0f8a7085e1c7b05b424e396e6be488c1fd`.

The final independent CLI run passed all declared checks: 1,943 exact split
boundaries, 3,884 exact encoding comparisons, 496,640 exact raw pieces,
496,654 exact text-decode mode cases, all 248,320 token classes, all five EOG
IDs and all 13 named special accessors. Unicode membership again passed all
4,448,256 checks. All six negative controls were detected; the case-fold
mutation caused two boundary mismatches and zero ID mismatches, reported
separately rather than conflated.

Earlier 1,942-boundary runs omitted the separately preregistered witness and
are retained as partial-scope results. They do not close the 1,943-case gate.
The first full-boundary independent receipt was
`430b3102060c126f2f17a8414815249b215165dddaf30103b571d5ac0434c7b5`.
Its numerical/structural observations stand, but it omitted the Python binding
version. An initial preparation profile incorrectly named 0.22.2; the live
version guard rejected it before staging. That approval was revoked and the
profile disabled while the qualifier was corrected. This was a registration
error, not a numerical failure.

The corrected qualifier binds CPython 3.13.13, module and distribution
`tokenizers` 0.23.3 matching `uv.lock`, and the compiled extension's installed
RECORD digest. Extension SHA-256:
`83c58f9ec4937f86521c76ba5222215d3e7707a25818d5ea69604e230fd4a96c`.
Both version-mismatch controls reject an incorrect version. The repeated final
comparison passed unchanged; environment-bound receipt SHA-256:
`d74501849a329954290a58bb376bc3d59b513cc5af01a04b1da40e3e94052ca0`.
Qualifier source SHA-256:
`35b7f55bddabe068574c380fb9b6d058db5276ea7d683b4900fab7aef7afc178`.

The accepted profile is limited to Python source preparation with this source,
asset, program and proof. Rust 0.22.2 tokenizer loading, generation streaming,
native model execution and complete upstream frontend equivalence remain
unqualified. The profile does not enable app admission.

Reproduce with the exact CPU library/receipt inputs described above:

```sh
uv run --project tools/converter --locked --python 3.13.13 python scripts/validate_edge0_tokenizer.py --source runtime/edge0-local-evaluation.native-download --oracle-library runtime/edge0-native-tokenizer-2026-11-12/libedge0_oracle.dylib --metadata-only-gguf runtime/edge0-native-tokenizer-2026-11-12/edge0-metadata-only.gguf --oracle-receipt runtime/edge0-native-tokenizer-2026-11-12/oracle-feasibility.json --candidate-revision edge0-native-tokenizer-v2-unicode --property-oracle-library runtime/edge0-native-tokenizer-2026-11-12/libedge0_oracle_v3.dylib --property-oracle-receipt runtime/edge0-native-tokenizer-2026-11-12/boundary-oracle-receipt.json --output runtime/edge0-native-tokenizer-2026-11-12/qualifier-new.json
```
