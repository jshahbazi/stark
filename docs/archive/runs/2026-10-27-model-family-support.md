# Model-family inventory and installation checks, 2026-10-27

This work starts from `f3ded18d0555bce3d0830c1464be6380836d8104` on
`codex/qwen35-vision-artifact-contract`. The requested scope is a maintained
inventory of supported text, Decision and multimodal families, with an early
warning when an installation requests an unsupported path.

## Implemented behavior

The converter adapter registry, CLI reports and menu bar app share one
schema-1 family inventory. It contains 13 architecture entries: 12 language
families and ModernBERT classifier conversion. Language text/Decision paths
remain conditional on full conversion, compatible metadata and successful
native execution. Classifier conversion has no desktop runtime mode. Image
support remains confined to the implemented dense BF16 Qwen3.5 family.

Raw-source installs inspect small metadata before weight downloads. A
conditional result continues automatically; unsupported and inconclusive
results stop with distinct explanations. Available alternatives are limited
to modes not excluded by the inspected source. A source-wide incompatibility
does not suggest another mode that has the same incompatibility.

Metadata requests and local reads have size limits. Remote metadata receives
an immutable-revision HEAD size check before payload download. The isolated
preflight child has a 64 KiB report limit, cancellation and a deadline. Actual
conversion repeats metadata admission and pins the primary source revision.
External catalog metadata is independently resolved and checked; it is not
claimed to reuse every byte or revision from the preliminary check.

The Swift install uses a fresh correlation identifier and discards stale
reports after source/mode changes, cancellation and restart. Rejection preserves
the running model. Converted folders retain native admission, including Chat
imports of existing vision artifacts. Reports do not grant runtime image
capabilities, checkpoint quality, reference qualification or calibrated Decision
probabilities. Source/receipt serialization and native workers are unchanged.

Current behavior and report fields are documented in
[model-family support](../../MODEL_SUPPORT.md).

## Evidence boundaries

The real public `Qwen/Qwen3.5-4B` metadata check resolved revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and returned conditional image support.
It fetched four metadata files totaling 28,017 bytes and cached no weight files.
The public `Qwen/Qwen3-0.6B` image check resolved revision
`c1899de289a04d12100db370d81485cdf75e47ca`, read its 726-byte configuration and
returned unsupported image use with conditional text/Decision alternatives.
Spy-Hub tests additionally assert that rejected metadata never invokes weight
downloads or weight HEAD requests.

All private candidate bundles, models, synthetic fixtures, captures and raw logs
remain in the ignored `runtime/model-family-support-validation-2026-10-27/`
directory. Earlier source-bound validation records are preserved. This run does
not extend the exact-reference image catalog or repeat numerical parity for
unchanged native code. The installed user application is not replaced.

## Completed verification

| Check | Result |
| --- | --- |
| Full locked Python 3.13.13 converter suite, final frozen source, Hub offline | 4,803 passed; 88 skipped. |
| Independent registry/source/host review | 159 passed; no remaining actionable findings. |
| Focused converter/source regression pass | 519 passed. |
| Exact-source Swift controller checks | 43 passed. |
| Application builder tests | 23 passed; 12 subtests passed. |
| Swift compilation | All application sources typechecked and the final executable built successfully. |
| Family-sheet visual checks | Collapsed inventory, scrolling to ModernBERT and expanded family details inspected; no horizontal overflow. Captures use a temporary exact-source harness, not installed-app gesture automation. |
| Repository policy, converter Ruff and changed-Python formatting | Passed. Broad formatting inspection also found four unchanged baseline files needing formatting; they are outside this patch. |
| Developer ID candidate | Signed and verified, including 37 Mach-O files, hardened runtime and secure timestamp. Not notarized. |
| Packaged raw-source installation | Existing DeskForge Qwen3.5-4B checkpoint converted fully offline; conditional preflight preceded conversion, and local image checks preceded library publication. |
| Packaged model execution | Run reused the installation receipt; HTTP image completion and token-usage checks passed. Generated text was not logged. |
| Packaged rejected installations | Unknown family, unsupported image mode and unidentified metadata all stopped before conversion/publication. The loaded model, library, image receipt and live image API capability were preserved. |
| Private media cleanup | No staged attachment or transient media files remained. |

The fresh raw-local installation reproduces all five artifact component hashes
from the previous configuration-driven run. This includes weights
`96696f438f1abeb7b651d0c9492101900d05de7934d6467c558e59a91d1b4025`
and configuration
`ad673b9a7eaf6211f4bd0745fb5040da06132be767b58f98d498383ef8645c7c`.
This is conversion determinism and local execution evidence for that checkpoint;
it does not establish general answer quality or qualify every family entry.

## Candidate and private evidence identities

| Item | SHA-256 |
| --- | --- |
| Unsigned generation worker, unchanged native build | `0c5756fda8540df51e21d5a900612a38befec77093f904ba5a7a215e396dda2d` |
| Signed candidate generation worker | `3e0f6e7f411d36847ff329d49cc4a6e3d066880e970dbc7510bf02771882317f` |
| Signed Swift executable | `e6cee8930cdb894f8df87381ea1aee09eb9aaf220ca8101c4ea84243ce03ec5d` |
| Candidate application manifest | `b7422961b96860075d763a8249ea891d1d3960f61caa89d24809b8d71a8dd1f4` |
| Unchanged exact image execution catalog | `2222b577fe5a88407ad0377e44cfbdbaeb4af7e0a51c03768a0d8f679bb5316d` |
| Final full Python-suite log | `bb107e97410d9b17ae9c01a43f1e1fdef8c4dc03cc5fa6dcfe0fb5676a5b7132` |
| Final Swift-controller log | `2a608f7bc5c66ec526d1edd5170fc2311d2b15dca19dfec235610ca1e50e1a0a` |
| Builder-test log | `3ece0b88c3b29dad8b5fc33519c6cd31a9ce0c79e4fcaed8f40dc73f3e55eb20` |
| Packaged source-install/HTTP aggregate report | `3ccbabeff3eb887ab7111a74ff2899ffc16f2a4c74898388d0a17e2aa056a93b` |
| Candidate signature/identity report | `e9ea250050c39570455a8c0b0107c77a3fae0e721e80c356d51bf97fdd7d7166` |
| Real remote 4B metadata preflight | `2e56cff29c41eca43d72c6d014e0309c1b3034f08f2eebfa71f6468d2af67a07` |
| Real remote unsupported-image preflight | `cdc885623226f40edeb2b59d4cdd42a009a4e5fa627f152baa60338918878a1b` |
| Exact-source sheet capture provenance | `a6b7f08faa25d56d64cd2a9d8cdcd3ee9db681c30c568465eabe383cba580dc5` |

An initial regression run retained an obsolete assertion for the new immutable
revision error wording; that assertion was corrected without weakening the
rejection. Independent review also found misleading cross-mode alternatives and
an early mode rejection that masked shared exclusions. Both were fixed and
covered before the final full run. Earlier logs and candidate identities remain
separate; the table above identifies the final candidate and passing evidence.
