# Config-derived Qwen3.5 families, 2026-10-27

This run extends merged BF16 dense Qwen3.5 admission to configuration-derived
decoder and vision dimensions. The schema-3 profile is
`qwen35-dense-bf16-merged-v2`, version `2`. Its supported operations, geometry,
indexing and storage limits are defined in the
[current family contract](../../VISION_FAMILY_CONTRACT.md). Repository names and
published size labels do not select an admission exception.

The implementation starts from `43cbeedf216ae63e42109bfeacfb729739c09c63` on
`codex/qwen35-vision-artifact-contract`. Actual 4B conversion, native execution
and signed-package checks are distinguished below from metadata, synthetic
header and CPU operator coverage for other dimensions. Passing local checks
does not certify fine-tune quality or transfer the official model's output
claims to a new checkpoint. The existing exact-reference catalog remains
unchanged.

## Governing scope and conformance

The baseline is `stark-multimodal-research.md`, SHA-256
`8f2d6864056665a981c6f6f77ee5a9b929693f5552402029c820b70d0be6c3d2`,
especially native/artifact changes, staged acceptance, private IPC and versioning.
The [earlier family run](2026-10-27-qwen35-family-installation.md) records the
user-approved extension from exact-only availability to locally tested merged
fine-tunes. The current user request is: “Proceed with making the Qwen3.5 family
contract configuration-driven so larger dense sizes and merged BF16 fine-tunes
work automatically, and double check your work.” The accepted scope derives
geometry from validated metadata/tensor headers without a per-repository or
per-size allowlist. This broadens structural coverage; it does not weaken the
official 2% numerical gate or authorize arbitrary tokenizer programs, operations
or architectures.

| Requirement | Governing source | Evidence and status |
| --- | --- | --- |
| Versioned full-model contract with complete required tensors and actual source metadata | Research native/artifact change 1; latest family scope | **Conforms.** Converter and native reader independently derive a bounded dense graph. Actual 4B exports contain 603 tensors; official offline replay matches all five native components. |
| Generalize supported dimensions without per-model exceptions | Latest user direction; current family contract | **Conforms.** Validated configuration determines dimensions, schedule, tied/untied readout and tensor shapes. Native reconstruction rejects unsupported operation flags and storage/indexing overflow. |
| Exact tuple before public image capability | Research recommendation; approved family amendment | **Accepted exception.** Compatible family artifacts enable images as `locally_tested` only after local checks. Exact `reference_verified` evidence remains catalog-bound. |
| Preserve model-owned tokenizer/processor semantics | Research model/processor contract and versioning | **Conforms.** The two reviewed tokenizer programs retain intentional behavior differences. Both equivalent processor serializations are bound by their own raw hashes. Unknown processing programs and conflicting recipes fail. |
| Native numerical correctness under the existing gate | Research staged acceptance 2/4; native implementation | **Conforms for the measured fixtures.** Official 4B normal/boundary cases, DeskForge's two own-reference cases and legacy 0.8B regression cases are bit-exact: eight cases total. |
| Loaded-model capability intersection, private transport, fresh requests and recovery | Research staged acceptance 3 and private IPC | **Conforms for completed checks.** Actual family/version/vocabulary enter check identity; focused boundary tests pass. The signed official 4B candidate passes installation, repeated image use, cache reuse and cleanup. |
| Preserve old artifacts and source-bound history | Research versioning; repository provenance rules | **Conforms.** Real schema-2/version-1 reader checks pass. A fresh official 0.8B conversion preserves prior component bytes. Earlier archives and the exact execution catalog are unchanged. |
| Complete the selected larger-model application and boundary qualification | Latest work scope; research staged acceptance 4/5 | **Conforms within local qualification scope.** Official 4B boundary parity and DeskForge native, own-reference, signed-package, raw-install/HTTP and exact-source Swift checks passed. Public notarized distribution and gesture automation are outside this run's claim. |
| General visual/OCR, resource/latency and public-distribution certification for each new tuple | Research staged acceptance 4/5 | **Not applicable to this local-family completion claim.** No new exact catalog record, general accuracy/performance qualification or notarized-release claim is made. The measured numeric fixtures and local execution evidence retain their separate scopes. |

The implementation **conforms with the accepted availability exception above**.
The selected larger-model and merged-fine-tune checks are complete. This verdict
does not convert metadata coverage into runtime qualification for other sizes,
nor qualify a public notarized release.

## Version-2 implementation boundary

Version 2 retains the native Rust/Metal tower, feature fusion and Qwen3.5 M-RoPE
route. It derives complete tensor shapes from bounded scalar dimensions and
requires the supported attention/DeltaNet hybrid graph, dense SiLU MLPs, RMS
normalization and image processor recipe. It supports declared tied or untied
embeddings. Required matrices/convolutions remain BF16; the existing explicit
FP32 control-weight exceptions remain. Declared auxiliary MTP weights are omitted
under the existing conversion rule.

Bounds include a decoder width of 16,384, MLP width of 131,072, at most 256
decoder layers, at most 256 heads/groups, and at most 1,048,576 vocabulary rows
or declared positions. Vision bounds include 64 blocks, width 4,096, MLP width
32,768, 64 heads and 16,384 learned positions. The native weight file is bounded
at 128 GiB, each tensor at `2^32 - 1` elements, and configuration/header at 1 MiB
each. Divisibility, operation, schedule and full tensor-layout checks apply in
addition to these maxima.

Official 0.8B, 2B, 4B, 9B and 27B configurations and weight-index inventories
were inspected. Synthetic native header fixtures also cover these geometries and a size absent from published
model names. CPU execution checks cover the 72-wide vision attention heads,
4,304-wide vision MLP and 1-by-1 learned-position interpolation edge. These checks
establish supported metadata/operator behavior. They do not establish full-weight
conversion or runtime qualification for 2B, 9B or 27B.

The fixed version-1 0.8B family and exact schema-2 path remain readable. Older
readers reject the unknown version-2 family. The host binds the actual admitted
family/version and vocabulary to receipt identity, checks token IDs against that
vocabulary, and uses geometry-dependent state/scratch allowance for memory
preflight. Checker policy version 2 invalidates old check identities. Available
memory remains an observation, not a reservation or peak-memory guarantee.

## Reviewed processing variants

Version 2 accepts the stock reasoning template
`a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`
in addition to the earlier two reviewed templates. The existing boolean thinking
control is retained; the app and CLI explicitly select it and default off.

It also recognizes two complete tokenizer programs:

| Canonical tokenizer SHA-256 | Behavior |
| --- | --- |
| `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4` | Existing letter/combining-mark-aware splitting program. |
| `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` | Reviewed letter-only splitting program with different ByteLevel trimming/decoder flags; version 2 only. |

The programs share vocabulary, model and added special tokens. The latter is
preserved as its own program. It is never rewritten to the former. JSON key order
and spacing can vary only within a complete reviewed program fingerprint.
Three ASCII cases matched, all four image wrapper/special IDs were equal and
atomic, and two of six combining-mark cases intentionally differed. This is
tokenizer behavior evidence, not model quality or numerical parity evidence.
Schema 2 and family version 1 retain the original tokenizer gate.

The processor accepts recognized standalone or unified image metadata, including
both files when normalized recipes agree. Source class aliases and explicit
defaults normalize to the supported still-image recipe. Every actual source
file/hash remains in provenance. This does not admit arbitrary processors or
video execution.

## Actual checkpoint conversions

| Checkpoint | Immutable revision | Source weight bytes | Emitted tensors / weight bytes |
| --- | --- | --- | --- |
| [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 9,319,828,096 | 603 / 9,080,546,160 |
| [`docling-project/DeskForge-Qwen3.5-4B`](https://huggingface.co/docling-project/DeskForge-Qwen3.5-4B/tree/72337e2fe9e8489ee324c40ea16da49d14b480ff) | `72337e2fe9e8489ee324c40ea16da49d14b480ff` | 9,078,620,432 | 603 / 9,080,546,160 |

Both actual full checkpoints were downloaded and converted through their own
metadata without executing custom model code. The official conversion and its
offline manifest replay produced identical weights, configuration, tokenizer,
processor and template hashes. The independent source review separately checked
the actual converted artifacts and source identities.

Two independent complete tensor-range comparisons also establish that DeskForge
has changed decoder weights beyond safetensors metadata. The layer-0 MLP up
projection (94,371,840 bytes) and layer-12 MLP down projection (47,185,920 bytes)
both have different payload hashes from official 4B. The confirmation streams
at most 8 MiB at once and records only counts and hashes.

| Native component | Official 4B SHA-256 | DeskForge SHA-256 |
| --- | --- | --- |
| Weights | `a3b7889d7f83ce3cadf114110110f919db50091efca5c9373d30e703ce0570b2` | `96696f438f1abeb7b651d0c9492101900d05de7934d6467c558e59a91d1b4025` |
| Configuration | `6db041d32d78edea53e7a5422c1863f5362ece507909c778295f71fb20d76ad0` | `aef31e722fb01832e8f813955f3a0f051a15147d81c6314e63183304452134aa` |
| Tokenizer | `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4` | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| Processor | `6a970fd06f30e6943b3e2c14d5d3b42d49b06cf99b99103d56689bef462d90f8` | `6a970fd06f30e6943b3e2c14d5d3b42d49b06cf99b99103d56689bef462d90f8` |
| Template | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |

DeskForge's alternate tokenizer and both processor files were preserved and
validated. Their shared normalized processor recipe does not erase the raw
source provenance or imply equivalent generated answers.

The converter source digest is
`7cdd68f1cf618e4936bf23fa65795c193ac5cff9f55673c97e5fd4a202b69ac9`;
the locked Python dependency digest remains
`df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5`.
Conversion receipts report `supported_family_unverified` and
`execution: "unverified"`; conversion does not award reference or local execution
evidence.

## Official 4B reference and native execution

The CPU reference loads the actual official 4B source using pinned Transformers
revision `2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397`, including the root and
child numerical classes. Selected numerical symbols match the pinned source.
It verifies source sizes/hashes before and after execution and requires FP32
rotary buffers. The new reference manifest records the executed model, pinned
loader/helper, generator, source files and reused pixel-patch provenance.

Reusing the earlier verified BF16 image patches is appropriate because the 4B
source processor has the same resize, normalization, temporal and patch-order
recipe. The generator checks patch payload hashes before use and binds their
metadata. Its imports/source loading now fall inside a fixed-error guard;
incomplete or unpinned-child reference modes are refused. No exception payload
or generated content is logged.

| Complete-logit fixture | Input tokens | Image tokens | Vocabulary | Metal result |
| --- | --- | --- | --- | --- |
| 256-by-256 square | 83 | 64 | 248,320 | BF16 mismatches 0; relative RMS 0; cosine 1; top-1 match true |
| 47-by-81 rectangle | 96 | 77 | 248,320 | BF16 mismatches 0; relative RMS 0; cosine 1; top-1 match true |
| 1-megapixel boundary | 1,043 | 1,024 | 248,320 | BF16 mismatches 0; relative RMS 0; cosine 1; top-1 match true |

All three official 4B cases pass the existing 2% gate without changing its
threshold. The boundary reference uses the actual 4B weights and processor,
verified source pixel buffers, 4,096 vision patches and 1,024 image tokens. Its
1,043-token input crosses the prefill chunk boundary within a 2,048-token
context.

DeskForge separately loads its own full weights and `06b9509…` tokenizer through
the complete pinned root/child reference. The two square/rectangle fixtures
contain 83/96 input tokens and 64/77 image tokens. Both match native Metal
bit-exactly: relative RMS and BF16 mismatch count are zero, cosine similarity is
one, and top-1 matches. Reference weight identities match the verified DeskForge
source. This compares the fine-tune with its own reference, and measures only
those two fixtures; it is not a general fine-tune accuracy certification.

The new worker also replays the prior official 0.8B normal cases and 1-megapixel
boundary bit-exactly. Across official 4B (three), DeskForge (two) and legacy 0.8B
(three), all eight complete-logit cases pass. The source-bound reference records
remain separate, and no new tuple was added to the exact execution catalog.

## Signed package and compatibility

| Component | SHA-256 |
| --- | --- |
| Rust 1.98.1 release generator | `0c5756fda8540df51e21d5a900612a38befec77093f904ba5a7a215e396dda2d` |
| Separately built optimized scorer | `1bd74f87859f8d496a78024fb4caf60337acbb7ef14873adc67f9bff731665cd` |
| Developer ID signed generator | `67f80d504f44b69e926992407229704149fcb0ba4956babb69e76fe5ab7b725d` |
| Signed Swift executable | `4de63d9956af3782bc586edea4c9b2e010d3009d5afa5f5ff8a1d23583e3eeea` |
| Embedded app manifest | `03c25901e6224e6c69c88c4fffb1105f9a975ea37a8e4ddecc59eb446936e094` |
| Bundle inventory, 3,343 files | `72529173e66c9e01cee3dadf4ccee96742ce627a8262536958f30e6a1c5dc1a8` |

The signed official 4B candidate passed clean-copy/signature validation, embedded
interpreter imports, current native/adaptation notices, installation checks
before library publication, deduplication, unchanged receipt reuse on reinstall
and Run, two image turns with usage, same-image follow-up and staging cleanup.
It reported `locally_tested`; its ordinary exact catalog match remained false.
Reference numerics are recorded separately rather than inferred by the UI.

Real legacy startup checks also passed for the schema-2 official artifact and a
schema-3/version-1 family artifact under the new worker. The fresh official 0.8B
conversion preserved the five prior native component hashes. Models and raw
artifacts remain ignored; previous source-bound records were not rewritten.

The developer DeskForge worker passed automatic checks on first startup and
reused its private receipt on the second. It reported the actual version-2
family, vocabulary and alternate tokenizer identity. Owner-only receipt
permissions, normal-shutdown preservation and zero staged files passed.

The same signed candidate then passed three DeskForge application stages:

1. Install the converted artifact before library publication, deduplicate a
   repeated installation, reuse the receipt on Run, and complete two image turns
   with usage, a resupplied follow-up image and cleanup.
2. Convert the full raw local checkpoint offline, verify its own processor and
   tokenizer, finish checks before publication, and serve a live HTTP inline
   image completion with usage and `locally_tested` evidence. The four payload/
   sidecar hashes match the remote conversion; configuration hash
   `ad673b9a7eaf6211f4bd0745fb5040da06132be767b58f98d498383ef8645c7c`
   records the actual local source identity.
3. Compile the exact production Swift controller and verify its source against
   the package manifest; exercise preview, first/follow-up images, invalid-image
   recovery with a fresh valid image, reset, Stop during image work, Run recovery,
   token metrics and staging cleanup. The bundle remains unchanged.

All three stages passed with the same signed worker, Swift executable and bundle
inventory recorded above. The Swift source SHA-256 is
`b658c84f642debc3082aac121bc9e854c3e2601b90df96e947eaaee8b11a342e`.
The controller harness drives methods; SwiftUI gestures were not automated.

The candidate has Developer ID signing, hardened runtime and secure timestamp.
It is not notarized. This is local package validation, not a public-release
qualification. It was not installed over the user's existing app or model
library. Generated-answer quality was not assessed by the local or application
checks; measured reference parity remains the separate two-fixture result above.

## Verification and independent review

| Check | Result |
| --- | --- |
| Full converter/host Python suite | 4,702 passed, 88 skipped; 128.59 seconds |
| Full native engine suite | 570 passed, 30 ignored; 7 integration checks and 3 doctests passed; engine duration 199.28 seconds |
| Worker unit checks | 41 passed |
| Swift controller checks | 29 passed |
| Packaging checks | 52 passed |
| Policy, Ruff and scoped formatting | Passed |

The independent host review covered exact version pairs, v2-only stock-template
admission, actual vocabulary accounting, cache invalidation and bounded scalar
memory estimates. Its focused pass ran 186 checks successfully. Native and
converter reviews independently checked dynamic shape reconstruction and source
contracts. The reference-harness review required verified patch/helper
provenance and guarded startup errors before reference generation. Those fixes
were applied and independently inspected.

## Private evidence identities

Raw native/reference artifacts, models and binaries remain under ignored
`runtime/vision-config-driven-validation-2026-10-27`. Only aggregate results and
hashes are published here. Prompts, responses, token vectors, image bytes and
credentials are absent from these published records.

| Evidence | SHA-256 |
| --- | --- |
| Actual source metadata inventory | `4682e7a208749ea5fd7a341dccef8bf84636f2d2c895a9db0b973535dc2d2491` |
| Reviewed tokenizer behavior | `491b303ef3c3bcbb09499e010293ed23c03cd2eeb80cc1fe082a36ee4e52fb73` |
| Official 4B conversion receipt | `6b77ca071265216834e98ffe3924e41dd347e948ae0a7aae66972cb2212e0831` |
| Official offline replay receipt | `5cb0ce98ce78f8851ff1994912254a869631d60f2c72ea1a0be10621941bb970` |
| DeskForge conversion receipt | `957afa0a784b85ba1ed9f33e5e552861f60f22547dd6681faafea6eec4365aad` |
| Independent converted/source review | `af0e34019603e7a3bf308119758a22de032c97911de3ae23d50c84c663b33440` |
| Changed decoder tensor payload proof | `c350e6b4b290f795687883179893af798e3b02f87008ba7b39a200947cc43bcc` |
| Tensor-range verification reader | `ce3a175994c3d363e4c43ce9e68e0d041387369581bdb4ab7df69369c1a75588` |
| Complete official 4B reference manifest | `111f6a74a5b7affaea11884dcdcaa87fdf18e977227d4c47724ec1aa0e2fb3d3` |
| Official 4B normal-fixture Metal log | `5a40a70fb176b29db0950a55e357810035976458c7595d51d5b02c0ad5ac5ead` |
| Official 4B boundary reference manifest | `a159e3d5218876d8cbe0a6727446142c6979ff473d5aca6f8213a4fecfcc0cc5` |
| Official 4B boundary Metal log | `4265a9416676aaff36e909e6129113406720f448fde566218f9595dd67e59aa6` |
| DeskForge complete reference manifest | `34d664a4b55bd1364cb1ede30c38300af2194f602d73bcee53e60ef6199734dc` |
| DeskForge two-fixture Metal log | `77b42493a2addc2cbb07a48626c2969065c98efdc135370eba5ad42a52a7b639` |
| Legacy 0.8B normal-fixture Metal log | `b80d3f1397a584f75663831259c508a93d14f7a2d45a820090a413dd6adbc851` |
| Legacy 0.8B boundary Metal log | `e5b7d873af07a4a47fe074e9882b5acc71d5b8eeb2e5bdf7761041d5fa50eaaa` |
| Signed candidate identity | `2cf57d02773fb4ab41c1cc2392cb46548b3ab4a86af6240f8a768562972c920b` |
| Signed official 4B package checks | `073dcd4bc9704e94dd136e1e1d8fd52baa07ea9c6af58f396a8a60247a536d3a` |
| Developer DeskForge startup/reuse checks | `45af423c0b01b8552cdb9e4d6d946079e48e62693d11471f9ab35667ac94a96f` |
| Signed DeskForge converted-model checks | `8858647eaa737dae1ddb23097b8fed83cafbb8a375e6fa80c3ebf9eaf7642f84` |
| Signed DeskForge raw-install/HTTP checks | `be188be581828e4371af68bc8964fa867328c5047dbad65d3c33422a1091d424` |
| DeskForge raw-install/HTTP driver | `eeb69de350ec3b5f08476660bb2584163aed4208ea4ca53199c671e7fbe1183a` |
| DeskForge exact-source Swift qualification | `1909fd3cd7606213b26c2b12defdc98d4df494e15fbca7abb832ea0622540b43` |
| Swift qualification driver | `497993fbf0841729ff5a973688d04a7762d7afbaa95c0752c7b74ecb94d54240` |
| Three-stage DeskForge package runner log | `d90ce5085023460ab23153174150a3983cac648f514c1146206f41302c03aab9` |
| Legacy real-reader checks | `5f388130d18b51411256b815a1a64cd4eb00df46336f68b7ccf491dbf412f0a4` |
| Fresh schema-2 byte preservation | `cb0149e21e3cf11d031866e565dae6b4a1fc89178500e2e2845ac4d2d8bdc808` |
| Full Python log | `c260feefb3773e980a382cfd544326f6ac178ad2442f8a241b9a9609651ff6b0` |
| Full native engine log | `339ed833f6862b3f875899b286a5d4f89610f0be81b37b4691a83d6dcde5d818` |
| Worker unit log | `a46c8dbc4e8e2d1e2b4033a75bb56b184633036b1e4b678ad311224368e078e5` |
| Swift controller log | `4fa81d21993480c1f9b960b0e4687d1a11768dc99ae40714b06975b75ba3a239` |
| Packaging log | `f7bab492b95d2201c05d0e8c0fd1e511f604ab797ca5ef3177509cc6bd9f151c` |

The completed records bind actual checkpoint, processing, native and package
identities. Higher-size metadata/operator evidence remains scoped separately
from these measured full-model results.
