# Qwen3.5 family admission and automatic installation checks, 2026-10-27

This run exercises a genuine changed-weight Qwen3.5 checkpoint through remote
conversion, offline replay, native admission, automatic image checks, desktop
installation, the local HTTP API and the Swift controller. The implemented
profile is `qwen35-dense-bf16-merged-v1`, version `1`, limited to complete merged
BF16 models with the supported dense 0.8B geometry and recognized metadata.
The [current family contract](../../VISION_FAMILY_CONTRACT.md) defines admission
and receipt behavior.

The fine-tune receives `locally_tested` evidence. This run does not certify its
visual accuracy or compare its outputs with the official checkpoint. The
official 2% reference gate and exact execution catalog remain unchanged.

## Governing scope and approved amendment

The baseline is `stark-multimodal-research.md`, SHA-256
`8f2d6864056665a981c6f6f77ee5a9b929693f5552402029c820b70d0be6c3d2`,
especially its native/artifact changes, transport, private IPC, staged acceptance
and versioning sections. Its first milestone required an exact official tuple
before public image availability. The current
[native implementation](../../MULTIMODAL_IMPLEMENTATION.md) preserves the
native Rust/Metal route and the numerical/reference boundary.

The user then explicitly requested: “Proceed with the model-family contract and
automatic installation checks, starting with merged BF16 Qwen3.5 fine-tunes,
and double check your work.” This authorizes the scoped extension from exact-only
image availability to compatible family artifacts with local execution evidence.
It does not replace reference qualification, widen geometry/dtype coverage, or
permit a fine-tune to inherit the official model's accuracy or parity claim.

| Requirement | Governing source | Evidence and disposition |
| --- | --- | --- |
| Versioned full-model admission; own processor/template provenance; explicit text mode | Research native/artifact changes 1 and staged acceptance 1 | **Conforms.** Schema 3, 383 validated output tensors, source-bound metadata, ordinary remote conversion and identical offline replay. Native independently validates the emitted profile. |
| Exact tuple before any public image capability | Research recommendation and staged acceptance 4; explicit user amendment above | **Accepted exception.** Only the supported family can become `locally_tested` after checks; `reference_verified` still requires the unchanged exact catalog. |
| Preserve native fusion/positions and the official numeric acceptance gate | Research native/artifact changes 2–4 and staged acceptance 2/4; current implementation | **Conforms.** This change adds admission and checks without new kernel arithmetic. All three official final-logit cases remain bit-exact under the existing gate. |
| Bounded private media transport, fresh transactions, failure recovery and complete-request serialization | Research transport/private IPC and staged acceptance 3 | **Conforms for this extension.** The existing 1 MiB worker framing and private media path remain; focused boundary tests and real malformed-image, follow-up and Stop/Run recovery pass. Receipts contain metadata only. |
| Advertise loaded-model evidence; fail closed for unknown combinations | Research versioning; current family contract | **Conforms.** Unknown profiles/schemas, absent vision contracts and failed checks do not publish a schema-3 installation. Registry claims cannot enable live API images or reference verification. |
| Packaged execution and capability-aware Swift integration | Research staged acceptance 5 | **Conforms for this run's application scope.** The final candidate passes converted/raw-source installation, live HTTP images and exact-source Swift controller replay. Gesture automation and public notarized release are outside this run's completion claim. |

The family extension **conforms with the accepted exception above**. No
unresolved deviation remains within this conversion, local-check and packaged
execution scope. This verdict does not mark the full public-distribution stage
complete.

## Source and implementation identity

The implementation was developed on `codex/qwen35-vision-artifact-contract` from
`4a4f377e7ac3b2c7dcf2760b002afede63f289f2`. Candidate manifests bind the patched
production source files individually; production files were frozen before each
candidate build. The generator and optimized scorer were built separately with
locked Rust `1.98.1`, preserving their separate feature sets.

| Component | Identity |
| --- | --- |
| Source checkpoint | [`Ayansk11/FinSenti-Qwen3.5-0.8B`](https://huggingface.co/Ayansk11/FinSenti-Qwen3.5-0.8B/tree/9e209de33632a073235bab5e58482cb3f20b3103) |
| Source revision | `9e209de33632a073235bab5e58482cb3f20b3103` |
| Source weight bytes | 1,746,942,600 |
| Source weight SHA-256 | `b9a86d06daa533d06dab69e9f15651c6b3106388ed2a698f3f6dfc874fe69dc9` |
| Source index SHA-256 | `d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e` |
| Source processor file | `processor_config.json` |
| Source processor SHA-256 | `14932921ca485d458a04dafd8069fbb0a4505622a48208d19ed247115801385b` |
| Developer generator SHA-256 | `d82062ad04e2bc6dd5fea0507e93e6212bbe09d205cbcc5a7d7c810cde0d3737` |
| Optimized scorer SHA-256 | `1f4a8f7a7bde60ce4f977505ec609078fda92a375a9ee2c67f64564b84d65f83` |
| Developer ID signed generator SHA-256 | `579eec4f06ba18d1548ed53ce1061e1a84213cb945e02fd9303ee8dd40f8038e` |
| Final signed Swift executable SHA-256 | `cd87f9014e0515aed606f18fea89547eadebde4d04b38693f9ac55feb51343fb` |
| Final embedded app manifest SHA-256 | `f7a795f67e3476ebda49a620be9b149143278bf7ba84c4a641892f2f56e1d22f` |
| Final bundle inventory SHA-256 | `1de0434f7a867f1e18a95ca227ea75a76a2262a211ed0d578da8725ff7131d3e` (3,343 files) |
| Runtime | Apple M5 Max, 40 GPU cores, Apple GPU family 1010, MXU enabled; macOS 26.7.1 |
| Python | 3.13.13; Jinja2 3.1.6, MarkupSafe 3.0.4, tokenizers 0.23.3, psutil 7.2.2 |

The checkpoint contains full decoder and vision weights. No custom model code
was downloaded or executed. Its own normalized tokenizer, reviewed template
variant and unified processor metadata were used. No official processor or
template was substituted. The profile recognizes bounded serialization/default
annotations; tensor geometry, operations, required BF16 payloads and processor
recipe remain constrained.

The initial remote conversion and offline replay used converter source digest
`8b14812e16129fce9344f620a2be245db6b5da1b2f51149545ba68e1c9720142`.
Two Python files were subsequently formatted without functional changes. The
final candidate uses digest
`f341c18f8e3f1a9f191788e0a8879e138d5c8eb6d3dfecde8f3125b8a2553706`,
with lockfile digest
`df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5`.
It retains the signed native workers and repeats the raw-source conversion,
application and controller checks. Earlier candidate records remain preserved.

## Conversion and replay

The ordinary remote CLI invocation selected the family automatically without a
catalog entry or an explicit `--metadata-profile`:

```sh
uv run --project tools/converter --locked stark-convert \
  Ayansk11/FinSenti-Qwen3.5-0.8B \
  --revision 9e209de33632a073235bab5e58482cb3f20b3103 \
  --artifact-mode vision-language --dtype bfloat16 \
  --write-source-manifest runtime/converter-sources/finsenti-vision.json \
  --output models/finsenti-vision
```

Conversion emitted schema `3`, 383 tensors and 1,707,029,400 weight-file bytes.
The receipt reported complete source validation,
`native_loading: "supported_family_unverified"` and `execution: "unverified"`.
Offline replay from the private source manifest produced identical hashes for
all five native artifact files:

| Remote conversion and offline replay | SHA-256 |
| --- | --- |
| `model.safetensors` | `5068b0f3e3cb5a0366fdb0a37f5de60b76e5ab7b5fe52fa256b6d52bd43cef25` |
| `config.json` | `efe6d21c1686cc9f2af8dd46479a12c2b7e9c2ab98c683b46ae1adff364e3945` |
| `tokenizer.json` | `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4` |
| `preprocessor_config.json` | `6a970fd06f30e6943b3e2c14d5d3b42d49b06cf99b99103d56689bef462d90f8` |
| `chat_template.jinja` | `7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67` |

The packaged app also converted the downloaded raw checkpoint offline as a
local source. The four payload/sidecar hashes matched the remote conversion.
Its configuration SHA-256 was
`bb88d5cec8ffe77c35d345eb2c16e675ab69304d0ccae8f81c8f8a9464faac75`:
the configuration records the actual local source identity instead of claiming
the remote repository/revision.

## Automatic checks and application behavior

The developer worker's first startup ran text generation, malformed-image
recovery, two fresh valid-image transactions, image-token accounting and staging
cleanup. It reported `locally_tested`; the second startup reused the private
receipt. Normal shutdown preserved the receipt, owner-only permissions passed,
and both startups left zero staged files.

The signed candidate was tested through its embedded interpreter and copied
runtime resources in isolated storage. The model installation checks completed
before library publication. Reinstall preserved the model identity and receipt;
Run reused matching evidence. Two image turns passed with usage information and
the image resupplied on the follow-up. The ordinary exact execution-catalog
match remained false, and the evidence level remained `locally_tested`.

A separate raw-checkpoint installation exercised the packaged offline converter
and pre-publication checks. Its live HTTP API accepted an inline image,
generated a completion with usage, and reported local evidence from the active
worker. Run and Stop preserved the matching receipt, and cleanup left zero staged
files.

The Swift harness compiled the exact production controller source and checked
it against the candidate's packaged source manifest. Using isolated installed
resources, it passed image preview, the first image turn, a same-image follow-up,
invalid-image recovery with a fresh valid image, reset cleanup, Stop during an
image request, Run recovery with a fresh image, token metrics and staging cleanup.
The harness drove controller methods; SwiftUI gestures were not automated.

The candidate used Developer ID signing with hardened runtime and a secure
timestamp. It was not notarized, and this run does not qualify a public release.
It was not installed over the user's existing application or model library.

## Independent review and regression coverage

An independent host review found four defects before completion:

1. Desktop shutdown terminated the child before `Worker.close`, causing an
   expected SIGTERM exit to invalidate a successful receipt. `Worker.close` now
   owns process-group shutdown. A real subprocess/store regression distinguishes
   normal closure from unexpected exit.
2. Two installation config reopens bypassed the bounded reader. Both now use the
   bounded regular-file path; oversized and FIFO fixtures cover the boundary.
3. A missing vision `artifact_contract` could bypass family installation checks.
   Missing, null, malformed and unsupported contracts now fail admission.
4. Receipt identity omitted the staging and hashing helper implementations.
   Their hashes now participate, with regressions requiring fresh execution after
   either helper changes.

The independent post-fix pass ran 161 focused Python checks and 29 Swift
controller checks successfully. Fixture expectations were updated for the new
evidence fields and process-shutdown ownership. A controller command that lacked
the converter environment failed its import; the documented locked-project
command passed all 29 checks. A first numerical command lacked Cargo in `PATH`;
the corrected command passed. These harness/fixture corrections did not change
the reference gate or convert local execution into a quality assertion.

## Validation results

| Validation | Result |
| --- | --- |
| Final formatted-source converter/host Python suite | 4,631 passed, 88 skipped; 121.63 seconds |
| Reviewed packaging checks | 52 passed |
| Independent Swift controller checks | 29 passed |
| Repository policy and Ruff lint | Passed |
| Scoped Rust formatting | Passed |
| Final-source full native engine suite | 565 passed, 30 ignored; 7 integration checks and 3 doctests passed. Engine unit duration: 196.03 seconds. |
| Worker unit checks | 41 passed; worker sources unchanged by final metadata-only refinements. |
| Final-source family/component reruns | 4 family checks and 1 actual converted component/header check passed. |
| Final-source official complete-logit gate | All three official cases passed, including the 1-megapixel boundary: BF16 mismatch count 0, relative RMS error 0, cosine similarity 1, top-1 match true. |

The official logit cases used 83, 96 and 1,043 input tokens with a 248,320-entry
vocabulary. Their bit-exact result preserves the existing 2% acceptance rule.
The fine-tune was not added to the numerical/execution qualification catalog.
Prior hash-bound archives were retained unchanged.

## Private evidence identities

Raw artifacts remain under ignored `runtime/`; only aggregate results and
hashes are published here. This run persisted no prompt, generated response,
image bytes, token vectors or credentials in validation records.

| Evidence | SHA-256 |
| --- | --- |
| Remote conversion receipt | `72e6253ed6a42f683fa1bad64a4632191888576f71a0a46af63f11a9985d8654` |
| Offline replay receipt | `9dc71851bc8d848705937c01169de3d9de8171e21bd624c9495e7bc40629c8b4` |
| Developer worker family checks | `eed27b159bfa8819ea3bff02517066eee753aea2768cd4cf9d23c88feb867473` |
| Final formatted-source Python suite log | `ec166d9999f62c2dd85c61eed5ec2f97b0ca92ea9c9b4b35ecc0209f5f266e63` |
| Reviewed packaging log | `6d717b3629e1bb1cd298ed295552f6a64b249dc45f792c77c83070eae48821fb` |
| Final official normal-case log | `c6956990662ea74252dff4fbe165857f7210842c2905b0196499448056b9f000` |
| Final official 1-megapixel log | `0043429310cff94b2d6f9c272cf0d81451253d76caf6579a5d188a57799d6a73` |
| Final-source full engine log | `2a57b61a17b6cecceae0f13f1b79703a293a3811d9ddf004ff393911916374c1` |
| Worker unit log | `a64f1cbe0e03229a1bea24218d5e9d12f42dcb016747f059d41bef90bc3f4d49` |
| Final-source family rerun log | `8d5a00da62cd122d26120424dc47aaa9e08cfadac1d8b13053f516c555f85b6d` |
| Final-source real-component/header log | `ec1b00659d04dad5f99478928203e2b540f5b9511cfbb3348f802b93fc45393f` |
| Final signed candidate identity | `d786d00211568209d03736e80146ed720d58fa9ce933ae95af584a6ac2a5ac6b` |
| Final packaged converted-model checks | `1434bd1039435b7e9e32e69a397fd60cb6d1551a5a16e1185170d483199fdff2` |
| Final packaged raw-source installation and HTTP checks | `90af910f097f6847d07c5581e94f1557f7449cb057be77d51e116e644e9b76d3` |
| Raw-source installation/HTTP driver | `2461d1c34f17a51919127edd0d6ecfb3d3601a2e1daeef15874acee9be0f7fc9` |
| Final Swift controller qualification | `f05d073dc83ffb81fa7e4111dc5633c8cfdf93f50fcb3c95856fee7dfbfc77c3` |
| Exact Swift controller source | `0c47d6070dbbc8ae107dd279ba6178ab44ccc04cdfc22c9294f2a1c1399967db` |
| Swift qualification driver | `497993fbf0841729ff5a973688d04a7762d7afbaa95c0752c7b74ecb94d54240` |
| Final three-stage package runner log | `cd953246f4eb53422974ef8c8271849a4fc323bb774beeb41b0b2b6d4293005a` |

The final package records bind the same signed worker and final Swift executable
to the final inventory. The controller source hash matches the candidate's
source manifest. The three-stage runner completed with all stages passing.
