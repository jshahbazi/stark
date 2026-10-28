# Native Qwen3.5 preprocessing and vision tower

Date: 2026-10-26. Baseline: `a02ed0375d1e5bd7adb2e6e8554ee6f23f889715`, branch
`codex/qwen35-vision-artifact-contract`. Source/build identities below bind this
record to the accompanying implementation. Previous converter and fusion records
are preserved.

The private native pipeline executes PNG/JPEG decoding, preprocessing, the vision
tower, feature fusion, M-RoPE, and text generation with the exact official BF16
artifact. Component and regression checks pass. **Official-checkpoint numeric
qualification is not confirmed: the valid-image projector errors exceed the
unchanged 2% relative RMS gate.** Public workers and the Local API continue to
reject image execution; `images:false` and artifact qualification metadata are
unchanged. Functional execution is not a passing numeric qualification.

## Boundary and requirement evidence

The governing proposal is the 2026-10-26 first-image milestone, especially
“Native engine and artifact changes,” “Private IPC, decoding and cleanup,” and
the Stage 2 ownership/acceptance row. Current behavior is described in
[the implementation reference](../../MULTIMODAL_IMPLEMENTATION.md).

| Requirement | Implementation/evidence | Assessment |
| --- | --- | --- |
| Standalone native route, exact BF16 checkpoint | Private `Engine::load_vision_language_model`; five artifact byte identities checked on opened descriptors before tensor allocation and again before publishing the instance. All 153 vision tensors validated. | Conforms for this experimental tuple. |
| Shared bounded JPEG/PNG decode and preprocessing | Native Rust codecs; checked headers, dimensions, frame/scan counts, orientation, color/alpha policy, resize and patch layout. Independent decoded-pixel and complete patch replays. | Conforms for tested fixtures; hardware/resource qualification remains pending. |
| Patch projection, positions, twelve vision blocks and merger | Native CPU/Metal kernels, complete 12-layer synthetic model with 153 BF16 tensors, fifteen checkpoints per image, square/rectangular cases. | Component acceptance passes. |
| Fusion, positions, request ownership and lifecycle | Existing typed fusion path; exclusive model borrow spans the complete image stream; cheap ownership/sampling/context checks precede decoding. Context limits and fresh-state recovery exercised. | Conforms for tested boundaries and inspected native failure propagation. |
| Text/scoring behavior | Engine suites, worker checks and cross-build full-vocabulary digest/generation comparison. | Results below. |
| Source and validation provenance | Pinned reference revisions, artifact/source/build and harness hashes; raw validation data remains private. | Recorded separately from earlier runs. |
| Official checkpoint intermediate/projector numeric parity | Four official-weight cases against corrected-load pinned reference; unchanged numeric budgets. | **Not confirmed; gate fails.** |
| Transport, app UI and packaged qualification | Public loader, worker protocols and capabilities remain text-only. | Future stages; not implemented by this change. |

The source owners are `engine/vision/`, the shared language-model constructor and
stream/state preflight, and new vision-only CPU/Metal kernels. Existing text
normalization, attention and matrix-kernel bodies were not changed.

## Implementation checks and corrections

The tower preserves source Conv3d C/T/Y/X patch layout, merge-group row order,
bilinear learned-position interpolation with `align_corners=True`, axial spatial
RoPE, noncausal attention, twelve residual LayerNorm/MLP blocks and the patch
merger. The merger normalizes individual patches before their contiguous reshape.
Source BF16 weights remain unchanged.

Independent review and intermediate replays found and corrected:

- An extra BF16 rounding before LayerNorm bias. Vision normalization now keeps
  FP32 affine/bias precision until one final BF16 cast.
- Different attention precision boundaries in the inherited fused path. The
  vision path explicitly materializes BF16 scores and BF16 normalized
  probabilities to match the selected pinned eager reference.
- BF16 split-K partial matrix sums on Metal. Vision matrix operations now retain
  FP32 partial sums and use explicit final casts, with a separate FP32-sum plus
  BF16-bias cast for affine operations. The earlier hypothesis that the generic
  bias epilogue itself rounded prematurely was rejected by source/probe evidence.
- Requested image context capacity differing from the allocated state capacity,
  and a sampled lookahead escaping an exhausted image context. The wrapper now
  rejects inconsistent limits and missing output headroom before decoding; image
  iteration and cleanup cannot publish an output beyond capacity. Text streams
  without image position state retain their previous behavior.
- Compressed PNG profile/text metadata reaching codec initialization. iCCP, zTXt
  and compressed iTXt now fail header admission before decompression.

The original JPEG choice also missed the retained Pillow budget for subsampled
inputs. Production JPEG decoding was replaced with exact-locked pure-Rust
`jpeg-decoder 0.3.2`, default features disabled. Production `image 0.25.10` enables
PNG only. JPEG encoding on `image` remains a dev dependency for unit fixtures;
the normal generator dependency graph contains no `zune-jpeg` or `zune-core`.
Existing locked package versions and dependency edges were preserved.

## Reference construction and numeric gates

Reference code is pinned to Transformers
`2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397`. Private scripts execute extracted
reference classes/helpers with PyTorch 2.14.1; they are not product dependencies.
The selected attention implementation is explicit `eager`, rather than an
unspecified framework default. Pillow 12.0.0 decoder fixtures use an isolated
locked `uv` script environment.

The initial harness incorrectly applied a blanket `.to(bfloat16)` after model
construction, truncating fifteen of sixteen explicitly FP32 rotary frequencies.
The pinned loader instead constructs under its BF16 default-dtype context while
preserving explicit FP32 buffers. Corrected references reproduce that construction,
assign source BF16 parameters, and assert rotary buffers remain FP32. The old
harness, manifests, failed runs, SDPA comparisons and intermediate traces remain
private and are not relabeled as valid qualification evidence.

The numeric gates were not widened:

- Decoded PNG pixels and prepared BF16 patches: exact equality.
- JPEG pixels: maximum absolute error at most 3, RMS error at most 0.6.
- Tower patch projection: relative RMS at most 0.005.
- Later tower checkpoints: relative RMS at most 0.02; cosine similarity at least
  0.999. Shapes and finite values are mandatory at every checkpoint.

Five admitted preprocessing fixtures match completely, with one oversized result
rejected. Four true-downsample/mixed-resize uint8 fixtures are exact. Fourteen
PNG color/alpha/palette/orientation cases are exact; animated PNG is rejected.
Four baseline/progressive JPEG cases pass the original gate: 4:4:4 maximum error
2 and RMS 0.237234; 4:2:0 maximum error 3 and RMS 0.345577.

Both the two-layer and full twelve-layer small synthetic models pass every
checkpoint on CPU and Metal. The full model has 153 BF16 tensors and fifteen
reference stages per case. Its final merger relative RMS values are:

| Synthetic patch grid | CPU | Metal |
| --- | --- | --- |
| 4 × 4 | 0.00925343 | 0 |
| 4 × 6 | 0.01059913 | 0.01059913 |

The official-weight final merger results remain:

| Official-weight patch grid | CPU relative RMS | Metal relative RMS |
| --- | --- | --- |
| 2 × 2 component stress case | 0.26453912 | 0.03563311 |
| 2 × 4 component stress case | 0.01985681 | 0.01845531 |
| 16 × 16 canonical RGB image | **0.04872362** | **0.05060871** |
| 22 × 14 canonical processed RGB image | **0.04055180** | **0.04028932** |

The first two grids are below the canonical processor minimum and are component
stress cases. The latter two are real uint8 RGB preprocessing paths; the replay
checks production BF16 patches exactly before encoding. The combined official
replay fails on both backends. The remaining error is not fully explained by the
completed fixes. No output-quality, OCR, supported-memory or latency claim follows
from these results. Resolving the numeric gate is required before qualifying the
official image tuple or enabling public image serving.

## Native request and regression results

An ignored model-bound Metal test loads the full official artifact, decodes a
generated 256 × 256 PNG, produces 64 × 1024 finite projected features and executes
two greedy language outputs. Physical token/cache length is 70 and M-RoPE delta
is -56. A fresh request repeats the same sequence privately. Malformed image/span,
foreign/nonfresh state, vocabulary, inconsistent-context and output-headroom
cases fail at their intended early boundaries. Request content and outputs are
not logged. Native backend fault injection and cancellation during tower work
remain part of later worker/lifecycle qualification.

Physical Apple M5 Max, 128 GiB, macOS 26.7.1. Engine/formatting toolchain:
`nightly-2026-10-01`; release workers: Rust 1.98.1. All dependency commands use the
checked-in locks. Private artifacts live under owner-only
`runtime/vision-tower-validation-2026-10-26/`.

| Final check | Result |
| --- | --- |
| Engine CPU/grammar | 367 library tests passed, 6 ignored; 7 integration tests and 3 doctests passed. |
| Engine CPU/grammar/RoPE cache | 369 library tests passed, 6 ignored; 7 integration tests and 3 doctests passed. |
| Physical CPU/Metal suite | 546 library tests passed, 12 ignored; 7 integration tests and 3 doctests passed. |
| Converter regression suite | 4,181 passed, 88 skipped. |
| Private preprocessing/decoder and synthetic tower replays | Passed as described above. |
| Official image request replay | Passed on the final source, including two repeated fresh requests and early boundary rejection. |
| Official tower numeric replay | **Failed on CPU and Metal; retained as an unmet gate.** |
| Worker release builds, unit/process checks | Both builds passed; generator 26 tests, scorer 39 tests, generation process 6 cases and scoring lifecycle 8 checks passed. |
| Cross-build native text comparison | Two generation sequences and four complete-vocabulary digests identical to the prior build; short and 1,025-token inputs, one cache hit and three native score calls. |
| Public full-vision worker startup | Still rejected: exit 1, `ready:false`, `model_load_failed`. |
| Repository policy, pinned formatting, patch whitespace | Passed. |

Engine commands are `cargo +nightly-2026-10-01 test --locked -p stark-engine`
with the listed feature sets and `-- --test-threads=1`. Private replay filters in
`engine/vision/reference_tests.rs` use `--lib FILTER -- --include-ignored
--test-threads=1`, `STARK_VISION_REFERENCE_PATH` and `STARK_TEST_MODEL_PATH`.
Reference roots are `eager_fp32_buffers`, `synthetic_component` and
`synthetic_12layer`; model paths and all vectors remain private. Worker packages
are built and tested separately with Rust 1.98.1, using `--features optimized`
only for the scorer. Converter checks use the locked Python 3.13.13 environment
with `HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false`.

## Source and artifact identities

The existing converter `native_build_provenance` helper hashes the native source
manifest, including new kernels/tests. Documentation is outside that manifest.

| Identity | SHA-256 |
| --- | --- |
| Native source manifest | `7b91157a1891c1d24d534265a0252404709ae03e2b1398b4d1181f02504a819c` |
| Cargo.lock | `2254da0dd179db6b926492f2bec5f63b756f6508132b34f5ac772ec9d877aaa2` |
| rust-toolchain.toml | `72da9f9e503464ba7393560f6f51a69fb06be72633d52ae7725fa0ba59dfda22` |
| Generator executable | `1b05e9b16a39e65a5730a6889f74955feaeda2f044d8df93f9bcf23f68c20a52` |
| Optimized scorer executable | `94537b99cc15580e49a8ff4c8525683cb8a24044cbd9fa359a8becb417000838` |
| Optimized digest executable | `f595cb093504afb23a5e94d9451c9cdd47ee3c910bd5736c3ec649143eeef860` |
| Official converted configuration | `26583eb8a5d2a86aac80d9205eceda6133f0e63ac9ff7085364f8be65fd6139b` |
| Official converted weights | `7c5960e7225b3b36844fa397a85c5e20d281d732e382475062f552c0737c79d4` |
| Corrected official reference manifest, including rectangle | `e5dceac11aa5fd42bea86ca50b4aa1a9d48d8bfd99f08cd43c5fe950225330ae` |
| Two-layer synthetic reference manifest | `397c00f2aeeccdff099fc55f85a42bb5e56d152505ee881d48727ad9b79ce5da` |
| Twelve-layer synthetic reference manifest | `199c7da6af7c8a8bb04d52e91900aab1b1296b2af061547078a11d36136daefb` |
| Twelve-layer synthetic weights | `4b23c448cbdc555ad8b9e1b03435aab13aa81f6f2ec1e25c2b8ab0386c486994` |
| Decoder reference manifest | `f93d67955b369d832e3b1341fd36a0778c691e15b414379c95df80460a81c054` |

The previous text build baseline is commit `a02ed037`, native manifest
`fec9cdbe3f7f4db609651666e8b8fa52ea3175cc4b05fc92b8e261152548befa`.
Its generator executable SHA-256 is
`a141d5435e11e29f4492fdf3acf91db4238da673670cfce13f2df042ef1d0045`;
its digest executable SHA-256 is
`dbaa9bd4b4561b9f94d7ae75c248238224e07dd4dea0499340d4bf533b6b15d8`.
The cross-build replay used the cached BF16 `Qwen/Qwen3-0.6B` text artifact at
`c1899de289a04d12100db370d81485cdf75e47ca`, whose identities were recorded in the
[preceding fusion run](2026-10-26-qwen35-native-fusion-mrope.md).

## Primary references and harness provenance

- [Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397/src/transformers/models/qwen3_5/modeling_qwen3_5.py), copyright 2025 The Qwen Team and The HuggingFace Inc. team, Apache-2.0; source SHA-256 `af05d0acfe79fff8aea1c75f93516a6a72bc80f6f901329a73e66a3866fe1e7c`.
- [Vision interpolation/position helpers](https://github.com/huggingface/transformers/blob/2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397/src/transformers/vision_utils.py), copyright 2026 The HuggingFace Inc. team, Apache-2.0; source SHA-256 `bcecd5a92b3266b9926272a549d2b1a0f1fe7646c698c0fa19bd96f976085356`.
- [Pinned image processor](https://github.com/huggingface/transformers/blob/2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py); source SHA-256 `ebdb484798a6aefebc5dbe5431d3e4af8f460404d95d4cb67165fdff624ed333`. Native preprocessing independently implements its resize and patch equations.
- [Pinned loader dtype handling](https://github.com/huggingface/transformers/blob/2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397/src/transformers/modeling_utils.py), source SHA-256 `251aaae7733affbb7cdb5aa05a84157c46cac02518d868898cd6d6282cc14571`.
- Resize coefficient arithmetic follows [PyTorch CPU antialias interpolation](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cpu/UpSampleKernel.cpp), version `v2.11.0`, under the [PyTorch BSD-style license](https://github.com/pytorch/pytorch/blob/v2.11.0/LICENSE). The independently implemented native resize is checked against PyTorch 2.14.1 execution.
- Reference resize dispatch was checked in [Torchvision geometry](https://github.com/pytorch/vision/blob/a034671605659e24ff1de1142079bf318ac3b123/torchvision/transforms/v2/functional/_geometry.py), revision `a034671605659e24ff1de1142079bf318ac3b123`, [BSD 3-Clause](https://github.com/pytorch/vision/blob/a034671605659e24ff1de1142079bf318ac3b123/LICENSE), copyright 2016 Soumith Chintala; source SHA-256 `619db678a3318cff253774fc0d13006f2ed263971bcd8ebc8e753fcbdbb8b31d`.
- Native codecs retain their crate licenses: `image 0.25.10` and `jpeg-decoder 0.3.2`, MIT OR Apache-2.0. No reference runtime/model implementation is imported into the product. Existing [Uzu/Lalamo provenance](../../../provenance/UPSTREAM.md) remains unchanged.

Private harness code identities: corrected-load generator
`7ff41e75a8a812e120c9dbe49242501063dac1f60b67781f55d2b148dae23d42`;
twelve-layer generator
`4a10a159abdd5d0dee608ecc4a745a301f5b7c8b6afcc1424fc34c220eb6b29f`;
decoder generator
`32af09e7acf1516b1184484a3922e8e07fbfe9cffb2b69b26e82bd4eaa6e46b1`,
its lock `fb4524b8cbb8a82d80b15d93f709df8e8adf177bb3107f1bb093ffa2e4a3c437`;
first-block generator
`4beef0cf727556d5d60bb0659f08f1202add4edd3b8535506253d99205846376`.

Only aggregate results and source/build identities are published. No image bytes,
request text, responses, token/feature vectors, probability outputs or raw native
validation data are committed.
