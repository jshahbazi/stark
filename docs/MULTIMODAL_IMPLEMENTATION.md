# Native image support implementation

The first target is one static image with the official BF16 `Qwen/Qwen3.5-0.8B`
checkpoint at `2fc06364715b967f1860aea9cf38778875588b17`. The implementation
follows the native Rust/Metal tower and decoder route. The converter has an
experimental full-vision artifact mode. The engine now has a private BF16
projected-feature fusion and Qwen3.5 M-RoPE core, plus experimental native image
decoding, preprocessing, and vision-tower execution. The private loader admits
one exact BF16 artifact. Worker transport and app attachments remain
unimplemented. Exact-tuple product qualification is pending, and the Local API
continues to report `images:false`. The official-weight tower passes the unchanged
2% numeric gate: all fifteen checkpoints across four fixtures are bit-exact on
CPU and Metal. The [parity validation record](archive/runs/2026-10-26-qwen35-vision-parity.md)
records this result and the remaining qualification boundaries. The earlier
[native vision record](archive/runs/2026-10-26-qwen35-native-vision-preprocessing.md)
retains the source-bound failing result.

## Implemented conversion boundary

`stark-convert --artifact-mode vision-language` is opt-in and accepts only this
checkpoint and BF16 output. The default `text` mode retains the existing text
conversion layout, scalar RoPE, source manifests, and native qualification
matching. Its receipt explicitly records text input and `text-generation-v1` for
language models. Existing text-only Qwen3.5 artifacts cannot gain vision support
without reconversion.

The vision mode:

- Pins the model configuration, tokenizer, tokenizer configuration, and processor
  and template files by revision and SHA256. It also checks the exact source
  safetensors payload and index identities, including for a local source folder.
- Validates the image/video/wrapper special-token IDs against the tokenizer and
  its added-token metadata.
- Preserves every required visual tensor under `vision.*`, in the source layout
  and BF16 dtype. Required shapes cover patch projection, learned positions,
  all twelve attention/MLP blocks, and the patch merger. Missing, extra,
  malformed, or unsupported visual tensors fail conversion. Declared auxiliary
  MTP weights remain omitted, as in the text adapter.
- Copies the exact `preprocessor_config.json` and `chat_template.jinja` into the
  output and embeds their recipe and identities in the model configuration.
  The pinned inline and standalone templates must agree.
- Uses a private, atomic conversion directory and rechecks every source before
  publication. Source manifests use schema 3 and bind `artifact_mode`; text and
  vision manifests cannot be substituted for each other.

The pinned 0.8B revision has no `generation_config.json`. Vision artifacts derive
stop IDs from its text configuration and retain unset sampling overrides. They
do not inherit the existing text catalog's 27B generation metadata.

An experimental conversion can be prepared with:

```sh
uv run --project tools/converter --locked stark-convert Qwen/Qwen3.5-0.8B \
  --artifact-mode vision-language \
  --write-source-manifest runtime/converter-sources/qwen35-08b-vision.json \
  --output models/qwen35-08b-vision
```

This command downloads model weights. A completed conversion establishes artifact
validation only; it does not execute or qualify the model. Replay requires the
same `--artifact-mode vision-language` argument. `--require-native-support`
rejects this mode before source resolution.

## Versioned artifact contract

The root model tag is `Qwen35VisionLanguageModelConfig`. Public model resolution
and the text worker loader reject it. The crate-private experimental vision
loader verifies the exact configuration, weights, tokenizer, processor, and
template byte hashes before tensor allocation, keeps their opened descriptors,
and rechecks them before publishing the loaded instance. The original artifact
tag is retained. This loader does not establish worker capability or converter
qualification.

The root and receipt embed `artifact_contract` with:

| Field | Meaning |
| --- | --- |
| `type`, `schema_version` | `Qwen35VisionLanguageArtifact`, version 1 |
| `input_modalities`, `dtype` | Artifact represents text and image inputs in BF16 |
| `source` | Exact checkpoint revision, metadata hashes, source payload/index sizes and hashes |
| `special_token_ids` | Validated vision wrappers, image placeholder, video placeholder |
| `vision_config`, `processor_config` | Exact source architecture/projector and processor metadata |
| `position_contract` | M-RoPE sections and interleaving; attention configs use `Qwen35MRoPEConfig` |
| `tensor_layout` | `hf-qwen35-vision-v1`; visual tensors keep their source shape and axis order |
| `transport` | `unsupported`; no worker image input protocol is implemented |
| `runtime_supported`, `qualification` | `false`, `unverified` |

The receipt reports `native_loading: runtime_blocked`, `execution: unverified`,
and no native qualification. Artifact modalities describe stored weights and
metadata; they cannot enable service capabilities. The normalised tokenizer and
complete model configuration remain bound by the receipt's output hashes.

## Native fusion and positions

The native fusion and position core accepts one already projected image through
crate-private, owned input types. It binds the features to the loaded model's
identity, BF16 dtype, model width, processed grid, and exact contiguous placeholder
span. Host validation checks finite features, vocabulary and reserved tokens,
wrapper placement, context capacity, shape, and checked offset arithmetic before
device allocation. Public clients cannot submit embedding tensors.

The decoder looks up text embeddings and copies the projected rows into precisely
the image span using command-buffer-owned storage. Generated-token lookup keeps
the existing path. `Qwen35MRoPEConfig` implements the pinned interleaved
temporal/height/width frequency mapping, including a rotary prefix smaller than
the attention head. Explicit per-token positions survive chunked prefill; a
signed continuation delta lives in the request state. Physical causal and KV
indices continue to count tokens independently of spatial positions. Spatial
tables are computed per submission and are not retained in the scalar RoPE cache.

Image prefill requires a fresh state and disables speculation. Tensor capture is
suppressed for the image request and its continuation. Failed image execution
fences the state; pending submissions drain before their owned storage is
released. A fresh state is required after a failure. Scoring remains text-only.

Synthetic native fixtures check feature replacement, M-RoPE coefficients, partial
head rotation, whole/chunked prefill, a span crossing the 1,024-token boundary, and
decode continuation. The [native core validation record](archive/runs/2026-10-26-qwen35-native-fusion-mrope.md)
records the checks, tolerances, and source/build identity. These checks do not
establish preprocessing or projector numerics. Those have separate reference
replays for the native preprocessing and tower work below.

## Native preprocessing and vision tower

`engine::vision` provides a private native path from compressed image bytes to
projected features and the existing multimodal stream. The stream's exclusive
model borrow covers preprocessing, tower execution, language prefill, and decode.
State/model ownership, sampling, context agreement, and output headroom are
checked before image decoding. Context exhaustion cannot return the decoder's
lookahead token. Recoverable image validation errors leave native state untouched.
A tower failure fences the loaded instance before language-state mutation;
decoder execution failures fence the loaded instance and request state.

The shared decoder accepts one static 8-bit PNG or baseline/progressive JPEG.
PNG decoding uses exact-locked `image 0.25.10`; JPEG uses exact-locked
`jpeg-decoder 0.3.2` with default features disabled. Both are native Rust, with no
Python or system codec dependency. JPEG support on `image` is enabled only for
test-fixture encoding. Header validation precedes codec construction and checks
dimensions, frame/scan structure, arithmetic, and encoded length. Animated,
multiple, 16-bit, CMYK, and unsupported JPEG encodings fail. Compressed PNG
profile/text metadata is rejected before decompression.

EXIF orientation is applied. Grayscale expands to RGB and transparency composites
over white; ICC color conversion is not performed. Smart-resize dimensions follow
the pinned processor, with ties-to-even rounding. Antialiased bicubic uint8
resizing, fused rescale/normalization, static-frame temporal repetition, and
merge-group patch ordering feed owned BF16 patch rows to the tower. The official
processor's 65,536–16,777,216 pixel recipe remains unchanged. Internal admission
rejects a recipe result beyond 1,048,576 pixels or 4,096 encoder patches instead
of changing that resize result. Other internal bounds are 20 MiB compressed,
32 million source pixels, dimension 16,384, aspect ratio 200, and 64 JPEG scans.
These are implementation admission guards; they are not measured hardware or
public service capabilities, and do not claim a strict global allocator ceiling.

The tower consumes all 153 preserved `vision.*` tensors. It implements the Conv3d
patch projection as an equivalent flattened matrix operation, bilinear learned
position interpolation with `align_corners=True`, axial spatial RoPE, twelve
noncausal attention/LayerNorm/MLP blocks, and the pre-shuffle-normalized patch
merger/projector. Vision LayerNorm follows the pinned BF16 reference's FP32
Welford reduction and affine sequence before one final BF16 cast. Matrix
operations preserve its FP32 accumulation order: thirty-two streams for dense
projections and QK, four for patch projection and probability/value products.
Biases enter before the final BF16 cast. Vision-only rotary, softmax, and GELU
arithmetic follows the pinned SLEEF and PyTorch ARM implementation, including
explicit contraction and reciprocal boundaries. Metal GELU preserves tiny BF16
half-value rounding with integer arithmetic where FP32 flush-to-zero would lose
the result. The explicit eager attention path preserves BF16 score and
probability boundaries. Existing text kernels retain their behavior.

Independent private replays cover decoded pixels, complete patch tensors, and
all fifteen tower checkpoints against pinned reference code and official BF16
weights. Numeric limits and results belong to the dated validation record;
functional image execution alone does not establish numeric parity or output
quality. Public serving remains disabled until the complete tuple is qualified.

## Remaining dependencies and acceptance

1. **Transport and lifecycle:** support inline image data URLs through bounded
   HTTP bodies and private staging references. Preserve the 1 MiB worker frame
   limit, authentication, serialization, epoch fencing, cancellation, and cleanup.
   Choose resource budgets using measurements before advertising them.
2. **Exact-tuple qualification:** extend the passing tower fixture gate to
   end-to-end decoder logits and visual task quality. Measure peak memory and encode/prefill
   latency, test failure recovery, and publish aggregate evidence only.
3. **Swift and distribution:** add attachment input and progress through the
   common media pipeline, then qualify a packaged installation. Enable image
   capabilities only for the intersection of the qualified artifact, worker,
   and service policy.

Synthetic converter tests and header inspection do not establish preprocessing,
native inference, numeric parity, output quality, or supported hardware budgets.
The official-weight end-to-end conversion and offline manifest replay now pass.
All 153 emitted vision tensor payloads match their source shapes, dtypes, and
byte hashes. The normalised special-token IDs and six M-RoPE layer descriptions
also pass inspection. Experimental native tower execution is recorded separately
from these earlier converter-only results; it does not qualify public image
inference.
No model weights or native image outputs are checked into the repository.
The [follow-up review record](archive/runs/2026-10-26-qwen35-vision-artifact-review.md)
binds the real conversion and regression checks to the converter and artifact
hashes. The [earlier converter record](archive/runs/2026-10-26-qwen35-vision-artifact-contract.md)
preserves the initial synthetic and header-only evidence.

## Primary references and provenance

The source metadata comes from the [exact checkpoint revision](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17).
Source payload identity comes from the [immutable tree's LFS metadata](https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17?recursive=true&expand=false).
The tensor name/shape inventory follows the public architecture declarations in
[Transformers Qwen3.5](https://github.com/huggingface/transformers/blob/2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397/src/transformers/models/qwen3_5/modeling_qwen3_5.py),
revision `2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397`, copyright 2025 The Qwen Team
and The HuggingFace Inc. team, Apache-2.0. The converter implements an independent
shape inventory and direct tensor copies. The Rust fusion/position core
independently implements the reference's `get_rope_index` and text rotary
`recomposition_frequencies` equations. The explicit feature/position/delta boundary
also follows the design inspected in
[MLX-VLM Qwen3.5](https://github.com/Blaizzy/mlx-vlm/blob/e9af376626b04ba0cdad23b1476fd7a4a583d721/mlx_vlm/models/qwen3_5/qwen3_5.py),
revision `e9af376626b04ba0cdad23b1476fd7a4a583d721`, copyright 2025 Prince Canuma,
[MIT license](https://github.com/Blaizzy/mlx-vlm/blob/e9af376626b04ba0cdad23b1476fd7a4a583d721/LICENSE).
No Transformers or MLX runtime dependency or model code is imported. Existing Uzu
and Lalamo provenance remains in
[upstream provenance](../provenance/UPSTREAM.md) and the
[converter guide](../tools/converter/README.md#provenance-and-policy).
Native resize coefficient arithmetic also follows the independently implemented
[PyTorch CPU antialias equations](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cpu/UpSampleKernel.cpp),
version `v2.11.0`, under its
[BSD-style license](https://github.com/pytorch/pytorch/blob/v2.11.0/LICENSE).
The dated native vision record pins the additional processor, vision-helper and
Torchvision reference revisions, licenses, and validation harness identities.
