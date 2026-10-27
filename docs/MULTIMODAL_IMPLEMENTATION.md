# Native image support implementation

The first target is one static image with the official BF16 `Qwen/Qwen3.5-0.8B`
checkpoint at `2fc06364715b967f1860aea9cf38778875588b17`. The implementation
follows the native Rust/Metal tower and decoder route. The converter has an
experimental full-vision artifact mode. The engine has BF16 projected-feature
fusion, Qwen3.5 M-RoPE, native image decoding, preprocessing, and vision-tower
execution. A guarded public facade admits one exact BF16 artifact. The normal
generation worker, terminal client, and Local API implement one-image requests.
The developer worker, CLI, and Local API have exact execution qualifications for
the recorded BF16 artifact and worker builds on Apple M5 Max with macOS 26.7.1. Other
tuples continue to report `images:false`. The official-weight tower passes the unchanged
2% numeric gate: all fifteen checkpoints across four fixtures are bit-exact on
CPU and Metal. The [parity validation record](archive/runs/2026-10-26-qwen35-vision-parity.md)
records this result and the remaining qualification boundaries. The earlier
[native vision record](archive/runs/2026-10-26-qwen35-native-vision-preprocessing.md)
retains the source-bound failing result.
The [image execution qualification](archive/runs/2026-10-26-qwen35-image-execution.md)
records bit-exact complete decoder logits across three fixtures, including the
1-megapixel boundary, and release-worker visual and lifecycle checks.
The [activation replay](archive/runs/2026-10-26-qwen35-image-activation.md) verifies
fresh conversion, ordinary catalog-gated API/CLI use, and final regressions.
The [Swift and local signed-package replay](archive/runs/2026-10-27-qwen35-swift-packaged-app.md)
records production-controller checks, delivered-GUI image execution, recovery,
cleanup, source/notice audits, and the remaining public-release boundary.

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
  --require-native-support \
  --write-source-manifest runtime/converter-sources/qwen35-08b-vision.json \
  --output models/qwen35-08b-vision
```

This command downloads model weights. A completed conversion establishes artifact
validation only; it does not execute or qualify the model. Replay requires the
same `--artifact-mode vision-language` argument. `--require-native-support`
requires matching named-build evidence for the exact output. A record identifies
a worker build and runtime; serving checks the executable actually launched,
device class, and OS. After building the recorded worker, use the artifact with:

```sh
./scripts/stark chat --model-path models/qwen35-08b-vision --image /absolute/path/to/image.png
```

The Local API accepts ordered user content containing `image_url` with an inline
PNG/JPEG data URL. The Swift app accepts the same encoded formats through private
file references and the common native pipeline. Its Developer ID signed worker
has a [separate execution record](archive/runs/2026-10-27-qwen35-signed-worker.md).

## Versioned artifact contract

The root model tag is `Qwen35VisionLanguageModelConfig`. Ordinary text model
resolution rejects it. The generation worker selects `VisionLanguageModel` for
this explicit type. Its loader verifies the exact configuration, weights, tokenizer, processor, and
template byte hashes before tensor allocation, keeps their opened descriptors,
and rechecks them before publishing the loaded instance. The original artifact
tag is retained. This loader does not establish execution qualification.

The root and receipt embed `artifact_contract` with:

| Field | Meaning |
| --- | --- |
| `type`, `schema_version` | `Qwen35VisionLanguageArtifact`, version 2 |
| `input_modalities`, `dtype` | Artifact represents text and image inputs in BF16 |
| `source` | Exact checkpoint revision, metadata hashes, source payload/index sizes and hashes |
| `special_token_ids` | Validated vision wrappers, image placeholder, video placeholder |
| `vision_config`, `processor_config` | Exact source architecture/projector and processor metadata |
| `position_contract` | M-RoPE sections and interleaving; attention configs use `Qwen35MRoPEConfig` |
| `tensor_layout` | `hf-qwen35-vision-v1`; visual tensors keep their source shape and axis order |
| `transport` | `mediaref-v1`; bounded control messages reference privately staged bytes |
| `runtime_supported`, `qualification` | `true`, `unverified`; implementation and qualification are separate |

The receipt's `execution: unverified` describes conversion itself. Exact output
matches report `native_loading: qualified_tuple` and include the named-build
record; unmatched outputs report `supported_layout_unverified_tuple`. Conversion
does not start a model. Artifact modalities describe stored weights and
metadata; they cannot enable service capabilities. The normalised tokenizer and
complete model configuration remain bound by the receipt's output hashes.
Earlier experimental version-1 artifacts require reconversion. The version-2
configuration binds decoder BF16 operation boundaries as well as the image
transport; the exact serialized model weights remain unchanged.

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

`engine::vision::VisionLanguageModel` provides a guarded native path from compressed
image bytes to projected features and the existing multimodal stream. The public
API accepts encoded image bytes and returns an opaque owned preparation. Clients
cannot construct projected embeddings. Preparation decodes and preprocesses once;
generation consumes the object and validates its model identity and prompt span.
The stream's exclusive
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
quality. Public image capability requires a qualified complete tuple.

## Decoder reference arithmetic

The admitted vision artifact enables a private reference numerical profile.
Ordinary and Q/K RMSNorm use FP32 normalization and affine arithmetic before
their BF16 output boundary. Reference-enabled DeltaNet retains FP32 recurrent
state and preserves the source's BF16 convolution, sigmoid, core-output, and
gated-normalization boundaries. Attention preserves BF16 rotary products,
scores, probabilities, and output gating through an eager path. Other artifacts
retain the existing execution profile. End-to-end qualification compares final
decoder logits against the fully pinned reference, including its child model
classes; matching only the vision tower is insufficient.

Decoder BF16 linears and tied readout use the source's 32-stream FP32
accumulation followed by one BF16 cast. SiLU retains its source exponential,
division, and BF16 product boundaries. DeltaNet prefill uses 64-token chunks,
FP32 scratch, and source matrix/triangular-solve sequencing. Its CPU reference
accumulates cumulative decay in FP64; Metal emulates those IEEE additions with
integer arithmetic and rounds each prefix to FP32. This profile is selected
after exact artifact admission and keeps the Legacy profile for text artifacts.

## Worker and service transport

The host creates an empty owner-private staging directory before starting the
worker. Image JSON contains a random object ID, byte length, and MIME type. The
worker opens objects relative to a pinned directory descriptor, validates a
private regular single-link file, reads bounded bytes, and unlinks the claimed
name while retaining its open descriptor through completion. A prepared handle
is single-use and model-scoped. Invalid requests, cancellation, unrelated
operations, EOF, and failures release prepared state. Native cleanup failures
fence the worker before any success or cancellation result becomes visible.

Normal host shutdown removes the directory. Worker teardown removes only opaque
private staged files from the directory it verified empty at startup. Killing
both processes before claim can leave a staged object behind; no shared temporary
directory scan or persistent content cache is introduced.

The terminal client accepts `--image PATH`. The Local API accepts one inline
PNG/JPEG data URL in ordered user content, with 4 MiB decoded compressed bytes,
a 6 MiB request body, and a shared 24 MiB pending-body budget. Text requests keep
their 1 MiB body limit. These byte budgets do not claim a strict process RSS cap.
Public image capability requires an execution record matching all five admitted
artifact files, the launched worker executable, Metal device class, and macOS
version. The executable hash is checked before launch and at readiness. Image
transactions share one deadline across pipe writes, prepare, generate, streaming
reads, and discard. See [generation](GENERATION.md)
and [Local API](LOCAL_API.md) for protocol and service details.

## Swift image attachments

Chat mode enables **Attach Image** only after the loaded worker passes the exact
execution catalog and negotiates `private-file-v1` with the desktop bridge. The
composer accepts a file, a file drop, or a PNG/JPEG paste representation, previews
it, and supports removal before sending. TIFF-only clipboard representations
are unsupported. Original encoded bytes remain unchanged; ImageIO creates only
the UI thumbnail, and the native Rust decoder performs inference preprocessing.

One image is retained in memory for the conversation and resupplied on every
turn. The first image turn introduces one placeholder; follow-ups reuse it.
**New Chat**, advanced-option changes, Stop, mode changes, process failure, and
quit clear the image. Recoverable request errors restore the draft attachment.
Image loading has cancellation and guards against late drop/paste callbacks
overwriting a newer selection. Sending shows preparation/generation progress;
Cancel during generation stops the model, matching the app's existing Stop action.

Swift starts Python with an initially empty owner-only staging directory. Each
command references one opaque 32-hex identifier, MIME type, and byte length;
neither paths nor image bytes enter JSON-lines. The bridge pins the directory,
checks owner-only single-link regular files through descriptor-relative,
nonblocking, no-follow opens, verifies identity and size before/after reading,
and unlinks the claimed pathname before native decoding. The unchanged 1 MiB
command limit and 4 MiB encoded-image limit apply. Cleanup only touches private
opaque files in that pinned directory. Hard-killing both processes before claim
can leave a staging file; there is no shared-temporary-directory scan.

The model installer offers **Enable image input when converting** for the pinned
BF16 vision conversion. Existing converted folders are detected automatically.
The default conversion behavior remains text. Connection JSON and the connection
view report the running worker's qualified image capability.

## Remaining dependencies and acceptance

1. **Coverage:** additional models, dtypes, worker builds, hardware classes, or
   OS versions require their applicable implementation and exact execution
   evidence. The current native loader admits one pinned BF16 artifact.
2. **Distribution:** each delivered worker signature is a different executable
   identity. Preserve a qualified nested worker when resealing an application;
   changing it requires separate execution evidence. Public distribution also
   requires notarization and Gatekeeper acceptance for the final app.

The historical [converter review](archive/runs/2026-10-26-qwen35-vision-artifact-review.md)
binds real conversion and regression checks to their source and artifact hashes.
It verifies all 153 visual tensor payloads, normalized special IDs, and six
M-RoPE layer descriptions. The [initial converter record](archive/runs/2026-10-26-qwen35-vision-artifact-contract.md)
preserves its synthetic and header-only evidence. The current execution record
separately covers decoder numerics, output quality, memory observations,
cancellation, cleanup, and the supported runtime. Model weights and native
image outputs remain ignored.

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
