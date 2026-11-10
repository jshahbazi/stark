# Gemma 3 image artifacts

Stark implements the original dense BF16 Gemma 3 multimodal wrapper with its
SigLIP vision tower. The image sizes are 4B, 12B and 27B. Gemma 3 1B and 270M
are text models and do not acquire image support through this adapter.

| Size | Decoder width | Decoder layers | Vision tower | Execution evidence |
| --- | ---: | ---: | --- | --- |
| 4B | 2560 | 34 | SigLIP, 27 layers, width 1152 | Exact BF16 artifact passed native parity and worker/API replay; see the [run record](archive/runs/2026-11-07-gemma3-images.md) |
| 12B | 3840 | 48 | Same tower | Structural admission; full execution has not been qualified |
| 27B | 5376 | 62 | Same tower | Structural admission; full execution has not been qualified |

Conversion, structural native admission, numerical parity and a successful
host image check are separate results. An artifact starts with execution
`unverified`; a family entry or a converted checkpoint does not establish
answer quality or performance.

## Conversion

The original Google 4B instruction checkpoint used during implementation is
`google/gemma-3-4b-it` at revision
`093f9f388b31de276ce2de164bdc2081324b9767`. Run from the repository root:

```sh
uv run --project tools/converter --locked stark-convert google/gemma-3-4b-it \
  --revision 093f9f388b31de276ce2de164bdc2081324b9767 \
  --artifact-mode vision-language --architecture gemma3 \
  --output runtime/gemma3-4b-bf16 \
  --write-source-manifest runtime/gemma3-4b-source.json
```

The source must contain complete merged BF16 Safetensors weights, the original
multimodal configuration, tokenizer files and `preprocessor_config.json`.
Each consumed source file, including the optional generation configuration,
is bound to the source manifest. Offline replay preserves the selected image
processor and the original source signatures. Packed weights and changed
processor operations require a separate implementation profile.

The artifact type is `Gemma3VisionLanguageModelConfig`, with schema 5 and family
`gemma3-siglip-bf16-v1`. Its four component hashes bind weights, tokenizer,
normalized image processor and the exact reviewed chat template. The template
comes from the original tokenizer configuration; a standalone source template
must match it when present. The reviewed IT template shared by the three sizes
has SHA-256 `7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4`.

## Image execution

The processor admits one bounded upright 8-bit JPEG or PNG. The shared decoder
limits compressed input to 20 MiB, each dimension to 16384 and decoded pixels
to 32 million. RGB conversion discards alpha. Antialiased bilinear uint8 resize
uses the pinned CPU Torchvision processor's adaptive signed-int16 coefficient
precision and produces 896 × 896 pixels, normalized by mean and standard
deviation 0.5. It is covered by full patch hashes for nonuniform upsampling,
downsampling and each unchanged-size axis.
Pan-and-scan is disabled in this profile.

The native Conv2d patch order is channel, row, column inside each 14 × 14 patch.
The 4096 learned-position patches pass through 27 SigLIP blocks, final layer
normalization, spatial 4 × 4 average pooling, Gemma RMS normalization and a
learned projector. The result contains 256 private image embeddings.

The reviewed source template renders a single `<start_of_image>` marker.
Stark reproduces the processor's two newlines before BOI and after EOI, expands
256 `<image_soft_token>` slots and retokenizes with the admitted tokenizer.
Before expansion it verifies that decoding and re-encoding preserve the raw
token sequence. Surplus image markers and non-round-trippable raw token arrays
are rejected. Token counts depend on neighboring text; the fixed image-first
local-check prompt expands by 258 tokens in the original tokenizer.

The decoder uses ordinary scalar RoPE. Global layers permit causal attention
or attention within the same image; local layers apply the sliding-window
lower bound to that composition. The admitted 256-token image block fits
entirely inside the 1024-token local window, preserving complete same-image
visibility. Text remains causal. The scheduler keeps the
complete image block in one prefill chunk, including at a 1024-token boundary.
Each image request uses fresh decoder state and clears retained generation
prefix state. Image patches and projected embeddings are request-owned and
are not retained as reusable prefix evidence. Backend and numerical failures
after native work fence the loaded instance, requiring worker replacement.
Controlled cancellation between completed tower blocks releases private scratch
and leaves immutable model weights available for a fresh request.

Text-only generation through this image artifact can retain and restore a
completed decoder prefix snapshot. The snapshot includes each sliding KV
owner's ring payload and cursor state; shared-KV consumers refer to the owner's
restored state. Image requests still clear text prefixes and never publish
image-dependent decoder state. Artifact-bound qualification remains separate
from implementation support; see the dated image/text prefix records.

## Validation

`scripts/gemma3_image_reference.py` extracts private original-source canonical
CPU fixtures using unchanged Transformers 5.15.1 and Torch 2.14.1. It records
source-code hashes, input patches, projected features, full-vocabulary logits
and an eight-step greedy trajectory. The opt-in native replay test is
`gemma3_image_full_reference_metal`, using `STARK_GEMMA3_FIXTURE_PATH` and
`STARK_TEST_MODEL_PATH`. Fixture data stays in ignored private directories;
only aggregate metrics and hashes may be published.

When the preserved reference directory contains completed red/blue trajectories
and all three feature files, `--resume-fixture` with `--source-manifest` resumes
the missing pattern trajectory into a new output directory. It hashes the
preserved files, verifies every selected source payload and uses hardlinks for
completed evidence. A forward hook verifies that the unchanged full pattern
forward reproduces the preserved projector features exactly. Existing evidence
is never overwritten; the final manifest binds the resume-input receipt.

The initial fixed gates require exact BF16 preprocessing, feature relative RMS
error at most 0.02 and cosine at least 0.999, centered logit relative RMS error
at most 0.1, reference-to-native KL at most 0.03 and identical greedy decisions.
Checkpoint execution and worker lifecycle results belong in a dated run record.
The [2026-11-07 4B run](archive/runs/2026-11-07-gemma3-images.md) binds the
qualified artifact/builds, frozen numerical gates and actual HTTP lifecycle.

`scripts/gemma3_vision_stage_reference.py` loads only the original vision
weights for first-block diagnostics. It verifies the preserved full-source
receipt, captures unchanged module outputs and labels separately derived QK
stages. `gemma3_first_block_stage_reference_metal` compares bounded row
prefixes. These diagnostic stages locate arithmetic differences; they do not
replace whole-model reference gates.

`scripts/gemma3_image_worker_validation.py` exercises the shipped Python
Worker, an isolated Desktop instance and the actual loopback HTTP server.
It checks canonical greedy results, prepared-image single use, replay,
malformed-image recovery, request cancellation, HTTP client disconnects and
cached-check restart followed by image execution. Its private receipts bind
the worker, host modules, template helper, source receipt and canonical
fixture. The optional `--image-check-directory` requires an already valid
receipt for the same implementation. A successful local image execution
check is recorded separately from reference parity.

The architecture and reference behavior are described by the
[Google Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3),
[Transformers Gemma 3 implementation](https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/gemma3/modeling_gemma3.py)
and [SigLIP implementation](https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/siglip/modeling_siglip.py).
The model's Gemma license remains separate from the Apache-2.0 reference code.
