# Vision model families and local installation checks

Stark admits complete merged BF16 Qwen3.5 models through a versioned family
contract. The current profile, `qwen35-dense-bf16-merged-v2` version `2`, derives
the dense decoder and vision tower from validated configuration. It accepts
compatible sizes and changed-weight fine-tunes without a per-repository or
per-size allowlist. The converter checks the source, the native loader
independently reconstructs and checks the emitted graph and tensor layout, and
the host runs local execution checks before enabling image input.

This is the current implementation boundary. Successful local checks establish
that the artifact executes the supported text and image operations on this Mac.
They do not establish reference parity, visual accuracy, or preservation of the
base model's capabilities after training. The official checkpoint's existing
[2% reference gate](archive/runs/2026-10-26-qwen35-vision-parity.md) and exact
[execution catalog](../tools/converter/stark_convert/vision_runtime_qualifications.json)
remain separate.

## Config-derived family profile

Admission requires the complete supported decoder and vision tower, including
all required tensor names, shapes and dtypes. Dimensions and the hybrid layer
schedule come from the checkpoint; the implemented operations remain fixed.
Supported operations include gated causal attention with 256-wide heads,
DeltaNet with 128-wide key/value heads and a four-step convolution, SiLU dense
MLPs, supported RMS normalization and the existing Qwen3.5 M-RoPE recipe.
Tied or untied embeddings must agree with the configuration and complete weight
layout. The tower retains RGB input, 16-pixel patches, temporal patch size 2,
spatial merge 2, GELU blocks and a projector matching the decoder width.
DeepStack injection and alternate mixer/processor operations are unsupported.

| Geometry or storage | Admission bound |
| --- | --- |
| Decoder width / MLP width | At most 16,384 / 131,072; multiples of 32 |
| Decoder layers | 1 through 256; a supported hybrid schedule containing both mixer types |
| Attention and DeltaNet head/group counts | 1 through 256; head count divisible by group count, consistent geometry within each mixer type |
| Vocabulary / declared context | At most 1,048,576 each; tokenizer IDs and output rows must agree |
| Vision depth / width / MLP width | At most 64 / 4,096 / 32,768 |
| Vision heads / learned position count | At most 64 / 16,384; width divisible by heads, head width divisible by 4, square position grid |
| Converted weight file | At most 128 GiB, including its header |
| Elements per emitted tensor | At most `2^32 - 1` |
| Configuration / safetensors header | At most 1 MiB each |

These bounds describe implementation and indexing limits. They do not promise
that a model will fit the current Mac or that its outputs have been qualified.
Official dense 0.8B, 2B, 4B, 9B and 27B dimensions and a noncatalog size are
covered by geometry fixtures; that evidence alone does not establish runtime
execution for each size.

Required matrices and convolution weights must be BF16. Explicit norm-scale,
`a_log` and `dt_bias` control-weight exceptions may originate in FP32. Declared
auxiliary MTP tensors are omitted as in the existing text adapter.

Weights may differ from the official checkpoint. Metadata must use recognized
forms:

- One of two reviewed tokenizer processing programs. Their canonical output
  SHA-256 values are
  `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4`
  and, for version 2 only,
  `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`.
  Both retain the same vocabulary, model and added special tokens. They differ
  in processing: the first groups letters and combining marks with
  `[\p{L}\p{M}]+`; the second uses `\p{L}+` and also has different ByteLevel
  trimming/decoder flags. Combining-mark text can therefore tokenize
  differently. Conversion preserves the selected program and never rewrites
  the second into the first. JSON spacing and key-order variants are accepted
  only when their complete program fingerprint matches a reviewed form. The
  actual source hash and selected output component hash remain recorded.
  The tokenizer contains 248,070 IDs; embedding and output row counts come from
  the admitted configuration and may include unused padding.
- One of the reviewed Qwen templates, identified by SHA-256
  `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`
  or `7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67`.
  The second form changes tool-argument iteration to check for a mapping. Tools
  remain outside Stark's chat API. Inline and standalone templates must agree.
- Version 2 additionally recognizes the stock reasoning template with SHA-256
  `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`.
  Its existing `enable_thinking` control defaults on when omitted. Stark's app
  and CLI pass the selected boolean explicitly and retain their default of
  thinking off. This template is unavailable through the fixed version-1
  profile; arbitrary template programs remain unsupported.
- `preprocessor_config.json`, the recognized `image_processor` form in
  `processor_config.json`, or both files when their normalized recipes agree.
  Known explicit defaults and the equivalent `Qwen2VLImageProcessor`/`Fast`
  class spellings normalize to the supported native recipe. Every actual source
  processor filename and raw hash remains in provenance. Video metadata is
  retained only as source provenance; this path accepts still images.
- Self-contained source configuration, tokenizer metadata and template. No
  custom `auto_map` code, unresolved adapter weights or quantization metadata.

MoE, LoRA-only repositories, quantized vision weights, tokenizer programs outside
the reviewed forms, unrecognized templates/processors and other architectures
require additional implementation. Matching a repository name, model size or
architecture label alone cannot admit a model.

## Existing artifacts and readers

The exact official 0.8B conversion remains schema `2`. The fixed
`qwen35-dense-bf16-merged-v1`/version `1` profile continues to admit previously
converted 0.8B artifacts with its original graph, `87a7830d…` tokenizer and two
templates. Schema 2 retains that same original tokenizer gate. New family
conversion emits the config-derived version-2 profile. Current native and host
readers accept both family versions and schema 2. Older readers reject version 2
as unsupported; changing the family label on an old artifact is not a migration.

## Artifact and source identities

The existing exact official conversion remains schema `2`. Compatible family
conversions use `Qwen35VisionLanguageArtifact` schema `3`, with the same root tag
`Qwen35VisionLanguageModelConfig` and `mediaref-v1` transport. Schema `3` adds:

```json
{
  "family": {"profile": "qwen35-dense-bf16-merged-v2", "version": 2},
  "component_hashes": {
    "model.safetensors": "<sha256>",
    "tokenizer.json": "<sha256>",
    "preprocessor_config.json": "<sha256>",
    "chat_template.jinja": "<sha256>"
  }
}
```

The configuration's hash is calculated externally; it does not contain its own
hash. Source provenance binds the selected metadata bytes and every consumed
weight/index file to the resolved revision. Local sources use a logical `local`
identity when no repository identity is supplied; local paths are not exposed
in native readiness. Private source manifests retain the file locations needed
for offline replay.

The native loader derives the required tensor set from the admitted graph and
validates every emitted header, including storage coverage, before allocating
weights. The old 0.8B profile has 383 tensors; other graphs have their own counts.
It opens bounded regular files without following symlinks, hashes the selected buffers and files,
retains their descriptors, and rechecks their identity before publishing a loaded
model. Readiness reports the five admitted file hashes, schema and family
versions, source identity, runtime identity and supported limits. Its
`image_input.qualified` remains `false`: host evidence determines availability.

Conversion itself performs no inference. Family receipts report
`native_loading: "supported_family_unverified"` and `execution: "unverified"`.
`--require-native-support` continues to require an exact stored qualification;
it cannot turn a new family artifact into reference-qualified output.

## Automatic installation and startup checks

The shared [family/mode preflight](MODEL_SUPPORT.md) checks source metadata before
weight download. Its conditional result does not enable images or replace the
following native and execution checks.

For a schema-3 vision installation, the desktop bridge validates the family
version, stops the previously loaded model, and starts a temporary worker before
publishing the model in the library. The UI reports that image support is being
checked on this Mac. A failed or interrupted check does not publish the new
installation. Managed failed conversions are removed; source folders supplied
by the user remain intact. Only one model is loaded during this process.

Before native startup, the host compares available memory with the weight file
size plus an allowance for checks. Version 1 uses 512 MiB. Version 2 estimates
2048-token recurrent/KV state, decoder projections and MLP scratch, DeltaNet
prefill scratch and the fixed check image's vision scratch from bounded scalar
geometry, with a minimum allowance of 512 MiB. This rejects an obviously
insufficient budget. Available memory is an observation; it is not a reservation,
a peak-memory bound, or a guarantee for larger user-selected contexts.

After native admission, the checks use fixed public synthetic input and bounded
greedy generation:

1. Generate a nonempty text response within the reported token budget.
2. Tokenize a supported image prompt and validate its token accounting against
   the vocabulary reported by the admitted native worker.
3. Reject a malformed PNG recoverably.
4. Prepare, generate and discard the same valid 256-by-256 RGB PNG twice through
   fresh image transactions. Each must produce a nonempty bounded response,
   exactly 64 image tokens and the expected expanded prompt-token count.
5. Confirm consistent accounting, an empty staging directory, a live worker and
   unchanged executable and artifact identities before recording success.

The checks share a bounded deadline. They evaluate execution, accounting,
recovery and cleanup; no answer-content predicate asserts visual understanding.
Run performs native admission again and either reruns the checks or reuses an
identical private receipt. A rebuilt schema-2 official artifact/worker tuple
without a catalog match can also use this local-check path when it advertises
the supported family profile.

## Evidence and cache lifetime

The live host reports one of these values in `image_verification`:

| Value | Evidence |
| --- | --- |
| `reference_verified` | The exact schema-2 artifact, worker, Metal device and macOS tuple matches a passed external execution record. |
| `locally_tested` | The supported family passed the local execution checks, or an identical private receipt was reused. |
| `unverified` | No active passing evidence; image input remains disabled. |

The desktop stores receipts under its private `image-checks` directory. Directory
permissions are `0700`; receipt files are single-link regular files with `0600`
permissions. The receipt contains fingerprints and boolean check results. It
contains no prompt, response, image bytes, token vectors or credentials.

The receipt identity includes all five admitted artifact hashes, the exact
worker executable hash, actual family/schema versions, admitted vocabulary size,
image limits, Metal device and
macOS identity, Python version, and hashes of the checker, worker client,
readiness policy, private staging and hashing helper sources. It also binds the
versions of Jinja2, MarkupSafe, tokenizers and psutil. A change in any identity
requires new checks. Malformed, unexpected or inflated evidence is rejected.
Checker policy version 2 invalidates older check identities; receipt document
schema remains version 1.
Fatal native/protocol failures invalidate the receipt and the current process's
cache key. Normal Stop clears the live capability; the next Run must admit the
model and validate the receipt again.

Library metadata is informational. Attach Image and `capabilities.images` derive
from the current worker's passing evidence. Connection Details and the composer
distinguish locally checked models from reference-verified models. None of these
labels qualifies an arbitrary fine-tune against official model outputs.

The [initial installation run](archive/runs/2026-10-27-qwen35-family-installation.md)
records a genuine merged 0.8B fine-tune's version-1 conversion, offline replay
and application checks, with its source/artifact identities and evidence limits.
The [config-derived family run](archive/runs/2026-10-27-qwen35-config-driven-family.md)
records actual official 4B and merged 4B fine-tune conversion, reference and
application checks. Its 2B, 9B and 27B coverage remains metadata/index, synthetic
header and operator evidence rather than full-model runtime qualification.
