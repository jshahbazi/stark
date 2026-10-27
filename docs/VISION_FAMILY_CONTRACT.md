# Vision model families and local installation checks

Stark can admit changed-weight models through a versioned family contract. The
first profile, `qwen35-dense-bf16-merged-v1` version `1`, covers complete merged
BF16 Qwen3.5 fine-tunes with the implemented 0.8B geometry. The converter checks
the source, the native loader independently checks the emitted artifact, and the
host runs local execution checks before enabling image input.

This is the current implementation boundary. Successful local checks establish
that the artifact executes the supported text and image operations on this Mac.
They do not establish reference parity, visual accuracy, or preservation of the
base model's capabilities after training. The official checkpoint's existing
[2% reference gate](archive/runs/2026-10-26-qwen35-vision-parity.md) and exact
[execution catalog](../tools/converter/stark_convert/vision_runtime_qualifications.json)
remain separate.

## Initial family profile

Admission requires the complete supported decoder and vision tower, including
all required tensor names, shapes and dtypes. The decoder has width 1,024, 24
layers and a 248,320-row vocabulary; the vision tower has width 768, 12 blocks,
16-pixel patches and a width-1,024 projector. Attention, DeltaNet, tied embeddings,
normalization, M-RoPE and processor operations must match the implemented profile.
Required matrices and convolution weights must be BF16. Explicit norm-scale,
`a_log` and `dt_bias` control-weight exceptions may originate in FP32. Declared
auxiliary MTP tensors are omitted as in the existing text adapter.

Weights may differ from the official checkpoint. Metadata must use recognized
forms:

- The known source tokenizer or its normalized serialization; the emitted
  tokenizer must have SHA-256
  `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4`.
  The tokenizer contains 248,070 IDs; embedding and output storage retain the
  model's padded 248,320 rows.
- One of the two reviewed Qwen templates, identified by SHA-256
  `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`
  or `7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67`.
  The second form changes tool-argument iteration to check for a mapping. Tools
  remain outside Stark's chat API. Inline and standalone templates must agree.
- Exactly one source processor file: `preprocessor_config.json`, or the
  recognized `image_processor` form in `processor_config.json`. Known explicit
  defaults are normalized to the same supported image recipe. The checkpoint's
  actual processor filename and raw hash remain in source provenance.
- Self-contained source configuration, tokenizer metadata and template. No
  custom `auto_map` code, unresolved adapter weights or quantization metadata.

Larger Qwen3.5 sizes, MoE, LoRA-only repositories, quantized vision weights,
changed tokenizers, unrecognized templates/processors and other architectures
require a new implemented profile or an explicit extension. Matching a repository
name or architecture label alone cannot admit a model.

## Artifact and source identities

The existing exact official conversion remains schema `2`. Compatible family
conversions use `Qwen35VisionLanguageArtifact` schema `3`, with the same root tag
`Qwen35VisionLanguageModelConfig` and `mediaref-v1` transport. Schema `3` adds:

```json
{
  "family": {"profile": "qwen35-dense-bf16-merged-v1", "version": 1},
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

The native loader validates the full configuration and all 383 emitted tensor
headers, including storage coverage, before allocating weights. It opens bounded
regular files without following symlinks, hashes the selected buffers and files,
retains their descriptors, and rechecks their identity before publishing a loaded
model. Readiness reports the five admitted file hashes, schema and family
versions, source identity, runtime identity and supported limits. Its
`image_input.qualified` remains `false`: host evidence determines availability.

Conversion itself performs no inference. Family receipts report
`native_loading: "supported_family_unverified"` and `execution: "unverified"`.
`--require-native-support` continues to require an exact stored qualification;
it cannot turn a new family artifact into reference-qualified output.

## Automatic installation and startup checks

For a schema-3 vision installation, the desktop bridge validates the family
version, stops the previously loaded model, and starts a temporary worker before
publishing the model in the library. The UI reports that image support is being
checked on this Mac. A failed or interrupted check does not publish the new
installation. Managed failed conversions are removed; source folders supplied
by the user remain intact. Only one model is loaded during this process.

Before native startup, the host compares available memory with the weight file
size plus 512 MiB. This rejects an obviously insufficient budget. Available
memory is an observation; it is not a reservation or a peak-memory bound.

After native admission, the checks use fixed public synthetic input and bounded
greedy generation:

1. Generate a nonempty text response within the reported token budget.
2. Tokenize a supported image prompt and validate its token accounting.
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
worker executable hash, family/schema versions, image limits, Metal device and
macOS identity, Python version, and hashes of the checker, worker client,
readiness policy, private staging and hashing helper sources. It also binds the
versions of Jinja2, MarkupSafe, tokenizers and psutil. A change in any identity
requires new checks. Malformed, unexpected or inflated evidence is rejected.
Fatal native/protocol failures invalidate the receipt and the current process's
cache key. Normal Stop clears the live capability; the next Run must admit the
model and validate the receipt again.

Library metadata is informational. Attach Image and `capabilities.images` derive
from the current worker's passing evidence. Connection Details and the composer
distinguish locally checked models from reference-verified models. None of these
labels qualifies an arbitrary fine-tune against official model outputs.

The [2026-10-27 installation run](archive/runs/2026-10-27-qwen35-family-installation.md)
records a genuine merged fine-tune's conversion, offline replay and application
checks, with its source and artifact identities and evidence limits.
