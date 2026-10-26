# Native image support implementation

The first target is one static image with the official BF16 `Qwen/Qwen3.5-0.8B`
checkpoint at `2fc06364715b967f1860aea9cf38778875588b17`. The implementation
follows the native Rust/Metal tower and decoder route. The converter now has an
experimental full-vision artifact mode. Native image execution, image transport,
preprocessing, and app attachments remain unimplemented. The Local API continues
to report `images:false`.

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

The root model tag is `Qwen35VisionLanguageModelConfig`. The current native
engine rejects this unknown tag. It must not be renamed to `LanguageModelConfig`
to bypass the implementation boundary.

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

## Remaining dependencies and acceptance

1. **Native fusion and positions:** introduce typed image spans and projected
   feature inputs; separate token lookup from decoder execution; preserve causal
   cache indices separately from three-axis positions. Match whole/chunked
   prefill and continuation offsets against pinned references. Keep scoring
   text-only and speculative image prefill disabled.
2. **Decode, preprocessing, and tower:** select the shared JPEG/PNG decoder;
   specify orientation, color, alpha, and resizing; prove patch ordering and
   intermediate numerics. Implement the native vision blocks and merger before
   adding the model tag to the native loader.
3. **Transport and lifecycle:** support inline image data URLs through bounded
   HTTP bodies and private staging references. Preserve the 1 MiB worker frame
   limit, authentication, serialization, epoch fencing, cancellation, and cleanup.
   Choose resource budgets using measurements before advertising them.
4. **Exact-tuple qualification:** run complete conversion and reference/native
   parity with the official weights. Measure peak memory and encode/prefill
   latency, test failure recovery, and publish aggregate evidence only.
5. **Swift and distribution:** add attachment input and progress through the
   common media pipeline, then qualify a packaged installation. Enable image
   capabilities only for the intersection of the qualified artifact, worker,
   and service policy.

Synthetic converter tests and header inspection do not establish preprocessing,
native inference, numeric parity, output quality, or supported hardware budgets.
The official-weight end-to-end conversion has not yet been run; current evidence
is synthetic conversion tests plus official metadata and tensor-header inspection.
No model weights or native image outputs are checked into the repository.
The [converter validation record](archive/runs/2026-10-26-qwen35-vision-artifact-contract.md)
binds these checks to the converter source and dependency-lock hashes.

## Primary references and provenance

The source metadata comes from the [exact checkpoint revision](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17).
Source payload identity comes from the [immutable tree's LFS metadata](https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17?recursive=true&expand=false).
The tensor name/shape inventory follows the public architecture declarations in
[Transformers Qwen3.5](https://github.com/huggingface/transformers/blob/2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397/src/transformers/models/qwen3_5/modeling_qwen3_5.py),
revision `2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397`, copyright 2025 The Qwen Team
and The HuggingFace Inc. team, Apache-2.0. The converter implements an independent
shape inventory and direct tensor copies; no Transformers runtime or model code
is imported. Existing Uzu and Lalamo provenance remains in
[upstream provenance](../provenance/UPSTREAM.md) and the
[converter guide](../tools/converter/README.md#provenance-and-policy).
