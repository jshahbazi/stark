# Qwen retained image execution paths

The implementation adds two explicit paths. Execution qualification is pending
at this record's first revision; model-free checks establish conversion and
admission behavior only.

## Dense affine Qwen3.5 and Qwen3.8-27B

`qwen35-dense-affine-merged-v3`, version 3, retains canonical group-64 affine
decoder matrices and a complete BF16 visual tower. Native admission reconstructs
the same bounded dense graph as the BF16 v2 profile, requires folded RMS gains,
and validates every packed weight/scale/bias plane against its logical matrix.
Row partitions must cover that matrix exactly. BF16 activation, RMS, attention
and GDN boundaries remain explicit; existing affine matmul kernels execute the
packed projections. Image artifacts omit the separately declared MTP head.

MLX visual names are admitted only with their corresponding text wrapper.
MLX Conv3d `[H,T,P,P,C]` weights are restored bitwise to `[H,C,T,P,P]` using
bounded tiled transposes. Other BF16 visual payloads are copied unchanged.
Text-only exports cannot supply the missing tower.

A header-only plan of the existing local
`Jundot/Qwen3.8-27B-oQ4e-mtp` source (revision
`7dd276b77fcb27c982ebd8d8a434539345713fea`) has 1,613 output operations,
including all 333 BF16 visual tensors, and 17,675,742,688 payload bytes. This
planning check was followed by a complete offline conversion. The published
artifact is 17,676,008,168 bytes; its weights SHA256 is
`dff4809e84a628f772183293be7f6ba69d276ee5cf34571298dda04af5e84fa1`.
Native execution remains unverified at this revision.

The exact Qwen3.8 template was reviewed separately: image serialization still
emits one start/pad/end run, while assistant reasoning can be preserved and
the existing boolean thinking control defaults on. Its SHA256 is
`c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.
The source tokenizer differs from the existing reviewed tokenizer only in
merge serialization. Tokenizers normalizes it to the already admitted output
hash `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4`.

## Original-format Flash-Next

The versioned `qwen38-lily-retained-bf16-vision-v1` profile uses an explicit
schema-6 `stark-image-artifact.json`. It binds source identity, the shard index,
all model shards and any external n-gram table, configuration, tokenizer,
normalized fixed processor recipe and standalone template. Aggregate artifact
identity is SHA256 over sorted filename bytes, a zero delimiter, and each
64-byte lowercase ASCII payload digest. Native loading rehashes these files
and checks their file identities after model loading. No manifest means no
image capability. Existing text receipts and text loading remain valid.

Two retained formats have preparation paths:

- `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision
  `0f94842e3682b49acace4daeb4cf2e65a6f94f83`.
- `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`, revision
  `788c07a8725642bccf13a0a3b8a21954ab84b4f2`. Its 101 Safetensors shards use
  75,303,252,216 bytes and its external n-gram table uses 32,000,153,976 bytes.

An existing complete source folder can be explicitly upgraded without copying
its model payloads:

```sh
uv run --project tools/converter --locked python -m stark_convert.hybrid_prepare \
  qwen38-lily-q4-xl --output /path/to/prepared-model --upgrade-images
```

New native preparations accept `--artifact-mode vision-language`. The original
PipeNetwork format retains vision payloads, but its distinct text loader has no
image fusion bridge in this change.

The public model prepares opaque, instance-bound CPU patches, expands exactly
one complete image marker run, executes the retained native tower and prefill,
and returns target logits for ordinary continuation. Image MTP is disabled.
Image state cannot create or restore prefix snapshots. Cancellation is observed
before execution, after the bounded tower pass, and between completed text
prefill chunks. In-flight Metal commands finish before their resources are
released. A cancelled request resets safely; a device execution failure fences
the instance.

## Checks

- Native production generator `cargo +1.98.1 check --locked -p stark-generate`
  passed in the isolated worktree target.
- CPU engine test target compiled with the pinned nightly. Ten vision-family
  admission tests and one image expansion/control-boundary test passed.
- The focused dense conversion, family, geometry, MTP, runtime and metadata
  suite passed 284 tests. The focused native preparation, retained manifest and
  catalog suite passed 21 tests.
- Repository policy, Rust formatting, Python lint/formatting and diff whitespace
  checks passed.

## Shared sparse expert primitives

`encodable_block::mlp::routed::RoutedExperts` executes flattened expert banks
with the existing canonical full-precision or affine matrix kernels. Device
routing selects stable top-k IDs, expands them into bounded weight-row gathers,
and accumulates only selected expert outputs. The helper receives separately
normalized BF16 expert inputs and F32 router logits, allowing architectures to
own their router normalization. Router-logit, probability and weighted-product
BF16 boundaries are explicit policy fields. Shapes, buffer extents and expert
bank indexing are checked before command encoding. Non-finite router values
publish safe expert IDs and non-finite weights, allowing the existing target
readout guard to reject the request without unsafe gathers.

The Qwen MLP wrapper adds its sigmoid-gated shared expert. A two-row, four-expert
CPU fixture passed with both BF16 and packed Q4 banks; rows select different
expert pairs and ties have a defined lower-ID ordering. Production generator
type-check passed with the new CPU/Metal kernel bindings.

The versioned `qwen35-moe-affine-merged-v4` artifact profile now admits the
explicit Qwen3.5 MoE architecture, group-64 packed affine BF16 execution, its
normalized selected-expert probabilities and one sigmoid-gated shared expert.
The converter flattens the three-dimensional MLX expert banks, interleaves
up/gate rows per expert, and retains the ordinary BF16 vision tower. Native
geometry checks independently reconstruct every expert/shared/router bank and
validate bounded dimensions and physical tensor headers. Image requests omit
MTP and use the existing private prepared-image flow.

The full converter suite passed 6,158 tests with 90 skips. Native production
type-check and four focused CPU sparse-expert/geometry tests passed. The exact
source `mlx-community/Qwen3.5-35B-A3B-4bit` at revision
`1e20fd8d42056f870933bf98ca6211024744f7ec` was downloaded once (20,418,624,545
bytes). Header-only conversion planning admitted 1,520 operations, including
333 BF16 vision tensors, for a 20,393,738,208-byte output payload.

After explicit resource admission, the single streaming output completed in
36.33 seconds: 20,394,001,296 bytes, SHA-256
`2f762b2629aaefc2fea6eb37ff6caaa4747fdc23a040e36f331fa4a056c6cad7`.
The converter reported an 8 MiB transform workspace and 585,678,848-byte peak
RSS. Free disk remained 52.48 GiB. No duplicate source payload was created.

## Full-model checks in progress

The source-bound dense 27B artifact reaches readiness with the sealed worker
`32816d8494e29d0a5ca6858f4bf60f6203c18fec256befd4d3de5eda738f9449`.
Its first automatic **text** generation check returns `native_generation_failed`.
This happens before image execution. The worker is reaped after each failed
check; the artifact has no successful local image qualification receipt.
Further full-model retries are held for a typed, content-free native diagnostic.

The private logit replay now accepts dynamic admitted vocabulary and width.
New manifests bind all five admitted artifact hashes, exact source/revision,
family profile, greedy sampling and disabled MTP. Optional per-layer traces
have a 512 MiB aggregate allocation cap. The original immutable 0.8B replay
contract remains supported. Native type-check of this replay-only update is
pending the coordinated integrated build.

Original-checkpoint CPU vision references are ready for dense 27B and MoE 35B.
Each has a normal 256 by 256 RGB fixture and a small 32 by 32 numerical stress
fixture, with 30 stages per case. The recipe uses the pinned Transformers
`2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397` extraction, Torch 2.14.1,
four CPU threads, BF16 parameters and preserved FP32 rotary buffers. All 333
original BF16 visual tensors match each converted artifact bit-for-bit,
including the independent MLX Conv3D transpose check. Private manifest hashes:

- Dense 27B: `4c49cf1bedb0188b6f7ef7f195cb03f31c5b910de99b5f6b1e91c962e598f468`.
- MoE 35B: `6975234c15579f62d8286db6c86297dc34514d0e0699fb91c6377e3c755a3630`.

Native tower replay, full decoder numerical parity, and worker image lifecycle
qualification remain pending. Converter and fixture checks do not establish
end-to-end image support for these particular artifacts.

## Retained Lily memory preflight

Header/metadata inspection reads the existing immutable, hard-linked source;
it changes no source files or payload permissions. The retained artifact's
106,905,155,192 file bytes contain 32,000,153,600 bytes of paged PLE tables and
1,842,067,616 bytes of MTP tensors skipped for images. Selected resident
trunk plus vision tensors total 73,062,399,712 bytes. The loader reads these
directly into shared Metal buffers, while the PLE table stays a read-only file
mapping and copies selected rows into bounded staging buffers.

The code-pinned observation recipe applies only to this exact artifact,
configuration and index identity. It includes the adapter's initial 16,384-token
state allocation even when the check requests 2,048 context tokens. Twelve
attention KV/indexer stores plus 36 recurrent layers compute to 394,616,832
bytes; the decoder's largest dequant staging tensor is 62,914,560 bytes.
Conservative rounded buckets reserve 512 MiB for state, 256 MiB for decoder
scratch, 2 GiB for a 2,048-row prefill, 512 MiB for CPU/GPU image preparation at
the 4,096-patch cap, 1 GiB for the paged-table working set, and another 8 GiB for
driver, allocator and host-memory drift. Unknown identities retain the original
full-file-payload plus 2 GiB observation check.

This is a static allocation estimate, not a measured peak or a memory
reservation. Native payload admission and operation-specific bounds still
apply, and actual available memory must be refreshed before a full load.

## Primary references

- [Official Qwen3.8-27B configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json):
  its dense architecture is `qwen3_5`, with the same image tokenization and M-RoPE geometry.
- [Lily Q4-XL model card](https://huggingface.co/fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl):
  source format and retained BF16 visual tensor inventory.
- [MLX-VLM Qwen3.5 wrapper](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_5/qwen3_5.py):
  visual namespace mapping and placeholder fusion.
- [Transformers Qwen3.5 MoE](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py):
  architecture-specific MoE operation requirements.
