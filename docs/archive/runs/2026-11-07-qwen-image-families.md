# Qwen retained image execution paths

The implementation adds dense and sparse affine native paths and an explicit
retained Flash-Next path. The three representative artifacts below passed local
worker image lifecycle checks. Full decoder numerical qualification remains
in progress; the small packed reference currently fails the unchanged gate.

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
Local native execution and worker lifecycle checks passed as recorded below.

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

## Local execution and remaining numerical gate

The initial dense 27B automatic text check failed before image execution. A
14-row diagnostic isolated the reference BF16 MLP gate's downstream advisory
Int8 activation preference. The reference policy now forces its declared BF16
output; ordinary text defaults remain unchanged. The exact multirow MXU
admission regression and eight-step native text continuation passed on source
`224a112b`, engine executable SHA256
`4ee7629d07ed425efa2d0f7d9d4fa501d3e2ba783665b9030e92935687801859`.

The dense 27B, MoE 35B and retained Lily artifacts each passed the real worker's
automatic local checks and the same private lifecycle harness on source
`7f8b8379`, executable SHA256
`41baf746b8e0c112479c39cebff9322ae00ccd235347db0e41b12759e2fce53a`.
Each completed two distinct image cases with correct color responses, observed
cancellation requested after a real streamed delta, removed staged files,
recovered through an image-to-text request and a fresh image request, and kept
the worker healthy. Workers were reaped normally after qualification. These
are functional checks with greedy sampling and image MTP disabled; they do not
establish decoder parity or performance gains. Private receipts are
`qwen38-27b-7f8b8379-v2-image-qualification.json`,
`qwen35-moe-7f8b8379-image-qualification.json` and
`qwen38-lily-7f8b8379-image-qualification.json` in the ignored runtime folder.

The private logit replay now accepts dynamic admitted vocabulary and width.
New manifests bind all five admitted artifact hashes, exact source/revision,
family profile, greedy sampling and disabled MTP. Optional per-layer traces
have a 512 MiB aggregate allocation cap. The original immutable 0.8B replay
contract remains supported. The integrated replay compiled and executed.

Original-checkpoint CPU vision references are ready for dense 27B and MoE 35B.
Each has a normal 256 by 256 RGB fixture and a small 32 by 32 numerical stress
fixture, with 30 stages per case. The recipe uses the pinned Transformers
`2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397` extraction, Torch 2.14.1,
four CPU threads, BF16 parameters and preserved FP32 rotary buffers. All 333
original BF16 visual tensors match each converted artifact bit-for-bit,
including the independent MLX Conv3D transpose check. Private manifest hashes:

- Dense 27B: `4c49cf1bedb0188b6f7ef7f195cb03f31c5b910de99b5f6b1e91c962e598f468`.
- MoE 35B: `6975234c15579f62d8286db6c86297dc34514d0e0699fb91c6377e3c755a3630`.

Native tower replay on the sealed `224a112b` engine passed all 60 stages for
each artifact bit-for-bit: zero RMS and absolute error, cosine one. This
qualifies the retained tower independently of the packed decoder.

The bounded full-decoder control uses
`mlx-community/Qwen3.5-0.8B-MLX-4bit` at revision
`5d894f8cc4ef3e6c88537bf3746ed262f549da6a`. Its original 625,229,487-byte
source shard has SHA256
`f5a0d9dd3efa73510542a8023d610ff26be2b4b020d181cfc4bedaa1fcc5dd9e`;
the 626,210,248-byte converted artifact has SHA256
`7e127dfb8c78f22b94edd35e52717f50e3d70398fe3668abd2290d088b5b1343`.
The CPU producer retains the pinned Transformers graph, replaces every
parameter from the quantized checkpoint, binds operational text and vision
configuration, verifies reversible normalization gains, and hashes the actual
BF16 pixel tensor consumed. Its explicit comparison policy computes affine
weights in F32 and casts once to BF16 before each linear operation.

The two last-prompt logit comparisons on source `7f8b8379` produced relative
RMS errors 0.054535 and 0.030942. Both retained the same top prediction, but
both fail the fixed 0.02 RMS gate. First-layer output RMS already differs by
about 0.0018 and grows across the decoder. Failed references remain preserved.
Same-input GDN and MLP projection fixtures isolate packed matrix arithmetic;
their native replay is pending. A tiny independent stock-MLX control verified
131,072 dequantized weights bit-for-bit against the single-cast BF16 policy,
and stock regular matmul agreed with Torch. Stock quantized matmul differed
from both, so no alternative full-model reference policy has been promoted.

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

The static bound totals 86,215,737,056 bytes. During the successful retained
Lily worker lifecycle run, peak observed worker RSS was 55,519,461,376 bytes
and minimum observed available memory was 38,891,159,552 bytes. These sampled
measurements support the conservative bound and do not replace memory
admission or an allocator-level peak measurement. Native payload admission and
operation-specific bounds still apply, and actual available memory must be
refreshed before a full load.

## Affine decoder contraction follow-up

The original failed packed references remain preserved. Single-partition F32
GEMM accumulation removed BF16 scratch-partition rounding, and the explicit
affine GEMV coefficient boundary reduced the single-row projection controls
to three BF16 mismatches. Production full-decoder RMS still measured 0.054045
and 0.022619 on source `2eb2ad96`, so that implementation failed the fixed
0.02 gate.

A separate scalar packed-dot witness retains the pinned eight-by-four F32
contraction after materializing each effective coefficient in BF16. It matches
all frozen same-input projections exactly. A test-only, exact-artifact-bound
full-model override on source `202269cd` also matches both complete
248,320-element last-prompt readouts bit-for-bit: zero RMS and absolute error,
cosine one. Its executable SHA256 is
`70ad27c38886c838d9dd98a7ed92c62d5045cec40b8926aa8de83f620e32da9d`.
This establishes the remaining contraction boundary for this frozen policy.

The production affine-reference encoder is being qualified separately. It
handles unsigned group-64 4/5/6/8-bit coefficients, both metadata strides,
direct gathered bank rows, full-BF16 control matrices, and BF16 results widened
to F32 for routing. The pinned
[Qwen3.5 MoE router](https://github.com/huggingface/transformers/blob/2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)
uses a linear operation in the activation/weight dtype and requests F32 only
for softmax. This supports the BF16-result boundary under the admitted BF16
activation and effective-weight policy.

The encoder validates shape, byte extents, metadata strides, alignment, total
work and dispatch counts before allocating its sole output. It tiles output
coordinates while retaining complete K contractions. Each dispatch has at most
2^30 multiply terms and 65,536 output dots; each encode has at most 2^38 terms,
1,024 dispatches and 64 Mi output elements. The latter permits at most 128 MiB
of BF16 output or 256 MiB of widened F32 output. There is no dequantized weight
bank or accumulation scratch. Mixed row partitions include their existing copy
dispatches in the aggregate bound. Native prefill remains at its existing
1,024-token units, including the selected-expert expansion. Existing lifecycle
and cancellation boundaries remain in place.

Metal maps the 32 source partial sums onto one SIMD group, preserving the
stride-32 FMA chains and the source reduction tree. The scalar test witness
remains independent. The new path is selected only by `Qwen35AffineBF16`;
ordinary Legacy dispatch stays unchanged. Production numerical and lifecycle
replays are pending; no performance gain is claimed.

Explicit `materialize_affine_input_sum` or `source_output_widths` numerical
markers retain their existing backend route, including its original fold and
partition selection; neither the new encoder nor the earlier reference
accumulation/GEMV hook applies to those matrices. Plain mixed-width storage
partitions are eligible for the new encoder. A header-only inspection of the
27B artifact found 209 MLX specs and 49 mixed row-partition specs, with no
explicit source arithmetic markers. Its input lookup is Q8 `input_output`
storage `[248320,5120]`; the separate readout is Q6 `output_input` packed
storage `[248320,3840]` with BF16 metadata `[80,248320]`. Input lookup retains
its existing implementation, and the separate readout and unmarked decoder
matrices select the new contraction. A small factory fixture covers that
untied layout and a tied input-output table; native replay remains required.

### Production packed-reference qualification

Compiled source `2529098a2eabfb76fb77f9481b43e889c2a5dad1` passed all seven
focused CPU/Metal encoder tests. The sealed engine-test executable SHA256 is
`7c8e5edeebe1f7110bd50e2d44601521f07898225f17ca2cc841a4ec58efbeeb`.
The unchanged audited 0.8B reference manifest remains
`f22d7bde0e2473fad29dc62a57a72f1a1e23e0ea1ade886f6db321382d8d54bd`,
bound to artifact
`7e127dfb8c78f22b94edd35e52717f50e3d70398fe3668abd2290d088b5b1343`.

The actual production encoder passed both complete last-prompt logit cases
bit-for-bit: each has 248,320 logits, zero relative RMS and absolute error,
cosine one, zero BF16 mismatches, and matching top-1. The independent scalar
test-only full-model witness was replayed separately and was also exact.
Both production and scalar same-input projections were exact at M=1 and
M=83/96 for GDN input/output and MLP up/down (32 stage comparisons total).
The fixed RMS/cosine/top-1 gate was unchanged, and every prior failed receipt
and prior native output was preserved.

Six sequential correctness tests exited successfully without the resource
guard firing. The production full replay took 4.246 seconds with sampled
peak process RSS of 972,128,256 bytes; these are functional observations with
no performance claim. An unrelated Lily pilot worker was observed after the
replay; this was not a quiet-host benchmark. The private replay receipt SHA256
is `7fd2dc2d8124d0ae471ba88aef6bbd508f7391695b2bf235c63d388ddc34b9e3`.
Actual 27B and MoE worker lifecycle/cancellation rechecks on the production
encoder remain pending. The small exact dense result does not qualify those
larger checkpoints or explicit source-wide arithmetic policies numerically.

## Primary references

- [Official Qwen3.8-27B configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json):
  its dense architecture is `qwen3_5`, with the same image tokenization and M-RoPE geometry.
- [Lily Q4-XL model card](https://huggingface.co/fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl):
  source format and retained BF16 visual tensor inventory.
- [MLX-VLM Qwen3.5 wrapper](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_5/qwen3_5.py):
  visual namespace mapping and placeholder fusion.
- [Transformers Qwen3.5 MoE](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py):
  architecture-specific MoE operation requirements.
