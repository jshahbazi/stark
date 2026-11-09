# 20261109-GLMVIS-01: GLM image source admission and implementation inventory

## Identity and fixed gates

Registered 2026-11-09 before code changes. Owner: GLM image agent. Governing
work item: [IMGPAR-15](../../IMAGE_DEFAULT_PARITY_PLAN.md). Source baseline:
`8166e198`; branch `codex/glm-image-parity`. This first attempt inventories and
binds the missing image components. It does not qualify image execution.

| Field | Value |
| --- | --- |
| Kind / initial state | Compatibility and implementation; attempted |
| Initial evidence / promotion | Source/structural only; awaiting qualification |
| Decoder | `unsloth/GLM-5.3-Flash-GGUF` at `a38483c8cd5df544f53d70fb281afe97369d5ab6`, existing UD-Q2_K_XL four-shard text route |
| Metadata | `zai-org/GLM-5.3-Flash` at `04c4e9e95c5da8862dced7e5056455116f83a7e0` |
| Vision sidecar | `mmproj-BF16.gguf` from the same pinned Unsloth revision; 1,164,010,176 bytes; SHA256 `82b7adb44411dbb298b1c689567804e5ed0790d80f308ed8b5b6b8ac0527d2ac` |
| Sidecar identity evidence | Resolve HEAD `X-Linked-Size` and `X-Linked-ETag`, checked 2026-11-09; only the first 1 MiB range inspected, not the full payload |
| Native backend | Existing Rust/Metal GGUF path; optional EXL3 path is a separate implementation/qualification scope |
| Runtime/build | No new worker built or model loaded in this first phase |
| Resources | About 31 GiB disk headroom at intake; serialized build/GPU queue; no model deletion or full source download |

Fixed first-phase gates: recognize only the pinned source/profile, preserve the
source's one declared MTP layer as a declaration rather than execution proof,
reject altered architecture/tower/token contracts, and keep image capability
disabled until actual loading/fusion/serving is implemented and qualified.

## Observed implementation gaps

The [pinned source configuration][config] declares a conditional-generation
architecture and vision encoder. Current preparation requests only
`UD-Q2_K_XL/*.gguf`, excluding the separately published projector/encoder
sidecar. `Model::load_with_policy` explicitly rejects non-Lily image loading.
The GLM decoder currently obtains input rows only from its token embedding
matrix. The host catalog incorrectly describes the entire family as lacking
image input; the real limitation is Stark's current implementation.

GLM's image encoder cannot use the Qwen tower unchanged: its processor uses
aspect-ratio-preserving resize and black padding, 14-pixel spatial patches,
temporal duplication and a 2-by-2 merge. Its encoder includes a downsample
convolution and a projection/normalization/clamped-SwiGLU merger. Source
references: the official [Transformers model][model] and [processor][processor].
These moving reference pages are investigation references; pin their revision
and numerical fixtures before implementing or qualifying their arithmetic.

Ordinary native GGUF does not execute its included source MTP head. The
explicitly selected retained-BF16 EXL3 profile supports text MTP with three
drafts by default and head state in prefix snapshots. Neither route supports
images today. Adding images must preserve each route's actual text policy;
missing native GGUF MTP remains a concrete implementation gap.

## Implementation sequence and acceptance

1. Bind the pinned source/profile and sidecar identity in a fail-closed source
   plan. This plan must neither download payloads nor grant runtime capability.
2. Inspect the sidecar's bounded header and bind all encoder/projector tensor
   shapes, dtypes, spans and metadata to the chosen source. Add a content-bound
   image manifest and preparation/upgrade route that preserves the existing
   decoder bytes. Store the sidecar separately from the four top-level GGUF
   decoder shards: existing preparation and native source identity require an
   exact split inventory and must not reinterpret the sidecar as a fifth shard.
   Keep EXL3 sidecar ownership distinct and charge the sidecar in host totals.
3. Implement GLM preprocessing and tower/merger operators against pinned source
   fixtures, then admit the actual BF16 sidecar and run full image features.
4. Add request-owned embedding fusion before the unchanged GLM mHC expansion,
   preserving the existing text graph and all token/context/resource guards.
   Clear text prefixes and disable MTP only for the image transaction. The EXL3
   C ABI currently has no visual-embedding input and needs its own scoped
   extension, memory ownership and restoration proof.
5. Qualify image-loaded text snapshots: native GGUF recurrence/convolutions,
   MLA/indexer state and incomplete pooled tail at the 1,024-token grid; EXL3
   target/head snapshots at its own admitted grid. Include exact fresh/cached
   greedy and matched-mode seeded outputs, scope/context/MTP switches, image
   success/failure/cancel, delivery failure, reset/restart and idle invalidation.
6. Compare prior qualified text, candidate text and default image-loaded text
   with identical target/head weights, arithmetic, context, sampler and MTP.
   Predeclare bounds from repeated baseline variability before timing. Separate
   cache-cold and cached follow-ups, prefill/first-token, useful decode,
   end-to-end latency, loading and measured memory.
7. Verify real text MTP proposals with images resident on each implemented MTP
   route; image requests must report zero drafts/proposals/reuse. Following text
   must start fresh, restore the selected MTP policy and subsequently cache.
   Account resident tower/head payloads, scratch and snapshots in all admission.
8. Enable normal default image loading and honest live capabilities only after
   full artifact/image/lifecycle checks. Review, CI, release and installed-app
   behavior remain separate from branch qualification.

## Results and remaining work

Source planning and bounded header inventory completed. `glm_vision.py` now
checks bounded JSON, rejects duplicate keys and altered source/profile/token/head
contracts, and returns the required pinned sidecar identity. The returned plan
explicitly reports `weight_admission: not_checked` and
`image_execution: unimplemented`; it grants no runtime capability. The native
catalog now identifies the missing Stark implementation instead of describing
upstream GLM as lacking images.

The real retained metadata-only prepared configuration passed planning with
SHA256 `a09412710fc2b69eeb040ef57c1f358f74eef2a67f7b843c3c2379340d0bf7fe` and one
declared MTP layer. This folder has no weights; it is not a model-load result.

The first 1 MiB range of the pinned sidecar contains a complete 21,155-byte
header: 348 tensors, comprising 124 BF16 and 224 F32 tensors. Header metadata
reports `clip.projector_type: glm5v`, 24 blocks, 16 attention heads, width 1,024,
projection 4,096 and SwiGLU clamp 10. The private header inventory receipt has
SHA256 `bc3587f61392f25c5430229e784f9d987b12419d66ce022a9a80c7496c505eae`.
This checks declared inventory only; payload hashes, tensor spans against an
actual full file, numerical operators and execution remain unverified.

The sidecar's minimum/maximum pixel metadata is 12,544 / 6,272,000. Current
Transformers preprocessing expresses budgets through temporal token counts;
the conventions must be reconciled against pinned processor fixtures before
using either as the implementation reference. This unresolved source-version
question is not a numerical failure.

Validation: 21 focused source-plan/native-catalog tests passed with Python
3.13.13 and the existing trusted native template helper. An initial run had
20 passes and one existing native-install test failure because that helper was
absent from PATH; the corrected run passes. Repository policy and whitespace
checks pass. No Rust build, Metal test or full model run was performed.

Image execution, cache restoration with images resident, text performance,
MTP/image transitions and observed peak memory are **not run / not measured**.
The source planner is a foundation for preparation, not completed GLM image
support. Continue with the sidecar tensor contract and preprocessing/tower,
then embedding fusion and the complete qualification sequence above.

[config]: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/04c4e9e95c5da8862dced7e5056455116f83a7e0/config.json
[model]: https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py
[processor]: https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/image_processing_glm5_next.py
