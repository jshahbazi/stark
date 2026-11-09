# 20261109-GLMMEM-14: upstream Metal memory source review

State: pinned source research only. No new build, Metal execution, model load,
admission change or performance qualification. The actual compression failure
in [GLMVIS-10](2026-11-09-glm-functional-pageout-screen.md) remains unattributed.

## Source identities and licensing

| Project | Revision | Scope |
| --- | --- | --- |
| llama.cpp / ggml | `1b6935b227b9f9aa89deea8016c0e208f547446f` | Metal device, model range selection, model loader, POSIX mapping |
| MLX 0.32.4 | `36969c094354086e15d2a327a9c50feb20b9cf11` | Metal allocator, residency sets, library/pipeline cache |

Both revisions carry MIT licenses. The saved license hashes are
`94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d`
(ggml authors, 2023–2026) and
`ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01`
(Apple, 2023). No upstream code is ported by this record. Any later port must
preserve the corresponding copyright, license and exact patch provenance.
Private downloaded source copies, URL/file hashes and the findings receipt stay
ignored under `runtime/glm-image-source-2026-11-09/oss-pressure-research14/`.
The receipt hash is
`3e4cd9076f1b4e89bdb302b57cf33f8604e22dfb4f9aa85e94e18509bc11d6e6`.

## What the sources establish

llama.cpp's `llama_model::load_tensors` deliberately restricts Metal host-pointer
buffers to ranges containing the current context's weights. The source explains
that a Metal backend buffer pins the range it wraps. `get_mapping_ranges`
excludes weights belonging to other contexts, including lazy/CPU placements.
This is a reason to account the complete supplied range, even when a shader
reads only selected rows. Its POSIX mapping is read-only `MAP_SHARED`; lazy
ranges are excluded from eager prefetch. These are facts about that revision,
not proof that changing Stark's `MAP_PRIVATE` fixes its failure.
([model loading](https://github.com/ggml-org/llama.cpp/blob/1b6935b227b9f9aa89deea8016c0e208f547446f/src/llama-model.cpp),
[range selection](https://github.com/ggml-org/llama.cpp/blob/1b6935b227b9f9aa89deea8016c0e208f547446f/src/llama-model-loader.cpp),
[mapping](https://github.com/ggml-org/llama.cpp/blob/1b6935b227b9f9aa89deea8016c0e208f547446f/src/llama-mmap.cpp))

In `ggml_metal_buffer_map`, NoCopy buffers cover the page-rounded supplied
range; oversized mappings use overlapping views. Optional residency sets include
the buffers. Cleanup removes the set from its heartbeat collection, releases
buffers, ends residency, removes allocations and commits the set. A heartbeat
renews residency during an activity window. A source workaround submits tiny
GPU work for a reported residency-release problem. None of these operations
establishes an immediate total physical-memory ceiling or authorizes adding
residency to Stark's bounded recipe.
([Metal ownership and residency](https://github.com/ggml-org/llama.cpp/blob/1b6935b227b9f9aa89deea8016c0e208f547446f/ggml/src/ggml-metal/ggml-metal-device.m))

MLX separates live allocation, buffer cache and residency accounting. Its
allocator can recycle a freed buffer into the cache; only actual buffer release
erases the residency allocation. `ResidencySets` tracks inserted bytes, leaves
excess allocations outside its capacity, and commits removals. This is not a
measurement of the host's total wired memory. The observed Stark difference
between zero wrapper counts and retained task footprint therefore remains a
required telemetry distinction.
([allocator](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/allocator.cpp),
[residency](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/resident.cpp))

Both projects cache compute pipelines on demand. llama.cpp separately compiles
all per-kind source libraries in parallel; it is incorrect to call its complete
initialization lazy. MLX loads its default library and caches additional
libraries/kernels by key. Neither source demonstrates the cost of Stark's
different Safe-math source or compiler process peak.
([ggml pipeline/library creation](https://github.com/ggml-org/llama.cpp/blob/1b6935b227b9f9aa89deea8016c0e208f547446f/ggml/src/ggml-metal/ggml-metal-device.m),
[MLX library/kernel cache](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/device.cpp))

## Next bounded discriminator

Retain the exact [first-use proposal](2026-11-09-glm-first-use-pressure-audit.md):
one fixed, ignored Control→Q5_K native control, with no model or mapped weights.
Record library/PSO encoding before command submission and first completion
separately. Observe host counters, own RSS/footprint/Metal bytes and all compiler
service RSS continuously through drain. Preserve the proposed narrow hard
limits and obtain independent supervisor/source review before execution.

A pressure increase during no-file initialization would isolate a useful stage.
A passing result would leave GGUF-backed GPU views, later formats and subsequent
operators unresolved. Only then consider a separately registered file-size or
range contrast. No source fact here attributes the prior compression event.

Narrow selected-expert NoCopy views are a possible later design question:
current bounded `encode_indexed_matvec` registers a complete expert bank, while
the GPU router chooses indices inside the same command graph. Selecting smaller
host views would require an explicit routing/synchronization design and matching
arithmetic and speed controls. It is not an implemented remedy. Re-uploading all
99 GB of expert banks per decode token remains reference-only/nonpromotable.
No global wired-memory setting, generic admission divisor or production default
changes are proposed.
