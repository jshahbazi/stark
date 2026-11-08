# llama.cpp MoE CPU-share applicability assessment

Date: 2026-11-07. Stark source reviewed:
`6698d2c1ba89a85110b976202e6f677a4ac1e9ef`, the existing Flash-Next experiment
branch tip. Assessment branch: `codex/llama-moe-cpu-share-assessment`.
This is a source review. No CPU-share backend, model benchmark or new
performance qualification was produced.

## Decision

The idea could be adapted to Stark's Qwen3.8-Flash-Next, GLM-5.3-Flash and
Recursive Flint MoE paths. There is no demonstrated benefit for a Stark
checkpoint. The strongest reason to investigate is a measured expert staging
or registration bottleneck in a large MoE path. Existing resident Flash-Next
execution offers a weaker case because it already avoids expert uploads.
Dense models have no routed expert-cache misses for this mechanism to handle.

Borrow the calibration approach first. A CPU/GPU expert split requires a
separate implementation and artifact-specific numerical and lifecycle checks.
The current evidence does not justify changing an inference default.

## Upstream behavior and provenance

[llama.cpp PR #30966](https://github.com/ggml-org/llama.cpp/pull/30966),
"llama_moe_cache: run missed experts on CPU, add tuning script", was merged
on 2026-11-07. Reviewed merge:
`e95426f16982054c379ca7bdc8186996dd22f0fe`; final PR head:
`c0b02cd6fcad1f441c457f692611ad93896056ae`.
No upstream implementation was copied or patched into Stark.

The existing llama.cpp cache holds a subset of host-stored MoE experts in
device memory. This PR lets the CPU evaluate some uncached experts after the
cache fills, while the GPU handles cache hits and remaining misses. Its
[documentation](https://github.com/ggml-org/llama.cpp/blob/e95426f16982054c379ca7bdc8186996dd22f0fe/docs/moe-cache.md)
relates the gain to CPU memory reads being faster than uploading experts.
The tuner measures several cache sizes and CPU shares, then models additional
settings. The PR's CUDA examples establish no Apple Silicon speedup, and its
predicted rates must remain distinct from measured rates.

The implementation is eligible for GPU and integrated-GPU backends; it is
not restricted to CUDA. It requires host-resident expert tensors and has batch
and scheduler constraints. See the
[cache constructor](https://github.com/ggml-org/llama.cpp/blob/e95426f16982054c379ca7bdc8186996dd22f0fe/src/llama-moe-cache.cpp#L401-L447)
and [CPU-part admission](https://github.com/ggml-org/llama.cpp/blob/e95426f16982054c379ca7bdc8186996dd22f0fe/src/llama-moe-cache.cpp#L687-L718).
Metal identifies as a GPU and uses a default 32-token offload threshold:
[device classification](https://github.com/ggml-org/llama.cpp/blob/e95426f16982054c379ca7bdc8186996dd22f0fe/ggml/src/ggml-metal/ggml-metal.cpp#L758-L760),
[offload behavior](https://github.com/ggml-org/llama.cpp/blob/e95426f16982054c379ca7bdc8186996dd22f0fe/ggml/src/ggml-metal/ggml-metal.cpp#L850-L855),
[threshold](https://github.com/ggml-org/llama.cpp/blob/e95426f16982054c379ca7bdc8186996dd22f0fe/ggml/src/ggml-metal/ggml-metal-device.m#L1285).
Metal decode eligibility is not performance evidence.

The [graph integration](https://github.com/ggml-org/llama.cpp/blob/e95426f16982054c379ca7bdc8186996dd22f0fe/src/llama-graph.cpp#L2594-L2645)
combines complementary CPU and GPU contributions for the selected experts.
It does not achieve its savings by dropping selected experts. Moving
arithmetic and changing reduction order can still change rounding. Other
graph restrictions, including LoRA and pre-FFN weighting, also prevent treating
this as a universal MoE implementation.

## Stark model applicability

| Stark path | Relevant behavior at the reviewed revision | Assessment |
| --- | --- | --- |
| Qwen3.8-Flash-Next, Lily and official mixed-affine native backend | GPU router and expert products; resident expert banks. The served-cache resolver is currently a no-op. | Architecturally applicable, low initial priority for the existing resident path. Adding CPU work would require exposing routes to the host, synchronization and compatible CPU quantized kernels. |
| Qwen3.8-Flash-Next, original native hybrid path | Full bank views when admitted; bounded selected mappings otherwise. | A fallback-path experiment is plausible if mapping or working-set costs are measured. There is no existing llama.cpp-style miss queue to reuse. |
| GLM-5.3-Flash, native GGUF | CPU expert selection followed by registration of selected immutable Metal slices. | Best initial native candidate for a bounded investigation: it already exposes selected experts on the CPU. Establish that registration or expert access is material before implementing a split. |
| GLM-5.3-Flash, optional EXL3/FP8 bridge | Separate pinned MLX backend with transition-weighted expert retention and direct reads. | Placement calibration is relevant. CPU execution needs its own packed-format kernel and scheduler work; the GGUF path's implementation and validation would not qualify this pack. |
| Recursive Flint INT4 | Routed/shared experts execute through mapped-weight Metal kernels; expert selection is on the CPU. | Structurally applicable. Profile its expert stage first; no cache-miss transfer bottleneck has been established. |
| Dense Qwen2/Qwen3, dense Qwen3.5, dense Qwen3.8-27B and other dense paths | No routed expert inventory. | This expert-cache mechanism provides no direct optimization. General CPU/GPU layer placement is separate work. |

Source anchors: [Lily resident policy](../../../vendor/lily-core/src/moe_ffn.rs#L481),
[cache resolver and GPU decode](../../../vendor/lily-core/src/moe_ffn.rs#L994),
[hybrid bank admission and GLM routing](../../NATIVE_HYBRID_MODELS.md#execution-and-limits),
[GLM EXL3 profile](../../validation/GLM_EXL3_BACKEND.md#retained-bf16-native-profile)
and [Flint execution](../../FLINT.md#execution-limits).
Other MoE architectures do not gain Stark support from this proposal; the
[converter's MoE admission](../../../tools/converter/stark_convert/model_support.py#L327)
remains in force.

## Why the memory hierarchy changes the case

Stark's native hybrid backend constructs Metal buffers directly over retained
immutable mappings using `STORAGE_MODE_SHARED` and a no-copy constructor:
[implementation](../../../crates/stark-engine/src/engine/hybrid/weight_metal.rs#L338).
Its [weight-view policy](../../NATIVE_HYBRID_MODELS.md#execution-and-limits)
already avoids repeatedly uploading compressed weights. Full-bank admission,
transient registrations, physical residency and paging remain separate
concerns; retaining a view does not pin every page.

Consequently, the discrete-GPU upload bottleneck targeted upstream cannot be
assumed here. A CPU split might avoid some Metal registration or staging work,
but would add coordination and compete for shared memory bandwidth and power.
It supplies no additional physical RAM. A CPU mapping and a GPU view of the
same weights do not solve insufficient total system memory.

## Bounded next experiment

1. Profile an unchanged GLM native GGUF worker on its pinned pack. Separate
   expert registration, expert GPU work, routing and wait time, and observe
   paging and memory pressure. Use only aggregate timings and counts.
2. Proceed with a CPU split only if that profile identifies a material cost it
   can remove. Add an opt-in path for one supported packed format and retain
   the existing GPU path as the reference. Account for CPU scratch and any
   duplicated storage. Measure a placement/share sweep with zero CPU share
   included; replay predictions require separate confirmation measurements.
3. Keep all selected experts and their mixture weights. Check decoded packed
   values, expert outputs and final readouts against matching references;
   retain greedy and seeded sampled output controls. Qualify the actual
   checkpoint, quantization and precision policy separately.
4. Follow [model onboarding](../../MODEL_ONBOARDING.md), including prefix
   restoration, MTP on/off where supported, cancellation, failed requests and
   restart invalidation. Preserve complete-request serialization and buffer
   ownership through both CPU and GPU completion.
5. Compare sealed workers on the same artifact and request settings, with
   fresh-request timings, controlled cadence and passing host screens. Report
   prefill, useful decode throughput, first-token latency, full request time
   and peak memory. Promote only a repeatable end-to-end benefit.

The upstream tuner retains server logs and token-associated expert-routing
traces, including selected expert IDs:
[tuner outputs](https://github.com/ggml-org/llama.cpp/blob/e95426f16982054c379ca7bdc8186996dd22f0fe/scripts/moe-cache-tune.py#L491-L505),
[trace encoding](https://github.com/ggml-org/llama.cpp/blob/e95426f16982054c379ca7bdc8186996dd22f0fe/src/llama-moe-cache.cpp#L642-L677).
These IDs are not token IDs or decision probabilities. Do not publish these
runtime artifacts or adopt the trace-producing tuner unchanged. A Stark
calibration harness should retain aggregate measurements and hashes, without
logging prompts, responses, token vectors or probabilities.

## Verification and limits

Two separate read-only investigations checked local model/storage paths and
the pinned upstream implementation. The repository policy check passed.
This report introduces no engine code and runs no model. Existing
[Flash-Next projection and cadence evidence](../../benchmarks/2026-11-07-flashnext-moe-projections.md)
remains bound to its original workers and protocols; it does not measure this
CPU-share proposal.
