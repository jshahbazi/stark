# Experimental GLM batched matrices

The native GLM prefill graph has three independent matrix experiments: wider
Q5/Q6 matrix tiles, grouped IQ expert routes and batched MLA projections. Each
is disabled unless its selector is `1`. They share decoded weights across
independent input rows while retaining the selected reference arithmetic.
Defaults require artifact-bound numerical, lifecycle, prefix-reuse and
whole-request performance qualification. This page describes implementation
and gates; dated results belong under `docs/archive/runs/`.

Use the existing [optimized recipe](../../scripts/with_glm53_optimized.sh) as
the reference and retain its context and numerical selectors. The
[wide-prefill reference](GLM_WIDE_PREFILL.md) describes chunk scheduling,
workspace admission and overlap ownership. Restart the worker when changing
selectors.

## Routes and admission

| Selector | Admitted matrices | Execution |
| --- | --- | --- |
| `STARK_GLM53_PREFILL_DENSE_WIDE` | Packed Q5_K/Q6_K, rank two, `[8192,4096]` or `[4096,8192]`, exactly 32/64/128 input rows, Apple M5 Max | Reuses decoded weight fragments across the complete input tile with F32 SIMD-group matrix operations. Other shapes and tails retain the existing T8/T4/scalar routes. |
| `STARK_GLM53_EXPERT_PREFILL_BATCH` | Packed IQ2_XS/IQ3_XXS expert banks with 2048/4096 columns and the existing signed-factor policy selected | Groups selected routes by expert, builds jobs containing up to four input rows, and dispatches the projections indirectly on the GPU. Gate, up and down must all be admitted before grouping begins. Unsupported formats, including IQ4_XS, retain the existing layer route. |
| `STARK_GLM53_MLA_PREFILL_BATCH` | Compatible Packed F32/F16/BF16/Q8_0 rank-two matrices and Q8_0 head matrices; selected rank-two Q5_K/Q6_K MLA shapes | Batches independent projections around chronological MLA cache updates. The same compatible matrix helper can also serve rank-two projections elsewhere in the prefill graph. |

Shapes above use the Rust loader's `[output,input]` order, or
`[heads,output,input]` for head matrices. A positive selector does not establish
that a matrix was admitted. Pipeline availability, selected arithmetic,
serial-encoder requirements, physical buffer spans, aliases and workspace
bounds remain authoritative.

Expert grouping admits 1–128 tokens, at most 64 selected experts per token and
at most 1024 total routes. It retains raw integer route IDs without a floating
point conversion. Jobs keep the original token/top-k slot, and projection
results return to that slot before the existing weighted mixture. Router
selection, ties and mixture order are unchanged; no CPU route readback is
introduced.

MLA admits 1–128 tokens. Generic float/Q8 projections reuse each decoded weight
across up to eight tokens. The selected Q8 `[16384,1536]` grouped schedule and
Q5/Q6 `[1536,4096]` / `[4096,16384]` schedules reuse weights across up to four.
The Q5/Q6 routes require their existing grouped selectors. Q8 matrices whose
selected segmented or block schedule has no matching batched implementation
decline to the original route. Rank-three head projections preserve
`[token,head,channel]` input order.

MLA input/query projections, row normalizations, absorbed key/value head
projections and the output projection are independent across tokens. For each
token, cache append still precedes sparse index selection and attention, and
attention completes before the next token is appended. No query observes a
future cache row. Cache growth, sparse selection and tie handling use the
existing attention implementation.

## Precision and ownership

Encoded weights keep their original GGUF formats and scales. These kernels
decode into F32 and use F32 activations, products, accumulation and temporary
outputs. Dense-wide uses F32 matrix fragments. Float storage in F16 or BF16
does not imply an activation cast. Existing KV, indexer and recurrent-state
representations remain unchanged.

The numerical reference is the already selected optimized kernel, including
its grouped reductions and extreme-input fallback. It can differ from a
separately selected generic F32 reduction. Dense-wide retains the reference
K partition, partial-sum order and fallback decision for each original group
of eight inputs. Grouped experts retain the selected signed-factor sums and
take the source vector-eight fallback only for the affected input row. MLA
retains the selected generic or grouped tree and its input-row fallback scope.
Nonfinite input or output sets the graph's sticky failure status.

All route lists, jobs, indirect arguments, activations and immutable weight
owners remain graph-owned until completion. Host validation checks logical
geometry and actual backing spans before binding. Individual activation and
output buffers retain the 8-million-F32-element bound; all allocations also
participate in the existing command-owner and global scratch budgets. Fewer
dispatches or weight decodes do not prove lower memory traffic or faster
execution; register spills, occupancy and activation loads require measurement.

These paths do not add a new prefix-cache representation. Wide tiles stop at
the existing absolute snapshot boundary, and all writers must drain before
capture, restore, reset or failure recovery. Qualify fresh and restored
execution with identical artifact, selectors, context, precision and loaded
model instance, following [model onboarding](../MODEL_ONBOARDING.md).

## Qualification and attribution

Run each format's operator tests against the selected old GPU kernel and
decoded F64 products. Cover basis/random inputs, tails, nonzero offsets,
physical span and alias rejection, nonfinite inputs, extreme-input fallback
scope, and queued-graph ownership. Expert tests additionally cover raw route
bits, invalid IDs, empty/singleton/mixed expert jobs and original output slots.
MLA tests cover head boundaries, row normalization and causal cache growth.
The source tests are `decode_graph_dense_wide_tests.rs`,
`grouped_expert_prefill_tests.rs`, `decode_graph_mla_tests.rs` and the private
fixture test in `glm_mla_prefill_tests.rs`, under
`crates/stark-engine/src/engine/hybrid/`.

Then compare full-vocabulary model readouts and generation against matching
fresh references, followed by reset/replay, cancellation/failure fencing and
prefix capture/restore across a complete snapshot boundary. Operator bit
equality is insufficient to qualify a different checkpoint, quantization,
device or selected combination. A new mixed-precision variant requires its
own declared precision policy and numerical/quality gates.

Confirm effective admission with aggregate counters:

| Counter | Meaning |
| --- | --- |
| `native.glm.prefill.dense_wide.q5` / `.q6` and `.tokens` | Actual wide Q5/Q6 dispatches and admitted token totals. |
| `native.glm.prefill.expert_group.routes` / `.projections` | Grouped route construction and actual grouped expert projections. |
| `native.glm.prefill.mla_batch.tokens` | Entry into the batched MLA orchestration. |
| `native.glm.prefill.token_projection.tokens` / `.weight_bytes` | Actual compatible matrix dispatches; these may include projections outside MLA. Encoded byte accounting is not a measurement of device traffic. |

Measure each selector separately before combinations, using the same artifact,
source-bound executables, optimized recipe, context, sampling and workload.
Serialize GPU work, alternate repeated reference/candidate runs and screen the
host for interference. Keep retained-graph GPU stage profiling separate from
production throughput measurements. Publish aggregate results and hashes only;
keep native model data and validation artifacts private and ignored.

## Source provenance

Stark's batching and host scheduling add no dependency. The existing
GGUF layouts, lookup tables and numerical schedules retain their source-bound
license notices in `LICENSE.ggml-hybrid` and the shader headers.

- Dense-wide derives from llama.cpp revision
  `dde879bc4f954b809892fd0d660515f42536394a`,
  `ggml/src/ggml-metal/kernels/mul_mv_mma.metal`, through Stark's existing
  `packed_kquant_mma_matvec.metal`.
- Q5/Q6 grouped arithmetic and the IQ2_XS/IQ3_XXS schedules derive from
  llama.cpp revision `d81235049384534c167caea52b85a694f6103d14`,
  `ggml/src/ggml-metal/kernels/mul_mv.metal`, through the attributed grouped
  kernels. The new cross-input loops retain those selected numerical policies.
- MLA orchestration and generic cross-token weight reuse are Stark additions
  around existing graph and attention operations; they do not replace the
  cache or sparse-attention algorithms.
