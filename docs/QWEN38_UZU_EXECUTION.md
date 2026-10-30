# Qwen3.8 standard Uzu execution

## Current scope

The source implements the complete Qwen3.8-Flash-Next text architecture through
`Engine::load_language_model` and the standard `LanguageModel` generation and
scoring interfaces. Work is paused as an experimental branch checkpoint at the
user's request; the optimized existing executor remains the preferred path.
See the [checkpoint record](archive/runs/2026-10-30-qwen38-uzu-paused-checkpoint.md)
for measured performance, completed checks and outstanding qualification.
This page describes implemented behavior;
it does not claim parity, a throughput improvement or completion of the
[acceptance plan](QWEN38_UZU_EXECUTION_PLAN.md).

The supported input is a self-contained canonical Uzu text artifact with
`Qwen38LanguageModelConfig`. Raw `model_type=qwen4_exp` directories return
`model_conversion_required`. The implemented converter accepts the pinned
Lily-format affine checkpoint as source data. Other source packing, including
the PipeNetwork mixed export, requires a compatible conversion implementation.
Source provenance does not select an inference executor.

## Storage and matrix execution

`ParameterLoader` checks tensor spans, shapes and dtypes. Qwen admission checks
the schema-derived tensor inventory, explicit bindings, MLX specifications,
quantization geometry, zero affine padding and exact optional integer hash
controls. All required planes must be consumed before publishing a model.

Resident matrices use ordinary `WeightMatrix` storage and
`LinearMatmul`/`MatmulKernel` dispatch. Packed codes and BF16 group-major
scales/biases retain their canonical Uzu layout. Expert banks remain flattened
complete banks; GPU-selected global rows address their original quantization
parameters. Selected down-projection inputs are dispatched in groups of at
most eight rows. Weight quantization is unchanged. Loading performs no full-bank
dequantization or transpose into the former executor's parameter layout.

Weights and recurrent/cache buffers use normal backend allocation. Temporary
values use the standard scratch pools and command-buffer constants. The normal
backend owns encoding, dependency handling, submission, completion and resource
lifetimes. There is no production Lily model, Tensor, ComputePass, allocator or
command queue dependency.

## Complete layer graph

The decoder embeds each known input token, broadcasts it to the HC residual
streams, executes every configured layer, collapses the final HC streams and
projects the vocabulary readout. For the target artifact this is the complete
48-layer graph.

| Block | Execution |
| --- | --- |
| HC | Grouped normalization, standard down/up/injection projections, scaled SiLU, stream mixing and residual injection |
| GDN | Standard projections, causal depthwise convolution, normalized Q/K, F32 decayed-state delta update, gated normalization and output projection |
| QSA | Standard Q/K/V/indexer/output projections, rotary preparation, Q8 cache writes, pooled index keys, causal block selection plus the incomplete tail, split attention and output gating |
| MoE | F32 router, top-10 of 512 experts for the target model, standard gathered full-bank projections, shared expert, activation and ordered combination |
| PLE | Exact unsigned wrapping hash, bounded paged n-gram rows, standard projections, grouped norms/gating and dilated causal convolution |

Custom architecture operations are registered through the existing Uzu kernel
DSL, with CPU references for controls. Production Metal execution has no CPU
projection, expert-routing, attention or recurrent-update fallback.

## Precision and sampling

| Value | Policy |
| --- | --- |
| Weight codes and affine parameters | Source 4/8-bit codes and BF16 parameters preserved |
| Activations and projection outputs | BF16, except F32 router/shared-router and vocabulary readouts |
| Normalization and reduction arithmetic | F32 |
| Split-K matrix partial sums | F32 until the final output cast |
| GDN recurrent state | F32 |
| Convolution windows and indexer keys | BF16 |
| Attention K/V | Signed Q8 codes with F16 scales per 32 values |
| Attention statistics and partial sums | F32 |
| Floating controls | F32 preserved; BF16 widened exactly to F32 |
| Integer PLE controls | Exact raw 64-bit values |

The former executor used different precision policies for decode and prefill.
The standard Uzu path has these explicit differences from the sealed Lily
reference:

| Block | Precision difference |
| --- | --- |
| HC | Grouped normalization rounds to BF16 before down/injection projections. Lily's fused decode and batches of at most four retained that normalized projection input in F32; its larger prefill batches already materialized BF16. Its final prompt readout also used fused HC. |
| GDN prefill | Each causal step retains normalized Q/K and recurrent arithmetic in F32. Lily prefill staged normalized Q/K in BF16 and could use chunked kernels with additional BF16 intermediates. Both keep recurrent state in F32 and round the raw recurrent output to BF16 before gated normalization. |
| MoE prefill | Router and shared-router outputs remain F32, and the routed sum stays F32 through the shared-expert addition before one BF16 cast. Lily prefill rounded those router outputs and the intermediate routed sum to BF16; its decode already used the F32 policy. |
| Attention prefill | Split attention dequantizes Q8 cache values in F32 and retains F32 scores/partials. Lily tensor prefill rounded dequantized K/V and attention probabilities to BF16. Both retain Q8/F16 cache storage and BF16 attention output before gating. |

Standard quantized GEMM stages dequantized weights into BF16, while GEMV
retains F32 affine arithmetic. Matrix dispatch geometry and reduction order
also differ. Independent projection controls account for these branches.
Split-K partial sums remain F32; storing BF16 partials before reduction caused
an additional cancellation error. The
[failed precision record](archive/runs/2026-10-30-qwen38-split-k-precision.md)
preserves the baseline and attribution; the separate
[correction validation record](archive/runs/2026-10-30-qwen38-split-k-correction-validation.md)
records bounded checks and their limits.

Full-model numerical qualification remains pending, including the planned
full-vocabulary comparisons at 35 Qwen checkpoints and three ordinary-model
checkpoints. Those fixed-prefix comparisons measure numerical and distribution
drift; they do not establish general model accuracy or a quality threshold.

Qwen uses the normal backend sampler configured for F32 logits and its
`PRng` state. The removed CPU nucleus shortcut and SplitMix64 stream are not
used. Previous token-sequence hashes cannot establish parity for this sampler.
Forward comparisons must use matching fixed continuations and full readouts;
sampled generation requires its own deterministic-repeat and distribution
controls.

## State, failures and limits

Each request owns GDN/convolution state, QSA caches/indexer keys and PLE
history. Flat suffixes encode in causal token order. Acceptance must cover the
entire suffix in order; tree/subset acceptance fails explicitly. Partial
encoding failures poison the state. Submission, completion and early-drop
failures must fence reuse through the standard language-model lifecycle.

Paged PLE reads only the selected code/scale/bias rows from validated file
regions. Hashing receives known host token IDs. The stream resolves a sampled
GPU token before encoding its next PLE lookup; it never reads unfinished GPU
token storage on the CPU.

The implemented context cap is 65,536 tokens, bounded by the artifact's declared
maximum. Generation and scoring grant this ceiling only to the actually loaded
Qwen decoder; other model families retain their 16,384-token worker limit.
Desktop Chat/API and Decision mode use the worker's advertised ceiling. Old
scoring workers without that field retain the 16,384-token client fallback.
64K runtime and numerical qualification remains pending; the
[64K admission record](archive/runs/2026-10-30-qwen38-64k-admission.md) separates
implemented limits from unexecuted checks. A 65,536-token generation context
can contain 65,280 prompt tokens plus 256 sampled outputs; a full 65,536-token
prompt plus 256 outputs exceeds this capacity.

Prefill runs layer by layer in chunks
of at most 16 to bound command/scratch growth. Standard projections batch rows;
GDN recurrence and sparse-cache updates retain causal token order. Expert
gathers split their input rows into groups of at most eight. Only requested
rows reach the final HC/readout. Performance qualification is in progress.
Text input only is implemented; vision, MTP, speculation
and tree verification are disabled. The router currently requires
`norm_topk_prob=true`; other routing policy is rejected explicitly.

## Evidence and provenance

The [execution plan](QWEN38_UZU_EXECUTION_PLAN.md) tracks required admission,
state, whole-model, sampled-generation, scoring and independent-audit gates.
The [provenance record](QWEN38_UZU_PROVENANCE.md) records adapted equations,
source revisions and licenses. Dated studies retain the executor, source and
precision scope they measured. New runtime results must be recorded separately.
