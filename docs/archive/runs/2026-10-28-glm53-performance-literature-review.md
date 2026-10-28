# GLM-5.3-Flash native performance literature review

Date: 2026-10-28. Scope: GLM-5.3-Flash UD-Q2_K_XL on the Apple M5 Max with
128 GiB unified memory, executing in Stark's Rust/Metal engine.

This is a source and literature review. It ran no models or benchmarks and
copied no external implementation into Stark. The inspected checkout starts
at `8056d605d49e4876093206fb2d9fb18c1c799c25` plus the in-progress native GPU
changes. Its parent's preliminary full-worker measurement was 5.85 decode
tokens/second, compared with the preceding recorded 2.485. Final measurements
and qualification belong in the optimization benchmark record.

The most useful immediate work is faster Q5_K/Q6_K dense projection kernels,
followed by longer GPU execution chains. KDA fusion and absorbed MLA are already
present in the inspected Stark code. Repeating those architectural changes
would miss the remaining bottlenecks.

## Priorities for Stark

Impact and effort below are engineering judgments, not predicted speedups.

| Priority | Change | Why it matters here | Effort and acceptance |
| --- | --- | --- | --- |
| 1 | Tune Q5_K/Q6_K dense GEMV, then Q8_0 and the output head | The trunk's static Q5_K/Q6_K matrices total 5.105 GB; optimizing only the selected IQ experts leaves most static projection bytes on the older schedule | Medium. Measure actual shapes under sustained load; retain the existing numerical gates |
| 2 | Extend GPU chains through norms, router and MLA attention | Rust still reads intermediate arrays between major stages; a faster kernel cannot remove those CPU/GPU dependencies | High. Count submissions and readbacks per token, then compare complete fresh-state requests |
| 3 | Use bounded, granular residency policies | Registering entire expert banks changes the resources Metal must prepare. A near-limit full-bank policy can lose badly despite fitting an arithmetic budget | Medium. Compare bounded selected views, bank views and explicit residency separately, including first-request time and paging |
| 4 | Fuse expert gate/up/activation into a kernel and fuse weighted down reduction | The current FFN shares one command but still launches individual projections and materializes intermediates | Medium to high. Preserve routing order, clamping, shared-expert behavior and nonfinite rejection |
| 5 | Add chunked prefill with batched projections and expert grouping | Sequential token-by-token GEMV rereads weights for every prompt token | High. A separate prefill project; test every chunk boundary against recurrent prefix execution |
| 6 | Add native NextN speculation after a fast target path | The existing GGUF already contains a draft block; verification can amortize target weight reads | High. Requires correct KDA rollback, latent/pool-cache rollback and target-equivalent sampling |
| 7 | Move pooled-index scoring/top-k and indexed attention onto GPU for long contexts | These costs grow with history, while the current short benchmark is below the model's sparse selection budget | Medium to high. Add longer contexts before claiming a benefit |

## What the current checkpoint and code imply

The official model is a 320B-total, 18B-active model using hybrid linear and
sparse attention and mHC. The review uses the exact existing GGUF revision,
`unsloth/GLM-5.3-Flash-GGUF@a38483c8cd5df544f53d70fb281afe97369d5ab6`, and
the model definition at
`zai-org/GLM-5.3-Flash@04c4e9e95c5da8862dced7e5056455116f83a7e0`.
[Official model card](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/04c4e9e95c5da8862dced7e5056455116f83a7e0/README.md).

Independent header-only accounting reproduced these static trunk matrix sizes,
excluding the mHC projections and embedding/output tensors:

| GGML storage | Bytes |
| --- | ---: |
| Q5_K | 2,753,822,720 |
| Q6_K | 2,351,677,440 |
| Q8_0 | 536,444,928 |
| F32 | 203,970,560 |

These are source-storage bytes, not a measurement of DRAM transactions. The
parent's earlier profile found about 2.77 GB of selected expert ranges per
token. The inventory explains why speeding up IQ2_XS/IQ3_XXS alone does not
complete the projection work. Full rank-three source banks total
101,884,624,896 bytes, including the optional draft block's expert tensors.

The inspected `glm.rs` already has GPU KDA projection/recurrence/readout chains,
combined FFN submissions, and bounded decoded CPU mHC projections. MLA already
attends in the compressed latent space: it absorbs the query through K-B and
applies V-B after latent attention. Remaining CPU stages include router
sigmoid/top-k, latent normalization, pooled-index selection, attention softmax
and residual-stream handling. These are local code findings, not conclusions
borrowed from another engine.

## Kernel and execution evidence

### Packed dense projections

MLX's quantized kernels distribute output rows across SIMD groups and reuse
activation/quantization metadata. Their dispatch geometry is a useful baseline
for shape-specific tuning.
[MLX quantized kernels](https://github.com/ml-explore/mlx/blob/7d2d94a5561ebc3965884a0fc3dfc35b7b1bb17a/mlx/backend/metal/kernels/quantized.h).

The MIT-licensed mlx-kquant implementation is more directly relevant to GGUF
K-quants. Its Q5_K code uses packed-word loads and codec-specific fast GEMV;
Q6_K has a split-K matvec. Inspect those actual implementations before adding
more per-value decoder calls or applying one row schedule to every shape.
[Q5_K/Q6_K helpers](https://github.com/asher/mlx-kquant/blob/91b642ea0f779b0bbd559d9fbd3b9f403d5f195b/metal/mlx/backend/metal/kernels/kq_quantized_kquants.h),
[kernel registration and Q8_0 split-K](https://github.com/asher/mlx-kquant/blob/91b642ea0f779b0bbd559d9fbd3b9f403d5f195b/metal/mlx/backend/metal/kernels/kq_quantized.h).

Its authors report 450–465 GB/s for Q6_K/Q8_0 split-K at low sustained M5 Max
clocks, versus 385–395 GB/s for their per-row kernels. They also document shapes
where splitting regresses. These are component results on that implementation,
not Stark measurements or an end-to-end GLM target. The practical lesson is to
test steady-state clocks, row widths, output counts and batch geometry.
[mlx-kquant kernel measurements](https://github.com/asher/mlx-kquant/blob/91b642ea0f779b0bbd559d9fbd3b9f403d5f195b/docs/kernels.md).

### GPU chains and expert gathers

llama.cpp builds GLM's KDA, pooled indexer, absorbed MLA and MoE as graph
operations. Uzu's Metal command layer reuses command allocators and temporary
storage across longer operation chains. These supply execution and ownership
patterns; this review found no matched Uzu GLM throughput result.
[llama.cpp GLM graph](https://github.com/ggml-org/llama.cpp/blob/c02b5c641e23bc9c100f599466c23b46f8d5e2ef/src/models/glm5-next.cpp),
[Uzu command buffers](https://github.com/trymirai/uzu/blob/f1231705cc7e26a205819da7c78e0110cfbac306/crates/uzu-engine/src/backends/metal/command_buffer.rs).

mlx-kquant fuses routed gate/up and activation in one kernel, can include shared
expert slots, and supplies score-mixed down gathers. This removes intermediate
activation traffic and launches beyond merely sharing a command buffer.
[Fused K-quant MoE kernels](https://github.com/asher/mlx-kquant/blob/91b642ea0f779b0bbd559d9fbd3b9f403d5f195b/metal/mlx/backend/metal/kernels/kq_moe_glu_kq.h).

The gmlx GLM model keeps router logits in F32 because correction-biased
top-8-of-288 selection is sensitive to near ties; it uses a GPU sigmoid router
and GPU indexer score/top-k paths. Stark should retain its deterministic tie
and correction-bias contract when moving routing. Dynamic GPU expert selection
also requires a resource strategy that exposes the possible expert banks
without waiting for Rust to read the indices.
[MIT GLM model implementation](https://github.com/asher/gmlx/blob/9a09f9ed53beab6fabdbebd130e93f952d000f5d/gmlx/models/glm5_next/model.py).

Moving mHC back to a standalone GPU command would surrender the current tiny
CPU-projection latency gain. The useful later change is to keep the residual
streams on GPU and include mHC, normalization and the following block in the
same execution chain. The mHC paper explicitly treats memory traffic and kernel
fusion as infrastructure concerns.
[mHC paper, infrastructure section](https://arxiv.org/html/2512.24880v1#S4.SS3).

### Residency granularity

Apple documents no-copy buffers as wrappers around an existing allocation and
residency sets as groups of allocations Metal prepares for GPU access. A retained
buffer handle does not establish a measured physical-residency guarantee.
[No-copy buffer API](https://developer.apple.com/documentation/metal/mtldevice/makebuffer%28bytesnocopy%3Alength%3Aoptions%3Adeallocator%3A%29?language=objc),
[residency sets](https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets).

Current MLX splits its wired allocations into residency sets capped by default
at 5% of the recommended working set, with at most 32 sets. Its source explains
that residency decisions occur per set: splitting bounds the amount that must
be prepared again after residency is lost. It also records that macOS can drop
residency after GPU idle despite a standing request. This is a concrete policy
to study before constructing one set containing all of GLM.
[MLX residency policy](https://github.com/ml-explore/mlx/blob/7d2d94a5561ebc3965884a0fc3dfc35b7b1bb17a/mlx/backend/metal/resident.h),
[implementation](https://github.com/ml-explore/mlx/blob/7d2d94a5561ebc3965884a0fc3dfc35b7b1bb17a/mlx/backend/metal/resident.cpp).

MLX's wired limit defaults to zero and is separate from allocator/cache limits;
its allocator implements changes through the residency-set capacity. MLX-LM
temporarily sets a recommended wired limit for generation and flags models above
90% of the recommendation as potentially slow. The near-limit full-bank Stark
trial was still under investigation during this review; a nominal budget fit
does not qualify that configuration. Measure resource preparation, first-request
latency, page-ins/page-outs, memory pressure and sustained decode independently.
No global wired-limit or `sysctl` changes were made by this review.
[MLX wired-limit contract](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html),
[MLX allocator](https://github.com/ml-explore/mlx/blob/7d2d94a5561ebc3965884a0fc3dfc35b7b1bb17a/mlx/backend/metal/allocator.cpp),
[MLX-LM generation policy](https://github.com/ml-explore/mlx-lm/blob/1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7/mlx_lm/generate.py).

## Prefill, speculation and long context

Kimi Linear derives a chunkwise KDA algorithm and uses different prefill and
decode strategies. FLA supplies both chunk and fused-recurrent reference paths.
This supports a dedicated chunked prefill implementation, with batched weight
reads and expert grouping. It does not imply another large single-token gain
after Stark's fused recurrent KDA is already active. GLM's bounded decay gate
and GGUF-transformed decay tensor must remain exact model-specific inputs.
[Kimi Linear paper](https://arxiv.org/html/2510.26692v1),
[FLA KDA reference](https://github.com/fla-org/flash-linear-attention/blob/b0de6fc886300ef99b019591cd67abca3f77527b/fla/ops/kda/fused_recurrent.py).

Apple's M5 TensorOps support custom dequantization within matrix kernels and
matrix tiling. These units are promising for chunked prefill and multi-token
verification, where weights are reused across several activation rows. New
BF16/FP16 staging or matrix accumulation changes require separate numerical
qualification; a one-row memory-bound decode kernel need not improve simply
because it uses matrix hardware.
[Apple M5 machine-learning talk](https://developer.apple.com/videos/play/tech-talks/111432/).

Header inspection confirms an optional block 45 of 2,787,995,776 bytes, including
`nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm` and `nextn.shared_head_norm`.
The gmlx MIT MTP module documents collapsed trunk hidden state, a plain-residual
draft DSA block, latent/pool trimming and KDA replay for rollback. Its own
comments leave a hidden-state feed convention experimentally qualified, so it
is an implementation reference rather than an independent correctness oracle.
[GLM native MTP implementation](https://github.com/asher/gmlx/blob/9a09f9ed53beab6fabdbebd130e93f952d000f5d/gmlx/models/glm5_next/mtp.py).

Speculative decoding can preserve the target distribution through verification;
its benefit depends on draft acceptance and the cost of verifying multiple
rows. Add greedy equivalence and seeded-distribution tests before reporting
accepted target tokens per second. The existing immediately-consecutive score
cache policy does not authorize persistent prompt/KV caches.
[Speculative decoding paper](https://arxiv.org/abs/2211.17192v2),
[DeepSeek-V3 MTP design](https://arxiv.org/html/2412.19437v2#S2.SS2).

The current checkpoint selects up to 2,048 token positions through pools of
four. A short request below that budget does not stress sparse pruning. GPU
index scoring, radix top-k and indexed online-softmax attention should be
evaluated at increasing context depths; they should not be credited for gains
on the present roughly 46-token GLM benchmark. Official DSA work provides the
long-context mechanism and kernels, with architecture differences that require
checking against GLM's pooled NoPE contract.
[DeepSeek sparse-attention reference](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/tree/87e509a2e5a100d221c97df52c6e8be7835f0057),
[Metal pooled-index and sparse-attention kernels](https://github.com/asher/mlx-kquant/blob/91b642ea0f779b0bbd559d9fbd3b9f403d5f195b/metal/mlx/backend/metal/kernels/kq_dsa_sparse_attn.h).

## Published throughput and licensing boundaries

gmlx's 2026-09-02 changelog reports UD-Q2_K_XL on M5 Max increasing from 40.3 to
42.5 tok/s at depth zero and 36.0 to 37.2 at depth 4,096 after KDA/router/gather
work. The entry does not supply the complete checkpoint pin and benchmark
recipe or explicitly state speculation for those particular figures. Treat it
as author-reported evidence that a substantial gap exists; it is not a matched
Stark comparison or a promised result.
[gmlx changelog](https://github.com/asher/gmlx/blob/9a09f9ed53beab6fabdbebd130e93f952d000f5d/CHANGELOG.md#048---2026-09-02).

Sushi reports GLM 2.4-bpw at 71.5 tok/s at 4K and 63.8 at 128K. Its GLM usage
defaults include DFlash2 and 8-bit KV, and the weight format differs from this
GGUF. Those numbers cannot establish plain Stark decode throughput. Its
external drafter also carries a separate weight license.
[Sushi benchmark](https://github.com/beamivalice/sushi/blob/956fe27f4e19ccefbd4f6bddf7c32eb67d8cc71c/README.md#benchmarks),
[GLM launch defaults](https://github.com/beamivalice/sushi/blob/956fe27f4e19ccefbd4f6bddf7c32eb67d8cc71c/docs/usage.md).

| Source snapshot | Code-use boundary |
| --- | --- |
| mlx-kquant `91b642ea0f779b0bbd559d9fbd3b9f403d5f195b` | MIT; retain its notices and the included MLX/ggml attribution for derived kernels |
| gmlx `9a09f9ed53beab6fabdbebd130e93f952d000f5d` | Root is BUSL-1.1; the reviewed GLM `model.py` and `mtp.py` have explicit MIT headers. Imports and other files require their own review |
| MLX `7d2d94a5561ebc3965884a0fc3dfc35b7b1bb17a` | MIT; retain Apple notices |
| Uzu `f1231705cc7e26a205819da7c78e0110cfbac306` | MIT; Stark's inherited base remains separately recorded as Uzu 0.6.1 at `8d01581474760bd1fc7f4a547ff1d02f7dada437` |
| llama.cpp `c02b5c641e23bc9c100f599466c23b46f8d5e2ef` | MIT; this reviewed graph revision does not replace Stark's pinned GGML decoder provenance |
| mlx-serve `e1d2013c1a321ba9e04324d790254e3f5428265c` and Sushi `956fe27f4e19ccefbd4f6bddf7c32eb67d8cc71c` | MIT project code with additional notices for ported/vendor code; weights and assistants have separate licenses |

[mlx-kquant license](https://github.com/asher/mlx-kquant/blob/91b642ea0f779b0bbd559d9fbd3b9f403d5f195b/LICENSE),
[gmlx license and file-level exception](https://github.com/asher/gmlx/blob/9a09f9ed53beab6fabdbebd130e93f952d000f5d/LICENSE).

## Experimental decision rules

Each candidate needs an immutable worker hash and an interleaved baseline on
the exact checkpoint. Keep the numerical source gates unchanged; report greedy
agreement, error aggregates and reset/failure behavior. Intermediate route
comparisons can run privately and publish only aggregate agreement. Preserve
F32 controls and recurrent state while establishing the kernel/scheduling
baseline. BF16 activation staging, FMA changes, sigmoid lookup tables, KV
quantization and speculative decoding are separate experiments.

Record startup and first-request costs separately from warm decode; use long
enough runs to expose clocks, resource warming and expert diversity. Serialize
GPU tests, builds and measured inference, and report page activity alongside
memory-sensitive experiments. Neither process RSS nor registered buffer bytes
is total model memory. These checks are essential to deciding which literature
ideas actually improve this native engine.
