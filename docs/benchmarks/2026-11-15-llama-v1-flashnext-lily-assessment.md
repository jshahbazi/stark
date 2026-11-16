# llama.cpp 1.0.0 opportunities and Flash Next Lily baseline

Date: 2026-11-15. Attempts: `20261115-LCPP1-01`, `-02`, `-03`.

## Current reference update

The release audit and image-resident baseline below retain their recorded
scope. The later sampling-specific one-draft policy was reverted at
`2ac1b713`; enabled Lily requests again recommend two drafts. Its held-out
top-k endpoint had reduced median request latency by 14.66% with 8/8 paired
wins, and seven prefix checks matched fresh outputs. The subsequent natural
coding counterexample and planning tie prevented that result from supporting
the automatic policy. The [historical policy record](2026-11-15-lily-sampling-mtp-policy.md)
and [restoration record](../archive/runs/2026-11-15-lily-mtp-two-draft-default-restored.md)
preserve that chronology and evidence.

The rank-one verifier investigation has now completed a separate
[two-draft long-response study](2026-11-15-lily-small-row-verifier.md).
Packed-weight reuse in remaining generic projections passed 60 component
cases, but improved paired useful decode only 1.84% on coding and 4.54% on
planning, below the 10% practical minimum. The candidate is archived and
rejected for promotion. Production code and the installed app are unchanged;
no llama.cpp optimization is represented as adopted.

A later [very-long-context public benchmark case](2026-11-15-lily-long-context-infinitebench.md)
retains the same candidate and two drafts on 88,398 input tokens. It observes
+5.40% paired useful decode (8/8 wins) and −2.39% request time (7/8), with
identical outputs/work. This is additional evidence for a modest decode gain
under accepted host load, separately recorded without a production promotion.

On 2026-11-16 the user accepted that modest benefit and authorized
[adoption](../model-experiments/2026-11-16-lily-projection-reuse.md).
Group-64 verifier projection reuse is now the normal branch path after final
exactness and lifecycle controls, including the full original-prompt/book
replays. The earlier study decisions and source-bound data remain intact.
Two drafts remain recommended; the installed app is unchanged.

## Engineering judgment

The release is worth using as a source of narrowly scoped experiments. The
source audit does not support a wholesale backend change or an expectation of
a large automatic speedup for this Lily artifact. No optimization is adopted
by this study. The before/after comparison target is Stark against Stark using
the identical Lily payload, rather than a different GGUF quantization running
in llama.cpp.

The first assessment needed two material corrections:

1. The actual retained artifact executes `Forward::Lily`, including vendored
   `qwen4exp` kernels. Findings in Stark's separate `qwen_graph` implementation
   cannot establish coverage or missing work in Lily.
2. Lily already selects Q8 KV for this artifact. The nominal F16-to-FP8 or
   F32-to-FP8 capacity ratios do not describe the baseline. FP8 would replace
   an existing eight-bit representation and is a poor first performance bet.

The fresh baseline contains 128 measured requests under existing system load.
No positive MTP setting clears the declared practical gain criterion. Two
settings show clear slowdowns: three drafts at 4K input with top-k 20, and two
drafts at 512 input with nucleus-only sampling. This establishes a useful
before baseline and configuration cautions; it does not measure a ported
llama.cpp optimization.

Among the release changes relevant to this path, the clearest reported
throughput results are in CUDA few-row matrix kernels.
Their workload matches the idea of speculative verification, but their
instruction set, quantization and hardware differ. Lily already has one-to-four
row specializations. Whether a new Metal variant can beat them is an untested
question. The justified next step is a bounded bottleneck measurement with a
stop rule, rather than implementing a collection of speculative ports.

## Scope, revisions and evidence strength

Primary upstream boundaries:

- [llama.cpp 1.0.0](https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0),
  release commit `b1b770d`, 2026-11-13; its release notes compare with v0.9.0.
- [ggml 1.0.0](https://github.com/ggml-org/ggml/releases/tag/v1.0.0),
  release commit `417a024`, 2026-11-13; the standalone notes compare with v0.29.0.
  A standalone ggml release delta is not necessarily the same as the vendored
  ggml delta between two llama.cpp tags. The candidate list below records ideas
  without treating every standalone entry as newly introduced in llama.cpp.
- [llama.cpp 0.9.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.9.0)
  already included earlier MTP/device-copy and recurrent-tail work. In particular,
  #31506, #31443 and #30141 must not be counted as new 1.0 features.

This inventory covers inference throughput, initialization/host overhead,
memory/capacity, state movement, scheduling, precision and closely related
correctness changes in the release notes and inspected linked changes.
New architectures, audio functionality, UI, server features and non-Metal
backends are screened explicitly below. This is a performance assessment, not a
security audit or a claim of exhaustive source verification of every commit.

Evidence labels used here:

| Label | Meaning |
| --- | --- |
| Measured upstream | Author/reviewer supplied a workload and results; not independently reproduced here |
| Reported upstream | A performance statement without enough conditions or raw A/B data to assess it |
| Source overlap | Current reachable Lily code implements a related mechanism; does not establish equal performance |
| Hypothesis | A proposed measurement or implementation with no new Stark candidate result |
| Local baseline | Fresh execution of the exact artifact/build; busy-host conditions retained |

## What actually runs for the selected artifact

| Area | Current reachable implementation and consequence |
| --- | --- |
| Backend selection | Exact Lily/affine metadata makes `hybrid/mod.rs` construct `Forward::Lily`. The separate `Forward::Qwen` route is not the execution evidence for this artifact. |
| Vision and MTP loading | The image staging option selects `load_with_image_support(path, true)`, retaining the tower and MTP head. Text requests can select zero through three drafts; image requests disable MTP and invalidate text reuse. |
| Precision | Source weight quantization is preserved; BF16 activations, F32 reductions/recurrent state/logit readouts, and Q8 attention KV. Default affine weight groups are Q4/G64 with a separate Q4/G32 ngram policy and source-declared mixed tensors. This is not a uniform four-bit arithmetic engine. |
| Expert placement | Full packed expert stacks have device buffers. `slot_of` is `None`; the model's expert-cache statistics return `None` and cache-position setter is a no-op. There is no reached host-expert/cache-miss CPU split. This is a logical placement description, not proof that every mapped page is physically resident throughout a busy-host run. |
| Ordinary decode | Concurrent Metal4 encoding and explicit dependency barriers; Q4 projection/convolution and recurrence fusion; direct packed quantized GEMV. |
| Batched verification | Host bridge uses `verify_host_batched`, with at most four target rows for three drafts. Q4/Q8 small-row projection kernels and small-M expert kernels already exist. |
| Sampling | The regular Hybrid MTP branch calls host exact-rejection sampling. Each draft step waits and reads a complete F32 vocabulary row; target verification returns F32 rows for host acceptance/residual sampling. The separate core GPU speculative API is not proof that this generation branch samples entirely on GPU. |
| Target hidden state | MTP hidden states remain device-resident; the first draft reads `mst.hidden` as a device tensor and later draft states stay on GPU. The host bridge reads complete F32 logits for sampling. Device hidden residency itself is already present. |
| Reuse | One previous successful request can restore chunk-aligned recurrent state while retaining attention prefix rows in the same worker. Snapshot copying is not a general multi-slot checkpoint cache. This baseline deliberately exercises fresh requests with zero reused input tokens. |
| Prefill | 4,096-token logical chunks, reusable capacity, and different serial/concurrent encoding selections by row count. Multi-chunk paged PLE can double-stage ngrams when the ahead-staging path and alternate buffer are available. More concurrent encoding is not automatically faster. |

Owning source: [backend selection](../../crates/stark-engine/src/engine/hybrid/mod.rs),
[Lily adapter](../../crates/stark-engine/src/engine/hybrid/lily.rs),
[host MTP bridge](../../vendor/lily-core/src/qwen4exp/host_spec.rs),
[model scheduling/state](../../vendor/lily-core/src/qwen4exp/model.rs),
[packed quantization kernels](../../vendor/lily-core/src/kernels/quant.rs),
[expert kernels](../../vendor/lily-core/src/kernels/moe.rs),
[KV formats](../../vendor/lily-core/src/kernels/attention.rs),
[expert weights](../../vendor/lily-core/src/qwen4exp/weights.rs), and
[generation dispatch](../../crates/stark-generate/src/hybrid.rs).
These links describe the baseline source named in the identity section;
later repository changes require a fresh inspection.

## Ranked investigation decisions

Ranks describe expected engineering value for this exact artifact and the
user's noticeable-improvement requirement. They are not measured speedup
predictions. Cost estimates are engineering judgments and exclude full
qualification across other models.

| Rank | Candidate | Likelihood of a noticeable win | Judgment and bounded next check |
| ---: | --- | --- | --- |
| 1 | Small-row target verification, packed projection reuse and dispatch selection | Low/unknown; highest-ranked diagnostic, substantial existing Lily overlap | Worth one profiling budget only if it exposes a material poorly utilized stage. Measure the reached 2–4 row verifier, not ordinary decode. A new Metal kernel is a significant implementation/validation cost. Stop if the achievable whole-request ceiling is below the noise hurdle. |
| 2 | Existing MTP draft-depth choice, rather than a new release port | Historical potential in some opt-in top-k workloads; inexpensive to measure | Worth the fixed baseline matrix. Older matched runs showed meaningful benefit in some top-k workloads and little benefit in nucleus-only sampling. Preserve opt-in semantics; do not infer an adaptive policy or default change. |
| 3 | MTP synchronization and full-logit host readout/sampling | Low chance of clearing the practical hurdle; high correctness cost | Defer implementation. Upstream #31927's target-hidden residency is already represented in Lily. Moving complete exact sampling/acceptance to GPU is a separate larger project. Reconsider only with a measured critical-path overhead budget and exact distribution/RNG/residual controls. |
| 4 | Coalesce recurrent MTP snapshot/rollback copies | Low for Flash Next; upstream Flash Next showed no gain | Structural audit is complete. No further action unless a fresh profile exposes material copy cost. Separate per-round GPU rollback captures from request-prefix snapshots. The latter are outside this fresh-request baseline. |
| 5 | Prefill staging, unused-input suppression and reserve reuse | Low; existing staging and capacity reuse weaken the case | Inspect only if the 4K prefill baseline exposes a user-visible cost. Model scratch and the Metal allocator pool persist; prefill scratch grows with capacity. Earlier sampler-buffer reuse regressed slightly. No standalone allocator project is worthwhile without a trace showing exposed cost. |
| 6 | Shared multimodal scratch | Low for this one vision tower; possible capacity value in broader multimodal models | Defer unless peak allocation accounting shows duplicate simultaneously retained workspaces. #31945's vision/audio/generation subcontexts differ from this model. Keep the supported image tower loaded. |
| 7 | Metal small-K dispatch for K=32–63 | Very low without a matching shape | Reject as an implementation task until an actual reached matrix shape and current dispatch prove the same bottleneck. LoRA adapter rank and quantization group size are different dimensions. |
| 8 | More MoE routing/GDN fusion or changed concurrent cutoff | Very low incremental confidence; prior negative trials | Do not repeat mechanically. Existing fused routes and archived same-artifact failures require a new diagnosed cause before another candidate. |
| 9 | FP8 KV | Very low speed/capacity value versus existing Q8; high precision cost | Defer for this artifact. It changes representation at roughly the same bit width and needs custom Metal kernels, scaling and long-context numerical qualification. |
| 10 | Host expert cache, CPU miss execution, ClassLRU/LRFU | Very low for the reached full-bank path; very high architecture cost | Reject for this performance study. Reconsider only for a separately scoped model-capacity objective with actual memory pressure and placement measurements. |

Existing configuration selection is a separate, inexpensive opportunity:
measure draft depths one, two and three against the default MTP-off route.
That is included in the baseline matrix. It can reveal a useful existing option
without pretending that MTP was introduced in llama.cpp 1.0. No automatic draft
controller or default change is justified by one synthetic workload.

At registration, only rank 1 warranted a new optimization-specific diagnostic;
its bounded [follow-up](2026-11-15-lily-small-row-verifier.md) is now complete
and its tested projection candidate failed the practical performance gate.
Rank 2 is already covered by the draft-depth studies. The other rows are a documented
backlog of checks with explicit reasons to stop or defer. No row currently has
local evidence supporting an expected 10% implementation gain.

Historical profiling further lowers the priority of a GPU sampler rewrite:
the earlier MTP report attributed about 1.1% of its wall profile to target
sampling, 9.5% to drafting including GPU work/waits, and 86.8% to verification.
Those are source-bound historical figures, not today's profile. Eliminating
only the target sampler could not meet the current threshold. Measure a new
critical-path budget before revisiting it. Prefix capture was previously
coalesced into one pass for 74 recurrent tensors; its reported 7.3 ms is not a
general decode bottleneck. See the
[MTP optimization report](QWEN38_MTP_OPTIMIZATION_2026_10_30.md) and history.

Use an Amdahl bound before a kernel project. If a region occupies fraction `f`
of request time and can be sped up by factor `s`, the maximum whole-request
speedup under that model is `1 / (1 - f + f / s)`. Even eliminating a 5% region
cannot provide a 10% throughput gain. Halving a region requires roughly 18.2%
of total time to reach 10% throughput gain, or 20% to reduce total latency 10%.
Overlapping GPU/CPU spans cannot be added as independent fractions; measure the
critical path and use optimistic bounds only to reject low-ceiling ideas.

## Prior work that changes the ranking

- [Flash Next optimization pilot](2026-11-07-flashnext-optimization-pilot.md):
  draft depths and a concurrent-encoder cutoff change were already tried.
  Host interference prevented the older quiet-policy qualification. The
  64-row cutoff was not promoted. Its speculative verifier remains at 1–4
  rows, so changing a 16-row cutoff does not fix verifier dispatch selection.
- [MoE projection experiment](2026-11-07-flashnext-moe-projections.md):
  the candidate measured 81.23 versus 85.50 sampled decode steps/s with two
  drafts, and 75.63 versus 80.33 with MTP off, in its recorded observation
  window. The candidate was withdrawn; this does not establish a universal
  regression but does defeat a claim that another gate/up fusion is an
  unexplored obvious win. The source already fused SwiGLU and overlapped work.
- [Sustained diagnosis](2026-11-07-flashnext-sustained-diagnosis.md):
  the same baseline fell from 95.46 to 82.20 decode steps/s within a cycle,
  with verification GPU spans accounting for almost all of the increase.
  These historical observations motivate balanced order and preservation of
  every busy-host result. They do not set today's expected absolute rate.
- [Optimization history](QWEN38_OPTIMIZATION_HISTORY.md) records earlier
  sampling, state, kernel and failed MoE attempts. Repeat experiments only
  when new source changes or a measured bottleneck supply a new hypothesis.

## Release candidate inventory

The decisions below apply to the selected Lily study. A rejected speed
candidate may still be an important correctness fix or useful in another
backend. Numerical results in this table are attributed upstream observations,
not locally reproduced Stark gains.

### Direct Metal and numerical behavior

| Item and primary source | Evidence and exact limitation | Lily decision |
| --- | --- | --- |
| [MoE routing fusion #31615](https://github.com/ggml-org/llama.cpp/pull/31615) | Adds sigmoid/bias and sqrt-softplus routing chains. Apple M2/M5 Ultra DeepSeek/GLM tests report about 1–4% gains; prefill can be unchanged. Selection bias must not alter mixture weights. | Formula-specific source reference. Existing Lily fusion must be assessed on its own equations; small upstream gains would often fail this study's practical floor. |
| [Fusion-group command-buffer boundaries #31688](https://github.com/ggml-org/llama.cpp/pull/31688) | Prevents graph-size-dependent fused/unfused rounding. M5 Max benchmarks within 1%; no claimed speed increase. | Correctness/reproducibility audit, especially before relying on exact hashes. Lily uses explicitly encoded operations rather than ggml fusion discovery. |
| [Small-K matmul #30982](https://github.com/ggml-org/llama.cpp/pull/30982) | Batch >8, K=32–63 switches generic GEMV to GEMM. M4 Qwen3.5-0.8B with rank-32 LoRA: prefill 799→1,817 tok/s. No-adapter and other-rank cases essentially unchanged. | Conditional shape check only; no demonstrated relevant Lily shape. |
| [Deterministic radix top-k #31084](https://github.com/ggml-org/llama.cpp/pull/31084) | Stable tie order improves CUDA/Metal reproducibility; reported CUDA kernel geometric-mean speed ratio around 0.89 in listed cases. That is not a measured Metal regression. | Correctness property to verify, not a speed technique. Do not weaken deterministic ordering to win a benchmark. |
| [TRI masking with select #30711](https://github.com/ggml-org/llama.cpp/pull/30711) | Avoids infinity multiplied by zero creating NaN; no Metal speed evidence. | Bounded analogous-mask check if a matching operation is reached. |
| [FA output bounds #31662](https://github.com/ggml-org/llama.cpp/pull/31662), [SOLVE_TRI bounds #32103](https://github.com/ggml-org/llama.cpp/pull/32103), [GROUP_NORM layout #31852](https://github.com/ggml-org/llama.cpp/pull/31852) | Fixes implementation-specific output/scratch bounds and noncontiguous shapes. | Compare only corresponding Lily operations/layouts. These are not evidence of the same bug in Stark and have no speed claim. |
| [Accumulator/src1 precision #31851](https://github.com/ggml-org/llama.cpp/pull/31851), [precision gating/tests #31896](https://github.com/ggml-org/llama.cpp/pull/31896) | Restores reference precision and rejects/falls back for unsupported paths. | Preserve as numerical-validation lessons. No performance credit; a changed precision policy needs matched references. |
| [RoPE/KV shift #31683](https://github.com/ggml-org/llama.cpp/pull/31683), [YaRN truncation #30206](https://github.com/ggml-org/llama.cpp/pull/30206) | Model-specific position/rounding correctness, not measured faster attention. | Check when adding those exact RoPE/shift semantics; do not apply a generic shift to hybrid recurrent state. |

### Speculation, state movement and host overhead

| Item and primary source | Evidence and exact limitation | Lily decision |
| --- | --- | --- |
| [Few-row quantized matmul #31533](https://github.com/ggml-org/llama.cpp/pull/31533) | CUDA int8-MMA path for 2–24 columns and constrained K/alignment. Qwen3.8-27B UD-Q4_K_M plus DFlash2, seven drafts, 80 conversations, 8,192 output budget: reported 10.1–34.9% throughput gains across four NVIDIA systems. | Strongest conceptual lead for small-row verification. Artifact, draft count, quantization and ISA differ; Lily already specializes ≤4 rows. No transfer of percentages. |
| [BF16 few-row MMVF #31959](https://github.com/ggml-org/llama.cpp/pull/31959) | CUDA tiny-batch Qwen indexer/hyper-connection operations show operation-level gains, including >2× on some shapes. | Profile corresponding Lily BF16 shapes only; a fast CUDA op is not a Metal implementation. |
| [Device target hidden states #31927](https://github.com/ggml-org/llama.cpp/pull/31927) | Uses device rows for NextN state, building on earlier #31506. Reported 4–10% prefill statement lacks a sufficiently specified A/B workload in the PR body. | Lily draft input already reads device hidden state. Audit remaining copies before proposing work. Full device sampling would be a separate project. |
| [Recurrent conv snapshots #31660](https://github.com/ggml-org/llama.cpp/pull/31660) | Initial greedy RTX PRO 6000 report: GLM +7–11%, Qwen3.8-27B and Flash Next unchanged. Later GLM multi-request MTP comparison was approximately neutral/slower; workload differs. Flash Next reduced copies without a demonstrated speed gain. | Low-priority copy-budget check. Retain both upstream observations; do not select the best number as a universal result. |
| [Remove non-fused GDN paths #32078](https://github.com/ggml-org/llama.cpp/pull/32078) | Removes dead alternatives and ubatch-dependent graph budgeting after fusion was already used. | Not evidence of a newly faster GDN algorithm. Lily already has fused recurrence and Q4 input/convolution paths. |
| [Quantized KV state-copy size #30541](https://github.com/ggml-org/llama.cpp/pull/30541) | Corrects block-based copy byte counts; earlier code could copy only a fraction of q4/q8 state. | Audit units at matching copy boundaries; exact cache/state controls matter. No throughput improvement is claimed. |
| [Scheduler reuse #32009](https://github.com/ggml-org/llama.cpp/pull/32009), [graph-result reuse #32089](https://github.com/ggml-org/llama.cpp/pull/32089) | Avoid reserve-time destruction/reallocation. Tests are supplied; no matched before/after throughput. Submission-harness token rates are not benchmark evidence. | Cheap check for concrete repeated allocations. Lily is not using this scheduler. |
| [Untouched graph metadata pages #32043](https://github.com/ggml-org/llama.cpp/pull/32043) | Reported host memory reduction around 52 MiB on plain Windows master in the named configuration; ggml arenas and graph metadata. | Very low priority for a ~100 GB model without a corresponding allocator/page-touch issue. |
| [Skip unused input uploads #31419](https://github.com/ggml-org/llama.cpp/pull/31419), [prefetch dependencies #32100](https://github.com/ggml-org/llama.cpp/pull/32100) | CUDA scheduler tests report roughly 0.9–1.8% in #31419; correct split dependencies avoid unnecessary input movement. | Inspect actual staged inputs; Lily's double staging and unified memory change the mechanism. |
| [Sampling graph outputs #32015](https://github.com/ggml-org/llama.cpp/pull/32015) | Prevents output aliasing in backend sampling graphs. | Ownership/correctness audit for reused scratch; not a measured speedup. |
| [Immutable reference-counted checkpoints #30137](https://github.com/ggml-org/llama.cpp/pull/30137) | Multi-checkpoint server workloads avoid deep host copies; substantial latency/peak-RAM gains reported on SYCL/CUDA setups with many retained checkpoints. | Current single-predecessor ownership is moved/borrowed and attention rows remain resident. Check an actual duplicate copy; do not add a multi-slot cache for this study. |
| [Parent-slot-first shared prefixes #30779](https://github.com/ggml-org/llama.cpp/pull/30779) | Multi-slot server scheduling improves prefix opportunity. | Different workload and cache contract. Fresh-request baseline does not cover this. |
| [NextN public API #30565](https://github.com/ggml-org/llama.cpp/pull/30565) | Promotes experimental API/removes deprecated interfaces. | API stability is useful for libllama consumers; it does not accelerate Stark's independent runtime. |

### Memory, placement and quantization

| Item and primary source | Evidence and exact limitation | Lily decision |
| --- | --- | --- |
| [FP8 KV #28392](https://github.com/ggml-org/llama.cpp/pull/28392), [CUDA support #28413](https://github.com/ggml-org/llama.cpp/pull/28413) | Optional static scales and OCP FP8 cache support. GPU implementation is CUDA; no corresponding FP8 Metal attention specialization in the tag. | Existing Q8 weakens the memory case. New range/error tradeoff needs matched numerical and long-context qualification. |
| [CPU execution of expert-cache misses #30966](https://github.com/ggml-org/llama.cpp/pull/30966) | Trades CPU compute against host-to-device expert uploads; tuning results depend strongly on cache size/CPU share, RAM bandwidth and discrete-device transfer. | Current Lily has full expert-bank buffers and no explicit missed-expert path. Apple unified memory changes the tradeoff. Reject as a speed port. |
| [ClassLRU/LRFU #31233](https://github.com/ggml-org/llama.cpp/pull/31233) | Reported warm-cache gains about 6–7% and fewer uploads in selected Qwen/GLM cache traces. | Requires the expert cache that this path does not use. |
| [Avoid CPU waits on cached experts #31888](https://github.com/ggml-org/llama.cpp/pull/31888) | Gemma4/RTX5090 cache configurations report 7–9% decode gains; Qwen Flash Next did not gain in the described comparison. | No reached CPU-miss/GPU-hit split to reorder. General dependency diagnosis remains useful. |
| [Linearized MoE layout #31047](https://github.com/ggml-org/llama.cpp/pull/31047), [empty-tile skip #32067](https://github.com/ggml-org/llama.cpp/pull/32067) | CUDA MMQ/MMF tile bookkeeping and fixup work; small/noisy layout gains. | Consider only if Lily profiling exposes tile assignment or empty work. Source code cannot be transplanted into Metal. |
| [NVFP4 gate/up plus GLU #31296](https://github.com/ggml-org/llama.cpp/pull/31296), [scale epilogue #26048](https://github.com/ggml-org/llama.cpp/pull/26048) | CUDA NVFP4 prefill fusion; e.g. Qwen3.8-27B on DGX Spark, 16K prompt, +18.1% in #31296. | NVFP4 differs from the exact Lily payload. No requantization or tensor-core assumptions in this comparison. Lily already fuses relevant activation work in its own format. |
| [External PLE shards #31185](https://github.com/ggml-org/llama.cpp/pull/31185) | Packaging/loading flexibility for large embedding tables; no runtime speed result. | Potential distribution/storage feature; not a demonstrated performance change to resident Lily PLE. |
| [mmap/iGPU lazy fallback #31957](https://github.com/ggml-org/llama.cpp/pull/31957) | Large improvement in a Strix Halo/RADV Vulkan mmap case, tied to page-fault policy. | Diagnose actual page-fault/residency cost on Apple first. Different API/backend; no direct Metal flag fix. |
| [Skipped NextN fit accounting #31569](https://github.com/ggml-org/llama.cpp/pull/31569) | Corrects offload placement where a transformer layer accidentally remained on CPU. | Lily directly loads the intended GPU banks and has no matching `n_gpu_layers` fitter. |
| [Shared multimodal compute buffer #31945](https://github.com/ggml-org/llama.cpp/pull/31945) | Reuses workspace across vision/audio/generation subcontexts; no numeric result supplied. | Capacity audit only if duplicate live workspaces exist; no evidence for a text decode gain. |
| [Graph log-mel #31808](https://github.com/ggml-org/llama.cpp/pull/31808) | Avoids per-request audio thread creation; CPU custom graph op used for reference behavior. | No audio frontend in this selected text/vision workload. |
| [Qwen conversion tensor merge #32102](https://github.com/ggml-org/llama.cpp/pull/32102), [NVFP4 scale attachment #31449](https://github.com/ggml-org/llama.cpp/pull/31449) | Conversion/layout and source-scale correctness for other paths. | Do not convert/requantize the Lily baseline to create an apparent engine improvement. |
| [Windows unbuffered loading #26014](https://github.com/ggml-org/llama.cpp/pull/26014) | Windows counterpart to DirectIO model loading. | Platform-specific; no Apple text-decode benefit established. |

### Backend and feature coverage outside the selected execution path

The [ggml release notes](https://github.com/ggml-org/ggml/releases/tag/v1.0.0)
also cover CUDA graph reuse and VMM-pool sizing, HIP BLAS workspace and MLA
indexer, Vulkan prefill FlashAttention/dequantized KV/large tiles/transposes,
SYCL device MoE scheduling/oneDNN cache, Adreno OpenCL decomposed attention and
multi-column GEMV, WebGPU kernels, Hexagon operations, SpacemiT/RISC-V kernels,
and IBM/OpenVINO/RPC behavior. None is the backend executing this Lily baseline.
Their general ideas need a demonstrated matching bottleneck before a Metal
experiment. No cross-platform backend project is authorized by this assessment.

The [CUDA d_state=64 SSM_SCAN fast path #31818](https://github.com/ggml-org/llama.cpp/pull/31818)
reports Zamba2-7B throughput around 11→15.1 tok/s. Its CUDA scan shape and
architecture differ from Lily's GDN recurrence. It is explicitly screened out
rather than treated as an applicable recurrent-model speedup.

[ARM NEON 4×4 tile selection #32128](https://github.com/ggml-org/llama.cpp/pull/32128)
reports 7–8% on Cortex-A53 CPU inference. It does not establish a gain for
Metal inference on Apple silicon. New architecture support (Kimi-K3.1-Pro,
Ripple, xing4_0), TTS, additional vision/audio families, parser improvements,
web UI, packaging and server endpoints expand functionality; they are not a
before/after speed claim for Qwen Flash Next Lily. Security fixes should be
handled through the relevant separate security workflow when in scope.

## Preregistered baseline and decision rules

This study evaluates the performance-related delta in llama.cpp 1.0.0 and ggml
1.0.0 against Stark's actual Qwen3.8 Flash Next Lily execution path. It records a
fresh before baseline; no candidate optimization has been implemented or timed.

The user explicitly requested measurements during ordinary concurrent system
activity. Do not wait for quiet admission, stop other work, or discard slow
requests because the host is busy. Preserve all observations. Historical quiet
window and host-screened studies retain their own evidence and are not
superseded. These measurements are an operational screening baseline under
observed load, not an isolated estimate of kernel speed.

Baseline source: `cc8e47a824f802e24d442183ebbdd29fd25ecaaa`, built with Rust
1.98.1, `--release --locked -p stark-generate`, in a distinct Cargo target.
Exact artifact: `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`, using the installed image-capable
configuration. Keep its quantization, vision tower, supported MTP head and
precision policy unchanged for later before/after comparisons.

Workload matrix, fixed before execution:

- 512 and 4,096 input tokens, 256 sampled outputs, context input plus output.
- Temperature 1, top-p 0.95, seed 42; top-k 20 at draft depths 1, 2 and 3.
- Nucleus-only top-k 0 at draft depth 2, at both input lengths.
- Each cell compares MTP off and the named depth with eight paired blocks,
  randomized ABBA order, and one excluded warmup per route. Eight cells give
  128 measured requests and 16 warmups if all complete.
- Use `scripts/benchmark_qwen38_mtp.py --observe-host`. Retain its existing
  `timing_eligible=false` quiet-policy field; that field does not disqualify
  observations from this explicitly authorized busy-host screening study.
- Synthetic corpus: freeze the current vendored `qwen4exp/model.rs` bytes in
  ignored runtime storage and record their SHA-256. Historical corpus results
  remain separate. Tokenization is outside timed generation.
- No engine profiling flags, no image requests, and no deliberate prefix reuse.
  Verify actual route acknowledgements and cache behavior before interpreting
  timings. No generated completions, token vectors or probabilities are retained.
  Privacy correction, 2026-11-16: the original runner copied the source-code
  corpus into each ignored runtime output directory as `frozen-corpus.rs`.
  The earlier claim that no prompt material was retained was too broad. No
  corpus text was published. The current CLI instead keeps one snapshot in RAM;
  the [PR review record](../archive/runs/2026-11-16-lily-projection-pr-review.md)
  records the fix and model-free privacy controls. Original as-run source and
  measurements remain unchanged.

The practical improvement floor is 10% in the relevant median useful decode rate
or complete-request latency. A future candidate must also exceed observed
same-code variation: use the larger of 10% and the baseline cell's full relative
range `(max - min) / median` as the initial screening hurdle. Preserve min,
median, max, interquartile range and median absolute deviation. A small apparent
win within that range is a stop decision, not a reason to seek a quiet window.

A future adoption decision requires fresh interleaved baseline/candidate runs
with the same artifact, frozen corpus, settings, output budget and cadence;
at least seven of eight paired wins in the primary cell; confirmation in the
other input-length cell; and no greater than 5% median regression in another
declared cell. Historical baseline versus a later standalone candidate cannot
establish causality when host load changes. If paired variation remains larger
than the improvement, reject the performance case. Memory/capacity changes need
an explicit useful-capacity result as well as timing and numerical controls.

This range-based hurdle and paired-win count are conservative engineering stop
rules, not confidence intervals or a statistical significance claim. Before
production promotion, also confirm a held-out corpus/seed; the current matrix
contains one synthetic source-code corpus. Noisy observations are retained and
remain useful for rejecting small apparent improvements.

Scheduling-only candidates must preserve within-route seeded output hashes and
MTP work counters. Precision changes need matching-reference numerical testing;
hash equality is not a quality qualification. New state-affecting paths also
need exact prefix restoration, image-resident text/MTP parity and image
transition/cancellation controls before promotion.

Only this study's worker processes may be stopped on timeout/failure. Other
applications, workers and jobs remain running. Fixed per-worker timeout is
3,600 seconds from the existing harness. A failed setup or partial cell is
retained and cannot be represented as a completed baseline cell.

## Baseline identity and measurement boundaries

| Field | Before baseline |
| --- | --- |
| Source | `cc8e47a824f802e24d442183ebbdd29fd25ecaaa`; engine source unchanged, dedicated worktree/target |
| Lily upstream | `e22d58fa7cc488493fb1b5d810a198bd29730aeb`, with Stark's recorded local patches; see [UPSTREAM.md](../../vendor/lily-core/UPSTREAM.md) |
| Vendored Lily Git tree | `b7c775d44fa20b55359a16999d300ad969c465d9` |
| Engine / generator Git trees | `a2a83cc1c70f11b072155ed74ecd609b3d2016b2` / `78fdb18b675ec700106087e2759fec5b5e2aca8f` |
| Worker SHA-256 | `61b9c8cc724407573169eb2abad7f908e4943eead7ac3122eeec3ef841a48640` |
| Matrix runner SHA-256 | `bf98687bc7f2aa7829d30cd946978ac10b415530b0244c3693acd5ab334bee8c` |
| Final-cell continuation runner SHA-256 | `b14f6e24c87af64dc5f77bf9b3dfd8c6e3b00228bef3705107db3cb616e8a0d5` |
| Paired harness SHA-256 | `b340d6473ef5616f60c7ea070ebfc7e722d233c7c9cfe949c2bfde002c53a504` |
| Cargo.lock SHA-256 | `f836a99ba1c6bd017fb829cd45f108fd97f9f1fb40c1497564898b3eac625f5f` |
| Frozen corpus SHA-256 | `551b35f67a59130fb7df09c152d065fd8f1f7d6e58ff6a1350a4acf43bab3387` |
| Artifact content manifest SHA-256 | `97c11dd6f7ee5a69e23759b9774acb21a786c48552ca2ede9dac8b6be4b5bc98` |
| Configuration SHA-256 | `2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899` |
| Image artifact SHA-256 from native readiness | `18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4` |
| Hardware / OS | Mac17,7, Apple M5 Max, 40 GPU cores, 128 GiB physical RAM; macOS 26.7.2, build 25G332 |
| Power | Observed AC power; existing AC power-mode preference 2. No settings changed. Clock/residency behavior is not isolated. |
| Tools / build | rustc 1.98.1 (`48a229cea`, 2026-09-01), locked release generator only; uv 0.12.24, Python 3.13.13 |
| Candidate / adoption | No engine candidate, after result, installation, default change or PR |

All 55 model shards, 106,905,155,192 bytes, were freshly hashed. Their hashes
and sizes match the LFS metadata for the exact
[pinned source revision](https://huggingface.co/fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl/tree/0f94842e3682b49acace4daeb4cf2e65a6f94f83).
No new model payload was downloaded. The complete measured manifest contains
63 files totaling 106,918,429,816 bytes, including local prepared image metadata
and tokenizer/configuration files. Local image metadata has its own identity;
its preparation is not claimed to be an upstream source byte-for-byte copy.
Size, nanosecond mtime and inode guards are checked after every cell.

The tokenizer's SHA-256 and size also match the pinned LFS metadata.

The private HF metadata manifest SHA-256 is
`1030ca7377999a8cf2daa05ce0cd14eadd4b5df0e931ffb51b86887ae1c13c19`;
the 55/55 match receipt SHA-256 is
`af4f0bba64b482083325abf3b8075baf029c11185ad8df1c95f5e08909b072f9`.
After the run, a separate final revalidation binds the unchanged payload map to
the completed identity receipt and checks the tokenizer too; its private
receipt SHA-256 is
`fc9b0e898d0fca39bd0b587c6347c68e9826235721cc9de779eb7d778dea3f3c`.
The public dataset projects that receipt without local filesystem paths.

Each cell loads a new worker and excludes native startup and its two warmups.
The complete-worker-request timer includes generation and Python/native pipe
transport, but excludes corpus tokenization, chat-template rendering, API/UI
transport and loading. Native `prefill_seconds` is recorded separately;
streamed user-visible time to first token is not measured.

Both sampled decode steps and emitted useful decode rates are retained. Each
subtracts the first sample/token, which is outside `decode_seconds`. Equality
of these two rates must be verified from actual counts; they must not be
substituted when a model emits fewer tokens or stops early. MTP on/off hashes
may differ because exact rejection uses random draws differently; within-route
repeatability is the relevant baseline control.

Native readiness must advertise the retained BF16 vision profile and exact
batched MTP support together. No image request is timed, and this record does
not newly qualify image numerical behavior. Prefix support is enabled by the
normal Python worker, but unscoped raw-ID requests must report zero reused input
tokens. The receipt records that count rather than assuming that repeated
prompts are cache-cold. Engine profile/capture selectors are absent.

Host observations preserve external CPU, competing model CPU, compiler flags,
coarse thermal state, sample gaps/failures and pageout/swapout counter changes.
They do not establish exclusive GPU access, GPU clocks, peak process footprint
or an allocation reservation. No worker peak-memory result is claimed.

## Fresh baseline results

Completed: seven cells in attempt 02 plus the remaining cell in attempt 03,
2026-11-15 10:23:30–10:42:05 America/New_York (15:23:30–15:42:05 UTC), including
hashing/loading/continuation overhead. Eight measured off/on pairs per cell,
128 measured requests and 16 corresponding excluded warmups. Failed setup
attempts are retained separately below. The exact execution order is in the
[aggregate data](data/llama-v1-flashnext-lily-baseline-2026-11-15.json).

Every measured request emitted and sampled 256 tokens, ended at the length
limit, and reported the exact input count and zero reused tokens. Thus useful
and sampled decode rates happen to coincide here: both use 255/decode seconds.
All within-route repeated hashes, exact-MTP route/counter checks, image-loaded
readiness identities, execution-profile checks and artifact guards passed.
This is repeatability/lifecycle-scope evidence, not a new model-quality or
stock-runtime equivalence test.

### Throughput and whole-request time

Rates and seconds are medians of eight requests per arm. Percentage changes
are medians of the eight within-pair ratios, which need not equal the ratio of
the two displayed medians. Positive decode change favors MTP; negative wall
change means lower complete-worker-request latency. Top-k 0 means nucleus-only.

| Input | Top-k | Drafts | Off useful tok/s | On useful tok/s | Paired decode change | Paired wall change | Decode wins |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 20 | 1 | 79.82 | 87.46 | +9.93% | -8.32% | 8/8 |
| 512 | 20 | 2 | 79.96 | 81.69 | +2.97% | -2.42% | 7/8 |
| 512 | 20 | 3 | 80.01 | 82.19 | +3.50% | -3.03% | 6/8 |
| 512 | 0 | 2 | 78.43 | 69.26 | -11.78% | +12.38% | 0/8 |
| 4,096 | 20 | 1 | 73.25 | 77.57 | +5.39% | -1.58% | 8/8 |
| 4,096 | 20 | 2 | 67.12 | 72.35 | +9.21% | -3.99% | 8/8 |
| 4,096 | 20 | 3 | 75.60 | 65.25 | -13.75% | +11.88% | 0/8 |
| 4,096 | 0 | 2 | 72.61 | 77.85 | +7.00% | -3.10% | 8/8 |

| Input / top-k / drafts | Off request seconds | On request seconds | Off prefill seconds | On prefill seconds |
| --- | ---: | ---: | ---: | ---: |
| 512 / 20 / 1 | 3.506 | 3.227 | 0.301 | 0.306 |
| 512 / 20 / 2 | 3.499 | 3.433 | 0.307 | 0.312 |
| 512 / 20 / 3 | 3.490 | 3.417 | 0.302 | 0.308 |
| 512 / 0 / 2 | 3.554 | 3.994 | 0.300 | 0.309 |
| 4,096 / 20 / 1 | 5.287 | 5.109 | 1.809 | 1.830 |
| 4,096 / 20 / 2 | 5.856 | 5.683 | 2.047 | 2.124 |
| 4,096 / 20 / 3 | 5.098 | 5.722 | 1.706 | 1.789 |
| 4,096 / 0 / 2 | 5.255 | 5.128 | 1.731 | 1.835 |

One draft at 512 tokens is the closest positive result, at +9.93% paired
decode and -8.32% request latency. It remains below the fixed 10% criterion;
the 4K cell does not confirm a similar improvement. The +9.21% long two-draft
result is also inside that cell's approximately 18–21% observed rate range.
There is no basis for changing MTP defaults from this matrix.

The negative results are materially larger than their cell variability:
three drafts at 4K is slower on all eight pairs, as is nucleus-only two-draft
sampling at 512. Avoid calling these settings a general speed improvement.
They remain workload-specific findings; another corpus/seed may differ.

### Repeat variation and future screening hurdles

IQR and MAD are in useful tokens/s. The relative range is `(max-min)/median`.
Each future same-route candidate must exceed the larger of 10% and the named
baseline range for the chosen metric, then pass fresh interleaved controls.
These are observed engineering hurdles, not estimated confidence limits.

| Input / top-k / drafts | Route | Useful tok/s range | IQR | MAD | Relative range | Decode hurdle | Request-latency hurdle |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 / 20 / 1 | off | 78.01–80.74 | 1.57 | 0.79 | 3.41% | 10.00% | 10.00% |
| 512 / 20 / 1 | on | 84.97–88.63 | 1.53 | 0.66 | 4.18% | 10.00% | 10.00% |
| 512 / 20 / 2 | off | 69.34–82.70 | 4.38 | 1.82 | 16.71% | 16.71% | 18.17% |
| 512 / 20 / 2 | on | 71.49–89.87 | 8.97 | 5.73 | 22.50% | 22.50% | 22.91% |
| 512 / 20 / 3 | off | 77.70–82.21 | 0.93 | 0.58 | 5.63% | 10.00% | 10.00% |
| 512 / 20 / 3 | on | 79.61–87.49 | 3.48 | 1.89 | 9.59% | 10.00% | 10.00% |
| 512 / 0 / 2 | off | 77.94–79.36 | 0.49 | 0.30 | 1.82% | 10.00% | 10.00% |
| 512 / 0 / 2 | on | 68.47–70.57 | 1.01 | 0.55 | 3.03% | 10.00% | 10.00% |
| 4,096 / 20 / 1 | off | 70.25–77.10 | 5.00 | 2.54 | 9.35% | 10.00% | 11.18% |
| 4,096 / 20 / 1 | on | 74.15–79.17 | 1.62 | 1.01 | 6.47% | 10.00% | 10.00% |
| 4,096 / 20 / 2 | off | 63.12–75.34 | 8.22 | 3.67 | 18.19% | 18.19% | 20.38% |
| 4,096 / 20 / 2 | on | 67.71–82.61 | 9.92 | 4.52 | 20.59% | 20.59% | 21.57% |
| 4,096 / 20 / 3 | off | 74.47–76.62 | 0.76 | 0.45 | 2.84% | 10.00% | 10.00% |
| 4,096 / 20 / 3 | on | 64.41–66.09 | 1.01 | 0.56 | 2.59% | 10.00% | 10.00% |
| 4,096 / 0 / 2 | off | 69.13–73.11 | 1.27 | 0.49 | 5.49% | 10.00% | 10.00% |
| 4,096 / 0 / 2 | on | 76.58–78.24 | 1.08 | 0.22 | 2.13% | 10.00% | 10.00% |

All 128 requests are retained, including 26 that fail the legacy quiet-policy
host screen. Maximum observed external CPU was 376.50%; competing-model CPU
was 6.72%; coarse thermal state reached numeric state 1. Saved measured-request
observations have no sample gaps or missing samples, no compiler flags, 79
reported pageout counter increments and zero swapout increments. These counters
do not attribute interference to a particular process or prove exclusive GPU
use. There is no slow-request exclusion or automatic measurement retry.

### What the fresh timings imply for the ranking

The existing MTP timing counters attribute median per-request fractions of
decode time to verification of about 79.9–89.7%, drafting of 6.3–12.9%, and
rollback/finish of 2.6–3.6%, depending on cell. They include GPU work and waits;
they do not isolate host encoding, kernel arithmetic or transfer bandwidth.
Eliminating the entire recorded rollback stage would still fall short of a
10% decode win, which strengthens the decision to defer copy-only work.

Verification remains the largest plausible diagnostic target. Its large
fraction is not evidence of waste or a claim that a particular kernel can be
sped up. Limit the next investigation to a half-day profile of reached small-row
operations. Proceed to a prototype only if a concrete stage and realistic
improvement ceiling can clear the applicable hurdle. Otherwise stop. FP8,
expert-cache architecture and a complete GPU sampler rewrite do not become
worthwhile because an unrelated CUDA workload improved.

This initial study produced no after-optimization result. Its deliverable is a sealed before
baseline, a reproducible recipe, and ranked decisions with stop conditions.

### Setup follow-up before measured execution

`20261115-LCPP1-01` hashed 62 artifact files, then stopped before any timed
request with a sanitized `ChatError` during Python Worker readiness. No baseline
timings were collected. A direct native launch with the same image staging
option subsequently returned image readiness and exact batched MTP support.
Python's additional automatic image-execution qualification is a separate path
and is not needed to measure text with the tower resident; its exact failure
cause was not isolated. The first attempt does not establish an image defect.

`20261115-LCPP1-02` retains the same fixed matrix and decision rules. Its wrapper
temporarily hides the Python media handle while reading initial readiness to
skip automatic image requests, without changing the native launch flags. It
requires the native retained-Lily image contract/source/revision, and the paired
harness requires MTP readiness and real draft activity. Image execution quality
is explicitly not qualified by this text baseline. The full payload manifest
also includes `stark-image-artifact.json`; the first attempt's 62-file manifest
omitted that metadata file. Both attempts and their distinct runner hashes are
retained. There is no text-only load fallback.

`20261115-LCPP1-02` completed seven cells (112 measured requests). The eighth
cell stopped on `host_observations_missing` during its first warmup, after a
successful 4,096-input/256-output generation with zero reuse. No measured
requests were discarded from that cell. Its failure and source identity remain
immutable. The exact v2 runner is
[archived](../archive/experiments/2026-11-15-llama-v1-lily/benchmark_lily_release_baseline_v2.py).

`20261115-LCPP1-03` runs only the unfinished cell with the same frozen corpus,
worker, artifact and settings. The current runner changes only the cell filter,
artifact-completion guard and handling of the exact ancillary monitor error:
missing observations become `eligible=false, observation_status=missing`.
Other failures still stop execution. It does not change native generation or
relax output, MTP, identity or reuse checks. It repeats the full payload hash
outside timing, which can affect cache/host state; that condition is retained
as part of the run recipe. This is a setup continuation, not a retry of a slow
or unfavorable measured request. The continuation passed all 16 measured
requests; none actually needed the missing-monitor fallback. Both failed
attempts remain classified as setup failures for their incomplete cells.

## Reproduction and the later after comparison

Use the sealed before worker while it remains available, verifying its SHA-256
above. For an independent rebuild, build the engine at the named baseline
revision in a separate checkout and target directory; take the matrix runner
from this study. Do not silently rebuild the before executable from a later
engine revision. Cargo dependencies stay locked, and the scorer's optimized
features must not be unified into the generator.

The commands used for the original worker and matrix were equivalent to:

```sh
CARGO_TARGET_DIR="$PWD/runtime/llama-v1-baseline-target" cargo +1.98.1 build --release --locked -p stark-generate
uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_lily_release_baseline.py \
  --model "$HOME/Library/Application Support/Stark/models/qwen38-flash-next-lily-q4-xl-0f94842e" \
  --worker runtime/llama-v1-baseline-target/release/stark-generate \
  --corpus runtime/llama-v1-lily-baseline-20261115-v2/frozen-corpus.rs \
  --output-dir runtime/llama-v1-lily-new-observation
```

The output directory must not already exist. The current runner includes the
documented v3 monitor handling; use the archived v2 source to reconstruct its
exact earlier tooling. Pass `--cell input4096-topk20-drafts2` for the declared
single-cell continuation. Startup checks and full payload hashing are outside
request timing and can affect host/cache state. Use the same preparation and
cadence in a later comparison rather than hiding that effect.

Keep the frozen corpus bytes from this baseline. Using a later edited
`model.rs` changes the workload. For a genuine after comparison:

1. Implement one diagnosed candidate in an isolated checkout; preserve source
   attribution, licenses and patch provenance. Keep the exact Lily quantization,
   supported head, image tower, sampler and existing precision unless the
   candidate explicitly tests a separately qualified precision policy.
2. Run the relevant smallest numerical/state controls. Scheduling-only changes
   must retain matching seeded outputs, MTP work counts and input/output budgets
   against the before worker. Precision changes require matching-reference
   numerical gates rather than a demand for an all-F32 production engine.
3. Execute fresh before/candidate/candidate/before sessions with identical
   workload cells, balanced order and enough matched requests for the eight-pair
   rule. The current runner's off/on pairs compare MTP configurations within
   one binary; they are not baseline/candidate pairs. Do not call a later
   standalone candidate run a causal code comparison against today's rates.
4. Keep every measured request and host observation. Apply the declared
   practical/variability hurdles, paired-win condition and cross-cell regression
   limit. A gain inside the observed range is a rejection of the speed case.
5. Confirm a held-out corpus/seed and complete the applicable exact prefix,
   image-resident text/MTP, image-transition, cancellation/failure and restart
   controls before promoting a state-affecting path. Keep measured speed,
   correctness, code adoption and installed-build state separate.

Public source-bound aggregates and hashes are in
[the data record](data/llama-v1-flashnext-lily-baseline-2026-11-15.json).
Full native receipts, runtime copies of as-run source, model files and frozen
workload material stay in ignored runtime storage; the v2 tooling source is
also archived publicly for provenance. The public corpus identifier is a
hash of existing source bytes; no generated text, prompt/token vectors or
probabilities are published. Existing historical results remain unchanged.

## Outcome, validation and limits

| Scope | Result / disposition |
| --- | --- |
| Release/source assessment | Completed; candidate inventory and ranking reviewed independently by two agents |
| Busy-host before baseline | Completed across attempts 02/03, 128/128 measured requests retained and functional controls passed |
| Setup failures | Preserved: initial Python readiness/image-check failure and last-cell missing-monitor warmup failure |
| Noticeable positive configuration gain | None meeting the fixed criterion in this matrix; no default promoted |
| Noticeable negative configuration results | Two workload-specific slowdowns, each 0/8 decode wins and beyond its observed range |
| New optimization candidate / after measurement | Not implemented / not measured |
| Image numerical/quality qualification | Not run by this text baseline; native image-loaded contract is checked |
| Model-free harness checks | Paired MTP/pilot tests: 37 passed; Python compilation and repository policy checks passed |
| Delivery / installation | Documentation, aggregates and harness on the study branch; installed app and engine defaults unchanged |

Not covered: other checkpoints/quantizations, stock llama.cpp versus Stark,
greedy decoding, nucleus-only draft depths one/three, near-capacity or >4K
prompts, independent held-out tasks, full-model quality, image inference speed,
new image numerical parity, GUI/API TTFT, peak memory or exclusive-GPU behavior.
The first script's image-check failure is not diagnosed deeply enough to call
an image-runtime bug. No new path is represented as production-qualified.

## Follow-up

The initial 9.93% result compared one draft with MTP off; it did not establish a
gain over Desktop's enabled two-draft recommendation. The source-scoped
one-draft candidate and subsequent three-depth reconciliation are recorded in
the [one-draft study](2026-11-15-flashnext-lily-one-draft-default.md) and
[reconciliation](2026-11-15-lily-mtp-default-reconciliation.md). The primary
one-versus-two latency gain was below its preregistered 5% threshold, and the
long-nucleus guard regressed useful decode rate by 6.01%; the held-out top-k
cell favored one draft by 20.03% decode and 14.45% request latency. The
candidate was reverted because the artifact-wide criteria failed; two drafts
remain recommended, while explicit one-draft requests remain available. The
original baseline and conclusions above remain bound to their original source
and criteria. The
[reconciliation data](data/lily-mtp-default-reconciliation-2026-11-15.json)
retains all three-depth observations.
