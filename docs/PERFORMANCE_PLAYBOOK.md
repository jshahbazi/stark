# Performance playbook

What to try when making a model family faster on Stark, in roughly the order to try it, with what each idea returned on Qwen3.8-Flash-Next ("Lily"). Compiled 2026-11-19 and updated 2026-11-22 from the records listed at the end.

This page summarizes records; it is not one. Every number was measured on a named artifact, build, workload and host, and keeps that scope. Nothing here qualifies another artifact, and where this page and a dated record differ, the record is right. A technique that paid on Lily has to be measured again on the next family. One that failed on Lily failed for a stated reason, and the reason is the thing to check before trying it again.

Unless a row says otherwise, figures are for `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at revision `0f94842e`, on an Apple M5 Max with 128 GiB, with 4,096 input tokens.

For comparing a new family against it: Lily is a 48-layer hybrid, 36 GDN recurrent layers and 12 sparse-attention layers. Every layer has a routed mixture-of-experts feed-forward (512 experts, 10 routed per token, plus a shared expert) and four hyper-connection streams. It has an n-gram embedding table of about 30 GiB, a one-layer draft head and a vocabulary of 248,320. The pack is about 107 GB.

## Keeping this page current

Find the baseline, rejected candidates and source-bound speed history for the target family in [model research and benchmarks](MODEL_RESEARCH.md).

This page is maintained. Read it before optimizing a model or bringing up a new family, and update it in the same change as the experiment record whenever an attempt:

- changes what a lever returned, or tries a lever on another family or artifact;
- adds a rejected or failed approach, with the reason it failed;
- adds, settles or retires an entry under "Ideas not yet tried";
- changes a correctness gate, a measurement rule or the checklist.

Add the record to the table at the end and move the date in the first paragraph. Give a figure its own artifact, build and workload when they differ from the defaults above. Results from another family get their own rows, and their own section once there are enough of them; a Lily result is not reworded to fit another model.

## Terms

| Term | Meaning on this page |
| --- | --- |
| Serial step | One ordinary decode step: one token through the full model |
| MTP | Multi-token prediction. A small draft head proposes the next few tokens and the full model checks them in one batched pass |
| Round | One speculative cycle: draft steps, one verify pass, acceptance on the host, and a finish pass that rolls state back to the accepted tokens |
| Verify row | One position in the verify pass. A round with m drafts verifies m + 1 rows |
| Acceptance | Share of proposed draft tokens the full model accepts |
| Exact | The speculative path has the same output distribution as serial decode. On Lily the verify rows and the state after rollback are also bit-identical to serial decode |
| Useful decode rate | (sampled tokens - 1) / decode seconds, in tok/s |
| Same-loaded | Both arms run in one loaded worker process, alternating, so load-to-load variation is removed |

## The short version

- The large gains were structural and came first. Putting the whole token loop on the GPU with one completion per token took the first checkpoint from 0.67 to 22.9 tok/s. The precision policy and pack took it to about 77. A sampler that does not sort the vocabulary added 48 percent at top-k 20. Speculation with a batched verifier added 13 to 16 percent at top-k 20.
- After that, work on single kernels returned 1 to 6 percent per component and about 2 percent or less in the model. Most kernel candidates were rejected, and several were rejected twice.
- A profile says where the time goes, not whether it can be removed. Before building, find out what binds the stage: memory bandwidth, arithmetic, launch count or the host. The 2026-11-19 estimate of 15 to 19 percent from cheaper verify rows was wrong because it skipped that step.
- In speculation the unit is the round: expected tokens per round over cost per round. The verify pass is 80 to 90 percent of a round, so what matters is the cost of one more verify row against a serial step. On Lily that is about 5.7 ms against 14 ms.
- Host scheduling inside the round was worth more than the row-kernel work of 2026-11-19. The merged build is 9 to 18 percent faster with speculation on and no output changed; the one row change that was adopted accounts for 2 to 3 points of that.
- No fixed draft depth is right everywhere. Acceptance runs from about 50 percent to over 90 percent depending on how predictable the text is and how it is sampled. Zero, one, two and three drafts each win on some workload.
- So depth is now chosen per round. On 2026-11-20 a planner with prompt lookup replaced two fixed drafts as the default on the pinned artifact, after a gate that ran every schedule on the same inputs across prose and code, greedy and sampled, at three input lengths: 1.218 times MTP off against 1.155. Two fixed drafts, the default until then, had been running up to 5 percent below MTP off on corpus prose. On natural chat responses the two are level; the planner's gain is on text the model finds hard to predict.
- For chat and agent traffic the largest latency lever is not decode speed. Restoring state for a shared prompt prefix cut time to first token on follow-up turns by 5 to 7 times, and one extra snapshot inside the prompt cut it a further 40 to 46 percent at 12K.
- An idle GPU costs about 0.8 s on the next request. A keepalive signal every 500 ms removes it for almost no power.
- Kernels are compiled with fast math, so a restructured kernel can pass every synthetic byte test and still change values on the real model. The real-model control and the output-hash gate decide.
- Session drift of 3 to 15 percent is normal on this host, and request cadence alone moves decode by 10 percent. A gain smaller than that needs alternating pairs inside one loaded worker, and an arm the change cannot affect as a noise control.
- A decision needs enough valid samples in every cell it rests on. In the schedule gate, two cells with one and three valid samples said the planner lost to the old default by more than 5 percent. With nine each they did not, and the decision reversed.

## Step 1: build the time budget

Do this before choosing a lever. Each question has a cheap instrument.

| Question | Instrument | Lily's answer |
| --- | --- | --- |
| How long is a serial step, and where does it go by stage? | Stage profiler (`STARK_QWEN38_PROFILE_OUTPUT`) in a separate instrumented session | About 13.3 to 13.9 ms uninstrumented; split below |
| How much of a token is host work: sampler, encoding, waits, readback? | The same profile, plus sampler micro-timing on a real readout | 0.09 ms per top-k sampler call; 0.36 ms per greedy call before it was vectorized |
| For speculation: draft, verify, finish and host time per round; tokens per round; accepted and proposed counts | The worker's own `draft_seconds`, `verify_seconds`, `rollback_seconds` and MTP counters | Verify is 80 to 90 percent of a round |
| What does one more verify row cost, by stage? | Instrumented sessions at 1 to 3 drafts | About 5.7 ms; split below |
| How much of a long request is prefill? | Request timing | About 44 s of a 62 s request at 88,398 input tokens |
| How long from process start to ready? | Fresh-process timing | 10.5 to 11.9 s once payload digests are reused |
| What does the first request after an idle pause cost? | First-sample time after a 10 s pause | About 1.07 s against 0.30 s before the keepalive |

Throughput is quoted only from uninstrumented sessions. Instrumentation perturbs scheduling, parent and child stages overlap, and host spans do not add to GPU spans.

GPU time of one serial step and of a verify pass ([round profile][profile], instrumented):

| Stage | Serial step, ms | Share | Each extra verify row, ms |
| --- | ---: | ---: | ---: |
| Routed and shared experts (MoE) | 4.89 | 34% | about 2.6 |
| GDN recurrent layers | 3.36 | 23% | 0.55, 1.24, 1.65 at rows 2, 3, 4 |
| Hyper-connections | 2.91 | 20% | about 1.15 |
| Sparse attention (QSA) | 2.16 | 15% | about 0.54 |
| Vocabulary head | 1.13 | 8% | flat |
| N-gram embedding | 0.10 | 1% | flat |
| GPU command total | 14.56 | | about 5.7 |

The stage order has been stable since the first profile on 2026-10-28 (MoE 31 percent, hyper-connections 23, GDN 21, attention 16, head 8).

One sampled two-draft round on the base build: 29.39 ms, of which draft 2.91, verify 25.39, finish 0.88 and other host work 0.20, for 2.28 tokens.

### Estimate the ceiling before building

If a stage is a fraction f of the time and a change makes it s times faster, the whole gets `1 / (1 - f + f / s)` times faster. Removing a 5 percent stage entirely cannot give 10 percent. Halving a stage gives 10 percent more throughput only if the stage is about 18 percent of the time.

This rule stopped several projects before any code was written:

| Idea | Share of time | Decision |
| --- | --- | --- |
| Share vocabulary-head weight loads across verify rows | 5 to 6 percent | Rejected before build |
| Coalesce rollback and snapshot copies | 2.6 to 3.6 percent of decode | Stopped |
| Move exact target sampling to the GPU | About 1.1 percent of wall time | Deferred |

### Find out what binds the stage

The profile gives time. It does not say whether the time is weight bytes crossing memory, arithmetic per row, the latency of many small launches, or the host. Each needs a different fix, and a guess is expensive. Controls that settled it on Lily:

| Control | What it showed |
| --- | --- |
| Run the routed-expert kernels with every row choosing the same experts, then different experts ([MoE rows][moe]) | Gate/up costs 1.45 ms per extra row with distinct experts and still 0.95 ms with identical ones. Most of the slope is arithmetic, not bytes. Down/combine costs 0.85 ms per row either way |
| Count duplicate expert selections across rows before building a sharing kernel | 34 to 50 percent of selections repeat, yet the ceiling from sharing was only 0.3 to 0.4 ms per row |
| Time the existing kernels against shared-weight kernels in isolation ([HC rows][hc]) | Hyper-connection row cost is mostly re-reading and re-normalizing the input row, about twice the weight bytes |
| Add sub-stage markers to the verify branch ([GDN replay][gdn]) | The rising GDN slope at rows 3 and 4 was the input projection, not the state captures the first reading blamed |
| Run one group serially over the rows with cache-resident weights ([HC rows][hc]) | Byte-exact and slower at every row count: the stage is bound by latency, not by weight traffic |

## Levers

### 1. Take the host out of the per-token loop

This is where an unoptimized family loses an order of magnitude. Count command submissions, waits, uploads and bytes per token first. The first Qwen profile found 917 submissions and 6.02 GB of uploads per token.

| Attempt | Result | Decision |
| --- | --- | --- |
| Remove repeated uploads and submissions; map immutable weights once and keep the views alive | 0.67 to 4.32 tok/s | Kept |
| Chain stages on the GPU so intermediate buffers never return to the host | 4.32 to 12.52 | Kept |
| Encode the complete decode graph with one completion per token | 11.2 to 22.9 | Kept |
| Schedule independent projections together, cache keys and geometry checks, fuse finite checks | 23.2 to 43.7 | Kept |
| Encode the next step's commands while the GPU runs the current step | 74.0 to 77.2 | Kept |
| Poll for completion every 200 microseconds instead of blocking | 3.60 to 3.50, CPU time 1.01 to 2.92 s | Rejected |
| Residency hints attached to each command | 12.30 to 7.79 | Rejected |
| A cache of selected expert slices | 7.79 to 7.03, on GLM | Rejected |
| Looser memory admission for whole expert banks | No first request after 310 s, on GLM | Rejected |

Every row except encode-ahead is from the first, F32 checkpoint on a short greedy workload, or from GLM where marked; they are not on one curve with the later 4K figures. Source: [optimization history][hist], rows F01 to F18, P01 and P06.

Polling lost here and won later. In a 14 ms serial loop, polling cost CPU and gained nothing. Around a 25 ms verify pass on 2026-11-19, polled waits were the largest single host gain. Test both where the wait is long.

### 2. Choose the precision policy, and know which pack you are running

- The policy that held: BF16 activations, F32 reductions, logits and recurrent state, and an 8-bit attention cache. Full F32 execution is kept as reference evidence only.
- The jump from 43.7 to about 77 tok/s coincided with a change of pack, not only of engine. Two packs of the same model with similar total bytes differ per tensor: Lily stores GDN projections at 4 bits and embeddings at 8, and the official mixed pack does the reverse. The same engine peaked at 79.5 tok/s on Lily and 66.8 on the official pack.
- Storage format did not explain speed in Sushi either, the MLX-based engine compared on 2026-11-19. Its plain affine control pack decodes serially at about the same rate as its trellis-coded packs.
- A BF16 attention cache changed full-vocabulary readouts and was rejected. BF16 partial sums lost cancellation in a split reduction, and 136 of 1,088 outputs left their bounds until the partials were kept in F32.
- Matching a reference's arithmetic can cost speed before it gains any. The first reference-order paths for the official pack ran 41 to 43 tok/s; the optimized kernels that kept the same order reached about 66.
- Reduction order is part of the numerical contract. Wider lanes, extra accumulators and regrouped sums all change it.

Source: [optimization history][hist], sections 2, 6 and 8.

### 3. Make the sampler cheap

With a vocabulary of 248,320, a naive sampler costs milliseconds per token, on the host, after the GPU has finished. Check it early; it is cheap to fix and independent of the model.

| Attempt | Result | Decision |
| --- | --- | --- |
| Partial top-k selection instead of a full sort | 50.6 to 75.1 tok/s at top-k 20 (+48 percent) | Kept |
| Radix sort for the full ordering that nucleus sampling needs | 3.87 to 1.15 ms, then 0.64 ms | Kept |
| Contiguous prefix scan for small top-k | 0.274 to 0.073 ms | Kept |
| Vectorized finite-value check of the readout | 38.7 to 8.4 ms per request | Kept; no reliable model gain |
| Proof that the nucleus stops inside the top 64, with a full-sort fallback | 899 to 350 microseconds per draw | Kept; model rate unchanged |
| Parallel exponentials for large candidate sets | Sampler 2.43 to 1.87 ms | Kept; model gain inconclusive |
| Vectorized greedy argmax with a first-maximum rule | 0.36 to 0.02 ms per call | Kept ([host round trips][host]) |
| Four-bank radix histograms | 13 to 21 percent slower | Rejected |
| Reuse of sampler buffers | 0.5 to 1.4 percent slower | Rejected |

Every one of these preserves the exact ordering, tie rule, probability arithmetic and random-number draws. Source: [optimization history][hist], rows S01 to S13.

### 4. Kernel work on the dominant stages

Expect little. The accepted set of MoE, hyper-connection and GDN kernel changes on Lily measured +1.9 percent at top-k 20 and +0.05 percent with nucleus sampling in the same-loaded comparison. The patterns:

| Pattern | Examples | Outcome |
| --- | --- | --- |
| Fuse two stages and remove launches | Expert down/combine written straight into the hyper-connection residual, 48 launches removed | 5.6 percent off the component; model change inside noise |
| Fuse, but add a launch or a prepass | Three-launch hyper-connection variants; a cached gain and RMS prepass | Flat or slower (33.6 to 40.9 microseconds per read) |
| Specialize kernels to the real fixed dimensions | Expert gate/up and shared-down projections, 2026-11-07 ([MoE projections][moeproj]) | 2 to 3 percent off the component; in the model the sign reversed between groups; withdrawn |
| Raise the row count at which batches use the concurrent encoder | 16 to 64 rows ([pilot][pilot]) | Complete requests 1 to 8 percent slower; rejected |
| Repack or regroup weights for wider SIMD | Packed 4-bit layouts, row packing, wider lanes, four output rows at once | Flat to 3.7 times slower on the real shapes |
| Cache values derived from the input | Cached input sums, cached scaled inputs | Flat or slower |
| Change the reduction tree for speed | Pairwise reductions, four accumulators, two state values per thread | Not bit-stable; depends on the compiler's order |
| Port a kernel from the generic engine | Stride-aware matrix-vector port | Synthetic gains did not transfer |
| Keep values in registers instead of group memory | GDN output in registers | 1.2 percent off the component |

Three things recur. A synthetic win on a large or convenient shape does not transfer to the real expert shapes. A fusion pays only if it removes launches. And a kernel that passes synthetic parity can still change the model's output: one hyper-connection pairing passed 28 synthetic cases and changed the teacher-forced and sampled hashes.

A generic executor is not a shortcut. The standard engine path for the same converted weights reached 44.7 tok/s against 77.8 for the specialized graph, after a parallel expert router took it from 14.2 to 51.5 on a short prompt. That migration was paused.

A gain in a serial kernel does not reach speculation by itself. The verify pass runs its own row kernels.

Source: [optimization history][hist], rows K01 to K24, U06 to U30 and F07; [llama.cpp assessment][assess].

#### Inkling 2 Mini packed experts

For prefill batches, group routes by expert before the packed product. On original layer 10, the pinned 32×64×32 matrix tiles and FP32 replay preserved all 5,750,784 BF16 boundary words and reduced paired component GPU time to 0.083 of serial at 64 rows and 0.069 at 128. Only 15–16 experts were active in these concentrated controls. Route sorting was outside timing. Measure route diversity and the complete prefill graph before choosing a production threshold. [Grouped projection pilot](model-experiments/2026-11-21-inkling-mini-grouped-nvfp4.md)

The GPU-routing follow-up reproduced the same routed BF16 boundaries and exact bucket tables. With every expert active, paired component GPU ratios were 0.258 at 64 rows and 0.138 at 128 including GPU sorting. Use full activation coverage and realistic route density when measuring the integrated graph. Per-row caller copies and the remaining target stages sit outside these component timings. [GPU routing and uniform stress](model-experiments/2026-11-21-inkling-mini-grouped-nvfp4.md#20261121-inklinggroup-02--gpu-routing-and-diverse-expert-coverage)

Capture the actual dispatched kernel before emulating GPU accumulation order. The layer 11 short-prefix failure started at one BF16 gate-projection word. F64 accumulation and padded Thin NAX replay retained the mismatch. The released MLX runtime selected GemvWide, and matching its FP32 dot grouping and lane tree passed short and 4K connected parity. The paired 4K slice wall ratio of 0.965 cleared the no-regression gate while falling below the practical speedup threshold. [Short-prefix projection and paired budget](benchmarks/2026-11-21-inkling-mini-short-projection.md)

For the original Inkling draft, test the folded Markov weights directly on source ingress. Contiguous Metal tensor products and a load-time transpose corrected a stride-dependent fold failure and matched all four pinned short-prefix proposals. The serialized target verifier reproduced all 16 target rows and produced 10 agreements in 15 proposals while costing additional wall time. Measure the incremental target cost of a batched verification row before investing in rollback or deeper drafts. [Original DSpark controls](model-experiments/2026-11-21-inkling-mini-dspark.md)

On the original three-pass Mini, test the weight layout and the activation scale at the actual kernel boundary. At 32 rows the Unsloth Metal indexed kernel with an early activation outer scale missed the registered relative L2 limit on both original layer 6 projections (about `1.6e-4`). Keeping the four-over-six inner values through the dot and restoring the scale afterward reduced both errors to about `5e-8` on the same inputs. Preserve the BF16 projection round before SwiGLU and apply routing weights before down-input quantization. [Component and model benchmarks](benchmarks/2026-11-20-inkling-mini-m5-max.md)

Time the connected routed sublayer before multiplying an isolated graph latency by 44 logical visits. In a full-bank layer 0 MLX control, eight ordered singleton calls over distinct source inputs completed in 1.242 ms with one host wait (0.155 ms each). An individually synchronized singleton took 0.607–0.685 ms across sessions. The test reused a physical layer. A complete native token needs different layer weights, attention and the remaining stages. [Measurement boundaries](benchmarks/2026-11-20-inkling-mini-m5-max.md#original-weight-component-budget)

On the M5 Max, the native original-weight 44-visit slice took 27.35 ms median wall at 4K with one completion. It covered attention and cache mutation, cross-pass gates, block residuals and NVFP4. Test the expert path at exact BF16 ingress and again with its actual upstream producer. At physical layer 10, a 0.246% worst-row relative L2 difference before the router and quantizer grew to 8.96% after the expert sum. All 18,432 final BF16 expert values matched at identical ingress across the two visits in the same fixture. Check the upstream BF16 boundary with the pinned runtime before optimizing the packed products. [Native timing and parity controls](benchmarks/2026-11-21-inkling-mini-native-slice.md)

Capture the first differing boundary with live cache state. In the pinned Ollama follow-up, matching the residual mix and RMS order repaired the initial BF16 controller input; source-order controller expansion and attention passed the 4K connected gate. The 44-visit diagnostic moved from 27.23 to 24.84 ms on the same controlled workload, with a median paired wall ratio of 0.912 and one completion. One short global expert row still missed its gate. A padded BF16 projection candidate made identical-input parity worse than the single-row SIMD dot and was archived. [Live capture and paired budget](benchmarks/2026-11-21-inkling-mini-pinned-oracle.md)

Reordering Inkling's serial prefill rows by layer preserved every tested cache byte, target tap and continuation logit and measured a 0.995 paired wall ratio. Share matrix loads and launch work before relying on execution order for prompt speed. Match the reference's prefix chunk geometry when diagnosing singleton decode, since routed quantization and relative attention can change during batched prefill. [Full target and prompt scheduling](model-experiments/2026-11-21-inkling-mini-full-target.md)

Using the layer-major schedule to share four-row BF16 weights and group a 64-row NVFP4 tile gave an Inkling prefill wall ratio of 0.294 in eight paired windows. A complete 4K prefill took 32.090 s with 43.7 MiB additional Metal allocations. Full native serial prefill took 103.617 s on the same IDs and produced identical live cache state, taps and continuation logits. Four repeats of the same tile from a 4K snapshot gave a 0.316 paired wall ratio and exact continuation. The worker measurement remains separate from the target diagnostic. [Full target and prompt scheduling](model-experiments/2026-11-21-inkling-mini-full-target.md#20261121-inkprefill-02--64-row-shared-projection-and-grouped-experts)

The grouped worker ran 4K prefill in 32.98 s greedy and 34.28 s sampled, compared with 114.52 s and 114.63 s in earlier serial-worker screens. Decode cleared both stage budgets. The gap to the reference prefill floor remains much larger than the full worker's decode overhead. Measure dense products and relative attention on the connected prompt graph before choosing the next kernel. [Worker benchmark](benchmarks/2026-11-21-inkling-mini-native-worker.md)

Match prompt geometry and capture the first prompt forward before debugging post-prefill decode. On eight 512-row commands, matching the command width alone reproduced the B64 cross-runtime logit errors. Matching the source NAX fragments fixed the first full-model attention-gate difference. The source local attention also required a BF16 relative table and rounding the unnormalized probabilities before the value product. The combined repair matched every captured BF16 attention boundary through MLP ingress at visit 0. The first remaining visit 0 output mismatch moved from row 1 to row 52 in the expert path. [Matched 512-row prefill](model-experiments/2026-11-21-inkling-mini-matched-prefill.md)

On the 4K B64 graph, attention dense projections accounted for 11.81% of command GPU time and relative/cache attention for 13.40%. The relative/cache band doubled from 40 to 80 ms per tile as context grew, with dense steady at 60 ms. Halving both bands gives a 1.14× ceiling on this profile. Routed experts took 41.03% and other encoded work 33.25%. Subdivide those phases before committing another prefill kernel. [Connected M5 Max prefill profile](model-experiments/2026-11-21-inkling-mini-prefill-profile.md)

The separate reference-shaped 512-row profile measured 3.17 s in dense projections and 3.12 s in relative/cache attention over 4K. Its uninstrumented target prefill was 30.87 s and used 303.5 MiB of added Metal workspace, above the worker's 128 MiB reserve. Retain the profile as a numerical and performance diagnostic. The 64-row worker rerun took 33.05 s greedy and 33.52 s sampled and matched both output hashes. [Matched-width profile](model-experiments/2026-11-21-inkling-mini-prefill-profile.md#profile-after-the-matched-width-repair) · [Worker screening](benchmarks/2026-11-21-inkling-mini-native-worker.md#4k-worker-rerun-after-the-matched-trace)

Subdividing the B64 prefill graph found 4.35 s in the four MXFP8 visits and 10.15 s preparing attention and expert inputs. Grouping those expert visits, sharing controller expansion and dense projection loads, and using 128-row commands lowered exact native target prefill from 32.03 to 23.28 s at 4K. Grouped expert tails cut a 31-row tile from 567 to 224 ms. The combined graph used 89.8 MiB of additional workspace. Exact batching of relative/cache attention had a separate 2% marginal gain for 13.4 MiB of workspace and was dropped from the production candidate. [Faster prefill attempt](model-experiments/2026-11-21-inkling-mini-faster-prefill.md)

The B128 follow-up split 22.50 s of 4K command GPU time into 7.65 s of experts, 7.47 s of preparation and finish, 7.20 s of dense/relative attention and 0.18 s of embedding and gaps. The prefill floors allow about 6.5–6.9 s in total. Batched controller RMS and direct expert bindings took 21.15 s against 23.39 s in one exact same-session 4K pair. The 256- and 512-row geometry probes gained 5.6% and 8.4% in short screens and required about two and four times the workspace. A source-shaped dense variant changed logits, and compact NVFP4 removed its certified reduction replay. The next floor-scale comparison needs packed activations, batched projections and query-by-key attention across the connected graph. [4K prefill budget and attempts](model-experiments/2026-11-21-inkling-mini-prefill-budget.md)

Packing the activations across the original Inkling graph eliminated most of the per-token dispatches and brought the exact 4K target diagnostic from 20.67 to 11.09 s. The packed profile put 11.57 of 11.86 GPU seconds in projections, attention and experts. The first worker with packed activations and eight-route MXFP8 reached 473 tok/s greedy and 407 tok/s sampled on two cold 4K requests with matching output hashes. The combined matrix variant reached 771 tok/s in the target diagnostic and changed the continuation logits. Short and full-prompt controls retired the guarded matrix variants, wider 1,024-row geometry and the NVFP4 register-fragment kernel for lack of speedup. The finer profile put 2.99 s of 9.15 s in NVFP4 matrix products and 0.07 s in their quantization and ingress. Use the original runtime to qualify any change in accumulation order. [Packed prefill experiments](model-experiments/2026-11-21-inkling-mini-packed-prefill.md)

### 5. Speculative decoding

#### The design that held

- The draft head proposes; the full model verifies every proposed position in one batched pass.
- Acceptance is exact rejection sampling on the host from complete F32 rows: accept a draft token x with probability min(1, p(x)/q(x)), and on rejection draw from max(p - q, 0) renormalized. Greedy reduces to comparing argmaxes.
- Verify rows and the state after rollback are bit-identical to serial decode. That is tested, not guaranteed by construction.
- A seeded request need not match its MTP-off output, and must reproduce itself run to run.
- The draft head costs about 1.84 GB resident. It loads alongside the image tower with no measured text slowdown (104.7 against 105.5 tok/s on a short greedy prompt), image requests keep MTP off, and prefix snapshots are keyed on whether MTP is enabled, not on the draft count or the round schedule ([vision text MTP][lvmtp], [selector prefix reuse][selprefix]).

Three verifier designs were tried before one paid:

| Verifier | Result |
| --- | --- |
| One ordinary serial step per proposed position ([exact MTP][mtpexact]) | Slower than MTP off in every pair: 23 to 40 percent on Lily, 17 to 38 percent on the official pack. Kept as the strict reference |
| The batched prefill kernels ([batched MTP][mtpbatched]) | Failed numerical qualification. Prefill and decode differ in accumulation order, rounding boundaries and attention path, and exact rejection sampling needs the verifier's probabilities bit-equal to decode |
| The decode kernels' own bodies run along a row axis, layer by layer ([batched MTP][mtpbatched]) | Byte-identical readouts; +16.9 percent at two drafts in the first build, +1.5 at one, -13.0 at three |

After that, removing launches from the verifier did most of the work. Batched sparse attention removed 96 launches at three rows, and batched GDN projection, convolution and scatter removed 108 across the 36 GDN layers. The long runs measured 67.5 to 78.3 tok/s at top-k 20 (+16.1 percent) and later 73.0 to 82.9 (+13.5 percent), with nucleus sampling at +0.6 to +1.7 percent and no better in request time.

A verifier is tied to a precision policy. Lily's row kernels did not preserve the official pack's rounding boundaries, so that pack needed its own batched verifier (56.7 to 65.0 tok/s, +14.7 percent). Check the stored width of each tensor first: that pack's head is 8-bit where the default assumed 4.

Source: [optimization history][hist], rows X01 to X06; [MTP optimization][mtpopt]; [MTP combination][mtpcombo]; [official batched MTP][mtpofficial].

#### The round model

With `a[i]` the chance that draft position i is accepted given the ones before it were:

- Expected tokens of a round with m drafts: `E(m) = 1 + a[0] + a[0]a[1] + ... + a[0]...a[m-1]`
- Cost of a round: `m draft steps + verify(m + 1 rows) + finish + host`
- Rate: `E(m) / cost(m)`. A deeper round pays only if its extra expected tokens per extra millisecond beat the rate already achieved.

Measured on the base build ([round profile][profile]):

| Arm | Round ms | Tokens per round | Accepted / proposed | tok/s |
| --- | ---: | ---: | --- | ---: |
| Greedy, off | 13.93 per token | | | 71.9 |
| Greedy, 1 draft | 24.40 | 1.93 | 492 / 528 | 79.1 |
| Greedy, 2 drafts | 31.32 | 2.78 | 654 / 732 | 88.7 |
| Greedy, 3 drafts | 39.55 | 3.45 | 726 / 882 | 87.3 |
| Sampled, off | 13.26 per token | | | 75.5 |
| Sampled, 1 draft | 22.28 | 1.74 | 216 / 292 | 77.9 |
| Sampled, 2 drafts | 29.39 | 2.28 | 286 / 442 | 77.5 |
| Sampled, 3 drafts | 38.24 | 2.48 | 304 / 612 | 64.8 |

The third greedy draft cost about 8.2 ms for 0.68 more tokens, which is no better than the rate two drafts already had. At temperature 1 the third draft added 0.2 tokens and three drafts ran 14 percent below MTP off.

The two numbers to get for a new family are the cost of one more verify row as a fraction of a serial step (Lily: about 40 percent), and acceptance by position on real workloads. Everything else follows from them.

#### Making verify rows cheaper

| Attempt | Result | Decision |
| --- | --- | --- |
| Batch sparse attention across rows | 96 launches removed at three rows; 68.2 to 79.8 tok/s in a short on/off window | Kept ([hist], X03) |
| Batch GDN projection, convolution and scatter across rows | 108 launches removed; 66.7 to 80.4 in a short on/off window | Kept ([hist], X04) |
| Fuse expert down/combine with the hyper-connection write across rows, one launch and one barrier fewer per layer ([batched MTP][mtpbatched]) | +12.4 percent on against off at two drafts; the fusion's own share was not isolated | Kept |
| Share 8-bit vocabulary-head loads across rows, official pack ([official batched MTP][mtpofficial]) | 2.9 times faster as a component at three rows; on-against-off gain moved from +9.9 to +15.0 percent in separate windows | Kept on that pack. On Lily the head is already flat across rows |
| Read each packed weight once for all rows, with separate accumulation per row ([projection reuse][reuse]) | +1.8 percent on a natural coding response, +4.5 on planning, +5.4 at 88K input | Adopted after the owner accepted a gain below the 10 percent gate |
| Write only the final recurrent state in verify and roll back by replaying the accepted rows ([GDN replay][gdn]) | 113 MB of state writes removed per extra row; +1.6 to +3.4 percent on MTP arms | Adopted |
| Share routed experts that several rows selected ([MoE rows][moe]; earlier as expert union, [hist] X05) | Byte-identical, and a tie on gate/up and slower on down/combine. The earlier attempt was 0.67 to 1.02 times baseline depending on overlap | Rejected twice |
| Shared-weight hyper-connection read, down half ([HC rows][hc]; earlier on 2026-11-17, [kernel follow-up][fskernel]) | Byte-identical both times. 1.07 to 1.12 times faster as a component, then -2.0 percent in the model; on the second attempt 0.06 to 0.25 ms per pass and below resolution in the model | Not adopted, twice |
| Shared-weight hyper-connection read, up half | Three variants, none bit-exact; one passed a first random draw and failed a second | Rejected |
| Restructure the GDN input projection for rows 3 and 4 | The only faster variant passed 288 synthetic cases and changed 3,476,470 values in the real-model control | Rejected |

What generalizes: removing launches by batching rows worked; reading weights once worked a little; removing per-row state capture worked a little; anything that restructures per-row arithmetic under fast math is unlikely to stay bit-exact, and where the stage is arithmetic-bound, sharing weight reads does not help.

Sushi records added rows at about 3 ms ([Sushi assessment][sushi]). Its row kernels hold one output's weights in registers and stream every row through them, and each is byte-tested against its own one-row form. On Lily the one-row form is a different kernel body, and the attempts above show the two cannot be made to agree bit for bit after the fact. Getting there means replacing the serial kernels as well, which changes the serial reference bits and needs its own precision qualification. That has not been attempted. The official-pack path already compiles some kernels in a scoped safe math mode to match stock order ([hist], M11), which is the nearest precedent.

#### Host round trips inside a round

Measured together with GDN replay against the unmodified base, alternating sessions ([merged timing][merge]): MTP-off unchanged, greedy MTP arms +17 to +18 percent, sampled MTP arms +9 to +15 percent, every default output hash equal.

| Change ([host round trips][host]) | Effect |
| --- | --- |
| Wait for draft and verify passes by polling instead of a blocking wait | The largest sampled gain, about 2 ms of verify wall time per round. More than the documented wake latency explains; the cause is not established |
| Run all greedy draft steps in one GPU submission, feeding each step's argmax to the next, and read back only token ids | Draft phase 1.80 / 3.54 / 5.26 ms to 1.34 / 2.44 / 3.57 at 1 / 2 / 3 drafts in screening |
| Vectorized greedy sampler | Greedy host sampling per round 1.13 to 0.10 ms |
| Encode the next draft step and the verify graph while the previous pass runs | 0.1 to 0.3 ms per round with the next row |
| Read verify rows in place, and skip rows after the first rejection | Included in the row above |

One trap from this work: GPU comparisons flush subnormals, so a device argmax can differ from the host rule when the maximum is zero or subnormal. The chain flags that case and reruns the step on the host.

#### Draft depth

Depth interacts with how predictable the text is, how it is sampled, and how long the context is. Recorded cases, each against the alternative named:

| Workload | Result | Source |
| --- | --- | --- |
| 512 input, top-k 20, one draft against two | +4.3 percent decode, 12 of 12 blocks | [reconciliation][recon] |
| Same sampler, held-out corpus and seed, one against two | +20 percent decode | [reconciliation][recon] |
| 4,096 input, nucleus only, one against two | -6.0 percent, 0 of 12 | [reconciliation][recon] |
| 512 input, greedy, one against two | -0.4 percent, noise | [reconciliation][recon] |
| Natural coding response, top-k 20, one against two | -11.4 percent decode, 0 of 4; two-draft acceptance about 91 percent | [long coding][longcode] |
| Natural planning response, one against two | +1.0 percent, noise | [long planning][longplan] |
| 4,096 input, top-k 20, three drafts against off | -13.8 percent, 0 of 8 | [llama.cpp assessment][assess] |
| 512 input, nucleus only, two drafts against off | -11.8 percent, 0 of 8 | [llama.cpp assessment][assess] |
| Greedy prose, base build: one, two, three drafts against off | -4, -12, -28 percent | [merged timing][merge] |
| Greedy prose, merged build: one, two, three drafts against off | +13, +3, -17 percent | [merged timing][merge] |
| Prose, merged build, matched windows at 4,096 to 32,768 input: one, two, three drafts against off | Greedy +7 to +11, -4 to 0, -18 to -14 percent. Sampled +10 to +19, -5 to +1; three not run | [schedule gate][gate] |
| Code, same sessions | Greedy +29 to +30, +39 to +42, +39 to +42 percent. Sampled +22 to +28, +26 to +34; three not run | [schedule gate][gate] |
| 24 natural requests through the chat template, 1,024-token responses, two drafts against off | Coding +36 to +38 percent, planning +14 to +15, prose +20, greedy and sampled alike; acceptance 90 percent on coding and 71 to 72 on the others | [schedule default][sched] |

Two attempts to change the default from two drafts to one were adopted and then reverted. The first passed its functional checks and a 5 percent tolerance, then missed a preregistered 5 percent benefit gate and a guard cell. The second, narrowed to one sampling profile, passed every gate it set and then lost by 11 percent on a natural coding response. The held-out cell that favoured one draft had about 51 percent acceptance; the coding response had about 91.

A third change, from two fixed drafts to a per-round planner, was adopted on 2026-11-20 and is described under Adaptive depth.

Lessons for the next family:

- Compare a candidate default against the current default, not against off. One draft against off looked like +9.9 percent; against the two-draft default it was +4.3.
- Then compare the current default against off as well. Two fixed drafts survived both reverted changes and measured 0.95 to 1.01 times MTP off in every matched prose cell.
- Measure both the model's own writing and text it did not write. Continuing a novel, two drafts gave nothing; writing prose in answer to a request, the same depth gave 20 percent. A response the model composes is more predictable to its own draft head than a corpus is, so a corpus understates acceptance on chat traffic and a chat request overstates it for continuation, quotation and retrieval.
- Set guard cells across samplers and context lengths before measuring, and include a natural long response. Short fixed sequences did not predict it.
- Neither acceptance nor row count alone predicts the result. One draft does less speculative work; two drafts pay fewer round overheads.
- When row cost falls, the best depth moves. One draft on prose went from a loss to a gain between the base and merged builds without any change to the drafting.

#### Adaptive depth

Each round the planner picks the depth that maximizes `E(m) / cost(m)` from a running acceptance estimate per position and a static cost table measured on the build ([depth policy][depth]). It is a pure function of acceptance history, remaining budget and sampling mode, never of the clock, so requests reproduce.

Since 2026-11-20 it is the default, with prompt lookup, for a text request with MTP on that names no schedule on the pinned image-loaded artifact ([schedule default][sched]). `"mtp_policy": "fixed"` and `"mtp_lookup": false` on a request, or `STARK_LILY_MTP_ADAPTIVE=0` at load, restore the fixed schedule. Every other artifact keeps fixed depth and takes `mtp_policy: "adaptive"` as an option.

The gate that made it the default ([schedule gate][gate]) gave every arm the same corpus windows and seeds and took the median rate against MTP off in twelve cells: prose and code, greedy and sampled, at 4,096, 16,384 and 32,768 input tokens.

| | One fixed draft | Two fixed drafts | Planner | Planner with lookup |
| --- | ---: | ---: | ---: | ---: |
| Geometric mean of the twelve cells | 1.193 | 1.155 | 1.215 | 1.218 |
| Lowest cell | 1.070 | 0.952 | 1.060 | 1.061 |
| Prose cells | 1.07 to 1.19 | 0.95 to 1.01 | 1.06 to 1.13 | 1.06 to 1.14 |
| Code cells | 1.22 to 1.30 | 1.26 to 1.42 | 1.21 to 1.44 | 1.23 to 1.42 |

- It does not beat the best fixed depth for a workload; it avoids the wrong one. It matches two or three fixed drafts on code and stays 6 to 14 percent above MTP off on corpus prose, where two fixed drafts give nothing.
- On natural responses through the chat template it is level with two fixed drafts: 0.989 to 1.017 of it in six cells of coding, planning and prose, greedy and sampled, and 15 to 38 percent above MTP off ([schedule default][sched]). That is where both earlier default changes lost, and it is also where the planner has nothing to add, because acceptance is high and it chooses two drafts in 59 to 98 percent of head rounds. Its gain is on text the model finds hard to predict. In the fixed-corpus benchmark it read 113.2 tok/s on code against 113.3 to 116.9 at three fixed drafts ([merged timing][merge]).
- Its exploration costs something where one depth is right throughout. On prose it ran one draft in 72 to 89 percent of head rounds, two in 7 to 26 and none in 1 to 13, and one fixed draft beat it by 5 to 7 percent on sampled prose at 4,096 and 16,384 tokens.
- Input length changed little. The cost table measured at 4,096 tokens chose the same depths at 16,384 and 32,768.
- The draft limit matters little. The desktop app requests two drafts, so there the planner works under a limit of two: 1.193 against 1.206 at a limit of three over eight cells, with the gap in sampled code (1.22 against 1.32 at 4,096 tokens).
- The cost table must be re-measured whenever round costs change. With a table from the base build, the planner priced a one-draft round on prose as not worth taking and tied MTP off; after re-measuring it gained 9 to 10 percent.
- Seeded sampled output under the planner depends on its constants, so each table has an identifier and the response reports it as `stats.schedule.cost_table`. Making the planner the default changed seeded sampled output of MTP-enabled requests and left greedy output alone.
- A zero-draft round should be an ordinary serial step. The first design used a one-row verify, a head advance and a finish pass and cost 2 to 3 ms more per token.
- Things that mattered in tuning: how fast low acceptance moves the estimate at the start of a request, how often an abandoned depth is probed again, and a hysteresis band small enough that a 4 percent better depth is taken.
- Not established: sampled settings other than temperature 1 with top-k 20; multi-turn and tool-calling sessions; responses longer than 1,024 tokens; any other artifact.

#### Prompt lookup

A 4-gram index over the prompt and committed output proposes the tokens that followed the last match, as point-mass drafts with no draft-head forward. Verification and rejection are unchanged, so it stays exact. It is on by default with the planner on the pinned artifact and an option (`mtp_lookup: true`) elsewhere.

- Greedy copy-heavy input: +8 to +10 percent over the same depth without lookup, with 98 to 99 percent of lookup drafts accepted in screening. On the benchmark's code corpus, +5 to +9 percent over the adaptive arm.
- Sampled at temperature 1: mixed. The first screening found it neutral to negative, the continuation leaving the copy almost at once. A copy-heavy sampled arm later gained 7 percent ([merged timing][merge]), and in the schedule gate it was level with the planner alone on sampled code at 4,096 and 16,384 tokens and 4 percent ahead at 32,768.
- Prose and non-repeating code: nothing, and no measurable cost. On the gate's corpora lookup ran in 0 to 15 percent of rounds and moved the twelve-cell score by 0.3 percent.
- A round that skips the draft head must still write the head's cache row for the pending token, or later sampled output depends on earlier requests.

#### Raising the draft limit

Lily's limit is 3. A plan for 8 is in the [depth policy record][depth]. It is deferred: at Lily's row cost a fourth head draft does not pay, and the GDN input projection kernel exists only for 1 to 4 rows with a cost that climbs at rows 3 and 4. Deeper limits would mainly help lookup rounds, which pay no draft steps.

The desktop app is bounded lower still. Its paged-residency memory profile for the pinned artifact allows two drafts, and raising that needs a memory qualification ([schedule gate][gate]).

### 6. Prefix and state reuse

This does not change tok/s. It removes most of the prefill before the first token of a follow-up turn, which is what chat and agent users wait for.

| Attempt | Result | Decision |
| --- | --- | --- |
| Keep attention KV for the previous prompt's full chunks, on a dense model ([prefix reuse][prefix1]) | First token on follow-up turns 6.3 to 13.2 times sooner (7.57 to 0.85 s on turn 2); cold turn unchanged | Adopted |
| On the hybrid, also snapshot recurrent state, convolution windows, n-gram history and the draft-head hidden state at a 4,096-token chunk boundary ([hybrid prefix reuse][prefix2]) | First token 9.9 to 10.9 s down to 1.4 to 2.3 s on follow-up turns (4.8 to 6.9 times) | Adopted |
| The same in a live agent session ([live session][prefixlive]) | 79 percent of prompt tokens reused; first text 11.3 s cold, 1.9 to 3.6 s after | Observation; no same-session baseline |
| Halve the physical prefill chunk to 2,048 for a finer snapshot grid ([agent grid][fsagent], [fresh grid][fsfresh]) | Faster on all 38 requests that gained reuse, median 57 percent; fresh prompts 12 to 16 percent slower at 8K to 32K | Rejected as default |
| One extra recurrent-state snapshot captured inside the pass at a midpoint, chunk size unchanged ([midpoint][midpoint]) | At 12K, append-only follow-ups 40 to 46 percent sooner to first token; a 17-token suffix 4.95 to 0.11 s; capture plus follow-up 19.4 percent faster; fresh capture cost +2.6 to +5.5 percent | Adopted, on by default for the pinned artifact |

What generalizes:

- A recurrent model cannot rewind, so it can restore only where state was captured. The spacing of capture points sets how much a follow-up reprocesses: 2,500 to 4,200 tokens on a 4,096 grid.
- Do not buy a finer grid by shrinking the prefill chunk. The chunk size also sets cold prefill efficiency. Capture state inside the pass instead.
- One snapshot is about 110 MiB by the feasibility record and about 149 MiB by the release record; the two figures are not reconciled. One at every 2,048 boundary to 32K was estimated at about 1.6 GiB, which is why there is one midpoint and not many.
- Key the snapshot on everything that affects state: context size, model instance, cache scope and whether MTP is enabled. Clear it on failure, cancellation, image requests and restart, and never persist it.
- Leave out what does not affect state, and prove it. The round schedule is not in the key, because the snapshot is captured during prefill before any round runs. With the planner and lookup on, every prefix hit equalled the same request without reuse, in 67 of 67 cases as options and 77 of 77 as the default, and switching schedule between requests kept the entry ([selector prefix reuse][selprefix]).
- Qualify with three workers per case (cache on, cache off, baseline), compare hashes, check reuse counts against a simulation of the rule, and exercise every invalidation path after a warm hit.
- When the candidate's default differs from the baseline's, sampled cases have no baseline counterpart. Compare them with the uncached candidate only, check that the reply names the schedule it ran, and add cases that opt out of the new default: those must equal the baseline, sampled included.
- A single snapshot is displaced by any unrelated request. One 8,714-token request took 4.93 s to first text after displacement and 0.54 s on repeat ([speed review][fsreview]).

### 7. Prefill

Much less has been tried here than on decode, and the kernel candidates tried on Lily's cold prefill were not adopted.

- Cold first token on Lily was 4.4, 6.4 and 17.4 s at 8K, 12K and 32K input with two drafts ([fresh grid][fsfresh]). At 88K input, prefill is about 44 s of a 62 s request, so decode gains shrink at request level: +5.4 percent decode became 2.4 percent of request time.
- A fused grouped gate/up kernel for prefill, exact as a component, was 11.6 percent faster at 4,096 tokens and 5.4 percent slower at 8,700. Over six pairs the sign followed launch order. It stays off by default ([kernel follow-up][fskernel]).
- Raising the concurrent encoder from 16 to 64 rows gained up to 2 percent in prefill alone and lost in complete requests ([pilot][pilot]).
- On a dense model, prefill time fit about 0.22 s per chunk plus 17 ms per 1,024 tokens of existing prefix, with matrix products 55 to 60 percent and attention about 40. Host encoding was 21 to 26 ms in total, so pipelining chunk submission has nothing to win ([prefix reuse][prefix1]).
- A grouped attention kernel gained 3 to 4 percent of prefill and was not shipped: the scorer shares that kernel, and its bit-exact parity would break. Check who else uses a kernel before changing it.
- No stage profile of ordinary Lily prefill exists yet.

### 8. Startup and memory

| Attempt | Result | Decision |
| --- | --- | --- |
| Reuse validated payload digests, keyed on each file's device, inode, length and change times ([image startup][startup]) | Native readiness 10.5 to 11.9 s. Hashing the 55 shards once takes 11.3 s | Adopted |
| Faster full hashing on every start: assembly SHA-256, four shards at once | Ready at 31.8 s | Rejected |
| Vectorized transpose while loading affine planes ([hist], U05) | Process to ready 20.5 to 18.1 s | Kept |
| Plan memory from the non-paged payload, the state inventory counted once, a working set for the paged table and a reserve ([guided memory][mem]) | Plan 87 GiB; peak sampled footprint 72.9 GiB at 32K with two drafts | Adopted |
| Preflight on available memory with a cutoff for the pinned artifact ([startup memory][startmem]) | The full 88.7 GB estimate had rejected a host with about 85 GB available | Adopted for that artifact |

- Count a paged or memory-mapped table by its working set, not as resident. The first plan counted the 29.8 GiB n-gram table as resident and came to 111 GiB.
- Do not multiply an inventory that already counts overlapping owners.
- Admission limits and sampled memory do not prove a working-set guarantee. Record which one a number is.

### 9. Idle wake, cadence and host conditions

| Finding | Detail | Decision |
| --- | --- | --- |
| The first request after an idle pause is slow ([idle wake fix][idle]) | First sample after 10 s idle 1.07 s against 0.30 s. An empty GPU queue signal every 500 ms while idle removes 0.78 s. Decode is unchanged. While idle, reported GPU power rises from 0.003 to 0.006 W and worker CPU to 0.03 percent of a core | Adopted |
| The penalty depends on the backend ([cross-backend idle wake][idlex]) | About 0.8 s on the hybrid backends, 0.08 to 0.13 s on the dense 27B | Adopted for each |
| Request cadence moves decode rate ([MoE projections][moeproj], [sustained diagnosis][sustained]) | The same binary measured 95.4 tok/s when each request followed a 10 s quiet wait and 85.5 back to back. Over eight consecutive requests decode fell from 95.5 to 82.2 as the GPU clock fell from 1620 to 1488 MHz | Unresolved; the limiter is not identified |
| High Power mode, a thread priority override, a per-request activity assertion, a dedicated feedback queue ([hist], H03 to H07) | No measurable benefit from any | Not adopted for speed |

Lower decode rate and lower wall time can occur together, for instance when the first token arrives sooner. Report first token, decode rate and request time separately.

### 10. Benchmark conditions and other engines

Published rates are not comparable until the conditions match. The 2026-11-19 comparison found Sushi slower at serial decode (about 62 against 73 tok/s) and faster in its headline number (146.2 on code), because of cheaper and deeper speculative rounds measured greedy on a regular synthetic code corpus ([Sushi assessment][sushi]).

- Greedy against sampled, and code against prose, each move the result more than most optimizations. Sushi's own figures at one context size are 146.2 on code and 79.5 on prose. Stark's merged build records 113 to 123 on code and 86 to 87 on prose at its best settings.
- `ignore_eos` is length control, not a speed trick. It was a no-op on this corpus.
- A benchmark's second and third samples may reuse a cached prefix by design. Record reused input tokens.
- Hardware and speculation settings in a published figure have to be read from the source. One pack's advertised 157 tok/s was on different hardware with speculation on.
- [`scripts/benchmark_lily_fixed_corpus.py`](../scripts/benchmark_lily_fixed_corpus.py) reproduces Sushi's corpus generator and request ladder, so the conditions can be matched ([fixed-corpus benchmark][bench]). Even then it is not a matched comparison: the packs differ.

### Match the source call geometry before promoting a precision change

Inkling’s packed source graph reached 772 tok/s in an uncached 4K worker session after matching the source’s BF16 boundaries, global matrix attention, route weight normalization, and the different MXFP8 gate/up and down lane layouts. A single fused FP32 convolution operation changed one BF16 cache word and propagated into hundreds of thousands of downstream words. Compare the first unequal intermediate before changing another kernel. Record the scheduler’s effective prefill widths alongside forced trace widths and capture the final prompt token’s seed logits. A continuation-only comparison can include errors from the singleton implementation after a matched prefill. Source-aligned wide batches and singleton repairs reproduced all 16 logit vectors from four stock-schedule captures. The final long paired session measured 683–685 tok/s prefill medians with 16 wins, about 23% lower request time and about 6% lower decode throughput within budget. [Inkling source precision experiments](model-experiments/2026-11-21-inkling-mini-packed-prefill.md#20261121-inkprefill-08--source-arithmetic-and-embedded-library)

The stock Inkling 4K profile retained the two bulk commands, the three-row tail and the final seed. Grouped NVFP4 gate/up and down products used 2.78 s of 5.83 s instrumented command GPU time. Global mask generation used 0.017 s. Measure live expert tile utilization and exactness replay costs before rebuilding the grouped kernel. On the same restored 4K cache the source singleton took 2.39 ms more wall time than the legacy arithmetic at the paired median. The source router projection added 2.28 ms of instrumented GPU time, while selection and normalization were slightly faster. Screen a one-row projection specialization against the pinned arithmetic. [Stock stage profile](model-experiments/2026-11-21-inkling-mini-stock-profile.md)

The next Inkling control found 89% row fill on active 32-row runs at stock widths. Actual replay affected at most 0.33% of down output K-block entries. Removing replay saved 2.5–4.0% in the shadow timing, and bypassing the guard as well saved another 33–35% of exact shadow time with fewer telemetry updates. Test guard work before rebuilding the replay loop. Shared scales and a vector-loaded singleton router both passed short and stock 4K controls. The shared-scale worker trial improved 4K prefill by 1.61–1.77% at paired medians. The exact one-row router saved 0.92 ms in the projection stage and improved worker useful decode by 1.64–2.52%, with all eight worker pairs faster. Both were opt-in at the time of the screen. [NVFP4 and singleton router study](model-experiments/2026-11-21-inkling-mini-nvfp4-router.md)

Widening the Inkling NVFP4 run tile from 32 × 64 to 64 × 32 let four SIMD row fragments share decoded weight columns. Reusing the row and column metadata on broad guard misses preserved the replay rules. The exact stock 4K profiles removed 319–380 ms from the two NVFP4 matrix stages. The 64 × 64 and 128 × 32 shapes gave weaker whole-target screens. Combining the two source MXFP8 route loops retained exact output and slowed the down product. The 64 × 32 worker cut paired 4K prefill time by 5.52% greedy and 9.27% sampled, with output hashes preserved. The worker medians of 800.28 and 763.88 tok/s leave 1.03–1.28 s to remove at 4K. [Wider NVFP4 tile study](model-experiments/2026-11-21-inkling-mini-nvfp4-run-tiles.md)

Precompute immutable MXFP8 row norms at load and measure certification density before allocating a replay list. On the original Inkling checkpoint the initial scalar queue overflowed every stock bulk projection. One mask per weight column let four ambiguous routes share the recurrence and avoided that limit. Compensated K32 partials and vector loads reduced the MXFP8 part of two instrumented 4K comparisons to 0.777–0.814 s, with 19–25% of outputs replayed. A physical audit matched 3.78 billion certified BF16 words against the pinned recurrence. Eight matched cold worker pairs per mode cleared 1,000 tok/s, with medians of 1,058 greedy and 1,054 seeded sampled. Combining NVFP4 lane-scale sharing with the 64 × 32 tile gave mixed whole-target timings and was dropped. [MXFP8 certification and replay](model-experiments/2026-11-21-inkling-mini-mxfp8-tensor.md)

Promoting the Inkling tile and tensor combination kept 4K prefill at 1,020 greedy and 1,018 seeded sampled tok/s with all experiment flags unset. Check the loaded kernel selectors as well as pinned parity. A successful source fallback can pass the numerical gate without exercising the faster kernels. [Default-path verification](model-experiments/2026-11-21-inkling-mini-prefill-default.md)

The default one-row Inkling router saved 0.884 ms across the 44 NVFP4 projections at an identical 4K cache. The flags-unset worker's paired useful-decode changes were +3.15% greedy and +2.93% seeded sampled. Measure both the restored-state projection and sustained worker decode when promoting an exact singleton path. [Default-router verification](model-experiments/2026-11-21-inkling-mini-decode-default.md)

An Inkling NVFP4 block has one E4M3 activation scale across 16 FP4 values. Proving the inner products and all partial sums fit on an exact FP32 grid allowed four vector dots without changing the scaled lane recurrence. At cold 4K the opt-in worker reached 38.66 greedy and 36.75 sampled tok/s. The exact four-row grouped verifier still took 55–60 ms, and the native multi-mask draft had input-dependent acceptance. Measure conditional acceptance and complete round cost to choose a depth. [Decode and draft-depth screens](model-experiments/2026-11-21-inkling-mini-decode-60.md)

One SIMD exact top-eight selector removed the source singleton's serial scan and preserved its lowest-ID tie handling and F32 normalization. Same-cache selection went from 2.61 to 0.63 ms. With vector NVFP4 in both worker arms, the opt-in selector reached 44.24 greedy and 41.08 sampled tok/s in four cold 4K pairs per mode. [Parallel selector screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-04--parallel-exact-expert-selection)

With two to eight verifier rows, share decoded NVFP4 weights inside each same-expert run and keep each route’s original scaled FMA recurrence. At four rows the exact vector matvec cut 4K verification from 54–60 to 41–44 ms. The result still exceeds the two measured draft budgets at depth three and has only been qualified as a diagnostic. [Short-run verifier screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-05--nvfp4-weight-sharing-across-short-verifier-runs)

Replaying Lily’s 4-gram agreement rule on original Inkling’s eight stock 4K outputs offered drafts at 15 of 1,020 greedy steps and seven of 1,020 sampled steps. The opportunity was too sparse to justify building lookup for the 60 tok/s target. [Lookup eligibility screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-06--screen-lilys-prompt-lookup-on-inkling-continuations)

An Inkling one-row dense fusion removed 256 dispatches but saved only 0.17 ms per plain token command in the matched 4K profile. Exact FP16 NVFP4 coefficients and a narrower TensorOps router were slower. Keep component savings separate from useful throughput. [Singleton kernel screens](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-0709--three-single-token-kernel-screens)

For unrestricted seeded sampling, bound the mass omitted from a sorted prefix and take the original draw only when the full threshold interval falls in one token. Keep the original full sort and RNG state for fallback. On Inkling 233/256 draws certified and the CPU sampler fell from 0.865 to 0.291 ms in one instrumented pair. The worker screen measured +4.28% useful decode. [Certified sampler screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-10--certified-seeded-draws-with-a-sorted-prefix)

Initialize the draft cache from the same target precision and chunk schedule used for scoring. On original Inkling two pinned 4K cases with exact packed draft taps left 14.34 and 40.83 ms for depth-three verification and finishing. The wider spread changes when to enable speculation. [Source-precision draft screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-11--source-precision-dspark-acceptance)

Reuse the exact singleton selector across short grouped verifier rows. This removed about 2.1 ms from the four-row command while keeping exact logits and committed state. The favorable 4K input still has about 0.7 ms to recover before paying for rollback and host work. [Verifier routing screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-12--parallel-selection-in-the-short-verifier)

On Inkling, save only the local KV slots and convolution histories a short verifier can overwrite. Defer draft commits until acceptance. The exact four-row transaction needs the anchor convolution input because the history wraps at three rows. The worker reached 51.50 tok/s greedy at three proposals and 42.22 tok/s sampled at two, with matching output hashes and verified early-stop and cancellation handling. [Transactional worker screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-13--exact-short-round-rollback-and-worker-integration)

Existing packed local attention failed short-verifier F32 parity on Inkling. Sharing absolute-key reads while preserving each query’s singleton shard and chronological updates saved 0.52–0.60 ms per four-row command with exact logits and rollback. A fused grouped NVFP4 gate/up projection was slower. Keep shared KV opt-in while measuring useful throughput. [Verifier kernel screens](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-16--share-local-kv-reads-in-the-exact-short-verifier)

Lossless BF16 exponent packing reduced estimated rotating weight reads by 24.1% on Inkling, but GPU time rose from 5.41 to 7.30 ms. Test reconstruction overhead with rotating real banks before adopting compressed storage. [Lossless BF16 screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-15--lossless-bf16-attention-weights-on-metal)

Tabulating the F32 NVFP4 activation terms by exact BF16 input saved about 0.04 ms per four-row Inkling verifier command. The lookup kept exact outputs and rollback. It was retired because the full-command gain was small. [Gate nonlinearity table screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-18--table-the-exact-source-nvfp4-nonlinearity)

Match the NVFP4 short-verifier kernel width to the live expert run. Specializing the two- and three-route cases eliminated idle accumulators and per-route guards, saving 0.69–0.91 ms per four-row command on two original Inkling 4K inputs with exact logits and rollback. The specialization is opt-in. [Exact run width screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-19--specialize-the-short-nvfp4-product-by-expert-run-width)

Reusing Lily’s depth planner with measured Inkling round costs kept exact output. Four 4K pairs per sampling mode gave no sustained improvement over fixed depth three greedy and depth two sampled. Keep tuning inputs and costs tied to the artifact and target sampler. [Adaptive depth screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-20--replay-lilys-depth-planner-with-inkling-round-costs)

Decoded FP4 times E4M3 activations fit exactly in binary16. On original Inkling the four-SIMD verifier widened those coefficients before its F32 dots and saved 0.47–0.67 ms per four-row command. Different threadgroup sizes and polling gave little or inconsistent benefit. [Exact ingress and launch screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-21--store-exact-verifier-nvfp4-ingress-in-binary16)

For DSpark, shortlisting two base logits per SIMD group before the exact rank correction saved about 0.8 ms at three proposals. Fixed contiguous vocabulary caps removed most first-mask acceptances on both 4K inputs. Check proposal coverage before shrinking the draft vocabulary. [Draft shortlist screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-22--rerank-a-smaller-set-of-dspark-vocabulary-candidates)

Reusing binary16 activation loads across two adjacent NVFP4 output columns kept exact Inkling F32 readouts but added 1.22–1.58 ms when applied to both expert projections in the four-row verifier. [Adjacent column screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-23--share-nvfp4-ingress-across-adjacent-output-columns)

One TensorOps call per exact K16 block kept Inkling’s per-lane recurrence and matched every verified F32 row. The cooperative inputs required a 16-row physical tile. The fastest tested configuration took 107–122 ms for four verifier rows versus 37–40 ms for the vector control. Retire this layout for short verification. [K16 TensorOps screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-24--preserve-source-accumulation-around-tensorops-k16-dots)

Allowing shorter lookup suffixes with consensus among earlier continuations raised Inkling’s eligibility. With zero charged lookup cost, the strongest replay estimates over serial were +2.23% greedy and +1.20% sampled at cold 4K. That was too small for the throughput goal. [Short-context lookup replay](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-25--replay-short-context-lookup-with-past-continuation-agreement)

For the short Inkling verifier, sort up to 32 routes in one SIMD group and dispatch only live expert runs. This saved 0.28–0.29 ms at four rows with exact output and rollback. [Compact run screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-26--compact-live-nvfp4-runs-in-the-short-verifier)

Batch exact convolution finishing in token order within each channel. Combining the expert reduction, convolution, residual write and anchor export saved 0.55–0.56 ms per four-row Inkling command. The attention-side convolution and residual fusion added another 0.54–0.56 ms on those inputs. [Finishing fusion screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-27--batch-convolution-and-residual-finishing-across-verifier-rows)

On four held-out sampled Inkling trajectories, rank-boundary calibration did not improve on ordinary argmax proposals. Top-16 draft distributions and full vocabulary gave similar exact-rejection estimates around 56.4 tok/s before correction overhead. [Seeded calibration and sparse rejection screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-28--screen-held-out-seeded-calibration-and-sparse-rejection)

A three-comparison scalar FP4 threshold search preserved midpoint ties and saved 0.17–0.20 ms per four-row Inkling verifier command. A four-coefficient vector version was slower. The scalar search runs inside the opt-in binary16 ingress. [Quantization threshold screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-29--shorten-the-exact-fp4-activation-threshold-search)

Profile the live short verifier with undo copies and final F32 readout. Use the plain command to set the budget, folded timestamps for attention and expert totals, and a separate timing arm for finer expert stages. Four-row Inkling commands took 35.75–38.13 ms before draft or sampling. NVFP4 products were the largest region at 9.22–10.95 ms in the detailed arm. Router projection still took about 3.20 ms. [Live transaction profile](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-30--profile-the-live-exact-verifier-transaction)

Sharing the global K/V cache read across three query heads preserved per-head shard order on Inkling. Three-row verification saved 0.59–0.64 ms in eight paired commands per 4K input. Four cold worker pairs per mode showed paired gains of 2.41% greedy and 2.20% seeded sampled with exact output hashes. The worker path is opt-in. [Global K/V screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-31--share-global-kv-reads-in-the-exact-short-verifier)

Apply the vector-loaded singleton router's aligned weights to the two-to-four-row source verifier with the same tensor shape and split accumulation. On Inkling this saved 0.83–1.14 ms in stock 4K native comparisons, with exact F32 output and rollback. With global K/V sharing already on, the router moved paired worker throughput by +0.78% greedy and +1.91% seeded sampled. The sampled worker still needs 6.23–8.73 ms per round at the observed acceptance and fixed remaining costs. [Short verifier router screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-32--vector-load-the-exact-short-verifier-router)

Staging Inkling's short-verifier binary16 activations across eight- and sixteen-column tiles preserved exact outputs. Four-row 4K commands slowed by 0.79–1.71 ms with staging. The wider direct controls also slowed. Separate tiling and staging in matched comparisons before investing in an activation reuse path. [Wider NVFP4 tile screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-33--stage-nvfp4-ingress-across-wider-output-tiles)

Bound adaptive proposal depth with the actual native proposer on held-out trajectories. Inkling's exact seeded replay gave a 55.63 tok/s median at fixed depth two and 55.72 with a future-draw policy, before host and finish costs. Hindsight could reach 60.85–77.13 across inputs. The large hindsight gap comes from knowing future acceptance. The tested online signals leave the practical path short of budget. [Sampled depth bound](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-34--bound-adaptive-depth-on-exact-seeded-trajectories)

Separate weight-address reuse and arithmetic from exact kernel timing. Inkling's verifier shadow recovered 0.06–0.45 ms from repeated weight addresses and 4.83–5.57 ms when a checksum replaced float work with the reads intact. An exact affine FP4 decoder saved 2.06–2.30 ms in four-row uninstrumented transactions. [NVFP4 read and decoder screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-35--isolate-packed-expert-reads-and-simplify-exact-coefficient-decoding)

On the same Inkling verifier, exact FP16 element products removed 1.25–1.46 ms per four-row command. The affine F32 decoder removed 1.68–1.94 ms in those pairs and stayed the faster choice. Cold worker medians reached 60.82 tok/s greedy and 49.22 seeded sampled, with +5.57% and +5.62% median paired gains and exact outputs. Keep product precision and accumulation precision separate in comparisons. [Exact FP16 product screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-36--compare-exact-fp16-element-products)

Profile the DSpark command separately from the target verifier. At two and three proposals on stock 4K Inkling, the eight-row draft block used 2.12–2.13 ms and the shared vocabulary projection used 2.06–2.07 ms. The remaining shortlist rank and winner work used less than 0.09 ms. Exact proposals matched plain runs. [DSpark proposal stage profile](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-37--profile-the-stock-4k-dspark-proposal-command)

A smaller and wider DSpark vocabulary tile preserved every checked F32 score and proposal. The strongest shape saved 0.02–0.05 ms of the ordinary draft command on Inkling. The finer draft profile attributed 0.34 ms to attention and 1.14–1.16 ms to the FFN. Use those stage limits before investing in draft-side geometry. [DSpark vocabulary tile screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-38--shrink-and-widen-the-dspark-vocabulary-projection-tile)

Submit a GPU proposal command before encoding the dependent verifier. Read the proposed IDs directly from device memory and keep both commands on the same queue. Inkling's two-proposal native 4K pairs saved 1.49–3.03 ms, with exact scores and committed state across normal and wrap controls. A sampled worker screen gained 0.40% at the paired median and the four-pair repeat gained 3.82%, at 52.87 tok/s. Compare combined wall time because the separate spans overlap. [Overlapped verifier screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-39--encode-the-verifier-while-the-draft-command-runs)

Putting the Inkling draft and verifier in one command buffer preserved exact state and cost 0.38–1.80 ms more than the queued path at two proposals on stock 4K. The queued path already overlaps draft execution with CPU verifier encoding. [Single-buffer screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-40--compare-one-command-buffer-with-queued-verification)

Specializing the existing dense dot to three live rows saved 0.14–0.18 ms in a native Inkling verifier command. Keep the four-row noise control when screening such a change. [Live-width dense screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-41--specialize-dense-gemv-to-the-live-verifier-width)

Time the raw packed-to-tensor path before implementing exact replay. On Inkling M5 Max, the three-row affine NVFP4 vector shadows used 7.29–7.74 ms and the fastest staged K32 tensor shadows used 15.67–16.82 ms. Retire the staged layout before paying for a certificate. [NVFP4 tensor budget](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-42--measure-the-raw-k32-nvfp4-tensor-budget)

Folding each exact NVFP4 K16 inner product into one horizontal reduction saved at most 0.08 ms in the three-row product stage. Fully unrolling the source-lane loop added about 2 ms to three-row Inkling verifier commands. Preserve the affine vector path. [K16 accumulation screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-43--fold-k16-dot-reductions-and-specialize-the-lane-loop)

Removing threadgroup staging did not recover the K32 tensor budget. Three-row register-fed NVFP4 shadows used 59.78–61.75 ms on Inkling against 7.10–7.37 ms for the vector control. The packed per-coordinate decode remains inside the measured tensor kernel. [Register-fed tensor screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-44--time-register-fed-k32-nvfp4-tensor-tiles)

Charge alternate proposals their verifier rows before designing tree state. The sampled Inkling replay estimated 56.17 tok/s for the best fixed chain and 52.61 tok/s for two ranked roots with a primary child at representative costs. Even a free extra row left that measured fork below 60 tok/s. [Draft branch replay](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-45--replay-exact-seeded-draft-branches)

Quantize a draft-only projection when target verification can still use the original weights. On four Inkling 4K trajectories, BF16 activations with INT8 vocabulary weights saved 1.03–1.20 ms per multistep draft and changed the matching teacher prefix total by one token at each depth. Per-row INT4 lost more accepted proposals and gave a lower modeled rate than INT8. The repeated cold worker reached 53.40 tok/s seeded sampled with a 6.04% median paired gain and exact output hashes. Record the 440.36 MiB bank in startup and request admission. [Draft vocabulary quantization](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-46--quantize-the-multistep-dspark-vocabulary-head)

The local Metal 4 compiler accepts signed INT8 K16 products with INT32 results. The Inkling scalar screen preserves F32 per-block FMA bits by combining the two E4M3 scales. Time real routed kernels including packing and source-order accumulation before budgeting an integer verifier. [Integer block feasibility](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-47--check-exact-integer-nvfp4-block-arithmetic)

Test exact integer arithmetic with the activation pack and original weight layout in place. Inkling's exact packed pairs added 0.58–0.60 ms to three-row stock 4K verifier commands. Normalized FP16 and direct INT32 products cost more. Keep the affine vector kernel for these routes. [Real-route integer screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-48--time-exact-packed-integer-expert-products)

Separate projection fusion from weight precision. With the INT8 vocabulary head held fixed, the Inkling DSpark feed-forward INT8 arm reduced two-proposal command time by 0.49–0.63 ms. The fused BF16 control saved at most 0.04 ms. The INT8 worker's sampled median paired change was +0.04%. Archive the additional bank at this scope. [Draft feed-forward experiment](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-49--fuse-and-quantize-the-multistep-dspark-feed-forward-projections)

Retest polled waits with the target-only fence and a repeated fixture. Inkling's 20-microsecond poll saved 0.05–0.06 ms at the three-row stock 4K queued boundary. The larger first-short result included GPU execution drift and vanished on repeat. [Metal wait screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-50--poll-the-short-verifiers-metal-completion)

For small source routers, test direct device BF16 TensorOps while preserving exact accumulation boundaries. Inkling's two native router products retain the 2,048 + 256 split and join separate F32 partials. M16 × N8 saves 1.18–1.45 ms per three-row verifier command with exact routed logits. The sampled repeat gains 2.80% at the paired median and reaches 53.28 tok/s. [Native verifier router](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-51--split-the-exact-verifier-router-into-native-bf16-products)

Measure gathering and scattering when comparing device TensorOps against the short shared-weight GEMV. Three-row Inkling dense attention took 13.32–13.41 ms with the fastest native tile against 5.90–5.93 ms for GEMV. The native arm also changed BF16 projection words. Retire this layout. [Short dense TensorOps screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-52--screen-direct-bf16-tensorops-for-short-verifier-dense-attention)

Time byte-pair coefficient lookup before changing the FP4 reduction. Inkling's exact 1 KiB table saved 1.44–1.50 ms per three-row 4K verifier command on repeat. Keep E4M3 decoding and the F32 dot arithmetic. Sampled worker medians were 56.39 and 56.21 tok/s across two paired screens, with exact seeded outputs. [FP4 pair table](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-53--decode-packed-fp4-coefficient-pairs-with-a-constant-table)

Bound the cutoff predecessor before skipping scalar tail maxima in exact sampling. Inkling's chunked CPU certificate attempt fell from 160 to 84 µs on pinned temperature-one initial rows. All of these rows still fell back to the ordinary sampler. [CPU sampler screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-54--bound-and-scan-the-exact-sampler-tail-in-chunks)

Keep chronological per-channel K/V convolution and source BF16 casts when batching cache writes. Inkling's shared verifier fused cache path removed 336 dispatches and saved 0.56–0.58 ms at three rows. An exact seeded worker repeat gained 2.55% at the paired median and reached 55.70 tok/s. The fused cache path stays opt-in. [Fused verifier K/V](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-55--fuse-causal-kv-convolution-and-cache-writes)

Batch each row's original relative head tiles and gate work into two launches. On Inkling's shared verifier this removed 192 dispatches and saved 0.44–0.50 ms per three-row 4K command with exact logits. Sharing the 64 shard factors behind a threadgroup barrier was slower at three rows. [Verifier attention tail screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-56--batch-relative-tables-and-share-attention-gate-weights)

Measure teacher-forced probability overlap and conditional acceptance before building stochastic rejection. On two original Inkling 4K sample trajectories, three masks estimated 2.37 and 2.97 useful outputs per round. Only one trajectory’s current verifier could fit 60 tok/s after subtracting measured draft cost. [Stochastic feasibility screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-17--measure-stochastic-dspark-acceptance-before-changing-decode)

## Correctness gates

A speed change that is meant to leave output alone has to pass all of these, in this order of cost:

1. Component byte tests on the physical GPU against the one-row or unchanged kernel: real dimensions, edge geometries, zeros, subnormal-heavy inputs, cancellation, offset views and output canaries, over several independent random draws.
2. The real-model control: verify rows, rollback state and a host-serial reference with zero unequal values (`private_checkpoint_mtp_verify_and_rollback_match_serial`, which reads `STARK_TEST_MODEL_PATH`). Its draft-head phase is allowed to differ and reports 248,320 unequal values with no top-1 mismatch. Run the control by name: the prefix `private_checkpoint_mtp_` also selects a test that asserts an official-pack checkpoint and stops on Lily.
3. The output-hash gate: every default arm (greedy and sampled, 0 to 3 drafts) reproduces the baseline hash and the MTP work counters.
4. For a new request option: a greedy request reproduces the greedy baseline hash, and a sampled request gives one hash across repeats.
5. For a change of default: on the same inputs and seeds, the new default path equals the previous build with the option requested explicitly, hash for hash and counter for counter; the opt-out and the kill switch equal the old default; and the frozen baseline arms reproduce with the kill switch set ([schedule default][sched]).
6. For anything that touches state: prefix restoration, image-loaded parity, image transitions, cancellation and reset, per [model onboarding](MODEL_ONBOARDING.md).

Step 1 is necessary and not sufficient. Kernels are compiled with fast math, and the GPU compiler schedules identical per-row source differently depending on the surrounding code. Recorded cases: a kernel whose intermediate representation matched the one-row body per row and still differed in up to 300 of 2,560 outputs on subnormal streams; a variant that passed one random draw and failed the next; and a variant that passed 288 synthetic cases and failed the real model.

Equal hashes show the change is inert, not that it is faster. A precision change needs matching-reference numerics instead, and hash equality is not a quality qualification.

## Measuring

- Use one loaded worker and alternate arms, or alternate sessions in the order base, candidate, candidate, base. Report the median of per-pair ratios, not the ratio of medians. Never subtract medians from different sessions.
- Carry an arm the change cannot affect. MTP-off arms moved by 2 to 16 percent in several screening runs with no code change, which showed the other arms' movement was noise.
- Bracket a long comparison with a repeat of the first configuration. Two identical sessions run before and after the adaptive cells differed by up to 3.6 percent; within one session a worker has drifted 15 percent from its first requests to its last.
- Keep request cadence the same in both arms. Back-to-back requests ran 10 percent below requests separated by a quiet wait on the same binary.
- Run both launch orders. Several candidates changed sign with the order. One was 8.9 percent slower when the baseline ran first and 7.4 percent faster when the candidate ran first.
- Check that both arms did the same work. One sampled MTP request stopped at 127 tokens while its MTP-off pair ran to the 256 cap, which mixed answer length with speed.
- An unchanged worker has ranged from 53.3 to 61.1 tok/s across repeats. Baseline spread of 10 to 22 percent was recorded on busy-host cells.
- Fix the gate before the run. The gates used: a gain at least the larger of 10 percent and the baseline's own range; at least 7 of 8 paired wins; no declared cell more than 5 percent worse; a held-out corpus or seed before production. A small win inside the range is a stop decision.
- Component timing, instrumented stage timing and end-to-end timing are three kinds of evidence. A faster component has repeatedly failed to produce a faster request.
- Sampled arms at different depths generate different text and different lengths. Report throughput and request time separately, and do not rank sampled arms on a few 256-token samples. The schedule gate ranked them on five to nine matched windows per cell.
- Give every arm of a comparison the same inputs and seeds, and change the input and the seed between repeats. Repeating one input and one seed measures timing noise only, because a seeded arm writes the same text every time.
- Do not let a corpus window end where the text ends: the model stops at once and the window is lost. Sampled requests also stop early at long inputs, 4 of 15 on prose at 32,768 tokens. Drop such a window for every arm of that sampling mode, and plan spare windows.
- Set a minimum number of valid samples for each decision cell before the run. If samples are added afterwards, add them to every cell of that kind and write the plan down before looking at them ([schedule gate][gate]).
- Hold the incumbent to the guards the candidates are held to. The two-draft default failed the no-harm guard of the gate that replaced it.
- Keep a copy of each measured binary outside the build directory. The next build replaces it.
- Keep every invalid setup in the record as invalid: a wrong request type, a missing tool on the path, a concurrent build, a contended host.
- Give each experiment its own build target directory. A shared one once carried an experimental core into a supposedly clean worker.
- Only one process can hold a 100 GB model. Serialize model loads through one lock, and never stop a process the experiment did not start.
- Diff tools configured for display produce patches that do not apply. Export with `git diff --no-ext-diff`.

## Checklist for a new family

1. Count per token: command submissions, waits, uploads, bytes read back. Get to one submission and one completion per token before anything else.
2. Settle the precision policy against a reference, and record which pack and per-tensor widths are being run.
3. Time the sampler on a real readout for each sampling mode.
4. Profile one serial step by stage. For the top two stages, run a control that says what binds them.
5. Apply the ceiling rule to every idea. Drop anything whose stage is under about 10 percent, unless a gain of a few percent is worth its risk to the owner.
6. If the family has a draft head: build exact batched verification first, then measure the cost of an added verify row and acceptance by position on at least code, prose, greedy and sampled.
7. Remove host round trips inside the round before working on row kernels.
8. Reduce row cost by batching launches, reading weights once and avoiding per-row state capture, each behind the real-model control.
9. Choose depth from the round model, check it on a natural long response, and re-measure the cost table after each change to round cost. Gate a change of default against the current default over corpus, sampler and input-length cells with matched inputs.
10. Inventory every piece of state a request leaves behind, then implement prefix restoration with capture points inside the prefill pass. This is required for every new path by [model onboarding](MODEL_ONBOARDING.md).
11. Measure the first request after an idle pause, process start to ready, and peak memory against the plan.
12. Record every attempt, including the rejected ones, in the [experiment ledger](model-experiments/README.md).

## Ideas not yet tried

| Idea | Why it is open |
| --- | --- |
| Row-streaming kernels with ordered arithmetic for hyper-connections and the 4-bit projections, used by serial decode and verify together | The route to rows at about 3 ms. Changes the serial reference bits; needs its own precision attempt |
| A hyper-connection down kernel that stages the normalized input row once for several output rows | Would also help the serial step, where hyper-connections are 20 percent |
| Fold small launches: router with shared-expert gate, shared expert onto the routed-expert launches | Sushi went from seven launches to three per MoE layer and describes decode as bound by launch gaps |
| Expert down/combine latency (0.85 ms per row, not bound by bytes) | Dispatch shape, or fusion with gate/up |
| A coarse draft-head readout with exact re-scoring of the top candidates, and no logits readback | Draft steps are 1.0 ms of GPU time and 1.3 to 1.4 ms of wall time each |
| A sampled draft chain on the GPU, or top-k readback for sampled requests | Deferred; needs exact distribution, random-number and residual controls |
| Merge the finish pass into the next round's first submission | About 0.3 ms per round |
| Overlap host encoding in a zero-draft step inside MTP | Closes the 0.5 ms gap to MTP-off decode |
| A draft limit of 8, mainly for lookup rounds | Needs GDN projection kernels for rows 5 to 9 |
| A second draft chunk when the head is confident, serial fallback when rounds are losing | Present in Sushi; not assessed here. Sushi also keys its cost table by context size; on Lily the table measured at 4,096 tokens chose the same depths at 32,768, so that looks unnecessary |
| Less exploration by the planner where one depth is right throughout | One fixed draft beat it by 5 to 7 percent on sampled prose; rounds with no draft or two drafts were about one round in five there |
| The planner default in multi-turn and tool-calling agent sessions, and at other sampled settings | Single-turn natural requests are settled (level with two fixed drafts). Agent sessions mix copied tool output, where lookup should pay, with composed text, and have not been measured |
| A cost table and a schedule gate for each other pack of the family | The planner is the default on one pinned artifact. The official mixed pack takes Lily's table if asked and is unqualified |
| Three drafts in the desktop app | About 1 percent overall and 3 to 8 percent on sampled code. The paged-residency memory profile is bounded at two and needs a memory qualification |
| Whole-table warm of the n-gram embedding | The compared MLX server warms its 32 GB table before requests; never tried in Stark |
| Packed 4-bit layouts at 16 and 32 lanes | Prepared in October, never built |
| Prefix state for several conversations kept resident under a byte budget | Today one unrelated request displaces the snapshot |
| A stage profile of ordinary prefill | Prefill is most of a long request and has no attribution |
| The cause of the sustained-run slowdown | Needs privileged power metrics; clocks alone do not explain every slow request |
| The cause of the polled-wait gain | About 2 ms per round against a documented 0.09 ms wake latency |

## Records

| Period | Record | Covers |
| --- | --- | --- |
| 2026-10-28 to 10-31 | [Qwen3.8 optimization history][hist] | Host loop, precision, sampler, kernels, host conditions, standard-engine migration, matched-pack comparison, first MTP series |
| 2026-10-30 to 10-31 | [Exact MTP][mtpexact], [batched MTP][mtpbatched], [MTP optimization][mtpopt], [MTP combination][mtpcombo], [official batched MTP][mtpofficial] | Speculation design and first timings on both packs |
| 2026-11-01 to 11-02 | [Prefix reuse][prefix1], [hybrid prefix reuse][prefix2], [live session][prefixlive] | KV and recurrent-state restoration, prefill profile |
| 2026-11-07 to 11-08 | [Optimization pilot][pilot], [MoE projections][moeproj], [sustained diagnosis][sustained], [idle wake fix][idle], [cross-backend idle wake][idlex] | Stage profile, withdrawn projection kernels, cadence, idle wake |
| 2026-11-08 to 11-09 | [Vision-loaded text MTP][lvmtp], [guided memory][mem], [image startup][startup] | Draft head with the image tower resident, memory plan, startup |
| 2026-11-15 | [llama.cpp assessment and Lily baseline][assess] | Depth matrix, ranked stop and defer list, ceiling rule |
| 2026-11-15 | [One-draft default](benchmarks/2026-11-15-flashnext-lily-one-draft-default.md), [reconciliation][recon], [sampling policy](benchmarks/2026-11-15-lily-sampling-mtp-policy.md), [long coding][longcode], [long planning][longplan], [two drafts restored](archive/runs/2026-11-15-lily-mtp-two-draft-default-restored.md) | Default depth studies |
| 2026-11-15 to 11-16 | [Small-row verifier](benchmarks/2026-11-15-lily-small-row-verifier.md), [long context](benchmarks/2026-11-15-lily-long-context-infinitebench.md), [projection reuse][reuse] | Verifier stage shares and weight reuse across rows |
| 2026-11-17 | [Midpoint integration](model-experiments/2026-11-17-lily-midpoint-integration.md), [midpoint speed and release][midpoint], [startup memory][startmem] | Midpoint snapshot, memory admission |
| 2026-11-17, on a branch | [Speed review][fsreview], [speed experiments][fsexp], [agent grid][fsagent], [fresh grid][fsfresh], [kernel follow-up][fskernel] | Chunk size, prefill fusion, hyper-connection reuse, round-finish counts |
| 2026-11-19 | [Sushi assessment][sushi], [round profile][profile], [HC rows][hc], [GDN replay][gdn], [MoE rows][moe], [host round trips][host], [depth policy][depth], [fixed-corpus benchmark][bench], [merged timing][merge] | Round profile and the six workstreams it opened |
| 2026-11-19 to 11-20 | [Schedule gate][gate], [selector prefix reuse][selprefix], [schedule default][sched] | Matched-window gate over twelve cells, prefix reuse under the schedule options, the planner with lookup as the default on the pinned artifact |
| 2026-11-20 | [Inkling 2 Mini first milestone](benchmarks/2026-11-20-inkling-mini-m5-max.md) | Original-weight NVFP4 scale ordering, routed sublayer timing with and without a host wait per call, MXFP8 and attention costs |
| 2026-11-21 | [Inkling 2 Mini native decode slice](benchmarks/2026-11-21-inkling-mini-native-slice.md) | Single-stream 44-visit budget, exact-ingress expert replay and connected BF16 parity failure |
| 2026-11-21 | [Inkling 2 Mini pinned-runtime follow-up](benchmarks/2026-11-21-inkling-mini-pinned-oracle.md) | Live BF16 boundary, connected 4K parity and paired serial timing |
| 2026-11-21 | [Inkling 2 Mini short-prefix projection](benchmarks/2026-11-21-inkling-mini-short-projection.md) | Source-dispatched BF16 dot order, short and 4K connected parity, paired no-regression test |
| 2026-11-21 | [Inkling 2 Mini native worker](benchmarks/2026-11-21-inkling-mini-native-worker.md) | Grouped 4K worker prefill, greedy and sampled decode budgets, final slice reruns |
| 2026-11-21 | [Inkling 2 Mini packed prefill](model-experiments/2026-11-21-inkling-mini-packed-prefill.md) | Contiguous prefill buffers, exact kernel screens, matrix precision limits and cold worker comparison |
| 2026-11-21 | [Inkling 2 Mini stock 4K and singleton profile](model-experiments/2026-11-21-inkling-mini-stock-profile.md) | Stock prefill stage attribution, same-cache source router projection cost and selected follow-up screens |
| 2026-11-21 | [Inkling 2 Mini NVFP4 and singleton router](model-experiments/2026-11-21-inkling-mini-nvfp4-router.md) | Real-route tile occupancy, guard and FP32 replay attribution, plus exact shared scales and singleton projection trials |
| 2026-11-21 | [Inkling 2 Mini wider NVFP4 run tiles](model-experiments/2026-11-21-inkling-mini-nvfp4-run-tiles.md) | Same-loaded shape and MXFP8 fusion screens, exact 64 × 32 tile and matched 4K worker timing |
| 2026-11-21 | [Inkling 2 Mini tensor MXFP8 prefill](model-experiments/2026-11-21-inkling-mini-mxfp8-tensor.md) | Real-route certificate density, grouped source replay, compensated tensor partials, worker timing and wider-tile scale sharing |
| 2026-11-21 | [Inkling 2 Mini default fast prefill](model-experiments/2026-11-21-inkling-mini-prefill-default.md) | Flags-unset kernel admission, pinned parity, source-disable fallback and paired cold 4K worker timing |
| 2026-11-21 | [Inkling 2 Mini default one-row router](model-experiments/2026-11-21-inkling-mini-decode-default.md) | Flags-unset router selection, restored-cache projection and useful-decode timing |
| 2026-11-21 | [Inkling 2 Mini decode toward 60](model-experiments/2026-11-21-inkling-mini-decode-60.md) | Exact short verifier and rollback, NVFP4 sharing, shared local KV reads, certified sampling and stochastic draft overlap |

The 2026-11-17 branch records are not in this tree. They are on `codex/flashnext-speed-review-20261117` at commit `2e3058d1`, and the links above point there.

Performance work on other families is indexed in the [benchmark index](benchmarks/README.md): the dense Qwen3.8 27B [approach ledger](benchmarks/2026-11-06-qwen38-27b-approaches.md), GLM-5.3 [batched prefill](benchmarks/GLM53_BATCHED_PREFILL_2026_11_02.md) and the [GLM comparison with mlx-stream](benchmarks/2026-11-04-glm-mlx-stream-comparison.md). They are not summarized here. The full attempt index is the [experiment ledger](model-experiments/README.md).

[hist]: benchmarks/QWEN38_OPTIMIZATION_HISTORY.md
[mtpexact]: benchmarks/QWEN38_EXACT_MTP_2026_10_30.md
[mtpbatched]: benchmarks/QWEN38_BATCHED_MTP_2026_10_30.md
[mtpopt]: benchmarks/QWEN38_MTP_OPTIMIZATION_2026_10_30.md
[mtpofficial]: benchmarks/QWEN38_OFFICIAL_BATCHED_MTP_2026_10_31.md
[prefix1]: archive/runs/2026-11-01-generation-prefix-reuse.md
[prefix2]: archive/runs/2026-11-02-hybrid-prefix-reuse.md
[prefixlive]: archive/runs/2026-11-02-openwork-qwen38-live-prefix-reuse.md
[pilot]: benchmarks/2026-11-07-flashnext-optimization-pilot.md
[moeproj]: benchmarks/2026-11-07-flashnext-moe-projections.md
[sustained]: benchmarks/2026-11-07-flashnext-sustained-diagnosis.md
[idle]: benchmarks/2026-11-07-flashnext-idle-wake-fix.md
[idlex]: benchmarks/2026-11-08-cross-backend-idle-wake.md
[mem]: archive/runs/2026-11-08-qwen-guided-memory.md
[startup]: benchmarks/2026-11-09-lily-image-startup.md
[startmem]: model-experiments/2026-11-17-lily-startup-mode-memory.md
[midpoint]: model-experiments/2026-11-17-lily-midpoint-speed-and-release.md
[fsreview]: https://github.com/jshahbazi/stark/blob/2e3058d15fc67ccde7608861a80e5368b926acd8/docs/benchmarks/2026-11-17-flashnext-speed-review.md
[fsexp]: https://github.com/jshahbazi/stark/blob/2e3058d15fc67ccde7608861a80e5368b926acd8/docs/model-experiments/2026-11-17-flashnext-speed-experiments.md
[fsagent]: https://github.com/jshahbazi/stark/blob/2e3058d15fc67ccde7608861a80e5368b926acd8/docs/model-experiments/2026-11-17-flashnext-coding-agent-grid.md
[fsfresh]: https://github.com/jshahbazi/stark/blob/2e3058d15fc67ccde7608861a80e5368b926acd8/docs/model-experiments/2026-11-17-flashnext-fresh-coding-grid.md
[fskernel]: https://github.com/jshahbazi/stark/blob/2e3058d15fc67ccde7608861a80e5368b926acd8/docs/model-experiments/2026-11-17-flashnext-kernel-and-prefix-follow-up.md
[mtpcombo]: benchmarks/QWEN38_MTP_COMBINATION_2026_10_31.md
[lvmtp]: archive/runs/2026-11-08-lily-vision-text-mtp.md
[assess]: benchmarks/2026-11-15-llama-v1-flashnext-lily-assessment.md
[recon]: benchmarks/2026-11-15-lily-mtp-default-reconciliation.md
[longcode]: benchmarks/2026-11-15-lily-long-coding-response.md
[longplan]: benchmarks/2026-11-15-lily-long-planning-response.md
[reuse]: model-experiments/2026-11-16-lily-projection-reuse.md
[sushi]: benchmarks/2026-11-19-sushi-flashnext-lily-assessment.md
[profile]: model-experiments/2026-11-19-lily-mtp-round-profile.md
[hc]: model-experiments/2026-11-19-lily-hc-verify-rows.md
[gdn]: model-experiments/2026-11-19-lily-gdn-verify-replay.md
[moe]: model-experiments/2026-11-19-lily-moe-verify-rows.md
[host]: model-experiments/2026-11-19-lily-mtp-host-roundtrips.md
[depth]: model-experiments/2026-11-19-lily-mtp-depth-policy.md
[bench]: model-experiments/2026-11-19-lily-fixed-corpus-benchmark.md
[merge]: model-experiments/2026-11-19-lily-mtp-merged-quiet-timing.md
[gate]: model-experiments/2026-11-19-lily-mtp-schedule-default-gate.md
[selprefix]: model-experiments/2026-11-19-lily-mtp-selector-prefix-reuse.md
[sched]: model-experiments/2026-11-20-lily-mtp-schedule-default.md
