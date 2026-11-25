# Performance playbook

Use this guide when adding a model or improving its performance. The family speed histories answer “what happened on this artifact?” This page answers “what should I measure or try next?” Each comparison belongs to the build, workload and host in its dated report. Updated 2026-11-25.

## Choose a performance experiment

1. **Find the closest prior attempt.** Look up the exact checkpoint, quantization and backend, then search for the proposed technique. Read the last follow-up in the dated report. For a retry, name the condition that changed, such as the kernel shape, cache state or cost of an extra verifier row. [Model lookup and search](MODEL_RESEARCH.md) · [Branch-only studies](model-experiments/branch-research.md)
2. **Measure the slow part of a representative request.** Start a new family with the [bring-up checklist](#checklist-for-a-new-family). For an existing model, reproduce the workload and separate load time, first-token latency, useful decode and whole-request time. Record the image, draft and prefix-cache settings. Use the [time budget](#step-1-build-the-time-budget) to estimate how much request time the proposed change could save.
3. **Test one change against the incumbent.** Fix the correctness and timing criteria before the run. Use a stage profile to diagnose the cost, then time the uninstrumented path. Use paired runs on the exact artifact. For a model-level text-generation speed claim, use the standard server benchmark and its natural-text quality comparison. [Correctness gates](#correctness-gates) · [Paired measurement](#measuring) · [Standard benchmark](BENCHMARK.md)

| Observed symptom | What to try first | What happened on a recorded workload |
| --- | --- | --- |
| Cold prompts take too long | Time prefill with zero reused tokens. Compare batched projection widths at both short and longer prompt lengths. | On GLM-5.3-Flash UD-Q2_K_XL, 4- and 8-token batches tied at 14 input tokens. At 96 input and 8 output tokens, prefill rose from 34.895 tok/s on the control to 38.881 and 40.692 respectively. The 8-token recipe stayed opt-in. [Screened GLM runs](benchmarks/GLM53_BATCHED_PREFILL_2026_11_02.md) |
| A follow-up reports a cache hit but the first token is still late | Count the rows actually prefilled after restore. Compare fresh, prime and hit requests on a nonrepeating input near capacity. | Gemma 4 E2B and E4B reported 1,024 reused tokens while replaying the full prompt. Dispatching just the suffix cut median first-token time by 43–50% versus fresh on the two warmed workloads. [Warmed-cache repair](benchmarks/2026-11-09-gemma4-warmed-cache.md) |
| Serial decode is slow on a packed model | Count uploads and waits per token, then time the actual packed projections on rotating weights. Check whether the kernel repeats dequantization work. | On Ripple-65B Q4_K_M, grouped Q4_K/Q6_K projections plus cheaper attention indexing took worker decode from 4.559 to 7.460 tok/s across four short, unscreened warm requests per arm. The Q4 row-reuse variants were slower in component trials. [Saved-binary and component comparisons](benchmarks/RIPPLE_Q4_KERNELS_2026_11_12.md) |
| Decode repeatedly reads streamed expert weights | Count expert bytes read per request and time first use and a warm repeat at two cache sizes. Track memory pressure. | On the Hy4-flash Q3_K research branch, raising the arena from 28 to 36 GB cut warm decode reads by 37%. Warm throughput rose from 16.39 to 23.58 tok/s in one comparison. The larger arena stayed opt-in pending repeats and memory-pressure testing. [Hy4 branch measurements](model-experiments/speed-other-families.md#hy4-flash) |
| More draft tokens are accepted but generation slows | Measure committed tokens, target passes and wall time per round as draft depth changes. Compare with ordinary decode and the shipped draft policy. | On the dense Qwen3.8-27B oQ4e pilot, going from three to seven drafts raised advancement from 3.02 to 3.85 tokens per round. The second target scan pushed throughput from 48–50 down to 35–37 tok/s. Even full eight-row weight reuse lost in a separate sweep. The longer path was removed. [Draft-depth trials](benchmarks/2026-11-06-qwen38-27b-approaches.md#draft-depth-and-longer-verification) |
| Loading vision slows ordinary text | Hold the checkpoint and draft depth fixed. Check which text graph runs with the tower resident and whether it still batches verification. Time loading separately. | Dense Qwen3.8-27B's image reference graph lost 94% of three-draft decode speed on an eight-output test. Restoring the fast text graph with shared weights gave 48.309 tok/s text-only and 48.315 with vision resident in a separate 64-output comparison. [Failed control and repair](benchmarks/2026-11-09-qwen-dense-image-text-parity.md) |
| A decision request is slow | Split rendering, tokenization and worker forward time. If prefill dominates, test the actual batched linear shapes and validate the resulting choice and logits. | On Elephant 1 9.4B BF16, a wide-SIMD prefill kernel took a 526-token decision from 19.27 to 12.74 seconds median worker wall time (two samples). The choice matched with bounded logit drift. Generic BF16 GEMM ran about twice as slow in its trial. [Elephant prefill trials](model-experiments/2026-11-18-elephant-1.md#elephant-01e-bounded-prefill-wide-simd-acceleration) |

## Keeping this page current

Record every performance attempt, including regressions, invalid timings and experiments stopped before measurement. In the same change, add its baseline, candidate, measured change or timing limitation, and decision to the family speed history. A large study can use one linked matrix that calls out its wins, losses and reversals. [Recording workflow and index check](MODEL_RESEARCH.md#record-the-next-result)

When an attempt changes what a technique returned, extends it to another artifact, explains a failure, settles an open idea or changes a gate, update the relevant section here and link the dated record. For a rejection, give the measured reason or failed control and what would justify a retry. Move settled ideas out of the open list. Add the source under [Records](#records) and update the date above.

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

The following figures describe the Flash-Next work. Unless an example names another configuration, the reference is `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at revision `0f94842e`, with 4,096 input tokens on an Apple M5 Max with 128 GiB. Lily has 36 GDN recurrent layers and 12 sparse-attention layers. Each layer has 512 routed experts (10 selected per token), a shared expert and four hyper-connection streams. Its roughly 107 GB pack includes a 30 GiB n-gram embedding table and a one-layer draft head. The vocabulary has 248,320 entries.

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

When testing Inkling request-sized caches, record the advertised context and the physical allocation separately. Check exact warm and cold output across a physical resize within one logical window. Compare live decode growth against a fresh cache at the larger capacity with DSpark on and off. [Request-sized context experiment](model-experiments/2026-11-24-inkling-mini-request-context.md)

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

#### When a drafter proposes more rows than the target can verify at once

Use **wall time per emitted token** to compare designs. On a 40-GPU-core, 128 GiB M5 Max, DFlash 2 and DSpark produced identical outputs across 183 matched Inkling 2 Mini requests (20,825 tokens per arm). DFlash started 5,368 rounds to DSpark's 8,588, a 37.5% reduction. DFlash led on novel prose (+13.2% over DSpark and +7.5% over MLX-Serve). DSpark led at the 512, 4K and 8K context rungs in every block. Initial and sustained coding were unresolved between the two Stark arms. Keep DSpark as the ordinary Inkling default. [Image-loaded standard benchmark][ink-draft-live]

| Design choice | Performance tradeoff |
| --- | --- |
| Choose verification depth independently of draft width | A block drafter may generate all candidates for a fixed cost. Checking a shorter prefix saves target work. The useful depth depends on the chance of reaching each position and the measured cost of the next verifier call. |
| Verify a wide block in one pass or split it conditionally | One pass can amortize weight reads and dispatches. A split avoids the second pass when an early proposal fails. On one 4K trace, only 22 of 42 rounds reached the second call. That frequency determines how much a wider kernel can save. [Verifier replay](https://github.com/jshahbazi/stark/blob/a83752f22aec2139769ab8efae0d6b7729732dd1/docs/model-experiments/2026-11-24-inkling-mini-dflash2-traces.md) |
| Keep draft features and cache updates on the GPU | Passing committed feature rows through a host process adds copies and synchronization. Separate transfer time from draft compute before estimating the benefit of sharing device buffers. |
| Spend spare verifier rows on alternative continuations | A sibling candidate can rescue a round when the first choice fails. Its value depends on how often it recovers useful tokens and what an extra row costs. Committing a branch must restore the target state for that path and discard sibling state. [Splash implementation](https://github.com/incoai/splash/blob/2afc7337aacc044722fa95e4cb3da75c281a19a8/runtime/model/Runtime.cpp#L2541-L2591) |

Native round counts and target-call counts answer different questions because one round can span several calls. Recording both alongside draft, transfer, verification and commit time makes the cost visible. Verifier milliseconds per emitted token give a lower bound on the decode time. Compare that bound with the whole-token budget for the desired throughput.

#### Which drafter can repair the first miss?

On four greedy traces (one prose, three coding), each primary followed its own round starts. A catch required the other drafter to match the target on **every token through the primary's first error**. The original NVFP4/MXFP8 target verified native INT8 DSpark and BF16 DFlash 2 with F32 selector corrections. Training data, architecture and precision were not independently controlled, so the source of any training advantage remains unknown. [Matched traces and artifacts][ink-draft-traces]

The last three columns price a policy that offers a sibling at the first disagreement, dropping the last primary token when needed. Each divides the same modeled savings in target verification and commit by forks offered, siblings accepted, or all nonterminal draft rounds, respectively. [Cross-drafter replay and cost model][ink-draft-cross]

| Primary ← alternate | Workload | First misses caught, 3 slots | First misses caught, 7 slots | 3-slot ms per attempted fork | 3-slot ms per successful fork | 3-slot ms per draft round |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DSpark ← DFlash 2 | Prose | 39/79 (49%) | 46/90 (51%) | 8.09 | 16.83 | 5.26 |
| DSpark ← DFlash 2 | Coding | 72/149 (48%) | 80/186 (43%) | 4.32 | 9.27 | 2.17 |
| DFlash 2 ← DSpark | Prose | 7/47 (15%) | 6/56 (11%) | −1.01 | −7.06 | −0.66 |
| DFlash 2 ← DSpark | Coding | 22/102 (22%) | 22/134 (16%) | −0.23 | −1.02 | −0.11 |

**Use the denominator at which the implementation pays.** Calling DFlash on every DSpark round to discover disagreements gives a 5.26 ms target-time budget on prose and 2.17 ms on coding at three slots. The existing warm DFlash call including IPC took 8.71 ms on prose and 9.39 ms on 4K coding. The offline gate saw both completed proposals. An online gate that can decide *before* calling DFlash still needs a live test. At seven slots even the per-attempt budget fell to 3.97 ms on prose and 0.89 ms on pooled coding. On the 4K coding trace it was −1.99 ms because the branch increased physical target calls from 69 to 72. These estimates assume a sibling costs the same as a chain row and exclude ordinary draft work saved when rounds disappear. [Cost assumptions and sensitivity][ink-draft-cross]

#### What happened when the runner-up took the third slot?

The motivating paper trains a drafter to reconsider uncertain guesses with extra computation in parallel. Stark screened two ways to obtain an alternative without retraining. One ran the other drafter. The second reused the existing drafter's second-ranked token. The authors reported 3.5% to 9.2% higher speedup relative to their baselines. A trained reconsideration pass on either Stark model remains untested. [Parallel Reconsideration](https://arxiv.org/abs/2611.05724)

The Inkling and Flash Next experiments shared 12 held-out prose prompts and 12 held-out coding prompts. Each model used its own template and tokenizer, with thinking off and a 192-token greedy output cap. The gate checks draft positions one and two in order. At the first top-two logit gap of at most `ln(2)` (odds at most 2:1), the runner-up takes one of three proposal slots. The target verifies the anchor and up to three nodes. A sibling has no continuation of its own. Confident rounds keep all three primary choices. [Inkling policy and holdout][ink-draft-selector] · [Flash Next policy and holdout][flash-runner]

On ordinary Inkling round starts, the gated DFlash runner-up could recover 103/594 first misses on prose and 68/502 on coding. Replaying the whole candidate policy saved 19 prose rounds (904 → 885) and 10 coding rounds (840 → 830). Forcing a fork at position one on every round increased the counts to 970 and 933. After charging selector and readback, the modeled verifier savings left 0.46 ms per candidate draft round on prose and 0.27 ms on coding. The capture recomputed draft context at each anchor. The live worker updates the draft cache incrementally and reserves a final target draw. Adjusting for the output cap still left differences in the draft counts. The timings below use actual worker counts. [Held-out replay][ink-draft-selector] · [Worker accounting][ink-runner-live]

| Image-loaded target and workload | Live rounds, fixed three → runner-up, one pass | Fixed three, decode tok/s | Runner-up, decode tok/s | Paired decode change (95% interval) | Whole-request tok/s, fixed → runner-up |
| --- | ---: | ---: | ---: | ---: | ---: |
| Inkling 2 Mini, prose | 896 → 881 | 73.05 | 73.54 | +0.67% (−0.88% to +2.34%) | 67.72 → 68.17 |
| Inkling 2 Mini, coding | 833 → 823 | 78.64 | 78.65 | +0.02% (−1.24% to +1.32%) | 72.38 → 72.38 |
| Flash Next Lily Q4-XL, prose | 1,002 → 983 | 67.34 | 68.10 | +1.13% (−0.39% to +2.86%) | 63.88 → 64.53 |
| Flash Next Lily Q4-XL, coding | 710 → 703 | 95.92 | 96.04 | +0.12% (−1.04% to +1.19%) | 88.69 → 88.80 |

Each row compares adjacent requests in one loaded experimental worker with the vision tower present, reversing arm order on the second pass. A four-block Inkling test with separate worker launches changed sign across blocks on both workloads. The intervals resample whole prompts with both repeats together. Identical width-two controls moved −0.75% and +0.71% on Inkling prose and coding, and −0.51% and −0.46% on Flash Next. All 128 Inkling and 176 Flash Next timed completions in these comparisons matched their frozen targets. Inkling used streaming and Flash Next used nonstreaming requests. All four paired decode intervals include zero. The evidence supports keeping the runner-up opt-in on both models. [Inkling paired benchmark][ink-runner-live] · [Flash Next paired benchmark][flash-runner]

- **Inkling.** The live target chose the sibling on 98/388 prose forks and 57/295 coding forks. The net rounds saved could fund 0.60 ms extra per candidate round on prose and 0.43 ms on coding. The observed costs were 0.36 and 0.42 ms. For overhead paid only on packed forks, multiply ordinary ms/round by net rounds saved and divide by forks packed. That gives 1.35 ms per fork on prose and 1.19 ms on coding. Keep ordinary DFlash for deployments that select it. [Measured break-even calculation][ink-runner-live]
- **Flash Next.** On the short prose prompts the sibling recovered 72/763 first misses and displaced 46 correct third choices. On coding it recovered 33/251 and displaced 33. Across both repeats, verification saved 1.28 s on prose and 0.33 s on coding, while drafting plus rollback added 0.53 s and 0.28 s. Full rounds still paid for three autoregressive head steps. Shipped adaptive MTP with prompt lookup reached 83.04 tok/s on prose and 98.29 on coding. It beat the runner-up on all 12 prose prompts. The pooled coding difference was within paired uncertainty. Keep the shipped schedule. This control allowed up to three drafts and the desktop profile caps them at two. [Stage timings and shipped-policy comparison][flash-runner]

#### How to choose the next optimization

1. **Locate the time that must come out of the request.** Record native draft rounds, physical target calls, and draft, feature-transfer, verify and commit milliseconds per emitted token on workloads where the winner changes. Inkling's fixed 4K seven-proposal trace took 64 target calls for 42 rounds and 8.52 ms of verification per token. A hypothetical 1.7× over the measured 88.7 tok/s MLX-Serve rate allows 6.63 ms for the entire token. That target requires cheaper verification even if drafting becomes free. [Matched verifier replay][ink-draft-traces] · [Live baseline][ink-draft-live]
2. **Test how much of the primary chain to verify.** Replay a depth policy against fixed three and fixed seven on held-out anchors, then time it against the best ordinary schedule. Inkling DFlash produces seven proposals either way. On the four fixed traces, verifying seven instead of three saved 52 rounds, added eight physical target calls and broke even at 3.77 ms of avoided work per saved round. Seven won the separate live width screen by 2.6%. A native eight-row target pass would avoid a second call in 22/42 rounds of the 4K trace and do extra work after early rejection in the others. Price both outcomes before widening the kernel. [Width screen][ink-draft-live] · [Packet costs][ink-draft-traces]
3. **Spend engineering time where there is measured headroom.** If the MLX companion dominates, time feature copies, synchronization, draft execution and KV updates separately before moving feature exchange onto the GPU. Revisit alternatives when there is a cheaper gate, a better candidate, or cheaper branch verification and state commit. Replay each policy from its own anchors and count first misses rescued, correct continuations displaced, and net target rounds. Charge selection on every inspected round and branch-state work on every fork that incurs it. A trained corrector is worth prototyping when its projected net savings on fresh prompts cover its critical-path compute and state costs. [Cross-drafter budgets][ink-draft-cross] · [Live runner-up costs][ink-runner-live]
4. **Qualify the winning path and the next token.** Force sibling wins and rejections at both depths. Compare full target readouts and the next step after commit or rollback, including attention KV and model-specific convolution, recurrent or draft state. Check prefix reuse and image invalidation where supported. Flash Next passed 69 full-vocabulary comparisons, but its fork requires the packet to fit within the 2,051-token dense attention window. Its 4,096-token prefix-reuse check exercised ordinary-chain fallback. Inkling separately matched all 200,058 target logits, committed KV and convolution state, and the selected feature rows sent back to DFlash in its forced-branch tests. Sampled forks, sparse-context forks and peak memory need their own evidence. For a default change, repeat the adjacent comparison against the shipped policy on the intended workload and report both decode and whole-request rates. [Inkling state checks][ink-runner-live] · [Flash Next correctness and scope][flash-runner]

[ink-draft-traces]: https://github.com/jshahbazi/stark/blob/0237919982373a37b2c14b450c701c693415ead0/docs/model-experiments/2026-11-24-inkling-mini-dflash2-traces.md
[ink-draft-live]: https://github.com/jshahbazi/stark/blob/0237919982373a37b2c14b450c701c693415ead0/docs/benchmarks/2026-11-24-inkling-mini-live-dflash2.md
[ink-draft-cross]: https://github.com/jshahbazi/stark/blob/0237919982373a37b2c14b450c701c693415ead0/docs/model-experiments/2026-11-25-inkling-first-miss.md
[ink-draft-selector]: https://github.com/jshahbazi/stark/blob/0237919982373a37b2c14b450c701c693415ead0/docs/model-experiments/2026-11-25-inkling-dflash-self-correction.md
[ink-runner-live]: https://github.com/jshahbazi/stark/blob/0237919982373a37b2c14b450c701c693415ead0/docs/benchmarks/2026-11-25-inkling-live-runner-up.md
[flash-runner]: https://github.com/jshahbazi/stark/blob/786d1ea1269b982f0cbd2b7ce9b8cf8f92c3203d/docs/model-experiments/2026-11-25-flashnext-mtp-runner-up.md

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
- On Inkling 2 Mini, prefill has had more work than on Lily; see the Inkling paragraphs under the source call geometry section below. In short: removing an exactness guard that never fired and fusing the expert epilogues gave about 1.5 times while staying bit-exact, dropping MXFP8 certification under the noise-floor gate gave about 1.17 times more, a dense projection kernel and a packed draft commit about 1.11 times more, BF16 prefill attention about 1.02 times more, a gathered NVFP4 kernel in MLX-Serve's tile structure 1.13 times more on wide commands ([gathered kernel][inknax]), while stationary and staged kernels lost. In the first four-arm run on the committed tree (one repeat, not a standard result) Stark's 8-bit prefill was level with MLX-Serve's 8-bit pack (1.00 and 0.99), where run 4 had it at 0.76 to 0.80 ([engine matrix run 5][inkengine]); the three-repeat standard result then put it ahead on repeated-sentence prefill (1.13) and unresolved on the exact 8,800-ID cell (0.96), with decode ahead at 4K and 8K, behind on novel prose (0.86) and unresolved on five rows ([standard result][inkstd]); the [differences page](model-experiments/inkling-mlx-serve-differences.md) tracks each item ([exact fast prefill][inkfast1], [unguarded MXFP8][inkfast2], [prefill round 3][inkfast3], [prefill round 4][inkfast4], [prefill attribution][inkattr]).

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

The findings below are encoded as rules and checks in the [standard benchmark](BENCHMARK.md); use it for any new engine comparison.

Published rates are not comparable until the conditions match. The 2026-11-19 comparison found Sushi slower at serial decode (about 62 against 73 tok/s) and faster in its headline number (146.2 on code), because of cheaper and deeper speculative rounds measured greedy on a regular synthetic code corpus ([Sushi assessment][sushi]).

- Greedy against sampled, and code against prose, each move the result more than most optimizations. Sushi's own figures at one context size are 146.2 on code and 79.5 on prose. Stark's merged build records 113 to 123 on code and 86 to 87 on prose at its best settings.
- `ignore_eos` is length control, not a speed trick. It was a no-op on this corpus.
- A benchmark's second and third samples may reuse a cached prefix by design. Record reused input tokens.
- Hardware and speculation settings in a published figure have to be read from the source. One pack's advertised 157 tok/s was on different hardware with speculation on.
- [`scripts/benchmark_lily_fixed_corpus.py`](../scripts/benchmark_lily_fixed_corpus.py) reproduces Sushi's corpus generator and request ladder, so the conditions can be matched ([fixed-corpus benchmark][bench]). Even then it is not a matched comparison: the packs differ.
- Match the reasoning mode and check that both engines rendered the same input. A morning Inkling 2 Mini comparison on 2026-11-22 had Stark at 34.76 against 97.58 tok/s on short coding, but the client's `reasoning_effort: "medium"` was rejected by Stark and accepted by MLX-Serve, MLX-Serve was forced to full length while Stark stopped at end of sequence, and Stark's fast paths were not admitted at those lengths. None of it compared engines. With the mode matched (on Inkling, "off" renders effort level 0 and the model may still think), every arm received identical input token counts, which is the check to make ([engine matrix][inkengine]).
- Read what the other engine actually loads. MLX-Serve skips Inkling's draft head because the index does not name it, so it does no draft work in prefill while Stark does ([prefill attribution][inkattr]). Its headline Inkling figure is on the author's 8-bit pack, a different artifact from the original checkpoint ([engine matrix run 2][inkengine2]).
- Make the harness refuse a broken arm before timing and keep the other arms running when one fails. An 8-bit arm once failed only at server start because another session had deleted its weight shards; the matrix harness now checks every shard named in the index and records a failed arm instead of aborting ([engine matrix run 2][inkengine2]).

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

Batch each row's original relative head tiles and gate work into two launches. On Inkling's shared verifier this removed 192 dispatches and saved 0.44–0.50 ms per three-row 4K command with exact logits. Sharing the 64 shard factors behind a threadgroup barrier was slower at three rows. Dispatch batching gave two wins in four sampled worker pairs and was archived. [Verifier attention tail screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-56--batch-relative-tables-and-share-attention-gate-weights)

Timestamp the vocabulary readout separately before changing a padded tile. Inkling's eight-row target tile preserved exact logits and saved about 0.02 ms in the readout, which still traverses the full BF16 bank. The attention and expert inputs together account for about 4.53 ms at three live rows. [Target vocabulary tile screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-57--screen-an-eight-row-target-vocabulary-tile)

Measure the residual mix before expanding a barrier reduction across the controller path. Inkling's exact barrier-sharing variant saves 0.03–0.06 ms per three-row 4K command. Archive this scheduling change. [Residual mix barrier screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-58--share-residual-mix-barriers-across-input-sources)

Retest the dense stage before lifting the shared-weight GEMV to larger threadgroups. On Inkling, the loop interchange, static K and 128- or 256-thread launch sizes preserve exact logits and add time to dense attention at three live rows. [Dense verifier stream screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-59--stream-the-exact-dense-verifier-dot-across-live-rows)

Measure both prepare and finish when coalescing transactional copies. Inkling's address-table dispatch saved 0.36–0.65 ms at three rows when a suffix was rejected. The worker paired median moved +0.11% with two wins in four pairs. Archive the GPU address path at this scope. [Batched undo](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-60--batch-the-verifiers-cache-and-convolution-undo-copies)

Recheck depth after changing the verifier cost. The current fused-cache and FP4 lookup stack still loses all four seeded 4K pairs at depth three, with a −4.98% median paired change and exact outputs. [Sampled depth repeat](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-61--recheck-sampled-depth-on-the-current-verifier)

Bind tiny projection outputs to their final offsets before adding a packing kernel. For short Inkling verifier commands direct controller rank output preserved exact logits and removed 96 dispatches. The total saving was 0.17 ms at three rows, so keep the existing path. [Controller rank output](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-62--write-short-controller-rank-projections-directly-into-packed-rows)

Measure the entire boundary when appending accepted draft-tap commits to the next proposal command. Inkling's combined schedule preserved proposals, F32 logits and state, and saved 0.08–0.22 ms at two proposals on stock 4K. Archive the scheduling change at the native gate. [Commit scheduling](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-63--submit-the-accepted-suffix-commit-with-the-next-draft)

Price candidate construction and acceptance when shrinking a proposal vocabulary. Inkling's 16K context-selected INT8 head saved 0.86–0.88 ms including the gather and matched all checked base scores. The shortlist lost some accepted prefixes and host construction took 0.389 ms. The combined replay measured 58.93 modeled tok/s against 59.05 for the full head. [Causal vocabulary, projection and host costs](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-64--replay-a-context-dependent-proposal-vocabulary)

Measure table footprint and realistic indexing before replacing arithmetic with larger lookups. At three rows on Inkling, colocating neighboring expert weights and E4M3 bytes added 0.13–0.16 ms. The scale-indexed table added 0.54–0.56 ms and the four-coefficient table added over 13 ms. A separate packed product table increased product and quantization time. All native logits matched. [Weight layout screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-67--panel-nvfp4-weights-and-test-larger-coefficient-tables) · [Packed product table](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-68--look-up-two-coefficient-products-from-packed-fp4-operands)

Measure an improved one-proposal command at a fixed depth before making a depth policy choice. Inkling's queued INT8 path raised the one-proposal sampled median from 48.51 to 54.22 tok/s, with an 11.04% paired median gain and exact outputs. At the resulting costs, depth two still won three of four separate depth comparisons. [Queued one-proposal screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-69--queue-a-single-int8-proposal-and-retest-fixed-sampled-depth)

Charge the draft-completion fence when a confidence policy chooses the verifier width after scoring proposals. On Inkling that boundary cost 1.10–1.26 ms per round. The policy lost to the faster fixed depth on all four held-out seeded 4K trajectories. [Draft-confidence depth replay](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-70--price-draft-confidence-depth-selection-after-queued-int8-proposals)

Measure complete proposal time and accepted prefixes when shrinking a bidirectional draft block. Inkling's eight-row BF16 projection epilogue preserves proposals and saves at most 0.025 ms. Keeping four attention rows saves about 0.21 ms and improves only two of four seeded replays. [Draft block projection screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-71--reduce-the-padded-draft-block-and-fuse-projection-rounding)

Use the actual worker trajectories when a restored-cache round model exceeds the worker rate. The queued draft and target boundary accounted for 95.6–96.2% of Inkling's sampled worker time. Three batch workloads needed 1.27–1.71 ms less per round at their current acceptance. Streaming callbacks contributed at most 0.011 ms per round. [Live sampled worker budget](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-72--attribute-live-sampled-worker-time-on-the-four-fixed-workloads)

Test direct BF16 tensor inputs for a padded controller gate when the source products already use tensor operations. On Inkling an exact native 16 × 16 tile saved 0.19–0.27 ms per three-row stock 4K command, or about 0.27 ms in the summed input stages. [Short controller expansion](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-73--use-device-bf16-tensor-operands-for-short-controller-expansion)

Retest exact native savings together when separate worker A/B effects sit within run-to-run variance. Inkling's five short verifier changes improved all 12 sampled pairs across three sessions. The retained opt-in path reached 62.70 tok/s in the paired confirmation and 62.61 tok/s sampled and 77.76 tok/s greedy with one loaded worker. All eight single-worker workloads cleared 60 with exact hashes. [Combined short verifier](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-74--combine-exact-short-verifier-savings-in-the-live-worker)

Batch independent Q, K and V controller slots while keeping their RMS and BF16 boundaries. The Inkling prototype removes 192 dispatches and saves 0.48–0.57 ms in exact three-row 4K verifier commands. The initial sampled screen had two wins. The same-loaded repeat gained 3.05% at the paired median and won all four pairs. [Batched cross-pass controllers](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-75--batch-the-q-k-and-v-cross-pass-controllers)

The integrated Inkling controller batch preserved short and stock 4K parity and eliminated the same 192 dispatches. A paired M5 Max rerun on a quiet GPU gained 3.80% seeded sampled and 1.03% greedy, with exact hashes and four wins per mode. The batch stays opt-in within the combined short verifier. [Integrated QKV controllers](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-78--integrate-batched-q-k-and-v-controllers)

Batch independent residual mixtures across live verifier rows before redesigning their reductions. Inkling combines attention queries and cross-pass messages, and batches the expert input mix after the fused attention finish. The exact change removes 262 and 381 dispatches at three and four rows and saves 1.33–1.36 and 1.95–2.07 ms in plain 4K commands. The same-loaded worker gains 3.00% sampled and 4.66% greedy at the paired median. [Batched verifier residual mixing](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-83--batch-residual-mixing-across-verifier-rows)

Reuse proven vector load arithmetic at short verifier widths and measure each expert format separately. The existing source MXFP8 four-value loads save 0.97–1.17 ms at three rows and 1.10–1.45 ms at four on both stock 4K states, with exact logits and unchanged dispatch counts. This transfers to +1.55% sampled and +5.02% greedy at the paired medians. [Short verifier MXFP8 loads](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-84--vector-load-the-short-verifier-mxfp8-products)

Charge Metal certificate preparation to every verified row and host sampling to consumed rows. Inkling's standalone GPU certificate costs 17.48 µs more than the CPU proof including copy and wait, with matching draws. In the same-loaded worker test, the integrated path lowered the residual by 0.50 ms per round and added 0.37 ms to the queued interval at paired medians. Useful decode gained 0.57%. [GPU sampler certificate](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-76--prepare-an-exact-sampler-certificate-on-metal)

Score draft calibration on held-out trajectories at the same verifier width. Scaling the shipped Inkling Markov correction looked promising on one training fixture and used 388 held-out verification rounds against 386 for the stock draft. [DSpark Markov sweep](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-77--check-compatible-draft-availability-and-replay-proposal-calibration)

Measure the exact critical-path ceiling before implementing a greedy shortcut. Inkling's four-row QKV trace puts embedding and vocabulary readout at 2.14 ms. Returning the entire stage to the earlier paired greedy worker would model about 77.9 tok/s. Two-column NVFP4 workgroups also add 0.096–0.146 ms wall across both stock 4K prefixes. [One-worker repeat and verifier screen](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-80--screen-smaller-nvfp4-workgroups)

Price sampled proposal improvements on nonoverlapping accepted prefixes from live seeded trajectories. Inkling's privileged target-mode reference reduced 530 rounds to 461 and modeled 74 tok/s with the earlier measured per-prompt costs. The alternate seeds gave a similar estimate. The fitted Markov bias tied all four stock held-out round counts, so archive the fit. [Sampled proposal headroom](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-81--replay-a-privileged-sampled-proposal-reference)

Measure the remaining RMS work after batching verifier mixes. On Inkling, caching the immutable normalized block and carried vectors preserved exact logits. Three-row plain wall savings were −0.006 and +0.062 ms on the two stock 4K inputs, so archive the cache at the native gate. [Cached residual sources](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-85--cache-normalized-residual-sources)

Evaluate a causal prior on complete held-out prompt trajectories. Inkling's reranking uses the target row that produced the emitted anchor. Its signed prior and shortlist alternatives use 765 rounds against 769 for stock across eight test trajectories. [Causal draft ranking](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-86--score-draft-candidates-with-a-causal-target-prior)

Measure candidate coverage and calibrated selection on the same held-out trajectories. On Inkling's native INT8 shortlist, choosing the actual target from the top four when present lowered replay rounds from 769 to 665. The policy fitted to draft confidence and a copy of the target's seeded draw used 772 rounds. Archive this calibration. [Seed-aware draft ranks](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-89--replay-seed-aware-draft-ranks)

Measure complete commands after fusing a producer with its quantizer or epilogue. Inkling's exact NVFP4 boundary bundle removed 88 launches and moved three-row wall time by +0.198 and −0.017 ms on the two stock inputs. The immutable checkpoint census found zero zero-valued K16 scales among 1.27B. [NVFP4 boundary and sparsity screens](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-87--fuse-nvfp4-expert-buffer-boundaries)

Repeat depth comparisons after different speedups at adjacent verifier widths. On the retained MXFP8 vector worker, two drafts won all four sampled pairs against one and three of four against three. The third proposal still loses 6.75% at the paired median despite reducing rounds from 530 to 493. [Retained sampled-depth repeat](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-88--recheck-sampled-draft-depth-after-verifier-cuts)

When a faster runtime reports a short greedy peak on its own quantization, first compare the original weights at matched cold prompt lengths and record the output agreement. MLX-Serve reported a 116 tok/s peak on its 8-bit Inkling pack. The original-checkpoint 4K screen on M5 Max returned 69.69 tok/s greedy and 73.68 sampled with drafting requested. One of four completion texts matched its draft-off pair. Its verifier's packed expert reads and split attention warrant width-specific tests against Stark's exact reference. [MLX-Serve preview](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-90--compare-the-mlx-serve-inkling-preview)

Replay native drafts at the proposed width and price the extra target rows separately. On Inkling, four drafts reduced sampled replay rounds from 386 to 347, with only 0.21 ms of additional round cost available at 80 tok/s. Three-head local KV sharing saved 0.154–0.218 ms at the incumbent sampled width with exact logits and rollback. The direct wider graph adds several milliseconds before handling rollback, so defer the wider transaction. [Wider verifier budget](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-91--price-wider-verification-and-local-kv-sharing)

Time lossless weight reconstruction in the projection that will consume it. Inkling can reuse a packed BF16 high-byte decode across three rows and reduce estimated weight bytes by 24.1%. Even the fastest exact unpack adds 1.88–2.03 ms at the three-row native boundary. [BF16 high-byte packing](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-92--reconstruct-packed-bf16-bytes-across-verifier-rows)

Test concurrent projection encoders in the live queued round as well as standalone. Inkling saves 0.73–0.83 ms per standalone three-row target command and gains 2.20% in the first greedy worker comparison, with exact outputs and state. The sampled repeat loses 2.97%, and its queued verifier interval grows 0.78 ms per round. Archive this encoder layout at the worker gate. [Concurrent projections](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-93--run-independent-attention-projections-concurrently)

Check indirect dispatch arguments when replacing a serial encoder with a concurrent encoder and explicit buffer barriers. On Inkling the conservative target schedule preserves 1,644 F32 rows and 540 committed states. Three-row savings range from −0.48 to +0.22 ms when queued with the actual draft. Archive the single-encoder schedule. [Queued verifier schedule](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-94--compare-concurrent-schedules-in-the-queued-verifier)

Price a compact vocabulary readout with the rejected draw rate and the cost of submitting each stage. Inkling's coarse INT8 head and original BF16 tiles resolve 489 of 512 seeded draws in the one-row Metal bounds screen. All 152,800 selected-tile controls match bit-for-bit. The separate bounds command erases its modeled savings. A live trial needs to share the coarse head submission and deduplicate tiles across live rows. [Selected vocabulary readout](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-95--screen-an-exact-vocabulary-readout-with-int8-bounds)

Fuse the coarse head, bounds and compaction into the queued target command and time the exact tile union and any consumed-row fallback. Inkling's native probe gains 8.05% sampled and 5.01% greedy at 4K, with exact outputs, work and state across all short and 4K pairs. The sparse arm still needs a justified source arithmetic bound and worker qualification. [Queued sparse vocabulary readout](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-96--time-sparse-vocabulary-readout-inside-queued-decoding)

Keep the incumbent exact decoder in the comparison when adapting a packed-weight kernel from another runtime. MLX-Serve's scaled-half FP4 technique matches Inkling's coefficient and verifier controls on M5 Max. All three variants increase the sampled-width expert dot stage relative to the retained byte-pair table. [Scaled half coefficient decoding](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-97--decode-nvfp4-coefficients-with-scaled-half-bits)

Establish the implemented tensor product's arithmetic contract before promoting a bound that skips target logits. A float output type in Apple's TensorOps does not give the intermediate accumulation error used by Inkling's sparse certificate. Keep the sparse projection diagnostic while the contract is open. [Sparse arithmetic review](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-98--check-arithmetic-guarantees-for-sparse-vocabulary-readout)

Separate candidate ordering from confidence calibration in a seeded replay. Correct target ordering within Inkling's top four draft candidates reduces 769 rounds to 713 while retaining the draft's rank masses. Giving the true global sampled rank to the unchanged list reduces the count to 766. A held-out fit using draft gaps, tail mass and draw bins uses 771. [Draft ordering replay](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-99--separate-draft-ordering-from-sampled-rank)

Price epilogue fusion alongside the exact readout. Fusing Inkling's original soft cap into its BF16 vocabulary tile and grouping four SIMD tiles per threadgroup preserve raw and capped logits but change the three-row readout by at most 0.014 ms. The full BF16 weight traversal remains. [Fused vocabulary readout](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-100--fuse-the-original-vocabulary-readout-and-logit-cap)

Check a learned reranker on unseen prompts and both seeds before adding the scorer to the draft command. Inkling's pairwise linear fits with Markov, prior-score and hidden-projection features increase the held-out round count from 769 to 776. Both also lower agreement with the target's ordering of the existing shortlist. [Hidden-feature rerank](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-101--fit-a-causal-reranker-from-dspark-hidden-features)

Batch source RMS over verifier rows and share the reduction only when inputs coincide. Inkling eliminates 240 first-pass norm launches at three rows and 336 at four. The candidate with paired query/gate projection at three rows gains 1.12% greedy and loses 0.29% sampled over eight worker pairs per mode. Retain four-row norm batching alone on the opt-in residual verifier. [First-pass input batching](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-102--batch-first-pass-norms-and-share-query-and-gate-inputs)

Compare an exact weight permutation with the already coalesced source on real projection banks. Inkling's adjacent-output BF16 stripes and two-term vector loads change the three-row dense stage by at most 0.048 ms across four 4K captures, and queued command effects are mixed. [Dense weight layouts](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-103--reorder-exact-dense-weights-for-coalesced-verifier-reads)

Price shared draft K/V loads both alone and with the target already queued. Inkling's four-row draft sharing saves 0.473–0.591 ms alone at two proposals while the paired queued round ranges from a 0.097 ms loss to a 0.373 ms saving. All checked draft hidden words and target continuations agree. [DSpark K/V sharing](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-104--share-dspark-attention-reads-across-its-draft-block)

Charge ingress whenever a shorter integer dot needs another activation encoding. Exact packed multiplication reduces Inkling's FP4 pair table to 1 KiB. The three-row dot and activation packing both grow against the incumbent half-ingress kernel. [FP4 packed multiplication](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-105--compute-exact-fp4-pair-dots-with-packed-multiplication)

Reverse paired worker orders before qualifying a combination whose native timings track arm position. Inkling's three-row RMS, target local K/V sharing and DSpark K/V sharing gain 2.86% at the paired worker median and save 0.896 ms per round. All outputs and work counts agree. Retain the new geometry in the opt-in verifier at three rows and keep the incumbent local shader for the other widths. The retained sampled worker still needs to save 3.77 ms per round to reach 80 at its observed acceptance. [Sampled-width combination](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-106--combine-the-strongest-remaining-sampled-width-changes)

Price stale drafts over a complete trajectory. Inkling's reused DSpark mask rows remove about 2.62 ms from reused rounds versus fresh rounds within its two-round policy. The policy adds 110 verifier rounds over eight sampled 4K trajectories and raises median queued plus commit time by 16.3%. Refreshing on rejection adds 49 rounds and raises time by 5.8%. Both preserve exact outputs and committed state. [DSpark mask reuse](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-107--reuse-dspark-mask-rows-across-verifier-rounds)

When restored-state timing depends on arm order, repeat both orders at each anchor. Inkling's per-token NVFP4 down product removes routed intermediates while giving up shared weight reads. The first alternating-pair medians suggest a gain, and the same-anchor reversal adds 0.489–0.712 ms across stock 4K inputs with exact outputs and state. Archive the layout. [NVFP4 down route reduction](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-108--reduce-nvfp4-down-routes-inside-one-threadgroup)

Charge memory traffic before moving coefficient decode out of the hot path. Inkling's byte and binary16 expansions preserve all checked raw F32 products while adding 4.45–4.95 ms and 14.40–14.57 ms to three-row queued 4K verification. Archive both layouts. [Predecoded NVFP4 coefficients](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-109--expand-the-nvfp4-coefficients-before-verification)

Test shorter drafters at equal verification depth before charging their reduced layer count as a saving. Inkling's closest untrained DSpark subset keeps layers one and three and adds 48 verification rounds across eight seeded 4K trajectories. Applying paired queue costs raises its modeled median time by 4.9%. Archive all six untrained subsets. [DSpark layer ablation](model-experiments/2026-11-22-inkling-mini-decode-80.md#20261122-inkdecode-110--shorten-the-dspark-forward)

Measure teacher-forced probability overlap and conditional acceptance before building stochastic rejection. On two original Inkling 4K sample trajectories, three masks estimated 2.37 and 2.97 useful outputs per round. Only one trajectory’s current verifier could fit 60 tok/s after subtracting measured draft cost. [Stochastic feasibility screen](model-experiments/2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-17--measure-stochastic-dspark-acceptance-before-changing-decode)

Check whether an exactness guard ever changes a result before paying for it. On Inkling's prefill the NVFP4 guard and replay never changed an output on nine pinned captures; removing them gave 1.33 times at 4K on the engine diagnostic, still bit-exact. Fusing the gate/up and down epilogues into 64 × 64 kernels with a vectorized, double-buffered weight decode gave 1.08 times more, and reading BF16 rows in the MXFP8 replay 1.05 times. Together 1,085 → 1,634 tok/s at 4K (1.506×, 8/8 ABBA pairs) and 1.476× at 8,800 tokens. BK 64, 128-row tiles, and eight-row and row-major MXFP8 replays were slower. Keep verifier commands off a prefill fast path unless the rollback proof covers it: the first build broke it. [Exact fast prefill][inkfast1]

Decide an inexact change against a noise-floor control, not against zero. Once Inkling's prefill was bit-exact, one BF16 ulp raised in one residual element at one visit moved logits by a median 10.3 percent mean row L2 across trials, up to 87 percent on a row and up to 31 percent top-1 flips, so the old relative gate rejected any inexact change by chance. Unguarded MXFP8 (one F32 accumulation, no certificate or replay) measured 2.38 percent mean, 16 percent worst and 5.6 percent flips, inside the control span, and gave 1.169× at 4K and 1.174× at 8,800 over the exact build. It is the default; the exact kernels stay reachable through `--exact-prefill`. A mask-free global attention kernel derived from the pinned MLX kernel is bit-identical and removed 87 ms of instrumented time at 8,800 tokens. [Unguarded MXFP8 and the noise-floor gate][inkfast2]

Check row independence across command widths before relying on prefix snapshots. Inkling's source arithmetic switches split-K routing and the dense reduction at 1,024 rows and the thin projection tiling at 2,048, so a row's bits depend on its command width class; prefix reuse stays exact because snapshots sit on 2,048-row boundaries and replay the same schedule. [Split consistency][inkfast2]

Launch count is not the cost driver of Inkling's verifier. Fusions that removed 120 to 250 launches per verifier round left two-, three- and four-row commands at 0.989 to 1.002 of the pre-fusion time over six ABBA pairs; a four-row round spends 11.8 ms in experts, 5.8 ms in attention projections and 2.8 ms in attention of 27 ms GPU. The same kind of fusion on the single-row step (1,907 → 992 launches) made it 15.6 percent faster. Measure before fusing for launch count on multi-row commands. [Second fusion session][inkfuse2] The correction came later: what costs on the verifier is launches that repeat per row. Below 1,024 visible keys every verifier row ran its own attention (288 launches per row); one launch for all rows took the four-row verify 1.6 to 1.7 ms faster on both artifacts, 12 of 12 pairs, with exact output. Count per-row launches, not total launches. [Batched short-context attention][inkround]

Separate draft work and clock state before comparing a worker with its engine diagnostic. Inkling's worker was 19 percent slower than its diagnostic at 8,800 tokens: about 0.6 s was per-row draft-tap commits inside prefill (+11.5 percent GPU time on a 2,044-row command) and about 0.6 s was GPU clock state, because the diagnostic's first pass runs right after an idle model load. Snapshot, resizing and host gaps added about 0.1 s. Against MLX-Serve the gap is per-token work: 0.640 against 0.382 ms per token, with attention growth worth about 0.15 s at 8,800. MLX-Serve's larger 8,192-row chunks are worth about 0.3 s to it. [Prefill attribution][inkattr]

Byte-count a weight-format change before building it. Inkling's dense projections and readout read about 3.5 GB per token in BF16 and are bound by weight reads at one to four rows, so 8-bit affine weights would save an estimated 2.5 to 3.7 ms per round. The built Stark 8-bit artifact then measured 3.3 ms per single-row step (18.44 → 15.14 ms) and 2.9 to 3.4 ms per verify round in the engine, inside the estimate, with exact self-consistency and no prefill regression. Its memory did not fall, because the BF16 copies stay resident for prefill and the draft head. It is a different artifact from the original checkpoint. It became the default model choice on 2026-11-23 and met its image-parity, prefix-reuse, tool-calling and standard-benchmark gates; the app end to end is untested and nothing is installed. [8-bit dense weights][inkaffine8]

Inkling prefill round 3: a dense projection kernel with BF16 tensor operands, one F32 accumulation over full K and 128 × 64 tiles took the instrumented dense stage at 8,800 tokens from about 1,290 to 687 ms and passed the noise-floor gate; a packed draft-cache commit took drafting-on prefill from 5,722 to 4,679 ms; freeing the explicit global mask cut the extra admission at 32K from 1.10 GB to 0.23 GB. Together about 1.11 times over the previous default on the engine diagnostic. Rejected: an NVFP4 kernel modelled on MLX-Serve's `gather_qmm_nax` (gate/up 1,514 to 1,796 ms against 1,477 to 1,611 for the incumbent, cause unexplained) and a width-independent fixed-split router, which changed 2 of 1,768,704 expert selections and failed the gate (worst row 104.6 percent against an 87.3 percent control maximum). Measured, not adopted: 8,192-row commands gain 1.033× for about 2.9 GiB more workspace. A change of 2 routing decisions in 1.8 million can still fail the gate, because routing differences propagate like any other bit change. [Prefill round 3][inkfast3]

Inkling prefill rounds 4 to 6. Staging operands does not help a stage that is not limited by operand reuse: weight-stationary dense (713 against 670 ms), three dense kernels staging both operands in threadgroup memory (1,115 to 1,220 against 967 ms at 8,192-row commands) and weight-stationary NVFP4 at 2,048-row commands (wall 4.99 against 4.34 s, half-empty tiles at about 128 rows per expert) all kept hashes and lost. Reading NVFP4 weights pre-decoded to BF16 from device memory is no faster than decoding them (9.8 against 9.2 ms per visit); the product is decode and tensor work back to back (4.8 plus 4.7 ms), so the lever is overlap. A FLOP floor misled on dense: the stage looked four times its floor (687 against about 175 ms), but a decomposition on synthetic operands put production at about 2.7 ms per projection against 1.8 ms tensor-only and 2.1 ms loads, about 1.5 times its floor; decompose before trusting a FLOP floor. Fusing the prefill selection chain and sharing router weight fragments returned 1.001 to 1.006× (routing time is not in selection); the global causal skip returned nothing (little lies past the diagonal); both were kept because they are hash-identical. BF16 query operands in local attention and BF16 global attention with F32 scores and accumulation passed the natural-text gate (+0.19 points accuracy, 99.28 percent confident agreement) and returned 1.018× with a wide range, far below the 250 to 400 ms the MLX-Serve comparison suggested for attention. [Prefill round 4][inkfast4]

Do not copy another engine's speculation policy without its cost ratios. MLX-Serve verifies only drafts whose running confidence product clears a cost-derived bar and backs off to pipelined plain steps when drafting does not pay. Simulated on Stark traces with same-session costs, the confidence cut lost 0.3 to 3 percent to the best fixed depth (reading confidences first removes Stark's draft and verify overlap, and the draft command runs even when zero rows are kept) and the backoff lost 10 to 15 percent (a drafted round costs only 1.29 times a plain step on Stark, so drafting pays even at about 35 percent acceptance). Stark's history planner, within 1.2 percent of the best fixed depth on every trace and 8 percent ahead of fixed depth 3 on 8-bit prose, is the committed default. A higher prior and faster estimates kept it within 1.5 percent of the best fixed depth in replay; at worker level against fixed depth 3 it gained 8.8 percent on prose and lost 2.4 percent at 8K and 1.4 percent on code. Those losses reproduced in replay only after adding 10 percent noise to the measured round costs: noisy cost estimates talk an adaptive planner down from full depth. Holding full depth while every per-position acceptance estimate is at least 0.70 removed the long-context loss in replay; stiffer hysteresis removed the code loss but cost up to 10 percent elsewhere on its own, and about 5 percent combined with the hold rule. Replay adaptive policies with realistic cost noise before trusting a noise-free replay. [Depth policy][inkround], [planner tuning][inkplan]

Price a skipped stage as a semantic change, then measure it. MLX-Serve never applies the publisher's NVFP4 activation quantization to the expert inputs. Turning it off in Stark saved about 0.8 ms per step and per verify round on both artifacts, and on the original artifact natural-text accuracy moved from 73.64 to 73.76 percent with mean KL 0.0100 from the exact reference. The 8-bit artifact scored 73.99 percent against the same exact reference, prefill was level, and unquantized inputs became the default. Folding the remaining widening pass into the decode products was bit-exact and slower (0.15 ms per step, up to 1.2 ms per four-row round): a conversion moved inside a product repeats for every output channel, so a removed launch can cost more than it saves. [Expert inputs][inkround], [default and fold][inkquant]

Re-check a diagnostic harness on a known-good control after changing it. On Inkling a logit-agreement block added to the paired-control diagnostic (or the prefill changes of the same commit; not bisected) left the diagnostic hanging even on a control that had finished in 4 seconds that morning, so two launch merges tried at the wind-down were reverted unmeasured. The worker, scoreboard and tests completed on the same commit. [Inkling wind-down][inkwind]

A rewrite aimed at an instrumented stage figure can return nothing. The decode stage table put the one-row router at 1.11 ms; a replacement saved 0.06 ms per step and lost 0.56 ms at four rows. Splitting the encoder at stage boundaries inflates small kernels. [Router rewrite][inkround] Every difference from MLX-Serve and its current status is kept on the [Inkling MLX-Serve differences page](model-experiments/inkling-mlx-serve-differences.md).

Copy a faster kernel's structure completely, then time it on your own capture. A first Inkling port of MLX-Serve's gathered NVFP4 matmul (tensor views on both operands) was not faster. Timing MLX-Serve's own kernel on Stark's captured visit showed it 1.4 to 1.6 times faster than Stark's and faster than Stark's tensor-only variant, so the gap was in the kernel structure, not the decode. A second port with register-filled 16 × 32 × 16 fragments, vector activation loads from device, only dequantized weights in threadgroup memory, 128 of 192 threads loading and one barrier per K step was bit-identical but 6 times slower until every fixed loop was fully unrolled; then it was 20 to 35 percent faster per visit and 1.130 times on paired 8,800-token prefill, the default for commands of 768 rows or more. MLX-Serve's bare product is still about 6 ms per visit faster on gate/up. [Gathered NVFP4 kernel][inknax]

Compile shaders offline before trusting a timing run. On Inkling a runtime Metal library compile failed silently three times (a threadgroup array declared in a helper function, a kernel parameter named `thread`, mixed scalar and vector attribute inputs). Each time the run fell back to the old path with identical state, which reads as "correct and no faster"; an offline `xcrun metal` compile showed the error at once. The same session measured twice on stale binaries after a rebuild collided with another workstream's edit. Check the kernel-selection flag in the report and the binary's hash before reading a timing. [Session audit][inkaudit]

Two rewrites that fail the same gate with the same figures are one finding. Two width-independent Inkling prefill routers (a dense-tiled F32 router, then a fixed-split source router) changed the same 2 of 1,768,704 selections and failed the noise-floor gate identically, and neither was faster in pairs (0.94 to 1.00). The gate rejects the arithmetic class, not the kernel; a third rewrite in that class would fail too. [Session audit][inkaudit]

More output channels per SIMD group did not help a one-row product. Six bitwise-identical single-row Inkling NVFP4 variants (one, two or four channels, with and without half-precision activations) took 6.75 to 11.5 ms against 4.80 ms for the incumbent: register pressure and occupancy outweigh the shared loads. [Decode round][inkround]

## Correctness gates

A speed change that is meant to leave output alone has to pass all of these, in this order of cost:

1. Component byte tests on the physical GPU against the one-row or unchanged kernel: real dimensions, edge geometries, zeros, subnormal-heavy inputs, cancellation, offset views and output canaries, over several independent random draws.
2. The real-model control: verify rows, rollback state and a host-serial reference with zero unequal values (`private_checkpoint_mtp_verify_and_rollback_match_serial`, which reads `STARK_TEST_MODEL_PATH`). Its draft-head phase is allowed to differ and reports 248,320 unequal values with no top-1 mismatch. Run the control by name: the prefix `private_checkpoint_mtp_` also selects a test that asserts an official-pack checkpoint and stops on Lily.
3. The output-hash gate: every default arm (greedy and sampled, 0 to 3 drafts) reproduces the baseline hash and the MTP work counters.
4. For a new request option: a greedy request reproduces the greedy baseline hash, and a sampled request gives one hash across repeats.
5. For a change of default: on the same inputs and seeds, the new default path equals the previous build with the option requested explicitly, hash for hash and counter for counter; the opt-out and the kill switch equal the old default; and the frozen baseline arms reproduce with the kill switch set ([schedule default][sched]).
6. For anything that touches state: prefix restoration, image-loaded parity, image transitions, cancellation and reset, per [model onboarding](MODEL_ONBOARDING.md).
7. For a deliberate precision change measured against an exact incumbent: a noise-floor control. Perturb the exact path by one ulp at one element per trial, over several visits and positions on every reference input, and require the candidate's mean error, worst error and top-1 flip rate to lie inside the control span, with its mean at most twice the control median. Keep the exact path reachable as a diagnostic reference, and keep the self-consistency checks (drafting against serial, snapshot replay, determinism) exact. Introduced on Inkling 2 Mini prefill, calibrated on nine 4K captures ([noise-floor gate][inkfast2]). The gate is a sensitivity comparison: on synthetic captures its flip rates do not measure quality (see gate 8).
8. For any claim about output quality, including a precision change or a new quantized artifact: teacher-forced comparison on natural text. Synthetic or padded capture prompts can be nearly uniform; on Inkling's 4,150-token capture the model's top-1 matched the actual next token 0 times in 34, so agreement and flip rates there say nothing about real quality, although those captures stay valid for exactness and self-consistency. On ten natural inputs of 300 to 8,000 tokens (350 rows), the Inkling 8-bit artifact agreed with the original on 94.9 percent of top-1 choices and predicted the actual next token 256 of 340 times against 252 for the original ([8-bit artifact][inkaffine8]). Report agreement, next-token accuracy against the text, KL divergence and confidence-stratified agreement, pooled over rows; the bar is in [`docs/BENCHMARK.md`](BENCHMARK.md). Pooled over 2,580 natural rows, Inkling's default prefill scored 73.41 percent next-token accuracy against 73.64 for the exact reference, with 99.28 percent agreement on confident rows ([natural-text probe][inkround]).

Step 1 is necessary and not sufficient. Kernels are compiled with fast math, and the GPU compiler schedules identical per-row source differently depending on the surrounding code. Recorded cases: a kernel whose intermediate representation matched the one-row body per row and still differed in up to 300 of 2,560 outputs on subnormal streams; a variant that passed one random draw and failed the next; and a variant that passed 288 synthetic cases and failed the real model.

Equal hashes show the change is inert, not that it is faster. A precision change needs matching-reference numerics instead, and hash equality is not a quality qualification.

## Measuring

These rules are for optimization work on one engine. To compare engines, or to state how fast a model is, use the [standard benchmark](BENCHMARK.md): it fixes the scenarios, the metric definitions, the matched-artifact and matched-rendering rules, the host checks and the paired comparison, and it requires a quality result beside every speed result. State the level of every figure taken outside it (server, worker or engine diagnostic).

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
- When several agents share the host, put every build and every GPU job behind one lock, and time only in base, candidate, candidate, base process pairs inside it. On 2026-11-22 the M5 Max's 140 W adapter negotiated 94 W, the battery drained while "charging", the thermal state sat at Fair and the GPU moved between performance states. Identical requests varied by 10 to 15 percent, one ten-minute window ran 1.4 times slower, absolute times drifted 20 to 30 percent over long sessions, and the first request after idle was consistently faster. Ratios from paired runs stayed stable across pairs and launch orders; absolute numbers from separate sessions did not ([exact fast prefill][inkfast1], [prefill attribution][inkattr]).
- Treat a single pair that is slower on every stage, including stages the change does not touch, as host state. Inkling's fused build once measured 58.6 against 65.3 tok/s in a single-invocation benchmark and 5 to 17 percent slower on all stages in one pair; every later pair was level ([second fusion session][inkfuse2]).
- Treat instrumented sub-stage times as upper bounds. Splitting the GPU encoder at stage boundaries to time small stages inflates them: Inkling's one-row router read 1.11 ms instrumented, and removing most of its work saved 0.06 ms in paired uninstrumented timing ([router rewrite][inkround]).
- Keep long-session tooling, build caches and gated binaries under the repository's ignored `runtime/` directory, not in a temporary scratch directory, and check for other sessions' builds before a timed run. On 2026-11-23 a reboot emptied the session's scratch directory and took the lock script, build caches and the gated binaries with it ([Inkling wind-down][inkwind]).
- A lock wrapper must release only the lock it owns, and a timed job should check the lock owner at its end as well as at its start and be discarded if the owner changed. On 2026-11-23 a killed wrapper's armed exit trap removed a lock a timing job had just taken, and a cargo test and build ran alongside the job. Give GPU diagnostics their own deadline: a hung paired diagnostic held the shared lock for 13 minutes, and the harness still has none. After killing a large model process, wait for it to exit before timing anything ([Inkling wind-down][inkwind]).
- Decide policies on traces, not on interleaved live arms, when the host drifts. Inkling's interleaved depth arms moved by up to 25 percent between arms in one session; simulating each policy on recorded acceptance traces with round costs measured in the same session at the same positions gave stable answers, confirmed once at worker level ([depth policy][inkround]).
- Check that the inputs used for a quality figure are text the model can predict. Count how often the reference model's top-1 equals the actual next token; if that is near zero, the inputs measure exactness only. Use natural text for agreement and accuracy figures ([8-bit artifact][inkaffine8], [prefill round 3][inkfast3]).
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
12. Record every outcome, including rejected candidates, and check that the family speed history and playbook reflect the decision. [Experiment ledger](model-experiments/README.md) · [Recording workflow](MODEL_RESEARCH.md#record-the-next-result)

For AngelSlim Hy4-flash Q3_K, inspect the dispatch path on the actual mixed expert banks before interpreting shared-format timings. The pinned Strata grouped prefill requires IQ2_XXS gate/up and leaves the Q3_K gate/up route on per-entry products. The patched llama.cpp Metal reference groups the packed Q3_K/Q4_K products from 32 tokens and uses a 256-row microbatch in the matched diagnostic. Its projection precision and reduction order need their own parity gate. Measure the Q3_K path directly and split decode GPU work from expert and Engram I/O at the host cuts. [Hy4 native diagnostic and prior reference runs](model-experiments/2026-11-25-hy4-native-q3-k.md)

On the native Hy4 Q3_K release worker, grouped F32 and a single 4,096-row chunk reduced observed streamed K2 prefill from 19.497 to 15.665 s and expert reads from 232 to 87 GB. These are worker diagnostics with ambient compiler activity. Keep 1,024 rows for the default prefix grid. Read-only mapped banks and GPU routing improved observed decode with a cold page-fault cost. Consolidating the Engram submissions and switching the grouped operands to F16 produced inconsistent speed and stayed out of the worker. [Measured Hy4 attempts](model-experiments/2026-11-25-hy4-native-q3-k.md)

The native grouped and mapped pilots reproduced all 64 reference output IDs at 4K. On the mapped grouped prefill, 16.15 of 16.30 wall seconds were inside command GPU spans; CPU encode and Engram together took about 30 ms. The same allocation policy took 20 to 26 seconds to emit the first token on a short request in two fresh processes. Measure first use separately from a warmed 4K request, and test the reference's FP16 grouped operand fragments against the F32 native control. [Grouped and mapped attempts](model-experiments/2026-11-25-hy4-native-q3-k.md#20261125-hy4native-05--grouped-streamed-q3_k-prefill)

Increasing the streamed Hy4 group from 1,024 to 4,096 rows lowered software expert reads from 232.36 GB to 87.25 GB at depth two in separate same-binary observations. First-demand mapped decode spent 0.920 of 7.277 wall seconds inside measured command spans, despite zero software expert reads. Capture both stage wall and command spans when changing expert residency. The wider streamed path is opt-in and the mapped path remains a diagnostic option. [Final geometry and mapped comparison](model-experiments/2026-11-25-hy4-native-q3-k.md#20261125-hy4native-09--final-streamed-worker-and-prefill-chunk-size)

On the Hy4 4K repeat, mapped expert admission was followed by a stalled first prompt and 65,461 MiB of observed system swap. GPU budget admission alone did not predict first-use residency. At normal memory pressure, streamed stage sampling placed attention and index work at 6.57 s and routed experts at 5.31 s, with 0.10 s of expert read waits. Only 21% of grouped tiles were eligible for the smaller F32 descriptor and the candidate showed no prefill gain. A bounded 36 GB streamed arena reduced software expert reads 21% on first use and 37% warm, with decode at 17.20 and 23.58 tok/s against 14.49 and 16.39 tok/s for the initial 28 GB pair. Keep first-use and warm requests separate, observe system swap and measure the rate with stage sampling off. Exclude invalid command timestamps and report their count. [Hy4 profiling, tile and expert arena attempts](model-experiments/2026-11-25-hy4-prefill-decode-profile.md)

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
| Inkling: the remaining NVFP4 gap to MLX-Serve's bare product | About 6 ms per visit on gate/up after the gathered kernel; the fused epilogue's share is unmeasured (plain-store diagnostic queued); half-precision scale decode untested ([gathered kernel][inknax]) |
| Inkling: the 14 to 15 ms host gap between prefill commands with drafting on | Seen in the attribution trace; the packed draft commit did not remove it and it was not isolated ([prefill round 3][inkfast3]) |
| Inkling: overlap dense prefill loads with tensor steps | Dense runs at about 1.5 times its tensor floor (about 2.7 against 1.8 ms per projection on synthetic operands); staged and stationary tiles lost ([prefill round 4][inkfast4]) |
| Inkling: a TF32-class prefill router and fewer expert-input passes | Routing 371 ms plus quantization 136 ms at 8,800 tokens; fusing the selection chain returned nothing, so the projection itself is the target. Estimate 200 to 400 ms with the input passes (source comparison, not measured) ([prefill round 4][inkfast4]) |
| Inkling: remove the expert-input widening pass from wide prefill | About 136 ms of instrumented time at 8,800 tokens; expected hash-identical. Direct gathered kernels were compiled as a candidate and reverted unmeasured when optimization stopped; the decode-side fold was slower ([default and fold][inkquant], [wind-down][inkwind]) |
| Inkling: a planner rule that removes the code loss | Worker 0.986 and noisy replay 0.984 against fixed depth 3 on code; stiffer hysteresis fixes it with side effects of up to 10 percent on its own. The planner unit tests now pass after an attribute fix ([planner tuning][inkplan], [wind-down][inkwind]) |
| Inkling: pipelined plain steps (encode step N+1 while N runs) | MLX-Serve does this; estimate 0.3 to 0.8 ms per plain step (source comparison, not measured) |
| Inkling: fold the commit into the next round's draft command | About 0.3 to 0.45 ms per round from the worker-loop study ([worker loop][inkround]) |
| Inkling: remaining expert-tail and controller fusion on the single row (output gate in the projection, SwiGLU in up, one input kernel per visit) | Estimate 0.5 to 1.5 ms per token (source comparison, not measured) |
| Inkling: the single-row and verifier launch merges tried at the wind-down (router join into route selection, about 48 launches; NVFP4 up writing the down inputs, 44; slot-3 controller norm pair into the attention mix, about 24) | Reverted unmeasured after the paired diagnostic hung; fix or bisect `--paired-control` first ([wind-down][inkwind]) |
| Inkling: wider verification (4 or 5 drafts) at long context | Estimate +8 to +12 percent at about 8K if conditional acceptance stays at 0.88; several days of kernel work ([depth policy][inkround]) |
| Inkling: qualify the Stark 8-bit affine dense artifact | Owner decided it becomes the default; 3.3 ms faster per step in the engine and natural-text quality passes the bar. Catalog integration, image parity, prefix-reuse validation, tool calling and the standard benchmark are done; the app end to end and installation remain. A four-row dense kernel (4.2 ms against 2.6 ms at one row) and dropping the resident BF16 copies are open ([8-bit dense weights][inkaffine8]) |
| Inkling: 8-bit weights for the draft head's dense layers, then a cheaper vocabulary read | The draft command is 3 to 4 ms per round, almost all a fixed block forward that does not use the 8-bit path; it matters most on low-acceptance text. Not quantified ([session audit][inkaudit]) |

## Records

The [experiment selection table](#choose-a-performance-experiment) links the GLM, Gemma, Ripple, Hy4, dense Qwen and Elephant studies. The chronology below follows the Flash-Next and Inkling work.

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
| 2026-11-22 | [Inkling 2 Mini decode toward 80](model-experiments/2026-11-22-inkling-mini-decode-80.md) | Exact residual and projection batching, MLX-Serve comparison, local KV sharing and wider verifier budget |
| 2026-11-22 to 11-23 | [Exact fast prefill][inkfast1], [unguarded MXFP8 and the noise-floor gate][inkfast2] | NVFP4 guard removal and fused epilogues, faster exact MXFP8 replay, unguarded MXFP8 default under a perturbation-calibrated gate, mask-free global attention |
| 2026-11-22 to 11-23 | [Launch fusion, both sessions][inkfuse2] | Verifier and single-row fusions, paired timing showing launch count does not set verifier cost |
| 2026-11-22 | [Engine matrix runs 1 and 2][inkengine], [prefill attribution][inkattr] | Matched-mode engine comparisons with MLX-Serve and the 8-bit pack, per-token prefill fits, draft work and clock state in prefill |
| 2026-11-22 to 11-23 | [8-bit dense weights][inkaffine8] | Bandwidth estimate, byte-exact reproduction of the published 8-bit pack, the Stark 8-bit engine artifact and natural-text agreement |
| 2026-11-23 | [Prefill round 3][inkfast3] | Dense projection kernel, packed draft commit, mask-free workspace, width independence, rejected NVFP4 and router designs, command-width measurement |
| 2026-11-23 | [Prefill round 4 to 6][inkfast4], [decode round][inkround], [wide prefill worker][inkwidth], [MLX-Serve source comparison][inkmlxsrc], [differences page](model-experiments/inkling-mlx-serve-differences.md) | Rejected stationary and staged kernels, expert ceiling, BF16 prefill attention, dense decomposition; short-context verifier attention, rejected router rewrite, worker loop, depth policy against MLX-Serve rules, unquantized expert inputs, natural-text quality; wide commands and the cache fault; the maintained comparison with MLX-Serve |
| 2026-11-23 | [Expert inputs default and fold][inkquant], [planner tuning][inkplan], [gathered NVFP4 kernel][inknax] | Unquantized expert inputs by default and the rejected fold, planner retuning and hold rule, the gathered prefill kernel in MLX-Serve's tile structure |
| 2026-11-23 | [Inkling wind-down][inkwind] | Optimization stopped by owner decision; reverted unmeasured launch merges and prefill items, the paired-control hang, the lock incidents |
| 2026-11-23 | [Inkling 2 Mini standard benchmark][inkstd] | First standard result: three repeats, quality attached, standings against MLX-Serve on matched artifacts |
| 2026-11-23 | [Inkling session audit][inkaudit] | Attempts, intermediate figures and incidents of 2026-11-22 to 11-23 that the records lacked; corrections to earlier records; the merged tree's lint commit that the gates did not cover |
| 2026-11-24 | [Inkling DFlash 2 matched verifier traces][ink-draft-traces] and [live standard benchmark][ink-draft-live] | Proposal width, physical target calls and workload-specific throughput against DSpark and MLX-Serve |
| 2026-11-25 | [Inkling cross-drafter recovery][ink-draft-cross], [runner-up replay][ink-draft-selector] and [live paired timings][ink-runner-live] | First-miss coverage, time per correction and the cost of displacing a correct third choice |
| 2026-11-25 | [Flash Next three-slot MTP runner-up][flash-runner] | Exact branch commits, paired prose and coding results, and comparison with shipped adaptive MTP |
| 2026-11-25 | [Hy4-flash 4K prefill and decode attribution](model-experiments/2026-11-25-hy4-prefill-decode-profile.md) | Mapped first-use swap, prefill encoder stages, rejected smaller expert tile and bounded streamed arena sensitivity |

The 2026-11-17 branch records are not in this tree. They are on `codex/flashnext-speed-review-20261117` at commit `2e3058d1`, and the links above point there.

The family histories link the dated comparisons, including superseded results and the limits of incomplete trials. [Flash-Next and Lily](model-experiments/speed-flash-next.md) · [Inkling 2 Mini](model-experiments/speed-inkling.md) · [Dense Qwen and Qwen MoE](model-experiments/speed-qwen.md) · [GLM, Gemma, OLMo, Ripple and scoring models](model-experiments/speed-other-families.md) · [Model experiment ledger](model-experiments/README.md)

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
[inkfast1]: model-experiments/2026-11-22-inkling-mini-exact-fast-prefill.md
[inkfast2]: model-experiments/2026-11-22-inkling-mini-unguarded-mxfp8-prefill.md
[inkfuse2]: model-experiments/2026-11-22-inkling-mini-launch-fusion.md#20261122-inkfuse-02--second-session-paired-timing-and-single-row-fusions
[inkengine]: benchmarks/2026-11-22-inkling-mini-engine-matrix.md
[inkengine2]: benchmarks/2026-11-22-inkling-mini-engine-matrix.md#20261122-inkengine-02--second-run-with-the-author-8-bit-pack
[inkattr]: benchmarks/2026-11-22-inkling-mini-prefill-attribution.md
[inkaffine8]: model-experiments/2026-11-22-inkling-mini-affine8.md
[inkfast3]: model-experiments/2026-11-23-inkling-mini-prefill-round3.md
[inkfast4]: model-experiments/2026-11-23-inkling-mini-prefill-round4.md
[inkround]: model-experiments/2026-11-23-inkling-mini-decode-round.md
[inkwidth]: model-experiments/2026-11-23-inkling-mini-wide-prefill-worker.md
[inkmlxsrc]: benchmarks/2026-11-23-inkling-mini-mlx-serve-source.md
[inkquant]: model-experiments/2026-11-23-inkling-mini-expert-inputs-default.md
[inkplan]: model-experiments/2026-11-23-inkling-mini-planner-tuning.md
[inknax]: model-experiments/2026-11-23-inkling-mini-gathered-nvfp4.md
[inkwind]: model-experiments/2026-11-23-inkling-mini-wind-down.md
[inkstd]: benchmarks/2026-11-23-inkling-mini-standard-benchmark.md
[inkaudit]: model-experiments/2026-11-23-inkling-mini-session-audit.md
