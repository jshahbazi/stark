# Performance playbook

What to try when making a model family faster on Stark, in roughly the order to try it, with what each idea returned on Qwen3.8-Flash-Next ("Lily"). Compiled 2026-11-19 and updated 2026-11-20 from the records listed at the end.

This page summarizes records; it is not one. Every number was measured on a named artifact, build, workload and host, and keeps that scope. Nothing here qualifies another artifact, and where this page and a dated record differ, the record is right. A technique that paid on Lily has to be measured again on the next family. One that failed on Lily failed for a stated reason, and the reason is the thing to check before trying it again.

Unless a row says otherwise, figures are for `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at revision `0f94842e`, on an Apple M5 Max with 128 GiB, with 4,096 input tokens.

For comparing a new family against it: Lily is a 48-layer hybrid, 36 GDN recurrent layers and 12 sparse-attention layers. Every layer has a routed mixture-of-experts feed-forward (512 experts, 10 routed per token, plus a shared expert) and four hyper-connection streams. It has an n-gram embedding table of about 30 GiB, a one-layer draft head and a vocabulary of 248,320. The pack is about 107 GB.

## Keeping this page current

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
