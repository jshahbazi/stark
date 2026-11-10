# Benchmarks

Measured results, each bound to the source revision, build and host named inside it. This index says which result is current and which has been superseded. Earlier archived comparison artifacts are under [archive/experiments](../archive/experiments/). Each report states where its source-bound aggregates are retained.

Track each proposed optimization, failed approach and promotion decision in the
[model experiment ledger](../model-experiments/README.md), using its
[attempt template](../model-experiments/TEMPLATE.md). Keep the source-bound
measurements and current/superseded benchmark status here; ledger entries link
to them and must not broaden their claims. New studies need an attempt ID and
linked source-bound evidence. Existing reports retain their original
qualification; the ledger is not a completed historical backfill.

## Current result

**Remaining Qwen headless image-loaded text controls (2026-11-09).**
The [bounded comparison](2026-11-09-qwen-remaining-image-text-parity.md)
extends immutable shared fast text storage to BF16 0.8B, affine 0.8B and affine
MoE 35B. Matched MTP0 decode, first-token and request medians pass the registered
five-percent bound. Complete cache/lifecycle matrices, BF16 source heads, other
artifacts and retained ddalcu remain separate open gates.

**Qwen dense image-default parity (2026-11-09).** The complete 27B image artifact
now preserves its ordinary optimized text graph and batched MTP while the image
tower stays loaded. The [QDIP-07 comparison](2026-11-09-qwen-dense-image-text-parity.md)
measures 48.309 versus 48.315 decode tok/s at MTP3 on a matched 64-token workload;
MTP0 also shows no observed regression. Output hashes and all 14 nonperiodic
near-capacity cache/lifecycle controls pass. The earlier 94.44% regression is
preserved as a failed predecessor. This is bounded Jundot artifact evidence;
startup, broader workloads, other models and installed-release behavior remain
separate qualifications.

**Retained Lily vision-loaded text MTP (2026-11-08).** A
[short paired check](2026-11-08-lily-vision-text-mtp.md) observes 104.71 tok/s
on the previous text-only worker and 105.49 tok/s with vision and text MTP
loaded, identical outputs, zero image drafts and correct post-image text/cache
recovery. Two warm observations per arm; no host/thermal screening. This does
not supersede the longer Lily studies below.

**Qwen vision artifact text prefix reuse (2026-11-08).** On the official
Qwen3.5-0.8B BF16 artifact, 32 exactness cases pass and cached follow-ups reuse
1,024 tokens. Four observations per timing arm show follow-up first-token
medians of 0.93–1.20 seconds against 2.25–2.54 seconds fresh. Cache-cold medians
are 2.11 and 2.10 seconds. This is a bounded synthetic observation without
thermal/background-load screening. See the [paired timing and limits](2026-11-08-qwen35-vision-text-prefix.md).

**Cross-backend idle wake follow-up (2026-11-08).** Extends queue maintenance
to official mixed-affine Flash-Next generation, native Flash-Next scoring and
converted text generation on Metal4. The
[matched study](2026-11-08-cross-backend-idle-wake.md) binds separate artifacts,
baseline/candidate/candidate/baseline controls, score-cache and full-vocabulary
hash checks, prefix restoration, lifecycle and idle-resource results. This
extends the original Lily-generation study below; its source-bound results
remain unchanged.
Official first-sample latency falls from 1.12–1.13 to 0.31–0.32 seconds;
the fully screened dense repeat falls from 0.71–0.75 to about 0.62 seconds.
Fresh scoring is observed at about 0.30 seconds, with a documented host-coverage
gap in one primary candidate call.

**Flash-Next idle wake fix (2026-11-07).** The native Lily-policy generation
worker now signals its otherwise empty GPU queue every 500 ms while waiting
for a new request. It retains prefix state and its original expiry deadline.
Matched post-idle first-sample latency falls from 1.07–1.09 seconds to
0.30–0.31 seconds, with 16/16 host screens passing and matching seeded outputs.
Observed idle worker CPU is 0.033% of one core.
See the [matched latency and idle-resource validation](2026-11-07-flashnext-idle-wake-fix.md)
and [source archive](../archive/experiments/2026-11-07-flashnext-idle-wake/README.md).

**Flash-Next sustained throughput diagnosis (2026-11-07).** The unchanged
baseline reproduces a fall from 95.46 to 82.20 decode tok/s; about 98% of added
verification elapsed time is inside GPU command spans, with contemporaneous
clock reductions. An idle queue control separately avoids about 0.7 seconds
before the first sample. A single-model crossover establishes no reliable
projection gain; the original projection dispatch is restored. All 128
diagnostic arms are retained, 125 pass host screens, and timing remains
unqualified. See the [diagnosis](2026-11-07-flashnext-sustained-diagnosis.md),
[aggregates](data/flashnext-sustained-diagnosis-2026-11-07.json) and
[source archive](../archive/experiments/2026-11-07-flashnext-moe/README.md).

**Qwen3.8-Flash-Next MoE projection experiment (2026-11-07).** Fixed-geometry
expert gate/up and shared-first dispatch pass bounded numerical, scoring and
prefix-restoration controls. The final synthetic combined projection improves
by about 2.2%. A completed observed ABBA run measured **81.23 decode tok/s with
two MTP drafts versus 85.50 baseline**, and **75.63 versus 80.33 tok/s with MTP
off**. It has no quiet admission and four host-ineligible arms; the available
results do not support promotion. See the [experiment report](2026-11-07-flashnext-moe-projections.md),
[initial aggregate results](data/flashnext-moe-projections-2026-11-07.json),
[completed observed run](data/flashnext-moe-matched-observed-2026-11-07.json) and
[source patches](../archive/experiments/2026-11-07-flashnext-moe/README.md).
The later [cadence diagnostic](data/flashnext-moe-cadence-2026-11-07.json)
recovers 95.61 tok/s on the same final candidate after a ten-second pause,
versus 89.45 on the fourth consecutive request. All eight candidate arms
pass their host screens; no quiet-qualified kernel gain is established.
This adds separate candidate evidence and does not supersede earlier results.

**Qwen3.8-Flash-Next optimization harness pilot (2026-11-07).** Eighteen cells
and 144 measured requests passed bounded repeatability/candidate-hash checks.
Draft depths one through three and a 16-to-64-row concurrent-encoder cutoff
were tested on the pinned Lily artifact, with an independent corpus/seed case
and separate stage diagnostics. Host interference prevented timing qualification;
the encoder candidate is not promoted and no production default changed.
See the [pilot report](2026-11-07-flashnext-optimization-pilot.md) and
[aggregate results](data/flashnext-optimization-pilot-2026-11-07.json).
This is separate evidence and does not supersede the qualified historical
Flash-Next MTP records below.

**Dense Qwen3.8-27B oQ4e packed-weight follow-up (2026-11-06).** The final
host-screened run measured **51.48 useful decode tok/s median**, with all six
768-token responses passing on their first attempt. The **70.8 tok/s target
remains unmet**. An earlier balanced, unscreened packing-only comparison measured
48.96 versus 45.25 tok/s (+8.19%); it does not isolate the later predictor changes.
Final numerical, scoring and separate original/three-bit/four-bit prefix checks
passed. See the [packed-weight follow-up](2026-11-06-qwen38-27b-packed-followup.md)
and [public aggregate data](data/qwen38-27b-packed-followup-2026-11-06.json).
The [approach ledger](2026-11-06-qwen38-27b-approaches.md) records the successful,
rejected, optional and unimplemented approaches and the evidence for each decision.
The [original source-bound record](2026-11-06-qwen38-27b-oq4e-mtp.md) remains
unchanged. Neither report applies to the separate Flash-Next artifacts.

The follow-up's seven-draft trial advanced 3.85 tokens per round versus 3.02
with three drafts, while target GPU time rose from 47–50 ms to 79–85 ms and
throughput fell from 48–50 to 35–37 tok/s. That experiment was removed; the
shipped dense path is capped at three drafts.

**GLM-5.3-Flash focused prefill validation (2026-11-03).** The focused branch's
same-binary MMA off/on comparison measured 34.90 to 40.87 prefill tok/s (+17.1%)
and 2.970 to 2.565 seconds per warm request (13.7% lower), with four accepted
measurements per arm and exact generated IDs. This is a 96-input/8-output
greedy workload within the opt-in bounded profile, not a comparison against
main's normal defaults or MLX-Serve. See the [fresh validation record](../archive/runs/2026-11-03-glm53-prefill-merge-validation.md)
and [public aggregates](data/glm53-prefill-merge-validation-2026-11-03.json).
It supersedes the older result below for this branch; the historical source-bound
evidence remains unchanged.

**GLM-5.3-Flash batched prefill experiment (2026-11-02).** Historical v48 T8 receipts qualify a default-off recipe for the measured M5 Max workloads: screened 96-token prefill was 16.6% faster and request time 13.2% lower than the matched v46c control. The experiment's source receipt binds to its recorded v48 source revision, not this branch. See the [bounded experiment report](GLM53_BATCHED_PREFILL_2026_11_02.md) and [public aggregate data](data/glm53-batched-prefill-2026-11-02.json).

**Qwen3.8, identical checkpoint, maximum sampled throughput (2026-10-30).**
Stark peaks at **66.81 tok/s** versus **61.17** for stock MLX-Serve 26.10.3,
**9.21% higher**, with the same verified files, quantization and no speculation.
Ten measured 4K/256 requests per engine ran while the user reserved the machine
and confirmed other apps were closed. Exact component controls and retained
Lily output parity pass. Nine aligned next-token choices match; longer greedy
continuations still differ, so whole-model numerical and quality equivalence
remain unestablished. The [optimization record](QWEN38_MATCHED_OPTIMIZATION_2026_10_30.md)
records the 389-check aggregate audit, timing conventions and machine-readable
results. The earlier [58.13/60.95 baseline](QWEN38_SAME_CHECKPOINT_STARK_MLX_SERVE_2026_10_30.md)
is retained unchanged.

**Qwen3.8 exact MTP, two checkpoint packs (2026-10-31).** At 4K input and
256 sampled outputs with top-k 20, two drafts, and exact sampling, the optimized
Lily path measured **82.85 tok/s on vs 72.97 off** (+13.54%, 16/16 paired wins).
The official mixed-4/8-bit checkpoint measured **65.02 vs 56.69 tok/s**
(+14.69%, 16/16 wins). These are within-checkpoint Stark MTP on/off results,
not a new MLX comparison; the two weight packs remain separate. Lily nucleus
sampling gained 1.70%. MTP remains off by default, and these bounded runs do
not establish a general throughput floor. See the [Lily report](QWEN38_MTP_COMBINATION_2026_10_31.md)
and [official-checkpoint report](QWEN38_OFFICIAL_BATCHED_MTP_2026_10_31.md).

**Earlier Qwen3.8 comparison, different quantization profiles (2026-10-30).**
The Lily-format Stark checkpoint peaked at 79.52 tok/s against 61.00 for the
official MLX pack. Its [source-bound report](QWEN38_STARK_MLX_SERVE_2026_10_30.md)
retains pooled rates and block drift. Its two quantization profiles and
measurement window are separate from the current fixed-checkpoint comparison.

**Generation throughput, Qwen3.5-2B 4-bit, Apple M5 Max, one request at a time (2026-10-22).** Stark `97f15f38` measured 245.5 tokens/s aggregate against 218.8 for MLX-LM, 174.7 for vLLM Metal, 147.9 for llama.cpp and 125.6 for TensorFold, and led all six workloads. Four independent loads per configuration, three repeats per workload, 432 requests. Rates include input processing, generation, transport and cleanup; loading and warmup are excluded. [Full report](QWEN35_ENGINE_COMPARISON_2026_10_22.md), [machine-readable results](../../website/public/benchmark-results.json).

**Conversion time and memory against Lalamo.** Collected across converter milestones; see [conversion benchmarks](CONVERSION_BENCHMARKS.md).

**Native GLM-5.3-Flash and Qwen3.8-Flash-Next integration (2026-10-28).**
Qwen's initial mixed-precision Lily-format path measures 77.18 tok/s warm decode at 4K
input and 256 outputs, with speculation disabled. The original Pipe checkpoint
improves from 23.16 to 43.70 tok/s on its matched short F32 workload. See the
[mixed-precision study](QWEN38_MIXED_PRECISION_2026_10_28.md) for distinct artifacts,
precision policies and source validation. GLM's current 9.04 versus 8.13 tok/s
result remains in the [decode graph study](GLM_QWEN_DECODE_GRAPH_2026_10_28.md).
Earlier Qwen throughput and kernel measurements remain historical records for
their named builds.

**Qwen3.8 sampled generation at fixed quantization (2026-10-28).** Repeated
4K-input/256-output runs measure top-k 20 sampling at 75.08 tok/s versus 50.62
for the preserved worker, and nucleus-only sampling at 65.57 versus 49.73.
Both use temperature 1, top-p 0.95, seed 42 and the same Lily q4-xl checkpoint,
without speculation. All output hashes match within each sampling mode. The
[sampled BF16 study](QWEN38_SAMPLED_BF16_2026_10_28.md) retains both measurement
windows, profiles GPU/CPU stages and records exact sampler validation.

**Qwen3.8 HC and GDN kernels (2026-10-29).** Fixed-quantization paired component
controls reduce HC time by 5.40%, GDN input projection/convolution by 5.62%, and
the recurrent output wrapper by 1.15%. Retained routes match full-logit and
seeded sampled references. MoE trials are rejected. The [kernel study](QWEN38_KERNELS_2026_10_29.md)
records every sampled stage and separates component gains from unstable
between-load throughput measurements. A private same-loaded-model control
observes +1.93% pooled top-k 20 decode and effectively unchanged nucleus
throughput; earlier source-bound records are retained.

**Qwen3.8 sampled throughput during desktop activity (2026-10-29).** Real-readout
paired controls reduce top-20 selector time by 73.5% and full-vocabulary ordering
by 43.2%, preserving complete candidate orders and seeded outputs. The
[stability study](QWEN38_STABILITY_2026_10_29.md) records same-loaded-model
activity/QoS/feedback-queue controls and correlates throughput swings with
concurrent GPU work. These controls do not establish consistent 77–80 tok/s
during normal desktop multitasking. Earlier kernel and throughput records
remain source-bound historical evidence.

**Qwen3.8 AC power-mode control (2026-10-29).** The same worker measures 72.88,
66.79 and 65.41 pooled top-k 20 tok/s across sequential Automatic, High Power
selected, and restored Automatic phases. Desktop GPU load and clocks vary
materially, so the [power-mode study](QWEN38_POWER_MODES_2026_10_29.md) does not
isolate a causal mode effect. All 60 outputs match their references, the
77–80 tok/s target remains unmet, and the original preferences are restored.

## Host request latency

**Native chat-template isolation (2026-11-03).** A fresh Rust helper reduces
median short rendering from 48 to 9 ms and 64 KiB-history rendering from 59 to
12 ms in the matched component study. Decision scoring's two sequential renders
fall from 100 to 18 ms. The [native study](NATIVE_CHAT_TEMPLATE_2026_11_03.md)
records exact-output controls, resource limits, compatibility restrictions and
source hashes. The [initial Python baseline](CHAT_TEMPLATE_ISOLATION_2026_11_03.md)
is retained unchanged. End-to-end inference latency was not measured; earlier
engine throughput results remain bound to their original timing windows.

**Source-qualified OLMo 3 and Gemma 4 Q4 text (2026-11-02).** On Apple M5 Max, Gemma's source fusions improve median paired combined-work rate 2.57% over C3 and reach MLX parity (measured paired rate 0.44% above MLX). OLMo shows no material change from C3 (-0.05%) and measures 2.31% above MLX. Four loads per engine, three measured requests per load, all guards passing and matching 128-token hashes accompany the numerical qualification. Rates cover the complete 512-input/128-return request. See the [source fusion comparison](OLMO3_GEMMA4_SOURCE_FUSIONS_2026_11_02.md) and [aggregate results](olmo3-gemma4-source-fusions-2026-11-02.json). The [2026-10-29 C3 results](OLMO3_GEMMA4_Q4_REPAIRED_2026_10_29.md) remain historical.

## Engine comparison history

| Study | Date | Status |
| --- | --- | --- |
| [Qwen3.8 matched-checkpoint optimization](QWEN38_MATCHED_OPTIMIZATION_2026_10_30.md) | 2026-10-30 | Current: same files and quantization, no speculation; maximum 66.81 vs 61.17 tok/s (+9.21%); numerical equivalence unestablished |
| [Qwen3.8 same-checkpoint maximum throughput baseline](QWEN38_SAME_CHECKPOINT_STARK_MLX_SERVE_2026_10_30.md) | 2026-10-30 | Historical baseline before the matched-checkpoint optimization: maximum 58.13 vs 60.95 tok/s; source-bound evidence retained |
| [Earlier Qwen3.8 Stark / MLX-Serve comparison](QWEN38_STARK_MLX_SERVE_2026_10_30.md) | 2026-10-30 | Historical different-pack comparison: maximum 79.52 vs 61.00 tok/s; does not isolate engine efficiency |
| [Qwen3.5 engine comparison](QWEN35_ENGINE_COMPARISON_2026_10_22.md) | 2026-10-22 | Current: five engines, six workloads, 12.2% over MLX-LM |
| [TensorFold stock Qwen3.5 comparison](TENSORFOLD_STOCK_QWEN35_COMPARISON.md) | 2026-10-16 | Superseded by the 2026-10-22 comparison, which includes TensorFold |
| [TensorFold JevK5 scoring comparison](TENSORFOLD_JEVK5_COMPARISON.md) | 2026-10-16 | Scoring-path comparison; not repeated since |
| [Stock engine benchmark pilot](STOCK_ENGINE_BENCHMARK_2026_10_19.md) | 2026-10-19 | Pilot run; superseded by the 2026-10-22 comparison |
| [Qwen3.5 MLX quiet-window comparison](QWEN35_MLX_QUIET_COMPARISON_2026_10_20.md) | 2026-10-20 | Reported 7.36% over MLX-LM; superseded by the 2026-10-22 comparison |
| [Benchmark brief](STARK_BENCHMARK_BRIEF_2026_10_21.md) | 2026-10-21 | Summary of the 7.36% result; superseded |
| [Inference comparison evidence inventory](INFERENCE_COMPARISON_EVIDENCE.md) | rolling | Inventory of every recorded engine, reference and conversion result up to 2026-10-21 |
| [OLMo 3 and Gemma 4 Mac engine evidence](OLMO3_GEMMA4_MAC_ENGINES.md) | 2026-10-28 | Published benchmark evidence, comparison limits, source licensing and native implementation techniques |
| [OLMo 3 and Gemma 4 source fusion comparison](OLMO3_GEMMA4_SOURCE_FUSIONS_2026_11_02.md) | 2026-11-02 | Current for these pinned Q4 sources: Gemma +2.57% over C3 and at MLX parity; OLMo unchanged from C3; all guards and output hashes pass |
| [Repaired OLMo 3 and Gemma 4 Q4 comparison](OLMO3_GEMMA4_Q4_REPAIRED_2026_10_29.md) | 2026-10-29 | Superseded timing by the 2026-11-02 source fusion comparison; original qualification and paired Stark/MLX rates 1.0155 and 0.9785 preserved |
| [Matched native OLMo 3 and Gemma 4 comparison](OLMO3_GEMMA4_NATIVE_MATCHED.md) | 2026-10-28 | Historical diagnostic: original failed numerical qualification and timings preserved; superseded for these checkpoints by the repaired 2026-10-29 comparison |

## Optimization records

The [Qwen3.8 optimization history](QWEN38_OPTIMIZATION_HISTORY.md) catalogs 142
attempts and controls across the original F32 path, mixed precision, sampling,
Metal kernels, Uzu execution and matched-checkpoint work. It records retained,
rejected, inconclusive and deferred outcomes with source and evidence links.

| Study | Outcome |
| --- | --- |
| [Qwen3.8 mixed-4/8-bit exact batched MTP](QWEN38_OFFICIAL_BATCHED_MTP_2026_10_31.md) | Same MLX-Serve checkpoint: 65.02 tok/s on versus 56.69 off (+14.69%, 16/16 paired wins); separate original non-MTP worker ABBA +10.88%; quantization and exact sampling preserved; absolute rates drift and 77–80 tok/s remains unmet |
| [Qwen3.8 MTP with the optimized Metal branch](QWEN38_MTP_COMBINATION_2026_10_31.md) | Lily: 82.85 tok/s on versus 72.97 off (+13.54%, 16/16 wins); nucleus +1.70%; results are Stark on/off within each distinct checkpoint, not an MLX comparison; MTP remains off by default |
| [Qwen3.8 matched-checkpoint Metal optimization](QWEN38_MATCHED_OPTIMIZATION_2026_10_30.md) | Same-checkpoint maximum 66.81 vs 61.17 tok/s (+9.21%), no speculation; exact stock component controls and retained Lily parity; 77–80 tok/s and whole-model quality equivalence remain unestablished |
| [Qwen3.8 converted-format speed](QWEN38_UZU_SPEED_2026_10_30.md) | NEON affine restoration reduces median process-to-ready time from 20.54 to 18.09 seconds; exact nucleus-prefix work and sampled controls retain quantization; consistent 77–80 tok/s remains unqualified |
| [Qwen3.8 Uzu conversion and generation](QWEN38_UZU_FORMAT_2026_10_29.md) | Full quantization-preserving conversion, self-contained load and exact readout/sample parity; paired timing series has substantial throughput drift |
| [Qwen Metal hotpaths and finite readout checks](QWEN38_METAL_HOTPATHS_2026_10_29.md) | Quiet baseline 73.62 top-k / 69.85 nucleus tok/s; MoE, HC and GDN trials rejected or inconclusive; retained exact SIMD finite checks reduce guard elapsed time, with no reliable end-to-end gain or desktop target acceptance |
| [Qwen exact MTP sampled generation optimization](QWEN38_MTP_OPTIMIZATION_2026_10_30.md) | Previous Lily long run: 78.33 tok/s on versus 67.46 off across 16 pairs (+16.11%); superseded by the 2026-10-31 timing study, with source-bound numerical evidence and original implementation preserved |
| [Qwen batched exact MTP verifier](QWEN38_BATCHED_MTP_2026_10_30.md) | Historical initial batched build: 90 complete target readouts match ordinary decode byte for byte; the initial two-draft throughput windows are superseded by the longer study, source-bound numerical evidence retained |
| [Qwen serial exact MTP generation](QWEN38_EXACT_MTP_2026_10_30.md) | Historical serial verifier: 56 complete target readouts match; sampled MTP is 22.6–40.1% slower; throughput superseded by the long-run study, source-bound numerical evidence retained |
| [Qwen residual fusion and exact CPU sampling](QWEN38_FUSION_2026_10_29.md) | MoE down/injection component 5.55% faster; exact probability mapping 58.7% faster; nucleus same-loaded observation +1.3%, with broad overlap and an unmet 77–80 tok/s target |
| [Qwen iQ-MLX 4.7 bpw investigation](QWEN38_IQ47_INVESTIGATION_2026_10_29.md) | Exact pack unmeasured: storage and loader support pending; publisher's M5 Ultra result uses MTP; fresh existing-checkpoint Stark control is 72.08 greedy / 64.92 sampled warm tok/s |
| [Qwen AC power-mode control](QWEN38_POWER_MODES_2026_10_29.md) | Same inference build; 60 exact requests; selected High Power does not recover the target in this window; original settings restored |
| [Qwen desktop throughput stability](QWEN38_STABILITY_2026_10_29.md) | Current sampler and host diagnostics; exact selector gains; normal-desktop 77–80 tok/s acceptance not established |
| [Qwen HC and GDN kernels](QWEN38_KERNELS_2026_10_29.md) | Current kernels; exact references and component gains; same-loaded-model top-k observation +1.93%, nucleus unchanged; between-load throughput remains unstable |
| [Qwen sampled generation and BF16 profiling](QWEN38_SAMPLED_BF16_2026_10_28.md) | Sampler/scheduling baseline at fixed quantization: repeat-window warm gains of 48.33% with top-k 20 and 31.84% without top-k; exact seeded outputs |
| [Qwen mixed-precision native inference](QWEN38_MIXED_PRECISION_2026_10_28.md) | Initial Lily integration: 77.18 tok/s at 4K without speculation; original Pipe/F32 checkpoint 43.70 tok/s on the matched short workload |
| [Native GLM and Qwen decode graphs](GLM_QWEN_DECODE_GRAPH_2026_10_28.md) | Current GLM result; earlier Qwen throughput superseded, source-bound validation retained |
| [Native GLM and Qwen graph and kernels](GLM_QWEN_NATIVE_GRAPH_2026_10_28.md) | Earlier stage-chaining study; superseded throughput, source-bound validation retained |
| [Native GLM and Qwen optimization](GLM_QWEN_NATIVE_OPTIMIZATION_2026_10_28.md) | Initial no-copy and batching study; build identities and validation remain source-bound historical evidence |
| [MLX-informed generation optimization](MLX_GENERATION_OPTIMIZATION_2026_10_20.md) | Two decode changes accepted after bitwise validation; the throughput they produced is included in the current result |
| [DeepInfer resource reuse](DEEPINFER_ADOPTION_2026_10_21.md) | Two experiments measured and reverted; patches retained under the archive |
| [Qwen3 GGUF import experiment](QWEN3_GGUF_IMPORT_EXPERIMENT.md) | Experimental Q8_0 import route; see the [converter guide](../../tools/converter/README.md#experimental-qwen3-gguf-import) |
| [Qwen3 GGUF prefill optimization](QWEN3_GGUF_PREFILL_OPTIMIZATION.md) | About 15% higher prefill throughput for the imported model at 512 and 1,024 tokens, exact outputs on the tested cases |

## How to read these studies

Every study names its Stark commit, engine versions, model, quantization and host. Results apply to that build; they are not regenerated when the code changes. A new comparison should be added here with its date and should state which earlier rows it supersedes. Throughput studies do not measure answer quality; the 2026-10-22 report includes separate answer-format checks.
