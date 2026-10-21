# Stark comparisons: evidence inventory and future benchmark plan

Follow-up: [2026-10-20 MLX-informed generation optimization](MLX_GENERATION_OPTIMIZATION_2026_10_20.md) adds source investigation, before/after Stark measurements, current MLX-LM results, rejected kernel experiments and explicit timing exclusions. Its modified Stark build is a separate comparison from the earlier stock pilot.

Compiled: **2026-10-19**. Status: historical evidence and research for a future
benchmark. No new model runs or article publication were performed for this
document.

Follow-up: the [2026-10-19 stock engine pilot](STOCK_ENGINE_BENCHMARK_2026_10_19.md)
adds measured current MLX-LM, llama.cpp, TensorFold and vLLM Metal comparisons,
stock JevK5 admission and scoring, and a vLLM concurrency check. Its own protocol,
exclusions and conclusions govern those new results. The historical findings
below retain their original scope and source revisions.

This inventory covers the standalone Stark comparisons, its converter, and the
earlier MeeCP Uzu experiments that led to Stark. Each result retains its original
scope. Historical Uzu candidates, upstream Uzu, standalone Stark and installed
MeeCP packages are distinct measured implementations.

- [Direct engine and reference comparisons](#1-direct-standalone-stark-and-installed-stark-comparisons)
- [Stark representation and kernel comparisons](#2-stark-representations-and-kernel-versions)
- [Earlier Uzu lineage](#3-earlier-uzu-lineage-in-meecp)
- [Converter comparisons](#4-converter-comparisons-against-lalamo-and-earlier-stark)
- [Candidate engines](#5-other-engines-to-consider)
- [Future benchmark design](#6-proposed-design-for-a-publishable-benchmark)

## Conclusions supported by the existing evidence

1. The strongest existing comparison using unchanged supported engine paths is
   **Stark versus TensorFold 1.0.12 on Qwen3.5-2B MLX 4-bit**. Stark had
   1.53–1.81x higher end-to-end generation rates on three short fixtures. This is
   a bounded single-host result; output hashes matched on only one fixture and
   comparative answer quality was not assessed.
2. Stark's JevK5 advantage over the **custom TensorFold BF16 port** was
   2.20–3.00x on fresh scoring requests. Material numerical drift remains in the
   port. This does not characterize upstream TensorFold's supported engines.
3. The direct **installed Stark versus llama.cpp** result is a small, uncontrolled
   timing snapshot. Earlier full-population Uzu comparisons provide useful
   context but cannot be combined arithmetically into a Stark speed claim.
4. **No direct standalone Stark-versus-unmodified-upstream-Uzu benchmark was
   located.** Earlier modified-Uzu-versus-historical-Uzu comparisons measured
   improvements of a few percent under their specific protocols. They do not
   establish the current Stark fork's total improvement over Uzu.
5. Converter results are much broader than inference results. Stark substantially
   reduced sampled conversion memory, but latency wins depend on the model,
   representation and converter revision. Failed latency objectives remain part
   of the evidence.
6. A future public benchmark needs a newly frozen protocol, current pinned builds,
   a larger public workload set, explicit precision/cache controls and separate
   quality checks. Existing measurements are useful pilots, not a universal
   engine ranking.

## Evidence scope and provenance

The Stark report snapshot is commit
`8ecfed737368cd61a1d4d1a9f85343a791ea0f65`. Original reports and their linked
aggregates contain the full checkpoint revisions, source manifests, binary
hashes, dependency locks and individual scalar observations. Relative Stark
links below resolve within this repository; this inventory does not rebind old
results to today's executable.

MeeCP reports were read from clean tracked files at
[`88d031c1c1b9997ec5d9dec4789596f52b47ff5c`](https://github.com/jshahbazi/meecp/tree/88d031c1c1b9997ec5d9dec4789596f52b47ff5c).
Their immutable links below preserve the integration and predecessor evidence
without copying host application code into Stark.

Most inference runs used an Apple M5 Max with 128 GiB unified memory and macOS
26.7.1. The older converter packets identify Darwin 25.6.0/arm64, 18 CPUs and
128 GiB; preserve those recorded identities rather than assuming every packet
had the same OS/build state. Timing boundaries differ across packets. Values
from separate experiments must not be pooled into one throughput table.

Evidence dispositions used here:

| Disposition | Meaning |
| --- | --- |
| Measured, bounded | Completed under the cited protocol; conclusion applies to its tested artifacts and workload. |
| Descriptive snapshot | Timing exists, but isolation, ordering or sample size prevents a strong performance claim. |
| Correctness only | Numerical, functional or artifact checks; no comparative speed conclusion. |
| Excluded timing | An attempt is preserved for audit but contributes no accepted performance claim. |
| Proposed | Research or a future experiment; no measured Stark comparison. |

Only aggregate results and hashes belong in this report. Prompts, generated
answers, token vectors, per-case decision distributions, credentials, model
weights and native runtime captures remain outside Git.

## 1. Direct standalone Stark and installed Stark comparisons

### 1.1 Stock TensorFold: Qwen3.5-2B MLX 4-bit

Sources: [report](TENSORFOLD_STOCK_QWEN35_COMPARISON.md),
[aggregate](experiments/tensorfold-stock-qwen35-2026-10-16/aggregate.json).
Measured 2026-10-16.

Shared checkpoint: `mlx-community/Qwen3.5-2B-MLX-4bit`, revision
`93760be4f1f69842a46bc13dbdc0f19e291392a3`, affine 4-bit/group-64. TensorFold
used its original upstream 1.0.12 release binary. Stark used the existing
default `stark-generate` and normal converter, retaining the packed weights.
Neither engine nor converter was modified for the comparison.

| Workload | Input tokens | TensorFold default, tok/s | TensorFold plain, tok/s | Stark default, tok/s | Stark/default-TensorFold rate ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fibonacci function and explanation | 26 | 193.1 | 194.0 | 349.6 | 1.81x |
| GPU explanation | 31 | 194.2 | 194.3 | 349.0 | 1.80x |
| Copy/edit a Python file | 390 | 188.3 | 148.4 | 287.6 | 1.53x |

Rates divide exactly 64 generated tokens by client end-to-end wall time,
including prefill, generation, cleanup and transport; model loading is excluded.
There were six timed rounds after warmup, with rotating mode order, greedy
decoding, context 2048, identical input IDs and no retained prompt cache. The
normal TensorFold draft mode improved copy/edit by about 1.27x over its plain
mode. TensorFold's modes matched each other on all three output hashes; Stark
matched TensorFold on one of three. HTTP and private pipe transport differ.

**Conclusion:** a credible small unchanged-path comparison, requiring more
workloads, output-quality checks and fresh current-version runs before a broader
public claim. These are not isolated decode rates or evidence about TensorFold's
other families. Measured Stark source: `b5693c72b16ed62e30489810560b292747fb8b41`.

### 1.2 TensorFold JevK5 compatibility and custom port

Sources: [report](TENSORFOLD_JEVK5_COMPARISON.md),
[aggregate](experiments/tensorfold-jevk5-2026-10-16/aggregate.json).
Measured 2026-10-16.

JevK5 is the BF16 Qwen3.5-4B checkpoint `alibiserikbay/JevK5`, revision
`c4f7fdb3aeab5582336406e78d3bef11bf98833d`. The original TensorFold 1.0.12
binary rejected its `qwen3_5_text` type; a metadata-only wrapper probe then
reached `UnsupportedQwenGeometry`. Stark loaded its converted artifact. This
initial compatibility result supplies no speed measurement.

The later TensorFold fork added a specialized BF16 route and scoring worker.
Stark used a locked Rust 1.98.1 release build with `optimized` enabled.

| Input tokens | Custom TensorFold, median ms | Stark, median ms | TensorFold/Stark latency ratio |
| --- | ---: | ---: | ---: |
| 32 | 72.8 | 24.3 | 3.00x |
| 128 | 78.2 | 32.1 | 2.44x |
| 512 | 197.2 | 85.5 | 2.31x |
| 1024 | 378.4 | 171.7 | 2.20x |

Seven alternating-order rounds after warmup used the same IDs, sixteen
candidates, context 2048, temperature 1.22 and no output cache. Timing includes
fresh-state prefill, full-vocabulary normalization and pipe exchange, excluding
model load. Candidate argmax matched on 8/8 synthetic decisions; maximum
calibrated probability difference was 0.009182. Seven full-vocabulary fixtures
were finite and matched the MLX reference's top token, but RMSE reached 0.378
and maximum absolute logit error 2.205. MLX chunking differences do not establish
that the port's error is harmless or fully explained.

**Conclusion:** Stark was faster than this experimental implementation. The
port's distribution drift prevents a drop-in numerical-equivalence claim; the
large JevK5 quality campaign was not rerun. No generation-speed result follows.
Measured TensorFold source: `0a64bacfa05388a41f3986112ed4b0e2322b9577`;
Stark source: `1e8ef8c32a926e59adb1b38632f3b1cc12ebfcff`.

### 1.3 Installed Stark versus llama.cpp: JevK5 snapshot

Source: [MeeCP Stark report](https://github.com/jshahbazi/meecp/blob/88d031c1c1b9997ec5d9dec4789596f52b47ff5c/docs/STARK_JEVK5.md),
recorded 2026-10-11. Same JevK5 BF16 source revision
`c4f7fdb3aeab5582336406e78d3bef11bf98833d`, context 8192, temperature 1.22;
twelve length-selected cases and three warmups, with fresh and immediate
identical requests.

| Path | Mean fresh latency, ms | Mean identical-repeat latency, ms |
| --- | ---: | ---: |
| Standalone Stark exact-token IPC | 174.2 | 0.71 |
| llama.cpp application scoring path | 347.8 | 26.9 |
| Installed Stark through MeeCP API | 297.8 | 7.3 |
| Installed Stark through MeeCP MCP | 270.5 | 8.4 |

The installed MCP path's recorded ratios were approximately 1.29x fresh and
3.20x on repeats versus llama.cpp. Standalone Stark excludes rendering and
tokenization; the installed paths include additional host work. Short ordered
blocks ran with uncontrolled background activity. Maximum cross-engine option
probability difference was 0.0116377, and all twelve typed result objects
differed bitwise; ground-truth quality was not evaluated in this snapshot.

**Conclusion:** descriptive application-path timings, insufficient for a precise
engine-only ranking. Historical package acceptance is separate from this
comparison; current host integration acceptance requires its own replay.

### 1.4 PyTorch/Transformers reference: ColGrep-Agent-2B

Sources: [generation evidence](GENERATION.md#validation-status-and-replay),
[independent review](COLGREP_GENERATION_REVIEW.md),
[strict reference report](../provenance/COLGREP_GENERATION_VALIDATION.json).
Reviewed 2026-10-13, implementation `6bd0b979`.

Checkpoint `lightonai/colgrep-agent-2B`, revision
`f798a61129c6decb5166864e0a3c98373e62c764`, was compared with a BF16
PyTorch/Transformers reference under controlled greedy generation.

| Check | Result |
| --- | --- |
| Exact tokenizer/decoding/EOS/tool-call checks | Four reference cases passed functional checks. |
| Exact greedy generation | 3/4 token sequences matched. |
| Divergent case | 116 shared prefix tokens, then 171 native versus 173 reference tokens; valid finish locations matched. |
| Strict reference disposition | `passed: false`; complete token parity was not established. |
| Diagnostic comparison | Two candidate scores at one divergence; original incremental logits/kernel cause remained unobserved. |

**Conclusion:** controlled functional acceptance and partial exact-output
agreement. No comparative throughput, stochastic equivalence or broad quality
ranking was established.

## 2. Stark representations and kernel versions

### 2.1 Qwen3-0.6B: dense BF16, MLX 8-bit and imported Q8_0

Source: [compressed-import report](QWEN3_GGUF_IMPORT_EXPERIMENT.md), dated
2026-10-13, implementation `2642d705f63e8080b434d7ec835749d5c71ad77c`.

All three representations below ran inside Stark. MLX 8-bit names a checkpoint
encoding; it is not a run of the MLX inference runtime. The official Q8_0 GGUF
was repacked with its integer codes preserved and F16 scales adapted to BF16.

| Representation | Weight MiB | Peak RSS MiB | Tracked Metal MiB | Completed decode tok/s | 512-token prefill tok/s | Load seconds |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Dense BF16 | 1137.0 | 1313.3 | 1307.0 | 340.6 | 32640–34119 | 0.246–0.249 |
| Existing MLX 8-bit | 586.5 | 774.1 | 744.5 | 502.1 | 33478–34747 | 0.193–0.199 |
| Imported GGUF Q8_0 | 604.2 | 776.5 | 774.2 | 493.0 | 27715–28964 | 0.195–0.234 |

Two fresh-process rounds, six deterministic cases and three measured repeats
after warmup gave 36 scoring and 36 generation calls per representation.
Context was 1024 and output count 32. Decode rates count the 31 tokens after
the first and include stream-drain time; they exclude pipe serialization. RSS
and Metal allocations overlap in unified memory and must not be summed.

| Numerical comparison | First-token agreement | Mean top-32 overlap | Exact 32-token generation |
| --- | ---: | ---: | ---: |
| Imported Q8_0 versus Stark dense BF16 | 6/6 | 97.40% | 5/6 |
| MLX 8-bit versus Stark dense BF16 | 6/6 | 97.92% | 5/6 |
| Imported Q8_0 versus llama.cpp original GGUF | 6/6 | 97.40% | 5/6 |

The llama.cpp source reference was build 11146, commit `7fe450e19`; its timing
was not used as a performance baseline. Mean top-32 log-probability RMSE versus
llama.cpp was 0.09475, maximum per-case RMSE 0.21187. Independent import checks
found zero mismatches among 595,984,384 integer codes and 18,624,512 expected
converted scale coefficients. Scale adaptation introduced 0.16955% relative RMS
weight error. These distinct checks must not be collapsed into bit-exact runtime
equivalence.

**Conclusion:** compressed representations substantially improved this model's
memory/decode behavior relative to BF16. Q8_0 decode was 44.7% higher than dense
BF16 and 1.8% below MLX 8-bit. This is a representation/quantizer comparison,
with six-case correctness evidence and uncontrolled thermal/filesystem state.

### 2.2 Q8_0 prefill optimization and rejected alternatives

Source: [prefill investigation](QWEN3_GGUF_PREFILL_OPTIMIZATION.md) and
[aggregate](experiments/qwen3-prefill-2026-10-13/aggregate.json), 2026-10-13.
Selected C: `b4bebb5021fe68d3a6ba86ca8f363eb53f063f38`, restored with identical
engine files by `68374278ec2dc2bd07d02a44e587fadc5cbdcb67`.

Primary six-round median prefill times:

| Input tokens | Original Q8_0, ms | Selected C, ms | Stark MLX 8-bit, ms | Original/C latency ratio |
| --- | ---: | ---: | ---: | ---: |
| 32 | 5.301 | 5.036 | 5.227 | 1.053x |
| 128 | 6.916 | 7.357 | 7.700 | 0.940x |
| 512 | 19.544 | 16.881 | 15.539 | 1.157x |
| 1024 | 33.360 | 29.050 | 28.983 | 1.149x |

Independent four-round comparison:

| Input tokens | Original Q8_0, ms | C, ms | Rejected E, ms | Stark MLX 8-bit, ms |
| --- | ---: | ---: | ---: | ---: |
| 512 | 19.406 | 16.855 | 27.736 | 15.694 |
| 1024 | 33.207 | 28.956 | 49.357 | 28.872 |

C preserved full-vocabulary F32 logits and fixed 32-token hashes on all tested
cases (maximum error zero). Wider staging E was numerically correct but slower;
reduced split-K alternatives failed emulated correctness and were rejected.

**Conclusion:** C helped the longer tested prefill sizes by about 15%, while
128-token latency regressed in the primary table. At 1024 tokens C approximately
matched Stark's MLX 8-bit representation. The original mismatch against llama.cpp
remained; no new general quality claim follows.

## 3. Earlier Uzu lineage in MeeCP

Upstream base: Uzu 0.6.1,
`8d01581474760bd1fc7f4a547ff1d02f7dada437`. The experiments below used
versioned native workers and patches; their measured candidate names must stay
attached to their results. They are predecessor evidence, not current standalone
Stark measurements.

### 3.1 Qwen3.5 text generation versus llama.cpp

Source: [Uzu experiment](https://github.com/jshahbazi/meecp/blob/88d031c1c1b9997ec5d9dec4789596f52b47ff5c/docs/UZU_EXPERIMENT.md).
Measured 2026-10-10 on **Qwen3.5-0.8B**. Six samples per engine/fixture across
two reversed-order loads; all 36 generations returned 256 output tokens,
context 4096, one active request and no prompt caching. Input token hashes matched.

| Input tokens | llama.cpp median wall s | Uzu median wall s | llama.cpp decode tok/s | Uzu decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 51 | 0.894 | 0.613 | 292 | 425 |
| 436 | 0.907 | 0.642 | 292 | 416 |
| 1204 | 0.974 | 0.678 | 287 | 410 |

Uzu peak native RSS was 1.047–1.048 GiB, llama.cpp 1.087–1.088 GiB; adapter
memory is excluded. Startup ranges were 0.734–1.449 s and 0.601–0.604 s.
Artifacts differed: Uzu `trymirai/Qwen3.5-0.8B-L` at
`506caad05799df01d53f41ec3468db383e9e84ea` versus llama.cpp
`unsloth/Qwen3.5-0.8B-GGUF` at
`6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`, Q8_0. These derivative pins are
recorded in the [model provenance](https://github.com/jshahbazi/meecp/blob/88d031c1c1b9997ec5d9dec4789596f52b47ff5c/experiments/uzu-rust/MODEL_PROVENANCE.json);
their original official source-weight revision equality is unverified. Native
timing boundaries also differ.

**Conclusion:** descriptive end-to-end time reduction of 29–31% and decode
rates 42–46% higher for the Uzu configuration. Different artifacts prevent
attributing the whole difference to engine implementation.

### 3.2 Blind generation quality screen

Source: [quality report](https://github.com/jshahbazi/meecp/blob/88d031c1c1b9997ec5d9dec4789596f52b47ff5c/docs/UZU_QUALITY_EVALUATION.md).
One hundred stratified AlpacaEval prompts, two hundred answers, matched greedy
generation and one blinded Astra judge, with a 256-token cap.

| Metric | Uzu | llama.cpp |
| --- | ---: | ---: |
| Pairwise wins | 21 | 26 |
| Ties | 53 shared ties | 53 shared ties |
| Mean score / 5 | 1.98 | 1.95 |
| Answers scoring at least 4 / 5 | 9/100 | 9/100 |
| Answers reaching output cap | 68/100 | 69/100 |

**Conclusion:** no meaningful relative-quality advantage established. One
judge/sample, weak absolute scores and output truncation prevent equivalence or
general assistant-readiness claims.

### 3.3 JevK5 scoring: original precision mismatch and BF16 controls

Sources: [original scoring](https://github.com/jshahbazi/meecp/blob/88d031c1c1b9997ec5d9dec4789596f52b47ff5c/docs/UZU_JEVK5_SCORING.md),
[same-checkpoint BF16 control](https://github.com/jshahbazi/meecp/blob/88d031c1c1b9997ec5d9dec4789596f52b47ff5c/docs/UZU_JEVK5_BF16_CONTROL.md),
[optimized BF16 benchmark](https://github.com/jshahbazi/meecp/blob/88d031c1c1b9997ec5d9dec4789596f52b47ff5c/docs/UZU_OPTIMIZED_BF16_BENCHMARK.md).

All use 231 public JevBench v1.2 cases plus twelve separate synthetic checks.
The public cases contain gold decisions, not gold distributions. Strict option
probability tolerance is 0.001; the descriptive close tolerance is 0.01.

| Comparison | Cross-engine public argmax agreement | Strict / close probability matches | Mean / maximum option gap | Gold argmax accuracy, both engines |
| --- | ---: | ---: | --- | ---: |
| Historical Uzu BF16 export versus llama.cpp Q8_0 | 229/231 | 92/231; 189/231 | 0.003045 / 0.066516 | 202/231 |
| Uzu versus llama.cpp, same BF16 checkpoint | 229/231 | 98/231; 199/231 | 0.002262 / 0.027507 | 202/231 |
| Optimized Uzu versus llama.cpp, BF16 | 229/231, native Uzu output preserved | 98/231; 199/231 | Same existing BF16 differences | 202/231 |

Original Brier values: llama.cpp Q8_0 0.183197, Uzu 0.185154. In the BF16
control, llama.cpp Brier was 0.183678 and Uzu 0.185154; lower is better. Equal
argmax accuracy did not establish probability/calibration parity.

| Timing packet | Fresh decision result | Immediate identical-repeat result | Timing disposition |
| --- | --- | --- | --- |
| Original precision-mismatched screen | Paired ratio 1.62x initially and 1.73x in a second run, favoring Uzu | Kept separate from fresh calls | Local descriptive screen; compiler/UI activity and unequal artifacts. |
| Same-checkpoint BF16, 1944 measurements | Paired median ratio 1.55x favoring Uzu | Paired ratio 3.27x favoring llama.cpp | Interrupted-session blocks; fresh p50 rose 37.0% for llama.cpp and 44.4% for Uzu between rounds. |
| Optimized BF16, 3888 responses + 24 warmups | Quartet ratios 1.519111x and 1.647857x favoring Uzu | About 7.55x lower mean latency for llama.cpp | Completed and audited, but background activity and no continuous GPU/thermal isolation limit precision. |

Here, a **second run** means a new benchmark acquisition; an **identical repeat**
means the immediately repeated request. llama.cpp could reuse the preceding
prompt; these Uzu workers recomputed it. llama.cpp generated one readout token;
Uzu performed direct scoring. All timing statements therefore describe the
specific decision paths and cache policies.

**Conclusion:** fresh Uzu decision paths were faster in these local controls.
Cached-repeat behavior strongly favored llama.cpp. The BF16 control reduced
numerical gaps without eliminating them. These results neither rank today's
Stark nor isolate every effect of its optimization lineage.

### 3.4 Modified Uzu versus prior Uzu

Sources: [first Rust optimization](https://github.com/jshahbazi/meecp/blob/88d031c1c1b9997ec5d9dec4789596f52b47ff5c/docs/UZU_RUST_SCORING_OPTIMIZATION.md),
[deeper optimization](https://github.com/jshahbazi/meecp/blob/88d031c1c1b9997ec5d9dec4789596f52b47ff5c/docs/UZU_RUST_DEEP_OPTIMIZATION.md).

| Candidate/control | Fresh latency reduction | Identical-repeat reduction | Evidence |
| --- | ---: | ---: | --- |
| First combined candidate / original worker | About 2.58% | About 1.89% | Balanced 12-case control; full-population timing direction was inconsistent. |
| Deep candidate / earlier optimized worker | 3.12% | 2.15% | Balanced 12-case timing; exact outputs retained. |
| Deep candidate / historical Uzu | 3.94% | 4.12% | Separate direct historical comparison; exact outputs retained. |

Deep validation covered 243 cases, 48 boundaries and 534 vocabulary digests.
Against historical Uzu, fresh reductions by length bucket were 8.46% short,
5.00% medium and 3.16% long. Full-population incremental speed improvement was
not established. Never multiply these ratios by a different Uzu/llama.cpp run.

The first optimization also records a representation caveat: 196/972 historical
probability/typed-decision comparisons differed by at most `2.22e-16` because
the older harness recalibrated in Python. Historical token IDs/logits/log
probabilities matched; historical probability-object bit equality was not
claimed. Preserve that distinction from the later sealed-reference exactness
checks.

**Conclusion:** scoped exact-output optimization evidence exists, with small
local timing improvements. It is not a direct current-Stark/upstream comparison.

### 3.5 Sol optimization, excluded timings and consecutive cache

Sources: [Sol optimization and attempt dispositions](https://github.com/jshahbazi/meecp/blob/88d031c1c1b9997ec5d9dec4789596f52b47ff5c/docs/UZU_SOL_OPTIMIZATION.md),
[consecutive-cache validation](https://github.com/jshahbazi/meecp/blob/88d031c1c1b9997ec5d9dec4789596f52b47ff5c/docs/UZU_SOL_CONSECUTIVE_CACHE.md).

| Packet | Result | Permitted conclusion |
| --- | --- | --- |
| Three interrupted Sol timing attempts | CPU guard stopped runs with background process peaks of 265.1%, 262.9% and 291.2%. | All partial timing excluded; owned workers reaped. |
| Complete Sol acquisition | 5832 responses and 36 warmups passed scoring/artifact checks, but Mac sleep overlapped the run window. | Exactness evidence retained; whole timing result excluded. |
| Stored ratios from excluded acquisition | llama.cpp/Sol fresh ratios 3.4465x and 1.8171x; prior Uzu/Sol 1.0431x and 0.9881x. | Audit history only; no accepted speed claim. |
| Default-off consecutive-cache worker | 243 cold/hit pairs, 48 boundary pairs and 534 full-vocabulary cold/hit pairs exact; hits avoided native scoring dispatch. | Functional cache reuse validated, precise cache speed unmeasured. |

The cache worker used 292 native calls and 293 hits for 585 replies including
warmups; its digest probe used 535 calls and 536 hits for 1071 replies. Gold
public accuracy stayed 202/231, synthetic accuracy 12/12, Brier 0.185154 and
ECE-10 0.030787. These are preservation results against the sealed Uzu reference.
They are not improvements in task quality. The separate retained llama.cpp
trace showed 972/972 repeats evaluating four native prompt tokens; it must not
be described as the Uzu full-logit cache implementation.

**Conclusion:** retain the exactness/cache findings and excluded attempts. There
is no accepted new Sol speed claim to promote into a Stark headline.

## 4. Converter comparisons against Lalamo and earlier Stark

These are complete model-preparation commands, not inference benchmarks.
Lalamo reference revision:
`831accb8252a6b6606c378048caeb44178f1a050` (MIT). It remains an attributed
external oracle; the present Stark converter does not invoke it as a fallback.

Packets below used serial fresh Python processes and warm selected source files,
with setup/downloads and parity checking outside timed commands. Stark commands
include stronger source verification, output hashing and receipt work than the
Lalamo reference. RSS sampling every 10 ms can miss peaks or double-count shared
pages. The 8 MiB transformation workspace does not cap whole-process memory.
Exact tensor/spec/config/tokenizer parity and named-build **load-only** checks
must not be relabeled inference-quality or classifier-execution acceptance.
Tables show medians; complete ranges, source identities and scalar observations
remain in the cited packets. Passing a raw median hypothesis is not a
statistical-significance claim.

### 4.1 Prototype and milestone 1: 2026-10-11

Sources: [prototype](MODEL_CONVERSION.md),
[milestone 1](MODEL_CONVERSION_MILESTONE_1.md),
[review](MODEL_CONVERSION_MILESTONE_1_REVIEW.md). Three repetitions per backend.

| Packet / checkpoint | Stark CLI median s | Lalamo median s | Stark / Lalamo sampled RSS MiB | Conclusion |
| --- | ---: | ---: | ---: | --- |
| Prototype Qwen3-0.6B | 0.976 | 1.783 | 206.2 / 1981.8 | 1.83x complete-command ratio; historical prototype with a different integrity/fallback implementation. |
| M1 Qwen3-0.6B | 1.895 | 1.389 | 187.4 / 1887.7 | Latency hypothesis failed. |
| M1 Qwen2.5-Coder-0.5B | 1.250 | 1.193 | 123.2 / 1555.1 | Latency hypothesis failed. |
| M1 Llama-3.2-1B, Unsloth extension | 2.997 | 2.006 | 263.9 / 3393.5 | Latency hypothesis failed. |
| M1 Mistral-7B-Instruct-v0.3 | 15.435 | 6.568 | 97.0 / 14423.7 | Latency hypothesis failed. |

M1's within-packet initial-to-final Stark medians fell from 2.437 to 1.895 s
(Qwen3), 1.671 to 1.250 s (Qwen2), 3.702 to 2.997 s (Llama), and 24.080 to
15.435 s (Mistral). Sequential phases were not randomized together. All parity
and memory gates passed; all four final M1 Lalamo latency hypotheses failed.
A later final-source isolated Qwen3 observation was median 1.980 s, with no live
reference timing and therefore no comparative ratio. A concurrent smoke timing
was excluded. The prototype's 1.83x cannot describe M1 or later revisions.

### 4.2 Milestone 2: dense models and sharded-source optimization

Sources: [report](MODEL_CONVERSION_MILESTONE_2.md),
[review](MODEL_CONVERSION_MILESTONE_2_REVIEW.md), 2026-10-11. Three repetitions;
all sixty timed artifact comparisons passed.

| Checkpoint | Stark median s | Lalamo median s | Lower sampled RSS | Latency hypothesis |
| --- | ---: | ---: | ---: | --- |
| Qwen3-0.6B | 1.835 | 1.234 | 90.06% | Failed |
| Mistral-7B-v0.3 | 9.790 | 6.305 | 99.43% | Failed |
| SmolLM2-1.7B | 3.777 | 1.622 | 98.10% | Failed |
| SmolLM3-3B | 6.032 | 2.628 | 96.08% | Failed |
| FunctionGemma-270M | 1.047 | 1.338 | 69.51% | Passed, 1.279x ratio |
| Gemma2-2B extension | 5.707 | 2.406 | 95.94% | Failed |
| Gemma3-1B | 2.576 | 1.672 | 85.27% | Failed |
| Granite3.3-2B | 6.132 | 2.079 | 98.21% | Failed |
| Phi-4-mini | 6.656 | 3.375 | 97.70% | Failed |

Controlled earlier-Stark medians were 1.851 to 1.835 s for Qwen3 (about 0.9%,
within variation) and 15.663 to 9.790 s for Mistral (37.5% lower). This supports
the measured sharded-source improvement, with eight Lalamo latency objectives
still failing.

### 4.3 Milestone 3: existing quantized checkpoints and writer versions

Sources: [initial Qwen3 MLX packet](MODEL_CONVERSION_MILESTONE_3.md),
[Llama/SmolLM/Gemma continuation](MODEL_CONVERSION_MILESTONE_3_DENSE.md),
[later writer packet](MODEL_CONVERSION_MILESTONE_3_LATENCY.md),
[writer review](MODEL_CONVERSION_MILESTONE_3_LATENCY_REVIEW.md).
Initial packets used three repetitions; the later writer used five per
backend/tuple and ninety total comparisons.

The initial Qwen3 group-128 BF16-output packet measured:

| Source checkpoint | Stark median s | Lalamo median s | Lower sampled RSS |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B MLX 4-bit | 1.011 | 1.796 | 93.0% |
| Qwen3-0.6B MLX 8-bit | 1.506 | 1.818 | 93.3% |

The initial group-64 continuation measured:

| Source checkpoint | Stark median s | Lalamo median s | Latency hypothesis |
| --- | ---: | ---: | --- |
| Llama3.2-1B 4-bit | 1.845 | 2.292 | Passed |
| Llama3.2-1B 8-bit | 2.233 | 2.137 | Failed; ranges overlap |
| SmolLM3-3B 4-bit | 3.979 | 2.809 | Failed |
| SmolLM3-3B 8-bit | 6.019 | 3.544 | Failed |
| Gemma3-1B 4-bit | 1.991 | 2.444 | Passed |
| Gemma3-1B 8-bit | 2.655 | 2.171 | Failed |

The later writer experiment compared its own concurrently frozen earlier Stark
baseline; its baseline values differ from the earlier packet and are retained
separately:

| Source checkpoint | Frozen prior Stark s | New Stark s | Lalamo s | New/prior time reduction | Lalamo/new rate ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama3.2-1B 4-bit | 2.027 | 1.256 | 2.397 | 38.0% | 1.91x |
| Llama3.2-1B 8-bit | 2.697 | 2.202 | 2.604 | 18.4% | 1.18x |
| SmolLM3-3B 4-bit | 4.302 | 2.779 | 3.377 | 35.4% | 1.22x |
| SmolLM3-3B 8-bit | 7.019 | 5.083 | 4.233 | 27.6% | 0.83x, failed |
| Gemma3-1B 4-bit | 2.111 | 1.551 | 2.580 | 26.5% | 1.66x |
| Gemma3-1B 8-bit | 2.919 | 2.464 | 2.537 | 15.6% | 1.03x |

**Conclusion:** all six new writer medians improved 15.61–38.04% over their
frozen Stark baseline, preserving output bytes; five beat Lalamo by median.
Llama and Gemma 8-bit ranges overlapped Lalamo, and SmolLM3 8-bit remained
slower. These runs do not measure MLX runtime inference.

### 4.4 Milestone 4: dense and quantized hybrid models

Sources: [dense report](MODEL_CONVERSION_MILESTONE_4.md),
[dense review](MODEL_CONVERSION_MILESTONE_4_REVIEW.md),
[MLX report](MODEL_CONVERSION_MILESTONE_4_MLX.md),
[MLX review](MODEL_CONVERSION_MILESTONE_4_MLX_REVIEW.md), 2026-10-12.
Five repetitions per backend/tuple; twenty dense and forty MLX parity records.

| Source checkpoint | Stark median s | Lalamo median s | Stark / Lalamo sampled RSS MiB | Conclusion |
| --- | ---: | ---: | ---: | --- |
| LFM2-350M dense | 1.233 | 1.688 | 114.83 / 1168.86 | 26.95% lower median time; broad overlapping ranges. |
| Qwen3.5-0.8B text dense | 3.525 | 2.707 | 201.52 / 2348.45 | 30.21% higher median time; failed. |
| LFM2-350M MLX 4-bit | 0.532 | 1.853 | 113.0 / 1411.4 | 71.3% lower time, 3.48x ratio. |
| LFM2-350M MLX 8-bit | 0.623 | 1.581 | 113.2 / 1527.2 | 60.6% lower time. |
| Qwen3.5-0.8B text MLX 4-bit | 0.997 | 2.059 | 203.5 / 4012.3 | 51.6% lower time. |
| Qwen3.5-0.8B text MLX 8-bit | 1.393 | 1.940 | 203.5 / 4155.3 | 28.2% lower time. |

**Conclusion:** all four MLX hypotheses passed, with 92.0–95.1% lower sampled
RSS. Dense Qwen3.5 latency remained open. Load-only qualification and converter
coverage do not establish hybrid-model inference speed or quality.

### 4.5 Milestones 5 and 6: LLaMBA and classifier fixtures

Sources: [LLaMBA report](MODEL_CONVERSION_MILESTONE_5.md),
[review](MODEL_CONVERSION_MILESTONE_5_REVIEW.md),
[ModernBERT report](MODEL_CONVERSION_MILESTONE_6.md),
[review](MODEL_CONVERSION_MILESTONE_6_REVIEW.md), 2026-10-12.
Five repetitions per backend/tuple.

| Source | Stark median s | Lalamo median s | Stark / Lalamo sampled RSS MiB | Conclusion |
| --- | ---: | ---: | ---: | --- |
| LLaMBA-1B MLX 4-bit, group 128 | 2.142 | 2.225 | 165.0 / 4606.5 | 3.73% lower time; overlapping ranges. |
| LLaMBA-1B dense F32 | 5.791 | 2.105 | 167.0 / 5817.6 | 175.15% higher time; latency failed. |
| ModernBERT BF16-source public composite | 0.562 | 1.137 | 102.0 / 675.6 | 50.59% lower time, 84.90% lower RSS. |
| ModernBERT F32-source public composite | 0.861 | 1.312 | 101.9 / 977.9 | 34.36% lower time, 89.58% lower RSS. |

ModernBERT fixtures use public real weights with explicitly synthetic codec/chat
metadata overlays. They are separate from the inaccessible Mirai router catalog
identities. Classifier execution/quality remains unverified. Dense LLaMBA uses
97.13% less sampled RSS but takes about 2.75x as long as Lalamo.

**Converter conclusion:** useful memory and specific latency advantages exist,
with exact export checks. A public inference article should present conversion
as a separate preparation-cost section, retaining slow rows and revision labels.

## 5. Other engines to consider

Research checked **2026-10-19** against official sources. Rankings below are
recommendations about experiment value, not measured performance rankings.
Support for a family does not qualify an exact fine-tune, storage layout,
configuration or scoring API. Pin a release and full commit before any run.

Release pages observed during research: [MLX-LM v0.32.2](https://github.com/ml-explore/mlx-lm/releases/tag/v0.32.2),
[Uzu 0.6.9](https://github.com/trymirai/uzu/releases/tag/0.6.9),
[TensorFold 1.1.1](https://github.com/ashhart/TensorFold/releases/tag/v1.1.1),
and [SGLang v0.5.22](https://github.com/sgl-project/sglang/releases/tag/v0.5.22).
These are research snapshots, not tested builds or future execution pins. In
particular, the TensorFold measurements above remain attached to 1.0.12.

### Recommended comparison set

| Priority | Engine / backend | Why include it | Admission work before timing | Official sources |
| --- | --- | --- | --- | --- |
| First | MLX-LM / MLX | Direct Apple-silicon reference, existing affine quantized checkpoint ecosystem and model-level Python access. Largest missing mainstream baseline. | Check exact source config, tokenizer, dtype, quantization group and cache behavior; force lazy work to complete before stopping timers. Arbitrary candidate scoring may require a small external driver. | [Project](https://github.com/ml-explore/mlx-lm), [benchmark guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/BENCHMARKS.md), [server implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py) |
| First | llama.cpp / Metal | Established GGUF implementation, native benchmarking and existing JevK5/source-reference history. | Rebuild a pinned current version; record GPU offload and quantizer; separate raw engine, HTTP and installed-app costs. Validate coefficient/precision differences. | [Project](https://github.com/ggml-org/llama.cpp), [native benchmark documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) |
| First | Unmodified upstream Uzu / Metal | Measures the fork's relationship to its progenitor directly; missing from the existing evidence. | Compare historical base and current upstream as separate controls if both are relevant; validate exact artifacts and available generation/scoring interfaces without silently porting Stark patches. | [Upstream project and benchmark CLI](https://github.com/trymirai/uzu) |
| First | Current stock TensorFold / Metal | Extends the existing unchanged-path pilot; provides a draft-enabled comparison. | Repeat the stock shared-model preflight. Keep 1.0.12 history separate from a new version; run defaults and a supported plain-mode control. | [Project](https://github.com/ashhart/TensorFold), [releases](https://github.com/ashhart/TensorFold/releases) |
| Second | vLLM Metal | Apple-silicon plugin with MLX compute, vLLM scheduling and its own request-aware attention path; relevant to serving and concurrency. | Verify model matrix, package compatibility, exact quantized model, cache settings and full scoring coverage. Shared MLX components mean this also tests scheduling/attention design. | [Project](https://github.com/vllm-project/vllm-metal), [model matrix](https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md) |
| Second | SGLang Apple-silicon backend | Official Apple/Metal/MLX routes make it relevant to serving rather than excluding it as a CUDA-only system. | Pin the Apple backend and runtime separately; confirm model admission and single-request versus batched semantics. Do not transfer CUDA benchmarks to this Mac. | [Project](https://github.com/sgl-project/sglang), [releases](https://github.com/sgl-project/sglang/releases) |

The recommendation is to begin with **Stark, MLX-LM, llama.cpp, upstream Uzu and
TensorFold**, admitting each only on the models it already supports. Add vLLM
Metal and SGLang for a second serving/concurrency phase. This order is a judgment
based on relevance and existing evidence; it predicts no winner.

### Broader screening list

| Candidate | Potential role | Reason to defer from the first matrix | Official source |
| --- | --- | --- | --- |
| mistral.rs / Metal | Another Rust inference stack and a useful independent implementation. | Exact model/quantizer coverage and scoring semantics need preflight; avoid treating all Rust/Metal implementations as interchangeable. | [Project](https://github.com/EricLBuehler/mistral.rs) |
| Candle / Metal | Lightweight Rust tensor/model implementation; useful correctness or implementation comparison. | Model-specific examples and a matching driver may need more integration than the first group. | [Project](https://github.com/huggingface/candle) |
| MLC LLM / Metal | Compiled inference path with an Apple deployment target. | Conversion/compilation and supported architectures form a separate preparation track; qualify exact checkpoint and precision first. | [Project](https://github.com/mlc-ai/mlc-llm), [documentation](https://llm.mlc.ai/docs/) |
| Core ML / stateful transformer export | Apple-native deployment/runtime comparison if the intended workload includes app deployment. | Exported operators, state, device placement and precision can differ; requires a separately qualified model-export experiment. | [Apple stateful-model guidance](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html) |
| PyTorch/Transformers on MPS | Correctness/reference path already used for ColGrep. | Device, dtype and fallback operators must be explicit; reference correctness and optimized-serving performance answer different questions. | [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html) |
| Ollama | Real installed product-path latency and usability. | Record the selected runner/backend and bundled version to avoid counting the same compute engine twice. | [Project](https://github.com/ollama/ollama) |
| LM Studio / llmster | Product-path comparison with selectable runtimes. | Identify its loaded MLX or llama.cpp runtime and settings; product overhead belongs in a separately labeled table. | [App and runtime documentation](https://lmstudio.ai/docs/app) |

No installed or performance-qualified claim is made for these candidates. A
former Nexa SDK name should not be added from old Apple-platform descriptions:
the official [repository URL](https://github.com/NexaAI/nexa-sdk) now redirects to
Qualcomm's GenieX project; a current Mac candidate would require fresh evidence.

## 6. Proposed design for a publishable benchmark

These steps have not been executed. Engine installations, new model runs and
article publication remain future work.

### 6.1 Questions and comparison tracks

| Track | Question | Controls |
| --- | --- | --- |
| Single-stream generation | How quickly does each supported stack prefill and generate? | Same model revision, equivalent precision, exact rendered input IDs, context/output counts, sampling and declared cache policy. |
| JevK5 decisions | What latency and quality does the whole scoring decision require? | Full candidate coverage, correct full-vocabulary normalization, candidate temperature 1.22, no hidden generation/truncation or missing low-probability candidates. |
| Serving | How do queues, batching and concurrency affect latency/throughput? | Separate concurrency 1, 4 and 8 where supported; same arrival schedule and success criteria; disclose serial-only engines and errors. |
| Caching/drafting | What do native optimizations contribute? | Fresh, immediate identical, shared-prefix and A/B/A populations separated; defaults and documented cache/draft controls compared explicitly. |
| Preparation | What does conversion/compilation/loading cost? | Cold and warm file-cache conditions distinguished; model download excluded and reported separately; exact output/precision contracts preserved. |
| Quality/numerics | Does a speed gain preserve useful behavior? | Teacher-forced common histories, finite logits, top-k overlap, probability error, greedy divergence and held-out task quality assessed separately. |

Use two model-preparation cohorts. A common BF16 or otherwise demonstrably
equivalent representation isolates implementation differences where admitted.
A native-recommended quantized cohort measures practical stack choices and
includes their separate quality/cost consequences. Equal bit width alone is not
equal quantization; imported Q8_0's BF16 scale rounding already demonstrates why.

### 6.2 Candidate model matrix, pending preflight

| Candidate model | Purpose | Existing evidence | Outstanding admission requirement |
| --- | --- | --- | --- |
| Qwen3.5-2B MLX 4-bit, revision `93760be4f1f69842a46bc13dbdc0f19e291392a3` | Initial unchanged-path generation cohort. | Stock TensorFold 1.0.12 and Stark already ran the same packed source. | Recheck current TensorFold, MLX-LM, Uzu, vLLM Metal and SGLang against this exact geometry/config; do not infer universal overlap. |
| Qwen3-0.6B, pinned dense and source Q8_0 variants from the import report | Small dense-model, tokenizer/numerical and preparation controls. | Stark dense/MLX/Q8_0 and llama.cpp correctness history. | Verify common current-engine admission; track every quantizer and scale adaptation separately. |
| Qwen3-4B-Instruct-2507 or another pre-existing shared dense instruct checkpoint | Less tiny generation workload and stronger architecture breadth. | Stark adapter/catalog evidence; MLX-LM publishes an official benchmark example for this family. | Exact immutable checkpoint, dtype and native execution remain to be selected/qualified for every engine. |
| JevK5 BF16, revision `c4f7fdb3aeab5582336406e78d3bef11bf98833d` | Application-specific decision latency and quality. | Stark and historical Uzu/llama.cpp controls; custom TensorFold port is experimental. | Preflight exact flat `qwen3_5_text` admission and arbitrary-candidate readout. Stock family support does not suffice. |
| ColGrep-Agent-2B, revision `f798a61129c6decb5166864e0a3c98373e62c764` | Tool-oriented generation/reference correctness. | Stark/Transformers functional evidence with strict token divergence. | Confirm shared MiniCPM5-compatible execution and investigate numerical behavior before promoting broad parity. |

Start with two admitted generation models and the separate JevK5 scoring cohort.
Do not select only models favorable to Stark or add new engine implementations
to manufacture overlap. Unsupported exact tuples should be reported as such;
custom ports can be a separately named engineering study if later requested.
TensorFold's headline Nemotron family is not an existing Stark overlap.

### 6.3 Measurement and analysis requirements

1. **Freeze inputs first.** Record full commits, releases, source/model/tokenizer
   revisions, binary and consumed-file hashes, build flags, dependency locks,
   toolchains, OS/firmware, hardware, power mode and context/precision settings.
   Register the matrix, metrics, exclusions and quality criteria before timing.
2. **Preflight correctness outside timing.** Verify exact input IDs, stops,
   output counts, no truncation, finite readouts and identical candidate sets.
   A server's top-N logprobs must not substitute for arbitrary-candidate scoring
   when required candidates are absent. Preserve failures as results.
3. **Use balanced independent blocks.** Load only one measured engine at a time,
   avoid background GPU work, rotate whole engine blocks, include independent
   process restarts and warmups, and use at least ten measured requests per
   cell as a pilot floor. Increase independent blocks according to observed
   variance; ten samples alone do not support a reliable p95 claim.
4. **Control the host.** Use an owned sleep assertion and a sleep-inclusive wall
   deadline; inspect the entire power-log window. Predeclare CPU/GPU/thermal
   interference exclusions and retain every excluded attempt and cause. UI
   activity, compilation and competing model work must be accounted for.
5. **Separate timing boundaries.** Report client end-to-end latency, time to first
   token, prefill, synchronized decode, inter-token latency and request teardown
   separately. Synchronize lazy/asynchronous work. Raw engine and HTTP/product
   routes get separate tables, each including its actual costs.
6. **Exercise useful sizes.** Use a fixed public workload mix across code,
   explanation, copying and longer histories; initial input sizes 32, 128, 512,
   2048 and 8192 where all admitted engines fit. Outputs of 128 and 512 tokens
   give a stronger decode sample than the historical 64-token pilot. Record
   early EOS rather than silently comparing unequal completed work.
7. **Report distributions and uncertainty.** Keep scalar observations, block
   medians/ranges and paired ratios. Bootstrap at the independent block/workload
   level rather than treating tokens or cached repeats as independent runs.
   Publish confidence intervals only with sufficient independent evidence;
   avoid one blended headline across different models/precisions/workloads.
8. **Measure memory honestly.** Track resident process-tree memory and native GPU
   allocations with clear boundaries; unified-memory values overlap. Report
   load, steady state and peak separately, without adding overlapping counters.
   Energy testing is optional and needs its own instrumented protocol.
9. **Keep quality independent.** Use public held-out tasks and blind assessment
   with fixed stopping criteria, preserve all cases and invalid outputs, and
   distinguish greedy identity from semantic quality/calibration. JevBench
   development-set agreement alone does not establish deployment quality.
10. **Make replay possible.** Publish harness code, dependency locks, public
    dataset identifiers/selection seeds, source and artifact hashes, aggregates,
    confidence calculations and exclusion ledgers. Keep private prompts,
    answers, vectors and per-case distributions outside Git; public fixture
    generators or external dataset IDs can define replay inputs without logging
    request payloads. Verify redistribution rights for datasets and models.

### 6.4 Recommended next sequence

1. Review this inventory and select the initial questions and model tuples.
2. Run a bounded stock-support preflight for MLX-LM, current llama.cpp, current
   upstream Uzu and current TensorFold; record support failures before deciding
   the final matrix.
3. Freeze a versioned protocol and independent analysis before collecting new
   timings. An external driver is acceptable; engine/model-support changes must
   be disclosed and kept outside the stock comparison cohort.
4. Complete the single-stream and quality pilots; resolve timer/cache/output
   count issues, then collect an independent final run under the frozen protocol.
5. Add vLLM Metal/SGLang serving tests only after their model/runtime admission
   succeeds and the serial baseline is stable.
6. Review the evidence, exclusions and reproducibility packet before drafting
   any public article. Publication remains a separate later action.

## 7. Coverage boundary

This inventory includes the located tracked comparison reports in Stark and
MeeCP, their cited aggregate packets, and relevant older session conclusions.
It excludes benchmark harnesses without measured reports, model-free unit-test
microbenchmarks, unrun proposals and unrelated model-versus-model studies.
Separate Brekko MLX-versus-llama.cpp JevK5 results did not contain a Stark arm.

No additional direct Stark-versus-MLX-runtime, Stark-versus-Ollama or
Stark-versus-LM-Studio performance report was found. That is a search result,
not proof that no unindexed private experiment exists. A newly found result
should be appended with its original source/build/protocol identities rather
than silently substituted into an older table.
