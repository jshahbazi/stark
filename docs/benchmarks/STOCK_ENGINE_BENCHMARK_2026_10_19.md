# Stock inference engine pilot: Qwen3.5-2B and JevK5

Measured 2026-10-19 Pacific time (2026-10-20 UTC), Apple M5 Max, 128 GiB,
macOS 26.7.1. This is an internal pilot for a future public benchmark. No article
was published and no engine, model implementation or converter was patched.

The Qwen results do not establish a Stark performance lead. MLX-LM has higher
median end-to-end rates than Stark on all six fixtures. vLLM Metal and Stark are
close, with their ordering depending on the fixture. Stark exceeds TensorFold
1.1.1 default drafting on these fixtures. The llama.cpp arm uses a different
quantizer and has a separately recorded measurement-driver correction.

JevK5 remains a useful separate scoring and compatibility workload. Its existing
BF16 native and GGUF exports ran in Stark and llama.cpp. MLX-LM, TensorFold and
vLLM Metal rejected its unchanged flat Hugging Face source during stock admission.
Both admitted engines achieve only 59/231 public-case
accuracy on this sealed replay. Agreement between them cannot establish useful
decision quality.

## Evidence and scope

- [Frozen protocol and recorded amendment](../archive/experiments/stock-engines-2026-10-19/PROTOCOL.md)
- [Pinned provenance](../archive/experiments/stock-engines-2026-10-19/PROVENANCE.json)
- [Original generation observations](../archive/experiments/stock-engines-2026-10-19/generation-results.json)
- [Corrected llama.cpp repeat](../archive/experiments/stock-engines-2026-10-19/llama-generation-corrected-results.json)
- [JevK5 scoring observations](../archive/experiments/stock-engines-2026-10-19/scoring-results.json)
- [vLLM serving observations](../archive/experiments/stock-engines-2026-10-19/serving-results.json)
- [Admission history](../archive/experiments/stock-engines-2026-10-19/ADMISSION.md)
- [Cleanup receipt](../archive/experiments/stock-engines-2026-10-19/cleanup.json)

The earlier [comparison inventory](INFERENCE_COMPARISON_EVIDENCE.md) preserves
historical Stark, converter and predecessor-Uzu results. This experiment does
not rebind those results to current binaries.

## Engine and model controls

| Engine | Pinned configuration | Qwen3.5-2B representation | JevK5 admission |
| --- | --- | --- | --- |
| Stark | Existing default release, Rust 1.98.1, locked dependencies | Existing native repack of MLX affine 4-bit/group-64 | BF16 native scoring admitted |
| MLX-LM | 0.32.2, locked Python environment | Unchanged MLX checkpoint | Unchanged flat `qwen3_5_text` source rejected |
| llama.cpp | b11747, `f711c8f3038020d71f69b8ced09162b014fd7335` | Existing Unsloth Q4_K_M GGUF | Existing BF16 GGUF admitted |
| TensorFold | Original upstream 1.1.1 release, default drafting and built-in plain request mode | Unchanged MLX checkpoint | Unchanged flat source rejected |
| vLLM Metal | 0.31.0 plugin and 0.31.0+cpu core, upstream-pinned dependencies | Unchanged MLX checkpoint | Unchanged flat source rejected |
| MetalRT | Exact-model stock support unverified | Not measured | Not measured |

The shared MLX model is
[`mlx-community/Qwen3.5-2B-MLX-4bit@93760be4`](https://huggingface.co/mlx-community/Qwen3.5-2B-MLX-4bit/tree/93760be4f1f69842a46bc13dbdc0f19e291392a3).
The preexisting GGUF's SHA-256 matches
[`unsloth/Qwen3.5-2B-GGUF@f6d5376b`](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/tree/f6d5376be1edb4d416d56da11e5397a961aca8ae).
Its quantizer differs, and equality of the original weights behind the two
quantized derivatives is unverified. llama.cpp therefore measures an engine and
format stack. It cannot isolate an engine-only difference.

Stark's standard repack, stock release installation/builds, normal CLI settings
and thin external measurement adapters are the only setup work. The custom
TensorFold JevK5 port from earlier experiments was not used. vLLM Metal is the
Apple-silicon plugin stack; this experiment says nothing about CUDA vLLM.

Stark and MLX-LM use private pipes; llama.cpp, TensorFold and vLLM use loopback
HTTP. Primary time includes prefill, generation, readout, request teardown and
transport, with model loading excluded. Different interfaces and defaults form
part of these measured stacks. No exhaustive per-engine tuning was attempted.
MLX-LM 0.32.2 used MLX 0.32.4; the stock vLLM Metal lock pins MLX 0.32.1 and
MLX-LM 0.32.0 at its specified source revision. Those upstream dependency
differences are part of the stack comparison.

## Qwen generation

The cross-engine comparison uses concurrency one for every engine: one request
in flight and one loaded model. The vLLM concurrency-eight number below is a
separate diagnostic and contributes nothing to this comparison.

All engines received the same integer input IDs rendered from the source chat
template with thinking disabled. Stark, MLX-LM and llama.cpp independently
reproduced the tokenization. TensorFold and vLLM received the IDs directly and
returned matching input counts; their tokenizer output was not independently
echoed. Context was 8192, greedy decoding, seed zero where exposed, and retained
or prefix caches disabled through stock controls. Missing cache counters remain
unknown.

TensorFold reported zero cached prompt tokens on all measured requests. Stark
and MLX-LM constructed fresh state through their native paths. llama.cpp reported
full input evaluation on every accepted request; its nonzero `tokens_cached`
field describes resident slot state rather than proof of reused prefill. vLLM's
cache-hit counters were absent and remain unknown.

Six public fixture templates cover code, explanation, editing, medium/long
context and longer output. Five request 128 output tokens; the last requests
512. Four independent loads, one warmup per fixture/load and three timed
repetitions give twelve observations per normal cell. A paired order and its
reverse are followed by a rotated pair.

| Fixture (input tokens; output limit) | Stark | MLX-LM | vLLM Metal | llama.cpp repeat | TensorFold default | TensorFold plain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fibonacci-module (42; 128) | 300.2 | 322.4 | 283.4 | 200.6 | 162.9 | 156.6 |
| gpu-explanation (40; 128) | 284.8 | 317.1 | 282.4 | 189.0 | 164.3 | 152.1 |
| copy-edit (383; 128) | 260.6 | 293.1 | 267.5 | 184.0 | 237.1 | 132.9 |
| context-medium (522; 128) | 251.5 | 275.3 | 258.6 | 176.6 | 138.4 | 125.9 |
| context-long (2118; 128) | 191.2 | 200.4 | 202.9 | 136.7 | 87.2 | 78.1 |
| long-output (50; 512) | 287.5 | 304.5 | 265.0 | 197.1 | 170.5 | 152.1 |

Rates are output tokens/second. Accepted loads/samples per fixture: Stark, MLX-LM, vLLM and TensorFold default 4/12; TensorFold plain and the corrected llama.cpp repeat 3/9. Their first blocks were excluded for CPU interference. Independent-load median ranges are retained in [analysis.json](../archive/experiments/stock-engines-2026-10-19/analysis.json). Both generation campaigns have whole-campaign timing acceptance false; this table describes their accepted subsets.

Values are medians of actual generated tokens divided by client complete-request
seconds, including prefill. They are not isolated decode rates. Accepted samples
must reach the explicit output limit with limit termination. Early EOS and the
four short sanity checks do not enter this table.

The original report's whole-campaign `timing_accepted` is false: TensorFold plain
block zero exceeded the preset background CPU limit, leaving three accepted
loads and nine observations per fixture for that control. The excluded block
remains in the evidence. Other original configurations have four accepted loads.
No continuous GPU, thermal or power isolation was measured, and an existing
installed JevK5 server was left untouched.

The initial llama.cpp adapter read `stopped_limit`, which is absent from the
pinned native API. It consequently mislabeled every limit completion as stop.
Those original samples remain ineligible. The corrected adapter reads and
preserves `stop_type`, rejects unknown values, and passed tests for limit, EOS
and stop-word termination. A separate four-load repeat is reported above. It
does not recover the original paired order and is not silently merged into the
original report. The API schema is documented in
[the pinned server implementation](https://github.com/ggml-org/llama.cpp/blob/f711c8f3038020d71f69b8ced09162b014fd7335/tools/server/server-task.cpp).

TensorFold's default/plain comparison requests its documented `draft=true/false`
control. Native draft acceptance counters were not collected, so the result
does not explain its internal drafting behavior.

| Stack | Arithmetic | Prime | Sorting | Python |
| --- | ---: | ---: | ---: | ---: |
| stark | 0/4 | 0/4 | 4/4 | 0/4 |
| mlx_lm | 0/4 | 0/4 | 4/4 | 0/4 |
| vllm_metal | 0/4 | 0/4 | 4/4 | 0/4 |
| llama_cpp | 3/3 | 3/3 | 3/3 | 0/3 |
| tensorfold_default | 0/4 | 0/4 | 4/4 | 0/4 |
| tensorfold_plain | 0/3 | 0/3 | 3/3 | 0/3 |

Each accepted fixture has one decoded-text hash within its engine across repetitions. The common token hashing algorithm was independently visible for Stark, MLX-LM, llama.cpp and vLLM; TensorFold token hashes remain native-reported.

The four exact-answer sanity checks are too small to establish comparative
generation quality. Stable hashes establish repeated output identity under this
protocol; different hashes do not establish which answer is better. TensorFold
returns an opaque native token hash whose serialization was not independently
verified. Other engines use the shared driver's token hashing algorithm. Decoded
text hashes are computed consistently by the driver for every engine.

## JevK5 scoring

The model source is
[`alibiserikbay/JevK5@c4f7fdb3`](https://huggingface.co/alibiserikbay/JevK5/tree/c4f7fdb3aeab5582336406e78d3bef11bf98833d),
with the preexisting BF16 native and GGUF representations. All 243 sealed cases
were replayed without regenerating prompts, labels or tokenization: 231 public
JevBench v1.2 cases and twelve synthetic controls. Dataset/configuration hashes,
reference block hashes and the settled ownership seal were checked first.

Each score uses context 8192, fresh native state, raw full-vocabulary log
probabilities and candidate softmax temperature 1.22. There is no generated
readout token or output cache. Four loads per engine in AB/BA/AB/BA order yield
1944 measured requests, excluding three warmups per load. Timing includes native
context construction, prefill, normalization, cleanup and the private pipe.

| Scoring metric | Stark BF16 | llama.cpp BF16 |
| --- | ---: | ---: |
| Accepted loads | 2 | 4 |
| Accepted measured requests | 486 | 972 |
| Pooled median complete-request latency, ms | 68.63 | 133.13 |
| 0-255 input tokens (133 cases/load), median ms | 55.94 | 120.99 |
| 256-1023 input tokens (66 cases/load), median ms | 118.47 | 187.65 |
| 1024-8191 input tokens (44 cases/load), median ms | 598.26 | 893.67 |
| Public accuracy | 59/231 | 59/231 |
| Synthetic accuracy | 12/12 | 12/12 |
| Public Brier | 1.266523 | 1.266910 |
| Public ECE-10 | 0.617339 | 0.626079 |

Both engines completed all 243 cases in each of four loads. Stark timing blocks zero and one exceeded the preset single-background-process CPU limit (335.85% and 228.16%); they remain excluded. The accepted Stark block medians were 65.90 and 74.87 ms. llama.cpp block medians ranged from 130.27 to 135.99 ms. The unequal accepted load counts, host activity and whole-campaign timing acceptance false limit these to descriptive latency observations.

Cross-engine candidate argmax agreement was 241/243 (99.18%). Maximum/mean candidate probability difference was 0.026980/0.002134; maximum candidate raw-logit and full-vocabulary-normalized log-probability differences were 0.255001 and 0.234514. Maximum within-engine probability difference across all four loads was zero for both engines. These findings establish repeatability, with two cross-engine choice disagreements; they do not establish numerical identity.

Public-case accuracy is 25.5% for both engines. The synthetic controls are a
different population and cannot offset that result. Brier is the unnormalized
multiclass sum of squared probability error; ECE uses ten equal-width confidence
bins. The accounting uses the sealed label order and gold label directly.
These are development-set replay findings. Neither agreement nor repeatability
qualifies the model's decision quality or calibrated thresholds.

## vLLM serving extension

The throughput at concurrency eight sums work from eight simultaneous requests.
Comparing that number with any engine's single-request rate would be unfair.
This section has no cross-engine serving result. A serving comparison requires
the same offered workload and concurrency on every stack, with aggregate
throughput and per-request latency reported together.

| Concurrent requests | Accepted loads / waves / requests | Median aggregate output tok/s | Median request latency, ms |
| ---: | ---: | ---: | ---: |
| 1 | 3 / 15 / 15 | 264.4 | 484.0 |
| 4 | 3 / 15 / 60 | 738.1 | 693.3 |
| 8 | 3 / 15 / 120 | 953.4 | 1073.3 |

All three loads passed the preset CPU and whole-window sleep screens; this campaign's timing acceptance is true. All 195 timed requests and 39 warmup requests met fixed-work checks. All timed requests had one common output-token hash. Cache counters were missing for all 195 timed requests, so cache-hit coverage remains unknown despite the disabled prefix-cache setting. Continuous GPU/thermal isolation remains unverified.

This is a within-vLLM scaling check on the Fibonacci fixture. Concurrency 1/4/8
shares one stock server configuration: eight sequence slots, context 8192,
GPU memory utilization 0.1 and prefix caching off. Three independent loads rotate
the concurrency order; each has one warmup wave and five measured waves per
concurrency. All completed requests must generate exactly 128 tokens with length
termination and return 128 token IDs before their wave enters throughput.

Wave throughput measures total completed output tokens divided by dispatch-to-
completion wall time. Request latency is reported separately. This does not
compare the other engines' schedulers, establish an optimized serving frontier,
or support a tail-latency claim. Missing cache-hit counters remain unknown.

## What this changes for a future write-up

Use the shared Qwen checkpoint as the broad compatibility and generation arm.
Stark must be compared against current MLX-LM and vLLM Metal: the observed serial
results provide no basis for presenting Stark as the fastest engine. Include
JevK5 as a separate scoring and source-compatibility arm, with the measured low
public accuracy made explicit. Unsupported stock source configurations should
remain support outcomes.

Before publication, improve host isolation, use a larger public generation
quality set, freeze a fresh paired campaign, and assess numerical fidelity and
quality for the differently quantized GGUF. Repeat the same serving workload
on competing stacks if serving is part of the article. Resolve exact-model
MetalRT eligibility before including it. Preserve this pilot's failures and
protocol revisions; it is not a publication-ready universal ranking.

## Reproduction, validation and cleanup

Path-redacted configuration templates and dependency locks are retained beside
the aggregate evidence. Replace their path roles with local disposable runtimes
and the pinned model artifacts, then use the recorded drivers. The private sealed
JevK5 reference is required for that exact replay; aggregate hashes alone cannot
reconstruct its inputs. No private evaluation payloads are included here.

Validation passed: the locked default Rust release build; eleven model-free
benchmark adapter tests; 28 repository CI-unit tests; the C++ adapter's production
normalization self-test; Ruff lint/format; both dependency-lock checks; repository
policy checks; and the check for engine/converter source changes. Independent
reviews checked the numerical tables, exclusions, quality accounting, privacy
and retained provenance. No engine or converter diff was introduced.

The aggregate packet was committed as
`4adfae080a2d32152bfb3f362b26e6ab49da02ce` before cleanup.
The owned runtime was then removed after verifying its ownership marker, exact
directory type, zero open handles and no benchmark executable processes.
Its regular files totaled 7,196,723,512 logical bytes;
`du` reported 7,532,720,128 allocated bytes
(7.02 GiB).
The immediate filesystem free-space increase was
5,562,748,928 bytes
(5.18 GiB). APFS allocation and
other activity on the shared filesystem prevent treating that change as an
exact exclusive reclaim measurement.

The runtime root and temporary RPC directories are absent. The new benchmark
drivers' bytecode was removed. Existing Qwen/JevK5 model paths, preexisting
runtimes and the installed model server were preserved. Drivers, dependency
locks, configuration templates and aggregate evidence remain in Git.
