# Qwen3.8 GDN projection and recurrence follow-up experiments

Date: 2026-10-31. This follows the [recurrence/norm fusion study](../qwen38-gdn-fusion-2026-10-31/README.md). The three prototypes remain isolated research routes. No production dispatcher in this branch selects them. Earlier evidence is preserved unchanged.

Status: all three prototypes passed strict component controls and seeded full-model checks. Explicit Q8 verification weight reuse showed a small positive model difference in two windows and is the promising candidate. Ordinary projection/convolution and the 1,024-thread recurrence layout showed no reliable model gain. The observed MTP differences remain exploratory; this record establishes no sustained rate, precise causal gain or production qualification.

## Complete-model outcome

| Ordinary decode arm | Measured requests | Pooled useful decode tok/s | Median tok/s | Observed change |
| --- | ---: | ---: | ---: | ---: |
| Unchanged baseline | 6 | 54.5486 | 55.4344 | — |
| Q8 projection/convolution | 6 | 54.0272 | 54.0133 | −0.96% |
| Recurrence/norm, 1,024 threads | 6 | 55.6362 | 55.5456 | +1.99% |

All 24 ordinary calls, including six excluded warmups, matched the historical completion hash. The baseline's early and late blocks measured 51.4306 and 58.0691 tok/s (+12.91%), exceeding both candidate differences. Projection/convolution's own block rate rose 6.22%; recurrence/norm's rose 2.64%. These observations establish neither a reliable gain nor a precise causal regression. The 2.38% projection/convolution component improvement did not produce an observed pooled model improvement. The synthetic recurrence component regression and positive model difference also illustrate the limits of these timing windows.

| MTP enabled, two drafts, first window | Measured requests | Pooled useful decode tok/s | Median tok/s | Observed change |
| --- | ---: | ---: | ---: | ---: |
| Unchanged MTP baseline | 6 | 67.6394 | 67.9315 | — |
| Explicit Q8 weight reuse | 6 | 70.0920 | 69.9153 | +3.63% |

All 16 first-window MTP calls, including four excluded warmups, matched the MTP historical completion hash. Each reported 106 rounds, 209 proposed drafts, 149 accepted drafts and 315 verified rows. Baseline blocks measured 66.7283 and 68.5757 tok/s (+2.77%); reuse blocks measured 69.6975 and 70.4911 (+1.14%). Measured verifier time totals fell 4.00%, a descriptive observation. The result motivated one confirming window with identical workers, sources and protocol. These MTP rates are compared only within their own window, with two drafts enabled on both sides.

| MTP enabled, two drafts, confirming window | Measured requests | Pooled useful decode tok/s | Median tok/s | Observed change |
| --- | ---: | ---: | ---: | ---: |
| Unchanged MTP baseline | 6 | 74.6727 | 74.4842 | — |
| Explicit Q8 weight reuse | 6 | 76.2361 | 75.7290 | +2.09% |

The [confirming record](mtp-repeat-results.json) preserves all 16 additional calls and its separate source-bound evidence. All outputs and per-request MTP counters again matched, and every block seal passed. Baseline blocks measured 75.5590 and 73.8069 tok/s (−2.32%); reuse blocks measured 76.6725 and 75.8046 (−1.13%). Verifier time totals fell 2.30%. The monitor made 122 polls, recorded no CPU compiler observations and 27 Python observations, with no blocking or inspected GPU/model-mapping observation. No new build, source change or workload change occurred between windows.

Both MTP windows show a positive pooled difference, approximately 2–4%, with matching exactness and MTP activity. Their baseline drift is of comparable magnitude, request ranges overlap, and both use the same mirrored order. Absolute baseline throughput also changes materially between windows. This is a promising directional result, with no confidence interval or independent randomized-order replication. We retain the candidate for further qualification and keep the windows separate rather than pool them or adopt a production route. All **56 model calls** across this study matched their respective historical completion hashes: 24 ordinary and 32 MTP, including 14 excluded warmups.

The [initial aggregate record](results.json) preserves every request metric/hash, component sample and source identity. All ten initial model blocks passed their before/after seals. The ordinary monitor recorded 217 polls, 20 CPU compiler process observations and 42 Python observations; the first MTP monitor recorded 138 polls, no CPU compiler observations and 21 Python observations. Neither window had a blocking observation or an inspected Python record with GPU/model mapping markers. Some processes vanished before inspection, and process sampling cannot establish GPU exclusivity. The workers expose no precision metadata; precision and KV policy are qualified through the sealed sources rather than a runtime precision acknowledgment.

## Experiments

| Candidate | Hypothesis | Fixed reference |
| --- | --- | --- |
| Q8 projection/convolution (`q8conv`) | Execute the four convolution taps and SiLU in the projection's output lane; avoid the raw QKV write/read and one dispatch | Ordinary decode at `645ad5562f44926341b3a74001ed4243e40db3e9` |
| Recurrence/norm fusion (`fused1024`) | Use 1,024 threads per head to restore the original 16-state-scalar tile per thread while eliminating the separate norm/gate dispatch | Same ordinary reference |
| Q8 verification weight reuse (`q8reuse`) | Decode each weight code and affine pair once and reuse it across verification input rows | MTP enabled, two drafts on both arms, at `959ec981354acab313a9a8d709933fb2f2649b17` |

The projection/convolution kernel preserves the stock Q8 accumulation and SIMD reduction, its BF16 output boundary, and the original BF16 convolution and custom SiLU. It writes the z/a/b suffix unchanged. Each QKV channel updates its history once. The deployed kernel leaves the unused raw QKV prefix unwritten; a separate diagnostic build captures it for exactness checks.

The 1,024-thread kernel retains the preceding study's per-row recurrence and normalization arithmetic, BF16 boundaries, F32 recurrent state and precise inverse square root. One threadgroup owns each of 48 heads, with 32 SIMD groups and four value rows per group. The actual compiled pipeline supports 1,024 threads. Register allocation, spills and occupancy were not measured.

The reuse kernel hoists the eight codes and BF16 affine pair outside the input-row loop. Each row preserves the stock dot-product, affine expression, block accumulation, SIMD reduction and BF16 output. The source establishes explicit reuse; physical memory traffic and cache behavior were not measured. Integration changes only the qualified multi-row Q8 verification projection. Single-row dispatch stays stock.

## GPU correctness and component timing

All eight initial strict controls passed **912 whole-array comparisons**: 432 for recurrence/norm, 288 for projection/convolution and 192 for verification row counts one through four. Controls preserve poisoning, canaries, finiteness and repeated state/history steps. The two Q8 harnesses also verify unchanged inputs and weights; the recurrence harness checks guards and finiteness around its z/norm inputs without fingerprinting their contents. They compare against the unchanged stock paths with exact array equality. No tolerance or golden was relaxed. These synthetic fixtures and one seeded model workload do not establish arbitrary-input numerical or quality equivalence.

| Component | Stock GPU µs | Candidate GPU µs | Candidate time change |
| --- | ---: | ---: | ---: |
| Projection + convolution | 101.2890 | 98.8756 | −2.38% |
| Convolution + recurrence + norm/gate | 58.9657 | 63.0959 | +7.00% |

The reuse component has six shape/row-count comparisons. For the joined projection, candidate time changes by +7.37%, −4.10% and −20.78% for two, three and four rows. For the output projection, changes are +14.01%, −2.94% and −18.33%. The model trial uses two drafts, so its verification batches have at most three rows; the four-row result does not represent that workload.

Component timing uses actual command-buffer GPU time, paired controls and distinct weight/layer instances. Projection/convolution retains six measured samples per arm over 36 weight sets; recurrence retains eight over 36 layer instances. Both repeat ABBA ordering. Reuse rotates ABBA/BAAB ordering and retains six samples per geometry over nine weight sets and 36 serialized projections per pass. Each excludes its initial warmup cycle. Every timed arm has a matching aggregate result digest. Components omit the rest of the model and cannot establish a model throughput gain.

The first projection/convolution component attempt failed its capture control before timing. Its harness reused one `MetalContext` for deployed and capture shader sources with the same entrypoint. The existing pipeline cache key omits source text, so the capture control selected the cached deployed pipeline. Version two uses a fresh context for that control. Both strict controls were rerun and passed, and the shader, reference and goldens stayed unchanged. The failed source, binary and aggregate receipt remain retained separately; no numeric mismatch count was captured in its sanitized output.

## Model protocol and evidence

`prepare_trials.py` archives the frozen commits and patches only ignored, gitless source copies. `build_trial.py` builds each arm with locked Rust 1.98.1 into its own `CARGO_TARGET_DIR`. `benchmark_single.py` and `benchmark_mtp.py` serialize all worker loads and requests, validate source/build/worker identities, and recheck source, checkpoint and tooling seals around every block. They neither build nor download models.

Both model windows use the official mixed 4/8-bit checkpoint, revision `788c07a8725642bccf13a0a3b8a21954ab84b4f2`, its 113-file manifest and prior payload-hash receipt. Payload identity is bound through that receipt and unchanged file metadata; the 107 GB payload is not reread merely to hash it during timing. Tokenizer/config hashes are checked directly. Stored quantization, BF16 activation boundaries, F32 state and Q8-affine-64 KV policy stay fixed.

The pinned workload is 4,096 input tokens and 256 output tokens, context 4,352, temperature 1, top-k 20, top-p 0.95 and seed 42, without prefix reuse. Useful decode throughput is **255 / native decode seconds**, excluding prefill and transport. Each new worker receives one excluded warmup and three measured calls. Ordinary order is baseline/q8conv/fused1024/fused1024/q8conv/baseline (24 calls); MTP order is baseline/q8reuse/q8reuse/baseline (16 calls). There are six measured calls per arm. MTP requires two drafts and the exact rejection/batched verification acknowledgments on both sides; its throughput is compared only with its MTP-enabled baseline.

The ordinary historical completion hash is `cf67e22bcf0d71a09bd998471f860b6da8fd4383a75c82b0bc5a3a00a2b7dafb`; the MTP historical hash is `f83299da4a915ce999d38ab157da2225417349edb92e994178dffe92c98f7418`. The two windows are separate experiments and their rates must not be pooled.

The process monitor excludes named competing model/benchmark workloads and large Python processes with model/GPU mappings, records CPU compilation and smaller ambiguous library mappings, and stops only its own worker on a fatal observation. Process metadata cannot prove exclusive GPU use. No own builds run during timed windows. No per-process GPU utilization, clock, power or thermal telemetry is captured. Mirrored blocks expose temporal drift; six measured requests per arm do not establish statistical confidence or a sustained production rate.

The host is an Apple M5 Max (`Mac17,7`), 128 GiB unified memory, macOS 26.7.1. Research runtime records, source snapshots, binaries and model data remain ignored. Publication contains aggregate metrics, order and hashes only. The shaders derive from Stark's vendored Lily implementation at the frozen revisions and the preceding local fusion prototype; inherited source notices remain intact. No external kernel was imported.

Repository policy, Ruff and the three harnesses' pinned nightly Rust formatting checks pass. Independent read-only audits recomputed request counts, arithmetic, component samples, output hashes and source/build/checkpoint seals, and checked public exports against private receipts. The strict GPU controls and isolated model workers provide this experiment's numerical validation; the production dispatch is unchanged.
