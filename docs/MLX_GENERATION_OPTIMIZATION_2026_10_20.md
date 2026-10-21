# MLX-informed Stark generation optimization

Date: 2026-10-20. Apple M5 Max, 40 GPU cores, 128 GiB, macOS 26.7.1.

The modified default Stark build showed a **2.9% geometric-mean throughput gain** in the additional confirmation's three accepted paired loads. The six fixture gains ranged from 1.8% to 4.2%. Original and modified Stark retained identical output-token and decoded-text hashes on all ten fixtures, and the reference and modified scorers matched bit digests covering all 248,320 F32 vocabulary logits on 23 inputs.

These are descriptive results from accepted subsets. Every campaign's whole-window `timing_accepted` value is **false** because one or more loads exceeded the existing background-activity guard. The three-engine comparison retained only one load in which original Stark, modified Stark and current MLX-LM all passed that guard. It does not establish a reliable engine ranking or a publication-ready performance claim.

## Implementation and source findings

Upstream [MLX-LM](https://github.com/ml-explore/mlx-lm) was cloned at `72be8c955d9a0d3099bd01865af288e0e947c3fa`. Its locked runtime is MLX-LM `0.32.3.dev6+g72be8c955`, MLX/MLX-Metal 0.32.4 and Python 3.13.13. The earlier stock benchmark used MLX-LM 0.32.2. The Qwen3.5 model, DeltaNet, attention, cache and loader sources are unchanged between those revisions; tokenizer/text-state handling changed. The serial generation API remains structurally compatible. [Source comparison](experiments/mlx-optimization-2026-10-20/mlx-source-comparison.json), [package provenance](experiments/mlx-optimization-2026-10-20/mlx-current-provenance.json).

The investigation found two avoidable costs in Stark:

1. **A fresh scratch allocator on every decode step.** The global Metal allocator already caches physical buffers, so this was repeated allocation-pool, reference-count, locking and cache work. Commit `00f25344` alternates two decode pools. A pool becomes reusable only after every associated submission completes successfully; failures drain all submissions and withhold the pool. Live scratch handles continue owning their pages. Prefill, grammar and speculative paths retain their existing allocation lifecycle. No computed model result is cached or skipped.
2. **Small GEMV output tiles on two fused projections.** For Qwen3.5-2B's `N=8224, K=2048` DeltaNet projection and `N=12288, K=2048` MLP projection, the previous policy assigned two output rows per threadgroup. MLX's affine single-vector path uses eight. Commit `045d0ca6` changes Stark's independent output-row assignment to eight while retaining the K reduction lanes, quantization groups and arithmetic. Admission is restricted to Apple10 GPUs with at least 30 cores, BF16 scale/bias metadata and input/output, unsigned affine W4/G64, GroupOutput metadata, `M=1` and those exact dimensions. [MLX 0.32.4 dispatch](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp#L435-L507), [Stark policy](../crates/stark-engine/src/backends/metal/kernel/matmul/gemv/policy/quantized.rs).

The CPU stack sample spent 88.8% of main-thread observations in the completion-wait path and 9.5% in decoder encoding. Waiting includes GPU execution and callback scheduling; it is not a GPU-utilization measurement. Both engines already prepare the next token before returning the current one. Stark already fuses several projections and normalization/gating operations. The evidence does not attribute the complete historical MLX lead to one cause. [Profile summary](experiments/mlx-optimization-2026-10-20/cpu-profile.json).

## Confirmation of the final default build

The unchanged stock comparison driver used the existing Qwen3.5-2B affine W4/G64 source and its previously converted Stark representation. Settings were greedy decoding, seed 0, context 8192, one model and one request at a time, fresh request state, four independent loads per engine, one warmup per performance fixture/load and three measured repetitions. Client time includes prefill, decode, readout, transport and cleanup; model loading is excluded. These rates are not isolated decode rates.

In the additional Stark-only confirmation, optimized load 0 exceeded the guard. Loads **1, 2 and 3** passed for both builds, giving nine measured requests per cell below.

| Fixture | Input / output tokens | Original median tok/s | Modified median tok/s | Paired gain |
|---|---:|---:|---:|---:|
| Fibonacci module | 42 / 128 | 336.2 | 349.0 | 4.15% |
| GPU explanation | 40 / 128 | 317.8 | 326.2 | 4.05% |
| Copy/edit | 383 / 128 | 292.0 | 295.0 | 1.80% |
| Medium context | 522 / 128 | 279.6 | 285.2 | 2.32% |
| Long context | 2118 / 128 | 211.5 | 215.5 | 1.86% |
| Long output | 50 / 512 | 320.1 | 331.1 | 3.20% |

The two rate columns pool the nine accepted requests. The gain column is the geometric mean of the three within-load median-rate ratios, so it need not equal division of the pooled columns. Averaging the six fixtures' log ratios gives **2.89%**. All accepted performance samples reached their exact output limit with length termination. No quality claim follows from throughput or hash agreement. [Confirmation receipt](experiments/mlx-optimization-2026-10-20/stark-confirmation-results.json).

The preceding three-engine campaign's accepted original/modified Stark pairs were loads 0 and 2. They showed **1.28%** overall gain, with individual fixtures ranging from regressions to larger gains; its long-output gain was 10.84%. The confirmation used the same binaries and measurement protocol and was added after observing that variation. It is a follow-up, not a preregistered holdout. Do not combine selected faster cells across the two campaigns.

## Current MLX-LM comparison and its limits

All four MLX-LM loads completed generation through the unmodified current upstream runtime. Loads 1, 2 and 3 were excluded for background CPU activity. Original Stark passed all four loads; modified Stark passed 0 and 2. Only **load 0** passed for all three engines, with three requests per cell:

| Fixture | Original Stark tok/s | Modified Stark tok/s | MLX-LM tok/s |
|---|---:|---:|---:|
| Fibonacci module | 345.8 | 324.3 | 314.3 |
| GPU explanation | 341.9 | 311.3 | 314.9 |
| Copy/edit | 318.9 | 279.9 | 299.7 |
| Medium context | 281.6 | 268.0 | 285.2 |
| Long context | 198.8 | 207.4 | 217.2 |
| Long output | 303.3 | 319.9 | 270.5 |

The mixed ordering and variation prevent a broad speed conclusion. The earlier stock pilot's 5–12% MLX-LM lead remains its own source-bound result; this investigation does not replace it with a claim that Stark has overtaken MLX-LM. [Corrected three-engine receipt](experiments/mlx-optimization-2026-10-20/final-corrected-results.json).

## Exploration and rejected changes

Exploration preceded the final comparison. Results below use only paired accepted loads and the six performance fixtures. Whole campaigns remain rejected by the host guard.

| Experiment | Paired accepted loads | Geometric-mean rate change | Disposition |
|---|---|---:|---|
| Scratch pools versus original Stark | 2, 3 | +0.33% | Reduces allocator work; isolated timing contribution is within variation |
| GEMV tile eight versus scratch-pool build | 0, 1, 2 | +2.61% | Retained with narrow admission and bitwise tests |
| Packed fused DeltaNet versus scratch-pool build | 1 | -2.39% | Rejected |
| Packed split DeltaNet versus scratch-pool build | 1 | -2.03% | Rejected |

MLX's packed DeltaNet recurrence uses four lanes per value row; Stark's original decode kernel uses a full SIMD group per row. Two adaptations passed CPU/Metal numerical-tolerance tests but changed greedy token/text hashes on Fibonacci, medium context, long context and long output. They therefore matched only 6/10 fixtures, and the paired end-to-end timing did not justify promotion. Those differences are floating-point/output changes, not evidence of a measured quality regression. The adaptations are absent from the default engine and retained as a [rejected experiment patch](experiments/mlx-optimization-2026-10-20/rejected-packed-delta.patch) with Apple's MIT notice and source revision. It applies to `54d113a7034fc0d9bacb4c0653a7713500fb5bbe`.

The separate [synthetic GPU-duration diagnostic](experiments/mlx-optimization-2026-10-20/delta-gpu-diagnostic.json) had wide ranges and no host/clock controls. Its results do not qualify either adaptation's generation speed; its missing original test-executable hash is recorded explicitly.

## Correctness, provenance and validation

- Original and modified Stark matched token and decoded-text hashes on all ten fixtures, including loads excluded from timing. Within-engine repetitions were stable.
- The final reference/modified digest workers matched **23/23 complete-vocabulary F32 bit digests**, with 248,320 logits per input. The inputs are the ten fixtures plus thirteen prefix lengths spanning single-token and 1024-token chunk boundaries. Caches were disabled. This checks prompt scoring on those inputs, not every intermediate generated-token distribution or every supported model. [Numerical receipt](experiments/mlx-optimization-2026-10-20/final-numerics.json).
- Twelve GEMV policy tests passed, including private comparison of every BF16 output bit for both tuned shapes and fallback guards. Three allocator/completion tests, eleven generation-worker tests, four stock-worker adapter tests and twenty-eight repository-policy unit tests passed.
- The locked Rust 1.98.1 default release build, default engine check including grammar, nightly formatting, Python lint/format, locked environment checks and repository policy passed. Builds emitted the existing local `rust-objcopy` missing-LLVM stripping warning and still completed successfully.
- No model, converter, host-application or installed service was changed. No JevK5 speed improvement is established. There is no continuous GPU, thermal or power isolation, confidence interval, general-quality qualification or public-article acceptance.

Two setup errors are preserved separately. The first resumed three-engine attempt used an incorrectly resolved base-Python path and was stopped after MLX startup failures. The corrected campaign uses the literal venv executable. An earlier digest diagnostic supplied one candidate to an API requiring two to sixteen; the corrected and final diagnostic receipts are separate. Neither failure is a model-support or numerical-regression result. [Failure dispositions](experiments/mlx-optimization-2026-10-20/invalid-attempts.json).

The [protocol](experiments/mlx-optimization-2026-10-20/PROTOCOL.md), [source/build provenance](experiments/mlx-optimization-2026-10-20/PROVENANCE.json), [derived analysis](experiments/mlx-optimization-2026-10-20/analysis.json), raw aggregate receipts and redacted configuration templates are retained. The [analysis program](experiments/mlx-optimization-2026-10-20/analyze.py) recomputes accepted-load intersections, rates, paired ratios and hash agreement from those receipts. Raw prompts, responses, token vectors and logits are absent from the packet.

## Replay

Build the original default engine at `54d113a7034fc0d9bacb4c0653a7713500fb5bbe` and the modified default engine at `045d0ca6cca8eafae89f32a303e5123f001eaf43` with the locked Rust 1.98.1 release command in provenance. Use separate binary paths. Reuse the exact model artifacts identified by hashes; do not substitute a different quantization or source revision.

Create an owned Python environment from `tools/stock_benchmark/mlx_current` using its lock. Fill the placeholders in a configuration template with local paths and use the existing generation driver. Keep the literal venv interpreter path; resolving its symlink selects base Python. For numerical replay, use `scripts/verify_stark_optimization.py` with the two digest-worker binaries, model path and source-tokenizer path.

The retained source checkout and environments remain ignored under the task runtime. This document records engineering evidence and implementation limits. No article, benchmark release or external announcement was created.
