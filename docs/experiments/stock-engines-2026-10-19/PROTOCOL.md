# Stock engine pilot protocol

Frozen before the measured runs on 2026-10-19. Admission requests and driver
repairs are separate from measurement. This is a local pilot for a future
article; broad engine superiority and publication readiness are not assumed.

## Rules

- Same Apple M5 Max, 128 GiB, AC power. No engine/model implementation patches,
  model metadata edits, bespoke kernels or fresh quantization. Standard builds,
  release installs, standard conversion, and external measurement drivers only.
- Models and preexisting runtimes are borrowed read-only. New releases and Python
  environments live under the task-owned `runtime/stock-comparison-2026-10-19`.
- Pin source/model/tokenizer revisions, locked dependencies, executable hashes,
  driver hashes and engine flags. Keep payloads in memory/private IPC; persist
  aggregate outcomes, scalar timings and hashes only.
- Run one measured engine at a time. An existing installed JevK5 server is left
  untouched. Sample background CPU every 0.5 s, rejecting a block when any one
  non-owned process exceeds 200% CPU or a recognized model process exceeds 10%.
  Preserve excluded blocks. This is a CPU screen; continuous GPU/thermal/power
  isolation is unverified and Python-named competing inference can be missed.
- Hold an owned idle-sleep assertion. Check the whole power-log window and the
  wall/monotonic clock discrepancy. Sleep invalidates timing. Preserve failures
  rather than substituting successful fragments or selecting fast observations.

## Qwen generation

Common MLX source: `mlx-community/Qwen3.5-2B-MLX-4bit`, revision
`93760be4f1f69842a46bc13dbdc0f19e291392a3`, affine 4-bit/group-64.
Stark uses the existing repack. MLX-LM, TensorFold and vLLM Metal load the source
unchanged. llama.cpp uses the already-present Unsloth Q4_K_M artifact, whose
SHA-256 matches `unsloth/Qwen3.5-2B-GGUF` revision
`f6d5376be1edb4d416d56da11e5397a961aca8ae`. It is a different quantizer;
source-weight equality across the derivative quantized checkpoints is unverified.
This arm measures an engine-and-format stack.

Measured configurations: Stark default release; MLX-LM 0.32.2;
llama.cpp b11747 (`f711c8f3038020d71f69b8ced09162b014fd7335`);
TensorFold 1.1.1 default drafting and built-in plain mode; vLLM Metal 0.31.0.
MetalRT is not admitted because exact stock support for this model is unverified.

- Stock source chat template, thinking off, identical integer input IDs, greedy
  decoding, seed 0 where exposed, context 8192, retained/prefix caches disabled.
  Unknown cache counters remain null, with configured cache policy recorded.
- Six generated public fixture templates in `benchmark_stock_engines.py`: code,
  explanation, copy/edit, medium/long context, and longer code output. Five use
  a 128-token limit; the final fixture uses 512. Record actual input lengths.
- Four independent loaded-model blocks per configuration. Pair a chosen engine
  order with its exact reverse, then rotate the next pair. One untimed warmup
  per performance fixture/load; three timed repetitions give twelve per cell.
- Primary rate is actual generated tokens divided by client complete-request
  wall time, including prefill, decoding, native teardown and transport. Model
  loading is separate. HTTP and private-pipe routes are disclosed.
- Fixed-work ranking requires `length` termination and exactly the requested
  output count on every compared sample. Keep early stops and token-count
  differences visible. EOS counting differs on natural stops, so those rates
  cannot be pooled with fixed-output measurements.
- Four additional short arithmetic/format checks are sanity checks, one per
  load. They are not a generation-quality benchmark. Persist correctness counts
  and output hashes, never answer text. Report per-engine repeatability and
  cross-engine hashes separately from semantic quality.

## JevK5 scoring

Model: `alibiserikbay/JevK5` BF16, revision
`c4f7fdb3aeab5582336406e78d3bef11bf98833d`. Reuse the original source,
Stark-native BF16 repack, and the existing BF16 GGUF control. Compare unchanged
Stark and current llama.cpp through their ordinary native scoring APIs.
MLX-LM 0.32.2, TensorFold 1.1.1 and vLLM Metal 0.31.0 reject the unchanged flat
source during admission; no configuration shims or custom ports are permitted.

- Use all 243 sealed cases from the prior `uzu-optimization-combined` reference:
  231 public JevBench v1.2 and twelve synthetic controls. Verify the dataset,
  configuration, four blocks, two digest records and settled-ownership seal.
- Preserve the sealed input/candidate token IDs and gold label ordering. No
  prompt regeneration, truncation, changed candidate set or dataset filtering.
- Context 8192, candidate temperature 1.22, fresh native state per request and
  output caching off. Full-vocabulary log probabilities use raw logits;
  calibrated probabilities use softmax over candidate logits divided by 1.22.
- Four independent process loads per engine in AB/BA/AB/BA order. Three warmups
  per load, then one fresh request for each of the 243 cases. This yields 1944
  measured requests if both engines complete all loads.
- Primary timing includes context creation, prefill, normalization, readout,
  cleanup and pipe exchange, excluding model load. Bucket medians by input
  length; preserve all scalar timing samples. No hidden generated readout token.
- Assess public/synthetic argmax accuracy, public Brier and ECE-10, cross-engine
  maximum/mean probability error and repeatability. These are development-set
  findings. Agreement is separate from correctness; historical reference
  probabilities are not treated as ground truth.

## vLLM serving extension

A separate vLLM-only scaling check may measure concurrency 1/4/8 using its stock
scheduler. Use one fixed configuration, independent process loads, prefix caching
off, identical fixed-output work and rotated load order. Report wave throughput
and request latencies, host exclusions and actual token counts. This does not
rank other engines' serving stacks or establish an optimized serving frontier.

## Analysis and cleanup

Report medians, block ranges, exact work counts, cache coverage, output hashes,
quality limitations and support failures. Four process blocks provide limited
independent replication; do not claim reliable p95 or statistical significance
from the per-request repetitions alone. Keep excluded attempts separate.

After preserving tracked drivers, dependency locks, protocol and aggregate
evidence, stop/reap owned processes, remove task-owned temporary RPC directories
and the task-owned runtime root, and verify absence/open handles. Preserve
preexisting models, source downloads, app packages and runtimes. Record logical
task size and actual filesystem free-space change separately. No article or PR
is created by this pilot.
