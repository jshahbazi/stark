# DeepInfer-inspired Stark resource reuse: validation protocol

Frozen before the performance comparison. Upstream study: DeepInfer `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`, root Apache-2.0. No upstream code is executed or copied into Stark. The changes are independently written Rust adaptations of stable device inputs and reusable execution metadata.

## Candidate and reference

- Reference: the unchanged optimized Stark engine from `8be4b52d` (engine code from `045d0ca6`), existing Rust 1.98.1 locked default release binary SHA-256 `66c0aae91af87dadcfb461de536b27479822ea446ffddc78897e5bd98301ba89`.
- Candidate: the same default build with direct one-token buffer handoff in ordinary decode and successful-completion-gated reuse of Metal argument tables alongside command allocators/buffers. Grammar and speculator token-copy paths retain their previous implementation. No kernel arithmetic, sampling, model, conversion or dependency changes.
- Publish completed command resources to the cache before notifying waiters, and discard failed submissions from the cache. Tests exercise exclusive outstanding tables and rebinding of buffers, changing constant offsets and optional-input specializations. Actual GPU execution failure is not injected.

## Correctness gates

- Build release generation, scoring and scoring-digest tools using Rust 1.98.1 and locked dependencies; record binary hashes. Compile the default engine including grammar.
- Run the two new Metal cache tests and bounded existing sampler property tests. Test-harness bootstrap is permitted only for Rust's custom test runner; production builds do not use it.
- Compare complete-vocabulary F32-bit digests against the unchanged reference on the existing ten fixture inputs and thirteen boundary-prefix lengths, with scoring caches disabled. Preserve all 23 results separately from previous campaigns.
- Require output-token and decoded-text hash parity on all ten generation fixtures, including timing-excluded loads. Preserve early termination and hash failures; do not rank unequal work.

## Performance comparison

- Use unchanged `scripts/benchmark_clean_qwen.py`, `scripts/benchmark_stock_engines.py` and native worker protocol. Compare the two Stark binaries only. The prior MLX comparison remains separate.
- Existing Qwen3.5-2B affine W4/G64 checkpoint revision `93760be4f1f69842a46bc13dbdc0f19e291392a3` and unchanged Stark conversion; context8192, identical integer inputs, greedy sampling, seed0, fresh request state. One loaded task-owned worker and one in-flight request.
- Six paired independent model loads, five measured repeats per each of six performance fixtures, one measured repeat per four sanity fixtures. Warm all ten fixtures first. Rotate/reverse engine order using the existing algorithm, balanced three first positions per engine.
- Primary rate: actual output tokens divided by complete client request time, including prefill, decode, IPC/readout and cleanup; exclude loading, warmup and admission. Report paired-load median-rate ratios and their geometric mean. Do not attribute the combined result to either change individually.
- Before loading and after warmup, require ten seconds of fresh contiguous quiet samples: each external process at most80% CPU, aggregate at most150%, recognized model process at most10%, device GPU utilization at most5%, nominal thermal state. Admission timeout120seconds.
- Reject timed loads for external process CPU above100%, aggregate above200%, recognized model activity above10%, non-nominal thermal state, sample failures/gaps above2seconds, pageout or swapout increases, or unequal performance output work. Retain all loads and compare only matched accepted pairs. Whole-campaign acceptance requires all12 loads to pass plus sleep/clock checks.
- Hide Codex windows through macOS System Events and verify visibility before timing; restore previous visibility after the run. Finish all builds, tests and agent reads before measurement. Do not terminate unrelated services or developer jobs.
- Preserve all attempts and host attribution. Any further campaign after interference must be separately labeled; never pool favorable cells across attempts or relax gates after observing speeds.

Device GPU utilization cannot identify competing GPU clients during requests. Per-process GPU/power monitoring is unavailable without additional privileges. Thermal state does not establish identical clock frequencies. This is a bounded Apple M5 Max experiment, not qualification of other models, devices, grammar/speculative performance, general answer quality or a new Stark-versus-MLX ranking. Persist only scalar measurements, process basenames, counts and hashes.
