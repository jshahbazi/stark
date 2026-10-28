# Qwen3.8-Flash-Next performance literature and engine review

Date: 2026-10-28. Scope: native Stark execution on Apple M5 Max with 128 GiB.
This is a research review of the ongoing `codex/native-glm53-qwen38` work, based
on primary papers, pinned source, and existing aggregate local captures. It does
not add runtime code or run models. Later benchmark records supersede the local
candidate timings quoted here.

## Finding

The strongest remaining direction is to keep the complete decode graph on the
GPU, including routing, normalization, residual updates and sparse selection.
Stark has already combined several stages, but its Qwen path still completes
255 command submissions per token. A combined command containing separate
expert kernels is also an intermediate step toward kernels that gather experts
and fuse their arithmetic.

In the third replay of the private short profile, native command execution was
62.91 ms/token, of which recorded GPU execution was 28.05 ms. Their difference
is 34.86 ms/token. These counters are inclusive: the 60.55 ms wait counter must
not be added to either. This identifies substantial work between GPU spans and
completed commands; it does not prove that every millisecond of that difference
is removable. This instrumented exploratory capture is not a clean unprofiled
speedup estimate. Even eliminating the difference would leave kernel and CPU
work to improve.

The full worker candidate measured 12.06 decode tokens/s and 12.72 prefill
tokens/s on a 54-token input and up to 32 sampled tokens, with fresh request
state and exact repeated completion hashes. It is an intermediate candidate,
not a long-context or sustained-serving qualification. Its generation binary
SHA-256 is `7115dcbc6f9c62d45e7449a5bb360d6a7f3261d83fd1f769b47cb224973c1402`;
aggregate report SHA-256 is
`f53e91197f9d2b67d326f2bee6d0ef3a7afc6c37361be8c69a396db9cb9057b5`.
The short profile SHA-256 is
`19016425982275193b88547a03c4a436b83b6cf475f31c01338b40a7c895e840`.
Raw reports remain private and ignored.

## Papers and architecture constraints

The [Qwen architecture report](https://arxiv.org/html/2608.30320v1) describes
125B backbone parameters with 6B active per token and an additional 51B n-gram
embedding. Its hybrid attention and widened gated residual change the inference
cost distribution: optimizing ordinary transformer attention alone misses much
of the model. Host-side n-gram lookup is an intended architectural choice, so
moving the entire embedding table onto the GPU is not a prerequisite for fast
execution. The paper's FlashQLA results are on NVIDIA GPUs and do not establish
a Metal speedup.

[Gated Delta Networks](https://arxiv.org/html/2412.06464v3) supplies the
chunkwise formulation needed to replace token-by-token prefill with matrix
operations while propagating recurrent state between chunks.
[FlashQLA](https://github.com/QwenLM/FlashQLA/tree/5538c35f0df33ef804eacd90c77d9d9087c1d4e4)
is a practical fused chunk-prefill reference. Its reported 2–3× forward gains
over FLA concern Hopper/Blackwell workloads; its algorithms are relevant, while
its TileLang kernels and numerical choices require a separate Metal port.

[Speculative decoding](https://arxiv.org/abs/2211.17192) can preserve the target
sampling distribution through an acceptance/rejection correction. That result
does not make arbitrary draft acceptance, approximate sampling, or recurrent
state rollback correct automatically.

## What the engines demonstrate

| Primary implementation | Transferable evidence | Boundary |
| --- | --- | --- |
| [Lily Metal transport](https://github.com/fabiogreter/lily-qwen3.8-flash-next/blob/e22d58fa7cc488493fb1b5d810a198bd29730aeb/src/metal.rs) | Encodes a dependency graph in Metal 4 and supports GPU event control. This removes host wakeups from the ordinary decode loop. | A transport change alone cannot remove CPU dependencies left in Stark's operators. |
| [Lily architecture](https://github.com/fabiogreter/lily-qwen3.8-flash-next/blob/e22d58fa7cc488493fb1b5d810a198bd29730aeb/docs/architecture.md) | About 920–990 kernel dispatches share one concurrent pass; GPU routing, sampling and expert gathers retain intermediate state. Chunk prefill groups routed rows by expert and uses tensor-operation GEMMs. | Kernel dispatch count and command submission count are different. Its approximately 4.4 GB/token traffic estimate describes its conversion and arithmetic. |
| [mlx-serve MoE](https://github.com/ddalcu/mlx-serve/blob/e1d2013c1a321ba9e04324d790254e3f5428265c/src/plugins/qwen4_exp/trunk/moe.zig) | `gatherQmvGateUp` combines gate/up projection and activation. A fused down path applies router weights and reduces experts without materializing the complete expert-output array. It also has sorted grouped prefill paths. | Check each fast path's dtype, group size and quantization predicates. Combining existing commands does not reproduce these kernel fusions. |
| [MLX quantized kernels](https://github.com/ml-explore/mlx/blob/7d2d94a5561ebc3965884a0fc3dfc35b7b1bb17a/mlx/backend/metal/kernels/quantized.h) | Packed vector loads, reuse of scale/bias data, SIMD reductions and shape-specific schedules reduce decoder work and improve occupancy. | A winning shape is insufficient evidence for another shape; skinny projections, expert down projections and large heads need separate measurements. |
| [MLX-LM generation](https://github.com/ml-explore/mlx-lm/blob/1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7/mlx_lm/generate.py) | Chunked prefill evaluates cache state, then the decode loop asynchronously evaluates the next step before consuming the current result. | Native Stark must implement the corresponding dependency and lifetime contract; calling asynchronously around blocking operators is insufficient. |
| [Uzu Metal command buffers](https://github.com/trymirai/uzu/blob/f1231705cc7e26a205819da7c78e0110cfbac306/crates/uzu-engine/src/backends/metal/command_buffer.rs) | Reuses command allocators, retains allocations through feedback completion, and exposes one pending completion for encoded work. Stark already derives from this infrastructure. | Reuse the backend abstraction for hybrid graph encoding; an async receiver per individual matvec still retains the same dependency count. |
| [llama.cpp Qwen graph](https://github.com/ggml-org/llama.cpp/blob/ff0d396e3dda0f33e7ce5ad4267fce4f7690cfb1/src/models/qwen4exp.cpp) and [Metal GDN](https://github.com/ggml-org/llama.cpp/blob/ff0d396e3dda0f33e7ce5ad4267fce4f7690cfb1/ggml/src/ggml-metal/kernels/gated_delta_net.metal) | Independent graph and recurrent-kernel references help review operation ordering and backend boundaries. | Their presence establishes an implementation reference, not a throughput result for Stark's mixed MLX checkpoint. |

Lily's author reports **77–80 plain-decode tokens/s for q4-xl** across 4K–64K
contexts on a 40-core M5 Max with 128 GB, using fresh real-text HTTP requests,
256 greedy output tokens, three interleaved repeats and completed n-gram
preload. Speculative rows are separate. This is useful evidence that the
hardware/architecture pair can run much faster, but the conversion, activation
precision, context and procedure differ from Stark's candidate. It is not a
matched benchmark or a guaranteed target. See its
[method and performance history](https://github.com/fabiogreter/lily-qwen3.8-flash-next/blob/e22d58fa7cc488493fb1b5d810a198bd29730aeb/docs/performance.md).

## Recommended implementation order

| Priority | Stark change | Why it should help | Effort and gate |
| --- | --- | --- | --- |
| 1 | Introduce device tensor handles and encode a whole layer, then a whole decode step. Move router/top-k, norms, residual injection and QSA control onto the GPU. Read back only the result required by the worker. | Removes repeated CPU-visible completions and intermediate upload/readback boundaries. The observed command/GPU gap supplies a measurable target. | Large. Preserve source top-k tie order, causal visibility, failure status, reset and model serialization. Measure submissions and total latency separately. |
| 2 | Use bank-indexed expert gather dispatches, then fused gate/up/activation and down/weight/reduce kernels. | Qwen's retained banks permit expert selection through GPU indices. This removes per-expert bindings/dispatches and redundant input reads beyond the current combined FFN command. | Medium. Validate all supported 4/8-bit groups and scale dtypes, expert order, tails, duplicates, nonfinite intermediates and fallback paths. |
| 3 | Implement chunked prefill: dense GEMMs, expert-sorted grouped GEMMs, GDN chunk scan, batched QSA and final-row vocabulary readout. | Reuses weights across input rows and exposes matrix work to M5 tensor operations. Sequential GEMV prefill cannot approach optimized long-prompt prefill. | Large. Compare every chunk boundary against recurrent execution, including EOS/PLE history, convolution, partial sparse blocks and reset. Last-row readout skipping already exists. |
| 4 | Tune QSA split size and head/query grouping at long context. Keep per-query masks when reusing unions of selected blocks. | Improves occupancy when one query's sparse selection provides too few threadgroups; avoids rereading shared K/V tiles. | Medium. Benchmark 1K, 8K, 32K and the admitted context cap; check exact visible-token sets and partial tails. Short prompts cannot qualify this path. |
| 5 | A/B a request-scoped macOS activity assertion. | May prevent windowless sustained work from losing its compute performance envelope. | Small. [Lily's implementation](https://github.com/fabiogreter/lily-qwen3.8-flash-next/blob/e22d58fa7cc488493fb1b5d810a198bd29730aeb/src/activity.rs) ends the assertion between requests. Its measurements show clearer gains for prefill/speculation than memory-bound plain decode. Measure clocks/power and sustained latency; do not assume a decode gain. |
| 6 | Evaluate BF16 activation storage with F32 accumulation/state as a separately qualified execution policy. | May reduce activation traffic and register pressure and enable faster matrix paths. | Medium/high numerical risk. Keep the F32 baseline and current gates. Near-tie router/indexer decisions can change later trajectories; do not silently relax tolerances or narrow recurrent state. |
| 7 | Add matching MTP weights and batched verification after the base graph is efficient. | Amortizes target weight reads across accepted output tokens. | Large. The current checkpoint inventory has no identified MTP head. Establish artifact identity, exact greedy acceptance, sampling correction, and rollback of recurrence, convolution, attention and PLE history. Measure accepted tokens per wall second across several workloads. |

The easiest bounded next experiment is expert-indexed gather plus fused MoE;
the largest architectural opportunity is a complete GPU decode graph. Prefill
requires its own implementation and measurements.

## Numerical, memory and measurement limits

[mlx-serve's compiled-activation code](https://github.com/ddalcu/mlx-serve/blob/e1d2013c1a321ba9e04324d790254e3f5428265c/src/engines/mlx/compiled.zig)
records a generic SiLU fusion causing greedy divergence around token 55 and
uses a dedicated kernel instead. This is direct evidence against accepting a
fusion solely because a short completion looks plausible. Stark's existing
Qwen source gate remains `2e-5`; mixed-weight fixtures and four-stream fixtures
must both pass. Reordered reductions, FMA, approximate exponentials and BF16
casts need explicit comparison on identical prefixes, with aggregate errors
and private output hashes. Small fixtures do not establish general answer
quality.

Keep the n-gram table CPU-mapped. Its warm local short-profile phase was about
0.405 ms/token, so it is not the first decode target in that capture. Row
prefetch or bounded preload can improve cold faults, but measure cold time,
system pressure and effects on resident main weights separately. Retained
Metal wrappers and best-effort residency requests are not proof of wired
physical pages. Process RSS excludes significant GPU and OS file-cache memory.

Benchmark each candidate after builds and competing GPU work stop. Use longer
outputs and multiple fresh prompts, interleave candidates, separate cold load,
warm prefill, plain decode and speculation, and report token latency tails as
well as aggregate throughput. Record device model, power mode, screen/GPU
competition, paging, precision, checkpoint and binary hashes. Keep prompts,
outputs, token vectors and decision probabilities private. Persistent session
or cross-request prefix caches from other engines are outside Stark's current
consecutive-identical-request cache contract.

## Source and license inventory

No external runtime code was copied for this review. These are the inspected
revisions; root licenses were read. Any later adaptation must retain applicable
copyright/notices and inspect file-level provenance and bundled licenses.

| Project | Inspected revision | License boundary |
| --- | --- | --- |
| Qwen model repository | `69885871a64393807d988b27b1b5e380e8f28526` | [Model metadata](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json) declares Qwen Community 1.0; code and weight terms must be checked independently. |
| Lily Qwen fork | `e22d58fa7cc488493fb1b5d810a198bd29730aeb` | Apache-2.0 root license; adapted kernels retain upstream notices. |
| MLX | `7d2d94a5561ebc3965884a0fc3dfc35b7b1bb17a` | MIT, Apple copyright. |
| MLX-LM | `1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7` | MIT, Apple copyright. |
| mlx-serve | `e1d2013c1a321ba9e04324d790254e3f5428265c` | MIT root license plus Apache-2.0 and component notices; audit the actual imported files. |
| Uzu | `f1231705cc7e26a205819da7c78e0110cfbac306` | MIT, Mirai Tech copyright. |
| llama.cpp | `ff0d396e3dda0f33e7ce5ad4267fce4f7690cfb1` | MIT, ggml authors; bundled components have separate notices. |
| FlashQLA | `5538c35f0df33ef804eacd90c77d9d9087c1d4e4` | MIT; NVIDIA/TileLang implementation is an algorithmic reference for a Metal port. |

This review found no primary source establishing that an unchanged native F32
Stark implementation should achieve Lily's BF16-oriented rates. It found
concrete graph, kernel and prefill mechanisms that can be implemented and
measured without adopting another serving runtime.
