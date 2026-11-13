# 20261112-EDGE0-05: F32 Edge0 reference forward

## Identity and state

| Field | Value |
| --- | --- |
| Registered / owner | 2026-11-12 / Codex, local non-commercial evaluation only |
| Kind / scope | Phase 3 compatibility, component correctness and bounded checkpoint smoke |
| Baseline | `49d6ed10`, native reader/block component gate passed |
| State / result | Implemented; Phase 3 component and bounded checkpoint gates passed |
| Promotion | Explicit diagnostic F32 reference only; no app/catalogue availability |
| Predecessor | [20261112-EDGE0-04](2026-11-12-edge0-block-readers.md) |
| Governing plan | [Phase 3](../EDGE0_PLAN.md#phase-3-reference-forward-in-the-f32-graph) |

## Hypothesis and change

The existing native F32 Qwen trunk can consume the verified GGUF skeleton with
an explicit tensor/geometry adapter, FP8 table gather and grouped latent expert
stage. Retain source bytes, use bounded selected mappings and preserve routed
expert reduction order. Independently validate newly introduced arithmetic
before running the actual checkpoint. Existing Flash-Next qualification does
not establish this artifact's numerical behavior.

## Artifact, source and precision

- Checkpoint: `Edge0/Edge0-145B-q3k` at
  `d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`, prepared receipt SHA-256
  `8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`.
- Runtime format/arithmetic authority: Edge0 source
  `b7900cbb322926ccebdcd0c0c5a1280497ecff22` and its pinned llama.cpp
  `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Trace tensor transformations
  and normalization conventions from source, not tensor-name resemblance.
- Preserve Q8_0/Q3_K/IQ4_NL/FP8/BF16/F32/F16 payloads. Initial reference
  activations, accumulation, recurrence and cache values use the existing F32
  graph policy; routing retains the source's explicit F64 exponential/reduction
  stages and F32 probability boundaries. The upstream's 8-bit activation quantization is a separate
  measured control; floating activation execution does not inherit its proof.
- Build generation alone with locked Rust 1.98.1 and its distinct target.
  Build scorer separately if needed; never unify its optimized features into
  generation. Record executable/source and artifact seals for each actual run.

## Fixed structural and numerical gates

1. Check all trunk geometry fields against the sealed skeleton. Tensor aliases,
   fused-row views and supported reshapes must preserve raw bytes, row strides,
   shape and ownership. Inspect transformed recurrent parameters and centered
   norms explicitly. Reject incomplete/unrecognized layouts before readiness.
2. Map expert records through checked per-expert spans, retained source owners
   and bounded selected mappings; never expand the whole model to F32. Keep
   stored/logical table rows distinct and validate table format/global scale.
3. Apply `a_g = (D_g x) / s` once. Each expert consumes its labeled group,
   evaluates gate/up, SiLU/product and down; apply its group's U separately,
   then reduce in the original routed order. Grouped-up reassociation is
   excluded from this reference attempt. Preserve the full-width shared expert.
4. Register the independent F64 synthetic fixture and exact source/program/input
   hashes before comparison. It must exercise all new latent-stage operations,
   repeated/different group labels, routing and readout. State its trunk scope
   explicitly; a component fixture cannot claim whole-model parity.
5. For fixed normalized synthetic inputs/weights, require exact route/group IDs
   and finite outputs. Per-matrix limits remain the Phase 2 F64 bound
   `2e-5 * max(sum(abs(products)), 1)`; final fixture logits require maximum
   absolute error at most `0.002` and relative L2 error at most `0.0002`.
   Fix corpus, geometry, seeds and scales before observing results. Any changed
   fixture or threshold gets a distinct follow-up with the failure preserved.
6. Measure floating versus source-emulated 8-bit activations on the fixture and
   one bounded real layer, with unchanged weights/input. Report each measured
   difference and reduction/quantization policy; no source-equivalence claim
   follows merely from a floating-reference pass.
7. Qualify the derived tokenizer in the actual Rust 0.22.2 worker path before
   treating Python 0.23.3's preparation proof as runtime proof. Keep source,
   skip/render-special and UTF-8 boundaries explicit.

## State inventory and runtime controls

The new projections, labels, scales and expert/table lookups are stateless.
The implemented reused owners are `QwenModel.position` and retained n-gram
token history; each GDN's F32 V-head/value/key recurrent array and per-channel
convolution window; each attention layer's growing K/V and index-key prefix
rows; and PLE's convolution history, with `(kernel-1) * ngram_size * streams *
hidden_size` elements. Edge0 disables the Qwen device graph and retains its
recurrent owners on the host. Reset clears all of these owners/positions.
This bring-up retains fresh-per-request execution with
`prefix_cache.supported: false`; exact state restoration is unimplemented here
and remains Phase 6 work. MTP stays unavailable on the reference path. The
source has no image tower; image input is unavailable/N/A, not silently dropped.

The concrete restoration blocker is the absence of a consistent export/import
snapshot covering these F32 owners together with position and n-gram history.
This records an implementation gap; it does not classify exact restoration as
impossible. Group D activations and route buffers are local to a single latent
call and never become a cross-request cache.

Before a full-checkpoint run, admit a bounded configuration with context 512,
output budget 32, images off and MTP off. Fix a synthetic ordinary-chat case
and template/stop/sampling controls in the run recipe before execution. Run
greedy and seeded stochastic replay, reset/recovery and a fresh process.
Require readiness, successful finite generation, expected stop/length handling
and deterministic repeated output hashes under matching settings. Keep prompt,
completion and token vectors in memory; publish aggregates/hashes only.

Serialize model/GPU runs and own their workers. Screen available memory before
loading; bound the worker to 48 GiB RSS for this first reference smoke, stop on
resource failure and do not kill unrelated processes. Make any required source
load-memory estimate explicit before execution. No broad quality, speed,
full-context or low-memory-host qualification follows from this bounded run.

## Attempt history and failures

Native implementation, component qualification and the first bounded full
checkpoint smoke have run. The adapter retains the prepared receipt and original files. Loading
requires `STARK_EDGE0_REFERENCE=1`; the runtime configuration is derived in RAM
and advertises a diagnostic maximum of 512 tokens. Normal preparation identity
and catalogue admission remain disabled.

### Source conventions established before execution

- The pinned GGUF converter already changes `A_log` to `-exp(A_log)`, adds one
  to centered normalization gammas and tiles recurrent V heads and associated
  operands. The adapter retains those stored values. The reference recurrence
  consumes negative A directly and associates V head `h` with K head
  `h % key_heads`. Existing MLX execution retains its grouped-head convention.
- The PLE table width of 160 is per head. Sixteen heads give a 2,560-element
  gathered embedding; the single physical table retains 320,001,536 rows with
  320,001,446 logical rows. The source PLE layer index 1 becomes HF-style index
  2. Its history-reset token 248044 is independent of primary generation EOS
  248046; the generation stop set retains all five source EOG IDs.
- Full source softmax uses F64 reductions with F32 probability boundaries,
  stable top-k and selected-probability renormalization. Preserve the source
  probability tie behavior, selected order and per-route U products.

### Independent synthetic reference and activation controls

The component fixture fixes seed `0xED6E003`, widths H/latent/inner 256, three
groups, five experts, three selected routes and three input cases. Labels
`[0,1,0,2,1]` exercise shared and different projection groups. Stored matrices
use Q8_0 D/U/shared/readout, Q3_K gate/up and IQ4_NL down. Pinned GGML C supplies
quantization/dequantization; the independent scalar Python F64 forward supplies
60 finite matrix operations and latent/shared/readout expectations. It includes
no attention, recurrent trunk or PLE arithmetic.

The initial floating-versus-source-CPU activation control measured final
maximum absolute difference 0.0290203941 and relative L2 difference
0.0181715056. A separately registered Q8_1 activation-value control measured
0.02684639021 and 0.01765730721. Both retain F64 matrix/nonlinear arithmetic;
neither reproduces CUDA reductions, fast math or every BF16 boundary.

The bounded real layer-0 control fixes seed `0xED6E004` and a hypothetical
normalized synthetic input, with actual router, shared, D/U, expert and scale
bytes. Ten selected experts use six groups. It read 82,647,040 bytes of tensor
payload and peaked at 200,572,928 bytes RSS. Routes remain fixed across the
floating and quantized-activation controls. Both source CPU and Q8_1 value
controls measured maximum absolute difference 0.00207058786331 and relative
L2 difference 0.0122264079207, with finite outputs. This is an isolated
activation-policy measurement on actual layer weights, not an observed model
hidden state or full source/GPU numerical equivalence. These differences are
not candidate-versus-F64 fixture errors and do not use its 0.002 pass gate.

| Private aggregate record | SHA-256 |
| --- | --- |
| Initial fixture registration | `70d206a21775b928359dd593cf66df6988627e6cdd3ef149b7c0ad2140b4ecc4` |
| Fixture input seal | `7e2133ed975fee822accc64a9635c1ba33fc063bae9a4cf1ae7cf116e7965156` |
| Synthetic expected values | `40f2b8e62b9c071eebefd0c73095c8980952b11bf293232380f4b40cb446f2d4` |
| Synthetic reference report | `8cc855cdf87a72c7b5aa8b3c97a1ec799b5a5dca9369c3268c1b7f250a8c757b` |
| Supplemental Q8_1 registration | `2e2ae09dfa075ad53bdb9be85accb086eed68a342f192b8f117568f6fe60d08f` |
| Supplemental Q8_1 output | `f46b4af88ac421ecbdb7e21e6a8c9c4c34396e6679700862513aaec9745e2364` |
| Real-layer control registration | `08757349e926c06f0294c5022683af19d5264eb1684486550395d12ef8e7418e` |
| Real-layer control report | `660295d5a5e753bb79739268b2466b90e19ad2ad3d0365199bfc79c70b307428` |

The original executed fixture and supplemental-control helper bytes are retained
privately as `executed-v1` copies, SHA-256
`b2c5d9cfd927baf774507055a0ef60dea0a440f39b678b32196f366c1155dc28`
and `359c2532e4aa24c72438cca5a9d5b109679634a12014ee08a6eb0769326492d5`.
Subsequent tracked import/format cleanup does not rewrite their receipts.

### Locked Rust tokenizer helper gate

The Rust 1.98.1 release helper built in 87 seconds and exercises the worker's
locked `tokenizers` 0.22.2 dependency through private pipes, without an inference
model. All 3,884 encodes, 1,943 pre-tokenizer boundaries and 496,654 decode
comparisons passed with zero mismatches. Decode cases comprise 494,760 valid
UTF-8 and 1,894 replacement-boundary cases. All 3,750 complete ordinary UTF-8
incremental-decoder cases passed. Six independently mutated tokenizers were
detected, including the previously failing implicit Unicode class behavior.

This qualifies the Rust tokenizer parser/decoder helper against the sealed
source corpus. It does not qualify actual worker generation, template tool
parsing, all possible incremental byte endings or an upstream frontend's text
policy. The prepared tokenizer and previous Python receipts remain unchanged.

| Private record/build | SHA-256 |
| --- | --- |
| Rust helper executable | `032036972db02ab3d6fc3d9459565989783867a3bd25ce772ca78b933d90c954` |
| Rust helper registration | `89b66a8a8fc7ddf5a6ae043af82a5c645c86a85d69de43f17aea88a323a41bd2` |
| Rust helper report | `12736cfb51c32a4355d49f66d1668ead96a0dde2a96807901b8b9d7828aa078a` |

Before checkpoint execution, independent review found two validation gaps:
small prepared tokenizer/configuration bytes were not rechecked against the
sealed receipt, and progress frames could renew a per-frame timeout forever.
The adapter now verifies all bounded small receipt files; large payload hashes
remain the Phase 1 publication proof under the immutable-file assumption.
The smoke watchdog uses a single monotonic request deadline across all frames.

### Conditional Strata verifier outcome

Source inspection establishes that the pinned build exposes no CPU-only
registered latent-forward verifier for macOS arm64. This does not mean that
all of Strata is CUDA-only: a portable CPU library exists. The relevant
`latent_layer0_verify`, `latent_layer_pack_verify`, `latent_module_verify` and
`latent_cpu_verify` targets are all inside `STRATA_ENABLE_CUDA` in
`edge0-strata/CMakeLists.txt` lines 2573–2614; despite its name,
`latent_cpu_verify` links `strata_engine` and `CUDA::cudart`.
`ple_fp8_parity` is also CUDA-guarded at lines 2691–2700. Relevant CPU arithmetic
parity tools are x86-guarded at lines 2065–2127. The inspected CMake file is
209,191 bytes, SHA-256
`9ff7a9e06f1aa76ca227b3f64a3ec882f905bbe16a8e9ad7afa8d834ceb91e9a`,
at Edge0 revision `b7900cbb322926ccebdcd0c0c5a1280497ecff22`.

Those full verification targets were not built or run on this arm64 Mac. The
independent CPU-only pinned GGML decoder/quantizer bridge was built and used
for the component reference; it does not replace a full Strata verifier.

### Native component and regression gates

The production latent forward passed the preregistered private fixture on CPU
and the existing Safe Metal packed kernel. Each backend ran 60 independent
matrix comparisons and exact source routing checks. The maximum matrix error
was 0.0190752 of its fixed bound on CPU and 0.00472723 on Metal. Final readout
maximum absolute / relative L2 errors were `1.33304622e-6 / 7.93635726e-7`
on CPU and `3.60860715e-7 / 2.24968696e-7` on Metal, below the fixed
`0.002 / 0.0002` limits. The aggregate report SHA-256 is
`ab83e23fd2ce141ab77b1a95da1c3ac11eab86b7e74d1d8c0d14b1865162ada4`.
This tests quantized latent/shared/readout arithmetic, not a full external
trunk or mapped-owner numerical reference.

Additional native tests cover byte-preserving aliases and concatenation,
complete skeleton name/type/shape inventory, prepared metadata mutation/link/
path/count rejection, source routing ties, tiled-head negative-A recurrence
against an independent grouped F64 two-step reference, and unsigned PLE
semantics. The PLE negative-hash counterexample lies outside the pinned valid
token/multiplier range; it is not an observed pinned-artifact failure.

- All focused Edge0 tests passed: 24 passed, 3 deliberately ignored.
- The ignored actual-adapter test passed separately, validating the pinned
  artifact's five mappings, aliases, virtual table/integer fields and span
  boundaries. It reads small metadata and headers, not all payload bytes.
- The full CPU/Metal engine suite passed: 1,054 passed, 155 ignored, 278.56 s;
  C++/profile integration tests and documentation tests also passed.
- Rust 1.98.1 production Metal-library Clippy passed with `-D warnings`.
  Its first pass rejected the exact decimal table-scale spelling; the final
  code uses the identical F32 bits `0x39510000`. Focused Edge0 tests passed
  again after that representation-only change. Existing CPU-reference
  coefficient lint exceptions are outside this production-Clippy scope.
- All 21 Python helper/counterexample tests, Ruff, repository policy,
  whitespace and all six crates' formatting checks passed.

### First bounded checkpoint smoke

The sealed generation-only Rust 1.98.1 release ran on Apple M5 Max, 128 GiB
physical memory. It advertised vocabulary 248320, diagnostic context maximum
512, fresh-per-request state, prefix reuse unsupported and MTP unsupported.
The native template rendered one ordinary synthetic greeting request in memory.
The fixed output ceiling was 32; greedy and stochastic settings (seed 42,
temperature 0.7, top-k 40, top-p 0.9) each produced a nonempty two-token
completion followed by a source EOG token. Both reached EOS after three draws.
This run exercised EOS termination; it did not force length termination.

The same completion hashes matched after an intervening stochastic request,
request-error rejection/recovery and a new worker process under the same
settings. The error case rejects an invalid token ID at the request boundary;
it does not simulate a failure midway through a native layer. Every completion
passed the worker's finite-readout boundary. All workers were serialized and
reaped. No prompt, response, token vector or probability was persisted.

| Source/build/run identity | SHA-256 |
| --- | --- |
| Native source manifest (992 source/build-input files) | `ed58d8848256921eaaf427b4f1c6a47671c5ea25b63586b30c1cd1060f7dff5e` |
| Generation executable | `e3fa32d85c626fb949d086734819bf5747390b521b3d8d527ec213daa7301a50` |
| Build receipt | `ad5dd170490cabe7e76fbb5e2a5bc926ba620f5e11bedd68b6a54535dbc495ef` |
| Checkpoint smoke registration | `88e6d65ed49408bd23a4871e98f9ba67ee288336ce4081871532024c5db6bdec` |
| Checkpoint smoke aggregate report | `cc72b578466ca9e7ead665aeb739b9e53dcd4e8f871ac6776fcc66c95f0fc472` |

The reviewed static scalar workspace envelope is 6,553,600 bytes for the pinned
geometry. Cached D groups use 57,344 bytes; routing, expert temporaries and
single-command staging fit within that envelope, while recurrent/attention/PLE
owners are charged separately. This source review found no additional latent
allocation undercount. Phase 4 still needs complete residency/loaded-memory
qualification for its changed execution policy.

## Performance and memory

The smoke's sampled peak worker RSS was 571,424,768 bytes, below its 48 GiB
watchdog. RSS does not measure all GPU-backed/mapped residency and is not a
whole-model memory-footprint result. The observed short EOS completion does
not establish a long-decode rate. No matched speed measurement was run.
Production speed, mixed precision, resident banks, MTP defaults and prefix reuse
remain later gates.

## Evidence, decision and limits

Phase 3 ends only with the declared reference fixture and bounded real-artifact
chat/state smoke. Missing full-model external reference, upstream activation
differences and the unavailable sealed scoring replay remain explicit limits.
The recorded component and short ordinary-chat gates passed. Full-model
external numerical parity, broad quality, long generation, native mid-layer
failure recovery, full context and total resident-memory qualification remain
unverified. The scoped reference implementation does not complete model
onboarding or enable the app catalogue.

### Phase 3 plan-conformance review

Verdict: **Conforms** for the Phase 3 diagnostic component/ordinary-chat gate,
reviewing baseline `49d6ed10` plus this working tree. Governing sources are
[Phase 3](../EDGE0_PLAN.md#phase-3-reference-forward-in-the-f32-graph), this
preregistered attempt, repository rules and the applicable
[onboarding invariants](../MODEL_ONBOARDING.md). The owner's authorization
remains local non-commercial branch evaluation; it does not authorize product
admission or commercial model services.

| Applicable requirement | Implementation / observed evidence | Status |
| --- | --- | --- |
| Explicit geometry/byte-preserving adapter and bounded selected weights | Exact inventories/metadata, alias and mutation tests; actual adapter admission and worker load | Conforms |
| D/s once, per-expert U, routed-order reduction and full-width shared expert | Production latent path; sealed CPU/Metal F64 component comparisons | Conforms |
| Preregistered synthetic reference logits and fixed thresholds | Three cases, 60 matrix operations per backend; errors below fixed bounds | Conforms |
| Conditional second-oracle outcome | Pinned CMake guards establish the arm64 CPU-only verifier blocker; non-execution recorded above | Conforms |
| Activation-policy controls on fixture and one real layer | Separate registered measurements and numerical-scope limits | Conforms |
| Ordinary-chat greedy/stochastic, reset and restart | Sealed worker smoke with matching successful completion hashes; request-error recovery | Conforms |
| Prefix/MTP unavailable, image N/A and private evidence | Complete reference-state inventory, readiness assertions, retained source head files and aggregate-only run records | Conforms |

Full-trunk external numerical parity is unverified and is not inferred from the
component fixture. Mixed execution, MTP, prefix restoration, tools/catalogue and
performance/quality qualification are future phases, not completed by this gate.

## Follow-up history

No follow-up yet.
