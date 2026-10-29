# Native GLM-5.3-Flash, Qwen3.8-Flash-Next and Flint

Stark has a separate native Rust/Metal forward path for `glm5_next`,
`qwen4_exp` and `flint`. It reads the original quantized GGUF or MLX Safetensors storage.
There is no Sushi, llama.cpp, MLX, Python or Transformers inference process in
this execution path. Python is used to prepare download metadata and by the
optional reference validators.

## Checkpoints and preparation

The preparation command pins these sources:

| Choice | Checkpoint | Weight files |
| --- | --- | --- |
| `glm53-q2` | `unsloth/GLM-5.3-Flash-GGUF`, `UD-Q2_K_XL`, revision `a38483c8cd5df544f53d70fb281afe97369d5ab6` | 108,720,071,552 bytes, 101.25 GiB |
| `qwen38-mixed` | `pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit`, revision `b2c422f3c643e36f04227a64d61796b44a4b1029` | 106,195,031,237 bytes, 98.90 GiB |
| `qwen38-lily-q4-xl` | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83` | 106,905,155,192 bytes, 99.56 GiB |
| `flint-int4` | `RecursiveSI/flint-int4`, revision `3b93f9053b530488405ae9fca51d4900d6ce21f0` | 22,568,912,600 bytes, 21.02 GiB |

See [Recursive Flint](FLINT.md) for its architecture, installation command,
precision policy and validation. The GLM/Qwen-specific execution details below
retain their own scope.

GLM uses the configuration and tokenizer from `zai-org/GLM-5.3-Flash` at
`04c4e9e95c5da8862dced7e5056455116f83a7e0`. Its four GGUF shards retain their
quantized bytes. Qwen retains its packed 4-bit experts and n-gram embeddings,
8-bit matrices and floating control weights, including the exact integer hash
buffers. The Qwen export has its centered normalization offsets baked into the
stored weights. The official FP8 checkpoint does not satisfy this export
contract.

Run from the source checkout:

```sh
uv run --project tools/converter --locked --python 3.13.13 python -m stark_convert.hybrid_prepare glm53-q2 --output models/glm53-q2
uv run --project tools/converter --locked --python 3.13.13 python -m stark_convert.hybrid_prepare qwen38-mixed --output models/qwen38-mixed
uv run --project tools/converter --locked --python 3.13.13 python -m stark_convert.hybrid_prepare qwen38-lily-q4-xl --output models/qwen38-lily-q4-xl
cargo +1.98.1 build --release --locked -p stark-generate
cargo +1.98.1 build --release --locked -p stark-score --features optimized
./scripts/stark chat --model-path models/glm53-q2
./scripts/stark chat --model-path models/qwen38-mixed
```

Each output directory must be new. Interrupted downloads remain under the
adjacent `.native-download` directory and can be resumed with the same command.
`--offline` restricts preparation to existing Hugging Face downloads. Publication
uses an atomic rename that refuses an existing output directory.

Some Safetensors exports place compressed matrices at unaligned addresses.
Create a separate aligned artifact to enable the wider Metal loads:

```sh
uv run --project tools/converter --locked --python 3.13.13 python scripts/align_native_safetensors.py --model models/qwen38-mixed --output models/qwen38-mixed-aligned
```

This copies the checkpoint and pads JSON headers. Tensor offsets within the
payload, tensor bytes, quantization, configuration and tokenizer remain the
same. The tool verifies payload hashes, preserves the original source receipt,
and writes a separate alignment receipt. It needs space for another checkpoint.
Import the aligned folder or pass it to the worker as the model path. Its
preparation provenance does not itself qualify native execution or answer quality.

Preparation downloads metadata and weights without repository Python or draft
assistants. It adds the source chat template and the union of checkpoint EOS
declarations to the configuration, preserving the architecture. A
`native-source.json` receipt records the pinned source identities, configuration
hash and stored weight bytes. The receipt reports native loading and execution
as unverified. Source identity in the library describes preparation provenance;
it is not a weight-content qualification receipt.

The prepared folder can be imported through **Manage Models → Add from Source →
Local Folder** in an app build containing the new workers. Chat uses the existing
generation worker. These downloads are prepared with the command above; the family catalog continues to describe
the ordinary converter paths. An incomplete prepared folder is rejected before
registration. The native loader remains responsible for tensor admission.

## Execution and limits

GLM implements all trunk layers, including channel convolutions, Kimi Delta
Attention, compressed MLA, the pooled sparse indexer, mHC Sinkhorn mixing, and
routed/shared MoE. The GGUF reader validates metadata against the source
configuration and checks the transformed negative decay coefficients used by
GGML. The included MTP layer is not executed.

Qwen implements Gated DeltaNet, causal block-sparse QSA, gated residual streams,
routed and gated shared experts, and the n-gram PLE embedding and convolution.
Hash buffers retain their integer precision; EOS boundaries segment the n-gram
history. A Qwen3.5 adapter or model-name alias is not involved.

Matrix products execute on Metal directly from packed weights. GLM's dynamic
quantization includes Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS,
IQ3_XXS and IQ4_XS. Floating tensors support F32, F16 and BF16 storage. MLX supports
affine 4/8-bit matrices with group sizes 32, 64 or 128. Metal reads owned,
page-aligned views of the immutable weight mappings, avoiding repeated compressed
weight uploads. Selected expert products remain bounded to 256 MiB of logical
selected storage per projection. A bank view can cover additional immutable
experts without expanding or copying those weights.
The whole model is never expanded into BF16 or F32 weights. Mapping ownership
lasts through GPU completion, and completed temporary resources are released.
Static projections use a model-local FIFO cache of read-only mapped weight views,
limited to 8 GiB of page-rounded registrations. A separate registry retains full
rank-three weight-bank views only when the entire deduplicated inventory fits
within 75% of Metal's recommended working-set size minus the 8 GiB static
allowance. Every bank must
also fit the device's maximum buffer length and the kernels' 32-bit view
offsets. This admits the current Qwen expert banks while leaving its large
n-gram tables CPU-mapped. Larger inventories use transient selected views.
The registries count page-rounded resource spans, not physical resident pages.
Retained no-copy buffers alone do not wire memory.
Active command ranges have their separate bounds and may outlive an eviction
until completion. Dropping the loaded model releases its retained weight views.
This cache contains immutable weights; request state and score-cache rules below
are unchanged.

Mixed-precision inference is permitted. Activation, accumulation, recurrent-state
and weight precision are separate choices, selected through numerical validation
and measured performance. Existing F32 results remain source-bound reference
evidence; a new precision path is checked against a matching reference and gets
its own validation record. F32 execution is not required for production.

The Lily-format Qwen checkpoint selects an in-process native Metal 4 backend
extracted from Lily revision `e22d58fa7cc488493fb1b5d810a198bd29730aeb`.
Selection requires both `model_type=qwen4_exp` and
`lily.format=qwen4_exp-affine-v1`; other Qwen exports retain the F32 graph below.
This backend uses BF16 activations, F32 reductions, logits and GDN recurrent
state, and explicitly selected Q8 attention K/V caches. The q4-xl weights use
Q4 GDN projections and Q8 attention, shared-expert, readout and embedding groups.
Experts remain resident; the large n-gram embedding table is mapped read-only
and paged. Batched prefill processes the current request, and greedy decoding
reads back one GPU-selected token per step. Scoring and stochastic generation
read the complete F32 logits and use Stark's existing scoring and sampling.
Greedy and sampled requests overlap future CPU command encoding with the current
GPU step. Sampled requests retain full F32 readout and Stark's CPU sampler.
The future pass remains unsubmitted until the actual selected input token is
validated and staged. Early stops drain at most one submitted ordinary step
before state reset; cleanup never invokes the sampler or consumes randomness.
Completion feedback errors fence the request before a successful result can be
returned. Top-k up to 64 uses a sorted prefix while scanning contiguous logits,
falling back to partition selection after 1,024 replacements. Larger bounded
top-k uses partition selection. Full-vocabulary sampling uses three stable
11-bit radix passes over packed F32 keys and IDs. These preserve the existing
F32 total order, ascending-ID ties, serial F64 filtering and SplitMix64 draws.
HC decode computes the scaled BF16 activation once in the down kernel and
reuses it in an up/mix kernel retaining the original single-row dot order.
For Q4 GDN input projections with four convolution taps and an even output
width, projection and convolution share one dispatch, preserving the BF16
projection boundary, tap order and window updates. Other layouts retain
separate dispatches. The recurrent kernel keeps its BF16-rounded output local
before the F32 norm reduction. The [kernel study](benchmarks/QWEN38_KERNELS_2026_10_29.md)
records matching references, rejected trials and sampled benchmarks.
Every request resets logical state. GPU execution failures require model reload.
Hybrid generation and scoring hold a macOS user-initiated, latency-critical
process activity through request execution and GPU cleanup, then release it
between requests. This expresses the request's activity lifetime; it does not
reserve GPU resources or establish a fixed clock or throughput.
This path requires macOS 26 and Metal 4, retains Stark's 16,384-token limit,
and executes neither MTP nor vision. Source modifications and licenses are
recorded in [the vendored core provenance](../vendor/lily-core/UPSTREAM.md).

The existing native graph uses F32 forward arithmetic. Deferred Metal commands keep
dependent device vectors alive through completion and use a sticky failure
status to reject invalid intermediates, including values later hidden by
clipping. Existing CPU controls remain available in the staged execution path.
Temporary device vectors share disjoint 256-byte-aligned command arenas, capped
at 512 MiB per command. Persistent recurrent and attention buffers have separate
ownership and are never reused as command scratch.
Each token advance scopes Metal temporary objects to an autorelease pool,
including command transitions and error cleanup.
Vector kernels decode multiple packed values together and distribute output rows
across SIMD groups; their reduction schedule differs from the scalar oracle and
is checked against the existing numerical gates. GPU RMS uses an F32 SIMD
reduction, qualified with wide F64 oracles and the unchanged source-model gates.
Its sum order differs from the staged scalar path; it does not narrow activations.
Affine matrix products combine reduction and nonfinite checks in the vector
kernel. A typed unsigned-word load path is selected only when every mapped code,
scale and bias plane meets its alignment requirements; other layouts use the
checked byte/vector path. Both preserve the packed affine decoding and F32
reduction contract.

Qwen's graph keeps hyperconnection normalization and mixing, GDN, QSA caches and
attention, sparse routing, selected expert products, PLE control and convolution,
residual injection and readout on Metal. Stable routing uses descending F32 total
order and ascending expert indices for ties. Rust retains token history and the
exact I64 n-gram hashes, gathers all PLE embeddings before submitting device
work, and supplies the token embedding. QSA storage grows with the visible
prefix, copying valid cache rows on the device when capacity increases. The
graph is enabled only when the full expert inventory meets the bank-admission
and indexed-shape checks and the model selects at most 64 experts. Other
inventories use the staged path with bounded selected mappings. A graph error
fails the request and clears state.

Independent QSA, GDN, PLE and MoE projections are encoded in dependency levels.
Each level uses concurrent dispatch for independent projections, then a buffer
barrier makes their results visible before dependent work. Affine vector
reductions report nonfinite results through the sticky graph failure status;
misaligned affine planes fall back from typed unsigned-word loads to the checked
vector path. QSA preparation parallelizes query, key and value normalization,
rotation and cache writes. Completed pooled index keys are normalized and
rotated once, retained in the attention cache, and reused for later query scores.
Sparse block selection uses an exact 64-bit radix threshold key: its score word
preserves F32 `total_cmp` order and its ID word preserves ascending block IDs on
ties. PLE normalization and QSA query/key normalization use F32 SIMD reductions,
qualified against the existing source gates. Router selection is parallel;
its selected-score softmax retains the established scalar F32 order.
At prefixes of at least 256 tokens, QSA attention computes partial softmax/value
results over 256-token splits and combines them with the output gate. Smaller
prefixes and geometries exceeding the bounded workspace retain the scalar
attention path. The split reductions use F32 and are checked against F64
references and the source-model gates.
GDN's fused recurrent kernel uses a two-pass blocked state update, retaining
ordered F32 prediction and output accumulations while reducing recurrent-state
traffic. The staged recurrent kernel remains available as a diagnostic path.

GLM's graph keeps mHC projection and Sinkhorn mixing, normalization, KDA, pooled
index selection, absorbed MLA attention, shared and selected expert products,
residual streams and readout on Metal. Router logits cross a completion boundary
for CPU sigmoid, correction-biased selection and mixture weights; Rust then
registers the selected immutable expert slices before continuing the graph. This
preserves bounded admission for the current GLM inventory. The decoded mHC
control inventory remains capped at 8 million values per projection and 256 MiB
in total. The graph uploads those controls once and releases the production CPU
copy; larger layouts keep staged execution, whose bounded control cache uses
CPU NEON arithmetic on Apple silicon.

For the pinned original-format checkpoints, the code schedules one command completion
per token for Qwen and 43 for GLM: one for each of its 42 routed layers and one
final completion. Three four-step release profiles confirm these counts;
the counters and throughput measurements are recorded in the
[decode graph benchmark](benchmarks/GLM_QWEN_DECODE_GRAPH_2026_10_28.md).
Test builds can add layer readbacks for numerical diagnostics.

Set `STARK_QWEN38_DECODE_GRAPH=0` or `STARK_GLM53_DECODE_GRAPH=0` before loading a
model to select its staged reference path. The execution choice remains fixed
for that loaded model. `STARK_HYBRID_CONCURRENT_FFN=0` selects serial staged FFN
dispatch and serial GLM graph expert batches. The graph otherwise joins serial
dependent stages with concurrent expert encoders and explicit buffer barriers
inside the same command. `STARK_HYBRID_ORDERED_RMS=1` selects the slower ordered
GPU RMS diagnostic; this choice is captured once when the process initializes
the native Metal executor. Both paths clear request state on failure and reset.

The following opt-in environment flags select diagnostic variants before the
worker loads a model:

| Flag | Effect |
| --- | --- |
| `STARK_QWEN38_SERIAL_LEVELS=1` | Serializes otherwise independent Qwen projection levels. |
| `STARK_HYBRID_BYTE_LOADS=1` | Disables aligned typed unsigned-word affine loads and uses the checked vector-load path. |
| `STARK_HYBRID_VECTOR8=1` | Uses eight values per lane instead of the aligned 32-value affine schedule. |
| `STARK_HYBRID_SCALAR_UINT_LOADS=1` | Uses scalar unsigned-word loads instead of aligned `uint4` loads. |
| `STARK_HYBRID_SEPARATE_FINITE=1` | Uses separate finite-value scans instead of fused affine reduction checks. |
| `STARK_HYBRID_FACTORED_AFFINE=1` | Experimental group-factored affine reduction; disabled by default and separately tested against F64 controls. |
| `STARK_QWEN38_SINGLE_ENCODER=1` | Experimental single concurrent compute encoder with explicit dependency barriers. |
| `STARK_HYBRID_SCALAR_SHORT_MATVEC=1` | Uses the generic floating matrix schedule for short-column projections. |
| `STARK_QWEN38_SCALAR_QSA_PREPARE=1` | Selects scalar QSA query/key/value preparation. |
| `STARK_QWEN38_SCALAR_QSA_SELECTION=1` | Selects the scalar QSA visibility and top-k path. |
| `STARK_QWEN38_SCALAR_ATTENTION=1` | Selects scalar QSA attention at all prefix lengths. |
| `STARK_QWEN38_SCALAR_PLE=1` | Selects scalar PLE projection activation. |
| `STARK_QWEN38_SCALAR_ROUTER=1` | Selects scalar MoE router top-k. |
| `STARK_QWEN38_STAGED_GDN=1` | Selects the staged recurrent GDN kernel. |
| `STARK_QWEN38_PHASE_TIMING=1` | Adds per-phase GPU timings by ending, committing and waiting on commands between phases; synchronization changes execution and these timings are not throughput measurements. |
| `STARK_QWEN38_ENCODER_TIMING=1` | Samples GPU timestamps at encoder boundaries while retaining one command completion per token; diagnostic overhead still affects throughput. |

These switches support numerical and kernel diagnostics; benchmark comparisons
must record their values and use the production defaults unless the diagnostic
variant itself is under study.

For Lily-format BF16 stage diagnostics, set `STARK_QWEN38_PROFILE_OUTPUT` to a
new JSONL file before launching the generator. The file is created exclusively
with mode `0600`; existing files and symlinks are rejected. Records contain
static CPU/GPU categories and aggregate counts, durations and byte totals.
Precise GPU timestamp writes retain one command buffer per decode step and can
affect scheduling. Measure production throughput in a separate run with this
setting absent. For example, after building the worker:

```sh
mkdir -p runtime/qwen38-profile
chmod 700 runtime/qwen38-profile
STARK_QWEN38_PROFILE_OUTPUT="$PWD/runtime/qwen38-profile/stages.jsonl" uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_hybrid_context.py --model models/qwen38-lily-q4-xl --worker target/release/stark-generate --corpus vendor/lily-core/src/qwen4exp/model.rs --input-tokens 4096 --tokens 64 --replays 3 --sampling stochastic --temperature 1 --top-k 20 --top-p 0.95 --seed 42 --output runtime/qwen38-profile/run.json
```

The context benchmark supports greedy and stochastic generation; `--top-k 0`
omits top-k filtering. It emits counts, hashes and timing aggregates. The
benchmark study records the separate pinned corpus used for its exact workload.
Pass `--host-metrics` to attach optional CPU, GPU-client, VM and thermal
aggregates for each replay. Only monitoring intervals completely inside a
request contribute; coverage, boundary exclusions, source availability and
counter discontinuities are reported. Process names and IDs remain in memory;
reports contain fixed buckets. GPU-client counter units are unverified and
their raw deltas do not measure GPU clocks or prove exclusive access. This
monitor adds host work, so record its use in comparisons.

The F32 graph's prefill processes tokens sequentially and projects vocabulary logits only for
the final input token. Performance must be measured on the exact checkpoint and
host. Published Sushi or MLX throughput does not qualify Stark's throughput.
The earlier [GPU stage and kernel benchmark](benchmarks/GLM_QWEN_NATIVE_GRAPH_2026_10_28.md)
records the execution path before the deferred graph.
The [Qwen mixed-precision benchmark](benchmarks/QWEN38_MIXED_PRECISION_2026_10_28.md)
records the later F32 improvements, Lily-format qualification and 4K throughput.
The [sampled-generation study](benchmarks/QWEN38_SAMPLED_BF16_2026_10_28.md)
records the subsequent sampler, scheduling and BF16 profiling work at fixed
quantization.
The [desktop stability study](benchmarks/QWEN38_STABILITY_2026_10_29.md) records
later exact selector improvements, scheduling controls and concurrent GPU work.
The [AC power-mode control](benchmarks/QWEN38_POWER_MODES_2026_10_29.md) records
the subsequent temporary High Power test and restoration. Stark does not change
the host's power preferences.
The [Qwen research review](archive/runs/2026-10-28-qwen38-performance-literature-review.md)
and [GLM research review](archive/runs/2026-10-28-glm53-performance-literature-review.md)
record the remaining graph, kernel and prefill opportunities with source and
license boundaries.

These paths are text-only. The native loader does not execute vision, video,
audio, MTP or DFlash. The workers retain their 16,384-token context and
4,096-output-token caps; the upstream advertised maximum context is not a Stark
memory qualification. A weight set fitting within 128 GiB does not establish
available memory for every context or concurrent application.

Generation uses fresh state for every request. Scoring serializes the complete
token sequence, clears recurrent state after success or failure, and retains
only the existing optional cache of identical immediately consecutive inputs
with the same context size. No cache persists across worker restarts. Generic
candidate scoring does not establish decision calibration. The app's Decision
mode additionally requires a direct-answer template. GLM's official template
always opens a reasoning block and is rejected by that existing boundary check;
its boolean thinking control does not disable the upstream reasoning template.
Raw token scoring remains available through `stark-score`. Hybrid scoring's
prefill metric includes its vocabulary projection; its separate readout duration
is zero because that timing is not split out.

Raw original-format directories can also be passed directly to `stark-generate
--model-path` or `stark-score --score-model-path` when they contain the required
configuration, tokenizer and complete shards. The native loader honors EOS IDs
from both source and generation metadata, plus a declared tokenizer EOS token.
The prepared chat template is needed for the terminal client and desktop import.
Native admission errors use fixed worker error codes without source payloads.

## Validation

Operator tests compare every GLM packed block type with hashes from the pinned
GGML C decoder. Physical Metal tests compare packed, mixed affine, batched head
and nonconsecutive expert products with scalar references. These tests require
no production model or network:

```sh
cargo +nightly-2026-10-01 test --locked -p stark-engine --features cpu,metal hybrid:: -- --test-threads=1
```

The model-bound fixture generators use pinned upstream forward implementations
and write synthetic weights, inputs and reference logits only under an ignored
directory. They emit source identities and aggregate information. The Rust
example reports aggregate errors, output hashes, greedy agreement and state
reset checks, without printing inputs or logits:

```sh
uv run --project tools --locked python scripts/glm53_native_reference.py --help
uv run --project tools/stock_benchmark/mlx_current --locked python scripts/qwen38_native_fixture.py --output runtime/qwen38-reference
cargo +1.98.1 run --release --locked -p stark-engine --no-default-features --features cpu,metal --example hybrid_parity -- runtime/qwen38-reference
```

The reference checks exercise complete small-model forward paths, including
mixed quantization, grouped heads, recurrent decode and sparse attention beyond
the selection budget. Tied sparse scores use deterministic index ordering in
Stark; an upstream top-k implementation may choose a different valid tied set.
Oracle fixtures must avoid ambiguous cutoff ties when asserting numerical
parity. These checks establish implementation parity under the recorded F32
execution policy. They do not replace full-checkpoint execution, BF16 runtime
comparison, quality benchmarks or host memory measurements. Dated results belong
under `docs/archive/runs/`. The
[full-checkpoint validation record](archive/runs/2026-10-28-native-hybrid-full-checkpoints.md)
records the production smoke scope and the GLM format correction it exposed.

## Provenance

GGUF block decoding, lookup tables, names and transformed GLM layout follow
`ggml-org/llama.cpp` at `8ca4060b7e30e76baaa5670ee1ec1b760b7f8552` (MIT), with
its license preserved in `LICENSE.ggml-hybrid`. Qwen's forward contract follows
the pinned export's `qwen4_exp.py` and the PipeNetwork implementation at
`0352280b927a731fd35276ac622ef2f28da5c936` (Apache-2.0), preserved in
`LICENSE.qwen38-reference`. The GLM validator uses Transformers at
`95bafdcccfffe17ec383999fc39273867b11385e` and verifies the modeling source hash
before loading its forward modules. See [upstream provenance](../provenance/UPSTREAM.md).
