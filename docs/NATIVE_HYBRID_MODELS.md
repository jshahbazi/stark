# Native hybrid models and Qwen3.8 conversion

GLM-5.3-Flash and Recursive Flint use Stark's native hybrid Rust/Metal paths.
Qwen3.8-Flash-Next requires a typed Uzu conversion and executes through the
standard `Engine::load_language_model`/`LanguageModel` API. Its full decoder
implementation and current limits are documented in
[Qwen3.8 standard Uzu execution](QWEN38_UZU_EXECUTION.md). Numerical and
performance qualification of that implementation is pending.

No model path launches a Python, Transformers, MLX, llama.cpp or Sushi inference
process. Python handles preparation/conversion and optional reference checks.

## Pinned sources

| Choice / source | Pinned checkpoint | Source weight bytes |
| --- | --- | ---: |
| `glm53-q2` | `unsloth/GLM-5.3-Flash-GGUF`, `UD-Q2_K_XL`, revision `a38483c8cd5df544f53d70fb281afe97369d5ab6` | 108,720,071,552 |
| Qwen3.8 supported conversion layout | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83` | 106,905,155,192 |
| Qwen3.8 historical mixed layout | `pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit`, revision `b2c422f3c643e36f04227a64d61796b44a4b1029` | 106,195,031,237 |
| `flint-int4` | `RecursiveSI/flint-int4`, revision `3b93f9053b530488405ae9fca51d4900d6ce21f0` | 22,568,912,600 |

The PipeNetwork mixed packing is not accepted by the current Qwen Uzu converter.
Its historical preparation or F32 execution results do not make it a current
production input. Native `model_type=qwen4_exp` folders, including previously
prepared downloads and aligned copies, require a supported conversion.
The official FP8 checkpoint also falls outside the implemented conversion
contract.

GLM uses configuration/tokenizer metadata from `zai-org/GLM-5.3-Flash` at
`04c4e9e95c5da8862dced7e5056455116f83a7e0`. Its four GGUF shards preserve their
quantized bytes. See [Flint](FLINT.md) for its setup, precision and qualification.

## Prepare and load

Run from the source checkout:

```sh
uv run --project tools/converter --locked --python 3.13.13 python -m stark_convert.hybrid_prepare glm53-q2 --output models/glm53-q2
uv run --project tools/converter --locked --python 3.13.13 python -m stark_convert.hybrid_prepare flint-int4 --output models/flint-int4
./scripts/stark convert fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl --revision 0f94842e3682b49acace4daeb4cf2e65a6f94f83 --metadata-profile hf-generic --architecture qwen4_exp --dtype bfloat16 --chunk-mib 64 --output models/qwen38-uzu
cargo +1.98.1 build --release --locked -p stark-generate
cargo +1.98.1 build --release --locked -p stark-score --features optimized
./scripts/stark chat --model-path models/glm53-q2
./scripts/stark chat --model-path models/qwen38-uzu
```

Each output directory must be new. For an already downloaded compatible Qwen
source, replace the repository in the conversion command with its local folder
and add `--offline`. The converter emits one canonical `model.safetensors`,
a tokenizer, conversion provenance and typed `Qwen38LanguageModelConfig`.
It preserves weight quantization and omits vision/MTP tensors. Packed codes,
group-major affine parameters, direct controls and optional integer hash
controls are checked before publication. See the
[converter guide](../tools/converter/README.md).

Native GLM/Flint preparation adds source chat/EOS metadata and writes a
`native-source.json` provenance receipt. Interrupted downloads can resume from
the adjacent `.native-download` directory. `--offline` uses existing downloads.
Publication uses an atomic rename that refuses an existing output folder.
Preparation receipts do not establish inference quality or throughput.

Raw supported GLM/Flint directories can also be passed directly to
`stark-generate --model-path` or `stark-score --score-model-path`. Native hybrid
scoring includes the final vocabulary projection in `forward_seconds` and
reports `readout_seconds=0`; converted Uzu scoring records its readback
separately.

GLM retains opt-in diagnostics selected before model loading:

| Flag | Effect |
| --- | --- |
| `STARK_GLM53_DECODE_GRAPH=0` | Select the staged GLM reference path. |
| `STARK_HYBRID_CONCURRENT_FFN=0` | Serialize staged FFN dispatch and GLM graph expert batches. |
| `STARK_HYBRID_ORDERED_RMS=1` | Select ordered GPU RMS reductions for diagnostics. |

These settings require their own numerical and performance controls.

Generation and scoring select converted Qwen through the normal Uzu loader.
The former Lily runtime/dependency and production F32 Qwen constructors are
removed. Historical scalar/Metal Qwen controls compile only in private tests.
A source-format marker can remain in provenance; it does not select execution.

## Qwen3.8 execution and limits

The full 48-layer target graph includes HC, GDN, compressed-indexer QSA,
routed/shared experts and hashed paged PLE. Standard Uzu buffers, matrix
dispatch, commands and state acceptance own the complete path. Every
projection, including selected rows of full expert banks, uses the common
matrix dispatcher. Canonical group-major affine planes remain in that layout.

The precision policy is BF16 activations, F32 reductions/router/readout/GDN
state, and Q8 attention caches with F16 scales. HC now explicitly rounds its
normalized activation to BF16 before projection. Sampling uses standard
backend F32 sampling and its `PRng`, so old SplitMix64/CPU-sampler token hashes
do not qualify the new path. Matching forward readouts, sampled repeats and
performance measurements are separate pending controls.

Current execution is text-only and non-speculative, with an implemented
65,536-token context cap whose runtime qualification is pending. Other families
retain their 16,384-token public worker limits. Qwen prefills layer by layer in
chunks of at most 16 tokens. Projections
batch rows while recurrence and cache updates retain causal token order.
`norm_topk_prob=true` is required. Whole flat suffixes must be accepted in
order. Partial encoding or execution failures invalidate request state.
Paged n-gram reads use known host token IDs after GPU completion; unfinished
GPU token buffers are never read by the CPU. See the
[complete execution contract](QWEN38_UZU_EXECUTION.md).

The [standard stage profiler](QWEN38_STAGE_PROFILING.md) provides bounded,
opt-in timing attribution for the current path. Historical Lily/native-F32
diagnostic environment variables do not select production Qwen execution.
Existing dated studies retain
their source/build/executor scope:
[storage conversion](benchmarks/QWEN38_UZU_FORMAT_2026_10_29.md),
[sampled BF16](benchmarks/QWEN38_SAMPLED_BF16_2026_10_28.md),
[desktop stability](benchmarks/QWEN38_STABILITY_2026_10_29.md), and
[2026-10-30 speed experiments](benchmarks/QWEN38_UZU_SPEED_2026_10_30.md).
Their original commands require their recorded historical checkout; the
former extraction is available at
[revision 5a082f7](https://github.com/jshahbazi/stark/tree/5a082f7f2e98019ef8907e8c576b52a5c0c669af/vendor/lily-core).
New qualification and the independent final audit follow the
[execution plan](QWEN38_UZU_EXECUTION_PLAN.md).

## GLM/Flint native execution

GLM implements all trunk layers: channel convolution, Kimi Delta Attention,
compressed MLA, pooled sparse indexing, mHC Sinkhorn mixing and routed/shared
MoE. Its GGUF reader checks source metadata and transformed negative decay
coefficients. Included MTP tensors are not executed.

Packed products execute directly on Metal. GLM supports Q8_0, Q2_K, Q3_K,
Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS and IQ4_XS. Shared native affine
code also supports MLX 4/8-bit storage. Full weight banks are not expanded into
floating-point models. Immutable mapped views retain ownership through GPU
completion. Static mappings have an 8 GiB page-rounded FIFO allowance;
complete admitted rank-three banks have a separate working-set bound. Those
registration limits do not establish physical residency or pin pages.

GLM's graph retains mHC, normalization, KDA, index selection, absorbed MLA,
shared/selected expert products, residual streams and readout on Metal.
Router logits cross a completion boundary for CPU selection and mixture
weights, followed by registration of selected expert slices. This keeps
existing bounded admission. Decoded mHC controls are capped at 8 million
values per projection and 256 MiB total; larger layouts use staged execution.

`STARK_GLM53_DECODE_GRAPH=0` selects the existing staged reference path.
`STARK_HYBRID_CONCURRENT_FFN=0` serializes staged FFN and GLM graph expert
batches. Existing shared hybrid matrix diagnostic flags retain their numerical
scope. They do not select an alternate production Qwen executor.

The historical GLM graph schedules 43 completion boundaries per token: one
for each routed layer plus final completion. Recorded counters and throughput
belong to the [decode graph study](benchmarks/GLM_QWEN_DECODE_GRAPH_2026_10_28.md).
Published rates from other engines do not qualify current Stark throughput.

## Requests and validation

Generation starts fresh state for every request. Scoring serializes complete
decisions, clears recurrent state after success/failure, and optionally caches
only immediately consecutive identical token sequences with the same context
size. The cache never persists across restarts. Candidate scoring does not
establish decision calibration. GLM's reasoning template fails the app's
direct-answer Decision boundary; raw token scoring remains available.

Workers retain text/context/output limits independently of an upstream
advertised context maximum. Native admission failures use fixed error codes
without logging source payloads. A weight set fitting memory does not
establish capacity for every context or concurrent application.

Generic packed-format controls and physical Metal tests remain available:

```sh
cargo +1.98.1 test --locked -p stark-engine --no-default-features --features cpu,metal hybrid:: -- --test-threads=1
```

Existing source-bound results remain in
[full-checkpoint validation](archive/runs/2026-10-28-native-hybrid-full-checkpoints.md)
and dated benchmarks. Qwen standard-backend integration and numerical
qualification are tracked separately; this page claims no newly measured gain.

## Provenance

GGUF decoding and transformed GLM layout follow `ggml-org/llama.cpp` revision
`8ca4060b7e30e76baaa5670ee1ec1b760b7f8552` under the retained MIT notice.
Qwen adapted equations/shaders retain Apache-2.0 and Apple MIT attribution in
[Qwen Uzu provenance](QWEN38_UZU_PROVENANCE.md),
[LICENSE](../LICENSE.qwen38-reference) and
[NOTICE](../NOTICE.qwen38-reference). Archived extraction notices remain under
`vendor/lily-core/`; no executable crate remains there.
