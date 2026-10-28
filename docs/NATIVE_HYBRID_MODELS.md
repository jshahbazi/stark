# Native GLM-5.3-Flash and Qwen3.8-Flash-Next

Stark has a separate native Rust/Metal forward path for `glm5_next` and
`qwen4_exp`. It reads the original quantized GGUF or MLX Safetensors storage.
There is no Sushi, llama.cpp, MLX, Python or Transformers inference process in
this execution path. Python is used to prepare download metadata and by the
optional reference validators.

## Checkpoints and preparation

The preparation command pins these two sources:

| Choice | Checkpoint | Weight files |
| --- | --- | --- |
| `glm53-q2` | `unsloth/GLM-5.3-Flash-GGUF`, `UD-Q2_K_XL`, revision `a38483c8cd5df544f53d70fb281afe97369d5ab6` | 108,720,071,552 bytes, 101.25 GiB |
| `qwen38-mixed` | `pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit`, revision `b2c422f3c643e36f04227a64d61796b44a4b1029` | 106,195,031,237 bytes, 98.90 GiB |

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
cargo +1.98.1 build --release --locked -p stark-generate
cargo +1.98.1 build --release --locked -p stark-score --features optimized
./scripts/stark chat --model-path models/glm53-q2
./scripts/stark chat --model-path models/qwen38-mixed
```

Each output directory must be new. Interrupted downloads remain under the
adjacent `.native-download` directory and can be resumed with the same command.
`--offline` restricts preparation to existing Hugging Face downloads. Publication
uses an atomic rename that refuses an existing output directory.

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
IQ3_XXS and IQ4_XS. Floating tensors support F32, F16 and BF16 storage. MLX supports affine
4/8-bit matrices with group sizes 32, 64 or 128. MLA head projections and selected
experts are batched into individual GPU submissions. Only selected expert bytes
and their scale/bias planes are gathered, with a bounded compressed workspace.
The whole model is never expanded into BF16 or F32 weights.

This implementation uses F32 forward arithmetic. Small vector operations,
recurrent updates and sparse selection run in Rust on the CPU; matrix products
run on Metal. Prefill processes tokens sequentially. Compressed matrices are
uploaded for each call. Performance depends on these implementation choices and
must be measured on the exact checkpoint and host. Published Sushi or MLX
throughput does not qualify Stark's throughput.

Both paths are text-only. The native loader does not execute vision, video,
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
