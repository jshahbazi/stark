# Stock TensorFold and Stark: Qwen3.5-2B MLX 4-bit

## Result — 2026-10-16

Stark was 1.53–1.81x faster end to end on three generation workloads using the
same pinned Qwen3.5-2B MLX 4-bit checkpoint. TensorFold used the original
upstream 1.0.12 release executable. Stark used its existing default generation
build and standard catalog converter. No inference-engine, model-support or
converter code was changed for this comparison. The JevK5 TensorFold fork was
not used.

| Workload | Input tokens | TensorFold default | TensorFold plain | Stark default | Stark faster than TensorFold default |
| --- | --- | --- | --- | --- | --- |
| Fibonacci function and explanation | 26 | 193.1 tok/s | 194.0 tok/s | 349.6 tok/s | 1.81x |
| GPU explanation | 31 | 194.2 tok/s | 194.3 tok/s | 349.0 tok/s | 1.80x |
| Copy/edit a Python file | 390 | 188.3 tok/s | 148.4 tok/s | 287.6 tok/s | 1.53x |

These rates are generated tokens divided by client end-to-end wall time,
including prefill, generation, cleanup and transport, excluding model loading.
They are not isolated decode throughput. All requests produced exactly 64
output tokens. The table reports the median of six timed repetitions after an
untimed warmup per mode and fixture. The three mode orders rotate, so each mode
occupies every order position twice.

Median wall times were 331/330/340 ms for TensorFold default and
183/183/223 ms for Stark across the three rows. TensorFold's normal context-copy
drafting improved the copy/edit workload by about 1.27x relative to its built-in
plain-decoding mode. The short Fibonacci and explanation cases showed little
drafting benefit.

## Model and unchanged engine paths

The shared model is
[`mlx-community/Qwen3.5-2B-MLX-4bit`](https://huggingface.co/mlx-community/Qwen3.5-2B-MLX-4bit/tree/93760be4f1f69842a46bc13dbdc0f19e291392a3),
revision `93760be4f1f69842a46bc13dbdc0f19e291392a3`. It uses MLX affine
4-bit/group-64 weights, hidden size 2048 and 24 layers. It was already in
TensorFold's qualified model table and Stark's converter catalog. TensorFold's
[existing native recipe](https://github.com/ashhart/TensorFold/blob/b090e3b19014bfa1eb2b7f38ea3cc4d39abec9be/docs/recipes/qwen3.5-2b-native.md)
names this exact checkpoint revision and Apple M5 Max as an exercised platform.

TensorFold loaded the original checkpoint directly through its existing
Qwen3.5-2B engine. Stark's normal streaming conversion retained the existing
4-bit packed representation and produced a 1,060,485,616-byte native artifact.
No BF16 expansion or fresh quantization was introduced. The ordinary catalog
profile fetched its declared tokenizer and generation metadata from Qwen's
repositories; their immutable revisions and hashes are recorded in the receipt.
Native Stark tokenization was checked against the source tokenizer for every
rendered benchmark prompt and matched exactly.

The checkpoint is the existing supported overlap on this host. Stark has no
existing Nemotron 3.5 Lightning implementation, so the model behind TensorFold's
headline Nemotron speed claims could not be compared under the no-custom-work
constraint. These results do not rank TensorFold's other model families.

## Controls and limits

- Host: Apple M5 Max, 128 GiB unified memory, macOS 26.7.1.
- Same prompt token IDs, original source chat template, thinking disabled,
  greedy decoding, seed 0, 2048-token context and 64-token output limit.
- Serial requests against loaded models, fresh per-request state and no retained
  prompt cache. TensorFold's idle keep-warm work was disabled to avoid competing
  with Stark between requests. Neither engine's kernels or build features were
  tuned for this model.
- TensorFold default drafting and its existing per-request `draft: false`
  control were both measured. Its two modes had identical token hashes on all
  three fixtures. Each mode was repeatable over all six repetitions.
- Stark and TensorFold had identical token hashes on the copy/edit case and
  different hashes on the two other cases. This does not establish cross-engine
  token equivalence or comparative answer quality.
- TensorFold uses local HTTP and Stark uses its existing private JSON-lines
  worker interface. Wall times include those respective transport costs.
- All timed requests reached the output limit. Model loading, conversion time,
  multi-stream throughput, long-context behavior and broad quality are outside
  this measurement.

The comparison adds only a benchmark driver and aggregate documentation. Runtime
prompt/reply payloads and token vectors are not written to result artifacts;
the driver contains the public synthetic fixture templates. Model and runtime
files remain ignored. Results contain timing samples, counts and hashes.

## Provenance and replay

- TensorFold: upstream release `tensorfold-native 1.0.12`, original verified
  macOS arm64 archive. Executable SHA256:
  `97abc60e6b5bb4c48f5f58bfbe49f214a86bd44df25e064186a46aff4e0b8ff3`.
- Stark: default `stark-generate` release build, Rust 1.98.1, locked dependencies.
  The scoring worker's optional `optimized` feature bundle was not enabled.
  Debug stripping emitted the existing missing-LLVM-library warning; the build
  succeeded.
- Measured Stark checkout: `b5693c72b16ed62e30489810560b292747fb8b41`.
  This commit adds the benchmark harness; engine and converter sources are
  unchanged from the pre-comparison checkout.
- The [aggregate receipt](../archive/experiments/tensorfold-stock-qwen35-2026-10-16/aggregate.json)
  contains actual worker, source, converter, tokenizer and harness hashes,
  individual samples, finish reasons and output hashes.

Prepare the standard Stark artifact from the pinned source folder:

```sh
uv run --project tools/converter --locked stark-convert /absolute/path/to/source \
  --repo mlx-community/Qwen3.5-2B-MLX-4bit \
  --revision 93760be4f1f69842a46bc13dbdc0f19e291392a3 \
  --write-source-manifest runtime/stock-qwen35-source-manifest.json \
  --output models/stock-qwen35-2b-q4
cargo +1.98.1 build --release --locked -p stark-generate
uv run --project tools --locked python scripts/compare_tensorfold_stock.py \
  --source /absolute/path/to/source \
  --stark-model models/stock-qwen35-2b-q4 \
  --tensorfold /absolute/path/to/upstream-release/bin/tensorfold-native \
  --stark target/release/stark-generate --tokens 64 --rounds 6 \
  --output runtime/stock-qwen35-comparison.json
```

The driver starts and stops both processes, uses only their existing interfaces,
checks source/Stark tokenization, and keeps protocol contents in memory. The
required Python environment is the existing locked `tools` project. Avoid
competing inference, validation or compilation while measuring.
