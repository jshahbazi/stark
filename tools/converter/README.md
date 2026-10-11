# Stark model conversion prototype

`stark-convert` selects a bounded-memory streaming backend for verified dense Qwen3 checkpoints and the pinned upstream Lalamo backend for other registered models or weight formats. Conversion produces Stark's existing `config.json`, `tokenizer.json`, and `model.safetensors`; it does not introduce a new runtime format.

The [integration and coverage plan](../../docs/MODEL_CONVERSION_PLAN.md) defines the checklist, qualification gates, and parallel workstreams for expanding streaming support while retaining Lalamo fallback.

The streaming environment has no JAX, PyTorch, or Lalamo dependency. The fallback has its own Python environment and lockfile. Both use managed Python 3.13.13 through `uv`.

## Convert a model

Run from the repository root:

```sh
uv sync --project tools/converter --locked
uv sync --project tools/converter/lalamo --locked
uv run --project tools/converter --locked stark-convert Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --output models/qwen3-06b-stark
```

Repository inputs resolve to an immutable Hugging Face revision before downloading. Authentication uses Hugging Face's existing credential mechanism. Downloaded checkpoints stay in its local cache. For a local checkpoint, supply the original registered model identity to select Lalamo's architecture and chat metadata:

```sh
uv run --project tools/converter --locked stark-convert /absolute/path/to/checkpoint \
  --repo Qwen/Qwen3-0.6B --output models/local-qwen3-stark
```

`--backend auto` is the default. `--backend streaming` requires validated streaming coverage and fails otherwise; `--backend lalamo` explicitly uses upstream conversion. Existing output directories are refused. A complete staging directory is published with an atomic no-replace rename on macOS or Linux; failures remove partial output. The output directory has owner-only permissions.

The command emits aggregate timings, hashes, and backend selection. It writes a private `conversion.json` receipt containing the exact converter-source and dependency-lock hashes. Fallback source revisions are recorded privately. Native diagnostics and exception text are suppressed.

## Streaming coverage

The initial adapter supports dense Qwen3 with full-precision BF16, F16, or F32 tensors matching `--dtype`. BF16 is the default and the qualified format for Stark's current native loader. F16/F32 conversion tests establish format parity only; the current supported loader selects BF16 dense weights, and F16 embeddings are unsupported.

Verified metadata profiles are `Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B,32B}`, `Qwen/Qwen3-4B-Instruct-2507`, and `Qwen/Qwen3-4B-Thinking-2507`. Tied and untied embeddings, Q/K normalization, attention biases, and supported sliding-window settings have synthetic coverage. The real-model benchmark qualifies only the tested 0.6B checkpoint.

The writer plans tensor names, shapes, weight specifications, and byte offsets from checkpoint headers before writing. It copies full-precision embeddings directly, writes Q/K/V and up/gate matrices consecutively, and widens normalization scales to F32. The default copy workspace is 8 MiB. Tokenizer normalization and metadata parsing have separate memory costs. A bounded tiled transpose primitive is available for future adapters; ordinary Qwen3 does not need it.

Unsupported architecture, metadata profile, quantization, custom RoPE, or dense dtype conversion selects Lalamo automatically. Malformed checkpoints, unexpected tensors in the supported layout, and source changes fail without fallback. Lalamo's registry and transformation logic remain the authority for broader conversion coverage. Its cross-repository metadata references must retain their own source revisions. Conversion support does not establish runtime support: Stark currently rejects MoE, for example. Optional upstream dependencies and gated model access remain prerequisites where applicable.

## Validation and benchmark

```sh
uv run --project tools/converter --locked pytest tools/converter/tests -q
STARK_CONVERT_LALAMO_INTEGRATION=1 uv run --project tools/converter --locked \
  pytest tools/converter/tests -q
uv run --project tools/converter --locked ruff check tools/converter
uv run --project tools/converter --locked ruff format --check tools/converter
uv run --project tools/converter --locked stark-convert-benchmark \
  --source /absolute/path/to/checkpoint --repo Qwen/Qwen3-0.6B \
  --output runtime/converter-benchmark/run --repeats 3
```

The benchmark runs each backend in a fresh process, alternates execution order, warms source files before each run, and excludes dependency setup and downloads. It reports conversion time, complete CLI time including receipt hashing, and sampled process-tree RSS. RSS sums can count shared pages more than once, and 10 ms sampling can miss brief peaks. It verifies every output tensor's shape, dtype, and payload hash, weight specifications, complete configuration, and normalized tokenizer. Different safetensors header serialization can produce different whole-file hashes while every tensor byte matches.

Raw models, native validation output, and benchmark artifacts belong under ignored directories. Publish only aggregate results and hashes. See [benchmark evidence](../../docs/MODEL_CONVERSION.md).

## Provenance

The format, Qwen3 configuration, projection ordering, normalization dtype, and metadata conventions are adapted from [Lalamo revision `831accb8252a6b6606c378048caeb44178f1a050`](https://github.com/trymirai/lalamo/tree/831accb8252a6b6606c378048caeb44178f1a050). The fallback lockfile pins that exact revision. [Lalamo's MIT license](LICENSE.lalamo) and Mirai Tech Inc. copyright are preserved. Stark retains its Uzu base revision `8d01581474760bd1fc7f4a547ff1d02f7dada437`.

The adapted upstream paths are `model_import/loaders/huggingface.py`, `model_import/model_configs/huggingface/qwen3.py`, `model_import/common.py`, `model_import/huggingface_tokenizer_config.py`, `model_import/model_specs/qwen.py`, `modules/normalization.py`, `weight_matrix.py`, and `safetensors.py`. The streaming implementation and backend orchestration are maintained in this repository.
