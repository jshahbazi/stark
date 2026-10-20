# Stock benchmark runtimes

These directories preserve the exact dependency locks used for future
read-only stock-engine comparisons. They are separate projects so the root
tool environment and application dependencies remain unchanged.

Use a task-owned environment when replaying a runtime. For example:

```sh
UV_PROJECT_ENVIRONMENT=/absolute/path/to/task-runtime/mlx/.venv uv run --project tools/stock_benchmark/mlx --locked python -c 'import mlx_lm; print(mlx_lm.__version__)'
```

Set `UV_CACHE_DIR` to a task-owned cache path. Do not rely on globally
installed modules. Model loading and GPU execution require a separately
approved benchmark lane.

The MLX runtime uses stock `mlx_lm.utils.load` and `stream_generate`; its
worker keeps prompt and output payloads on a private pipe and writes only
aggregate hashes and scalar timings. Flat JevK5 admission must fail closed if
the stock loader does not support its exact `qwen3_5_text` configuration.

The vLLM lock is retained for a later Apple-silicon admission check. Its
planned RPC path must use a short local socket/HTTP timeout and record the
exact server/runtime identity. Cache flags, prompt caching, GPU offload and
server overhead must be explicit in any result.

Neither lock constitutes a qualification or benchmark result.
