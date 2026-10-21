# Stock benchmark runtimes

The `mlx` project preserves its exact dependency lock from the
[2026-10-19 stock-engine pilot](../../docs/STOCK_ENGINE_BENCHMARK_2026_10_19.md).
The `vllm` project contains the separately identified 2026-10-21 dependency
refresh below. The pilot's original vLLM 0.31.0 recipe and lock remain available
at packet commit
[`4adfae080a2d32152bfb3f362b26e6ab49da02ce`](https://github.com/jshahbazi/stark/tree/4adfae080a2d32152bfb3f362b26e6ab49da02ce/tools/stock_benchmark/vllm).
The original lock SHA-256 is
`5bc1702fca04bd4deb945a562d15918f28740f13856e3dcdd20961a934dff601`.
These separate projects keep benchmark dependencies isolated from the main
tool and application environments.

Use a task-owned environment when replaying a runtime. For example:

```sh
UV_PROJECT_ENVIRONMENT=/absolute/path/to/task-runtime/mlx/.venv uv run --project tools/stock_benchmark/mlx --locked python -c 'import mlx_lm; print(mlx_lm.__version__)'
```

Set `UV_CACHE_DIR` to a task-owned cache path. Do not rely on globally
installed modules. Serialize measured GPU work on the benchmark host and record
background activity and timing exclusions.

The MLX runtime uses stock `mlx_lm.utils.load` and `stream_generate`; its
worker keeps prompt and output payloads on a private pipe and writes only
aggregate hashes and scalar timings. Flat JevK5 admission must fail closed if
the stock loader does not support its exact `qwen3_5_text` configuration.

The separate `mlx_current` project pins upstream MLX-LM revision
`72be8c955d9a0d3099bd01865af288e0e947c3fa` for the 2026-10-20 optimization
comparison. It preserves that run's lock byte for byte and uses Python 3.13.
Replay it in a separate task-owned environment:

```sh
UV_PROJECT_ENVIRONMENT=/absolute/path/to/task-runtime/mlx-current/.venv UV_CACHE_DIR=/absolute/path/to/task-runtime/uv-cache uv run --project tools/stock_benchmark/mlx_current --locked python -c 'import importlib.metadata; print(importlib.metadata.version("mlx-lm"))'
```

The pilot's original vLLM lock was admitted on Apple silicon for the exact Qwen
checkpoint in the report. The updated recipe below has no native runtime or
model admission result. Worker RPC sockets must use an owned short temporary directory
(the shared driver creates `/tmp/svm-*` and removes it after server teardown).
The longer runtime-root path exceeds macOS socket-path limits. The driver keeps
caches under the task-owned root and records the exact server/runtime identity.
Cache flags, prompt caching, GPU offload and server overhead are explicit in the
report and configuration templates.

The path-redacted configuration templates in the experiment directory preserve
the actual CLI flags. Replace `<repository>`, `<task_runtime>`,
`<qwen_mlx_source>`, `<borrowed_models>` and `<sealed_reference>` recursively with
local absolute paths in private configuration copies. Never run the placeholder
templates directly. Use the corresponding generation, scoring or serving driver
with `--config PRIVATE_CONFIG --output AGGREGATE_JSON` through the locked `tools`
project. The corrected llama repeat uses the generation driver.

The scoring C++ adapter's header records its build command and exact upstream
header revision. Link it against the matching release libraries. These adapters
provide measurement and readout; they do not patch the engine implementations.

Locks alone do not qualify a result. Preserve the source-bound aggregate evidence,
host-screen dispositions and model hashes when reproducing the measurements.

## 2026-10-21 dependency refresh

The current `vllm` recipe pins the official vLLM `0.32.0+cpu` and vLLM Metal
`0.32.0` wheels. This addresses
[`PYSEC-2026-4319`](https://github.com/pypa/advisory-database/blob/main/vulns/vllm/PYSEC-2026-4319.yaml),
whose fixed version is vLLM 0.32.0. Both the main tools and vLLM benchmark locks
now require certifi 2026.10.19 or newer for the
[emSign trust-store advisory](https://github.com/certifi/python-certifi/security/advisories/GHSA-6q48-6c7w-9gw2).

The benchmark recipe explicitly overrides vLLM's setuptools 78.1.1 pin with
setuptools 83.0.0, the first fixed version for
[`GHSA-h35f-9h28-mq5c`](https://github.com/pypa/setuptools/security/advisories/GHSA-h35f-9h28-mq5c).
That also includes the earlier
[`GHSA-5rjg-fvgr-3xxf`](https://github.com/pypa/setuptools/security/advisories/GHSA-5rjg-fvgr-3xxf)
fix. The official vLLM 0.32.0 wheel's Python runtime sources contain no
setuptools or `pkg_resources` imports. vLLM Metal imports
`setuptools.Distribution` inside its optional `build_mps()` source-extension
builder. A focused Python 3.12.13 check of `Distribution` under setuptools
83.0.0 passed. This override is scoped to the benchmark project and is recorded
in its lock; it adds no advisory exception.

Locked resolution and model-free adapter tests were checked for this refresh.
No vLLM environment was synchronized, no native extension was built, and no
model was loaded. Import compatibility, native execution and benchmark results
for this updated recipe remain unqualified. The 2026-10-19 measurements retain
their original vLLM 0.31.0 identity and acceptance limits. Historical replay
requires the original source and a separately scoped environment; its archived
dependency versions have known advisories.

The [quiet-window Qwen follow-up](../../docs/QWEN35_MLX_QUIET_COMPARISON_2026_10_20.md)
uses `scripts/benchmark_clean_qwen.py` with the same workers and pinned artifacts.
It attributes external CPU activity by phase and requires fresh quiet admission
after full warmup. Its stricter measurement guards, per-attempt receipts and
accepted paired subsets must accompany any result. Regression checks run with
`python -m unittest discover -s scripts -p test_clean_qwen.py`.
