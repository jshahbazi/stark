# Recursive Flint

Stark executes Recursive Flint natively through its Rust worker and Metal
matrix kernels. The implemented text architecture includes per-head QK
RMSNorm, grouped-query attention with learned sinks, split-half RoPE on the
configured layers, sliding/full attention, dense SwiGLU layers, and sigmoid
top-k routed experts plus an always-active shared expert. Expert correction
biases affect selection only; selected original sigmoid scores determine the
normalized mixture weights.

## Install and run

Prepare the official Apple Silicon INT4 checkpoint from a source checkout:

```sh
uv run --project tools/converter --locked --python 3.13.13 python -m stark_convert.hybrid_prepare flint-int4 --output models/flint-int4
cargo +1.98.1 build --release --locked -p stark-generate
./scripts/stark chat --model-path models/flint-int4
```

The preparation command pins `RecursiveSI/flint-int4` at
`3b93f9053b530488405ae9fca51d4900d6ce21f0`. Its five weight shards total
22,568,912,600 bytes (21.02 GiB). Allow at least another 4 GiB of free disk space
for preparation, and additional space for source builds. Interrupted downloads
resume from the adjacent `.native-download` directory with the same command.
The command preserves packed weight bytes and records the source identity and
metadata hash in `native-source.json`. Preparation alone does not establish
native loading or model quality.

Use **Manage Models > Add from Source > Local Folder** to import the prepared
folder in an app build containing this change. Chat and the local API use the
generation worker. Flint is prepared through this command rather than the
ordinary converter catalog. Its upstream chat template and all declared EOS
IDs are preserved.

The weight reader accepts the official export's byte views of little-endian
MLX packed words, affine 4-bit projections/experts, and 8-bit embeddings, with
BF16 scale and bias planes. It also accepts the base checkpoint's separate
floating expert matrices. Original-format directories with complete weights,
configuration and tokenizer can be passed directly to the workers. App import
and terminal chat require the prepared metadata.

## Execution limits

Weights remain memory mapped in their original storage. Matrix products and
the fused routed/shared expert stage use Stark's existing native Metal kernels.
Attention, rotary transformations, normalization and expert selection use Rust
F32 arithmetic. Sliding caches retain at most `sliding_window` entries; full
attention retains the request prefix. KV state is F32, allocated as tokens
arrive, with a checked 32 GiB payload maximum over the configured architecture. This
differs from Recursive's optional quantized KV cache policy.

The workers currently cap requests at 16,384 context tokens and 4,096 output
tokens. Flint's upstream 262,144-token configuration is not a Stark context or
memory qualification. Prefill advances one token at a time and omits discarded
vocabulary projections. This implementation does not include Recursive's fused
decode megakernel; its published throughput does not describe Stark.

Each request starts with fresh state. Errors clear the full request state.
Raw candidate scoring uses the existing scorer and cache rules; Decision mode
still requires a suitable direct-answer template and independent calibration.
This path provides text generation and scoring. It does not add structured tool
call parsing to the existing API.

## Numerical checks and provenance

Model-free Rust tests compare complete synthetic forwards against an
independent F64 implementation, including both banked and separate experts,
sinks, grouped heads, rotary/no-rotary layers, window eviction, reset and
ingest. Private fixtures can also be generated from Recursive's hash-pinned
MLX model module:

```sh
uv run --project tools/stock_benchmark/mlx_current --locked python scripts/flint_native_fixture.py --output runtime/flint-reference --quantized --source-float-dtype bfloat16
cargo +1.98.1 run --release --locked -p stark-engine --no-default-features --features cpu,metal --example hybrid_parity -- runtime/flint-reference
uv run --project tools/converter --locked python scripts/test_hybrid_workers.py --help
```

Fixtures and native results remain ignored. Validators print aggregate metrics
and hashes; they do not print inputs or logits. Reference execution uses F32
arithmetic and floating KV state, including when stored weights/scales are
BF16. These checks establish bounded numerical compatibility rather than
quality, long-context behavior or throughput of the full checkpoint. Dated
results belong under `docs/archive/runs/`.

The equations and tensor contract follow
[`recursive-oss/rec`, `engine/src/rec_mlx/model.py`](https://github.com/recursive-oss/rec/blob/0e3150ca4262f572dee25c231d9ac7148414d2af/engine/src/rec_mlx/model.py)
at `0e3150ca4262f572dee25c231d9ac7148414d2af`, source SHA-256
`56363f7a1caea2b9db317feb192f4bfea32eccb2b6b190dffc3fa93ee09695b1`.
The Apache-2.0 license is preserved in `LICENSE.flint-reference` and packaged
with distribution notices. Native loading never imports repository Python.

Related: [native model preparation and worker behavior](NATIVE_HYBRID_MODELS.md).

Recorded checks: [2026-10-28 validation](archive/runs/2026-10-28-flint-native-support.md).
