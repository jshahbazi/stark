# Optional GLM EXL3/FP8 backend

The `glm-exl3` Cargo feature adds a serial macOS arm64 backend for the
`mlx-stream` GLM-5.3-Flash EXL3 K3/mul1 streaming pack. It uses a pinned MLX
engine through Stark's C ABI and returns full F32 vocabulary readouts to the
existing generation worker. The model's FP8/BF16/EXL3 arithmetic is retained.
This is an experimental engine path under artifact qualification.

The loader requires the explicit `glm5_next` architecture and streaming
manifest contract. It selects this backend before the GGUF loader. Ordinary
builds report the pack as unsupported. Current admission reserves a maximum
32,768-token context, caps the GPU budget at 112 GiB with a 2 GiB margin, and
chooses expert rows from the native phase bills. MTP and vision are unavailable.
The generator's optional native-head preload hint does not enable MTP.

The target artifact is
[`OpensourceWTF/GLM-5.3-Flash-streaming-repack-exl3-3.05bpw`](https://huggingface.co/OpensourceWTF/GLM-5.3-Flash-streaming-repack-exl3-3.05bpw/tree/50237d5c26ceb98bfd85f779990b0f5e6d7af144),
revision `50237d5c26ceb98bfd85f779990b0f5e6d7af144`. Other pack revisions or
precision policies require separate qualification. The
[upstream bridge record](../../vendor/glm-exl3-bridge/UPSTREAM.md) identifies
all engine/dependency pins, licenses, local patches and staging commands.

## Build and load

Prepare the bridge stage explicitly using the pinned local sources described
in its upstream record. Cargo verifies the source inventory, stage identity and
every native artifact hash. It performs no native dependency download or build.

```sh
MACOSX_DEPLOYMENT_TARGET=26.2 \
STARK_GLM_EXL3_STAGE=/absolute/ignored/glm-exl3-stage \
cargo +1.98.1 build --release --locked -p stark-generate --features glm-exl3

target/release/stark-generate --model-path /absolute/path/to/glm-exl3-pack
```

The worker embeds the validated MLX library directory as a runtime search path.
Keep that stage available when running the worker. Moving or distributing a
stage needs a separate relocation and packaging qualification. The feature
requires macOS 26.2 or later and Apple silicon. It does not alter the standard
worker build or the existing GGUF optimized recipe.

## Prefix state and qualification

The bridge implements a checkpoint at a complete 16,384-token upstream prompt
call. It deep-copies mutable KDA convolution/recurrent state and DSA open-pool
tails into private CPU memory. Valid attention/indexer backing storage remains
model-owned; restore changes its logical lengths before appending the suffix.
Owner, epoch, context and source-prefix checks prevent reuse after an
intervening fresh prompt, failed request, cancellation or model replacement.
Snapshot and restore peak memory are charged to admission.

Prefix reuse remains disabled in ordinary builds while this path is being
qualified. An explicit bridge `--experimental-prefix` stage must be paired with
`STARK_GLM_EXL3_VALIDATE_PREFIX=1` when building the engine. That build advertises
the boundary to the existing worker so its normal context, scope, publication
and invalidation checks can be exercised. It is a validation opt-in.

The `glm_exl3_validate` example accepts bounded token input through an anonymous
stdin pipe. It compares full F32 readouts and hashes of complete logical KDA/DSA
state between restored and fresh changed-suffix requests, followed by greedy
continuation. It also checks stale snapshots, failed token admission and a
cancelled prefill callback. It emits aggregate counts and hashes only.
`scripts/validate_generation_prefix_cache.py` covers the worker's consecutive
request and host-key rules. Use the actual artifact, prompts longer than the
advertised boundary, and a context that admits follow-ups. Preserve separate
source-bound receipts whenever tooling or precision changes.

Performance measurements must use uncached requests, one excluded warm-up,
screened measured runs and the effective admitted expert row counts. The
bridge benchmark includes final full-readout materialization. The historical
upstream binary and 16K fixture are unavailable publicly, so new measurements
must identify their source tuple and workload rather than claim an identical
reproduction.
