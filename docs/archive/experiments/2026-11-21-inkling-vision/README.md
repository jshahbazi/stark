# Inkling 2 Mini vision oracle

The original checkpoint's image tower consumes BF16 20×20 image patches, including bottom/right padding to a 40×40 merge tile. The source first composites alpha over its checkerboard in encoded orientation. It then applies its CLIP-to-BF16 rounding and renormalization. The tower retains the CLS row through its 23 blocks and merges each 2×2 block in row-major order before the text-width projection. [Pinned implementation](https://github.com/ollama/ollama/blob/6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8/mlxrunner/model/inkling2/vision.go)

## Reproduction

Use an isolated copy of the Ollama v0.40.17-rc3 source at `6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8`, Go 1.26.10, cgo, the shipped Metal v4 libraries, and the original imported checkpoint. The capture invokes the loaded model's actual media-preparation and vision-tower methods on the runtime's GPU worker.

```sh
umask 077
mkdir -p "$PRIVATE/vision/images" "$PRIVATE/vision/features"
uv run --no-project python "$VISION_DOCS/instrument.py" "$ISOLATED_OLLAMA_SOURCE" "$PRIVATE/vision/images"
ln -s "$OLLAMA_RUNTIME/mlx_metal_v4" "$PRIVATE/vision/mlx_metal_v4"
(
    cd "$ISOLATED_OLLAMA_SOURCE"
    GOTOOLCHAIN=local CGO_ENABLED=1 go build -o "$PRIVATE/vision/runner" ./cmd/inkling-vision-oracle
)
INKLING_VISION_IMAGES="$PRIVATE/vision/images" \
    INKLING_VISION_OUTPUT="$PRIVATE/vision/features" \
    OLLAMA_MODELS="$ORIGINAL_OLLAMA_STORE" OLLAMA_LLM_LIBRARY=mlx_metal_v4 OLLAMA_NUM_PARALLEL=1 \
    "$PRIVATE/vision/runner" --model inkling-m1-original --ctx-size 4352
cargo +1.98.1 build --locked --release -p stark-engine --features inkling-slice --example inkling_vision
target/release/examples/inkling_vision "$CHECKPOINT" "$PRIVATE/vision/images/opaque.png" \
    "$PRIVATE/vision/features/opaque.png.safetensors"
target/release/examples/inkling_vision "$CHECKPOINT" "$PRIVATE/vision/images/padded-alpha.png" \
    "$PRIVATE/vision/features/padded-alpha.png.safetensors"
```

Both fixture hashes bind the exact encoded bytes and the reference feature matrix. The component gate expects matching dimensions, finite embeddings and relative L2 at most 0.03 against the reference for each image. It checks tower arithmetic through the text-width projection. Prompt framing, decoder image interpretation, text prefix restoration, and throughput with the tower resident require separate worker evidence.

| Control | Source dimensions | Padded dimensions | Encoded PNG SHA-256 | Private oracle SHA-256 |
|---|---|---|---|---|
| Opaque | 40 × 40 | 40 × 40 | `0c83253c26b2a897204b1ce166cace2f3338850441cb72500fe19f4a80603a56` | `ddceec3a21a5f1b832f2dd87f0f8f666aecee656719dcf21bea8b4346649c3bf` |
| Padded alpha | 53 × 84 | 80 × 120 | `58d8da4b3f48a024fd0533829035882fb4b4dd3754aae4373f0b79d748afd693` | `679419098109af3f5b3c69cf638c8d14f75b5683739d709e84f12db5fa12a2ad` |

Capture executable SHA-256 `d894f165a8aff58e84e56436006be3526a44fd4a7c622be3e583d80bf0fa8526`. The runtime reported MLX `0.32.4-82-gfc344c9`. [Metal payload and original checkpoint identity](../2026-11-21-inkling-oracle/README.md)

## M5 Max results

Both images passed the component gate with finite embeddings of the expected shape. The native executable SHA-256 was `a95a53ce2829b6b8f6b3435080037307ab915420616feaabaa26359eb760f5c8`.

| Control | Patch grid | Output rows | Relative L2 | Cosine | Maximum absolute error | Encode wall time |
|---|---|---:|---:|---:|---:|---:|
| Opaque | 1 × 2 × 2 | 1 | 0.015921 | 0.999876 | 0.015625 | 19.757 ms |
| Padded alpha | 1 × 4 × 6 | 6 | 0.025454 | 0.999677 | 0.017578125 | 19.224 ms |

Each encode wall time is one observation including CPU preprocessing, scratch allocation and Metal completion, with the weights already loaded. [Attempt 20261121-INKLINGVISION-01](../../../model-experiments/2026-11-21-inkling-mini-vision.md)
