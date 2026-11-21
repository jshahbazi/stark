# Inkling 2 Mini decoder captures from Ollama MLX

The capture runner uses the Ollama v0.40.17-rc3 Go source at revision `6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8`, with the release's Metal v4 libraries. The runner loads the original mixed-precision target and vision weights through Ollama's manifest and tensor loading code. It drives the target with the pinned tokenizer, shipped projection kernels and model-owned caches. It evaluates four explicit singleton forwards on the same caches immediately after the prefix. No HTTP serving, draft execution or sampling occurs during collection. The runtime values support diagnosis of the two selected text decoder visits. [Ollama source and MIT license](https://github.com/ollama/ollama/tree/6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8) · [Original checkpoint and imported artifact](../../../benchmarks/2026-11-20-inkling-mini-m5-max.md)

The taps cover pass two at physical layers 10 and 11, logical visits 22 and 23, numbered from zero. The files carry the actual residual inputs, BF16 key/value and convolution state, attention inputs and outputs, loop controller values, router indices and weights, NVFP4 activation quantizer values, and MoE inputs and outputs. The merge retains the released runtime's tensor bytes and changes shapes and cache slot order for Stark. Each key/value row is head-major within the token. Each chronological convolution history is rotated into the three slots indexed by absolute position. An optional capture adds layer 0 embedding RMS and Q/K/V inputs and projections to the same five-step workload used for the short fixture.

## Reproduce

Use an isolated copy of the pinned Ollama source, the original model store and the unpacked release's `mlx_metal_v4` directory. Use Go 1.26.10, cgo and the macOS developer tools. The downloaded Darwin arm64 archive matched SHA-256 `0b5183492292ada170f9b561a834863808dd02edba599e4ad0e89899330020b3`. Keep the input JSON, logs and raw and merged captures private. [Go archive checksums](https://go.dev/dl/)

```sh
umask 077
mkdir -p "$ORACLE_DIR" "$ORACLE_DIR/4" "$ORACLE_DIR/4096"
ln -s "$OLLAMA_RUNTIME/mlx_metal_v4" "$ORACLE_DIR/mlx_metal_v4"
python3 "$ORACLE_DOCS/instrument.py" "$ISOLATED_OLLAMA_SOURCE"
cp "$ORACLE_DOCS/capture.go" "$ISOLATED_OLLAMA_SOURCE/mlxrunner/model/inkling2/oracle_capture.go"
mkdir -p "$ISOLATED_OLLAMA_SOURCE/cmd/inkling-oracle"
printf '%s\n' 'package main' 'import ("log"; "os"; "github.com/ollama/ollama/mlxrunner")' 'func main() { if err := mlxrunner.Execute(os.Args[1:]); err != nil { log.Fatal(err) } }' > "$ISOLATED_OLLAMA_SOURCE/cmd/inkling-oracle/main.go"
(
    cd "$ISOLATED_OLLAMA_SOURCE"
    go fmt ./mlxrunner ./mlxrunner/model/inkling2 ./cmd/inkling-oracle
    GOTOOLCHAIN=local CGO_ENABLED=1 go build -o "$ORACLE_DIR/runner" ./cmd/inkling-oracle
)
```

Each private `request.json` holds `Prompt`, `Start` and `Chunks`. Use the prompt from case `g512-00` and the chunk list `[4]` for the short fixture. Use `g4096-00` and eight 512-token chunks for the long fixture. Use `"Embedding": true` for the optional visit 0 capture. If the selected prompt ends before it supplies four teacher tokens beyond the prefix, the runner uses the prefix's last four token IDs again. The runner reports only the SHA-256 of the little-endian token IDs and prefill progress.

```sh
for start in 4 4096; do
    capture_dir="$ORACLE_DIR/$start"
    OLLAMA_MODELS="$ORIGINAL_OLLAMA_STORE" OLLAMA_LLM_LIBRARY=mlx_metal_v4 \
        OLLAMA_NUM_PARALLEL=1 INKLING_ORACLE_INPUT="$capture_dir/request.json" \
        INKLING_ORACLE_OUTPUT="$capture_dir" "$ORACLE_DIR/runner" \
        --model inkling-m1-original --ctx-size 4352 > "$capture_dir/run.log" 2>&1
    python3 "$ORACLE_DOCS/merge.py" "$capture_dir"
done
```

## Recorded runs

| Capture | Prefix chunks | Captured positions | Merged fixture SHA-256 |
|---|---|---|---|
| Short | 4 | 4–7 | `171f22c6edbbd3f7780f2cebe2437362238805c1c55f17ad7888d5d8568dc572` |
| Long | 8 × 512 | 4096–4099 | `15bc2cb21b66388258a7c45f63254fb3451005c4c3df59a21b9c81fc22c0ea31` |
| Short with visit 0 | 4 | 4–7 | `9a33c6efd1244121e1458935254a7e481826b5f9ea1bd49d190d60dc2423ab21` |

The first long attempt used 2,048-token chunks and exited with status 137 before producing a fixture. The successful retry bounded the working set with 512-token chunks and released per-forward temporary array handles before evaluating cache state. Width can affect routed expert computation. Use the chunked prefix above when reproducing this reference.

Discard initial merged fixtures with SHA-256 `7b787f61add91123d94856fd30d57c7b3ae6176bf094f0e879f5f67ba71646b2` and `a158e556dd358c6407d5c0d613492bd82816cc6d40cdb5f31cc1df1c6a7ef195`. Those exports copied convolution state in chronological order. The corrected merge rotated the existing runtime tensors into Stark's absolute-position slots.

The short and optional visit 0 captures share token sequence SHA-256 `fc2381d9f7a8a4d2af3b732e106584b70bd03cc03db1a45bf9770396f3e96fd5`. The long token sequence SHA-256 is `44b25d4c71b4521ee9f9b06a897cf2ed93be87438f89dd57313d2a89e4f7164a`.

The imported manifest SHA-256 is `bd9c92c3d4c4c1a6c6904dea6a51e8503cceb0f77f8073c2c9e5c63976e0aa4d`. The application reported MLX `0.32.4-82-gfc344c9`. The release's Metal v4 `libmlx.dylib`, `libmlxc.dylib`, and `mlx.metallib` have SHA-256 values `ca64874f5226232695e22f85c0d4b48a3df71a2a8f3f836df8442a7df2ab66ea`, `7a362d78cfa54e2adb003db825376bfaaaaad23fba6fe58c14391786f15d2512`, and `bb219cf229e00d2b7af3eb65c044fb0365f171d78a273164b92ac54a8ba9caf4`, respectively.
