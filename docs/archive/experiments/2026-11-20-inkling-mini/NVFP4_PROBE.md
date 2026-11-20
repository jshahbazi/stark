# Reproducing the real-weight NVFP4 probe

The probe links Unsloth's macOS arm64 llama.cpp shared libraries and executes the actual Metal backend's indexed matrix operation, expert-scale lookup and multiply. It loads the released GGUF and captures live expert inputs and route IDs for layers 6 and 11 on each of their three visits. A 32-token prefill and the following teacher-forced single-token decode supply the two execution shapes. An independent CPU reconstruction of layer 0 supplies an additional source-checkpoint control. Routing, activations, model values and token IDs remain in process memory.

## Dependencies and identities

Use the arm64 archive and its matching merged-source archive from Unsloth release `b12572-mix-8eebad4`. Their SHA-256 digests are `ef6c30328e5b24d2a282c627f0b2cd25a2869fd8b3444b12f62fdaf34b658667` and `da01168673b443123c1d714fe852711fe4228d2174dcf169f18db0168d239b59` respectively. The merged-source identifier in the release is `a5ba2dd76d1bb2173f6882bac30682f8ae55045e`. Compile with the release's headers and import its matching `gguf-py` for GGUF directory parsing. [Unsloth release](https://github.com/unslothai/llama.cpp/releases/tag/b12572-mix-8eebad4)

The activation shader comes from Ollama commit `6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8`. Supply a local copy of `mlx/nvfp4_activation.go`, SHA-256 `8a69afd0c9c24a239af288802ea32028c6be3fa06ca1a730959e76a3743fdfc2`. The harness extracts the original MIT-licensed Metal header and body and specializes them for the checkpoint widths. The wrapper uses Metal 4.0 and the safe math mode configured by Ollama's pinned MLX revision. [Ollama source](https://github.com/ollama/ollama/blob/6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8/mlx/nvfp4_activation.go) · [MLX compile options](https://github.com/ml-explore/mlx/blob/fc344c986c4169d99f76b5d95ae6b4a58a359a63/mlx/backend/common/custom_kernel.h)

## Running the probe

Set `UNSLOTH_SOURCE` and `UNSLOTH_LIB` to the extracted source and arm64 release directories. Set `PROBE_DIR` to this directory and `PROBE_PRIVATE` to a private scratch directory. Prepare the milestone's private workload fixture for the original-layer control. Use Python 3.13 with NumPy 2.3.4, PyYAML 6.0.3, tqdm 4.67.1 and tokenizers 0.22.1.

```sh
clang++ -std=c++17 -O2 -fobjc-arc -Wno-deprecated-declarations -dynamiclib \
    "$PROBE_DIR/nvfp4_probe.cpp" "$PROBE_DIR/activation_probe.mm" \
    -I"$UNSLOTH_SOURCE/include" -I"$UNSLOTH_SOURCE/ggml/include" \
    -L"$UNSLOTH_LIB" -Wl,-rpath,"$UNSLOTH_LIB" \
    -lllama -lggml -lggml-metal -lggml-base -framework Metal -framework Foundation \
    -o "$PROBE_PRIVATE/probe.dylib"

PYTHONPATH="$UNSLOTH_SOURCE/gguf-py" python "$PROBE_DIR/nvfp4_probe.py" \
    --lib "$PROBE_PRIVATE/probe.dylib" --activation-source "$PROBE_PRIVATE/nvfp4_activation.go" \
    --official "$OFFICIAL_SHARD" --gguf "$Q4_K_XL_GGUF" --output "$PROBE_PRIVATE/aggregates.json"
```

Add `--block6-screen --repeats 6` to reproduce the early physical-layer-6 control with eight experts and seeded inputs. That screening run used validated HTTP ranges of the same files followed by byte comparison between the original and GGUF. The [aggregate receipt](nvfp4-block6-screen.json) includes hashes of the selected weight and scale slices. The first model-free activation run used the default Metal math settings and failed exactness on 37 of 73,728 values. Matching the pinned MLX compiler settings preceded all reported real-weight results.

## Comparisons and timing boundary

- The independent CPU matrix reference decodes the publisher's original E2M1 nibbles and E4M3 scales, restores the per-expert outer scale, and takes float64 dot products for a reproducible spread of output rows from every exercised expert. Its weight checks compare actual codes, block scales and expert outer scales to the independently packed GGUF, accounting for the source's interleaved gate/up row order. All complete GPU outputs receive a finiteness check.
- One arm projects the observed floating-point inputs. The quantized arms round inputs to BF16 and evaluate the released four-over-six NVFP4 activation policy. The shader's per-row maxima and reconstructed inner values are compared elementwise to independently chosen nearest-even codebook entries and MAE scale candidates. One control restores the activation outer scale before the projection. The candidate keeps the inner values through the matrix product and applies the activation and expert outer scales in the Metal graph after the dot product. A further control replays the batched inputs as independent singleton products in one graph. Each arm is compared to its CPU reference. The difference between floating and four-over-six CPU references measures the numerical effect of skipping activation quantization.
- Projection wall times start immediately before GGML encodes and submits the prebuilt graph and stop after backend synchronization. This includes scheduling, GPU queue delay, execution and host wait. The graph includes the indexed dot product, expert-scale gather, optional activation outer-scale multiplication and final scaling. The eight selected experts share a single indexed operation per projection. Graph construction, buffer allocation, weight staging, input transfers, readback and reference work occur outside the interval. The singleton cells rotate through actual routes and inputs from independent token positions, checking the decode token as the first fixture. The batched cell replays the prefill tile. Both shapes use two warmups and twelve timed repetitions by default.
- The eight-singleton projection control runs eight independent products with distinct input/route pairs, per-expert scaling and output copies inside one GGML graph. The reported wall interval is divided by eight. Each complete group cycles as a unit. This tests submission amortization on one resident expert bank. Processing eight consecutive layers would use different banks and have inter-layer dependencies.
- Activation wall timing begins before creation of a Metal command buffer and encoder and ends after the CPU waits for completion. Metal GPU timestamps give the command-buffer execution interval. The eight-singleton activation control binds disjoint rows and issues eight dispatches in one encoder, dividing each interval by eight. Compilation, pipeline setup, allocation, BF16 input conversion and all copies occur outside the timed regions.
- The activation and projection libraries use separate command queues. Their measured wall intervals each include a submission and wait. Summing these standalone intervals estimates two serialized submissions under the harness and cannot identify the latency of a fused production graph. The CPU-visible bridge transfers reconstructed rows outside both intervals. Instrumented spans from a separate GGUF decode use a scheduler callback after the router and after each complete layer. Synchronization changes scheduling and those spans include host work. End-to-end throughput comes from the separately benchmarked model servers.
