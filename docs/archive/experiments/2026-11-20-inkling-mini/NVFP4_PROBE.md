# Reproducing the real-weight NVFP4 probe

The probe links Unsloth's macOS arm64 llama.cpp shared libraries and executes the actual Metal backend's indexed matrix operation, expert-scale lookup and multiply. It loads the released GGUF and captures live expert inputs and route IDs for layers 6 and 11 on each of their three visits. A 32-token prefill and the following teacher-forced single-token decode supply the two execution shapes. An independent CPU reconstruction of layer 0 supplies an additional source-checkpoint control. Routing, activations, model values and token IDs remain in process memory.

## M5 Max result

On macOS 26.7.2 (25G332), delaying the activation outer scale until after the indexed dot passed all 26 projection cells. The largest relative L2 error was `6.00e-8` and the largest maximum absolute error relative to the reference maximum was `1.05e-7`. The 155,520 checked outputs cover 48 rows per exercised expert. The layer-0 control exercised 37 experts. Layers 6 and 11 each exercised 71 experts across three visits and two shapes. Each individual 32-row tile used 54–61 distinct experts. All full GPU outputs were finite. All 4,529,664 comparisons of activation inner values matched, including repeated rows across experimental shapes. Row maxima also matched. [Complete aggregate receipt](nvfp4-routed.json)

Restoring the activation outer scale before the dot failed the relative L2 gate in 11 of the 12 captured 32-row cells. The largest error was `1.31e-4` against a `1e-4` gate. The 32-singleton replay of the same inputs passed each cell. Dropping activation quantization moved the independently computed products by 5.96%–9.06% in relative L2 across the GGUF-sourced cells. The GGUF's top-8 routing and floating activations describe the conversion's execution. This experiment measures the original quantization policy applied to those actual routed inputs and to the separate original layer-0 fixture. [Source-checkpoint ingress reconstruction](original_layer0.py)

The table gives the median of six layer/pass medians in milliseconds, with twelve warm measurements per cell.

| Measured stage | Isolated single | Eight-singleton submission, per single | Grouped 32-row prefill | Same 32 rows as serial products |
| --- | ---: | ---: | ---: | ---: |
| Gate/up indexed dot and scales | 0.2095 | 0.0763 | 0.7534 | 1.8188 |
| Down indexed dot and scales | 0.1877 | 0.0612 | 0.4575 | 1.4412 |
| Gate/up activation, wall | 0.1452 | 0.0198 | 0.1503 | — |
| Down activation, wall | 0.1495 | 0.0209 | 0.1527 | — |

Multiplying the two single-product medians by 44 NVFP4 visits gives 17.48 ms. Two separately submitted, synchronized activations per visit add 12.97 ms, for 30.44 ms of the isolated work. Applying the same arithmetic to the eight-singleton controls gives 6.05 ms for products and 1.79 ms for activations. This 7.84 ms calculation reuses one bank within each eight-operation submission. A complete token visits different banks and depends on intermediate results. Both calculations omit router selection, gated nonlinearities, expert reduction, the other four layers' expert format, attention, residuals, vocabulary projection and sampling. Qualification requires a whole-token graph on the target.

The GGUF graph callback also timed one teacher-forced decode after 33 input tokens. The 48 intervals ending at router logits totalled 24.106 ms. These include attention, convolution, residual mixing, loop gates when active and the router projection. Medians for local/global visits without messages were 0.421/0.498 ms, versus 0.552/0.606 ms on visits with messages. The 48 router-to-layer-end intervals totalled 15.487 ms. Of these, 14.097 ms covered the 44 NVFP4 visits and 1.390 ms the four Q8_0 epilogue visits. These intervals include selection, expert math, the post-FFN convolution, residual addition and callback/scheduling overhead. They stop before the final vocabulary head.

The first profiling run used prefix-only name matching and counted suffixed view nodes, generating 380 events (48 router and 332 layer-end matches). Those stage timings were rejected. Requiring complete node-name matches produced 96 alternating boundaries across visits 0–47. The two runs produced identical hashes for all 24 captured inputs and their route IDs, plus the last-position prefill, teacher-forced decode and separately instrumented decode readouts. Each readout contained 200,058 finite logits. The archived receipt contains per-boundary elapsed times and aggregate readout counts and hashes. Whole-model throughput comes from the uninstrumented server runs.

## Dependencies and identities

After transfer, an independent CPU pass hashed both complete artifacts. The original shard was 19,526,019,168 bytes with SHA-256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`. The Q4_K_XL GGUF was 17,629,759,456 bytes with SHA-256 `e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15`. All 20 original expert weight/scale tensors used by the routed probe matched their pre-run digests. The layer-0 replay reproduced both the activation-array digest and the route-ID digest, along with the four original gate/up tensor digests and the three early GGUF header/weight/scale digests. All 12 early physical-layer-6 slice hashes matched their corresponding regions in the complete artifacts.

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
