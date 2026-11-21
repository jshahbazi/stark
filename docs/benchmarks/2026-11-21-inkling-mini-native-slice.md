# Original Inkling 2 Mini native Metal decode slice

The native Stark slice fit the M5 Max 4K serial step budget. The connected local-to-global numerical gate failed. Keep the implementation as an opt-in engine diagnostic and resolve the first BF16 producer difference before adding the full generation worker. [Port plan](../INKLING_2_MINI_PLAN.md) · [Aggregate data](data/inkling-mini-native-slice-2026-11-21.json)

## Scope and build

The slice ran physical layers 0–19 through the fixed triple-pass schedule. The 44 visits include 37 local attention visits, seven global visits and all 24 cross-pass reads with four learned gates each. The command includes the residual block mixing, post-attention and post-expert convolutions, all KV and convolution writes, and original-weight routing, gate/up and down projections. Each logical visit owns separate state. All intermediates stay on the GPU between visits. The weights are shared by physical layer.

The original `thinkingmachines/Inkling-2-Mini` revision is `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`. The original target shard is 19,526,019,168 bytes with SHA-256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`. Rust 1.98.1 built the locked `inkling-slice` engine example in release mode. It uses a retained Metal queue, one serial compute encoder and Metal 4 shader safe math. Packed weights use E2M1 and E4M3. Attention cache, convolution history and explicit projection boundaries use BF16. Accumulation, residual mixing, cross-pass expansions and attention use F32. The source reference is pinned Ollama `6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8`. The reference generator uses the upstream activation shader at its registered source hash, independent NumPy equations and MLX 0.31.2 gathered expert projections. [Native implementation](../../crates/stark-engine/src/engine/inkling_slice/mod.rs) · [Reference fixture generator](../archive/experiments/2026-11-21-inkling-native/reference.py) · [Prior source and controls](../archive/experiments/2026-11-20-inkling-mini/NVFP4_PROBE.md)

## Parity

Compare layer 10 directly followed by layer 11 on the second core pass. The 4K diagnostic starts at position 4,096 with synthetic but source-shaped incoming residual, KV and convolution state. It runs four consecutive positions, wrapping four slots of the local ring and appending four positions to global KV. The teacher-forced diagnostic derives eight tokens from an existing private workload and independently computes the prelude, first core pass and second pass through layer 11. Native validation starts at token position 4 with real state from the previous four rows. Neither fixture is a callback capture from the pinned Ollama binary.

Require identical unordered route sets, finite tensors and untouched BF16 cache data. The per-row relative L2 and peak-normalized absolute error limits are `2e-5` for F32 cross-pass messages, `0.008` for BF16 attention, input and state boundaries, and `0.015` for connected expert output. Exact-expert-ingress replay checks the quantized activation reconstructions, route sets, weighted intermediate and final expert output. The global ingress control feeds layer 11 the reference layer 10 partial, preserving the global visit’s own KV and convolution state.

| Check | 4K synthetic state | Short teacher-forced state |
| --- | ---: | ---: |
| Layer 10 post-attention worst row relative L2 | 0.342% | 0.135% |
| Layer 10 MoE input worst row relative L2 | 0.246% | 0.114% |
| Layer 10 expert output worst row relative L2 | 8.96% | 5.57% |
| Layer 11 expert output worst row relative L2 with native upstream output | 45.77% | 14.05% |
| Layer 11 post-attention worst row relative L2 with reference ingress | 0.151% | 0.0585% |
| Connected routing sets matching | 7 of 8 | 8 of 8 |
| Identical-ingress routing sets matching | 8 of 8 | 8 of 8 |
| Identical-ingress BF16 expert outputs matching exactly | 18,432 of 18,432 | 18,432 of 18,432 |
| Untouched BF16 KV elements changed | 0 of 10,993,664 | 0 of 10,993,664 |

Both independent ingress controls passed on both fixtures. The reconstructed four-over-six inputs to both expert products were identical at every compared value after substituting exact MoE ingress. The local attention cache writes and cross-pass messages passed their local gates. Layer 10 expert output was the first missed numerical gate. The exact-ingress replay shows the effect of supplying the reference BF16 input, and the global ingress control shows the effect of restoring the reference partial. Specific upstream projection differences remain to be isolated.

A second reference used MLX dense projections padded to the source’s eight-row short path and kept its explicit pointwise and attention equations. The Python MLX wheel is different from the pinned Ollama shared libraries. This control also failed connected parity and passed identical-ingress replay. On the 4K fixture, switching NumPy to the MLX dense producer alone changed the layer 10 MoE input by up to 0.181% relative L2, the input to the down dot after four-over-six by 9.54%, and the final expert output by 8.81%, with all four layer-10 route sets intact. This supports checking the dense producer’s operation and BF16 boundaries against the actual source runtime before changing packed-expert arithmetic.

## 4K decode budget

Measure 32 singleton steps per process after eight warmups across three fresh loads. Each step rotates through four BF16-derived ingress vectors and starts from populated, independent cache buffers with synthetic history. The timed positions are 4,104–4,135. Times include command encoding and one submit/completion wait. The diagnostic writes 9,216 bytes of shared ingress per step before the wall interval and reads no model data between submission and completion. Metal GPU timestamps report the command execution window separately. Weight loads, fixture reads, and parity readback are outside the timing interval.

| Measurement | Value |
| --- | ---: |
| Native slice wall median across process medians | 27.35 ms |
| Wall p10–p90 envelope across processes | 27.26–27.60 ms |
| Median GPU time range across processes | 26.42–26.48 ms |
| Dispatches, command buffers and waits per step | 1,756 / 1 / 1 |
| Resident Metal allocation during the run | 12.94 GB |
| Original full worker at 4K, greedy / sampled | 38.28 / 39.52 ms per useful token |
| Corresponding full-worker 90% throughput ceilings | 42.53 / 43.91 ms per useful token |
| Remaining wall budget at those ceilings | 15.18 / 16.56 ms |

The host was an 18-core, 40-GPU-core M5 Max with 128 GiB and macOS 26.7.2 (25G332), on AC in high-power mode. All 24 host samples across the three process windows had nominal thermals and normal memory pressure. The native process accounted for 81.8% of observed Metal counter increments. The system recorded 42 additional pageouts and no swapouts over the sampled interval. The measurement is a diagnostic parts budget on a shared host. [Prior matched 4K reference](2026-11-20-inkling-mini-m5-max.md)

The slice omits the final three local attention visits and one global visit, four MXFP8 expert calls, token embedding, final normalization and vocabulary projection, sampling and worker I/O. It does not load the image tower or the DSpark head. It does not measure the 4K prefill throughput floor or compare useful decode throughput. At a complete prefill boundary the resumable state must hold each visit’s KV/ring slots and positions and four convolution histories. Cross-pass residuals are within-token scratch. A chunked prefill path and worker admission are necessary to test generation prefix reuse, vision and DSpark. The global cache here is bounded to 4,352 positions, below the 8,192 log-scaling threshold.

## Reproduce

Use the pinned original shard and source tree from milestone 1. Install Python dependencies in a private environment using `uv` with `mlx==0.31.2`, `numpy==2.5.3` and `tokenizers==0.23.3`. Set `ORIGINAL_SHARD`, `OLLAMA_SOURCE`, `INKLING_PRIVATE`, and `INKLING_PYTHON` to the corresponding local paths. Keep fixture files and detailed data in the ignored private directory.

```sh
cargo +1.98.1 build --release --locked -p stark-engine \
  --no-default-features --features inkling-slice --example inkling_slice
"$INKLING_PYTHON" docs/archive/experiments/2026-11-21-inkling-native/reference.py \
  "$ORIGINAL_SHARD" "$OLLAMA_SOURCE/mlx/nvfp4_activation.go" \
  "$INKLING_PRIVATE/native-input.safetensors"
target/release/examples/inkling_slice \
  "$ORIGINAL_SHARD" "$INKLING_PRIVATE/native-input.safetensors"
```

Add `--workloads "$INKLING_PRIVATE/workloads.json"` to produce the short teacher-forced fixture, `--mlx-projections` to switch the dense producer, and `--parity-only` to skip timing in the native example. The example returns a nonzero exit status on a parity failure and still prints the aggregate timing from an eligible 4K fixture. Use the [milestone host sampler](../archive/experiments/2026-11-20-inkling-mini/host.py) to record host conditions.
