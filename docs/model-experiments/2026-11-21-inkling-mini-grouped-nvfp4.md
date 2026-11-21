# 20261121-INKLING-14 · Grouped original-weight Inkling NVFP4

## Method

Registered November 21, 2026. This diagnostic asks whether sorted expert routes can make cold prefill faster on an M5 Max while preserving Stark’s current BF16 expert boundaries. Target decoder, prefix restoration and DSpark stay outside the component measurement.

Both arms load the original snapshot `6e7c8248f75265bdd4fbab5273304243b5d8c2cb` and use physical layer 10’s packed gate/up and down banks. The 64- and 128-row inputs use deterministic BF16 convex combinations of eight source-bound layer 10 activations captured at short and 4K prefixes. The actual original router supplies each token’s eight IDs and scores. This isolates the effect of regrouping a fixed real-weight routing distribution. [Oracle provenance](../archive/experiments/2026-11-21-inkling-oracle/README.md)

The private short fixture SHA-256 is `171f22c6edbbd3f7780f2cebe2437362238805c1c55f17ad7888d5d8568dc572`. The private 4K fixture SHA-256 is `15bc2cb21b66388258a7c45f63254fb3451005c4c3df59a21b9c81fc22c0ea31`.

Each arm times both activation quantizers, both projections, the expert outer scales, BF16 projection rounding, SwiGLU with routing weight and the final original-order sum on one command buffer. The serial arm runs existing per-token SIMD dots. The grouped arm uses 32×64×32 matrix tiles and the pinned FP32 ambiguity checks and replay. Its GPU lane gather and bounds calculation are timed. Router calls, host sorting and GPU upload of route metadata happen before timing. Each shape has one warm run per arm and five alternating pairs. [Pinned grouped kernel and MIT source](https://github.com/ollama/ollama/blob/6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8/mlx/nvfp4_matmul.go)

The numerical gate requires every weighted BF16 SwiGLU, BF16 down projection and BF16 expert sum word to match the serial arm. The practical speed gate is a median paired candidate-to-serial GPU-time ratio at most 0.90 for each measured shape. A passing result warrants an end-to-end trial with a GPU route planner and real prompt chunks. A numerical miss keeps the candidate diagnostic-only until the first affected boundary is resolved.

```sh
cargo +1.98.1 build --locked --release -p stark-engine --features inkling-slice --example inkling_grouped_nvfp4
target/release/examples/inkling_grouped_nvfp4 "$CHECKPOINT" "$ORACLE/4/fixture.safetensors" "$ORACLE/4096/fixture.safetensors"
```

## Result

Executed November 21, 2026 on the Apple M5 Max with 128 GiB, macOS 26.7.2, Metal 4, and a locked Rust 1.98.1 release build. The Stark working tree was based on `a5a803408fb3977fc06cf05319aa7b79dad33d56`. The executable SHA-256 was `bda2fbc06abafbcdc2793505ae8103fdbee76e22e1bfe650827541640431fdbd`. The aggregate result SHA-256 was `7399997f65b6a30d3e17ff1e1fe4b29baa83cb4a35520e850e042d19b07da4a6`.

| Input rows | Experts used (of 128) | Most rows per expert | BF16 words compared | Different | Serial median GPU / wall | Grouped median GPU / wall | Median paired GPU ratio (range) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 15 | 62 | 1,916,928 | 0 | 14.430 / 14.670 ms | 1.196 / 1.440 ms | 0.0829 (0.0827–0.0880) |
| 128 | 16 | 126 | 3,833,856 | 0 | 28.855 / 29.207 ms | 1.982 / 2.277 ms | 0.0687 (0.0685–0.0690) |

The grouped arm lowered dispatches from 194 to 72 at 64 rows and from 386 to 136 at 128 rows. The combined process peak physical footprint was 772,113,464 bytes. This includes both arms, one pair of physical weight banks and all diagnostic buffers. The OS reported zero change in pageouts, swapouts and compressor occupancy across the run. GPU foreign-activity counters were unmeasured.

Both isolated gates passed. The eight anchors produced concentrated routing, and the result applies to that distribution. A uniform- or diverse-route control, GPU-side route sorting, layer scheduling, exact prefix restoration and end-to-end cold speed remain separate work. The kernel remains a diagnostic. [Performance playbook](../PERFORMANCE_PLAYBOOK.md#inkling-2-mini-packed-experts)

## 20261121-INKLINGGROUP-02 · GPU routing and diverse expert coverage

The follow-up adds native row-batched routing, GPU stable expert bucketing, run construction and the final batched route sum. The matrix geometry and packed weights stay the same. The routed cases use the identical input-generation recipe and time each arm’s router. The stress cases use the same activations and original packed weight banks with fixed, normalized route weights and a deterministic permutation covering every expert evenly. This gives four routes per expert at 64 rows and eight at 128 rows. The fixed-route arms begin at the given routes and include GPU bucket construction.

Criteria set before execution preserve the first attempt’s numerical and speed gates. The two routed shapes additionally require exact serial/grouped router ID and FP32 score words. All four cases require the GPU permutation and run table to match a stable CPU bucket reference. The standalone component command excludes caller row copies and decoder graph scheduling.

The follow-up passed every gate on the same host and checkpoint. The executable SHA-256 was `e8d71a5e73820c8b090b2796d648ed74744ccf2f13943972f0f87cbcb9008bfc`. The aggregate result SHA-256 was `3dde8c8b461a43d180e53b23b548c7ebfa3711c7b5403042dbfb936e1ae17326`. No route IDs, scores or tensors were retained in the public receipt.

| Control | Rows | Experts used | Serial GPU / wall median | Grouped GPU / wall median | Median paired GPU ratio (range) | Dispatches |
|---|---:|---:|---:|---:|---:|---:|
| Original router | 64 | 15 | 18.234 / 18.523 ms | 1.267 / 1.518 ms | 0.0695 (0.0691–0.0709) | 322 → 12 |
| Original router | 128 | 16 | 36.430 / 36.777 ms | 2.065 / 2.305 ms | 0.0566 (0.0559–0.0570) | 642 → 12 |
| Uniform expert stress | 64 | 128 | 14.523 / 14.799 ms | 3.750 / 4.015 ms | 0.2583 (0.2579–0.2588) | 194 → 10 |
| Uniform expert stress | 128 | 128 | 29.033 / 29.340 ms | 3.998 / 4.239 ms | 0.1375 (0.1375–0.1378) | 386 → 10 |

The comparison covered 11,501,568 BF16 words across weighted SwiGLU, down and merged outputs, with zero differences. All four permutation and run tables matched. The two native router cases matched expert IDs and FP32 weights exactly. Peak physical footprint was 792,314,984 bytes for the combined diagnostic; pageouts, swapouts and compressor occupancy remained unchanged. The reusable prefill workspace is available on the opt-in M5 Max path for 64–128 rows. Full-graph parity, throughput and the default selection require separate evidence.
