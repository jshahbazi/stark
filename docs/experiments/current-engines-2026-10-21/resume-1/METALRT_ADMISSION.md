# MetalRT admission check

Date: 2026-10-22

Disposition: **exact-model-support-unverified**. No MetalRT timing result was collected.

The official [MetalRT binary distribution](https://github.com/RunanywhereAI/metalrt-binaries) documents signed, prebuilt MetalRT binaries for RCLI on macOS arm64 and lists macOS 14+ and Apple Silicon platform requirements. It documents the engine files and installation route, but it does not publish a Qwen3.5-2B model catalog or an exact model-loading contract for this benchmark.

The official [RunAnywhere SDK model and engine documentation](https://github.com/RunanywhereAI/runanywhere-sdks/blob/main/README.md) identifies MetalRT as the proprietary GPU engine powering RCLI on Apple silicon. In the same documentation, Qwen3.5-0.8B/2B/4B appears under the **QHexRT** model table, whose text explicitly describes Qualcomm Hexagon NPU bundles and Snapdragon device requirements. The Apple Silicon cross-platform model section names MLX and llama.cpp, without establishing that those Qwen3.5 bundles are MetalRT inputs.

This distinction matters: QHexRT support for Qwen3.5-2B is evidence for a Qualcomm NPU bundle, not evidence that MetalRT can load the benchmark's Apple Silicon MLX safetensors representation. The official sources establish that MetalRT binaries exist for macOS Apple Silicon, but they do not establish exact stock support for this Qwen3.5-2B model, its MLX affine W4/G64 representation, or a reproducible MetalRT invocation compatible with the retained harness.

Accordingly, MetalRT remains an explicit admission gap for this campaign. It must be reported as an exact-model-support-unverified result until an official exact-model path and a successful model admission receipt establish a reproducible stock setup. It is not silently treated as a zero-throughput engine and no prior unmeasured status is converted into a timing claim.

