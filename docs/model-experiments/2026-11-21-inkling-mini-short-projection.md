# 20261121-INKLING-07 · Short-prefix attention projection

Executed 2026-11-21 on the original fixed-three-pass Inkling 2 Mini using its packed NVFP4 diagnostic slice. The source-order M5 Max BF16 projection passed the connected and identical-ingress gates at both depths. Adopted in the native slice on the implementation branch. Full-model, worker and installed-build qualification are separate. [Predecessor](2026-11-21-inkling-mini-pinned-oracle.md) · [Benchmark](../benchmarks/2026-11-21-inkling-mini-short-projection.md) · [Ledger](README.md)

| Artifact | Identity |
| --- | --- |
| Original target checkpoint SHA-256 | `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1` |
| Source runtime | Ollama `6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8`, released MLX v4 libraries |
| GemvWide arithmetic source | Apple MLX `fc344c986c4169d99f76b5d95ae6b4a58a359a63`, existing `LICENSE.mlx-reference` |
| Stark baseline | `8211f6e783feeb80db239394d693fac610d470d1`, executable SHA-256 `828b4f147320cc9eaa37ed1daa9dca8babce86656a1d9c914129850023cc1dd1` |
| Candidate | Baseline plus projection and private trace diagnostics, executable SHA-256 `fe099421bd7b135d0e2b2d43c23f046b3c836c3f7f905c5e7cfefdbf01d532aa` |
| Candidate projection patch SHA-256 | `81a0c033feb34a2bb127241810b81790b55a5e7f10323fa4d426b126fd538084` |
| Short / 4K live fixture SHA-256 | `171f22c6edbbd3f7780f2cebe2437362238805c1c55f17ad7888d5d8568dc572` / `15bc2cb21b66388258a7c45f63254fb3451005c4c3df59a21b9c81fc22c0ea31` |
| 4K timing seed SHA-256 | `7a55cbe224772e650ff584c662bca2a14f551f7498ab9be5c3e53d27d0ceff0b` |
| Build and host | Locked Rust 1.98.1 release, `stark-engine --no-default-features --features inkling-slice --example inkling_slice`, 128 GiB M5 Max with 40 GPU cores, macOS 26.7.2 (25G332) |

1. Single-stage replay replaced captured BF16 outputs in the old native command. The first effective replacement was the layer 11 attention gate. Replacing the query or first controller rank left the failure intact. F64 CPU replay and padded Thin NAX retained one mismatch on the same gate input.
2. Replay through pinned MLX selected GemvWide. Copy its FP32 dot grouping and lane reduction with existing BF16 ingress and output. The two connected visits at positions 4–7 and 4,096–4,099 passed the preceding attempt's gates with exact route sets and zero BF16 expert mismatches. Device detection confines this result to M5 Max.
3. Compare eight alternating baseline/candidate pairs with the same 4K seed, eight warmups and 32 singleton steps. Private diagnostics were inactive during timing. The paired ratio was 0.965 against a maximum of 1.05, with one completion, unchanged dispatches and eligible host measurements. Keep the projection for correctness. The measured gain falls below the meaningful-speedup criterion. Peak memory was unmeasured.

The diagnostic scope ends at the connected slice and 44-visit budget. The target-graph, worker prefill, prefix restoration, image and draft gates remain separate. Reuse the pinned live fixtures in the target-graph test. [Updated performance playbook](../PERFORMANCE_PLAYBOOK.md#inkling-2-mini-packed-experts)
