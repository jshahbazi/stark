# 20261121-INKLINGVISION-01 · Inkling 2 Mini native vision tower

## Identity and state

| Field | Value |
|---|---|
| Registered and executed | November 21, 2026, Codex |
| Kind and state | Component correctness, completed |
| Acceptance criterion | Finite, dimension-matched projected image rows with relative L2 at most 0.03 per fixture |
| Result and disposition | Passed for two synthetic controls on the original checkpoint. Adopted on the implementation branch within the component scope. |
| Related attempts | [Native decoder](2026-11-21-inkling-mini-native-slice.md), [short-prefix projection](2026-11-21-inkling-mini-short-projection.md) |

## Artifact and method

| Field | Pinned reference | Native candidate |
|---|---|---|
| Checkpoint | Original Inkling 2 Mini snapshot `6e7c8248f75265bdd4fbab5273304243b5d8c2cb` | Same original checkpoint |
| Runtime | Ollama `6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8`, MLX `0.32.4-82-gfc344c9`, Metal v4 | Stark dirty implementation based on `8211f6e783feeb80db239394d693fac610d470d1`, `inkling_vision` release example, Rust 1.98.1, locked dependencies |
| Executable SHA-256 | `d894f165a8aff58e84e56436006be3526a44fd4a7c622be3e583d80bf0fa8526` | `a95a53ce2829b6b8f6b3435080037307ab915420616feaabaa26359eb760f5c8` |
| Host | Apple M5 Max, 128 GiB, macOS 26.7.2 | Same host |

The pinned reference runs the loaded model’s `PrepareMedia` and `EncodeMedia`. Stark loads the original BF16 tower and encodes 23 blocks, 2×2 merge and the text-width projector on one Metal command buffer. The tested path uses tiled BF16 matrix multiplication with FP32 accumulation. Attention uses row-local streaming softmax with linear scratch. The scalar GELU uses the shared precision `erf` implementation because Metal compilation rejected the initial intrinsic call. The fallback SIMD matrix path compiled and requires device-level verification. [Reproduction, fixture hashes and aggregate results](../archive/experiments/2026-11-21-inkling-vision/README.md)

## Results and qualification scope

The 40×40 opaque and 53×84 RGBA fixtures passed with relative L2 of 0.015921 and 0.025454. Both used the original weights. The second fixture exercised encoded alpha checkerboard compositing, bottom/right padding to 80×120, rectangular position interpolation and six projected rows. The one-observation native encode times were 19.757 and 19.224 ms with weights loaded. Paired throughput and peak memory were unmeasured.

The normal image-loaded worker owns the tower for its lifetime. Image embeddings have request scope. Matching generated text, prefix-state lifecycle, loaded-tower text budget and preservation of compatible draft heads require worker evidence. JPEG, larger images, other Metal devices and the Unsloth artifact remain outside this component result. [Image default gates](../IMAGE_DEFAULT_PARITY_PLAN.md)
