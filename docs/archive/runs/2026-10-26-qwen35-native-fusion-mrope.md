# Native projected-image fusion and Qwen3.5 M-RoPE

Date: 2026-10-26. Source baseline:
`491a36c3f1911bde7df8de6cb0ead98171a922ee` on
`codex/qwen35-vision-artifact-contract`. The native source manifest and worker
hashes below identify the implementation accompanying this record. Earlier
converter validation records remain unchanged.

## Scope and acceptance

This implements the native fusion and position core of the image-support plan.
It accepts one already projected BF16 image through private Rust types, validates
the model/span/grid/feature contract, replaces exactly its embedding rows, and
carries three-axis positions and a signed rotary continuation delta through
prefill and decode. Causal token counts and KV offsets remain physical indices.
Owned uploads and completion guards retain request storage until GPU completion;
failure fences the state. Image prefill requires a fresh state and does not use
speculation or tensor capture.

| Acceptance item | Evidence and limit |
| --- | --- |
| Input ownership and validation | Host tests cover identity, BF16, width/count, finite features, grid/merge arithmetic, reserved tokens, wrappers, vocabulary, context, and chunk intersections. |
| Feature replacement | CPU and Metal synthetic decoders compare projected rows against independently selected token-embedding rows, bit for bit, with sensitivity checks for missing replacement. |
| M-RoPE | Official `[11,11,10]` sections and 64-wide rotation match an independent f64 equation oracle. Scalar/diagonal rows remain bit identical to unscaled RoPE. Rectangular row order and 1,024-row slicing are checked. |
| Partial rotation | Native CPU/Metal attention preparation rotates only 64 of 256 channels, preserves the suffix and values, and agrees with the independent rotation oracle. |
| Prefill and continuation | Whole/chunked same-dispatch fixtures agree within the stated BF16 budget. A 1,029-token request with image rows 1,022 through 1,025 crosses the real 1,024-token boundary and matches two greedy lookup-control outputs. Physical cache length is 1,031 after those outputs; the rotary delta remains -2. |
| Failure recovery | An injected internal nonfinite feature fault fails decoder setup, clears continuation state, and prevents reuse. A fresh request succeeds. Submission-owner tests check abandoned work and draining all submissions after a completion failure. |
| Text regression | Full engine CPU suites and model-bound generation/scoring smoke checks are recorded below. |
| Patch order and projector output | Pending preprocessing and vision tower implementation. Features in these fixtures are already projected; this record makes no projector or raw-image claim. |
| Official vision artifact | Still rejected by the native loader. No image capability, worker transport, or app attachment is enabled. |

The original proposal's Stage 2 row also mentions new vision blocks, patch order,
and projector numerics. Those broader acceptance items remain open. This record
establishes the requested fusion/M-RoPE core and does not mark the complete image
milestone or exact-checkpoint qualification complete.

## Numeric controls and limits

The coefficient oracle uses f64 equations independent of the Rust f32 table
calculation, with absolute error below `2e-6`. Partial native rotation uses an
absolute BF16 output budget of `0.004`; untouched channels and values must match
exactly. Whole/chunked decoder comparisons use
`abs(actual - expected) <= 0.00390625 + abs(expected) / 128`. Identical batch
shapes must match bit for bit. The fixture has model width 128, two 64-wide
attention heads, a four-channel rotary prefix, and generated BF16 parameters.
All fixtures are synthetic and removed after each test.

The first full Metal run reported 528 passed, one failed, and five ignored. Its
nine-token whole batch selected the existing GEMM attention path, while chunks
of 3/2/4 selected single-pass attention. An independent lookup-only control
reproduced the difference: maximum absolute logit error `0.5`, RMS error
`0.050133444` with explicit image axes. Ordinary scalar positions also reproduced
it, with maximum absolute error `0.5` and RMS error `0.049965527`. Chunked feature
fusion was bit identical to chunked lookup.
The numerical whole/chunked fixture now uses eight tokens, with chunks of 2/2/4,
so both executions use the same dispatch family and retain the original tight
budget. A separate nine-token control retains fusion-versus-lookup checks across
that threshold. This does not establish tight numeric agreement between the two
inherited attention paths. Cross-dispatch and official-checkpoint numeric
qualification remain open; no production kernel was changed.

Neither Transformers nor MLX was executed for these native fixtures. Reference
agreement here means the independently implemented position/frequency equations
and synthetic embedding control, rather than official-model logit parity.

## Environment and checks

Physical Apple M5 Max, 128 GiB memory, macOS 26.7.1. Engine tests and formatting
use `nightly-2026-10-01`; release workers use Rust 1.98.1. Dependencies remain
locked. Native logs and converted models are private ignored artifacts under
`runtime/stage2-native-validation-2026-10-26/`.

| Final check | Result |
| --- | --- |
| Engine `--no-default-features --features cpu,grammar` | 351 library tests passed, 2 ignored; 7 integration tests and 3 doctests passed. |
| Engine `--no-default-features --features cpu,grammar,rope-cache` | 353 library tests passed, 2 ignored; 7 integration tests and 3 doctests passed. |
| Engine `--features cpu,metal` on the physical GPU | 530 library tests passed, 5 ignored; 7 integration tests and 3 doctests passed. |
| Generator release build and tests | Build passed; 26 tests passed. |
| Optimized scorer release build and tests | Build passed; 39 tests passed across its targets. |
| Generation process boundaries | 6 cases passed against the rebuilt worker. |
| Scoring process lifecycle | 8 checks passed with the existing `stark-validation-reap` helper. |
| Repository policy, pinned nightly formatting, `git diff --check` | Passed. |
| Official full-vision artifact startup | Rebuilt generator rejected it: exit 1, `ready:false`, `model_load_failed`. |

Engine commands use `cargo +nightly-2026-10-01 test --locked -p stark-engine`
with the feature sets above and `-- --test-threads=1`. The generator was built
with `cargo +1.98.1 build --release --locked -p stark-generate` and tested with
the equivalent `test` command. The scorer was separately built/tested with
`-p stark-score --features optimized`; its features were not unified into the
generator. Process checks use Python 3.12.13 through `uv`,
`scripts/test_generation_process.py --binary target/release/stark-generate`, and
`scripts/test_scoring_process.py --reaper runtime/stark-validation-reap`.

The native source identity uses the existing converter
`native_build_provenance` helper: a sorted SHA-256 manifest of `crates/`, `.cargo/`,
`vendor/`, `Cargo.toml`, `Cargo.lock`, and `rust-toolchain.toml`. It includes the
new fixture source. Documentation is outside that manifest.

| Source/build identity | SHA-256 |
| --- | --- |
| Native source manifest | `fec9cdbe3f7f4db609651666e8b8fa52ea3175cc4b05fc92b8e261152548befa` |
| Fusion fixture Rust source | `79f7e81a50574e0d289bffe6d2b52e11879c1b37da7d2a208896993601dd2ae5` |
| Cargo.lock | `fcd9a5645c077b14e493d2e37df05d8335f423950853151081f80c6992ed8426` |
| rust-toolchain.toml | `72da9f9e503464ba7393560f6f51a69fb06be72633d52ae7725fa0ba59dfda22` |
| Generator executable | `a141d5435e11e29f4492fdf3acf91db4238da673670cfce13f2df042ef1d0045` |
| Optimized scorer executable | `d6a12d984ecf99036d2048d62b2dbaef426b9510c4c4f3d31cc9da1cb7bbd7a7` |

## Real-model text smoke

The BF16 text model is `Qwen/Qwen3-0.6B` at
`c1899de289a04d12100db370d81485cdf75e47ca`. Conversion ran offline from the cached
official snapshot through a private hardlink staging directory, using the locked
converter and Python 3.13.13. It emitted 226 tensors with text-only contract
`text-generation-v1`.

The generator handled two identical requests in one loaded process using
`fresh-per-request` state. Both produced the same greedy sequence: 11 input tokens,
eight sampled tokens, finish reason `length`. The scorer handled four requests
with consecutive cache enabled: one hit, three native calls, finite values, and
identical repeated logits. Changing context size forced a miss; returning to the
older context also missed. Request/reply content remained in memory and was not
logged. These are native text execution checks, not image or scoring parity
qualification.

| Converted artifact identity | SHA-256 |
| --- | --- |
| Configuration | `dfa5763dc220e871b86a6119536f5c1c2862a5137a473eaf9d812be3cc53924e` |
| Tokenizer | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| Conversion output identity | `87c3344a80b7376392cc5c44d5369680b68936e6930bb8ac476ca65a136f64c7` |

The conversion receipt's earlier qualification metadata does not qualify the new
native build. The worker hashes below identify this run separately.

## References and provenance

- [Transformers Qwen3.5](https://github.com/huggingface/transformers/blob/2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397/src/transformers/models/qwen3_5/modeling_qwen3_5.py), revision `2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397`, copyright 2025 The Qwen Team and The HuggingFace Inc. team, Apache-2.0. Inspected source SHA-256: `af05d0acfe79fff8aea1c75f93516a6a72bc80f6f901329a73e66a3866fe1e7c`. The Rust code independently implements `get_rope_index` and text rotary `recomposition_frequencies` equations.
- [MLX-VLM Qwen3.5](https://github.com/Blaizzy/mlx-vlm/blob/e9af376626b04ba0cdad23b1476fd7a4a583d721/mlx_vlm/models/qwen3_5/qwen3_5.py), revision `e9af376626b04ba0cdad23b1476fd7a4a583d721`, copyright 2025 Prince Canuma, [MIT license](https://github.com/Blaizzy/mlx-vlm/blob/e9af376626b04ba0cdad23b1476fd7a4a583d721/LICENSE). Its explicit feature/position/delta interface informed the private boundary; no runtime dependency or implementation was imported.
- [Current implementation boundary](../../MULTIMODAL_IMPLEMENTATION.md) and [upstream provenance](../../../provenance/UPSTREAM.md). Existing Uzu/Lalamo notices and source-bound validation history are preserved.

Only aggregates and source/build/artifact identities are published here. Model
weights, native binaries, request content, token/feature vectors, logits, and
probability outputs remain outside version control.
