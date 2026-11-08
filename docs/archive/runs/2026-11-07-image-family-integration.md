# Image family integration checkpoints

This record separates implemented artifact paths, functional model checks and numerical qualification. The family contract and exact source identity remain necessary for every loaded artifact. No PR or release is part of this work.

## Coverage

| Path | Implementation and evidence |
| --- | --- |
| Qwen 3.5/3.8 dense affine, schema 3 v3 | Packed decoder matrices and a complete BF16 tower. The pinned 27B extracted tower and fresh release-worker lifecycle pass. Both quantized 0.8B full-model reference cases now match every logit bit for bit on the production path. |
| Qwen 3.5 affine MoE, schema 3 v4 | Router, selected expert banks, sigmoid-gated shared expert and BF16 tower. The pinned 35B-A3B extracted tower and fresh release-worker lifecycle pass. These do not transfer numerical qualification to other checkpoints. |
| Qwen 3.5 BF16-source MoE, schema 3 v5 | Distinct image-only conversion, full-bank admission, indexed reference dots and BF16 router/route boundaries. Native controlled component checks pass; full official 35B loading, parity and worker execution are unverified. |
| Retained Qwen Flash Next, schema 6 | Exact-source Lily image contract and native worker lifecycle. A new vision-only comparison fails one unchanged full-tower feature gate and remains under investigation. |
| Gemma 3, schema 5 | Original 4B/12B/27B BF16 wrapper admission and SigLIP processing. The pinned 4B numerical and worker/API lifecycle checks pass; full 12B/27B execution is unverified. |
| Gemma 4, schema 4 | Pinned E2B/E4B, encoder-free 12B, 26B-A4B MoE and 31B dense profiles. E2B/E4B have separate execution evidence; 12B embedder and 26B MoE component checks pass. The 31B feature gate and full larger-checkpoint generation remain open. |

Image requests clear and bypass prefix reuse and disable image MTP. Gemma text ring-state restoration remains a documented prefix-reuse blocker. See the [Qwen family record](2026-11-07-qwen-image-families.md), [BF16 MoE record](2026-11-07-qwen35-bf16-moe-images.md), [Gemma 3 record](2026-11-07-gemma3-images.md), and [Gemma 4 scope](2026-11-07-gemma4-image-family-scope.md).

## Integrated regression checks

The converter source tree `9029cb045338e4d29c9ada457d99fd1aceb1e3e8` passed **6,307 tests, with 96 skipped**, using the locked Python 3.13.13 environment. The sealed native template helper was explicitly discoverable on `PATH`; its SHA256 is `913ec72e2e94ae98cc75d098315f2d42abe02af068718c384662ab75cd236f6f`. A preceding run without a discoverable helper had invalid test setup and is not counted as regression evidence.

Engine source `6fb15e6d21479d313b1bbbcba14b8b428d7a3a7c` passed **924 CPU/Metal native tests, with 113 ignored and six benchmark cases filtered**, on the pinned nightly. The sealed test executable SHA256 is `1de43487d13e0fa754a35ed22375a393f07c87f96f692e301e59add929943d4e`. Ignored model-bound checks require their separate source-bound receipts.

The subsequent CPU/Metal suite on `a1649a6aa9f580b60e8701c6c2a463f38e6ad76d` passed **928 tests, with 124 ignored and six benchmark cases filtered**. Its sealed executable SHA256 is `ff1c7d870c917dd293b09dc708cdcf39cef229c89e168b164c4d34f1fe0bb0ee`. This build disables default features and debug information to reduce cache size; it does not rerun the grammar coverage in the preceding default-feature suite.

The Rust 1.98.1 release worker sealed from `5b8d40258ad93631dd2a1217b3d07acd6282c748` has SHA256 `5985412ddb84abc32fd3338c5384d394e0c52b5c95392152d61027ae407784b4`. It passed six content-free startup checks, and its reference-test selectors are absent. The release generation suite subsequently passed **122 tests, with two ignored**, on `1d0fe455ea22490159257fe61bb50252894d4ccf`; intervening Lily changes were test controls. The sealed host package contains 68 individually hashed source/data files and the verified template helper.

Release generation Clippy passed with all targets and `-D warnings` on `96b6da7d11244774a0d4ca0339369206d216b661`. Repository policy and its **41 tests** also pass. The standalone `cargo-deny` tool is absent locally; its dependency/license gate has not been run here. These local checks do not establish hosted CI or Security completion.

That fresh worker and host package passed both actual dense 27B and quantized MoE 35B lifecycle replays: automatic image checks, red/blue discrimination, stream cancellation after a delta, staged-media cleanup, text recovery and a fresh image. Both workers exited, and neither memory guard fired. Receipt SHA256 values are `3893a10dd6d491e3eda3a016a55f2bcd46bb06dc8dba512d7a88b7c240d767d1` and `927fdb52e4ce4b1f1806052179a84aad81fe748a6e69079d7fbe88dc2e6761fc`, respectively. These are functional observations, not quiet-host performance measurements.

The BF16 MoE whole-component and cumulative scratch-request assertions pass on that executable. The indexed reference-dot and selected full-bank route controls also pass on the preceding `cb7ae445` executable; its production engine code differs only in the subsequently corrected test fixture. The scratch counter measures requested bytes and excludes allocator rounding and overhead. It is a controlled component bound, not a full-model memory measurement.

## Numerical qualification

The affine Qwen production encoder now preserves the pinned source's BF16-effective coefficients and F32 contraction order without a dequantized weight bank. Both fixed 0.8B cases match all 248,320 logits bit for bit on sealed build `2529098a`: relative RMS zero, cosine one and zero mismatches. A separate scalar witness is also exact; all 32 projection comparisons are exact. The receipt SHA256 is `7fd2dc2d8124d0ae471ba88aef6bbd508f7391695b2bf235c63d388ddc34b9e3`. Explicit source arithmetic policies retain precedence, and the new encoder is selected only by the Qwen affine numerical profile. Earlier failed receipts remain preserved.

The retained Lily replay uses the existing 333 source visual tensors without loading the complete decoder. Measured vision-only repairs make all captured linear contractions, nonlinearities, position additions and production rotary controls exact; both GELU variants also match all 65,280 finite BF16 inputs. The smaller case is exact through 27 blocks. After rotary promotion, the larger case still fails at relative RMS `0.0274135105`. Its first captured production divergence is attention. On sealed `7e537d9d-lily`, all seven independent attention-order controls match their source tensors bit for bit, with zero observer mismatches; promotion and the unchanged full-tower replay are pending. Native peak RSS stays below 1 GiB and resource guards do not fire. The 54 additional normalization controls intentionally exercise the legacy baseline and do not measure the production Welford path. Full retained-model numerical behavior remains unqualified.

The larger Gemma tower has exact patch preparation and captured prepared Q/K/V, with zero padding tails. Measured QK contraction, fused denominator and compensated scalar-exponential repairs now make the complete captured first-layer attention bit-exact across 2,757,888 BF16 values on sealed `f67ab9d5` (SHA256 `3fc4760fd66bc61153d58a822038a08eded217bfea14f989e5ccb6cc2c8f3328`). Both prefix fixtures and all four canonical Flash/F32-QK tests pass. The independent scalar tests cover 8,215 inputs: CPU matches its pinned source libm; Metal matches the 100-digit nearest-F32 oracle, with 42 reported one-ULP differences from that libm. This is a bounded tested policy, not a universal correct-rounding claim.

The unchanged three full-feature cases now have relative RMS `0.0091937972`, `0.0132609557` and `0.0264942683`. The first two pass; the pattern case still fails the fixed 2% limit. Its first existing block-prefix divergence is at block 5. A direct current-main GELU finite-domain control is pending before adding a bounded layer capture. Same-input MLP down projection is exact. The trained-gain no-PLE synthetic control passes its source-bound CPU cases; the sharper unit-gain stress failure remains recorded. These controls do not qualify full 12B/26B/31B generation.

New qualification runs must retain distinct executable, component and reference identities. Performance and installed desktop UI execution are not established by these checks.
