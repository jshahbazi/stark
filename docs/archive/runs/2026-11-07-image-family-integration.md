# Image family integration checkpoints

This record separates implemented artifact paths, functional model checks and numerical qualification. The family contract and exact source identity remain necessary for every loaded artifact. No PR or release is part of this work.

## Coverage

| Path | Implementation and evidence |
| --- | --- |
| Qwen 3.5/3.8 dense affine, schema 3 v3 | Packed decoder matrices and a complete BF16 tower. The pinned 27B worker lifecycle and extracted tower checks pass on their recorded builds. The separate quantized 0.8B decoder reference gate remains open. |
| Qwen 3.5 affine MoE, schema 3 v4 | Router, selected expert banks, sigmoid-gated shared expert and BF16 tower. The pinned 35B-A3B worker lifecycle and extracted tower checks pass on their recorded builds. These do not transfer numerical qualification to other checkpoints. |
| Qwen 3.5 BF16-source MoE, schema 3 v5 | Distinct image-only conversion, full-bank admission, indexed reference dots and BF16 router/route boundaries. Native controlled component checks pass; full official 35B loading, parity and worker execution are unverified. |
| Retained Qwen Flash Next, schema 6 | Exact-source Lily image contract and native worker lifecycle. A new vision-only comparison fails one unchanged full-tower feature gate and remains under investigation. |
| Gemma 3, schema 5 | Original 4B/12B/27B BF16 wrapper admission and SigLIP processing. The pinned 4B numerical and worker/API lifecycle checks pass; full 12B/27B execution is unverified. |
| Gemma 4, schema 4 | Pinned E2B/E4B, encoder-free 12B, 26B-A4B MoE and 31B dense profiles. E2B/E4B have separate execution evidence; 12B embedder and 26B MoE component checks pass. The 31B feature gate and full larger-checkpoint generation remain open. |

Image requests clear and bypass prefix reuse and disable image MTP. Gemma text ring-state restoration remains a documented prefix-reuse blocker. See the [Qwen family record](2026-11-07-qwen-image-families.md), [BF16 MoE record](2026-11-07-qwen35-bf16-moe-images.md), [Gemma 3 record](2026-11-07-gemma3-images.md), and [Gemma 4 scope](2026-11-07-gemma4-image-family-scope.md).

## Integrated regression checks

The converter source tree `9029cb045338e4d29c9ada457d99fd1aceb1e3e8` passed **6,307 tests, with 96 skipped**, using the locked Python 3.13.13 environment. The sealed native template helper was explicitly discoverable on `PATH`; its SHA256 is `913ec72e2e94ae98cc75d098315f2d42abe02af068718c384662ab75cd236f6f`. A preceding run without a discoverable helper had invalid test setup and is not counted as regression evidence.

Engine source `6fb15e6d21479d313b1bbbcba14b8b428d7a3a7c` passed **924 CPU/Metal native tests, with 113 ignored and six benchmark cases filtered**, on the pinned nightly. The sealed test executable SHA256 is `1de43487d13e0fa754a35ed22375a393f07c87f96f692e301e59add929943d4e`. Ignored model-bound checks require their separate source-bound receipts.

The BF16 MoE whole-component and cumulative scratch-request assertions pass on that executable. The indexed reference-dot and selected full-bank route controls also pass on the preceding `cb7ae445` executable; its production engine code differs only in the subsequently corrected test fixture. The scratch counter measures requested bytes and excludes allocator rounding and overhead. It is a controlled component bound, not a full-model memory measurement.

## Open numerical comparisons

The affine Qwen split-K correction has a hard production-path cancellation witness. Same-input projection diagnostics improve to relative RMS `1.48e-5`–`3.68e-5`; those observational diagnostics do not themselves assert a numerical gate. The unchanged full quantized 0.8B logit gate still fails at relative RMS `0.05406` and `0.02264` on `cb7ae445`. Both cases retain their source references and failed receipts.

The retained Lily vision-only replay on source `c2d7d6cd` uses the existing 333 source visual tensors without loading the complete decoder. Zero/one-block controls are within the fixed gate. At 27 blocks, relative RMS is `0.02525` and `0.01884`; the first exceeds the unchanged 2% limit. Native peak RSS is below 1 GiB, and the resource guard does not fire. This does not qualify the full retained model's numerical behavior.

The larger Gemma tower replay has exact patch preparation and zero padding tails. Small normalization/attention differences accumulate across the 27 blocks; the existing full-feature failure is preserved. A measured normalization candidate and independent pinned-source fixtures are awaiting native replay. The trained-gain no-PLE synthetic control passes its source-bound CPU cases; the sharper unit-gain stress failure remains recorded. Neither control qualifies full 12B/26B/31B generation.

New qualification runs must retain distinct executable, component and reference identities. Performance and installed desktop UI execution are not established by these checks.
