# Gemma 4 12B encoder-free vision controls

The pinned `google/gemma-4-12B-it` source at
`43c9d014b231474fa7212dbb48a762566c2b255d` now has a separate schema-4
`gemma4-12b-bf16-v1` converter/runtime profile. Its native image path uses raw
rescaled RGB patches of size 48, patch LayerNorm, biased projection, another
LayerNorm, factorized learned positions and positional LayerNorm, then
weightless RMS and the final projection. The teacher grid remains 16px with a
3x3 merge, preserving image-token accounting. Sliding layers use image-block bidirectional
prefill with the source lower window bound; global layers remain causal. A square 256px check image resizes to 768px and produces 256 image tokens.

The immutable checkpoint stores the patch/position parameters under
`model.vision_embedder.*` and the final projection under `model.embed_vision.*`.
Transformers 5.15.1 nests the final projector inside its vision module. The
reference performs an explicit namespace map and strict state loading for all
ten tensors. It never loads an incomplete model with random missing weights.
Audio/video remain outside image execution.

Only selected image tensors were read using strict HTTP byte ranges: 99,844,608
payload bytes, from source header SHA-256
`d8b49f990225888acc5734bd55e2c269404d24b5a7dc2d3122be4f75500d8fb4`.
The extracted raw file SHA-256 is
`2abaac53d40eb97238806db9c64b83a82b1eb22aca7cef82f0e38a9d77913d3a`;
native image-only tensor file SHA-256 is
`6cd4e8775165e1f1e9459d2246e1ad693709017cdaadc7ad693196f7e812ae3c`.
No full 12B checkpoint was downloaded or converted.

The canonical CPU reference uses Torch 2.14.1, Transformers 5.15.1 and unified
model source SHA-256
`c52d9b27fc83b6ced166aee816b235e1db057fde10df6c1eff64bbc91fb16f39`.
Its three-case reference manifest SHA-256 is
`186201280c6156ae1530d66043808d97ef3967bb83e078976774bd39b75caac8`.
The native Metal component comparison passed all three images: processed
patches were exact, feature relative RMS error was at most 0.00007343 and cosine
similarity at least 0.9999999973. Private source, vectors and aggregate receipts
remain under ignored `runtime/gemma4-12b-*` paths. This is image-feature
component evidence from the implementation worktree; final-source replay is
recorded separately.

The host recognizes this profile's own tokenizer, processor and template hashes.
Its memory preflight accounts for raw 48px patch buffers and projected/normalized
buffers, without inventing transformer vision layers or score tensors. The
native file cap is 32GiB. The separate 26B profile uses the checked sparse/dense
composite and larger standardized vision encoder, with a 60GiB file cap.
These caps are admission bounds, not peak inference-memory estimates.

Validation includes a production CPU/Metal generator check, 20 focused CPU
Gemma tests (one opt-in fixture ignored), 309 focused host/converter tests, and
the full Python suite (6,247 passed, 96 skipped). The shared per-layer image-mask
policy has independent CPU/Metal grouped-softmax oracle coverage. The no-PLE
converter emits `windowed_same_image` for local layers and `causal` for global
layers; image admission rejects absent or conflicting choices. Online language
attention keeps F32 scores and probabilities, with BF16 activation/cache/output
boundaries. The explicit rotary control materializes BF16 coefficients/products.

Full 12B/26B image generation and full-checkpoint language parity are unqualified.
The six-layer unit-Q/K no-PLE precision stress control remains unresolved, and
the disk reserve prevents retaining full larger source/artifact pairs. See the
[residual](2026-11-07-gemma4-nople-residual-controls.md) and
[MoE](2026-11-07-gemma4-moe-component-controls.md) controls for that scope.

The added width-512 six-layer unit-Q/K control also remains a failing stress
case: lengths 1 and 9 pass, while 31 and 63 fail. The default CPU SDPA reference
manifest is `2431a5c29c4b08d2699a2591d509ef7366f1b3e1f9dffa02549033d904b8f02b`.
Native F32 online scores/probabilities produce centered logit RMS errors
0.318832/0.312848 on the failing lengths. Explicit BF16 score/probability
materialization increases them to 0.434641/0.472615 and remains disabled for
canonical Gemma. A separately retained Torch SDPA math reference still fails
(0.276836/0.259122). A direct text-mask probe matches the native lower-window
predicate. These observations narrow the investigation without attributing
the complete drift to a single intermediate precision choice.

The opt-in layerwise CPU diagnostic captures raw post-layer residuals through
the existing decoder feature API. All six one-token residuals are exact. The
nine-token first layer is exact, increasing to relative residual RMS 0.0271 at
layer five. For length 31, the first-layer relative RMS is 0.001935 and grows to
0.51951 at layer five. First differences occur at row one, including 1,381
elements in the first 16 rows; they do not begin precisely at the window
boundary. These fixtures disable shared KV layers and start from fresh state.
This narrows the investigation without ruling out later cache or numerical
effects. The diagnostic reports measurements and does not apply a parity gate.

The retained E4B source provides a separate trained-scale calibration: its first
six learned query-norm RMS values are approximately 0.949–1.023, local key-norm
values 0.123–0.132, and the sixth/global key norm 0.0610. Both-unit Q/K synthetic
stress and both-0.125 topology controls should therefore remain distinct from
trained-scale component evidence and from any full no-PLE checkpoint claim.
