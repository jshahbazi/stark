# Retained Lily image resolution

Attempt ID: `20261110-LILY-SIZE-01`. Baseline: `92289590`.
State: implemented, component and bounded model execution passed; signed local
build installed. Not merged or released. Full 16 MP model execution unqualified.

The Playground report persists in a new chat. The supplied error screenshot is
1344 x 1788 pixels; it is evidence of the error UI, not necessarily the exact
image originally submitted. Native preprocessing rejects normal images above
1,048,576 rounded pixels before decoding. The retained Lily kernels separately
reject more than 8,192 patches. The source processor permits 16,777,216 pixels.

Remove the extra host restriction for the retained Lily profile and extend its
checked row dispatches to the source processor's 65,536 patches / 16,384 merged
image tokens. Keep the pinned smart-resize algorithm and source processor
parameters, source bounds, format checks, context admission and memory checks.
Other families retain their separately admitted limits. No precision changes.

Acceptance: preserve small-image resize and arithmetic; cover rectangular and
maximum grids, max-plus-one rejection, metadata/host handshake, checked memory
inventory, and an actual screenshot-size image request. Full 16 MP model quality
and performance are separate from component bounds and short execution smoke.
Image requests retain zero drafts and clear text prefix state; do not enable
image prefix reuse. Text paths and their qualification remain separate.

Reference: pinned processor configuration and resize implementation already
recorded in `crates/stark-engine/src/engine/vision/preprocess.rs`; retained native
kernel provenance in `vendor/lily-core/UPSTREAM.md`. Model artifact, build hashes,
actual checks and outstanding qualification will be recorded below. No new
performance or full-model numerical parity claim is intended.

## Results and identities

On 2026-11-10, Apple M5 Max, 128 GiB RAM, macOS 26.7.2, Rust 1.98.1
(nightly-2026-10-01 for the engine harness), locked converter Python 3.13.13:

- Repository policy, Rust formatting, converter lint/format and generation
  clippy passed. Generation worker: 144 passed, five private-artifact controls
  ignored. Image component CPU suite: 51 passed, including the opt-in private
  screenshot preparation. Original 8000 x 4000 expected resize in the new test
  was incorrect (5760 x 2880); the pinned algorithm gives 5792 x 2880. Corrected
  the test oracle and reran the selected suite.
- Converter suite: 6,937 passed, 94 skipped. Additional loaded-limit propagation
  control passed in the 67-test attachment suite. Native controller harness:
  140 checks passed, including switching advertised limits and clearing stale
  limits on Stop. Existing image transaction/recovery tests remain passing.
- Retained allocation inventory control passed at 65,536 patches, including
  invalid extents. Metal normalization and GELU executed over all 65,536 rows
  and produced the expected constant output. This is an indexing/allocation
  component check, not full tower or decoder parity at that resolution.
- Actual retained Lily source revision
  `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, Q4-XL decoder and resident BF16
  tower (27 layers, width 1152, intermediate 4304, 16 heads): private 1344 x 1788
  PNG became 1344 x 1792, 9,408 patches and 2,352 image tokens. With a synthetic
  prior text turn, greedy image generation at context 4096/output 16 returned
  2,391 input tokens, 16 output tokens, `length`, and zero reused tokens. The
  complete prepare/generate/discard transaction and worker teardown succeeded.
  Output SHA-256: `daf89d7efa3cdd175b368139b8f0792c0584cbe2bff29986430db98bf8e9fd37`.
  This establishes execution, not correctness of the image description.

| Identity | SHA-256 |
| --- | --- |
| Model config | `2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899` |
| Processor | `022f906548d5a89d3a18c61c0a87de760db8c81d618437501baa3dcdc14d4ecd` |
| Image artifact manifest file | `cd1e467d66062bdeeea841123ccdef48a84a3a60f468f3a88004e129c9768092` |
| Tested source worker | `c0c467bec3062fbfb5651a1c5ba8fcd9007af9e7875464c85cb238687556e47d` |
| Installed signed worker | `316c434d0899428330bc39d5a71b477eeccd0a7cb66abe8aa4eb363560669c5b` |

Local app build `20261110134459258851` uses Developer ID, hardened runtime and
secure timestamp; deep/strict signature verification passed after installation.
Its manifest records the baseline revision, dirty source file hashes and worker
overlays. Signed bytes differ from the pre-signing source-worker hash above.
The installed app loaded the retained model to Ready. Playground's live
accessibility state showed 16,777,216 processed pixels and attachment help
reported 32,000,000 source pixels / 16,384 image tokens with memory/context
conditions. The model was left Ready in Playground.

## Model-specific application limits and remaining evidence

The loaded worker advertises its implementation's source-pixel, processed-pixel,
patch and merged-token limits. Desktop forwards these values; Playground shows
the current processing ceiling and attachment help describes source/context
bounds. It replaces the displayed limits on each Ready/Configured event and
clears them when image support stops. It does not substitute a global 16 MP
limit for other families. Generic Qwen and Gemma retain their existing backend
and family limits; this attempt does not qualify wider eager-attention shapes.
The 4 MiB app transport bound is separate from processor resolution.

The retained 16,777,216-pixel ceiling is conditional on the source format bounds,
single-buffer device limit, available memory and remaining context. Full tower
latency scales quadratically despite streaming storage; 16 MP latency, peak
RSS, full-resolution reference parity, image-loaded text timing, and fresh
large-image-to-text MTP/cache recovery have not been measured in this attempt.
The other retained mixed-precision checkpoint has source/contract coverage
only, without a new execution run. Image prefix reuse remains disabled because
image embeddings and rotary state lack a qualified restore path.
