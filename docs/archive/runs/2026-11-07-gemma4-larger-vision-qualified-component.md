# Pinned Gemma 4 31B vision component qualification

The extracted official 31B image tower and projector pass all three original
canonical feature gates on sealed source `e5dade43`. Full decoder execution and
image-conditioned language generation remain unqualified. The released 26B
weights require separate qualification despite sharing this tower architecture.

## Exact identities

| Item | Identity |
| --- | --- |
| Checkpoint | `google/gemma-4-31B-it@842da3794eaa0b77d5f08bae87a17459d91ff475` |
| Engine source tree | `d3b0348be50831472f09aa92a5c739f6a6caf767` |
| Sealed test executable SHA-256 | `85a98ef19fb3d6e20036105f5662014fc472b2106c50c89b024044eadbb49112` |
| Original canonical manifest SHA-256 | `fbe74d13fc83f0da2868fb289bf1716dd560fd1591fe15c98808cdffd231eb71` |
| Extracted source vision weights SHA-256 | `ae20f331d92a3347e21cbe885430213c60c9c98a4529319d2b39c53a45657ba8` |
| Corrected native component SHA-256 | `adcc1e2aca1cd1de0ffe84b2396ac4def0d6f3cc74e71dba04d7bd09afdb8be1` |
| Private qualification receipt SHA-256 | `ee85488d484611679c84508539efebd729417e159f3f0e3d5d530c59b0f6975e` |

Source and native weight hashes were freshly rechecked after execution. The
corrected component preserves all original tensor payloads and offsets; its
clone-only namespace correction and original artifact have separate receipts.
Private source images, tensors, executable and earlier failed receipts remain
retained outside version control.

## Unchanged gates and controls

The frozen gates require exact preprocessed patches, feature relative RMS at
most `0.02`, and cosine similarity at least `0.999`.

| Image | Relative RMS | Cosine similarity | Patches |
| --- | ---: | ---: | --- |
| Red | 0.0010468731 | 0.9999994520 | Exact |
| Blue | 0.0010244444 | 0.9999994753 | Exact |
| Pattern | 0.0011068880 | 0.9999993874 | Exact |

The full blue-image first-layer and pattern-image layer-3 Q/K/V and attention
outputs match their frozen BF16 references exactly. Same-canonical-input
attention replays also match exactly. Both prefix pairs and all four existing
Flash/F32-QK canonical tests pass. The new scalar source policy matches all
8,220 independent source-libm outputs on CPU and Metal; the original 8,215-case
nearest policy remains separately passing against its original goldens.

The larger-only path normalizes 72 logical channels before zero padding, uses
the measured RMS cascade and four-part F32 QK reduction, physically stores BF16
unnormalized exponential tiles, and keeps denominator/PV state in F32. Its
scalar denominator contraction and separately rounded PV rescaling have
independent controls. Scalar rescaling uses a mathematically derived
128-power/quadratic formula with generated constants and compensated F32 pairs.
The [Flash record](2026-11-07-gemma4-larger-flash-controls.md) preserves its
derivation, source controls and known limit: one one-ULP source-libm discrepancy
in 106,525 broader CPU cases. It does not establish universal libm equivalence.

The full component process maximum RSS was 1,237,057,536 bytes. This is a bounded
component observation, not full-model inference memory or a performance result.
The new scalar dispatch is restricted to Gemma 4 larger vision widths/model
dimensions; E2B/E4B, encoder-free 12B, Gemma 3 and text arithmetic are unchanged.

## Remaining scope

The [family assessment](2026-11-07-gemma4-image-family-scope.md) distinguishes the
implemented 12B/26B/31B converter/runtime paths, passing 12B and 31B image
components, and controlled MoE fixtures. Full checkpoint decoder/logit,
generation and worker lifecycle qualification remains pending for those larger
families. The original six-layer unit-Q/K synthetic stress failure is retained.
No component result removes that limit or qualifies released 26B expert/vision
weights. No PR or publication accompanies this checkpoint.
