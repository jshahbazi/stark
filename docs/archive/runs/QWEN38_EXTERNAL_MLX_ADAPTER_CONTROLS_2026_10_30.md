# Qwen3.8 external MLX adapter controls, 2026-10-30

This record qualifies loader fixtures, GPU components and the retained Lily
route. **Full official-checkpoint admission and a matched throughput comparison
remain pending the checkpoint transfer.** It supplies no new throughput claim.

Follow-up: the [same-checkpoint benchmark](../../benchmarks/QWEN38_SAME_CHECKPOINT_STARK_MLX_SERVE_2026_10_30.md)
records the subsequently completed transfer, real model checks, numerical
differences and throughput. The component evidence below remains bound to
its original validation stage.

## Implementation and scope

The native mixed-precision route now recognizes the external-table affine
storage contract used by
`ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit` at revision
`788c07a8725642bccf13a0a3b8a21954ab84b4f2`. Recognition requires the implemented
architecture, affine Q4/group-64 defaults and `ngram_table.bin` in
Q4/group-32 storage. The core independently validates that contract and the
individual tensor widths and shapes.

Tensor aliases retain source files, byte intervals, dtypes and shapes. The
external n-gram table becomes one mapped logical shard. Small BF16 gates stay
BF16 and use F32 accumulation. No source files or quantized weight values are
rewritten. Original Lily inputs retain their existing Q4/Q8 dispatch and
delta-weight normalization. The external pack uses separate folded-gain
attention/indexer variants; GDN's plain-gain normalization is unchanged.

Loading remains text-only, with resident experts, paged n-gram rows, BF16
activations, F32 reductions/logits/GDN state, Q8 attention KV and speculation
disabled. Unsupported contracts and alias collisions are rejected.

The PLE hash-control derivation follows MLX-Serve 26.10.3, revision
`046b647686419df06c6c7132fb1d15d76185c3e5`. Its MIT notice is retained and
included in distribution notice packaging. See
[core provenance](../../../vendor/lily-core/UPSTREAM.md).

## Validation

On the Apple M5 Max host, with Rust 1.98.1 release builds:

- Seven official-adapter controls passed: format refusal, PLE reference hash
  controls, source-byte aliases, BF16 projection, HC gain conventions, fused
  Q/K gain conventions with BF16/Q8 KV, and folded QSA entry points.
- Three retained GPU controls passed exact parity for cached HC reads, fused
  GDN projection/convolution and fused MoE down/injection.
- Two engine format-selection tests passed with the pinned nightly harness.
- The core suite before adding the final QSA control had 22 passes and nine
  ignored tests. The later seven-control filter above includes that QSA test.
- Rust formatting and repository policy checks passed. The focused
  distribution-notice test passed.
- A real Lily q4-xl control generated 256 sampled tokens from the previous
  4,096-token corpus prefix, without speculation, and reproduced the previous
  output-token SHA256 exactly:
  `621ca90417af420aaf9defdc957f7cc1f57f318c08d908be764834abb382b0d0`.
  Downloads were active, so this run does not qualify throughput.
- A header-only audit covered 460 projection matrices across 50 available
  shard headers with zero precision/layout deviations. Remaining headers and
  complete weight payloads were not yet qualified by that audit.

Independent review found a missing folded-gain case in fused attention decode.
It was repaired before the final build and Q/K GPU controls. The repaired
source review found no further blocking correctness issue.

## Artifact identities

The patch is based on commit `5fcd0e9cad94a5733a1430609754269fff7b5f50`.
Private controllers, manifests and aggregate validation artifacts remain
ignored; prompts, responses, token vectors and probabilities are not published.

| Artifact | SHA256 |
| --- | --- |
| Rebuilt generation worker | `0f95ff0d43cebedcc8f622b16d7b3e15eafb80b08abb4f03f8e8202241169fe0` |
| Source-file manifest v2 | `5a3c6a40e83b2bffb5d2a7b87af53072ef04cea03d9ee415a202aaa1ed307889` |
| Source/build receipt v2 | `d7abf3e1d65e699db4f11808490fe461540802cf5b6dce9ee9124f4f0ffa2735` |
| Partial header audit | `4940f91f1867e5486fe21f5502295922efce3ecea1276ed23ba8b1c765cd754a` |
