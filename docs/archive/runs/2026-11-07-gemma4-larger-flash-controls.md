# Larger Gemma vision Flash source controls

The checked larger-tower RMS correction moves the first measured difference to
attention on sealed source `fcadc29f`. The frozen full feature gates still fail.
No tolerance is changed. E2B/E4B and unified12B do not select this larger-tower
attention implementation; text attention is separate.

Pinned Torch 2.14.1 CPU Flash attention uses logical 72-channel F32 QK, 512-key
tiles, the NEON third-degree `fexp_u20` approximation, BF16 unnormalized
exponential values before PV, F32 running max/denominator/output state and a
final reciprocal. This differs from ordinary F32 probabilities and from BF16
globally normalized probabilities. Canonical processor padding extends the key
axis to 2520 (a multiple of eight), so compressed ragged valid tails must retain
the source vector exponential policy and eight-lane denominator fold.

Replaying the exact canonical prepared head-zero Q/K/V with the pinned Torch
operator reproduces all three captured attention prefixes exactly. An independent
Python implementation of the above source equation also reproduces every value
exactly; replacing its fast exponential with ordinary exp leaves differences.
The private aggregate is `runtime/gemma4-31b-attention-stage-oracles-v5.json`.

The candidate runtime has four separate kernels: full logical 72 QK for one
fixed 512-key plane, exponential/state update, four-stream PV and final BF16
output. Integer RNE preserves BF16 exponential materialization, followed by a
physical BF16 buffer passed into a separately dispatched PV pipeline. Q/K/V
physical 128-channel tails are excluded from dot products. No NxN plane is
allocated. At 2520 queries and 16 heads, requested score/exp/state/accumulator
planes round to 128 + 64 + 0.5 + 16 MiB under `PoolAllocator`, plus 8 MiB output. The four
scratch handles are allocated once outside the key-tile loop and dropped before
the attention method returns. Pool drops return pages immediately, and Metal's
range access tracker inserts Device dispatch barriers before overlapping reuse,
including later layers. The existing image-check allowance already includes
approximately 388 MiB of attention storage plus 210 MiB of other vision scratch.

The independent model-free fixture calls pinned Torch's
`aten._scaled_dot_product_flash_attention_for_cpu` directly with masked 2520-key
tensors. It covers 31, 511, 513, 1025, 2394 and 2520 valid keys, 8 query rows, two distinct
heads, logical 72/physical 128 layout and nonzero padding canaries. Its outputs
contain 776 rounding witnesses against ordinary F32 probabilities. Every expected
BF16 value matches a separate Python transcription of the fixed 512-plane
equation. The fixture manifest SHA-256 is
`19bbe422ed239b641fe7f1d07b878c686a67570d798498988f70d9b2c0c8b010`.
No checkpoint vectors are committed.

Native CPU/Metal fixture execution and type-check are pending at this source-only
checkpoint. The coordinator will build the checkpoint before any larger-tower
stage or full-feature replay. These source controls do not qualify full 31B/26B
generation, image features or model throughput.
