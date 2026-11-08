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

The candidate subsequently passed both six-case native CPU/Metal fixtures on
sealed source `32b33e16`, executable SHA-256
`aa97099bbfc19cc87b66282d9fe45ef7c2dc25e47a2aa3d07313721c68525b12`.
Full larger-tower feature gates remain failing, with relative RMS
0.05385, 0.07353 and 0.04158. First-block attention prefix RMS improves to
0.00000509, 0.00008753 and exactly zero; case 2's block-zero prefix is also
exact. These results do not establish all-head or later-row parity. Observed
maximum process RSS is 1,318,469,632 bytes for the stage diagnostic and
1,242,267,648 bytes for the full feature replay; this excludes any claim about
total GPU allocations or quiet-host performance.

A new source-bound diagnostic captures all 16 heads and all 2,394 valid patch
rows for case 1's first prepared Q/K/V and attention output. These four F32
stage planes total 44,126,208 bytes, below the unchanged 64 MiB stage cap. All
three original image, patch and feature hashes remain unchanged. Its manifest
SHA-256 is
`0c37a150fa44dfcddf17c6dd30f0418b7032d6a57eecaa4a6bd7271700667b19`.
The new ignored Metal split test compares tower-prepared values with those source
planes, then runs the native Flash component on the exact canonical Q/K/V. It
prints aggregate comparisons only. Native type-check and split execution are
pending for this diagnostic-only checkpoint; no arithmetic policy changes.

The full split passed its diagnostic execution on sealed source `36414711`,
SHA-256 `8b997f9d72b5200818169477da7ab2995b800a8970fa0f7f2471833bafb6faa1`.
All 2,757,888 prepared Q values, K values and V values match exactly. Native
attention differs in 513 BF16 elements, beginning at image-patch row 3, with
relative RMS 0.00007203. Replaying the exact canonical Q/K/V through the native
Flash component reproduces the same 513 differences, isolating attention.
The old split report and full feature failures remain preserved separately.

The next witness distinguishes CPUBlas' F32-output and BF16-output QK routes.
Pinned `BlasKernel.cpp`'s no-downcast F32 output uses the generic four-part sum;
its 32-stream `compute_dot` specialization accepts BF16 output. An independent
Python replay of the first eight canonical query rows across all heads has five
differences with the 32-stream QK schedule (heads 14 and 15), and none when only
QK changes to the four-part F32-output schedule. Direct pinned Torch attention
also matches every expected prefix value. No other equation changes in this
comparison.

The candidate changes only larger-Flash QK's CPU/Metal sum to that source
four-part schedule. The six model-free output cases and their 776 probability
rounding witnesses are retained byte-for-byte. A new independent score witness
uses one unmasked key in a padded 2520-key pinned Torch Flash call: its F32
logsumexp equals QK exactly. This exposes 52 F32 differences from the old
32-stream schedule across 256 queries without reimplementing QK as the oracle.
The updated model-free manifest SHA-256 is
`50e5a3de58978897ef3ff059e13f0ea9fbea3be99cf3ba60eaa848ab4372e919`.
New native score and all-head-prefix tests, plus full attention/feature replay,
are pending for this source-only QK checkpoint. E2B/E4B, unified 12B, text and
all BF16-output dense projections retain their paths and all gates remain fixed.

The affected-row diagnostic on sealed source `a1649a6a` isolates six image query
rows: CPU differs in one BF16 value, while Metal differs in six. Direct pinned
Torch Flash matches the frozen outputs for these rows with 1, 2 and 4 CPU
threads and both original-like and head-contiguous strides. Thus source query
batching does not explain the mismatch. An independent CPU equation control
has one difference with a separately rounded denominator update and none with
`fmaf(factor, previous_sum, tile_sum)`. Prior PV rescaling stays separately
rounded before the dot/add, matching the source loop structure.
The private control report SHA-256 is
`714fec6d7d3d839959d9e374380ac905a754b81e92363d39cb16d7beeb23072d`.

The next candidate makes only that larger-Flash denominator contraction explicit
on CPU and Metal. Native validation is pending. A diagnostic-only reciprocal
kernel and no-op observer support private F32 QK/state/PV, BF16 exponential and
F32 reciprocal traces for only affected row/head pairs. Trace storage is capped
at 256 KiB per backend and reuses the existing canonical tensors; it introduces
no new oracle, large fixture or production scratch plane. Only aggregate counts
and hashes are printed. The remaining Metal-specific differences still require
these controls before any additional arithmetic change.
