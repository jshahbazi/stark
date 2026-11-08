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

Sealed source `96b6da7d` verifies the fused denominator update: both prefix
controls and all four canonical Flash/F32-QK controls pass. The affected CPU
replay is exact; full same-input Metal attention improves from six to four BF16
differences. All captured F32 QK and physical BF16 exponential tiles agree across
CPU and Metal. The first divergence is the scalar rescaling exponential, before
PV: nine of twenty factors differ by one F32 ULP. CPU matches Darwin `expf` and
100-digit decimal exponential rounded to F32 for all twenty factors. Replaying
only CPU versus Metal factors reproduces respectively zero versus four final
BF16 differences, with all native denominator/PV states reproduced bit for bit.
No separate reciprocal defect is demonstrated.

The next larger-only Metal candidate replaces that scalar exponential with
compensated range reduction and a degree-12 Taylor polynomial. CPU retains libm;
the BF16 tile exponential is unchanged. Integer rounding in subnormal units
preserves underflow values without depending on arithmetic subnormal storage.
The independent private fixture contains 8,215 inputs (seeded random, uniform
grid, twenty captured factors, zero and negative infinity), totaling 98,580
tensor bytes. Its SHA-256 is
`fb77820532a29404499e2f17287d48464061c1b1d7bb5a693dbeae04d6972ad6`.
It freezes both 100-digit decimal-nearest and actual source-libm outputs.
Darwin libm differs from decimal-nearest by one ULP in 42 cases; the twenty
observed factors agree. The Python candidate matches decimal-nearest in every
case. Native CPU must match its retained source-libm policy; native Metal must
match its declared nearest-decimal policy. Source differences are reported with
a one-ULP bound. These finite controls do not establish universally correct
rounding. Native validation and the unchanged full-attention/full-feature gates
remain pending for this source-only scalar candidate. Historical failed gates
are retained.

Sealed source `f67ab9d5` (engine tree
`52cdc08f047e6e93b8c58bb700912229e5c410aa`, executable SHA-256
`3fc4760fd66bc61153d58a822038a08eded217bfea14f989e5ccb6cc2c8f3328`)
passes both 8,215-case native scalar controls, all four Flash/F32-QK controls
and both canonical prefix controls. The full blue-image first-layer attention
now matches all 2,757,888 BF16 values exactly; Q/K/V remain exact.

The unchanged three-image full-feature gates remain **failed**. Relative RMS
errors are `0.0091937972`, `0.0132609557`, and `0.0264942683`; cosines are
`0.99995776`, `0.99991207`, and `0.99964903`. Patches are exact. The first two
images pass the frozen RMS `0.02` and cosine `0.999` gates; the pattern image
fails RMS. Existing stage captures show its first eight-row block-prefix drift
at block 5 (zero-based), after exact prefixes for blocks 0–4. These prefixes do
not establish full-row agreement in earlier blocks. Larger-tower qualification
therefore remains incomplete; no tolerance or source-reference change is made.

Whole-pattern block replay subsequently establishes blocks 0–2 exactly, with the
first full-block difference at block 3. The layer-3 split on sealed `b003b291`
finds exact Q/K/V and one BF16 attention difference. Its CPU replay is exact;
the Metal nearest policy differs once. Here the source scalar libm itself is one
ULP from decimal-nearest. Substituting only source versus Metal factors reproduces
zero versus one final BF16 difference, with all denominator/PV states exact.
The original nearest helper and its independent passing tests are retained.

A subsequent source-compatibility candidate uses mathematically generated
128-entry powers and quadratic `1+r*(a+a³/32+r*a²/2)`, `a=ln(2)/128`, evaluated
with compensated F32 pairs. This is a generic independently derived formula;
no proprietary host code, tuned constants, table bytes or input-specific fixes
are copied. Licensed ARM table/range-reduction references and the derivation are
recorded in `provenance/UPSTREAM.md`. Host source identity is not claimed.

The candidate agrees with all original 8,215 source-libm goldens plus five actual
layer-3 factors. Their separate immutable fixture is 98,640 bytes, manifest
SHA-256 `b4f24f125d2fc08966c6f3b29e9b8fa17db002494d1cbdc832cadff0963d46c5`.
The old goldens are unchanged. Broader CPU coverage has 106,525 inputs: the
compensated prototype matches the F64 mathematical formula everywhere, and both
differ from source libm in one one-ULP result. That strict-source counterexample
is retained privately, SHA-256
`1bb4d3b41ee73d5a9c43d4ce1092a86181ea30e58d0331f89ac161fdfde7d8ce`.
This is finite numerical evidence with a known counterexample, not universal
libm emulation. Native source-bit controls and the unchanged three full-feature
gates remain pending for this source-only candidate.
