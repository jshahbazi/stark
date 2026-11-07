# Larger Gemma MLP source controls

Sealed source `214c19a9`, executable SHA-256
`57d4922eec60a56d634ad3482ecc7088770a4bf073a98d24b62fcd92e2f1a804`,
passes independent F32 QK witnesses and six Flash cases on CPU and Metal. Its
canonical first-eight-query prefix matches all 16 heads exactly on both backends.
Full first-layer attention has six differing BF16 values out of 2,757,888,
relative RMS 0.00000552, with exact prepared Q/K/V. Full feature gates remain
failing: relative RMS 0.03685, 0.04938 and 0.03614 against the unchanged 0.02
limit. All original failed receipts remain preserved.

The next diagnostic measures first-block pre-FF norm, gate before/after GELU,
up projection and product prefixes, plus full 4304-wide down-projection input
and 1152-wide raw output. It also runs raw down projection directly on the
canonical source input to distinguish projection rounding from upstream drift.
Observer copies introduce no arithmetic changes; ordinary execution keeps a
no-op observer. These measurements do not relax numerical gates.

The source-bound one-case stage capture totals 52,834,432 bytes, below the
unchanged 64 MiB cap. All three original image, patch and feature hashes match
the earlier canonical reference. Its manifest SHA-256 is
`509d2d48ddb13d6c19753633f90aae6989054c9c0ee5375ee51d902c18b30212`.
All vectors remain private in the ignored runtime directory.

An independent CPU equation check of the first eight canonical input rows
matches every source raw down output with the existing 32-stream accumulation
and 16-channel tail. Direct pinned BF16 Torch Linear also matches exactly; the
four-stream control differs in one value. This prefix evidence does not establish
full-row projection parity. No dense projection policy changes are proposed.

Native diagnostic type-check and Metal execution are pending at this source-only
checkpoint. Full 31B/26B image features and full-checkpoint generation remain
unqualified; earlier 12B image-component and E4B checkpoint evidence keep their
separate scope and source identities.

The existing full-attention split now preserves the remaining mismatch
coordinates in a private runtime JSON file and replays only the affected image
query rows, across all 16 heads and full canonical KV, on CPU and Metal. The
diagnostic caps this replay at six distinct query rows and reuses the frozen
full-attention fixture; no new oracle or tolerance is introduced. Only counts,
hashes and aggregate errors are printed. This separates source-order differences
from GPU-only rounding at ties missed by the first-eight-row prefix.
