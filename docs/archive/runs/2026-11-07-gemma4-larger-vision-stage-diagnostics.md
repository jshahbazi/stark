# Larger Gemma vision stage diagnostics

The integration engine test seal from source `af356919`, SHA-256
`1df910fb8508e7df8933bc2f63ff4f64837382c22874e121edf7e0dc23bcce03`,
passed all three 12B image-feature comparisons with exact patches and maximum
feature relative RMS 0.00007343. Full 12B decoder generation remains unqualified.

The 31B tower initially rejected the extracted native component because the
official source retains `*.linear.weight` wrappers even with clipping disabled.
Converter correction `a5eaa732` validates those source names and maps them to
plain native weight leaves. The original extracted artifact remains preserved.
A clonefile-only derivative changes 189 header names with unchanged payload
offsets and verifies every one of 356 tensor payload hashes against the original.
Measured physical allocation delta was 65,536 bytes, below the 64MiB bound, with
the 30GiB reserve retained. Corrected component SHA-256 is
`adcc1e2aca1cd1de0ffe84b2396ac4def0d6f3cc74e71dba04d7bd09afdb8be1`;
its padded header SHA-256 is
`efdbe7491b8e0c4c5fde7b086fd4ecd16cf5e114c0bea843c5709be0fee3a42f`.
The separate private receipt is
`runtime/gemma4-31b-vision-source-v1/native-vision-plain-names-v2.json`.

The corrected component loads and executes on the same seal but fails frozen
feature gates: relative RMS 0.07467, 0.09914 and 0.05464; cosines 0.99721,
0.99508 and 0.99851. All processed patches are exact. The 0.02 RMS/0.999 cosine
gates were retained. This result does not qualify either full 31B generation or
the shared larger vision tower used by 26B.

The next diagnostic captures patch projection/positions, first-layer raw and
logical 72-channel normalized Q/K/V, prepared head-zero rotary values, block
prefixes, the full encoder output, standardized output and projector input.
Native captures verify zero tails across all 128-padded heads. CPU controls
compare patch projection accumulation schedules and run pooling/standardization
from the canonical encoder output to isolate that step. Ordinary execution uses
a no-op observer; this checkpoint changes no arithmetic policy.

The source-bound three-case stage reference manifest is
`9730c112d000fb17f4891c8185c31390a11ad1be38193ca3294756f9f5721ec8`.
Stage vectors are capped at 64MiB; the total private reference folder is 88MiB
including ordinary patch/features. Original feature and patch hashes are
unchanged. Source generation and Python lint pass, and Rust formatting passes.
Native type-check and stage execution are pending at this observer checkpoint;
the integration coordinator will batch its build to preserve the disk reserve.

Independent CPU controls use the captured canonical stage inputs, without a
native build. Sequential F32 RMS with logical width 72 reproduces every captured
first-block Q/K/V normalized BF16 value in all three cases. The 1,152-wide input
and projector norms have at most seven differing elements in an 8-row prefix,
with relative RMS at most 0.000121. Sequential pooling/standardization fed the
canonical encoder output differs by relative RMS 0.000354–0.000375. These controls
do not explain the full feature failure and do not establish native stage parity.
Their private aggregate receipts are
`runtime/gemma4-31b-normalization-stage-oracles-v1.json` and
`runtime/gemma4-31b-pool-stage-oracles-v1.json`.

Separate sealed replay receipts preserve each outcome and freshly verify the
binary and component identities. The 12B receipt
`runtime/gemma4-12b-sealed-component-af356919-v1.json` has SHA-256
`0a0f1bd30f98b84e0bbd47b09afb6ccd468cdf2c77586dd0b82560772fb67be4`.
The failed 31B receipt
`runtime/gemma4-31b-sealed-component-af356919-v1.json` has SHA-256
`87e9371cb8ca31b35ad5af03b22069eac08a8f6877d571d03bf88cd7eb330b06`.
