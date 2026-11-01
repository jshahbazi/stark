# IQ2_XS/IQ3_XXS vector-factored trial

Status: rejected. The candidate passed its numerical tests and increased GPU
time in each paired component comparison. Its shader, flag and test-only
support were removed from active code. Other packed-kernel changes remain.

The original Stark candidate split packed grids into two 32-bit words,
converted their bytes into float4 coefficients, applied signs to activation
bits, and multiplied the local eight-value dot by its shared scale. This
changes F32 association. Inputs above the proven bounded range used the
unchanged vector8 reference helper so factoring could not hide an overflowing
product. Stored weights and quantization were unchanged.

The component workload used 64 experts in eight rotating groups of eight,
covering 155 MB for IQ2_XS and 205 MB for IQ3_XXS. The gate shape was
2048 by 4096; the down shape was 4096 by 2048. Each math mode collected
12 warm GPU-duration samples per variant with alternating order. Uploads
were excluded. These are component times for eight selected experts.

| Math mode / format | Original vector8 | Factored float4 |
| --- | ---: | ---: |
| Safe / IQ2_XS | 85.62 us | 106.35 us |
| Safe / IQ3_XXS | 78.46 us | 91.82 us |
| Fast / IQ2_XS | 73.45 us | 90.31 us |
| Fast / IQ3_XXS | 155.67 us | 187.09 us |

Only comparisons within a math-mode run are paired evidence. The difference
between Safe and Fast remains a separate investigation. Numerical comparisons
and the candidate's targeted tests passed in both modes. No complete-model
gain is claimed.

Private aggregate logs are `runtime/glm53-optimization-20261101/iq-components-safe-v1.log`
and `iq-components-fast-v1.log`. They contain scalar measurements and errors.
The original implementation and its tests are retained in `rejected.patch`.
The patch applies to the immediate removal snapshot and passed `git apply --check`.
These snapshot digests record the workspace at removal; concurrent surrounding
optimization work means they are not retroactive build identities for the logs.

| Removal snapshot artifact | SHA-256 |
| --- | --- |
| Patch | `daaf66560f62fea06a2b743a1224e21c90e6861c991b4a6e8fb72338f08ff15a` |
| Candidate weight_metal.rs | `84693cea75ee0ce9154259a7f6ffe5c831a9284ed992f92c71156bb15a6abee4` |
| Restored weight_metal.rs | `3a5066e6e96c7bf8605be8f7689b4aefe58765344d2eca35a0a296f53d8a6dea` |
| Candidate shader | `41b5b945b1c4df029b757e749e702b08f52a06d7d791503bb271ff24bd44990a` |
| Hybrid bundle before removal | `c9f4bb1fe11c0963ccab936357a20558f83acf29e87a930e03cccfcfe1e53f20` |
| Hybrid bundle after removal | `7ec6d764e597ab4e0e984cdd3534996b90a2dd4ae4c3871293181f736bab6523` |

Bundle hashing sorts `.rs` and `.metal` files directly under the hybrid source
directory; each contributes relative UTF-8 path, NUL, bytes and NUL. GGML table
and layout provenance remains in `LICENSE.ggml-hybrid`. The trial is original
Stark code and adds no external implementation or dependency.
