# Gemma 4 no-PLE residual controls

The no-PLE BF16 layer now explicitly materializes the post-attention residual,
then adds the normalized MLP output and applies the learned layer scalar once.
`materialize_post_layer_scalar` defaults to false. The converter selects it only
for no-PLE BF16 Gemma layers. Existing PLE and legacy configurations retain their
selection. The optional MLP residual-input hook supplies the original residual
to architectures whose router and expert branch require it.

The private synthetic canonical control uses Torch 2.14.1, Transformers 5.15.1,
and `modeling_gemma4.py` SHA-256
`4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3`.
It has six no-PLE layers, production head widths 256/512, alternating local/global
attention, global K=V, and scalars -0.5, 0.625, -0.75, 0.875, 1.125, -0.25.
The controlled Q/K norm weights are 0.125, isolating residual topology from sharp
random attention transitions. This is component evidence.

CPU comparisons passed lengths 1, 9, 31 and 63 with greedy agreement in every
case. Centered logit RMS error was at most 0.002327 and centered maximum absolute
error at most 0.006566. The one-token result was exact. The identical-weight
control with legacy distributed scalar selection failed the same gates.
The reference manifest SHA-256 is
`fa12a9e0fd83526b37370c1f5cedf21bc55bdf8bb0d5e593d8e9517a610d6791`.
Receipts are retained under ignored `runtime/gemma4-nople-dense-*` paths.
The production CPU/Metal generator check and 28 Gemma converter tests passed.

The original unit-Q/K random fixture remains a failing stress control:
the one-token result is exact, but multi-token contexts diverge. A diagnostic
BF16 RoPE materialization reduced the nine-token error without resolving longer
contexts. The current no-PLE converter explicitly selects that default-false
materialization flag. Full 31B, 26B and 12B decoder qualification still requires
representative numerical evidence and exact released-artifact tests. These
controls do not establish full-checkpoint image generation or throughput.

A separate width-512 control uses the actual first-six-layer Q/K gains from
`google/gemma-4-31B-it` revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`. A bounded immutable range extraction
retains 285,363 metadata/header bytes and 7,168 tensor bytes. All 12 norms are
uniform over their logical channels: Q gains range from 0.94921875 to 1.03125,
local K gains from 0.12109375 to 0.1318359375, and the global K gain is
0.0615234375. These values differ materially from the original unit-gain stress.
The exact file/range/hash receipt is
`runtime/gemma4-31b-head-norm-source-v1/receipt.json`, SHA-256
`a72df06960003d00da71385bc4669232096ee7bb29fa8bf9ac234634fda1e496`.

`gemma4_nople_hf_reference.py --head-norm-source` creates a separately named
synthetic control. It preserves the alternating synthetic topology and chooses
the nearest extracted source layer with the matching head width; each synthetic
global layer therefore uses source layer 5. Tensor-by-tensor comparison verifies
that only the 12 Q/K norm tensors change from the original width-512 fixture.
The source model otherwise remains synthetic, including all attention and MLP
projection weights and layer scalars.

The source-informed control passes unchanged CPU logit gates on sealed engine
source `af356919` for lengths 1, 9, 31 and 63, with greedy agreement throughout.
Maximum centered relative RMS is 0.02158 and centered absolute error 0.03451;
the one-token result is exact. Its canonical reference manifest SHA-256 is
`8f7d638ce49a76ab53fdcefb09e884f1a963aa136c50a52678b5b9032974025e`.
The separate native receipt is
`runtime/gemma4-nople-trained-norm-width512-cpu-af356919-v1.json`, SHA-256
`9b9f18108d345d2aaf71d208050ec78a9f56a9427f27af33bfd3007280a23677`.
The original unit-Q/K failures and all numerical gates remain unchanged. This
adds component evidence at trained gain magnitudes; full-checkpoint decoder and
image generation remain unqualified.
