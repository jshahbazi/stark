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

The original unit-Q/K random fixture remains a failing precision stress control:
the one-token result is exact, but multi-token contexts diverge. A diagnostic
BF16 RoPE materialization reduced the nine-token error without resolving longer
contexts. That diagnostic change is excluded from this checkpoint. Full 31B,
26B and 12B decoder qualification requires resolving this attention precision
gap and testing the exact released artifacts. These controls do not establish
full-checkpoint image generation or throughput.
