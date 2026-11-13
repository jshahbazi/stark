# GLM KDA segmented reduction candidate: declined numerical qualification

This record preserves a superseded experiment; it is not evidence for the
replacement chronological implementation. The source base was
`e0b12e2f3b6af47d39cc30e4018b9e73b190caa4` plus uncommitted experimental
prefill changes. No model data or vectors are included.

The first register KDA candidate split each 128-key dot into eight ordered
16-key partials followed by an XOR tree. It always used ordered q/k and output
RMS norms, including when the existing optimized reference selected parallel
norms. Activations and recurrent state remained F32. Synthetic operator
tolerance evidence did not establish full-model qualification.

The coordinating validation run reported the following full-checkpoint
513-token comparison against the existing optimized T8 path, with the new
candidate using T64, batched hyper-connections and register KDA:

| Metric | Reported result |
| --- | ---: |
| Maximum absolute readout difference | 0.6934890747 |
| Relative L2 readout difference | 0.03130855 |
| Declared per-element acceptance | `1e-3 + 1e-3 * abs(reference)` |
| Numerical acceptance | Failed |
| Reported top-1 agreement | 5/5 |

Shorter 17-, 65- and 129-token comparisons against the scalar path had reported
maximum absolute differences approximately `2e-5` to `4.5e-5` and relative L2
approximately `1e-6`. The coordinating run also reported a similar 513-token
difference against scalar execution. The matched control then compared
T64 plus batched hyper-connections with register KDA disabled against old T8
at 513 tokens: all `5 * 154880` readout values matched exactly, with maximum
absolute difference zero, relative L2 zero and identical readout hashes.
This diagnostic isolates the difference to the first register KDA path's
arithmetic policy. The harness source changed while the model was running,
so these coordinating results are diagnostic evidence rather than a final
qualification bound to a frozen harness and executable. Top-1 agreement does
not waive the declared readout threshold.

The superseded source files were retained privately under ignored
`runtime/glm53-kda-segment-tree-20261103/` before replacement:

| File | SHA-256 |
| --- | --- |
| `glm_kda_prefill.rs` | `9b97dca7cf45c4a8c39903af4f81e761b27e6cc8e62a4e43e7be821bfc0d20cc` |
| `glm_kda_prefill.metal` | `de15db0ae07fa8e0ff9479363b741aecda059e8db248a119b2727cff4f3ba09b` |
| `glm_kda_prefill_tests.rs` | `366b36f0abefe52a1c49b2776d859408c8ac156ca6defee4fbfda487e9d3df65` |

The replacement retains register state and restores chronological dots plus
the existing selected norm policy. Its operator, full-model and performance
results require separate validation.
