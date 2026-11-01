# Rejected mHC fused Control experiment

The opt-in `STARK_GLM53_CONTROL_FUSED=1` candidate combined Sinkhorn mixing, weighted input, and the final RMS into one 256-thread dispatch. It preserved stream and Sinkhorn chronology and the existing eight-SIMD RMS reduction order; ordered-RMS diagnostics retained the original path.

The candidate passed `fused_control_preserves_pre_post_bits_and_nonfinite_recovery` after the poison fixture was corrected to write nonfinite values after allocation. Coverage compared pre output, mixing coefficients, weighted input, and post output bits at four geometries and 1/8/16/32 chunks, checked nonfinite inputs/projections, and checked fresh-command recovery.

The isolated stage experiment used hidden 4096, four streams, ten controls per command, four measured replays after warmup, and a serial command encoder. Median GPU microseconds per control were:

| Projection chunks | Unfused pre/post | Fused pre/post |
| --- | ---: | ---: |
| 8 | 111.7563 | 120.8250 |
| 16 | 101.2938 | 109.6063 |
| 32 | 96.5979 | 97.0542 |

The candidate showed no admitted component speed gain and was removed from active runtime and tests. No whole-model performance or quality claim was established for this candidate. The retained `CONTROL_CHUNKS` choices and unfused stage timing harness remain active.

`rejected.patch` recreates the fused-only candidate from the restored source identities recorded in `aggregate.json`. That file retains aggregate measurements and SHA-256 identities for the candidate sources, restored sources, patch, and private evidence logs. Source copies and full logs stay in the ignored runtime directory. The hardware clock policy was not independently recorded, so these component observations do not establish a fixed-clock effect size.

This is original Stark source. The active runtime retains the separately measured chunk schedules and unfused timing harness. No candidate binaries, model data, or private vectors are included in this archive.
