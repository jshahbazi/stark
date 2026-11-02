# GLM-5.3 T2 Compact-Timer Probe v2 (2026-11-02)

## Result

This is a compact component-timing refinement of T2v1, not a full-request result. The layer-major two-token path had a 68.854958 ms median wall time versus 71.3373125 ms for two serial steps, a 3.48% component-only median reduction. Both timed variants perform the same GPU work. V2 moved full-vocabulary CPU copy and argmax work outside the timer; it did not time any full-logit readback rows. The difference in GPU medians is unmatched and unexplained, and is not attributed to moving CPU work outside the timed scope.

The measurement does not include head catch-up or rollback and does not establish speculative decoding throughput, model-wide speedup, runtime memory admission, or a production change.

## Parity and timed scope

Before timing, the run passed 192 complete hidden-state and full-logit comparisons. After timing, it passed 192 compact choice and hidden-state comparisons. It also checked eight ordinary serial API alignment rows, four MLA pool residues, and four pooled alignment pairs. There were 36 measured serial pairs and 36 measured layer-major pairs after one warmup round; no full-logit readback rows were timed and no rollback calls were made.

| Aggregate metric | Two compact serial steps | Compact layer-major two-token graph |
| --- | ---: | ---: |
| Median wall time (ms) | 71.3373125 | 68.85495800000001 |
| Arithmetic mean wall time (ms) | 72.03129744444445 | 69.27951258333331 |
| Median GPU time (ms) | 64.79193750419654 | 61.195354152005166 |
| Arithmetic mean GPU time (ms) | 65.31519676193258 | 61.57700925299044 |

The arithmetic means are the recorded totals divided by 36 measured pairs. The 3.48% change refers to the wall-time medians. The v1 and v2 corpus, authoritative continuation hash, selector hash, four GGUF identity guards, six metadata-file identities, model and weight identities, and memory gauges match. The v1 full-logit timer and v2 compact timer have different CPU cost scopes, so v2 does not revise or replace v1's full-logit measurement. The change in GPU medians is unresolved; it is not evidence that the CPU copy or argmax caused a GPU timing change.

## Memory

The v2 memory gauges match v1: graph scratch high-water was 80,740,352 bytes; required peak was 108,692,065,600 bytes against a 109,676,289,225-byte cap; sampled Metal peak was 105,772,580,864 bytes. Static registration peaked at 6,408,552,448 bytes over 594 entries. These are guarded component observations, not runtime admission evidence.

## Validation and provenance

The model-free parser suite passed 19 cases. A release no-run build completed in 1m08s with `nightly-2026-10-01`; it ran no native tests and no model/GPU work. The guarded GPU attempt passed and confirmed unchanged source, binary, driver, launcher, model metadata and weight identity. The full source freeze, archive and run receipt hashes are in [compact-v2.json](compact-v2.json).

Source is frozen at base `b4f1836d97c9fcfb154d803ba93f96b8255d18e4`. The run used engine binary SHA-256 `57da554650f76f08168853372c85be302cb512aa9a20699e728f63299c689222` and driver SHA-256 `750979c9fa4a929c2810e14f92a2d4cc489b7e5be13ef873fd546ed00ed0bb23`. The recorded precision policy is `existing_safe_f32_legacy_projections`, unchanged from v1.

The public [incremental v1-to-v2 patch](compact-v2.patch) changes only three files. It was regenerated against the public v1 patch state from the exact six v2 frozen files, using `--no-ext-diff`, `--no-textconv`, `--full-index`, and `--binary`. Its SHA-256 is `7877af6992fe9630be985ad0ca0390731807e94ee884bed49641d3a421094674`. Applying public v1 plus this delta to clean base `b4f1836d97c9fcfb154d803ba93f96b8255d18e4` reconstructed all six v2 source hashes. Reversing the delta restored all six v1 hashes; reversing the v1 patch then returned the checkout to a clean base. The source freeze's private incremental patch hash is recorded separately in `compact-v2.json`; the regenerated public patch is not byte-identical to that private patch. No model payloads or run outputs are published.
