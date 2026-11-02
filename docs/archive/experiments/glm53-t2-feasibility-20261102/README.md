# GLM-5.3 T2 Feasibility Probe (2026-11-02)

## Status

Measured component feasibility only. This record covers one guarded T2v1 GPU attempt against the source frozen at base revision `b4f1836d97c9fcfb154d803ba93f96b8255d18e4`. The probe is not integrated into engine defaults. It does not establish model-wide speedup, head catch-up behavior, rollback behavior, runtime admission, or production readiness. The prototype remains an experiment.

## Measurement

The comparison measured two serial token steps against a layer-major two-token graph across three cases. Four rounds included one warmup and three measured rounds, giving 36 measured serial pairs and 36 measured layer-major pairs. Eight ordinary serial alignment rows, four pool residues, and four pooled alignment pairs were checked. All 192 complete hidden-state and logit comparisons were bitwise equal. No rollback calls occurred.

| Aggregate metric | Two serial steps | Layer-major two-token graph | Change |
| --- | ---: | ---: | ---: |
| Median wall time (ms) | 73.082854 | 67.8808955 | -7.12% |
| Median GPU time (ms) | 65.72679166856688 | 59.16143750073388 | -9.99% |

This is a component result for the measured two-token operation. The timed scope includes full-vocabulary logits, hidden-state readback, greedy readback, and CPU finite/argmax work. It excludes head catch-up and rollback. The target cost differs from the earlier MTP greedy protocol; that protocol difference remains under investigation and is not evidence of a causal regression or an MTP comparison.

The measurement reported 80,740,352 bytes of graph scratch high-water. The guarded required peak was 108,692,065,600 bytes against a 109,676,289,225-byte cap; sampled Metal peak was 105,772,580,864 bytes. These measurements document the attempt and do not establish runtime memory admission.

## Source and validation provenance

The source freeze is pinned to base revision `b4f1836d97c9fcfb154d803ba93f96b8255d18e4`, with freeze receipt SHA-256 `17056778607b7b3add25d078d7e28810d61af4260fef9bc82210362894df79ca`. The six frozen source paths and their byte counts and SHA-256 values are recorded in `results.json`.

The private source archive SHA-256 is `349ed42ec3dac17f16959e01e35029e791e32bd26a7b54f9572a771dfd695970`. Its frozen embedded patch hash was `c126d36f8de2462b9adf29e298618110c6cb58b1d676d4eb987dffc5f323be3c`. The public `prototype.patch` was regenerated from the exact six archived source files in a disposable checkout at the pinned base using Git diff with external diff and textconv disabled. Its SHA-256 is `70f4202e9a31a994a461c71c74d083a5fb66b127905066d47fefc04c310dc95d`. It declares all six source paths, including four new-file sections. Forward application reconstructed all six frozen source hashes; reverse application returned the validation checkout to a clean base. The patch contains source code and authored fixed test fixtures only. It excludes binaries, runtime receipts and request/response payloads, token vectors, and probability data.

The model-free parser fixture receipt records 17 passing cases. The no-run release build receipt records a successful build in 3m25s and sealed test-binary SHA-256 `5126afa655d445d20c9b72d668a4c4a1e900da243c95d48896d96bad08dc9081`; no native tests were invoked by that build receipt. The GPU attempt passed its guards and reported unchanged frozen sources, driver, binary, wrapper, model artifacts, and weight identity. See `results.json` for receipt hashes and aggregate fields.

The recorded precision policy is `existing_safe_f32_legacy_projections`. No new mixed-precision or production precision qualification is claimed here. Engine defaults and benchmark data are outside this archive's scope; the prototype does not imply a runtime enablement or rollback path.

## Follow-up

The small measured sample does not answer whether T2 improves a full request after head catch-up and associated overhead. A diagnostic of the differing target cost is pending. Any follow-up must separately measure those costs and preserve the exact reference and guard protocol before drawing a broader conclusion.

The compact-timer v2 follow-up is recorded separately in [compact-v2.md](compact-v2.md) and [its aggregate JSON](compact-v2.json); the v1 results above remain unchanged.
