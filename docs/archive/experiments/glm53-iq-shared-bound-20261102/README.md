# GLM-5.3 IQ Pair-Local Shared-Bound Candidate

Date: 2026-11-02. Status: **source and component correctness gates passed; component timing is unqualified**.

## Candidate

The isolated branch `codex/glm53-iq-shared-bound` contains commit `ef48dcaeaec4bdc5024b411aef47f086e1bbded7`, based on `cd03dc7c44bb7dbf44711b797634fb3d864e4b12`. It adds an opt-in `STARK_GLM53_IQ_SHARED_BOUND=1` path. The flag remains default-off, and the optimization branch does not merge or enable this candidate. No production wrapper changed.

For gate and up projections consuming the same input vector, the candidate computes their shared input bound once. The two GEMVs remain unchanged. Seven correctness GPU tests, one whitelist/budget fallback GPU test, and one GPU component test passed. The source audit found no actionable issue, and the CPU no-run build passed. No full-model generation or integrated engine comparison was run.

## Component result

The component capture emitted four aggregate rows for IQ17 and IQ18, each with shared-bound off/on. Each projection used 2,048 rows and 4,096 columns per expert, selecting eight experts from a bank of 288. Each command contained eight pairs with rotating scattered expert groups. Every full pair kept two GEMV dispatches; the shared-bound mode reduced bound dispatches per pair from two to one. Gate/up/activation output bits matched, and 512 sampled gate/up rows per command passed the F64 relative absolute-sum tolerance of `2e-6`.

| Kind | Shared bound | Min (µs/pair) | Median (µs/pair) | Max (µs/pair) |
| --- | --- | ---: | ---: | ---: |
| IQ17 | Off | 104.260416 | 106.510417 | 176.609377 |
| IQ17 | On | 101.453126 | 102.250000 | 168.265626 |
| IQ18 | Off | 115.234376 | 115.658853 | 181.713542 |
| IQ18 | On | 110.734374 | 111.088541 | 202.937499 |

The median component times are about 4% lower with shared bounds. The capture was unscreened and reports a 57.5%–83.0% range-over-median across rows. Although commands alternated ABBA/BAAB, per-round observations were not emitted, so control drift cannot be calculated. These results do not establish a stable performance effect, full-model gain, or an integrated speedup.

The maximum observed F64 error divided by absolute sum was `6.031283195230824e-9` for IQ17 and `4.504282453766835e-9` for IQ18. Both were below the `2e-6` gate. Counts, ranges, dispatches, flags, and receipt hashes are in [the aggregate results](results.json).

## Source and validation provenance

The pushed source commit is pinned to `ef48dcaeaec4bdc5024b411aef47f086e1bbded7` with parent `cd03dc7c44bb7dbf44711b797634fb3d864e4b12` and tree `378abbefbbd85375f1446a852cfe3bcdf82ac494`. The five source paths and their SHA-256 values are recorded in `results.json`; the commit's five blobs matched the original freeze. The canonical full-context patch is retained privately with SHA-256 `60db73420f3a747dddff8867111c9d32e1bb191e45e32e9aa16b5f2e58d5c095`. The public record binds the pushed commit and file hashes instead of duplicating the 485,297-byte patch.

The library no-run build passed with `nightly-2026-10-01`, `--locked`, `cpu,metal`, and two build jobs in 341.033 seconds. Its sealed test binary SHA-256 is `cf17550d585601a3a177a7086110e4564bc404fb3d5af94558baa85bb1c40388`. Seven matched correctness tests and a separate whitelist fallback test passed; the component test passed and emitted four rows. Receipt identities and launcher pins are in `results.json`.

Capture v1 failed before Cargo launched because the managed Python `PATH` did not contain Cargo; its sanitized failure receipt and four tiny model-free fixture results remain private. V2 used an absolute Cargo path and supplied the passing build and seven-test gate; v3 added a whitelist fallback test; v4 captured the single component test using a strict known `libtest` prefix. Historical capture receipts remain private and unchanged. Raw child stdout/stderr were not persisted; the capture cap was 8 MiB.

This archive documents an isolated candidate branch. The optimization branch remained at `3bdbf15ea357bb966bd6681a77921c5fb4e24f88` when this record was prepared and does not contain the candidate. No pull request, full-model run, or production promotion is claimed.

## Privacy

The public record contains source hashes, aggregate test and timing results, and receipt hashes only. Raw child streams and any runtime payloads are not published.
