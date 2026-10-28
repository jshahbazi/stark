# Family quantization tabs, 2026-10-27

This Swift-only change starts from `3570bb25c5a71a5c82b9dbf89abbc8d7251a5001`
on `codex/qwen35-vision-artifact-contract`.

Each selected family opens on **All models**, followed by tabs for the formats
present in its full current catalogue, such as **4-bit**, **8-bit** and **BF16**.
Tab selection intersects with search before existing grouping and sorting.
Immutable row IDs, downloaded status, download actions and image choices are
preserved. Switching families resets to All models; a format removed by refresh
also resolves to All models. Browsing tabs does not request another catalogue.

## Validation

| Check | Result |
| --- | --- |
| Swift controller checks | 84 passed, including dynamic/unique tab labels, empty and single-format lists, missing-format fallback, search/format/sort intersection, downloaded markers and exact download targeting. Independent repeat passed. |
| Builder checks | 23 passed; 15 subtests passed. |
| Swift typecheck, final compilation, Ruff, scoped formatting, repository policy and diff checks | Passed. |
| Actual installed GUI | Qwen3 and Qwen3.5 showed All models, 4-bit, 8-bit and BF16. Format changes filtered rows while retaining the typed size search. All models restored multiple formats. Entering Qwen3.5 after selecting Qwen3 8-bit reset to All models with an empty query. |
| Preserved data | Two models retained; registry byte-identical; selected model, mode and all three saved sort choices preserved. No model was downloaded or removed for QA. |

The qualified Developer ID signed local build is running at
`~/Applications/Stark.app`. A verified rollback copy was retained. The installed
tree, deep strict signature, compiled Swift source and all 58 packaged converter
source files match the candidate. It has hardened runtime and a secure timestamp
and is not notarized. Converter, backend and native worker code are unchanged;
their full suites and numerical parity were not repeated for this UI filter.
Private bundles, backups and logs remain under ignored
`runtime/model-quantization-tabs-validation-2026-10-27/`.

## Identities

| Item | SHA-256 |
| --- | --- |
| Installed application tree | `6c3de39de2fa70b5f2fdfa75685aff163cb1bad45c1af66b61cd29e96899cfe9` |
| Signed Swift executable | `4ab5f537790bf70a52fa1e0a08b8b20c05377e1d484ee0cbf23b177dc4553a2b` |
| Application manifest | `a75cf53bbd0fd56c72918b9d9c355626bc435f90a7ff5c7fcd7dac713758d442` |
| Swift source | `3c5d94f739b9141e199127025e2327af6cf8b04175b004b2f1504fa848590819` |
| Installed-state and GUI aggregate | `7ea5838a91b768d638e65db4a07baf54d0a8463c5b9b05c01a041ca432db2d33` |
| Controller log | `5056927c7a825a33cc434be99618f2f85ee78172cb17b333926debdf66c7cee8` |
| Builder log | `c0c1715a188f90ada3bc40fdebbadb39228fa131044328a8fa0942308400f888` |
