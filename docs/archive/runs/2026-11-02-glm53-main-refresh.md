# GLM-5.3 Main Refresh Merge Validation (2026-11-02)

## Change

Merged `origin/main` at `23abde2ff041ee5a4b0e313a3793a0a4cb6fad00` into the optimization branch. The resulting merge commit is `cd03dc7c44bb7dbf44711b797634fb3d864e4b12`, with tree `8fe66daa6baa23e5aaebdf4fd49e3ea4d97ac49b`. The conflict in `hybrid.rs` was resolved by passing the checked `input.get(reused..len - 1)` slice to `ingest_prefix`, preserving the incoming Qwen restore/capture path alongside the GLM batch and native profile behavior. An independent source review found no actionable issue.

## Validation

Rust 1.98.1 checks passed against the unchanged tree above. `stark-generate` reported 112 passed and 2 ignored; its release build passed. The optimized `stark-score` test command emitted three suite totals of 1, 25, and 25 passed; these are separate suite outputs, not a unique combined test count. Its release build passed. The selected Python checks reported 9 passed. Formatting, repository policy, diff checks, and the native wrapper shell check passed. The five private command receipts and their hashes are listed in [the aggregate JSON](2026-11-02-glm53-main-refresh.json).

The merged-head native smoke passed with `repeat_exact: true`: 14 input tokens, 512 context tokens, 32 greedy output tokens, one warmup and one replay. The warm and replay output IDs had the same SHA-256 as the sealed v48 result. The input identity also matched the receipt. The run was unscreened and establishes no performance rate or comparative performance claim. The weight receipt reused prior verified hashes and checked file size, modification time, and inode before and after execution; no 108 GB rehash was performed. See [the aggregate JSON](2026-11-02-glm53-main-refresh.json) for the fields and private receipt bindings.

The qualified T8 benchmark record is unchanged. Separate MTP v7 and shared-bound candidate work remains unqualified and does not enable production behavior.

## Privacy

Only aggregate results, source-tree identifiers, and receipt hashes are published here. Runtime payloads and raw native output were held in memory and were not persisted. Private aggregate receipts are in ignored runtime storage.
