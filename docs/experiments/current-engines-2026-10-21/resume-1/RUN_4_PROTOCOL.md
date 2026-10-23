# User-directed observed run

On 2026-10-22, after the quiet preflight timed out, the user explicitly directed: "just run it", followed by "try again". Campaign 4 therefore starts without waiting for pre-load or post-warmup quiet windows. It remains a complete fresh 24-load run: six configurations, four independent loads, unchanged engine builds/models/sampling/fixtures/repeats and serialized requests. No previous timing samples are reused.

CPU, GPU, thermal, paging, sleep, clock, fixed-work and cleanup observations remain enabled. Interference exclusions are retained. This observe-only mode always records `timing_accepted: false`, including when individual loads pass their screens, and cannot produce a certified quiet-host public comparison. The previous frozen clean protocol and failed/interrupted receipts remain unchanged.

Config SHA-256: `9dd9888b7ae4e6649c67f52a7ec021856315eae190b41291ac03cdd236e3e59e`. Acquisition harness SHA-256: `3eb193c53b2a06990cc263de8efa84c1e82124a9eefc57c4fb154af290193665`. Engine/model/lock identities remain those in `ARTIFACTS.json`; main `bba01710d7996c10713c4968a758288966ca50f6` leaves the verified Stark native build inputs unchanged. The model-free regression checks passed: 14 tests and 6 subtests.
