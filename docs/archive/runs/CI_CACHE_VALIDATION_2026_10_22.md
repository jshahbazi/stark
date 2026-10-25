# CI scheduling and cache validation (2026-10-22)

Hosted-run measurements recorded when the CXX test moved to its own job and the Rust build cache was introduced. They are kept as a record; current workflow behaviour is described in [CI](../../CI.md).

## Scheduling change

Validated on 2026-10-22 UTC with identical native and TypeScript source, feature flags, test commands and toolchain pins; only the workflows changed between these commits.

| CI run | Source commit | Elapsed time |
| --- | --- | --- |
| [Before: pull request](https://github.com/jshahbazi/stark/actions/runs/39704758381) | `a60e554ce65eadd2f8f679729ed6590d161a74fd` | 31m26s |
| [After: manual branch validation](https://github.com/jshahbazi/stark/actions/runs/39708539187) | `09ede8daec2f8debf764a44c1be2f36254011b37` | 16m24s |

Both runs passed. Elapsed time includes runner queues and the completion gate; the reduction was 48%. The CXX test step took 12m31s in both runs; in the second run its separate job queued for 2m35s and finished while the compatibility job was still running. TypeScript passed on Ubuntu in 23s compared with 33s on macOS.

## Cache validation

The [cold population run](https://github.com/jshahbazi/stark/actions/runs/39874355777) and the immediately following [warm run](https://github.com/jshahbazi/stark/actions/runs/39879305262) both passed on source commit `0f73fa9b557a131938d037478329a6190e5dd871`. All five Rust jobs saved caches after their cold builds and restored their exact primary keys on the warm run; every build, Clippy and test step executed. [Security also passed](https://github.com/jshahbazi/stark/actions/runs/39874358763) on this commit.

| Cached job | Cold execution | Warm execution |
| --- | --- | --- |
| Repository checks | 1m57s | 2m13s |
| Scorer, default | 10m16s | 10m35s |
| Scorer, optimized | 7m12s | 6m27s |
| Inherited dependency compatibility | 18m12s | 4m35s |
| CXX panic safety and grammar compatibility | 16m24s | 2m42s |

Execution includes job setup, cache transfers and cleanup and excludes queue time. Summed execution across these parallel jobs fell from 54m01s to 26m32s. Default scorer execution did not improve because Cargo rebuilt the workspace crates and their Metal work after the fresh checkout. Whole-workflow elapsed time was 47m48s cold and 38m16s warm; runner queues dominated both (CXX waited 31m15s cold and 34m01s warm).
