# Decider Flash v1.3 packaged-app qualification

Date: 2026-11-16. Attempt: `20261116-DF13APP-01` on
`codex/decider-flash-v13`. See the [attempt ledger](../../model-experiments/2026-11-16-decider-flash-app.md)
for gates, failed controls and resource deferrals. The earlier
[native qualification](2026-11-16-decider-flash-v13.md) retains its own scope.

The signed, relocated app's embedded Python adapter and native worker matched
all five reference decisions, including three image cases. Repeat/reset,
private attachment cleanup, decision-only admission and GPU telemetry passed.
This is actual packaged-backend execution plus separate Swift controller tests;
it is not a completed manual GUI session or an installed-build claim.

## Build and source identities

| Binding | Identity |
| --- | --- |
| App implementation | `641f7f7f45bea07ec764c46279536e672b958aa5`; clean source at build |
| Model source | `perplexity-ai/pplx-decider-flash-v1.3` at `d06fb3f2d97e0a52325a0d463c450cf0338f44fe` |
| Native configuration SHA-256 | `d859a249a3089252f164c755d2a584e41c1041ab91dc32751a742ddad29e8337` |
| Native weights SHA-256 | `c3ab2961a63c3342ff3390b41036c6029d62095414db4cffbd59a8944c754c97` |
| Unsigned Decider worker SHA-256 | `ca58a20531797dfca877aef26f2986798a24a75539ac7a9a95643af34ca23579` |
| Signed bundled Decider worker SHA-256 | `e29af88f88e4d32be5ee23938244910788b044b9e21e8cceb87fee47ee00f6b6` |
| Signed app executable SHA-256 | `f96967ac4287eea55e2f2a4717ce2b90b347d4a56a092061482bd2b61f38b9af` |
| App build number | `20261116192037701994` |
| App manifest SHA-256 | `b9b00e71db2d4e534bc5b89b7357e90021a7381d107999a898f68fa312f451cb` |
| Bundle inventory SHA-256 | `df2e73ab61a91950455015e19a3a3a982a56afa3cb1bb3f99f811c32d5792e4d` |
| Reference source aggregate SHA-256 | `6a1555c5998ffcaba54a299599b45591002a7941b5d383bcef5bf35bf4d49481` |
| Reference runner SHA-256 | `2e94e02c84feea14186e57ee70a734f35f7f3072c63d4191795eb1f85c03f54c` |
| App qualification script SHA-256 | `ac63d04f55f77d96d51ea98c3ecdc7fa94a6c2f18537f0583555ef146ae18554` |
| Fixture SHA-256 | `3c302223fcd3491aaa47389dc91743a11627f0b8f61b555ffce6877e59969d4c` |
| Aggregate receipt SHA-256 | `420ff029a55ffd3417f80f56a37135cabb04224a8588d709faf6dc1135855021` |

The inventory covers 4,097 regular files and is hashed as UTF-8 JSON with sorted
keys and compact separators. The app is a Developer ID signed development build
with hardened runtime and secure timestamp; deep/strict signature verification
passed. No notarization or release publication was performed. Signing changes
the executable bytes, so both worker identities are retained.

Host: Apple M5 Max, 128 GiB RAM, macOS 26.7.2 (`25G332`). Four native binaries
were built separately with Rust 1.98.1, release and locked dependencies.
`stark-score` alone uses `optimized`; `stark-decide` and generation use default
features. The Decider worker change adds private-FD GPU telemetry without
changing inference arithmetic. Its BF16 weights/activations and F32 reduction,
convolution-history and recurrent-state policy match the native implementation.

The same unmodified, verified official reference was loaded separately on MPS
with FP32, `fast=False`, `locate=False`, separate questions, 4,096 positions and
eight CPU threads. Reference and app models did not run concurrently. No
timings or speed claims are inferred from this correctness run.

## Actual packaged-backend checks

`scripts/qualify_decider_app.py` copied the complete signed app to a private path
containing spaces, checked every file hash and signature, then launched its
embedded Python with `-I -B` from an unrelated private working directory. It
used an isolated home/model library, offline HF settings, invalid external
Python paths and the app's four bundled worker paths. No checkout Python module
or native worker was used by the running desktop backend.

The existing qualified prepared artifact was imported through the normal local
install command. Registration retained Decision-only mode and images; an
attempted Chat load was refused. Each image crossed the app's opaque private
attachment transport, then a separate private native-worker lease. Raw model
inputs/results remained in memory and private pipes.

| Gate | Result |
| --- | --- |
| Reference choices | 5/5; two text and three image cases |
| Image-only state | Passed in one of the three image cases |
| Maximum absolute probability error | 0.000790242319442, below the fixed 0.02 limit |
| Mean absolute probability error | 0.000146810055544 |
| Answer order and checkpoint calibration metadata | Passed for every decision |
| Repeat text after intervening images | Exact choice, labels, probabilities and input-token count |
| Reset followed by repeat | Exact same fields |
| Attachment consumption and reset/stop/shutdown cleanup | Passed; staging empty |
| GPU memory telemetry | 13 events through the private native FD |
| Ready capabilities | Decision-native, images enabled, one image, 4,096 expanded positions |

These are bounded synthetic controls using existing private fixtures, including
simple image colour controls. They do not establish general task or vision
quality, probability calibration on real tasks, or another artifact's parity.
No generation, tools, Locate or MTP path was enabled. Every native forward has
fresh state; bidirectional attention still blocks causal prefix reuse.

## Regression checks and delivery

- Packaging/build/signing checks: 99 passed.
- Swift Decider controller harness: 10 passed; existing controller regressions:
  156 passed. Full Swift typecheck passed for arm64 macOS.
- Pinned source/catalog/support tests: 279 passed. Source preflight on the real
  local snapshot stopped at the disk margin before full weight hashing. Fresh
  download/conversion is implemented and mock-tested, resource-deferred here.
- Initial full converter suite: 7,875 passed, 95 skipped, one test-double
  interface failure. Focused input-budget/Decider rerun after fixing only the
  test double: 23 passed. Final complete converter rerun: 7,878 passed,
  95 skipped in 338.26 seconds, with offline HF and tokenizer parallelism off.
- Repository policy, Rust formatting, Python lint and diff checks passed.

The signed development app remains an ignored local build under
`artifacts/menu-bar/Stark.app`. Its source manifest stays bound to the clean
implementation commit even though the external validation harness was corrected
later. The user's installed app, model library and credentials were not changed.
No PR was opened. Branch push status is recorded in the final delivery response.
