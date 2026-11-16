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

## Installed GUI follow-up: 20261116-DF13APP-02

The user subsequently requested installation and an actual-app image check.
The original `DF13APP-01` build was installed and verified unchanged. CUA could
not initially attach to the menu-bar popup; the user opened Manage Models.
The visible Add from Source flow then exposed a missing control: it sends no
explicit artifact mode and inherits Chat's text support purpose. That payload
was rejected for the detected Decision artifact. Earlier backend checks had
explicitly requested Decision. This was a GUI-path failure before inference,
and does not erase or invalidate the earlier explicitly scoped backend results.

The fix routes an auto-detected prepared or pinned Decider source to Decision
when the source form has not explicitly selected an artifact mode. Explicit
conflicting modes are still refused. New regressions mirror the visible form,
and the external packaged validator now exercises its inherited default payload.

| New binding | Identity |
| --- | --- |
| Implementation | `9bef02aed7354e5f8b79597b91e32b452eb0e59c`, clean at build |
| Signed development build | `20261116200836904445` |
| App manifest SHA-256 | `4f1392231ad17346bb870bc5f604e78ca70290df99207f0795ecd7a5339f5437` |
| Signed app executable SHA-256 | `5890f7cd5e076c916146f85eece1d6636d1e91be1a835c933052bc03a6290647` |
| Signed Decider worker SHA-256 | `258245aa6674397c21f9677e03e04aa2f5051df35ee6a4011f2bf5bb2b272bc3` |
| Installed and staged inventory SHA-256 | `0dd5445a3fe81b97c3b6ec1821ae15579ea8dac21a1e722f60bb28245fe471a3` |
| Updated app validator SHA-256 | `888d00003211765614089d087661dac268dc1d91cc9f0ffb32deec27c3085a9f` |
| New private reference receipt SHA-256 | `98f2502db36309d659c87f438127f9d2aae05fe56eb9866f8ab5035a76e9e2b6` |
| Private installed-GUI receipt SHA-256 | `aca425af2a0521e1be65d6ec93210eb6ce863c6aa5ad945d1fb73b0d1f8fe3e5` |
| GUI test image SHA-256 | `c07d7cc870c2345d68f7bb38ee814d16d0a9cadb41520dd25f97459f683ebed0` |

Model configuration, weights, reference source/runner and fixture identities
remain as above. Native inference code and the unsigned worker did not change;
new signatures change delivered executable hashes. All 4,097 installed files
matched the new staged app, including after GUI execution. Deep/strict signature
verification passed. This build was installed at `~/Applications/Stark.app`.

The new relocated packaged-backend run passed all five choices (two text, three
image), maximum absolute probability error 0.000790242319442 and mean error
0.000146810055544. Repeat after images and reset were exact, private attachment
cleanup passed, and 14 GPU memory events were received. Reference/native models
ran sequentially on the same host and precision policy as `DF13APP-01`.

The installed GUI was then exercised through visible controls:

- Manage Models → Add from Source → Local Folder imported the same prepared
  artifact. The row showed Decision and images and no tools. Run automatically
  selected Decision and loaded the native model.
- The Playground accepted one private 256×256 synthetic PNG with an empty text
  state. Run decision produced the expected top answer, three rendered
  probability rows, the saved-calibration caption and Ready status. GPU memory
  was visibly reported. This observes the result presentation, not a separate
  numerical comparison of rounded GUI percentages.
- Remove Image cleared the preview and result. Stop left the app stopped,
  removed the worker and cleared GPU telemetry. The backend's private
  attachment directory was empty afterward. Test inputs were cleared.

The imported model remains available as `Native V2`; its prepared folder must
remain available. The four pre-existing model entries and credentials were
retained. There is no standalone Reset button in this Decision view, so GUI
reset is not claimed; the external packaged check exercised native reset.
Raw GUI accessibility output containing inputs/results was inspected in memory
and reduced to aggregate checks; no result screenshot or raw probability record
was saved or published.

Verification after the fix: 59 focused regression tests and 363 common desktop
tests passed, with one common test skipped. Repository policy, lint and diff
checks passed. The earlier complete 7,878-test run retains its earlier source
binding. No performance or broad vision-quality promotion, PR, notarization or
release publication was performed.

## Installed catalogue and link check: 20261116-DF13APP-03

The user's newer installed app came from another branch and lacked Decider
catalogue support. Its local model row also lacked a source link: the historical
prepared artifact's receipt does not satisfy the newer display-provenance
binding. A separate optional, validated `upstream_repo` display field now links
the typed model family to Hugging Face. It does not change artifact grouping,
the unbound source revision, numerical qualification or the historical receipt.

| Binding | Identity |
| --- | --- |
| Implementation | `8d1c8658c2b1de7ced7453a6f113c7e45e1e4d3b`, clean at build |
| Signed development build | `20261116213746314865` |
| Manifest SHA-256 | `7d16456d0fcb6d27b5963547237bc92d6a0a0efdff8ad1374bba35c0db6a0578` |
| Installed/staged inventory SHA-256 | `51ebbb37a7b60c726137624b8e471b7253e125c6adb2c788535b250ffea7f2b8` |
| Signed app executable SHA-256 | `c2ced8a9f1ce9c6a2a8136a80e941d58f6d08a5fce63074f97ad0a0c9692e6a3` |
| Signed Decider worker SHA-256 | `9b9187021ff5dde3350c9fd469cf76ee483f1f8f44f85e9def8354c6a274b767` |
| Private aggregate GUI receipt SHA-256 | `e44af292863c1cdf0627e4c8680c17e17d520e01baf1a56cc686a575f19c709a` |

The prior installed app was preserved in ignored local backups. The new signed
Decider branch build was installed at `~/Applications/Stark.app`; all 4,097
delivered files matched staging, and deep/strict signature verification passed.
Actual visible controls confirmed:

- Installed row: `Decider Flash v1.3`, Decision and images, with a link to
  `huggingface.co/perplexity-ai/pplx-decider-flash-v1.3`.
- Model Catalog: a separate **Decider Flash v1.3** family, alongside Qwen3.5.
- Opening that family: pinned revision `d06fb3f2d97e0a52325a0d463c450cf0338f44fe`,
  BF16, 4B, 9.08 GB, Decision and images, official Hugging Face link and Download
  control. No download was started.

Seventy focused Python inventory/Decider desktop checks passed, followed by a
46-test inventory rerun. Eleven Swift checks passed, including the upstream-link
fallback without changing qualification/grouping, malformed repository-string
rejection and bound-group precedence. Policy, lint, diff and signed build checks
passed. No inference, new numerical comparison or performance test was run for
this display-only change; earlier source-bound model results remain separate.
