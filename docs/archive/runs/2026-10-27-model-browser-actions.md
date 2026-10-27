# Family downloads and model removal, 2026-10-27

This work starts from `120215d2a1963edac1bd897a30b5312c1b6025f4` on
`codex/qwen35-vision-artifact-contract`. It adds direct downloads to the family
browser and per-version removal to **On this Mac**.

## Delivered behavior

Family cards open a model list with sizes, quantizations, publishers and source
versions. A downloaded version is identified by its bound repository, revision
and artifact mode. Supported Qwen3.5 BF16 choices offer **Text** or **Text and
images**. Downloads keep progress and errors in the browser and leave model
selection unchanged, including the first download into an empty library.

Discovery combines registered sources with bounded live metadata inspection:
at most 45 candidates and 30 returned choices. It preserves representative
sizes and formats, excludes known embedding/reranking and internal test models,
and never fetches weight payloads while browsing. The list is a representative
subset of available sources. Installing a choice resolves the backend's cached
entry, pins its primary revision and repeats compatibility checks before weight
transfer. Image installations retain their required local execution checks and
can stop the loaded model for those checks.

Each saved version has a trash button and a confirmation identifying the model,
quantization and version when available. Exclusively owned managed folders go
to macOS Trash. Imported, shared and unowned folders offer **Remove from Stark**
and keep their files. The backend rechecks ownership and the confirmed action,
stops only the removed model's worker, and restores the registry and attempts
file rollback if persistence fails. Shared image-check receipts remain intact.

## Completed checks

| Evidence | Result |
| --- | --- |
| Full locked converter suite, Hub offline | 4,936 passed; 88 skipped. |
| Exact-source Swift controller suite | 71 passed; independent repeat also passed. Covers stale catalogue results, mode-specific downloaded markers, progress ownership, empty-library selection, rejected download/removal requests, stopping and removal confirmations. |
| Builder suite | 23 passed; 15 subtests passed, including packaging both new modules. |
| Swift, Ruff, formatting, repository policy and diff checks | Passed; final app compiled and signed. |
| Independent review | Catalogue ordering, optional-mode fallback, concurrent library projection and correlated admission errors corrected; final review found no remaining functional blocker. |
| Live metadata discovery | 26 Qwen3 choices and 24 Qwen3.5 choices; BF16, 4-bit and 8-bit represented. Six Qwen3.5 choices offered images. All official dense 0.8B/2B/4B/9B/27B choices appeared. No weights were cached in the isolated discovery run. |
| Packaged backend, private library | Downloaded/converted Qwen3-0.6B at the cached catalogue revision; support preceded active progress. Output was 1,192,276,800 weight-file bytes. Native loading passed; repeat download retained the same entry and loaded worker. |
| Real Trash and imported removal | The private converted model went through Foundation Trash with its directory identity preserved. Its matching worker stopped. Files were restored and then imported; unregistering the import preserved its weights. Private library ended empty. |
| Actual installed GUI | Family navigation, Qwen3 BF16/4-bit/8-bit choices, existing-version Downloaded status and Qwen3.5 Text/Text and images picker verified. Managed and imported removal dialogs inspected and canceled. |
| Existing user data | Two models retained; registry byte-identical; selected model and mode preserved. No user model was downloaded or removed for QA. |

The private qualification driver initially counted the UI's waiting placeholders
as active transfer. It was corrected to check support before running progress;
the original attempt and corrected driver remain separate private evidence.
An earlier signed candidate is also retained separately and was not installed.

The final Developer ID signed local app was installed at
`~/Applications/Stark.app`, with a verified rollback copy, exact candidate/tree
checks and deep strict signature verification. It is running and is not
notarized. The actual GUI download button was not used to add models to the
user's library: controller action tests and the real packaged private-library
download establish that path separately. Native engine/worker code and the
image qualification catalogue are unchanged; numerical parity was not repeated.
Private bundles, model fixtures, raw logs and reports remain under ignored
`runtime/model-browser-actions-validation-2026-10-27/`.

## Final identities

| Item | SHA-256 |
| --- | --- |
| Installed application tree | `40d3a0dfe5a82cf87c2791120f5b078a87894a94112929995f5d7b4e70e75766` |
| Signed Swift executable | `20813d90caa682e7f6e2f7d7fb7504a27c3cde6d8538a505817e2ebbe2b22549` |
| Application manifest | `ee40164b1253e312d979806fc0ef1c3217ab8045cb7bf339638a8cd130dfd6af` |
| Swift source | `91f742de1e0b3064a4ddbca8d075f28c8bd3cd34a44b13bb69bc541c350a833c` |
| Catalogue source | `db6f793a522210b52dc059585e21169dbee878e55925f830cf489f06809da430` |
| Removal source | `7ae0b8eab547a1b9e17edbbff7e9cc5712d94d00e5ccdf730d0b31fd96f0f69e` |
| Installed-state and GUI aggregate | `3101281e5a0bb265c6d59f9ae714b63f9d254b34802baefd10de50732c366d10` |
| Packaged-backend aggregate | `29899fc44ae16abe4d8e29f102032f4b2fdff803eb08cb9c743dbf03ec278f70` |
| Packaged qualification driver | `96a2c773c6df9a997b96097f2c2fd25ec48e1c261d28d2597a4a9e0072284fdc` |
| Full converter-suite log | `964ac3e1ac73d61e0db02d706dc4d947717046ab42cf64daeedf27459e32e274` |
| Controller log | `654d8614624c943201b6e29af425d7726277ad5b9fe0f2afda2c3e3036594859` |
| Builder log | `e7e7fbb3605cfea6fa8170c3d855b15e0f4a5b4c05bc7981a27ff259895c48cf` |
