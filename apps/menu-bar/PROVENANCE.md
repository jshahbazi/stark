# Menu bar app provenance

## Artwork

`Resources/Stark.jpg` is the owner-supplied original cat artwork; the app icon is a padded, resized conversion of it. `Resources/MenuBarTemplate.png` was edited from the original in Pixlr Express on 2026-10-21: the gray background, gold eyes and white highlight were removed, pupils retained, and the remaining cat flattened to black so macOS can render it as a template image in light and dark menu bars. No generative changes were used.

| Source asset | SHA-256 |
| --- | --- |
| Original JPG | `bfaecfd420af69ef3bcf6b5c92ae672a54e169eee59c8fe9ad4c8d9b119383f1` |
| Pixlr template PNG | `e51bf75a34ecfd83803714b613b1c4fd5d78c0d2feaf91b66379a9c936e5cfc5` |

## Build manifests

The app carries the base CLI runtime's licenses, notices and source manifest. `scripts/build_menubar.py` packages the complete converter wheel with its source provenance, licenses and distribution metadata, and adds Jinja2 and MarkupSafe from the converter lockfile with required hashes; building may fetch the pinned build backend and locked wheels.

`Contents/Resources/menu-bar-manifest.json` records the source revision and hashes, whether the checkout was dirty, the base runtime manifest hash and every file the builder adds, including the Python package files. Its `converter_package` record identifies the wheel, build backend, source hashes and installed converter provenance. `artifacts/menu-bar/Stark.app.manifest.json` additionally records the signed executable hash.

`--worker-build-record PATH` and `--score-worker-build-record PATH` embed optional worker build records after checking their `worker_sha256` against the selected binaries, preserving the distinction between a fresh locked Cargo build and a development worker that reused compiled engine libraries. The builder checks both workers' fixed invalid-argument replies before packaging.

The builder checks imports of the attachment staging, common media pipeline and
image runtime qualification modules using the embedded interpreter. It also
checks the desktop entry point's `--attachment-root` option. These are packaging
contract checks; they do not load a model or establish image execution.

Worker overlays also refresh the native license inventory using Rust 1.98.1 and
the current locked Cargo dependency metadata through the existing distribution
notice collector. This includes the native image decoder dependencies and the
verified PyTorch, SLEEF, Arm, GGML, Transformers and MLX adaptation notices.
The MLX notice is retained as `share/stark/licenses/LICENSE.mlx-reference` and
is required by the packaged image qualification checks. The old
CLI runtime's notice files are retained, while current notices and their Cargo
lock identity are recorded in the app manifest. Building the app therefore needs
the pinned Rust toolchain even when the worker executables already exist.

`scripts/qualify_image_app.py` is an explicit, model-bound qualification command.
It copies the candidate app into a fresh private installation with spaces in its
path, verifies the copy and its code signature, and invokes that copy's embedded
interpreter, converter and worker with an empty model registry and isolated home
and temporary directories. It performs two image conversation turns through the
production desktop pipe and requires the ordinary runtime catalog to admit the
delivered worker. The source checkout, Rust, uv and system Python are unavailable
to the checked runtime. Its report contains aggregate checks and delivered-file
hashes; generated text, credentials, images and model paths are not included.

This command does not install over an existing app, enable image qualification,
sign or notarize the candidate, or exercise the Swift controller. A separate
controller replay and signature/notarization checks must identify the exact
delivered bundle. Re-signing a worker can change its hash and requires new
source-bound image execution evidence before the ordinary catalog admits it.
