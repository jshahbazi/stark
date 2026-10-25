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
