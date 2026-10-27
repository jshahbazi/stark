# Model repository links, 2026-10-27

Starting from `ee50fdd6954ea4d313c33bbd2b1b9d7741e96aaa`, each family-browser
variant row now has a **link** immediately after its download size. It uses the
row's exact Hugging Face repository, including a quantized export's publisher.

The existing 84 controller checks, repository policy and diff checks passed.
The final app compiled and was Developer ID signed. Independent review found no
issue with URL construction, row identity or available layout width. Native GUI
inspection confirmed the placement and exact BF16/4-bit/8-bit link destinations,
including Qwen base sources and their mlx-community exports. Image pickers and
Download controls fit alongside the new link.

The signed local candidate is installed and running at `~/Applications/Stark.app`.
Its tree, deep strict signature, Swift source and 58 packaged converter source
files match the candidate. Both models, selected model, mode and sort choices
are preserved. No model was downloaded or removed. Backend/engine code is
unchanged. This local build is not notarized. Private evidence remains under
ignored `runtime/model-repository-links-validation-2026-10-27/`.

| Item | SHA-256 |
| --- | --- |
| Installed application tree | `d8092303b96bfa7ac90c41f769073954cf134c48b57a91dc056903c27886bc4a` |
| Signed Swift executable | `76308dbec6cdfcda4b0a48ce562be23f6c7efa172d8739d2fecbfc05d7fc7df0` |
| Application manifest | `9bf6d31b5ada1e20c96c15d8035aa043c1231801792e642d9009fb9b1f02061e` |
| Swift source | `191b56160afc1b89e8931180b964d7d0f371ba601d57e18c6994cf3bfa314a8e` |
| Installed-state and GUI aggregate | `93abe25bb403cf182fcb8c5d4ca90abcf10562c84ccb4e9d51c1a5c72a40fb59` |
