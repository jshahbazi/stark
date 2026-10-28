# Three-tab model management, 2026-10-27

Starting from `0600efe8fac1236e55a696ff1daabcdc1df1f0b8`, Manage Models has
Installed, Download and Install Model tabs. Installed and Download replace the
previous saved-model and family tab labels. The main menu no longer has a
separate Install Model button. Command-I and the application menu command select
the third tab in the shared window; empty-library startup uses the same route.

The install form retains its source, source type and image option in app-owned
memory across tab changes and window reopening. Entering the tab does not reset
progress. The footer keeps Install, Done and cancellation accessible; Return
belongs to Install on this tab. Catalog progress remains available across tabs.
Manual installs have a progress link and cancellation on the other tabs.
Completion no longer invokes the old installer-window/Playground navigation.

All 100 production-controller checks passed in the independent final rerun.
Existing tab expectations now verify the new labels and order. The new lifecycle
case verifies that leaving a catalog during installation preserves its progress,
worker, ownership and busy state, rejects a late catalog reply, and releases the
original metadata cache on completion without auto-loading or selection changes.
Full Swift type-checking and compilation, Ruff, scoped formatting, repository
policy and diff checks passed. Independent source and documentation review found
no remaining blockers or stale current UI descriptions.

Actual installed-app GUI checks confirmed all three tabs, embedded Hugging Face
and Local Folder forms, persistent source/image drafts across tabs and reopening,
Command-I routing, visible footer controls and safe Return on an empty form.
Download family detail retains the top-level tabs and quantization tabs. A
private empty local fixture was rejected inline; its failed progress survived
tab switching. An existing saved model then loaded and stopped successfully in
Installed, retaining the pinned active-row behavior. The form was restored to
empty Hugging Face with images off, and management was left open in Installed
with no model running. No model weights were downloaded or removed. A complete
model installation and a long-running install tab switch were not performed in
the GUI; controller evidence and source review cover those lifecycle paths.

The Developer ID signed app at `~/Applications/Stark.app` matches the candidate's
tree, deep strict signature, Swift source and 58 packaged converter source files.
A verified rollback copy was retained. The two-model registry is byte-identical;
selected-model, mode and sort preferences are preserved. Backend/engine source
is unchanged. The local build is not notarized. Private evidence remains in
ignored `runtime/model-management-tabs-validation-2026-10-27/`.

| Item | SHA-256 |
| --- | --- |
| Installed application tree | `596a696bd76646d0218e2d1a30149c3d2bafd6265962c5a8992ba45d25f0418b` |
| Signed Swift executable | `c68131fa9730ab9290a0a0908bccc2ef76bac8f92f665b76ead46a5e213739cd` |
| Application manifest | `84c179be4efb883f4a37cc17d3e9b10df43d750a3d8595c5a295302ec128fea3` |
| Swift source | `22305c2aeea0d1f07322a4db41926700caea7b57589571462f61cd46359c5607` |
| Controller test source | `1ddb397fc46014bb7259586db038f201a6270fb98ced34a2f9bb98384118d8f5` |
| Installed-state and GUI aggregate | `6a46feaba5bd76a2986a26c6ab9ac593174cb72d411e03f20de3d19e059248b4` |
