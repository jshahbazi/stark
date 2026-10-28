# Manage Models navigation and saved-row Run, 2026-10-27

Starting from `45f0461651f0eab57db0fd07acae06d460c1b646`, each saved-model row
has a **Run** button. It resolves that version's private ID, inherits the current
Chat or Decision mode and opens the Playground. A matching ready model is reused
without clearing its conversation. Busy and stale rows are rejected; a failed
runtime send leaves management open with an error.

**Manage Models** appears immediately to the right of the model dropdown in the
menu and Playground. The install dialog's former **Models…** action uses the same
name. These actions explicitly open an independent management window. Install
Model no longer owns a model-management sheet. The original reported first-open
failure was not reproduced or attributed to a proven cause before this change.

All 90 production-controller checks passed, including five new saved-row Run
cases: exact identity with duplicate names, busy/stale guards, reuse of a ready
model, mode mismatch and an unavailable runtime. Full Swift type-checking and
compilation, Ruff, scoped formatting, repository policy and diff checks passed.
Independent final source review found no actionable regressions and confirmed
that source hashes match the successful controller run.

Actual GUI checks used the installed signed app. The first **Install Model…**
button click opened only the install form; a subsequent opening also left
management closed. Hugging Face remained the default. The menu, Playground and
install management buttons opened the independent window. Done and native close
both closed management. Both compact saved rows retained their controls without
clipping, including an inline Running indicator. One removal confirmation was
opened inadvertently and canceled; no removal was performed.

Run on an existing saved model reached Ready in the Playground. Repeating Run
for that ready model returned to the ready Playground. The model was stopped
after QA, and management was left open on **On this Mac**. No inference prompt
was submitted; this is UI and model-load evidence, not numerical qualification.

The installed app at `~/Applications/Stark.app` matches the candidate's tree,
deep strict Developer ID signature, Swift source and 58 packaged converter source
files. A verified rollback copy was retained. Both models remain registered with
byte-identical registry contents, and selected-model, mode and sort preferences
are preserved. No models were downloaded or removed. Backend/engine code is
unchanged. The local build is not notarized. Private evidence remains in ignored
`runtime/manage-models-run-validation-2026-10-27/`.

| Item | SHA-256 |
| --- | --- |
| Installed application tree | `2db407d5339936edf4bf8774ebf3636fc704e36b60cf58c94599c4e4942f00d9` |
| Signed Swift executable | `a17d5bf4a67b568bc9356d889ae56dbc0bcfae78b75f280a1917eccc76de1b87` |
| Application manifest | `15959bc0883bdf4dc6cfa5276b64325d3b079960845b893112c6fc50c75a94f0` |
| Swift source | `525f952d90c9eaef05b2065fa05efd96c9ae79598a16f3f8caadcbcbc41e56f1` |
| Controller test source | `2bf6473c1a01903c5624175a238019a5201ba9c34ac9d1c0df32073d8d942d61` |
| Installed-state and GUI aggregate | `d89f128b608bf9c7d3a13b386453a5b13e2acb66d5b37cbe931b45753bcdde44` |
