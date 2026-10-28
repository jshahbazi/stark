# Compact saved-model rows, 2026-10-27

Starting from `a10ca1d7b00abceae93bb665b3668369d93893d7`, **On this Mac** now
places each saved name, quantization, file size, repo link and removal button on
one compact line. Single variants omit the old card heading; multiple variants
retain a compact shared heading. Family, image support and the full source
revision remain in row help. Running state stays inline.

Name sorting uses the saved name for singleton rows and the visible heading for
multi-variant groups, including when search reduces a group to one version.
Repository URLs and removal actions retain their existing per-model identity.

All 85 controller checks passed, including a new alias/filter sorting regression.
Full Swift compilation, Ruff, scoped formatting, repository policy and diff
checks passed. Independent review confirmed row identities, link targets,
removal validation, sorting and full revision retention. Hover formatting was
factored into a string helper to avoid a Swift type-checking timeout. Earlier
candidates and diagnostics remain separate from the final identities below.

Actual installed GUI inspection confirmed that both model rows and all requested
controls fit on one line. Each removal button opened the correct confirmation;
both were canceled. The model registry is byte-identical, with both models and
model/mode/sort preferences preserved. No models were downloaded or removed.

The Developer ID signed local app is installed and running at
`~/Applications/Stark.app`, with a verified rollback copy. Its tree, deep strict
signature, Swift source and 58 converter source files match the candidate.
Backend/engine code is unchanged. The local build is not notarized. Private
evidence remains in ignored `runtime/compact-saved-models-validation-2026-10-27/`.

| Item | SHA-256 |
| --- | --- |
| Installed application tree | `9fdeba22995319fdbb2201a69ce86b62b4c5b267abaf5774c508375f8e06bcef` |
| Signed Swift executable | `9e8964a34016301e40178a9b2da1c90da3f6412cbcdf0d0c2cadebfcfd30d7e9` |
| Application manifest | `cdc7cc1b96fa8c56549773b89aeb988c25aa1502a94a14269ecd66622ba67327` |
| Swift source | `3f05cb7232b967adf5355eeccf6f554cb2a23a8f24ee94c27719fdddff89189b` |
| Installed-state and GUI aggregate | `be7ef7b99595b3b1f98937649058b6fd901c28600043c7ac4bed22f61a2ff266` |
