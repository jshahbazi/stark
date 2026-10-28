# Explicit model tab names, 2026-10-27

Starting from `f878203137eca7f527e46cba9eea79270dacfea3`, the management tabs
read Installed, Model Catalog and Add from Source. The application-menu command
also reads Add from Source; Command-I retains its existing route to that form.
Current reference guides and existing label expectations were updated. Runtime
actions and internal navigation cases retain their existing behavior.

All 100 controller checks, full Swift compilation, Ruff, scoped formatting,
repository policy and diff checks passed. Independent review confirmed naming
consistency in the current app and reference documentation. Actual installed-app
GUI inspection verified that all names fit, Installed shows the saved models,
Model Catalog opens the family browser, and Command-I opens Add from Source with
Hugging Face selected before Local Folder. The menu command also has the new
name. No models were loaded, downloaded or removed for this rename check.

The Developer ID signed app at `~/Applications/Stark.app` matches the candidate's
tree, deep strict signature, Swift source and 58 packaged converter source files.
Its rollback copy was verified. Both models remain registered with byte-identical
registry contents; selected-model, mode and sort preferences are preserved. The
local build is not notarized. Private evidence remains in ignored
`runtime/explicit-model-tabs-validation-2026-10-27/`.

| Item | SHA-256 |
| --- | --- |
| Installed application tree | `e4b4036a7914c9229d2107ff86b6d14b712043fe438bbc2559b13371b9d81e6d` |
| Signed Swift executable | `65bbf64b9b0e5e9c950f85674ab057ba8db0ed4abdf8ed69ab127e07a5327a27` |
| Application manifest | `d1739ac21b8ac4f7fe8fbf69d6b76cb2b1d0550113142eee2ad7855dddb7f48e` |
| Swift source | `b18a304d962b1066ca9487b79644a86ccc29846f2e3fd928781c24d9b290e628` |
| Controller test source | `cd9eafd3678c4d3c5ea08995de88224a6ffae3e4e943a9bc84a1ec0d87aa735c` |
| Installed-state and GUI aggregate | `c0dcc6c0a19acca666b5ad1b5d4e7dd4800bd963a7e4b28807fae219c4e7bd5d` |
