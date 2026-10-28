# Model controls and separate running section, 2026-10-27

Starting from `b5f8a11947257663d2aa2021b7cdcfb52f156308`, saved-row Run keeps
Manage Models open, including the current search. The active model has Stop;
starting another saved version moves Stop and the pinned row to that version.
Stop cancels the model load or request, keeps management open and restores Run
when the worker stops. Private IDs guard stale row actions and duplicate names.
Row Stop cannot cancel an unrelated installation or removal.

The active version appears once in a separate first section, titled Running,
Starting or Stopping as appropriate. Other models retain their source grouping
and sort by their remaining size or visible name, excluding the active version.
Search filters both sections. Without a matching active model, the list uses
normal grouping and sorting.

The menu and Playground use Manage Models to select and run models. Their model
dropdowns and standalone Run actions are removed; the running name remains
visible. Quick Stop controls remain available. Install Model still opens the
install form independently.

All 98 production-controller checks passed in the independent final rerun.
Eight new cases cover model switching, pending-load cancellation and late-ready
rejection, wrong/removed private identities, generation cancellation, unrelated
operation protection, single extraction with metadata/action preservation,
remaining-group sorting, and search/idle layouts. Full Swift type-checking and
compilation, Ruff, scoped formatting, repository policy and diff checks passed.
Independent source review confirmed the lifecycle and partition behavior. The
visible-name ordering of a group reduced to one version is intentional.

Actual GUI checks of the installed signed candidate confirmed that Run stays in
management, two existing saved models can be switched, Stop follows the active
model, and the running row moves above the alphabetical list into a separate
section. Searching for the other model hid the running section; starting the
filtered model preserved the query. Clearing search restored both sections.
The final menu and Playground show Manage Models and the correct running name
without model dropdowns. Row Stop restored the normal list without navigation.
Management was left open with no model running after QA. No prompt was submitted;
these checks qualify UI/model loading, not inference accuracy.

The Developer ID signed app at `~/Applications/Stark.app` matches the final
candidate's tree, deep strict signature, Swift source and 58 packaged converter
source files. A verified rollback copy was retained. Both models remain
registered with byte-identical registry contents; selected-model, mode and sort
preferences are preserved. No models were downloaded or removed. Backend and
engine source are unchanged. The local build is not notarized. Final private
evidence remains in ignored
`runtime/pinned-model-controls-validation-2026-10-27/`; earlier candidates and
checks remain separate.

| Item | SHA-256 |
| --- | --- |
| Installed application tree | `36fb182b243f5597b36badb1c690a633855a9c20a2206b2a8b6cec1c16ac6f77` |
| Signed Swift executable | `7f78ecdafca69757e4ffe25d5d043b948ee88ca24f6640c2d152b8d8fda922f0` |
| Application manifest | `5eaac1f985270c2da2fa71007e1880b20f928f4fe5579753c735544ab2dd7335` |
| Swift source | `5f4b67513eb193def76557fc79cfd7631fbabff27af58659610efdd7587885df` |
| Controller test source | `79b2439917647477e51ff0189ad1286240e51c44d08e41437f28743425dd6992` |
| Installed-state and GUI aggregate | `0725f82534a73040bc33d4d37d4f4d823bc7875f81bc55a03a4bdfc8f464abbb` |
