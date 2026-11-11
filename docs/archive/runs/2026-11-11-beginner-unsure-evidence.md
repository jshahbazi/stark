# Beginner unsure recommendation correction

Date: 2026-11-11. Source: the commit containing this record, based on
`75000202c89705b70ed60efb9ce86557ef6e7767`.

“I’m not sure” with Balanced previously let an unscored model retain first
place through the parameter/memory heuristic. It now prioritizes one eligible
published IFEval cohort, including when only one parent is covered. Within
that cohort, recorded native chat/tool workflow checks precede score. Missing
results remain unknown. Without eligible references, the recorded starters
precede other checked workflows, then unchecked sources ranked by resource
cost. This path does not rank parameter counts.

Regression coverage includes Flint on synthetic 128 GiB hardware, one covered
parent, zero scores, checked workflows, exclusions, resource admission,
explicit priorities, fallback copy and visible upstream-artifact caveats.
Synthetic memory geometry isolates ranking; it does not establish actual model
fit or quality. Existing benchmark provenance and model validation are unchanged.

Validation:

- Offline converter suite: 7,099 passed, 94 skipped in 281.97 seconds. The
  additional packaged-IFEval/Flint regression was added after suite collection
  and passed in the focused run below.
- Focused guided recommendation and benchmark tests: 121 passed.
- Native Beginner checks: 14 passed.
- Native controller checks: 141 passed.
- App builder tests: 46 passed.
- Repository policy, Ruff and whitespace checks passed.

No models were downloaded, started or evaluated. The installed application was
not rebuilt or replaced.
