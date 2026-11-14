# Guided alternatives and missing stronger candidates

This record describes the initial implementation at `a3c1a382`. The
available-memory recommendation policy below was superseded by the user's
installed-RAM direction; see the [capacity-policy follow-up](2026-11-13-guided-capacity-recommendations.md).

Date: 2026-11-13. Base source: `fab5ae05` (`origin/main` when this work began).
Changes and regressions are on `codex/competitive-guided-alternatives`.
The installed app inspected was Stark 0.21.1. Its guided ranking, discovery
and leaderboard files matched the base source byte-for-byte. No model worker
was started and no weights were downloaded for this investigation.

## Reproduced recommendation gap

The packaged Coding / 128 GiB snapshot, captured 2026-11-11, contains:

| Parent model | Cached reference score |
| --- | ---: |
| Qwen3.8 Flash Next | 76.1 |
| Qwen3.8-27B | 73.7 |
| Qwen3.6-27B | 65.3 |
| Qwen3.5-27B | 62.3 |
| Qwen2.5-Coder-14B-Instruct | 51.0 |
| Qwen2.5-14B-Instruct | 48.4 |

The installed ranker reproduced the screenshot's three-model ordering from
the three corresponding candidates. The report simply truncated the sorted
eligible list to three; it imposed no relative quality requirement. Swift
then used identical leaderboard recommendation text without displaying the
score on the card. The exact quantized export was not benchmarked by this
ranking. These scores are separate parent references, not accuracy percentages.

The fix keeps Quality alternatives within 10 cached task-score points of the
best admitted candidate, removes repeated parent-model exports, and does not
fill remaining slots with distant or unknown scores. With no cached coverage,
Quality offers a single explicitly unknown fallback. Legacy task/preferences
retain their ordering. Cards display the cached score and its quantization
limitation.

## Why stronger candidates were absent

### Current available memory

An aggregate observation on this Mac found 128 GiB physical memory, a Metal
recommended working-set budget of 115,448,725,504 bytes, about 72.35 GiB currently
available RAM and about 196.98 GiB free disk. Available RAM changed during
later observations; these are observation-bound values, not a reconstruction
of the screenshot's original request.

| Fixed native source | System/startup plan | GPU plan | Observed reason excluded |
| --- | ---: | ---: | --- |
| Flash Next Lily Q4-XL | 87 GiB | 81 GiB | Current available RAM below 87 GiB; total RAM, GPU and disk passed |
| Flash Next MLX-Serve mixed | 88 GiB | 82 GiB | Current available RAM below 88 GiB; total RAM, GPU and disk passed |
| Flash Next original mixed | 111 GiB | 111 GiB | GPU plan exceeds 85% of the Metal budget; current RAM also insufficient |
| GLM Flash Q2 | 111 GiB | 111 GiB | GPU plan exceeds 85% of the Metal budget; current RAM also insufficient |
| Flint INT4 | 29 GiB | 29 GiB | Resource checks passed; no cached Coding parent score |

Flash Next's two admissible GPU plans reference the 76.1 parent score, above
the screenshot's 73.7 main choice. Recommendations correctly require the same
current-memory predicate as startup. The fix names a scored option blocked
only by current available memory, displays required and available memory and
offers a fresh check. The option remains outside installation/start actions.
When the selected fallback is unscored, the message makes no comparative
quality claim. Synthetic regression checks restore the stronger offer after
available memory recovers and reject advice for GPU, disk or excluded sources.

### Bounded search policy

Guided discovery previously requested planning rows with score prioritization
disabled, inspected at most 30 candidate repositories on a page and stopped
after finding any downloadable model. Downloads/popularity order and fixed
seeds therefore determined which metadata reached final ranking. Browsing
already supported a bounded, score-prioritized 300-position listing window.

Guided discovery now uses that window with the selected General Work/Coding
task. Exact scored parent repositories and revision-reviewed exports are
inspected ahead of inferred quantized variants. Search priority never grants
a guide score; planning rows remain unscored and final ranking retains exact
source/revision bindings. General and Coding discovery caches are separate.
The existing family deadlines, page limits and three-family concurrency remain.

Deterministic regressions place a scored parent after 35 popular unscored
sources or 35 inferred exports and verify that it reaches guided inspection.
Separate task-score fixtures verify that Coding does not reuse General order.

Two sequential live metadata-only scans completed with partial coverage:

| Scan | Catalogue rows | Complete guide recipes | Families checked | Failed families |
| --- | ---: | ---: | ---: | ---: |
| Installed baseline | 332 | 31 | 19 | 2 |
| Updated Coding discovery | 347 | 37 | 19 | 1 |

Both reports marked discovery incomplete. These sequential network observations
are not a controlled coverage or performance comparison. The updated scan
retained the exact Qwen3.8-27B export at revision
`7dd276b77fcb27c982ebd8d8a434539345713fea` and Qwen3.5-27B at
`fc05daec18b0a78c049392ed2e771dde82bdf654`. The latter's 62.3 score is outside
the new 10-point distance from 73.7 and no longer fills an alternative slot.

### Explicit support and evidence blockers

Qwen3.6-27B was also checked directly through a single-repository metadata
catalogue scan. Its public Hub record existed at
`8f33b98937c6031bb49e713dee3e8e185aafd26f`, with the expected Qwen3.5-family
architecture hint and Safetensors inventory. The built-in Stark record has
`adapter: null`, `conversion: planned` and `qualification: pending`.
`model_catalog.examine` excludes a registered record whose adapter does not
match the requested family, before configuration/preflight inspection. Search
ordering cannot promote this pending checkpoint. Its qualification is outside
this recommendation-policy change.

Gemma 4 has conditional Advanced conversion support, but its layout is absent
from the guided memory planner and recipe allowlist. Adding it requires a
route-specific supported memory plan; this change does not whitelist it as
generic dense attention. GLM Flash and Flint have no matching cached Coding
score in this snapshot. Unreviewed third-party quantized exports also do not
inherit guide scores merely from their names or Hub tags.

## Validation scope

The regression checks cover ranking distance, unknown scores, distinct parents,
legacy behavior, current-memory advice/recovery, selected-task discovery,
source inspection priority, subprocess arguments and task-specific caching.
Swift checks exercise native report parsing, card score presentation and
memory advice using synthetic data. A full app source typecheck also passed.
Final test counts are recorded in the implementation PR.

At the initial implementation (`a3c1a382`), the full converter suite reported
7,746 passed / 95 skipped, and 80 app/setup/build/window harness checks passed.
The final memory-message adjustment at that revision was separately checked
by 213 affected converter tests and a 19-test beginner harness rerun. These
results describe the initial selection policy; the capacity-policy follow-up
records validation for the user's revised direction separately.

No answer-quality benchmark, model load, generation, installation, release or
installed-app replacement was performed. This record establishes recommendation
policy and metadata-discovery behavior, not model quality or native qualification.
