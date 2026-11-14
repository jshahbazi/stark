# Guided recommendations use installed RAM

Date: 2026-11-13. Base: `a3c1a382`, the initial implementation on
`codex/competitive-guided-alternatives`. This record supersedes the
available-memory selection policy in the [initial investigation](2026-11-13-guided-alternatives.md).

## Final policy

The user's 128 GiB of installed RAM describes hardware capacity. Other apps
changing available memory must not hide stronger models or change the ranking.
`recommendation_allowed` now checks installed physical RAM, the existing
2 GiB physical reserve, Metal budget, source eligibility and setup disk space.
It does not depend on the available-memory probe, including a missing probe.
The full planned runtime must fit physical RAM even for legacy clients that
do not provide GPU metadata.

`installation_allowed` retains the existing 2 GiB available-memory allowance
for a new preparation job. `execution_allowed` rechecks the model's actual
startup requirement after releasing the previous worker. Current-memory
pressure can add the existing card reminder to free memory before starting.
It no longer excludes that model from recommendations. The initial separate
memory-blocked-model explanation was removed because the stronger model is
now an ordinary offer.

The existing task-specific cached-score order, 10-point Quality alternative
distance, parent deduplication and bounded discovery prioritization remain.
GPU capacity, disk and model qualification remain eligibility requirements.

## Model-free observation

On the observed 128 GiB Mac, with its Metal budget of 115,448,725,504 bytes
and sufficient current disk space, native Coding/Quality reports selected
`fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`. Repeating the same report with
available memory set to missing, 0, 2, 72 and 100 GiB retained that choice.
These are controlled aggregate metadata inputs, not model executions.
The 87 GiB startup plan and 81 GiB GPU plan are unchanged. This observation
does not assert that the model can start when less than its startup memory
requirement is available.

## Validation

- 303 affected converter checks passed across guided models, native profiles,
  installed models, discovery, leaderboard and benchmark-reference handling.
- Regressions cover stable Quality and Speed offers under low, zero or unknown
  current memory, native 128 GiB offers, physical/GPU/disk exclusions and
  the preserved startup boundary and preparation working allowance.
- The Swift report check retains a low-memory offer and its startup reminder.
- 80 app/setup/build/window harness checks passed; the full menu-bar Swift
  source typecheck passed with no diagnostics.
- Ruff lint/format, repository policy and diff checks passed.

No model worker, weight download, new backend, artifact qualification or
installed-app replacement was performed for this follow-up.
