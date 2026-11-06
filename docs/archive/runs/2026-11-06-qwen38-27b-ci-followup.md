# Dense Qwen27 CI and catalog follow-up

PR #86 at `f0316d1b` failed two code-quality gates in Actions run
`42241881616`. The model's source-bound benchmark records remain unchanged.

## Repairs

- Repository checks: Ruff lint passed, but formatting failed for
  `adapters/qwen35_mtp.py`, `model_catalog.py`, `model_inventory.py`,
  `tests/test_guided_models.py`, `tests/test_model_catalog.py` and
  `tests/test_qwen35_mtp.py` under `tools/converter`. The pinned formatter
  reformatted those six files; Python ASTs before and after matched exactly.
- Scorer (default): the generator release build passed; its Clippy step failed
  on `collapsible-if` in `native.rs`. The replacement let-chain preserves all
  conditions: scope and prefix position must be present, the position must
  equal the captured boundary, and it must be positive before retention.
- Main was merged cleanly to retain the current website and onboarding copy.
  No model artifact or recorded benchmark output was changed.

## Local validation

| Check | Result |
| --- | --- |
| Ruff check, complete converter tree | Passed |
| Ruff format check, complete converter tree | 146 files formatted |
| AST comparison for six reformatted files | All unchanged |
| Generator release Clippy, all targets, warnings denied | Passed |
| Generator release tests | 119 passed, two ignored |
| Focused converter/catalog/inventory/guided-model tests | 234 passed |
| Independent review of the seven-file repair | No semantic regression found |

These are repair checks, not a new throughput qualification. The
[51.483 tok/s measurement](../../benchmarks/2026-11-06-qwen38-27b-packed-followup.md)
continues to identify its original source bundle and worker executable.
Hosted CI must be checked at the subsequently pushed head before reporting it
as green.

## Model chooser

A metadata-only call through the actual `family_catalog("qwen3_5")` helper
returned the exact Jundot checkpoint as downloadable on the first catalog page:

| Field | Observed value |
| --- | --- |
| Family in Advanced Model Catalog | Qwen3.5 dense |
| Item name | Qwen3.8-27B-oQ4e-mtp |
| Publisher | Jundot |
| Size / quantization | 27B / mixed 4/5/6/8-bit |
| Artifact mode | Text |
| Source revision | `7dd276b77fcb27c982ebd8d8a434539345713fea` |
| Source weight bytes | 17,517,432,678 |

This proves catalog offering through the updated source. It does not update an
installed app, download the weights, or transfer runtime qualification to a
different build. Users need a build containing this PR for the new native MTP
and packed-weight support. The measured native artifact's converted weight file
is 17,078,038,512 bytes; source download size and runtime memory requirements
are separate quantities. Decision-mode suitability still requires its own
template, label and checkpoint checks.
