# Lily one-draft experiment tooling

These are exact as-run tooling snapshots for
[the recommendation study](../../../benchmarks/2026-11-15-flashnext-lily-one-draft-default.md).
Hashes in the aggregate record bind them to the measurements.

- `compare_depths.py`: explicit one-versus-two depth comparison on the sealed before worker.
- `compare_defaults.py`: before/candidate/candidate/before comparison using `mtp: true`.
- `validate_policy.py`: text/image readiness, selectors and image-to-text recovery.

They were executed from `runtime/lily-one-draft/` and derive the checkout root
from that location. Restore them there to reproduce the recorded path semantics.
They include the exact local model/binary locations used in this run; another
machine needs a separately recorded path/source adaptation and matching binary,
corpus and artifact hashes. They are provenance snapshots, not installed tools.

The existing prefix validator remains under `scripts/`; its source hash and
arguments are recorded in the report. Binaries, weights, full runtime receipts,
frozen corpus copies and native state remain ignored/private. The public data
contains counts, timing observations and hashes only.
