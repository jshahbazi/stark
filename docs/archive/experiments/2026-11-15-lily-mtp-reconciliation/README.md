# Lily MTP comparison reconciliation tooling

Exact as-run source for
[the reconciliation study](../../../benchmarks/2026-11-15-lily-mtp-default-reconciliation.md),
attempt `20261115-LILY-MTP1-03`. Public data binds source, worker, artifact,
corpus and receipt hashes. No model files or generated content are included.

- `run_initial.py`: the initial balanced 0/1/2 runner. It completed the 18
  before-worker measurements, then failed to aggregate an empty clock sample
  list. The executable's clock schema lacked required source intervals.
- `run_continued.py`: the separately hashed continuation. It reprocesses those
  saved 18 measurements, verifies their per-depth hashes/work and artifact
  identities, and measures only the five remaining candidate cells. Unavailable
  clocks produce null summaries. No completed native measurement is rerun.
- `publish.py`: checks the 198 measurements, fixed exclusions, cross-build
  primary output/work identity and the predeclared decision rule, then publishes
  counts, timings, host observations and hashes.

The executed locations were `runtime/lily-mtp-reconcile-20261115/run.py`,
`run-v2.py` and `publish.py`. Restore the first two snapshots under those names;
the initial-source hash check also expects an identical `run-as-initial.py`
copy. Paths derive the checkout root from that runtime location. Preserve the
original `results/` receipt directory before continuing to `results-v2/`.
Native requests are not resumed or reconstructed from published aggregates.

These snapshots intentionally retain the exact local binary, artifact and
corpus paths. Reproduction elsewhere needs a separately recorded path/source
adaptation and the matching identities. The existing source-bracket clock
helper is unchanged; ordinary timestamps are not substituted for its missing
interval schema. Clock data cannot be recovered for these requests.

All observed host conditions and all measured requests are retained. Fixed
warmups and selector probes are labeled separately. Full runtime receipts,
binaries, weights and frozen workload copies stay ignored/private. These
scripts are provenance snapshots, not installed product tools.
