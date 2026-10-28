# Rejected native execution experiments

These source patches are excluded from production. Each applies independently
to the final production source; `git apply --check` passed for all three after
the experiments were removed. Replay one experiment at a time.

- `complete-bank-admission.patch` restores the optional near-limit full-bank
  admission trial. The normal 75% working-set policy remains in production.
- `selected-slice-cache.patch` restores the bounded FIFO for selected immutable
  ranges. It passed correctness and ownership checks and reduced measured GLM
  throughput.
- `sharded-residency.patch` restores the optional residency prototype with
  per-command attachment of used shards. That prototype reduced measured Qwen
  throughput. Queue-attached execution with a complete GPU graph remains
  untested by this experiment.

The [benchmark index](../../../benchmarks/README.md) links the measurement
records and their qualification boundaries.
