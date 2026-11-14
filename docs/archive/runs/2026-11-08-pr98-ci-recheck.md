# PR 98 CI portability fix and independent recheck

Date: 2026-11-08. Reviewed source:
`4fe11fad295ed081c888f0976b6c924ff4e8d1b6`, followed by the
process-metrics validation and tests recorded in this change.

## Failure and fix

The Ubuntu repository-check job failed in the newly enabled model-free
benchmark suite: `snapshot(0)` returned `unsupported` on Linux, while its
test expected `invalid_pid`. The macOS-only local run had passed because
the helper checked the platform before validating the argument.

Validate the PID before selecting the platform. Invalid arguments now return
the same static metadata on Linux and macOS, and valid PIDs on unsupported
platforms still return `unsupported` without loading a native library.
Regression coverage explicitly mocks both platforms, checks invalid PID
types and values, and confirms that a native-library failure exposes only
`api_unavailable`.

## Validation

- All ten model-free benchmark/telemetry test files from CI pass locally:
  116 tests and 12 invalid-input subtests. The regression tests exercise
  Linux behavior on the development Mac without calling native APIs.
- Repository policy and whitespace checks pass.
- An independent read-only review of the production Rust diff found no
  concrete regression in idle polling, retained-state expiry, failure
  cleanup, EOF handling or queue maintenance.
- A separate independent review of the benchmark harness, aggregate results
  and latest report confirmed the position-specific medians and the stated
  host-coverage limits. Quiet-admission failure stops the harness before
  measured requests; the digest endpoint uses the dedicated worker binary.

This follow-up changes argument validation and model-free tests. The
measured inference sources, private workers, numerical controls and published
benchmark artifacts remain bound to their original source and tooling hashes.
No new model measurement or GPU work is part of this recheck. The coordinated
benchmark chat approved the short CPU-only checks; heavy suites and compilers
are left to hosted CI during its reservation.

The historical [cross-backend record](../../benchmarks/2026-11-08-cross-backend-idle-wake.md)
retains its original 114-test local result. The additional tests above are
new validation and do not retrospectively qualify its missing host samples.
Hosted CI status must be checked against the pushed head; the previous
commit's other completed jobs passed, while its aggregate CI gate failed
because of the repository-check failure described above.
