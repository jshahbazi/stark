# Lily sampling-specific policy experiment sources

Exact as-run snapshots for
[the policy implementation and qualification](../../../benchmarks/2026-11-15-lily-sampling-mtp-policy.md).
The aggregate record binds sources to worker, patch, artifact and corpus hashes.
No weights, generated content or token arrays are included.

| Snapshot | Executed location and purpose |
| --- | --- |
| `validate_policy_initial.py` | `runtime/lily-topk-policy/validate_policy.py`: 44 successful checks, then image-helper setup failure. An identical source copy at `validate_policy_initial.py` supports the continuation import. |
| `continue_image_policy.py` | `runtime/lily-topk-policy/continue_image_policy.py`: binds the retained initial receipt and runs only the missing three-check image cycle with the existing template helper on PATH. |
| `run_performance_initial.py` | `runtime/lily-topk-policy/run_performance.py` initially; preserved there as `run_performance-05-as-run.py`. Its exact-directory precondition rejected nine ancillary files before any worker started. |
| `run_performance.py` | `runtime/lily-topk-policy/run_performance.py` continuation: keeps the original 63-file guard and separately pins nine ancillary files before the first 64-request comparison. |
| `publish_performance.py` | `runtime/lily-topk-policy/publish_performance.py`: validates the complete fixed matrix and emits all sanitized observations, per-cell statistics and predeclared checks. |
| `publish_qualification.py` | `runtime/lily-topk-policy/publish_qualification.py`: binds the initial/continued functional receipts, prefix proof and complete performance aggregate into one content-free public record. |
| `qualify_sampler_prefix_transition.py` | `runtime/lily-mtp-policy-qualification/qualify_sampler_prefix_transition.py`: sequential cached/fresh/before workers, automatic 1↔2 transitions, image invalidation and reuse recovery. |

Restore the snapshots at their executed runtime locations and original names
to reproduce their path semantics. They intentionally retain exact local
binary/model/corpus paths; another environment needs a separately recorded
adaptation and matching identities. Read the benchmark plan before running;
do not overwrite a completed receipt or repeat an unfavorable timing result.

The image and prefix tools need the sealed `stark-template` on PATH. Its source
did not change in this implementation. Use the pinned converter environment
with `uv`, the sealed before/candidate binaries, the prior payload identity
receipt, frozen corpora, and `build-identity.json` containing the native and
Python patch hashes. Binaries, native state, raw runtime receipts and frozen
workload copies remain private/ignored.
The combined publisher also consumes the privately retained initial publication
and publisher-source copy to record an attempt-label metadata correction;
their hashes are in `publication_metadata_revision`. No native requests were
repeated for that correction.

The prefix tool's as-run `corpus_sha256` field comes from the prior benchmark
identity receipt. Its 4,615-token prefix is generated separately in memory by
the source-bound template/session builder; that field is not the hash of those
synthetic input tokens. The combined public record labels it as the prior
benchmark corpus identity. All seven generated-output hashes and exactness
comparisons are retained.

Source clocks are unavailable; no incompatible observer is launched. Host
observations contain numeric/boolean aggregates and never gate measurements.
Only per-depth matches are required across stochastic runs; different draft
depths have different valid random-draw schedules and output hashes.
