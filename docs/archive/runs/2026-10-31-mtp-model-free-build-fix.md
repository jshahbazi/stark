# MTP model-free build follow-up, 2026-10-31

Implementation: `c2910489112d0e6e7e2f3f5879f511970e810985`.
This adds feature-matrix coverage to the preceding
[PR review](2026-10-31-mtp-pr-review.md), whose passing Metal configuration
did not establish model-free build compatibility.

## Finding and fix

The common generation protocol imported its MTP descriptor from the optional
`stark-engine` dependency. `cargo test --locked -p stark-generate
--no-default-features` failed with E0433 because that dependency is absent.
The initial hosted Repository checks job also failed at its model-free
generation step.

The common protocol now owns its descriptor, and the Metal adapter translates
the loaded model's four capability fields explicitly. Shared protocol code
therefore compiles independently of the native engine. Draft bounds,
recommendations and sampling/verification labels retain their values.
Dependency locks and feature defaults are unchanged.

## Validation

- Model-free generator tests: **49 passed**.
- Metal release generator tests: **85 passed, two ignored**.
- Release generator Clippy passed with warnings denied. Rust formatting,
  repository policy and whitespace checks passed.
- Independent review confirmed the field mapping and absence of unguarded
  optional-engine references in model-free modules.
- The release worker passed the nine-request real-checkpoint toggle smoke
  again. All enabled and disabled route hashes and capability values match
  the pre-fix worker exactly, including invalid-request recovery and repeated
  on/off/on selection.

Tests ran on the fixed working tree subsequently committed above; the private
test runner retains parent HEAD `283d84bd7a3eabbd48ad4d06b1770e87b18de49b`.
The rebuilt worker SHA256 is
`8ea0b89a10a14346b62d123154041a44a8326c19a2e698826eaf387fc1ae1cd3`.
The private smoke validator SHA256 is
`56bd8a1abc5805ee7d2984f0da2dac8af3a67746145451c144c5891c515fc0bf`;
only its worker and output-receipt paths differ from the preceding smoke.

The Python implementation, inference kernels, quantization and exact sampler
did not change. The preceding 5,094-pass Python run remains its separate
validation record. Earlier workers and reports are preserved; no new
throughput measurement is claimed. Native payloads remain private and ignored.
