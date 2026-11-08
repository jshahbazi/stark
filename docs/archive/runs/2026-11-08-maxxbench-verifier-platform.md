# MaxxBench verifier platform qualification — 2026-11-08

## Result and scope

A controlled stock Anko oracle passed all 20 feature checks and 104 regression
checks, with reward 1. The only execution change was a disposable verifier
config selecting the captured source image's `linux/amd64` platform. Default
verifier builds on the same ARM64 Docker host repeatedly failed before grading.
This establishes a verifier platform handoff defect for this task.

This is environment qualification. No model ran, and the result is not a model
score or an 89-task MaxxBench score. The earlier infrastructure failures remain
preserved as separate unscored attempts. The committed implementation adds
strict metadata/conflict guards to the controlled mechanism; it still requires
fresh oracle qualification before a full campaign uses it.

## Bound source and experiment

- Caller: `e4b7e29e80f45c29e1e6470155df905d129e626a`.
- MaxxBench: `1a0df5e9c5d4bccbae131b01830d478088fea947`, 89 released tasks.
- Pinned upstream handoff source SHA-256:
  `0a9ac622f135fbfa3953e9edc46031ce0e731b55cf498614b718a8d2ec887bb1`.
- Runner SHA-256:
  `0797a521a83ac670a8bf1f7224585e11e0824b4a2cfaa5dd529b697ffd10b573`.
- Pre-change verifier helper SHA-256:
  `a33cf0ef6b013c0f507c46e742c03de77ba54e393ff93452d8a9c91cc2a5299a`.
- Controlled observer SHA-256:
  `1796b79c4c1cad8acc179aa8c28f626e65cb11df22724a04e758840cb11f712d`.
- Committed platform helper SHA-256:
  `fa3d7472c339b9ba0e2affc43ccef4e58f084b9e595425ca7b8d00fbd60bc0b4`.
- Python 3.12.13; locked Pier 0.3.1. Existing dependencies were reused.
- Existing native worker SHA-256:
  `a589fb72b94ffd8381642ab898f348d53bc3d78d5e59c36ddf294e023ebb2d79`.
- Existing template-helper SHA-256:
  `1765a2c28309db99b8095da015035b89acc2cf4759752fba931079121421eb97`.

The native binaries were hashed for provenance and were not executed by the
oracle. Recorded task/native/helper/source file hashes stayed unchanged during the
controlled attempt. The shared benchmark lock serialized the complete run.

## Diagnosis without content dumps

The source environment and reference solution completed in the failing
attempts. A diagnostic wrapper delegated original Docker command arguments,
checks and timeouts unchanged, retaining fixed classifications, numeric exit
codes, timing and allowlisted traceback code identities only. Synthetic secret
tests covered command arguments, output, exception text and image metadata.

The failure was verifier `build`, return code 1, before grading. In-memory
predicates matched base-image resolution failure and denied registry fallback.
The generated `BASE_IMAGE` reference matched the actual source reference and
resolved to the same captured immutable image ID immediately before build.
Both reference hashes were
`f8cc915f3e1622838e9e0db2059f686724fc475b77d413a16966ee55f4cddcd7`.
Wrong-tag handoff was therefore excluded. An explicit platform-mismatch error
predicate did not match; the controlled intervention established causation.

The actual source image was `linux/amd64`; its inspected metadata did not
include a variant. The daemon reported `aarch64`, and no global
`DOCKER_DEFAULT_PLATFORM` was present. Public manifest inspection also found
that Anko's pinned task base publishes only AMD64. The passing Abs/Adaptix base
publishes both AMD64 and ARM64.

## Controlled intervention and guards

The observer deep-copied the disposable view's trial config, including
`environment.env`, and added `DOCKER_DEFAULT_PLATFORM=linux/amd64`. Pinned Pier
forwards that config to only the separate verifier's Compose subprocesses and
verifier exec environment. Source config/environment, global variables, task
files, native options and the stock upstream hook remained unchanged.

Both the actual source and verifier images were AMD64 after this change. The
oracle ran from `09:51:25.057891` to `09:53:21.974131` UTC. Source image ID:
`sha256:cb79ddc5c0e0d1cc5edd4f5d5f23fb994f8d85acc81bc2e0ca25aeceec5bad31`.
All feature and regression checks passed. Cleanup succeeded, removing one
verified source image and 18 eligible new cache entries while preserving all
61 protected volumes. The lock was released.

The implementation keeps the existing `{id, reference}` ownership receipt
unchanged. A single bounded, quiet inspection verifies the ID and reads
platform metadata in memory. Missing/unsupported metadata fails closed.
Explicit conflicting platform settings in process, trial, task or Pier
`persistent_env` kwargs also fail closed. Deep copies preserve source mappings,
including Pier's kwargs precedence. This applies only to `ARG BASE_IMAGE`
verifiers. Other verifier paths receive the original trial/config unchanged.

Model-free validation passed 111 runner, verifier, resource and bridge tests,
including platform metadata, copied-config immutability, conflicts, absent
metadata, kwargs precedence, cancellation and source-file preservation. Ruff,
repository policy and diff checks passed. No native build or model inference
was performed for this change.

Per-attempt force-build/no-cache source policy and independent image IDs remain
in effect. No byte equivalence between separately rebuilt images is asserted.
Only aggregate results and hashes are published; prompts, responses, patches,
raw Docker/verifier output, credentials and exception dumps remain excluded.
