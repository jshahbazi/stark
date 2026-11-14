# Guided native and recurrent model coverage

Date: 2026-11-08  
Implementation: `86df441888db61488cde5d7147fbf716efec9fd3`  
Final staging guard: `cdb445ecfaf7e71a37e7b810c0e91f3dedad70cf`  
Text-only reuse guard: `8c93fd4aa9f6b17fe785a8659ea933d64339a643`  
Base: `dca0b182351cbd2602349e8cb41c6df02dd52fe9`

## Implemented scope

Beginner now considers every fixed native preparation source: Flint INT4, GLM
Flash UD-Q2_K_XL, the original Qwen Flash Next mixed export, Lily Q4-XL and
MLX-Serve mixed. The catalog's Qwen3.5-family text path also has a DeltaNet
planner, including declared MTP attention, F32 recurrence/convolution, verifier
state and retained prefix snapshots. Other unplanned layouts still fail closed.

Flint is exposed as a fixed native catalog family. Guided installs dispatch to
the existing native preparer, retain offer/operation fencing and recheck
resources. Installed native reuse verifies weights and metadata revisions,
quantization, prepared configuration, template/parser, shard counts and sizes.
It does not hash model payloads during recommendations. Loading still performs
native tensor admission.

The packaged profiles preserve configuration and template hashes, exact payload
filenames/sizes/ETags, selected-download manifest hashes and parameter-card
revisions/hashes. Published parameter counts are architectural metadata, not
measurements of answer quality. Qwen's 125B count is the source's conventional
parameter count, excluding its separately reported n-gram table and MTP head;
the memory plan still charges those stored bytes and the loaded head state.

## Memory and storage policy

Native request inventories follow the checked source formulas in
`engine/hybrid/{flint,glm,qwen}.rs` and the Stark-authored vendored
`qwen4exp/request_memory.rs`. The Python planning port includes conservative
growth, graph and dense-SDPA scratch bounds, plus a 3 GiB loading/workspace
allowance. It is a planning estimate, not a measured RSS peak or a native
qualification at every context. Source precision policies are preserved.

At 32,768 context tokens the rounded plans are:

| Pinned choice | System estimate | GPU estimate | New setup space |
| --- | --- | --- | --- |
| Flint INT4 | 29 GiB | 29 GiB | 26,998,097,624 bytes |
| GLM Flash UD-Q2_K_XL | 111 GiB | 111 GiB | 113,149,256,576 bytes |
| Original Qwen mixed | 111 GiB | 111 GiB | 110,624,216,261 bytes |
| Qwen Lily Q4-XL | 111 GiB | 82 GiB | 111,334,340,216 bytes |
| Qwen MLX-Serve mixed | 111 GiB | 82 GiB | 111,732,591,216 bytes |

The paged n-gram table remains fully charged to system memory for Lily and
MLX-Serve. Their GPU plans charge row staging rather than the full CPU file
mapping. GLM and the original mixed export receive no full-weight GPU discount;
the default per-encoder bank cap is not a total residency bound. They can fail
Beginner's 85% budget on a 128 GiB machine while retaining their Advanced paths.

Native setup uses one original weight copy, bounded metadata and the preparer's
4 GiB reserve. Partial/completed downloads receive credit only for allocated
blocks tied to the exact source's reusable payload or Hub incomplete paths.
The bounded scanner acquires the existing lease, rejects symlinks/nonregular
metadata and ignores unrelated files, sparse logical capacity, stale Hub
metadata and duplicate copies of the same payload. Reports expose remaining
setup bytes without local paths, and install admission refreshes the credit.

## Validation

- Full offline converter suite at the implementation commit: 6,717 passed,
  96 skipped in 266.54 seconds.
- Both final guards passed the 14-test native/recurrent suite afterward,
  including dangling output symlinks and rejection of image-capable artifacts
  under a text-only memory plan. The remaining implementation was unchanged.
- Native Beginner state/report harness: 11 passed, including legacy reports,
  separate GPU/system values and bounded remaining-space estimates.
- Controller harness: 134 passed. Its friendly-family expectation was updated
  for Flint; its remaining consumer, request fencing and lifecycle checks passed.
- App builder tests: 46 passed. The packaged runtime sanity check constructs
  native recipes so missing profile assets fail before installation.
- Full Swift app type-check passed for `arm64-apple-macosx26.4`.
- Native generation worker built with locked Rust 1.98.1 dependencies in the
  isolated checkout. No engine or inference-state code was modified.
- Repository policy, converter Ruff lint/format and whitespace checks passed.
- Independent read-only reviews checked native inventory formulas, source
  matching, install fencing, cancellation/retry and storage-credit handling.

Metadata-only checks against the actual registered library matched both
installed Qwen native artifacts to their exact profiles, including the complete
prepared-config hash. No library records or model files were changed.

A live metadata scan checked 18 families with zero family failures, returned
228 catalog entries and yielded 41 unique planned candidates with the pinned
starters. Under the reported 128 GiB physical / 107.52 GB Metal / 63.72 GB setup
envelope, Coding + Balanced selected Flint INT4, Coder 14B Instruct and Coder 7B
Instruct. Lily and MLX-Serve passed installed-memory admission but needed more
space for fresh downloads. These are selection replays, not quality results.

## Qualification and remaining work

No weights were downloaded and no new model generation or benchmark run was
performed. Existing native tool workflow evidence remains bound to its original
builds/artifacts, including the 8K-context records under
`2026-11-07-recommended-model-tool-qualification.md` and
`2026-11-07-glm-flint-tool-qualification.md`. This change does not qualify a new
precision path, enable images, enable Flint prefix reuse or alter the existing
native prefix-cache implementations. Existing image-capable artifacts do not
inherit this text-only memory plan. The installed app was not replaced.

The [recommendation evidence policy](../../MODEL_RECOMMENDATIONS.md) proposes
LiveCodeBench, a later SWE-bench track, BFCL/local tool checks, IFEval and local
writing/file tasks. No score import, evaluator or benchmark ranking is claimed.
