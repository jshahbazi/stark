# Model catalogue family admission validation

Date is 2026-11-10. Implementation starts from `fc4c3153` on
`codex/catalog-family-admission`.

## Implemented behavior

- Interleave architecture aliases and family-name searches, with repository
  deduplication and continuation offsets. Configuration identifies the family.
- Retain unsupported family exports with a reason and disable installation.
- Reuse the production converter planners against pinned Safetensors headers
  before enabling generic downloads. Validate tensor inventory, shapes, dtypes,
  affine packing and shard-index membership without fetching tensor payloads.
- Preserve the pinned native GLM and Flint entries while discovering other
  exports and explaining their importer limits.
- Continue guided discovery beyond pages containing only unavailable rows,
  within ten pages and a shared ten-second family budget.

Header-only checks defer duplicate tied-tensor byte equality to ordinary
conversion. Native execution and numerical qualification require their own
artifact checks. Each remote header has a 4 MiB inspection limit. Oversized
headers and servers ignoring HTTP ranges produce unavailable rows.

## Local checks

- Full offline converter suite passed with 7,084 tests and 94 skips.
- Final focused discovery, HTTP header and guided hybrid checks passed with
  55 tests after adding pinned-entry recovery and header-budget regressions.
- Header planner tests include exact production-plan equality for twenty
  synthetic layouts, sixty corrupted inventories, complete Gemma 3 and Gemma 4
  image inventories, and the retained source-backed tied-tensor byte check.
- Ruff lint and format checks passed across `tools/converter`.
- Locked converter dependency resolution passed.
- Repository policy checks and all 41 policy tests passed.

## Live metadata and header checks

First-page checks returned ready for Qwen3, Qwen3.5 dense, Qwen3.5 MoE,
Gemma 4, GLM-5.3 Flash and Flint. Supported Qwen and official BF16 Gemma
sources continued to pass their tensor planners.

The LM Studio Gemma 4 12B MLX 4-bit and 8-bit exports appeared on the first
page with the no-PLE global K=V decoder's BF16 requirement. The 4-bit export
previously appeared at candidate offset 11,340. Unsupported mixed-precision,
packed and GGUF variants retained explanations.

The Qwen3.8 external-table checks returned ready on offsets 0, 30 and 60.
`ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw` at revision
`a0e89db090a3e4b2e8ea2118fe9d0324d3854aca` was downloadable on offset 30.
The 3.3 bpw export on offset 60 was unavailable with its expert W3/group128
versus required W4/group64 explanation.

GLM and Flint retained their pinned downloadable entries. Other discovered
Safetensors and GGUF exports displayed their native importer limitations.
Their preparation code currently binds exact source artifacts.

## Conversion and generation smoke

The updated converter prepared `Qwen/Qwen3-0.6B` at revision
`c1899de289a04d12100db370d81485cdf75e47ca` and produced 226 tensors totaling
1,192,276,800 output weight bytes. The installed generation worker loaded the
artifact, returned readiness and completed the bounded generation request with
`finish_reason=length` and process exit code zero. Only aggregate status and
hashes were retained.

| Bound input or output | SHA-256 |
| --- | --- |
| Converter source at smoke time | `54a930acd3d7f011757a43d053c9344f2a061242d7ebdab23fde76579b10f4a1` |
| Converter lockfile | `df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5` |
| Source manifest | `e199732d889006d43907f5ebe0b09d136f3da4bb27482efd2b735a3128d0d5c7` |
| Output weights | `87c3344a80b7376392cc5c44d5369680b68936e6930bb8ac476ca65a136f64c7` |
| Output config | `dfa5763dc220e871b86a6119536f5c1c2862a5137a473eaf9d812be3cc53924e` |
| Output tokenizer | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| Installed generation worker | `a0611f77a05e1805c71955c664ae3a90c87a13e9422bc7982afebe0303338bf8` |

The worker came from the user's installed app. This run exercised the converter
output with that binary. Subsequent catalogue recovery and inspection-budget
changes passed the final focused tests. The full 4.7 bpw checkpoint was assessed
through its headers during this work.
