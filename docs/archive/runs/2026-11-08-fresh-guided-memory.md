# 20261108-MEM-02: fresh-install resident memory planning

This model-free correction follows [MEM-01's actual-artifact observations](2026-11-08-qwen-guided-memory.md).
Owner: Codex, 2026-11-08. Branch: `codex/beginner-hardware-ranking`, PR #104.
Kind: recommendation compatibility and memory planning. Implementation and
validation are model-free. No model is loaded or run for this correction.

## Problem and acceptance

The resident plan was applied only while inspecting installed copies. A fresh
128 GiB Mac with 100 GiB available and adequate setup storage therefore still
excluded both pinned Qwen3.8 paged preparations using their 111 GiB fallback.
This reproduced with synthetic host resources and an empty model registry;
running a model is unnecessary to test the recommendation decision.

Acceptance: the same pinned recipes must plan 87 GiB for Lily / 88 GiB for
Serve before download. Resource checks must reject insufficient physical RAM,
current memory, Metal budget or setup storage. Installed reuse can waive setup
storage. An existing artifact that fails exact resident checks must be reported
and started with the conservative estimate. Unknown or altered recipes cannot
inherit the source-bound plan. Benchmark ordering must still operate after
resource admission.

## Change and provenance

The same shared memory inventory now includes a canonical SHA-256 binding to
each complete `guided_native_profiles.json` source row. This covers immutable
weight/metadata revisions, stored sizes, geometry, template and preparation
metadata. `native_recipes()` obtains its fresh estimate from that binding.
`source_profile()` also checks source identity, weight/paged sizes and context.
No machine identity or installed directory is required for source planning.

The existing local config/receipt/index hash and complete payload filename/size
checks remain independent. `with_resident_profile()` requires both source and
artifact validation at startup. Discovery downgrades matching installed copies
when none passes artifact validation, rather than retaining the source-only
estimate. All source/payload checks remain layout checks; payload bytes are not
freshly rehashed. Pinned preparation metadata generation is deterministic,
without timestamps; unexpected prepared bytes still lose the reduced plan.

No inference math, backend precision, sampler, cache key or prefix-restoration
rule changes. The native request gate's existing policy is unchanged. MEM-01's
measurements remain bound to its recorded source/artifacts; this correction
adds no measured memory, performance or model-quality result.

## Validation and disposition

Model-free tests cover fresh and installed cases, multiple RAM/GPU budgets,
available-memory and setup-space boundaries, source/layout drift and changed
local artifacts. Final commands/counts, source and installation status are
recorded below after validation. Native profile parser/scope checks use the
nightly CPU/grammar configuration and synthetic temporary files only.

Related [experiment ledger](../../model-experiments/README.md).

The focused converter files passed **73 tests**. The native profile/scope suite
passed **4 tests** with Rust nightly CPU/grammar and 585 unrelated tests filtered
out. Ruff, formatting and repository policy checks passed. The complete
converter suite is replayed before publication.

The resource matrix uses explicit per-artifact expected outcomes rather than
recomputing the admission predicate. It covers 16, 32, 64, 96, 128, 192 and
256 GiB physical/GPU fixtures; independent GPU budgets 48, 80, 96, 97 and
128 GiB; fresh setup space 32, 64, 80, 96, 128, 192 and 256 GiB; and exact
available-memory thresholds with one-byte-below rejection. Lily's GPU plan is
81 GiB, Serve's 82 GiB. The earlier MEM-01 narrative's shared 82 GiB GPU figure
was an over-rounding; this correction records the actual per-recipe estimates.
For example, a synthetic 96 GiB Metal budget admits Lily under the 85% rule
and rejects Serve. These are synthetic budgets, not measurements of another
Mac's recommended working-set budget.

Integration checks invoke `Desktop.publish_guided_models` with an empty model
registry, 100 GiB available, ample library space and no catalogue/network
results: both Qwens appear with 87/88 GiB estimates and empty installed IDs.
A matching but incomplete local copy loses the lower estimate and is excluded
at that same headroom. Source drift cases cover revisions, weight and paged
sizes, geometry and template metadata. No benchmark scores are added or changed.
