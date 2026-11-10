# Model-family double-check before pull request

Attempt ID: `20261110-FAMILY-RECHECK-01`.
Baseline branch head: `cbd8b0224440ec7cd3bb2084c37ed0c5f26141fb`.
Status: independent reviews completed; corrections and validation in progress.
Registered before running revised native controls. No installation or merge.

Three independent reviews cover native execution/sampling/state, image handling
and memory admission, and converter/metadata compatibility. The converter review
found no concrete regression; its targeted tests pass 42 cases. Initial source
review found the following gaps before a PR was opened:

- The snapshot worker regression did not isolate several invalidation keys:
  context was tested after clearing the entry, and later divergence/scope misses
  also changed other inputs. Each revised gate must prime and publish a matching
  baseline, prove a 1,024-row plan, then change one condition. Cached/fresh output
  comparisons remain exact and content-free. Test both synthetic full-attention
  and sliding-ring targets, greedy and seeds 0/7/42.
- A finite reciprocal does not ensure finite temperature-scaled logits.
  Temperature 1e-38 with finite values can still overflow. The candidate must
  avoid silently choosing a lower logit and avoid a request-triggered model
  reload, while preserving existing arithmetic and draws in ordinary ranges.
  The precise repair and numerical controls will be recorded before adoption.
- The original tower estimate summed a loose block bound across every layer,
  despite scratch-pool reuse within a command. A square Gemma 4 E2B image was
  charged about 5.3 GiB of tower workspace. The revised inventory must cover
  allocator-rounded live scratch classes, persistent constants, patches and
  readbacks without multiplying reusable storage by depth. Compare the bound
  with observed production-path pool allocation on a synthetic tower; no speed
  or full production image-artifact qualification follows from that control.

Existing source-bound validation is preserved in
`../archive/runs/2026-11-10-model-family-triage.md`. Revised receipts will bind
new source/executable identities separately. Prompts, responses, token vectors
and probability arrays stay private and ignored; publish aggregates/hashes.
The full image acceptance matrix and actual Gemma 2 reference replay remain
separate qualification gaps and are not closed by this review.

## Attempt history before the final candidate

The first integrated compile failed in the new allocation controls: private
allocator access, ambiguous backend inference and a missing trait import.
The control was changed to use the normal Metal command allocator and its
existing reservation probe, preserving production module visibility.

The first centered-temperature candidate failed the combined CPU/Metal control
with an invalid-token assertion (zero passed, one failed), and the actual dense
worker survival control also failed (zero passed, one failed). The original
kernel assertion did not identify its backend/dtype; that failed run alone does
not localize the failure. Revised assertions include content-free labels.
The next candidate reconstructs a positive subnormal temperature reciprocal
from its stored mantissa using normal-range arithmetic, then applies the same
centered retry. Exponent-nonzero reciprocals retain the existing operation.
The physical rerun is required before adoption. The first dirty source patch was
not sealed, so its failure receipt binds the executables and named mechanism,
not an immutable source commit; final candidate source will be committed.

The three allocation controls already pass on the first executable. Actual
constant/pool reservation remains within its inventory and pool reservation is
unchanged between one and three synthetic production-tower layers. Both revised
cache fixtures also pass the isolated-key controls. Exact allocation results
are in [the workspace record](../archive/runs/2026-11-10-vision-workspace-inventory.md).
