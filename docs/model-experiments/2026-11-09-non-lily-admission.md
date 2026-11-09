# 20261109-ADM-04: non-Lily resident-memory accounting

## Identity and acceptance

Codex, 2026-11-09. Memory/lifecycle infrastructure follow-up to
[ADM-01/02/03](2026-11-09-resource-admission.md), starting at `429dc7c8` after
merging main through `799558e9`. Implementation and validation are in progress;
no new model qualification or installed-build result is claimed.

The user requested correction of the remaining non-Lily memory overestimates,
an independent check and a pull request after validation. The scope includes
generic dense, dense MTP and supported vision decoder state, plus GLM, converted
Qwen and Flint hybrid execution. Existing Lily allocation accounting is the
predecessor, not evidence for these different artifact/backend paths.

Acceptance criteria set before candidate validation:

- Admission counts actual new allocations and required replacement overlap.
  It may credit only verified, reusable allocations owned by the loaded model
  and compatible state; live aliases, in-flight buffers and failed allocations
  never become speculative free memory.
- Cold, warm, growth and failure controls exercise allocator/state ownership,
  model/context/scope identity, MTP selection, snapshot lifetime, device limits
  and fresh-state fallback. Unknown or overflowing inventories fail safely.
- Auto planning and explicit generation use consistent request-specific
  accounting. Legacy count-only planning remains compatible and receives no
  unproven prefix credit. The planner handles an isolated reusable-context
  candidate without treating its discontinuous cost as a monotonic search.
- Preserve model context and Metal buffer/index bounds, numerical policies,
  exact prefix restoration and cleanup. No arbitrary token ceiling is added.
- Run meaningful pinned-toolchain checks and bounded native controls on an
  available small generic Qwen artifact. Record untested artifacts separately.
  Open a PR only after implementation, review and required local checks pass.

## Source findings and design

Generic finite prefix reuse currently admits a full fresh state while its KV
allocation remains resident. Auto knows only an input count, so it cannot prove
the prefix, scope or selected MTP path; a loaded dense draft head can also be
charged when the request disables MTP. Generic allocator retirement and free
page caching mean dropping a state does not by itself prove OS headroom has
already increased. Persistent hybrid decode controls/state/scratch likewise
need actual-owner accounting rather than a blanket subtraction from a cold
formula. Scorer consecutive-cache hits already bypass native admission.

The candidate adds allocator-backed resident accounting and request-aware
planning metadata, gated by an advertised capability. Generic and hybrid
implementation are delegated independently, followed by cross-review. Public
reference pages change with the final contract. This ledger preserves failed
attempts and validation limits as work completes.

## Artifact and environment boundary

Rust 1.98.1 and locked Cargo dependencies; nightly formatting/engine harness
`nightly-2026-10-01`; Python 3.13.13 through locked uv. Use the worktree's
distinct target directory. Generate and optimized-score builds remain separate.

Available generic smoke artifact:
`Qwen/Qwen3-0.6B-MLX-4bit` revision
`173234aa840d113125e9f2271100ddbaf16c9620`, converted payload 316,959,744 bytes.
Its predecessor smoke identities remain in the earlier dated run record; new
candidate source/executable hashes and observations will be recorded separately.
No model download or installed-app replacement is required by this follow-up.
Hybrid synthetic geometry/lifetime controls are not full-model qualification.

No inference precision change is planned. Weights, activations, accumulation,
recurrent state and caches retain each artifact's current policy. No speed or
task-quality gain is claimed; allocator reclamation overhead is a possible
tradeoff to record rather than a benchmark result.

## Planned controls and stop conditions

Use synthetic, memory-only inputs, seed 42 and greedy sampling for native
functional controls. Exercise a chunk-aligned shared prefix, changed scope,
larger allocation and an invalid request followed by same-worker recovery.
Dense MTP/image-specific artifact execution may remain unrun where no admitted
artifact is available; source and model-free controls must identify that gap.
Publish counts, booleans, hashes and aggregate memory observations only.

Run one owned native validation worker at a time. Keep unrelated processes and
the installed app untouched. Stop the test worker on deadline, protocol failure
or unsafe resource observations; retain the failed/deferred result. Full-window
generation, long-output quality and matched throughput are outside this run.

## Attempt history and results

### Implementation and review controls

- Generic state credit uses actual dense payloads and successfully mapped
  sparse pages. Shared KV views carry no duplicate credit. Credit is capped to
  persistent state, preserving fresh scratch and new snapshots. Plain requests
  with a loaded dense MTP head inventory only the selected path.
- GLM inventories persistent decode controls, KDA state, attention capacity and
  optional scratch owners. Converted Qwen credits only owners surviving its
  graph reset; Flint credits retained descriptors rather than discarded rows.
  New/replacement owners retain their real overlap costs.
- Removed Flint's aggregate 32 GiB full-window startup rejection and converted
  Qwen's aggregate 512 MiB recurrent-state rejection. Checked arithmetic and
  actual per-buffer/index guards remain. Scalar backends no longer carry an
  unrelated fixed 512 MiB workspace pad.
- Generic allocation reclamation waits for resource-drop callbacks and drops
  fully free allocator pages outside the allocator lock. Only observed OS
  headroom is used after reclamation; estimated release bytes can trigger a
  retry but cannot admit it. Live, aliased and in-flight owners remain excluded.
- Modern Auto planning includes actual token identity, scope, sampling and
  selected MTP. A compatible retained context is checked separately from the
  monotonic cold search. Legacy count-only requests remain supported.
- Scoring has a separate inventory without generation history, sampler or new
  prefix snapshots. Consecutive identical logits-cache hits already bypassed
  native admission; an explicit regression now preserves that behavior.
- Greedy hybrid requests with MTP disabled omit absent stochastic sampler
  arrays. Their readout allowance covers the previous F32 host row, replacement
  F32 device row and new F32 readback row. Stochastic/MTP and legacy count-only
  requests retain their conservative independent-array allowance.

Independent review found two correctness defects in intermediate accounting:

1. Sparse prefill preparation maps a full largest batch at the last chunk's
   start, even when its final chunk is short. A fresh 1,025-row input can map
   2,048 rows. The estimate now follows the padded text/scoring chunk grid;
   image layouts with irregular chunk boundaries retain a conservative bound.
2. Subtracting retained history length from a cold context multiplier did not
   represent Vec replacement overlap. The corrected history class uses actual
   capacity, chunked prompt appends and single-token pushes. A retained
   1,024-element owner growing past its capacity charges the complete new
   2,048-element allocation. Already sufficient capacity costs zero additional
   history storage. Prompt copies and the exact preallocated returned-ID Vec
   remain separate classes. Dense MTP owns no ordinary u64 token history.

Both defects were corrected before promotion. Pure boundary controls include a
real Vec capacity-transition oracle; they do not rely solely on restating the
inventory formula. A related dormant hybrid helper edge at capacities one to
three was also corrected and receives explicit boundary assertions.
The final review also found a warm-state overestimate: the persistent-state
credit cap used suffix-only geometry while the request estimate used the full
retained sequence. Both now use the same padded-prefix geometry, preserving the
transformer-only credit boundary.

### Remaining conservative inventories

This change removes confirmed duplicate charges and arbitrary aggregate startup
ceilings; it does not claim exact prediction of every transient allocation.
Converted Qwen's DecodeGraph uses its actual 512 MiB command-arena maximum as a
bound because a complete static operator inventory is not available. Experimental
GLM wide schedules retain their real owner limits; scalar activation/copy
workspace also remains bounded conservatively. Stochastic/MTP host sampling
arrays retain their independent cold bound. These are workspace estimates, not
additional token ceilings.

Experimental GLM encoder-bank residency still reserves the configured encoder
state allowance when selecting resident expert banks. Those banks persist while
future requests can grow state; dynamic bank re-admission/resizing would be a
separate lifecycle change. Ordinary GLM request admission does not carry that
experimental full-window bank reserve. Full GLM/Qwen/Flint model execution and
vision artifacts are not qualified by synthetic owner/geometry controls.

### Preliminary validation and resource deferrals

The first complete CPU/Metal engine run on an intermediate source had 970
passes, two failures and 137 ignored tests. Both failures were new ownership
tests using an unsupported Qwen 3.5 multimodal-RoPE fixture; the real reuse guard
correctly rejected it. The fixture was changed to supported scalar RoPE. This
intermediate run is not final-source qualification. A test-build borrow error
in the first private warm-planning control was corrected before execution.

Before the final history correction, the full CPU/grammar engine suite passed
574 tests with 30 ignored; integration groups passed seven and two tests, and
three doctests passed. The generation no-default-feature suite passed 95 tests.
The final Python source passed the complete converter suite: 6,804 passed and
96 skipped. Ruff, repository policy and pinned nightly formatting checks passed.

An independent native numerical investigation reserved this machine's GPU and
then all native test execution. A CPU-only engine test PID initially triggered
that investigation's supervisor; the already-running suite was allowed to
finish, and subsequent native execution was held. Source review, compile-only
checks and Python validation continued. Earlier small generic Qwen and dense
MTP toy controls passed, but they predate the final sparse/history corrections
and will be repeated after the reservation is explicitly released.

Final source/build identities, post-correction test results and PR disposition
are pending. The installed app remains the separately recorded ADM-03 build.
