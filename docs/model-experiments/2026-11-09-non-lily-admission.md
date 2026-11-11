# 20261109-ADM-04: non-Lily resident-memory accounting

## Identity and acceptance

Codex, 2026-11-09. Memory/lifecycle infrastructure follow-up to
[ADM-01/02/03](2026-11-09-resource-admission.md), starting at `429dc7c8` after
merging main through `799558e9`. Implementation is committed at
`08fb0075cf8d821178eaca1ed449e95e3f969fad`; final-source checks and bounded
native controls passed. No new installed-build result is claimed.

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

### Final-source validation

All following checks use implementation `08fb0075`:

- Full physical-Metal engine suite, pinned nightly and `cpu,metal`: 979 passed,
  zero failed, 137 ignored; 253.20 seconds. Integration groups passed seven and
  two tests; three doctests passed. This includes actual mapped-page/owner
  credit, page-sensitive padded-prefix geometry, real queue-gated retirement,
  live-allocation preservation and token-history capacity-transition controls.
- Generation, Rust 1.98.1 release: 139 passed, three ignored. The private native
  warm-context test was additionally executed: one pass, proving a reusable
  context can fit when the cold minimum does not fit the same allowance.
- Scoring and digest entry points: 34 tests each, plus one protocol test, in
  both separate default and optimized release configurations; all passed.
- Dense MTP private toy controls: two explicitly executed ignored tests passed.
  They cover active/off admission, retained owner credit, full-sequence/max-four
  geometry, foreign-model/context/cancellation guards, greedy equivalence and
  exact prefix restoration. This synthetic artifact is not a production model
  quality qualification.
- Separate locked Rust 1.98.1 release builds for generator (default Metal),
  optimized scorer and template passed. Separate release Clippy checks with
  warnings denied passed for generator, optimized scorer, template and resource
  budget. Final formatting, Ruff and repository policy checks passed. The
  earlier complete Python result applies to unchanged final Python source.
- Model-free process boundaries: generator six cases passed; scorer's exact
  invalid-argument JSON, exit status and empty stderr check passed.

The generic Qwen worker completed seven bounded greedy seed-42 generations on
one worker: cold/warm, changed scope, 1,152-to-3,072 input growth, a fresh growth
reference, changed context and recovery after a rejected context. Both warm
requests reused 1,024 input tokens. Fresh/reference output hashes matched.
Request-aware Auto selected context 8,192 and allowance 7,040; actual generation
was deliberately limited to eight output tokens. The resized control used
16,384 context. The artifact reports 38,912 positions, and 38,913 was rejected.
This is not full-window execution or a memory/performance benchmark.

The optimized scorer completed four successful 1,152-input requests on one
worker. An identical consecutive request hit the enabled cache; changed context
and a request after a rejected short context missed it. All successful logits
hashes matched. Both private workers were closed afterward, and a process check
found no remaining owned model/native-test process or running Stark app.

Two initial private harness controls were invalid: generation's negative check
expected an exception instead of inspecting the structured error reply, and the
scoring harness expected a cache hit without enabling `--consecutive-cache`.
Their failed aggregate receipts were preserved. Corrected harnesses passed
without changing product source or binaries.

An extra distribution-verifier invocation rejected the checkout scorer's
Mach-O minimum OS of 11.0 (SDK 26.4); that check requires minimum OS 26.4 and
also enforces the separate packaging contract.
Source-checkout builds here used the documented Cargo commands, without the
packager's explicit `MACOSX_DEPLOYMENT_TARGET=26.4` environment. The direct
startup-protocol check passed.
These executables were not packaged, installed or promoted as distribution
artifacts. The installed app remains the separately recorded ADM-03 build.

### Reproducible identities

The runtime source manifest contains 1,285 tracked files from the source trees
and manifest/lock inputs selected by the private receipt helper. Sort paths;
concatenate UTF-8 path, NUL, binary SHA-256 of file bytes (link-target bytes for
symlinks), LF; hash that stream. Its SHA-256 is
`1afdaee651310da5494ab07512e89d92a4630a0dd4c1c66149f061e311e2f139`.

| Build input/output | SHA-256 |
| --- | --- |
| Generator | `08f2d530ad813b68ce645d726fb19aa61ee8c0b4b816d44e6c7496b03f250bbb` |
| Optimized scorer | `04f9f40df98e2b589afa0ce81907d174348d9cfd6bb0f2d9240689ef5c71a11c` |
| Template | `84400ce69a958528402e3a2d6c3dc2e2ba641a0504b9f9e62d0251c749c2cfe4` |
| Cargo lock | `85c73ea6bdb1ece383aca9502820095be4dba8818d6d7e348caca938507e18d6` |
| Converter lock | `df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5` |
| Generation aggregate receipt | `2663d5561fb93afb254fe49eadbf3e1b10e4d208a489c912233c5102794aee0c` |
| Scoring aggregate receipt | `b8df04ebbeda7ca1e13ee21315776b99b71926972711131d39c4173969ce4f50` |

Qwen payload was freshly streamed for hashing: 316,959,744 bytes, source
`Qwen/Qwen3-0.6B-MLX-4bit@173234aa840d113125e9f2271100ddbaf16c9620`, based on
`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`.
The dense-MTP toy payload is 430,432 bytes with no source-revision metadata;
its identities qualify only this serialized private regression fixture.

| Artifact | Configuration SHA-256 | Payload SHA-256 | Tokenizer SHA-256 |
| --- | --- | --- | --- |
| Qwen 0.6B Q4 | `60d4fb79e35aa0a6a72e7d71e53cd701df9f446930aa37d7e85390213194388b` | `eedff76a3d7466f2031dd4bae2a13138f025cb501c0316243ff31581d858805a` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| Dense MTP toy | `a879f860ee69a60a6e512b66bde0ea6aa0c8c5b9ba6ed59ab4689f79f4fcec04` | `156f79e885165762545012fde03c4c453d6631e5c1cc633acb44d8fe4dbc95ab` | `0ac7663391818bb8eaa3e6da828ed81a2738eec4bb8c3c2a0a4d2078b6c2c76e` |

### Promotion decision

Independent generic/hybrid source reviews are clear after the recorded fixes;
required local checks and bounded controls passed. Adopt `08fb0075` on the
resource-budget branch and open the authorized implementation PR. No merge,
new app installation, full-model hybrid qualification, full-window quality or
speed promotion is included.
