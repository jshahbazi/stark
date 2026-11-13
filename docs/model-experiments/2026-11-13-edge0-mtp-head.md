# 20261113-EDGE0-08: native MTP head component

## Identity and state

| Field | Value |
| --- | --- |
| Registered / owner | 2026-11-13 / Codex, local non-commercial evaluation only |
| Baseline | `e91a3fdf`, qualified Phase 4 and native MTP foundation |
| State / result | Bounded head-component numerical and actual-artifact lifecycle gates passed; worker MTP remains unavailable |
| Governing plan | [Phase 5](../EDGE0_PLAN.md#phase-5-mtp), native Qwen Metal graph |
| Predecessor | [20261113-EDGE0-07](2026-11-13-edge0-mtp.md) |

## Candidate and fixed scope

Execute the original stored draft weights through a private full-vocabulary
F32 head. Keep activation, accumulation, recurrence-independent head K/V and
readout precision F32. The pinned source uses Q8_1 activation quantization;
this floating-activation candidate requires its own qualification and does not
claim bitwise source execution.

Preserve global RMS over the four-stream input residual, per-stream hidden
projection, normalized embedding projection and broadcast addition. The three
hyperconnections use per-stream RMS, source low-rank activation, read gates and
write injection. Stored F32 norm values are final gamma. Use the source head's
own dense causal K/V, half-split RoPE at the cell position, ordinary full-width
four-plane Q2 experts plus shared expert, its final mixer and the main vocabulary
matrix. No latent D/U or group scales belong to this head.

Cell `i` pairs target residual `i` with input token `i+1`. Catch-up appends K/V
without executing the head FFN/readout. Chained draft input uses the preceding
head residual. Limit the diagnostic to 512 cells; full sparse-index behavior is
outside this attempt. Preserve default target commands and worker capabilities.

## Registered numerical gates

Generate the independent F64 oracle before candidate evaluation. Bind helper,
source, geometry, raw payload and schema hashes before the oracle runs. Preserve
failed/incomplete records and never overwrite an existing private fixture.

- Synthetic geometry: hidden/intermediate 64, streams 4, low rank 8, heads 4,
  KV heads 2, head dimension 16, rotary 8, experts 4, active 2, vocabulary 128,
  context 16 and four cells. Retain all 29 dense names and exact physical planes.
- Include unchanged raw Q8_0 embedding/readout bytes separately from the head's
  dense records, plus synthetic F32 residuals and private input IDs.
- Compare all observed stages with absolute difference at most
  `3e-5 + 3e-5 * max(abs(reference stage))`. Require exact routed IDs and finite
  inputs, decoded selected values, intermediates and outputs.
- Complete readouts additionally require maximum absolute difference at most
  `0.002` and relative L2 at most `0.0002`. Independent isolated Q2 dot controls
  use `2e-5 * max(sum(abs(products)), 1)`; original-plane fused GU is compared
  with its F64 activation, with the same final absolute/L2 bounds.
- Cover four ordinary cells, two front-only catch-up cells followed by the
  matching full suffix, and a chained fourth cell using the third head output.
  The oracle registers 132 stage records including independent Q2 product bounds.
  Fused gate/up preactivations are not observable head stages; do not fabricate
  their comparison from expected values or infer all-stage execution from them.
- Before saving a fixture, require hand controls and mutations that distinguish
  global/stream input normalization, adding gamma bias twice, wrong gate/up row
  layout, wrong scale plane and wrong RoPE cell position.

## Ownership, memory and lifecycle gates

Raw buffers retain original mapped owners. Checked dense descriptors are not
inserted as aliases into the target tensor map. The 824 MB draft files consume
the existing 8 GiB static-view allowance; head state/scratch requires a separate
allowance of at least 512 MiB. Bound target banks, static reserve, loaded target
dynamic allowance and head reserve by the existing complete cap. This is mapped
diagnostic admission, not physical-residency proof. Explicit residency remains
rejected for this new head until separately qualified.

Copy target residuals before final mixing into dedicated owners only on the
private diagnostic path. Bind residuals to loaded-model identity, reset epoch
and absolute row. Head K/V checkpointing must restore independent mutable
owners, validate identity/policy/cell chronology and invalidate on failure.
Complete target-plus-head controller integration and its pending residual are
still required before MTP capability promotion.

Current source admission is one-shot per loaded target. A fresh bound head after
target reset/restore requires model reload; reusable head leasing and combined
rebinding are unfinished integration work. Synthetic unbound head reset/replay
does not qualify loaded-model request recovery.

Tracing materializes extra GPU calculations for ungated attention and weighted
shared output; these are not separately retained normal-path intermediates.
Require ten traced/untraced readout, wide-vector and routing pairs to match
exactly across the registered serial/catch-up/chain cases before actual use.

Run native operator/component and regression checks only after source freeze.
Serialize GPU/model work. Actual-artifact head execution requires a separate
preregistered case, 64 GiB available before process start, owned-process RSS
watchdog/deadline, source/executable hashes and private pipes/RAM. Synthetic
component agreement does not establish actual head acceptance, sampling,
throughput, prefix reuse or full model equivalence.

## Fixture registration and pre-evaluation history

The independent helper's 14 CPU controls pass. The first generated fixture
was sealed before native evaluation, then repository Ruff found structural
Python style issues. Preserve that helper/test revision and `fixture-v1`
privately. The corrected helper uses literal dictionaries, sorted imports and
adjacent-pair iteration without changing equations, cases, seed or gates.
Ruff and all 14 controls pass after cleanup.
Oracle generation uses the private pinned source-text cache and intake Git-tree
record; the native consumer uses the sealed fixture. These are qualification
inputs and are not inference-worker dependencies.

The replacement `fixture-v2` registers the corrected helper before evaluation.
Its dense binary, dense manifest, expert bank, inputs and expected binary are
each byte-identical to v1. No numerical gate was adjusted and no candidate GPU
result existed when the cleanup occurred.

| Binding | SHA-256 |
| --- | --- |
| Archived v1 helper | `29c1736c47322e54f9f25e40c1deadbf2ab9b988a35f18a55e42b86a4a251913` |
| Archived v1 helper tests | `199360749425117d63e49a3243e60829968846d6ca31413acee956c6bd11396c` |
| v1 fixture manifest | `387118248b2e65e796a560523b769c7753924c3c8c08aed482789f93b8746d27` |
| Corrected helper | `76e418f3488d2f150763ddd2415b6f12bc6f0866be18a5fe53f8cb67e9d8f689` |
| Corrected helper tests | `2282a19259c345a7a1eba1db43861c36a67f527c1b6b839d1b285ebb03cd9e10` |
| v2 fixture registration | `414d99374d6b2aa2590176ec88100981fe51ee688bbba9630da1e95e44f38f07` |
| v2 fixture manifest | `d9079cfb1b23492279a39393a6ed9085948084840272b3e9bd332d583f24839f` |
| Identical expected binary, v1 and v2 | `5196ec1baf13d1474d57ec71861c6766c1f7fad432a87b4e95e32f90cda50513` |

Before native test execution, a production-library check compiled the head
implementation but Clippy rejected a constant-size byte chunk iterator in the
new gamma reader. The reader now uses typed four-byte chunks. Preserve the
initial log; the correction changes no byte interpretation or arithmetic.
The CPU/grammar feature check passed.

Review also identified a new residual-lineage concern: owner, reset epoch and
absolute row alone cannot distinguish an old speculative residual after a
successful target restore and a different continuation at the same row.
Residual tags must additionally bind a target restoration branch generation,
without making reusable target checkpoints single-use. A previously bound head
must reject after target restoration until an explicit combined target/head
restore can safely rebind it. That composite integration remains unfinished.

## Registered actual-artifact component case

After the synthetic head and lifecycle gates pass, run one separately bound
ignored actual-artifact diagnostic with virtual Metal registration. Keep MTP
sampling and worker admission disabled. Use one loaded model and an ordinary
36-token input kept in RAM.

First ingest 32 tokens and collect four ordinary target readouts. Reset the
target, create the bounded diagnostic head and advance all 36 target tokens
through the private wide-output path. Require the last four target readouts
bit-exact to the ordinary path. Catch up head cells 0–30 and retain the dedicated
target residual for cell 31. Run that head cell and one chained cell, then
restore the head checkpoint and replay both. Require finite complete vocabulary
rows and bit-exact head readouts and dedicated wide vectors.

Finally restore a valid target checkpoint, thereby changing its residual branch
generation. Restore the still-valid head checkpoint and submit the abandoned
target residual at a chronologically valid head row. Require tag rejection,
head fencing and refusal of the old head checkpoint. This deliberately does
not implement combined target/head rebinding or request recovery.

Fixed work is 72 target advances, 40 target readout rows, 31 front-only head
cells and four full head rows (35 successful head cells). Compare four target,
two head readout and two head-wide vector pairs exactly. Require the unchanged
prepared receipt, 64 GiB available before the owned process tree, a 48 GiB RSS
watchdog, a 600-second deadline, one named test passing and immutable source/
executable/recipe bindings. Timing includes test/build overhead and is not a
performance measurement. This case establishes actual component integration
and deterministic replay only, excluding exact MTP sampling, acceptance rate,
external full-model/source numerical parity and production capability.

## Results and disposition

### Native and independent numerical controls

At `151bb30c`, all 12 selected MTP operator/storage/admission tests passed,
including four original-plane Q2 tests. The complete head matched 402 observed
F64 stage rows and all ten routed rows across serial, catch-up and chained
cases. Maximum stage-bound ratio was `0.005921221`; maximum independent Q2 down
dot-bound ratio was `0.000041197`. Complete readout maximum absolute error was
`6.276948e-8`, maximum relative L2 `1.447396e-7`, both below the unchanged gates.
Fused GU scalar preactivations remain unobserved; their fused operator controls
are separate evidence, not 402-stage claims.

Synthetic head checkpoints passed seven bit-exact replay pairs plus cross-head,
reset-stale, malformed-position and nonfinite-state refusal. At `6f8745a8`, ten
traced/untraced readout pairs, ten wide-vector pairs and ten routing pairs were
exact; both front-only pairs retained no readout. Thus the extra diagnostic
trace passes did not change the tested untraced outputs. The new target-wide
restoration-branch counterexample and overflow controls passed too.

The full native CPU/Metal suite at `151bb30c` passed: 1,087 passed, 159 ignored,
219.84 seconds, plus seven integration, two profile and three documentation
tests. The later source adds two ignored qualification cases and documentation;
the full run remains bound to its original revision. Production Rust 1.98.1
Metal Clippy passed with `-D warnings`, CPU/grammar checking passed, and pinned
formatting and repository policy passed. The 14 Python oracle controls and Ruff
passed. No full-model external parity or generation quality is inferred.

### Actual artifact integration

The one-model component case passed at `6f8745a8`: 72 target advances and 40
target readout rows, 31 front-only head cells and four complete head rows.
All four target readout pairs, two head readout pairs and two dedicated-wide
pairs were bit-exact. A target restore invalidated the abandoned residual at
an otherwise valid head cell; the bound head fenced itself and rejected its
old checkpoint. Sampling remained absent.

Actual-head-1 passed these functional assertions with source pre/post bindings,
but its helper did not seal the executable before launch. Preserve that result
as source-bound evidence with a post-run executable seal. It does not satisfy
the preregistered executable-binding requirement on its own.

Actual-head-2 repeated the unchanged case by directly invoking the frozen native
test binary. The replacement helper checks its SHA-256 before registration and
launch, checks it again afterward, and rechecks all source hashes. It requires
the exact named pass and all aggregate fields/types/counts. This strict repeat
passed with exit code 0 and closes the component gate.

The repeat began with 89,822,871,552 available bytes and peaked at 711,606,272
sampled owned-process RSS bytes under the 48 GiB watchdog. Wall time was 19.30
seconds including test startup/model load, below 600 seconds. These are process
watchdog aggregates, not complete model/device memory or throughput. Immediately
after reap, available memory was 60,720,906,240 bytes; normal process exit is not
proof of synchronous GPU-page reclamation. Device allocation/residency claims
retain the separate Phase 4 scope. The host remains the same 128 GiB Apple M5 Max.
A later system sample reported 87,894,769,664 available bytes, above the
next-process admission gate; this is not attribution of every reclaimed page.

| Evidence binding | SHA-256 |
| --- | --- |
| Component source manifest at `151bb30c` | `7f60987ea4e0e36ca34fe94ab3ac2c13101a1041fc6e006b0c19b2b9ca510497` |
| Component native binary at `151bb30c` | `854053a2d1f9e9f3fa3d92672ce2a34d6a3ad1611c30b39808b4fc30bd4b9ccb` |
| Component registration | `795694c2add85eb68e76fed743ca06206ce71cd03849705a4bb0511aadd1fee0` |
| F64 head result log | `d18798401c976e2fe86bb23b388b20ec0986533ddfc91727d88cd4d093681523` |
| Full native result log | `ae9cdc1eec1448650c869e87fff7cac1249d99c9fe963798d6ce1fcc4dbd5243` |
| Head checkpoint result log | `6a7efabb126d14e527964e4cce95aa0097d7258d71f4ebf6fcce132e3eb3e858` |
| Tracing equivalence result log | `dfc4b4e789c254f4c054cf3dc070ce3e3aaf1d67a8e2d332e41fbe53edfddcd5` |
| Actual/test source manifest at `6f8745a8` | `0104b5351ee115701c4365a06e5b0e9cdddbaa3e1b74dc410cbb4a78ca4e6db5` |
| Actual frozen native binary | `62ccfaf1dec139b329eb56572f5977dcce43588714651fb4a88b8c27c51fc215` |
| Actual-head-1 registration, incomplete executable binding | `2b2100875e811bfca160196657db0062f58175a9546f61bdfc02eee6d6373c4c` |
| Actual-head-1 result | `4ba96bd5fcb70a393de437a4e74b28656251e2360dc9f8828c2b7d69d3dacc54` |
| Actual-head-2 registration | `c4d149f3f27103129b7cda80ea4293da1a3372d7389849cdb824341364af6b06` |
| Actual-head-2 result | `7ceef6005abb2205e16d8c6dbd1711f0c8896ccd5c9a6c56730df3f9b880dcef` |
| Actual-head-2 log | `cd0eaab638374feeb1b59ab19f20e0648739127a5289b5ce24e0681c1e7ae6fa` |
| Final component evidence receipt | `5f7adc631cde84bf579660724eab05298523f34e37de474df86022551ec2893a` |

### Scoped conformance and remaining work

| Obligation | Evidence | Disposition |
| --- | --- | --- |
| Original storage and source normalization | Checked descriptors/planes, pinned source trace, independent hand/mutation controls and F64 head stages | Conforms for the F32 candidate |
| Complete bounded head and catch-up/chain semantics | 402 synthetic stage rows, exact routes and real 31-cell catch-up/two-cell replay | Conforms within the registered cases |
| Finite, ownership and loaded-model branch safety | Q2 poison controls, scoped keys, dedicated residual owner and actual abandoned-branch refusal | Conforms |
| Head K/V restoration | Synthetic zero/positive/failure controls and actual same-head replay | Conforms for component checkpoints |
| Prospective diagnostic memory and private evidence | Complete mapped allowance, virtual-only admission and strict actual-head-2 resource/binding gates | Conforms within stated memory metrics |
| Default target preservation | Optional wide copy only, actual four ordinary/wide target pairs exact, full native regression | Conforms within tested scope |
| Combined controller, resource leasing/rebinding and exact sampling | Not connected; one-shot head admission requires reload after target binding changes | Pending before MTP promotion |
| MTP on/off default/performance and prefix reuse | No sampling or benchmark run; 512-token cap, worker prefix integration/4,096-token gate unfinished | Pending later gates |

Decision: adopt the private original-byte head component for continued local
evaluation. Attempt 08's bounded component gate passes. Phase 5 remains
incomplete: combined target/head/pending-wide state, resource leasing, request
recovery/cancellation, real one-to-three-draft exact sampling and matched on/off
measurement are still required. Worker MTP and prefix reuse remain unavailable;
app/catalogue and distribution admission are unchanged.
