# 20261113-EDGE0-08: native MTP head component

## Identity and state

| Field | Value |
| --- | --- |
| Registered / owner | 2026-11-13 / Codex, local non-commercial evaluation only |
| Baseline | `e91a3fdf`, qualified Phase 4 and native MTP foundation |
| State / result | Component implementation and independent oracle in progress; MTP unavailable |
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
