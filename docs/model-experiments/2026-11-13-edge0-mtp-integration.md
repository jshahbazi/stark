# 20261113-EDGE0-09: combined restoration and exact sampling

## Identity and scope

| Field | Value |
| --- | --- |
| Registered / owner | 2026-11-13 / Codex, local non-commercial evaluation only |
| Baseline | `90746d474e3ab42553ebc23077f7abb090d421db`, qualified private head component |
| State / result | Implementation in progress; untested |
| Governing plan | [Phase 5](../EDGE0_PLAN.md#phase-5-mtp), owner-approved native Qwen Metal graph |
| Predecessor | [20261113-EDGE0-08](2026-11-13-edge0-mtp-head.md) |
| Promotion | Private evaluation only; worker MTP, prefix reuse and product admission remain unavailable |

The owner requested combined state restoration and exact sampling on 2026-11-13.
This attempt connects the existing original-byte F32 head and ordinary single-token
target to Stark's backend-independent rejection sampler. It also replaces one-shot
head admission with an exclusive, bounded lifetime lease. No default or performance
decision is part of this attempt.

## Fixed correctness and lifecycle gates

- A combined checkpoint owns the complete target snapshot, head K/V and the pending
  target-wide residual. Successful restoration rebinds only the saved residual to
  the restored target branch; residuals from abandoned branches remain invalid.
  Failure fences the entire session and invalidates its checkpoint handles.
- Resource admission stays virtual-only, with complete target/static/head/checkpoint
  allowance before allocation. Only one head lease may be active per loaded model.
  Releasing it must drain owned work before a fresh session can be admitted.
- Verification executes ordinary single-token target steps. No batched-prefill
  numerical tolerance substitutes for exact target-distribution verification.
- Reuse `MtpSampler` and the generator's `ExactMtpSampler`, with one, two and three
  drafts and the complete vocabulary. Four drafts and restricted vocabulary remain
  excluded. Same-seed MTP-on/off identity is not an exact stochastic-sampling gate.
- Deterministic fake backends must cover every acceptance length for depths one to
  three, short output tails, rejection/correction replay, early stream drop,
  explicit finish, cancellation and injected backend/sampler/rollback failures.
  Delivered boundaries must match ordinary causal consumption bit for bit; cleanup
  must not draw randomness. Invalid readouts or IDs must fail closed.
- Actual-artifact controls must exercise combined target/head/pending-wide replay,
  stale-branch and stale-session refusal, lease release/recreation, sampled depth
  one to three, bounded output tails and post-failure/reset recovery. Exact replay
  means bit equality of tested complete readouts/wide vectors, not top-1 agreement.
  A deterministic recorded sampler additionally compares each verification readout
  with the ordinary target continuation along its delivered token sequence.
- Existing head numerical gates and ordinary target behavior must retain their
  qualifications. Run focused controls first, production Clippy and appropriate
  native regression tests after integration. Record compile/setup failures too.

## Artifact, arithmetic and resource boundary

The artifact remains `Edge0/Edge0-145B-q3k` at
`d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`, prepared under
`runtime/edge0-native-v2-evaluation/`. Its `native-source.json` SHA-256 is
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`.
Original weights, four-plane Q2 head experts, F32 activations/accumulation/state/KV,
512-token cap and source pins retain attempt 08's boundary. The preparation receipt
contains the complete source-payload hash validation; this run does not imply a new
103 GB rehash. Source payload immutability remains an explicit assumption.

Native build sources and the executable must be SHA-256 sealed before launch and
checked afterward. Exact named test passes and aggregate counts are required;
exit code alone is insufficient. Private source/fixture inputs, IDs, readouts and
sampler decisions remain in ignored local runtime storage or RAM. Public evidence
contains aggregate results and hashes only.

Actual model processes run one at a time on the same 128 GiB Apple M5 Max host.
Require at least 64 GiB available memory before launch, a 48 GiB sampled owned-RSS
watchdog and a 600-second deadline per registered case. Owned RSS is not complete
GPU allocation/residency evidence. No OS wired-memory setting is changed. Stop on
resource failure, nonfinite state, any replay mismatch or executable/source drift.
Register each concrete actual-case workload and helper hash before its launch.

Performance and quality are not measured in this attempt. Worker request dispatch,
prefix scope/lifetime integration, the wider-context finite concern and the
4,096-token reuse gate remain later work.

## Attempt history, results and disposition

Before execution, independent review identified two controls that needed changes:
the ordinary prompt reference now uses serial steps throughout, and restore charges
the complete padded target replacement plus temporary host storage rather than
the smaller saved-prefix byte count. A two-checkpoint actual case would exceed
the conservative 512 MiB head allowance when capturing/restoring. The concrete
case below uses one reusable combined snapshot, matching the controller's lifetime.
Multiple live full snapshots remain bounded by admission; they are not promised.
The three 64 MiB graph-segment reserve remains unchanged.

### Registered actual cases

Both cases use one loaded model and fixed ordinary text tokenized privately.
These are functional controls; elapsed time is a watchdog metric only.

1. `edge0_mtp_actual_combined_restore_rebind_and_resource_lease`: serial 31-token
   prompt, two continuation cells and replay from one reusable combined checkpoint.
   Require 36 total target advances, two exact complete target pairs, two exact
   complete head/wide pairs and one exact pending-wide pair. Repeated restore must
   stale the previous branch. Malformed saved state must fence both owners, and
   the otherwise repaired old checkpoint must still be refused after reset/rebind.
   An abandoned residual must fail execution. Concurrent heads, checkpoint-held
   admission and restored-wide-held admission must all refuse a second head.
   Dropping those owners must allow head recreation and reset/rebind without
   reloading the model. Sampling is absent in this case.
2. `edge0_actual_exact_sampling_and_restoration`: 12-token serial prompt;
   `ExactMtpSampler::new` using greedy and temperature 0.7 / top-p 0.9 / no top-k
   or min-p, each at draft depths 1, 2 and 3, seed `0x4544474530`, five outputs
   each. Additional depth-three greedy tails have budgets 0 through 3 (seed 17).
   A depth-three clean callback stop after two of five outputs (seed 23) must
   preserve the next ordinary target readout. Callback cancellation after two
   outputs and injected first-draft sampler failure (seed 29) must propagate;
   each is followed by one exact recovery output (seed 31) on the same model.
   Require 15 cases, 41 bit-exact complete serial target pairs, all three depths,
   six full sampled cases, four tails, one early stop, two failures and two
   recoveries. Proposal/acceptance totals are observed aggregates, with positive
   proposals and accepted no greater than proposed. Capabilities stay unavailable.

Candidate revisions, results and immutable receipt hashes will be appended after
the presealed executables run under the registered watchdog.
