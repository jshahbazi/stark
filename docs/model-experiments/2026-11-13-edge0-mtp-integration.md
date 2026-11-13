# 20261113-EDGE0-09: combined restoration and exact sampling

## Identity and scope

| Field | Value |
| --- | --- |
| Registered / owner | 2026-11-13 / Codex, local non-commercial evaluation only |
| Baseline | `90746d474e3ab42553ebc23077f7abb090d421db`, qualified private head component |
| State / result | Bounded actual combined-restoration and exact-sampling gates passed |
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

The build identities and run receipts below bind these results to code revision
`18089b37` and the presealed executables.

### Build and registration bindings

Candidate `18089b37` uses Rust 1.98.1 for the generator and production Metal
Clippy, and `nightly-2026-10-01` for the CPU/Metal engine unit binary. Dependencies
remain locked. A source inventory of 1,234 files includes Rust/Metal code, Cargo
inputs, vendor sources and external compile-time fixtures. The first build seal
omitted external compile-time inputs; review corrected the inventory before any
actual-artifact launch, and sealed Cargo checks revalidated both binaries. The
original build record is retained as incomplete evidence rather than overwritten.

The default-feature Clippy invocation reported 26 diagnostics in unchanged CPU
reference files and one new slice-clone lint. The new lint was corrected; the
production configuration, `--no-default-features --features metal`, passed with
`-D warnings`. CPU/grammar checking, formatting and repository policy passed.
The controller's 12 tests and the generator's 15 exact-sampler tests passed.
The final native unit regression and both actual cases passed as detailed below.

| Prelaunch evidence | SHA-256 |
| --- | --- |
| Source manifest | `bacab74c5496de0d97e53333177388d925fe9b218ae18362346ff3cff24cb825` |
| Corrected build seal | `ce111b254f951892b8b18d9f1d933e683d83266a7efb90c697489a7ebe1447ef` |
| Engine unit executable | `27bd04975492c26255fa0f654ccf99f29f8f68953f72496f1476997412e4b37d` |
| Generator unit executable | `0256c76ed8d2a709dcd7a05a59c193e24b74c92f89fdb0f0fb4aef16f8627f61` |
| Watchdog helper | `c1b05aad215c44701541b62a109a02da57655c7aefc456576ce992857de5fe90` |
| Combined-state registration | `15f789e04229d50b6ca7e264e42c69e41b53cfb921efe0980d41ed6d508d393b` |
| Exact-sampling registration | `542d02f1375c5e0684690fc9d024976e2bf739f5f35b7e71c14a257a0de0f732` |

The private callback interface observes cancellation at delivered outputs. It
does not qualify asynchronous worker cancellation during prompt setup or a
submitted target step. Checkpoint and restored pending-wide owners retain their
lease; arbitrary diagnostic `DeviceVector` clones are outside the adapter's
bounded lifetime claim.

## Results and scoped conformance

The sealed native CPU/Metal unit binary at `18089b37` passed 1,094 tests with
162 ignored in 252.59 seconds. This is the complete unit binary, without a new
integration-test or documentation-test run. All four private head controls passed:
402 F64 stage rows and ten exact route rows under the unchanged numerical bounds,
seven checkpoint replay pairs, ten traced/untraced readout and wide pairs, ten
route pairs and two front-only pairs. Maximum final absolute error remained
`6.276948e-8`, relative L2 `1.447396e-7`. All 15 generator exact-sampler controls
and all 14 Python oracle controls passed. The initial direct Python script call
collected no tests; only the later pytest collection is qualification evidence.

Both actual cases passed their exact named one-test gates and unchanged aggregate
contracts. The helper checked sources, executable, configuration and helper seals
before and after each process; independent review verified the raw logs and all
1,234 source hashes against the corrected build registration.

| Actual case | Registered functional result | Resource watchdog observations |
| --- | --- | --- |
| Combined state | 36 target advances; two target and two head/wide pairs plus one pending-wide pair bit-exact; repeat restore, stale/malformed refusal, joint fencing, retained leases, recreation and reset/rebind all passed | 91,649,474,560 available bytes before launch; 710,606,848 sampled peak owned RSS bytes; 22.23 seconds; exit 0 |
| Exact sampling | All 15 cases and 41 full target pairs bit-exact; three draft depths, tails, clean stop, both injected failures and both same-model recoveries passed; six full sampled cases proposed 24 drafts and accepted nine | 92,600,549,376 available bytes before launch; 772,145,152 sampled peak owned RSS bytes; 69.38 seconds; exit 0 |

The host was verified as Mac17,7, Apple M5 Max, 137,438,953,472 RAM bytes,
macOS 26.7.2 build 25G332, with uv 0.12.24. Both cases met the 64 GiB launch,
48 GiB sampled owned-RSS and 600-second limits. The combined and sampling cases
reported 92,650,799,104 and 84,968,161,280 available bytes after reaping. These
observations do not measure total GPU allocation, residency or synchronous page
reclamation. The case wall times include model loading and controls; they are
not inference throughput measurements or acceptance-rate benchmarks.

| Final evidence | SHA-256 |
| --- | --- |
| Combined-state report | `c9b767762e6e125b7d29863204dbbe3865a9221f79a0f528fdc969282e848026` |
| Combined-state log | `3a1c86d3428afb4d3d6158067a7025e91b615679c16befd55d18b54faf9d2879` |
| Exact-sampling report | `47c2a939db5e46928d85a146009101bd74d94a842ae5727b701cd11e20a03c7b` |
| Exact-sampling log | `a59d34cd83dec6283147a4d42742b7f73d112b5ca5a0a885f8f9e71f0d0701c1` |
| Native unit regression log | `8b642bf3b5f8093bca1428f8a5d22b40f4e5255ec63a47a545ddbf50e779fffa` |
| Head-component log | `71b8d5fbdaac681201b6625db49ca7360dfd1df3aab2fa514bd5c444d8b38093` |
| Exact-sampler unit log | `6b30c3d2f7ea50a97ff87a4086990a6b6b59df62f55b269cb341ed7369280351` |
| Python oracle log | `e80cd43714047dff16570b481f3efce991c33bc3cb6b5fc11610f4cdc59124e2` |
| Production Metal Clippy log | `39b9ba4a692b0fcd2850fd2b936fd79b85665e1dc25a0d921d3e19be9966e142` |
| Final integration evidence receipt | `c6e78a1aae4795862e4eb96f91cfaafcf587c1d035e684b90f225409b03d1f7a` |

| Obligation | Evidence | Disposition |
| --- | --- | --- |
| Complete target/head/pending-wide restoration and ownership | Combined checkpoint inventory, actual bit-exact pairs and abandoned/recovered-session refusal | Conforms within the registered context and one-checkpoint lifetime |
| Reusable bounded head resources | Capacity-aware replacement accounting, lease counter/drain controls, actual held-owner refusal and recreation | Conforms for the private evaluator |
| Exact one-to-three-draft sampling and serial target distributions | Existing sampler math/filter controls, exhaustive fake backend matrix, 15 actual cases and 41 complete target pairs | Conforms within registered cases |
| Tails, cleanup, cancellation and failures | Every acceptance/finish/drop/cancel fake boundary, actual callback stop/error and sampler failure with same-model recovery | Conforms for synchronous private callbacks |
| Original head arithmetic and ordinary target behavior | Unchanged F64 head gates, bit-exact actual serial target rows and full native unit regression | Conforms within tested scope |
| Worker integration, asynchronous cancellation, performance/default selection | Not connected or measured | Pending before worker MTP promotion |
| Prefix reuse, wider context, quality and product admission | 512-token cap, no worker prefix scope, wider-context finite concern and no quality/admission run | Pending later gates |

Decision: adopt combined restoration and exact sampling on the branch for continued
local non-commercial evaluation. Independent review found no outstanding issue in
attempt 09's bounded scope. Phase 5 remains incomplete until worker integration and
matched on/off/default qualification pass. Worker MTP and prefix reuse stay
unavailable; no app/catalogue, distribution or installed-build promotion is made.

Follow-up: [attempt `20261113-EDGE0-11`](2026-11-13-edge0-mtp-batched-worker.md)
subsequently qualified the local virtual Metal worker path, aligned ordinary
prompt arithmetic and strengthened actual cancellation/recovery controls. Its
matched requests completed with no eligible timing group; default remains off.
The original attempt 09 results and source boundaries above remain unchanged.
