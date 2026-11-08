# 20261108-MEM-01: Qwen3.8 Beginner memory admission

## Identity and acceptance

Memory attempt, 2026-11-08, Codex. Bounded qualification passed on
`codex/beginner-hardware-ranking`; installation pending. Related attempts:
[context admission](2026-11-08-context-resource-budget.md) and
[experiment ledger](../../model-experiments/README.md).

The user reported a memory warning on a 128 GiB M5 Max with no model running.
Beginner admitted the model using physical/GPU fit, displayed a 111 GiB system
estimate, then startup demanded that estimate from currently available memory.
The estimate treated the complete 29.8 GiB paged n-gram payload as resident.
A second, independent native request gate reserved four times an inventory
that already includes overlapping state/scratch ownership.

Candidate acceptance, set before the candidate runs: exact installed text
artifacts must load and finish 1K, 8K, 16K and near-32K requests with default
two-draft MTP, plus near-32K MTP-off requests. Repeats must restore the existing
prefix cache and match fresh-scope greedy output hashes within each mode.
No safety guard may trigger: available memory >=6 GiB, swap growth <=1 GiB,
worker physical footprint <=96 GiB. Recommendations and fresh startup must use
the same current-memory predicate and revalidate artifact identity. This is a
bounded memory/lifecycle observation, not task-quality or numerical-reference
qualification.

## Workload and policy

One native worker at a time; no concurrent model or compilation workload.
Synthetic code-like input, rendered with the installed template; tokenized
prefix plus final 64-token suffix. Context 32,768; output budget 64; greedy,
seed 42; requests and outputs retained only in memory. Per-mode near-full
requests include immediate repeat and fresh scope. Private reports under
ignored `runtime/guided-memory-20261108/` retain only aggregates and hashes.

The shared `guided_memory_profiles.json` binds exact config, source receipt
and index SHA-256 plus the complete payload filename/size inventory. Payload
bytes are not freshly rehashed. It applies only to the pinned installed Lily
Q4-XL and MLX-Serve mixed text artifacts, <=32K context. Unknown or changed
artifacts and images retain conservative accounting.

Candidate resident plan: nonpaged payload + the native 32K request
inventory + 1 GiB paged working set + 8 GiB host reserve, rounded up to GiB.
The native request gate uses the same factor and reserves after loading.
The paged allowance covers up to 4,096 rows * 16 heads * 16 KiB pages per
prefill chunk; earlier clean mapped pages remain reclaimable. Lily plans 87 GiB
system and Serve 88 GiB; both plan 82 GiB GPU. The original full-payload fallback remains
111 GiB system. Physical reserve and 85% Metal-budget checks remain active.

## Attempt history

| Trial | Source and observation | Disposition |
| --- | --- | --- |
| Serve v1 | Rust 1.98.1 worker `c0179e2d20e8d5921287dc349a9495c18ecfa409501350fb58162acf886cb6ce`; loaded in 11.78s, completed 1,024+64 tokens, then 8K request rejected. Peak sampled footprint 76,307,345,368 bytes; minimum available 29,415,030,784; no swap growth or safety trigger. | Failed long-request acceptance. Initial client translated the terminal error, so v2 records its code. |
| Serve v2 | Same worker; load and 1K request succeeded; raw terminal code `resource_budget_exceeded` at 8K. | Reproduced native admission failure; no memory override used. |
| Serve v3 | Initial candidate charged twice the native inventory plus 9 GiB reserves (95 GiB planned). Host available memory was only 75.46 GiB before loading because a separate Docker workload was active; diagnostic measurement still ran under the 6 GiB guard. Loaded and completed 1K; 8K rejected safely. | Failed acceptance. Superseded: source review confirmed that the inventory itself already charges overlapping old/new owners. No Docker work was stopped. |
| Candidate policy | Charge that complete inventory once, retaining 9 GiB reserves (87/88 GiB plans). Also validate the external Serve n-gram payload and limit the qualified request policy to MTP depth <=2. | Adopted on branch after the final runs below. |
| Candidate compile | An MTP-bound helper accepted `usize` while request options use `u32`; Rust rejected the build. | Corrected helper to the request protocol's `u32`; no model executed with this failed build. |
| Serve v4 | 88 GiB plan; baseline available was 85.82 GiB, below that plan. Completed 1K and 8K, then 16K was rejected with about 16 GiB available. Peak sampled physical footprint 72.88 GiB; no guard trigger or swap growth. | Diagnostic run under competing Docker pressure; not a successful qualification. Subsequent harness requires startup headroom for five seconds before loading. |
| Model-free checks | One engine test incorrectly expected a same-size payload edit to revoke a size-bound memory layout; one ranking fixture omitted the now-required resident binding. | Corrected tests to match the documented contract. Payload byte identity is explicitly outside this layout check; altered metadata and payload sizes must fail. |
| Serve v5 | All nine requests passed, including near-32K MTP on/off, immediate reuse and fresh-scope hash agreement. The harness's telemetry callback had the wrong argument signature, so GPU samples were silently dropped by the production client. System/process samples are valid; complete measurement acceptance failed. | Preserved as functional/process-memory evidence; repeat with corrected callback. Added a model-free check through the production telemetry reader. |

Measurements sample macOS `proc_pid_rusage` RSS/physical footprint at 10 Hz,
system available memory and swap at 10 Hz, and private worker Metal telemetry
at approximately 1 Hz. Sampled peaks can miss brief spikes; lifetime maximum
is process-wide rather than phase-local. Existing swap is distinguished from
growth. Source, harness, worker and artifact-stat hashes are sealed in each
receipt. Final environment, aggregate results, checks and installation evidence
will be appended without rewriting failed trials.

## Final artifact-bound observations

[Public aggregates and hashes](data/2026-11-08-qwen-guided-memory.json) preserve
all seven trials, including failures. The private receipt hashes identify the
original ignored reports. No prompts, completions, token vectors or process
identities are published. Earlier harness versions were not committed as
standalone source snapshots; their hashes and limitations remain in the receipts.

| Field | Value |
| --- | --- |
| Mac / OS | Apple M5 Max, 128 GiB physical RAM, macOS 26.7.2 |
| Toolchains | Rust 1.98.1 (`48a229cea`), uv 0.12.24, Python 3.13.13; Xcode 26.4.1 build 17E202; driver version not separately recorded |
| Backend provenance | Stark native Lily integration; upstream `e22d58fa7cc488493fb1b5d810a198bd29730aeb` plus checked-in adaptations; no inference math or precision changed here |
| Candidate worker before app signing | `01d9bb0923cbca16b24011d679d70033d361dae22fd6cc9facca46c7caf900af` |
| Final harness SHA-256 | `15ae2dde8ef8975d31117e0b45cbc55ae140fe326f9d46f23fb421c0f01ec378` |
| Cargo lock SHA-256 | `85c73ea6bdb1ece383aca9502820095be4dba8818d6d7e348caca938507e18d6` |
| Converter lock SHA-256 | `df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5` |
| Source identity | Reports record merge `b009e0f1` plus dirty source-file hashes; the final policy/helper/worker/harness files match those sealed hashes. Tests and documentation were finalized afterward. |
| Artifact identities | Exact repositories/revisions/config/index/receipt hashes and payload manifests in `guided_memory_profiles.json`; tokenizer/template metadata hashes in the aggregates. Payloads were not freshly rehashed. |
| Precision | Pinned mixed 4/8-bit stored weights, BF16 activations, F32 GDN recurrence, Lily Q8 KV / external MLX-Serve Q8Affine64 KV; existing accumulation policy unchanged |

| Measurement | Lily `lily-v1` | Serve `serve-v6` |
| --- | ---: | ---: |
| Startup plan | 87 GiB | 88 GiB |
| Initially available | 95.57 GiB | 97.69 GiB |
| Sampled peak physical footprint | 72.90 GiB | 73.28 GiB |
| OS lifetime maximum physical footprint | 73.31 GiB | 73.70 GiB |
| Sampled peak RSS | 73.47 GiB | 73.88 GiB |
| Sampled peak Metal allocation | 72.50 GiB | 72.90 GiB |
| Minimum system available | 20.71 GiB | 22.20 GiB |
| Additional swap used | 0 bytes | 0 bytes |
| Requests completed | 9 / 9 | 9 / 9 |
| Guard triggered | No | No |

Both final trials completed 1,024, 8,192, 16,384 and 32,672 input tokens with
two-draft MTP and a 64-token output budget. Each mode's near-full request,
immediate repeat and fresh-scope replay returned identical greedy output
hashes. The repeated near-full request restored 28,672 input tokens; the fresh
scope restored zero. MTP-off ran the same three near-full requests. All nine
requests sampled 64 tokens. Artifacts and worker bytes were unchanged after
each trial. Existing swap was present before the runs; no growth was observed.
Docker remained running, with no unrelated work terminated.

These are single-run functional/memory observations for each exact text
artifact. They do not establish cold-load speed, a throughput improvement,
coding quality, quantization equivalence, reference-logit agreement, images,
deeper MTP, larger contexts or all possible host-pressure conditions. Native
admission remains active and can reject later requests when headroom changes.
No speculative credit for already-owned scratch was added. Cancellation/reset
behavior was covered by model-free tests, not a new actual-artifact cancellation
trial here; worker close/reload and distinct cache scopes were exercised.

## Validation and installation

The generator suite passed 134 tests with two existing ignored tests. The four
new engine profile/scope tests passed using the nightly CPU/grammar harness.
The first converter run passed 6,763 tests, skipped 96 and found the stale
ranking fixture recorded above. The corrected ranking/profile files passed
32 tests; guided/start/profile tests passed 95; production telemetry callback
integration passed one. Final complete converter replay and signed installation
results will be appended below.
