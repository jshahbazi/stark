# Pi-assisted conversion adapter builder

Status: proposed implementation; no builder, command or new qualification is delivered by this document.

Prepared: 2026-10-12. Stark baseline: `451b2a76`, the current `main` revision inspected for this plan. The inspected Meecp Pi implementation is pinned to `f818b73a00161b880fe544397108abb52b76db82`. Implementation must record its own exact source and dependency hashes.

## Objective and decision

Build an optional developer tool that uses Pi to create missing Stark conversion adapters from pinned model architecture sources. It must produce a reviewable source patch and host-generated verification evidence. Accepted adapters become ordinary deterministic converter code; later conversions require no agent or provider access.

Keep the builder in this repository under `tools/adapter-agent/`, with a separate Python supervisor and Node/Pi worker. The Rust/Metal engine and installed converter retain their current dependency boundaries. A later opt-in converter command may launch the builder through the same process protocol. Meecp can consume that protocol from its own repository without moving host application integration into Stark.

The principal uncertainty is whether Pi can generate correct architecture transformations with less developer effort. A bounded pilot must measure that before broad automatic generation. Successful compilation, agent-authored fixtures or native readiness alone cannot establish semantic correctness.

## Current foundation

The [converter](../tools/converter/README.md), [replacement plan](MODEL_CONVERSION_PLAN.md) and [typed contracts](../tools/converter/stark_convert/contracts.py) already provide:

- Explicit adapter dispatch, complete configuration/tensor validation, bounded streaming transformations, immutable source resolution and offline replay.
- Separate conversion implementation, source validation, native loading and execution status.
- Private receipts, initial/final content verification and atomic publication that refuses existing output directories.
- An explicit `hf-generic` route for uncatalogued checkpoints with implemented architectures and self-contained metadata.
- Independent tensor/config/tokenizer comparisons and exact named-build load-only qualifications.

At the inspected baseline, 85 of 110 pinned Lalamo built-ins and four separate Stark extensions are implemented. Coverage includes LLaMBA/Mamba2 and ModernBERT classifiers; those are existing work, not missing-adapter pilot claims. The catalog remains an inventory of checkpoint profiles, not a list of arbitrary future models. This plan does not close the replacement plan's open coverage, latency, runtime or execution gates.

Meecp supplies an implementation pattern: an explicitly configured in-memory Pi session, brokered tools, controlled subprocesses, fixed error categories, mandatory independent checks and bounded repair. Its app setup routes, native UI, MCP package schema and installation machinery are outside Stark's scope.

## Routing unknown inputs

Resolve the input and inspect its declared architecture, model kind, quantization and required operations before starting Pi. Do not use model-name similarity as dispatch evidence.

| Classification | Host action | Permitted outcome |
| --- | --- | --- |
| Existing architecture and accepted layout/metadata | Use the registered profile or explicit `hf-generic` route. | Deterministic conversion with existing qualification rules; no Pi request. |
| Existing architecture with incomplete or conflicting metadata | Return a precise metadata/access disposition. | A reviewed metadata proposal when authoritative source information exists; no invented template, parser, EOS or label defaults. |
| Missing architecture adapter or supported-format variant, expressible with existing engine operations and streaming primitives | Start a bounded adapter-generation job. | Candidate code, tests, provenance and independently measured conversion evidence. |
| Architecture needs an absent native operation, incompatible dimensions or unsupported backend path | Record `needs_engine_work`, including the operation and owning runtime contract. | A technical gap report; a separately scoped engine implementation is required. |
| Ambiguous/custom architecture or unsafe/unreadable source | Return `needs_specification`, `source_access_blocked` or `unsupported_source`. | Retain the bounded disposition without changing converter coverage. |

Preflight is advisory until full source and layout validation passes. A repository's `model_type` must agree with any architecture assertion. Existing catalog profiles remain authoritative; generic mode cannot override them. `modernbert` retains explicit classifier routing and its supported pooling/activation semantics.

An invalid source for a supported architecture must not trigger a repair agent that weakens its validator. Genuine unsupported variants receive a new, explicit contract. A new adapter can export conversion-only artifacts where justified, but must retain runtime-blocked/unverified status.

## Scope

The first implementation supports Python adapters producing Stark's existing `ConversionPlan` and `TensorOperation` contracts for dense language models with BF16/F16/F32 conversion. This includes the existing unquantized LFM2 short-convolution/attention layout used by the controlled pilot. It can reuse reviewed streaming operations and accepted metadata validators. BF16 is the current native loading path; F16/F32 export must retain its conversion-only status.

Subsequent packets can cover existing affine MLX 4/8-bit formats, broader hybrid/state-space adapters and classifiers. Each expands the contract and adversarial tests independently. Quantized inputs must preserve existing packed payload semantics; new quantization, unsupported compression and invented weight approximations are out of scope.

Initial exclusions are native Rust/Metal modifications, unrestricted runtime plugins, automatic catalog/native-ledger edits, model training, production model quality claims, arbitrary upstream `trust_remote_code` execution, automatic Git pushes/PRs/merges, hosted services and distribution/signing/release infrastructure. The tool's output is a candidate patch. Maintainer integration uses the repository's normal contribution workflow.

## Components and ownership

| Component | Responsibilities | Trust boundary |
| --- | --- | --- |
| Python supervisor | Preflight, job policy, source snapshots, lifecycle, budgets, artifacts and receipts. | Owns all state transitions and final dispositions. |
| Node/Pi worker | Selected provider/model session, coding loop and requests for named broker tools. | Untrusted planning output; no direct authority over the checkout or verification decisions. |
| Capability broker | Bounded source inspection, candidate writes, fixed local checks and controlled source acquisition. | Enforces filesystem, process, network and credential rules outside the model. |
| Candidate runner | Imports and tests generated Python code in a restricted fresh process. | Generated Python is executable code even if it uses approved imports. |
| Independent verifier | Host-owned fixtures, oracle comparison, source/toolchain hashes and probe outcomes. | Read-only to Pi; independent of candidate tests and claims. |
| Integration/export step | Produces a sealed candidate patch against an exact converter base. | Cannot modify the active checkout, catalog or qualification ledger automatically. |

The worker protocol follows Meecp's one-start-frame/typed-tool-call/typed-result/fixed-terminal-frame pattern. Credentials travel through private pipes, never command-line arguments. No transcript frames are persisted. The supervisor emits only fixed progress stages, public-safe aggregates and hashes.

Use Pi's SDK with an explicit credential store, model runtime, resource loader, settings manager and in-memory session manager. Disable built-in tools, global/project discovery, user extensions, skills, MCP discovery, credential-file reads, telemetry and catalog refresh. Implement and verify these controls against the pinned SDK; a setting name in documentation is insufficient evidence that it works.

### Proposed layout

These paths are planned and do not exist yet:

```text
tools/adapter-agent/
  README.md
  pyproject.toml
  uv.lock
  .python-version
  stark_adapter_agent/
    cli.py
    supervisor.py
    contracts.py
    preflight.py
    broker.py
    confinement.py
    verify.py
    export.py
    receipts.py
  worker/
    worker.mjs
    worker.test.mjs
    package.json
    package-lock.json
    provenance.json
    licenses/
  schemas/
  tests/
```

Keep schemas and immutable verification fixtures host-owned. Add package build/install resources only when needed for this local tool. The converter's installed wheel must remain usable without Node, Pi or the builder environment.

### Proposed CLI

These commands are interface targets, not runnable commands today:

```sh
stark-adapter-agent inspect my-org/new-model --revision <immutable-commit>
stark-adapter-agent build --job-spec /private/path/job.json
stark-adapter-agent verify --candidate /private/path/candidate
stark-adapter-agent export --candidate /private/path/candidate --output /private/path/patch
```

`inspect` never starts Pi. `build` requires an explicit provider/model configuration and a frozen job policy. `verify` reruns checks in fresh processes without a coding provider. `export` writes to a new directory and refuses replacement. Cancellation must stop the worker, outstanding provider work where supported, candidate processes and verification children.

The standalone CLI is the first entry point. After acceptance, an optional `stark-convert` flag/subcommand may invoke it. Default conversion remains deterministic; an unsupported conversion does not automatically send source material to a provider or start spending tokens.

## Data contracts and status

All contracts are versioned, reject duplicate JSON keys/unknown fields and bound strings, arrays, nesting and file sizes. Paths are private host references, not public report fields.

| Contract | Required contents |
| --- | --- |
| `JobSpec` | Source identities/revisions/hashes, model kind, architecture assertion, input/output formats, converter base/source/lock hashes, selected provider/model, source disclosure policy, budgets and requested verification level. Secret values are excluded. |
| `ArchitectureAssessment` | Required operations/config semantics, mapping to existing runtime/streaming contracts, references to pinned public source files, unsupported fields and routing disposition. Host validates every claimed mapping. |
| `CandidateManifest` | Adapter entry point, declaration of accepted architecture/config/layout/format, complete file hash/mode/size inventory, supported/unsupported variants, dependency policy and provenance. |
| `VerificationReport` | Candidate/base/source/oracle/build hashes, independent checks with pass/fail/not-run outcomes, aggregate mismatch counts, resource observations, sanitized failure codes and scoped claim. |
| `ExportReceipt` | Frozen candidate hash, exact base revision/hash, exported patch hash, verifier/report hash, review status and revalidation requirements. |

Generation states are `queued`, `preflight`, `generating`, `checking`, `repairing`, `verifying`, `candidate_ready` and fixed terminal dispositions such as `failed`, `cancelled`, `budget_exhausted`, `needs_engine_work` or `needs_specification`. Persist only content-free lifecycle records needed for recovery. Candidate code and private evidence are separate artifacts.

`candidate_ready` means the requested checks passed and the patch is available for review. It does not mean conversion coverage, native support or execution quality has been promoted. Keep these statuses outside `CapabilityResult`; its existing semantics remain unchanged. Any future contract extension needs its own compatibility review.

## Adapter and patch boundary

The candidate adapter must validate the complete semantic config and tensor inventory before writing. It must account for every source tensor, including explicit verified omissions; validate tied heads, aliases, shapes, dtype, fusion order, RoPE/attention/recurrent settings and normalization/scaling rules; and reject unsupported variants.

Use existing typed streaming/compressed primitives and their workspace accounting. No imports of Lalamo, JAX, PyTorch, executable checkpoint code or new production dependencies. Static import checks are useful diagnostics but cannot establish runtime confinement or semantic correctness.

The candidate workspace contains a pinned converter snapshot, reviewed helper sources and an adapter scaffold. Shared streaming code, source resolution, publication, qualification, dependency locks, verifier tests and schemas are read-only. Pi can write adapter code, candidate-owned tests, a declarative registration/metadata proposal and a rationale containing public source references. If a needed operation is absent, stop with a helper/engine gap report; do not edit protected code to make checks pass.

Host validation materializes the declared registration in an ephemeral converter copy, using bounded, host-generated edits. Integration points are currently `registry.py`, `adapters/__init__.py`, `cli.py`, architecture/config helpers and metadata/profile routing. A generated adapter must be registered consistently across architecture selection, model-kind routing, planner dispatch and generic capability reporting.

Catalog changes, checkpoint-specific parser/reasoning defaults and external metadata routing require separate authoritative evidence and review. Never add native qualification records from agent claims. Candidate execution cannot rewrite receipts, source manifests or the native ledger.

Freeze the candidate inventory before independent verification. Run from a relocated workspace without access to the authoring workspace or `.git` history, then rehash before export. Reject base/source/toolchain/candidate mutation. The exported patch contains source/tests/documentation and public provenance only; it contains no model files, runtime bundles, private receipts or evaluation material.

## Capability broker

Expose narrow operations whose policy is enforced by the host:

| Tool family | Allowed behavior |
| --- | --- |
| `inspect_checkpoint` | Host-parsed bounded config fields, tensor names/shapes/dtypes and declared metadata capabilities; no tensor payloads, token IDs/vectors or vocabulary dump. |
| `read_source` / `list_source` | Selected public architecture implementations and approved converter helpers from immutable snapshots. Paginated reads with explicit disclosure policy. |
| `read_candidate` / `write_candidate` | Regular files within the candidate subtree, with size/type/path limits and serialized writes. |
| `check_candidate` | Fixed syntax/lint/synthetic checks through the pinned runner. Candidate tests cannot replace host tests. |
| `verify_candidate` | Requests a host-owned parity/load check; returns fixed categories and aggregate counts without oracle contents. |
| `fetch_source` | Host fetch of approved public code/metadata with identity/hash binding, redirect/size limits and no remote execution. |
| `capabilities` | Exact available operations, supported confinement and current job budget. |

Initial jobs have no general shell, package manager or background-service tools. Fixed runner profiles use explicit argv, pinned executables and host-selected working directories. A future general command capability would require a separate boundary review. Fetches and model acquisition are broker operations; candidate processes have no network.

Worker/provider communication uses a host-owned authenticated provider relay. The worker can connect only to that job's fixed loopback relay address/port; direct provider/network access is denied by confinement. The relay forwards only the selected provider's declared API origins/paths, validates TLS and DNS/address results at connection time, rejects redirects and bounds requests/responses. Private/link-local destinations are denied unless the job explicitly selects an exact local-provider endpoint. Endpoint configuration alone is not enforcement: B1 must test rejected hosts, DNS/address changes, redirects, unrelated local services and direct worker egress.

Source fetching is a separate process and policy. Reject credential-bearing URLs and uncontrolled redirects. Checkpoint access tokens stay with the source resolver. Upstream provider keys stay with the relay; the worker receives only a random per-job relay credential through private stdin for its in-memory SDK store. Neither credential enters model context, tool arguments/results, artifacts or diagnostics. Qualify the relay against the selected provider's SDK transport, streaming and cancellation behavior before enabling that provider.

Public model code, config and tensor metadata may still be sent to the selected coding provider under the job's disclosure policy. Private checkpoints require an explicitly suitable disclosure configuration, including a local provider when appropriate. Do not assume that public weights make every local overlay or metadata file public. Tokenizer vocabularies, chat-template text, example conversations, native inputs/outputs and raw numerical results are not automatically forwarded.

## Confinement and privacy

The Node process, an isolated directory and tool descriptions do not create an OS sandbox. Generated Python may execute during import or tests. On the initial Apple-silicon macOS target, the host must apply and test an OS confinement profile to the worker and every candidate/check subprocess. Profile-application failures stop the operation. Unsupported hosts can inspect/export inert artifacts but cannot execute untrusted candidates until an equivalent boundary is implemented and qualified.

Confinement allows only the job's exact workspace/runtime/source paths and required read-only system libraries. Deny access to the active checkout, user home/configuration, credentials, unrelated models, Git credentials/sockets, private oracle files and other jobs. Candidate runners deny network and process escape. The Pi worker can reach only its host-owned relay and cannot launch children or read arbitrary files. Verify enforcement on the actual supported macOS version; Meecp's Seatbelt code is a reference, not proof that Stark's profiles work.

Reject path traversal, symlink/hardlink escapes, special files and unbounded archive extraction. Use descriptor-based operations and identity rechecks to prevent check/use races. Owned process groups, deadlines and bounded pipes must reap descendants after failure, cancellation and supervisor termination. A host-side reap helper must not depend on the model cooperating.

Do not persist prompts, responses, compaction summaries, token vectors, logits/probabilities, credentials, provider error bodies, command output containing sensitive data or exception dumps. Broker frames are private task transport and must not become logs. Filter errors to fixed categories at every subprocess boundary. Generated code must not embed copied private input text, secrets, vocabularies or expected private model outputs; inspect exports for these fields and accidental private paths. Adapted approved public architecture code retains its source revision, copyright and license provenance.

Native verification uses a host-controlled runner that discards raw stdout/stderr and compares execution values in memory. Prohibited execution content cannot be persisted even in private validation artifacts; retain permitted aggregate results and hashes only. Feedback to Pi contains check IDs, safe structural categories and aggregate mismatch counts. Public evidence publishes aggregate results and hashes only. The exact checkpoint source paths and permitted private fixtures stay under ignored storage.

## Budgets and recovery

Freeze explicit limits in each `JobSpec`. Proposed pilot defaults are implementation targets, to be measured and adjusted with new evidence:

| Resource | Initial limit / policy |
| --- | --- |
| Coding session | 30 minutes, 128 model calls, 512 broker calls and 6 repair cycles; stop at the first exhausted limit. |
| Token use | 250,000 cumulative input tokens and 32,000 cumulative output tokens, counting retries and compaction. Reserve each request's maximum output before dispatch. |
| File access | 128 KiB per paginated read, 1 MiB per candidate file, 64 MiB candidate/source-code closure and 2 GiB non-model scratch. |
| Local checks | 300 seconds per fixed check; separate predeclared limits for long private parity/load runs. |
| Memory | Terminate candidate process trees at a 2 GiB sampled aggregate RSS threshold. Record sampling interval and overshoot; this is a termination policy, not a proven hard memory cap. |
| Real weights | Acquisition is a separate preflight stage with declared total bytes, free-space reserve and owned cache/output paths. Refuse acquisition if the bound cannot be determined. |
| Concurrency | One writer per candidate/job and one native model-validation owner per host; queue memory-heavy conversions and measurements. Independent authoring/synthetic jobs use disjoint workspaces within measured capacity. |

Provider context/output limits are separate from cumulative budget. Before each request, bound the remaining input/output allowance; enforce cancellation as usage approaches it. Capture actual reported usage when available. If the provider cannot supply reliable usage or request bounds, return `usage_unverifiable` rather than promise a token cap. An optional monetary budget needs a recorded pricing snapshot and conservative request reservation; otherwise report token bounds and do not claim a billing ceiling.

After a provider failure, cancellation or supervisor restart, dispose of the in-memory conversation and stop all owned processes. Do not auto-resume a partial coding job. A new job can reuse a sealed candidate after rechecking its base, source and inventory; it cannot resurrect a transcript or promote stale evidence. Terminal verification failures return a fixed disposition after the repair budget. Changed protected inputs require a fresh job or explicit revalidation.

## Independent verification and promotion

Verification levels are cumulative where applicable. Every report identifies skipped, inaccessible and unsupported checks explicitly. A native runtime block does not erase a successful conversion-only comparison, and conversion-only success cannot be presented as runtime support.

| Level | Required evidence | Claim permitted |
| --- | --- | --- |
| V0: candidate integrity | Complete regular-file inventory; accepted schema/base/dependency policy; no protected-file changes; source/license provenance; private-path/content export checks. | Inert source candidate can be reviewed. |
| V1: synthetic correctness | Fresh confined process; host-owned positive/adversarial fixtures; full tensor-consumption/config checks; candidate tests; lint; existing affected regression suite; cancellation and memory/workspace checks. | The declared synthetic layouts pass. |
| V2: conversion parity | Pinned independent oracle; complete tensor names/shapes/dtypes/payload hashes/specs, logical config and normalized tokenizer comparisons; verified source and metadata dependencies; initial/final hashes. | Exact checked checkpoint/config/dtype tuples match the reference conversion. |
| V3: native loading | Exact BF16 artifacts and named Rust 1.98.1 build; private load-only readiness/EOF, full tensor consumption, zero scoring/generation/classification calls and process cleanup. | The exact artifacts load in the recorded build/backend. |
| V4: execution semantics | Separate predeclared private numerical equivalence protocol, reference implementation, cases/tolerances and backend/build identity; aggregate results/hashes only. | Only the checked execution behavior meets its declared criteria. |
| V5: quality/performance | Separate representative evaluation and serial reproducible performance protocol with source/build/workload binding. | Only the measured quality or performance result. |

The first pilot requires V0–V3 for its exact small dense tuple. It makes zero native scoring, generation or classification calls. V4/V5 are later scoped validation work and remain unverified in the pilot. A future claim that an unfamiliar model runs correctly requires V4, with short/long contexts, positional/attention masks, recurrent state resets and model-kind-specific behavior where applicable. Do not conflate artifact parity with an independently correct engine implementation.

### Host-owned oracles

Freeze expected outputs and validation programs before Pi receives the task. For currently referenced architectures, use pinned Lalamo in a separate locked reference environment or archived independently verified fixtures. Lalamo/JAX/PyTorch remain excluded from converter and builder runtime dependencies; they may be used by a separately controlled reference generator when necessary.

For genuinely new architecture sources with no Lalamo adapter, specify a reference loader/forward implementation and a trusted mapping to Stark's artifact semantics. Synthetic reference transforms must be written or reviewed independently of the candidate. If a complete independent mapping cannot be established, V2 is `not_established`; keep the candidate experimental and do not claim qualified support. Two independently generated implementations agreeing is useful investigation evidence, not a substitute for an authoritative semantic contract.

Pi cannot access oracle code, hidden fixtures, expected payloads or verifier directories. Return check IDs, fixed reasons and aggregate mismatch counts only. Maintain undisclosed holdout fixtures and run them after freezing the final candidate. A mismatch cannot be repaired by changing the oracle, loosening tolerances, ignoring tensors or downgrading a check.

Candidate tests supplement independent tests. Host tests must exercise:

- Exact config semantics, positional/attention variants, projection ordering, tied/untied embeddings, aliases, metadata routing and full source-tensor accounting.
- BF16/F16/F32 edge behavior through existing reviewed primitives; dtype mismatches and incorrect fusion/scaling rejection.
- Malformed headers, missing/extra tensors, contradictory declarations, mixed packed representations, wrong dimensions and unsupported configuration variants.
- Source mutation before/during conversion, missing shards/sidecars, replay tampering and changed toolchain/dependency identities.
- Output collisions, interruption, bounded workspace, relocation, import-time side effects and attempted protected-file/network/credential access.

MLX expansion additionally checks complete packed weight/scale/bias triplets, bit/group identity, payload byte preservation, coefficient transforms, partial/mixed fusion rejection and source-specific normalization conventions. Classifier expansion checks explicit model kind, labels, pooling, activation and supported tokenizer/config semantics, using the classifier harness rather than language-worker readiness.

### Maintainer integration

Review the sealed patch and evidence against its exact base. Apply into a separate development branch, run affected converter tests and independently review new dispatch/metadata semantics. Recompute converter source provenance after integration; the candidate sandbox source hash cannot silently substitute for the final integrated package hash. Rerun affected checks when any semantic input changes. Unchanged reference bytes may be reused only with an explicit identity/equivalence record.

Only reviewed catalog/profile additions change implementation coverage. Exact native qualification still requires the existing catalog identity, source-manifest/output/config/tokenizer hashes and complete named-build receipt. Generic checkpoints remain native-unverified and cannot satisfy `--require-native-support`; this plan does not change that policy. Broadening generic qualification would require a separate schema/compatibility decision.

Execution evidence stays separate from the current load-only ledger. An accepted adapter defaults to the current capability rules rather than inheriting a model family's prior qualification. Retain previous source-bound records unchanged and append new evidence for new tooling or source state.

## Pilot design and success criteria

The pilot is a controlled adapter-development experiment. It must not count regeneration of existing support as additional catalog coverage.

Before coding, freeze the task set, source revisions, oracle hashes, hidden fixtures, model/provider, budgets, success criteria and the control converter revision in a public-safe protocol. Choose the smallest accessible checkpoint that fits declared disk/memory bounds. Current model cleanup means old manifests may point to deleted weights; reacquire exact immutable inputs and verify hashes rather than assuming historical outputs remain available.

Use three task types:

1. A compatible uncatalogued Qwen3 checkpoint/fixture demonstrates deterministic reuse. It must make zero provider requests and retain generic qualification status.
2. Withhold the existing dense LFM2 adapter from a controlled converter snapshot and ask Pi to reconstruct the architecture mapping from pinned public sources and approved shared helpers. Keep the original adapter, architecture-specific tests, result fixtures, documentation containing its implementation and Git history outside the agent-visible closure. This exercises short-convolution/attention semantics with an independently known oracle. It is explicitly a withheld-adapter experiment.
3. Frozen negative tasks demonstrate an unsupported native configuration, ambiguous architecture source and malformed supported checkpoint. They must yield bounded dispositions without code that weakens the existing validator or changes the native engine.

Run three independent coding sessions for the withheld-adapter task under the same frozen policy. Record provider/model identity, available generation controls, SDK/runtime/lock hashes, attempts/repairs, usage, elapsed time, interventions, candidate hashes and every pass/fail disposition. Do not retain prompts or responses. If the provider does not expose a stable model revision/seed, record that limitation; do not describe sessions as reproducible model outputs.

Pilot acceptance requires all of the following:

- All deterministic and negative routing tasks have the expected dispositions; no unsupported runtime case is falsely promoted.
- Each of the three withheld-adapter sessions produces V0–V2 passing candidates within its budget, with no developer implementation edits. Record failed sessions and developer assistance separately rather than resetting the denominator.
- Host-owned positive/negative/holdout fixtures cover at least two structural configurations, complete tensor consumption and every declared transform/rejection rule.
- The exact selected dense BF16 real-weight tuple passes V2 and V3 for each candidate, with zero native inference calls. F16/F32 can pass synthetic/conversion checks and retain runtime-blocked status.
- Attempted checkout/credential/oracle/network access is denied; cancellation/crash tests leave no live children or partial published artifact. No prohibited content appears in public or private logs.
- Source/candidate mutation and incompatible-base exports fail. Candidate imports do not expand dependencies. Existing deterministic conversion and qualification behavior remains intact.
- Aggregate cost, tokens, elapsed time and reviewer/intervention effort are reported. Compare developer effort with a predeclared manual task or a recorded comparable adapter packet; if no credible comparison exists, report benefit as unproven.
- All job-owned large model outputs and sources are cleaned after validation/review under the ownership policy below. Retained evidence remains sufficient to reacquire and reverify exact inputs.

A passing pilot establishes the controlled workflow only. The first genuine missing-adapter follow-up must be selected from a fresh inventory assessment, map to existing native operations, have accessible authoritative sources and a usable independent oracle. Freeze that task before generation. Unsupported engine work must not be disguised as a converter success to produce a favorable result.

If the pilot fails, preserve aggregate failures and candidate hashes, identify the owning boundary, and decide whether a narrowly scoped repair justifies a new protocol. Do not automatically increase budgets, loosen gates or expand model scope. There is no broad auto-generation rollout until the first genuine missing-adapter packet passes the same independent acceptance process.

## Implementation phases and gates

All tasks below are pending. Builder gates B0–B6 are separate from the existing conversion plan's G0–G5 gates.

### Phase 0: freeze contracts, task set and provenance

- [ ] Define schemas, state machine, error categories, disclosure rules, limits and supported-host requirements.
- [ ] Record exact Stark/Meecp/Pi/runtime source pins and review copyright/license ownership before copying host or worker code. Meecp has component license records; no repository-wide license was established during this inspection.
- [ ] Inspect the latest remaining conversion inventory and classify native/helper/metadata gaps.
- [ ] Freeze pilot sources/oracles/holdouts and predeclare all claimed qualification levels and metrics.
- [ ] Specify candidate entry-point/registration materialization and protected inputs without changing existing runtime contracts.

Gate B0: independently reviewed contracts and pilot protocol; each task has a bounded route, source access plan and independent oracle or an explicit blocked disposition.

### Phase 1: supervisor, worker and confinement

Dependencies: B0.

- [ ] Implement the separate CLI, worker protocol, in-memory SDK configuration and deterministic preflight.
- [ ] Build brokered file/source/check tools, fixed runner profiles, sanitized diagnostics, budgets and owned process reaping.
- [ ] Qualify worker/provider and candidate-runner confinement on the supported Apple-silicon macOS host, including denial tests and failure-to-apply behavior.
- [ ] Pin Node/Pi and Python environments, retain licenses and source provenance, and verify runtime inventory before launch.
- [ ] Use a synthetic loopback provider to test SDK sessions, tool routing, mandatory verification, repair bounds, compaction, provider failures and cancellation without a paid request.

Gate B1: no real model/provider data required; all boundary tests pass, transcripts/credentials are not persisted, and unsupported hosts cannot execute untrusted candidates.

### Phase 2: candidate contract and independent verifier

Dependencies: B0 interfaces; generation execution waits for B1.

- [ ] Implement candidate manifests, read-only base snapshots, typed registration proposals and sealed export.
- [ ] Implement host-owned syntax/lint/adversarial checks and the V0–V2 report, with private oracle isolation and mutation detection.
- [ ] Integrate existing source snapshots, converter provenance, parity comparison and publication checks through fixed host runners.
- [ ] Add relocation, protected-input tampering, schema/version mismatch, attempted test bypass and incompatible-base tests.
- [ ] Implement content-safe evidence/feedback and complete post-verification rehashing.

Gate B2: synthetic generated candidates can be accepted/rejected/exported in fresh confined processes without weakening converter or oracle code; a forged success frame cannot bypass independent verification.

### Phase 3: controlled pilot

Dependencies: B1 and B2, frozen B0 protocol, independently reviewed integration.

- [ ] Run the deterministic and negative routing controls, then the three bounded withheld-adapter sessions.
- [ ] Perform separate exact real-weight V2 comparisons and V3 load-only checks; serialize model activity and measurements.
- [ ] Publish aggregate outcomes, intervention effort, budgets, scope and hashes, with a review disposition for every failure.
- [ ] Remove job-owned large weights after review and record cleanup/reacquisition metadata.

Gate B3: the complete pilot acceptance list passes. No new catalog/runtime/execution claim follows solely from this gate.

### Phase 4: first genuine missing adapter

Dependencies: B3 and a separately frozen, source-accessible architecture/format packet with existing native operations.

- [ ] Select one unmet adapter/variant, establish independent mapping/fixtures and run the bounded builder.
- [ ] Independently review and integrate source/tests/registration/profile semantics into a development branch.
- [ ] Recompute final converter provenance and rerun affected synthetic, parity and named-build loading checks.
- [ ] Record exact implementation/qualification scope and update catalog/evidence only when their existing requirements are met.

Gate B4: a real conversion gap is closed by reviewed code with independent source-bound evidence; all unrelated open conversion/runtime gates retain their statuses.

### Phase 5: opt-in converter integration and format expansion

Dependencies: B4.

- [ ] Add a documented opt-in converter entry point using the standalone process contract, explicit provider/disclosure/budget selection and safe failure handling.
- [ ] Verify normal converter installation/import/conversion works with no Node, Pi, provider credentials or network.
- [ ] Add MLX, hybrid or classifier packets independently, with their format/model-kind-specific tests and private harnesses.
- [ ] Add V4 execution validation only under a separate predeclared private protocol; report untested execution as unverified.

Gate B5: opt-in integration is useful on the accepted scope, the deterministic path remains intact, and expanded formats each have independent evidence.

### Phase 6: maintenance and distribution decision

Dependencies: B5 and demonstrated usage.

- [ ] Define engine/converter/schema compatibility and supported provider/runtime upgrade policy.
- [ ] Maintain regression corpora and rerun affected boundary/parity checks on SDK, runtime, helper, profile or reference changes.
- [ ] Evaluate local package installation and supply-chain inventory without enabling inherited release workflows.
- [ ] Propose any public distribution, signing or release implementation separately, with its own scope and validation.

Gate B6: documented maintenance responsibilities and compatibility/revalidation policy. Public release infrastructure is not authorized or implemented by this plan.

## Dependency-aware parallel execution

Use the environment's actual agent capacity. With four available slots, assign the coordinator plus three disjoint workstreams after B0:

| Workstream | File ownership | Dependencies / integration rule |
| --- | --- | --- |
| Worker and SDK tests | `worker/` | Stable protocol; no supervisor/broker edits. |
| Supervisor, broker and confinement | Supervisor/broker/lifecycle modules and their tests | Frozen schemas; actual provider/source policies tested before candidate execution. |
| Candidate verifier and export | Verifier/export modules and host-owned fixtures | Frozen candidate contract; can develop from synthetic fixtures while confinement work proceeds. |
| Coordinator | Schemas, policy integration, docs and gate evidence | Integrates packets, resolves ownership conflicts and dispatches independent reviews. |

Replace a completed implementation workstream with an independent reviewer. Delegate small read-only lookups to lower-cost agents. Shared converter modules have one owner; changes to common contracts are coordinated before dependent work resumes. Do not run native qualification, weight-heavy conversion or timing comparisons concurrently with competing model activity. Concurrency improves development throughput; it does not remove dependency or memory constraints.

## Dependency and build policy

Use `uv` with a locked managed Python 3.13 environment compatible with the converter. The inspected Meecp worker uses Pi SDK 1.0.2 and Node 24.15.0; these are starting reference pins, not automatic choices for a new implementation. Select an exact validated runtime/SDK pair at B0, record source/integrity/license hashes and account for current advisory checks. Never float dependencies to a moving latest release.

Install Node dependencies with the checked-in lock and scripts disabled. Any exceptional build step must be explicit, isolated and reviewed. Generated adapters cannot add dependencies; a needed new primitive or dependency is a separate implementation packet.

When adding the builder locks, extend repository policy and advisory coverage together. Today `scripts/ci/check_repository.py` does not include npm `package-lock.json` in its lockfile inventory. The implementation must add that format, the builder's Python and Node locks, and corresponding explicit OSV scan/report-gate inputs. Include runtime/SDK license and source checks. Do not infer security coverage merely because a lockfile exists.

Local implementation checks will include locked builder Python tests/lint, locked real-SDK mock-provider tests, confinement/lifecycle tests and affected existing converter tests. These existing commands remain relevant:

```sh
uv run --locked --python 3.12.13 --script scripts/ci/check_repository.py
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false uv run --project tools/converter --locked --python 3.13.13 pytest tools/converter/tests -q
uv run --project tools/converter --locked --python 3.13.13 ruff check tools/converter
uv run --project tools/converter --locked --python 3.13.13 ruff format --check tools/converter
uv lock --check --directory tools/converter
```

Public CI uses synthetic inputs and mock providers only. Real checkpoint acquisition, oracle/reference generation, native loading, execution and performance evidence stay in separately controlled private runs. CI must not upload model-dependent artifacts, prompts, transcripts or private source manifests. The inherited cloud/signing/release infrastructure stays disabled.

## Artifact lifecycle and cleanup

Job workspaces, runtime bundles, provider transports, model files, private sources/manifests and verification artifacts live under ignored `runtime/`, `models/`, `artifacts/` or `test-artifacts/` roots. Apply owner-only permissions. Update ignore rules for Node dependencies and builder caches before introducing them; binaries, distributions and dependency closures are never tracked.

Retain small source receipts, provenance, schemas, candidate source patches, aggregate evidence and hashes according to their public/private classification. Runtime transcripts are never a retained artifact. Approved adapter source/tests and sanitized provenance enter Git only through maintainer review.

Follow the existing [model cleanup policy](MODEL_CONVERSION_ARTIFACT_CLEANUP.md) and [conversion plan](MODEL_CONVERSION_PLAN.md#model-artifact-cleanup): remove job-owned large generated weights and acquired source weights after validation/benchmarks/review, while preserving the evidence needed to reacquire exact immutable sources. No recursive deletion of shared caches or user-supplied models. Use an ownership inventory and identity/open-file rechecks; retain active/shared/uncertain files and report the disposition. Cancellation may remove owned scratch after processes are reaped, while retaining a sealed candidate when the user requests it.

Receipts must distinguish missing cleaned-up payloads from retained evidence. Revalidation reacquires sources, verifies recorded hashes and creates a new run; it does not pretend an old manifest can replay deleted files. Disk estimates include input, candidate, reference and verification outputs simultaneously.

## Risks and unresolved implementation choices

| Risk or choice | Required resolution |
| --- | --- |
| A generated adapter passes self-authored tests while implementing wrong semantics | Independent mapping/oracle, adversarial tests, hidden holdouts, full parity and scoped claims. |
| Source code or metadata instructs the coding agent to bypass checks | Treat selected source as untrusted data; host capabilities and immutable verifier enforce authority regardless of agent instructions. |
| Python import or tool execution escapes its job | Actual OS confinement, exact-path access, no candidate network, lifecycle ownership and denial tests before real generation. |
| Unknown architecture cannot be expressed by existing engine operations | `needs_engine_work` with evidence and separate native implementation; no guessed mapping. |
| Agent repairs weaken existing validation or invent metadata | Protected shared code/locks/oracles, declarative proposals and independent semantic review. |
| Hosted provider discloses private checkpoint material or cannot meet budget controls | Explicit source disclosure policy, selected compatible provider, usage-verification gate and local-provider option where suitable. |
| Worker/runtime source has incomplete license provenance | Establish ownership/permission and preserve notices before copying; implement equivalent behavior where provenance cannot be established. |
| Native load-only proof is marketed as execution correctness | Independent V0–V5 levels and unchanged ledger semantics. |
| Budget, disk or cancellation handling is ineffective under children/retries | Request reservations, process-tree tests, declared acquisition bytes, interruption cleanup and measured threshold limitations. |
| SDK/provider upgrades silently alter tools, storage or protocol | Exact pins, real-SDK mock tests, new provenance and affected boundary revalidation. |
| Pilot uses already-solved tasks and overstates usefulness | Label withheld experiments, report effort limitations and require a genuine missing-adapter packet before rollout. |

At B0, select the exact Pi/Node pair, supported macOS confinement mechanism, provider usage-accounting contract and pilot reference environment. These are routine implementation decisions with recorded evidence, not reasons to weaken acceptance. If a hard boundary cannot be implemented on the target host, retain inspect/export-only behavior and report the execution limitation.

## Completion criteria

The initial deliverable is complete at B4: an optional, separately runnable builder; pinned dependencies/licenses; tested process/data boundaries; immutable candidate export; mandatory independent verifier; successful controlled pilot; and one genuinely missing adapter integrated with precise source-bound conversion/loading evidence. It must operate without changing the installed converter's normal dependencies or qualification rules.

Converter UX integration, MLX/hybrid/classifier expansion, execution/quality claims and public distribution remain separately gated follow-ups. A full release of the planned scope requires B5 and the B6 maintenance policy; unfinished gates stay visible.

## Sources and reuse provenance

- Stark [converter contracts](../tools/converter/stark_convert/contracts.py), [registry](../tools/converter/stark_convert/registry.py), [planner/publication](../tools/converter/stark_convert/cli.py), [sources](../tools/converter/stark_convert/sources.py), [provenance](../tools/converter/stark_convert/provenance.py) and [qualification matching](../tools/converter/stark_convert/qualifications.py).
- Stark [replacement plan](MODEL_CONVERSION_PLAN.md), [CI policy](CI.md) and [model cleanup record](MODEL_CONVERSION_ARTIFACT_CLEANUP.md).
- Meecp at `f818b73a00161b880fe544397108abb52b76db82`: [worker/protocol](https://github.com/jshahbazi/meecp/blob/f818b73a00161b880fe544397108abb52b76db82/agents/pi/README.md), [SDK worker](https://github.com/jshahbazi/meecp/blob/f818b73a00161b880fe544397108abb52b76db82/agents/pi/worker.mjs), [component provenance](https://github.com/jshahbazi/meecp/blob/f818b73a00161b880fe544397108abb52b76db82/agents/pi/provenance.json), [host confinement](https://github.com/jshahbazi/meecp/blob/f818b73a00161b880fe544397108abb52b76db82/backend/meecp/pi_dependencies.py) and [independent verification](https://github.com/jshahbazi/meecp/blob/f818b73a00161b880fe544397108abb52b76db82/backend/meecp/preparation_verifier.py). Reuse behavior selectively; copying code requires recorded ownership/license provenance.
- Pi's official [SDK](https://pi.dev/docs/latest/sdk) and [extensions](https://pi.dev/docs/latest/extensions) documentation describe the integration surface and process permissions. These moving documentation pages are orientation sources; implementation evidence must bind the selected version's actual APIs and source hashes.
