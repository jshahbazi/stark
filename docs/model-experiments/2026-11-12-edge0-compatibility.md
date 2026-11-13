# 20261112-EDGE0-01: Edge0-145B compatibility intake

This is the Phase 0 record for the [Edge0 native support plan](../EDGE0_PLAN.md).
It registers the compatibility hypothesis and its prerequisites. No weights
were downloaded, implementation changed, build produced or model executed for
this attempt. Metadata intake does not establish Stark support.

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | `20261112-EDGE0-01` / 2026-11-12 / no runtime execution / Codex records; repository owner decides the model-license disposition |
| Kind | Compatibility; Phase 0 provenance and license intake |
| Attempt state | Proposed; Phase 0 gate unmet |
| Result and acceptance criterion | Planned / untested. Phase 0 requires this attempt record merged and the owner's license decision recorded; neither is established by this working document. |
| Evidence scope | Source/metadata intake reported by the pinned plan and the [license intake](../archive/runs/2026-11-12-edge0-license-intake.md); no operator, full-model, lifecycle, quality or timing evidence |
| Promotion decision | Awaiting qualification; no capability or default enabled |
| Disposition | Experiment record only; merge, release and installation not established |
| Predecessor / related attempts / governing issue / PR | No predecessor; governed by [EDGE0_PLAN.md](../EDGE0_PLAN.md) and [model onboarding](../MODEL_ONBOARDING.md); no PR |

## Hypothesis and change

- **Problem and trigger:** the pinned `Edge0/Edge0-145B-q3k` release has a
  Flash-Next-derived `qwen4exp` trunk, but uses a Strata expert pack and GGUF
  skeleton instead of Stark's existing artifact layouts. Its grouped latent
  experts, scale conventions, FP8 n-gram table and draft bank require separate
  compatibility work. An existing Flash-Next path does not admit these bytes.
- **Concrete hypothesis:** a native path can preserve the pinned payload bytes,
  adapt the trunk tensor names and implement grouped down-projection, selected
  latent experts, group up-projection and routed reduction. A scalar reference
  and separately qualified mixed-precision path could then establish bounded
  correctness. This is a proposed implementation route, not an execution result.
- **Baseline and candidate difference:** baseline is Stark
  `a171326271f6c5eaf31f4ea49bc21905f7e7a813`; no candidate implementation exists
  in this attempt. The proposed candidate is described by the plan. This
  Phase 0 change adds records only.
- **Qualification and reference sources:** the Q3_K checkpoint pin below is the
  sole first-artifact target. The pinned Edge0 source is a format/arithmetic
  authority. The proposed synthetic F64 oracle must be implemented independently
  from that specification in a later attempt; it does not exist as evidence
  here. No public Mac latent-path reference or repeated upstream numerical
  baseline is available in this intake.
- **Current stop:** the owner's model-license decision is pending and this
  attempt record has not been merged. No later phase starts on this record.

### Gates reserved for later attempts

These are planned acceptance requirements, not completed tests. Each later phase
gets its own attempt ID, sealed identities, controls and fixed limits before
execution. No numerical tolerance, speed-regression allowance, repetition count,
memory/time guard or workload budget is fixed by this intake.

| Scope | Planned acceptance requirement and status |
| --- | --- |
| Source preparation | Exact revision/file hashes, sizes and geometry; manifest/range/label consistency; byte equality for duplicated tensors; tokenizer IDs, merges, token types and special IDs agree. Exact equality required; full preparation receipt and tokenizer proof absent. |
| Readers, reference and precision | Trace scale conventions; independent decoder/forward references; operator controls, activation comparison and bounded real-artifact smoke. Numerical tolerances, precision policies and fixtures remain unfixed for their later attempts. |
| MTP, reuse, tools and lifecycle | Confirm draft format, preserve the head and qualify verification; exact successful-output/hash agreement and expected reused counts under matching options; complete onboarding state/lifecycle and tool continuation controls. No implementation or execution evidence. |
| Performance and admission | Matched cold/warm, prefill/decode and MTP on/off requests with repeats, host/paging screens and memory accounting. Workload, variability, resource limits and acceptance thresholds remain unfixed for later attempts. |

## Artifact, source and environment

The following pins and Hub-reported payload identities are carried from the
[pinned intake](../EDGE0_PLAN.md#pinned-intake). They have not been verified
against locally downloaded payloads in this attempt.

| Field | Baseline | Candidate |
| --- | --- | --- |
| Model family, exact checkpoint and upstream revision | No Edge0 baseline execution | Intended first artifact: `Edge0/Edge0-145B-q3k` at `d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`; Flash-Next-derived `qwen4exp` |
| Format/quantization and config/weight/tokenizer/processor/template hashes | No locally sealed model | Strata pack + GGUF v3 skeleton; default Q8_0 latent projections, Q3_K gate/up and IQ4_NL down expert records; FP8 E4M3 table. Payload hashes below. Full config/tokenizer/preparation manifest not sealed; selected license/notice hashes are recorded in the linked license intake. Template SHA-256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`. No image processor in the pinned inventory. |
| Backend, vendor revision and local patch provenance | Existing Stark source baseline only; Edge0 not loaded | No backend implemented. Format authority: `Edge0-AI/Edge0` at `b7900cbb322926ccebdcd0c0c5a1280497ecff22`; Strata import `f1933b1e4993fecc10d2e9360d85569619a2c42d`, merge base `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`; ggml from llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`. No code/layout/table port in this attempt. |
| Stark commit, dirty patch/runtime manifest and executable hash | `a171326271f6c5eaf31f4ea49bc21905f7e7a813`; no baseline executable or runtime manifest | Documentation-only working change based on the same commit; no candidate executable or runtime manifest |
| Toolchains, dependency lock hashes and build flags | Not recorded for a build; no build run | Not recorded; future scoring builds require Rust 1.98.1 and locked dependencies; Python tooling uses `uv` |
| Weight, activation, accumulation, recurrent-state and KV precision | No Edge0 execution policy | Storage types identified by intake; activation/accumulation/state/cache precision not selected or qualified. Upstream 8-bit activation blocks and proposed floating activations require a measured comparison. |
| Selected kernel and materialization policies | No Edge0 path selected | Proposed byte-retaining native path; kernels, paging, mappings and materialization not implemented |
| Hardware, physical RAM, OS/driver/Metal toolchain | No execution environment sealed | Intake plan reports a 128 GiB host; device/OS/Metal identities not recorded for this attempt |
| Available headroom, observed peak memory and measurement method | Not measured | Plan's historical intake free space was 317 GiB; not a current admission measurement. Peak memory, RSS, reservations and paging not measured. |

| Intended default payload | Bytes | Hub-reported SHA-256 |
| --- | ---: | --- |
| `experts.bin` | 40,076,574,720 | `947d15f3d69ecdb45d602a62bc84803451da779a22294e1db5b3568290b42879` |
| `ple-fp8.gguf` | 51,200,246,112 | `c7615c59b060277ddf6f84729fa8f7469cfc82c7fb983d6e7eab5a2f7446f170` |
| `model-skeleton.gguf` | 5,991,569,248 | `ab49fbe7590cdcda45d7a2bcae4be0bd42c6a70b951d594a00eb158d7df0b72f` |
| `latent_dense_q8.bin` | 3,743,416,320 | `fbb25caaa0dcf45ef62f1a4945dcf98a53a166934a3b461b166fbafa831d87ca` |
| `dense.bin` | 1,538,625,024 | `f9bcc7f36eac4e59a7b1cb1962e038ba12dc831a41fb193761f1d1e72657fb62` |
| `mtp/experts.bin` | 707,788,800 | `09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f` |
| `mtp/dense.bin` | 116,099,072 | `c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a` |
| `labels.bin` | 98,304 | `ce9b4a20b92c5c866e65d84a3884c104986522471cf11cc5f797a153affbcc06` |

The eight listed binaries total 103,374,417,600 bytes (96.27 GiB), including
`labels.bin`; small metadata files are not yet size-totaled. This is artifact
size accounting, not a download, working-set or supported-memory measurement.
Small manifests, scales and tokenizer remain revision-bound pending a full
preparation receipt; the linked license intake separately hashes selected
license/notice files. Optional
`latent_dense_q4.bin` is excluded from this first-artifact target.
`Edge0/Edge0-145B-NVFP4` at
`6aab5f4337b4c7146dd69130dad4c719f83fca03` is an intake comparison pin only;
NVFP4, Edge0-120B, 35B and 8B are outside this attempt's qualification target.

## Workload and controls

- **Install/load and images:** no installation or load. The pinned intake
  reports no vision/projector tensors among 1,079 skeleton tensors, no projector
  file and a `text-generation` pipeline tag. Image-on/off and real-image
  transition arms are N/A for this source inventory. Pairing a Flash-Next tower
  is unqualified and outside this attempt. Text cache, performance and MTP gates
  remain required.
- **MTP inventory:** intake reports one draft layer and a 512-expert bank; the
  Q2_0 expert layout is inferred from sizes and the pack tool, awaiting source
  confirmation. No drafts generated, head loaded or default chosen.
- **Execution recipe:** not run or selected. Fixture/corpus hashes, budgets,
  sampling/seeds, stop rules, cache modes, repeats/order, host/paging screens,
  deadlines and resource guards belong to later attempt records. No command,
  download, preparation run or owned model/GPU worker exists for this attempt.

## Attempt history and failures

| Trial / source | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| 2026-11-12 Phase 0 registration; Stark baseline above | Read governing plan, onboarding rules and ledger template; register the compatibility target and pending gate | Source/metadata intake only; no preparation, compile, operator or native execution attempted | Valid documentation scope. Compatibility untested; pending owner license decision and record merge. No runtime failure reproduced. |

## Correctness and lifecycle

No numerical, model-free, artifact or lifecycle tests were run. Tolerances are
not fixed; payload/template hashes above identify the target rather than outputs.
The plan proposes a 4,096-token mixed-precision snapshot of recurrent/convolution,
attention/indexer, n-gram and draft state. State tracing and a restoration proof
are absent; the proposed F32 bring-up keeps reuse and MTP unavailable. Context
and resource rejection, scope/options, cancellation, timeout, reset/restart,
recovery and scratch/device limits all remain untested. Documentation review
covers pins, links, missing identities and the Phase 0 gate.

## Performance and memory

Not measured: tokenization/rendering/planning, first token/prefill, useful decode,
end-to-end rate, peak memory and retained-state payload. There are no repetitions
or eligible timing arms. Upstream claims do not qualify Stark; file sizes do not
establish resident memory or supported host capacity.

## Evidence, decision and limits

- Public evidence: [pinned plan](../EDGE0_PLAN.md) and
  [license intake](../archive/runs/2026-11-12-edge0-license-intake.md). No private
  runtime receipt was produced by this attempt.
- What is established: a source-bound first-artifact target and documented
  compatibility work. Native loading, numerical correctness, tools, reuse,
  performance, memory admission and deployment remain untested.
- Model license: pending owner decision; see the license intake for pinned terms
  and provenance boundaries. This record supplies no license clearance.
- Adoption/default/build status: no implementation, enabled capability,
  installation or model-support documentation update. A merged record will close
  only the record portion of Phase 0.
- Next action: record the owner's model-license decision and merge the Phase 0
  records. Reassess the gate before beginning Phase 1. Preparation, source ports
  and later phases require their own attempt records and evidence.

## Follow-up history

### 2026-11-12: local evaluation decision

After reviewing the Phase 0 gate, the owner selected **local non-commercial
evaluation only** and instructed continued plan work with parallel Sol agents.
This authorizes local branch progression before the Phase 0 record is merged;
merge status remains pending. The original registration above preserves its
then-pending state. No commercial permission, product admission or runtime
qualification is established. Source preparation has successor attempt
[`20261112-EDGE0-02`](2026-11-12-edge0-source-preparation.md).
