# WORKSTREAM-YYYYMMDD-AREA-NNN: short model change or experiment title

## Question and scope

| Field | Value |
| --- | --- |
| Registered / executed | Dates separately; retrospective entries must identify themselves |
| Owner / PR | Author and implementation/review link |
| Kind | Compatibility, correctness, precision, performance, memory, state/lifecycle or admission |
| Problem / hypothesis | Observed defect and proposed explanation; distinguish facts from hypotheses |
| Predecessor | Prior attempt ID and immutable evidence link, or none |
| Exact target | Repository, immutable checkpoint revision, format and size |
| Artifact identity | Weights/config/tokenizer/template/processor and conversion/preparation hashes |
| Evidence scope | Structural, operator/component, full-model numerical, lifecycle, quality or end-to-end timing |

## Bound comparison

| Field | Baseline | Candidate |
| --- | --- | --- |
| Source commit / tree / patch | | |
| Executable / reference hashes | | |
| Compiler, dependencies and flags | | |
| Device, OS, memory and backend | | |
| Weight / activation / accumulation / cache precision | | |
| Selected kernel and materialization policies | | |
| Context, input/output lengths and batch | | |
| Sampler, seed, MTP/draft depth and prefix reuse | | |

Describe the one intended change and any unavoidable differences. Keep sources
and qualification artifacts distinct from the source used to generate a
reference. Write `not recorded` where evidence is unavailable.

## Fixed gates and procedure

- Correctness metric, threshold, controls and reference; record these before
  examining candidate results.
- For speed: warm/cold/reuse state, paired order, repetitions, distributions and
  eligible host screens. State the useful-work definition and measurement scope.
- Memory admission, resource/time limits and stop conditions; identify owned
  processes. Do not change another task's workers.
- Relevant prefix-cache, MTP, cancellation, malformed-input and restart checks.
- Privacy: only aggregate results and hashes enter committed output.

## Results

| Control / arm | Eligibility | Baseline | Candidate | Gate / outcome | Evidence |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

Include failures, rejected arms, invalid setup and guard stops. Report precision
counterexamples and excluded observations separately. Link immutable receipts,
source-bound public aggregates and the canonical detailed run record. Do not
include private evaluation content or raw logs.

## Decision and remaining work

| Field | Value |
| --- | --- |
| Outcome | Planned/untested, passed, failed, invalid setup, resource-deferred or inconclusive; name the exact gate |
| Promotion | Promoted within named scope, default-off, diagnostic only, rejected, deferred or awaiting qualification |
| What the result establishes | Exact supported statement |
| What it leaves unverified | Other artifacts, full-model execution, quality, timing, host coverage or numerical domains |
| Next test / owner | Concrete follow-up; include a blocker if it cannot run |

## Follow-up history

Append dated follow-ups and successor IDs. Preserve the original experiment and
receipts. When a later attempt changes the conclusion or shipping decision,
update the index with a link to that new evidence rather than rewriting the old
measurements.
