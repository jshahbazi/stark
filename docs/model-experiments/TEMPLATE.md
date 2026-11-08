# YYYYMMDD-AREA-NN: model and attempted change

Copy this template to a dated validation record or benchmark report. Replace every placeholder; use `not recorded`, `not run` or `not measured` when appropriate. Add the attempt to the [ledger](README.md). Adjust that link in the copied file: use `../../model-experiments/README.md` from `docs/archive/runs/`, or `../model-experiments/README.md` from `docs/benchmarks/`.

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / date / owner | |
| Kind | Compatibility, correctness, memory, precision, kernel, sampling, prefix reuse or performance |
| Attempt state | Proposed / attempted / completed / blocked / superseded |
| Result and acceptance criterion | Passed / failed / inconclusive / not measured; scope and criterion |
| Disposition | Experiment only / optional / adopted on branch / merged / released / installed / rejected / superseded |
| Related attempts / governing issue / PR | |

## Hypothesis and change

- Concrete problem, trigger and expected behavior:
- Baseline and candidate difference:
- Acceptance criteria set before the run:
- Why the attempt was stopped, rejected or superseded, if applicable:

## Artifact, source and environment

| Field | Baseline | Candidate |
| --- | --- | --- |
| Model family, exact checkpoint and upstream revision | | |
| Format/quantization and config/weight/tokenizer/processor/template hashes | | |
| Backend, vendor revision and local patch provenance | | |
| Stark commit, dirty patch/runtime manifest and executable hash | | |
| Toolchains, dependency lock hashes and build flags | | |
| Weight, activation, accumulation, recurrent-state and KV precision | | |
| Hardware, physical RAM, OS/driver/Metal toolchain | | |
| Available headroom, observed peak memory and measurement method | | |

Do not repair historical identities using the current filesystem. Record missing/malformed identities and their effect on confidence.

## Workload and controls

- Task/mode and privacy-preserving corpus/fixture identifier/hash:
- Exact input/context/output budgets; sampled versus emitted/useful-token definitions:
- Sampling, seeds, stop rules, MTP/draft depth and other state-affecting options:
- Cold/warm, loading/warmup exclusions, prefix-cache state and reused counts:
- Run order, repeats, concurrency, deadlines and host-eligibility/paging screens:
- Exact command/config receipt identifier or hash; private artifacts stay ignored:

## Attempt history and failures

| Trial / source | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| | | | |

Include setup/admission/compile failures that block execution, numerical mismatches, regressions, invalid timing arms and discarded ideas. Distinguish a code-review risk from a reproduced runtime failure. Preserve invalidated results with the reason they cannot support a claim.

## Correctness and lifecycle

- Reference, numerical tolerance and tested scope; aggregate errors/hashes:
- Model-free versus actual artifact execution:
- Prefix reuse support/restoration proof or concrete blocker, key/options and exactness:
- Context/resource rejection, cancellation, timeout, reset/restart and same-worker recovery:
- Buffer ownership, scratch lifetime, peak-state assumptions and device/index limits:
- Tests actually run, their revision/configuration and ignored/skipped coverage:

## Performance and memory

| Metric and measurement boundary | Baseline | Candidate | Repeats/dispersion | Eligibility and conclusion |
| --- | --- | --- | --- | --- |
| Tokenization/rendering/planning | | | | |
| First token / prefill | | | | |
| Useful decode / end-to-end rate | | | | |
| Observed peak memory / retained payload | | | | |

Leave performance explicitly `not measured` when only functional tests were run. Separate component gains from whole-request gains; separate available-memory estimates, payload accounting, RSS and actual reservations. Record exclusions and regressions even when correctness passes.

## Evidence, decision and limits

- Source-bound public aggregates and private receipt hashes:
- What worked, what did not, and which exact model/artifact/options each result covers:
- Adoption/default decision and supporting commit/PR/build/installation evidence:
- Untested models, formats, contexts, encoder paths, quality or lifecycle cases:
- Next action and required evidence; link any superseding attempt without rewriting this result:

Keep prompts, completions, token vectors, probabilities, credentials, exception dumps and model/private evaluation files out of this document.
