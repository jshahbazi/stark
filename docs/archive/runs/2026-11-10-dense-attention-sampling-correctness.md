# Dense attention and sampling correctness validation

Attempt [20261110-DENSE-01](../../model-experiments/2026-11-10-dense-attention-sampling-correctness.md)
is based on `746e482f`. The implementation is on
`codex/model-family-correctness`; this record initially registers pending checks
and must be updated with the tested source and executable identities after
execution. No benchmark, quality comparison, release or installed-app change
has been performed by this attempt.

## Confirmed source defects and implementation

| Finding | Source disposition |
| --- | --- |
| Gemma 2 attention soft cap ignored | Confirmed. The cap reaches the common attention config and the CPU/Metal online score transform. Capped prefill/decode use this path; uncapped routing remains unchanged. |
| Tiny positive temperature kills worker | Confirmed. Request validation now rejects an overflowing F32 reciprocal before native dispatch. |
| Dense MTP changes combined top-k/top-p filtering | Confirmed. Dense Qwen MTP and Gemma assistant MTP now use full-vocabulary nucleus mass before intersecting top-k; hybrid policy remains separately unchanged. |
| Ordinary dense NaNs become token 0 | Confirmed. CPU/Metal sampling emits a sentinel for nonfinite raw or temperature-scaled logits, or an empty mask. Host stream resolution and drop cleanup reject/fence it, and the worker checks the request-state fence before replying or retaining a prefix. |
| Dense MTP copies a snapshot that cannot be retained | Confirmed for requests with no cache scope. The worker explicitly disables fresh-prefill capture in that case. |

## Validation state

Pinned Rust formatting completed after implementation. Integrated build and
execution are coordinated by the root task and are pending in this initial
record. The registered gates and tolerances are in the attempt record. The
full suite, production Gemma 2 logits, default image-loaded Qwen MTP, broad
contexts and performance are not established by source inspection or synthetic
tests.

Required engine filters are
`capped_attention_matches_independent_scaled_softmax_oracle`,
`native_sampling_rejects_nonfinite_logits_and_empty_masks` and
`failed_device_sampling_never_commits_a_token_to_history`. Required generator
filters are
`dense_mtp_keeps_full_vocabulary_nucleus_mass_when_top_k_is_enabled` and
`bad_request_does_not_dispatch_native_generation_or_poison_following_request`.
The ignored artifact test `dense_mtp_toy_greedy_and_prefix_metal` additionally
tests optional capture and exact continuation when explicitly supplied with
the private toy artifact.

The first integrated no-run build failed before execution: the CPU ancestor
attention helper's direct call lacked the new disabled-cap arguments, and the
new oracle test lacked the `Context` trait import. Both compile failures were
corrected; they provide no passing runtime evidence.

The initial cap, sampling-sentinel and host-history component filters passed on
CPU and Metal. Independent review then identified three follow-up gaps: an
ordinary grammar lookahead owner did not drain local submissions on early
sample errors; Drop-only text failures were invisible to worker cache
publication; and finite logits could overflow during temperature scaling and
produce inconsistent CPU/Metal results. The candidate now drains submitted
owners for all modes, exposes/checks a content-free request-state fence, and
rejects scaled nonfinite candidates. The earlier three passing filters do not
qualify these later edits; rebuild and repeat execution are required.

New lifecycle filters are
`failed_sample_resolution_drains_abandoned_text_lookahead` and
`ordinary_sampling_failure_fences_lookahead_state_cpu` / `_metal`. They cover
early-error draining and real tiny-model streams whose completed next-sample
slot is replaced by the sentinel, including Drop-only failure discovery.
Production Gemma 2 weights are not retained locally, so its artifact-level
logit and speed gates remain pending; no download was started.

## Remaining limits

Capped attention intentionally selects a correctness path with F32 online
score/softmax accumulation. Gemma 2 prefill and long-context speed have not been
measured. Existing records asserting both caps were applied remain historical
source-bound evidence and are not silently rewritten as passing this candidate.

Dense ordinary and dense MTP filtering now share the full-vocabulary nucleus
definition. Their RNG implementations and rounding remain distinct, as before;
same-seed output equivalence across MTP modes is not an acceptance claim.
The worker still conservatively admits snapshot memory even when capture is
omitted; this change removes unused copies rather than reducing admission
limits. Peak memory savings have not been measured.

## Final integrated results

The final source is `b41201523b547ea646f1378104319e7f501b3ac7`. The sealed
CPU/Metal test executable passes capped attention against its independent
F64 oracle, raw/scaled nonfinite and empty-mask sampling, sentinel resolution,
abandoned submitted lookahead draining, and real-stream next/drop failure
fencing on both backends. Sampling property controls also pass. The explicit
synthetic dense MTP fixture passes two controls, including optional snapshot
capture and exact continuation. No tolerance was relaxed.

Independent review found three cleanup issues before finalization: a local
ordinary grammar lookahead owner did not drain on early resolution failure;
ordinary worker cleanup ignored the newly fenced state; and Metal could drop an
overflowed candidate and choose a lower finite logit. All were fixed and their
new regression controls passed. The reviewer checked the corrected source and
reported no remaining concrete issue in that scope.

Release generator tests pass 144 with four ignored. Six process-boundary cases
pass; both scorer configurations pass 34 tests. Clippy with all generator
targets and warnings denied passes. Actual converted Qwen3-4B Q4 returns
recoverable invalid_sampling_request for temperature 1e-40 and then produces
four greedy tokens successfully in the same process. This is a bounded recovery
smoke, not a Gemma 2 numerical or image MTP qualification.

The [aggregate validation record](2026-11-10-model-family-triage.md) owns exact
binary/lock/artifact hashes, environment, setup failures and remaining gates.
A subsequent Clippy-only correction replaced constant-size chunk iteration in
the PNG converter; the sealed executable reran all selected engine controls.
