# Drex Max v1 Q4 Human100 registration

Registered 2026-11-08 before any Human100 Drex forward pass. This is a local,
fixed-checkpoint diagnostic replay requested by the owner. No training, model
editing, prompt search, threshold fitting, provider switch or production change
is authorized by this experiment.

## Fixed instrument

- Human100: the frozen 100 cases, human-label revision 121, with 67 accepted
  and 33 rejected references. This is reused private development data.
- Reuse the exact verdict-only SystemOne request snapshots from the completed
  SubQ Snap Human100 instrument. Validate its original seal, input hashes,
  completion/verdict hashes and every referenced private wire-file hash.
- J0 uses the frozen question, gold fact, supplied answer, rubric and critical
  entities. E1 adds the frozen original-source context. Do not alter criteria,
  option order, instructions or state. Change only the model identity field.
- Run both contexts twice, preserving the registered 400-record request order.
  Infer every request with fresh state and no output/prefix cache. Full
  noncausal attention prevents ordinary causal prefix reuse.
- Pin Drex source `d2fd26cdf211f87b9a6b1261995528629c9e1401` and affine-Q4
  snapshot `23f21ac6e664d3edb32faa7b55b0c8a7fb09134a`, using the upstream
  custom MLX scorer and embedded readout temperature. Do not use causal text
  generation. Keep the existing locked MLX environment.
- Preflight all 200 distinct prompts against the upstream 16,384-token limit.
  No truncation, omitted cases or changed source windows. Refuse an unsupported
  request before Human100 inference. A public synthetic warmup is permitted.

## Rulers, statistics and decision branches

Only native argmax `correct` accepts. Both `incorrect` and `partial` reject.
Primary results are binary acceptance agreement per 200 judgments/context,
false accepts and false rejects. Also report results per repeat, distinct-case
repeatability, verdict frequencies, binary Brier/log loss/ECE10 from P(correct),
and observed scoring latency. Do not conflate winning-option probability with
validated acceptance confidence.

Reuse compatible frozen baseline rows without rerunning competitors. Keep
their exact model/runtime identities and request-scope limitations. For
matched differences, pair case/context/repeat and cluster both repeats by
case: 10,000 percentile bootstrap draws, seed 20260929. Use first-repeat
discordances for an exact two-sided McNemar diagnostic. No multiple-comparison
adjustment is claimed. With only 100 distinct cases, small differences may
remain uncertain; repeat judgments do not double independent sample size.

The existing Human100 development screen of at least 88% J0 agreement is
reported descriptively. Passing advances only to fresh human confirmation;
failing does not justify this recipe for answer acceptance. No production
replacement can be qualified here because this run does not test Recall224,
fresh domains, deployment thresholds or the existing cascade gates. No bar or
metric changes after observing Drex results; no discretionary rerolls.

## Evidence and failure handling

Freeze the loader, harness, profile, input/wire inventory, registration and
model hashes before forwarding a Human100 request. Preserve the original
reference files. Store new run metadata under the same tenant/corpus research
scope. Only aggregate measurements and hashes may be published in Stark.
Prompts, source passages, responses, token vectors, native probabilities and
credentials stay in memory/private pipes; do not persist or log them.

Checkpoint aggregate coverage, confusion counts and usage after every successful
judgment. Keep per-request readouts in memory for the full analysis. An
independent analysis process consumes those readouts through a private pipe,
recomputes metrics and validates source/input seals. Publish its aggregate
audit receipt only. A model or instrumentation failure keeps an incomplete
receipt and is not a full result. Any necessary correction gets a separately
dated deviation and fresh validation; do not silently replace earlier results.

No noisy-host speed ratio is claimed. Retain sampled host activity, RSS,
pageout/swapout and sleep observations. Accuracy and artifact provenance are
the primary evidence in this task. Historical benchmark records stay separate.
