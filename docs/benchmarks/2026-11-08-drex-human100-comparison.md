# Drex Max v1 Q4 on Human100

Drex agreed with the frozen human acceptance labels on **86% of J0 judgments**
without source evidence and **87% of E1 judgments** with source evidence.
The saved JevK5-MLX baseline scored 90% and 88%, respectively. All **400
registered judgments completed**, with identical native probability vectors
across the two repeats. The independent analysis passed 809 counted checks
with zero errors; its confusion counts exactly match the live checkpoints.

This is the same 100 private Human100 cases, human-label revision 121:
67 accepted and 33 rejected. Each is judged in both contexts and repeated
twice. Only native argmax `correct` accepts; `partial` and `incorrect` reject.
These are binary acceptance-agreement scores, not human three-way verdict
labels. This reused diagnostic population is not a fresh confirmation set.

## Agreement and acceptance errors

| Context | Drex agreement | Saved JevK5-MLX agreement | Drex false accepts / false rejects per repeat | MLX false accepts / false rejects per repeat |
| --- | ---: | ---: | ---: | ---: |
| J0, no source evidence | 172/200, 86% | 180/200, 90% | 8 / 6 | 8 / 2 |
| E1, frozen source evidence | 174/200, 87% | 176/200, 88% | 7 / 6 | 9 / 3 |

Each repeat contains 100 distinct cases. False accepts are out of the 33
human-rejected cases; false rejects are out of the 67 human-accepted cases.
The table's per-repeat error counts are identical in both repeats. Over 200
judgments, Drex has 16 false accepts and 12 false rejects in J0, and 14/12
in E1. Its false-accept rates are 24.24% J0 and 21.21% E1; false-reject rate
is 8.96% in both contexts.

Drex rejects more human-accepted answers than the saved MLX recipe in both
contexts. Source evidence reduces Drex's false accepts by one distinct case
and leaves its false rejects unchanged. The registered J0 development bar
of at least 88% is not met. This run does not test Recall224, fresh domains,
production selection thresholds or existing cascade gates, so no production
replacement is qualified.

## Uncertainty and repeatability

Both repeats are clustered by case, not treated as 200 independent examples.
The registered 10,000-draw percentile bootstrap uses seed 20260929.

| Context | Drex agreement 95% interval | Drex minus saved MLX | Paired 95% interval | First-repeat exact McNemar p |
| --- | ---: | ---: | ---: | ---: |
| J0 | 79%–92% | −4 percentage points | −10 to +1 points | 0.2890625 |
| E1 | 80%–93% | −1 percentage point | −7 to +5 points | 1.0 |

The first J0 repeat has two cases only Drex gets right and six only the saved
MLX recipe gets right. E1 has four and five, respectively. Both paired
intervals include zero. Point agreement is lower on this cohort; the small
reused set does not establish a general model ranking or non-inferiority.
No multiple-comparison adjustment or formal prospective power target is
claimed. There are 100 distinct facts, not 400 independent samples.

Native maximum probability differences across repeats were **0.0** for both
contexts. There were no binary or three-way verdict flips. Drex returned
69 `correct`, 23 `incorrect` and eight `partial` verdicts per J0 repeat;
E1 returned 68, 24 and eight. Human binary labels do not establish which
three-way rejecting verdict is correct.

| Binary acceptance calibration | J0 | E1 |
| --- | ---: | ---: |
| Brier score from P(correct) | 0.112135 | 0.101473 |
| Log loss, clipped at 1e-15 | 0.429060 | 0.374663 |
| Equal-width ECE, ten bins | 0.104465 | 0.098751 |

These are descriptive calibration diagnostics on reused cases, with no
temperature or threshold fitted to Human100.

## Timing and host observations

On this Apple M5 Max with 128 GiB memory and macOS 26.7.2, observed median/p95
prepared-input scoring latency was **0.753/1.183 seconds for J0** and
**5.141/6.338 seconds for E1**. Tokenization and model loading are excluded;
native scoring, readout, upstream response formatting and private pipe
exchange are included. Input lengths were 320–488 tokens in J0 (median
385.5) and 2348–4308 in E1 (median 2828). All 200 unique prompts fit the
16,384-token limit without truncation.

**Timing is unqualified for comparative speed claims.** The sampled
background CPU screen exceeded its threshold, reaching 978.7% for a process;
the whole-host counters recorded 4,016 pageouts and 13,037,548 swapouts.
These counters cannot attribute activity to a particular process. There
were 2,108 host samples, zero sampling failures, no detected active background
model above the model-CPU threshold and no sleep events. Physical GPU
exclusivity, clocks and continuous thermal/power state were not established.
Historical competitor latency is not a fresh matched speed baseline.

Sampled process-tree RSS peaked at 14,988,443,648 bytes. This sums the harness
and descendants every 50 ms; shared pages can be counted twice and this is
not an exact Metal allocation or minimum-memory requirement.

## Frozen instrument and provenance

The [registration](2026-11-08-drex-human100-registration.md) and complete
harness/auditor were committed as `4730f1f45ba785d6e17923cd787df8980a5fb40c`
before any Human100 forward pass. The worker uses the upstream Drex custom
MLX noncausal scorer and its embedded readout temperature, with affine-Q4
group-64 weights and mixed BF16 controls/F32 readout. Source is
`d2fd26cdf211f87b9a6b1261995528629c9e1401`; Q4 snapshot is
`23f21ac6e664d3edb32faa7b55b0c8a7fb09134a`. This is Drex Max v1, the same
artifact used in the earlier JevBench comparison.

All five weight/readout/config/tokenizer hashes match the earlier committed
benchmark manifest and were verified again after this run. The locked
environment uses Python 3.13.13, MLX 0.32.4, MLX-LM 0.32.2 and Transformers
5.22.0. Transformers emits a generic Mistral-regex warning for this local
configuration's absent version metadata; source review confirmed that its
default warning branch does not modify the pinned Qwen pretokenizer. No
tokenizer flag or model artifact was changed.

The loader verifies the original SubQ Snap seal, completion/verdict hashes,
all protected input hashes and all 400 referenced wire-file hashes. It reuses
the exact `state` and single `verdict` question from those snapshots, removing
only the provider model-identity field for local inference. Independent
source review also recomputed all 400 canonical and ordered request hashes
against the original instrument, normalizing only the historical model alias
as Brekko's registered request-hash recipe specifies. There are 200 distinct
requests and 400 judgments in the original order. Every forward uses fresh
state without output or prefix caching; noncausal attention blocks ordinary
causal prefix restoration.

The reused comparator is **SirSahOl/JevK5-chat-mlx-16bit**, model revision
`39e89a917c5a20e819af0b8ff8bd9f944eedd9c6`, evaluated by the historical
BF16 bounded-continuation MLX recipe. Its complete verdict artifact SHA-256
is `874aa7a59d98afbeef71f317f43f657f5ff09741ed14b675639cc1510d23c0f7`.
It is distinct from converted JevK5-4B's 89%/88% historical result and the
native Stark artifact used in JevBench. No competitor was rerun. Differences
in model size, precision, prompt rendering and runtime remain part of this
comparison; it does not isolate any one architectural mechanism.

## Evidence and checks

- [Preflight seal](data/drex-human100-preflight-2026-11-08.json)
- [Complete aggregate receipt](data/drex-human100-2026-11-08.json)
- [Independent analysis/audit](data/drex-human100-audit-2026-11-08.json)
- [Harness and reproduction instructions](../../scripts/drex_jevk5/README.md)

All 400 native judgments were validated and checkpointed as aggregate confusion
counts. The separate audit process received per-request readouts through a
private pipe, reverified protected input/reference and comparator hashes and
all 400 record identities/distributions, independently recomputed the confusion
counts, and computed calibration, intervals and paired differences. Parent and
auditor confusion counts agree exactly. Later source review checked the formulas
and arithmetic; private readouts were not retained for another postflight
numerical reconstruction. Twenty-four model-free tests, lint, formatting
and repository policy checks pass; independent read-only review verified the
instrument and final arithmetic.

New run metadata stays under the original tenant/corpus research scope.
Public artifacts contain aggregate measurements and hashes only. Prompts,
source passages, answers, token vectors and native decision probabilities
were not persisted or published. No model, backend, provider, threshold or
production default was changed. This adds Human100 evidence; the earlier
JevBench results and source-bound records remain unchanged.
