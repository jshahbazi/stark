# Mercury Decide, JevK5 and Drex Max on JevBench

The complete fresh Mercury Decide evaluation scored **207/231 public
JevBench cases (89.61%)**, with **11/11 supported synthetic controls**.
All 242 requests succeeded in one sequential pass, with no rate limits,
failures or retries. The [complete receipt](data/mercury-jevbench-2026-11-08/complete.json)
contains the source hashes, coverage, usage and metrics.

| Public JevBench, 231 cases | Correct | Accuracy | Brier sum | ECE, ten bins |
| --- | ---: | ---: | ---: | ---: |
| Drex Max v1 Q4 / upstream MLX | 212 | 91.77% | 0.120900 | 0.043681 |
| Mercury Decide / hosted Inception API | 207 | 89.61% | 0.162322 | 0.060516 |
| JevK5 BF16 / Stark | 202 | 87.45% | 0.184880 | 0.029446 |

Mercury answered five more public questions correctly than JevK5 and five
fewer than Drex. These are descriptive differences on a reused public
development corpus. No paired significance test between Mercury and the other
models was calculated. The models differ in size, precision, prompt,
attention, readout and runtime; this does not isolate an engine advantage or
establish accuracy or calibration on a fresh workload.

The local results come from the earlier
[complete Drex/JevK5 comparison](2026-11-08-drex-jevk5-comparison.md). A local
JevK5 refresh during this Mercury investigation also scored 202/231 public
cases. Its worker and weight hashes match the earlier comparison.

## Coverage and method

The sealed corpus contains **all 231 public JevBench cases** and 12 synthetic
controls. Mercury's [documented score limit](https://docs.inceptionlabs.ai/capabilities/decisions)
is ten levels. One synthetic 16-level score control is unsupported and
explicitly excluded. No public case was excluded. The original synthetic
question was preserved without adapting it to another question type. Mercury
answered all remaining 11 controls correctly; the local models also answered
that supported subset correctly in their earlier 12-control runs.

The run used [`POST /v1/decisions`](https://docs.inceptionlabs.ai/api-reference/decisions/create-a-decision)
with hosted alias `mercury-decide`, one named question per request, original
state/instructions/criteria order, no local output cache and a one-second
minimum dispatch interval. The hosted API provides no immutable weight digest.
Provider-side caching and infrastructure were not controlled. The dataset
SHA-256 is `acf56888f47486fd8eba45d17a46372d65d911e2f57bc390474e7b41f37aae58`.

All three answer types were validated for named question, served model,
option coverage, normalization, finite probabilities, native choice argmax,
score weighted mean and token usage. The existing canonical boolean aliases
and shuffled-choice mapping were used. The final metric helper stringifies
both labels and expected answers before evaluating correctness.

## HTTP latency and usage

Successful HTTP calls had **312.6 ms median** and **478.2 ms p95** latency;
their summed elapsed time was 79.59 seconds. Pacing intervals are excluded.
These measurements include connection setup, network and hosted provider
work, so they are not directly comparable to the local prepared-input scorer
timings. The earlier Drex timing measurements also failed host screens.
No qualified cross-engine speed ratio is claimed.

The API reported **152,625 input tokens** and **242 output tokens**. At the
[published pricing](https://docs.inceptionlabs.ai/get-started/models) checked
2026-11-08 ($0.02 per million input tokens and free output), estimated input
charges for this complete pass are **$0.0030525**. This is arithmetic from
reported usage, not a settled invoice or a total for earlier attempts.

## Recovery and validation

The initial rate-limited attempt lost its in-memory quality readouts. The
subsequent unanswered-only completion could not establish a full score. The
owner then explicitly requested a fresh complete evaluation; the primary
result above uses only that successful full pass. Earlier attempts remain
separate in the [historical failure and coverage record](2026-11-08-mercury-jevbench-attempt.md).

The raw receipt's auxiliary `accuracy_progress` counter undercounted numeric
score answers: it compared string labels directly with integer expected
values. It reports 193 public and ten synthetic correct, while the
authoritative final `quality` fields correctly normalize both sides and
report **207 and 11**. Independent review confirmed this exact code path and
the label types. The original receipt is preserved unchanged; the progress
comparison is corrected separately with a regression test. The final metric
calculation is unchanged by that correction.

The [harness and setup](../../scripts/drex_jevk5/README.md) use locked Python
dependencies. The primary command selects one complete round and reuses the
existing local baseline comparison:

```sh
uv run --project scripts/drex_jevk5 --locked python scripts/drex_jevk5/mercury.py \
  --config runtime/drex-jevk5-config.json --output runtime/mercury-complete.json \
  --rounds 1 --skip-local-baseline
```

Supply `INCEPTION_API_KEY` in the process environment or an explicit local
key file. Successful requests are never automatically retried within a run;
only HTTP 429 can receive a bounded retry after backoff, retaining earlier
answers in memory. The completed run needed none. Aggregate progress is
saved after each successful answer. Existing output paths are refused.
Credentials, prompts, responses, token vectors and individual decision
probabilities are not logged or published.
