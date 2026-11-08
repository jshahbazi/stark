# Mercury Decide on JevBench: coverage completed, full score unavailable

This historical record covers the initial failed and partial attempts only.
The later [fresh complete evaluation](2026-11-08-mercury-jevbench-comparison.md)
successfully scored all 231 public cases and supersedes this incomplete status.

Mercury Decide returned valid responses across all **231 public JevBench
questions** and **11 supported synthetic controls** on 2026-11-08. A complete
corpus accuracy, Brier score or calibration result cannot be recovered from
this attempt: the initial process retained readouts only in memory and lost
them when an HTTP 429 terminated it. No full Mercury score or ranking against
JevK5 or Drex is claimed.

After the owner requested only unanswered questions, the overlapping fresh
pass was stopped and **34 previously unanswered questions** were sent once
each. All 34 succeeded. That remaining subset scored **21/24 public cases**
and **10/10 synthetic controls**. These selected tail cases are not a
representative substitute for the full 231-case benchmark.

## Coverage and call accounting

The sealed dataset is the same one used in the
[Drex/JevK5 comparison](2026-11-08-drex-jevk5-comparison.md), with SHA-256
`acf56888f47486fd8eba45d17a46372d65d911e2f57bc390474e7b41f37aae58`.
It contains 231 public cases and 12 synthetic controls. Mercury's
[official decision schema](https://docs.inceptionlabs.ai/capabilities/decisions)
supports at most ten score levels. One synthetic 16-level score control was
excluded with an explicit coverage record. Its question was not rewritten.
All public cases are supported.

| Stage | Requests started | Recorded valid responses | Outcome |
| --- | ---: | ---: | --- |
| Initial API admission probe | 4 | 4 | All question types and largest choice accepted |
| Initial full pass | 208 | 207 | HTTP 429; no full score retained |
| Paced API admission probe | 4 | 4 | Passed |
| Overlapping fresh pass | 28 | 27 | Stopped after owner requested unanswered questions only; one in-flight outcome unknown |
| Unanswered-only completion | 34 | 34 | Completed without failures |
| Total | 278 | 276 | 242 unique supported questions have a recorded valid response across stages |

The first 207 successful responses in the initial sequential pass establish
its completed prefix. The admission probes add one distinct later control,
bringing prior unique coverage to 208 questions. The interrupted fresh pass
covered questions already answered in that prefix. Recovery excluded the
successful prefix and successful probe identities, leaving 24 public questions
and ten controls. It sent exactly those 34 requests once, with a one-second
minimum interval and no retries. No previously answered question was resent
by the unanswered-only completion.

Recovery checks the exact serialized request-population SHA-256 of the
failed run and both probes before deriving the remaining set. The probes have
identical request sets; duplicate probe receipts do not increase unique
coverage. The pending request in the interrupted overlapping pass was already
inside the earlier successful prefix, so its unknown outcome does not affect
unique coverage.

Each stage's aggregate receipt is preserved separately:

- [Initial probe](data/mercury-jevbench-2026-11-08/probe.json)
- [Rate-limited full attempt](data/mercury-jevbench-2026-11-08/rate-limited.json)
- [Paced probe](data/mercury-jevbench-2026-11-08/paced-probe.json)
- [Interrupted overlapping pass](data/mercury-jevbench-2026-11-08/interrupted-overlapping-pass.json)
- [Unanswered-only completion](data/mercury-jevbench-2026-11-08/unanswered-only.json)

## Retained subset results

| Remaining subset | Correct | Brier sum | ECE, ten bins |
| --- | ---: | ---: | ---: |
| 24 public cases | 21 | 0.161438 | 0.062950 |
| Ten synthetic controls | 10 | 0.000004802 | 0.000956 |

The final 34 requests reported 20,446 input and 34 output tokens. The
[published pricing](https://docs.inceptionlabs.ai/get-started/models) checked
2026-11-08 is $0.02 per million input tokens and free output, giving an
estimated input charge of $0.00040892 for this final stage. This is usage
arithmetic, not a settled invoice. Initial successful usage was also lost on
failure, so a complete token or charge total is unavailable. The interrupted
in-flight request can have an unknown charge.

The refreshed local JevK5 baseline again scored 202/231 public cases, using the
same BF16 checkpoint revision `c4f7fdb3aeab5582336406e78d3bef11bf98833d` and
worker SHA-256 `5dcc03fe336ca9f3c265dd6d6516227529d7e74c6ced9c23cdb830041a72ee79`
as the earlier comparison. Its retained aggregate does not recover Mercury's
missing answers. Mercury HTTP latency, including network and provider time,
would also differ from the earlier prepared-input local scoring measurement.
No comparative latency claim is made here.

## Method, limitation and verification

Requests used the official
[`POST https://api.inceptionlabs.ai/v1/decisions`](https://docs.inceptionlabs.ai/api-reference/decisions/create-a-decision)
and hosted alias `mercury-decide`, without an immutable weight digest.
Original state, instructions and choice order were preserved. Responses were
checked for served model, exact named answer, native type, option coverage,
finite normalized probabilities, choice argmax, score weighted mean and usage.
The existing canonical boolean and shuffled-choice mappings were retained.

The first harness saved successful-request counts before moving to the next
question but deferred quality aggregation until a complete pass. This was a
failure-recovery defect: successful response coverage survived, while their
quality did not. The completion tool checkpoints cumulative correct counts
and usage after each response without saving individual readouts. A successful
coverage ledger cannot reconstruct the earlier missing accuracy.

The owner explicitly requested avoiding answered-question replays. No further
full pass was performed after that instruction. Existing answered cases were
excluded using the hash-validated coverage ledger. A future complete quality
evaluation would need a separately requested fresh run or retrieval of the
original responses through a supported provider mechanism.

The dedicated research tools and setup are described in the
[harness README](../../scripts/drex_jevk5/README.md). Twenty model-free tests
pass, covering all hosted answer types, invalid responses, the provider's
score limit, case-order alignment, and excluding answered prefixes/probes.
Lint and format checks pass. The original Drex harness and source-bound
results remain unchanged. No new inference backend or product integration was
added. Credentials, prompts, responses, token vectors and individual decision
probabilities are not logged or published; output artifacts contain aggregate
measurements and hashes only.
