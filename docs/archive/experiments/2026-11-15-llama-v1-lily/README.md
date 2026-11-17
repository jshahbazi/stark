# Historical Lily release-baseline runner

`benchmark_lily_release_baseline_v2.py` is the immutable as-run source for the
[llama.cpp assessment](../../../benchmarks/2026-11-15-llama-v1-flashnext-lily-assessment.md).
Its original hash and results remain unchanged.

Privacy correction, 2026-11-16: this archived runner copied its source-code
corpus into the ignored runtime output as `frozen-corpus.rs`. Its aggregate-only
docstring therefore overstates the absence of retained input material. No
corpus text or generated model content was published. The current CLI uses a
RAM-only snapshot and has model-free privacy regression controls, recorded in
the [PR review](../../runs/2026-11-16-lily-projection-pr-review.md).
This source is an execution record, not the current benchmark entry point.
