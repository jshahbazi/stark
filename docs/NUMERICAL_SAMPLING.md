# Numerical sampling

Converted Qwen3.8 models execute through the standard Uzu `LanguageModel`
and backend sampling implementation. The decoder returns F32 vocabulary
logits, and its sampling instance uses that same data type. The former
Lily-scoped CPU nucleus shortcut is removed from production. Historical
measurements remain in the
[2026-10-30 speed study](benchmarks/QWEN38_UZU_SPEED_2026_10_30.md).

## Native hybrid CPU sampler

The remaining native hybrid families use `stark-generate`'s CPU sampler.
Stochastic candidate order is descending F32 total order with ascending token
IDs for ties, including distinct signed-zero ordering. Greedy selects the
first maximum by token ID.

Logit subtraction and temperature division remain F32. The rounded value
converts to F64 before exp. Probability sums, filter comparisons and CDF
accumulation use F64 in candidate order. SplitMix64 advances once for a
successful stochastic draw. Greedy and rejected nonfinite readouts do not
advance the random stream.

Top-k selection sorts only the retained prefix. Full-vocabulary sampling uses
stable radix ordering. At most 64 candidates use bounded insertion selection,
with the existing partial-selection fallback when its replacement budget is
exhausted. Large probability maps may use a worker-owned pool of at most four
threads; mapping finishes before filtering or drawing.

## Validation scope

The generic CPU sampler retains its differential ordering, probability and
seeded-draw controls. Qwen3.8 forward-pass and sampler qualification belongs
to the [standard execution plan](QWEN38_UZU_EXECUTION_PLAN.md) and its recorded
runs. Different sampler implementations can use different arithmetic and
random streams; forward-pass comparisons use matched readouts and numerical
controls rather than assuming identical sampled token sequences.
