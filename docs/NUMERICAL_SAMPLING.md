# Native hybrid sampling

Native hybrid generation computes stochastic draws on the CPU. The nucleus
shortcut is enabled on Qwen3.8's Lily sampled-decode path, for native-source
and converted Uzu artifacts. Other hybrid models retain the reference sampler.
Greedy, top-k,
min-p and requests outside the nucleus shortcut's eligibility use the reference
sampler. The reference sorts by descending F32 total order with ascending token
IDs for ties, including distinct signed-zero ordering.

Both reference and nucleus paths use the same scalar exp function. Logit
subtraction and division remain F32; the rounded result converts to F64 before
exp. Probability sums, filter comparisons and CDF accumulation use F64.
SplitMix64 advances once for a successful stochastic draw. Greedy and rejected
nonfinite readouts preserve their existing RNG behavior.

## Proving a small nucleus

The shortcut accepts a nonempty finite vocabulary of at most 1,000,000 entries,
finite positive temperature, no top-k, finite top-p below 1, and no min-p. Worker
request validation applies its existing limits. Internal calls with top-p at or
below zero preserve the reference's first-candidate behavior.

It selects the exact first 64 candidates using the existing total-order
comparison and ID tie break. The bounded insertion budget is 1,024 replacements.
Exceeding that budget enters the reference sampler before mapping probabilities.
This avoids a full-vocabulary allocation for prefix selection on adversarial
input orders.

The shortcut maps the same rounded exp terms in token-ID order and brackets the
reference total. For identical nonnegative terms with exact sum `S`, recursive
summation satisfies `|computed - S| <= gamma_(N-1) * S`, where
`gamma_k = k*u / (1-k*u)` and `u = EPS/2`. See Nicholas J. Higham,
[*The Accuracy of Floating Point Summation*, section 2, equation (2.6)](https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf),
SIAM Journal on Scientific Computing 14(4), 783–799 (1993),
[DOI 10.1137/0914050](https://doi.org/10.1137/0914050).

Applying this bound to the sorted and ID orders gives the following conservative
relation, using `a = N*u` and the computed ID-order sum `s`:

```text
abs(sorted_total - s) <= 2*a/(1-2*a) * s
```

For `N <= 1,000,000`, the implementation uses a larger error with outward
rounding:

```text
error = next_up(4 * N * EPS * s)
lower_total = next_down(s - error)
upper_total = next_up(s + error)
```

The exact maximum produces a term equal to 1, and all terms lie in `[0,1]`, so
the total is finite and at least 1. The argument assumes round-to-nearest-even
and gradual IEEE underflow. A subnormal result of nonnegative binary64 addition
is exact on the subnormal grid; other additions satisfy the relative-error
model. Exp underflow changes neither the premise nor the ordering because both
paths use the identical rounded terms.

For a bit-identical ordered reference prefix sum `C`, positive-denominator division
is monotone. Outward division brackets the rounded reference comparison:

```text
lower_ratio = next_down(C / upper_total)
upper_ratio = next_up(C / lower_total)
```

The shortcut stops when `lower_ratio >= top_p` and continues when
`upper_ratio < top_p`. An overlap is uncertain. Each test precedes adding the
next candidate, matching the reference. A proved prefix retains the reference
probability bits, cumulative sum and CDF order, so it uses the same random draw
and selects the same token.

## Exact fallbacks and scratch lifetime

An uncertain threshold or a nucleus larger than 64 uses the existing complete
ordering. It reuses the already-computed ID-order probabilities, then recomputes
the total, filter and CDF in the exact reference order. It does not repeat exp.
Proof and fallback selection have no RNG argument; only the final CDF draw
advances the stream.

The worker retains only a prefix vector and a full-vocabulary F64 probability
vector. The full fallback order is local and drops after the draw. The prefix
has at most 64 IDs, including on selector-budget fallback. At a vocabulary of
248,320 entries, the probability payload is 1,986,560 bytes, plus the prefix and
allocation overhead. Full fallback ordering adds temporary storage through the
existing selector.

Vector lengths clear after draws, errors and request boundaries. Capacity reuse
stores no semantic cache. Ineligible requests enter the reference sampler
before shortcut profile scopes, probability allocation or an extra finite scan.

The numerical tests compare the previous exp expression, shared scalar helper,
ID-order map, reference sorted totals, exact kept-prefix sums, sampled tokens
and RNG states. They cover underflow, signed zeros, ties, threshold boundaries,
adversarial selection, both reuse fallbacks, unsupported settings and failures.
Performance measurements belong in the dated benchmark records and must be
repeated when the compiled sampler changes.
