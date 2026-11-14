# Image decoder budget follow-up

Date: 2026-11-08

## Source and scope

Baseline: `a2670011203009b9b72fc8dc5c61b1ead80d143c`, PR #99.

Runtime-source manifest SHA-256: `52b42de048f327f65b2878edc887c26ab4d7ab51d7a710c5455fbde55c5f02dd` (1,075 files), using the selection and binary-hash encoding from the [initial PR review](2026-11-08-context-budget-pr-review.md).

The existing image qualification memory check uses a short 2,048-token decoder window. It does not admit the user request's complete output span. Native Qwen/Gemma vision variants and retained hybrid image generation now check decoder allocation using the actual expanded prepared-token count and requested output before state creation/prefill.

Sparse physical growth and selected virtual cache capacity are checked separately. The latter includes ring/full cache geometry, suffix capacity, dtype, device buffer size and shader index range. This also protects short explicit text/scoring requests whose selected sparse address reservation is larger than their active span. Dense MTP predictor context bounds remain enforced.

Image preparation context binding, encoder execution, precision and qualified row bounds are unchanged. Image Auto can reject a bound context that does not fit instead of resizing that prepared transaction. The decoder check does not provide a complete encoder peak estimate.

## Validation

- Pinned-nightly CPU/grammar engine suite: 555 passed, 30 ignored; seven C++ string tests, two profile tests and three doctests passed.
- Generation worker: 133 passed, two ignored. Resource-budget tests: five passed. Default and optimized release scorer suites: 65 passed each.
- Image transaction and undeclared-context Python tests: 45 passed, including buffered and streaming memory rejection, prepared-state discard, same-worker image retry and subsequent text tokenization.
- Release generator/scorer builds, release Clippy for generator and both scorer configurations, scoped formatting, Python lint, repository policy and diff checks passed.

On physical Apple silicon, rebuilt dense and hybrid scorers each completed two exact-fill one-token requests in the same worker without truncation. The rebuilt synthetic undeclared-context HTTP smoke again returned two 200 responses, kept capacity undeclared and retained the same worker/PID. Its selected allowance and EOS-only result match the earlier record. Allowances were not filled. No prompts, responses, token vectors or probabilities were recorded.

| Artifact | SHA-256 |
| --- | --- |
| Release generator | `edb956b4795ee8fc6b45324e13f4bcb4a78245925dd59d6b0957849eea6aed3e` |
| Release default scorer | `0674dec40502ff1581159d797060beb966b240bc728ef1bc2ba01002f8f61145` |

Model artifact hashes remain those in the preceding validation records. The most recent full converter run before this Rust-only guard and its two added Python cases passed 6,699 tests with 96 skipped.

## Limits

No converted image-capable artifact at or below 15 GiB was available in the bounded local inventory, so this follow-up's image checks are model-free protocol and allocation-inventory checks. No image model was downloaded or converted. Real-image hardware qualification and full-window stress/performance are not established by this run. Admission estimates current headroom, without a reservation or hard RSS ceiling. Earlier source-bound results remain separate; the installed app was not updated.
