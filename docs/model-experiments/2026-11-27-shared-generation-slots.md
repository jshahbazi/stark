# GEN-SERVE-01 · Shared desktop text slots beyond Hy4

This change builds on the [Hy4 STQ four-slot default](2026-11-27-hy4-stq-four-slots.md).
Desktop uses one admission queue and one group of request, response and
cancellation pipes for all generation backends. A backend opens the slots it
can own safely. Each auxiliary serves text only, retains request state only
while it runs, and reports its context capacity. The primary remains the
ordinary path for solo requests, images and consecutive prefix reuse.

Native language and vision-language decoders share the existing loaded
parameters, decoder and Metal queue. Each request allocates its own KV and
recurrent state, with a live group reservation over outstanding requests.
Finite and undeclared context planning stay available. If the required state
exceeds live headroom, the request fails before native state allocation or
growth. The per-token decode loop is unchanged. A completed primary prefix
can be reused again when subsequent requests stay on the primary without
intervening auxiliary activity.

Flint and Ripple share the mapped weight inventory and give each slot its own
request graph or cache. Their fitter accounts for lazy weight residency once
before partitioning the available context budget. The API uses the common
capacity of the slots the backend admits. Small CPU fixtures verify that
interleaved lanes produce the same values as an independent reference and
that a reset or failure on one leaves the other usable. Full Metal checkpoint
validation remains open because the local model store lacks full copies.

## Coverage and remaining backend work

| Backend | Slot support in this change |
| --- | --- |
| Hy4 Sushi STQ | Existing shared-weight path and fixed per-slot contexts, with compatibility for the legacy pipe protocol. |
| Native language and vision-language decoders | Shared loaded decoders, independent native sequence state and per-request reservation. Image work uses the primary. |
| Flint and Ripple | Shared mapped weights and independently sized caches. Synthetic CPU isolation and Metal compile coverage. |
| Native GLM and generic Qwen hybrid | Current paths remain serial. GLM needs one-time expert residency setup and independent recurrent and attention state. Generic Qwen needs to separate layer parameters from mutable GDN, KV and index state. |
| Lily | Serial in this rollout. See the [Lily follow-up](2026-11-27-lily-shared-generation-slots.md) for shared resident weights and independent request slots. |
| GLM EXL3, Inkling, and Hy4 AngelSlim GGUF | Current paths remain serial. These loaders bind mutable execution state to loaded runtime owners. GLM EXL3 also claims a single thread-affine bridge model per process. |
| Decision | Continues to serialize each complete decision against the loaded model. |

## Measurement

For native Qwen3 4B MLX 4-bit, start the same release worker through the
Desktop API with one slot and with the default. Use four distinct synthetic
prompts and compare each overlapping response with its own fresh serial
reference. Measure solo decode on the primary while auxiliary pipes are
idle. Preserve the request output budget and compare effective context and
toolchain within the timing bracket. For image-loaded inference, qualify
the cached original Qwen3.5-0.8B artifact after offline conversion.

The registered 4B artifact's weights SHA-256 was
`7c2af9a45e5b0d196f06a5cbfd6fc5cc95adf29df79885d0540f2206fb3d40d8`, and
its config SHA-256 was
`5b505594bf00ef9682b373c6a292b88f94f1e1ceab449784c12accb835806c7f`.

Initial Qwen3 HTTP runs admitted one and four slots at the same 40,960-token
capacity. All four overlapping requests used distinct slots and matched
serial output hashes. The preliminary one-to-four timing reported median
solo rates of 116.25 and 96.92 tok/s across three requests per arm while
compiler work overlapped the run. That bracket establishes output parity.

The final Rust 1.98.1 release worker (`metal,hy4`) used SHA-256
`75c93b3ada818eb8f2509548f7acdb712b069b5ac355550e61989f35f02bd057`.
On the M5 Max, four separate loads alternated one, four, four and one
enabled slots. Each load warmed once, then ran three 96-output solo
requests. Both configurations advertised 40,960 tokens. Median solo
useful decode was **169.00 tok/s** with one slot and **174.67 tok/s** with
four enabled. Per-load medians ranged from 162.70 to 175.57 tok/s. The
run did not show a solo decode regression at this precision. Each
four-request overlap matched the serial response hashes and peaked at
four active lanes, finishing in 3.71 and 4.00 seconds respectively.

The cached Qwen3.5-0.8B publisher snapshot at
`2fc06364715b967f1860aea9cf38778875588b17` converted offline to the native
BF16 vision-language layout. Its merged weight digest was
`836eaf8b6b21d3a96f41e2ab1391de63f69fe2e4fa4c858ccf2fb5fe0b55f9d2`.
With the image tower loaded, four distinct concurrent text replies
matched their serial references. An image queued behind those requests
ran its prepare, generate and discard transaction on the primary. Its
reply and the following fresh text reply matched the serial controls.
MTP was off in this bracket. The initial comparison failed because the
HTTP usage comparator treated an auxiliary's omitted cache field as
different from an explicit zero on the primary. The corrected validation
normalized this single zero field and kept exact output hash, finish
reason, token counts and freshness checks.

Validation also passed 190 release generator tests with six ignored,
510 broader Python chat and desktop cases with one skipped, and 143 API,
media and Decision cases. The final focused host path passed 21 cases.
Flint and Ripple passed independent CPU output/reset fixtures and the
hybrid worker's budget ladder test. Other available model artifacts each
need their own slot qualification.

| Private receipt | SHA-256 |
| --- | --- |
| Qwen3 alternating one/four-slot bracket | `3306ec3ea7b8fa989cb26790a648938c35a6022df9857d1c14f1e0000fe40ff3` |
| Qwen3.5 four-slot text and image lifecycle | `e869e4e474eec60e854d8e30605aff080da39cad50b327f3c2db6fb4596d32b8` |
