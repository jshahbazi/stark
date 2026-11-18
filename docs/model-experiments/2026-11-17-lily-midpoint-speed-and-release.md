# Lily midpoint prefix speed and release record

**20261117-FLASHSPEED-11–16 · November 17, 2026**

## Decision

Enable 2,048-token midpoint prefix snapshots by default only for the retained-vision `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` generation artifact at revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83` and artifact SHA-256 `18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4`.

The optimization still requires `--prefix-cache` and the worker's existing cache-scope, loaded-model, context-length, MTP and image invalidation checks. `STARK_LILY_PREFIX_MIDPOINT=0` restores the stock 4,096-token boundary grid for this artifact. Other Lily and mixed-affine artifacts remain unchanged.

## Baseline

The stock Lily generation prefix cache retained the start of a request's final 4,096-token physical prefill chunk. At a 12,245-token prompt, the retained position was 8,192 tokens. A following append-only request therefore prefetched the remaining 4,053 tokens plus its newly appended suffix.

The original midpoint patch advertised a 2,048-token grid while retaining 4,096-token GPU passes. It could capture at 10,240, but rejected restoration when fewer than 128 suffix tokens remained because those shapes selected different GDN or small-matrix arithmetic.

## Changes

The release path:

- Keeps physical Lily prefill chunks at 4,096 tokens and captures one midpoint state within a chunk.
- Captures the F32 GDN state directly from the recurrent kernel at the selected 64-row boundary.
- Fuses the GDN and PLE convolution-window rollback into the corresponding convolution kernels, removing separate capture dispatches per layer.
- Forces the original 64-row chunked GDN route for restored 64–127-row tails.
- Allows midpoint continuation with 17 or more suffix tokens; the 17-token floor avoids the 1–16-row dense small-m projection route selected by the containing 4,096-token pass.
- Allocates output-only snapshot buffers whose complete payload is overwritten before read, avoiding redundant 155,762,688-byte host zeroing.
- Adds aggregate request-profile counters for cache attempt, restore and capture attempt/storage.
- Keeps one in-memory snapshot and never persists or logs it.

The capture payload remains the same recurrent state required for exact restore: per-GDN F32 recurrence and BF16 convolution window, PLE history and window, and the draft head's last hidden row when MTP is loaded. Attention, indexer and draft-head prefix caches remain in the loaded model instance.

## Correctness evidence

All following comparisons used the installed retained-vision artifact.

### Full model

Twenty capture/restore comparisons matched fresh execution:

- all 248,320 prompt logits by raw F32 bits;
- every draft and target logit digest;
- every emitted token;
- two-draft MTP rounds, proposed, accepted and verified-row work.

The cases covered 8,192-, 12,288- and 32,768-token source fixtures, checkpoints at 2,048, 6,144, 10,240 and 30,720, and suffixes of 17, 64, 127, 128, 293 and 2,469 tokens. The added 12K case restored from 10,240 with 17 suffix tokens.

### Metal components

Exact physical-Metal tests covered GDN capture and continuation at suffix lengths 1, 17, 63, 64, 65, 127 and 128. The fused GDN and PLE capture kernels matched ordinary convolution output, terminal window state and the pre-existing standalone rollback bytes.

### Lifecycle transitions

One 40.4-second installed-model lifecycle run passed:

- restoring a 2,048 snapshot and replacing it with a captured 6,144 snapshot;
- cancellation after the capture-bearing command, with no published snapshot and a healthy next request;
- MTP-off restore with exact prompt logits and next greedy token;
- a real PNG image transition with a full 248,320-logit target readout, stale text-snapshot rejection, clean reset and exact subsequent fresh text behavior.

## Performance evidence

### Representative append-only 12K usage

Three paired requests per arm used the same 12,245-token source prompt. The comparison was midpoint mode versus the stock 4,096-token grid in the same worker binary.

| Appended tokens | Stock reused | Stock prefill | Midpoint reused | Midpoint prefill | Follow-up improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 | 8,192 | 4,309 | 10,240 | 2,261 | 2.341 s to 1.262 s, 46.1% |
| 1,024 | 8,192 | 5,077 | 10,240 | 3,029 | 2.903 s to 1.737 s, 40.1% |

Output hashes and MTP work matched. The combined fresh-capture plus follow-up prefill cycle improved 19.4% in both cases.

These are first-token prefill results, not decode throughput measurements.

### Short-suffix stress case

A 10,257-token prompt following a 12,245-token capture reused 10,240 tokens and processed 17 suffix tokens. Median first-token prefill fell from 4.953 seconds with the original midpoint guard to 0.112 seconds with the new continuation path, a 97.7% reduction. This is an edge case and is not the normal-usage estimate because ordinary chat follow-ups append rather than shortening a prompt.

### Allocation attribution

A synthetic single-allocation probe for the 155,762,688-byte snapshot payload measured 7.034 ms median zeroing overhead versus 0.016 ms output-only allocation. This micro-result explains one small source of capture cost; it is not an end-to-end latency claim.

## Interpretation

At around 12K tokens, the practical benefit is approximately:

- 40–46% faster cached follow-up first-token time for 256–1,024 appended tokens;
- 19.4% faster for a fresh capture plus follow-up cycle;
- unchanged decode speed;
- no extra benefit on a cache miss, scope/context/model/MTP mismatch, or image request.

The improvement comes primarily from avoiding 2,048 additional suffix tokens at cache hit time, not from changing model arithmetic or increasing physical chunk size.

## Release boundary

This evidence promotes only the exact retained-vision Lily Q4-XL artifact identified above. It does not qualify:

- another revision, quantization, conversion or source repository;
- an artifact lacking the loaded vision tower or the compatible two-draft head;
- other mixed-affine layouts or model families;
- a default-off or altered MTP policy.

Operator rollback requires only `STARK_LILY_PREFIX_MIDPOINT=0`; the worker then retains its stock 4,096-token prefix grid.

Release readiness was checked on the installed artifact with the same `--vision-staging-root` and `--prefix-cache` path used by the image-loaded worker: no environment override advertised 2,048 tokens, while `STARK_LILY_PREFIX_MIDPOINT=0` advertised 4,096 tokens. The release worker SHA-256 was `0f42dbfc04585f57660aa07b56bdfa2b43ee1ed675650b65f7c7c71e2f6cbfbc`; the source patch SHA-256 at that check was `577941414604be2b73691101a3f7f2e19f7f6fc7b44952df121da1d42e8e2fa6`.

## Validation records and privacy

The implementation passed:

- 16 prefix-cache worker tests;
- 97 Lily core unit tests;
- the two physical midpoint Metal tests;
- the installed-model full-logit/MTP suite;
- all four installed-model lifecycle transitions;
- formatting and diff checks;
- the locked Rust 1.98.1 release worker build.

The private token fixture and prompt data were not committed or published. Aggregate correctness, timings and executable/fixture hashes are recorded in [the integration record](2026-11-17-lily-midpoint-integration.md). Upstream attribution and patch provenance are recorded in [the vendored Lily provenance note](../../vendor/lily-core/UPSTREAM.md).
