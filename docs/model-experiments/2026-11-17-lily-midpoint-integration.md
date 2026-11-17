# Optional Lily midpoint prefix capture

**20261117-FLASHSPEED-11–14 · November 17, 2026**

The generation worker accepts `STARK_LILY_PREFIX_MIDPOINT=1` for the admitted retained-vision `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` artifact at revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83` and artifact digest `18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4`. This adds the midpoint of each physical 4,096-token chunk as an eligible snapshot position. The worker keeps one snapshot. It uses fresh prefill when a midpoint restore would leave fewer than 128 suffix tokens. Existing cache scope, token, context and MTP checks still apply. The option is off by default.

The experiment build passed 12 full-model capture and restore comparisons with exact prompt, draft and target logits. All 18 fresh comparison pairs and all 94 paired native Pi requests matched emitted tokens and two-draft MTP work. In 16 Pi sessions, median paired turn time fell 15.34% and the midpoint arm won all eight matched session pairs. First-token time fell 55.32% at the median on the 50 native requests that reused an additional 2,048 tokens. The same four of eight sessions per arm passed the full grader. [Source-bound exactness and A/B results](https://github.com/jshahbazi/stark/blob/b11719993c113409bae88107b3f666b77bbe626c/docs/model-experiments/2026-11-17-flashnext-midpoint.md)

Fresh first-token medians increased by 2.63%, 5.51% and 4.98% at 8K, 12K and 32K. The owner accepted the 12K result when considering integration. A subsequent 12K profile measured 0.55 ms incremental allocation and zeroing over stock. The capture-bearing and untouched GPU passes varied across launch orders. [Attribution report](https://github.com/jshahbazi/stark/blob/2e3058d15fc67ccde7608861a80e5368b926acd8/docs/model-experiments/2026-11-17-flashnext-midpoint-attribution.md)

The integration patch is based on `2cc7345369d3ae34d7f293b2a6ac20db87bc21cb`. It passed a Rust 1.98.1 locked release build, six prefix-cache tests and the physical Metal capture fixture, which matched 155,762,688 bytes in three geometries. The integration executable SHA-256 is `dda01cbfa57d81582bbfe0d8511753bae1cc47ef1075316f258ae99761d8a207`. The full-model measurements retain their original experimental build identity.

The recommended landing scope is the guarded worker option for the pinned artifact. Wider availability needs tests for a later midpoint replacing a restored one, cancellation during capture, a real image transition, and MTP-off continuation. [Worker protocol](../GENERATION.md#prefix-reuse)
