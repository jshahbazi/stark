# Regular Decider v1.3 201-token cold timing

## Scope

This is one private, local M5 Max smoke timing, not numerical qualification or
a production throughput claim. The request contained one text choice question
with 201 expanded input tokens. No image, generation, MTP or prefix reuse was
used. Prompts, answers and token vectors were not retained.

| Field | Value |
| --- | --- |
| Host | MacBook Pro, Apple M5 Max, 128 GB unified memory |
| OS | macOS 26.7.2 build `25G332` |
| Toolchain | Rust 1.98.1 (`48a229cea`) |
| Source | `perplexity-ai/pplx-decider-v1.3` at `9c2ebb27470d3dec2ad7ab727727e40d142c7213` |
| Prepared model SHA-256 | `8029720978d100a0a26fe15503400efc30d31206d09fc9398ba2a084b593b4eb` |
| Prepared config SHA-256 | `7b829079acaf459d0456af3897ece45050ca1e7a8e2c162e9e8d026fd74ebc66` |
| Worker source revision | `ad125f791c7b51656454403ad52f2037ce3ee1df` plus the current uncommitted Decider patch |
| Worker SHA-256 | `76b836034f0b9f2afa371b049fcda965844c439625d4f6adcedf49bea0c1317a` |
| Private request hash | `7bb8e3d80645316ac98428da44df38edb3369a9e302992dab055b118fb7a5e94` |
| Precision | BF16 weights/activations; F32 normalization, accumulation and recurrent state |

## Observation

| Metric | Result |
| --- | --- |
| Worker load | 10.042 s |
| State creation | 0.148 s |
| Prefill | 50.241 s |
| Readout | 0.000015292 s |
| Native request total | 50.392 s |
| Frontend request total | 50.406 s |
| Prefill throughput | 4.001 processed input tok/s |
| Native end-to-end throughput | 3.989 processed input tok/s |
| Frontend end-to-end throughput | 3.988 processed input tok/s |

Decider emits a distribution rather than generated output tokens, so these rates
are processed-input throughput, not decode/output-token throughput.

## Limits

- One cold observation only; no repeats, host screening, image case or warm-run
  variability analysis.
- The native load succeeded and the request completed structurally, but no
  upstream Python reference comparison was performed in this smoke test.
- The 27B checkpoint remains numerically unqualified. The earlier Flash result
  does not transfer.
- Peak unified-memory use was not measured. Model weight residency is about
  48.6 GiB before working state.

Private receipt: `runtime/decider-v13/benchmark-300token.jsonl`.
