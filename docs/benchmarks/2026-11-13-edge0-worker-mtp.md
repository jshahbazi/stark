# Edge0 local worker MTP: matched requests, unqualified timings

Attempt [20261113-EDGE0-11](../model-experiments/2026-11-13-edge0-mtp-batched-worker.md),
2026-11-13. All 60 normal-worker requests completed with matched useful work.
**No timing group qualified. MTP remains off by default.** These observations
support no throughput ranking or performance promotion.

The candidate is `bdf3b62e`, Rust 1.98.1, on a 128 GiB Apple M5 Max,
Mac17,7 / macOS 26.7.2 build 25G332. Release worker SHA-256:
`056839034938df8d4ffa19ec55ee0be0923e4559004acbc688dc1035ff07ede2`.
The exact prepared artifact is `Edge0/Edge0-145B-q3k` at
`d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`, receipt SHA-256
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`.
This is local non-commercial evaluation only. The prior complete payload receipt
and immutability assumption were reused; no new 103 GB payload rehash is claimed.

## Matched work and controls

One loaded normal JSONL worker, context 512, original weights and F32 arithmetic,
virtual mappings, prefix reuse disabled. Greedy seed 42; prompt lengths 32/128;
maximum 16 new tokens; off paired with depths one, two and three. Each group used
one warmup per route and four balanced ABBA pairs: 12 warmup requests and 48
measured arms across six groups. Loading and cold first-request time are excluded.
The native target uses identical prompt batches in both modes and ordinary
single-token verification. Complete readout exactness has its separate gates.

The private corpus SHA-256 was
`1a681566d9fcf584e42df629c610513a786648976d5365687a2e6465b6fc504c`.
Every warm/measured arm in a group matched emitted token hashes, sampled/emitted
counts and stopping reason. The 32-token prompts stopped naturally at EOS after
seven sampled tokens, six emitted; 128-token prompts emitted all 16 and stopped
at the budget. No EOS suppression, prompt selection or arm retries were used.

Useful emitted decode excludes the first output and EOS:
`(output_tokens - 1) / decode_seconds`. Raw sampled decode additionally counts
the EOS draw. Request wall includes IPC and cleanup; decode excludes prefill.
MTP-off had zero speculative activity; every on arm had real proposals and valid
verification counters. Speculative counters may include queued work beyond a
later EOS. They are not counts of useful emitted drafts.

## Observations excluded from performance qualification

All rows below are **timing-ineligible observations**. Rates are route medians;
wall time is median seconds per request. They are retained to make the failed
qualification reviewable, and must not be used to rank draft depths.

| Prompt / drafts | Off emitted decode tok/s | On emitted decode tok/s | Off wall s | On wall s | Timing disposition |
| --- | ---: | ---: | ---: | ---: | --- |
| 32 / 1 | 6.45 | 2.75 | 3.08 | 4.22 | Ineligible |
| 32 / 2 | 6.45 | 2.69 | 3.18 | 4.32 | Ineligible |
| 32 / 3 | 5.23 | 2.26 | 3.82 | 5.00 | Ineligible |
| 128 / 1 | 6.32 | 2.84 | 13.37 | 16.38 | Ineligible |
| 128 / 2 | 6.55 | 3.00 | 12.75 | 15.65 | Ineligible |
| 128 / 3 | 6.65 | 3.04 | 12.56 | 15.41 | Ineligible |

Ten-second quiet admission preceded each group; every measured arm was screened.
System-wide decompressions increased in 46 of 48 arms; pageouts increased in nine.
One arm exceeded the external CPU ceiling of 150%. Compression/swapout counters
did not increase and thermal state remained nominal. Only two individual arms
were eligible, with zero complete eligible pairs or groups. No arms were removed
to manufacture a valid comparison. The off wall-spread checks passed, but cannot
override the failed host gates. Global paging counters do not attribute those
events to this worker.

The default-selection gate required complete eligible groups at both prompt
lengths, at least 5% median paired decode and wall improvement, no more than 5%
prefill regression, and at most 15% off wall spread. No depth qualified, so the
preregistered decision retains default zero. Explicit one-to-three-draft request
overrides remain available for evaluation. A future qualified speed claim needs
a fresh fully screened run under unchanged or separately registered conditions.

The worker exited normally on stdin EOF with code zero. Sampled peak owned RSS
was 802,897,920 bytes under the 48 GiB watchdog; available memory before launch
was 86,582,296,576 bytes. RSS does not measure complete GPU memory/residency.

## Evidence and predecessor

| Binding | SHA-256 |
| --- | --- |
| Source manifest, 2,182 files | `3626c3b1a05bcd9ef5965cead2e168a8978963cc45888dc8f94cf273fd7912fd` |
| Registration | `32074cb0a2e3781af6f4039b124302fa13da7073648b75334658355fe08e335f` |
| Private aggregate report | `45622b35d83b6615c8f06d2f03206cbbc23a945ceb395ef52631c42141d3dd34` |
| Harness | `baa704aa7c442d894261b2a7d932cabacff2f9cc27ad484367e95ced0b8c985c` |
| Complete worker evidence receipt | `b42e7755e7fab57e28749ed2ee288758cc03c712843a5e659e406c6b80351d08` |

[Attempt 10](../model-experiments/2026-11-13-edge0-mtp-worker.md) required 16
emitted tokens; its first off warmup failed that gate, before any measured pair.
This successor registered natural-stop matching separately and preserves that
failed setup. It does not supersede Phase 4's target-only measurements or establish
quality, wider-context, low-memory, resident-mode MTP or product admission.
