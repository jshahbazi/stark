# 20261113-EDGE0-18: observed Edge0 token speed

## Result

The short run observed **6.35 decode tok/s with MTP off** and **2.77 tok/s
with one draft**. Including prefill, end-to-end rates were **1.20 / 0.98 tok/s**.
These are medians of two measured requests per setting on this workload.

| Measurement | MTP off | MTP on, one draft |
| --- | --- | --- |
| Decode median | 6.35 tok/s | 2.77 tok/s |
| Individual decode rates | 6.46, 6.23 tok/s | 2.93, 2.61 tok/s |
| End-to-end median, including prefill | 1.20 tok/s | 0.98 tok/s |

Decode uses `(emitted_tokens - 1) / decode_seconds`; the first token belongs
to prefill. End-to-end uses `emitted_tokens / parent_request_wall_seconds`.
All six requests emitted and sampled 16 tokens with a length stop, so EOS
accounting does not change these decode rates.

## Recipe and outcome

Executed 2026-11-13 for local non-commercial observation at the user's request
for practical token rates. One worker used 128 input tokens, context 512,
maximum 16 outputs and greedy seed 42 with natural EOS handling. One warmup
per setting preceded measured order **off, on, on, off**. Warmups are excluded
from the medians. All six requests matched the predecessor output hash, counts
and stop condition; on requests performed real speculative work. The worker
exited zero on stdin EOF and was reaped. Sampled peak owned RSS was
599,638,016 bytes; this does not establish complete Metal residency.

The registered observational recipe has no quiet wait, full host Monitor or
host-eligibility screen. Endpoint global paging is recorded without subtraction.
This run provides observed speeds under the recorded conditions. It establishes
no strict performance qualification, broader speed ranking or default-policy
promotion. [Attempt 17](../model-experiments/2026-11-13-edge0-arm-attribution.md)
and its existing strict qualification gates remain unchanged; MTP remains off
by default.

## Paging and evidence

The following are summed request-endpoint deltas, excluding load and gaps
between requests. They are global events without process ownership attribution.

| Request brackets | Compressions | Decompressions | Pageouts | Swapouts |
| --- | ---: | ---: | ---: | ---: |
| Two warmups | 1,641,796 | 667,243 | 414 | 0 |
| Four measured requests | 178,966 | 240,594 | 668 | 0 |
| All six | 1,820,762 | 907,837 | 1,082 | 0 |

Reuse `Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`,
the unchanged worker, virtual Metal, F32 state/activation/accumulation/cache and
selected-record bindings. Prefix reuse remains unavailable. Prior complete
payload hashes retain their immutability assumption; no payload rehash or new
build occurred. Source commit is `ca0b2e5b47d5857be9b733cc4c1d9d01fbc8a9a7`.

Private receipts stay ignored under `runtime/edge0-observed-speed-2026-11-13/`.

| Binding | SHA-256 |
| --- | --- |
| `run_speed.py` | `b78959a346bc34d4221115ef221034580b47a6dd4301d4af3f291466ff36e5f3` |
| `observed-speed-1.registration.json` | `62ed29428235cd76b2d90275c4c58e51d61b15b33ad1ec0d334dbbdadef4b24c` |
| `observed-speed-1.report.json` | `15fb58df44fcff25e1d4ccb5f5ebdec6a73206a17671e8f7962e032db1c94af7` |
| Worker | `0f6db70121c424f86457023de5330635b37ad677d8ebe3179d12e37cb9240055` |
| Source manifest | `72cd9009cc1e5d1fbaae55465b1c6367b4343633e48630f9c4ce912ce32b5b47` |
| Native receipt | `8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e` |
