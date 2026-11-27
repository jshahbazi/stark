# 20261126-HY4OC-03 · Sushi 4bpw and resident Stark through OpenCode

## Identity and controls

This run compares Sushi's native Hy4 deployment with the earlier
[resident AngelSlim Q3_K worker](2026-11-26-hy4-resident-opencode.md)
on the same 128 GiB M5 Max, macOS 26.7.2 and OpenCode 2.4.2. The Sushi arm
used [Sushi 1.8.1](https://github.com/beamivalice/sushi/releases/tag/v1.8.1)
and the [208.4 GB STQ 4bpw pack](https://huggingface.co/beamster/Hy4-Flash-Sushi-STQ-4bpw/tree/61528ba635a623f127c5a7b1e1f4ac2c2b0d4885)
at revision `61528ba635a623f127c5a7b1e1f4ac2c2b0d4885`. All 99 files matched
the pinned manifest digests after the interrupted transfer.

Sushi served a 118,784-token window with KV8, vision loaded, DFly2 and 4,096-row
prefill. The Stark worker fitted the same loaded capacity with the BF16
projector, 4,096-row prefill and greedy NextN K2. Stark's measurements
precede the compensated sparse MLA default. The recovered harness used the
same synthetic source generator, user instructions and source lengths as the
earlier Stark runs. Both clients submitted 12 tool definitions and generated
96 tokens per turn with greedy settings. There were no tool calls.

The two arms used their own templates, weights, caches and draft systems.
Rendered lengths differed by 32–33 positions. Each continued turn included
its own model's first reply. The earlier Stark logs were unavailable after
reboot, so fixture equivalence was checked from the harness source and
invocation settings. The runs were in separate sessions on the same Mac.

## OpenCode timings

| Turn | Sushi SSD cache | Total input, Stark / Sushi | Reused, Stark / Sushi | Prefill tok/s, Stark / Sushi | Decode tok/s, Stark / Sushi | OpenCode seconds, Stark / Sushi |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fresh 16K | Off | 16,000 / 15,967 | 0 / 0 | 484.59 / 1,012.80 | 38.67 / 56.01 | 36.171 / 18.135 |
| Fresh 16K | 8 GiB | 16,000 / 15,967 | 0 / 0 | 484.59 / 806.42 | 38.67 / 43.12 | 36.171 / 22.608 |
| Continued 27K | Off | 26,950 / 26,918 | 12,288 / 0 | 500.06 / 886.68 | 35.74 / 42.92 | 32.864 / 33.212 |
| Continued 27K | 8 GiB | 26,950 / 26,918 | 12,288 / 16,063 | 500.06 / 731.50 | 35.74 / 41.85 | 32.864 / 17.713 |
| Fresh 27K | Off | 27,000 / 26,967 | 0 / 0 | 495.62 / 786.69 | 40.77 / 36.49 | 57.563 / 37.463 |

Sushi rates are from the final OpenAI SSE usage frame and match the phase
times and token counts. Prefill divides newly evaluated positions by phase
time for both workers. Sushi counts all 96 output tokens in its decode rate.
Stark counts from the second output token. Applying that numerator to
Sushi's phase timing gives 55.43, 42.67, 42.47, 41.42 and 36.11 tok/s in
table order. OpenCode seconds measure the entire client invocation.
[Sushi timing implementation](https://github.com/beamivalice/sushi/blob/v1.8.1/src/generation_stream.zig#L2025-L2047)

The initial server refused a model-scoped prefix-clear request with HTTP
409 before any timed prompt. Sushi requires its SSD tier for that operation
and for Hy4 prefix restoration. Cold runs then used fresh server processes
and reported zero reused tokens. Each continuation shared the process and
OpenCode session of its corresponding 16K turn. With SSD enabled, Sushi
evaluated 10,855 positions on the continuation versus 14,662 in Stark.
[Sushi cache clear](https://github.com/beamivalice/sushi/blob/v1.8.1/src/prefix_cache_admin.zig#L183-L217)

All sampled memory pressure levels were normal and swap stayed at 1.0 MiB.
Sushi took 106.49 s to become ready on the initial attempt, including served
warmup. It reused the pipeline and warmup manifest on subsequent starts,
which took 20.10–22.11 s. The two fresh 16K runs produced different greedy
output digests across cache settings, with the same rendered lengths and
zero cache reuse. Each setting has one sample. Host, pipeline and cache
effects need paired repeats before attributing that change to any one cause.

## Disposition

Sushi completed both cold fixtures faster, with 1.59–2.09× the observed
tuned-resident prefill rate with SSD off. Its fresh 27K decode was slower
in this run. The SSD-enabled continuation completed in 54% of the earlier
Stark wall time, with both more prefix reuse and a higher rate on the
remaining prefill. These are end-to-end observations across two packs.
A same-artifact stage profile is still needed to price the attention
product, expert prefill, and prefix boundaries separately on Stark.
Natural-text quality and the combined Stark defaults remain open.

| Private evidence | SHA-256 |
| --- | --- |
| Sushi release archive | `01a2110d6f6bf664bc1598a37fd35337b0486cc44bd29c674ad89eb481f139f6` |
| Sushi executable | `13feaab689501082fd166f6a77e80683362b83ac5b37b527dee0d88875d3bf57` |
| OpenCode harness | `283eacf9252895e3c904948c159234a0ab2530cdd03e3a1ee201409d82a6d323` |
| Initial checksum and cache-clear receipt | `ff689226ac41d5779cb538ce3091bd57a25e253a2d085ba51b21003fc0297c88` |
| SSD-off timing aggregate | `9670688182190979ca2492e9bc4b2ae14a5456f7cf0c606ecfb492389282fdb2` |
| SSD-on timing aggregate | `032243482832abe4245c2c8c38eee30706f557143ba36e71a752cda456294e6c` |
