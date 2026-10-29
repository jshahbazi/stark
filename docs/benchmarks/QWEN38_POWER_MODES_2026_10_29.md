# Qwen3.8 sampled generation with AC power settings

This follows the [desktop stability study](QWEN38_STABILITY_2026_10_29.md).
The inference source is `9204b0b9add0e6312618431854e5c5b5cfc2cf9f`; no inference
code changes during this experiment. The purpose is to test whether selecting
AC High Power changes the sustained clock reduction observed in Automatic mode.

## Fixed workload and control

The host remains an M5 Max with 40 GPU cores and 128 GB, macOS 26.7.1 (25G241),
connected to AC. The checkpoint is the same Lily q4-xl revision
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`, 55 shards totaling
106,905,155,192 bytes. The generator SHA-256 is
`ff4df5d10ccb95dd3ab3f9453be1bcbce830bbc6f99b27c079deced20d5e354e`.
Quantization, BF16/F32/Q8 execution, resident experts and request reset behavior
are unchanged. MTP, vision and speculation remain disabled.

Every request uses the same 4,096-token public prefix, 256 outputs, temperature
1, top-p 0.95 and seed 42. Top-k 20 and disabled top-k alternate, reversing
their order on successive cycles. Each phase loads the same worker afresh and
runs ten cycles: 20 requests, one warmup per sampling case and nine measured
requests per case. Decode rates count 255 steps after the prefill draw. Pooled
rates divide 2,295 measured steps by summed decode time for each case/phase.

The sequence is Automatic, High Power, then restored Automatic. Only the AC
Energy Mode is changed; battery mode remains Automatic. UI values and
`pmset -g custom` are checked at transitions. Desktop applications remain
running; there is no quiet-window filter. Builds and other GPU work belonging
to this experiment remain stopped during each phase.

Clock telemetry uses the same private, verified macmon v0.10.0 binary and
rootless IOReport reader as the preceding study. It records active-residency
mean GPU frequency, active/scaled residency, temperature and memory counters.
Actual consecutive source-timestamp intervals must be completely inside a
request to contribute. Independent CPU, GPU-client and VM counters record
desktop activity; raw GPU-client counter units remain unverified.

This is a sequential control on one machine. Desktop load, starting temperature
and fan/power-management hysteresis can differ between phases. The restored
Automatic phase helps expose drift; it does not make the order randomized or
establish a long-term performance guarantee. Each phase also has a separate
model load, a known source of variation in the preceding study.

## Results

All three phases finish with the expected 256-output hashes. Warmups are
excluded from the following table; all nine measured requests per case/phase
are retained.

| AC preference | Top-k 20 pooled tok/s | Top-k 20 range | No top-k pooled tok/s | No top-k range |
| --- | ---: | ---: | ---: | ---: |
| Automatic, before | 72.8832 | 71.0148–75.2111 | 67.3876 | 64.6633–69.9452 |
| High Power selected | 66.7898 | 66.3964–67.2011 | 62.5704 | 61.9249–63.8829 |
| Automatic, restored | 65.4143 | 60.9371–68.4515 | 61.4384 | 58.7097–64.6857 |

No measured request reaches 77 tok/s. Selecting High Power does not restore the
target in this window. The restored Automatic phase is slower still, so these
results do not establish that High Power itself causes lower throughput.

| AC preference | Top-k mean active GPU MHz / GPU degrees C | No top-k mean active GPU MHz / GPU degrees C |
| --- | ---: | ---: |
| Automatic, before | 1,295 / 88.7 | 1,330 / 88.8 |
| High Power selected | 1,248 / 85.8 | 1,308 / 85.7 |
| Automatic, restored | 1,201 / 87.3 | 1,257 / 87.0 |

The GPU remains almost continuously active: mean active residency is 1.000 for
top-k and approximately 0.999 without top-k. High Power selection does not hold
the sampled active frequency near the device's 1,620 MHz maximum. Temperature
is lower in that phase, but the experiment does not isolate cooling from other
policy and workload changes.

WindowServer's observed raw GPU-client counter rate rises from about 54–55
million units/second in the initial phase to 384–401 million in the latter two
phases. Other client activity also increases. These counters are not GPU
utilization percentages, and their units and client-continuity limits prevent
a precise resource attribution. They show material load differences between
the phases, which preclude a clean power-mode effect estimate. No measured
request records pageouts or swapouts.

The original AC preference is restored to Automatic immediately after the
High Power phase and before the final control. Both UI selectors then show
Automatic. The final `pmset -g custom` output exactly matches the initial
snapshot, including all other preferences. Battery mode is never changed.

## Interpretation limits

Only the selected preference and its `pmset` value are verified at the mode
transitions. System Profiler's high/low-power flags disagree during the
High Power selection, reporting High Power off despite the UI selection and
AC `powermode=2`. Those flags are not used as evidence of the effective GPU
performance state. Actual clocks and residency are measured independently.

The screen is found locked after the final control. Its transition time is not
recorded, so a lock during part of the final phase cannot be excluded. An
optional additional API cross-check is abandoned before making any further
setting change. Automatic remains restored.

## Integrity and provenance

All 60 requests produce exactly 256 outputs, finish at length and match the
expected seeded hashes for their sampling cases. All phases use the same
worker and measurement-script hashes, public input hash and sampling settings.
The 54 measured requests account for 13,770 decode steps. No warmup or measured
request is removed because of contention or low throughput.

The clock captures contain 126, 131 and 133 samples respectively, with median
cadence about 1.008 seconds and observer delay below 0.2 ms. The full-request
join retains 95, 99 and 102 complete intervals, four to six per request.
Approximate decode-only joins remain separate from these complete-request
clock summaries. OFF-state zero is valid; the capture helper's below-338 MHz
flag is a known false positive for OFF and is not treated as a failed sample.

The macmon v0.10.0 release asset SHA-256 remains
`0ccff60b7bf67bcb9e09cc8f804656a1cafa7eeae7297ee83d10e460ead7d43d`;
its binary SHA-256 is
`57259ce6845d0904be247c8e860fcc923300bab52e28a01f476c22c852051286`.
The source revision is `5df67a54e082e0b30a0aa2b48b7c7f78a36f1795`.
The MIT license is retained with the private tool. It is not a Stark dependency.

| Private aggregate | SHA-256 |
| --- | --- |
| Automatic-before report | `4062d9df8cb1325a6dd0355e6657cdc65983b31cce58d84382be60034b1fb7e2` |
| High-selected report | `d5fb42b94ea8bbc491884998c314bba191caa799d8ce190bf76317b3f745ba80` |
| Automatic-restored report | `1dc92212372a52ef8c2314ab6f20b95093593714e1dfe7e99b5695dc12d8f914` |
| Three-phase analysis | `c1531fb88c2b2e4a113f0008594d792cec4f714be6c37ea6f5eb31f11eacdad4` |
| Final weight/worker/restoration check | `c5fe5ea3e4a68a27dca6a4e48f15cbad7f68e2a558d87d2c7f0b2863431c662c` |

All 55 checkpoint stat identities still match the preceding complete SHA-256
verification. The final worker hash remains the measured `ff4df5d1` above.
Raw protocols, telemetry, tool binaries and analysis artifacts remain private
and ignored. This experiment changes documentation only; the inference code,
precision policy and quantization remain as qualified in the preceding study.
