# Qwen3.8 sampled throughput and desktop contention

This study starts at `908ae21f4894d841cf2e1ad44451643439543443`, following the
[HC/GDN kernel study](QWEN38_KERNELS_2026_10_29.md). The acceptance target is
77–80 sampled decode tokens/second during normal desktop multitasking, including
other active GPU applications. Measurements retain competing activity; no quiet
window filter is applied.

## Fixed workload

The model remains `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at revision
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`: 55 shards, 106,905,155,192 bytes.
Weights and quantization are unchanged. Execution uses BF16 activations, F32
reductions/logits/GDN state and Q8 attention K/V. Speculation, MTP and vision
remain disabled; experts remain resident.

The host is the same M5 Max, 40 GPU cores, 128 GB, macOS 26.7.1 (25G241), on AC.
The public corpus, tokenizer and input hashes match the preceding study. Each
request uses 4,096 input tokens, 256 outputs, temperature 1, top-p 0.95, seed 42,
and either top-k 20 or disabled top-k. Reported decode rates count 255 steps
after the prefill draw. All retained comparisons use the same expected output
hashes, including fresh request state and RNG initialization.

Lily's [77–80 tok/s reference](https://github.com/fabiogreter/lily-qwen3.8-flash-next/blob/e22d58fa7cc488493fb1b5d810a198bd29730aeb/README.md#performance)
uses greedy generation. Sampled throughput is measured explicitly here.

## Measuring the instability

The unchanged preserved production worker starts at a 56.6968 tok/s warm median
for top-k 20, with measured warm requests spanning 53.3238–61.0991 tok/s. All
five completion hashes match the earlier, faster windows. This reproduces the
instability without a code or checkpoint change.

Private diagnostic workers retain the normal one-command-buffer decode graph
and verified completion/error handling. Their CPU profiler is enabled without
GPU stage timestamp writes. Completed Metal feedback provides command start/end
times; host marks separate GPU completion, callback entry/publication and
accepted completion. These are diagnostic builds, not production throughput
claims. GPU command spans can include scheduling interruptions.

The monitor samples CPU time, VM counters, Foundation thermal state and GPU
client counters. `IOGPUDeviceUserClient` exposes `AppUsage.accumulatedGPUTime`
and its creator process without administrator access. The counter's units and
exact hardware accounting are not established here; comparisons use raw deltas
and ratios between observed clients. They are not GPU clock measurements or
proof of exclusive GPU ownership. Counter resets, disappearing clients,
unavailable sources and request-boundary coverage must be recorded.

All raw runtime records remain private and ignored. Published evidence contains
aggregate measurements and source/output hashes. Builds and GPU work belonging
to this experiment are serialized. Other desktop applications and independent
workloads remain active.

## Scheduling controls

Lily holds a per-request macOS process activity assertion; Stark's production
worker previously omitted it. A same-loaded-worker ABBA control toggles that
assertion between requests. It performs one warmup per route/case followed by
two ABBA blocks per case: 20 requests total, 16 measured. Every request matches
its expected output hash and selected-route acknowledgment.

| Control | Top-k 20 pooled tok/s | No top-k pooled tok/s |
| --- | ---: | ---: |
| Activity assertion off | 53.5065 | 49.4293 |
| Activity assertion on | 52.4748 | 48.3627 |

The direction changes between individual ABBA blocks. This control establishes
no performance benefit for the assertion. The host is under material competing
load: external CPU peaks reach 1,238% of one core, thermal state remains fair,
and some requests record VM activity. The retained assertion expresses the
request's user-initiated lifetime; it is not claimed to lock GPU clocks or solve
the throughput swings.

A second control tests a temporary user-initiated thread QoS override. Actual
worker queries report `USER_INTERACTIVE` (0x21), relative priority zero, before,
during and after every request. The override is below that existing priority;
it cannot improve this launch's classification. No thread QoS override is
retained in production.

The QoS control's pooled warm rates are 59.8593 versus 58.7093 tok/s for
top-k 20 and 58.2942 versus 57.3179 for nucleus-only sampling. These are
separate measurement windows from the activity and queue controls.

The GPU telemetry in that control correlates the later slowdown with increased
WindowServer GPU work. In matched complete monitoring intervals, its observed
GPU-counter share rises from 6.1–6.5% to 22.3–31.6% across the top-k blocks;
top-k throughput falls from roughly 71–72 to 48–57 tok/s. Other GPU clients and
CPU activity also contribute. These are concurrent observations, not an isolated
causal intervention on WindowServer.

Metal's default feedback callbacks also report `USER_INTERACTIVE`. A separate
same-loaded-model control switches between the default Metal queue and a queue
with an explicit serial user-initiated feedback queue. Both queues share the
same weights, mappings, scratch, residency set and monotonically ordered fence;
switches occur only after drain/reset and verified completion.

| Feedback queue | Top-k 20 pooled tok/s | No top-k pooled tok/s |
| --- | ---: | ---: |
| Default Metal callback queue | 70.0958 | 66.5940 |
| Explicit serial callback queue | 70.2200 | 67.2694 |

The differences are small, and the explicit queue lowers observed callback QoS
from 0x21 to 0x19. It is not retained as a priority fix. Callback validation is
never bypassed to reduce latency.

Warm mean GPU-end-to-accepted-output latency is about 0.112 ms on the default
queue and 0.111 ms on the explicit queue. The roughly 1.6 microsecond change is
too small to explain the observed end-to-end differences. The earlier activity
window contains a 267.945 ms maximum completion delay. The later split window
does not reproduce that delay: its maximum GPU-end-to-callback-entry delay is
7.686 ms and callback-publication-to-acceptance maximum is 0.142 ms. The earlier
extreme cannot be assigned retrospectively to either subcategory.

## Exact CPU selection

Top-k up to 64 now scans contiguous logits while maintaining a sorted candidate
prefix. Equal keys keep the earlier, smaller token ID. More than 1,024 prefix
replacements falls back to the existing partition selector, bounding the extra
work on adversarial ascending inputs.

Full-vocabulary ordering uses three stable 11-bit radix passes over packed F32
keys and IDs. It avoids random logit gathers in later passes. Descending F32
total order, ascending-ID ties, signed zeros, F32 temperature arithmetic, serial
F64 probability sums/filtering, and the SplitMix64 draw remain unchanged.

A paired control compares the starting selectors and candidates on each of 256
real complete-model readouts, alternating AB/BA. Every top-20 prefix and complete
vocabulary order matches. Measurements use the same 4K input and nucleus-only
seeded trajectory; only hashes and timing aggregates leave the test.

| Selector | Previous mean / median ms | Candidate mean / median ms | Mean reduction |
| --- | ---: | ---: | ---: |
| Top-20 prefix | 0.274468 / 0.273875 | 0.072637 / 0.072125 | 73.5% |
| Complete vocabulary order | 1.118435 / 1.104188 | 0.635396 / 0.629354 | 43.2% |

These are selector timings, not total-token gains. The preserved full-sort
reference also matches all complete orders. CPU-only controls include finite
IEEE edge patterns, signed zero, ties, ascending/reverse inputs, the fallback
boundary, seeded choices and RNG advancement. The generator release suite
passes 52 tests; the real-readout test passes separately.

## Rejected MoE input-sum cache

A separate trial caches the input-only F32 sums used in Q4 affine gate/up dots:
80 sums, 320 bytes for the production hidden width. It preserves quantization,
dot expressions, reductions and BF16 round sites. The proposed prepass runs at
the existing input-only dependency level beside the router and shared projection.

Exact controls pass 42 cases and 258,034 compared bytes, including complete
gate/up outputs and down/shared-expert combination, offsets, canaries, group
sizes, extremes and cancellation. Their aggregate SHA-256 is
`69b010c7ad2e74bc43db62e6f2bf4da5878b2ba08d482fe961d05b6d431d61bf`.

The paired 16-sample chain with distinct weights larger than GPU caches is
slower with the prepass included: gate/up mean 35.346 versus 35.956 us;
gate/up plus down mean 56.740 versus 58.094 us. A subsequent intrinsic-kernel
control with the sums prepared outside timing is flat: 34.8286 versus
34.8344 us mean, median 34.8021 versus 34.5885 us. Hiding the prepass has no
established useful headroom. The entire production trial is reverted; its
patch, controls and complete ordered timings remain private.

Two further private trials retain the same Q4 weights:

- Guarded masked unpacking uses exact power-of-two activation rescaling only
  inside a conservative BF16 range, with the original inline unpacking outside
  it. A helper-based fallback initially changes 48 complete-kernel BF16 outputs
  despite matching isolated word controls; copying the original fallback body
  restores complete parity. This rejected intermediate remains separate.
- Caching the guarded, scaled F32 inputs and original F32 sums removes repeated
  per-row conversion/scaling, at a cost of 10,880 scratch bytes reused across
  layers. The prepass is included in the measured chain.

Both final variants pass 1,828,864 exact F32 word cases and the 42 complete
kernel cases above. Their word-control SHA-256 is
`687e2f84cbc903bb456c7fe318adc27268cfc93512d86320915d33580dae4911`.
Their measured gate/up-plus-down chains provide no speedup:

| Private MoE trial | Original mean us/layer | Candidate mean us/layer |
| --- | ---: | ---: |
| Input-sum cache | 56.740 | 58.094 |
| Guarded masked unpacking | 56.665 | 61.406 |
| Cached scaled F32 inputs | 90.308 | 90.856 |

The final window has large timing swings. Its median chain is 66.729 versus
69.766 us; all 16 samples are preserved. None of these three trials is retained.
The aggregate receipt SHA-256 is
`29aca8b81d5a5f08fa512f8c62ee5258862fd3a93e69c858162d7e162225d40d`.

## Production sampled replay and sustained clocks

The retained sampler/activity worker SHA-256 is
`ff4df5d10ccb95dd3ab3f9453be1bcbce830bbc6f99b27c079deced20d5e354e`.
Five production requests per case, with the first excluded, measure:

| Sampling | Warm median tok/s | Warm range tok/s |
| --- | ---: | ---: |
| Top-k 20 | 74.1846 | 71.0075–76.5263 |
| No top-k | 62.8178 | 59.4460–66.2312 |

All ten outputs match their expected hashes. These sequential windows have
optional host monitoring enabled and do not isolate an end-to-end selector
speedup. No paging is recorded, all thermal samples are nominal, and raw GPU
client counters vary with desktop activity. Nominal thermal state alone does
not establish a fixed GPU clock.

A subsequent single-loaded-worker run alternates the two sampling cases for
ten cycles: 20 requests, with one warmup per case. It adds independent rootless
frequency/residency measurements using [macmon v0.10.0](https://github.com/vladkens/macmon/releases/tag/v0.10.0).
The official release asset is verified as SHA-256
`0ccff60b7bf67bcb9e09cc8f804656a1cafa7eeae7297ee83d10e460ead7d43d`;
the tool remains private and is not a Stark runtime dependency.

The pinned source reads IOReport's `GPU Stats / GPU Performance States / GPUPH`
channel and the device's `pmgr` DVFS table. The observed active table has 13
states from 338 to 1,620 MHz; zero is OFF. Reported frequency is the
residency-weighted mean over active states, not an instantaneous clock.
Active residency and frequency-scaled residency are recorded separately.
These private interfaces are not independently checked against privileged
`powermetrics` on this host.

There are 125 clock samples. Cadence is about 1.013 seconds, with no gap over
1.5 seconds and maximum source-to-observer delay 0.186 ms. Actual consecutive
sample boundaries, fully contained in absolute request boundaries, yield 94
matched intervals: four to six per request. Approximate decode-only joins are
retained separately; the table below uses complete-request intervals.

| Sampling, nine measured requests each | Pooled tok/s | Median tok/s | Range tok/s | Mean active GPU MHz |
| --- | ---: | ---: | ---: | ---: |
| Top-k 20 | 72.77 | 72.10 | 70.34–75.44 | 1,296 |
| No top-k | 67.63 | 67.47 | 65.14–70.82 | 1,320 |

The initial top-k warmup reaches 77.89 tok/s at approximately 1,614 MHz; the
nucleus warmup reaches 71.09 at 1,577 MHz. Later rates decline as matched GPU
means reach about 1,220–1,226 MHz and temperatures approach 93–94 degrees C,
then partially recover. GPU temperature peaks around 95.8 degrees C. Frequency
and throughput are strongly correlated within each small time-ordered case,
but that does not isolate thermal or power management as a cause.

The GPU remains almost continuously active. No request records pageouts or
swapouts. Existing swap usage stays fixed. This run observes little other
GPU-client counter activity, unlike the earlier WindowServer-heavy window.
External CPU activity remains present. The evidence supports two contributors
to investigate separately: competing work and sustained frequency changes.
The long-run analysis SHA-256 is
`61c4a49bedb255b13f995a691185388a8b26e3d077adba684fd8d54a95ec95de`.

macOS AC and battery Energy Mode both read Automatic after this run. High Power
mode has not been tested; no result for that mode is claimed.

## Validation and remaining work

The activity, QoS and queue controls each complete 20 requests with matching
hashes and route acknowledgments. A further six-request diagnostic separates
callback delivery from host acceptance. Source manifests and preserved binaries
bind each private control to its measured inputs.

- Native CPU/Metal engine: 666 tests passed, 45 ignored; seven C++ integration
  tests and three doctest checks passed. This includes the new activity scope,
  unwind and recovery control in the repository's custom test harness.
- Generator release: 52 tests passed, one ignored; the ignored 256-readout
  paired selector control was run explicitly and passed.
- Optimized scorer release: 39 tests passed. Generator and optimized-scorer
  Clippy checks pass with warnings denied. CPU-only engine admission compiles.
- Fresh real-worker integration: 13 generator and 12 scorer requests passed;
  workers loaded serially and closed cleanly. Checks include seeded repeat and
  isolation, streaming/cancellation/recovery, natural EOS, finite candidate
  scoring, full-vocabulary log-partition consistency and consecutive-cache
  rules. It does not independently sum every vocabulary probability.
- Benchmark and host-reader tests: 11 passed. Ruff, formatting, repository
  policy and whitespace checks pass. The opt-in readers also passed a live
  model-free smoke test. An independent read-only review found no actionable
  issue in the retained activity, sampler or monitor changes.

The final source inventory covers 736 compiled/build/test inputs against the
starting revision. Its SHA-256 is
`3b726d526591e270217777b81df55a81c7b92dbe33847fbc655f2010ca73b624`.
The optimized scorer SHA-256 is
`c7eb6e3fb7dca8f6b06b4ecacc28bb3b70606814522bd719bf63ef9c92c98322`.
A final generator rebuild retains the measured `ff4df5d1` executable bytes.
All 55 shard stat identities still match the preceding complete SHA-256
verification. Dependency locks and production vendored-core sources are
unchanged by this study. Runtime binaries, full records and experimental
patches remain ignored and private.

The 77–80 tok/s acceptance target during normal desktop multitasking has not
been established. Selector gains are measured independently from overall
throughput; rejected scheduling and MoE candidates remain private.
