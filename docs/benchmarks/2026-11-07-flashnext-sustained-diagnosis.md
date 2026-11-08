# Flash-Next sustained throughput and idle wake diagnosis

Date: 2026-11-07. This follows the [MoE projection experiment](2026-11-07-flashnext-moe-projections.md).
The [aggregate receipt](data/flashnext-sustained-diagnosis-2026-11-07.json)
retains all 128 measured diagnostic requests, source and executable hashes,
host screens, clock coverage and native phase aggregates. Twenty-four declared
warmup or settling requests are excluded from measured summaries. These are
observed diagnostics without quiet admission; 125/128 requests pass the host
screen. No release throughput or candidate speedup is qualified.

## Findings and disposition

The earlier 95 tok/s result is reproducible after a pause. It is not sustained
throughput for this workload. The original baseline also slows during repeated
requests. Most added verification time is inside GPU command execution spans;
host scheduling and unchanged MTP work counts do not explain that increase.
GPU performance-state reductions are observed contemporaneously. The evidence
does not identify the firmware's exact thermal, power or memory limit reason,
and clock reductions do not explain every slower request.

A separate idle effect adds about 0.7 seconds before the first sample. An empty
GPU queue signal every 500 ms during a ten-second pause removes that delay in
an ABBA control, while prefill GPU work remains about 0.292 seconds. This
implicates queue/residency wake-up work in the host/driver path. No background
keeper or diagnostic operation is added to the production worker.

The single-loaded-model crossover does not reproduce the earlier apparent 5%
MTP penalty as a consistent code effect. Neither projection candidate has an
accepted whole-model win. Both are withdrawn from the current engine: generic
gate/up and the original shared-down dispatch order are restored byte-for-byte
from engine baseline `72b5aaef96c939809bc906d10d45a0eef70386c0`. The candidate
patches, synthetic controls and original source-bound results remain archived.

## Fixed workload and machine

The only artifact is `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`: 55 shards, 106,905,155,192 bytes.
The existing full-payload hash receipt is reused; shard size, nanosecond mtime
and inode are checked before and after each run. This is not a new payload
rehash. Quantization, BF16 activation boundaries and F32 accumulation/state
policies are unchanged. The official mixed-affine artifact is not tested here.

The machine is Mac17,7, M5 Max with 40 GPU cores and 128 GiB physical memory,
macOS 26.7.2, on AC. The existing AC power-mode preference is 2 (High Power);
no power settings are changed. The console remains locked. Display power state
is unconfirmed, so these results do not establish a lock-screen/display cause.
Workers are release builds with Rust 1.98.1 and locked dependencies. GPU/model
jobs run serially, without compiler work during timing.

Each request uses 512 source-corpus tokens and 256 sampled outputs, context
768, temperature 1, top-k 20, top-p 0.95 and seed 42. MTP is off or uses two
drafts. The corpus is `vendor/lily-core/src/qwen4exp/model.rs`, SHA-256
`33dcb30e762d63db4f16aaef21034fbaab93dd56dfd9b555e6da38f5db3238e3`.
Configuration, input and output hashes are in the receipt. Decode throughput
is 255 decoded steps divided by native decode seconds, excluding the first
sample. Full request wall time and time to first sample are recorded separately.
Repeated identical seeded outputs are a bounded consistency check, not a
whole-model quality or numerical equivalence claim.

All MTP-on measured requests retain 108 rounds, 214 proposed tokens, 147 accepted
tokens and 322 verified rows. Same-route output hashes match across variants.
No prompts, responses, token vectors or probability vectors are written.

## Sustained baseline control

The telemetry grid declares worker order baseline, final candidate, gate-only
candidate, baseline. Each owns one loaded model, two warmups and two cycles of
eight requests, with a ten-second pause before each cycle. The ordinary host
screen interval remains between requests. All 64 measured requests are retained;
63 pass their host screen. One gate-only arm exceeds the external CPU threshold.
No pageouts, swapouts or external model CPU activity are observed in this grid.

The first baseline session gives a direct same-code comparison:

| Cycle 0 request | Decode tok/s | Active GPU MHz | Verification elapsed, s | Verification GPU span, s | Elapsed minus GPU span, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| First after pause | 95.46 | 1620 | 2.2865 | 2.1922 | 0.0943 |
| Second | 95.55 | 1620 | 2.2846 | 2.1931 | 0.0915 |
| Eighth | 82.20 | 1488 | 2.6758 | 2.5730 | 0.1028 |

From first to eighth, verification grows by 0.3893 seconds; GPU span grows by
0.3808 seconds, about 98% of that increase. Warm request CPU instruction counts
remain near 2.905 billion, CPU time near 0.16–0.18 seconds and process footprint
near 75.62 GB. The GPU is active for essentially the entire covered decode
interval. This localizes the dominant observed delay to GPU execution spans.
Those spans include stalls or preemption; they are not a pure ALU-work measure.

The second cycle again starts at 95.09 tok/s and 1620 MHz, ending at 83.94 tok/s
and 1329 MHz, with peak reported GPU sensor average 98.15 C. The second baseline
session, with the same source and executable, starts at 77.15 tok/s and later
recovers to 92.26 after a pause. Cross-worker pooled rates are consequently
confounded by time and hardware operating state:

| Session | Pooled MTP-on decode tok/s |
| --- | ---: |
| Baseline, first | 88.69 |
| Final candidate | 85.85 |
| Gate-only candidate | 72.30 |
| Baseline, last | 84.89 |

The gate-only session runs at roughly 1081–1268 MHz while reported GPU averages
are only about 65–73 C. Some requests in the earlier clock probe slow near
1595–1618 MHz. Temperature alone and core clock alone are insufficient to
explain every loss. There is no direct limiter-reason signal in these records.
The system's coarse process thermal state is often nominal during downclocking.
No meaningful competing graphics-client GPU counter activity is observed in
the sampled intervals; incomplete coverage does not prove exclusive access.

The separate initial clock-only probe retains 16/16 host-eligible measurements.
It starts at 95.66 tok/s and 1620 MHz after a pause and ends at 89.75 tok/s and
1453 MHz. Its last peak reported GPU average is 97.62 C. It supports the cadence
observation; its v2 harness source snapshot was not retained byte-for-byte, so
the recorded harness hash and receipt are retained with that provenance limit.
The later v3 and v4 source snapshots are available.

## Idle queue/residency control

The private prototype changes no weights, model state or cache policy. It only
signals/waits on the existing empty GPU queue during the ten-second pause.
Four cycles use plain, keepalive, keepalive, plain order; each then runs four
measured requests. All 16 host screens pass. The first request per cycle is:

| Pause condition | Time to first sample, s | Prefill GPU span, s | Worker CPU time, s | CPU instructions, billions | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain | 0.9958 | 0.2912 | 0.8740 | 17.754 | 95.22 |
| Keepalive | 0.3056 | 0.2921 | 0.1765 | 2.940 | 95.26 |
| Keepalive | 0.3014 | 0.2921 | 0.1721 | 2.939 | 95.24 |
| Plain | 1.0113 | 0.2932 | 0.8865 | 17.675 | 95.01 |

Mean first-sample time is 1.0035 seconds for plain pauses and 0.3035 with queue
signals: 0.7000 seconds avoided. Each keepalive pause makes 20 signals, totaling
only 0.00045–0.00055 native seconds. The GPU remains mostly idle during both
pause conditions. The worker's Darwin background flag is 0 before and after
these requests; the slow wake is not evidence that it became a background task.

This control separates idle latency from sustained decode loss. It does not
qualify a production keepalive policy, its energy cost, or a driver-internal
function as the precise source of the cold cost. The private protocol patch is
archived for reproduction; production does not accept this operation.

## One-model projection crossover

A separate private worker retains all three projection routes in one binary
and one loaded model: original generic/shared-after (0), fixed gate/shared-after
(1), and fixed gate/shared-first (2). The generic shader is copied verbatim from
the baseline. Every serialized variant setter requires reset model position and
clears retained prefix state before the next request. No state is reused across
a variant change. The official policy is unchanged.

Six condition warmups and six baseline MTP-on settling requests precede eight
groups. For each candidate and MTP route, one ABBA group and one BAAB group
compare against their own adjacent baseline arms. Group order is shuffled with
seed 42. All 32 measured arms are retained. Two candidate MTP-off arms fail the
external CPU screen (276.2% and 243.4%; one also reports thermal state fair).
All MTP-on arms pass. No compiler, paging or external model CPU activity is
observed. Timings remain diagnostic without quiet admission.

| Route | Candidate | Change vs own baseline, group 1 | Group 2 | Geometric mean |
| --- | --- | ---: | ---: | ---: |
| MTP off | Fixed gate | -1.65% | -0.60% | -1.13% |
| MTP off | Fixed gate, shared first | -1.99% | -1.29% | -1.64% |
| MTP on | Fixed gate | +1.65% | -0.40% | +0.62% |
| MTP on | Fixed gate, shared first | +2.86% | -1.12% | +0.85% |

MTP-on changes reverse sign between the two groups. MTP-off provides no win
and includes failed candidate host screens. These small samples do not establish
a reliable gain or a confirmed regression. Withdrawal follows the absence of
an accepted end-to-end benefit. It is not a claim that the earlier cross-worker
5% difference is caused by the candidate.

## Instrumentation and reproducibility

The [source archive](../archive/experiments/2026-11-07-flashnext-moe/README.md)
contains independent worker patches, source bases and the MIT-licensed macmon
interval patch. Private worker operations are prototypes, not product APIs.
All raw receipts, native profiles, executables and dense clock samples remain
under ignored `runtime/flashnext-moe-2026-11-07/`. Public data contains aggregate
measurements and hashes only. Earlier records are preserved separately.

The engine addition reads the existing process-local union of completed GPU
command spans around prefill, MTP draft, verify and finish. It records fixed
aggregate counters only when native profiling is enabled. It changes no kernel,
precision, model state or synchronization. These runs use
`STARK_NATIVE_PROFILE_OUTPUT`, without GPU stage timestamp writes. Profiling
and host monitoring add work; they are not mixed with earlier unprofiled timing
claims. The telemetry grid uses frozen cadence v3; the idle control uses v4;
the single-model crossover has its own v1 harness.

The rootless sampler uses macmon 0.10.0 at source revision
`5df67a54e082e0b30a0aa2b48b7c7f78a36f1795`. Its temporary patch brackets actual
IOReport source reads; original emission timestamps are not used as interval
boundaries. The Python join accepts only source intervals fully inside the
request or approximate decode window, retaining duration, coverage and boundary
uncertainty. At 500 ms sampling, decode coverage is partial; the receipt records
each request's coverage. Frequency is active-residency-weighted GPU performance
state MHz. GPU temperature is the reported average of valid GPU sensors, not
the hottest junction. No privileged `powermetrics` cross-check was available.

The production cadence harness accepts optional `--clock-sampler`,
`--process-metrics` and `--native-profile-output` arguments. The idle flags and
`benchmark_qwen38_moe_crossover.py` require their respective archived private
workers. Exact source hashes, artifact guards, workload options, ABBA plans,
native profile hashes and executable identities are in the aggregate receipt.

For a telemetry-grid cell, set paths to the exact artifact, independently built
worker and patched sampler, and choose a fresh private output directory:

```sh
uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_qwen38_cadence.py \
  --model "$MODEL_DIR" --worker "$WORKER" \
  --corpus vendor/lily-core/src/qwen4exp/model.rs \
  --weights-receipt "$WEIGHTS_RECEIPT" \
  --cycles 2 --consecutive 8 --rest-seconds 10 \
  --clock-sampler "$CLOCK_SAMPLER" --process-metrics \
  --native-profile-output "$STARK_DIAGNOSTIC_RUN/native.jsonl" \
  --output "$STARK_DIAGNOSTIC_RUN/cadence.json"
```

Use the frozen v3 harness for an exact telemetry-grid source replay. The v4
idle control changes to four cycles of four requests and adds
`--idle-keepalive-ms 500 --alternate-idle-keepalive`, requiring the idle worker.
The crossover harness uses `--repetitions 2` with its separate private worker;
it fixes the workload and constructs the shuffled ABBA/BAAB plan itself.

## Validation of the restored engine

The restored release generator builds with Rust 1.98.1 and locked dependencies,
SHA-256 `6f54108f8271feb1fa8922b4cfd71211fe3b5e72ef9d2a4f4676faf2759186b1`.
The executable contains neither private diagnostic operation nor the withdrawn
fixed-geometry kernel name. Restored MoE source hashes match baseline bytes.

- Full physical-machine engine suite: 817 passed, 91 ignored, zero failed;
  nine integration tests and three doctests pass. The first build attempt could
  not find existing CMake through the shell PATH; retrying with
  `/opt/homebrew/bin` in PATH succeeds without dependency changes.
- Restored-source model-bound prefix checkpoint control: one passed. Capturing
  and restoring the recurrent prefix matches fresh-prefill readouts bit-for-bit
  on its bounded fixture. No new state-affecting production path is introduced.
- Model-free Python harness/telemetry checks: 76 passed. Repository policy,
  pinned Rust formatting and whitespace checks pass. Archived worker patches
  and the macmon patch reverse-check against their measured source copies.
- Restored-source model smoke: eight measured requests plus two warmups pass
  artifact, executable and same-route golden-hash checks; all eight host screens
  pass. Profiling is disabled. Continuous pooled throughput is 81.21 tok/s with
  MTP off and 84.99 with two drafts. Timing remains unqualified without quiet
  admission. This is a restored-path smoke, not a new candidate comparison.

The new receipt records validation log hashes and smoke aggregates separately
from the 128 diagnostic measurements. Original candidate checks remain bound
to their original sources and executables.
