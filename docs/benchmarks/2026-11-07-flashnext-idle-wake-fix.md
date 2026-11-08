# Flash-Next idle wake latency fix

Date: 2026-11-07. This implements the queue-maintenance control from the
[idle wake diagnosis](2026-11-07-flashnext-sustained-diagnosis.md).
Production source base: `5f974a939f9fa97d7d5935ea690b82c334e8397e`.
The later [PR 98 main integration](../archive/runs/2026-11-07-flashnext-pr98-integration.md)
records a rebuilt worker and separate post-merge controls. Original ABBA
measurements below retain their `753884b8` source binding. On newer checkouts,
reproduction requires the corpus bytes from that revision because main changed
the source file used as the corpus.

## Implementation

The generation worker now polls for the next request with a 500 ms timeout
when its loaded native model uses the Lily numerical policy. On timeout, the
owner thread submits the existing empty Metal queue signal. The queue remains
warm while the worker waits for input; no model inference is run. The official
mixed-affine policy, GLM, Flint and other generation backends keep their existing
idle behavior. This is enabled automatically for the qualified Lily path.

No background thread, public protocol operation, extra dependency, inference
activity assertion or power-mode change is added. Maintenance runs between
complete requests. Input/EOF wakes the poll immediately. Shutdown stops
maintenance with the worker. A queue fault or input wait failure clears retained
prefix state and exits the worker so the host can reload it.

An idle tick changes no token positions, sampling state, KV cache or recurrent
snapshot. The original ten-minute prefix-cache deadline is measured from the
start of the idle wait, including time spent in maintenance; ticks cannot renew
it. Native model request execution remains serialized. The Metal residency set
was already explicitly requested and attached to the queue; the fix uses the
measured queue signal rather than adding another residency request.

The guarantee covers an awake worker waiting for a complete new request frame.
System suspension and a client stalled while writing a partial frame can still
leave the queue idle. The worker does not prevent system sleep in production.

## Matched validation protocol

The old restored generator and the fixed release generator are built separately
with Rust 1.98.1 and locked dependencies. A predeclared ABBA sequence loads one
worker at a time: old, fixed, fixed, old. Each session warms MTP off and on, then
runs one cycle per route. Each cycle requires ten seconds of host quiet
admission, a further fixed ten-second idle pause, and two measured requests.
The first request measures idle wake latency; the second is a warm control.
All 16 measured requests and eight warmups are retained. No builds or other
model jobs run during the matched window. Host screens are retained per arm.

The artifact is `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`, 55 shards totaling 106,905,155,192
bytes. The prior full-hash receipt is reused with size, nanosecond mtime and
inode guards before/after each session. No new full payload rehash is claimed.

The workload remains 512 corpus tokens, 256 sampled outputs, context 768,
temperature 1, top-k 20, top-p 0.95, seed 42, with zero or two MTP drafts.
The corpus is `vendor/lily-core/src/qwen4exp/model.rs`, SHA-256
`33dcb30e762d63db4f16aaef21034fbaab93dd56dfd9b555e6da38f5db3238e3`.
The host is Mac17,7, M5 Max/40 GPU cores, 128 GiB physical memory, macOS 26.7.2,
AC with the existing High Power preference, verified after the run. Profiling
and GPU stage timestamp
writes are disabled. Benchmark-only `caffeinate -i` prevents system idle sleep.

Cadence harness v5 adds optional quiet admission and alternating MTP routes.
Per-position summaries and hash checks keep each MTP route separate. The
patched rootless clock sampler and process helper record idle CPU time and
GPU power/activity during each fixed pause. Clock joins accept only fully
contained source intervals; coverage remains partial. GPU power comes from
private IOReport APIs and is not cross-checked against privileged powermetrics.

## Results

The [aggregate receipt](data/flashnext-idle-wake-fix-2026-11-07.json) records
every final arm, admission, source/executable hash and idle resource interval.
All **16/16 host screens pass**, with no paging, compiler activity or external
model CPU activity during measured requests. Artifact, worker, configuration
and source guards pass. Same-route output hashes match the existing golden
hashes across all workers, requests and warmups. These are bounded seeded-output
checks; they do not establish general model quality or numerical equivalence.

Median seconds, with two observations per worker/route/position:

| MTP | First sample after idle, old | Fixed | Seconds avoided | Warm control, fixed | Full request after idle, old | Fixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Off | 1.0747 | 0.2965 | 0.7781 | 0.2965 | 4.1618 | 3.3800 |
| Two drafts | 1.0919 | 0.3053 | 0.7866 | 0.2979 | 3.7769 | 2.9886 |

The fixed first-sample latency is close to its warm control. The measured idle
wake penalty is removed in both MTP routes. Post-pause decode is effectively
unchanged: 82.82 versus 82.74 tok/s off and 95.23 versus 95.12 with MTP. This
fix addresses idle latency; it does not resolve the separately documented
sustained GPU execution slowdown. The generic cadence receipt retains
`timing_eligible: false` as its general decode-promotion guard. The accepted
endpoint here is matched first-sample latency after a fixed idle interval.
The harness's private keepalive flags remain off: all fixed-worker queue
maintenance is internal, without diagnostic protocol commands.

Each worker contributes four ten-second idle intervals. Aggregate measurements:

| Idle metric | Old | Fixed |
| --- | ---: | ---: |
| Worker CPU seconds / approximately 40 seconds idle | 0.00073 | 0.01323 |
| Worker CPU, percent of one core | 0.0018% | 0.0330% |
| GPU active fraction, covered source intervals | 0.303% | 0.782% |
| Reported GPU power | 0.0028 W | 0.0062 W |
| Clock interval coverage | 37.95 s | 36.93 s |

The queue stays mostly idle. These short-window, rootless GPU power readings
are descriptive and do not qualify long-duration battery or total-system
energy costs. The policy continues while the model's worker remains loaded,
including before its first request and during long idle periods.

## State, lifecycle and source controls

The fixed release generator has SHA-256
`8c50ec262db31c0aace8c3c47983fccf74ec0a05656a559262ea770a62769f25`.
The old restored generator is
`6f54108f8271feb1fa8922b4cfd71211fe3b5e72ef9d2a4f4676faf2759186b1`.
The [production patch](../archive/experiments/2026-11-07-flashnext-idle-wake/README.md)
has SHA-256 `82b8787f2d32cfd6de1ff4a251304580e5276d24729ac0ea0ffa9098d0986169`.
It changes the adapter/worker scheduling and tests, with no vendored shader or
precision changes. The exact four Rust file hashes are in the receipt.

- Native worker unit suite: 124 passed, two private model benchmarks ignored.
  New tests cover idle ticks with no retained prefix, ready input/EOF, retained
  prefix reuse, expiry during maintenance, input arriving after expiry, and
  queue/input failure cleanup.
- A separate model-bound control uses the same fixed release worker with 4,608
  input tokens and 64 sampled outputs. After ten seconds idle, 4,096 prefix
  tokens are reused. Capture, reuse and fresh-reference requests have identical
  output hashes within each MTP route. All six requests pass; payloads stay in
  memory. This qualifies preservation of existing exact prefix restoration for
  this artifact and bounded fixture.
- Closing worker input while idle exits successfully in 0.503 seconds, including
  model teardown, without terminate/kill fallback. No maintenance thread remains.
- Portable worker configuration: 82 passed. Python benchmark/telemetry tests:
  77 passed. All six real-generator startup boundary checks pass. Repository
  policy, pinned Rust formatting, whitespace and archive reverse-apply checks
  pass. No dependencies or lockfiles change.

A preliminary run was stopped after two complete sessions when review identified
an expiry edge: input becoming ready during maintenance could bypass the prefix
deadline. The final source checks expiry both after maintenance and when input
is readable. The corrected binary restarts the entire ABBA plan. All eight
completed preliminary arms and partial aggregate log hashes remain in the
receipt under a separate source binding; no performance-based removal occurs.

Raw receipts, clock timelines, executables and native validation logs remain
private under ignored `runtime/flashnext-moe-2026-11-07/`. Public data contains
aggregate measurements and hashes. Earlier measurements remain unchanged.

To reproduce one final session with the exact old or fixed executable, choose
a fresh private output directory and the pinned sampler from the prior study:

```sh
uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_qwen38_cadence.py \
  --model "$MODEL_DIR" --worker "$WORKER" \
  --corpus vendor/lily-core/src/qwen4exp/model.rs \
  --weights-receipt "$WEIGHTS_RECEIPT" \
  --cycles 2 --consecutive 2 --rest-seconds 10 --alternate-routes --screened \
  --clock-sampler "$CLOCK_SAMPLER" --process-metrics \
  --output "$STARK_DIAGNOSTIC_RUN/cadence.json"
```

Run the four sessions in old/fixed/fixed/old order with no builds or overlapping
model jobs. Keep each source checkout's build target separate. The receipt binds
cadence v5 and its helpers by hash; new tooling requires a separately recorded run.
