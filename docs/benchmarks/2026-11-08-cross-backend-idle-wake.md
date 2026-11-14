# Cross-backend idle wake follow-up

Date: 2026-11-08. Production source base:
`640a6a7b6375c54e3cb87e12952bd4c7c77af464`.
This extends the [Lily generation fix](2026-11-07-flashnext-idle-wake-fix.md).
Its measurements and the earlier sustained-throughput diagnosis retain their
original source bindings.

Empty queue maintenance removes the measured official-generation wake penalty
of about 0.81–0.82 seconds and the smaller dense penalty of 0.08–0.13 seconds.
Fresh scoring also falls to about 0.30 seconds, with one host-coverage gap in
its primary latency comparison. Complete-vocabulary, cache, prefix-restoration
and lifecycle controls pass. The report retains every rejected host screen.

## Implementation

Official mixed-affine Flash-Next generation now uses the existing Lily empty
queue maintenance. Flash-Next scoring and the digest probe poll for input on
the owner thread and maintain that same queue every 500 ms between complete
request frames. Successful ticks leave the immediately consecutive score cache
and its counters unchanged. Read-ahead bypasses polling; EOF clears the cache.
Queue/input failures discard retained state and terminate for reload.

Converted text generation maintains its Metal4 queue at the same interval.
The engine lazily creates one private shared event, signals increasing values
on its existing command queue, and waits at most five seconds. This submits no
model work or command buffer and changes no model allocation, precision,
sampling, KV cache or recurrent state. A queue failure ends the worker. A
request arriving during a stalled signal wait can wait until that timeout.
Vision wrappers, dense scoring and other native hybrid backends retain their
existing scheduling. No dependencies, shaders or quantization policies change.

Generation retains the original ten-minute prefix expiry, including time spent
inside maintenance. The guarantee covers an awake worker waiting for another
complete frame. System suspension and a stalled partial frame can still leave
the queue idle. Maintenance stops with worker shutdown and adds no production
sleep assertion or background thread.

## Matched protocol

The predeclared sequence is baseline/candidate/candidate/baseline. Each block
runs dense generation, official generation and Lily scoring serially, with
separate loaded workers and build targets. Two additional baseline/candidate
digest sessions check complete vocabulary-logit hashes. Generation has one
warmup for each MTP route, a ten-second quiet admission, a fixed ten-second
pause, then post-idle and warm requests. Scoring resets its cache before every
fresh native call and additionally checks a deliberately identical cache hit
after idle. Those sub-millisecond cache controls establish correctness and are
excluded from sampled host timing qualification.

All final arms are retained. Quiet admission requires ten continuous seconds
with external CPU at most 150% of one core, each external process at most 80%,
external model CPU at most 10%, device GPU utilization at most 5%, and thermal
state at most 1. Each primary request retains its own host screen, paging
deltas and process attribution. No builds, tests or other model jobs run in
the timed window. The concurrently active recommended-model benchmark chat
uses the same exclusive advisory lock; ownership includes model teardown.

Workers use Rust 1.98.1 and locked dependencies; scorer builds enable
`optimized` separately from generator builds. Source and executable hashes,
the production patch and aggregate reports are sealed by the
[public receipt](data/idle-wake-cross-backend-2026-11-08.json).
GPU stage profiling is disabled. The hardware is Mac17,7, M5 Max/40 GPU cores, 128 GiB physical
memory, macOS 26.7.2, AC with its existing High Power preference.

| Target | Artifact and revision | Weight payload |
| --- | --- | ---: |
| Official Flash-Next generation | `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`, `788c07a8725642bccf13a0a3b8a21954ab84b4f2` | 101 shards plus ngram table; 107,303,406,192 bytes |
| Flash-Next scoring/digest | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, `0f94842e3682b49acace4daeb4cf2e65a6f94f83` | 55 shards; 106,905,155,192 bytes |
| Dense Metal4 generation | `Jundot/Qwen3.8-27B-oQ4e-mtp`, `7dd276b77fcb27c982ebd8d8a434539345713fea` | One converted native weight payload; 17,078,038,512 bytes |

The dense target is the previously qualified `coarse3` conversion with its
three-bit draft-only readout. Configuration SHA-256 is
`9385bac7134505d9da94d048797e7ea03583c1d8c4eb06ce9d13cf778bb395b1`.
Its weight file is the same unchanged inode that the coordinated benchmark
freshly hashed. No payload rehash or model conversion is repeated here.

Fresh content hashes from the coordinated benchmark's fingerprint pass match
the pinned weight receipts for all 102 official payloads, 55 Lily shards and
the dense payload. Installed official metadata is not claimed byte-identical
to its old Hugging Face snapshot. The current metadata hashes are separately
sealed. Full file-identity tuples and metadata hashes are checked before and
after every session; private fingerprint maps remain ignored.

All targets use 512 tokens from the source corpus frozen at `753884b8`, SHA-256
`33dcb30e762d63db4f16aaef21034fbaab93dd56dfd9b555e6da38f5db3238e3`.
Generation samples up to 256 outputs at context 768, temperature 1, top-k 20,
top-p 0.95 and seed 42, with zero or two MTP drafts. Readiness and per-request
acknowledgements verify each route. Scoring uses four fixed candidates,
temperature 1.22 and context 768. Token and score vectors exist only in memory;
published identities are hashes. Each artifact is compared against itself.

## Results

All 14 planned sessions pass correctness and EOF controls. Same-route output
fingerprints match across baseline and candidate builds, including scoring's
complete vocabulary-logit digest. All artifact, source, executable and
configuration guards pass. These are bounded fixture checks, with seeded
generation and one scoring sequence; they establish no general quality claim.

The primary campaign retains all 40 latency requests and four digest requests.
**38/44 host screens pass**: 35/40 latency requests and 3/4 digest requests.
Two dense requests and one official warm control overlap external CPU bursts.
Three short scoring/digest calls have no captured host observation and remain
unqualified. First-response screens pass for 8/8 official requests, 3/4 scoring
requests and 7/8 dense requests. The receipt keeps whole-session timing flags
separate from these endpoint counts; it does not declare the entire campaign
timing-qualified.

All-arm medians in seconds, two observations per worker/route/position:

| Endpoint | Baseline after idle | Candidate after idle | Candidate warm control | First-response screens |
| --- | ---: | ---: | ---: | --- |
| Official generation, MTP off | 1.1226 | 0.3174 | 0.3127 | 4/4 |
| Official generation, two drafts | 1.1305 | 0.3131 | 0.3121 | 4/4 |
| Lily scoring, fresh native call | 1.0810 | 0.3028 | 0.3001 | 3/4; one candidate lacks coverage |
| Dense generation, MTP off | 0.7267 | 0.6328 | 0.6341 | 3/4; one candidate has CPU interference |
| Dense generation, two drafts | 0.7438 | 0.6346 | 0.6301 | 4/4 |

Generation measures native time to the first sample; scoring measures the
complete client-observed fresh score call. Official generation avoids about
0.81–0.82 seconds after idle in both MTP routes. The scoring observation is
consistent with the same improvement, with limited host coverage for one fast
candidate call. All scoring cache hits retain identical fingerprints and the
same cumulative native-call count across idle. The earlier
[sustained GPU slowdown](2026-11-07-flashnext-sustained-diagnosis.md) remains
unresolved; the endpoint here is latency after a fixed idle pause.

A separate dense-only ABBA repeat is triggered solely by the failed host
screens. It keeps the same source, workers, artifact, input, sampling and idle
pause, with 64 output tokens and context 576 to fit a six-minute reservation.
The original 256-output campaign remains intact. This additional control
measures first-response latency with its own output fingerprints and timing
qualification.

The repeat completes in about 253 seconds. All four sessions pass correctness,
source/artifact guards and clean EOF teardown; **16/16 host screens pass**.
The two post-idle observations per worker/route give these median seconds:

| Dense repeat | Baseline after idle | Candidate after idle | Seconds avoided | Candidate warm |
| --- | ---: | ---: | ---: | ---: |
| MTP off | 0.7057 | 0.6229 | 0.0828 | 0.6256 |
| Two drafts | 0.7542 | 0.6247 | 0.1295 | 0.6278 |

This qualifies removal of the smaller dense wake penalty for the named
512/64 workload. It is separate from the 512/256 observations above and adds
no decode-throughput claim. The full-vocabulary scoring control independently
measures 1.1091 versus 0.3032 seconds after idle, with both first-response host
screens passing; its subsequent baseline warm request lacks coverage.

Each build/target contributes about 40 seconds of recorded fixed idle pauses.
Duration-weighted values below are baseline / candidate:

| Target | Worker CPU, % of one core | GPU active, % | Reported GPU watts | Covered clock seconds |
| --- | ---: | ---: | ---: | ---: |
| Official generation | 0.0022 / 0.0334 | 0.344 / 0.660 | 0.0032 / 0.0045 | 36.91 / 37.41 |
| Lily scoring | 0.0023 / 0.0316 | 0.343 / 0.823 | 0.0025 / 0.0045 | 37.43 / 37.46 |
| Dense original campaign | 0.0017 / 0.0348 | 0.175 / 0.906 | 0.0016 / 0.0070 | 37.39 / 37.40 |
| Dense 64-output repeat | 0.0014 / 0.0346 | 0.362 / 0.732 | 0.0023 / 0.0050 | 37.95 / 38.44 |

The sampler joins only fully contained source intervals; GPU coverage remains
partial. Rootless IOReport GPU power is descriptive and was not cross-checked
against privileged powermetrics. These short AC windows do not qualify battery
life or long-duration total-system energy. Maintenance continues while the
worker remains loaded, including before the first request and during long idle
periods.

## State, lifecycle and source controls

Candidate generation separately captures, restores after idle and recomputes
a fresh prefix reference in each MTP route for both generation artifacts.
Scoring compares candidate-output fingerprints and complete vocabulary-logit
digests across builds, and checks that a post-idle hit increments no native
call counter. Every session closes stdin and requires successful EOF teardown
without terminate/kill fallback.

The native generation suite passes 127 tests with two private benchmarks
ignored. Scorer and digest each pass 30 tests; the separate probe passes one
test. Five new scorer tests cover timed maintenance, bit-preserving cached
replies, buffered lines, EOF, input/queue failures and oversized input.
The benchmark and telemetry suite passes 114 tests, including twelve generic
idle-harness tests. Pinned Rust formatting and repository policy checks pass.

Preliminary tooling attempts remain separate from the final comparison:

- Harness v1 stopped in official generation warmup because it expected flat
  MTP counters; the hybrid worker emits them under `mtp.stats`. No measured
  requests completed.
- Harness v2 completed one official baseline session, then scoring failed
  without a durable partial result. The v3 harness treats very short cache
  hits as correctness controls, marks any unsampled fresh request ineligible,
  and writes aggregate checkpoints after every stage. The completed v2
  official baseline remains a separate observation, outside final ABBA.
- The first v3 attempt timed out quiet admission before any measured request
  while an external Docker scan used about one CPU core. The owning benchmark
  chat stopped its specific scan. The final sequence restarted in full with
  unchanged source and workers.
- A second v3 sequence completed official and scoring baselines, then the
  selected historical dense `long3` fixture failed strict config admission:
  it contains an unsupported experimental `mtp_config.max_drafts` field.
  Cleanup of its exiting process group raised macOS `EPERM` and masked that
  startup rejection. The harness now reaps an exiting leader before signaling
  its remaining group, including the short exit race. A real subprocess
  regression check preserves the original static startup error. Harness v4
  adds setup-stage checkpoints and numeric errno diagnostics. The final
  sequence uses the admitted `coarse3` artifact, which has identical weights
  and target configuration with that unsupported field absent. Its metadata
  is sealed separately; both release workers pass direct startup/EOF admission
  before the timed sequence. Earlier complete baselines remain separate.

Raw receipts, builds, partial aggregate checkpoints and native artifacts stay
private under ignored `runtime/idle-wake-cross-backend-2026-11-08/`.

## Reproduction

Build the baseline and candidate in distinct target directories, pin the
corpus bytes above, prepare a private full-hash artifact receipt, and acquire
the shared benchmark lock before loading a model. Run the three targets in
each ABBA block and keep all arms. For one cell:

```sh
uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_idle_wake.py \
  --endpoint generation --model "$MODEL_DIR" --worker "$WORKER" \
  --artifact-receipt "$ARTIFACT_RECEIPT" --corpus "$FROZEN_CORPUS" \
  --screened --rest-seconds 10 --output "$PRIVATE_RUN/cell.json"
```

Use `--endpoint score` for the latency/cache comparison and `score-digest` for
the full-vocabulary control. Add `--prefix-control` to the first candidate
generation session for each artifact. The optional pinned `--clock-sampler`
records covered idle GPU activity/power intervals. New tooling or source
requires a separately identified run.
