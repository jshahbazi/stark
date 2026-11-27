# HY4STQ-SERVE-01 · Two active text slots with shared Hy4 STQ weights

The [current native STQ defaults](2026-11-27-hy4-stq-catalog-defaults.md)
are the single-slot control. A long OpenCode prefill holds the worker while
other agent requests wait. This experiment admits two text requests and
alternates GPU work at prefill and decode boundaries.

## Artifact and controls

| Field | Value |
| --- | --- |
| Artifact | `beamster/Hy4-Flash-Sushi-STQ-4bpw@61528ba635a623f127c5a7b1e1f4ac2c2b0d4885` |
| Artifact identity | `1ed7e181a27f78ffb1e93f9bef2eb00be3c9c5933cf5d7f50494320b45e9a04f` |
| Host and toolchain | M5 Max, 128 GiB, Metal 4, Rust 1.98.1 release, OpenCode 2.4.2 |
| Precision and draft policy | Original STQ weights, F16 KV, 4,096-row prefill and five greedy DFly2 drafts |
| Control | Stark `90054b7e9`, packaged worker SHA-256 `539a68b235378e209172448273d2b92a04a3af933a12aca61f2195bd1eea0669` |
| Candidate | Stark `25b6754c4`, release worker SHA-256 `6d33485155065548d5263f9b18f626345a4a228e8f6847cf24a911794d8043c4` |
| Disposition | Opt-in candidate through `STARK_HY4_CONCURRENT_SLOTS=2` |

The second slot shares the model weight buffers, compiled Metal library,
Engram source, vision tower and DFly2 scratch. Its forward graph owns its KV,
IndexPool and intermediate buffers. The native phase queue lets a request
waiting for a GPU phase proceed ahead of a later phase. Text requests keep a
logical context within their admitted slot capacity. The load may admit one
slot when both context reservations and the second scratch arena do not fit.

The first slot serves image requests exclusively on the existing image path.
The second slot has no retained-prefix capability. Before reusing a primary
prefix after overlap, the desktop clears any state retained during the overlap.
The single-slot flow retains the existing immediately preceding, exact-prefix
policy. Both slots keep separate cancellation channels and output streams.

## Qualification workload

- Use the fresh 16,229- and 27,229-token synthetic review turns from the
  [tuned OpenCode fixture](2026-11-27-hy4-stq-open-code-performance.md)
  with the same 96-token output budget. Start one, two and four independently
  granted OpenCode CLI sessions per window. Isolate client storage and record
  only aggregate timings, counts and hashes.
- Hold the loaded cap and runtime settings equal where both allocations fit.
  Verify loaded capacity, actual request input counts, expected token hashes,
  draft and prefix behavior and which worker slots executed. Preserve the
  warmed one-agent decode rate and target at least 48 useful tok/s at 27K.
- Compare elapsed time to first nonempty API delta, separation between API
  admission and worker execution, worker prefill and useful decode, and wall
  time until all clients finish. Report native wall time including any wait
  for GPU phases. Record measured concurrency and memory pressure.
- Check flexible windows, a cancelled peer, and image exclusivity separately
  from fan-out timing. An isolated transport or model-free fixture does not
  establish real-weight timing or precision.

## Single-agent control

Both loads advertised a 49,152-position API window. The candidate advertised
the same capacity for its second slot. Single-agent calls used only the primary
slot and matched the reference output hashes. The useful rate uses 95 of the
96 sampled tokens and excludes the first sample from decode time.

| Fixture | Serial prefill | Two-slot prefill | Serial decode | Two-slot decode |
| --- | ---: | ---: | ---: | ---: |
| 16,229 tokens | 20.88 s | 20.71 s | 50.72 tok/s | 51.29 tok/s |
| 27,229 tokens | 30.94 s | 30.18 s | 52.78 tok/s | 51.59 tok/s |

Each cell is one observation. The 16K cell was first use and the 27K cell
followed the 16K groups in each load. Startup is excluded. The serial load
crossed a host clock discontinuity before any timed request.

## OpenCode fan-out

All 28 recorded completions matched the existing generated-token hashes
for their fixture. Request counts, rendered input hashes, output budgets and
finish reasons also matched. The raw candidate recorder classified groups
with a second slot as invalid because its readiness disables prefix reuse
and therefore its replies omit the reused-token count. The offline check
accepted omission only on the second slot and required zero reuse on the
primary. Both logs preserve the original evidence.

Times are wall seconds. First delta starts at client launch. Admission delay
is API arrival through dispatch of the worker's generation operation, so it
includes request handling overhead. Rates are per-agent medians including
any wait for GPU phases. Each group is one observation.

| Fixture and agents | All clients finished, serial / two slots | Median first delta, serial / two slots | Median admission, serial / two slots | Prefill tok/s, serial / two slots | Decode tok/s, serial / two slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16K × 2 | 46.64 / 46.29 | 33.48 / 37.69 | 11.61 / 0.35 | 793.89 / 455.32 | 52.00 / 16.63 |
| 16K × 4 | 84.68 / 83.54 | 52.94 / 54.53 | 32.11 / 20.26 | 866.39 / 504.32 | 54.06 / 17.34 |
| 27K × 2 | 73.34 / 72.30 | 52.44 / 64.39 | 17.38 / 0.34 | 804.54 / 433.68 | 49.46 / 16.96 |
| 27K × 4 | 154.16 / 150.13 | 96.15 / 104.88 | 57.10 / 37.58 | 759.45 / 414.55 | 50.40 / 16.90 |

One PID served both slots. At four agents, two requests waited for admission
and the active intervals for the slots overlapped by 80.95 seconds at 16K
and 147.46 seconds at 27K. The last first delta arrived 2.86 seconds sooner
at 16K and 5.90 seconds sooner at 27K. Rendered requests shared the same
long prefill, and the scheduler interleaved their work at 4,096-row
boundaries. Median first delta increased in all four groups. The 0.7–2.6%
decrease in total completion time across groups is too small to attribute
to the scheduler from these observations. Candidate memory-pressure samples
stayed normal through the timed matrix and existing swap use decreased.
These boundary samples do not provide a peak unified-memory measurement.

## Lifecycle and disposition

- The original managed 16K→27K continuation restored 12,288 positions and
  produced the same hash as the zero-reuse replay on the candidate. The
  primary reused 0, 4,096 and 0 positions across request windows of 8,192,
  8,192 and 12,288 with the same output hash. Both slots independently
  produced that hash at 8,192 and 12,288.
- A real image request ran on the primary with drafting off. Text before
  and after the image matched, with reuse of 0, 4,096 and 0 across the three
  text calls. The vision tower stayed loaded during the text matrix.
- The focused transport and Desktop tests covered peer cancellation,
  shared-worker EOF, Stop fencing, single-slot fallback and image admission.
  The wider Python suite passed 640 tests with one skipped. The final
  lease-fencing refinement passed the focused twelve-test rerun.
- Keep the feature opt-in. The median latency increase under matched
  fan-out outweighs the observed change in total completion time. A follow-up
  should compare shorter prefill slices with mixed-length jobs and keep
  single-agent decode and exact output as promotion gates. Measure actual
  cross-request batching separately for throughput.

| Private aggregate receipt | SHA-256 |
| --- | --- |
| Serial OpenCode matrix | `77d4687122282a67b9d4a384897e148e4da4f1d0e43445f274adc032d687c9e3` |
| Two-slot OpenCode matrix and lifecycle | `dff5bdafb993d2121836fabf30b4e19e3d70299a770afa3d68463407d986d802` |
| Comparison with omitted reuse field normalized | `926e0f33aa535847a9fca39ea70615b765d25d4e2190fec40a6566d4733e4aac` |
