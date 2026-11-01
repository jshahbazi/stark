# Qwen3 long-prompt timing attempts — 2026-11-01

**No demonstrated speedup and no accepted timing window.** The latest completed
three-cycle run observed a median **6.117 seconds to first text for the baseline
and 6.218 seconds for the candidate**. Its CPU guard failed, so these are
descriptive measurements from a CPU-guard-rejected window, not a qualified regression.
The candidate remains uninstalled.

This is a separate measurement record after the
[candidate qualification study](QWEN3_LONG_PREFILL_CANDIDATE_2026_11_01.md).
It does not supersede that study's numerical evidence or erase its earlier
unexplained generation mismatch. No runtime or measurement-controller source
changed for these attempts.

## Fixed inputs and builds

- Baseline source: `36e855ee5bca4ce8c1bd585173bad44cac04b7a1`.
- Candidate source: `c5071d102d8ee3d9f9b5a324b50eeef19c5e038c`.
- Candidate native source-tree SHA-256:
  `fe9c4736fa37c40a5b09359a05c1ae97368f067ae0474805aa5efed650f5151b`.
- Baseline worker SHA-256:
  `facebd20b6a9b1242b16e2fb4b5c64e1971f99a98d5b80f9a14f8b83601cad81`.
- Candidate worker SHA-256:
  `e4f679005ecc1a78f319fd413abc5ecd461be7e9a43092fd1fac89cba9aae83b`.
- Controller SHA-256:
  `d842559920065e3926c14a9de899a6f3d1ccd7949f8daf5f0b0ed56daf7cc559`.
- Model: `mlx-community/Qwen3-4B-4bit`, revision
  `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`, MLX affine 4-bit/group-64.
  Weights SHA-256:
  `7c2af9a45e5b0d196f06a5cbfd6fc5cc95adf29df79885d0540f2206fb3d40d8`.
- Host: Apple M5 Max, 128 GB, macOS 26.7.1 (25G241). Rust 1.98.1 workers;
  locked uv/Python 3.13.13 measurement environment.

The workload preserves **18,544 tokens, eight complete tools, five history
messages and one linked tool result**, rendered by the pinned model's chat
template. Context capacity is 40,960. Timing requests generate one token;
separate generation-parity requests generate up to 32. Every request uses fresh
native state. A 49-token chat is an independent control.

These are native-worker measurements through private pipes, not OpenWork UI
measurements. `prefill_seconds` is the worker's wall-clock first-sample proxy,
including state creation, stream setup, the first GPU wait and possible
lookahead encoding. `first_text_seconds` also includes private-pipe delivery
and decoding before the first nonempty text callback. Neither is isolated GPU
kernel time.

## Observations, all unqualified

Each ABBA cycle executes baseline, candidate, candidate, baseline serially with
both workers loaded. Loading and warmup are excluded. Values below are medians
of individual samples within each attempt; different windows are not pooled.

| Attempt | Protocol | Samples per build / fixture | Baseline first text | Candidate first text | Rejection |
| --- | --- | ---: | ---: | ---: | --- |
| 01 | One load, three ABBA cycles, then full parity/lifecycle checks | 6 | 6.308 s | 6.358 s | CPU and thermal guards |
| 02 | One load, three ABBA cycles, timing only | 6 | 6.117 s | 6.218 s | CPU guard |
| 03 | One load, one ABBA cycle, timing only | 2 | 5.607 s | 5.739 s | CPU guard |
| 04 | Quiet admission | 0 | — | — | `quiet_window_timeout` |

Native first-sample medians were respectively 6.307155 → 6.357561,
6.116210 → 6.216848 and 5.606077 → 5.737762 seconds. The candidate's observed
latency was higher in every completed cycle. Host screening rejected every
window, so this does not establish the size or cause of a regression.

There is substantial drift. Attempt 01's three baseline/candidate cycle
medians were 5.627/5.825, 6.276/6.358 and 6.751/7.386 seconds. Attempt 02's were
5.554/5.840, 6.094/6.217 and 6.200/6.299. In attempt 03, the opening and closing
baseline samples spanned 5.073–6.139 seconds. Apparent short-chat gains also
have strong opening/closing order effects; the 49-token case does not enter
the changed materialization route and supplies no evidence of an optimization.

After attempts 01–02, a three-load protocol was declared before collecting new
data: three separate one-cycle blocks, fresh warmup and quiet admission for
each, with all three required to pass before pooling six samples per build.
Attempt 03 failed its CPU guard; attempt 04 timed out before retaining any
samples. A separate Stark inference worker was then observed running from the
GLM optimization worktree after the final quiet-admission timeout. Attempt 05 was
not started. This protocol is
**incomplete**, with zero accepted blocks and no qualified pooled result.

## Checks and limits

Completed attempts 01–03 preserve identical source, worker, model, tokenizer,
configuration, rendering, controller and helper bindings through their end
checks. All one-token timing responses match. Attempt 01 additionally passes
all ten full-generation/stop comparisons and both workers' cancellation,
context-overflow and exact-recovery controls. Timing-only attempts 02–03 do
not repeat those full controls.

For attempts 01–03, sampling has no failures or excessive gaps; paging and
sleep counters remain zero. There are 122, 123 and 38 fully measured monitor
intervals. The CPU threshold fails in all three, and the thermal threshold
also fails in attempt 01. Attempt 04 retains only sanitized failure metadata;
it has no timing samples or end seals to audit.

Live process observations separately saw Rust compilation during attempt 01
and another Stark inference worker after the final quiet-admission timeout. The JSON
guards do not identify the offending CPU processes; no process identity is
inferred for attempts 02–03. Quiet-window GPU utilization and process/CPU/
thermal screening cannot prove whole-device GPU exclusivity.

An independent reviewer recomputed medians, sample counts, seals and rejection
conditions. The honest outcome is **no observed benefit worth promoting**;
no guarded comparison was accepted. After observing another active inference
worker, no further attempt was started. The
earlier generation concern, Developer ID upgrade checks and real OpenWork
validation remain separate incomplete qualifications. No app was installed or
credential accessed for this measurement.

## Retained aggregate records

Primary JSON records are private, mode 0600, under the isolated checkout's
ignored `runtime/`, named `prefill-timing-20261101-01.json` through `-04.json`.
They contain timings, counts, hashes, booleans and sanitized failure categories;
no prompt, reply, token-vector or credential payloads are retained.

| Attempt | Record SHA-256 |
| --- | --- |
| 01 | `40b80f68dc22155367fb03a43e9f25d9d1be19cd087ebb9d7530211cd4d85d94` |
| 02 | `279eab0da3fe98d604a2ce1a0c32a1342d20abf64213bc0159f2b3709e9b17a3` |
| 03 | `a436eae7021d433b55159c021aa6b0d28b67bf9445641fdee10863c441f015c1` |
| 04 | `fc3b502cb524dcb0ed04778f02dc088415957c0bca7ec09a06df2c6f69735328` |
