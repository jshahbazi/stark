# Flash-Next optimization pilot

Date: 2026-11-07. Scope: Qwen3.8-Flash-Next Lily Q4 XL only.
Engine base: `464049330d472f970decc6591692f7d31cb51590`.

Disposition: the harness pilot completed, with 18 cells and 144 measured
requests passing the bounded correctness checks. No performance result is
qualified because quiet admission failed and the subsequent measurements ran
with host observations only. The 64-row encoding candidate is not promoted.
The production encoder cutoff remains 16 rows.

## Objective and experimental boundaries

This pilot extends Stark's existing paired MTP benchmark into a bounded
experiment runner. It screens draft depths one through three, compares an
isolated encoding-policy candidate, and collects separate stage diagnostics.
The [Isoquant Inference Engineering Benchmark](https://isoquant.ai/benchmark)
motivated the bounded-task structure and the requirement to measure complete
requests after checking correctness. No Isoquant implementation is imported.

The checkpoint is `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`. All 55 installed weight shards,
106,905,155,192 bytes, were rehashed against the pinned upstream LFS manifest
before execution. File size, nanosecond mtime and inode guards bind the subsequent
experiments to those verified payloads. Configuration, corpus, executable and
harness hashes are recorded separately. This artifact does not qualify the
official mixed-4/8-bit pack or dense Qwen3.8-27B.

The host was an Apple M5 Max with 128 GiB unified memory and macOS 26.7.2. Exact hardware/OS
identity, build hashes and results are in the
[public aggregate record](data/flashnext-optimization-pilot-2026-11-07.json).

The precision policy is unchanged: BF16 activations, Q8 attention cache, F32
reductions and recurrent state, with the Lily head's existing F32 readouts.
No weight requantization or sampling implementation change is involved.

## Harness

`scripts/benchmark_qwen38_pilot.py` calls
`scripts/benchmark_qwen38_mtp.py` for each declared input-length/draft-depth
cell. Each cell owns one loaded worker and executes one warmup per route,
then randomized ABBA pairs of MTP off and on. Decode rate counts sampled steps
after the first token. Wall time includes prefill, decoding and worker transport;
loading and warmup are excluded. Tokenization is outside the timed requests.

The pilot removes inherited hybrid diagnostic environment settings. The paired
harness records helper-source hashes and checks that execution settings remain
unchanged. Screened runs use the existing quiet-admission and host-screening
helpers. Both arms of every pair must pass the host screen for the cell to be
eligible; arms are never selectively dropped. Ineligible observations remain in
the receipt. Runs have fixed work and no automatic measurement retries.

The runner checks output count, route acknowledgement, exact-rejection verifier,
MTP counters and deterministic hashes within each route. A scheduling candidate
must also match the baseline's output hashes at the same draft depth and input.
MTP on/off trajectories may differ because rejection sampling consumes random
numbers differently. These checks do not establish whole-model task quality or
equivalence to another runtime. They are suitable for this scheduling-only
candidate; a new precision policy would need separate numerical qualification.

Candidate-worker comparisons use four loaded sessions in baseline/candidate/
candidate/baseline order, reversing cell order on alternate sessions. This
reduces order bias but does not remove every source of machine drift. Four
paired blocks per screening cell are exploratory evidence. An independent
workload is required before proposing promotion.

The completed grids used the harness at commit `f5750888`; their receipts
retain its exact source hashes. The final tooling also strips the generic
`STARK_NATIVE_PROFILE_OUTPUT` alias, with separate model-free regression tests.
Both profile-output aliases were absent during the measured grids, so that
defensive follow-up does not change their interpretation. Existing historical
harness records remain separate.

## Candidate

The isolated candidate changes `CONCURRENT_ROWS_MAX` from 16 to 64 in
`vendor/lily-core/src/qwen4exp/model.rs`. Batches through that threshold use the
existing concurrent encoder and explicit dependency barriers. Larger batches
retain serial encoding. This affects the complete batched graph, including
expert projections, and cannot establish an isolated MoE-kernel speedup.
Speculative verification has at most four rows and retains its existing encoder
selection in both builds.

Baseline and candidate were built with Rust 1.98.1, locked dependencies, and
separate Cargo target directories. The baseline has no engine-source changes
from the named base commit. The candidate's only engine-source change is the
threshold patch. The production source retains its original cutoff while the
experiment is evaluated.

The [candidate patch](../archive/experiments/2026-11-07-flashnext-pilot/concurrent64.patch)
is preserved for reproduction. The baseline executable SHA-256 is
`7a5d3f33b39c54515963fe81d9547b21a3a80d9def0a717201de75b96fa23380`;
the candidate is
`8e5dde3bc4dff401e566f1e9b489c16a8580115ec38f64b937f44f085797421e`.
The build receipt binds both to the named engine base, the one-line patch,
locked release command and separate build directories.

The initial build receipt captured the local diff display formatter. Final
artifact checks regenerated an applyable unified patch with external diff and
text conversion disabled, and verified reverse application against the built
candidate source. The v2 build receipt retains the initial display hash, the
portable patch hash and the unchanged executable hashes. No source or binary
was changed by that receipt repair.

## Results

All completed grids use stochastic sampling, temperature 1, top-k 20 and
top-p 0.95. Every cell has four paired blocks and two warmups. The depth and
encoder grids use seed 42 and the exact corpus-derived prefixes from
`vendor/lily-core/src/qwen4exp/model.rs`, with 64 sampled outputs. The held-out
case uses `vendor/lily-core/src/qwen4exp/config.rs`, seed 7, 48 input tokens and
128 sampled outputs. Context capacity is input plus requested output tokens.
These raw code prefixes are synthetic throughput workloads, not chat-quality
evaluations.

### Host conditions

Two initial attempts stopped before collecting measured pairs. The first
version retained only a generic failure code. A smaller diagnostic after
adding sanitized partial receipts established `quiet_admission_timeout`:
205 host samples observed external CPU between 176.75% and 639.99%, compiler
activity, and GPU utilization between 0% and 94%. Both generation warmups
completed successfully before that timeout.

The complete grids then used explicit `--observe-host`, which records the same
request screens without waiting for quiet admission and always leaves timing
unqualified. Of 144 measured arms, 84 failed their request-time host screen.
All observations remain in the aggregate record; no arm was removed to improve
a result. Paging and compiler activity are host observations and are not
attributed to Stark. No qualified speedup or general throughput floor follows
from this run.

### Draft-depth screening

These are observations under the host conditions above. Decode rates pool the
four measured requests per route. Wall speedup is the median of four within-pair
off/on wall-time ratios; it is a different statistic from pooled decode rates.

| Input tokens | Drafts | Off decode tok/s | On decode tok/s | Median paired wall speedup | Decode wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 81.55 | 92.75 | 1.116x | 4/4 |
| 32 | 2 | 72.62 | 78.81 | 1.079x | 4/4 |
| 32 | 3 | 78.75 | 66.42 | 0.856x | 0/4 |
| 512 | 1 | 76.92 | 89.04 | 1.109x | 4/4 |
| 512 | 2 | 76.73 | 90.35 | 1.123x | 4/4 |
| 512 | 3 | 78.01 | 81.15 | 1.070x | 3/4 |

All six cells passed route acknowledgements, requested output counts and
within-route repeatability, including warmups. Three drafts were slower for
the shorter prefix; one and two drafts are candidates for a quiet-window
rerun. The observations do not justify an online controller or a changed
production recommendation. Baseline rates vary between cells, which reinforces
the need for paired controls and independent confirmation.

### Encoding-policy comparison

Each row below pools eight measurements per worker from its two ABBA sessions.
Times are median complete-request wall seconds. The ratio is baseline divided
by candidate, so values above one favor the 64-row candidate.

| Input tokens | MTP drafts | Baseline seconds | Candidate seconds | Observed ratio |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0 | 0.8958 | 0.9167 | 0.977x |
| 32 | 2 | 0.8097 | 0.8824 | 0.918x |
| 64 | 0 | 0.9348 | 0.9409 | 0.993x |
| 64 | 2 | 0.9746 | 1.0091 | 0.966x |

All eight cells passed, and every same-route output hash matched between the
two executables. Prefill-only median ratios ranged from 1.004x to 1.023x;
the complete requests showed no improvement. Host interference prevents
attributing the observed decode differences to this prefill cutoff. The
candidate has no demonstrated benefit and is excluded from production.

The held-out corpus/seed case added four cells and 32 measured requests. All
same-route hashes matched across baseline and candidate, with 128 sampled
outputs per request. Its timing remained unqualified. This extends the bounded
scheduling-correctness evidence and does not establish general answer quality,
prefix-cache qualification, or a performance win.

### Separate stage diagnostic

A fresh baseline worker ran three 64-input/32-output requests with aggregate
stage timestamps enabled and MTP off. The first was warmup; both warm diagnostic
requests repeated the same output hash. No instrumented rate is used in the
throughput tables above.

Across the 62 warm decode passes, the recorded GPU command span totaled
0.9045 seconds. Its coarse timestamp shares were:

| Component | Share of profiled command span |
| --- | ---: |
| MoE | 35.1% |
| GDN | 24.1% |
| Hyper-connections | 20.8% |
| QSA attention | 10.8% |
| Vocabulary head | 8.3% |

The expert gate/up plus shared-down group accounts for 16.8% of the command
span; expert down/combine accounts for 9.3%. GDN input projection/convolution
accounts for 13.1%. These identify useful targets for another bounded task.
They are instrumented timestamp shares, not GPU utilization or measured
end-to-end gains. Coarse GDN/MoE counters include their detailed children;
adding parents and children would double-count those intervals. Multiple
counter calls per pass do not imply additional decode passes.

## Outcome and validation

The reusable runner now supports bounded grids, candidate ABBA sessions,
independent output-hash comparisons, artifact/binary guards, sanitized partial
failures and explicit observed-versus-screened host modes. Reports contain
hashes, counters and aggregate timings only. There is no automatic promotion.

The next useful engine experiment is an actual-shape gate/up/shared-down
dispatch candidate, followed by numerical checks and a quiet, matched
whole-request comparison. A more aggressive global concurrency cutoff has
no demonstrated benefit in this pilot. The existing MTP recommendation and
precision policy are retained.

Validation after the tooling follow-up: 60 tests passed across the paired MTP,
pilot and neighboring context harness suites. Coverage includes route/count
validation, candidate output regressions, failed host screens, observation-only
qualification, payload identity changes, profiler-alias restoration, sanitized
failure receipts, bounded grid validation and report privacy. Repository policy
and whitespace checks passed. Neither production engine sources nor dependency
locks changed.

## Reproduction

Use the exact artifact and verified payload receipt named in the aggregate
record. Run from the repository root with the locked converter environment:

```sh
uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_qwen38_pilot.py \
  --model /absolute/path/to/prepared-lily-checkpoint \
  --worker /absolute/path/to/baseline/stark-generate \
  --corpus vendor/lily-core/src/qwen4exp/model.rs \
  --weights-receipt runtime/flashnext-pilot-2026-11-07/verified-weights.json \
  --input-lengths 32 512 --draft-depths 1 2 3 \
  --tokens 64 --blocks 4 --screened \
  --output runtime/flashnext-pilot-reproduction/depth-screen.json \
  --cells runtime/flashnext-pilot-reproduction/depth-cells
```

For a scheduling comparison, provide `--candidate-worker` and use the candidate
comparison's input lengths and draft depth. The output and cell directory must
be new. Raw model data, prompts, responses, token vectors and probability data
are never written by this harness. Native receipts and binaries remain ignored;
public results contain aggregate timings, counters and hashes.

The encoder comparison used `--input-lengths 32 64 --draft-depths 2`; the
held-out case used `--input-lengths 48 --draft-depths 2 --tokens 128 --seed 7`
and the config corpus above. All completed grids used `--observe-host` in place
of `--screened`. A qualifying rerun must restore `--screened` and obtain complete
eligible pairs. Native receipts are retained under the ignored
`runtime/flashnext-pilot-2026-11-07/` directory. The public aggregate record seals
each receipt's SHA-256, executable identities, corpus/input/output hashes,
sampling settings, context sizes and per-cell host-screen results.
