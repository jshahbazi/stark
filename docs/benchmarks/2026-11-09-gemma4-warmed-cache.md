# Gemma 4 warmed cache follow-up and suffix-dispatch fix

Date: 2026-11-09. Attempt: `20261109-G4IP-02`. Apple M5 Max, 40 GPU cores.
Warmed alternating controls exposed a worker dispatch bug: Gemma restored a
1,024-token boundary and then replayed the entire input. Passing only the suffix
fixes duplicated context and makes the measured cache hits materially faster.
Both retained E2B/E4B artifacts also pass nonperiodic near-capacity and image
transition controls. Gemma assistant MTP remains unimplemented; see its
[staged implementation plan](../GEMMA4_ASSISTANT_PLAN.md).

This supersedes the cached-worker qualification in the [original record](2026-11-09-gemma4-image-text-parity.md).
Its short unscoped image-loading measurements retain their original source-bound
scope. Engine ring-copy component tests also remain valid. The old periodic
worker comparisons were insufficient evidence for correct suffix dispatch.

## Fixed comparison before measurement

Use the same existing image artifacts and numerical policies as the original
record. One image-loaded, prefix-enabled worker stays loaded per artifact.
Context is 4,096; sampling is greedy, seed 47, MTP off, output budget eight.
The two original synthetic workloads have 1,792 and 1,824 input tokens, with a
1,024-token retained boundary. Two warmup cycles precede six measured cycles.
Cycles alternate fresh→prime→hit and prime→hit→fresh. Each cycle begins with
reset. Fresh requests are unscoped, prime requests capture a boundary with zero
reuse, and hit requests must reuse 1,024 rows. Every request's output hash must
match its same-workload reference. Priming cost is recorded separately.

Before execution, the regression bound is fixed at 5% for first-token and
end-to-end medians. Eligible measured blocks require zero additional pageouts,
swapins and swapouts, and at most 200% aggregate CPU from other processes at
block boundaries. GPU/build ownership is serialized. Loading/hashing/warmup
are outside the measured block. All reported blocks pass these guards. No
thermal/performance warning level is recorded at their boundaries; full
temperature telemetry and peak RSS were not collected. Scope remains these
bounded workloads on the named host/artifacts.

## Original binary: demonstrated failure

The original `81b85f88` release worker has SHA-256
`71280ab529af3994cc3920ced6b18f15381b7129a1250eda035135166e0b2e82`.
Warmed hit first-token medians regress 7.03%/7.24% for E2B and 2.52%/5.89% for
E4B across the two workloads. Reused counts alone conceal the replay.

An independent nonperiodic E2B control uses 1,792 tokens in a 2,048-token context.
Priming succeeds and produces eight tokens. Its following cache hit returns
`native_generation_failed`: the restored 1,024 rows plus full replay exceed
capacity. The ordinary fresh request fits. The old binary and receipt are
preserved. This establishes a real request/state defect, beyond small snapshot
copy or synchronization overhead.

## Corrected execution

Gemma's text stream now receives `input[reused..]`, matching the existing
Qwen/Text dispatch contract. A text-only native invariant compares the engine's
actual prefilled count with checked `total_input - reused`; a mismatch drains
the stream and fails the request before exposing output. Image metric contracts
remain unchanged.

The measured suffix-fix worker SHA-256 is
`ddb28007d6e2283b3d37fb3c94f37f4be8789056f539aac58d2ed433f49b8558`.
It is built from `81b85f88` plus the suffix fix and actual-prefill invariant.
The receipt's `source_commit` field identifies that base; the executable hash
identifies the tested candidate. Final review narrows the invariant to text and
uses checked subtraction, preserving the tested text execution path. Final
source/build checks are recorded in the attempt ledger. Numerical kernels,
target/tower weights and ring snapshot copies are unchanged.
The final scoped-guard build is
`567898a851702f0dfc944ddee884f5050a387fb0b1822240a58e9d6555c59bb9`
and passes 134 worker tests, two ignored. Timing remains bound to the measured
`ddb28007…` executable; the final review adjustment received source/unit checks.

| Artifact / input tokens | Fresh first token (ms) | Prime first token (ms) | Hit first token (ms) | Hit change | Fresh end-to-end (ms) | Hit end-to-end (ms) | Hit change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E2B / 1,792 | 86.035 | 86.179 | 45.972 | -46.57% | 106.443 | 66.107 | -37.89% |
| E2B / 1,824 | 89.897 | 92.179 | 51.061 | -43.20% | 170.895 | 132.052 | -22.73% |
| E4B / 1,792 | 230.450 | 229.185 | 114.687 | -50.23% | 380.758 | 264.337 | -30.58% |
| E4B / 1,824 | 259.483 | 259.492 | 138.454 | -46.64% | 419.202 | 293.663 | -29.95% |

All medians use six observations per mode. First-token time is the worker's
native wall-clock proxy, including restore/setup and its documented lookahead.
End-to-end here is native `generation_seconds`; IPC wall-clock medians are also
retained and show comparable relative changes. E2B's 1,792-token workload emits
one token before EOS, while its 1,824-token workload and both E4B workloads emit
eight. Output counts and hashes match across modes.

Prime-minus-fresh is an inclusive request observation covering snapshot
allocation/copy/synchronization and measurement variation. Component copy times
were not isolated. The replay fix leaves those operations intact and removes
the extra target input work. No copy/synchronization change is justified by
these results.

## Stronger correctness controls

For each corrected artifact, a deterministic nonperiodic 1,792-token input and
its changed tail run at context 2,048. Fresh/prime/hit greedy and seeded
temperature-0.7/top-k-20/top-p-0.95 requests match exactly; hits reuse 1,024 rows.
All 12 near-capacity requests pass per artifact. They cannot hide a full replay
behind repeated input patterns or spare capacity. A real image request then
clears the entry; the same-context nonperiodic text first reuses zero rows and
then reuses 1,024 on repetition, matching fresh output. Assistant MTP stays off.

Each corrected recipe passes 207 aggregate checks, covering warmed timing,
actual reuse, exact output, nonperiodic/capacity and image transitions. This
does not cover all contexts, longer answers, new quantizations, mid-GPU fault
injection or the larger Gemma profiles. Runtime adoption, required CI, release
and installed-app validation remain separate.

## Private evidence

Prompts, responses, token vectors, decision probabilities and image pixels are
excluded from the aggregate receipts. The corrected recipe SHA-256 is
`e9ffc7e9f1edad6dcc317793844f02a9aebe0626cdda6aced20ebbb37d171eb0`.

| Receipt | SHA-256 |
| --- | --- |
| Original E2B warmed `runtime/gemma4-e2b-warm-cache-v1.json` | `b18b4083979add52f29e2c0bc2606daf55540a3d66870126043e42abf8c60a53` |
| Original E4B warmed `runtime/gemma4-e4b-warm-cache-v1.json` | `589f37bdcdd6d0ad7ad3b418871f9675161d6458f4b98bae05b38faf21000c5c` |
| Old nonperiodic capacity reproducer | `4fb6aebae53c29b6a0502912d3749d8170ccd373324ec940083c4dc2f14d9843` |
| Corrected E2B `runtime/gemma4-e2b-warm-cache-v2.json` | `951ddceb819fbd11c7e24ce9cabb99cf195ec453ba6ab90c019eba8fc6420c6e` |
| Corrected E4B `runtime/gemma4-e4b-warm-cache-v2.json` | `3d90529437f44f742fc3fbcc93bb44d4e38e3ad9d6cb4d7c309eef22629c902f` |
