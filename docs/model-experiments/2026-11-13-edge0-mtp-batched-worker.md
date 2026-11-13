# 20261113-EDGE0-11: shared prompt arithmetic and matched stopped requests

## Identity and boundary

| Field | Value |
| --- | --- |
| Registered / owner | 2026-11-13 / Codex, local non-commercial evaluation only |
| Baseline | `64d3ab12`, serial-prompt worker candidate from [attempt 10](2026-11-13-edge0-mtp-worker.md) |
| State / result | Correctness and worker lifecycle passed; all matched timing groups ineligible |
| Governing plan | [Phase 5](../EDGE0_PLAN.md#phase-5-mtp) |
| Adoption | Local opt-in worker adopted on branch; default remains off; performance qualification pending |

Attempt 10's worker lifecycle and serial-target numerical controls passed, but its
first off-route benchmark warmup failed the fixed 16-emitted-token gate. It has
no qualified performance result. Its decode cancellation also occurred after the
initial prefill output, before a verified decode output was demonstrated.
Keep those receipts intact and qualify the narrower observations separately.

Independent review also found that serial MTP prompt ingestion and ordinary
eight-row prompt ingestion do not establish identical target distributions.
The earlier serial-reference oracle does not close that gap. This attempt retains
ordinary MTP-off arithmetic and uses that same prompt schedule for MTP-on.

## Fixed implementation and correctness gates

- Retain the ordinary off schedule exactly: `input[..len-1]` in chunks of at
  most eight, followed by a single final input/readout. Capture a compact wide
  residual for each batched prefix row using copies after the existing target
  arithmetic. No extra target advance or change in numerical reduction order.
- Catch up the draft front in chronological order from those retained rows and
  the already-known next prompt inputs. Preserve head K/V, row tags, single-token
  target verification, complete rewind, cancellation and exclusive resource leases.
- At eight rows the additional compact owner is at most 327,680 bytes, within
  the existing 16 MiB head reserve. Runtime head admission remains 512 MiB and
  worker planning remains conservatively 768 MiB. Ordinary off requests allocate
  no wide capture. Source weights and F32 arithmetic remain unchanged.
- Rebind the 15-case / 41-complete-row oracle to ordinary off-route batched prompt
  ingestion, followed by ordinary single-token decode. Require bit equality at
  the initial readout and every verified continuation; do not substitute greedy
  identity or tolerances. Add actual prompt boundaries around eight-row chunks.
  The separate actual prefix case uses lengths 1 through 10, 17 and 32, requiring
  36 exact complete target-readout pairs, 92 head-front cells, all eight prefix
  tails and empty/oversized-batch refusal on one loaded model.
- Repeat the 26 normal-worker lifecycle cases under a new sealed helper. The
  decode cancellation trigger now requires `sampled_tokens >= 2`, proving at
  least one target-verified tail output before cancellation. Preserve request B
  recovery on the same worker, stale-ID refusal, prompt cancellation, channel EOF,
  stops, tails, streaming, planner and negative controls. Normal EOF exit 0 is
  required; forced teardown is failure evidence.
- Retain the independent head numerical controls, appropriate engine/generator
  regressions, production Metal Clippy, CPU feature check, format and policy gates.

## Successor performance recipe

This is a separate preregistered recipe; attempt 10's fixed-work gate is not
changed retroactively. Keep its corpus, prompt lengths 32/128, context 512,
greedy seed 42, requested maximum 16, depths one through three, warmups, four
balanced pairs per group, privacy, resource limits and host screens.

Use the ordinary worker's natural EOS handling. Both arms and every warm/measured
request in a group must have identical emitted token hashes, emitted/sample counts
and finish reason. Require at least two emitted tokens for a decode-rate claim.
Early EOS is matched useful work only when both arms stop identically; no EOS
suppression, prompt selection, arm retry or forced continuation is allowed.
Retain invalid/ineligible groups without ranking them.

Report useful emitted decode `(output_tokens - 1) / decode_seconds`, raw sampled
decode `(sampled_tokens - 1) / decode_seconds`, prefill, generation and IPC-inclusive
wall separately. The default decision retains attempt 10's fixed 5% median paired
decode/wall gain, at most 5% prefill regression, off wall spread at most 15%, and
all host/correctness gates on both prompt lengths. Select a qualifying depth by
its minimum wall speedup; otherwise retain off. Warmup/loading/cold first requests
remain outside these warm same-process summaries.

## Artifact and resource constraints

Use the same pinned `Edge0/Edge0-145B-q3k` artifact and receipt SHA-256
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`.
The prior complete source-payload hash receipt and explicit immutability assumption
remain in force; this is not a new large-payload rehash. Seal source/build/helper
identities before launch and verify afterward. Rust 1.98.1 and locked dependencies;
same 128 GiB M5 Max host; one model process at a time; 64 GiB available launch gate,
48 GiB sampled owned RSS, 180-second request / 1,800-second normal-worker limits.
The private numerical binary uses the existing 600-second test limit.

No wider-context, prefix-reuse, resident-mode MTP, quality, app/catalogue,
distribution or installed-build admission is part of this attempt.

## Results and disposition

Candidate `bdf3b62e` passed the complete native unit binary: 1,099 passed,
163 ignored, 226.72 seconds. Generator units passed 150 with six ignored;
four private head controls retained their 402 stage rows, ten exact routes,
seven checkpoint and ten trace/readout/wide replay pairs. Production Metal
Clippy, CPU/grammar checking, pinned formatting and repository policy passed.
Six startup process-boundary controls passed. Integration/documentation tests
were not rerun; ignored artifact cases below ran separately.

The exact prepared artifact passed 12 prompt-boundary cases with 36 full target
readout pairs bit-exact, 92 chronological head-front cells, all eight batch tails,
and empty/oversized refusal. The fresh 15-case sampler oracle compared 41 complete
target rows with ordinary off-route prompt ingestion and single-token continuation;
all were bit-exact. It observed 24 proposals and nine acceptances. These tests
close the serial-prompt limitation of attempt 10 within the registered controls.

The normal release worker passed all 26 lifecycle cases and nine greedy hash
comparisons. Prompt cancellation fired at sampled count zero; decode cancellation
fired after sampled count two, establishing a verified tail output. Both requests
recovered on the same loaded worker with the reference output hash after a stale
cancellation ID was queued. Channel EOF, stopping, streaming, admission and negative
controls passed. The worker exited normally on stdin EOF, code zero.

All actual processes were serialized. The boundary, sampler and lifecycle cases
began with 86,917,234,688 / 88,930,189,312 / 86,743,760,896 available bytes and
sampled peak owned RSS of 592,576,512 / 821,854,208 / 847,003,648 bytes. Boundary
and sampler test wall times were 35.80 and 52.03 seconds, within the 600-second
watchdog. These metrics are resource controls, not complete GPU memory or speed
claims. The same 128 GiB M5 Max / macOS 26.7.2 build 25G332 host was used.

### Matched measurement and default

The [benchmark record](../benchmarks/2026-11-13-edge0-worker-mtp.md) preserves all
60 requests: 12 warmups and 48 measured arms. Hashes, stopping and useful work
matched. The short prompt stopped at EOS with seven sampled/six emitted tokens;
the long prompt emitted all 16. Every on arm performed real proposals; off had
zero MTP activity. Normal EOF exit was zero.

Only two arms, zero pairs and zero groups met all host screens. Global
decompressions increased in 46 arms, pageouts in nine, and external CPU exceeded
the ceiling in one. No compression/swapout increase or thermal failure was
observed. All six timing groups are ineligible; no throughput ranking, qualified
speedup or speed regression is promoted. The preregistered no-qualifying-depth
decision retains default zero and preserves explicit request overrides.

### Immutable bindings

| Evidence | SHA-256 |
| --- | --- |
| Release worker | `056839034938df8d4ffa19ec55ee0be0923e4559004acbc688dc1035ff07ede2` |
| Source manifest, 2,182 files | `3626c3b1a05bcd9ef5965cead2e168a8978963cc45888dc8f94cf273fd7912fd` |
| Build seal | `6c6c1b89ba6af3c65c16b55dbe102a723efd45e7cc13fd565651863f8c345d84` |
| Boundary report | `b578dbf06596f98c80570f422ad23c538313e9529c447f3466a7ccca1114e590` |
| Sampling report | `4cd00607a94b99fe17f90b3dff6c0a4e9b4852879e0aba840d8fe12bba47f932` |
| Worker lifecycle report | `c35626b569345cc8aeea5f7ae62a71c2823378dcbd0b4d9bcfe7d26204d07358` |
| Matched measurement report | `45622b35d83b6615c8f06d2f03206cbbc23a945ceb395ef52631c42141d3dd34` |
| Native unit log | `b18b798df518338f215bb1052af27e0977fd76376061ce3c9263a8db29ff56a4` |
| Generator unit log | `9c8ab36e8c088b0bad73c05eb9d22f3046c4055997e755afa305ca314295bff6` |
| Complete worker evidence receipt | `b42e7755e7fab57e28749ed2ee288758cc03c712843a5e659e406c6b80351d08` |

Helpers verified source, executable, corpus/configuration and registration seals
before and after execution and process reap. Independent Sol review verified the
receipts, exact counts, all matched work, screen predicates and default decision.
Predecessor failures and weaker cancellation observations remain in attempt 10.

Verdict: **conforms for bounded local worker correctness and lifecycle**.
Measurement execution is complete; performance qualification is unverified.
Adopt the opt-in virtual Metal worker path on this branch, retaining default off.
Phase 5's performance qualification and phases 6–8 remain unfinished; no resident
MTP, prefix reuse, wider-context, quality, product or installed-build claim follows.
