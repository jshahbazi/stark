# 20261113-EDGE0-11: shared prompt arithmetic and matched stopped requests

## Identity and boundary

| Field | Value |
| --- | --- |
| Registered / owner | 2026-11-13 / Codex, local non-commercial evaluation only |
| Baseline | `64d3ab12`, serial-prompt worker candidate from [attempt 10](2026-11-13-edge0-mtp-worker.md) |
| State / result | Implementation and qualification in progress |
| Governing plan | [Phase 5](../EDGE0_PLAN.md#phase-5-mtp) |
| Adoption | Pending; MTP remains off by default |

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

Not run. Preserve predecessor failures, then append source-bound results and the
matched default decision after the registered gates execute.
