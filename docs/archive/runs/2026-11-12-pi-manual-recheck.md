# Pi manual foundation recheck, 2026-11-12

Verdict: **Not confirmed for milestone 1 completion.** The foundation follows
the plan's architecture and manual-authentication scope. The independent review
found concrete setup/request defects, which were corrected and rechecked.
Live artifact qualification and two client protocol gates remain pending.

Reviewed baseline: integration `f319ac5021e9edd99a8d033d926c439e2bfa19e0`, Stark
tracking `088eb90ed80ae759d21029148a2dfdc8eb5c4182`. Governing plan and AGENTS.md:
Stark `0bdcf4b3b4cc316fd7fd4f4ec2f6bc3a18268420`. The earlier
[foundation record](2026-11-12-pi-manual-foundation.md) remains source-bound to
the initial 22-test implementation.

Corrected integration:
[`8fdd7bf52c913df9fe2abd65c30bb7d4cc4b841e`](https://github.com/jshahbazi/stark-agent-clients/tree/8fdd7bf52c913df9fe2abd65c30bb7d4cc4b841e).
Its [review report](https://github.com/jshahbazi/stark-agent-clients/blob/8fdd7bf52c913df9fe2abd65c30bb7d4cc4b841e/docs/runs/2026-11-12-pi-foundation-recheck.md)
contains the requirement map and new source/test manifest:
`46c8d4f2d57c020bbde26b45840456130e5e604dd3269bd1ec851eb365ffdf4e`.
Pi CLI/AI remains pinned to 1.5.2; lockfile, installed Pi/schema and settings
hashes remain unchanged from the original record. Checks ran on Node.js
26.13.0, macOS 26.7.2, arm64.

## Reproduced findings and fixes

| Finding | Effect | Correction |
| --- | --- | --- |
| Pi inferred long-cache support on loopback | Inherited `PI_CACHE_RETENTION=long` added `prompt_cache_key` and `prompt_cache_retention`; Stark rejects these fields | Explicitly disable long retention and prompt cache keys; SDK and actual CLI regressions cover inherited preferences |
| Duplicate JSON keys accepted by `JSON.parse` | A second `providers` object silently replaced earlier unrelated providers | Reject duplicate keys, including escaped spellings, in merge and discovery JSON |
| Merge read occurred before size validation | Oversized or special input could consume excess memory or block | Require a bounded regular file; nonblocking open and capped reads cover special files and growth after stat |

The primary cache and duplicate-key findings were independently reproduced
against the initial commit. Corrections were independently reviewed. The final
review found no additional blocking code issue within this foundation scope.

## Verification and remaining gates

- Fresh locked install and original 22-test suite passed before correction.
- Corrected suite: **29 passed**, 0 failed, 0 skipped, including actual CLI
  tools with inherited long retention. The independent reviewer also ran all
  29 checks. A final nonfinite-JSON counterexample was added to an existing
  config check; all 15 configuration tests passed again afterward.
- New checks cover escaped duplicate keys, input bounds/special files, the
  five-second whole-request discovery deadline and proxy environment refusal.
- `git diff --check` and Stark repository policy checks passed.

Pi's successful result with missing usage remains an open accounting gate.
The recheck additionally established that Pi accepts clean EOF after finish
reason and usage without Stark's `[DONE]` sentinel. Both checks characterize
client limitations; passing their test assertions does not satisfy the plan's
acceptance requirements.

No real credential was obtained, model loaded, native inference invoked or
performance measured. Actual CLI/native cancellation, artifact/task quality,
context/compaction, lifecycle, interleaving and prefix reuse still require the
live matrix. No managed grant, adapter or distribution scope was added.
