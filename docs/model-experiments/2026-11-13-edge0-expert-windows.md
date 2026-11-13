# 20261113-EDGE0-14: expert binding extent and decompression churn

## Identity and scope

| Field | Value |
| --- | --- |
| Owner / authorization | 2026-11-13 / Codex; identify/reduce worker churn and rerun matched MTP |
| Scope | Local non-commercial evaluation only |
| Baseline | `543ea9bf`; preceding diagnostics from `590b185d` |
| Predecessor | [Attempt 13](2026-11-13-edge0-worker-memory.md) |
| Governing plan | [Phase 5](../EDGE0_PLAN.md#phase-5-mtp) |
| State / promotion | Implementation/control preparation; new binding policy opt-in, MTP default off |

## Source findings and hypothesis

Parallel source review found read-only file mappings, retained no-copy Metal
wrappers and small CPU control vectors; no complete CPU weight expansion or
per-request payload scan was found in the inspected graph path. Per-layer router
boundaries allocate fresh Shared scratch, a separate untested allocation-churn
candidate. This attempt changes only the target expert binding extent.

The old policy binds an entire 834,928,640-byte expert layer buffer even when a
kernel reads one expert's gate/up/down record. Across 48 layers the complete
expert bank is 40,076,574,720 bytes, plus 3,743,416,320 projection bytes. The
selected record is 1,630,720 bytes. Large resource extents are a plausible source
of driver/VM work, but source inspection does not prove OS decompression behavior.

## Candidate and safety contract

`STARK_EDGE0_EXPERT_WINDOWS=1` freezes a selected-expert-record policy during
Metal admission. Unset/other values retain the old layer policy. Projected
weights, stored formats, numerical kernels, F32 policy, routing, sampling and
request state remain unchanged. Expert gate/up/down views address the same
immutable mapped bytes through page-aligned no-copy wrappers. Each wrapper
retains its file mapping through GPU completion. Residency combined with this
policy remains unqualified and is rejected; no host memory or power setting
changes.

Prospective admission includes all 24,576 expert records plus 48 projection
windows. At 16 KiB pages these register 44,210,061,312 bytes, 390,070,272 more
than the old 43,819,991,040-byte complete bank inventory because page edges
overlap. Request and MTP admission use that actual complete allowance under the
same 56 GiB/recommended-working-set cap and static/dynamic reserves. Actual
registration remains lazy. Smaller selected resources do not imply that complete
prospective registration bytes or physical memory decrease.

The existing operator gate also enables an aggregate
`backend_memory_diagnostics` operation. It reports window policy/count/bytes,
admitted allowance and Metal allocated resource bytes. It preserves prepared and
prefix state, emits no content or per-expert identities, and makes no physical
residency claim. Both diagnostic flavors and all detailed receipts remain private
during evaluation; publish aggregates and hashes only.

## Preregistered diagnostic control

Use one sealed Rust 1.98.1 binary/source/artifact/corpus and fresh worker sessions
with policy order **0, 1, 1, 0**. Each session uses the same 32-token prompt,
context 512, greedy seed 42, maximum 16 outputs and matching natural EOS:

1. Record loaded diagnostics; run off then one-draft warmups with endpoint probes.
2. Wait a fixed 120 seconds with the same approximately one-second diagnostic
   cadence. Never shorten the idle interval when the host becomes quiet.
3. Repeat off then one-draft requests with the same endpoint probes.
4. Close stdin, require normal zero exit, reap and verify sealed identities.

Compare corresponding sequence positions across binding policies, preserving
input/output hashes, sampled/emitted counts, stopping/verification metadata and
task/gauge/global endpoint brackets. This bounded control can diagnose churn;
its timings have no performance eligibility. Fixed off-before-on order cannot
rank MTP routes, and ABBA does not eliminate every uncontrolled host difference.
No task/global subtraction or attribution percentage is allowed.

## Preregistered performance rerun

After numerical/lifecycle checks and the control, use the candidate policy in a
separate matched rerun: prompts 32/128, context 512, original weights/F32,
greedy seed 42, maximum 16 outputs with natural EOS, depths one through three,
one warmup per route and four balanced pairs per group. Intended work is
12 warmups and 48 measured arms. Preserve all
[mandatory screens](2026-11-13-edge0-mtp-paging.md#unchanged-mandatory-gates):
zero Pageouts/Swapouts/Compressions/Decompressions per measured arm; unchanged
CPU/GPU/thermal/compiler/gap/failure gates; fresh contiguous ten-second combined
admission, 120-second group limit, boundary coverage and final monitor join.
Diagnostic latency must not stale first-arm admission. Default-selection rules
are unchanged, with no selective retries or baseline subtraction.

Reuse the exact pinned artifact/receipt and explicit large-payload immutability
assumption; no new full payload rehash is planned. Keep 64 GiB minimum available
before launch, 48 GiB sampled owned RSS, 180-second request and 1,800-second
worker limits. Serialize all model/GPU work. Seal code, binary, corpus, observer,
helpers/tests, SDK and registration before and after runs. Keep context 512,
unavailable Edge0 prefix reuse and local non-commercial scope. The resource
extent changes no causal state format or restoration; Phase 6's blocker remains.

## Results and disposition

Preflight checks passed: the native engine suite, 52 final focused Edge0 tests
(11 artifact-dependent cases ignored), 163 worker tests (seven ignored),
55 private observer/control tests, six startup-process cases, formatting,
repository policy and production-target Clippy with warnings denied. The new
tiny-file GPU test verifies byte-identical products, mapped page prefixes/tails
and mapping ownership through both graph completions after file unlink.
Independent source reviews checked the complete inventory, lifetime and budget
accounting, unchanged screen predicates and the fixed-idle control schedule.

Pending sealed controls and matched rerun. Preserve every predecessor receipt.
No decompression reduction or performance benefit is established by source alone.
