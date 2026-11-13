# 20261113-EDGE0-14: expert binding extent and decompression churn

## Identity and scope

| Field | Value |
| --- | --- |
| Owner / authorization | 2026-11-13 / Codex; identify/reduce worker churn and rerun matched MTP |
| Scope | Local non-commercial evaluation only |
| Baseline | `543ea9bf`; preceding diagnostics from `590b185d` |
| Predecessor | [Attempt 13](2026-11-13-edge0-worker-memory.md) |
| Governing plan | [Phase 5](../EDGE0_PLAN.md#phase-5-mtp) |
| State / promotion | Bounded churn reduction and lifecycle validated; six timing groups deferred; record policy opt-in, MTP default off |

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

Preflight checks passed: the native engine suite, final focused Edge0 checks
(52 passed, 11 artifact-dependent cases ignored), worker checks (163 passed,
seven ignored),
55 private observer/control tests, six startup-process cases, formatting,
repository policy and production-target Clippy with warnings denied. The new
tiny-file GPU test verifies byte-identical products, mapped page prefixes/tails
and mapping ownership through both graph completions after file unlink.
Independent source reviews checked the complete inventory, lifetime and budget
accounting, unchanged screen predicates and the fixed-idle control schedule.

The [separate result record](../benchmarks/2026-11-13-edge0-expert-windows.md)
binds implementation `7c900c69` and release worker SHA-256
`0f6db70121c424f86457023de5330635b37ad677d8ebe3179d12e37cb9240055`.

The four-session ABBA control passed with all 16 requests matching the same
input/output hashes, six emitted/seven sampled tokens and EOS. Post-idle off
requests recorded **458,036 / 0 / 0 / 446,679 task decompressions** for policies
0/1/1/0; following on requests recorded zero in every session. Every fixed idle
wait exceeded 120 seconds. All four workers exited normally and were reaped.
Both record-policy sessions retained 11,956,895,744 bank-registration bytes
versus 43,819,991,040 for the baseline, a 72.714% reduction for this workload.
Sampled Metal allocated resource bytes after on were 18,214,731,776 versus
50,077,827,072. These are registration/allocation gauges, not physical residency.

This intervention identifies binding extent as an actionable source of excess
worker-execution churn in the bounded control. It does not identify individual
pages or the complete driver/OS mechanism. Source bytes, kernels and useful work
are unchanged. Fixed off-before-on route order, four sessions and uncontrolled
host differences prevent a route-efficiency ranking or speed claim. Neither
task/global subtraction nor diagnostic baseline subtraction is permitted.

The new policy also passed 26 actual worker lifecycle cases: nine greedy hash
comparisons, sampled paths, stopping/tails/streaming, two asynchronous prompt/
decode cancellations with two same-worker recoveries, cancellation-channel EOF,
three rejected invalid requests and two memory plans at context 512. These
cases have no worker-memory trace and are separate from memory measurements.

The preregistered performance attempt completed 12 golden-matching warmups but
deferred all six groups before any of 48 intended measured arms. All 657
diagnostic snapshots were valid, with task decompressions staying zero throughout
the observed span, including all warmups and admission waits. Six admission
windows totaled 19,986 global decompressions and zero compression, pageout or
swapout increases over 720.392 seconds. The sampled counter-only quiet maximum
was 9.759404 seconds, and the combined maximum was 8.600685, below ten. Every
window also had CPU/GPU gate violations; thermal remained nominal. The source
of residual global activity and complete GPU/driver accounting remain unresolved.

With the 128-token prefix, retained registration grew to 19,286,458,368 bytes
across 9,488 windows, while sampled Metal allocation reached 25,544,294,400 bytes.
The full admitted allowance stayed 44,210,061,312 bytes; arbitrary future prompts
can populate more of that inventory. The short control is not a model-wide
memory bound. No cache limit or admission reserve was reduced.

Independent review regenerated all 2,185 source entries and verified binary,
SDK, private helper/test, corpus, artifact receipt, input and output identities.
All 472 control and 657 performance snapshots and their derived deltas/brackets
were rechecked without discrepancies. Final performance monitor joined with
1,365 samples and zero failures. Lifecycle and performance workers each exited
on stdin EOF with code zero and were reaped within the registered limits.
Every predecessor receipt is preserved. Keep the resource policy opt-in;
performance qualification remains open and MTP stays off by default.

## Evidence bindings

Only aggregates and hashes are published. The locked Python 3.13.13 environment,
psutil 7.2.2, six SDK headers, interpreter, native fixture and private source/test
identities are bound inside the immutable registrations. No fresh large-payload
rehash is claimed.

| Binding | SHA-256 |
| --- | --- |
| Source manifest, 2,185 entries | `72cd9009cc1e5d1fbaae55465b1c6367b4343633e48630f9c4ce912ce32b5b47` |
| ABBA registration | `930c3a51a0a2a4c87a3b6dafba54f26e6b863216816a2c68e7fe69c22e5ab16c` |
| ABBA aggregate report | `dbc8f16c33c9836db0146ea786cc77343ddbd3d504b49579d8ced9e284b81ad9` |
| Lifecycle registration | `f5a0f29f69e4c81ee5ff5b0ea6c719aa209b5e3d94d62fdb6bd7f42ac440e799` |
| Lifecycle aggregate report | `976a1656e938d818fffe1b60b4d068e53d9261103198624214a41a1318d8afcd` |
| Performance registration | `a891aaaaef2c7bf8b45e335ddd66a1cb802ddcca97291a0f2b277f1712da33b0` |
| Performance aggregate report | `d0a4c8d15ea07490eb66afd23aca79fc1595b67e868fec64f733189cdeaa3b4d` |
| Control harness | `37e253e72b0ce0697bcd3d3459a5d445d0e95e2f37efb42c6446a210739d4478` |
| Performance/lifecycle harness | `c29a55e8cf1ecaa95e75619fec29edf604f510a580c8d44ee76aa74bfbd51c6b` |
| Worker/backend observer | `eb2bcf6b46d713b1ff46ca326640c7c458c833acf2b83e69fb7444052a4c25e5` |
| Final focused native test log | `ef0a00e0db29ac9c9af66cc0d3d228d3a98037e0d56a8345d3bfb0be6107e051` |

## Remaining qualification work

A qualified MTP decision still requires complete eligible matched groups on a
host that reaches every unchanged admission condition. Smaller bindings address
the measured worker churn, while the residual global counter and CPU/GPU gates
remain. No foreign application was closed and no OS memory/power policy was
changed. Wider prompts, arbitrary long-running cache histories, explicit residency,
prefix reuse, other artifacts and product admission require separate evidence.
