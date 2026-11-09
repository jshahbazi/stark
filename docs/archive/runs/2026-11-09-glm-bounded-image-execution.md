# 20261109-GLMVIS-09: bounded GLM image/text execution

Registered before the new request-accounting or execution changes. Source
baseline `cc4e5aa2`; predecessor:
[GLMVIS-07](2026-11-09-glm-image-admission.md). State: proposed/source work;
no new model, lifecycle or timing result is claimed.

## Fixed recipe and acceptance

Use the same pinned schema-7 decoder and sidecar, arithmetic, context and
sampler as the earlier records. For both text-only and image-loaded arms,
select existing bounded-command execution with `STARK_GLM53_BOUNDED_GRAPH=1`
and `STARK_GLM53_BANK_GROUP_LAYERS=1`. All other GLM experiment selectors are
unset: no persistent encoder registry/residency, wide encoder, graph overlap,
chunked prefill or pooled/unretained graph. Preserve the generic request
divisor four and its reserve; no path-specific relaxation is proposed.

Correct request accounting must include the live state, old/replacement
recurrent snapshots, all host attention copies and separately selected vision
work. Ordinary text with the tower loaded retains its weight charge but must
not be charged inactive vision scratch. Image entry clears text snapshots and
has zero MTP. Increasing previously omitted state accounting is required
before relying on the plan.

The header/source plan receipt is
`e3446e0fd1551a0aa314f8586edbe994398963512f2f9ae7eedb9054a963aadd`.
Its initial conservative active inventories at context 4,096/input 1,025 are:

| Component | Bytes |
| --- | ---: |
| Largest actual one-layer bank group | 3,133,194,240 |
| Complete static allowance | 8,589,934,592 |
| Text dynamic, including two KDA snapshots and three host attention copies | 1,787,057,680 |
| Image dynamic, no text snapshots and including vision scratch | 1,815,455,120 |
| Resident vision sidecar in image-loaded arms | 1,164,010,176 |
| Extra CPU/GPU control-weight overlap during startup | 150,699,960 |
| Separate minimum host margin | 17,179,869,184 |

This projects 13,660,886,472 bytes for text-only, 14,824,896,648 bytes for
image-loaded text and 14,853,294,088 bytes for an image request, before the
separate host margin. Summed bank spans and an active inventory are not an
RSS or driver-commitment guarantee. Mapped CPU/file-cache pages must be
distinguished from wired allocations and private physical footprint.

## Execution and timing gates

Before full-model execution, obtain independent review of the complete source
formulas, nonoverlapping group completion/autorelease cleanup and an owner-only
watchdog. It must monitor available memory with at least a 16 GiB floor,
wired/physical footprint against the complete reviewed plan, swap/pageouts,
compression and process RSS separately, and stop the owned run on a violation.
No model run or heavy build is authorized by this record alone; the serialized
resource slot remains mandatory.

Qualify actual image generation and image-to-text recovery, exact native
1,024-token fresh/cached results, scope/context/MTP changes and image
success/failure/cancellation/restart. Compare matched text-only and
image-loaded arms with identical bounded execution, predeclare tolerance from
baseline repeats, and keep the earlier resident-mode baseline separate.
Record finite/nonempty execution, counts/hashes, preparation, first token,
useful decode, end-to-end, loading and memory independently. Failed resource
screens do not qualify correctness or performance.

Source-head MTP remains required. Any implementation must execute its retained
weights, preserve exact target verification and request/cache state, and add
head state/payload/snapshots to this plan before execution. A serial verified
head route would establish supported execution only; it would not establish
acceleration. Existing EXL3 text MTP/default-three policy remains separate.

## Source accounting and bounded CPU controls

The request-accounting change is implemented on the branch. Ordinary GLM text
now charges live/previous/replacement host attention copies and two complete
KDA snapshot copies. Image admission drops the retained text prefix first and
uses its own inventory with one live host attention copy and bounded vision
scratch. Both loaded-image modes retain the complete sidecar-weight charge.
The worker's generic divisor four and 256 MiB reserve are unchanged.

Independent source review found no blocker in that accounting split or the
existing bounded groups' completion/drop/autorelease ordering. It did not run
an engine or establish actual footprint. The immutable reviewed patch is
`be590f0a037162abd39d3193517a0bddf498e4e0a54fede83ca4c86b2f6345a8`.

Eight CPU controls passed using the actual source-extracted accounting
functions, compiled directly with Rust 1.98.1 without an engine/dependency
build. They check pinned text/image totals, snapshot difference, context/input
growth, minimum rounding and overflow. Their private receipt is
`890e864b26d22a15c64dd0754ab1cc43627588fef88939c21dbcf7bc0ee056da`.
This is bounded CPU arithmetic evidence; the engine test target and model path
are not yet built or executed for this change. Formatting, whitespace and
repository policy pass.

A private owner-only watchdog is prepared with a 64 GiB entry screen, 32 GiB
available-memory/RSS/physical-footprint limits, 32 GiB host wired-growth ceiling,
less than 1 GiB compressor growth and zero pageout/swap growth. It attaches
before readiness, closes only its worker's cancellation pipe on breach and
then terminates/waits only for that owned process if cancellation does not
finish. Nine injected screen controls pass without a worker/model. Live
supervisor cleanup, source compilation and actual memory qualification remain
pending the serialized slot and independent watchdog review. A 32 GiB RSS
rejection remains a recorded resource failure even if reclaimable file pages
explain it; the limit is not raised automatically.

## Compiled candidate and next execution checkpoint

Both fresh isolated builds now pass at exact code revision
`b76f560d54d251f207cdca870b1136dec7076e2b`: Rust 1.98.1 release worker/helper and
the pinned-nightly release native engine test target. All 1,262 recorded source
files stayed unchanged during both builds. Their common source-graph digest is
`a2c9fad27a697206d70f0661b2116675d14f4fec4d1648d78729360b819d81f6`.

The actual freshly compiled native accounting control passes: one test,
zero failures. Its private receipt is
`7573f9d5c9a8fb29b8f66e391a54479656118f8e24dbf6ab694a3680d4455ec7`.
This updates the earlier source-extracted CPU evidence without relabeling it.
Worker SHA256:
`c8e528dc017809edb9c717b55c6c2dce6533e34259a4175988e9c8d4aa506f3d`;
helper SHA256:
`7d8f92e9552a9fd4e898113d674dd3ad235f423f7ce8a03d1ed03a2f6380164d`.

The exact supervisor now also continuously rejects foreign native workers and
the installed Stark app, while terminating only its owned process. Its SHA256
is `7848ec39a115a41bd517cb36d1499fe84dc2a0b523412c5a4af6ae8e9cdd0c37`.
Five live tiny-child cases pass on that exact source, including cooperative
cancel, termination, constructor/base-close failure and early exit. Receipt:
`1c2a360ab9a5bb62a8b9305ed5c669acfa53d6e8f2ad4df3970daae2401f587a`.
Success requires positive sample count, no violation, owned child reaped and
supervisor joined.

Before starting a model, the private checkpoint binds the exact next command,
driver, fixed environment, source/binaries, artifact and watcher identities:
`1ff4591950b3ec059560a9d9f2df0130fdd2c942e1aa44b9429533c6ef7b1ce4`.
Driver SHA256:
`01a0969158ff333cd0a3ad74da8060ece645a6eca070bd3ef8cbae9ff8f2e13c`.
The next authorized control is sequential text-only eight tokens, image-loaded
text eight tokens with exact output-hash agreement, an actual PNG request with
one output token, then eight-token text recovery. It grants no cache, MTP or
performance promotion. The first retained-fixture AST preflight failed before
any model; corrected fixture extraction and the newly built helper pass their
separate CPU preflight.

A background shader build in the separate resource-budget checkout was
identified from executable/cwd metadata. The initial functional control may
proceed under the unchanged strict memory/foreign-model screens; no quiet-host
or performance claim may be derived from that run. Benchmarks require a later
quiet slot. No model has run at this checkpoint.
