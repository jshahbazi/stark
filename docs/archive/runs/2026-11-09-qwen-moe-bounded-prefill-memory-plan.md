# Qwen affine MoE bounded-prefill memory proposal

The 256-token physical schedule is implemented and passes two CPU grid/tail
controls, 16 tiny native state/cache cases and four actual 1,026-token MoE
prefix cases under the existing hard guards. Its cold output matches the
frozen worker, and the short 64-token image/text comparison passes fixed 5%
checks. The larger cache matrix remains unqualified. The inventory below
preserves the earlier planning basis, without adding a hard threshold.

## Direct runtime result

The production worker is
`7f41246a4e265ad2ba4060c64a57888c4d0541a758f19ec4e0c9f80511c5700a`.
The actual image-loaded MoE passes cold/repeat and changed-tail/fresh checks,
with 1,024 reused tokens and matching outputs. Peak sampled footprint is
25,884,543,840 bytes, minimum available 76,490,129,408, and all paging,
compression and swap growth is zero. Receipt:
`c6813324f7eb3fdf598a20faed58f73898a07c5c3188bdda7dce332ba7eb48e2`.
The frozen worker matches the same eight-token cold output, receipt
`e753f4748357b8004b38a209030c7270a6527b43f8bfd891a6fcd779f2fca999`.

The [complete continuation record](2026-11-09-qwen-remaining-image-text-parity.md)
contains exact build/control identities and the short comparison. All workers
close after each run; the 64 GiB floor and 40 GiB process ceilings stay intact.

## Exact scope and source

- Existing headless Qwen3.5-35B-A3B affine4 image artifact, with its image tower
  loaded and the ordinary text graph retained.
- Payload: `20,394,001,296` bytes,
  `2f762b2629aaefc2fea6eb37ff6caaa4747fdc23a040e36f331fa4a056c6cad7`.
- Config: `1fe4929c5df2c9c5dae321e82f5cf3e181c05136e385fb9a016436ba3146c525`.
- Forty routed layers, 256 experts / eight active per token, model dimension
  2,048 and expert hidden dimension 512. Thirty mixers are DeltaNet with F32
  recurrent state; ten are attention.
- Source base: `d2cdde180b2331eb7e5dd01530d21e8919acdbc5`, plus the reviewed
  physical-prefill patch `78cf769fedab7eadcc71241f05e2bda81d4d92280ccc03844bcc6762e170f907`
  and separate completion-order patch
  `83d5ef1ba2c6955e0a4cd978e15a4dd4f9a849e02c10a509cff579977a228ee6`.
- Combined test build: `f013042d899e3ee7191bba76c0a2a49a97093697b4f4ab88ebd328d8ea6c7664`;
  compiled/source seal:
  `e308d690c9c97337e494e8e54de66dd386ab663f4834213872a7bb1d9a609e24`.

The 256-token cap applies only to ordinary routed-Qwen text without image
context or a generic speculator. It submits and awaits each physical chunk,
including the final sample, while preserving the absolute 1,024-token snapshot
grid. A snapshot stays private until the whole prefill succeeds; partial setup
failure removes snapshot markers and fences the request state. The completion
callback publishes its cached allocator before waking a waiter. Image geometry
and speculative verification retain their existing schedules.

## Evidence and remaining attribution gap

The frozen `4010b86e` run crossed its unchanged 64 GiB actual available-memory
floor before completing its first 1,792-token case. Its receipt remains failed:
`31b0c7bed7a4362e56669a061b56ba097e54675853ac00bb64b37f3b3ebeab92`.
An audit of that exact source rules out per-token full recurrent-state histories;
route scratch returns immediately to pool bins. At least 3,440,640 routed
dispatches are encoded for that complete input into one command buffer.

The subsequent small affine operator measures command allocator heaps directly:
53,862,976 bytes for 256 rows and 213,983,808 for 1,024 rows. Four smaller
submissions exactly reproduce the large operator result and reuse its allocator
without growth, retaining the larger high-water allocation. Curve receipt:
`f1d4cb775622feca5e5b65df5959b626c0c89fe30a3a8329085a7a803044fb3c`.

**That probe uses the `Qwen35AffineBF16` image/reference profile at dimension
64. The ordinary fast-text loader explicitly switches to `Legacy`.** Its
kernel dispatch choices, activation rounding and route accumulation policy have
not been measured by that probe. Scaling its curve by 40 layers is an estimate,
not an actual-text bound or attribution of the failed full-model peak.

An additional Legacy expert-component calibration was proposed but not
implemented. The user directed the work to stay focused on fixing the issues;
that extra fixture/rebuild is abandoned. Actual-model checks will use the
existing supervisor and start with the smaller 1,026-token prefix case.

## Conservative planning inventory

The following allowance is provisional pending actual runtime checks. It
deliberately charges plural large command heaps: decode
and later requests can retain more than the one allocator observed in a
sequential operator test. Reset marks reuse and does not promise release.

| Component | Planning bytes | Basis |
| --- | --- | --- |
| Resident parameter payload | 20,393,085,408 | Prior exact-artifact storage receipt on worker `4010b86e`; no representation copy. New build storage/peak remains unverified. |
| Loader/driver margin | 1,073,741,824 | Allowance beyond parameter bytes; the earlier stable loading plateau was about 20.72 GB RSS. |
| Three retained command heaps | 8,053,063,680 | Three allowances of 2.5 GiB each; larger than the dimensional extrapolation, still unmeasured on the actual Legacy path. |
| Target state, snapshots and payload scratch | 1,073,741,824 | Full selected-context KV, F32 recurrent state, up to two snapshot copies and rounded/recycled payload pages. Existing admission estimates remain unchanged. |
| Tiny vision preparation scratch | 536,870,912 | Restricted 32×32 fixture and expanded language input at most 128 tokens; no general-resolution claim. |
| CPU/file-cache allowance | 2,147,483,648 | File-cache effects are not assumed zero or automatically reclaimable. |
| **Provisional working inventory** | **33,277,987,296** | Planning sum; not a measured or guaranteed peak. |

Adding 2 GiB of host drift gives an illustrative entry planning estimate of
`104,144,947,680` available bytes. This is not an additional hard guard or a
proven peak ceiling. The existing 64 GiB entry/runtime floor remains unchanged;
fresh psutil available bytes and observed runtime counters govern the bounded
check. A reported memory-pressure percentage is not used to calculate bytes.

## Remaining scope

1. The existing 1,792-token / 30-case matrix remains pending; no run is made in
   the current window. No additional calibration fixture is required.
2. Retain the exact new worker and source-bound receipts, including failed
   setups. Do not claim numerical equivalence from the small reference-profile
   curve or transfer older timings to the new worker.
3. Continue to use 40 GiB RSS/physical-footprint/wired-growth ceilings, 64 GiB actual
   available floor, 8 GiB disk floor and 600-second deadline. Keep zero swap
   used and Swapins/Swapouts growth, compressor growth strictly below 1 GiB,
   positive process samples, foreign-owner rejection and owned drain/join.
   A hard stop ends that attempt without an automatic retry.

Any future functional-only Pageouts policy remains ineligible for performance
claims. Matched timing needs its own quiet, zero-paging controls. The retained
107.3 GB ddalcu installation and unavailable official BF16 MoE payload are
outside this proposal and remain separately held/blocked.
