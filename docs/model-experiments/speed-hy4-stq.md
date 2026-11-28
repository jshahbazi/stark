# Hy4-flash Sushi STQ 4bpw performance history

The pinned catalog export is
`beamster/Hy4-Flash-Sushi-STQ-4bpw@61528ba635a623f127c5a7b1e1f4ac2c2b0d4885`.
Stark loads the original 208,397,806,219-byte export natively with the BF16
vision tower and DFly2 draft head. The original AngelSlim Q3_K port remains a
local preparation path with a separate [performance history](speed-hy4.md).
Its GGUF, FP8 Engram and NextN results apply to that artifact. The STQ catalog
qualification used the full verified local payload through the real install
boundary with copy-on-write clones. [Catalog qualification](2026-11-27-hy4-stq-catalog-defaults.md)

## Bring-up and what paid off

Stark started with Strata's Hy4 graph and its own Metal worker, context fitting,
prefix reuse, images and cancellation. Matching the 4K AngelSlim fixture to
patched llama.cpp established a numerical starting point. Packed grouped
experts helped prefill, while prepared expert residency and the Q5_0 NextN
head helped decode. On the actual longer OpenCode turns the resident AngelSlim
worker measured 35.74–40.77 useful decode tok/s. The shorter 4K peak came
from a different workload. Stark also added a parser for the exact pinned
Hy4 template and verified typed calls, tool-result replay and streamed
reasoning boundaries in the local API. [AngelSlim build and controls](speed-hy4.md) ·
[Resident OpenCode record](2026-11-26-hy4-resident-opencode.md) ·
[Tool-calling qualification](2026-11-26-hy4-tool-calling.md)

On the STQ export, reuse the architecture and host lifecycle and adapt the
Sushi v1.8.1 PAIR1/SPEC1 packed products, register sparse MLA, positional
Engram reads and DFly2 schedule to Stark. The port kept the original matrices
and the pack's precision boundaries where qualified. The first native record
binds codecs, Metal kernels, vision and drafts to the exact artifact and
upstream notices. [Native component and trunk record](2026-11-27-hy4-stq-native.md)

| Change and evidence | Disposition |
| --- | --- |
| Tensor MLA for contiguous ranges, register MLA for sparse rows, GPU verification planning, split SPEC1, PAIR1 finish and cached Engram reads. At fresh 27K, sampled MLA fell from 18.16 to 5.77 s. Across the bundled uninstrumented repeat, resident prefill/decode moved from 577.81/28.59 to 748.60/49.93 tok/s. [Initial OpenCode work](2026-11-27-hy4-stq-open-code-performance.md) | Adopt the tuned path. Treat the profiler as attribution and the uninstrumented run as whole-request evidence. |
| Dense attention's direct packed K32 product and fused gate preparation saved 2.49 s in a fresh 27K same-binary pair. The wide routed K32 PAIR1 and register H128 path moved a repeated matched 27K pair from 34.20 to 32.93 s. [Dense](2026-11-27-hy4-stq-dense-attention-prefill.md) · [Routed and sparse attention](2026-11-27-hy4-stq-routed-attention-prefill.md) | Keep the dense and routed paths. Absorbed-head prototypes and alternate long-context MLA scheduling stayed out. |
| The apparent wide-routing decode loss did not reproduce in an alternating same-load test. At 27K with 24,576 reused positions, the medians were 49.15 with wide routing on and 46.58 with it off, and wide won two of four adjacent pairs. Narrow verifier work and GPU draft gathering had separate exact component and integrated evidence. [Decode attribution and verifier work](2026-11-27-hy4-stq-decode-prefill-followup.md) | Keep wide prefill and qualify decode on fresh requests as well as warm continuations. |
| The attention-to-FFN boundary occupied 1.014 s of GPU spans on the profiled fresh 27K request. Exact norm and residual fusion plus batched draft correction measured 31.87 s prefill and 51.88 tok/s decode. A separate-load control measured 32.42 s and 51.62 tok/s. [Boundary fusion](2026-11-27-hy4-stq-attention-ffn-boundary.md) | Keep the fused and batched paths after the catalog qualification. The component wins placed an upper bound on expected end-to-end savings. |
| Register-only preparation, finishing and fused gates lowered fresh 27K prefill median from 32.10 to 31.38 s at a 49,152 cap. First-use 16K wall got slower despite lower GPU spans. Wider 64-row PAIR1 jobs cut job count by 44% but increased routed projection time by 31–35%. [Wide dense IO and tile screens](2026-11-27-hy4-stq-kv-width-dense-io.md) | The catalog build enabled the qualified dense/register path after the opt-in screen. Keep 32-row routed groups. |

## Read the Sushi comparison at the right scope

Both engines used the pinned STQ pack on the 128 GiB M5 Max with vision
loaded. The OpenCode providers rendered their own prompts and used their own
cache widths and draft schedules. In the initial fresh 27K same-cap comparison,
Stark's tuned repeat measured 748.60 prefill and 49.93 useful decode tok/s
and Sushi measured 862.46 and 45.92. Sushi had a prefill lead in that run.
A later 4.29 s prefill gap compared separate runtime renderers and loads.
Within Sushi's own same-binary cache-width bracket, fresh 27K prefill took
29.08 s with KV8 and 27.90 s with BF16. The 27K output hashes differed across
widths. KV8 may save storage, but that experiment showed a prefill cost and
left the cross-engine prefill gap open. [Matched workload and phase-clock
details](2026-11-27-hy4-stq-open-code-performance.md) ·
[KV width controls](2026-11-27-hy4-stq-kv-width-dense-io.md)

The product path uses prepared mapped residency, 4,096-row prefill, F16 KV,
cached Engram reads and up to five DFly2 proposals for greedy text. The earlier
single-slot catalog run fitted 249,856 positions and measured 35.49–38.35 s
fresh 27K prefill and 45.73–51.25 useful decode tok/s. The 49,152-cap result
and the uncapped run differ in load and context. Use the caps and workload
labels when quoting these OpenCode diagnostics. The [standard
benchmark](../BENCHMARK.md) still needs matched rendering and same-artifact
natural-text quality evidence. [Defaults and lifecycle](2026-11-27-hy4-stq-catalog-defaults.md)

## OpenCode agent fan-out

The two-slot preview shared one load and scheduled GPU work at existing phase
boundaries. At fresh 27K, two slots cut median dispatch delay for two OpenCode
agents from 17.38 to 0.34 s. Their median first delta went from 52.44 to 64.39 s
as prefill work competed for the GPU. Four agents on the two-slot worker took
150.13 s to finish, compared with 154.16 s serial. The matched solo control
stayed near 52 tok/s. [HY4STQ-SERVE-01](2026-11-27-hy4-stq-dual-slot.md)

Sharing phase-local scratch made four slots fit on the same Mac.
The default four-slot worker fitted 45,056 API positions and 68,608 primary
positions. Each request kept the option to choose a smaller logical window.
All output hashes matched. Each row is one observation with the 96-output-token
budget. Solo decode on the same four-slot load was 51.15 tok/s at 27K.

| Fresh OpenCode inputs | Four-request wall | Median first delta | Median useful decode per agent |
| --- | ---: | ---: | ---: |
| 16,229 tokens each | 96.46 s | 81.52 s | 6.60 tok/s |
| 27,229 tokens each | 139.44 s | 126.97 s | 7.85 tok/s |

The two- and four-slot runs used different fitted windows and request order.
Four-slot evidence establishes simultaneous admission, exactness and the solo
rate at that load. Smaller prefill scheduling slices and batching compatible
decode work are open performance experiments. The same Desktop queue and
worker transport now serve other backends through the [shared-slot
design](../GENERATION_SLOTS.md). [HY4STQ-SERVE-02](2026-11-27-hy4-stq-four-slots.md)
