# Hy4-Flash Sushi STQ native bring-up

2026-11-27 · HY4STQ-NATIVE-01 · Native component and trunk diagnostics

## Artifact and build

The source is `beamster/Hy4-Flash-Sushi-STQ-4bpw` at revision
`61528ba635a623f127c5a7b1e1f4ac2c2b0d4885`. The exact 208,397,806,219-byte
pack was verified separately for the Sushi run. The Rust reader checks its
configuration, index, inventory, file identities, all 89 shard headers and
the Engram token map. These open-time checks do not rehash all weight payloads.
The diagnostic borrows the original immutable release files throughout GPU
execution. [Artifact inventory and quantizer identifiers](2026-11-26-hy4-sushi-stq-preflight.md)

The implementation reuses the Hy4 GGUF backend's hyper-connections, MLA,
IndexPool, head, context sizing and rewind. It maps the PAIR1 routed banks
and SPEC1 dense matrices directly and converts the joint SPEC1 K/V
projection into the graph's absorbed BF16 views at load. Engram reads the
four original BF16 tables via positional reads. The separate STQ gate
preserves the order of the independent key and query RMSNorm weights.

The packed codecs and Metal kernels adapt Sushi v1.8.1 at
`b7b7975cc017acba1b4c0877d65543c4c5a4a381`. The MIT and Apple MLX notices
ship alongside the kernels. Tests ran on an M5 Max with 128 GiB, Metal 4
and Rust 1.98.1. GPU runs held the shared Hy4 lock.
[Sushi source and notices](https://github.com/beamivalice/sushi/tree/v1.8.1)

## Component evidence

| Check | Result and scope |
| --- | --- |
| Pinned inventory | 3,179 tensors and 1,256 trunk aliases. Bindings covered 123 PAIR1 K4 expert banks, 420 SPEC1 K6 trunk projections and 44 SPEC1 K4 draft projections. |
| CPU tile codec | Three upstream fixtures, 96 seeded tiles, 256 real tiles and four 128-by-128 blocks matched the independent decoded reference. Component output SHA-256 `b8eb84d26a7e6491d5c3cfb8a0e61deaf71cbeaff0d46bb3851c6d91d06b6b94`. |
| Absorbed KV | The actual 32-head and 64-head projections matched an independent reconstruction on 22,020,096 BF16 output words. The CPU oracle uses the same documented rounding policy. |
| PAIR1 Metal | Three real layer-one projections matched the rounded stage references at relative RMS at most `1.55e-5`. The mapped fused gate/up/down component reached `5.80e-5` relative RMS against the independently rounded CPU chain. All 4,608 weighted combine values matched in router-slot order. |
| SPEC1 Metal | Real K6 trunk and K4 DFly2 query projections at 3,072 by 1,024 passed singleton and 39-row cases. Relative RMS at 39 rows stayed below `2.07e-5` against the matched rounding reference. Full chain warm command medians were 0.433 and 0.427 ms respectively. |
| Packed embedding | 36,864 F32 values for 12 real ID occurrences, with a relative RMS of `2.14e-7` against CPU block reconstruction. Repeated rows matched exactly. Out-of-range inputs were bounded. Warm command medians were 0.034, 0.127 and 1.600 ms at 1, 32 and 512 rows. |
| Engram | All 320 sampled rows matched independent positional reads after closing the original descriptors. The independent gate check covered 12 streams, four exact zero-dot 0.5 gates and a maximum finite-normal absolute difference of `9.32e-9` in the gate. |

The first PAIR1 shader compile used a reserved Metal identifier. The
corrected shader compiled and passed the real-bank checks. An extreme
subnormal Engram gate stress case exceeded the initial absolute
tolerance, with an absolute gate difference of `2.91e-6`. The finite-normal
reference passed and distinguishes independent key/query products from a
swapped weighting.

A private contiguous dense prepare prototype preserved exact output bits
and reduced median 4,096-row GPU command time from 0.476 to 0.201 ms on
one 3,072-wide shape. This is component evidence. The forward integration
initially uses the validated general prepare to keep its materialization
and schedule changes independently testable.

## First full native trunk runs

The 1,024-row graph opened with a fitted 4,096-position context. An
eight-token diagnostic produced finite vocabulary logits and completed
two greedy decode steps. Load, first prefill and decode took 41.730,
16.798 and 0.134 seconds respectively.

The 4,096-row graph opened with an 8,192-position context and completed
a 4,096-token synthetic prefill and eight decode steps. It took 38.308 s
to load, 26.985 s to prefill and 1.208 s to decode, or 151.8 and 6.62
tok/s on this first request. Of the prefill wall time, 6.227 s were
reported inside GPU command spans and 1.343 s in Engram preparation
and reads. Zero software expert-read bytes are expected with mapped
banks and do not measure file-backed page faults. Output SHA-256
`69b44c11ad119a268c13216bbe3a97709ccccb6e534260b6e366b6cbdcb154b3`.
The fused MoE's indirect command matched the direct command bit for bit
on the independent real-layer component fixture.

## Qualification boundary

The native entry is a development diagnostic. The catalog remains a Sushi
entry while the exact pack's vision tower, DFly2 execution, same-pack
output checks and 16K/27K OpenCode matrix are pending. Trunk diagnostics
use deterministic synthetic inputs and emit aggregate timings and hashes.
