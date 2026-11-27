# Hy4-Flash Sushi STQ metadata preflight

2026-11-26 · HY4STQ-PREFLIGHT-01 · Local format preflight passed

## Artifact and evidence scope

The local `beamster/Hy4-Flash-Sushi-STQ-4bpw` pack has internal revision
`capture-2a0815359e97` and matches the expected layout for source revision
`61528ba635a623f127c5a7b1e1f4ac2c2b0d4885`. Its 99 files total
208,397,806,219 bytes and include 89 Safetensors shards. The preflight
checks the config, index, headers and file sizes. It reads zero bytes of
tensor payload. The earlier whole-pack verification remains separate.

| Pinned file | SHA-256 |
| --- | --- |
| `config.json` | `c1c5af8449a8e7db0dc5c5e8aa768c80673c94604997cfe13c6ee6aa98e9a3e3` |
| `model.safetensors.index.json` | `6a3d938ac63427349ff8e647edd894cdfa00d3075c45cda4d1e7b662a99359ca` |

## Observed format inventory

- The index lists 3,179 tensors and 208,387,139,446 tensor bytes. All
  indexed names and shard assignments match the headers.
- The 42-layer trunk has 41 routed MLP layers. Its 123 gate, up and down
  expert groups use PAIR1 K4, window 16 and pair-major tiles. The first gate
  plane is U16 `(512, 48, 96, 2, 64)` with `.suh` shape `(512, 1536)`.
  The codebook is `pair1-w16-v1` with digest prefix `c089b4eb`.
- The trunk uses SPEC1 K6, window 15 and codebook `spec1-w15` with digest
  prefix `37f72431`. The first indexed query projection is U16
  `(192, 64, 96)`. The vocabulary head is U16 `(192, 7552, 96)`.
- DFly2 has five layers and five speculative tokens. Its SPEC1 K4 query
  and gate examples are U16 `(192, 64, 64)` and `(192, 576, 64)`.
- The vision tower and projector have 441 tensors. Its BF16 patch embedding
  has shape `(1152, 3, 16, 16)` and its projector example
  `(4608, 1152, 2, 2)`.
- The four BF16 Engram tables are 96 channels wide and have 165,008,942,
  165,004,086, 164,999,008 and 164,994,380 rows in shard order.

`model-dfly.safetensors` has a six-byte zero alignment gap and
`model-globals.safetensors` has a four-byte gap. Those two shards opt in
to a check that permits at most eight zero bytes between tensors. The
other shards use the strict reader, and all shards require an exact end
offset.

## License provenance

The pack identifies `tencent/Hy4-flash` as its base and includes the
Tencent Hy License V1.0. The format preflight adds a Stark reader check
for the observed layouts. A later decoder port must preserve the notices
of the source it adapts. [Pack license](https://huggingface.co/beamster/Hy4-Flash-Sushi-STQ-4bpw/blob/61528ba635a623f127c5a7b1e1f4ac2c2b0d4885/LICENSE) · [Sushi source notices](https://github.com/beamivalice/sushi/blob/v1.8.1/NOTICE)

## Result and remaining loader work

Native use requires PAIR1 and SPEC1 decoders, integration of the STQ tensor
views and Engram row reads into the Hy4 graph, and execution of the
vision tower and DFly2 head. After the first model load, qualify image,
tool and text behavior, context sizing, cache restoration and performance
on this exact pack. Catalog installation stays unavailable until those
paths are qualified.

## Implementation and checks

Run `python -m stark_convert.hy4_stq_preflight <model-directory>` from the
converter's locked environment. It checks the pinned config and index
hashes, file inventory, quantizer fields, tensor mappings and shard
metadata before returning a report.

Checks run on 2026-11-26:

- The preflight passed against all 89 local weight-shard headers.
- Both focused tests passed. Ruff formatting and lint passed.
