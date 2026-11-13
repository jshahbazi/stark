# Retained ddalcu default-image memory plan

Attempt: `20261109-QNIP-05`. Source/header review only; no model was loaded for
this plan. The installed source remains untouched. Execution stays held pending
independent review and an exclusive hardware slot.

The exact source is
`ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit` at
`788c07a8725642bccf13a0a3b8a21954ab84b4f2`. Its complete 101 shards total
75,303,252,216 bytes; `ngram_table.bin` adds 32,000,153,976 bytes, for
107,303,406,192 bytes overall. An ignored private view hardlinks payloads and
copies only metadata. It has no admitted image manifest yet. Native full-payload
hash admission and the ordinary strict image upgrade remain prerequisites.

## Actual storage and loading

The header inventory covers all 102 files, with per-header hashes and source
hashes in private receipt
`dcd593dab1b589a4cf013506324aedb3d042a8b2e79891c1080de11e9e1a1716`.
These categories count source bytes once:

| Component | Tensors | Payload bytes |
| --- | ---: | ---: |
| Target experts | 432 | 67,947,724,800 |
| Other target tensors | 2,330 | 4,945,279,000 |
| Predictor, including its own experts | 72 | 1,511,927,296 |
| BF16 image tower | 333 | 897,862,112 |
| Paged n-gram planes | 3 | 32,000,153,600 |

This Lily-derived loader reads target, expert, predictor and vision tensors
directly into new, zero-filled shared-storage Metal buffers. It requires
resident experts; the vendored loader rejects expert-slot serving. Fused
projections concatenate directly into their destination, and their segment
views share that buffer. GDN F32 promotion adds 12,672 bytes to the selected
stored payload. Counting every byte of every non-table shard, including its
headers, gives a conservative resident-weight upper inventory of
**75,303,264,888 bytes**. This includes the whole predictor even at MTP0 and
the whole tower with ordinary text. No active-expert residency assumption is
used.

The n-gram table is a read-only shared file mapping, not a 32 GB Metal tensor.
Gathering stages selected row codes/scales/biases into bounded GPU buffers.
The serving load does not invoke the optional whole-table preload. Its file
pages can nevertheless remain cached, and the exact paged-text policy's one
GiB working-set allowance does not enforce a one-GiB resident cap. The peak
scenarios below conservatively count the **entire 32,000,153,976-byte mapping**
as resident. File cache from hashing or other processes remains a separate
host observation, never an assumed zero.

## State and request peak

The serving wrapper initially allocates capacity **16,384**, even when a
request asks for context 2,048 or 8,192. This floor is retained in the inventory.
The loaded head counts as an additional attention cache and allocates its
verification scratch even when the current request selects MTP0. The core
request estimate includes the live recurrence, two retained/replaced snapshots,
attention/indexer caches, two growth generations of state and scratch,
prefill routed-expert buffers, PLE staging, and host sampler/history allowances.
The wrapper's independent host workspace is counted again as its own owner.

| Input tokens | Prefill allocation rows | State inventory | Scratch inventory | Combined request upper bytes |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 16 | 649,277,440 | 442,278,144 | 2,232,460,288 |
| 1,025 | 2,048 | 649,277,440 | 1,841,630,720 | 5,031,197,728 |
| 4,098 | 4,096 | 649,277,440 | 3,196,235,264 | 7,740,505,152 |
| 7,168 | 4,096 | 649,277,440 | 3,196,235,264 | 7,740,603,392 |

The combined column includes two growth generations and host allowances; it is
not the sum of only the two preceding columns. A retained cache test needs
more than 4,096 prompt tokens because this backend's snapshot grid is 4,096.
The normal image tower has separate persistent, growing scratch and CPU image
staging; reserve another 512 MiB for its bounded 4,096-patch serving limit.
This inventory also keeps two GiB for allocator/driver overhead and sixteen
GiB for the host.

With all n-gram pages resident, the short 16-token case totals
**129,400,102,896 bytes** including those margins. The 7,168-token scenario
totals **134,908,246,000 bytes**, leaving only 2,530,707,472 bytes beneath
128 GiB physical memory. The margins are planning allowances, not a proven
upper bound on driver allocations or a reservation of currently available RAM.

## Review and execution boundary

Do not approve the complete lifecycle/ABBA run from the apparent free-plus-
inactive value alone. First independently check the header categories, mirrored
request arithmetic, the 16,384-capacity floor, loaded predictor buffers, tower
scratch and file-cache treatment. Full-image preparation must preserve the exact
source/config/index identities and create its manifest only in the private view.

If the coordinator approves a bounded first probe, begin with load/readiness
and one short request. Use an owned-process supervisor from before payload
hashing through shutdown, monitoring both RSS and physical footprint, global
available memory, compressed-page growth, pageouts and swap. Proposed starting
limits are 96 GiB RSS/footprint, a 32 GiB observed available-memory floor, zero
pageout/swap growth and at most one GiB compressor growth. These limits may
resource-defer a legitimate large mapping; that is an observed outcome, not a
reason to relax them. Continuously reject foreign model workers and terminate
only the owned process on a guard failure.

Only observed, eligible startup/request results can support a later near-
capacity request. The proposed full-mapping scenario has little remaining
margin, so complete cache, MTP2 and image execution cannot be assumed safe or
qualified. MTP3 is a separate option-switch control outside the exact paged
text policy's maximum depth two. No result from retained Lily, BF16 dense,
affine MoE or an installed worker transfers to this exact artifact.

Source boundaries: `vendor/lily-core/src/weights.rs` directly allocates and
reads tensors; `qwen4exp/weights.rs` loads every resident expert and selected
head/tower; `qwen4exp/ngram.rs` owns the separate file mapping and gathered
staging; `qwen4exp/request_memory.rs` defines the mirrored request inventory;
`qwen4exp/vision.rs` defines the tower scratch; and
`crates/stark-engine/src/engine/hybrid/lily.rs` sets the initial capacity and
loaded-head/KV policies. These runtime source hashes are sealed in the private
receipt above. There is no new download, model deletion or full export/copy.
