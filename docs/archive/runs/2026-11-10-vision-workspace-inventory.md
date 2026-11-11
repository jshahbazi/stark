# Vision workspace inventory correction

This is the allocation portion of attempt `20261110-FAMILY-RECHECK-01`, based on
branch head `cbd8b022`. Implementation and controls are registered before root's
integrated compilation and serialized native execution. No inference arithmetic,
source/weight/activation/cache precision, preprocessing geometry or image bounds
change. No new model payload is downloaded or installed.

The initial `20261110-IMAGE-01` estimate summed a loose block bound across every
layer. A normal square Gemma 4 E2B image resizes to 768 by 768 pixels and was
charged approximately 5.3 GiB of tower workspace. That estimate did not reflect
the pool's actual storage lifetime and could falsely reject requests.

## Source-backed allocation contract

`PoolAllocator::allocate` rounds each scratch buffer to a power-of-two size
class. Dropping `PoolAllocation` returns its page to that class immediately.
Metal's command access tracker inserts read/write barriers before the same
physical range is reused, so correctly ordered dispatches can reuse scratch
within one command. The upstream Metal block allocator has a 16 KiB minimum.
Each inventory rounds individual buffers using both allocator rules.

The revised bound counts every scratch allocation in one full block, plus every
initial and final stage allocation. This deliberately exceeds simultaneous
storage: it bounds each retained size class even if a block holds all its local
results until scope exit. Later identical blocks have the same shape and
allocation schedule, with only their incoming hidden owner retained. The
initial-stage inventory already covers that owner. Scratch does not multiply
by layer count.

Constants use a separate 256 KiB minimum-page bump allocator. They persist for
the submitted command, including after a Rust constant handle is dropped. The
inventory counts the actual upload sequence, alignment and unused page tails;
Qwen's per-block scalar constants include the full depth. CPU prepared patches,
temporary position/rotary vectors, previous zero features, device readback and
new CPU features are also charged. An explicit 32 MiB plus 1 MiB per active
command block remains as command/argument-table and general allocation margin.
This is payload admission with margin, not a measured peak RSS guarantee.

| Source-admitted tower | Scratch inventory | Persistent constants and owners |
| --- | --- | --- |
| Qwen BF16/affine vision tower | Initial projection; one block's raw F32/BF16 linears, normalizers, Q/K/V, eager score pair and residuals; complete merger | Patch, interpolated learned-position and rotary uploads; per-block scalar constants; CPU staging and projected-feature copies |
| Gemma 3 fixed SigLIP | Initial submission, one full 4,096-row block and final pooling/projector submission summed; each block waits before the next | One patch upload and block rotary constants; persistent hidden global buffer, CPU patches and projected copies; cached stage pages conservatively covered by the sum |
| Gemma 4 E2B/E4B | Initial upload/projection, one block including clipped linear input/output boundaries and online attention, final pooling/projector | Patch, position, rotary and unit-normalizer uploads; BF16 CPU vectors and projected copies |
| Gemma 4 larger encoder | Same stage structure with unclipped linears, padded Q/K/V and the actual fixed FlashAttention buffers | Four flash buffers have shapes `[heads, rows, 512]` F32 and BF16, `[heads, rows, 3]` F32, and `[heads, rows, 72]` F32; each is rounded separately and reused across key tiles; no N-squared score plane |
| Gemma 4 unified 12B | Entire encoder-free patch normalization/projection, position and output projection sequence | Patch upload, prepared CPU patches and projected copies; no encoder block or attention scratch |

Unsupported geometries, zero shapes, scratch buffers above the pool's 2 GiB
limit and checked arithmetic overflow return no estimate, preserving recoverable
request rejection. Existing resident weights remain outside request-owned
workspace; normal load admission and current headroom already account for them.

## Fixed controls

The profile inventory test checks normal E2B admission stays below 1 GiB, covers
all admitted profile shapes and rejects invalid geometry/overflow. A separate
constant test compares the estimator with the real bump allocator across large
uploads and enough scalar appends to cross its shared page boundary.

`vision_tower_workspace_bounds_pool_reuse_and_constants_metal` constructs a
small synthetic production Qwen tower at depths one and three, executes its
normal encoding path, and reads actual command constant and scratch-pool
reservations. Rectangular 24-row geometry and non-power-of-two widths exercise
rounding. Each actual reservation must be below its corresponding inventory;
pool reservation must be identical across repeated blocks, while constants
remain separately accounted. The test compares implementation allocations,
not a second transcription of the bound's formula. Its synthetic zero weights
and features stay private and are not model-quality evidence.

Root owns final build/executable identities and passing/failing outcomes in the
[double-check record](../../model-experiments/2026-11-10-family-double-check.md).
No image-artifact loading, full-model image generation, default-image text/MTP
qualification, measured throughput or RSS comparison is established by these
bounded allocation controls. Every production artifact retains its own gates.

The first integrated compile rejected test-only access to the private allocator
module, ambiguous backend inference for the synthetic tower constructor, and a
missing Buffer trait import. The constant-page control now exercises the normal
Metal command allocator through its existing test probe, preserving production
module visibility. The constructor and import were corrected. These compile
failures occurred before runtime validation and establish no passing result.

## Coordinated allocation results

All three revised controls passed in the root-coordinated CPU/Metal executable
on 2026-11-10. The profile control already emits content-free computed budgets
through its `vision-workspace-inventory` aggregate record.

| Computed request-owned workspace | Bytes |
| --- | ---: |
| Gemma 4 E2B square image: 2,304 rows, output width 1,536 | 412,221,440 |
| Gemma 4 larger encoder: 2,520 rows, output width 5,376 | 832,783,872 |
| Gemma 3 4B fixed tower/projector | 829,095,936 |
| Gemma 4 unified maximum 280 image tokens | 98,078,720 |

The revised square E2B admission is 393.125 MiB, compared with the original
approximately 5.3 GiB bound. These are computed allocation budgets, not observed
production-artifact RSS or inference-speed measurements.

The production Qwen tower path used synthetic zero weights at one and three
blocks. Both runs observed 262,144 bytes of command-constant pages and 125,952
bytes of pool reservation. Their respective inventory bounds were 262,144 and
475,136 bytes. Full admission including host storage and margin was 35,359,872
bytes at depth one and 37,457,024 at depth three; only the explicit command
margin changed with depth. The observed scratch reservation did not grow when
identical blocks repeated. The independent constant control also passed across
the scalar bump-page boundary.

The root's private logs are `runtime/family-double-check-tower-{0,1,2}.log`;
final executable/source identities remain in the linked double-check record.
