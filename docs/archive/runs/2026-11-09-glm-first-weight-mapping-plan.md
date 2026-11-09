# 20261109-GLMMEM-16: first-weight mapping source plan

State: source and one pinned header audited. No tensor payload read, file
mapping, new native build or Metal execution. Full GLM execution remains held.
This follows [GLMMEM-15](2026-11-09-glm-first-use-control-diagnostic.md), which
passed its no-file tiny Control/Q5 discriminator without reproducing pressure.

## Actual route and geometry

The cleared-selector first layer calls `Weights::encode_matvec` for
`blk.0.attn_q.weight`. It calls `mapped_matrix` with index zero and
`cacheable=true`, then `metal_view`/the static FIFO, `Mapping::metal_slice`,
`metal_span`, `weight_metal::mapped_slice` and `graph.mapped`.

`Mapping::new` owns a whole-shard read-only `MAP_PRIVATE` mapping. The Metal
NoCopy buffer covers only the page-rounded tensor range, while its deallocator
keeps the entire Mapping owner alive. The static cache retains this wrapper.
The cleared recipe does not request explicit residency for it.

| Quantity | Bytes |
| --- | ---: |
| Whole second-shard file | 49,294,975,936 |
| Logical Q5_K tensor (8,192×4,096) | 23,068,672 |
| Page-rounded Metal view | 23,085,056 |
| Logical offset within Metal view | 9,408 |
| Host page | 16,384 |
| Pinned header read for this audit | 42,155 |

The tensor formula is `8192 × (4096 / 256) × 176`. Header bytes match the earlier
source-bound inventory. Private descriptor identity, source hashes and exact
file offsets are in `mapping-source-audit16-receipt01.json`, SHA-256
`328d8d183893344185c8ee66eaecb05d4f6fa9e1d89cc31f17227e3aef64a30d`.

The 23 MB Metal length does not establish a 23 MB physical ceiling. A dedicated
range mapping into the original 49.3 GB file still shares its backing object.
Neither original-file GPU arm is granted. This source relationship does not
attribute the earlier compression event to a VM entry or file pager.

## One bounded experiment proposal

Read exactly the logical tensor through an owned descriptor with `F_NOCACHE`,
checking file identity before/after and sealing the bounded bytes. Do not map
the original GGUF or give its file-backed pages to Metal. The only original-file
payload operation would be this 23 MB bounded read.

Use those identical bytes, a fixed synthetic F32 input and the actual first
Q5_K shape/kernel in capped private fixtures. Keep Metal view length 23,085,056
and logical offset 9,408 identical in every arm, including owned storage. Padding
is initialized and immutable. Possible arms are owned storage, whole private
64/256/512 MiB fixture mappings, and a dedicated view into the same 512 MiB
fixture. The last contrast separates VM extent from fixture backing extent.
Each arm needs a fresh child; the earlier four-case blit toy shared a process
whose retained footprint confounded cross-case comparisons.

The CPU oracle decodes only one packed block at a time and uses F64 products;
it does not expand a full matrix. Require finite outputs and identical packed
bytes, input, shape, pipeline family and numerical output across arms. Preserve
an explicit wrong-range negative control. All artifacts remain private; public
records contain counts and hashes.

Proposed source-managed payload is about 23 MB host bytes, a 23 MB GPU view or
owned buffer, small input/output/oracle storage and bounded fixture-writing
chunks. File-backed VM reservation remains separately reported, never equated
with committed memory. All fixture extents are capped at 512 MiB. No causal or
total-physical ceiling follows from these source quantities.

The separate proposed supervisor retains the owner start gate, phase markers,
positive per-PID compiler-service growth, cleanup and post-drain observations.
Keep 1 GiB own RSS/footprint/host-wired/service-growth caps, 64/32 GiB available
floors, zero swap and compressor growth, and a bounded deadline. Global Pageouts
remain separately recorded with no performance claim. Any hard guard stops the
experiment without continuing to later arms or retrying. Exact source, CPU
controls, fixture bounds and supervisor must be independently reviewed before
any new build or execution grant.

This is one pressure discriminator. It implements no backend, upload policy,
mapping default, admission relaxation or production text-performance change.
Growth with bounded fixture extent would be evidence to investigate, without
clearing a 49.3 GB backing object or the full 109 GB model.
