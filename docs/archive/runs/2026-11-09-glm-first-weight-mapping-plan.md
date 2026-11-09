# 20261109-GLMMEM-16: first-weight mapping source plan

State: source/header audit and separately authorized CPU preparation pass.
Minimal three-way native control source is frozen, unbuilt and unexecuted.
No actual file mapping or Metal execution. Full GLM execution remains held.
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

## Authorized CPU preparation

After review of this scope, the private Python preparer read exactly 23,068,672
original-file bytes with successful Darwin `F_NOCACHE`. Descriptor identity
matched the header audit before and after reading. It did not map the source
or invoke Metal/native code.

Three read-only private sparse files have 64/256/512 MiB extents and the same
23,085,056-byte initialized view, 9,408-byte logical offset and exact tensor
payload. Actual allocated disk totals 69,255,168 bytes. Four invalid-range
controls rejected before source I/O, and all three fixtures rejected an actual
one-byte wrong-offset hash comparison.

The private preparation receipt is
`5e112a4b7ab2325a86a3496fef09a804136e1c1359ae76d8f20f6f5fb7a686e3`;
the logical packed payload hash is
`6985b8b6a850af5095bbe7696e5049af2ce7aa1dbbfc81341dd15d8dd442cf1b`.
Fixtures and raw bytes remain ignored/private. Native control and watcher
implementation, source review, build and execution are still pending.

## Final minimal scope

Review identified that sparse holes can use shared zero pages. The unexecuted
sparse fixtures remain separate. A new private set writes deterministic
SHAKE256 bytes seeded by absolute chunk index throughout its 64/256/512 MiB
extents, then restores the identical 23 MB view. All 872,415,232 bytes are
materialized. It derives from authenticated private bytes and reads zero new
bytes from the original GGUF. Receipt:
`16a90b68f2a9a6b265464687bfab5c48ac4f31d10a7d453d92a5586a31390624`.
Preparation can warm file backing; it does not establish a cold-paging control.

Following the instruction to keep the fix focused, the frozen control uses only
the existing 512 MiB fixture in three fresh children: owned buffer, dedicated
range NoCopy, and full-region NoCopy. The dedicated arm also retains a full CPU
mapping, matching production's simultaneous CPU owner. There is no additional
format, backend, upload policy or fixture variant to implement.

The control reuses the reviewed first-use supervisor `125d1a40…`, tightening
only compressor growth to zero. Twenty-two reused/injected CPU controls pass,
receipt `40ae58673deb5d7669c0d0a97fe589d25a5f66893dfdd0b6f9a0c3e06d73dbe5`.
Frozen source registration is
`da149ce31e5f46d8b207a257a807d58c5dd73dbbb1deec66ec3a8eee604d1a6b`;
native patch `4be38da7145dd0ca10b7c0850254312c53de4ec3015477aa3e52d8c5d9da29b7`;
driver `f1809553d7a975bf93ca71dc85bc97c83c7ed73c1dad07ce348bd57284c05921`.
Formatting, repository policy and diff checks pass. Build and Metal execution
require a separate grant. If this bounded contrast cannot choose a minimal
per-view ownership fix, stop with the concrete pressure blocker. It does not
clear the original 49.3 GB backing object or full model by itself.

## Completed bounded comparison

The separately granted build reused this checkout's existing target and archived
the prior `8e7abaee…` test binary with an APFS clone. A private build-driver path
correction selected that existing binary as the archive source; the original
driver and frozen native registration remain preserved. The build passed in
95.87 seconds, all 1,272 source hashes remained unchanged, and minimum free disk
was 20,105,019,392 bytes against the 8 GiB reserve. The 22 injected supervisor
controls passed again. Build seal:
`101aff79bf048a5165e67f0aefdfa4cd96e87c4d9873b20fdc2945e9e122292b`;
binary:
`0067192c9ee0f7594c191d792786de0b338550b82b08859780fd2debc572cbc1`.

All three fresh children passed all 11 phases and the independent projection
oracle, with identical output hashes. The original GGUF was never mapped by a
child. Each guard reported positive live samples, four post-drain samples, no
violation, successful drain and supervisor join. Pageout, swap and compressor
growth were zero in every arm.

| Arm | Live samples | Sampled peak RSS, bytes | Sampled peak footprint, bytes | Sampled host wired growth, bytes |
| --- | ---: | ---: | ---: | ---: |
| Owned | 2 | 19,677,184 | 8,913,496 | 13,074,432 |
| Dedicated range plus full CPU owner | 1 | 8,847,360 | 3,031,400 | 785,268,736 |
| Full region | 1 | 8,863,744 | 3,047,784 | 0 |

These sparse live samples do not bound brief GPU or process peaks. In every
arm, phase-attested Metal allocation peaked at 24,608,768 bytes and returned
to 458,752 bytes after test owners drained, matching the library-ready phase.
This remaining allocation and the owner counters do not prove physical
reclamation. CPU mapping counters returned from two in the dedicated arm and
one in the full arm to zero, with zero unmap failures. Sampled positive
compiler-service RSS growth was 59,162,624 bytes in the owned arm and zero in
the other arms. Service RSS and host-global wired growth are distinct metrics;
the dedicated arm's host-global increase has no established attribution.

Private aggregate receipt:
`4654bf3635ffd78b0d25dd2fb26986af823c9f649411e5697f074df71b6fd704`.
Backing and shader cache warmth remain unknown, and this is not a performance
comparison.

The contrast did not reproduce the full-model pressure or distinguish a
production ownership fix. Investigation stops at this agreed boundary. No
mapping default, backend, upload policy, admission limit or production
arithmetic changed. The concrete blocker remains the real full-model first
request's compression growth, with its cause unresolved. The original 49.3 GB
backing object and full 109 GB model remain held; image generation, default
promotion, cache, GGUF MTP and matched performance qualification remain open.
