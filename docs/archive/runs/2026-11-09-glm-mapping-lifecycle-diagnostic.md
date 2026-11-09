# 20261109-GLMMEM-11: nonmodel mapping lifecycle diagnostic

State: preregistered source and CPU geometry controls. Native compilation and
Metal execution have not run. A later explicit resource-slot grant is required.
Predecessor: [GLMVIS-10](2026-11-09-glm-functional-pageout-screen.md).

## Problem and hypothesis

The one functional GLM attempt stopped on hard host compressor growth before
its first text completion. Its worker RSS and footprint stayed below 0.8 GB
while host available memory fell substantially. The active-resource source
inventory and bounded wrapper ownership did not establish the complete host
physical residency ceiling. No cause has been attributed.

The current source maps whole immutable files with read-only `MAP_PRIVATE` and
wraps bounded page-aligned ranges with shared Metal no-copy buffers. This
diagnostic asks whether mapping extent, mapping mode or explicit-copy storage
changes observable allocation/residency/release behavior on a small synthetic
fixture. It does not assume copy-on-write, a driver leak or shader compilation
caused the full-model interruption.

Apple's [no-copy buffer API](https://developer.apple.com/documentation/metal/mtldevice/makebuffer%28bytesnocopy%3Alength%3Aoptions%3Adeallocator%3A%29?language=objc)
requires page-aligned memory within one VM region and supplies a deallocator
callback. Its contract does not establish immediate physical-page reclamation.
Apple's [I/O Kit memory documentation](https://developer.apple.com/library/archive/documentation/DeviceDrivers/Conceptual/IOKitFundamentals/DataMgmt/DataMgmt.html)
separately describes preparation/page-in/wiring, completion/unwiring and
descriptor release. The [mmap contract](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/mmap.2.html)
defines private modifications as copy-on-write. These support investigating
the lifecycle; they do not establish what happened in the failed GLM run.

## Fixed small control

Use one owned 64 MiB synthetic file, opened read-only without following a
symlink. Its deterministic byte pattern and SHA256 are private receipt inputs.
Each case copies fifteen sequential 4 MiB logical ranges through completed
Metal blit commands. A 128-byte non-page-aligned offset exercises the rounded
mapping span while preserving blit alignment. No model weights, inference,
custom shader kernel or model downloads are involved.

| Case | CPU mapping | Metal storage |
| --- | --- | --- |
| Full private | Whole 64 MiB file, read-only `MAP_PRIVATE` | Bounded no-copy view |
| Dedicated private | One rounded range, read-only `MAP_PRIVATE` | Bounded no-copy view |
| Full shared | Whole 64 MiB file, read-only `MAP_SHARED` | Bounded no-copy view |
| Explicit copy | One rounded range, read-only `MAP_PRIVATE` | New bounded shared buffer copying the bytes |

One 4 MiB destination buffer persists across cases. At a 16 KiB page size the
dedicated source mapping is 4,210,688 bytes. The explicit-copy case additionally
owns its 4 MiB Metal copy. Metal/framework/driver allocations remain measured,
not inferred from those payloads. The fixture can warm at most 64 MiB of file
data; later arms share warm cache state, so this is not a performance comparison.

Record Metal `currentAllocatedSize`, actual view `length`/`allocatedSize`,
no-copy wrapper/deallocator counts and bytes, mapping/unmapping counts/errors,
`mincore` resident pages and Mach task RSS, footprint, internal, external and
compressed bytes. No-copy counters do not count explicitly copied buffers;
that case is observed through Metal allocation sizes and pool cleanup. Every
source/destination copy must match an independent synthetic checksum. Require
zero old wrappers before the next group, zero dedicated mappings after drain
and no unmap failures. These controls still cannot prove a full-model peak.

## Owner-only safety and evidence

The retained functional supervisor is unchanged, SHA256
`9860befa761a1b7fabc563ddb873e6854fcbeb072e52b7fe6755bb83bb096b6e`.
The toy driver narrows its RSS, physical-footprint and host wired-growth
ceilings to 1 GiB each, compression growth to less than 256 MiB and wall time
to 30 seconds. It retains the 64 GiB entry floor, 32 GiB available floor,
zero swap/Swapouts growth and continuous foreign-worker failure screen.
Pageouts remain timestamped diagnostics; all performance eligibility is false.

Only the driver-owned child can be terminated and reaped. Passing requires
positive process samples, no hard violation, the child drained, supervisor and
wall timer joined, sixty matching copies and all four case drains. Cleanup
removes only the originally created fixture with matching device/inode,
including a partially written fixture. Failed controls remain retained and
do not authorize larger inputs or relaxed limits.

Native source SHA256:
`02b8be84049fa94c328cfc05ac27de4ec2cc6a33af9def112efb38d91365523d`.
Driver SHA256:
`779d86f523d5d74fcc186d98315eb414e8944543c3dd48ba7e741dc62e977935`.
Private source registration:
`78e62a736bf345c54956b41619861a3cfbead88daeda7fca584e8d5ff61d9bad`.
It seals the exact future compile/run commands and records 45 passing CPU
geometry/pattern cases across 4, 16 and 64 KiB pages plus Python AST validation.
Those controls do not constitute native syntax, compilation or GPU results.

Independent source review, a compiler gap, successful native compilation and
an explicit small GPU slot remain required. The full GLM model remains held;
no production mapping/admission change, retry, cache/MTP or default promotion
is authorized by this proposal.
