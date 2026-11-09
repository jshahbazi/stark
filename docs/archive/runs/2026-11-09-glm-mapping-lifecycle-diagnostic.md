# 20261109-GLMMEM-11: nonmodel mapping lifecycle diagnostic

State: preregistered source/CPU controls and the one independently reviewed
tiny Metal lifecycle run pass. The full-model compression failure is not
reproduced or explained; full-model execution remains held.
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
symlink. Each 64-byte record starts with its absolute little-endian 64-bit
record index and ends with 56 deterministic bytes. Its SHA256 is a private
receipt input. The original prepared pattern repeated every 256 bytes;
independent review found that wrong 4 MiB-spaced ranges could pass its checksum.
That prepared source never ran on Metal and its registrations are retained.
The revised pattern distinguishes those absolute ranges.
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

Before any Metal work, a CPU negative control reads the next actual fixture
range. It must reject the first range's expected hash and match its own
absolute-position hash. This adds no GPU blit. A separate native
`--cpu-control` compares the distinct absolute-position expected hashes and
exits before device creation.

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
`902bf4d53c3d911343fc2544fef12d022979242ca53584722a4c909f45035460`.
Driver SHA256:
`3647d5e023bf9a7ecdc7e4f7636e9f0acce85d2e38ff3dc73c6d61f63f60bbfd`.
Private source registration:
`39f7d634c2bbf05ba7a7db85d727f04221c19145849082ad8a74f1a0498dc19a`.
It seals the exact future compile/run commands and records 45 passing CPU
geometry/pattern cases across 4, 16 and 64 KiB pages plus Python AST validation.
The tiny native build separately passed with `-O2 -Wall -Wextra -Werror`, zero
warnings/errors, and its CPU wrong-offset expected-hash control passed without
creating a Metal device. Native binary SHA256:
`303664cc6171d01112b127aa5899f2a20d4393cdcaa4609d0dc8df9b3c0ebb0d`;
build/CPU receipt:
`df478fe230ff6f12152be6a077897e05e3df6ce99bf666c6fbc5dbd65f5bcac7`.
The driver checks those exact source/binary/build-receipt identities before
fixture or child creation and adds received wall/monotonic timestamps to the
native aggregate events. These CPU controls do not qualify Metal execution.

Independent review of the revised sealed recipe and an explicit small GPU slot
were completed before execution. The pre-execution checkpoint is
`001d6d46ea583aafb91c7ff00b54558470a12d6d1e63a989ebea641858685d89`.

## Actual small lifecycle result

The one granted run passes all four cases, sixty matching completed blits and
the actual wrong-offset CPU rejection. It records 306 aggregate phase events,
45 no-copy deallocator calls and zero unmap failures. The fixture was removed;
the child drained, supervisor joined and wall timer joined. The exclusive slot
was released directly to Qwen. Full private receipt:
`9c9daec77cbc6ef58845fd54ca1ea1c590b5fc1550222f6a72d2fbc2604c9994`.

The supervisor has three process samples, no violation and zero Pageouts,
Swapouts, swap-used or compressor growth. Minimum available bytes were
106,728,243,200; peak host wired growth was 368,001,024 bytes. These observations
do not constitute a performance result.

| Case | Peak Metal allocated bytes | Peak native task footprint bytes | Maximum mapped resident pages |
| --- | ---: | ---: | ---: |
| Full private no-copy | 8,863,744 | 135,430,744 | 4,096 |
| Dedicated private no-copy | 8,863,744 | 135,561,816 | 257 |
| Full shared no-copy | 8,863,744 | 135,692,888 | 4,096 |
| Explicit copy | 8,847,360 | 144,196,184 | 257 |

At 16 KiB per page, those mapping observations represent 64 MiB and 4,210,688
bytes of resident backing respectively. The fixture is warm, and those file
pages must not be added to footprint as if they were unique committed copies.
The no-copy views report both length and allocated size as 4,210,688 bytes;
the explicit copy reports 4,194,304. Metal allocated bytes return to 4,653,056
after every case.

The first private view changes Metal allocated bytes from 4,653,056 to
8,863,744 while task footprint stays near 8.57 MB. After the first completed
blit, task footprint is 135,266,904 bytes. Pool drain returns its wrapper
count/bytes to zero and Metal allocated bytes to 4,653,056, while task footprint
remains 135,266,904. This directly distinguishes wrapper deallocation from
complete task-footprint reclamation. It does not identify the retained
footprint's implementation category or attribute the full-model failure.

No compression failure was reproduced. The result does not select private,
shared, dedicated or copied storage as a production fix. The full GLM model
remains held. No production mapping/admission change, retry, cache/MTP,
performance or default promotion is authorized by this result.
