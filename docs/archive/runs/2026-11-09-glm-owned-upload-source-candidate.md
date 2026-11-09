# 20261109-GLMMEM-12: owned packed-weight upload source candidate

State: bounded private CPU prototype controls pass. No production integration, Metal
execution, model run or memory-policy change is authorized by this record.
Predecessor: [GLMMEM-11](2026-11-09-glm-mapping-lifecycle-diagnostic.md).

The small mapping control did not reproduce the full-model compression
failure. It also did not establish a physical ceiling from wrapper counts.
This candidate investigates explicit ownership of packed-byte GPU inputs:
read an admitted descriptor range with caching disabled into one separate
page-aligned anonymous allocation, then eventually retain that allocation
through the Metal deallocator. Normal source arithmetic and packed bytes must
remain unchanged. No prior failure cause is assumed.

The first bounded increment is an ignored standalone Rust CPU primitive with
no model or Metal API. It must validate regular/no-follow descriptor identity,
checked range/rounding arithmetic, an explicit live-byte budget, unchanged
source signatures across exact positional reads, and physical unmapping
before its budget reservation is released. Failed cleanup must fence the
budget. Tests use only a tiny owned synthetic file and preserve all existing
model/diagnostic receipts. No source weights are loaded.

Apple's [fcntl contract](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html)
supports disabling descriptor data caching with `F_NOCACHE`. Its
[file-system guidance](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/FileSystem/Articles/FilePerformance.html)
describes this control for streaming reads. This is a proposed I/O ownership
mechanism, not evidence that it will solve the observed GLM failure or preserve
normal-text speed.

The complete target expert-bank inventory is 99,033,808,896 bytes. Reuploading
every complete bank for each target token would require at least that much
input I/O per token, before statics and other work. That whole-bank route is
reference-only and nonpromotable as the unchanged normal-text path. This
bounded CPU primitive is useful for ownership/byte-preservation reasoning;
it is not a whole-backend implementation or a performance-recovery plan.

The bounded private Rust source is prepared and formatted, SHA256
`6ef774e2293c0ccd91ccf6d62aed2de47c24843fb6cd9d77d3dc80825f11c7ad`.
Its source registration is
`1d9f4d1f622d99fd4fcf453428c049113b68549d148b6a27b43f86fc1a57f155`.
Four CPU controls cover exact bytes/alignment/live-budget release,
invalid ranges, descriptor mutation/fencing and symlink rejection, using at
most a 128 KiB synthetic file per control. A fresh zero-dependency Rust 1.98.1
compile and all four controls pass; source identity remained unchanged and
no fixture residue remains. CPU receipt:
`0ede57b7df6b04d84b0e9d51694f5210d28101de9529adde94a4d889ba48915c`.
Darwin FFI constants were checked against the installed SDK. No new package,
engine source change, Metal object or model process is involved.

This bounded prototype gate is complete. It is not being expanded into a
whole-bank backend; primary diagnosis returns to the actual first-use
library/native-kernel pressure path.

Any later integration must be independently reviewed as a separate opt-in
candidate, cap retained statics and nonoverlapping expert groups, include all
CPU/GPU overlap and driver/host margins, and retain the production request
guard. It must earn small actual lifecycle checks, exact source/reference
numerics, cache/MTP/image transitions and matched loading-mode timing before
adoption. Full GLM execution remains held.
