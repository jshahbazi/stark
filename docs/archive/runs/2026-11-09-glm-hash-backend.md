# 20261109-GLMVIS-08: pinned macOS/aarch64 artifact hashing

## Identity and decision

This is a separately scoped continuation of the hash investigation registered
in [GLMVIS-07](2026-11-09-glm-image-admission.md), recorded after crash recovery.
Baseline worker source: `fecc27d551773d8197f0f9b2c55c6dba21fb9444`.
Kind: artifact-admission compatibility/correctness. Result: bounded digest
controls passed; full decoder replay and startup performance remain unmeasured.
Disposition: hash feature adopted on this branch; model/image admission and
default promotion remain awaiting qualification. See the
[attempt ledger](../../model-experiments/README.md).

## Change and fixed scope

The cached pinned RustCrypto `sha2` 0.10.9 source selects its aarch64 SHA-256
backend only with the `asm` feature; `asm-aarch64` enables that feature. Its
backend checks CPU SHA2 support and retains the software fallback. Enable the
feature only for the engine's macOS/aarch64 target dependency. The existing
`sha2` version stays exactly 0.10.9; optional `sha2-asm` 0.6.4 is added to the
locked graph. Other targets and build dependencies keep their prior features.

Full content checking remains mandatory. This change affects hashing only;
model arithmetic, precision, inference options, request budgets and state
restoration have no source changes. No new image or text capability is granted.
No installed build was changed.

## Controls and recovery evidence

The saved pre-crash hash-only control passed four vectors with the accelerated
and force-soft builds: empty input, `abc`, one million repeated bytes and a
deterministic 65,537-byte fixture streamed in 513-byte chunks. Their digests
match the independent Python SHA-256 reference. This receipt predates recovery
and is retained without relabeling it as a new run.

A new recovery control used Rust 1.98.1, offline locked dependencies and a fresh
distinct private target. A bounded 64 KiB streaming hash tool checked five
complete files against their existing source pins: finished config, tokenizer,
template, processor and the 1,164,010,176-byte vision sidecar. Both accelerated
and force-soft binaries produced identical expected digests; an independent
Python `hashlib` replay also matched. File descriptor signatures were unchanged
across the runs. The sidecar digest remains
`82b7adb44411dbb298b1c689567804e5ed0790d80f308ed8b5b6b8ac0527d2ac`.

The private recovery receipt is
`706c3cd31fb93e637d3f4ac940cff7ae2fbb3493fbe5e5105f91fc3eea4ad799`.
The 7.32-second aggregate includes two tiny builds and three file-hash methods;
it is a correctness-control duration, not a startup-speed comparison. Locked
offline Cargo metadata and the selected feature tree parse successfully;
repository policy and whitespace checks pass.

## Limits and next gates

The four 109 GB decoder payloads were not rehashed during this recovery control.
Their prepared/original hardlink signatures and pinned sizes still match;
earlier full-content admission retains its own source/build-bound receipt.
Full decoder digest equivalence must precede a new startup/admission claim.

The pre-crash native07 engine build used the earlier target directory and the
subsequently rejected memory-policy draft. Its successful build receipt is
preserved, but it is not current-source validation. No engine or model build
was performed in the recovery hash control. A fresh source-bound worker build
and native admission remain deferred behind the resource and memory review.
Actual image generation, cache correctness, matched text performance and
source-head MTP execution remain open under GLMVIS-06/07.
