# Menu bar install progress validation — 2026-10-22

## Scope and source

The local app built from clean revision `94b706432b7cf4e5eff6938b4d4c66dc46c7a2df` shows separate Download and Conversion progress bars. The converter sends fixed phase/state names and numeric byte counts over a private inherited pipe. Download counts selected source bytes available locally, including cache hits. Conversion counts output weight bytes, including the Safetensors header. Completion follows source validation and atomic output publication; the desktop controller additionally waits for model validation and registry publication.

Local and already converted sources skip the corresponding phases. Cancellation, failed installs, stale events, pipe cleanup and retained descendant descriptors have dedicated coverage. The complete current converter wheel is included in the app with its provenance capsule, license and distribution metadata. Dependency locks are unchanged.

This report was added after the build. Earlier menu bar and chat validation reports and their private artifacts remain unchanged. Work was isolated from concurrent changes in the primary checkout.

## Verification

| Check | Result |
| --- | --- |
| Locked converter suite | 3,855 passed; 88 skipped |
| Dedicated converter progress tests, included above | 13 passed |
| Desktop controller tests, included above | 38 passed |
| Model-free menu bar packaging tests | 14 passed |
| Converter Ruff lint and formatting | Passed |
| Swift typecheck and optimized app build | Passed |
| Local ad-hoc signature verification | Passed |
| Packaged native runtime harness | 17 checks passed |

The packaged harness used a temporary private model library. Registering a local converted model skipped both stages. Installing cached `Qwen/Qwen3-0.6B` produced six download events ending at 1,514,733,679 bytes and 21 conversion events ending at 1,192,276,800 bytes. Counts were monotonic, totals matched, and intermediate conversion counts were observed. Conversion completion followed publication of a valid model and registry. The converted model loaded and generated a synthetic reply with positive input/output token rates. The harness retained no conversation content, left the user's registry unchanged, and removed temporary models and child processes.

HTTP resume, cache reuse, selected-file totals, Xet reconstruction accounting with progress globally disabled, random output writes, failure and cancellation were covered by tests. The packaged native run reused the Hugging Face cache; it did not repeat a full uncached network download. This is a scoped functional check, not a new model qualification or throughput benchmark.

The installed app at `~/Applications/Stark.app` was inspected through native accessibility and screenshots. Command-I opened the install dialog. Both source modes displayed separate accessible Download and Conversion bars with readable labels and unclipped controls. Local mode showed Download as Not needed; Hugging Face mode showed both stages Waiting before installation. Native progress events were verified separately by the packaged harness.

## Artifact identity

| Artifact | SHA-256 |
| --- | --- |
| Embedded app manifest | `cd74a88309ee8c02e7cff03a09703e5bce1c45b4ba98eea7b51eb943122dfea5` |
| Signed app executable | `1f65e80395690cb19d9f9aa64b3af84dd8e4e3ebaa09a32d023308a2b3e15fd1` |
| Converter wheel | `c7180c8290ae0f628069a1db46f23e71de9ae76d8ef201e6d307b4db3d369c83` |
| Canonical converter toolchain provenance JSON | `607bd099ead2d2a2a072698122f33935d859db89e354b7ac4fd5d64057d85b1a` |
| Generation worker | `f12fd9849d81610170c2b893e98360b25ab553310cd5276e266f6d5e583e320f` |
| Conversion receipt | `196dc9f525a8cc17333e50010e3abca4d1f9b21f095a7ab9f41f44fc149058ea` |
| Converted weights | `87c3344a80b7376392cc5c44d5369680b68936e6930bb8ac476ca65a136f64c7` |

The generation worker is the previously recorded development build that reuses cached engine libraries; it was not rebuilt for this UI change. Its build record remains embedded. The app is locally ad-hoc signed and has no new release or notarization infrastructure.

Private aggregate results and the harness remain under ignored `artifacts/menu-bar/progress-final-validation/`. The preliminary run against `3be99e07` remains separately under `artifacts/menu-bar/progress-validation/`. Models, app bundles and native artifacts are not committed.
