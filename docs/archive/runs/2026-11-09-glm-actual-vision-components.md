# 20261109-GLMVIS-04: actual pinned GLM vision-sidecar execution

Registered 2026-11-09 before fetching the complete vision sidecar or executing
Metal components. Predecessor: [GLMVIS-02/03](2026-11-09-glm-native-vision-components.md).
Starting source `1d399016`, branch `codex/glm-image-parity`.

## Intended scope and fixed gates

- Fetch only the pinned 1,164,010,176-byte `mmproj-BF16.gguf`, expected SHA256
  `82b7adb44411dbb298b1c689567804e5ed0790d80f308ed8b5b6b8ac0527d2ac`, from
  Unsloth revision `a38483c8cd5df544f53d70fb281afe97369d5ab6`.
- Maintain an 8 GiB disk reserve after the fetch and anticipated isolated
  Metal build. Stop new allocations if that reserve becomes unsafe; preserve
  all model weights and source-bound executable/evidence receipts.
- Compile the current native components with `cpu,metal,grammar`, exercising
  actual mapped BF16/F32 operators. No optional EXL3 backend in this scope.
- Admit the complete pinned sidecar, including actual mapped-descriptor identity
  and all 348 tensor contracts, then execute its complete tower.
- Compare a deterministic nonuniform F32 full-tower component input against an
  independent private PyTorch F32 replay using the same actual GGUF weights.
  Retain the predeclared maximum-absolute and relative-RMS error bounds of 1e-3;
  report failures without widening those bounds or inheriting BF16 qualification.
- Separately execute a native prepared synthetic PNG through the actual tower,
  checking decoder-width output, finite values and cancellation. This is not a
  full GLM language-decoder/image-generation result.
- Recheck complete synthetic GLM fusion and continuation with the Metal graph.
  Default image loading/host dispatch, actual image generation, image-loaded
  text caching/performance and text MTP remain separate required gates.

Results and exact source/executable/reference identities will be appended after
execution. No inference-speed claim is intended from these component checks.

## Payload/reference preparation and first resource gate

The complete sidecar was fetched and verified at exactly the pinned byte count
and SHA256 above. No language-model source or additional checkpoint was fetched.
The independent PyTorch 2.14.1 CPU replay uses all actual sidecar weights with
F32 activations, and a deterministic 16-patch / 4-by-4 grid component input,
producing 16,384 finite projected values. This deliberately bypasses the native
image-preprocessing minimum and tests the full tower operators; it is not a
normal admitted image request or image-quality check.

The private fixture SHA256 is
`34296aef1f3b4d15bf69a5037c8b514d002030287781dc57dc8cf1fc2d94886d`;
its replay script SHA256 is
`06b4318311bfee1bf442e5c3a6cfa5b4139339ecea30bbd2a2c67445035cc231`.
The candidate native Metal replay remains unexecuted at this point.

The first guarded compact Metal build was **resource-deferred before compiler
startup**: 10,796,040,192 bytes free minus the anticipated 2.4 GiB build would
leave less than the required 8 GiB reserve. No model deletion or reserve
reduction was attempted. Normal debug Metal builds were estimated more
conservatively at approximately 4.5 GiB from a separate worktree's observed
build size; that target cannot fit this headroom. The intended compact build
uses an isolated ignored target, `--release`, debug disabled and incremental
compilation disabled, with an owned-process disk guard throughout compilation.

The next build narrows to `--lib`: every requested component check is an engine
unit test, so compiling a separate regular library and integration executables
is unnecessary for this scope. The revised anticipated footprint is 2.2 GiB,
using the observed approximately 341 MiB regular-engine artifact as the avoided
output. The 8 GiB reserve is unchanged and an automatic owned-process guard
stops compilation with 64 MiB safety margin. This build cannot establish a
worker/integration result.

## Actual Metal results

The compact unit build succeeded with the pinned nightly, `--release --lib`,
`cpu,metal,grammar`, release debug disabled, incremental compilation disabled and
two Cargo build jobs. The isolated target occupied approximately 945 MiB by
`du`; retiring it later reported 933.4 MiB of reproducible build files. The
automatic guard's minimum observed free space was 10,160,402,432 bytes, above
the 8 GiB reserve throughout compilation. No model artifact was deleted.

| Gate | Observed result |
| --- | --- |
| Full pinned sidecar content and mapped descriptor | Passed exact byte count/SHA256 and all 348 tensor shape/storage contracts |
| Actual complete F32 tower versus independent private PyTorch replay | Passed: 16 nonuniform patch rows, 16,384 output values; maximum absolute error `2.8312206268310547e-6`, relative RMS error `1.90591745194979e-5`; both below the fixed `1e-3` limits |
| Encoded synthetic PNG through native preparation and actual tower | Passed: 32-by-32 PNG becomes the native 112-by-112 canvas, producing 16 decoder-width rows / 65,536 finite values; immediate cancellation rejected |
| Normal GLM vision component controls | Five passed; three model/reference checks remain ignored in the default filter and were selected separately where named here |
| Native floating convolution and mapped-descriptor substitution controls | Passed individually with the Metal executor |
| Complete synthetic GLM embedding fusion/continuation and cancel/reset | Passed exact matched text-row controls with the native Metal decoder |
| Existing pinned-upstream synthetic causal reference | Passed 94 positions: maximum absolute error `4.5299530029296875e-6`, relative RMS error `1.1092299704819467e-6` |
| Retained executable after its build target was removed | Passed a fresh standalone Metal convolution control; embedded kernels remain usable |

These are component and synthetic-decoder results. The PNG check reaches the
actual vision tower, not an actual 320B language-model image request. The
nonuniform full-tower reference bypasses preprocessing admission as stated
above; it does not qualify every image/resize or the source BF16 trajectory.
No text inference speed was measured.

The private source/build/artifact/result receipt is
`runtime/glm-image-source-2026-11-09/metal-validation-receipt.json`, SHA256
`59a6a35c6bae39bb8dae0c51b857ae021bd5606eca522a807a107904247f42c0`.
The retained qualified Metal test executable has SHA256
`274a4aeaac43667ee469069152755cf0c1c7c4ce9c38c53715d4124b59d0a9f7`.
All GPU/test processes were closed before handing the resource slot to the
next Qwen workstream. Only this attempt's disposable Metal target was retired;
sidecar, reference data, receipts, logs and qualified binaries were preserved.

## Required continuation

Implement and qualify complete native GLM image generation and normal default
artifact/host/worker integration. Then qualify image-loaded text caching,
matched prior-text/candidate-text/default-image-loaded performance, and text
MTP where supported, including image→text restoration. Ordinary GGUF source
MTP execution remains missing; the optional EXL3 image ABI and preservation of
its qualified native-stage/default-three-draft policy remain separate work.
The engine still advertises no GLM image capability at this checkpoint.
