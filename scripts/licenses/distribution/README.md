# Supplemental license references

`pytorch-LICENSE` is the exact PyTorch license file from tag `v2.14.1`, resolved
to commit `5c4886908584029761b579af026dcfb627c84070`. It is included because the
vision-only BF16 LayerNorm variance implementation adapts the logical Welford
moments cascade and scalar final contraction from
`aten/src/ATen/native/cpu/moments_utils.h`. The adaptation does not add a PyTorch
runtime dependency. The `source_adaptations` entry in `manifest.json` records
the source hashes, license hashes and pinned revision. The BF16 dot-product
schedule is documented in PyTorch's `BlasKernel.cpp`,
`ReducedPrecisionFloatGemvFastPathKernel.cpp` and `functional_base.h` at the
same pinned revision.

`ggml-LICENSE` preserves the complete MIT notice embedded in PyTorch's
`ReducedPrecisionFloatGemvFastPathKernel.cpp`. That source explicitly attributes
its reduction helpers to llama.cpp's ggml dot-product functions. The notice is
included with the PyTorch BSD license because Stark adapts the documented
reduction schedule; the manifest records the PyTorch source hash containing
that notice. No direct llama.cpp source revision is claimed.

`sleef-LICENSE.txt` preserves the Boost Software License 1.0 text from the SLEEF
revision pinned as PyTorch's `third_party/sleef` gitlink. `sleef-ATTRIBUTION.txt`
records the copyright statements in the pinned SLEEF SIMD implementation and
ARM AdvSIMD helper. The manifest records source hashes for the single-precision
SIMD math routines, double-float helpers, and trigonometric reduction table.

The pinned PyTorch `vec128_float_neon.h` identifies its `exp_u20` vector
exponential as copied from Arm Optimized Routines. The PyTorch `Gelu.h`,
`Activation.cpp`, and NEON header hashes are recorded for the vision GELU
adaptation. `arm-optimized-routines-LICENSE.txt` preserves the full upstream
MIT OR Apache-2.0 WITH LLVM-exception license, and
`arm-optimized-routines-ATTRIBUTION.txt` records the copyright and exact
upstream `expf.c` revision reviewed for its polynomial and range-reduction
schedule. PyTorch's pinned implementation adjusts special-case behavior for
its finite GELU range.

`objc2-LICENSE.md` is the exact historical `LICENSE.md` from each locked
objc2 source revision. The locked revisions did not contain full license text;
that file describes the package license selection and links to the canonical
texts.

`objc2-LICENSE-MIT.txt` and `objc2-LICENSE-ZLIB.txt` are exact files from the
later upstream revision `ee9a7ada2131f5944b8750428e265c15632f2a19`, which added
license text to the repository. They are supplemental references only and are
not asserted to have existed in the locked crate source archives.

The Apache-2.0 text is already preserved at
`vendor/cxx/LICENSE-APACHE`; the manifest records the later upstream source
URL and revision for the complete Apache reference.
