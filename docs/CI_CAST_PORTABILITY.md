# Converter cast portability repair

Date: 2026-10-11.

## Failure and cause

PR [#1](https://github.com/jshahbazi/stark/pull/1) at source revision `a51ef143f2d262427c4d4006211d6c3ebcc5f1ea` failed the Linux `Repository checks` job in [CI run 38195504810](https://github.com/jshahbazi/stark/actions/runs/38195504810/job/114637122324). Its model-free converter suite reported 14 failed, 833 passed, and 37 skipped cases. All failures were exact frozen-reference payload hash mismatches involving F16 conversion: F16 to BF16, BF16 to F16, F16 to F32, and F16 normalization widening. The same pre-repair suite passed 847 cases with 37 skipped on Darwin ARM64.

Native NumPy and ml_dtypes half-conversion paths differ across architectures in NaN quieting and payload handling. The frozen independent JAX CPU oracle was qualified on Darwin ARM64. Finite-value casting alone does not define the exact exceptional-value bytes required by that oracle.

## Repair

The converter reconstructs F16-involving NaN outputs from their source integer bits after the native cast. F16 to F32 and BF16/F32 to F16 preserve sign and shifted or truncated payload while setting the target quiet bit. F16 to BF16 produces a signed canonical quiet NaN. Finite casts retain native rounding. Same-dtype copies and BF16 normalization bit widening retain their existing bytes.

Dense casts, fused casts, normalization widening, and scalar-transform cast stages share this policy. Workspace accounting includes the one-byte-per-element NaN mask; integer reconstruction uses scalar temporaries instead of allocating payload or index arrays. The normalization helper rejects a budget smaller than one complete element.

The independent frozen fixtures, their source attribution, and historical qualification/benchmark results remain unchanged. This repair changes converter provenance and requires new source-bound validation; it does not establish new model, inference, or performance qualification.

## Validation

Both platforms used the unchanged converter lockfile, NumPy 2.5.3, and ml_dtypes 0.6.0. Tests use synthetic inputs and publish only aggregate outcomes and hashes. No model conversion, model inference, or private reference-artifact validation is part of this repair.

| Run | Environment | Result |
| --- | --- | --- |
| Original native-cast replay | Linux x86_64 under Docker emulation, CPython 3.13.11; replace only the new cast helper in memory with the prior direct native cast | The 42 frozen cast/normalization comparisons reproduced the same 14 failures; 28 passed |
| Repaired full converter suite | Darwin ARM64, managed CPython 3.13.13 | 860 passed, 37 unsupplied reference-artifact cases skipped |
| Repaired full converter suite | Linux x86_64 under Docker emulation, CPython 3.13.11 | 860 passed, 37 unsupplied reference-artifact cases skipped |
| Lint, format, and repository policy | Locked local tools | Passed |

The 13 new regressions cover all 65,536 F16 or BF16 source patterns for each of the three failing cast pairs, the normalization route, minimum workspace acceptance/rejection, and simulated native loss of NaN sign/payload/quiet bits across all four F16-involving cast pairs. Expected exhaustive outputs use independent IEEE integer arithmetic; the original frozen-reference hashes remain mandatory.

The Linux image was `ghcr.io/astral-sh/uv@sha256:531f855bda2c73cd6ef67d56b733b357cea384185b3022bd09f05e002cd144ca`, run with `--platform linux/amd64`. The repository was mounted read-only, dependencies were synchronized with `--locked --no-install-project`, and the converter was imported from that mount. Its available interpreter was 3.13.11, so this is supplemental Linux validation. Hosted CI retains managed Python 3.13.13 and is checked separately after publication.

Source-bound SHA-256 identities for these repair checks:

| Input | SHA-256 |
| --- | --- |
| `tools/converter/stark_convert/streaming.py` | `3884180013b44d6e402cca8f78d8174137571ec26130e88312e001ea24ed3223` |
| `tools/converter/tests/test_casts.py` | `cfdc3c8b684d590074b79512e6e17557a83692e7f82fcd8ab190ba3bc392c2ab` |
| Unchanged `tools/converter/tests/fixtures/dense_casts.json` | `091c4263a5474ee60cb660eca196654d9d717054b77ab8cc8a1fd343477758eb` |
| Unchanged `tools/converter/uv.lock` | `c7a77fc0e6451524d8276c3aba584929ef0e0907a2f255ad1f160af69ac7e0d5` |
