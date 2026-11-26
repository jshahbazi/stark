# 20261126-HY4DESKTOP-01 · Hy4-flash catalog install and packaged generation

| Field | Value |
| --- | --- |
| Attempt ID / registered date | 20261126-HY4DESKTOP-01 / 2026-11-26 |
| Scope | Compatibility and install-to-generation lifecycle on Apple Silicon |
| Predecessor | [Native Q3_K record](2026-11-25-hy4-native-q3-k.md) and [flexible context verification](2026-11-25-hy4-context-ladder.md) |
| Implementation | [PR #205](https://github.com/jshahbazi/stark/pull/205) |
| Artifact | AngelSlim/Hy4-flash-GGUF `c50d4dd30be42f2b91138e5731de0e2f32ad56b2`, two Q3_K shards and the BF16 projector; Tencent metadata `eccd3123eec73f2040f4c9ed0de1883593470510` |
| Runtime | Locked Rust 1.98.1 Metal worker with `hy4` and the packaged Python desktop backend; M5 Max with 128 GB RAM, macOS 26.7.2 |

The source-only path required a manually prepared directory and a feature-enabled
worker. The catalog path should install exactly the same artifact, retaining the
projector, NextN and load-time context sizing. It uses native download staging so
an interrupted install can resume before the model is published to the library.
The managed download is 156,415,403,072 bytes. A derived NextN expert sidecar is
built on the first load.

## Acceptance checks

- Open the exact pinned catalog row in a fresh library and install it from the
  bundled desktop backend. Reuse the two already downloaded shards through the
  Hub's completed-file resume path, fetch the projector and Tencent sidecars
  from their pinned revisions, and check all three GGUF digests before publish.
- Start the installed artifact with the ordinary desktop load settings and the
  bundled worker. Check the live context capacity, NextN and image readiness.
- Complete text requests at 2,048 and 4,096 context settings in the same loaded
  worker, then submit a synthetic image through private attachment staging.
- Keep benchmark timings and numerical evidence in the existing native records.
  This smoke test verifies the app's lifecycle and configuration.

## Observed result

The first package attempt reused local template and Decider binaries with an
outdated deployment target. Recompiling those binaries under the distribution
builder's macOS 26.4 settings passed the package checks.

The Developer ID–signed development archive built from
`c90f88d710ffac6c11ce78d314c66506bee935e5` passed local signature and
relocation checks. Archive SHA-256 was
`b68eb58c36d6be64c052418af92db0b60bf2052ad9fd390c44ac0a297a049915`.
The signed bundled generator's SHA-256 was
`ac58d01ec27ba8afcc19999413e4de3c63cdd03549d98bcc7cd76d1791e68881`.

The bundled desktop backend started with an empty registry and a resumable
stage containing APFS clones of the two original verified shards. It downloaded
the 1,044,397,440-byte BF16 projector, the weights license and Tencent sidecars.
The installer verified all three GGUF digests and passed the local image check
before publishing the model. This run took 100.69 seconds from the Catalog
install command through publication. A cold transfer of both weight shards was
outside this run's scope.

The installed worker reported a 452,608-token load capacity, image readiness
and native NextN support. It returned successful text replies at request windows
of 2,048 and 4,096 without changing loaded capacity. It answered a synthetic
image request and removed the staged attachment. These short requests qualify
the packaged lifecycle and two window settings.

A copy of the same app with an isolated bundle identity and model library
showed the pinned Catalog row as Downloaded. Starting the model via Manage
Models reached Ready and a local Playground chat returned the expected answer.
The generic Catalog memory hint still compared the 156.42 GB of files with
physical RAM, despite the model running on the tested 128 GB Mac. The UI fix
leaves admission to the streamed worker's live memory and context checks.

This was a local development archive. Stable distribution and notarization
use the signed release workflow after merge.
