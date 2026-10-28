# Native GLM5-Next and Qwen4-Exp implementation validation

Date: 2026-10-28. Branch: `codex/native-glm53-qwen38`.
Base source: `b9f9a7b5`; implementation source is the commit introducing this
record. Host: Apple M5 Max, 128 GiB, macOS 26.7.1.

This record covers native implementation, source-format admission, small-model
reference comparisons and worker integration. Full production checkpoints were
still downloading when this implementation checkpoint was recorded. Header-only
admission does not establish production forward execution, model quality,
throughput or memory qualification.

## Implemented paths

- GLM-5.3-Flash: all text trunk layers, KDA, compressed sparse MLA, pooled indexer,
  mHC, routed/shared MoE, GGUF dynamic quantization and configuration agreement.
- Qwen3.8-Flash-Next: Gated DeltaNet, QSA, gated residuals, routed/shared MoE,
  exact integer n-gram PLE, original MLX mixed affine storage and grouped heads.
- Native Metal packed matrix products, batched heads and selected experts.
  Scalar control and recurrent updates use Rust F32 arithmetic.
- Existing generation and raw scoring protocols, fresh-request state, EOS union,
  cancellation, stopping, recovery and strictly consecutive scoring cache.
- Pinned original-format preparation, local desktop import, family display names
  and Jinja loop controls needed by the actual GLM template.

Production inference does not import or launch the reference runtimes. These
paths are text-only, without vision, MTP or speculative assistants. GLM's official
template always opens reasoning; the app's direct-answer Decision boundary
continues to reject that template. Raw scoring is implemented independently of
that application-template eligibility.

## Source and precision boundaries

The GGUF layout and quantization oracle use llama.cpp/ggml at
`8ca4060b7e30e76baaa5670ee1ec1b760b7f8552`. GLM's independent full-prefill
reference uses Transformers modeling source at
`95bafdcccfffe17ec383999fc39273867b11385e`, source SHA-256
`7870e7ce6ca477d26de3ec79b31d41553b75a09a726c661f04c2e26e790bdf36`.

Qwen's reference uses the PipeNetwork model module shipped with checkpoint
`b2c422f3c643e36f04227a64d61796b44a4b1029`, source SHA-256
`6fae4ec0decbf77ca4a4571de683bc5580ec75e84325ecb432dfcd2fc81df75e`.
The reference module, configuration and tensor contracts were inspected before
execution. Synthetic captures remain under ignored runtime directories.

Both complete-model comparisons use F32 execution. The Qwen mixed fixture stores
floating parameters and affine planes in BF16, then both implementations execute
their decoded values in F32. This is not a BF16-intermediate execution
qualification against the publisher's default runtime.

## Results

| Check | Result |
| --- | --- |
| Full physical CPU/Metal engine suite | 601 passed, 33 ignored; 7 CXX tests and 3 documentation tests passed |
| Final focused hybrid suite | 31 passed, 3 source-bound tests ignored by default |
| Converter, chat and desktop Python suite | 4,962 passed, 88 skipped |
| Generation worker release unit suite | 43 passed |
| Scoring worker release unit suite | 19 passed; digest binary repeats the same 19 tests; experiment protocol test passed |
| Menu bar controller | 100 checks, zero failures or skips |
| Menu bar build tests | 23 passed |
| Distribution construction/license tests | 19 passed |
| Release worker Clippy, pinned Rust format, Python lint/format, repository policy and whitespace | Passed |

All nine packed GGUF formats in UD-Q2_K_XL match hashes from the independently
compiled pinned C decoder. Physical Metal checks cover those formats, floating
storage, mixed 4/8-bit affine planes, nonconsecutive selected experts, batched
heads, invalid dimensions and release of completed matrix uploads. The 64-head
component benchmark reduced 64 submissions to one with identical results; this
does not measure whole-model throughput.

GLM quantized complete-forward comparison covers 94 positions across three
independent sequences and repeated resets. It includes IQ2_XXS/IQ2_XS expert
matrices, Q8 matrices and F32 controls. CPU maximum absolute logit error was
`2.26498e-5`, relative L2 `1.80378e-6`. Metal maximum absolute error was
`3.09944e-5`, relative L2 `2.35569e-6`. Invalid-token failure clears state.

The GLM reference generator verifies that strict-parity comparisons have no
ambiguous sparse cutoff ties. Minimum margins were `0.124091625`, `0.124091625`
and `0.412294328`. Separate native tests cover zero-score ties with deterministic
index ordering, visible tails, and nonuniform per-channel four-token pooling.

Qwen's mixed BF16-storage fixture covers 13 positions, grouped attention heads,
a three-to-one GDN head ratio, sparse attention beyond its budget, EOS boundaries
and n-gram PLE. CPU maximum error was `3.76254e-7`; Metal maximum error was
`2.98023e-7`. Greedy agreement was 13/13, with exact reset and failure recovery.
Selected-expert batching preserved the previous output hashes.

The exact production Qwen headers cover 11 shards and 3,215 tensors. Their
106,195,031,237-byte layout admits correctly, including 128 n-gram shards of
logical width 160 and an unused rank-five vision tensor. The parser was corrected
to admit bounded ranks through eight while retaining operation-specific rank
checks. Rank nine, zero dimensions, overflow and truncated spans are rejected.
Only 296 actual control/hash tensors were fetched for the sparse admission
fixture. No forward was run on that fixture. Three geometry mismatches fail.

The exact four-shard production GLM headers also pass configuration agreement;
altered metadata is rejected. Its sparse header fixture was not executed.

## Worker identities and integration

The final source-checkpoint release binaries used for both private fixture
worker replays have these SHA-256 identities:

- `stark-generate`: `7c122ac8ee5435e9aaab8cbf462e9738d4320c05f740147a9128528633ad077d`.
- Optimized `stark-score`: `aa304649517ff3ee0e2d3eb87db0336d2d2601950b944bc257917abb95ddbefd`.

Both families pass tokenization/decode, union of all three EOS metadata sources,
four greedy source checks, A/B/A generation isolation, invalid-request recovery,
strictly consecutive scoring cache hits/misses, and clean process shutdown.
Maximum source scoring-logit errors were `1.45435e-5` for GLM and `2.53320e-7`
for Qwen. Source configurations remain unchanged by the worker fixture harness;
synthetic tokenizer metadata and small copied weights live separately.

Reference readout hashes:

- GLM first-sequence fixture: `870ae079c9fccbf88294b42ee14e01b98740df90ae738467ff74082f222ac899`.
- Qwen grouped mixed fixture: `4e1a5c21bf59d354cf88bdc1c106e3ea74a64855b57c2d1b17cd2a605152fdaf`.
- Private final engine test log: `26e4c47124559bf05c8acc358b8412a935d3f4a8a4bc7748cb02864950de43f4`.

Reproduction commands and preparation constraints are in
[native hybrid models](../../NATIVE_HYBRID_MODELS.md). Models, binaries, raw
reference readouts, token vectors and runtime artifacts remain ignored. Only
aggregate results and hashes are recorded here.
