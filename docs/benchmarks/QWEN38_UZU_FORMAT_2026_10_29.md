# Qwen3.8 Uzu conversion and generation comparison (2026-10-29)

Stark now converts the pinned Lily Qwen3.8 text checkpoint into a self-contained
Uzu artifact and executes it through the specialized Qwen Metal graph. Weight
quantization and the existing precision policy are preserved. Full-model
readout and sampled-generation checks are recorded separately from the
ordinary-matrix kernel component comparison below.

All storage, full-vocabulary readout and sampled-output controls pass. The
timing series establishes no Uzu speed gain or consistent 77–80 tok/s result:
throughput declines substantially in both formats during sustained execution.

## Full artifact and runtime validation

The full conversion completed with a 64 MiB writer workspace. All 3,431 text
planes were checked against the consumed source: 1,046 code planes retained
their exact bytes, 1,834 group-major parameter planes inverse-transposed
bit-for-bit, and 551 direct planes retained their bytes. Every padded value
was zero. Complete selected source files were hashed and checked again before
atomic publication.

The single weight file is 104,165,407,864 bytes, including its header. Text
payload is 104,164,714,240 bytes, with 23,040 bytes of output-row padding.
The text adapter omits 333 vision planes (897,862,112 bytes) and 81 MTP planes
(1,842,067,616 bytes). Neither omitted component participates in this
non-speculative text graph. Configuration and tokenizer are separate files;
no original checkpoint path or shard index is required.

The initial 8 MiB conversion attempt was interrupted. Large expert parameter
planes triggered the generic writer's row-at-a-time fallback, making billions
of tiny reads possible. Repeating with 64 MiB admitted contiguous source-row
reads and completed the conversion. This workspace change does not change
quantization or output layout.

Full-model controls use the same freshly built digest worker for both
formats. With 32 and 4,096 corpus-derived input tokens, every F32 vocabulary
logit bit matches across all 248,320 outputs. The converted model also passes
both readouts under a macOS child sandbox that denies all reads from the
original checkpoint directory. A denied-read control confirms that restriction
is active. Python local-import admission, chat metadata and mixed-quantization
inventory display also accept the actual converted folder.

The conversion receipt retains its original `unverified` runtime fields.
These dated local checks do not silently promote the converter's automatic
qualification registry or establish decision calibration.

## Sampled-generation protocol

One release generator executes both formats at 4,096 input tokens, 256 sampled
tokens and a 4,352-token context capacity. Sampling uses temperature 1,
top-p 0.95 and seed 42, with top-k 20 or top-k disabled. The existing policy is
BF16 activations, F32 reductions/logits/GDN recurrent state, Q8 attention cache,
resident experts and paged n-gram storage. Speculation is disabled.

Formats are measured in native/Uzu/Uzu/native order, with a fresh worker per
block. Case order alternates between blocks. Each case has five replays per
block; the first is a warmup, leaving eight measured rates per format/case.
The reported decode rate covers 255 continuation steps after the initial
sample. Sampled output hashes must match the established native-source hashes
on every replay. No prompts, outputs, token arrays or probabilities are saved.

All 40 generations complete with the established output hashes. The eight
measured rates per format/case give these medians and ranges:

| Format | Top-k 20 tok/s | Range | Nucleus tok/s | Range |
| --- | ---: | ---: | ---: | ---: |
| Native source | 63.53 | 44.78–77.28 | 57.43 | 46.37–67.90 |
| Uzu | 53.61 | 47.05–62.55 | 54.61 | 46.95–72.32 |

The block medians show substantial time dependence:

| Block | Format | Top-k 20 tok/s | Nucleus tok/s |
| ---: | --- | ---: | ---: |
| 0 | Native source | 75.64 | 65.52 |
| 1 | Uzu | 58.48 | 62.79 |
| 2 | Uzu | 48.56 | 49.13 |
| 3 | Native source | 50.63 | 48.33 |

The host is an Apple M5 Max, 40 GPU cores, 128 GiB, running macOS 26.7.1
(25G241). Each block reports the same maximum observed Metal allocation,
74,989,060,096 bytes. Startup-to-ready measurements
include Python launch, executable hashing, metadata and model initialization;
they are not isolated native load timings or controlled cold-cache tests.
Metal allocation telemetry is sampled at 1 Hz and excludes CPU-mapped pages
and short-lived peaks. Other applications were closed by the user, AC Energy
Mode remained Automatic, and no instrumented Metal stage profiling was enabled.
This run does not establish throughput during normal desktop multitasking.

### Observed follow-up

A separate native/Uzu/Uzu/native diagnostic uses three replays per case/block,
leaving four warm rates per format/case. It adds the existing host monitor and
rootless macmon 0.10.0 clock capture. All 24 generations match the expected
hashes. Its pooled medians are 66.83 / 64.09 tok/s for native and
60.57 / 61.81 for Uzu, for top-k / nucleus respectively. Instrumentation and
the different time window prevent pooling this series with the first.

Consecutive clock intervals are included only when both observed monotonic
endpoints lie inside a generation call. These intervals include prefill and
decode. The GPU frequency is the device's active-residency-weighted frequency;
it is not a decode-only or instantaneous clock.

| Block | Format | Top-k tok/s / whole-call GPU MHz | Nucleus tok/s / whole-call GPU MHz |
| ---: | --- | ---: | ---: |
| 0 | Native source | 76.22 / 1,398 | 69.90 / 1,328 |
| 1 | Uzu | 63.75 / 1,280 | 65.94 / 1,257 |
| 2 | Uzu | 60.57 / 927 | 59.64 / 949 |
| 3 | Native source | 57.88 / 856 | 57.99 / 930 |

Lower throughput accompanies falling frequency in both formats. System thermal
state changes from nominal to fair, while GPU temperature is not monotonic;
these observations do not prove thermal throttling. Uzu warm requests record
246 and 173 pageouts across its two blocks, with no swapouts. The final native
block remains slow with zero pageouts. Existing swap usage stays fixed.
Other-client raw GPU counters rise later in the series, but their units and
client attribution cannot establish a precise contention cost.

The Uzu arms have lower pooled medians in both series. Strong sequential
frequency changes prevent an isolated estimate of a storage-format effect.
Conversion alone has not delivered the throughput target. Both routes retain
the same architecture kernels; the separate ordinary-matrix component gains
below have not been integrated into the complete Qwen graph.

## Validation

- Converter suite: 5,016 tests pass, 88 skipped; 81 focused Qwen, catalog and
  benchmark tests also pass. An earlier invocation used umask 077, which
  made a synthetic public-directory fixture private and invalidated that
  fixture's expectation. Repeating with standard umask 022 passes the full
  suite without a code change; private evidence files retain mode 0600.
- Storage admission: six tests pass, including exact Q4/Q8/expert/embedding
  restoration, malformed headers, duplicate fields, quantization-policy
  aliases, inventory, spans and padding.
- Native Metal engine: 669 tests pass, 45 ignored, plus seven C++ tests and
  three doctest checks. This run precedes mechanical Clippy condition rewrites;
  final builds, Clippy and full-artifact controls run after those rewrites.
- Final worker suites: 55 generator tests pass, two ignored; scorer/probes
  complete 41 tests. Both workers pass release Clippy across all targets with
  warnings denied, and workspace Rust formatting passes.
- Repository policy and all 41 policy self-tests pass. Converter Ruff checks
  pass; benchmark scripts use the converter's existing 120-column style.
- Full-artifact checks cover readouts at 32 and 4,096 tokens, 4K sampled
  generation, and source-directory-denied execution. Other context lengths
  and decision calibration are outside this validation.

## Why a typed architecture extension is still required

The standard Safetensors/MLXSpec representation is a storage contract. It
does not describe Qwen3.8's hybrid graph, including GDN, hyperconnections,
QSA attention, expert routing, or n-gram PLE. The generic dense decoder does
not implement that graph.

The artifact uses standard matrix storage inside a typed
`Qwen38LanguageModelConfig` extension. Typed tensor bindings present the
logical tensors expected by the already implemented Qwen3.8 hybrid compute
path. The format adapter changes physical storage layout; it does not
substitute the generic dense decoder for the architecture-specific runtime.
The typed config and bindings are implemented in the
[Qwen3.8 Uzu adapter](../../crates/stark-engine/src/engine/hybrid/uzu.rs),
which dispatches through the existing
[Lily hybrid compute path](../../crates/stark-engine/src/engine/hybrid/lily.rs).
Generation and scoring select this extension through `hybrid::Model::load`.
Direct `Engine::load_language_model` calls still use the dense decoder.

For linear and expert matrices, the converter plan uses standard MLXSpec
`output_input` storage: packed code bytes retain their values while U32
storage is represented as U8, and BF16 scale/bias planes are transposed into
group-major form. Output rows are padded to a multiple of four with zeros.
Embedding layouts and direct floating/integer control tensors have separate
bindings. The conversion itself does not dequantize or requantize the matrix
planes. The converter accepts only BF16 artifact configuration in this first
version; that selection does not convert these BF16 parameter planes.

## Component protocol

The checkpoint is `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at revision
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`. The fixture exporter checked
before/after stat identities for the 55 source shards, but did not compare
their complete contents to a prior full-content receipt. Selected projection
plane hashes bind the matrices used here.

The fixtures contain one Q4 GDN matrix (`10,240 × 2,560`, group size 64) and
one Q8 attention matrix (`12,288 × 2,560`, group size 64). The comparison
loaded each matrix through both paths with the same packed codes, BF16 scales,
BF16 biases, and deterministic BF16 input. Standard output-row padding was
zero rows in both cases because both actual row counts are divisible by four.
The fixture manifest records exact source-plane hashes and the code/scale/bias
preservation checks.

Both kernels accept BF16 activations and write BF16 outputs with F32
accumulation. Their affine quantization and group size match. The underlying
Metal kernels use different decompression/dot and reduction implementations,
so equal input bytes do not imply bit-identical output. Each arm was warmed
for two seconds, then measured in 16 paired alternating AB/BA rounds. Each
round encoded 48 distinct copies of that matrix in one command buffer/pass per
arm, with separate output buffers. Times below are medians divided by 48;
wall time includes host encoding and command completion. Each path therefore
issued 768 matrix dispatches across its measured rounds. Outputs were checked
outside the timed rounds. The bounded F64 reference covered the first 64 rows.
The sealed component test completed successfully: one test passed, with 713
other engine tests filtered out in that invocation.

## Projection component results

| Projection | Stark standard GPU µs/matrix | Lily GPU µs/matrix | Stark standard wall µs/matrix | Lily wall µs/matrix | BF16 output agreement | Relative L2 | Max absolute difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4 GDN, 10,240 rows | 24.859 | 28.028 | 38.756 | 36.202 | 10,238 / 10,240 (99.9805%) | 6.9441e-5 | 0.00390625 |
| Q8 attention, 12,288 rows | 56.958 | 61.430 | 70.617 | 70.129 | 12,285 / 12,288 (99.9756%) | 5.0481e-8 | 1.90735e-6 |

For Q4, the standard path's median GPU time was 11.3% lower, while its median
wall time was 7.1% higher. For Q8, its GPU time was 7.3% lower and wall time
was 0.7% higher. These are component observations from one run; they do not
predict model-level speed.

The fixture manifest's exact plane checks establish storage preservation for
these selected projections, not the whole checkpoint. Different output
hashes are expected: the two kernels reduce in different orders. On the
bounded 64-row oracle, maximum absolute differences were 0.0035814 for each
Q4 path and 0.00380365 for each Q8 path. Those oracle differences are not
whole-matrix error bounds.

The benchmark aggregate does not include host/device identity. Its timings
should be treated as measurements of the recorded run, not portable
performance claims.

## Evidence bindings

| Evidence | SHA-256 |
| --- | --- |
| Component aggregate report | `771b0795b390dbbe744e12f58497630bbcfc3e29e19788ff86d810118042830e` |
| Component v4 harness source | `d620632e9a52ef02fbc51fba6998ed264e2153f7c0b2fc81ce1661bf76b03b70` |
| Projection fixture manifest | `4037236592110beeead93e258ff72e1da4a7b289153920fb1771f4647b0dac37` |
| Converter design note | `9edf2dcd2416eea5393e02fdb9587fbdab6c3c9f39b220dbb21f8b9964994e23` |
| Runtime contract | `107f5d928b324917578a66dcb6f51e2f45e5871b2a759d78d06ad88b42d61f24` |
| Complete conversion source manifest | `4bce8ad08a02057e549cdbab0cc401c28849b9b4e4fd306c7fff9649a7a2823d` |
| Converted model weights | `c493d4a3f876f6495f42127744b6415fe4cd3287f017fbc4b12ecd6d1245e657` |
| Converted configuration | `cd9fef82e5016e282dc1920600776a1fee5c3560b86bc7787b883b83c732410e` |
| Converted tokenizer | `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4` |
| Final native source manifest | `1d4e849d35e4eb1a17434dc96b9c7e8a38137face0d051ecd693360bd1024c8d` |
| Generator | `50ea16ecb8ebe5ec1d51e5a992c37209299a8c2568ac267e1f8ddc8c66f70d83` |
| Digest worker | `d2cc6fe5512094e0e4ff3a5761df30716fc8856d061961eda54be87be6b6323f` |
| Full-model readout aggregate | `465e9caf23dc1d7a5d8c3bcffcc0a1fa6cc9d3c51918acd90e99ab7954f66c7f` |
| Source-denied execution proof | `78f3b8404b788dbeae2a7d9dc3aff18c432bae06927bb90aba0a622c167446ca` |
| Uninstrumented generation aggregate | `84c2caf03f9a222e5f79a6870c057325694aa8ef36035dcade28d269cc9f62a5` |
| Observed generation aggregate | `ffe206fb338698653dca406a6869e25aab3d91fa6f9bb076f607a9ef0dc76292` |
| Clock/paging analysis | `b5a67561cfeb400433454d632cf58cdf96f6db5165aa5d3a4416a72551d5fda6` |

The component harness was based on Stark commit `edf92a029bfa56cef871720c2dcd497d27251809` and is sealed separately from the full-model source above. Source manifests, executable archives, model files and scalar/hash validation receipts remain private. Rust 1.98.1 builds each release worker separately; engine unit tests and formatting use nightly-2026-10-01, with locked Python 3.13.13 converter tooling.
