# Qwen3.5 native image execution qualification

The native BF16 `Qwen/Qwen3.5-0.8B` path passes the unchanged final-logit gate
on the exact model, worker, and runtime below. All 744,960 final vocabulary
values across three fixtures match the pinned reference bit-for-bit. The
release worker also reproduces all eight normalized reference answers on the
frozen visual questions. Both implementations answer six correctly; the source
model's counting and OCR misses remain visible.

This qualifies the developer worker, terminal client, and Local API on the
recorded runtime. Swift attachments and a signed packaged installation require
separate qualification. Earlier failing records and private diagnostic runs
are retained.

## Qualified identity

| Input | Identity |
| --- | --- |
| Model | `Qwen/Qwen3.5-0.8B` |
| Model revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| Artifact | Schema 2, BF16, `mediaref-v1` |
| Worker | Rust 1.98.1 release `stark-generate`, built separately from the optimized scorer |
| Backend | Metal |
| Device class | Apple M5 Max, Apple GPU family 1010, 40 GPU cores, MXU enabled |
| macOS | 26.7.1 |

The external qualification catalog matches all five artifact files, the worker
executable, the public device-class properties, and the macOS product version.
The host snapshots the executable hash before launch and checks it again at
readiness. Other builds, artifacts, devices, and OS versions fail closed.
Conversion receipts identify a named build; serving checks the build actually
launched. Signing or rebuilding a worker can require a new execution record.

| File or build input | SHA-256 |
| --- | --- |
| `model.safetensors` | `7c5960e7225b3b36844fa397a85c5e20d281d732e382475062f552c0737c79d4` |
| `config.json` | `3f0cecb7cb3633bc271eabf69b06ac9f74b377b9ec17b441f5ceac331a71f42a` |
| `tokenizer.json` | `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4` |
| `preprocessor_config.json` | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `chat_template.jinja` | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| Generation worker | `5fc92516c9f803df90316599a3930c1e2104cf320ae23361ff807e48bde08ec6` |
| Native source manifest | `a96c7fa72958a68244bd517b3682ea286c713da2e8a443557b0a54f50bc96bee` |
| `Cargo.lock` | `2254da0dd179db6b926492f2bec5f63b756f6508132b34f5ac772ec9d877aaa2` |
| Rust toolchain file | `72da9f9e503464ba7393560f6f51a69fb06be72633d52ae7725fa0ba59dfda22` |

The source manifest uses `native_build_provenance`: sorted file hashes beneath
`crates/`, `.cargo/`, and `vendor/`, plus the root Cargo files and toolchain file.
The serialized weights are unchanged from the earlier tower qualification.
Schema-1 experimental artifacts require reconversion.

## Reference and numerical result

The target uses Transformers modeling revision
`2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397`, PyTorch 2.14.1 at
`5c4886908584029761b579af026dcfb627c84070`, eager attention, BF16 parameters,
FP32 rotary buffers, and eight CPU threads. Root and child model classes are
bound to the pinned source. AST, tensor, token, image, and position identities
are checked. An installed Transformers child-model substitution was identified
and corrected in new reference directories; its earlier diagnostics were
preserved and are not the acceptance target.

| Case | Prompt tokens | Vocabulary values | Relative RMS error | Cosine | Top-1 |
| --- | ---: | ---: | ---: | ---: | --- |
| Normal square | 83 | 248,320 | 0 | 1 | Equal |
| Normal rectangle | 96 | 248,320 | 0 | 1 | Equal |
| 1024 × 1024 boundary | 1,043 | 248,320 | 0 | 1 | Equal |

The gate remains relative RMS error at most 2%, cosine at least 0.999, and
matching top-1. The boundary image produces 4,096 encoder patches and 1,024
image tokens, crossing the decoder's 1,024-token prefill boundary. An independent
raw-bit audit also finds zero differences across all 48 normal-case decoder
layer checkpoints: 4,399,104 BF16 values, including zero signs. Matched-input
DeltaNet convolution, decay, core, FP32 state, and complete mixer output are
bit-exact. All 48 MLP gate checks match across 15,396,864 values.

| Private reference or audit | SHA-256 |
| --- | --- |
| Canonical full-model manifest | `c32ee8e39b89a1bae59526be310848a9bc7983b9908ab745d0e4ca93dfae7002` |
| Boundary full-model manifest | `d2b7bc3e3dc905d07e10023c6e3ddc20c35657adb5051d7b0510befaf2f4a785` |
| Visual QA reference report | `c62855b797ff8a0d45d153131d38966d65bc9fa5e6621a170c5e3a770a99bea7` |
| Frozen visual fixture AST, Python 3.12 | `87c7947de54266ca02f7287ae80d8632b8bb8cf9b34312c309dbe19a163551b8` |

These numeric replays run through the engine's opt-in test readbacks; production
workers expose no tensor capture. Release-worker behavior is checked separately
below. The earlier [tower parity record](2026-10-26-qwen35-vision-parity.md)
retains its independent CPU/Metal intermediate results.

## Arithmetic required for parity

The exact vision loader selects a private reference numerical profile after
five-file admission. Existing text artifacts keep the Legacy profile. The
profile preserves source BF16 materialization boundaries, 32-stream decoder
linear accumulation, source RMS reductions and affine sequencing, eager causal
attention, rotary products, sigmoid gates, and SiLU products.

DeltaNet prefill follows the pinned 64-token chunk algorithm with FP32 scratch
and recurrent state. Sequential FP32 fused multiply-add schedules match the
measured Accelerate matrix products and triangular solves on the frozen inputs;
this is measured compatibility with that runtime. Prefill cumulative sums use
the source CPU's FP64 accumulator. Metal emulates the required IEEE additions
with integer arithmetic and rounds each prefix to FP32. Source exponential and
softplus arithmetic is retained. Single-token continuation uses the recurrent
path with the same source boundaries. No tolerance or reference target was
weakened to obtain the result.

Pinned source revisions, licenses, and adaptation scope are recorded in
`provenance/UPSTREAM.md` and the distribution notice manifest, including
Transformers' Apache-2.0 license and Qwen/Hugging Face attribution.

## Release-worker and service checks

The visual QA reference uses pinned modeling and cache classes with the
Transformers 5.15.1 greedy generation utilities, PyTorch 2.14.1, and Python
3.12.13. The pinned cache source SHA-256 is
`36c4cd52d66f39fdb51fa2fc92cfc43e2e8b152e07f7c2a17186eac1e81237e7`;
executed cache classes are checked. The native comparator uses the same frozen
fixtures and input token counts, comparing normalized answer hashes without
printing responses. A Python 3.13 AST serialization mismatch was rejected;
replaying with the reference interpreter reproduced the exact fixture identity.

- Visual QA: 8/8 normalized answers agree; native and reference each score 6/8.
  Color, shape, and spatial questions pass. Counting and OCR fail in both.
- Buffered and SSE image requests agree, including usage. Image prompt counts
  include the expanded image span: 89 tokens for the small color fixture.
- A 4,197,244-byte HTTP request succeeds with 1,049 prompt tokens. The observed
  request wall time is 3.13 seconds. Disconnect cleanup takes 1.70 seconds and
  a following image request succeeds on the same worker.
- Malformed-image recovery, single-use/foreign handles, context admission,
  controlled cancellation stages, and fresh request state pass. Cleanup failure
  fences before publishing success or cancellation.
- No staged files remain after requests. Actual host-death tests pass before
  claim for valid and partial files and after native prepare; normal close
  removes the staging directory.
- Candidate HTTP measurements observe 2,076,508,160 bytes peak child RSS and
  130,138,112 bytes peak API RSS. These are observations, not allocator ceilings
  or idle-machine performance guarantees.

The HTTP candidate run temporarily enables qualification only inside its test
process. Activation is followed by an ordinary catalog-matched API/CLI replay.
The control pipe stays bounded at 1 MiB. Service policy admits one PNG/JPEG,
4 MiB compressed bytes, a 6 MiB image body, and a shared 24 MiB pending-body
budget. Text bodies retain their 1 MiB limit. Native processing caps the result
at 1,048,576 pixels and 1,024 image tokens without changing the source resize
recipe. The image transaction uses one deadline across writes, preparation,
generation, streaming reads, and discard; unresponsive workers are reaped.

Worker teardown removes only opaque private files in its pinned directory,
which must be empty at startup. Killing both processes before claim can still
leave a staged file. No shared temporary-directory cleanup is performed.

## Regression checks

- Engine: 560 passed, 29 ignored; seven grammar integration checks and three
  doctests passed. Includes public source goldens for BF16 operations, a
  65-row chunk boundary with nonzero FP32 state, and FP64-prefix edge cases on
  CPU and Metal.
- Model-free generation worker: 40 passed. Generation-process checks: six passed.
- Guarded public engine API: actual-model cancellation/recovery checks passed.
- Converter: 4,327 passed, 88 skipped before the final executable-identity guard;
  its additional targeted image/runtime suite passed 71 checks.
- Packaging and launcher: 55 checks passed; pinned notice hashes also verified.
- Existing text fixture: two greedy requests and four complete vocabulary
  digests equal baseline `9eadf771`; consecutive-cache hits remain
  `false, true, false, false`. Generator and optimized scorer were built in
  separate Rust 1.98.1 commands.

Raw images, token/logit/feature arrays, runtime binaries, and detailed private
reports remain ignored. This record publishes aggregate results and hashes.
