# Optional GLM EXL3/FP8 backend

The `glm-exl3` Cargo feature adds an optional macOS arm64 backend for the
`mlx-stream` GLM-5.3-Flash EXL3 K3/mul1 streaming pack. It uses a pinned MLX
engine through Stark's C ABI and returns full F32 vocabulary readouts to the
existing generation worker. The model's FP8/BF16/EXL3 arithmetic is retained.
This is an optional experimental engine path. Model-bound restoration and worker
checks are recorded for the pinned artifact below.

The loader requires the explicit `glm5_next` architecture and streaming
manifest contract. It selects this backend before the GGUF loader. Builds
without the feature report the pack as unsupported. Current admission reserves a
maximum 32,768-token context, caps the GPU budget at 112 GiB with a 2 GiB margin, and
chooses expert rows from the native phase bills. Vision is unavailable. The
`headless-reference` stage keeps the MTP head unloaded and reports MTP unavailable. A separate private
native MTP and head-prefix candidate uses native decode extension API version 1.
It requires a source-bound stage prepared with `--private-native-caps` and
explicit `STARK_GLM_EXL3_VALIDATE_NATIVE=1` during the Cargo build. Prefix support
also requires `--experimental-prefix`. Ordinary stages keep native capability
bits disabled and retain headless loading. This
candidate is not a production capability.

## Retained BF16 native profile

Prepare a stage with `--profile bf16-native-experimental` using the pinned
source/dependency command in the [upstream record](../../vendor/glm-exl3-bridge/UPSTREAM.md).
This explicitly selected profile includes prefix support and the retained BF16
draft head, depth 3, gate 0.5, native target-draw-match acceptance, original
transition-weighted expert retention and direct expert reads. Cache rows are
chosen by strict normal admission; the observed 129/156 counts are not forced.

Build the generation worker against that stage without validation-only flags:

```sh
STARK_GLM_EXL3_STAGE=/absolute/ignored/bf16-native-stage \
cargo +1.98.1 build --release --locked -p stark-generate --features glm-exl3
target/release/stark-generate --model-path /absolute/path/to/glm-exl3-pack --prefix-cache
```

For this profile, omitted MTP request settings select three drafts. Readiness
reports `mtp.default_drafts: 3`; `"mtp": false` or `"mtp_drafts": 0` selects
serial execution for that request. Other model families and the default
`headless-reference` stage retain their existing opt-in MTP behavior. The public
profile and `--private-native-caps` are mutually exclusive; private validation
stages continue to require their explicit validation build flag.

This remains an experimental profile. The earlier worker matrix passed 41/41,
and the quality smoke passed 22/24 across both modes; the all-pass gate remains
unmet. The retained 922.7 prefill / 20.3 decode observation failed CPU screening
and had unverified clocks. Selecting this profile does not qualify production
quality or throughput. Every new artifact retains separate validation evidence.
The [retained-profile setup record](../archive/runs/2026-11-05-glm-retained-profile.md)
binds the named stage and worker to its actual-model defaults and prefix smoke.

The bridge exposes serial steps and full readouts, and its native profiles
add a chosen-token MTP round interface with committed-round semantics
and draft-head state in prefix snapshots. The greedy 128-output and sampled
1,024-output checks match the native policy; the candidate's prefix mechanism
and invalidations pass. The actual Rust worker passes a 41-case contract matrix.
A separate objective smoke run passed 22 of 24 arms; both modes exhausted the
same greedy task's original output budget. Its all-pass gate remains unmet,
and production quality and throughput remain unqualified. See the
[native mechanism validation](../archive/runs/2026-11-04-glm-exl3-native-mtp-prefix.md)
and [GLM mlx-stream comparison](../benchmarks/2026-11-04-glm-mlx-stream-comparison.md)
for source-bound evidence; this reference does not duplicate their artifact
details or hashes.

Admission uses the upstream host's memory census and pack file-cache eviction.
Remaining foreign file cache stays charged to the baseline. After loading, the
bridge sets a process-local MLX wired capacity bounded by the admitted model's
bill, remaining budget and Metal's recommended working set. Teardown releases
model buffers before restoring the previous capacity. A pinned observational
MLX getter makes that restoration safe when a setter fails after mutation.

The target artifact is
[`OpensourceWTF/GLM-5.3-Flash-streaming-repack-exl3-3.05bpw`](https://huggingface.co/OpensourceWTF/GLM-5.3-Flash-streaming-repack-exl3-3.05bpw/tree/50237d5c26ceb98bfd85f779990b0f5e6d7af144),
revision `50237d5c26ceb98bfd85f779990b0f5e6d7af144`. Other pack revisions or
precision policies require separate qualification. The
[upstream bridge record](../../vendor/glm-exl3-bridge/UPSTREAM.md) identifies
all engine/dependency pins, licenses, local patches and staging commands.

## Build and load

Prepare the bridge stage explicitly using the pinned local sources described
in its upstream record. Cargo verifies the source inventory, stage identity and
every native artifact hash. It performs no native dependency download or build.

```sh
MACOSX_DEPLOYMENT_TARGET=26.2 \
STARK_GLM_EXL3_STAGE=/absolute/ignored/glm-exl3-stage \
cargo +1.98.1 build --release --locked -p stark-generate --features glm-exl3

target/release/stark-generate --model-path /absolute/path/to/glm-exl3-pack
```

The worker embeds the validated MLX library directory as a runtime search path.
Keep that stage available when running the worker. Moving or distributing a
stage needs a separate relocation and packaging qualification. The feature
requires macOS 26.2 or later and Apple silicon. It does not alter the standard
worker build or the existing GGUF optimized recipe.

## Prefix state and qualification

The bridge implements a checkpoint at a complete 16,384-token upstream prompt
call. It deep-copies mutable KDA convolution/recurrent state and DSA open-pool
tails into private CPU memory. Valid attention/indexer backing storage remains
model-owned; restore changes its logical lengths before appending the suffix.
Owner, epoch, context and source-prefix checks prevent reuse after an
intervening fresh prompt, failed request, cancellation or model replacement.
Snapshot and restore peak memory are charged to admission.

A bridge stage prepared with `--experimental-prefix` advertises snapshot
support. An ordinary `glm-exl3` worker linked to that stage exposes the
16,384-token boundary through its existing `--prefix-cache` option. A serial-only
stage keeps reuse unavailable. `STARK_GLM_EXL3_VALIDATE_PREFIX=1` enables the
expensive state-hash diagnostics for qualification; worker reuse does not
require that build setting. Each artifact and precision policy still requires
separate model-bound qualification.

The `glm_exl3_validate` example accepts bounded token input through an anonymous
stdin pipe. It compares full F32 readouts and hashes of complete logical KDA/DSA
state between restored and fresh changed-suffix requests, followed by greedy
continuation. It also checks stale snapshots, failed token admission and a
cancelled prefill callback. It emits aggregate counts and hashes only.
`scripts/validate_generation_prefix_cache.py` covers the worker's consecutive
request and host-key rules. Use the actual artifact, prompts longer than the
advertised boundary, and a context that admits follow-ups. Preserve separate
source-bound receipts whenever tooling or precision changes.

For the raw pack, provide `--download-receipt` with its completed pinned flat-pack
receipt. The harness binds both expert banks, residents and metadata to the
previously verified hashes and checks sizes, modification times and inodes
before and after execution. It renders the shipped HF template through the
native evaluator using an in-memory codec, and records template/helper hashes.
It does not modify the pack. A 32,768-token context and the default first-turn
target of 18,560 tokens admit the synthetic qualification session.

Performance measurements must use uncached requests, one excluded warm-up,
screened measured runs and the effective admitted expert row counts. The
bridge benchmark includes final full-readout materialization. The historical
upstream binary and 16K fixture are unavailable publicly, so new measurements
must identify their source tuple and workload rather than claim an identical
reproduction.

The [2026-11-04 qualification record](../archive/runs/2026-11-04-glm-exl3-qualification.md)
binds the residency integration and ordinary prefix checks to their source and
artifact. The private native MTP and head-prefix candidate's current status is
recorded in the [GLM mlx-stream comparison](../benchmarks/2026-11-04-glm-mlx-stream-comparison.md).
The qualified bridge throughput benchmark uses serial greedy decoding.
The historical upstream recipe enables MTP with three draft tokens per round
and uses different sampling, output length and expert-row settings. Its decode
rate needs a separate matched control before attributing a speed difference.
