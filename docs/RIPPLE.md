# Ripple-65B Preview

Stark implements Ricursive Intelligence's `ricursive` text architecture in
Rust, using native Metal matrix kernels over memory-mapped weights. The model
has sliding attention with split-half RoPE, global attention without RoPE,
per-head Q/K RMS normalization, sigmoid attention gates, four normalization
stages per layer, embedding-conditioned value projections and clipped SwiGLU.
The value-residual seed is the initial scaled embedding and stays fixed across
all layers of each token's forward pass. The publisher describes the design as
Gemma 3N-like, developed from an OLMo 3 baseline. Its released model type is
`ricursive`; the attention gates, value residuals, normalization and activation
clipping require Ripple-specific execution.

## Prepare and run

Run from a source checkout containing Ripple support:

```sh
uv run --project tools/converter --locked --python 3.13.13 python -m stark_convert.hybrid_prepare ripple-q8 --output models/ripple-q8
cargo +1.98.1 build --release --locked -p stark-generate
./scripts/stark chat --model-path models/ripple-q8
```

The Q8_0 recipe pins `AtomicChat/Ripple-65B-Preview-GGUF` at
`57066fdb2348a53f2261f096e0a793b9554b4106`. Its two shards total
68,840,113,568 bytes (64.11 GiB). Configuration, tokenizer and the tools-aware
chat template come from `ricursiveAI/Ripple-65B-Preview` revision
`e27e171792f303d74c048184e8259ba663cfa0d5`. Preparation preserves the packed
weights and records both source identities. Interrupted downloads can resume
using the same command. The output directory must be new.

Original floating Safetensors checkpoints use the same native architecture;
`ripple-bf16` prepares the pinned official release. Its weights alone occupy
about 120.7 GiB, making Q8_0 the practical starting point on a 128 GB Mac.
Successful preparation does not qualify loading, generation quality or speed.

The smaller `ripple-q4-k-m` recipe selects the standard (not AD) Q4_K_M file
from AtomicChat revision `df550a7178a87bbfb6ee2bdbd37603c553d536ee`.
It occupies 38,798,820,576 bytes (36.13 GiB) and preserves the mixed Q4_K/Q6_K
blocks. It uses the same pinned official metadata as Q8:

```sh
uv run --project tools/converter --locked --python 3.13.13 python -m stark_convert.hybrid_prepare ripple-q4-k-m --output models/ripple-q4-k-m
./scripts/stark chat --model-path models/ripple-q4-k-m
```

Native execution uses the existing packed Q4_K/Q6_K kernels and the Ripple
device graph, with F32 controls, accumulation and KV state. The independent
full reference helper also decodes these block types. Q8 validation does not
qualify this artifact; record Q4_K_M numerical and generation checks separately.
The released Q4_K projection shapes use the grouped32 two-row, 64-thread
kernel. The Q6_K down, KV and output-head projections use grouped16 with four
rows per 64-thread group. Both schedules have Ripple-local shape policies and
separate pipeline caches. They preserve packed codes, F32 accumulation and
Safe Metal math, while changing reduction grouping. Extreme finite inputs use
the original block arithmetic to preserve overflow behavior. No diagnostic
environment flag is needed.

## Precision and limits

Q8_0 weights retain their original block scales and integer codes. Matrix
products accumulate into F32 outputs. Residuals, normalization, rotary
transformations, attention, gates and KV state use F32 arithmetic. When the
complete projection inventory is admitted on Metal, these operations stay in a
device graph through each token, with a single
completion before returning its readout. Growing the KV allocation adds
separate bounded copy commands at capacity boundaries. This is a
separate precision policy from the publisher's BF16 inference and benchmarks.
Floating reference fixtures store F32 or BF16 weights and execute in F32 to
isolate architecture and storage behavior.

The released Q8 projection geometries use a measured, grouped two-row matrix
schedule within the Ripple device graph. The schedule retains Q8 block scales,
F32 accumulation and Safe Metal math, while changing reduction grouping. Other
shapes and storage formats use their established matrix kernels. This policy
is local to Ripple's graph.

Ripple retains the complete set of immutable Metal projection views when its
page-rounded size fits three quarters of the device's recommended working set,
after an 8 GiB dynamic reserve. Embeddings remain CPU mappings. Admission is
lazy and changes buffer lifetime only. A model that does not fit keeps the
existing staged execution with an 8 GiB streaming view cache. On the tested 128 GiB M5 Max, the Q8
projection inventory occupies 63.604 GiB within a 72.640 GiB policy limit. Request
KV state remains subject to the independent bounds below.

KV allocation grows with the actual prefix. Sliding layers retain at most
1,024 positions in the released architecture, while global layers retain the
request prefix. For the 80-layer checkpoint, F32 cache payload is approximately
512 MiB plus 128 KiB per prefix token: about 4.5 GiB at 32K tokens and 16.5 GiB
at 128K. Runtime buffers and macOS memory are additional. The configured
2,097,152-position boundary does not establish available memory or long-context
quality. A checked 32 GiB cache-payload bound stops growth before allocation
exceeds that limit; a failure clears the request state. Full-length execution
requires separate validation and a suitable memory policy.

The Metal cache includes persistent attention probability workspace and spare
capacity from amortized growth. Its checked 32 GiB bound includes these buffers
and the old layer allocation while that layer's replacement is copied. The
payload estimates above exclude this overhead.

Prefill currently advances one token at a time on the device and omits discarded
vocabulary projections. Each request starts with fresh state; errors clear the sequence.
Cross-request prefix reuse, image input and DFlash2 speculative decoding are
not implemented for Ripple. Source chat-template preservation does not itself
establish structured tool-call parsing or agent performance.
Both Q8 and Q4_K_M graphs own request-local cache buffers without a qualified
snapshot/restore interface. Exact restoration of those buffers is the concrete
blocker for cross-request prefix reuse; reuse remains disabled.

## Reference and validation

The implementation follows the publisher's
[modeling source](https://huggingface.co/ricursiveAI/Ripple-65B-Preview/blob/e27e171792f303d74c048184e8259ba663cfa0d5/modeling_ricursive.py)
and
[configuration](https://huggingface.co/ricursiveAI/Ripple-65B-Preview/blob/e27e171792f303d74c048184e8259ba663cfa0d5/configuration_ricursive.py).
The source revision, hashes and license are recorded in
[upstream provenance](../provenance/UPSTREAM.md). Private numerical fixtures
and runtime outputs stay ignored; validation reports contain aggregate errors,
timings and hashes.

Generate a small source-bound fixture and compare native Metal readouts:

```sh
uv run --project tools --locked --python 3.12.13 python scripts/ripple_native_fixture.py --output runtime/ripple-reference-q8 --q8
cargo +1.98.1 run --release --locked -p stark-engine --example hybrid_parity -- runtime/ripple-reference-q8
```

`--bf16` selects BF16 storage with F32 reference
execution; omitting both options creates an F32 fixture. Fixtures exercise
window eviction, global attention, nonzero value-residual coefficients,
embedding scaling, nonunit normalization weights and both clipping bounds.

For a prepared Q8 checkpoint, the full reference helper independently decodes
one layer at a time and runs the publisher's unchanged PyTorch decoder on CPU.
It avoids allocating an expanded copy of the entire model:

```sh
uv run --project tools --locked --python 3.12.13 python scripts/ripple_full_reference.py --model-path models/ripple-q8 --output runtime/ripple-full-reference
cargo +1.98.1 run --release --locked -p stark-engine --example hybrid_parity -- runtime/ripple-full-reference 0.001
uv run --project tools/converter --locked --python 3.13.13 python scripts/validate_ripple_generation.py --model models/ripple-q8 --worker target/release/stark-generate --output runtime/ripple-generation.json
```

The full reference compares the same quantized weights under F32 execution.
It does not establish parity with the original BF16 checkpoint's inference
policy or reproduce the publisher's quality scores.

Recorded checks: [native validation](archive/runs/2026-11-04-ripple-native-support.md).
The later [GPU performance report](benchmarks/RIPPLE_65B_GPU_2026_11_05.md)
records graph/Q8 validation, the matched worker comparison and the installed
development build.
The separate [Q4_K_M report](benchmarks/RIPPLE_Q4_K_M_2026_11_12.md) records
artifact-specific reference, generation and timing checks.
The later [K-quant kernel report](benchmarks/RIPPLE_Q4_KERNELS_2026_11_12.md)
records grouped Q4/Q6 dispatch and attention indexing validation, plus the
saved-binary performance comparison.
