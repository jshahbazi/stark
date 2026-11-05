# Ripple-65B Preview

Stark implements Ricursive Intelligence's `ricursive` text architecture in
Rust, using native Metal matrix kernels over memory-mapped weights. The model
has sliding attention with split-half RoPE, global attention without RoPE,
per-head Q/K RMS normalization, sigmoid attention gates, four normalization
stages per layer, embedding-conditioned value projections and clipped SwiGLU.
The value-residual seed is the initial scaled embedding and stays fixed across
all layers of each token's forward pass.

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

## Precision and limits

Q8_0 weights retain their original block scales and integer codes. Matrix
products accumulate into F32 outputs. Residuals, normalization, rotary
transformations, attention, gates and KV state use F32 arithmetic. This is a
separate precision policy from the publisher's BF16 inference and benchmarks.
Floating reference fixtures store F32 or BF16 weights and execute in F32 to
isolate architecture and storage behavior.

KV allocation grows with the actual prefix. Sliding layers retain at most
1,024 positions in the released architecture, while global layers retain the
request prefix. For the 80-layer checkpoint, F32 cache payload is approximately
512 MiB plus 128 KiB per prefix token: about 4.5 GiB at 32K tokens and 16.5 GiB
at 128K. Runtime buffers and macOS memory are additional. The configured
2,097,152-position boundary does not establish available memory or long-context
quality. A checked 32 GiB cache-payload bound stops growth before allocation
exceeds that limit; a failure clears the request state. Full-length execution
requires separate validation and a suitable memory policy.

Prefill currently advances one token at a time and omits discarded vocabulary
projections. Each request starts with fresh state; errors clear the sequence.
Cross-request prefix reuse, image input and DFlash2 speculative decoding are
not implemented for Ripple. Source chat-template preservation does not itself
establish structured tool-call parsing or agent performance.

## Reference and validation

The implementation follows the publisher's
[modeling source](https://huggingface.co/ricursiveAI/Ripple-65B-Preview/blob/e27e171792f303d74c048184e8259ba663cfa0d5/modeling_ricursive.py)
and
[configuration](https://huggingface.co/ricursiveAI/Ripple-65B-Preview/blob/e27e171792f303d74c048184e8259ba663cfa0d5/configuration_ricursive.py).
The source revision, hashes and license are recorded in
[upstream provenance](../provenance/UPSTREAM.md). Private numerical fixtures
and runtime outputs stay ignored; validation reports contain aggregate errors,
timings and hashes.
