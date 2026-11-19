# Elephant 1

Stark admits `baselabs/elephant-1` as a pinned, native Decision-only model at
revision `b3f8a93462fb1051a9de031e94a7ff380e254072`. The checkpoint is a 9B
Qwen3.5 multimodal full fine-tune with the MIT license declared in its model
card. The serving protocol comes from OneJev
`81ce62f1597c91e46767d4d02d6ac2e18534fe94`.

## Current scope

Elephant 1 has a separate catalog entry and a pinned Hugging Face/local-source
installation path. The native artifact:

- retains the Qwen3.5 vision tower and accepts one PNG or JPEG;
- runs full-attention layers with a bidirectional state prefix and a
  bidirectional question suffix;
- keeps the hybrid recurrent layers causal;
- scores the 256 OneJev answer labels through the checkpoint's BF16 LM rows,
  with F32 accumulation and output;
- uses the 32-lane MLX BF16 dot schedule for 2+-row reference-prefill matrix
  products, chunked at 1,024 rows; one-row and unrelated profiles retain their
  existing paths;
- uses the 16,384-token trained request limit;
- is Decision-only and does not expose Chat, generation, Locate or function
  tools.

Function-tool qualification is not applicable to this non-generative
checkpoint. The ordinary tool parser is never selected.

Preparation preserves the original README, config, tokenizer, processor, patch,
setup and readout metadata. It records source and output hashes, the MIT
declaration and the exact 18,819,722,392-byte source weight identity. The
prepared converter includes the OneJev source revision and patch provenance.

## Prefix reuse and images

Prefix reuse is disabled. The block-bidirectional full-attention representation
of the shared state is reusable only among suffix branches of the same request;
it cannot be retained as a causal generation prefix because each branch's
future question tokens determine the full-attention values. Each request starts
with fresh full-attention, recurrent and convolution state. No completed
distribution is cached or persisted.

Image support remains enabled on the normal model. The source processor admits
up to 16,777,216 processed pixels, while the final expanded text-and-image
sequence must still fit 16,384 tokens. Image requests never reuse state. Text,
image and text-after-image qualification remain separate artifact-bound work.

## Installation and limits

In **Manage Models**, open **Model Catalog**, choose **Elephant 1**, and select
its pinned BF16 entry. **Add from Source** also recognizes
`baselabs/elephant-1` or a complete pinned local snapshot. Prepared artifacts
must retain their `config.json`, tokenizer, template and converted weights.

Run the installed model to select **Decision** mode. Enter a question and
distinct ordered answers, optionally with one image. Multiple API questions are
executed separately; each branch receives the exact trained state prefix.
Inputs are refused rather than truncated.

Direct preparation:

```sh
cargo +1.98.1 build --release --locked -p stark-score --bin stark-decide
cargo +1.98.1 build --release --locked -p stark-template --bin stark-template
uv run --project tools/converter --locked --python 3.13.13 python \
  -m stark_convert.elephant_install prepare \
  baselabs/elephant-1 --kind huggingface \
  --revision b3f8a93462fb1051a9de031e94a7ff380e254072 \
  --output runtime/elephant-1-prepared
```

## Qualification status

Pinned metadata, hash pins, all 256 tokenizer slots, exact conversion and the
native worker load passed on Apple silicon. Bounded synthetic text and
one-image decisions completed with normalized distributions, and repeat,
reset and worker-restart checks were identical. Those controls are not a
reference-quality result: source-model numerical choices/probabilities,
broader image cases, full lifecycle, memory under requests and
matched-performance evidence remain pending. The current code does not inherit
Decider Flash, another Qwen3.5 checkpoint's numerical result, or Base Labs'
published quality numbers.

The first matched local speed run on Apple M5 Max measured median warm native
wall time reductions of about 20% for a 113-token text request, 34% for a
526-token text request and 32% for a 32x32 one-image request. The 113-token and
image slot logits were identical to the previous BF16 reference; the 526-token
slot logits differed by at most one BF16 step and retained the same selected
option. These aggregate controls do not transfer to longer prompts, other
hardware or upstream numerical qualification.

The compatibility attempt is recorded in
[20261118-ELEPHANT-01](model-experiments/2026-11-18-elephant-1.md).
