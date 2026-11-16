# Decider Flash v1.3

Stark provides a dedicated native Rust/Metal path for
`perplexity-ai/pplx-decider-flash-v1.3`, pinned to
`d06fb3f2d97e0a52325a0d463c450cf0338f44fe`. It supports BF16 binary decisions,
custom answer choices, ordinal scores and one PNG/JPEG image per request.
The vision tower is loaded by default. Qualification results and their exact
scope are recorded in [20261116-DF13-01](model-experiments/2026-11-16-decider-flash-v13.md).

The [DF13-03 run](archive/runs/2026-11-16-decider-flash-v13.md) passed all 15
reference decisions, including five image cases, with maximum probability
error 0.00596. Repeat/error/reset/cancellation/restart controls passed. The
matched text residency timing was inconclusive because baseline variability
exceeded its fixed limit. Long requests remain slow on this initial path.

This is a standalone engine/CLI integration. The desktop catalogue, existing
generic Decision workflow and HTTP chat endpoint do not dispatch this model.
Its `DeciderModelConfig` artifact cannot be loaded as a causal language model.

## Prepare and run

Obtain the complete pinned Hugging Face snapshot, including its source metadata,
tokenizer, template, backbone shards and separate `readout.safetensors`. Keep it
and converted output in an ignored directory. Build using the pinned toolchain:

```sh
cargo +1.98.1 build --release --locked -p stark-score --bin stark-decide
cargo +1.98.1 build --release --locked -p stark-template
uv run --project tools/converter --locked python -m stark_convert.decider prepare \
  --source /absolute/path/to/source-snapshot \
  --output /absolute/path/to/prepared-decider
```

Preparation preserves the tower, maps the separate 255-row readout, retains the
source licence/notice and records source and output hashes. It rejects existing
output directories, incompatible processors, old/causal protocols, unexpected
tensor namespaces and incomplete heads. Quantized exports are not supported by
this path.

The JSONL frontend accepts the upstream decision request schema:

```sh
uv run --project tools/converter --locked python -m stark_convert.decider decide \
  --model-path /absolute/path/to/prepared-decider \
  --worker "$PWD/target/release/stark-decide"
```

Each input line contains `state` and a nonempty `questions` object. Question
types are `noul` (binary), `choice` (a map of option names to descriptions) and
`score` (an ordered list of level descriptions). Add an `images` list containing
one absolute local image path for image classification. Responses contain
`answers` and input-token usage; no output tokens are generated. Probabilities
use the checkpoint's saved temperature, normalized over the valid options.

Python callers can use `stark_convert.decider.Decider(model_path, worker_path)`
as a context manager and call `decide(request, temperature=None)`. Calls are
serialized. A deadline or broken worker terminates that process; callers must
create a new instance. Request content and native exceptions are excluded from
diagnostics. Keep response traffic private.

The Rust worker's lower-level JSONL protocol accepts `operation: "decide"`,
`input_token_ids`, one final `readout_positions` entry, one `option_counts`
entry, optional `images`, and `context_tokens`. Use the Python frontend to
construct the checkpoint's exact protocol; ordinary chat prompts are invalid.

## Execution and limits

- At most 4,096 expanded input tokens per question, including template and image
  positions. Inputs are rejected without truncation.
- One static PNG/JPEG image per request. Native source dimension/byte admission
  also applies. The Qwen processor resizes to the saved 1,638,400-pixel maximum
  and 65,536-pixel minimum, subject to patch alignment and aspect-ratio bounds.
- Each question is processed separately with its complete state, matching the
  upstream default `mode="separate"`. Optional joint/isolated execution is not
  implemented.
- Full-attention layers see the whole sequence. DeltaNet layers remain causal.
  BF16 inference uses the existing Qwen reference operators and F32 reductions
  and recurrent state where those operators specify them.
- Every forward pass has fresh KV, convolution and recurrent state. Prefix
  reuse is disabled because earlier full-attention representations depend on
  later question tokens. No completed result cache is retained. Images, errors
  and reset therefore cannot inherit previous request state.
- Generation, function tools and MTP are unavailable for this decision-only
  checkpoint. The inherited MTP configuration does not provide executable
  predictor tensors or a generative vocabulary readout in this export.
- Locate span extraction uses a separate adapter and is explicitly unsupported.
  It is unnecessary for the classification/scoring and image workflow.

The worker's readiness reply reports the admitted capabilities and limits.
`--text-only-validation` is a diagnostic tower-residency control for matched
text measurements; normal loading always retains image support.

## Reference qualification

The [locked reference runner](../tools/validation/decider_flash/README.md) uses
Perplexity's unmodified, hash-verified Python source. On CPU/MPS that reference
selects FP32; native Stark selects BF16. This comparison does not qualify the
upstream CUDA BF16/compiled path or another checkpoint/quantization.

Run `scripts/qualify_decider_flash.py` through the converter environment with
`--snapshot`, `--model`, `--worker`, a private `--fixtures` JSON file and an
ignored `--output` path. The fixture contains a `cases` list of `request`
objects. The validator loads reference and native models sequentially, compares
responses in memory, checks repeated/reset/error/restart behavior, and writes
aggregate errors, counts and identity hashes. It never saves raw responses.
