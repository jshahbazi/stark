# Gemma 4 E4B BF16 image qualification

Date: 2026-11-07

The pinned official `google/gemma-4-E4B-it` image path passes conversion, canonical image-feature and last-logit comparisons, bounded worker lifecycle checks, local client cold/cached checks, and controlled cancellation recovery on Apple M5 Max. This qualifies the exact artifact and builds below on three private RGB PNG fixtures. Each fixture generated one output token; this is a bounded functional and numerical check. Installed desktop UI execution, long-form image generation, JPEG end-to-end generation and image performance remain unmeasured.

## Source, artifact and build identities

| Component | Identity |
| --- | --- |
| Official source revision | `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c` |
| Source weights SHA-256 | `cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503` |
| Source implementation | `e3da5f8a` |
| Converted weights SHA-256 | `58e1ed2e61286ccc58223bc46a2efe909f11b0b8eca6c4ccb03f00cdb435ea5a` |
| Artifact config SHA-256 | `977693a5a4bb1378e26db4e4eaff457eef788240c57c5f0c94a369d57b3f302e` |
| Tokenizer SHA-256 | `91d9cf86906f78e197021fda88214d448e5c120665d0243a8e776e046986c6bc` |
| Processor SHA-256 | `32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c` |
| Template SHA-256 | `0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5` |
| Generation worker source | Integration commit `f2edef50` |
| Sealed generation worker SHA-256 | `32816d8494e29d0a5ca6858f4bf60f6203c18fec256befd4d3de5eda738f9449` |
| Numerical test source | `aba886fa` |
| Sealed numerical test executable SHA-256 | `5cc75e75264bf6205e9400c7cca4ebf1d06d0da7fb9814af7e23f4419ab82058` |
| Canonical reference manifest SHA-256 | `0befed870aa83410aff0985dafd9a0aa0f9fc3664579961927f0a6b527e9378f` |

The source download reused the shared Hugging Face cache, without an extra copied source directory. Conversion produced 1,234 tensors and 15,266,007,916 bytes of packed artifact output. Measured transformation workspace was 8MiB and peak converter RSS was 634,290,176 bytes. The artifact excludes audio execution. E4B has its own `gemma4-e4b-bf16-v1` schema-4 profile and tokenizer identity.

## Canonical comparison

The reference uses Torch 2.14.1, Transformers 5.15.1 and Pillow 13.0.0 on CPU, through the previously locked reference environment. `scripts/gemma4_image_hf_reference.py` checks immutable source identities and re-tokenizes each fixture prompt using E4B's tokenizer. No E2B image features, token sequences or language outputs are reused as E4B references.

| Fixture index | Image tokens | Patch equality | Feature relative RMS | Feature cosine | Last-logit centered RMS | Last-logit KL | Last-logit greedy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 266 | Exact | 0.0114521 | 0.9999344 | 0.0942021 | 0.0000344773 | Agree |
| 1 | 266 | Exact | 0.0130552 | 0.9999148 | 0.0914297 | 0.0000000083 | Agree |
| 2 | 256 | Exact | 0.0147230 | 0.9998917 | 0.0796816 | 0.0000175079 | Agree |

All three cases pass the existing independently calibrated `gemma4-canonical-bf16-image-v2` gates: feature relative RMS at most 2%, cosine at least 0.999, and bounded centered logit error, common offset, extrema, KL and greedy agreement. Image-feature argmax is not a language-generation metric and is not required by the feature gate.

## Worker, client and cancellation behavior

`scripts/gemma4_image_worker_validation.py` passed three canonical greedy comparisons, fresh replay, restart replay, malformed-image recovery, single-use prepared images, staged-file cleanup and the red/blue output counterfactual. The same harness's client route passed a cold local image check and a subsequent cached check using the exact worker/artifact identity.

`gemma4_completed_projection_cancellation_preserves_loaded_instance` passed preparation cancellation and cancellation immediately after completed image projection, then generated text using the same loaded instance. Gemma's current controlled cancellation checks bracket the synchronous projection; they do not fence immutable weights at those safe boundaries. Backend, allocation, nonfinite-output and failed GPU-completion errors retain the failure fence.

Images, prompts, logits, token vectors and generated text remain in ignored private fixtures. Published values are aggregate checks and hashes. Other Stark processes were present, so observed functional timings do not qualify quiet-host latency or throughput.
