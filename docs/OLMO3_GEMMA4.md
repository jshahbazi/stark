# OLMo 3 and Gemma 4 support

Stark has native dense text paths for OLMo 3 and Gemma 4, plus a separate BF16 image artifact for the official Gemma 4 E2B checkpoint. Conversion support, native execution paths, and exact checkpoint qualification are separate states. The Gemma image implementation is wired through the converter, worker, CLI, desktop family handling and schema-4 artifact admission. Canonical HF image-feature comparison, bounded generation-worker lifecycle checks, and local cold/cached chat-client checks pass. The installed desktop UI path and image performance have not been measured.

## Convert the pinned official checkpoints

Run from the repository root. The revisions are immutable Hugging Face commits recorded in the catalog and image contract.

```sh
uv run --project tools/converter --locked stark-convert allenai/Olmo-3-7B-Instruct \
  --revision 6e5971d9eba42665f5bd5a0fcf047f299ce1dccc \
  --output models/olmo-3-7b-instruct

uv run --project tools/converter --locked stark-convert google/gemma-4-E2B-it \
  --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
  --output models/gemma-4-e2b-it
```

For the Gemma 4 E2B image artifact, select the full vision-language conversion explicitly:

```sh
uv run --project tools/converter --locked stark-convert google/gemma-4-E2B-it \
  --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
  --artifact-mode vision-language \
  --output models/gemma-4-e2b-image
```

This image conversion reads the official `processor_config.json` and writes its normalized recipe to the artifact's `preprocessor_config.json`. It consumes the checkpoint's own metadata and weights; it does not borrow image sidecars from another repository. Access to Google's official checkpoint may require accepting its Hugging Face terms.

## Native text and weight formats

| Family path | Current implementation boundary |
| --- | --- |
| OLMo 3 text | Dense `allenai/Olmo-3-7B-Instruct` layout, including full/sliding attention and configured YaRN. BF16 and MLX affine Q4/Q8 import are implemented. The pinned W4/group-64/BF16 source profile passes 14 strict numerical cases in the Q4 record below. |
| Gemma 4 text | Dense PLE E2B/E4B layouts. The catalog pins official E2B. E2B affine Q4 conversion emits 901 tensors in a 2.605 GB native packed artifact. The pinned W4/group-64/BF16 E2B profile passes 14 strict numerical cases; Q8 and other variants require separate qualification. |
| Gemma 4 image | Official E2B BF16 image artifact only, schema 4, family profile `gemma4-e2b-bf16-v1`. It requires BF16 source matrices and activations. The E2B/E4B text path does not imply E4B image support. |

For text generation, `gemma4_text` supports text-only PLE decoder weights. Quantized matrix payloads use affine 4-bit or 8-bit storage where the selected source and matrix layout pass conversion. Quantization applies to matrix storage; activations and other supported tensors use BF16 on these current native paths. The E2B Q4 output size is an artifact measurement, not an inference-speed result or qualification.

Gemma 4 image conversion is restricted to the official pinned E2B source and BF16 weights. Gemma 4 31B, 26B MoE, unified 12B, audio, video, and E4B image paths are unsupported. The image artifact excludes audio execution and video-frame processing even though their token identifiers are recorded for rejecting unsupported content.

## Affine Q4 source precision

The converter automatically selects the audited MLX BF16 arithmetic profile for imported affine four-bit, group-64 matrices with BF16 activations. Reconvert older Q4 artifacts to receive the explicit precision settings; loading an older artifact does not add them. The source checkpoints used for the repair are:

```sh
uv run --project tools/converter --locked stark-convert mlx-community/Olmo-3-7B-Instruct-4bit \
  --revision d732c91ae02e90cd5d86e810fbeb9741794b4dd9 \
  --output models/olmo-3-7b-instruct-q4

uv run --project tools/converter --locked stark-convert mlx-community/gemma-4-e2b-it-4bit \
  --revision 238767527555cb75a05732a84dff5d6ba0dd6809 \
  --output models/gemma-4-e2b-it-q4
```

Packed Q4 weights remain packed. The profile preserves source BF16 activation, affine input-sum, normalization, rotary and partial-result materialization, with F32 accumulation where the source uses it. Prefill preserves the original logical projection widths when selecting its matrix kernel and reduction schedule. Gemma additionally preserves its BF16 embedding, approximate GELU, PLE scalar, softcap and bounded materialized-attention boundaries. Its short, wide-head prefill QK products use two F32 split-K partials before their final BF16 store. OLMo preserves its source sigmoid, score scaling, online attention reduction and YaRN prescaling. Source single-token attention pins the audited F32 arithmetic schedule. The native implementation adapts MIT-licensed MLX source; it does not invoke MLX at runtime.

Source cached attention also preserves physical sliding-window traversal across wrap and bulk updates. The bounded Gemma D512 M=1 route follows the source dense wide-vector QK reduction, with F32 dot accumulation followed by BF16 scores. These policies avoid changes in the reduction order that can affect later greedy choices.

The source RMS reduction is an explicit compatibility policy. The canonical native result can agree more closely with an F64 control at individual rounding boundaries. Default-false metadata keeps existing canonical and unrelated model paths on their earlier policy. Q8 and other structurally supported imports do not inherit this W4/group-64 numerical qualification. Exact checkpoints, devices, compiled binaries and validation workloads are recorded in the [Q4 precision investigation](archive/runs/2026-10-28-olmo3-gemma4-q4-precision.md).

Contiguous single-token Q/K normalization reuses the existing source RMS kernel. OLMo treats each 4,096-element full-width Q or K vector as one row. Gemma's bounded per-head route batches eight query rows or one key row, each with 256 or 512 elements; shared-KV consumers supply query rows only. Each row uses the same F32 gain vector, configured epsilon, four-contiguous-value source reduction and precise reciprocal square root. Normalized values and gains materialize in BF16 before multiplication. Stored Gemma Q/K gains use scale offset zero.

This routing requires BF16, the explicit source reduction setting, `OnlyNormalization`, learned scales, no bias or mean subtraction, and zero scale offset. Gemma additionally requires eight query heads and at most one KV head. Weightless value normalization, multi-token input, canonical BF16 profiles and other geometries retain their existing routing and validation. The host selects contiguous buffer subranges without changing the artifact format or precision policy.

## Gemma 4 E2B image artifact and execution

The converter emits a `Gemma4VisionLanguageModelConfig` with artifact-contract schema 4 and the `gemma4-e2b-bf16-v1` family profile. Native admission validates five components before loading tensors: `config.json`, `model.safetensors`, `tokenizer.json`, `preprocessor_config.json`, and `chat_template.jinja`. Their hashes, source identity, schema, geometry, tokenizer IDs and processor recipe must match the contract.

The runtime accepts one bounded PNG or JPEG image per request. CPU preprocessing decodes RGB, resizes by the aspect-ratio patch budget, and produces no more than 2,520 16-by-16 patches before 3-by-3 pooling to at most 280 soft tokens. The Rust/Metal path runs the learned vision encoder and projection. Image features enter the decoder after text embedding scaling; image PLE IDs are zero and image positions use the model's sequential RoPE behavior. Gemma 4 uses its own processor, token and artifact contract alongside the Qwen3.5 image path.

Conversion and runtime code implement this path, and the CLI, terminal chat, desktop host and schema-4 family admission are wired. The canonical Hugging Face CPU image-feature comparison passes its 2% relative-RMS gate on three fixtures, and bounded generation-worker and local chat-client cold/cached checks pass. Installed desktop UI execution and performance benchmarking remain unmeasured; no image latency or throughput claim is made here. See the [2026-10-28 validation record](archive/runs/2026-10-28-olmo3-gemma4-native.md) for parity, image, worker and quantized-path results and limits.

The canonical HF reference is `scripts/gemma4_image_hf_reference.py`, using the official checkpoint, its Transformers image processor and vision implementation on CPU with Torch 2.14.1 and Transformers 5.15.1. `scripts/gemma4_image_reference.py` is a separate comparison against pinned MLX-VLM commit `53ebd1647cb83549b482a031ea70ba9d9ea60a04`; it uses MLX 0.32.4 and Transformers 5.14.0. The MLX profile's documented preprocessing and precision differences remain distinct from the canonical HF profile.

Text reference validation is separate: `scripts/dense_native_validation.py` and the `crates/stark-engine/examples/dense_parity.rs` example support model-bound decoder comparisons. The text MLX profile uses MLX 0.32.3 and its own frozen environment. The image and text dependency locks are intentionally separate; do not compare or merge their results as if they used one reference stack.

## Gemma 4 BF16 precision policy

Gemma 4 BF16 exports set `reference_bf16` on the per-layer embedding producer and consumers. The native path follows canonical Hugging Face text-model materialization boundaries: projection scaling before RMS normalization, token/projection sum before input scaling, post-attention residual before pre-FFN normalization, and PLE residual before the layer scalar. Token-embedding scale buffers are rounded to activation dtype; PLE projection and input scalars retain FP32 opmath. MLX's weak BF16 scalar conversion is a separate comparison profile. The flag defaults to false for existing artifacts, preserving their prior arithmetic.

These precision choices have model-bound evidence only within the references and fixtures named by their validation records. They do not establish every output configuration, quantized checkpoint, device, or end-to-end chat path.

The [original validation record](archive/runs/2026-10-28-olmo3-gemma4-native.md) preserves the BF16 qualification and earlier Q4 failures. The [Q4 precision record](archive/runs/2026-10-28-olmo3-gemma4-q4-precision.md) records the repaired profiles and their source-bound results. The [repaired Q4 comparison](benchmarks/OLMO3_GEMMA4_Q4_REPAIRED_2026_10_29.md) records the final matched text measurements and remaining Gemma gap; the earlier [diagnostic Q4 timings](benchmarks/OLMO3_GEMMA4_NATIVE_MATCHED.md) remain historical. External measurements and licensing are in the [Mac engine evidence](benchmarks/OLMO3_GEMMA4_MAC_ENGINES.md).
