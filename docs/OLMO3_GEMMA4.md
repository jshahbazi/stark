# OLMo 3 and Gemma 4 support

Stark has native dense text paths for OLMo 3 and Gemma 4, a bounded Gemma 4 MoE composite, and separate BF16 image profiles for the official Gemma 4 E2B, E4B, 12B, 26B-A4B and 31B checkpoints. Conversion support, native execution paths, and exact checkpoint qualification are separate states. The Gemma image implementation is wired through the converter, worker, CLI, desktop family handling and schema-4 artifact admission. E2B canonical HF image-feature comparison, bounded generation-worker lifecycle checks, and local cold/cached chat-client checks pass. E4B has separate source and execution validation. The installed desktop UI path and image performance have not been measured.

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

For the Gemma 4 E2B/E4B image artifacts, select the full vision-language conversion explicitly:

```sh
uv run --project tools/converter --locked stark-convert google/gemma-4-E2B-it \
  --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
  --artifact-mode vision-language \
  --output models/gemma-4-e2b-image

uv run --project tools/converter --locked stark-convert google/gemma-4-E4B-it \
  --revision 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c \
  --artifact-mode vision-language \
  --output models/gemma-4-e4b-image
```

This image conversion reads the official `processor_config.json` and writes its normalized recipe to the artifact's `preprocessor_config.json`. It consumes the checkpoint's own metadata and weights; it does not borrow image sidecars from another repository. Access to Google's official checkpoint may require accepting its Hugging Face terms.

## Native text and weight formats

| Family path | Current implementation boundary |
| --- | --- |
| OLMo 3 text | Dense `allenai/Olmo-3-7B-Instruct` layout, including full/sliding attention and configured YaRN. BF16 and MLX affine Q4/Q8 import are implemented. The pinned W4/group-64/BF16 source profile passes 14 strict numerical cases in the Q4 record below. |
| Gemma 4 text | Dense PLE E2B/E4B and dense no-PLE/global-K=V 31B layouts. The catalog pins official E2B/E4B and adds a pinned 31B extension. E2B affine Q4 conversion emits 901 tensors in a 2.605 GB native packed artifact. The pinned W4/group-64/BF16 E2B profile passes 14 strict numerical cases; Q8 and other variants require separate qualification. |
| Gemma 4 image | Official pinned E2B/E4B/12B/26B-A4B/31B BF16 artifacts, schema 4. Profiles are `gemma4-e2b-bf16-v1`, `gemma4-e4b-bf16-v1`, `gemma4-12b-bf16-v1`, `gemma4-26b-a4b-bf16-v1` and `gemma4-31b-bf16-v1`. Each requires its own source identity and separate execution qualification. |

For text generation, `gemma4_text` supports text-only PLE decoder weights. Quantized matrix payloads use affine 4-bit or 8-bit storage where the selected source and matrix layout pass conversion. Quantization applies to matrix storage; activations and other supported tensors use BF16 on these current native paths. The E2B Q4 output size is an artifact measurement, not an inference-speed result or qualification.

The image family profiles require their exact pinned official sources and BF16 matrices. Unified 12B, dense 31B and the 26B MoE composite are implemented. Extracted 12B and 31B image components pass their canonical three-image comparisons; full 12B/26B/31B checkpoint generation and released 26B expert/vision weights remain unqualified. The [family assessment](archive/runs/2026-11-07-gemma4-image-family-scope.md) records immutable identities, evidence and remaining blockers. Audio execution and video-frame processing are rejected even though their token identifiers are recorded for that validation.

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

Source RMS normalization retains each thread's four BF16-rounded values in registers and reuses them for normalization, avoiding another buffer read on both OLMo and Gemma. The source row limit remains 4,096 elements and the reduction order is unchanged.

Gemma's source PLE path also fuses the post-attention residual addition into pre-FFN RMS normalization. Each residual sum rounds to BF16 before statistics. The source PLE final residual addition and layer scaling share one dispatch, with an explicit BF16 sum before the final scaled BF16 store. These fusions require BF16, `reference_bf16` PLE and the explicit source RMS policy. Canonical BF16 and legacy PLE profiles retain their separate operations.

The [source fusion validation](archive/runs/2026-11-02-olmo3-gemma4-source-fusions.md) records fresh numerical, text/image worker and native-suite checks. The [qualified source fusion comparison](benchmarks/OLMO3_GEMMA4_SOURCE_FUSIONS_2026_11_02.md) measures Gemma's complete-request work rate 2.57% above the preceding C3 build and at MLX parity on the pinned 512-input/128-return workload. OLMo shows no material change from C3 and remains 2.31% above MLX on that workload.

## Gemma 4 E2B/E4B image artifacts and execution

The converter emits a `Gemma4VisionLanguageModelConfig` with artifact-contract schema 4 and a size-specific `gemma4-e2b-bf16-v1` or `gemma4-e4b-bf16-v1` family profile. E4B uses a 2,560-wide decoder projector and its own tokenizer hash; E2B uses width 1,536. The E2B weights cap remains 12GiB and E4B admits at most 20GiB. Native admission validates five components before loading tensors: `config.json`, `model.safetensors`, `tokenizer.json`, `preprocessor_config.json`, and `chat_template.jinja`. Their hashes, source identity, schema, geometry, tokenizer IDs and processor recipe must match the contract.

The runtime accepts one bounded PNG or JPEG image per request. CPU preprocessing decodes RGB, resizes by the aspect-ratio patch budget, and produces no more than 2,520 16-by-16 patches before 3-by-3 pooling to at most 280 soft tokens. The Rust/Metal path runs the learned vision encoder and projection. Image features enter the decoder after text embedding scaling; image PLE IDs are zero and image positions use the model's sequential RoPE behavior. Gemma 4 uses its own processor, token and artifact contract alongside the Qwen3.5 image path.

Conversion and runtime code implement this path, and the CLI, terminal chat, desktop host and schema-4 family admission are wired. For E2B, the canonical Hugging Face CPU image-feature comparison passes its 2% relative-RMS gate on three fixtures, and bounded generation-worker and local chat-client cold/cached checks pass. E4B qualification remains separately recorded. Installed desktop UI execution and performance benchmarking remain unmeasured; no image latency or throughput claim is made here. See the [2026-10-28 validation record](archive/runs/2026-10-28-olmo3-gemma4-native.md) for parity, image, worker and quantized-path results and limits.

The canonical HF reference is `scripts/gemma4_image_hf_reference.py`, using the official checkpoint, its Transformers image processor and vision implementation on CPU with Torch 2.14.1 and Transformers 5.15.1. `scripts/gemma4_image_reference.py` is a separate comparison against pinned MLX-VLM commit `53ebd1647cb83549b482a031ea70ba9d9ea60a04`; it uses MLX 0.32.4 and Transformers 5.14.0. The MLX profile's documented preprocessing and precision differences remain distinct from the canonical HF profile.

Text reference validation is separate: `scripts/dense_native_validation.py` and the `crates/stark-engine/examples/dense_parity.rs` example support model-bound decoder comparisons. The text MLX profile uses MLX 0.32.3 and its own frozen environment. The image and text dependency locks are intentionally separate; do not compare or merge their results as if they used one reference stack.

## Gemma 4 BF16 precision policy

Gemma 4 BF16 exports set `reference_bf16` on the per-layer embedding producer and consumers. The native path follows canonical Hugging Face text-model materialization boundaries: projection scaling before RMS normalization, token/projection sum before input scaling, post-attention residual before pre-FFN normalization, and PLE residual before the layer scalar. Token-embedding scale buffers are rounded to activation dtype; PLE projection and input scalars retain FP32 opmath. MLX's weak BF16 scalar conversion is a separate comparison profile. The flag defaults to false for existing artifacts, preserving their prior arithmetic.

These precision choices have model-bound evidence only within the references and fixtures named by their validation records. They do not establish every output configuration, quantized checkpoint, device, or end-to-end chat path.

The [original validation record](archive/runs/2026-10-28-olmo3-gemma4-native.md) preserves the BF16 qualification and earlier Q4 failures. The [Q4 precision record](archive/runs/2026-10-28-olmo3-gemma4-q4-precision.md) records the repaired profiles and their source-bound results. The [source fusion comparison](benchmarks/OLMO3_GEMMA4_SOURCE_FUSIONS_2026_11_02.md) records the current matched text measurements. The [preceding C3 comparison](benchmarks/OLMO3_GEMMA4_Q4_REPAIRED_2026_10_29.md) preserves its remaining Gemma gap, and the earlier [diagnostic Q4 timings](benchmarks/OLMO3_GEMMA4_NATIVE_MATCHED.md) remain historical. External measurements and licensing are in the [Mac engine evidence](benchmarks/OLMO3_GEMMA4_MAC_ENGINES.md).

## Generation prefix reuse

Image requests clear and bypass generation prefix reuse. E2B/E4B text artifacts currently disable prefix reuse because their owned sliding-attention layers use ring caches, and exact restoration of the overwritten ring rows is not implemented. Shared-KV consumers own no separate cache. Their source-specific physical ring traversal and canonical/source precision policies remain intact. A future ring snapshot implementation needs independent numerical and cached trajectory qualification before enabling this capability.

## Larger dense Gemma 4 image path

The 31B converter imports its no-PLE decoder with separate local and global KV head counts. Global K=V reuses the raw K projection for both native K and V rows, then applies learned K RMS/RoPE and independent weightless V RMS. The larger image encoder has 27 layers, width 1,152, head width 72 and ordinary linears. Native execution pads each vision head to 128 elements for the online attention kernel, sets all padded Q/K/V elements to zero and removes padded outputs before projection. Pooling preserves a BF16 spatial mean followed by F32 square-root scaling and learned standardization before its final BF16 store.

The 31B image block uses scalar positions. Global decoder layers remain causal; local layers permit same-image future keys while retaining the strict sliding lower bound. Image requests clear prefix reuse. Its artifact admission cap is 72GiB; this is a file bound and does not establish inference memory. The extracted pinned 31B tower/projector passes all three unchanged canonical feature gates on sealed source `e5dade43`, with relative RMS at most `0.00110689`. Logical-72 normalization precedes zero padding. Its larger-only attention policy preserves F32 QK, physical BF16 unnormalized exponential tiles and F32 accumulation, with separately validated scalar rescaling. The [qualified component record](archive/runs/2026-11-07-gemma4-larger-vision-qualified-component.md) preserves exact source/artifact identities and the known scalar rounding limit. Full-checkpoint language generation remains unqualified.

## Encoder-free 12B and sparse 26B image paths

The pinned 12B path uses 48px raw RGB patches and its encoder-free LayerNorm/projection/learned-position pipeline. It retains the 16px teacher grid with a 3x3 merge for image accounting and scalar positions. Its global layers are causal; local layers use window-bounded same-image visibility. Tokenizer and processor hashes are separate from the efficient-family profiles. The pinned 26B path combines the larger standardized vision tower with a no-PLE decoder, global K=V and the separately normalized dense/expert MLP branches. Expert routing applies source channel scales and learned expert gains after F32 top-k renormalization, then materializes source-ordered BF16 accumulation.

No-PLE BF16 artifacts explicitly select final residual/scalar and rotary product materialization; old no-PLE configurations without those controls fail image admission. Controlled residual/MoE fixtures pass CPU/Metal comparisons, and the extracted 12B image embedder passes its canonical three-image feature comparison. The six-layer unit-Q/K synthetic stress control remains a precision blocker. Full 12B/26B/31B checkpoint generation is unqualified. See the [12B component record](archive/runs/2026-11-07-gemma4-unified-vision-controls.md) and [MoE component record](archive/runs/2026-11-07-gemma4-moe-component-controls.md).
