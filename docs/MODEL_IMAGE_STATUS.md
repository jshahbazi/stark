# Model image capabilities and remaining work

Reviewed 2026-11-08 against main `2c1e75b1` and the open [text-prefix PR #110](https://github.com/jshahbazi/stark/pull/110). Saved on `codex/catalog-image-capability` as requested. This is a source/evidence audit; no new models were run for this inventory. It does not describe the capabilities of an installed release.

The [default-image parity worklist](IMAGE_DEFAULT_PARITY_PLAN.md) turns this
inventory into current implementation/qualification tasks, includes the later
Lily text-MTP evidence on PR #112, and defines the standing requirements for
future models. The source-bound snapshot below retains its original scope.

The scope is all 18 generic converter families plus the two additional native families, their distinct image paths, and all 120 pinned repository entries. Live Hugging Face discovery can return more compatible checkpoints; those cannot be exhaustively enumerated and need individual preflight and validation.

**Added** means implementation exists. **Verified** names the actual evidence. **Qualify** means run the missing checks; it does not imply a known numerical defect. **Implement/fix** identifies missing code or metadata. A text-only artifact can come from an image-capable source. Its format and preparation route determine what Stark can load.

## Image-capable models and artifact paths

2026-11-12 update: Add Source now uses repository-independent retained-image
admission for supported Flash Next Lily, external-table affine and indexed PLE
layouts. The actual `nace-ai/Qwen3.8-Flash-Next-MLX-KQuant` artifact passed the
Add Source controller's installation and local image checks, plus 14 explicit
image/text/MTP/prefix lifecycle controls. This adds evidence for that exact
artifact; older rows retain their original checkpoint-specific evidence.
See the [implementation and qualification record](archive/runs/2026-11-12-flashnext-general-images.md).

| Model / artifact | Images added in Stark? | Verified evidence and limits | Text optimizations with images loaded | Remaining action |
| --- | --- | --- | --- | --- |
| Qwen3.5 dense BF16, official 0.8B | Yes: native image conversion, tower, fusion and serving | Source-bound numerical/image-worker baseline exists. Small image-answer checks are bounded evidence, not general quality proof. [Qwen ledger][qwen] | Prefix snapshots implemented in open PR #110; 32 exact cases and CPU/Metal checks pass for this artifact at context 8,192. Vision MTP unavailable. | Merge/ship the reviewed caching change after checks; retain artifact-specific qualification. |
| Qwen3.5 dense BF16, other sizes/fine-tunes | Implemented compatible family path | Does not inherit the 0.8B numerical, caching or performance result. | Same eligible snapshot implementation in PR #110; other artifacts unqualified by that run. | Qualify each exact checkpoint/format. |
| Qwen3.5 dense affine 4/8-bit decoder + BF16 tower | Yes, supported layouts | Pinned 0.8B packed-reference cases matched source logits exactly. Arbitrary quantizers and quantized towers are not covered. [Qwen ledger][qwen] | Eligible converted decoder snapshot path in PR #110; affine artifact caching still needs its own replay. Converted vision artifacts omit MTP heads. | Qualify caching per quantization; separate work to retain and qualify text MTP. |
| Dense Qwen3.8-27B, `Jundot/Qwen3.8-27B-oQ4e-mtp` | Yes: separate converted image artifact | BF16 tower/stage checks and actual image/cancellation/recovery lifecycle pass. Full 27B decoder image-logit parity and image speed remain unverified. [Qwen ledger][qwen] | Eligible converted snapshot implementation; no 27B image-artifact cache replay in PR #110. Text artifact MTP does not transfer to its vision conversion. | Qualify full decoder/image path and cached text; add/qualify vision-artifact text MTP separately. |
| Qwen3.5 35B-A3B, affine-source MoE profile v4 | Yes | Tower, tensor inventory and worker lifecycle evidence exist; full decoder parity/throughput remain bounded or unverified. [Qwen ledger][qwen] | Eligible snapshot primitives are implemented; this artifact has no new caching qualification. | Full decoder/reference and cache validation; reconcile legacy catalog metadata. |
| Qwen3.5 35B-A3B, BF16-source MoE profile v5 | Structural conversion/admission and operators added | Router/expert/scratch component controls exist. Complete official checkpoint image execution and parity are unverified. [Vision contract][vision] | Eligible snapshot primitives alone do not establish full-model restoration. | Run full artifact admission, image execution, decoder parity, memory and caching checks. This is unfinished qualification. |
| Qwen3.8 Flash Next, retained `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, schema 6 | Yes: explicit retained image preparation/upgrade | All 27 vision-tower layers exact on two source fixtures; actual image-worker lifecycle passes. Full retained-decoder parity and image throughput remain unverified. [Retained images][retained] | Engine already supports a 4,096-token text snapshot when no image transaction is active. PR #110 makes the host request this capability for vision workers. Loading in image mode disables MTP heads even for ordinary text. | Qualify text reuse and speed with the tower loaded; separately enable/qualify text-only MTP on image-loaded instances. |
| Qwen3.8 Flash Next, retained `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`, schema 6 | Explicit preparation and retained image path added | This distinct complete artifact lacks image execution qualification in the cited work. [Retained images][retained] | Lily-backed retained state has text snapshot support; host enablement is in PR #110. Actual artifact/readiness and performance need checking. Image-mode MTP disabled. | Qualify actual image execution, text caching, speed and memory before treating it like the pinned Lily result. |
| Qwen3.8 Flash Next, indexed affine `nace-ai/Qwen3.8-Flash-Next-MLX-KQuant`, revision `05cb5c56` | Yes: shared Add Source preparation, preserving BF16 tower, PLE ranges and MTP | Complete real-source header admission, actual controller install, automatic image checks and 14 bounded lifecycle controls pass. Full source-reference decoder parity and general image quality remain unassessed. [Run record](archive/runs/2026-11-12-flashnext-general-images.md) | Actual MTP0/MTP2 text equality, proposal counters and exact 4,096-token restoration pass with images loaded. Short paired timing meets the 5% bounds but fails the strict clean-endpoint screen; no general speed claim. Images use zero drafts/reuse and clear text prefixes. | Qualify MTP1/MTP3, broader workloads, sustained performance and other artifacts separately. |
| Qwen3.8 Flash Next, typed Uzu `qwen4_exp` text conversion | No image components in this artifact | Separate implemented text route; converter intentionally omits vision and MTP. [Native reference][native] | Existing typed-text reuse qualification is separate from retained images. | Use the explicit retained image route for images. Adding images to this typed format would require new implementation and qualification. |
| Qwen3.8 Flash Next, original PipeNetwork/F32 text route | No image-fusion bridge | Text implementation exists; retained vision payload alone does not enable images. [Native reference][native] | F32 graph keeps prefix capability disabled. | Implement a bridge and exact state restoration if this route is to support them; use retained image preparation where applicable. |
| Gemma 3 4B BF16, `google/gemma-3-4b-it` | Yes | Exact processor patches; canonical feature/logit limits, greedy trajectories and worker/API lifecycle pass. Full-model logits are within fixed tolerances, not universally bit-exact. [Gemma ledger][gemma] | Physical-ring snapshot/restore and scoped cache lifecycle pass for the pinned 4B artifact. [Continuation][gemma-next] | Larger/quantized artifacts and installed UI execution remain separate; prior-text baseline timing is ineligible. |
| Gemma 3 12B / 27B BF16 | Structural conversion/admission added | Full actual-checkpoint numerical replay and image-worker execution unverified. [Gemma ledger][gemma] | Ring snapshot implementation exists; these artifacts lack their own replay. | Qualify full models and cache lifecycle; reconcile stale 27B catalog status. |
| Gemma 3 quantized image checkpoints | No qualified native quantized image path in this inventory | BF16 Gemma 3 results do not qualify 4/8-bit exports. | No image cache qualification. | Implement/admit the intended quantization and qualify it; do not label these exports ready from the BF16 results. |
| Gemma 4 E2B / E4B BF16, pinned instruct image profiles | Yes, separate profiles | Source/execution evidence and retained canonical worker replays exist for each. This is not universal full-logit parity. [Gemma ledger][gemma] | Physical-ring snapshots and source-bound ordinary-text cache controls exist for both small profiles. E2B assistant remains staged. [Continuation][gemma-next] | Broader/default assistant context and companion provisioning remain open; staged serial MTP is slower, so preference defaults off. |
| Gemma 4 12B BF16 unified | Encoder-free patch/embedder implementation added | Extracted three-image component comparison passes; full 12B decoder generation unverified. [Gemma ledger][gemma] | Ring-state snapshot blocker. | Full model execution/parity/memory qualification, then caching. |
| Gemma 4 26B-A4B BF16 | Tower and bounded MoE/decoder implementation added | Component/synthetic controls exist; released expert weights and complete generation unqualified. [Gemma ledger][gemma] | Ring-state snapshot blocker. | Validate actual expert/decoder weights and complete requests; implement caching restoration. |
| Gemma 4 31B BF16 | Pinned tower/projector and decoder admission implemented | Three source feature gates pass; component similarity is not full 31B generation proof. One broad scalar control counterexample remains documented. [Gemma ledger][gemma] | Ring-state snapshot blocker. | Full checkpoint/reference execution; resolve or bound remaining numerical differences; implement caching restoration. |

All current image transactions clear/bypass text-prefix state and disable image MTP. Prefix reuse here concerns ordinary text requests on the loaded image artifact, never reuse of image-derived state.

The quick Lily regression in this chat used its existing **text-loaded** worker. Identical output and roughly 83.7 versus 84.3 tok/s demonstrate a short text-path sanity check. It did not measure text speed with the vision tower loaded. The separate small Qwen3.5 vision-artifact benchmark cannot establish Lily performance.

## Complete family inventory

“No image route” describes Stark's admitted artifact, not every model sharing the upstream brand. Vision siblings such as Qwen-VL or unrelated Llama/Mistral multimodal architectures require their own onboarding.

| Family / registry ID | Current Stark coverage | Images in this route | Status / work needed |
| --- | --- | --- | --- |
| Qwen3 dense `qwen3` | Implemented text/conditional Decision | No image route | Existing text support; no image repair inferred. Add a separate supported vision architecture if requested. |
| Qwen2 dense `qwen2` | Implemented text/conditional Decision | No image route | Qwen2-VL is not covered by this text adapter. |
| Llama dense `llama` | Implemented text/conditional Decision | No image route | Multimodal Llama needs separate implementation. |
| Mistral dense `mistral` | Implemented text/conditional Decision | No image route | Multimodal siblings need separate implementation. |
| Gemma 2 `gemma2` | Implemented text/conditional Decision | No image route | Text coverage; image support would be new scope. |
| Gemma 3 text `gemma3_text` | Implemented text/conditional Decision, including compatible derivatives | Text artifact omits images | Use the separate Gemma 3 image adapter where supported; source model capability must be inspected. |
| Gemma 3 images `gemma3` | Image-artifact conversion; text prompts can run through admitted image model | BF16 paths above | 4B has source-bound ring/cache evidence; 12B/27B remain independently unqualified. |
| Gemma 4 `gemma4_text` and aliases | Dense text paths plus distinct pinned BF16 image profiles | Five named image profiles above | Small profiles have ring/cache evidence and staged E2B assistant controls; larger profiles remain independently unqualified. |
| OLMo 3 `olmo3` | Implemented native text/conditional Decision | No image route | The Gemma work does not add images to OLMo. |
| Phi-3 dense `phi3` | Implemented text/conditional Decision with constraints | No image route | Phi vision variants require separate support. |
| Granite dense `granite` | Implemented text/conditional Decision | No image route | No image implementation/qualification established by this adapter. |
| SmolLM3 `smollm3` | Implemented text/conditional Decision | No image route | Vision sibling support would be new scope. |
| Qwen3.5 dense `qwen3_5` | BF16/eligible affine text and image routes; dense Qwen3.8 extension | Yes, compatible artifacts | Merge/ship #110 after review; qualify other artifacts independently. |
| Qwen3.5 MoE `qwen3_5_moe` | Affine and BF16-source image profiles; BF16-source text/Decision unsupported | Yes, evidence levels differ | Complete BF16 full-model checks, broader parity/cache qualification and metadata reconciliation. |
| Qwen3.8 Flash Next `qwen4_exp` | Typed text conversion and separate retained native backends | Route-dependent | Retained Lily image support exists; typed text conversion and Pipe/F32 are separate. See detailed rows. |
| LFM2 `lfm2` | Implemented text/conditional Decision | No image route | LFM vision and planned LFM2.5 entries do not inherit support. |
| LLaMBA `llamba` | Implemented text/conditional Decision | No image route | No image implementation established. |
| ModernBERT `modernbert` | Classifier conversion | No app chat/Decision/image workflow | Do not advertise as a supported chat model. |
| GLM-5.3 Flash `glm5_next` | Native pinned GGUF text path; optional EXL3 profiles separate | No image execution | Text support only. Full GLM-5.3 `glm_moe_dsa` is a different, unsupported architecture. |
| Flint `flint` | Native pinned INT4 text path | No image execution | Image support would require implementation. Prefix restoration remains separately blocked/unqualified. |

Decision mode additionally requires a compatible direct-answer template, unique single-token labels and checkpoint checks. Family recognition or conversion does not certify Decision accuracy, runtime execution or image support. [Registry][registry] [Native registrations][native-registry]

## Concrete next actions

| Action | Models / paths | Type |
| --- | --- | --- |
| Finish review/CI and ship #110; qualify additional artifacts | Eligible converted Qwen vision decoders; vision-worker host capability request | Added on PR, not merged as of this audit |
| Test image-loaded text caching and speed | Retained Lily Q4-XL first, then distinct retained mixed-4/8 artifact | Qualification gap; existing text-only benchmark does not cover it |
| Restore text-only MTP availability while images are loaded | Retained Lily/mixed: loader currently disables heads; converted Qwen vision: conversion omits predictor weights | Implementation plus separate state/key/numerical qualification; image-request MTP remains separate |
| Qualify remaining ring/cache artifacts | Larger/quantized Gemma 3/4 | Physical snapshot implementation exists; each additional artifact still requires its own cache/lifecycle proof |
| Complete full-model image checks | BF16-source Qwen MoE; larger Gemma 3/4; retained mixed artifact; other precise quantizations | Qualification, with numerical repair only if a test demonstrates a defect |
| Reconcile stale pinned catalog rows | Official Gemma 3 4B/27B and Qwen3.5-35B-A3B legacy “planned” entries | Metadata fix; distinguish actual implemented profile from unqualified variants |
| Reconcile reference copy about retained images | Generic model-support mode table still summarizes fewer image paths than the implementation | Documentation fix; explicit retained and Gemma paths must remain visible |

## Pinned checkpoint inventory

The following appendix enumerates all 120 entries in the audited `catalog.json`: 110 built-ins and 10 extensions. Ninety-seven have an adapter entry (including two classifiers); 23 retain a null/planned entry. Those 23 are not automatically supported downloads. Some legacy statuses lag the newer family implementation, as marked below. “Conversion added” is an implementation claim; the catalog's generic runtime/qualification fields do not establish a current artifact replay.

The specialized native/retained paths and image-profile sizes above are included separately because the generic pinned list alone misses them. Additional discovered compatible fine-tunes remain conditional.

| Pinned repository | Size | Adapter / route | Image disposition | Current action |
| --- | --- | --- | --- | --- |
| [allenai/Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct) | 7B | `olmo3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [cartesia-ai/Llamba-1B](https://huggingface.co/cartesia-ai/Llamba-1B) | 1B | `llamba` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [cartesia-ai/Llamba-1B-4bit-mlx](https://huggingface.co/cartesia-ai/Llamba-1B-4bit-mlx) | 1B | `llamba` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | 1.5B | `qwen2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [empero-ai/Qwen3.8-4B-Distill](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill) | 4B | `qwen3_5` | Image family path if complete supported vision tensors/processor are retained | Select/qualify exact image artifact; see Qwen rows above |
| [EssentialAI/rnj-1-instruct](https://huggingface.co/EssentialAI/rnj-1-instruct) | 8B | `gemma3_text` | This is a text artifact; separate image route where compatible | Conversion added; runtime/checkpoint qualification remains separate |
| [fishaudio/s1-mini](https://huggingface.co/fishaudio/s1-mini) | 0.8B | `null / planned` | Exact entry image support not established | Pinned entry planned; do not advertise ready |
| [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it) | 270M | `gemma3_text` | This is a text artifact; separate image route where compatible | Conversion added; runtime/checkpoint qualification remains separate |
| [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) | 2B | `gemma2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | 1B | `gemma3_text` | This is a text artifact; separate image route where compatible | Conversion added; runtime/checkpoint qualification remains separate |
| [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) | 27B | `null / planned` | Separate BF16 image route implemented | Reconcile stale null/planned row; see exact-size qualification above |
| [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | 4B | `null / planned` | Separate BF16 image route implemented | Reconcile stale null/planned row; see exact-size qualification above |
| [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) | 12B | `gemma4_text` | Named BF16 image profile implemented | See size-specific evidence and remaining work above |
| [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | 26B-A4B | `gemma4_text` | Named BF16 image profile implemented | See size-specific evidence and remaining work above |
| [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | 31B | `gemma4_text` | Named BF16 image profile implemented | See size-specific evidence and remaining work above |
| [google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B) | 5B | `null / planned` | Family image route exists; exact base artifact unqualified | Inspect/register exact base profile; instruct proof does not transfer |
| [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | E2B | `gemma4_text` | Named BF16 image profile implemented | See size-specific evidence and remaining work above |
| [google/gemma-4-E4B](https://huggingface.co/google/gemma-4-E4B) | 8B | `null / planned` | Family image route exists; exact base artifact unqualified | Inspect/register exact base profile; instruct proof does not transfer |
| [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | E4B | `gemma4_text` | Named BF16 image profile implemented | See size-specific evidence and remaining work above |
| [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | 1.7B | `llama` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 3B | `smollm3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [ibm-granite/granite-3.1-2b-instruct](https://huggingface.co/ibm-granite/granite-3.1-2b-instruct) | 2B | `granite` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [ibm-granite/granite-3.1-8b-instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct) | 8B | `granite` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [ibm-granite/granite-3.3-2b-instruct](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) | 2B | `granite` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [ibm-granite/granite-3.3-8b-instruct](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct) | 8B | `granite` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Jundot/Qwen3.8-27B-oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp) | 27B | `qwen3_5` | Image family path if complete supported vision tensors/processor are retained | Select/qualify exact image artifact; see Qwen rows above |
| [LiquidAI/LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B) | 1.2B | `lfm2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [LiquidAI/LFM2-2.6B](https://huggingface.co/LiquidAI/LFM2-2.6B) | 2.6B | `lfm2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [LiquidAI/LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) | 350M | `lfm2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [LiquidAI/LFM2-700M](https://huggingface.co/LiquidAI/LFM2-700M) | 700M | `lfm2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) | 1.2B | `null / planned` | Exact entry image support not established | Pinned entry planned; do not advertise ready |
| [LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit) | 1.2B | `null / planned` | Exact entry image support not established | Pinned entry planned; do not advertise ready |
| [LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit) | 1.2B | `null / planned` | Exact entry image support not established | Pinned entry planned; do not advertise ready |
| [LiquidAI/LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) | 1.2B | `null / planned` | Exact entry image support not established | Pinned entry planned; do not advertise ready |
| [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) | 350M | `null / planned` | Exact entry image support not established | Pinned entry planned; do not advertise ready |
| [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 8B | `llama` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 1B | `llama` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 3B | `llama` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | 30B | `null / planned` | Exact entry image support not established | Pinned entry planned; do not advertise ready |
| [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | 3.8B | `phi3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mistralai/Codestral-22B-v0.1](https://huggingface.co/mistralai/Codestral-22B-v0.1) | 22B | `mistral` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mistralai/Devstral-Small-2505](https://huggingface.co/mistralai/Devstral-Small-2505) | 24B | `mistral` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) | 7B | `mistral` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | 7B | `mistral` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mlx-community/gemma-3-1b-it-4bit](https://huggingface.co/mlx-community/gemma-3-1b-it-4bit) | 1B | `gemma3_text` | This is a text artifact; separate image route where compatible | Conversion added; runtime/checkpoint qualification remains separate |
| [mlx-community/gemma-3-1b-it-8bit](https://huggingface.co/mlx-community/gemma-3-1b-it-8bit) | 1B | `gemma3_text` | This is a text artifact; separate image route where compatible | Conversion added; runtime/checkpoint qualification remains separate |
| [mlx-community/gemma-3-27b-it-4bit](https://huggingface.co/mlx-community/gemma-3-27b-it-4bit) | 27B | `null / planned` | Exact entry image support not established | Pinned entry planned; do not advertise ready |
| [mlx-community/gemma-3-27b-it-8bit](https://huggingface.co/mlx-community/gemma-3-27b-it-8bit) | 27B | `null / planned` | Exact entry image support not established | Pinned entry planned; do not advertise ready |
| [mlx-community/gemma-3-4b-it-4bit](https://huggingface.co/mlx-community/gemma-3-4b-it-4bit) | 4B | `null / planned` | Exact entry image support not established | Pinned entry planned; do not advertise ready |
| [mlx-community/gemma-3-4b-it-8bit](https://huggingface.co/mlx-community/gemma-3-4b-it-8bit) | 4B | `null / planned` | Exact entry image support not established | Pinned entry planned; do not advertise ready |
| [mlx-community/LFM2-1.2B-4bit](https://huggingface.co/mlx-community/LFM2-1.2B-4bit) | 1.2B | `lfm2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mlx-community/LFM2-1.2B-8bit](https://huggingface.co/mlx-community/LFM2-1.2B-8bit) | 1.2B | `lfm2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mlx-community/LFM2-2.6B-4bit](https://huggingface.co/mlx-community/LFM2-2.6B-4bit) | 2.6B | `lfm2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mlx-community/LFM2-2.6B-8bit](https://huggingface.co/mlx-community/LFM2-2.6B-8bit) | 2.6B | `lfm2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mlx-community/LFM2-350M-4bit](https://huggingface.co/mlx-community/LFM2-350M-4bit) | 350M | `lfm2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mlx-community/LFM2-350M-8bit](https://huggingface.co/mlx-community/LFM2-350M-8bit) | 350M | `lfm2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mlx-community/LFM2-700M-4bit](https://huggingface.co/mlx-community/LFM2-700M-4bit) | 700M | `lfm2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mlx-community/LFM2-700M-8bit](https://huggingface.co/mlx-community/LFM2-700M-8bit) | 700M | `lfm2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mlx-community/LFM2.5-1.2B-Thinking-4bit](https://huggingface.co/mlx-community/LFM2.5-1.2B-Thinking-4bit) | 1.2B | `null / planned` | Exact entry image support not established | Pinned entry planned; do not advertise ready |
| [mlx-community/LFM2.5-1.2B-Thinking-8bit](https://huggingface.co/mlx-community/LFM2.5-1.2B-Thinking-8bit) | 1.2B | `null / planned` | Exact entry image support not established | Pinned entry planned; do not advertise ready |
| [mlx-community/Llama-3.1-8B-Instruct-4bit](https://huggingface.co/mlx-community/Llama-3.1-8B-Instruct-4bit) | 8B | `llama` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mlx-community/Llama-3.2-1B-Instruct-4bit](https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-4bit) | 1B | `llama` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mlx-community/Llama-3.2-1B-Instruct-8bit](https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-8bit) | 1B | `llama` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mlx-community/Llama-3.2-3B-Instruct-4bit](https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit) | 3B | `llama` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mlx-community/Llama-3.2-3B-Instruct-8bit](https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-8bit) | 3B | `llama` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mlx-community/Qwen3.5-0.8B-MLX-4bit](https://huggingface.co/mlx-community/Qwen3.5-0.8B-MLX-4bit) | 0.8B | `qwen3_5` | Image family path if complete supported vision tensors/processor are retained | Select/qualify exact image artifact; see Qwen rows above |
| [mlx-community/Qwen3.5-0.8B-MLX-8bit](https://huggingface.co/mlx-community/Qwen3.5-0.8B-MLX-8bit) | 0.8B | `qwen3_5` | Image family path if complete supported vision tensors/processor are retained | Select/qualify exact image artifact; see Qwen rows above |
| [mlx-community/Qwen3.5-27B-4bit](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) | 27B | `qwen3_5` | Image family path if complete supported vision tensors/processor are retained | Select/qualify exact image artifact; see Qwen rows above |
| [mlx-community/Qwen3.5-27B-8bit](https://huggingface.co/mlx-community/Qwen3.5-27B-8bit) | 27B | `qwen3_5` | Image family path if complete supported vision tensors/processor are retained | Select/qualify exact image artifact; see Qwen rows above |
| [mlx-community/Qwen3.5-2B-MLX-4bit](https://huggingface.co/mlx-community/Qwen3.5-2B-MLX-4bit) | 2B | `qwen3_5` | Image family path if complete supported vision tensors/processor are retained | Select/qualify exact image artifact; see Qwen rows above |
| [mlx-community/Qwen3.5-2B-MLX-8bit](https://huggingface.co/mlx-community/Qwen3.5-2B-MLX-8bit) | 2B | `qwen3_5` | Image family path if complete supported vision tensors/processor are retained | Select/qualify exact image artifact; see Qwen rows above |
| [mlx-community/Qwen3.5-4B-MLX-4bit](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit) | 4B | `qwen3_5` | Image family path if complete supported vision tensors/processor are retained | Select/qualify exact image artifact; see Qwen rows above |
| [mlx-community/Qwen3.5-4B-MLX-8bit](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-8bit) | 4B | `qwen3_5` | Image family path if complete supported vision tensors/processor are retained | Select/qualify exact image artifact; see Qwen rows above |
| [mlx-community/Qwen3.5-9B-MLX-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit) | 9B | `qwen3_5` | Image family path if complete supported vision tensors/processor are retained | Select/qualify exact image artifact; see Qwen rows above |
| [mlx-community/Qwen3.5-9B-MLX-8bit](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-8bit) | 9B | `qwen3_5` | Image family path if complete supported vision tensors/processor are retained | Select/qualify exact image artifact; see Qwen rows above |
| [mlx-community/SmolLM3-3B-4bit](https://huggingface.co/mlx-community/SmolLM3-3B-4bit) | 3B | `smollm3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [mlx-community/SmolLM3-3B-8bit](https://huggingface.co/mlx-community/SmolLM3-3B-8bit) | 3B | `smollm3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Nanbeige/Nanbeige4.1-3B](https://huggingface.co/Nanbeige/Nanbeige4.1-3B) | 3B | `llama` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | 20B | `null / planned` | Exact entry image support not established | Pinned entry planned; do not advertise ready |
| [POLARIS-Project/Polaris-4B-Preview](https://huggingface.co/POLARIS-Project/Polaris-4B-Preview) | 4B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [prism-ml/Bonsai-4B-mlx-1bit](https://huggingface.co/prism-ml/Bonsai-4B-mlx-1bit) | 4B | `null / planned` | Exact entry image support not established | Pinned entry planned; do not advertise ready |
| [Qwen/Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) | 0.5B | `qwen2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | 1.5B | `qwen2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) | 14B | `qwen2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) | 32B | `qwen2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) | 3B | `qwen2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 7B | `qwen2` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 0.6B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-0.6B-MLX-4bit](https://huggingface.co/Qwen/Qwen3-0.6B-MLX-4bit) | 0.6B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-0.6B-MLX-8bit](https://huggingface.co/Qwen/Qwen3-0.6B-MLX-8bit) | 0.6B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 1.7B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-1.7B-MLX-4bit](https://huggingface.co/Qwen/Qwen3-1.7B-MLX-4bit) | 1.7B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-1.7B-MLX-8bit](https://huggingface.co/Qwen/Qwen3-1.7B-MLX-8bit) | 1.7B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | 14B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-14B-MLX-4bit](https://huggingface.co/Qwen/Qwen3-14B-MLX-4bit) | 14B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-14B-MLX-8bit](https://huggingface.co/Qwen/Qwen3-14B-MLX-8bit) | 14B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 32B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-32B-MLX-4bit](https://huggingface.co/Qwen/Qwen3-32B-MLX-4bit) | 32B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-32B-MLX-8bit](https://huggingface.co/Qwen/Qwen3-32B-MLX-8bit) | 32B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 4B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 4B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-4B-MLX-4bit](https://huggingface.co/Qwen/Qwen3-4B-MLX-4bit) | 4B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-4B-MLX-8bit](https://huggingface.co/Qwen/Qwen3-4B-MLX-8bit) | 4B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) | 4B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 8B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-8B-MLX-4bit](https://huggingface.co/Qwen/Qwen3-8B-MLX-4bit) | 8B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3-8B-MLX-8bit](https://huggingface.co/Qwen/Qwen3-8B-MLX-8bit) | 8B | `qwen3` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 0.8B | `qwen3_5` | Image family path if complete supported vision tensors/processor are retained | Select/qualify exact image artifact; see Qwen rows above |
| [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | 27B | `qwen3_5` | Image family path if complete supported vision tensors/processor are retained | Select/qualify exact image artifact; see Qwen rows above |
| [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 2B | `qwen3_5` | Image family path if complete supported vision tensors/processor are retained | Select/qualify exact image artifact; see Qwen rows above |
| [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | 35B | `null / planned` | Separate MoE image profiles implemented | Reconcile legacy row; BF16 full-model qualification pending |
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 4B | `qwen3_5` | Image family path if complete supported vision tensors/processor are retained | Select/qualify exact image artifact; see Qwen rows above |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9B | `qwen3_5` | Image family path if complete supported vision tensors/processor are retained | Select/qualify exact image artifact; see Qwen rows above |
| [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | 27B | `null / planned` | Exact entry image support not established | Pinned entry planned; do not advertise ready |
| [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 35B | `null / planned` | Exact entry image support not established | Pinned entry planned; do not advertise ready |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B | `null / planned` | Exact entry image support not established | Pinned entry planned; do not advertise ready |
| [RekaAI/reka-flash-3.1](https://huggingface.co/RekaAI/reka-flash-3.1) | 21B | `llama` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |
| [trymirai/chat-moderation-router](https://huggingface.co/trymirai/chat-moderation-router) | 0.15B | `modernbert` | No app image workflow | Classifier conversion only; no chat workflow |
| [trymirai/health-related-router](https://huggingface.co/trymirai/health-related-router) | 0.15B | `modernbert` | No app image workflow | Classifier conversion only; no chat workflow |
| [unsloth/Llama-3.2-1B-Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) | 1B | `llama` | No image route in this text adapter | Conversion added; runtime/checkpoint qualification remains separate |

## Audited sources

The family table covers 20 registrations; the appendix covers every pinned entry
exactly once. Null adapter rows remain visible to expose stale metadata and
planned sources rather than silently counting them as implemented support.
No source-model modality is inferred solely from a brand or repository name.

[registry]: https://github.com/jshahbazi/stark/blob/2c1e75b1/tools/converter/stark_convert/model_support.py
[native-registry]: https://github.com/jshahbazi/stark/blob/2c1e75b1/tools/converter/stark_convert/native_catalog.py
[catalog]: https://github.com/jshahbazi/stark/blob/2c1e75b1/tools/converter/stark_convert/catalog.json
[qwen]: https://github.com/jshahbazi/stark/blob/2c1e75b1/docs/model-experiments/2026-11-08-qwen-images.md
[gemma]: https://github.com/jshahbazi/stark/blob/2c1e75b1/docs/model-experiments/2026-11-08-gemma-images.md
[vision]: https://github.com/jshahbazi/stark/blob/2c1e75b1/docs/VISION_FAMILY_CONTRACT.md
[retained]: https://github.com/jshahbazi/stark/blob/2c1e75b1/docs/NATIVE_HYBRID_MODELS.md#retained-lily-image-artifacts
[native]: https://github.com/jshahbazi/stark/blob/2c1e75b1/docs/NATIVE_HYBRID_MODELS.md

Pinned appendix source: [catalog.json][catalog]. The [image integration review](https://github.com/jshahbazi/stark/blob/2c1e75b1/docs/archive/runs/2026-11-07-image-family-pr-review.md) and [retrospective](https://github.com/jshahbazi/stark/blob/2c1e75b1/docs/model-experiments/2026-11-08-image-support-retrospective.md) preserve the scope and remaining failures. Prefix implementation and its named artifact proof are in [PR #110](https://github.com/jshahbazi/stark/pull/110).

[gemma-next]: model-experiments/2026-11-09-gemma-next-image-parity.md
