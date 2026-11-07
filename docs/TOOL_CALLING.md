# Tool-calling coverage

Stark supports tools through the exact loaded chat template and its reviewed
parser. Architecture conversion alone does not enable tools. A supported wire
format establishes that Stark can render definitions/history and parse calls;
it does not prove that every checkpoint using it chooses tools reliably.

## Current implementation

| Supported family or path | Tool protocol in Stark | Boundary |
| --- | --- | --- |
| Qwen2 dense, including matching Qwen2.5 Instruct/Coder templates | Hermes JSON, `qwen3-hermes-v1` | The reviewed Qwen2.5 program is admitted; other Qwen2 templates need review. Model-bound qualification of the newly admitted program is pending. |
| Qwen3 dense | Hermes JSON, `qwen3-hermes-v1` | The reviewed 0.6B and MLX 4B programs and syntax-equivalent programs are admitted. |
| Qwen3.5 dense and compatible Qwen3.8 dense/distill artifacts, including JevK5 | Qwen XML, `qwen35-xml-v1` | Requires the reviewed program; this includes the matching official Qwen3.5-4B template. Tools cannot be combined with images in one request. |
| Qwen3.8 Flash Next, Lily and matching PipeNetwork mixed export | Qwen XML, `qwen38-xml-v1` | Only matching reviewed programs; quantization and backend qualifications stay separate. |
| Native GLM-5.3 Flash | GLM argument-key/value XML, `glm53-xml-v1` | The pinned native-preparation template is admitted. Synthetic rendering, parser and API tests pass; model-generated workflow qualification is pending. |
| Native Flint INT4 | GLM argument-key/value XML, `glm53-xml-v1` | The pinned Flint template shares the grammar; native rendering/history tests pass. Model-generated workflow qualification is pending. |
| Mistral-7B-Instruct-v0.3 | JSON list after `[TOOL_CALLS]`, `mistral-json-v1` | Reviewed serializer correction and render-only nine-character ID mapping. Public API IDs are preserved. Model-generated qualification is pending. |
| Llama 3.1 / 3.2 Instruct matching reviewed templates | JSON object, `llama-json-v1` | One call per assistant turn; mixed call/text history is rejected. Source templates were obtained from pinned public mirrors; qualification does not transfer between artifacts. |
| Granite 3.0 / 3.1 / 3.3 Instruct | JSON list, `granite30-json-v1` / `granite33-json-v1` | Native call/result roles and content are adapted per template. Model-generated qualification is pending. |
| LFM2 and matching LFM2.5 Instruct | Bounded Pythonic calls, `lfm2-pythonic-v1` | Syntax is parsed as data without execution. Tool-end markers inside strings remain data; documented trailing prose is preserved. LFM2-350M passed narrow call/argument and result-use checks; the exact ordinary-chat check failed. LFM2.5-350M remains a planned conversion route. |
| SmolLM2-1.7B-Instruct | JSON list in `<tool_call>`, `smollm2-json-v1` | Adds upstream's published function-calling instructions for tool requests; an empty list means no call. Model-generated qualification is pending. |
| SmolLM3-3B | Hermes JSON, `smollm3-hermes-v1` | Uses native function definitions and content-based history. Python-code mode is unavailable. Native template rendering passes; model-generated qualification is pending. |
| Phi-4-mini-instruct | JSON list, `phi4-json-v1` | Uses the native system-message tools field and call/result framing. This does not enable tools for Phi-3. Model-generated qualification is pending. |
| rnj-1-instruct | Hermes JSON, `qwen3-hermes-v1` | Reviewed rnj-1 program admitted; model-generated qualification is pending. |
| FunctionGemma-270M matching reviewed templates | Native function-call grammar, `functiongemma-v1` | Protocol tests pass. The tested BF16 public-mirror artifact fails native model loading, so native tool execution remains blocked for that artifact. Restricted concrete schemas and representable values are required. |
| Other Gemma 2 / Gemma 3 text, Phi-3, LLaMBA and compatible fine-tunes | Unavailable unless their actual template matches an admitted program | Assess each checkpoint's training and complete template. No family-wide tool capability is inferred. Some need a different checkpoint or training rather than an adapter. |
| ModernBERT classifier | Unavailable | Classifier conversion does not provide generative chat/tool execution. |

An unregistered template keeps tools disabled even if it emits familiar marker
strings. Conversely, a compatible fine-tune can inherit protocol compatibility
through the reviewed program without inheriting another checkpoint's task
qualification. This table covers implemented families; it is not a promise about
every repository discoverable in the catalogue.

## Behavioral evidence

Existing records establish narrow tool-result workflows for exact artifacts:

- Qwen3-0.6B BF16: [native buffered/streamed smoke](archive/runs/2026-11-01-tool-template-compatibility.md).
- MLX Qwen3-4B 4-bit: [bounded OpenWork task](archive/runs/OPENWORK_KEYCHAIN_E2E_2026_10_31.md).
- Qwen3.8 Flash Next Lily Q4-XL: [bounded OpenWork acceptance](archive/runs/2026-11-01-openwork-lily-acceptance.md).

These records retain their original model/build identities and limitations.
They do not qualify all models in those families. Newly admitted Qwen2.5, GLM
and Flint programs still require their own model-generated runs.

The [additional-format validation record](archive/runs/2026-11-07-additional-tool-protocols.md)
records native rendering, parser/API checks, the LFM2-350M partial behavioral
result and the FunctionGemma native-loading blocker. Protocol registration
alone leaves the Installed row's `model_behavior` as `unverified`.
FunctionGemma remains excluded from Guided Setup's general-assistant
recommendations; its task-specific tool-routing workflow requires separate
qualification. Manual import and protocol compatibility remain available.

## Import and adding support

Installed rows show **Tools available**, **Tools unavailable** or **Tools
unchecked**. The backend derives this from bounded reads of the artifact's actual
codec every time it describes the library, including locally imported and
previously installed versions. The install-completion event also includes the
assessment. Missing, unreadable or oversized metadata is unchecked; a readable
unregistered program is unavailable. Saved names and claimed capability flags
cannot enable the parser. These labels do not load a model or record a successful
behavioral test.

Follow the required [tool acceptance workflow](MODEL_ONBOARDING.md#assess-and-qualify-function-tools)
when adding/importing a new model for agent use. It includes upstream provenance,
native template/history tests, adversarial parser tests and a separate
artifact-bound smoke command. Unsupported tools do not prevent an otherwise
valid ordinary-chat import. Unknown templates are never rewritten automatically.
For the reviewed Mistral v0.3 program only, runtime rendering applies a packaged,
hash-checked JSON-serialization correction. The original imported source stays
unchanged; qualification records hash the original and effective renderer
separately. The [fixture provenance](../tools/converter/tests/fixtures/tool_templates/README.md)
records all additional source programs, licenses and this patch.

The exact registered sources, schemas, streaming rules and tool-choice contract
are in the [Function tools reference](LOCAL_API.md#function-tools).
The [2026-11-07 validation record](archive/runs/2026-11-07-tool-calling-coverage.md)
records the new registrations, tests and pending model-bound checks.
The [follow-up review](archive/runs/2026-11-07-tool-calling-review.md) records
corrections to GLM generation-prefix handling and qualification provenance.
