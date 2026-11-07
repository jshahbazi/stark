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
| Llama dense, Mistral dense, Granite dense, LFM2 | Unavailable for their unregistered native tool formats | Tool-trained checkpoints may gain support through their own reviewed renderer/history integration and parser. Existing Qwen parsers do not cover these formats. |
| Gemma 2, Gemma 3 text, Phi-3, SmolLM3, LLaMBA and other compatible fine-tunes | Unavailable unless their actual template matches an admitted program | Assess each checkpoint's training and complete template. No family-wide tool capability is inferred. Some need a different checkpoint or training rather than an adapter. |
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

The exact registered sources, schemas, streaming rules and tool-choice contract
are in the [Function tools reference](LOCAL_API.md#function-tools).
The [2026-11-07 validation record](archive/runs/2026-11-07-tool-calling-coverage.md)
records the new registrations, tests and pending model-bound checks.
The [follow-up review](archive/runs/2026-11-07-tool-calling-review.md) records
corrections to GLM generation-prefix handling and qualification provenance.
