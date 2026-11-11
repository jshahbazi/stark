# Stark local API

The menu bar application's bridge serves a loopback-only, OpenAI-style HTTP API so that other applications on the same Mac can chat with the loaded model. It supports buffered or streamed text completions and one-image requests through one loaded model. Images require passing host evidence for the [supported Qwen3.5 family](VISION_FAMILY_CONTRACT.md). The API binds to `127.0.0.1`, requires bearer authentication, and rejects browser-origin requests. Clients may use the persistent Keychain bearer key or, for the managed OpenWork and OpenCode Desktop integrations below, a short-lived per-client capability.

## Connection and authentication

Open **Connections** in Stark. **Apps** groups OpenWork and OpenCode Desktop with independent **Connect** buttons and status; **Manage** reveals each configured app's **Reconnect** and **Disconnect** actions. OpenCode's per-engine **Allow** and **Deny** prompt stays separate. The shared model header identifies the model currently running in Chat mode. API startup and credential-recovery messages remain visible when setup needs attention.

Select **Connect your own app** to see the base URL (for example `http://127.0.0.1:8000/v1`), the API key and the Model ID. The model picker inspects saved model IDs and the settings copied as JSON; it does not run a model or change the model published to managed apps. Use **Manage Models** and **Run** in Stark to serve a different model. **Advanced settings** contains the saved listening port, API retry, explicit key regeneration, JSON and curl export, and capability details. Manually configured native clients send the copied master key on every request, including health checks:

```text
Authorization: Bearer <key>
```

The Swift host loads a random 256-bit key from macOS Keychain and sends it to the backend through its private control pipe. It generates a key only when Keychain reports that the item is missing, and activates it only after storage succeeds. Concurrent first launches use the winning stored value. Relaunches, worker restarts, model changes and app upgrades reuse this credential. Secrets are absent from process arguments, environment, UserDefaults, configuration files and logs.

**Show** and **Copy** retrieve the existing key without rotating it. Key and connection-JSON copies retain the sensitive clipboard markers. **Regenerate API Key** explicitly replaces the stored credential after confirmation; clients configured with the copied master key need the replacement, while managed OpenWork and OpenCode capabilities refresh through Stark. A storage failure preserves the working key. Locked, denied, corrupt or unavailable Keychain access produces recovery instructions in Connections; authentication is never disabled. Unlock or allow access in Keychain Access and choose **Retry API**. A corrupt entry can be replaced using the explicit regeneration action. Do not save copied connection JSON in ordinary configuration files.

The installed, Team-ID-signed app uses service `com.starkengine.menubar.local-api`, account `bearer-key-v1`. Canonical unsigned/ad-hoc development builds use the separate `.development` service. Other bundle identifiers receive a separate development service suffix; native tests use disposable services. A signing identity or Keychain access-policy change can require access recovery. [Developer ID package upgrades](archive/runs/OPENWORK_KEYCHAIN_E2E_2026_10_31.md) reused a credential under an isolated bundle identifier; production-identifier upgrades and notarized distribution require their own validation.

Keychain operations run on a dedicated background queue, with at most one outstanding operation. If macOS waits for legacy Keychain access approval, Connections reports the pending operation while model management and chat remain usable. Retry, port changes and regeneration wait for that operation to finish. A pending regeneration is not reported as failed because macOS may still commit it. Approve access in Keychain Access; if it remains pending, quit and reopen Stark afterward. Rebuilt ad-hoc development apps can require renewed access approval; a stable service name does not bypass Keychain's signing checks.

The non-secret listening port is saved separately in UserDefaults, initially 8000, and reused. Stark never silently chooses another port. If it is occupied, Connections explains the conflict; close the other server or enter a port from 1 through 65535 and choose **Apply Port**. Update clients after an intentional port change. A failed port change keeps any previous listener active and displays its actual Base URL. Model management and the playground remain usable when the API cannot start. The curl example prompts without echo and passes authentication through stdin, without embedding a key in arguments or exporting it.

Request rules enforced before any endpoint runs:

- The `Host` header must name `127.0.0.1`, `localhost` or `::1`.
- Requests carrying an `Origin` header are rejected, so browser pages and webviews cannot use the API. `OPTIONS` preflight returns 501 because there is no CORS handler.
- Duplicate `Host` or `Authorization` headers are rejected.
- Text-only request bodies use the [automatic shared resource budget](WORKER_PROTOCOLS.md#resource-budget): 16–64 MiB of logical content and 98–386 MiB of encoded JSON, selected from available memory at startup. When `capabilities.images` is true, a request containing an image part has 6 MiB of additional body allowance; the compressed image remains limited to 4 MiB. Every request must carry exactly one `Content-Length` and must not use `Transfer-Encoding`. When `Content-Type` is present it must be `application/json`. JSON with duplicate keys, `NaN` or infinities is rejected. A structural scan rejects more than 65,536 nodes or 64 nested containers before allocating the parsed tree.

Responses carry `Cache-Control: no-store` and an `X-Request-Id` header, except the connection-limit response described under [limits](#limits-queueing-and-cancellation). Stark writes no access log and never logs prompts, replies or keys.

## Endpoints

| Endpoint | Behaviour |
| --- | --- |
| `GET /v1/health`, `GET /health` | Application status: whether a model is loaded, whether a request is running, queue depth and limits, the model list, and capability flags |
| `GET /v1/capabilities` | The same status document |
| `GET /v1/models` | The running model as a one-item list, or an empty list when no model is running in Chat mode |
| `GET /v1/models/{id}` | The matching running model, or 404 |
| `POST /v1/chat/completions` | One completion, buffered or streamed; one inline PNG/JPEG image when `capabilities.images` is true |

Each model entry has `id`, `object`, `created`, `owned_by`, `name`, `context_length`, `context_capacity_known`, `max_output_tokens`, `tool_calling`, `capabilities.mtp` and `defaults.mtp`. Embeddings, the Responses API, structured output and batching are not available. Function tools require the supported template/parser below.

When a supported template/parser produces function calls, Stark validates and returns them in the completion response. The API does not execute the named functions. The client decides whether to execute a returned call, applies its own approval policy, and sends any result back as conversation history. Treat returned calls and arguments as model output, not as trusted instructions.

Public model IDs use the full Hugging Face repository ID (`owner/model`) for models installed from Hugging Face, or the installed model name for local folders. The connection modal, model listing, completion responses and streamed chunks use these readable IDs. Model detail routes accept percent-encoded IDs, including encoded slashes and spaces. Private library IDs and managed model paths remain stable; existing installations require no registry migration. Only the running model is exposed, so installations sharing a public name refer to whichever copy is running.

## Image qualification

`capabilities.images` becomes true after the live worker passes one of two host evidence paths. The status document's `image_verification` identifies the result:

| Value | Meaning |
| --- | --- |
| `reference_verified` | The schema-2 `mediaref-v1` contract matches a passed external execution record for the exact artifact, worker and runtime. |
| `locally_tested` | The admitted family passed bounded local text/image, token-accounting, recoverable-error and cleanup checks, or reused an identical private receipt. |
| `unverified` | No passing evidence is active; images are disabled. |

The host checks all five loaded file hashes—weights, config, tokenizer, processor and template—plus the worker executable hash, Metal device properties and current macOS version. It hashes the executable before launch and again when readiness arrives. Local receipts also bind the checker, Python/dependency versions, actual family/schema versions, admitted vocabulary and image limits. Installation metadata cannot enable the capability. Native readiness reports `qualified: false` because the host supplies the evidence decision. See the [family contract and cache rules](VISION_FAMILY_CONTRACT.md).

The [reference catalog](../tools/converter/stark_convert/vision_runtime_qualifications.json) contains the developer `5fc92516…` and Developer ID signed `741a3ea7…` worker tuples for the same BF16 Qwen3.5 artifact on an Apple M5 Max with 40 GPU cores, Apple GPU family 1010 and MXU enabled, running macOS 26.7.1. Their records bind the [developer execution evidence](archive/runs/2026-10-26-qwen35-image-execution.md) and [signed-worker evidence](archive/runs/2026-10-27-qwen35-signed-worker.md). The [Swift and local signed-package replay](archive/runs/2026-10-27-qwen35-swift-packaged-app.md) covers that delivered GUI. Public notarized distribution remains pending. A rebuilt, signed or modified worker requires matching reference evidence or fresh local checks. `locally_tested` does not assert reference parity or visual accuracy for a fine-tune.

## Chat completions

Accepted request fields: `model`, `messages`, `stream`, `stream_options`, `max_tokens`, `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, `n`, `presence_penalty`, `frequency_penalty`, `logprobs`, `top_logprobs`, `response_format`, `user`, `metadata`, `mtp`, `tools`, `tool_choice`, `parallel_tool_calls` and `store`. Unknown fields are rejected.

**Storage.** Omit `store` or use `false`/`null`. Stark does not store completions. `store: true` and non-boolean values return `unsupported_parameter`; they do not enable retention.

**Model and messages.** `model` may be omitted, or may be the running model's public ID, installed name or previous private library ID; any other value returns 404 `model_not_found`. `messages` must be non-empty. Roles `system`, `developer` (normalised to `system`), `user`, `assistant` and `tool` are accepted (tool-result history requires a supported template/parser). Content is a string or an array of ordered parts. Text parts use `{"type":"text","text":"..."}`. When `capabilities.images` is true, one user message may also include `{"type":"image_url","image_url":{"url":"data:image/png;base64,...","detail":"auto"}}`; `detail` may be omitted or set to `auto`. The URL must be an inline canonical Base64 PNG or JPEG data URL; remote URLs and image parts on non-user roles are rejected. Only one image is accepted per request. Image bytes are passed to the worker separately from the rendered prompt, which receives only an image placeholder. An optional `name` of up to 64 ASCII characters is validated and ignored. Assistant messages may carry `reasoning_content` as a string, which is passed to the chat template; Stark 0.4.0 and earlier reject that field. Function calls use the contract below. Legacy `functions` and `function_call` fields are rejected. The model's chat template renders the conversation; a template that rejects the messages returns 400 `invalid_messages`.

The service accepts PNG or JPEG file bytes up to 4 MiB after Base64 decoding the data URL. This is a compressed file-size limit, not a decoded-pixel limit. The engine transport has a wider 20 MiB file limit, but the HTTP service applies its 4 MiB cap before staging. The parser does not decode pixels in Python; the native processor applies its source dimension, source pixel, processed pixel, patch and image-token limits. See the [request parser](../tools/converter/stark_convert/api_request.py), [image validator](../tools/converter/stark_convert/media.py) and [worker qualification gate](../tools/converter/stark_convert/vision_runtime.py).

**Context limit.** `/v1/models` reports a declared finite model maximum as `context_length`, with `context_capacity_known: true`. The combined rendered input and requested output must fit that context. A backend without a declared finite maximum reports `context_length: null` and `context_capacity_known: false`; its unsigned 32-bit protocol ceiling is not advertised as a model capacity. For these models, text requests reserve only the rendered input plus requested output, while any custom playground conversation budget still applies. Image requests require a declared finite context. There is no global 16,384-token context cap. Request byte limits and available memory still apply; a published context capacity does not establish full-length performance or memory qualification on this Mac.

**Output limit.** `max_tokens` and `max_completion_tokens` are aliases; supply at most one. When both are omitted, Stark tokenizes the rendered conversation and uses the remaining finite context as the output budget, bounded by the worker’s output-storage allowance. Generation stops on EOS, a stop condition, the context boundary, cancellation or a deadline. Explicit output limits are honored and must fit beside the actual input. `/v1/models` reports `max_output_tokens` as the smaller of context capacity minus one and the resource budget’s four-byte-per-token storage allowance. For models without a declared finite maximum, that storage allowance bounds the unsigned 32-bit protocol range. There is no separate 4,096-token output cap or 512-token default. Generated content remains subject to the shared resource budget. `n` must be 1. A completion that reaches the output or context limit reports `finish_reason: "length"`.

For a model with no declared finite context, omitted output limits use [native memory planning](WORKER_PROTOCOLS.md#automatic-execution-for-undeclared-contexts) after tokenization. The worker selects a finite execution allowance from the loaded model's cache/state geometry and current memory headroom. The model continues to report `context_length: null`; the planned allowance is a resource decision. Stark never reserves the unsigned 32-bit protocol ceiling. No context or output configuration is required. A worker predating memory planning reports recoverable 503 `automatic_output_unavailable` and needs updating; explicit output limits remain compatible with that worker.

**Sampling.** `temperature` defaults to 0, which selects greedy decoding. A positive finite temperature up to 2 selects stochastic sampling with `top_p` (greater than 0, at most 1), `top_k` (1 through the vocabulary size), `min_p` (0 through 1) and `seed` (unsigned 64-bit, default 0). These values are accepted without effect at temperature 0. Non-zero `presence_penalty` or `frequency_penalty`, `logprobs: true`, `top_logprobs` above 0, and any `response_format` other than `{"type":"text"}` return 400 `unsupported_parameter`; the neutral values are accepted. `user` and `metadata` are accepted within their size bounds and discarded.

**Stop sequences.** `stop` is one string or up to four non-empty strings of at most 1,024 characters. The worker matches them against decoded text, including across chunks, and withholds the match and everything after it. Usage counts still include the token that completed the match.

**Streaming.** With `stream: true` the response is Server-Sent Events of `chat.completion.chunk` objects followed by `data: [DONE]`. The first chunk carries the assistant role. `stream_options: {"include_usage": true}` adds a final usage-only chunk. Validation errors keep their HTTP status because they occur before streaming starts; a failure during streaming emits an error object in the stream and closes without `[DONE]`.

**Usage.** `usage` reports `prompt_tokens`, `completion_tokens` and `total_tokens`. When the running worker supports [prefix reuse](#prefix-reuse), it also reports `prompt_tokens_details.cached_tokens`: the prompt tokens served from the retained prefix of the previous request instead of being processed again. `prompt_tokens` always counts the whole prompt.

## MTP capabilities and per-model preferences

The live worker supplies `capabilities.mtp` in `/v1/capabilities` and each
running `/v1/models` entry. `supported` determines whether a caller can offer
an MTP toggle. Support is established by the loaded backend, including its
admitted head and implementation. Model names and saved preferences do not
enable support.

```json
{"supported":true,"max_drafts":3,"recommended_drafts":2,"default_drafts":0,"sampling":"exact-rejection-v1","verification":"batched-affine-decode-exact-v1","operations":["generate"]}
```

Each backend chooses its maximum and recommended draft count. Unsupported or
missing worker capabilities produce `supported: false`, zero counts and an
empty operation list. Capabilities are cleared when the worker stops or the
model changes; Decision mode does not expose MTP. The app shows a
**Multi-token prediction (MTP)** toggle under **Advanced Options** when the
loaded model supports it. **Apply and New Chat** saves the choice with the other
options; **Restore Defaults** selects MTP on.

`defaults.mtp` is the active model's effective saved choice. Registered models
default to on when the loaded worker supports MTP. Each model remembers its own
boolean preference in the private model registry across stops, switches and
application restarts. Existing registry entries without a saved choice also
default to on; an explicitly saved off choice stays off. The preference becomes
inactive when the loaded worker lacks support and is retained for a later
supported worker. The preference-only command below preserves the chat; applying
Advanced Options starts a new chat without reloading the model.

A completion may send `"mtp": true` or `"mtp": false` to override that default
for one request. `true` selects the model's recommended count. Omission
inherits the saved choice; overrides do not change it. `null` and non-booleans
return 400 `invalid_request`. Unsupported enable requests return 400
`mtp_not_supported` before worker dispatch and keep the model loaded. This
also applies to streaming. Selection is snapshotted when a request joins the
queue, and model changes still fence queued requests.

Current MTP operations are text-only. Image requests default to MTP off,
preserving the model's text preference; an explicit `mtp: true` with an image
returns 400 `mtp_not_supported`. Exact sampling and stored quantization remain
unchanged. A supported worker's completion acknowledgement must match its
selected mode, depth and policy; a contradiction closes that worker.

The desktop's private JSON-lines backend accepts the following command to save
the preference, using the internal `model_id` from its `ready` event:

```json
{"command":"configure_mtp","model_id":"registered-model-id","enabled":true}
```

It requires the same loaded model in idle Chat mode, validates live support,
and atomically persists only its preference. Stale model IDs, malformed
options, running/queued operations and save failures leave the setting
unchanged. The `mtp_configured` acknowledgement includes `model_id`, `mtp`
capabilities and `mtp_enabled`. Desktop `ready` events expose those settings
alongside the existing limits. These private commands use library IDs;
HTTP completions continue using the public IDs described above.

The app's `configure` command can include both `model_id` and boolean
`mtp_enabled` to save MTP with the other Advanced Options before starting a new
chat. It validates the loaded model, live support and all options before saving;
a save failure leaves the preference, conversation and other options unchanged.
Omitting both fields preserves the MTP preference.

## Function tools

`/v1/capabilities` reports `tool_calling`, `tool_parser` and `tool_choices` for the running model. Tool support means that the loaded chat template has an implemented, tested wire-format parser. Stark recognizes the reviewed originals below by source SHA-256 and also accepts syntax-equivalent Jinja programs, independently of the model name or saved installation metadata:

| Parser | Supported template sources | Template SHA-256 |
| --- | --- | --- |
| `qwen3-hermes-v1` | Qwen/Qwen2.5-0.5B-Instruct, revision `7ae557604adf67be50417f59c2c2f167def9a775`; matching program in reviewed Qwen2.5-Coder-3B-Instruct | `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f` |
| `qwen3-hermes-v1` | Official Qwen3-0.6B | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| `qwen3-hermes-v1` | `mlx-community/Qwen3-4B-4bit`, revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` | `87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5` |
| `qwen38-xml-v1` | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83`; identical template in installed `pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit` | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| `qwen35-xml-v1` | `alibiserikbay/JevK5`, revision `c4f7fdb3aeab5582336406e78d3bef11bf98833d` | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| `glm53-xml-v1` | `zai-org/GLM-5.3-Flash`, revision `04c4e9e95c5da8862dced7e5056455116f83a7e0`, used by native preparation | `34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891` |
| `glm53-xml-v1` | `RecursiveSI/flint-int4`, revision `3b93f9053b530488405ae9fca51d4900d6ce21f0` | `7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466` |

Additional reviewed sources use the protocols below. Full pinned revisions,
original and normalized hashes, program fingerprints and licenses are in the
[source fixture manifest](../tools/converter/tests/fixtures/tool_templates/README.md).

| Parser | Reviewed sources | Framing |
| --- | --- | --- |
| `mistral-json-v1` | Mistral-7B-Instruct-v0.3 | `[TOOL_CALLS]` followed by a JSON list |
| `llama-json-v1` | Llama-3.1-8B-Instruct and Llama-3.2-1B-Instruct public-mirror templates | Bare JSON object with `name` and `parameters`; optional `<\|python_tag\|>` |
| `granite30-json-v1` / `granite33-json-v1` | Granite 3.0, 3.1 and 3.3 Instruct | `<\|tool_call\|>` followed by a JSON list |
| `smollm2-json-v1` | SmolLM2-1.7B-Instruct | JSON list inside `<tool_call>`; empty list means no call |
| `smollm3-hermes-v1` | SmolLM3-3B | Hermes objects inside `<tool_call>` |
| `lfm2-pythonic-v1` | LFM2-1.2B and LFM2.5-1.2B-Instruct / 350M template programs | `<\|tool_call_start\|>[name(key=value)]<\|tool_call_end\|>` |
| `phi4-json-v1` | Phi-4-mini-instruct | Bare JSON list or `<\|tool_call\|>` / `<\|/tool_call\|>` |
| `functiongemma-v1` | FunctionGemma-270M public-mirror templates | `<start_function_call>call:name{key:<escape>value<escape>}<end_function_call>` |
| `qwen3-hermes-v1` | rnj-1-instruct | Hermes objects inside `<tool_call>` |

The [coverage matrix](TOOL_CALLING.md) lists family boundaries and separately
recorded task evidence. Newly admitted Qwen2.5, GLM and Flint programs have native
template, parser and synthetic API coverage; model-generated workflow
qualification is pending. Their unchanged upstream fixtures and licenses are
recorded in [fixture provenance](../tools/converter/tests/fixtures/tool_template_additions.provenance.md).

The compatibility check parses templates using the same locked Jinja version, immutable sandbox and whitespace settings as chat rendering. A versioned program fingerprint ignores source locations, comments, equivalent string quoting/escapes, redundant parentheses and expression spacing. Adjacent literal text split by a comment is joined without changing its contents. Newline/indentation changes are accepted only when Jinja produces the same program. Literal output, tool instructions/markers, emitted whitespace, values, lookup style, filters, calls, branches, loops and their ordering remain part of the fingerprint. Changed instructions or a new tool format still require review and an updated registration or parser; model names and marker presence never establish support. Matching does not compile or execute the candidate. Sources are limited to 256 KiB of UTF-8, with bounded traversal work/depth and an eight-entry in-memory template cache; malformed or over-limit candidates remain unsupported.

The 4B template is preserved as an upstream fixture with source and license provenance under `tools/converter/tests/fixtures/`. XML paths use synthetic parser fixtures and separately recorded rendering checks against the installed templates. Separate model-loading and vision-artifact integrity/evidence checks still apply: recognizing a compatible tool program does not authorize a changed vision artifact to load. Other templates keep ordinary chat and reject tool definitions/history with `tools_unsupported`. A matching program establishes protocol support; reliable tool selection, argument quality and task completion require separate model-bound evidence. Models sharing a supported program are not automatically qualified for agent tasks. See the [template compatibility validation record](archive/runs/2026-11-01-tool-template-compatibility.md).

- `tools` accepts up to 64 function definitions with unique ASCII names, optional descriptions, and object parameter schemas. Schemas are model hints; schema-constrained generation is unavailable and `strict: true` is rejected. The client must validate execution arguments against its schema.
- `tool_choice` accepts `auto`, `none`, `required`, or `{"type":"function","function":{"name":"supplied_name"}}`. Named choice renders only that definition. Required/named output is checked after generation; failure returns `tool_choice_not_satisfied`, without synthesizing a call. `none` omits definitions from rendering and rejects generated tool markers.
- `parallel_tool_calls` is a boolean, default true. False permits at most one generated call; a violation fails the completion. At most 16 calls are returned, in generation order, with distinct stable IDs within the response and JSON-object argument strings.
- Assistant history may contain null or omitted text alongside `tool_calls`. Each call requires `id`, `type: function`, and a function name/argument string. Tool-result messages require string/text-part content and the matching `tool_call_id`. All outstanding results must be returned before another assistant/user turn. Duplicate, orphaned and incomplete transactions are rejected. The loaded template receives complete history, argument objects, definitions, and results ordered by their call IDs.
- Hermes calls contain JSON objects inside `<tool_call>`. Qwen XML calls contain `<function=NAME>` and named `<parameter=NAME>` blocks. GLM calls place the function name directly after `<tool_call>`, followed by paired `<arg_key>`/`<arg_value>` blocks; strings retain exact contents and non-string values use JSON. Only complete calls from a normally terminated generation become executable requests. Duplicate keys/parameters, nonfinite numbers, unknown names, invalid argument representations, truncated markers, quoted examples, extra trailing text and limit violations fail. Code fences, blockquotes, HTML comments and HTML code/quotation containers in the visible prefix are rejected conservatively, including completed blocks. Balanced inline-code mentions and completed thinking before an actual call remain supported. No guessed repairs occur. Malformed output does not unload an otherwise aligned worker.
- XML schemas require directly named root properties. Parameters can declare types, constrain types through JSON `enum`/`const` literals, or use typed `anyOf`/`oneOf` alternatives or bounded local JSON Pointer references. Parameter names use the same bounded ASCII identifier syntax as function names. Root references/composition, dynamic parameter names, malformed schemas, remote/cyclic/unresolved references and resolution-budget exhaustion return `tool_schema_unsupported` before generation, with a fixed unsupported-form category.
- Empty or annotation-only parameter schemas are forwarded unchanged as model hints, so an unused unconstrained parameter does not block ordinary chat or other tool calls. XML cannot determine the representation of a generated unconstrained value: using such a parameter fails the entire completion with `tool_argument_type_unsupported`, without returning any calls. Types are never guessed from generated text.
- Mistral v0.3 rendering applies a hash-checked packaged correction to its upstream JSON serializer. Render-only IDs use nine alphanumeric characters and results are JSON-escaped; API IDs retain their original values. Llama permits one call per assistant turn. Mistral, Llama and Granite reject history combining visible text with a call when the native template cannot retain both, using `tool_history_unsupported` before queue admission.
- LFM Pythonic syntax is parsed as bounded data and never executed. Attribute access, nested calls, positional arguments, unpacking and executable expressions are rejected. Function and root parameter names must be Python identifiers. Its documented trailing prose is returned as text; other formats require clean call framing. Llama holds a possible initial JSON object during streaming; Phi holds a possible initial JSON list.
- FunctionGemma requires concrete parameter types and representable named keys. References, schema composition, dynamic properties, reserved schema-key parameter names and native delimiter strings are rejected with `tool_schema_unsupported` or `tool_history_unsupported`. Its required response marker is tokenized to one native stop-token ID for tool requests, preserving the allowance of four client stop strings. An incompatible tokenizer returns `tool_tokenizer_unsupported`. Protocol compatibility does not establish native model loading: the tested BF16 artifact currently fails that step.
- XML strings retain their original contents; structured values use JSON. When a parameter's schema permits a boolean, all XML parsers accept top-level `true` and `false` in any capitalization, with optional surrounding whitespace, and return JSON booleans. This includes `True`, `FALSE` and mixed-case spellings. String parameters retain these spellings as text. Nested object/array values still require strict JSON, and the Qwen3.5 template additionally supports its top-level `None` null representation. Ambiguous string/non-string union values and strings containing reserved wire delimiters are rejected rather than coerced. These checks decode types; clients still validate the complete schema before executing anything.
- XML value rejections retain HTTP 502 and `invalid_tool_call`, with a fixed diagnostic tag in the error message: `[xml_value_json]` means a non-string value failed strict JSON decoding (including duplicate keys); `[xml_value_type]` means the decoded type is not allowed; `[xml_value_ambiguous]` means both a raw string and a non-string interpretation fit. The ambiguity tag identifies a wire-format limitation and does not establish model error. Clients may display the tag in buffered or streamed errors. Other failures retain their existing reasons.
- These three categories are also recorded in `~/Library/Logs/Stark/tool-diagnostics.jsonl`. Each JSON line contains only `timestamp` (UTC) and `category` (one of the three fixed tags, without brackets). No prompt, generated text, tool name, argument name, value or exception is recorded. The private file uses mode `0600` and resets before exceeding 64 KiB, retaining the newest record. Logging is best effort: unavailable storage, unsafe file paths or file-lock contention do not change tool-call rejection or expose exception details. Synthetic tests use isolated temporary log directories.
- This temporary diagnostic branch additionally captures rejected XML argument values at the user's explicit request in `~/Library/Logs/Stark/tool-arguments.jsonl`. It records timestamp, category, parser, expected types, the rejected value (up to 1,048,576 characters), and a truncation flag. It does not capture full responses, prompts, conversation history, tool names or argument names. The separate private file uses mode `0600` and resets before exceeding 8 MiB. This branch's argument capture is intended for local diagnosis and must be removed after the investigation; captured files stay private and outside Git.
- Definitions together, each argument object and parsed tool output use the shared logical byte budget, selected automatically from 16 through 64 MiB. Tool results use the aggregate message-text budget without a separate per-result cap. Tool JSON depth remains limited to 32 and item count to 16,384. The shared encoded-body/worker-frame budgets and actual model context also apply. Tool use and image input cannot be combined.

Buffered completions return `message.tool_calls`, optional text (null when absent), and `finish_reason: "tool_calls"`. Tool-enabled streaming delivers ordinary text as it is generated. Possible tool markers and their arguments stay private through the terminal worker frame and strict parsing, then indexed `delta.tool_calls` are emitted exactly once, followed by the terminal reason, optional usage and `[DONE]`. Trailing whitespace is held until it can be distinguished from the separator before a call. Invalid or truncated calls produce a stream error after any provisional text; they never produce executable tool deltas. Usage includes all generated framing/argument tokens and tokens completing a stop sequence. Cancellation/deadline checks remain active throughout generation.

An `invalid_tool_call` message ends with one fixed reason naming the rule that rejected the reply, for example "the call names a tool that this request does not offer or allow" or "text before the call contains a code block, quotation or unclosed reasoning". The reasons cover:

- generation stopping at the output or context limit, or ending abnormally
- tool calls not allowed for the request
- call markup outside a complete call
- a code block, quotation or unclosed reasoning before the call
- a call that breaks the model's format, or text after it
- a tool the request does not offer or allow
- an undefined or repeated argument, or a value that does not match its schema
- arguments over the size or nesting limits
- more calls than the request allows

Reasons never include generated text, tool or argument names, or values. Clients should act on `code` rather than the message text.

An agent client can still take longer to produce its first token than a short Playground chat. Its system instructions, tool schemas and conversation history are part of the model input and must be processed before decoding begins. Live text streaming removes the additional wait for the entire answer; it does not shorten the client's instructions. [Prefix reuse](#prefix-reuse) removes most of the repeated work on later steps of the same conversation, but the first request of a conversation is still processed in full. The [OpenWork first-text latency record](archive/runs/2026-10-31-openwork-first-text-latency.md) separates native first-token work from the rest of a measured client request.

Qwen/Qwen3-0.6B with unquantized BF16 weights completed a narrowly prompted two-tool result-use task locally. Looser variants failed through an unwanted parallel call or premature stopping; this checkpoint is not a general-purpose agent recommendation. Protocol fixtures and task-competence evidence are recorded separately.

The pinned MLX Qwen3-4B 4-bit/group-64 artifact completed a bounded two-read task in OpenWork, including client-side execution, result return and a correct final calculation. A looser variant repeated a read until cancelled. Use explicit task limits and client approvals; this is a narrow workflow qualification. The [desktop validation record](archive/runs/OPENWORK_KEYCHAIN_E2E_2026_10_31.md) separates protocol, model behavior and signed-package evidence.

When the rendered generation prompt opens a thinking block, the host carries
that state into tool-call validation. Generated text must close the block before
an executable call. The flag comes from the rendered prompt; client parameters
and worker reply fields cannot enable it. Templates such as Flint that do not
open a thinking block continue to reject unmatched closing tags. Generated
pre-call text retains the existing content/streaming representation.

## Limits, queueing and cancellation

- The server accepts at most 16 concurrent connections. Excess connections receive a bare 429 with `Retry-After: 1` and no `X-Request-Id` header.
- One completion runs at a time. Up to eight more wait in FIFO order; a ninth receives 429 `queue_full`, and a queued request that waits more than 30 seconds receives 429 `queue_timeout`.
- After its body has been read and parsed, each completion has a 300-second deadline covering queueing and execution; reaching it returns 504 `request_timeout`.
- The handler polls the client connection every 50 ms while a request runs. A closed connection cancels the native generation at the next safe boundary and, when a response can still be written, returns 499 `request_cancelled`. Closing only the write side of the socket counts as a disconnect.
- Stopping or changing the model in Stark fails waiting and running requests with 503 `model_changed`; cancellation and deadlines keep the model loaded.
- Body admission, logical content and rendered worker framing each apply the [automatic shared policy](WORKER_PROTOCOLS.md#resource-budget). Normal app use requires no configuration; `STARK_REQUEST_BUDGET_MIB` is a developer/operator override accepting 1 through 64 before startup. Worker requests include the rendered prompt (system prompt, template framing and all messages) or token IDs, JSON encoding and the terminating newline. A request that exceeds the effective encoded line limit returns 400 `request_too_large` before being written to the worker. The model stays loaded, and a smaller request can be retried.
- The server reserves declared raw body bytes against an aggregate budget of two maximum image bodies, automatically 208–784 MiB, before reading. It holds that reservation through completion and releases it on success, parse failure, disconnect or cancellation. This bounds concurrent request bodies, not total process RSS: JSON parsing and decoded image bytes add memory beyond the raw-body reservation. Structural admission also bounds compact arrays and objects before parsing.
- The image service starts the worker with an empty, owner-private 0700 staging directory and stages each accepted image as a 0600 regular file with an opaque identifier. The worker validates the reference, owner, type, private permissions, size, link count, stable metadata and image signature, then immediately unlinks the name. It retains the opened-file lease through completion. Host cleanup removes remaining staged names when the transaction exits; worker teardown removes only opaque, owner-private, single-link regular files in its pinned directory. Media references and image bytes are not logged.
- Image preparation, generation, streaming reads, frame writes and discard share one worker transaction deadline, with a 300-second default. The timeout does not reset for each operation. Recoverable cancellation drains native work and clears prepared state before reuse; a protocol or cleanup failure closes and reaps the worker before another request can run.

### Prefix reuse

Each request is tokenized in full. For eligible text models, the worker can then reuse the state of the prompt prefix that the request shares with the immediately preceding request, and process only the remainder:

- Full-attention models such as Qwen3 dense rewind their attention cache on a 1,024-token grid. A later step of an agent conversation waits only for its new messages plus at most about 1,024 tokens.
- Native GLM-5.3 resumes from a snapshot at the start of the previous prompt's final 1,024-token reuse chunk. It restores recurrent/convolution state and the incomplete MLA/indexer pool while keeping completed attention rows in the loaded model.
- Native mixed-precision Qwen3.8 resumes from a recurrent-state snapshot at the start of the previous prompt's final 4,096-token prefill pass. Lily-format and admitted external mixed-affine models, including `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`, use this route. MTP must be on or off for both requests.

For either native snapshot route, a follow-up processes its new messages plus the previous prompt's final chunk when the entire snapshot prefix is unchanged. An earlier change processes the whole prompt. Reuse follows the worker's prefill grid and must preserve fresh-state outputs. The [generation contract](GENERATION.md#prefix-reuse) describes the rules; the [Qwen3-4B record](archive/runs/2026-11-01-generation-prefix-reuse.md), [Lily-format Qwen3.8 record](archive/runs/2026-11-02-hybrid-prefix-reuse.md) and [GLM/external mixed-affine record](archive/runs/2026-11-03-glm-qwen-prefix-reuse.md) retain their own artifact and build scope.

The cache holds one entry, in worker memory only. It is never written to disk, logged or returned. Each principal has its own random, in-memory scope:

- each Playground conversation
- the master API key within one API session
- each managed client grant within one API session

A request never reuses another principal's entry, so first-token timing cannot reveal what another principal sent. Clients that share the master key share its scope. The entry is released:

- when the next request uses a different scope or context size
- on any worker error, cancellation, client disconnect or deadline
- on any image request
- on Disconnect/revocation, a key change or an API restart
- after 600 seconds without requests
- when the model stops or changes

It costs the KV memory of the previous request (about 2.7 GiB after an 18,500-token step on Qwen3-4B) while idle. Native GLM-5.3 and Qwen3.8 add a snapshot to the completed attention caches the loaded model already keeps; the pinned Lily-format Qwen3.8 snapshot is about 110 MiB. Snapshot size depends on the admitted graph. No more than one request's state is held at a time.

## Progress in Stark

While a completion runs, the Stark menu shows that an app is generating, and the readout box in the menu and Playground follows the request: input and output rates as soon as they can be measured, the first-token time, the prompt token count and elapsed time. When the worker reports reused prompt-prefix tokens, the prompt line also shows how many were reused; workers without prefix reuse omit it. Once timing is calibrated for the model and request settings, elapsed time appears against an estimated total with a progress bar. Buffered requests are streamed from the worker internally for this purpose; the HTTP response is unchanged. When generation finishes, the box keeps the request's final rates, first-token time, prompt count and total time, including for requests whose tool calls then fail strict parsing.

These app events carry counts and timings, a random per-request identifier and the loaded model's library identifier only. They never contain prompts, responses, tool arguments, client identifiers, capability scopes or keys, and they are not written to disk. The worker serves one request at a time, so a request's progress ends before the next request starts, and progress from a stopped or replaced model is discarded. Completion estimates use the same bounded, in-memory calibration as Playground chats, with separate profiles per model load and request settings; see the [generation contract](GENERATION.md#generate).

## Error codes

Error responses are `{"error":{"message":...,"type":...,"code":...}}`. `type` is `authentication_error` for 401, `rate_limit_error` for 429, `server_error` for 5xx and `invalid_request_error` otherwise.

| Status | Code | Cause |
| --- | --- | --- |
| 400 | `invalid_request` | Duplicate headers, missing or invalid `Content-Length`, incomplete or non-object body, invalid field values |
| 400 | `length_required` | `Content-Length` missing (411 with the same code when `Transfer-Encoding` is present) |
| 400 | `invalid_json` | Malformed JSON, duplicate keys, `NaN`, infinity or excessive JSON structure |
| 400 | `unsupported_parameter` | Penalties, log probabilities or response formats the API does not implement |
| 400 | `mtp_not_supported` | Explicit MTP enable request on an unsupported model/worker or an image request |
| 400 | `invalid_messages` | The chat template rejected the messages |
| 400 | `images_unsupported`, `too_many_images`, `invalid_image_url`, `invalid_image_detail`, `invalid_image` | No active passing image evidence, multiple images, remote or malformed data URL, unsupported detail, invalid or mismatched image bytes |
| 400 | `context_length_exceeded` | Prompt plus output limit exceeds the context budget |
| 400 | `context_capacity_unknown` | Image inference requires a model with a declared finite context capacity |
| 503 | `automatic_output_unavailable` | Worker lacks automatic memory planning; rebuild or update the worker |
| 400 | `request_too_large` | Input, encoded worker request or generated output exceeds the resource budget |
| 400 | `tools_unsupported` | Running model/template has no supported parser |
| 502 | `invalid_tool_call`, `tool_choice_not_satisfied` | Invalid, truncated, unknown or disallowed generated call, or unsatisfied required/named choice; no executable call returned |
| 401 | `invalid_api_key` | Missing or wrong bearer key |
| 403 | `origin_not_allowed`, `host_not_allowed` | `Origin` header present, or non-loopback `Host` |
| 404 | `not_found`, `model_not_found` | Unknown path, or `model` is not the running model |
| 413 | `body_too_large` | Body exceeds the configured encoded body budget |
| 413 | `text_too_large` | Aggregate logical message text exceeds the configured content budget |
| 413 | `image_too_large` | Decoded image exceeds the 4 MiB service limit |
| 415 | `unsupported_media_type` | `Content-Type` other than `application/json` |
| 429 | `server_busy`, `queue_full`, `queue_timeout`, `body_budget_full` | Connection limit, queue limit, queue wait expired, or shared pending raw-body budget reached |
| 429 | `memory_pressure` | Available RAM is too low to admit a large body before reading or parsing it; a smaller request can be retried |
| 499 | `request_cancelled` | Client disconnected |
| 503 | `model_not_loaded`, `model_changed`, `server_stopped`, `server_shutdown`, `request_stopped`, `streaming_unavailable`, `stop_sequences_unavailable`, `generation_failed` | No model in Chat mode, model stopped or changed, Stark quitting, worker lacks a capability, or native generation failed |
| 504 | `request_timeout` | 300-second deadline reached |
| 500 | `internal_error`, `http_error` | Unexpected failure |

## Managed OpenWork connection

On macOS, open **Connections → Apps** and choose **Connect** on the OpenWork row. Open OpenWork with a local workspace first. Stark installs its bundled OpenCode V2 connector into the native global `opencode.jsonc` selected by the verified OpenWork engine's `OPENCODE_CONFIG_DIR`. It adds only the connector's local package URL, private Unix-socket path, public expected API URL and Stark ownership marker. **Manage → Reconnect** repairs a recognized Stark-owned entry after its app location or credential namespace changes, including a switch between signed installed and ad-hoc development builds. It updates only those public fields; credentials and unrelated settings are preserved. Unknown formats, conflicting ownership and duplicate Stark entries are rejected.

If configuration changes, **quit OpenWork completely and reopen it**, then select **Stark** and the running model in its model picker. Closing the OpenWork window does not restart its engine. No copied API key, manual JSON, terminal command or credential helper is required. OpenWork lists the model running in Stark's Chat mode; the Model ID menu under **Connections → Connect your own app** only inspects saved models and does not run them. After a normal model switch, the connector refreshes the running model within a few seconds. A previously saved selection can show unavailable until the new running model is selected.

If OpenWork settings already define a manual provider with ID `stark`, remove only that provider entry before connecting and keep your other provider entries. The managed connection does not migrate existing client credentials.

The managed connection is distinct from the **Copy All as JSON** and key-copy workflows for other clients. The API's master bearer key remains in macOS Keychain. A separate device-only Keychain item records whether OpenWork is approved; it does not contain the API key. For an approved request, Stark creates a random 256-bit bearer capability associated in its admission table with the verified OpenCode sidecar's process identity and current API session, gives it a renewable 120-second admission lease, and returns it over the private owner-only Unix socket. The broker renews the same client token when it is requested after 30 seconds. The connector requests that capability when making an API request and places it only in the in-memory Authorization header. As an HTTP bearer token, it is transferable if exposed while valid; the process association gates issuance at Stark and is not a kernel binding on later HTTP use. Expiring an admission lease does not cancel a request that was already admitted; explicit Disconnect revokes OpenWork's grants in the session and cancels their active requests. The capability is not written to the OpenWork config, auth store, environment, process arguments or logs. The Swift broker keeps active client capabilities in memory; the API admission table stores only their digests. Only models with declared finite context and output limits are published; OpenWork handles its own tool execution and approval flow.

The accepted app identity is the regular macOS application with bundle ID `com.differentai.openwork` and Apple Team ID `F5DJWB4CCV`; its engine peer must be the signed `opencode2` sidecar at the expected path inside that app and descend from the verified app process. This verifies the official app and engine binary. It does not isolate plugins running inside OpenWork: plugins share the authorized client process and may be able to inspect request data and the bearer capability in memory. Install only OpenWork plugins you trust.

Choose **Manage → Disconnect** on the OpenWork row to disable the approval and revoke its capabilities for the current API session, including admitted requests. Every OpenWork grant uses the client ID prefix `openwork.`, and the backend revokes only that prefix, so a connected OpenCode Desktop keeps working. A later **Connect** action on that row enables it again. Stark restarts and API-session changes invalidate prior capabilities; the connector refreshes status and can obtain a new capability for its next request. If Stark's saved port changes, choose **Manage → Reconnect** to update the public expected URL, then reopen OpenWork. A failed or occupied port does not silently change the URL; use the Base URL shown under **Connect your own app** and connect again after the listener is available.

The stock helperless connection has narrow validation with official OpenWork 0.18.63 / OpenCode 2.0.26 and the pinned Qwen3-4B artifact, including chat, follow-up, client-side tool-result exchange, restart/reconnect and signed test-package upgrades. Model task evidence includes successful bounded result use, formatting misses and a failed final numeric check. See the [managed-connection validation record](archive/runs/2026-10-31-openwork-managed-connection.md).

The pinned Lily q4-xl artifact also completed streaming chat/follow-up, two sequential client-side reads with result use, cancellation/recovery and reconnect after a signed Stark relaunch. Direct GUI chat/follow-up were checked with the local model selected. This qualifies a bounded read workflow, not broad agent reliability. OpenWork can retain a different model for an existing session; choose Stark and the running model in that session's model menu before sending. When Stark switches models, the previous saved selection may become unavailable while the new model appears under Stark. If OpenWork retains an unavailable-model state after recovery, choose its Retry or Refresh action, then select the running Stark model.

Agent clients send substantial system instructions and tool definitions, so Stark may process a much longer input than a Playground chat before producing its first token. The Lily validation observed roughly 19,000 input tokens and about 8–20 seconds to the first OpenWork text event for short chat across runs; these observations do not isolate setup, input processing and host contention, or promise a response time. Use explicit tool budgets and client approvals. See the [Lily acceptance record](archive/runs/2026-11-01-openwork-lily-acceptance.md) for the exact build, model, timings and limitations.

The earlier [helper-based desktop run](archive/runs/OPENWORK_KEYCHAIN_E2E_2026_10_31.md) remains historical evidence for its separate credential path; it is not the basis for the managed-connection result above.

## Managed OpenCode Desktop connection

Under **Connections → Apps**, choose **Connect** on the OpenCode Desktop row for the official OpenCode Desktop macOS app, version 2.2.5 or later. Stark opens OpenCode if it is not running, waits for its background engine, and installs the same bundled connector entry used for OpenWork. The entry goes into the global `opencode.jsonc` that engine reads: its `OPENCODE_CONFIG_DIR`, otherwise `$XDG_CONFIG_HOME/opencode`, otherwise `~/.config/opencode`. OpenCode Desktop copies the variables your login shell exports into its engine, so a value set in your shell profile counts. Select **Stark** in OpenCode's model picker. No copied API key, manual JSON or terminal command is required.

OpenWork and OpenCode can read the same global configuration, so they share one connector entry and one private socket. Connecting either client leaves an entry the other already installed unchanged, and Stark identifies each peer before answering. If OpenCode's settings already define a manual provider with ID `stark`, remove only that provider entry before connecting.

OpenCode Desktop runs its engine differently from OpenWork. It copies its signed `opencode-cli` to `~/Library/Application Support/ai.opencode.desktop/cli/<version>-<arch>/` and starts that copy detached as a background service. The engine's parent is launchd. It keeps running after you quit OpenCode, and later OpenCode launches reuse it. With no app ancestry to check, Stark approves each engine instance (process ID and start time) once:

- **Connect** approves the engine Stark verified while installing the connector.
- A later engine, for example after an OpenCode update, a restart of its background service or a new login, gets `not_connected` until you choose **Allow** next to it in Connections. **Deny** ignores that engine until Stark quits; **Manage → Reconnect** approves the current engine again.
- A separate device-only Keychain item (account `opencode-approval-v1` in the `.desktop-connections` service) records whether OpenCode is connected and up to eight approved engine identities. It holds no credential. **Disconnect** clears it.

Stark accepts an OpenCode peer only when all of these hold:

- The peer is the signed `opencode-cli` engine (Apple Team ID `5NZ4Q7NXJ4`, Anomaly Innovations), checked through its socket audit token.
- Its executable resolves to `opencode-cli` directly inside one staging folder under `~/Library/Application Support/ai.opencode.desktop/cli/`. The staging folder name uses only letters, digits, `.`, `_` and `-`, up to 64 characters.
- It was started as `serve --service`, the way OpenCode Desktop starts its background engine, without interpreter, inspector or debugger arguments, and without non-empty `BUN_OPTIONS`, `BUN_BE_BUN`, `BUN_INSPECT*`, `DYLD_*`, `ELECTRON_RUN_AS_NODE` or `OPENCODE_DESKTOP_REMOTE_DEBUGGING_PORT`. `NODE_OPTIONS` is allowed in the engine, which ignores it, because OpenCode copies it from many developers' shell profiles; the OpenCode Desktop app itself must not have it.
- A verified OpenCode Desktop app (bundle ID `ai.opencode.desktop`, Team ID `5NZ4Q7NXJ4`) is running and holds an open loopback TCP connection that the engine process itself accepted, so the engine is the one Desktop is using. While OpenCode is closed, its engine cannot obtain capabilities. An engine copy that Desktop is not connected to is never accepted, and **Connect** never chooses it.

Capabilities work as for OpenWork: renewable 120-second leases held only in memory, client IDs `opencode.<pid>.<start time>`, and at most eight active OpenCode capabilities. The backend admission table holds up to 24 managed clients, enough for OpenWork's 16 and OpenCode's 8. **Disconnect** revokes only grants with the `opencode.` prefix and cancels their admitted requests; OpenWork is unaffected.

The approved engine is shared. While it is approved and OpenCode Desktop is open, any client attached to that background service can use Stark through it, including a terminal `opencode` session that reuses the same service. Plugins loaded by the engine share its process and may read request data or the bearer capability in memory. Install only plugins you trust. A standalone or Homebrew `opencode` that runs its own engine is not a managed client: any process can start it with any configuration, so configure it with the copied API key instead. If you restart the background service from a standalone or Homebrew `opencode` binary, Stark will not accept that engine; stop it with `opencode service stop`, then reopen OpenCode Desktop so it starts its own engine.

A running OpenCode 2.2.5 engine reloads its configuration when the connector entry is written, so Stark normally appears in OpenCode's model picker within a second of **Connect**, without restarting anything. If it does not, quit OpenCode first (an open OpenCode restarts a stopped engine), stop the background engine with `opencode service stop` (or quit `opencode-cli` in Activity Monitor), reopen OpenCode and choose **Allow** when Stark asks about the new engine. If Stark's saved port changes, choose **Manage → Reconnect** to update the public expected URL.

OpenCode Desktop's background engine listens on a fixed default port. Two OpenCode Desktop profiles under one account cannot run their engines at the same time unless one sets another port with `opencode service set port`; this matters for isolated test profiles, not normal use.

A desktop run with a signed test build and official OpenCode Desktop 2.2.5 qualified Connect, live configuration reload, chat, per-engine **Allow** and **Disconnect**. It did not cover tool calls, **Deny**, the production bundle identifier or a notarized build. See the [OpenCode desktop run](archive/runs/2026-11-02-opencode-desktop-live.md) and the [implementation record](archive/runs/2026-11-01-opencode-managed-client.md).

## Client compatibility

Clients must send authenticated requests from native or backend code on the same Mac and stay within the capabilities shown by `/v1/capabilities`. The client checks below validated the text subset against the published 0.3.0 bridge and worker on 2026-10-23 ([record](archive/runs/LOCAL_API_CLIENT_VALIDATION_2026_10_23.md)); they do not qualify image input:

| Client | Configuration | Result |
| --- | --- | --- |
| `llm` 0.36 | Custom OpenAI model, key from environment, buffered and streaming | Passed, including three concurrent requests |
| `aichat` 0.30.0 | Custom OpenAI endpoint, buffered and streaming | Passed for single-turn text |
| Cherry Studio 2.1.7 | Custom OpenAI provider, Chat mode, streaming on, overrides off | Discovery and fresh chats passed; follow-up turns returned 400 because the client sent `reasoning_content`, which 0.3.0 rejects and current source accepts |
| Browser page | Cross-origin JSON request with bearer key | Blocked, as designed |

A client appearing here does not mean its agents, tools or other workflows are supported.

Goose's native CLI independently completed chat and a client-side read-only tool task with the same artifact, using its supported `auth.command` route and an isolated Keychain reader. Its desktop GUI remains untested. The validation record includes credential-flow review and disposable-profile cleanup; client logging/session options alone do not guarantee that a client writes no local request data.

## Compatibility boundary

The surface follows the `/v1/models` and `/v1/chat/completions` shape used by local engines such as [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) and [LM Studio](https://lmstudio.ai/docs/developer/openai-compat), restricted to supported chat templates, one model and a bounded worker. Image input is conditional on the live host evidence gate; other multimodal content, JSON-schema output, embeddings, the Responses API, batching and vendor-specific sampling fields are unsupported. Tool requests require a supported template program and parser; clients own execution, schema validation and approvals.
