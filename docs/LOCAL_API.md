# Stark local API

The menu bar application's bridge serves a loopback-only, OpenAI-style HTTP API so that other applications on the same Mac can chat with the loaded model. It supports buffered or streamed text completions and one-image requests through one loaded model. Images require passing host evidence for the [supported Qwen3.5 family](VISION_FAMILY_CONTRACT.md). The API binds to `127.0.0.1`, requires bearer authentication, and rejects browser-origin requests. Clients may use the persistent Keychain bearer key or, for the managed OpenWork integration below, a short-lived per-client capability.

## Connection and authentication

Open **Connection Details** in Stark to see the base URL (for example `http://127.0.0.1:8000/v1`), the API key and the Model ID. Manually configured native clients send the copied master key on every request, including health checks:

```text
Authorization: Bearer <key>
```

The Swift host loads a random 256-bit key from macOS Keychain and sends it to the backend through its private control pipe. It generates a key only when Keychain reports that the item is missing, and activates it only after storage succeeds. Concurrent first launches use the winning stored value. Relaunches, worker restarts, model changes and app upgrades reuse this credential. Secrets are absent from process arguments, environment, UserDefaults, configuration files and logs.

**Show** and **Copy** retrieve the existing key without rotating it. Key and connection-JSON copies retain the sensitive clipboard markers. **Regenerate API Key** explicitly replaces the stored credential after confirmation; clients configured with the copied master key need the replacement, while managed OpenWork capabilities refresh through Stark. A storage failure preserves the working key. Locked, denied, corrupt or unavailable Keychain access produces recovery instructions in Connection Details; authentication is never disabled. Unlock or allow access in Keychain Access and choose **Retry API**. A corrupt entry can be replaced using the explicit regeneration action. Do not save copied connection JSON in ordinary configuration files.

The installed, Team-ID-signed app uses service `com.starkengine.menubar.local-api`, account `bearer-key-v1`. Canonical unsigned/ad-hoc development builds use the separate `.development` service. Other bundle identifiers receive a separate development service suffix; native tests use disposable services. A signing identity or Keychain access-policy change can require access recovery. [Developer ID package upgrades](archive/runs/OPENWORK_KEYCHAIN_E2E_2026_10_31.md) reused a credential under an isolated bundle identifier; production-identifier upgrades and notarized distribution require their own validation.

Keychain operations run on a dedicated background queue, with at most one outstanding operation. If macOS waits for legacy Keychain access approval, Connection Details reports the pending operation while model management and chat remain usable. Retry, port changes and regeneration wait for that operation to finish. A pending regeneration is not reported as failed because macOS may still commit it. Approve access in Keychain Access; if it remains pending, quit and reopen Stark afterward. Rebuilt ad-hoc development apps can require renewed access approval; a stable service name does not bypass Keychain's signing checks.

The non-secret listening port is saved separately in UserDefaults, initially 8000, and reused. Stark never silently chooses another port. If it is occupied, Connection Details explains the conflict; close the other server or enter a port from 1 through 65535 and choose **Apply Port**. Update clients after an intentional port change. A failed port change keeps any previous listener active and displays its actual Base URL. Model management and the playground remain usable when the API cannot start. The curl example prompts without echo and passes authentication through stdin, without embedding a key in arguments or exporting it.

Request rules enforced before any endpoint runs:

- The `Host` header must name `127.0.0.1`, `localhost` or `::1`.
- Requests carrying an `Origin` header are rejected, so browser pages and webviews cannot use the API. `OPTIONS` preflight returns 501 because there is no CORS handler.
- Duplicate `Host` or `Authorization` headers are rejected.
- Text-only request bodies are limited to 1 MiB. When `capabilities.images` is true, a request containing an image part may use up to 6 MiB; text-only bodies remain limited to 1 MiB. Every request must carry exactly one `Content-Length` and must not use `Transfer-Encoding`. When `Content-Type` is present it must be `application/json`. JSON with duplicate keys, `NaN` or infinities is rejected.

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

The service accepts images up to 4 MiB compressed, with PNG or JPEG signatures. The engine transport has a wider 20 MiB file limit, but the HTTP service applies its 4 MiB cap before staging. The parser does not decode pixels in Python; the native processor applies its source dimension, source pixel, processed pixel, patch and image-token limits. See the [request parser](../tools/converter/stark_convert/api_request.py), [image validator](../tools/converter/stark_convert/media.py) and [worker qualification gate](../tools/converter/stark_convert/vision_runtime.py).

**Context limit.** `/v1/models` reports a declared finite model maximum as `context_length`, with `context_capacity_known: true`. The combined rendered input and requested output must fit that context. A backend without a declared finite maximum reports `context_length: null` and `context_capacity_known: false`; its unsigned 32-bit protocol ceiling is not advertised as a model capacity. For these models, text requests reserve only the rendered input plus requested output, while any custom playground conversation budget still applies. Image requests require a declared finite context. There is no global 16,384-token context cap. Request byte limits and available memory still apply; a published context capacity does not establish full-length performance or memory qualification on this Mac.

**Output limit.** `max_tokens` and `max_completion_tokens` are aliases; supply at most one. The default is the smaller of 512 and the model's limit. Requests may go up to the model's limit, which is capped at 4,096 and shown as `max_output_tokens` in `/v1/models`; a larger value returns 400 rather than being clamped. `n` must be 1. A completion that reaches the output or context limit reports `finish_reason: "length"`.

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
model changes; Decision mode does not expose MTP. These fields prepare the
backend for a UI toggle; no new UI control is implemented here.

`defaults.mtp` is the active model's effective saved choice. Registered models
start off, and each model remembers its own boolean preference in the private
model registry across stops, switches and application restarts. The preference
becomes inactive when the loaded worker lacks support and is retained for a
later supported worker. Saving it does not reload the model or clear the chat.

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

The desktop's private JSON-lines backend accepts the following command for
the future UI, using the internal `model_id` from its `ready` event:

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

## Function tools

`/v1/capabilities` reports `tool_calling`, `tool_parser` and `tool_choices` for the running model. The parser is `qwen3-hermes-v1`, admitted only for the exact official Qwen3-0.6B template SHA-256 `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` or the official `mlx-community/Qwen3-4B-4bit` template SHA-256 `87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5` from revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`. The 4B template is preserved as an upstream fixture with source and license provenance under `tools/converter/tests/fixtures/`. Other templates keep ordinary chat and reject tool definitions/history with `tools_unsupported`. A model name alone does not qualify a template.

- `tools` accepts up to 64 function definitions with unique ASCII names, optional descriptions, and object parameter schemas. Schemas are model hints; schema-constrained generation is unavailable and `strict: true` is rejected. The client must validate execution arguments against its schema.
- `tool_choice` accepts `auto`, `none`, `required`, or `{"type":"function","function":{"name":"supplied_name"}}`. Named choice renders only that definition. Required/named output is checked after generation; failure returns `tool_choice_not_satisfied`, without synthesizing a call. `none` omits definitions from rendering and rejects generated tool markers.
- `parallel_tool_calls` is a boolean, default true. False permits at most one generated call; a violation fails the completion. At most 16 calls are returned, in generation order, with distinct stable IDs within the response and JSON-object argument strings.
- Assistant history may contain null or omitted text alongside `tool_calls`. Each call requires `id`, `type: function`, and a function name/argument string. Tool-result messages require string/text-part content and the matching `tool_call_id`. All outstanding results must be returned before another assistant/user turn. Duplicate, orphaned and incomplete transactions are rejected. The pinned template receives complete history, argument objects, definitions, and results ordered by their call IDs.
- Only complete `<tool_call>` JSON objects from a normally terminated generation become executable requests. Duplicate JSON keys, nonfinite numbers, unknown names, invalid argument JSON, truncated markers, extra trailing text and limit violations fail. No guessed repairs occur. Malformed output does not unload an otherwise aligned worker.
- Definitions together are limited to 256 KiB; each argument object to 64 KiB; each tool result to 256 KiB; parsed tool output to 512 KiB. JSON depth is limited to 32 and item count to 16,384. The ordinary 1 MiB body/worker-line limits and model context limits also apply. Tool use and image input cannot be combined.

Buffered completions return `message.tool_calls`, optional text (null when absent), and `finish_reason: "tool_calls"`. Tool-enabled streaming delivers ordinary text as it is generated. Possible tool markers and their arguments stay private through the terminal worker frame and strict parsing, then indexed `delta.tool_calls` are emitted exactly once, followed by the terminal reason, optional usage and `[DONE]`. Trailing whitespace is held until it can be distinguished from the separator before a call. Invalid or truncated calls produce a stream error after any provisional text; they never produce executable tool deltas. Usage includes all generated framing/argument tokens and tokens completing a stop sequence. Cancellation/deadline checks remain active throughout generation.

An agent client can still take longer to produce its first token than a short Playground chat. Its system instructions, tool schemas and conversation history are part of the model input and must be processed before decoding begins. Live text streaming removes the additional wait for the entire answer; it does not shorten the client's instructions. [Prefix reuse](#prefix-reuse) removes most of the repeated work on later steps of the same conversation, but the first request of a conversation is still processed in full. The [OpenWork first-text latency record](archive/runs/2026-10-31-openwork-first-text-latency.md) separates native first-token work from the rest of a measured client request.

Qwen/Qwen3-0.6B with unquantized BF16 weights completed a narrowly prompted two-tool result-use task locally. Looser variants failed through an unwanted parallel call or premature stopping; this checkpoint is not a general-purpose agent recommendation. Protocol fixtures and task-competence evidence are recorded separately.

The pinned MLX Qwen3-4B 4-bit/group-64 artifact completed a bounded two-read task in OpenWork, including client-side execution, result return and a correct final calculation. A looser variant repeated a read until cancelled. Use explicit task limits and client approvals; this is a narrow workflow qualification. The [desktop validation record](archive/runs/OPENWORK_KEYCHAIN_E2E_2026_10_31.md) separates protocol, model behavior and signed-package evidence.

## Limits, queueing and cancellation

- The server accepts at most 16 concurrent connections. Excess connections receive a bare 429 with `Retry-After: 1` and no `X-Request-Id` header.
- One completion runs at a time. Up to eight more wait in FIFO order; a ninth receives 429 `queue_full`, and a queued request that waits more than 30 seconds receives 429 `queue_timeout`.
- After its body has been read and parsed, each completion has a 300-second deadline covering queueing and execution; reaching it returns 504 `request_timeout`.
- The handler polls the client connection every 50 ms while a request runs. A closed connection cancels the native generation at the next safe boundary and, when a response can still be written, returns 499 `request_cancelled`. Closing only the write side of the socket counts as a disconnect.
- Stopping or changing the model in Stark fails waiting and running requests with 503 `model_changed`; cancellation and deadlines keep the model loaded.
- The body limit and the worker's own 1 MiB request line are separate. Worker requests include the rendered prompt (system prompt, template framing and all messages) or token IDs, JSON encoding and the terminating newline. A request that exceeds this encoded line limit returns 400 `request_too_large` before writing that request to the worker. The model stays loaded, and a shorter conversation can be retried.
- The server reserves declared raw body bytes against a shared 24 MiB pending-body budget before reading. It holds that reservation through completion and releases it on success, parse failure, disconnect or cancellation. This bounds concurrent request bodies, not total process RSS: JSON parsing and decoded image bytes add memory beyond the raw-body reservation.
- The image service starts the worker with an empty, owner-private 0700 staging directory and stages each accepted image as a 0600 regular file with an opaque identifier. The worker validates the reference, owner, type, private permissions, size, link count, stable metadata and image signature, then immediately unlinks the name. It retains the opened-file lease through completion. Host cleanup removes remaining staged names when the transaction exits; worker teardown removes only opaque, owner-private, single-link regular files in its pinned directory. Media references and image bytes are not logged.
- Image preparation, generation, streaming reads, frame writes and discard share one worker transaction deadline, with a 300-second default. The timeout does not reset for each operation. Recoverable cancellation drains native work and clears prepared state before reuse; a protocol or cleanup failure closes and reaps the worker before another request can run.

### Prefix reuse

Each request is tokenized in full. For eligible text models (full-attention models such as Qwen3 dense), the worker can then reuse the attention cache of the prompt prefix that the request shares with the immediately preceding request, and process only the remainder. A later step of an agent conversation therefore waits only for its new messages plus at most about 1,024 tokens. Reuse follows the worker's 1,024-token prefill grid, so outputs are identical to processing the whole prompt; the [generation contract](GENERATION.md#prefix-reuse) describes the rules and the [validation record](archive/runs/2026-11-01-generation-prefix-reuse.md) the measurements.

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

It costs the KV memory of the previous request (about 2.7 GiB after an 18,500-token step on Qwen3-4B) while idle. No more than one request's state is held at a time.

## Error codes

Error responses are `{"error":{"message":...,"type":...,"code":...}}`. `type` is `authentication_error` for 401, `rate_limit_error` for 429, `server_error` for 5xx and `invalid_request_error` otherwise.

| Status | Code | Cause |
| --- | --- | --- |
| 400 | `invalid_request` | Duplicate headers, missing or invalid `Content-Length`, incomplete or non-object body, invalid field values |
| 400 | `length_required` | `Content-Length` missing (411 with the same code when `Transfer-Encoding` is present) |
| 400 | `invalid_json` | Malformed JSON, duplicate keys, `NaN` or infinity |
| 400 | `unsupported_parameter` | Penalties, log probabilities or response formats the API does not implement |
| 400 | `mtp_not_supported` | Explicit MTP enable request on an unsupported model/worker or an image request |
| 400 | `invalid_messages` | The chat template rejected the messages |
| 400 | `images_unsupported`, `too_many_images`, `invalid_image_url`, `invalid_image_detail`, `invalid_image` | No active passing image evidence, multiple images, remote or malformed data URL, unsupported detail, invalid or mismatched image bytes |
| 400 | `context_length_exceeded` | Prompt plus output limit exceeds the context budget |
| 400 | `context_capacity_unknown` | Image inference requires a model with a declared finite context capacity |
| 400 | `request_too_large` | Rendered worker request exceeds the 1 MiB encoded line limit |
| 400 | `tools_unsupported` | Running model/template has no supported parser |
| 502 | `invalid_tool_call`, `tool_choice_not_satisfied` | Invalid, truncated, unknown or disallowed generated call, or unsatisfied required/named choice; no executable call returned |
| 401 | `invalid_api_key` | Missing or wrong bearer key |
| 403 | `origin_not_allowed`, `host_not_allowed` | `Origin` header present, or non-loopback `Host` |
| 404 | `not_found`, `model_not_found` | Unknown path, or `model` is not the running model |
| 413 | `body_too_large` | Body exceeds the active body cap: 1 MiB normally, or 6 MiB when image capability is enabled |
| 413 | `image_too_large` | Decoded image exceeds the 4 MiB service limit |
| 415 | `unsupported_media_type` | `Content-Type` other than `application/json` |
| 429 | `server_busy`, `queue_full`, `queue_timeout`, `body_budget_full` | Connection limit, queue limit, queue wait expired, or shared pending raw-body budget reached |
| 499 | `request_cancelled` | Client disconnected |
| 503 | `model_not_loaded`, `model_changed`, `server_stopped`, `server_shutdown`, `request_stopped`, `streaming_unavailable`, `stop_sequences_unavailable`, `generation_failed` | No model in Chat mode, model stopped or changed, Stark quitting, worker lacks a capability, or native generation failed |
| 504 | `request_timeout` | 300-second deadline reached |
| 500 | `internal_error`, `http_error` | Unexpected failure |

## Managed OpenWork connection

On macOS, Stark's **Connection Details** includes **Connect to OpenWork**. Open OpenWork with a local workspace, then choose that button. Stark installs its bundled OpenCode V2 connector into the native global `opencode.jsonc` selected by the verified OpenWork engine's `OPENCODE_CONFIG_DIR`. It adds only the connector's local package URL, private Unix-socket path, public expected API URL and Stark ownership marker; the installer preserves unrelated configuration and rejects a conflicting Stark connector. If configuration changes, reopen OpenWork and select **Stark** in its model picker. No copied API key, manual JSON, terminal command or credential helper is required.

If OpenWork settings already define a manual provider with ID `stark`, remove only that provider entry before connecting and keep your other provider entries. The managed connection does not migrate existing client credentials.

The managed connection is distinct from the **Copy All as JSON** and key-copy workflows for other clients. The API's master bearer key remains in macOS Keychain. A separate device-only Keychain item records whether OpenWork is approved; it does not contain the API key. For an approved request, Stark creates a random 256-bit bearer capability associated in its admission table with the verified OpenCode sidecar's process identity and current API session, gives it a renewable 120-second admission lease, and returns it over the private owner-only Unix socket. The broker renews the same client token when it is requested after 30 seconds. The connector requests that capability when making an API request and places it only in the in-memory Authorization header. As an HTTP bearer token, it is transferable if exposed while valid; the process association gates issuance at Stark and is not a kernel binding on later HTTP use. Expiring an admission lease does not cancel a request that was already admitted; explicit Disconnect revokes the session's grants and cancels their active requests. The capability is not written to the OpenWork config, auth store, environment, process arguments or logs. The Swift broker keeps active client capabilities in memory; the API admission table stores only their digests. Only models with declared finite context and output limits are published; OpenWork handles its own tool execution and approval flow.

The accepted app identity is the regular macOS application with bundle ID `com.differentai.openwork` and Apple Team ID `F5DJWB4CCV`; its engine peer must be the signed `opencode2` sidecar at the expected path inside that app and descend from the verified app process. This verifies the official app and engine binary. It does not isolate plugins running inside OpenWork: plugins share the authorized client process and may be able to inspect request data and the bearer capability in memory. Install only OpenWork plugins you trust.

Choose **Disconnect** in Connection Details to disable the approval and revoke capabilities for the current API session, including admitted requests. A later **Connect to OpenWork** action enables it again. Stark restarts and API-session changes invalidate prior capabilities; the connector refreshes status and can obtain a new capability for its next request. If Stark's saved port changes, choose **Reconnect** to update the public expected URL, then reopen OpenWork. A failed or occupied port does not silently change the URL; use the Base URL shown in Connection Details and connect again after the listener is available.

The stock helperless connection has narrow validation with official OpenWork 0.18.63 / OpenCode 2.0.26 and the pinned Qwen3-4B artifact, including chat, follow-up, client-side tool-result exchange, restart/reconnect and signed test-package upgrades. Model task evidence includes successful bounded result use, formatting misses and a failed final numeric check. Use explicit tool budgets and client approvals. See the [managed-connection validation record](archive/runs/2026-10-31-openwork-managed-connection.md).

The earlier [helper-based desktop run](archive/runs/OPENWORK_KEYCHAIN_E2E_2026_10_31.md) remains historical evidence for its separate credential path; it is not the basis for the managed-connection result above.

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

The surface follows the `/v1/models` and `/v1/chat/completions` shape used by local engines such as [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) and [LM Studio](https://lmstudio.ai/docs/developer/openai-compat), restricted to supported chat templates, one model and a bounded worker. Image input is conditional on the live host evidence gate; other multimodal content, JSON-schema output, embeddings, the Responses API, batching and vendor-specific sampling fields are unsupported. Tool requests require the pinned supported template/parser; clients own execution, schema validation and approvals.
