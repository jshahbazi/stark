# Stark local API

The menu bar application serves a loopback-only, OpenAI-style HTTP API so that other applications on the same Mac can chat with the loaded model. It covers text chat completions, buffered or streamed, through one loaded model. It is not a hosted service: it binds to `127.0.0.1`, requires a per-launch bearer key, and rejects browser-origin requests.

## Connection and authentication

Open **Connection Details** in Stark to see the base URL (for example `http://127.0.0.1:8000/v1`), the API key and the Model ID. Send the key on every request, including health checks:

```text
Authorization: Bearer <key>
```

The key is a random 256-bit value generated when Stark launches, held in memory only, and never written to disk by Stark. **Copy** places it on the system pasteboard. Stark prefers port 8000 and falls back to a free loopback port, so always use the URL shown. After reopening Stark, copy the new key and check the URL again.

Request rules enforced before any endpoint runs:

- The `Host` header must name `127.0.0.1`, `localhost` or `::1`.
- Requests carrying an `Origin` header are rejected, so browser pages and webviews cannot use the API. `OPTIONS` preflight returns 501 because there is no CORS handler.
- Duplicate `Host` or `Authorization` headers are rejected.
- Bodies are limited to 1 MiB, must carry exactly one `Content-Length`, and must not use `Transfer-Encoding`. When `Content-Type` is present it must be `application/json`. JSON with duplicate keys, `NaN` or infinities is rejected.

Responses carry `Cache-Control: no-store` and an `X-Request-Id` header, except the connection-limit response described under [limits](#limits-queueing-and-cancellation). Stark writes no access log and never logs prompts, replies or keys.

## Endpoints

| Endpoint | Behaviour |
| --- | --- |
| `GET /v1/health`, `GET /health` | Application status: whether a model is loaded, whether a request is running, queue depth and limits, the model list, and capability flags |
| `GET /v1/capabilities` | The same status document |
| `GET /v1/models` | The running model as a one-item list, or an empty list when no model is running in Chat mode |
| `GET /v1/models/{id}` | The matching running model, or 404 |
| `POST /v1/chat/completions` | One text completion, buffered or streamed |

Each model entry has `id`, `object`, `created`, `owned_by`, `name`, `context_length` and `max_output_tokens`. Embeddings, the Responses API, image input, structured output, tool and function calls, and batching are not available.

## Chat completions

Accepted request fields: `model`, `messages`, `stream`, `stream_options`, `max_tokens`, `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, `n`, `presence_penalty`, `frequency_penalty`, `logprobs`, `top_logprobs`, `response_format`, `user` and `metadata`. Unknown fields are rejected.

**Model and messages.** `model` may be omitted, or may be the running model's ID or installed name; any other value returns 404 `model_not_found`. `messages` must be non-empty. Roles `system`, `developer` (normalised to `system`), `user` and `assistant` are accepted. Content is a string or an array of `{"type":"text","text":"..."}` parts, which are concatenated. An optional `name` of up to 64 ASCII characters is validated and ignored. Assistant messages may carry `reasoning_content` as a string, which is passed to the chat template; Stark 0.4.0 and earlier reject that field. Tool and function-call fields are rejected. The model's chat template renders the conversation; a template that rejects the messages returns 400 `invalid_messages`.

**Output limit.** `max_tokens` and `max_completion_tokens` are aliases; supply at most one. The default is the smaller of 512 and the model's limit. Requests may go up to the model's limit, which is capped at 4,096 and shown as `max_output_tokens` in `/v1/models`; a larger value returns 400 rather than being clamped. `n` must be 1. A completion that reaches the output or context limit reports `finish_reason: "length"`.

**Sampling.** `temperature` defaults to 0, which selects greedy decoding. A positive finite temperature up to 2 selects stochastic sampling with `top_p` (greater than 0, at most 1), `top_k` (1 through the vocabulary size), `min_p` (0 through 1) and `seed` (unsigned 64-bit, default 0). These values are accepted without effect at temperature 0. Non-zero `presence_penalty` or `frequency_penalty`, `logprobs: true`, `top_logprobs` above 0, and any `response_format` other than `{"type":"text"}` return 400 `unsupported_parameter`; the neutral values are accepted. `user` and `metadata` are accepted within their size bounds and discarded.

**Stop sequences.** `stop` is one string or up to four non-empty strings of at most 1,024 characters. The worker matches them against decoded text, including across chunks, and withholds the match and everything after it. Usage counts still include the token that completed the match.

**Streaming.** With `stream: true` the response is Server-Sent Events of `chat.completion.chunk` objects followed by `data: [DONE]`. The first chunk carries the assistant role. `stream_options: {"include_usage": true}` adds a final usage-only chunk. Validation errors keep their HTTP status because they occur before streaming starts; a failure during streaming emits an error object in the stream and closes without `[DONE]`.

## Limits, queueing and cancellation

- The server accepts at most 16 concurrent connections. Excess connections receive a bare 429 with `Retry-After: 1` and no `X-Request-Id` header.
- One completion runs at a time. Up to eight more wait in FIFO order; a ninth receives 429 `queue_full`, and a queued request that waits more than 30 seconds receives 429 `queue_timeout`.
- Each request has a 300-second deadline from the time its body is read; reaching it returns 504 `request_timeout`.
- The handler polls the client connection every 50 ms while a request runs. A closed connection cancels the native generation at the next safe boundary and, when a response can still be written, returns 499 `request_cancelled`. Closing only the write side of the socket counts as a disconnect.
- Stopping or changing the model in Stark fails waiting and running requests with 503 `model_changed`; cancellation and deadlines keep the model loaded.
- The body limit and the worker's own 1 MiB request line are separate. A request whose rendered prompt (system prompt, template framing and all messages) exceeds the worker line limit currently fails with 503 `generation_failed` and unloads the model, which then needs **Run** again. Keep conversations well below 1 MiB of rendered text.

Each request is tokenized and prefilled from scratch by the worker; there is no prompt or prefix cache between requests.

## Error codes

Error responses are `{"error":{"message":...,"type":...,"code":...}}`. `type` is `authentication_error` for 401, `rate_limit_error` for 429, `server_error` for 5xx and `invalid_request_error` otherwise.

| Status | Code | Cause |
| --- | --- | --- |
| 400 | `invalid_request` | Duplicate headers, missing or invalid `Content-Length`, incomplete or non-object body, invalid field values |
| 400 | `length_required` | `Content-Length` missing (411 with the same code when `Transfer-Encoding` is present) |
| 400 | `invalid_json` | Malformed JSON, duplicate keys, `NaN` or infinity |
| 400 | `unsupported_parameter` | Penalties, log probabilities or response formats the API does not implement |
| 400 | `invalid_messages` | The chat template rejected the messages |
| 400 | `context_length_exceeded` | Prompt plus output limit exceeds the context budget |
| 401 | `invalid_api_key` | Missing or wrong bearer key |
| 403 | `origin_not_allowed`, `host_not_allowed` | `Origin` header present, or non-loopback `Host` |
| 404 | `not_found`, `model_not_found` | Unknown path, or `model` is not the running model |
| 413 | `body_too_large` | Body above 1 MiB |
| 415 | `unsupported_media_type` | `Content-Type` other than `application/json` |
| 429 | `server_busy`, `queue_full`, `queue_timeout` | Connection limit, queue limit, or queue wait expired |
| 499 | `request_cancelled` | Client disconnected |
| 503 | `model_not_loaded`, `model_changed`, `server_stopped`, `server_shutdown`, `request_stopped`, `streaming_unavailable`, `stop_sequences_unavailable`, `generation_failed` | No model in Chat mode, model stopped or changed, Stark quitting, worker lacks a capability, or native generation failed |
| 504 | `request_timeout` | 300-second deadline reached |
| 500 | `internal_error`, `http_error` | Unexpected failure |

## Client compatibility

Clients must send authenticated requests from native or backend code on the same Mac and use the text-only subset above. The following were checked against the published 0.3.0 bridge and worker on 2026-10-23 ([record](archive/runs/LOCAL_API_CLIENT_VALIDATION_2026_10_23.md)):

| Client | Configuration | Result |
| --- | --- | --- |
| `llm` 0.36 | Custom OpenAI model, key from environment, buffered and streaming | Passed, including three concurrent requests |
| `aichat` 0.30.0 | Custom OpenAI endpoint, buffered and streaming | Passed for single-turn text |
| Cherry Studio 2.1.7 | Custom OpenAI provider, Chat mode, streaming on, overrides off | Discovery and fresh chats passed; follow-up turns returned 400 because the client sent `reasoning_content`, which 0.3.0 rejects and current source accepts |
| Browser page | Cross-origin JSON request with bearer key | Blocked, as designed |

A client appearing here does not mean its agents, tools or other workflows are supported.

## Compatibility boundary

The surface follows the `/v1/models` and `/v1/chat/completions` shape used by local engines such as [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) and [LM Studio](https://lmstudio.ai/docs/developer/openai-compat), restricted to text templates, a single model and a bounded worker. Client libraries that require tools, multimodal content, JSON-schema output, embeddings, the Responses API, batching or vendor-specific sampling fields need a configuration that stays within the text-only subset.
