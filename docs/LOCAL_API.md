# Stark local API

Stark exposes a private, loopback-only HTTP API from the menu-bar application. The API is intended for local OpenAI-style clients and is bounded by the loaded model and the native generation worker. It is not a hosted service and does not enable inherited cloud, release, or signing infrastructure.

## Connection and authentication

The application emits a connection record containing a base URL, such as `http://127.0.0.1:<port>/v1`, and an ephemeral bearer key. The key is held in memory and must be sent as:

```text
Authorization: Bearer <key>
```

The server accepts loopback hosts only (`127.0.0.1`, `localhost`, and `::1`) and rejects requests carrying an `Origin` header. Duplicate `Host` or `Authorization` headers are rejected. Responses include an `X-Request-Id` value and `Cache-Control: no-store`. Authentication failures use an authentication error type; malformed requests use an invalid-request type; server failures use a server-error type.

The API accepts JSON request bodies up to 1 MiB. Chunked requests are rejected. When `Content-Type` is supplied, it must be `application/json`. Duplicate JSON keys and JSON `NaN` or infinity values are rejected.

## Endpoints

The following endpoints are available under the emitted `/v1` base path:

| Endpoint | Behavior |
| --- | --- |
| `GET /v1/health` and `GET /health` | Returns the current safe application status. |
| `GET /v1/capabilities` | Returns safe capability metadata for the loaded worker. |
| `GET /v1/models` | Lists the active model, or an empty list when no model is running. |
| `GET /v1/models/{id}` | Returns metadata for the matching running model, or 404. |
| `POST /v1/chat/completions` | Runs one text completion, buffered or streamed. |

The API does not expose embeddings, the Responses API, image input, structured output, tool calls, function calls, or batching. It runs one completion at a time through one loaded model.

## Chat completions

Requests accept `model`, `messages`, `stream`, `stream_options`, `max_tokens`, `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, `n`, `presence_penalty`, `frequency_penalty`, `logprobs`, `top_logprobs`, `response_format`, `user`, and `metadata`.

`model` may be omitted when the application selects the active model. The active model ID and its installed name are accepted. `messages` must be non-empty. Message roles `system`, `developer`, `user`, and `assistant` are accepted; `developer` is normalized to `system`. Content may be a string or an array containing only `{ "type": "text", "text": "..." }` parts. Text parts are concatenated. An optional ASCII identifier `name` up to 64 characters is validated and ignored because the native templates do not consume message names. Tool and function-call fields are rejected.

The default output limit is the smaller of 512 and the worker limit. `max_tokens` and `max_completion_tokens` are aliases and cannot both be supplied with values. The requested limit must be at least one and must fit the loaded worker. `n` must be 1.

Temperature defaults to 0 and selects greedy sampling. A positive finite temperature up to 2 selects stochastic sampling. `top_p` may be supplied as a finite value greater than 0 and up to 1; it is passed to stochastic sampling and accepted without effect for greedy sampling. `top_k` accepts a positive integer within the model vocabulary and `min_p` accepts a finite value from 0 through 1. These filters apply to stochastic sampling; the temperature-0 path accepts them without effect. `seed` defaults to 0 and is an unsigned 64-bit value. Nonzero presence and frequency penalties, requested log probabilities, and response formats other than `{ "type": "text" }` are rejected. Neutral compatibility values such as `logprobs: false`, `top_logprobs: 0`, and zero penalties are accepted. `user` and `metadata` are accepted within their bounds and are not logged or retained as conversation state.

`stop` may be one string or a list of up to four non-empty strings, each at most 1024 Unicode characters. Stop sequences are applied by the native worker to decoded text and can span generation chunks. Matching stop text and later generated text are withheld. The completion usage count includes the sampled token that completed a stop sequence, while the returned text excludes the stop sequence.

For `stream: true`, the response is Server-Sent Events using OpenAI-style completion chunks, followed by `[DONE]` on successful completion. The first chunk carries the assistant role. `stream_options: { "include_usage": true }` adds a final usage-only chunk. Streaming responses disable proxy buffering with `X-Accel-Buffering: no`.

## Limits, queueing, and cancellation

The HTTP server admits at most 16 connections. Excess connections receive HTTP 429 with `Retry-After: 1`. At the application layer, up to eight completion requests wait in FIFO order behind the active generation. A queued request waits at most 30 seconds and then receives HTTP 429. Model epochs fence queued and running requests when the model is stopped or changed.

For cancellable application requests, the HTTP handler monitors the client every 0.05 seconds after the body is read. A peer close or the 300-second request deadline sets the cancellation event. Native generation checks cancellation at safe GPU generation boundaries. A deadline is reported as a gateway-timeout response; a client cancellation is reported as a client-closed request when a response can still be written. The loaded worker is preserved after a normal request cancellation. Older workers without the cancellation channel are fenced and may need Run to reload; reinstall the app for cancellation with model reuse.

The Python worker uses a private inherited pipe for cancellation. Each generate request receives a monotonically increasing unsigned 64-bit request ID. Cancellation frames are exactly eight little-endian bytes containing that ID. The native worker polls the pipe, handles partial frames, scopes cancellation to the matching request, and treats owner EOF as cancellation. The pipe and request state are private and are not persisted.

## Compatibility boundary

The surface follows the common `/v1/models` and `/v1/chat/completions` shape used by local engines such as [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) and [LM Studio's OpenAI compatibility layer](https://lmstudio.ai/docs/developer/openai-compat), while retaining Stark's text-template, single-model, bounded-worker constraints. Client libraries that require tools, multimodal content, JSON-schema output, embeddings, Responses, batching, or vendor-specific sampling fields must use a client configuration that targets the supported text-only subset.

Model-free tests and a local real-worker smoke cover native stop filtering, seeded sampling on one Qwen3 checkpoint, cancellation before prefill and during streaming, worker reuse, stale cancellation fencing, and capability discovery. These results do not establish broader model quality or performance equivalence. Model state is serialized against the single loaded model. This API does not introduce persistent prompt caching or a prefix cache; conversation content, generated text, token vectors, decision probabilities, credentials, and exception payloads are not logged.
