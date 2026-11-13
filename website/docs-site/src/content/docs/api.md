---
title: "HTTP API reference"
description: "Discover capabilities and find the exact request contract."
---

## Connection

The menu bar app serves authenticated HTTP on `127.0.0.1`. Read the actual Base URL and key from **Connections → Connect your own app**. Every endpoint requires bearer authentication.

Start with [your first request](/docs/first-request/) before implementing retries or tools.

## Endpoints

| Endpoint | Use |
| --- | --- |
| `GET /health` | Check the service |
| `GET /v1/models` | Read the running model ID and reported limits |
| `GET /v1/capabilities` | Discover the current API capabilities |
| `POST /v1/chat/completions` | Send buffered or streamed chat |

[Endpoint and authentication contract](https://github.com/jshahbazi/stark/blob/main/docs/LOCAL_API.md#endpoints)

## Request behavior

Send a non-empty `messages` array. The `model` field may be omitted to use the running model. Use either `max_tokens` or `max_completion_tokens` for an explicit output limit. Input and requested output must fit the applicable context and resource budget.

Read limits at runtime. A reported `context_length: null` means the model has no declared finite maximum; it does not promise unlimited execution.

[Chat fields and limits](https://github.com/jshahbazi/stark/blob/main/docs/LOCAL_API.md#chat-completions)

## Streaming and errors

Streaming uses Server-Sent Events and terminates successfully with `data: [DONE]`. Handle an error object during a stream and a connection closing before completion. Validate the result before acting on it.

[Errors, queueing and cancellation](https://github.com/jshahbazi/stark/blob/main/docs/LOCAL_API.md#limits-queueing-and-cancellation)

## Tools and images

Check the loaded model's capabilities before sending tools or images. The client executes tools and returns tool results; Stark renders supported templates and parses supported tool-call formats.

Image requests accept one inline PNG/JPEG data URL, with a 4 MiB decoded-file limit at the HTTP service. Remote image URLs are rejected. Image input requires matching reference-verified or locally tested execution evidence. Check `capabilities.images` on the running model.

[Tool contract](https://github.com/jshahbazi/stark/blob/main/docs/LOCAL_API.md#function-tools) · [Image qualification](https://github.com/jshahbazi/stark/blob/main/docs/LOCAL_API.md#image-qualification)

## Unsupported API features

Embeddings, the Responses API, structured output and batching are unavailable. Browser-origin requests and connections from remote devices are rejected.

The linked [Local API contract](https://github.com/jshahbazi/stark/blob/main/docs/LOCAL_API.md) owns the detailed schema, defaults and error codes. Check the release notes when your installed version differs from current source.

