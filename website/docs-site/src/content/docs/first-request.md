---
title: "Make your first API request"
description: "Send an authenticated request to the model running in Stark."
---

## Prerequisites

[Install Stark](/docs/install/), run a model in **Chat** mode and keep the app open. You need a terminal on the same Mac.

## Send the request

1. Open **Connections → Connect your own app → Advanced settings**.
2. Choose **Copy curl Example**.
3. Paste the generated command into your terminal and run it.
4. When prompted, paste the API key from Connections. The terminal hides your input.

The generated command uses your current connection details and sends authentication to curl through standard input. It does not put the key in command arguments or export it as an environment variable.

## Understand the request

The HTTP request has this shape. The example below illustrates the protocol; use the app-generated curl command to execute it.

```http
POST /v1/chat/completions
Authorization: Bearer <API_KEY>
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "Reply with OK."}
  ],
  "stream": true,
  "max_tokens": 32
}
```

The model may answer differently. A successful stream contains `data:` lines with JSON objects, followed by `data: [DONE]`. Response text arrives in `choices[0].delta.content`; a chunk may carry a role or other metadata without text.

For a buffered JSON response, set `stream` to `false`. Read the result from `choices[0].message.content`. Treat generated content as untrusted input in your application.

## Fix common failures

| Result | What to check |
| --- | --- |
| Connection refused | Stark is open, its API started, and the URL matches Connections |
| 401 `invalid_api_key` | Use the current key from this installed build |
| 404 `model_not_found` | Omit `model` or use the running model's ID |
| 503 `model_not_loaded` | Run a model in Chat mode |
| 400 `context_length_exceeded` | Shorten the input or reduce the requested output |

Do not send keys or conversation content in bug reports. Include the error code and app version.

## Next steps

Use the [HTTP API reference](/docs/api/) for model discovery, capability checks, tools and streaming errors. Choose [native workers](/docs/workers/) if you need to supervise the inference process directly.

