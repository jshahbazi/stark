---
title: "Connect another application"
description: "Configure a native client with Stark's local API settings."
---

## Configure the provider

1. Run a model in **Chat** mode in Stark.
2. Open **Connections → Connect your own app**.
3. In your client, choose an OpenAI-compatible provider.
4. Copy **Base URL**, **API key** and **Model ID** into the corresponding fields.
5. Send a short message and confirm the client receives a response.

Use the displayed Base URL. The default is `http://127.0.0.1:8000/v1`, but the port can be changed.

Some clients ask for a base URL; others ask for a complete endpoint. Check the client's instructions to avoid adding `/v1` twice.

## Check compatibility

Stark implements a subset of the OpenAI API. A client that requires embeddings, the Responses API, batching or structured output will need another supported route. See the [API reference](/docs/api/).

The API listens on this Mac and rejects browser-origin requests. A hosted website or another computer cannot use it directly.

The connection-window model picker inspects IDs; it does not switch the running model. Use **Run** in Stark before changing the client to a different model ID.

## Keep the key private

The key is a credential. **Copy All as JSON** includes it even if the window hides it. Paste it only into trusted client settings and keep it out of source control, screenshots and support reports.

If you regenerate the key, update clients that use the copied key. Managed OpenWork and OpenCode connections refresh through Stark.

[Client compatibility details](https://github.com/jshahbazi/stark/blob/main/docs/LOCAL_API.md#client-compatibility)

