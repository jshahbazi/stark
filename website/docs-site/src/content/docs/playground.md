---
title: "Use Playground"
description: "Chat with the loaded model and understand where the conversation lives."
---

## Start a conversation

1. Run an installed model in **Chat** mode.
2. Open **Playground** or **Open chat** from Stark.
3. Send a short message and wait for the response before sending a follow-up.

Playground talks to the model running in Stark. Changing a model selection in connection settings does not start that model; use **Run** in model management.

## Work with a longer conversation

Each follow-up needs conversation context. If a request exceeds the context budget or available memory, shorten the conversation or start a new one. Keep important instructions concise.

Supported text paths can reuse an unchanged prefix from the immediately preceding request. This reduces repeated input processing when the required settings match. It does not increase the model's context capacity or make its answers more accurate.

## Ask about an image

Use one PNG or JPEG with a model whose image input is enabled after qualification. Image availability depends on the exact artifact and build. If attachment is unavailable, check [image compatibility](/docs/compatibility/#images) before importing another model.

## Decision mode

Decision mode compares supplied choices using a compatible scoring model. The reported choice probabilities are relative to the supplied candidates. They are not a measured probability that an answer is factually correct.

See the [scoring reference](https://github.com/jshahbazi/stark/blob/main/docs/SCORING.md) for the probability definitions and limits.

## Conversation lifetime

Playground conversations stay in memory. App updates and restarts end the active conversation. Save any work you need before quitting or upgrading.

For more controls and settings, see the [app reference](https://github.com/jshahbazi/stark/blob/main/apps/menu-bar/README.md).

