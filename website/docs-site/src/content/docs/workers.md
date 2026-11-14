---
title: "Native worker reference"
description: "Run generation and scoring processes under your application's control."
---

## Choose the worker interface

Use the [HTTP API](/docs/first-request/) when the menu bar app should manage the running model and render chat templates.

Native workers expose inference over private stdin/stdout pipes. Your application owns process supervision, templates, tool execution and its agent loop. No Stark SDK package is published.

## Start a worker

Install the terminal tools using the [installation reference](https://github.com/jshahbazi/stark/blob/main/docs/INSTALLATION.md#install-the-app-and-terminal-tools), then launch the appropriate worker with a prepared model:

```sh
stark generate --model-path /path/to/prepared-model
```

For scoring:

```sh
stark score --score-model-path /path/to/prepared-model
```

The paths are placeholders. Use an artifact prepared for the corresponding mode.

## Follow the process lifecycle

1. Read the readiness reply before sending a request.
2. Write UTF-8 JSON-lines messages using the worker's schema.
3. Serialize complete request transactions against a loaded model.
4. Consume streaming events through the terminal reply.
5. Handle failure and process exit before admitting another request.

Do not treat worker stdout as ordinary application logging. It carries protocol messages.

## Reference contracts

| Reference | Contents |
| --- | --- |
| [Worker protocols](https://github.com/jshahbazi/stark/blob/main/docs/WORKER_PROTOCOLS.md) | Readiness, framing, versioning and shared errors |
| [Generation](https://github.com/jshahbazi/stark/blob/main/docs/GENERATION.md) | Requests, streams, stops and cancellation |
| [Scoring](https://github.com/jshahbazi/stark/blob/main/docs/SCORING.md) | Candidate IDs, logits and probability definitions |

Keep prompts, responses, token vectors and scoring probabilities out of logs. Artifact support and numerical qualification belong to the exact model and build being used.

