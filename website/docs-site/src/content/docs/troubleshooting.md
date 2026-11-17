---
title: "Troubleshooting"
description: "Check the failing stage before repeating setup."
---

## Stark will not install or open

Confirm Apple silicon and macOS 26.4 or newer. Install the signed release from the official release page. For older drag-and-drop images, quit the previous copy before opening the installed copy.

Keep the specific installer error. Do not disable Gatekeeper or remove quarantine attributes to bypass a failure.

## A model will not download or run

Check the reported storage and memory requirements. Downloads and prepared models consume disk space; running consumes unified memory. Free space alone does not solve a memory error.

Close memory-heavy workloads or try a smaller model. Confirm the exact format is supported before importing. See [model selection](/docs/models/) and the [converter errors](https://github.com/jshahbazi/stark/blob/main/tools/converter/README.md).

## Playground works but the API does not

Open Connections and read the API status. The API can fail while chat remains usable.

If the port is occupied, close the conflicting server and choose **Retry API**, or choose another port under **Advanced settings → Apply Port**. Use the displayed Base URL and update clients after changing ports.

If Keychain access is pending or denied, allow access in Keychain Access and choose **Retry API**. If a pending operation remains after approval, quit and reopen Stark. Regenerate a corrupt key only through the explicit recovery action; copied-key clients need the replacement.

## An application cannot find Stark

Confirm a model is running in Chat mode. Check for pending approval in Connections. Follow the recovery steps for [OpenWork](/docs/openwork/) or [OpenCode Desktop](/docs/opencode/).

## Chat works but file tasks fail

Check **Tool use available** for the loaded model. Check the connected application's workspace, permissions and tool approvals. Then try reading one small file.

A valid connection does not establish that a model can reliably finish an agent task.

## Requests fail partway through

Check for model changes, cancellation, context limits and memory pressure. A streaming client must detect a stream ending without `[DONE]`. See [API error codes](https://github.com/jshahbazi/stark/blob/main/docs/LOCAL_API.md#error-codes).

## Report a problem

Include the app version, macOS version, Mac chip and memory, exact model variant, failing step and error code. Remove credentials, prompts, responses and private file contents. Contact [john@starkengine.app](mailto:john@starkengine.app) or use the [repository](https://github.com/jshahbazi/stark).

