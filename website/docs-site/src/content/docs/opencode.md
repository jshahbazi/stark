---
title: "Connect OpenCode Desktop"
description: "Use Stark for chat and coding tasks in OpenCode Desktop."
---

## Before you connect

Install the official [OpenCode Desktop app](https://opencode.ai). Run a model in Stark's **Chat** mode. For coding tasks that use tools, check **Tool use available** in Connections.

The managed connection applies to OpenCode Desktop. A standalone or Homebrew CLI running its own engine uses the [manual connection](/docs/other-apps/).

## Connect

1. Open **Connections → Apps** in Stark.
2. Choose **Connect** on the OpenCode Desktop row. Stark opens Desktop if needed and configures its connection.
3. In OpenCode, open a project and select **Stark** and the running model.
4. If Connections shows **Allow** and **Deny** for a new engine, review it and choose **Allow** to connect.

Stark approves each verified engine instance. An update or a new engine after login can require approval again.

## Verify the connection

Send a short chat request. Then ask OpenCode to explain one small source file and compare the answer with that file. Check that it can actually read the file before starting a larger coding task.

## If the model is missing

Check the running model in Stark and any pending approval in Connections. If the picker still does not update, quit OpenCode, run `opencode service stop` in a terminal, reopen Desktop and approve the new engine.

Stopping the background service interrupts sessions using that engine. Save ongoing work first.

## Disconnect

Use **Manage → Disconnect** on the OpenCode Desktop row. This revokes OpenCode's grants and clears approved engines. OpenWork keeps its own connection.

The approved engine is shared with sessions and plugins using it. Those components share the authorized process and its access.

[OpenCode connection reference](https://github.com/jshahbazi/stark/blob/main/docs/LOCAL_API.md#managed-opencode-desktop-connection)

