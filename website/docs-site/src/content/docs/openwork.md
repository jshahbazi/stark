---
title: "Connect OpenWork"
description: "Use Stark's running model from the OpenWork Mac app."
---

## Before you connect

Install the official [OpenWork Mac app](https://openworklabs.com). Open a local workspace in OpenWork. In Stark, run a model in **Chat** mode and check **Tool use available** in Connections before trying file tasks.

## Connect

1. Open **Connections → Apps** in Stark.
2. Choose **Connect** on the OpenWork row.
3. If Stark asks you to reopen OpenWork, quit it completely and launch it again.
4. Select **Stark** and the running model in OpenWork's model picker.

This managed connection does not require copying the API key.

## Verify the connection

Send a short chat message in OpenWork. Once that works, ask it to read a small file in the selected workspace and check its answer against the file.

Chat success alone does not verify tool execution. OpenWork controls tools and file approvals; the model must also produce supported tool calls.

## If Stark does not appear

Confirm a model is running in Chat mode. Reopen OpenWork after connection setup. If you previously created a manual provider named `stark`, remove only that provider entry before connecting; preserve other providers.

After changing Stark's API port, choose **Manage → Reconnect** on the OpenWork row, then reopen OpenWork.

## Disconnect and privacy

Choose **Manage → Disconnect** to revoke OpenWork's managed connection and cancel its active requests. OpenCode Desktop has a separate connection.

Connected applications and plugins have their own data practices. OpenWork plugins share its authorized process, so install only plugins you trust.

[Managed connection reference](https://github.com/jshahbazi/stark/blob/main/docs/LOCAL_API.md#managed-openwork-connection)

