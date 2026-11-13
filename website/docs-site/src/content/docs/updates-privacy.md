---
title: "Updates and privacy"
description: "Understand what an upgrade preserves and where data goes."
---

## Update Stark

The integrated updater checks stable release metadata and can download and verify a newer app in the background. Choose **Upgrade to VERSION** to install it and restart.

Save work before upgrading: active requests stop and the in-memory Playground conversation ends. Models, preferences and the Keychain API credential are preserved. The app updater does not update the terminal tools.

If preparation fails, read the displayed error and retry the upgrade. For manual installation and uninstalling, see the [installation reference](https://github.com/jshahbazi/stark/blob/main/docs/INSTALLATION.md#update).

## Local processing

Stark runs inference on your Mac. Its local API does not log prompts or responses, and Playground conversations stay in memory.

The API listens on loopback, requires authentication and rejects browser-origin requests. The persistent API key is stored in macOS Keychain. Keep copied credentials and connection JSON private.

## Network access

Model downloads and app updates use the internet. Connected applications, plugins and tools can contact external services under their own settings and policies. Review those separately before using private data.

The website adds no analytics or advertising trackers. Documentation search uses a static index shipped with the site.

## Connection permissions

Managed OpenWork and OpenCode connections use renewable capabilities. Disconnect through Stark to revoke that client's managed access. Plugins in an authorized client process share that process's access.

[API and privacy boundaries](https://github.com/jshahbazi/stark/blob/main/docs/LOCAL_API.md) · [Security policy](https://github.com/jshahbazi/stark/blob/main/SECURITY.md)

