---
title: "Install Stark and run your first model"
description: "Install the Mac app, load a model and verify chat works."
---

## Before you begin

You need an Apple silicon Mac with macOS 26.4 or newer, internet access for downloads, and free storage. You do not need Rust, Xcode or Python to use the released app.

## Install the app

1. [Download Stark.dmg](https://github.com/jshahbazi/stark/releases/latest/download/Stark.dmg).
2. Open the disk image. If it contains **Install Stark**, open that launcher and wait for its completion message.
3. For an older drag-and-drop image, copy **Stark.app** into Applications, quit any running old copy, then open the installed copy.
4. Open Stark from its cat icon in the menu bar.

## Download and start a model

1. Open **Beginner → Guided setup** from the main menu.
2. Choose your intended work and follow the setup questions. Review the recommended model and its download requirements.
3. Choose **Download and set up**. Keep Stark open while the download and preparation finish.
4. Follow the prompt to run the model, then open chat.

Setup wording varies by version. Newer development builds include **General Work**, **Coding**, and **Quality** or **Speed** choices. If your release shows a different flow, use its guided steps. You can also use **Advanced → Manage Models → Model Catalog**, download a model, then choose **Run** in **Installed**.

## Check that it works

Open **Playground** in **Chat** mode and send a short message. You should see a response from the running model. Response wording varies; a completed response confirms the basic chat path.

If preparation fails, read the error before retrying. Check storage and available memory, then try a smaller model. See [loading problems](/docs/troubleshooting/).

## Next steps

[Choose a model](/docs/models/) explains memory and capability choices. To work with files or code, follow the [OpenWork](/docs/openwork/) or [OpenCode Desktop](/docs/opencode/) guide.

For terminal tools, source builds and uninstall instructions, use the [installation reference](https://github.com/jshahbazi/stark/blob/main/docs/INSTALLATION.md).

