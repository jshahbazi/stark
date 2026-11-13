---
title: "Manage models and storage"
description: "Download, import, run and remove models."
---

## Add a catalog model

Open **Advanced → Manage Models → Model Catalog**, choose a supported model variant and download it. After preparation finishes, choose **Run** in **Installed**.

Downloading, preparing and running are separate steps. Wait until the app reports the model is running before connecting another application.

## Import a checkpoint

Use **Add from Source** for a supported checkpoint. Check its architecture, weight format and template against the [converter reference](https://github.com/jshahbazi/stark/blob/main/tools/converter/README.md) first.

Adapter-only files must be merged with their base model before import. Successful conversion establishes that the artifact was prepared; it does not qualify answer quality, tool behavior or image support.

## Switch models

Stop the current work before running a different model. Requests waiting or running against the old model can fail when it changes. Recheck the model shown in **Connections** and in your client after switching.

## Remove a saved model

Use the model-removal action in model management and review the confirmation. Removing a saved model and uninstalling Stark are different operations. Do not remove files from the model library while a model is running.

Keep space available for both downloads and prepared artifacts. If setup reports insufficient space, use the paths and estimates shown by the app to decide what to remove.

The [model-management reference](https://github.com/jshahbazi/stark/blob/main/docs/MODEL_SUPPORT.md#browsing-and-removing-models) describes storage and removal behavior.

