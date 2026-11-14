---
title: "Introduction"
description: "Run a model, chat with it, or connect an application."
---

Stark runs open-weight language models on Apple silicon Macs. The menu bar app downloads and starts models. Playground provides chat, and the local HTTP API lets other applications use the running model.

## Start with a working model

Follow [Install Stark and run your first model](/docs/install/). You will finish with a model loaded and a response in Playground.

Once that works, you can [connect OpenWork](/docs/openwork/) for file tasks, [connect OpenCode Desktop](/docs/opencode/) for coding, or [send an API request](/docs/first-request/) from your own program.

## How the pieces fit together

| Part | What it does |
| --- | --- |
| Stark app | Installs models, starts the engine and manages connections |
| Model | Supplies the learned behavior; quality and supported inputs vary |
| Playground | Sends chat or decision requests to the loaded model |
| Connected application | Provides its own interface, file access, tools and approval flow |
| Local API | Accepts authenticated requests from applications on the same Mac |

A model that can chat may still lack a supported tool format or image input. Check [compatibility](/docs/compatibility/) before choosing one for a particular application.

## Requirements

Use an Apple silicon Mac running macOS 26.4 or newer. Downloads require internet access and disk space. Running a model also requires available unified memory; the download size alone does not tell you whether it will fit.

[Choose a model](/docs/models/) explains the practical tradeoffs. [Troubleshooting](/docs/troubleshooting/) covers installation, loading and connection failures.

[Stark website](/) · [Performance studies](/technical.html#benchmarks)

