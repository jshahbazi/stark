---
title: "Model, tool and image compatibility"
description: "Check capabilities for the exact model and installed build."
---

## Check your installed build first

The website and repository can describe features newer than your installed release. Read the [release notes](https://github.com/jshahbazi/stark/releases/latest), then check the capability shown by the running model.

Family names alone are insufficient: checkpoint, weight format, template and build all affect support.

## Chat and tools

Chat requires a template Stark can render. Tools additionally require a reviewed wire format and matching parser. Check **Tool use available** in Connections before using an agent application.

Some newer builds show **Tools available**, **Tools unavailable** or **Tools unchecked** for installed models. An unchecked artifact has not established tool compatibility.

[Tool-calling coverage](https://github.com/jshahbazi/stark/blob/main/docs/TOOL_CALLING.md)

## Images

Use one PNG or JPEG only when the loaded artifact has passing local image evidence. Image input can be enabled by a matching reference-verified execution record or passing local checks. Use the running model's `capabilities.images` value. Successful text generation or conversion does not enable image input.

Qwen and Gemma image coverage varies by artifact and build. The status reference distinguishes implementation, component comparisons and full-checkpoint execution evidence. A development-branch result does not establish availability in a released app.

[Image status by family and checkpoint](https://github.com/jshahbazi/stark/blob/main/docs/MODEL_IMAGE_STATUS.md) · [Vision admission contract](https://github.com/jshahbazi/stark/blob/main/docs/VISION_FAMILY_CONTRACT.md)

## Imported models

Use a complete checkpoint in a supported format. Merge adapters with their base model before importing. Each new export needs its own compatibility and numerical assessment.

[Supported families and formats](https://github.com/jshahbazi/stark/blob/main/docs/MODEL_SUPPORT.md) · [Import reference](https://github.com/jshahbazi/stark/blob/main/tools/converter/README.md)

## Performance evidence

Compare measurements with matching model, quantization, hardware and workload. Throughput and short output agreement do not establish answer-quality equivalence.

[Benchmark overview](/technical.html#benchmarks) · [Source-bound reports](https://github.com/jshahbazi/stark/blob/main/docs/benchmarks/README.md)

