---
title: "Choose a model"
description: "Match a model to your task, available memory and required capabilities."
---

## Start with Guided setup

Use **Beginner → Guided setup** for an initial recommendation. Stark checks model compatibility, memory and storage. Review the exact model variant before downloading.

Recommendations help narrow the catalog. They do not establish that a model will handle every task well. Try a small example of your actual work before committing to a large download or long task.

Newer development builds offer **Quality** and **Speed** priorities. Their rankings use a bundled whatmodel.sh snapshot and parent-model scores. Those scores do not measure every quantized export. See the [recommendation policy](https://github.com/jshahbazi/stark/blob/main/docs/MODEL_RECOMMENDATIONS.md) for that build's rules.

## Decide what you need

| Task | Check before downloading |
| --- | --- |
| Chat or writing | A usable chat template and enough memory for your conversation |
| File tasks or coding agents | Tool support for the exact model and template |
| Image questions | An admitted vision artifact with image input enabled |
| Decision scoring | Compatibility with Decision mode and the scoring workflow |

## Understand memory and context

Models share your Mac's unified memory with macOS and other applications. A model that fits by itself may fail when other workloads consume memory.

Context is the input the model can consider in one request, including conversation history. Longer conversations require more working memory. A published context limit does not establish that the full length is practical on your Mac.

Quantization reduces weight storage and often memory use. Its effect on quality and speed depends on the model and implementation. Compare exact checkpoints and quantizations when reading measurements.

## Compare models using your work

Run a short task with a clear expected result. For coding, check whether the code runs and whether tool calls work. For writing, check accuracy and whether revisions follow instructions. Record the model variant when comparing results.

Use [model compatibility](/docs/compatibility/) to check support, then [manage your installed models](/docs/manage-models/).

