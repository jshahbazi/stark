# Conversion benchmarks

Complete-command conversion time and sampled process-tree memory for `stark-convert` against the pinned Lalamo reference converter, collected across converter milestones. All rows target BF16 output. The speed ratio is the Lalamo median divided by the Stark median; values above 1 favour Stark. Each row links to the milestone record under the archive, which holds the exact checkpoint, converter revision, full ranges and review.

## Existing MLX 4-bit and 8-bit checkpoints

| Checkpoint | Stark median | Lalamo median | Speed ratio | Lower sampled RSS |
| --- | ---: | ---: | ---: | ---: |
| [LFM2-350M 4-bit](../archive/conversion/MODEL_CONVERSION_MILESTONE_4_MLX.md) | 0.532 s | 1.853 s | 3.48 | 92.0% |
| [LFM2-350M 8-bit](../archive/conversion/MODEL_CONVERSION_MILESTONE_4_MLX.md) | 0.623 s | 1.581 s | 2.54 | 92.6% |
| [Qwen3.5-0.8B text 4-bit](../archive/conversion/MODEL_CONVERSION_MILESTONE_4_MLX.md) | 0.997 s | 2.059 s | 2.07 | 94.9% |
| [Qwen3.5-0.8B text 8-bit](../archive/conversion/MODEL_CONVERSION_MILESTONE_4_MLX.md) | 1.393 s | 1.940 s | 1.39 | 95.1% |
| [Qwen3-0.6B 4-bit](../archive/conversion/MODEL_CONVERSION_MILESTONE_3.md) | 1.011 s | 1.796 s | 1.78 | 93.0% |
| [Qwen3-0.6B 8-bit](../archive/conversion/MODEL_CONVERSION_MILESTONE_3.md) | 1.506 s | 1.818 s | 1.21 | 93.3% |
| [Llama3.2-1B 4-bit](../archive/conversion/MODEL_CONVERSION_MILESTONE_3_LATENCY.md) | 1.256 s | 2.397 s | 1.91 | 94.2% |
| [Llama3.2-1B 8-bit](../archive/conversion/MODEL_CONVERSION_MILESTONE_3_LATENCY.md) | 2.202 s | 2.604 s | 1.18 | 96.6% |
| [SmolLM3-3B 4-bit](../archive/conversion/MODEL_CONVERSION_MILESTONE_3_LATENCY.md) | 2.779 s | 3.377 s | 1.22 | 95.4% |
| [SmolLM3-3B 8-bit](../archive/conversion/MODEL_CONVERSION_MILESTONE_3_LATENCY.md) | 5.083 s | 4.233 s | 0.83 (slower) | 96.2% |
| [Gemma3-1B 4-bit](../archive/conversion/MODEL_CONVERSION_MILESTONE_3_LATENCY.md) | 1.551 s | 2.580 s | 1.66 | 90.7% |
| [Gemma3-1B 8-bit](../archive/conversion/MODEL_CONVERSION_MILESTONE_3_LATENCY.md) | 2.464 s | 2.537 s | 1.03 | 90.8% |
| [LLaMBA-1B 4-bit](../archive/conversion/MODEL_CONVERSION_MILESTONE_5.md) | 2.142 s | 2.225 s | 1.04 | 96.4% |

The sequential coefficient writer introduced in the [latency milestone](../archive/conversion/MODEL_CONVERSION_MILESTONE_3_LATENCY.md) reduced conversion medians by 16 to 38 percent against the previous Stark implementation across the six Llama, SmolLM3 and Gemma tuples while preserving exact output bytes. MLX conversion repacks an already quantized checkpoint; it does not compute new quantization.

## Dense models and classifier fixtures

| Checkpoint | Stark median | Lalamo median | Speed ratio | Lower sampled RSS |
| --- | ---: | ---: | ---: | ---: |
| [LFM2-350M dense](../archive/conversion/MODEL_CONVERSION_MILESTONE_4.md) | 1.233 s | 1.688 s | 1.37 | 90.2% |
| [Qwen3.5-0.8B text dense](../archive/conversion/MODEL_CONVERSION_MILESTONE_4.md) | 3.525 s | 2.707 s | 0.77 (slower) | 91.4% |
| [LLaMBA-1B dense, F32 source](../archive/conversion/MODEL_CONVERSION_MILESTONE_5.md) | 5.791 s | 2.105 s | 0.36 (slower) | 97.1% |
| [ModernBERT, BF16 source fixture](../archive/conversion/MODEL_CONVERSION_MILESTONE_6.md) | 0.562 s | 1.137 s | 2.02 | 84.9% |
| [ModernBERT, F32 source fixture](../archive/conversion/MODEL_CONVERSION_MILESTONE_6.md) | 0.861 s | 1.312 s | 1.52 | 89.6% |

The ModernBERT rows use unchanged public classifier weights with recorded synthetic chat-template overlays, because the two catalog router repositories are not accessible.

## Slow cases and method

SmolLM3 MLX 8-bit is 20.1 percent slower than Lalamo, dense Qwen3.5 is 30.2 percent slower, and dense LLaMBA takes 2.75 times as long while still cutting sampled memory from 5,817.6 MiB to 167.0 MiB. The main reason dense conversion is slower is deliberate: Stark hashes every consumed checkpoint byte before and after conversion to verify the source, on top of the conversion read itself, while the Lalamo command performs weaker source checks. Several smaller differences (Llama 8-bit, Gemma 8-bit, LLaMBA MLX, ModernBERT F32, dense LFM2) have overlapping ranges.

Each measurement is the median of serial fresh processes with warm source files on one Apple silicon host (three repeats for the initial Qwen3 packet, five for the others). Setup, downloads, warming, parity comparison and native loading are excluded from command time. Process-tree RSS was sampled every 10 ms, which can miss peaks or double-count shared pages; the 8 MiB transformation workspace bound does not include headers, tokenizer processing, hashing or interpreter memory. These figures describe conversion only. Inference speed and answer quality are measured separately.
