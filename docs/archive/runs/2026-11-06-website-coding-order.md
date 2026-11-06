# Website Text model ordering

The user requested coding-benchmark ordering without displaying scores, with
comparable published models first and other families afterward. The new dense
27B performance tab was removed from the website before publication; the
repository's source-bound benchmark reports are preserved.

## Ordered entries

1. Qwen3.8-Flash-Next.
2. Qwen3.8-27B oQ4e MTP.

The publisher evaluates both reference models in the same comparison table.
Flash-Next leads the dense 27B model on the listed competitive-coding and
software-engineering rows. This establishes the relative reference-model order;
it does not qualify coding accuracy for Stark's quantized artifacts.

Primary sources checked on 2026-11-06:

- [Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/7ecef55a2e549814c62143fcf8b5be457695e3b9/README.md),
  Benchmark Results: LiveCodeBench v6, DeepSWE 1.1, SWE-bench Pro,
  SWE-bench Multilingual and NL2Repo-Bench.
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/8c0b3d30f3ae96f5c7f774de704bf0ada83e422a/README.md),
  Benchmark Results, as a cross-check of the dense reference model.

No numeric scores, score attributes, coding leaderboards or new performance
dataset are included in the website. Its short ordering note identifies the
publisher-reference scope and the possibility of different local quantized
results.

## Other entries

The remaining fourteen Text entries retain their relative order in a separate
unranked group. Several contain multiple sizes or fine-tunes, and the inspected
sources do not establish one shared benchmark rank for those entire groups.
This does not mean they lack coding ability or published evaluations.

In particular, GLM's agentic evaluations use their own harness/context settings;
the inspected SmolLM3 card reports a different LiveCodeBench version; Gemma 3
text includes both FunctionGemma and rnj-1 rather than a single model. Those
results are not flattened into a fabricated family-wide score.

The app's hardware-fit recommendations and Stark's decode-throughput records
are also not coding-quality scores. This change reorders the website's Text
directory only. Image and Decision groups, the app's catalog order, and the
three existing website benchmark tabs retain their behavior.
