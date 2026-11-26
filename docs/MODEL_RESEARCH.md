# Stark model research and benchmarks

Start with the model below to see what changed, how much time it saved or cost, and why a candidate was kept or dropped. Follow a result to its dated report for the checkpoint, worker build, precision, workload, host conditions and correctness controls.

| Model or question | Where to look |
| --- | --- |
| Qwen3.8-Flash-Next, Lily Q4-XL, Pipe and official mixed packs | [Decode, prefill, MTP, cache and startup changes](model-experiments/speed-flash-next.md) · [October optimization history](benchmarks/QWEN38_OPTIMIZATION_HISTORY.md) · [All indexed Flash-Next reports](model-experiments/source-index.md#qwen38-flash-next-and-lily) |
| Dense Qwen3.8-27B, Jundot oQ4e MTP | [Packing, draft depth, idle wake and image-loaded parity](model-experiments/speed-qwen.md#dense-qwen38-27b) · [Kernel and scheduling decisions](benchmarks/2026-11-06-qwen38-27b-approaches.md) |
| Qwen2, Qwen3, Qwen3.5 dense and 35B-A3B MoE | [GGUF prefill, prefix reuse, engine comparisons and image-loaded text](model-experiments/speed-qwen.md) · [Admission, vision, tools and artifact records](model-experiments/source-index.md#dense-qwen-qwen-moe-and-gguf) |
| GLM-5.3-Flash | [GGUF prefill, decode and EXL3 results](model-experiments/speed-other-families.md#glm-53-flash) · [Branch kernel, memory and image experiments](model-experiments/branch-research.md#glm-53-flash) |
| Tencent Hy4-flash, AngelSlim Q3_K | [Performance write-up, retained profiles and failed approaches](model-experiments/speed-hy4.md) · [Per-attempt results](model-experiments/speed-other-families.md#hy4-flash) · [Dated records](model-experiments/source-index.md#glm-flint-and-native-hybrids) · [Standard benchmark profile](../scripts/benchmark_profiles/hy4-flash.toml) |
| Gemma 2/3/4 and OLMo 3 | [Q4 repairs, fusions, image residency, cache and assistant MTP](model-experiments/speed-other-families.md#gemma-and-olmo) · [Image components, precision and runtime records](model-experiments/source-index.md#gemma-and-olmo) |
| Ripple-65B Preview | [Q8_0 GPU graph, Q4_K_M kernels and attention decode](model-experiments/speed-other-families.md#ripple-65b-preview) |
| Decider Flash, regular Decider, Elephant and JevK5 | [Native scoring latency, HTTP and prefill changes](model-experiments/speed-other-families.md#decision-models-and-scoring) · [Drex, Mercury and JevBench comparisons](model-experiments/branch-research.md#task-scoring-and-conversion-comparisons) · [Frozen JevK5 scoring parity](validation/SCORING_PARITY.md) |
| Edge0-145B | [Native and wider-prefill trials on closed PR #174](model-experiments/branch-research.md#edge0-145b) |
| Inkling 2 Mini | [Prefill, decode, DSpark and prefix reuse speed history](model-experiments/speed-inkling.md) · [Benchmarks, attempts and retrospective](model-experiments/source-index.md#inkling-2-mini) · [Prefill and decode proposals](INKLING_2_MINI_PERFORMANCE_FOLLOWUP.md) · [MLX-Serve engine matrix](benchmarks/2026-11-22-inkling-mini-engine-matrix.md) · [Prefill attribution against MLX-Serve](benchmarks/2026-11-22-inkling-mini-prefill-attribution.md) · [Every difference from MLX-Serve, what worked and what is open](model-experiments/inkling-mlx-serve-differences.md) · [Standard benchmark against MLX-Serve](benchmarks/2026-11-23-inkling-mini-standard-benchmark.md) |
| Recursive Flint, Phi-3.5, Mistral, ColGrep and Maxwell | [Native execution and qualification limits](model-experiments/speed-other-families.md#other-model-paths) · [Maxwell compatibility](model-experiments/branch-research.md#other-intake-and-qualification) · [Flint and GLM task pilot](model-experiments/branch-research.md#task-scoring-and-conversion-comparisons) |
| LFM2, Llama, LLaMBA, SmolLM3, ModernBERT and other converter targets | [Conversion time, memory and regressions by checkpoint and quantization](benchmarks/CONVERSION_BENCHMARKS.md) · [Milestones and reviews](model-experiments/source-index.md#conversion-plans-milestones-and-reviews) |
| Quality, tool use, context limits or images across families | [Runtime, artifact and tool studies](model-experiments/source-index.md#shared-runtime-catalog-and-image-qualification) · [Image capability inventory](MODEL_IMAGE_STATUS.md) · [External task-score provenance](benchmarks/2026-11-14-catalog-coding-ranking.md) |

**When adding a model or trying to improve its performance**, start with its family history and the playbook's guide to choosing an experiment. Search for earlier failures as well as wins, including on related architectures if the checkpoint is new. A superseding report may change the decision on the same artifact. [Choose a performance experiment](PERFORMANCE_PLAYBOOK.md#choose-a-performance-experiment) · [Experiment ledger](model-experiments/README.md) · [Benchmark supersession index](benchmarks/README.md) · [Historical Uzu and MeeCP comparisons](benchmarks/INFERENCE_COMPARISON_EVIDENCE.md)

**For a new speed result or engine comparison**, run the [standard benchmark](BENCHMARK.md) with the model's profile, so the result can be read next to every other model's, and index the record under the model above.

## Find a particular study

Browse the [source index by model and topic](model-experiments/source-index.md) for every benchmark report and dated model experiment in this tree, selected model-related archived runs, conversion records, validations and aggregate data. The [branch and PR index](model-experiments/branch-research.md) preserves useful work at its recorded revision, including unmerged experiments.

From the repository root, search by checkpoint, attempt ID, technique or metric. Each result contains all the terms. Quoting a phrase searches it literally.

```sh
uv run --python 3.13.13 python scripts/model_research.py ripple attention
uv run --python 3.13.13 python scripts/model_research.py "coarse" "27B" --limit 5
uv run --python 3.13.13 python scripts/model_research.py 20261119-LILYROWS-03
```

The search reads tracked Markdown in the documentation tree and the upstream provenance page. Branch results are discoverable through their local index and open at the linked commit. Archived reports can contain later follow-ups, so read the last dated entry before repeating a failed experiment. The speed tables select comparisons for quick reference. Reports preserve the complete matrices, earlier builds, excluded samples and aggregate receipt links.

## Research proposals with no Stark timing

| Prior assessment | What would settle it | Sources |
| --- | --- | --- |
| 2026-11-03 · Distilled latent sparse attention (D-LSA Flat) for GLM-5.3-Flash. The proposed experiment targets the cost of scoring pooled keys at long contexts. The published accelerator measurements belong to the paper's implementation. | Check checkpoint compatibility, then measure prefill, decode, cache memory and retrieval quality at context lengths that fit the Mac. The assessment produced no Stark implementation or speed result. | [Latent Sparse Attention paper](https://arxiv.org/abs/2610.42147) · [Authors' reference implementation](https://github.com/Prefrontal-AI/latent-sparse-attention) |

## Record the next result

1. Register the attempt and baseline using the [experiment template](model-experiments/TEMPLATE.md). Record the exact artifact, builds, host, timing boundary and workload once in the dated report.
2. Add the report under its model in the [source index](model-experiments/source-index.md). For a performance attempt add a row to the applicable family speed history. Give baseline and candidate in the same units, or the source-reported paired change, plus the decision and a link. A failed gate, invalid timing or unmeasured change gets that explicit outcome. Link a large per-attempt matrix once and call out material gains, losses and reversals.
3. Link superseding work from the older entry. Keep component and end-to-end figures distinct. For a simultaneous artifact or workload change, name both arms and report what can be attributed. Use commit-pinned links for evidence on another branch.
4. Read back the family history and relevant playbook section against the dated report. Check the comparator, measurement boundary, outcome and decision, including rejected variants and later reversals. For work on another branch, link its dated report at the measured revision.
5. Stage new records and run the index check below. It checks direct bibliography links for tracked benchmark reports and dated model experiments, and local file destinations on the research index pages. An attempt that was never written up is invisible to that check. The [Inkling session audit](model-experiments/2026-11-23-inkling-mini-session-audit.md) shows how comparing session evidence with the reports recovered omissions.

```sh
uv run --python 3.13.13 python scripts/model_research.py --check
```
