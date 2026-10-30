# Qwen3.8 sampled generation: Stark and MLX-Serve, 2026-10-30

The optimized Stark path measured **74.13 tok/s** pooled decode against
**58.53 tok/s** for stock MLX-Serve 26.10.3, a 26.65% difference in this run.
This compares two supported engine/checkpoint combinations with comparable
weight budgets. Their quantization profiles differ, particularly Q4 GDN in
Stark's Lily checkpoint versus Q8 GDN in the MLX pack. The result does not
isolate engine efficiency or establish equivalent model quality.

The machine was a MacBook Pro with an Apple M5 Max, 40 GPU cores, 128 GiB unified
memory and macOS 26.7.1 (25G241). AC power mode was Automatic (`powermode=0`).
The user confirmed other applications were closed. Downloads and builds had
finished before timing; complete requests and model loads were serialized.

## Results

Each engine has ten measured requests across two fresh loads. All requests
used 4,096 input tokens and produced 256 sampled tokens.

| Engine/checkpoint | Pooled decode tok/s | Median decode tok/s | Decode range | Pooled prefill tok/s | Median whole-call seconds |
| --- | ---: | ---: | --- | ---: | ---: |
| Stark / Lily q4-xl | 74.13 | 75.42 | 69.08–79.52 | 1,912.95 | 5.410 |
| MLX-Serve 26.10.3 / official mixed 4/8 | 58.53 | 58.01 | 56.41–61.00 | 1,948.23 | 6.534 |

Pooled rate is total measured tokens divided by total measured phase time.
Whole-call wall times include a worker pipe for Stark and HTTP for MLX, so
their transport paths differ. Phase rates use each engine's internal timing
boundaries rather than GPU hardware counters.

| Fresh-load block | Engine | Pooled decode tok/s | Pooled prefill tok/s |
| --- | --- | ---: | ---: |
| 0 | Stark | 78.04 | 2,086.70 |
| 1 | MLX-Serve | 58.64 | 2,025.12 |
| 2 | MLX-Serve | 58.43 | 1,876.97 |
| 3 | Stark | 70.60 | 1,765.91 |

Stark's final block was 9.54% slower than its first; the two MLX block rates
differed by 0.37%. The cause is unestablished. After removing the temporary
MLX checkpoint, a separate three-request Stark control reproduced its output
hash and measured 78.96 tok/s warm decode. An earlier standalone retention
control measured 77.18. Neither control replaces the paired result above.
The standalone controls use the worker's 255 decoded-step numerator; the
paired table consistently divides 256 output tokens by the decode interval
for both engines. This convention accounts for approximately 0.4% and should
be retained when reproducing or comparing these records.
These observations do not establish consistent 77–80 tok/s, or performance
during normal desktop multitasking.

## Checkpoints and precision

Stark used [Lily q4-xl](https://huggingface.co/fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl)
at revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, loaded directly through
the existing optimized executor. The full-Uzu experiment remains preserved on
`codex/qwen38-uzu-execution` and was not used for these measurements.

MLX used the unmodified [26.10.3 release](https://github.com/ddalcu/mlx-serve/releases/tag/v26.10.3)
and [official mixed-4-8bit pack](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit/blob/788c07a8725642bccf13a0a3b8a21954ab84b4f2/README.md)
at revision `788c07a8725642bccf13a0a3b8a21954ab84b4f2`. The published snapshot
is 107,326,639,344 bytes. Its license metadata names `qwen-community-1.0`.
The user authorized comparable quantization for this comparison.

| Representative weights | Lily | Official MLX mixed pack |
| --- | --- | --- |
| Routed experts | Q4 / group 64 | Q4 / group 64 |
| GDN projections | Q4 / group 64 | Q8 / group 64 |
| HC, QSA/indexer, shared experts, PLE projections, vocabulary head | Q8 / group 64 | Q8 / group 64 |
| Token embedding | Q8 / group 64 | Q4 / group 64 |
| MoE router | Q8 / group 64 | Q8 / group 64 |
| HC block-inject projections and shared-expert gate | Q8 / group 64 | BF16 |
| PLE ngram table | Q4 / group 32 | Q4 / group 32 |

A bounded audit checked 28 representative matrices across eight official
Safetensors headers against the Lily inventory. It confirmed the listed
stored widths; the model card's BF16-router description is stale for the
sampled router. This is not an exhaustive precision or checksum audit of every
tensor. Activations, reductions and recurrent arithmetic also depend on each
executor. Additional GDN weight traffic matters when interpreting the rates.

Direct stock-MLX loading of Lily failed with `EmbeddedPleIndex`: that release's
embedded-PLE naming branch interprets Lily's shard spelling as NVFP4 and
rejects its affine planes. The accepted run used the supported official pack.
No engine code or checkpoint values were modified for the comparison.

## Workload and controls

- Identical 4,096 corpus-derived token IDs were passed directly to both
  engines. Their compact-JSON SHA256 is
  `fd1222c74a823b3a6cf5ed3ffd3dd22cb627afde17c21cef8b2150ac9e14b322`.
  Both tokenizers also produced this prefix from the same public source corpus.
- Each call requested 256 outputs, temperature 1, top-k 20, top-p 0.95,
  min-p 0 and seed 42. Repetition, frequency and presence penalties were off.
  Context capacity was 4,352 tokens. Both engines used Q8 KV cache.
- Four fresh-load blocks ran in Stark/MLX/MLX/Stark order, six calls each.
  Each block's first call was warmup, leaving ten measured calls per engine.
- Speculation was disabled. MLX used `--no-mtp --no-drafter --no-pld` and
  matching false request flags. Prefix entries, disk prefix reuse and tokenize
  caching were disabled. Each model occupied an isolated process; MLX bound
  an owned loopback port.
- All 12 calls per engine, including warmups, repeated their respective
  hashes. Stark hashes token IDs; MLX hashes decoded response text. Different
  RNG and numerical paths preclude a cross-engine output-parity claim.
- Native phase durations came from the worker. Each MLX completion was
  bracketed by `/metrics.json` snapshots. Prefill/decode histogram sum deltas
  supplied seconds; both count deltas and the successful-request delta had
  to equal one. A further scrape checked stability. Upstream records these
  metrics before signaling request completion.
- Checkpoint revisions, original file identities, executable/library hashes
  and MLX settings hashes were checked before and after. All 113 official
  file sizes matched the pinned manifest; the 32 GB ngram file's full SHA256
  matched `c8ab74bc343408cf3923d7d64b3698fbeb3e78c07ce7f85a650a8278731251d2`.
  Other full weight payloads were not freshly rehashed.

MLX protocol and timing sources:
[server](https://github.com/ddalcu/mlx-serve/blob/v26.10.3/src/server.zig),
[scheduler](https://github.com/ddalcu/mlx-serve/blob/v26.10.3/src/scheduler.zig),
[metrics](https://github.com/ddalcu/mlx-serve/blob/v26.10.3/src/metrics.zig).

## Provenance and cleanup

| Artifact | SHA256 |
| --- | --- |
| Stark worker | `3fca1cb56dc0ab2d8ab9b870a2c5e0ae9a92289900e88cf70768f237f3445544` |
| MLX 26.10.3 executable | `908e87168f7fbced9a0a40fdd0469e18b510594ca8f76b72c32c69392be918c8` |
| Official MLX release archive | `d1781c89c206f0d077d6d1abbcab43aa9ec65280202c83eac908f957956b652b` |
| Aggregate ABBA report | `0c280cca7a0366a63e66de8f2bc28bda9b984d40f5a01258561d0be5f17553df` |
| Independent aggregate audit | `152ae23206d5e710101f704990e68cc7ac706e72e0269698ecd55a4cffdc5f3e` |

The independent audit passed 31 checks covering row counts, block order,
duration arithmetic, repeats, input/output counts, privacy and source guards.
Private aggregates, the controller, profiles, manifests and metadata remain
under the primary checkout's ignored `runtime/qwen38-mlx-comparison-20261030/`.
Prompts, responses, token vectors and probabilities were not persisted.

The temporary official MLX model was removed after the completed run. Its
configuration, model card, index, tokenizer metadata and hashes were retained.
The original Lily checkpoint and GLM checkpoint remain available.
