# Stark: native inference on Apple silicon, measured against MLX-LM

*Engineering preview · October 21, 2026*

Stark is an open-source Rust/Metal inference engine for teams building local AI features on Apple silicon. It combines native text generation, direct probability scoring and a streaming model converter. It is an experimental fork of [Uzu](https://github.com/trymirai/uzu), with upstream history and attribution preserved.

In our latest local comparison, **optimized Stark delivered 7.36% higher end-to-end Qwen3.5-2B generation throughput than a pinned, unmodified MLX-LM build**. Four of six planned pairs passed the frozen host guards; Stark led all six workloads in every accepted pair. Answer-quality equivalence remains unqualified.

## The results

Both engines used the same Qwen3.5-2B 4-bit checkpoint on an **Apple M5 Max with 40 GPU cores and 128 GiB of memory**. Requests used identical input token IDs, greedy sampling, fresh request state and one request at a time.

| Workload | Input / generated tokens | Stark, tokens/s | MLX-LM, tokens/s | Stark throughput advantage |
| --- | ---: | ---: | ---: | ---: |
| Fibonacci module | 42 / 128 | 358.2 | 321.5 | 11.40% |
| GPU explanation | 40 / 128 | 358.7 | 322.1 | 11.34% |
| Copy editing | 383 / 128 | 330.4 | 305.3 | 8.23% |
| Medium context | 522 / 128 | 315.0 | 296.7 | 6.13% |
| Long context | 2,118 / 128 | 238.8 | 230.6 | 3.53% |
| Long output | 50 / 512 | 344.2 | 331.6 | 3.83% |

**Overall paired geometric-mean advantage: 7.36%.** The overall lead ranged from 6.09% to 7.95% across the four accepted pairs. Those ranges describe the observed runs; they are not confidence intervals. [Full results and reproducible evidence](https://github.com/jshahbazi/stark/blob/d49e6b599ce5948f9fd90390a309cd7143b6d670/docs/QWEN35_MLX_QUIET_COMPARISON_2026_10_20.md)

These rates include prefill, generation, IPC/readout and request cleanup. Loading and warmup are excluded. Each cell covers 20 requests: five per independent load across four accepted loads. Rates are geometric means of within-load medians; the overall percentage uses all 24 unrounded paired workload/load ratios with equal workload weight.

## What developers get

- **Native local generation.** A Rust/Metal worker exposes tokenization, decoding and generation through a JSON-lines pipe. Applications own their chat templates, tool execution and user experience. [Generation interface](https://github.com/jshahbazi/stark/blob/d49e6b599ce5948f9fd90390a309cd7143b6d670/docs/GENERATION.md)
- **Direct probability scoring.** Applications can request candidate logits, full-vocabulary-normalized log probabilities and temperature-scaled candidate probabilities. A separate recorded JevK5 BF16 scoring build passed 243 cases and 486 cold/cache-hit replies against frozen accepted Uzu scorer references. [Scoring](https://github.com/jshahbazi/stark/blob/d49e6b599ce5948f9fd90390a309cd7143b6d670/docs/SCORING.md) · [validation](https://github.com/jshahbazi/stark/blob/d49e6b599ce5948f9fd90390a309cd7143b6d670/docs/VALIDATION.md)
- **Verified model preparation.** The streaming converter handles supported dense and existing MLX 4/8-bit checkpoints, with source manifests and output checks. A separate LFM2-350M 4-bit study measured median conversion times of 0.532 seconds for Stark versus 1.853 seconds for pinned Lalamo across five warm-file repeats, with 92.0% lower sampled process-tree RSS. These are conversion measurements. [Conversion study](https://github.com/jshahbazi/stark/blob/d49e6b599ce5948f9fd90390a309cd7143b6d670/docs/MODEL_CONVERSION_MILESTONE_4_MLX.md)

Teams building on-device assistants or local decision features can control model preparation and native execution through these interfaces. Stark currently requires building from source. Supported model configurations and validation status are documented separately.

## How to interpret the benchmark

This is a preliminary result from one machine and one model. Two pairs were excluded under predeclared rules: concurrent JavaScript tooling affected one Stark load, and system pageouts affected one MLX load. All attempts remain available; whole-campaign `timing_accepted` remains `false`. The four accepted pairs have balanced engine order.

Output-token hashes matched between engines on 6/10 fixtures and decoded-text hashes on 5/10. Both engines passed only sorting among four strict sanity checks; arithmetic, prime and Python failed the exact-reply checks. The receipts cannot distinguish semantic failures from formatting failures. Equal token counts establish comparable work; application quality needs separate evaluation.

This experiment measured direct workers at concurrency one. MLX Serve/`mlx_lm.server` HTTP performance, concurrency, time to first token, power consumption and other models or Macs remain unmeasured here. Per-process GPU activity was unavailable; nominal thermal state does not establish identical clocks. Broader production qualification remains open.

## Try the measured build

The comparison used Stark `045d0ca6`, MLX-LM `72be8c955` (`0.32.3.dev6+g72be8c955`) and MLX/MLX-Metal 0.32.4. The checkpoint was `mlx-community/Qwen3.5-2B-MLX-4bit`, affine W4/G64, with an unchanged Stark repack, context 8192 and thinking disabled. The host ran macOS 26.7.1; Stark used a Rust 1.98.1 locked release build. Full revisions and model hashes are in the evidence packet.

Start with the [source and build instructions](https://github.com/jshahbazi/stark/blob/d49e6b599ce5948f9fd90390a309cd7143b6d670/README.md#quickstart), check the supported configuration for your model, and replay your application's requests. The [benchmark protocol, model/build hashes and all attempts](https://github.com/jshahbazi/stark/tree/d49e6b599ce5948f9fd90390a309cd7143b6d670/docs/experiments/clean-qwen-2026-10-20) are available for inspection. The [comparison inventory](https://github.com/jshahbazi/stark/blob/d49e6b599ce5948f9fd90390a309cd7143b6d670/docs/INFERENCE_COMPARISON_EVIDENCE.md) also links earlier TensorFold, llama.cpp, vLLM Metal and predecessor-Uzu investigations, with their original support limits and timing exclusions. They are separate experiments and are not combined into this table.
