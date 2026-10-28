# Upstream provenance

- Project: [lily-qwen3.8-flash-next](https://github.com/fabiogreter/lily-qwen3.8-flash-next)
- Pinned revision: `e22d58fa7cc488493fb1b5d810a198bd29730aeb`
- Revision URL: <https://github.com/fabiogreter/lily-qwen3.8-flash-next/tree/e22d58fa7cc488493fb1b5d810a198bd29730aeb>
- Upstream metadata identifies Lily as Apache-2.0, Copyright 2026 Perplexity AI, Inc.; the modified lily-qwen3.8-flash-next distribution also attributes Copyright 2026 Fabio Greter. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Extraction and local changes

Stark extracts the minimum in-process Qwen4Exp model and native Metal 4 execution path into the `stark_lily_core` package, whose Rust library crate is `stark_lily_core`. The extraction is a modified subset, not an unmodified mirror. Stark removes server and command-line integration, dashboard and HTTP code, tools, session persistence, probes, and ablation controls. It supports resident experts only and excludes the disk-backed expert cache. The checkpoint-backed n-gram table may be paged read-only. This load path does not support MTP or vision.

Upstream environment-variable selection and diagnostics are disabled in the Stark integration. Its selected precision policy is BF16 input and activations, F32 reductions, logits and GDN recurrent state, and explicitly selected Q8 attention K/V. This records intended configuration, not a source-parity or numerical-equivalence result.

The fourteen upstream Metal shader files retain their pinned source bytes. Stark adds `src/kernels/metal/stark_finite_readout.metal`, an original integer-bit validation kernel. After an ordinary GPU draw, it marks the output ID invalid if any F32 logit has exponent 255 (NaN or infinity). Stark's adapter rejects that sentinel before accepting a decision. This check covers both prefill and decode through the shared sampling helper and does not change the upstream model arithmetic.

Host patches also sanitize framework errors and retain submitted command memory until GPU completion feedback arrives after a fault. Missing completion feedback keeps fault cleanup blocked; it cannot safely release live GPU memory. Unused capture, profiling and standalone vision helpers are removed from the extracted Rust modules.

Every successful pass wait also awaits all asynchronous completion feedback before accepting outputs, then checks the context's sticky fault. A signaled GPU fence alone is insufficient: framework error feedback can arrive afterward. Command resources remain held until these checks succeed, and failed checks retain the fault-safe cleanup drain.

## Validation status

This provenance document records no build, runtime, numerical, source-parity, or performance validation. Do not infer upstream parity or a performance improvement from the extraction. Record future validation separately with its toolchain, revision, configuration, and results.
