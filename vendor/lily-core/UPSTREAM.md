# Archived Lily extraction provenance

The executable Rust/Metal extraction and its Cargo manifests were removed on
2026-10-30 as part of the full standard Uzu execution implementation. This
directory retains licenses, attribution and this historical source-bound
record. It is not a Cargo package and is not linked into production workers.

Current Qwen3.8 execution is implemented under
`crates/stark-engine/src/encodable_block/qwen38/` and the corresponding
standard backend kernel folders. Its implementation provenance is recorded in
[Qwen3.8 Uzu provenance](../../docs/QWEN38_UZU_PROVENANCE.md).

The record below describes the former extraction at its historical revision.
Its source paths and validation commands apply to that historical checkout;
they are not current production execution instructions.

---

# Upstream provenance

- Project: [lily-qwen3.8-flash-next](https://github.com/fabiogreter/lily-qwen3.8-flash-next)
- Pinned revision: `e22d58fa7cc488493fb1b5d810a198bd29730aeb`
- Revision URL: <https://github.com/fabiogreter/lily-qwen3.8-flash-next/tree/e22d58fa7cc488493fb1b5d810a198bd29730aeb>
- Upstream metadata identifies Lily as Apache-2.0, Copyright 2026 Perplexity AI, Inc.; the modified lily-qwen3.8-flash-next distribution also attributes Copyright 2026 Fabio Greter. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Extraction and local changes

Stark extracts the minimum in-process Qwen4Exp model and native Metal 4 execution path into the `stark_lily_core` package, whose Rust library crate is `stark_lily_core`. The extraction is a modified subset, not an unmodified mirror. Stark removes server and command-line integration, dashboard and HTTP code, tools, session persistence, probes, and ablation controls. It supports resident experts only and excludes the disk-backed expert cache. The checkpoint-backed n-gram table may be paged read-only. This load path does not support MTP or vision.

Upstream environment-variable selection and diagnostics are disabled in the Stark integration. Its selected precision policy is BF16 input and activations, F32 reductions, logits and GDN recurrent state, and explicitly selected Q8 attention K/V. This records intended configuration, not a source-parity or numerical-equivalence result.

The upstream Metal shaders originate at the pinned revision. Stark modifies four of those files:

- `src/kernels/metal/hc.metal` adds a cached-activation up/mix entry point copied from the original single-row body, preserving its weight staging and dot/reduction expressions. Decode pairs it with the existing one-row down/activation kernel and reuses the existing activation scratch.
- `src/kernels/metal/gdn.metal` keeps the gated output's explicit BF16 round in a thread-local value, removing a shared-memory store, own-element read and redundant barrier. The F32 recurrent-state helper and norm reduction remain unchanged.
- `src/kernels/metal/quant.metal` adds GDN Q4 two-row input projection with a convolution epilogue. It calls the existing packed dot helper, preserves the BF16 projection boundary and copies the original convolution expression, tap order and window update. Other layouts retain separate projection and convolution.
- `src/kernels/metal/moe.metal` adds a decode down/combine entry point whose row owner also performs the following hyperconnection injection. It copies the original dot and slot-ordered combine, retains the BF16 branch output, and applies the original F32 injection expression to that rounded branch. Decode orders both the branch output and residual streams before their next consumers. The original down/combine and injection kernels remain available for reference controls.

The other ten upstream shader files retain their pinned source bytes. The original HC and generic projection/convolution entry points remain available for reference controls. These patches change scheduling and reuse without changing checkpoint quantization or the selected precision policy. Numerical and performance qualification is recorded separately in the [kernel study](../../docs/benchmarks/QWEN38_KERNELS_2026_10_29.md) and [injection fusion study](../../docs/benchmarks/QWEN38_FUSION_2026_10_29.md).

Stark adds `src/kernels/metal/stark_finite_readout.metal`, an original integer-bit validation kernel. After an ordinary GPU draw, it marks the output ID invalid if any F32 logit has exponent 255 (NaN or infinity). Stark's adapter rejects that sentinel before accepting a decision. This check covers both prefill and decode through the shared sampling helper and does not change the upstream model arithmetic. `src/kernels/metal/stark_gdn_step_reference.metal` is a test-only copy of the original gated-output wrapper, with a distinct entry-point name; it retains the shared-memory round trip for differential controls and is not included in production shader sources. Kernel controls adapted from upstream tests identify the pinned revision in their source headers.

Host patches also sanitize framework errors and retain submitted command memory until GPU completion feedback arrives after a fault. Missing completion feedback keeps fault cleanup blocked; it cannot safely release live GPU memory. Unused capture, profiling and standalone vision helpers are removed from the extracted Rust modules.

Stark adds a canonical Uzu storage admission path in `src/safetensors/uzu.rs`
and refactors Qwen weight/model loading so native-source and typed Uzu inputs
share the existing specialized kernels. The host's
`Qwen38LanguageModelConfig` contains explicit text architecture fields and
logical tensor bindings. Admission requires one regular local
`model.safetensors`, canonical MLXSpec matrices, bounded headers, checked byte
spans, a bijective tensor inventory and valid layout/dtype/shape combinations.
U32 packed codes are exposed from identical U8 storage bytes; BF16 group-major
linear/expert parameter planes are inverse-transposed into owned logical
buffers, with zero padding checked. Embedding row-group planes retain their
paged access layout. Direct BF16/F32 and I64/U64 controls preserve their bits;
optional PLE hash buffers must match the typed configuration as exact raw
64-bit words. The converted folder supplies the configuration, tokenizer and
weights, with no fallback to an original checkpoint folder. The typed text
artifact omits vision and MTP. These are host loading and storage changes;
their numerical and performance validation is recorded separately in the
[Uzu-format study](../../docs/benchmarks/QWEN38_UZU_FORMAT_2026_10_29.md).

Stark adds an explicit, disabled-by-default Metal 4 stage profiler in `src/metal_stage_profile.rs`. Fixed phase markers use a bounded per-pass timestamp heap, retain the existing concurrent dispatch and dependency barriers, and require one command buffer per profiled decode. The heap remains owned through verified completion and fault draining. Resolution checks entry sizes, timestamp order, device-reported frequency and agreement with independent command feedback. Only aggregate phase durations leave the profiler; no model values or request payloads are captured. Precise timestamps can still change scheduling, so instrumented rates are diagnostic and separate from ordinary throughput runs. Upstream diagnostic environment controls remain disabled.

Every successful pass wait also awaits all asynchronous completion feedback before accepting outputs, then checks the context's sticky fault. A signaled GPU fence alone is insufficient: framework error feedback can arrive afterward. Command resources remain held until these checks succeed, and failed checks retain the fault-safe cleanup drain.

## Validation status

This provenance document records no build, runtime, numerical, source-parity, or performance validation. Do not infer upstream parity or a performance improvement from the extraction. Record future validation separately with its toolchain, revision, configuration, and results.
