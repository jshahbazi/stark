# Rejected Qwen3.8 F32 hyperconnection fusion

The two-stage F32 experiment passed its numerical checks but regressed the warm component benchmark. It was removed from the production source on 2026-10-28. The existing hyperconnection implementation and the accepted QSA split-attention changes remain active.

## Scope and provenance

This is original Stark code. The two-stage structure was informed by Lily's hyperconnection implementation; no external source text was copied into these experimental kernels. Stage one fused RMS normalization, affine down projection, SiLU, and optional floating-point injection. Stage two fused affine up projection, sigmoid, normalized stream mixing, and injection-gate copying. Activations and accumulation remained F32 with Metal Safe math. Down/up projections retained the existing eight-values-per-lane SIMD accumulation order. RMS retained the existing 256-thread reduction tree. Injection used a 32-lane reduction and retained the unchanged `2e-5` error gate.

Only matching affine 4-bit or 8-bit down/up matrices were admitted. Floating-point injection supported F32, F16 and BF16 weights. The path was opt-in through `STARK_QWEN38_FUSED_HC=1`; unsupported layouts retained the existing path. Raw dot products and normalized values were checked for nonfinite values before saturating activations.

## Reproduction

`reapply.patch` restores only this experiment: the two kernels, their tests, the host methods, the opt-in branch, and the two pipeline caches. It excludes other agents' graph counters, factored affine work and QSA attention changes. The patch was generated from before/after removal snapshots of the shared working tree, whose Git HEAD was `9e786991885d69506f5d4740c8af11535844c7de`. That HEAD alone does not include the other accepted working-tree changes required by the patch. Apply to the source containing this archive, after checking for intervening changes:

```sh
git apply --check docs/archive/experiments/qwen38-f32-hc-2026-10-28/reapply.patch
git apply docs/archive/experiments/qwen38-f32-hc-2026-10-28/reapply.patch
cargo +nightly-2026-10-01 test --locked -p stark-engine --features cpu,metal qwen_fused_hc -- --include-ignored --test-threads=1
```

Run on an otherwise idle Apple silicon Mac with Metal available. The ignored microbenchmark does not require a model. `git apply -R` removes the restored experiment. The archive's `source-hashes.json` records the source hashes on both sides of removal, the patch hash, and the private aggregate test-log hash. The final Rust source and tests were formatted using `nightly-2026-10-01` before removal. Patch application and reversal were checked against the removal snapshot without running a new GPU workload.

## Results

The targeted run passed all four tests, including the ignored benchmark: `4 passed; 0 failed; 701 filtered out`. Coverage included affine 4/8-bit weights, F32/F16/BF16 metadata, aligned and unaligned byte views, one/four streams, broadcast/per-stream normalization, optional injection, production dimensions, a dimension smaller than the stream count, and nonfinite/overflow rejection. Hidden outputs matched the existing vector-eight reference bit for bit. Injection error remained within `2e-5`.

The component benchmark used an Apple M5 Max MacBook Pro with 128 GB memory, four streams, width 2560, latent width 320, affine 8-bit weights with BF16 scale/bias metadata, group size 64, byte-unaligned weight planes, and no injection. Each sample encoded 16 hyperconnections in one command. Timing measured completed-command wall time divided by 16, excluding allocation and encoding. These measurements do not establish full-model throughput or GPU-only duration.

| Sample | Existing vector-eight path | Fused F32 path | Change |
| --- | ---: | ---: | ---: |
| Cold, excluded | 378.997 us | 95.826 us | Compilation/cache warm-up |
| Warm 1 | 43.130 us | 50.729 us | 17.6% slower |
| Warm 2 | 40.781 us | 50.643 us | 24.2% slower |

This final run supersedes earlier unstable component measurements. There was no measured speed gain to justify retaining the experiment. Recomputing F32 normalization for each down-projection tile adds input/normalization reads and arithmetic; that is a plausible contributor, rather than an established isolated cause. The result does not qualify a mixed-precision fusion, which requires its own numerical and performance evidence.

No model weights, prompts, responses, token vectors, logits, probabilities or private tensors are included in this archive.
