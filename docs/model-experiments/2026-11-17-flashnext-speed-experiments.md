# Flash Next prefill and decode experiments

**November 17, 2026 · Lily Q4-XL on a physical M5 Max, 128 GiB**

These attempts follow the [installed-worker speed review](../benchmarks/2026-11-17-flashnext-speed-review.md). The baseline is the 4,096-token prefill grid and the existing grouped MoE and MTP verifier kernels. One sealed experimental worker runs the baseline and changes one option at a time. Runtime flags are private experimental controls. The production defaults are preserved.

| Attempt | Candidate | Pre-run correctness and speed criteria |
| --- | --- | --- |
| 20261117-FLASHSPEED-02 | Set `STARK_LILY_PREFILL_CHUNK=1024` or `2048` at worker launch. The engine and worker share the chosen boundary. | Confirm reported geometry, exact reuse counts and same-grid cold/resumed output identity. Compare cold prefill and growing follow-up first-token time at each length. Treat a cross-grid output change separately from cache correctness. |
| 20261117-FLASHSPEED-03 | Set `STARK_FLASHNEXT_PREFILL_GROUPED_SWIGLU=1`. Fuse routed Q4 gate/up and SwiGLU in the grouped prefill tile. | Require exact BF16 intermediate/activation component behavior and same full-model emitted output at matched settings. Measure cold prefill and observe the rerouting threshold. |
| 20261117-FLASHSPEED-04 | Set `STARK_LILY_VERIFY_HC_DOWN_REUSE=1`. Reuse Q8 HC down weights across exact verifier rows. | Exact component intermediate and output bytes. Compare equal-output, equal MTP-work generations at short and long contexts. Record component timing separately. |
| 20261117-FLASHSPEED-05 | Instrument full, partial, rejected, no-draft and interrupted MTP round finishes to inform a GDN state replay design. | The profile must contain aggregate counts/time only. Assess exposure and future rollback design independently of a speed claim. |

Repeat timing rows with a single model resident. Exclude explicit warmups. Retain host observations and variation alongside group medians. Small net effects or effects within the observed variation need a separate paired confirmation before any default changes. A fresh input uses an unscoped request. A reuse input follows its identical scoped prefix with the same model, context size and MTP setting. Full-model runs require a complete artifact and sufficient headroom for one worker. The installed service is stopped only for that campaign and restarted afterward.
