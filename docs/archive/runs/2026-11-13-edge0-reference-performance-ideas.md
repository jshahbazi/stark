# Edge0 reference performance ideas

2026-11-13. Bounded source review for Stark's local non-commercial evaluation
of the original Edge0-145B Q3_K artifact. No model, build or benchmark was
launched for this review.

## Source and applicability

Reviewed [Edge0-AI/Edge0 at
`b7900cbb322926ccebdcd0c0c5a1280497ecff22`](https://github.com/Edge0-AI/Edge0/tree/b7900cbb322926ccebdcd0c0c5a1280497ecff22).
The pinned [platform table](https://github.com/Edge0-AI/Edge0/blob/b7900cbb322926ccebdcd0c0c5a1280497ecff22/README.md#platforms)
assigns the 145B/120B models to NVIDIA/CUDA Strata; its native Metal engine
supports the older 35B model. The [Strata origin record](https://github.com/Edge0-AI/Edge0/blob/b7900cbb322926ccebdcd0c0c5a1280497ecff22/edge0-strata/EDGE0.md)
also records removal of its inherited Metal directory. Upstream CUDA throughput
does not establish expected speed for Stark's Apple GPU implementation.

## Three concrete opportunities

1. **Specialize the actual quantized operation and workload shape.** The
   [latent CUDA kernels](https://github.com/Edge0-AI/Edge0/blob/b7900cbb322926ccebdcd0c0c5a1280497ecff22/edge0-strata/src/kernels/cuda/latent.cu#L29-L38)
   use coalesced packed loads and warp reductions for Q8 projection work. Stark's
   new opt-in singleton Q8/Q3/IQ4 and fused-Q3 kernels remove dynamic token
   accumulators and runtime format addressing while preserving its existing
   F32 column and reduction order. **Implemented in this change:**
   `STARK_EDGE0_SPECIALIZE_SINGLE=1`. The singleton implementation predates this
   source review; this mapping is retrospective. Validation and measured speed
   belong to the separately bound local experiment record. A further aligned
   packed-load variant needs byte-offset, tail and bitwise parity gates.

2. **Batch complete expert work with device job tables and fused epilogues.**
   Strata's [Q3_K latent prefill operator](https://github.com/Edge0-AI/Edge0/blob/b7900cbb322926ccebdcd0c0c5a1280497ecff22/edge0-strata/src/prefill/lat_q3k_fused.cu#L1-L31)
   combines gate/up, SwiGLU and activation quantization before the IQ4 down
   stage, with two launches for the A stage. Its [prefill executor](https://github.com/Edge0-AI/Edge0/blob/b7900cbb322926ccebdcd0c0c5a1280497ecff22/edge0-strata/src/prefill/prefill.cpp#L1472-L1478)
   explicitly avoids gathering expert weights for that route. **Future work:**
   replace Stark's per-expert gather/scatter copies and encoder boundaries in
   `edge0_metal::encode` with bounded job-table dispatches over existing selected
   record bindings. Preserve every original route slot and apply U separately.
   CUDA tensor-core/int8 activation arithmetic changes the numerical policy;
   a Metal equivalent requires separate matched F32 qualification. Simply
   doubling the existing matvec batch did not demonstrate a useful speed gain
   in this local investigation.

3. **Reuse execution plans and reduce host/device fences.** The
   [35B native forward path](https://github.com/Edge0-AI/Edge0/blob/b7900cbb322926ccebdcd0c0c5a1280497ecff22/edge0-infer/src/models/edge0_35b/forward.cpp#L780-L840)
   retains compatible decode plans, checks reuse and rebinds weights; its
   [executor](https://github.com/Edge0-AI/Edge0/blob/b7900cbb322926ccebdcd0c0c5a1280497ecff22/edge0-infer/src/exec/execution.cpp#L135-L157)
   defines compatibility checks. Strata's [145B verifier](https://github.com/Edge0-AI/Edge0/blob/b7900cbb322926ccebdcd0c0c5a1280497ecff22/edge0-strata/src/core/verify.cpp#L2855-L2888)
   captures CUDA graphs. **Future work:** investigate reusable bounded Metal
   dispatch plans around Stark's current `routes()` fences. Host F64 routing
   currently requires each score readback; removing that dependency needs an
   independently validated exact routing implementation. Existing measurements
   identify substantial GPU execution time, so plan reuse alone has limited
   expected benefit. Keep prefix reuse disabled for the present artifact.

## Provenance

This document summarizes source and proposes follow-up work; it copies no
upstream implementation. Existing component licenses and import provenance are
recorded in [license intake](2026-11-12-edge0-license-intake.md): root
Apache-2.0, Strata MIT and its separately tracked dependencies. Any later code
port must retain the applicable notices and record exact source files and
revisions. No new numerical policy or default selector is promoted here.
