# Qwen3.8 standard Uzu execution

## Authority and scope

The user's 2026-10-30 instruction requires a full Uzu execution path for
Qwen3.8 and an independent final check that production execution has moved
off Lily. The starting revision is `5a082f7f2e98019ef8907e8c576b52a5c0c669af`.
The existing conversion and validation records remain historical evidence.
Repository rules in `AGENTS.md` and toolchains in `CONTRIBUTING.md` apply.

Weight quantization stays identical to the existing converted artifact.
Inference remains non-speculative. BF16 activations, F32 reductions, logits
and GDN recurrent state, and Q8 attention caches are the matching reference
precision policy. Any necessary numerical-policy change requires separate
qualification and a recorded explanation. Native model data, readouts and
validation artifacts remain private; publish aggregates and hashes only.

## Required execution architecture

`Engine::load_language_model` admits the typed Qwen3.8 configuration through
the same `LanguageModel` API used by standard converted models. A dedicated
`Qwen38Decoder<B>` implements its architecture using the normal Uzu backend:

- `ParameterLoader` validates and owns the converted artifact's tensor spans.
- `WeightMatrix` consumes canonical packed codes and group-major affine
  parameters. Projection dispatch uses standard `LinearMatmul`/`MatmulKernel`
  policies, including expert-bank row selection. It never restores the
  resident matrices to Lily's layouts or dequantizes complete weight banks.
- Normal backend buffers, scratch allocation pools, command encoding,
  submission, completion and state acceptance own the complete execution.
- Qwen-specific HC, GDN, QSA, expert routing and hashed PLE are encodable
  blocks with kernels registered in the standard backend infrastructure.
  Existing compatible primitives are reused. Extensions preserve the exact
  architecture, tensor boundaries and state semantics.
- The full 48-layer graph, embedding, final HC reduction/norm and vocabulary
  readout execute through that decoder. Required blocks cannot be bypassed.

The standard language-model decoder/state interface gains explicit variants
for the existing decoder and Qwen3.8. The common generation and scoring APIs
continue to own requests. Qwen3.8 supports flat sequential batches; unsupported
tree verification, speculation and multimodal operations fail explicitly.

Hashed PLE retains bounded paged reads of the large n-gram table. Host token
IDs are staged only after their producing work completes. A model capability
prevents the standard stream from encoding a future unknown GPU token before
that synchronization. No layer performs CPU projection, expert routing,
attention, or recurrent-state computation as a production fallback.

## Work ownership

| Area | Owner |
| --- | --- |
| Engine/decoder-state seam, streams/scoring, worker admission, integration | Root |
| Strict canonical weight loading and bounded paged file regions | Weight-loader agent |
| Decoder composition, state, HC, GDN and MoE | Core-block agent |
| QSA sparse attention and hashed paged PLE | Attention/PLE agent |

Only root runs builds, model-bound validation and GPU experiments. Performance
runs have exclusive model/GPU use and no concurrent compilation. Every source
workspace uses its own Cargo target. References use sealed old binaries, not
a production fallback.

## Acceptance map

| ID | Requirement | Required evidence | Initial status |
| --- | --- | --- | --- |
| U1 | Standard Engine/LanguageModel entry point | Dispatch test and actual worker load | Pending |
| U2 | Standard parameter and matrix infrastructure | Complete tensor admission; canonical plane hashes; standard matmul dispatch | Pending |
| U3 | Complete Qwen architecture | Component, layer and complete-model controls for all block families | Pending |
| U4 | Standard request-owned state and commands | Reset/accept/bounds/error tests and command completion checks | Pending |
| U5 | No Lily production executor | Cargo dependency audit, call-site audit, linked-symbol check and actual route evidence | Pending |
| U6 | Same weight quantization and no speculation | Artifact identities and configuration/dispatch checks | Pending |
| U7 | Numerically qualified execution | Matching precision controls against sealed reference and bounded independent references | Pending |
| U8 | Sampled generation and scoring work | Complete-model sampled runs, deterministic repeats, readout/scoring controls | Pending |
| U9 | Performance measured honestly | Fixed workload before/after measurements, warm controls and clock context where needed | Pending |
| U10 | Independent final conformance review | Requirement-to-implementation map against final revision | Pending |

Architecture conformity and throughput are evaluated separately. Conversion
alone, renamed types, a new wrapper around the old executor, or a successful
one-layer pilot cannot satisfy U1–U5. The historical 77–80 tok/s objective
remains a performance target; measurements determine its status.

## Implementation and verification sequence

1. Implement canonical weight admission and standard matrix access, plus the
   decoder/state seam. Keep the existing standard model behavior intact.
2. Implement all Qwen block families and the complete decoder. Adapt source
   algorithms only with preserved copyright, licenses and recorded revisions.
3. Exercise synthetic primitive and layer controls, strict negative admission,
   state/reset/error boundaries, and standard model regression tests.
4. Route converted-model generation and scoring through the standard engine.
   Remove the Lily execution dependency and production dispatch. Native-source
   inputs that need conversion must receive an explicit conversion requirement.
5. Build pinned production workers. Check source/artifact identities, compare
   full-vocabulary readouts at short and 4K contexts, and run sampled generation
   with top-k 20 and unbounded nucleus, seed 42, 256 tokens and no speculation.
   Cross-engine sampler differences are separated from forward-pass numerical
   comparisons; do not present unmatched random streams as exact token parity.
6. Profile and benchmark the full path. Record failures and rejected variants.
7. Have an independent reviewer trace the final production call graph and
   runtime evidence against U1–U10. Correct deviations and repeat affected
   checks before claiming completion or pushing a completed implementation.
