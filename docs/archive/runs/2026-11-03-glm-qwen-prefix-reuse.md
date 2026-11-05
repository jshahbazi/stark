# GLM-5.3 and external mixed-affine Qwen3.8 prefix reuse

Date: 2026-11-03. This record extends the
[converted-model run](2026-11-01-generation-prefix-reuse.md) and the
[Lily-format Qwen3.8 run](2026-11-02-hybrid-prefix-reuse.md). Existing records
remain bound to their original sources and tooling.

## Implementation

The generation implementation is `5ec52a2ff1633a57f14b9f5e90886854c1c6bd95`.
GLM-5.3 now captures a snapshot at the previous prompt's final 1,024-token
boundary. It copies all KDA recurrent matrices and convolution history, restores
the MLA/indexer cursor and incomplete pooling tail, and retains completed
attention rows in the same loaded model. The 1,024-token grid preserves the
existing scalar and admitted four/eight-token prefill grouping.

The external mixed-affine Qwen3.8 artifact already selects the native
mixed-precision snapshot path. This run qualifies that exact artifact using its
4,096-token grid. It adds no quantization or numerical policy.

Both snapshot routes require the whole retained prefix to match and at least
two remaining input tokens. Model instance, host scope, context and MTP selection
must match. The shared worker retains publication, invalidation, idle expiry and
failure fencing. See [generation](../../GENERATION.md#prefix-reuse).

## Environment and artifacts

- Apple M5 Max, 128 GiB unified memory, macOS 26.7.1.
- Workers built with Rust 1.98.1, release mode and locked dependencies, in
  separate source worktrees and target directories.
- Validation uses uv and Python 3.13.13. Workers run sequentially.
- The GLM full matrix uses the existing
  [`with_glm53_optimized.sh`](../../../scripts/with_glm53_optimized.sh) recipe:
  eight-token prefill groups, transposed KDA state, admitted encoder/resident
  banks and its specialized native kernels. Every reference and candidate
  inherits the same environment. The context stays within its 4,096-token cap.
  This compares prefix reuse under that recipe; it does not compare the complete
  optimized recipe numerically with the default graph.

| Model | Source and revision | Precision/storage |
| --- | --- | --- |
| GLM-5.3-Flash | `unsloth/GLM-5.3-Flash-GGUF`, `a38483c8cd5df544f53d70fb281afe97369d5ab6` | Original four-shard UD-Q2_K_XL; native F32 recurrent state, activations and accumulation under the stated optimized recipe. |
| Qwen3.8-Flash-Next | `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`, `788c07a8725642bccf13a0a3b8a21954ab84b4f2` | Admitted external mixed-affine 4/8-bit weights, BF16 activations, F32 GDN state, affine-64 attention cache route. |

GLM configuration and tokenizer metadata come from
`zai-org/GLM-5.3-Flash` at
`04c4e9e95c5da8862dced7e5056455116f83a7e0`, separately from the GGUF weights.

| Worker | Source | SHA-256 |
| --- | --- | --- |
| Fresh-state baseline | `6eb240d29ae51a01d64ff1fe92e8f6addc4ddff5` | `cd7136372f44b044441e053afa9eaa3d0e698797b41f07fa1dbdab63cf8a1107` |
| Candidate | `5ec52a2ff1633a57f14b9f5e90886854c1c6bd95` | `1041eae891a2992e6e74452efe3a15c2dfe07dd42d65e49c6bc01c23bd8aa0da` |

## Method

The requested artifacts' tool-history templates also need adjacent-loop
metadata, declared single-underscore temporaries and explicit
`tojson(ensure_ascii=False)`. The bounded native renderer now implements those
constructs. Private root arguments are omitted; nested JSON data and genuine
local bindings remain usable, while private member access and dunder names
remain rejected. Synthetic tool-history renders were compared in memory with
the model-runtime-compatible Jinja/JSON control. The GLM renderer changes are
in `f7adc0ff`; Qwen used the earlier adjacent-loop build identified below.

[`validate_generation_prefix_cache.py`](../../../scripts/validate_generation_prefix_cache.py)
renders deterministic synthetic tool histories with each artifact's template.
Prompts, token vectors, responses and scope values remain in memory. The
candidate with reuse enabled is compared with the same candidate without reuse
and the independent baseline binary. Comparisons cover generated-token and text
hashes, finish reasons and tool parsing where admitted.

The sequence covers cold/growing/repeated/shrinking prompts, early and final-pass
divergence, one/two-token snapshot remainders, context and scope changes, MTP
transitions when supported, errors, unscoped requests, reset, image operations,
cancellation, tokenize/decode preservation and a same-scope worker restart.
Each reuse count is checked against the boundary rule. Equal error replies
cannot qualify correctness.

Timing uses seed 12, four generated tokens, two alternating worker-order blocks
and one round per block. Every first turn has a fresh scope. There are two
samples per worker and turn. The reported figure is worker `prefill_seconds`,
which includes native state setup and the first sample. It excludes template
rendering, tokenization and model loading. Timing outputs must also match.
Before each timed worker arm, one unscoped request processes up to 2,048 input
tokens and one output token. The timed first turn has fresh request state and
no reusable prefix, with kernels and some immutable weight views already warm.

## External mixed-affine Qwen3.8 results

All **36 cases passed**: generated token IDs, text and finish reasons matched
both fresh references, and every reuse count matched the rule. MTP transitions,
all invalidation paths and same-scope restart passed. Output budget: 16 tokens;
context: 24,576. Seed-11 input lengths were 9,039, 9,197, 9,310 and 9,590 tokens.
The synthetic session used eight tools, 12 instruction paragraphs and follow-up
sentence counts 8/5/16, with a first-turn target of 9,216 tokens.

| Timed turn | Input tokens | Baseline median | Reuse median | Reused tokens |
| --- | ---: | ---: | ---: | ---: |
| Cold | 8,945 | 6.363 s | 5.678 s | 0 |
| Follow-up 1 | 9,135 | 8.954 s | 0.860 s | 8,192 |
| Follow-up 2 | 9,250 | 8.140 s | 0.994 s | 8,192 |
| Follow-up 3 | 9,545 | 8.208 s | 1.273 s | 8,192 |

Follow-up median ratios were approximately 10.4, 8.2 and 6.4. Baseline/reuse
ranges were 8.600–9.308/0.808–0.912 s, 7.997–8.283/0.983–1.005 s, and
8.027–8.388/1.253–1.292 s. Cold ranges were 6.266–6.459/5.631–5.724 s. The cold
difference was not isolated as an implementation effect. Host contention and
thermal conditions were not independently measured.

| Qwen artifact/evidence | SHA-256 |
| --- | --- |
| Weights, including external n-gram data | `817c5d4a8171f599b40fc3075792734c2b726c028310ce7cae6ea36be2643189` |
| Configuration | `84a69b2bcb20a57b033c3827bc98fa21a5db4d18b55e0640dd3687c99a1b765f` |
| Tokenizer | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| Chat template | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| Renderer executable, source `43155df6` | `6d6e27cd31138c525d4abfa74316b267892c4b5ac3a73c61a5924763883214e1` |
| Harness | `c4ef1fdebf4fcef0db34bd513f8cf523321730a73e796511c24582ce103f5041` |
| Private validation record | `196f530c9be85e7937959eadf8b9968f73ce25e76521619a514d230ff8132112` |

## GLM results

All **32 checks passed** under the stated optimized recipe. Every case generated
eight output tokens; token IDs, text and finish reasons matched both fresh
references. Every reuse count matched the rule. Errors, unscoped requests,
explicit reset, image operations and cancellation each cleared a previously
warm 2,048-token entry. Tokenize/decode preserved it. The same-scope worker
restart reused zero tokens and matched both references. GLM MTP is unsupported
and was not enabled.

The context was 4,096 tokens. Seed-11 input lengths were 1,221, 1,342, 1,499 and
2,183 tokens, exercising capture at both 1,024 and 2,048 positions. The
one-token-remainder case reused zero; the two-token remainder reused 2,048.
The full harness completed in 3,776.8 seconds, including artifact hashing,
loads, comparison passes and timing. Its model-bound record is schema v3;
this run uses harness revision `29fc8169`, separately from the Qwen run above.

Reproduction, using the exact model and worker identities in this record:

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  scripts/with_glm53_optimized.sh \
  uv run --project tools/converter --locked --python 3.13.13 python \
  scripts/validate_generation_prefix_cache.py \
  --model /absolute/path/to/prepared-glm \
  --baseline /absolute/path/to/baseline/stark-generate \
  --candidate /absolute/path/to/candidate/stark-generate \
  --reuse boundary --context 4096 --first-tokens 1536 \
  --tools 1 --instruction-paragraphs 1 --follow-up-sentences 8 8 40 \
  --max-new 8 --timing-new 4 --timing-blocks 2 --rounds-per-block 1 \
  --progress --output runtime/glm-prefix-reuse-new-run.json
```

| Timed turn | Input tokens | Baseline median | Reuse median | Reused tokens |
| --- | ---: | ---: | ---: | ---: |
| First turn, fresh state | 1,258 | 40.992 s | 43.660 s | 0 |
| Follow-up 1 | 1,385 | 45.775 s | 13.715 s | 1,024 |
| Follow-up 2 | 1,521 | 51.430 s | 18.597 s | 1,024 |
| Follow-up 3 | 2,161 | 79.682 s | 45.884 s | 1,024 |

Follow-up median ratios were **3.34, 2.77 and 1.74**. Baseline/reuse ranges
were 44.375–47.175/12.875–14.556 s, 50.349–52.512/17.949–19.246 s and
78.586–80.778/44.988–46.780 s. The fresh-turn median increased by 2.669 s
(6.5%); its ranges overlapped (39.630–42.354/40.661–46.659 s). The snapshot's
isolated contribution to that difference was not measured. All timing outputs
matched. These are two samples per worker/turn on this host, with ambient
contention and thermal conditions unmeasured.

| GLM artifact/evidence | SHA-256 |
| --- | --- |
| Four-shard weights manifest | `a3b35a5e80c39e699a176d57d50da6d50d1cbbce674f3e60b0e45daaefb93cc3` |
| Configuration | `d1e229307a382fc1deae8e4a48b6ffc2f47fc47769ab8789c023bdbd806072fd` |
| Tokenizer | `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` |
| Tokenizer configuration | `98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc` |
| Chat template | `34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891` |
| Native-source receipt | `7b8a61782577efbc30f6791f26e91d539a0934f14671f82d9548a313ec67903d` |
| Renderer executable, source `f7adc0ff` | `8c071dc49fed2f94d0120b4aed87af6c738c0b6b4c992f73898755dbf1fdc856` |
| Harness, source `29fc8169` | `32b5981d56552c3731d7eb881331a0587c2b2c7ed477651a49c6595881512531` |
| Optimized wrapper | `be7e0a5400c9325b11800819a57e56ab946f35bd4264b129327340f304dbfcab` |
| Private validation record | `29bf669dab913b88d05c9a0450abadba4b49c524fe37e3e29959acc6169d049d` |

The private completed record is
`runtime/prefix-reuse-2026-11-03/glm53-ud-q2-optimized.json`, mode 0600.
Native component tests additionally cover both KDA physical layouts and
incomplete sparse pools after cache growth and tail overwrite. The private
small-fixture full-forward test remained ignored because its fixture was
unavailable; the real-checkpoint worker comparison above was run instead.

The default-graph attempt reached worker readiness in 1.238 seconds. Its
1,221-token cold case and first follow-up completed successfully; later cases
were interrupted to use the optimized recipe for the complete matrix. There is
no complete default-graph numerical-comparison result. A private one-second CPU
profile sampled prefix ingestion and Metal command-completion waits; it did not
show sustained model admission or pipeline compilation. The profile cannot
attribute time to GPU arithmetic versus driver scheduling.

## Retained-state accounting

These figures count snapshot payload from admitted geometry and allocation
types. They are not measured process RSS or allocator/transient-copy peaks.
Existing model-owned attention caches and immutable weights are separate.

- GLM: 34 KDA layers, each holding
  `3 * 8192 * (4 - 1) + 8192 * 128` F32 values. Snapshot payload is
  **152,633,344 bytes (145.56 MiB)**. The checkpoint's sparse pool size is four,
  so its incomplete-pool payload is empty at a 1,024-token boundary. The general
  implementation also copies a nonempty tail when the admitted pool needs one.
- Qwen: 36 F32 GDN matrices, BF16 convolution windows, PLE window and optional
  draft hidden state contribute **110.29–110.31 MiB**. The snapshot's own
  `bytes()` accounting sums these tensors. Attention/indexer caches stay in the
  loaded model.

## Checks and limits

- Native CPU/Metal engine suite: 763 passed, 79 ignored; integration and
  documentation tests passed.
- CPU/grammar engine suite: 433 passed, 21 ignored; integration and
  documentation tests passed.
- Generation worker: 112 release tests passed, two ignored; 72 tests passed
  without native features. Worker process harness: six cases passed.
- Converter/app-backend suite before template changes: 5,923 passed, 89 skipped.
- Final renderer checks: 19 Rust tests and 80 Python template-runtime tests
  passed; independent release-helper controls passed. Seven model-free
  qualification-harness tests passed.
- CI-scoped worker/template Clippy, Ruff, formatting and repository policy
  checks passed. A separate, broader engine Clippy invocation with the CPU
  feature reports 24 existing reference-constant precision/approximation
  diagnostics. No reference arithmetic was changed to silence those checks.
- Final complete converter/app-backend suite: 5,958 passed, 89 skipped.

This is bounded synthetic numerical and lifecycle qualification for the exact
artifacts, kernels and contexts above. It does not establish long-context
quality, real client task competence or a universal latency improvement. The
installed app was not rebuilt or signed as part of this run.

The separate [fast-engine research](2026-11-03-glm-fast-engine-research.md)
records source-backed follow-up experiments for prefill and their matching
requirements. Those experiments are proposals.

Private records are created exclusively with mode 0600 under ignored
`runtime/`. No prompt, output, token vector, probability, credential or scope
value is logged or persisted.
