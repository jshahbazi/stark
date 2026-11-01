# GLM-5.3-Flash native optimization, 2026-11-01

Status: v39 wins the screened, matched warm-decode comparison by **1.216%**.
Its median useful decode rate is **34.800 tok/s**, versus **34.382 tok/s** for
the refreshed MLX-Serve 26.10.3 control. This is a narrow result for one short
greedy request; it does not establish broad answer quality or a whole-model
quality win. Native prefill remains much slower, so the full request is not
faster. The GLM path and its experimental flags remain opt-in; no production
default changed.

The [aggregate JSON](data/glm53-native-optimization-2026-11-01.json) contains
scalar measurements and hashes only. Ignored runtime receipts retain detailed
screen records. Prompts, completions, token IDs, logits and probabilities are
not published. See the [matched MLX baseline](GLM53_MLX_BASELINE_2026_11_01.md)
for the original control provenance.

## Final screened comparison

Both engines used the same four-shard `unsloth/GLM-5.3-Flash-GGUF`
`UD-Q2_K_XL` checkpoint at revision
`a38483c8cd5df544f53d70fb281afe97369d5ab6`, 14 input token IDs, context limit
512 and 32 greedy output tokens with speculation disabled. The input token-ID
hash, tokenizer-configuration hash and decoded-text hash match. The native
receipt also records the rendered-template hash; the MLX receipt records the
rendered prompt and token-ID hashes rather than a chat-template file hash.
All four shard prior content hashes, sizes, modification times and inode
identities match between receipts. Full 108.7 GB payload hashes were not
recomputed for these runs.

Native v39 had four warmups, four accepted measured requests and one excluded
attempt with a paging delta. MLX had one warmup and four accepted measured
requests, with no exclusions. Each accepted request passed its host screen,
including the same 10-second quiet-admission policy. Both produced the same
decoded-text hash across accepted replays. The MLX receipt does not include
generated token-ID hashes, so token-ID equality of the outputs is not claimed.

Useful rate is `(sampled_tokens − 1) / decode_seconds`: the first of 32
sampled tokens is produced during prefill, leaving 31 useful decode tokens.
All-sampled rate uses all 32 tokens over the same decode interval.

| Engine / build | Useful median | All-sampled median | Prefill median |
| --- | ---: | ---: | ---: |
| Stark v39 | **34.800 tok/s** | **35.923 tok/s** | 31.691 tok/s |
| MLX-Serve 26.10.3 | 34.382 tok/s | 35.491 tok/s | 93.126 tok/s |

The useful-rate difference is 0.418 tok/s (+1.216%). Native and MLX define
their decode timer boundaries independently, so this is a screened engine
comparison, not isolated GPU timing. Median full-request wall time was about
1.346 seconds for native and 1.053 seconds for MLX; the short-request prefill
cost erases the warm-decode advantage. The result applies to this 14-token
prompt and 32-token output, not long prompts, multimodal requests, sustained
thermal behavior or general answer quality. Receipts:
`runtime/glm53-optimization-20261101/final-screened-v39.json` (SHA-256
`ae4b7019996e5578c90d4acb5da57612d1a6e534270e5c179ef2609b7229c90e`) and
`runtime/glm53-optimization-20261101/mlx-final-control-v38.json` (SHA-256
`076421a11c9038110989f611ec0d1991cc4716343ff7a5a54e407cf8f3d8ce1d`).

## Build and qualification

The generator and optimized scorer are separate Rust 1.98.1 locked builds.
The source-bound receipt is
`runtime/glm53-optimization-20261101/build-receipt-v39.json` (SHA-256
`18e80daf9605ccba47f3498c8eb924aafffb1666ba8abbc540ab008943531800`). It
records unchanged source around the build and hybrid bundle
`c832148ac5e543f56779a3abb6b28dae26651b060b319d7bc46ac9981e18c731`. The
generator hash `0191ff9aff0eac1022a433604ca3d71de929fb557c4945c49201cfdad00db44c`
matches the screened run; scorer hash
`5a4af2f26271a265938eebec8f7a48ed9ae8b7c480cb95b3bc0c6fc0e3a2cc58` matches
the optimized worker used by the v38 score-parity receipt.

The scorer comparison covers 8 public rendered contexts and up to 16
candidate IDs per context (128 normalized next-token log-probability points)
against the archived original-F32 GLM scorer, not an MLX teacher. Maximum
absolute log-probability delta was `1.120e-5`, RMS delta `4.270e-6`, against a
`0.001` limit; candidate mean NLL differed by `2.36e-7`. This is narrow score
agreement, not broad answer-quality evaluation.

The v39 focused hybrid suite reports 173 passed, 0 failed and 53 ignored. The
GPU greedy readout checks every logit for finiteness, orders finite F32 values
with integer IEEE-754 keys to avoid denormal-flush ambiguity, and preserves
the first index for equal values, including signed zero. All-poison input
fails through the graph failure status. Greedy continuation reads back one
selected index; full-logit scoring is unchanged. Three focused GPU readout
tests passed. Q4 actual-admission tests covered a nonzero byte offset; context
growth reached position 67 on both normal and test-only lazy-initialization
paths. The source-oracle pools passed 94 positions each (pool 2 max absolute
logit delta `2.77e-5`; pool 4 `3.19e-5`) under the `1e-4` logit gate. Paired
worker checks passed greedy reference cases, isolation, EOS/recovery and cache
invalidation. Both workers passed Clippy with warnings denied. Scorer release
unit tests passed (21); generator library release tests passed (89, with 2
ignored). The generator binary harness itself contains no tests. The complete
repository suite was not completed: a broader engine run was interrupted in
large, unrelated BF16 CPU reference cases. No whole-suite pass is claimed.

## Earlier timing trials

| Build | Useful median | All-sampled median | Status |
| --- | ---: | ---: | --- |
| v37 | 34.710 tok/s | 35.830 tok/s | Historical only: the float GPU argmax raised a denormal exactness concern, fixed by integer IEEE ordering in v38. |
| v35 | 33.991 tok/s | 35.087 tok/s | Prior screened and narrowly scorer-qualified result. |
| v32 | 33.868 tok/s | 34.960 tok/s | Superseded. |

The v37 run had four warmups, four accepted measurements and one excluded
attempt. Its faster rate is not treated as the current qualified result because
the float reducer could conflate distinct subnormal values on a denormal-
flushing GPU path. v38 changed the reducer to integer ordering, then passed
subnormal, ULP, tie and all-poison checks. Its scorer executable is byte-
identical to v39's scorer; v39's source adds a test-fixture admission
correction, with no release-worker binary difference. Component timings and
prior receipts remain in the ignored runtime directory; v35 and v37 aggregates
are summarized in the JSON.

## Runtime policy and limits

Activations, recurrent state and caches remain F32. Packed GGUF weights and
quantization are unchanged. Opt-in F32 reassociations include grouped Q4_K,
Q5_K and Q6_K matrix products; IQ2_XS/IQ3_XXS signed-grid factoring and IQ4_XS
vector reductions; split/fused parallel control projections and RMS; 16K
parallel RMS partial/apply; and Q8 grouped-eight dot products with the block
scale applied after the local dot. These paths use F64/original-source
checks and safe fallbacks where required; the 16K RMS path is not bitwise
equivalent to the serial reduction. The 4K RMS prefetch component is bitwise
checked. KDA state-layout transpose and fused state have bitwise component
checks; KDA projection math is unchanged. Router output matches the prior GPU
implementation, with CPU-reference exponential ULP differences retained as
measured.

The Q4_K kernel is Stark's specialization of the MIT-licensed Q5_K schedule
in `ggml-org/llama.cpp@d81235049384534c167caea52b85a694f6103d14`
(`ggml/src/ggml-metal/kernels/mul_mv.metal`). The adapted shaders retain
their upstream copyright, revision and license notices. Q4 arithmetic has
its own F64 and trained-model score checks.

The GLM encoder path caps context at 4096 tokens; this comparison used 512.
The explicit wide-encoder experiment uses a 95% global working-set cap only
with its opt-in selectors. The active profile records 99.04 GB of expert banks,
an 8 GiB static reserve and a 1.60 GB dynamic-state allowance, totaling
109.23 GB against a 109.68 GB cap on a 115.45 GB recommended working set.
Default 75% per-encoder admission and OS wired-memory limits are unchanged.
Admission is fail-closed, and Metal residency requests do not guarantee that
all resources remain resident. The separately opt-in scratch heap reserves
1 GiB, bounded by two graph allocations of 512 MiB each; it does not alter
weight allocations.

The exact v39 experiment flags are captured in
`runtime/glm53-optimization-20261101/final-screened-v39.json` and the wrapper
[`with_glm53_optimized.sh`](../../scripts/with_glm53_optimized.sh). Contexts
above 4096 are unsupported by this experimental encoder path.

The [IQ factored path](../archive/experiments/glm53-iq-factored-2026-11-01/README.md),
[fused control path](../archive/experiments/glm53-control-fused-2026-11-01/README.md)
and [packed finite fusion](../archive/experiments/glm53-packed-finite-2026-11-01/README.md)
were rejected and remain archived with their test evidence. No production
engine default changed.
