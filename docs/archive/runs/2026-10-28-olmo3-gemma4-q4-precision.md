# OLMo 3 and Gemma 4 Q4 precision investigation

Investigation dates: 2026-10-28 through 2026-10-29. Pre-repair branch revision: `9257d80ae6fab1c83439c46c845326bbebfcb0c7`. Final production implementation: `598b3f6eaddf2b0b87a08032b92d2e26e4288358`. Both families pass their 14-case stock-MLX numerical suites with unchanged gates. Fresh native, CPU, worker and canonical BF16/image checks pass. The [final matched comparison](../../benchmarks/OLMO3_GEMMA4_Q4_REPAIRED_2026_10_29.md) measures OLMo at 1.0155 times MLX's work rate and Gemma at 0.9785 times on the pinned pp512/tg128 workload. Gemma retains a 2.15% work-rate gap on this host.

This record tracks repairs to the affine-Q4 numerical failures and the measured MLX performance gap. The [original native validation record](2026-10-28-olmo3-gemma4-native.md) remains unchanged, including its passing BF16 results, failed Q4 gates and diagnostic timings. This investigation does not change acceptance tolerances. Primitive agreement and diagnostic controls are recorded separately from complete model qualification.

## Sources, host and reference policies

The host is Apple M5 Max with 128 GiB unified memory, macOS 26.7.1 build 25G241. The primary Q4 oracle is stock MLX-LM `1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7` with MLX 0.32.3. Inspected core MLX v0.32.3 source is `64ea011cb65f14d9ce2737e60db9a4ae91ed7441`. Both source projects carry Apple's MIT license. The executed reference environment uses Python 3.13.13 and a frozen lock with SHA256 `caa51ccf8c64acf6a850765121808930be0caf88075ef9a6095925953b62063b`.

| Family | Immutable Q4 source | Source weights SHA256 |
| --- | --- | --- |
| OLMo 3 7B Instruct | `mlx-community/Olmo-3-7B-Instruct-4bit` at `d732c91ae02e90cd5d86e810fbeb9741794b4dd9` | `a97f02f033af038a222c4ae3cd22a3df580160d559dc985c6a6a2d512b5210cd` |
| Gemma 4 E2B | `mlx-community/gemma-4-e2b-it-4bit` at `238767527555cb75a05732a84dff5d6ba0dd6809` | `038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e` |

Both exports use affine four-bit weights with group size 64. The repair retains packed weights and matches the reference's phase-specific precision: F32 product/dot accumulation on the audited GEMV route, BF16 partial scratch and typed reduction on the source QMM route, and explicit BF16 activation materialization. Activation, accumulation, cache and weight precision are separate policies. Existing full-precision evidence remains a reference; full F32 inference is not a production requirement.

Canonical Hugging Face Gemma controls use their own pinned model/processor and locked environment. Their FP32 opmath and BF16 materialization differ from stock MLX. A canonical-HF or deliberately modified-MLX control pass does not qualify the stock-MLX Q4 path. CPU and GPU primitive controls also retain distinct provenance.

## Preserved failures and intermediate controls

At the pre-repair revision, OLMo Q4 passed four of five numerical cases. Its one-token case had raw RMSE 0.127956 against a 0.1 cap, and centered RMSE 0.047431 against a case bound of 0.047090. Gemma Q4 passed three of five cases; the 23- and 129-token cases failed its unchanged numerical profile. Next-token greedy agreement held in all ten cases. Existing worker replay successes did not resolve those numerical failures. Diagnostic combined work rates remained 14.0% below MLX for OLMo and 2.4% below for Gemma.

The following later comparisons remain failed or diagnostic evidence:

| Intermediate artifact / reference | Numerical cases passed | Meaning |
| --- | ---: | --- |
| OLMo materialized-sigmoid control / stock MLX | 4/5 | The one-token failure remained. |
| OLMo RMS-plus-sigmoid control / stock MLX | 3/5 | One- and 50-token failures remained; matching individual primitives was insufficient. |
| Gemma embedding materialization v2 / stock MLX | 2/5 | 23-, 129- and 517-token failures remained. |
| Same Gemma artifact / canonical HF | 5/5 | Confirms agreement with that control profile only. |
| Gemma embedding-plus-softcap v3 / stock MLX | 2/5 | The same three source-profile failures remained. |
| Same softcap artifact / canonical HF | 5/5 | The canonical HF control remained valid. |
| Gemma MLX-profile v3 / stock MLX | 2/5 | Scalar adjustments alone did not close source disagreement. |
| Same artifact / MLX canonical-scalar control | 4/5 | The 517-token failure remained. |
| Same artifact / HF-opmath control | 5/5 | Demonstrates the distinct arithmetic profile; it is not the primary Q4 acceptance result. |
| Gemma uniform-F32-partial candidate / stock MLX, five cases | 2/5 | 23-, 129- and 517-token failures remained. This candidate was rejected. |
| Same F32-partial candidate / stock MLX, nine sequence positions | 3/9 | Contexts 512, 544, 560, 576, 592 and 608 failed. This candidate was rejected. |
| Source-aligned prefill v1 / stock MLX, OLMo five cases | 2/5 | One-token output is exact; 46-, 50- and 4,101-token cases fail. Final cached-token computation remains unqualified. |
| Same OLMo candidate / nine fresh-prefix sequence positions | 6/9 | Contexts 528, 560 and 640 fail. All 14 next-token greedy choices agree. |
| Source-aligned prefill v1 / stock MLX, Gemma five cases | 2/5 | The 23-, 129- and 517-token cases fail; the last greedy choice also differs. |
| Same Gemma candidate / nine fresh-prefix sequence positions | 4/9 | Complete 511-row prefill is exact at embedding, all 35 layer outputs and final hidden state. The following populated-cache M=1 step remains unqualified. |

The final production choice must pass the primary reference with the existing gates. No tolerance is enlarged to convert an intermediate failure into a pass.

## Isolated causes and repairs

### Affine Q4 GEMV input-sum materialization

The largest shared discrepancy was in the affine bias correction. MLX's BF16 single-vector kernel sums each four-input group through BF16 additions before converting that sum to F32 for the bias contribution. Stark previously summed those input values in F32. On this GEMV route, dot products, quantization scales and cross-lane reductions remain F32 while the input sum follows the source's BF16 boundaries.

An actual 128-row OLMo Q-projection control differed from the source in 92 outputs under the original expression, with maximum absolute difference 0.00048828125. Reproducing the MLX expression reduced that control to zero differing outputs. A change to the wide accumulation fold alone left all 92 disagreements. The shared repair is an explicit `MLXSpec.materialize_affine_input_sum` option, defaulting to false, threaded into the selected BF16/Q4 affine GEMV route. It is enabled only for the declared source profiles. Existing metadata preserves its previous policy by default.

Implementation locations include `config/weight_matrix/mlx_spec.rs`, common matmul arguments, the Metal GEMV affine input-sum path, and the CPU reference. A focused cancellation-sensitive test checks source BF16 group boundaries and the legacy default. This primitive result does not establish whole-model acceptance.

### Matrix execution phase and expression schedule

Later source controls showed that matching the four-input bias sum alone was insufficient. The source M=1 path also needs the generic eight-value lane form, four-product subtotals, and the source F32 expression fold. Its wide F32 dequantization expression must preserve the source operation schedule. F32 storage or accumulation by itself does not fix a changed expression tree.

With those source operations, the OLMo single-token control matches all 35 recorded components and all 100,278 final logits exactly. Its layer-16 operator control also matches all eight recorded operators. Gemma's corresponding single-token control matches all 262,144 final logits; one element in layer 11 differs by 0.000030517578125 and that difference disappears in the next layer. These are bounded M=1 controls, not complete five-case or sequence qualification.

Multi-token source QMM follows another precision schedule: BF16 partial-result scratch and the source's typed eight-lane strided reduction. Original projection widths influence that schedule, so fusing projections must preserve their individual dispatch/reduction geometry. The uniform-F32-partial alternative failed the complete Gemma controls above. Source GPU and CPU small-reduction controls separately reproduce the three audited projection branches under the typed reduction policy.

Native source prefill preserves each logical projection's original width while retaining the packed tensor's physical scale/bias strides. It applies the source architecture/size thresholds to choose wide-vector or matrix execution. Split QMM uses classic SIMD-group matrices with BF16 shared operands widened to F32 registers, BF16 partial scratch, and the source reduction. Unsplit M5 QMM uses the native matrix unit with BF16 operands and F32 accumulation. Actual Gemma QKV controls match all 46,080 values at M=18, all 327,680 at M=128, and all 1,308,160 at M=511. These isolated controls precede the whole-model acceptance result.

### OLMo SwiGLU and RMSNorm

MLX's BF16 sigmoid materializes the exponential, denominator, reciprocal and sign-dependent result before the SwiGLU multiplication. A fused F32 activation expression changes those boundaries. On 11,008 actual first-layer gate values, the original native expression differed in 4,682 elements; a generic fused expression differed in 5,646. Reproducing the complete MLX sigmoid expression yielded zero differences in the control.

Stock MLX RMSNorm computes statistics/normalization in F32, then materializes the normalized BF16 value before applying its learned scale. This corresponds to `OnlyNormalization`; canonical Gemma HF uses the separately recorded `FullLayer` policy. An OLMo control had 1,071 differences under `FullLayer` and zero under `OnlyNormalization`. The OLMo affine-BF16 conversion path selects the source norm and sigmoid policies. These policies are included in the final numerical qualification below.

### Gemma embedding, scalar and GELU boundaries

- Quantized token embeddings must materialize the BF16 dequantized value before applying the BF16 embedding scale. Scaling a wider dequantized value first changes the result.
- The BF16 softcap profile materializes division, `tanh` and multiplication as separate operations. A single wider fused softcap does not preserve this source expression.
- Stock MLX's approximate GELU uses BF16 weak coefficients and intermediate polynomial/tanh results. The implementation adds an explicit materialized-polynomial activation while retaining the canonical activation profile. The source GPU control uses actual 6,144-element MLP gate input and 256-element PLE gate input. Metal tests pass for both the actual-source input control and the independent primitive-boundary test.
- PLE scaling and learned RMSNorm scaling follow the selected profile. On 1,536 actual Gemma norm values, `FullLayer` differed in 368 elements, with maximum absolute difference 0.0625; `OnlyNormalization` had zero differences against stock MLX.

Gemma's explicit `mlx-bf16` planner profile combines the source-compatible norm, GELU, scalar and affine-input-sum policies. The final five-case and nine-position sequence results below qualify the tested artifact and binary against stock MLX. Canonical HF remains a separate profile with preserved controls and evidence.

### Source RMS reduction and value-normalization inheritance

Matching the normalization storage boundary also requires matching its reduction tree. The source regular RMS kernel assigns four contiguous inputs per thread; the older native kernel distributes strided inputs across up to 1,024 threads. On an actual Gemma layer-1 post-attention input, the existing native result matched the F64 control at every element, while stock MLX differed at seven BF16 rounding boundaries. The repair therefore introduces an explicit source reduction profile; it does not replace the canonical normalization policy. A reciprocal-square-root substitution alone had no effect and was rejected.

The default-false `NormalizationConfig.mlx_bf16_reduction` selects the audited BF16 source reduction, normalization and learned-scale schedule. Packed query/key normalization reproduces the source's logical SIMD-group partials before the final sum. Actual Gemma layer-1 regular and packed normalization controls become exact. A synthetic BF16 midpoint control independently distinguishes the source and legacy trees, and verifies the legacy result after a source invocation. Unsupported normalization combinations fail validation.

Gemma value normalization is constructed at runtime from the key-normalization policy. It previously discarded the source reduction/upcast settings. Inheriting those explicit settings reduces the global layer-4 value-normalization control from five differing values to zero across 261,632 values. Artifacts without the source setting retain the earlier runtime policy.

### Attention score, probability and online reduction schedules

OLMo's source NAX attention scales completed F32 QK scores. The earlier native path scaled Q before the matrix product, changing BF16 operands. A source-only score-scale specialization eliminates the layer-0 disagreement across 200,704 attention outputs. The last layer-1 disagreement comes from the online softmax row reduction: the source pairs four local columns, shuffles XOR 1 then XOR 8, and folds each fragment into the already-rescaled mass. Reproducing that tree eliminates the remaining one-value disagreement while preserving the exact layer-0 result. An independent positive-mass midpoint test distinguishes source and legacy reductions.

The next isolated OLMo discrepancy appeared at layer 11: one of 200,704 attention outputs differed despite exact prepared Q/K/V operands. The source NAX finish first rounds a reciprocal in F32, then multiplies the F32 output accumulator and materializes BF16. Keeping that complete expression in a source-only Safe, contraction-enabled helper eliminates the difference and preserves the exact layer-0 and layer-1 controls. An independent reciprocal-boundary oracle passes. Earlier finish and scalar-helper controls that had no effect at layer 1 remain preserved as rejected intermediate evidence.

Gemma's audited multi-token source attention materializes BF16 QK scores and BF16 probabilities before the value product. The source-compatible fallback is bounded to causal, unit-scale BF16 D=256/512 attention with 9 through 1,023 query rows. On the actual 511-row layer-0 control, it eliminates 686,908 disagreements across 1,046,528 outputs. The next layer exposed nine probability differences despite exact QK scores; an explicit global-maximum softmax with the source four-contiguous-value reduction eliminates all nine across 2,088,968 probabilities and all 52 downstream attention differences. Independent masked, tail-row and F64 probability controls pass. Other attention profiles retain their existing dispatch.

The complete 511-token Gemma prefill subsequently matches embedding, all 35 layer outputs and final hidden state exactly, but the next populated-cache M=1 step first differs at global layer 4. All 11 same-input linear/regular-normalization operators and all five prepared Q/K/V norm/rotary stages match at absolute position 511. Direct vector attention with the source Q and full 512-position K/V cache differs in 1,595 of 4,096 outputs. The owning source eligibility check explains the dispatch mismatch: MLX rejects fused D=512 vector attention below its default 1,024-key minimum before reaching the generic vector selector. The explicit source policy therefore also selects materialized BF16 attention for D=512 M=1 with fewer than 1,024 keys. D=256 M=1 retains vector attention. The production-route cached control then matches all 4,096 source outputs exactly. An independent two-key rounding oracle and routing checks at 1,023/1,024 keys pass, including legacy and D=256 exclusions. These controls precede the complete model qualification.

### Source vector compiler policy and dense QK split precision

After the complete 49-token OLMo prefill became exact, its cached final token first differed at local layer 2. All same-input normalization and rotary stages were exact; identical source Q and 50-position K/V operands isolated one differing BF16 attention value among 4,096. Gemma's cached local layer 16 similarly isolated one differing value among 2,048. MLX compiles its vector attention with `-fno-fast-math`, while the earlier native vector kernel used relaxed arithmetic. A separate source-only Safe, contraction-enabled kernel closes both controls. The existing vector kernel and its default dispatch remain unchanged; the OLMo legacy control reproduces its original output hash and one-value disagreement.

The source vector route is limited to BF16, D=64/128/256, one flat causal query, fewer than 1,024 keys, no trie, and no sinks. An independent seven-key uniform-average oracle checks a BF16 midpoint where precise F32 division and multiplication by a separately rounded reciprocal choose different BF16 endpoints. It passes at all three supported dimensions. With the production source route, the complete OLMo cached-token trace is exact through embedding, all 32 layer outputs, final hidden state and all 100,278 logits: 35 of 35 recorded components. Gemma's complete cached-512 trace is exact across all 38 recorded components, including its 35 layers and logits.

The remaining 23-token Gemma case exposed one valid BF16 QK-score disagreement in shared global layer 24. All other same-input operators and source-probability value products were exact. An independent F64 dot-product control followed by BF16 rounding agrees with stock MLX at the contested boundary. Forcing different output batching alone does not fix the score. Matching the source's two-way dense split-K with F32 partial scratch and F32 reduction, followed by one BF16 score materialization, makes all 2,024 valid scores, probabilities and 90,112 attention outputs exact.

This dense QK repair is restricted to the explicit materialized BF16 source profile, D=512, eight query heads and one KV head, query rows 9 through 1,023, an MXU-capable device, and the audited source split predicate after wide-vector exclusion. The inner dimension is 512 with two 256-element K partitions. Quantized source QMM retains its separately validated BF16 partial-scratch and typed-reduction policy. Independent source-QK routing and rounding tests pass; these controls do not broaden qualification to other dense shapes.

### Cached trajectory: physical window order and dense-wide QK

The initial precision repair at `b53320a1122358ef95740d6178e0ef6d5429ca97` passes both 14-case suites and the bounded generation/scoring worker checks described below. The longer matched benchmark nevertheless exposes a Gemma output-hash disagreement across 128 returned tokens. A separate stock-MLX cached trajectory locates the first difference at absolute input position 512, local layer 8, after an exact 511-token prefill and exact first cached step.

All prepared Q/K/V values agree. The source replaces its rotating-cache slot before attention; native storage appends the pending suffix and masks the old slot until acceptance. Those representations contain identical visible values, but distribute them differently across the source online F32 reduction. The old 513-entry representation differs in one of 2,048 BF16 attention values. Compacting those same native buffers into the source physical 512-entry order eliminates the difference. The native repair instead maps the source traversal onto the existing storage without a KV copy. It records the source origin after multi-token acceptance and retains that origin across M=1 accepts. Independent powers-of-two cancellation controls cover origins 0, 4 and 511, later single-token phases, pre-wrap state and legacy negative controls.

The corrected ring makes the first eight source forwards exact through all 35 layers and logits. The next difference, at absolute position 519 in global layer 4, is separate: all prepared operands agree, but two of 4,160 BF16 QK scores differ, leading to one probability and four of 4,096 attention-output differences. The value product is exact when given source probabilities. Switching the generic dense engine or scratch precision does not reproduce the source. An F64 classification favors the source score at one boundary and the earlier native score at the other; this is a compatibility repair for the declared arithmetic profile.

The source collapses eight D512 query heads into M=8 and selects its dense `GemvWide` kernel before GEMM. A bounded native adaptation preserves its staged float4 dots, four query vectors per group, local accumulation and halving shuffles before the BF16 score store. The new production route makes all 4,160 scores and probabilities and all 4,096 attention outputs exact on the causal control. It is restricted to the validated BF16 source profile, one query, eight query heads, one KV head, unit scale, no ring or sliding window, 2 through 1,023 keys, and the qualified MXU-capable host path. The complete public 512-input/128-return stream then matches the stock-MLX output hash `03f17cff595a19aba064fb7ddf00547b52fd0cdf3c908b77db96f2727aba52a1` in all four repeated runs of binary SHA256 `90daa697db02c3ec7077e2e00d6361538ff4df6dd34d85c52a07e5b9e1ae05ca`. Those runs establish output agreement; their timings are excluded because other CPU work was active.

### GPU rotary coefficients, precise powers and YaRN prescaling

The earlier native path calculated sine/cosine tables on the CPU. The source calculates GPU frequencies and uses fast GPU trigonometric functions. An explicit source preparation path reproduces those operations for the supported unscaled, proportional and YaRN configurations. OLMo YaRN also materializes its weak BF16 attention factor before rotation; folding the wider factor into a CPU trigonometric coefficient changes the result.

Custom proportional/YaRN frequency generation exposed another compiler boundary. Under Stark's Metal compiler defaults, `metal::pow` is selected through `__METAL_MATH_FP32_FUNCTIONS_FAST__`; MLX's precompiled kernel build disables that default. A Safe pragma after the Metal headers cannot change their preprocessor selection. The source path now calls `metal::precise::pow` explicitly. The actual Gemma proportional frequency control changes from 23 differing coefficients to zero across 256. Its global layer-4 rotary outputs then match all 2,093,056 Q values and 261,632 K values. OLMo's remaining 11 Q and six K YaRN differences also become zero; all eight prepared-attention stages match across 200,704 values each.

These source-only flags are opt-in and are emitted automatically only for imported affine W4/group-64 matrices with BF16 activations. Generic affine import coverage does not establish the same source arithmetic profile for other bit widths, group sizes, checkpoints or GPU architectures.

## Hypotheses ruled out or insufficient

- The audited first OLMo QKV and MLP projection codes, scales and biases match the source exactly. No packed-weight corruption was found in that bounded audit.
- Across 131,072 scalar dequantization values, the original wide and typed expressions both matched the source. Blanket BF16 dequantization introduced 76,800 differences. The repair therefore targets the identified embedding boundary and GEMV bias-correction sum rather than indiscriminately rounding every dequantized weight.
- Changing the F32 reduction fold alone did not fix the 128-row projection control. Source inspection and operation-level controls isolated the four-input bias correction.
- A uniformly F32 partial-result buffer did not reproduce source QMM: the candidate passed only two of five Gemma cases and three of nine sequence positions. The repair must preserve source partial storage, reduction order and original projection geometry.
- Matching sigmoid or RMSNorm alone did not pass the full OLMo suite. Matching canonical HF opmath did not pass stock-MLX Gemma. These controls locate causes without replacing the primary oracle.
- Algebraic equivalence and CPU-only primitive results were insufficient for the Gemma activation decision. Actual source GPU gate values were tested on Metal before accepting the materialized GELU implementation.

## Performance investigation

The earlier cold-weight synthetic Q4/group-64 GEMV sweep covered 71 configurations and remains historical evidence for that tested kernel. The later source-v4 sweep covers 16 matrix shapes under both legacy and source arithmetic policies, with 286 recorded tile/policy cases. Every v4 comparison is bitwise equal within its named policy on the 268,435,456-byte synthetic working set.

The new large-M5 source-only dispatch table selects these six projection shapes:

| Output rows N | Input width K | SIMD groups / output rows per SIMD group |
| ---: | ---: | ---: |
| 4,096 | 4,096 | 4 / 2 |
| 4,096 | 11,008 | 4 / 2 |
| 22,016 | 4,096 | 8 / 2 |
| 12,288 | 4,096 | 2 / 4 |
| 100,278 | 4,096 | 2 / 4 |
| 12,288 | 1,536 | 2 / 4 |

The dispatch is restricted to the explicit source profile, Apple10 large-GPU geometry and group size 64. Existing legacy metadata keeps its legacy policy. That default-policy boundary is not a universal bitwise guarantee across recompilations: an audited legacy Gemma single-token trace changed one element in layer 23 when the compiler context changed. Report each compiled variant's actual control results instead of inheriting an earlier exactness claim.

This sweep is kernel evidence. It does not establish an end-to-end speedup. The final comparison will use the guarded aggregate-only driver, alternating engine order and screening host contention, with identical source/input identities, one warmup and three measured requests per process. Each request includes 512 input tokens, 128 returned greedy tokens, 128 decode forwards and final lookahead drain. The primary rate is `640 / median_request_seconds`; a standalone decode rate uses a different timing boundary.

The comparison driver snapshots benchmark executables and the MLX harness before a campaign, freezes command/environment selection, and binds source, model, precision, device and output identities across all process loads. It rejects the campaign if an executable, fixture, harness or bound native artifact changes. The precision-matched baseline and candidate must differ only in the implementation change being measured.

Sequence-reference protocol `dense-q4-sequence-v3-cache-state-only-prefill` now matches the benchmark's discarded prefix readout and cache-state-only prefill evaluation. Cached-generation-trajectory references and freshly recomputed-prefix references carry separate profile labels. Fresh native prefill comparisons must use the recomputed profile; cached trajectory diagnostics retain their original meaning. Earlier v1/v2 receipts remain unchanged.

## Initial repaired numerical qualification

The final stable Metal comparison binary is `source-qk-safe-v1-dense-parity`, SHA256 `f7940b16e2afda301fc104d583e2003cae1a6dc93a7e15cf4caf7e507a10176e`. The corresponding physical-control test binary is `source-qk-safe-vector-v2-test`, SHA256 `a693948c970a7d967da20233ecded91991c5d64adf33822f4f89a8c3657cf391`. These immutable binary identities bind the numerical results before the final repair commit is assigned.

Both families pass the original five-case suite and the nine fresh-prefix sequence positions under the existing gates. Each score evaluates the prefix and then the last token with the populated cache. The nine-position references use protocol `dense-q4-sequence-v3-cache-state-only-prefill`. The OLMo gate label is `bf16-centered-logits-v3`; Gemma retains the existing `gemma4-canonical-bf16-v4` gate label while comparing against the pinned stock-MLX Q4 references described above. That label does not change the reference oracle.

| Family / suite | Strict cases passed | Greedy agreement | Full-vocabulary bit-exact cases | Maximum centered RMSE | Maximum centered absolute error | Maximum reference-to-native KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OLMo, original five cases | 5/5 | 5/5 | 4/5 | 0.0358055215 | 0.184105578 | 0.00000172058 |
| OLMo, nine sequence positions | 9/9 | 9/9 | 9/9 | 0 | 0 | 0 |
| Gemma, original five cases | 5/5 | 5/5 | 5/5 | 0 | 0 | 0 |
| Gemma, nine sequence positions | 9/9 | 9/9 | 2/9 | 0.145445013 | 0.712216981 | 0.00498084664 |

The OLMo original lengths are 1, 46, 50, 129 and 4,101 tokens; only the 4,101-token case is non-exact. Its raw RMSE is 0.0971913874, within the existing 0.1 cap, and its centered and distributional errors also pass. Gemma's original lengths are 1, 19, 23, 129 and 517 tokens. Both sequence suites cover 512, 528, 544, 560, 576, 592, 608, 624 and 640 tokens. These results qualify the tested checkpoint, precision profile, binary and host; they do not establish arbitrary-context or task-quality coverage.

The final qualification runs were serialized. OLMo's initial `olmo3-qk-safe-*-v1` calls briefly overlapped after a child process yielded its session identifier. Those reports remain preserved and are excluded from qualification. The `-v2` original-five and sequence-nine reruns each completed with exit code zero before the next GPU process started; the separate serialized-qualification receipt records this distinction. All failed and intermediate numerical records above remain unchanged.

### Converted artifact identities

Both tested v5 artifacts were produced through the normal converter CLI with fresh configuration and conversion receipts. Complete exported weight hashes were verified before identical prior weights were used for APFS copy-on-write storage deduplication. The converted weights retain the source affine W4/group-64 representation.

| Artifact field | OLMo 3 7B Instruct v5 | Gemma 4 E2B v5 |
| --- | --- | --- |
| Native directory | `olmo3-q4-source-bf16-v5` | `gemma4-e2b-q4-source-bf16-v5` |
| Tensor count | 519 | 901 |
| Weights SHA256 | `86eca5f98623ae556c4af1e681529579c10fac9bc710801bf6d182679a265400` | `6b340c2c5de885e6a29fd72671cd89ab878d7d582863a6ef1775ee4da81038ea` |
| Configuration SHA256 | `6c55239ab67ac5c3bb7bc6605d507123657c4187a1f69d105c1e7d2726af2dc1` | `db1850dce7370c6cb797f703e48609e4b112eebb8d18ff86ce0f1bb6628875e0` |
| Source manifest SHA256 | `c973ee635c50c06eb17be245d101f6e3ea7b7f735937b1b927d4cff7f38d2ba4` | `b657d8531553412e4d04bdd14f4fa771571a07a2e021af95819df090c22d61ae` |
| CLI conversion report SHA256 | `34b05ebe4ce0aaf9d452becd238323066f07da98fac374ae10e28073e56e9ffb` | `1362c2458c6c9f0b10bc9e1fc29607f83732048c2ab94c9220b4d86ea890e7a5` |
| Artifact `conversion.json` SHA256 | `c67580f7a4fc15c213b228ba30f8a8b4cde32e2f3882876593e4cdf65fc4e3c1` | `b248e76393ba22adba4877b300661a43f8560f7c63e5c5b8518aed3899289b62` |

The CLI reports and artifact receipts contain identical JSON values with different serialization. Worker and benchmark receipts bind the exact bytes of each artifact's `conversion.json`.

## Final qualification and benchmark status

| Required evidence | OLMo Q4 | Gemma Q4 |
| --- | --- | --- |
| Production implementation revision (C3) | `598b3f6e` | `598b3f6e` |
| Normal-CLI v5 converted-artifact identities | Verified above | Verified above |
| Stock-MLX numerical suite, five cases, unchanged gates | Pass, 5/5 | Pass, 5/5 |
| Stock-MLX teacher-forced sequence suite, nine positions from context 512 through 640 | Pass, 9/9 | Pass, 9/9 |
| Generation/scoring workers, exact replay, restart and cache-failure recovery | Pass, 2 generation and 2 scoring cases | Pass, 2 generation and 2 scoring cases |
| Canonical BF16 text and Gemma image regressions | Pass, 5/5 text | Pass, 5/5 text and 3/3 images |
| Full physical native unit suite | Pass, 715 tests | Pass, 715 tests |
| Accepted quiet matched benchmark and output-hash agreement | Pass, all 4 paired blocks | Pass, all 4 paired blocks |
| Native/CPU, policy, Ruff and format checks | Pass, recorded below | Pass, recorded below |

The final normalization section below binds these results to its production binaries and separate numerical, worker, regression and performance receipts. Earlier diagnostic measurements and failed numerical results remain historical.

## Selected aggregate receipt identities

All paths below are relative to ignored `runtime/olmo3-gemma4/`. Only aggregate counts/errors and hashes are published; source tensors, logits, private inputs and generated sequences remain private.

| Receipt | SHA256 |
| --- | --- |
| `olmo3-q4-initial-weight-audit.json` | `c50c76a1b6f9baace7f08bf8b7ade43363b1d3b22402c8ef18fc72aa435fd93f` |
| `q4-dequant-control.json` | `f8b9744d781faad19d1926a475289e0ef2e1971f9b3d47bc0934b3ecc5561234` |
| `q4-dot-control-v2.json` | `bd51c36883de960163bdc41546f61f3435b721f3769d083e8ac55305141248a3` |
| `q4-swiglu-control-v2.json` | `417b7043e2bca0576260be579e500bec315f23bed7c5c09e16d6cc12643c37b4` |
| `olmo3-q4-rms-source-control.json` | `0fb18850a17ca0f57524bb489a3c1b83e0079febf15dfb297da9d6e77c00500b` |
| `olmo3-q4-rms-sigmoid-parity-v1.json` | `511287ed270ae81229439e8ed35e90fc6be9a2425db0d22090b393f96e74a8df` |
| `gemma4-q4-embedding-v2-mlx-parity.json` | `b6a24c09a7022c1f0f945aaec424e4165f1541cb6cc2abf2cd921b25947cc1a3` |
| `gemma4-q4-embedding-v2-hf-parity.json` | `32c812f9fb674c26fe23dc647d237d4431b960df1d4810fe87905000027c443d` |
| `gemma4-q4-softcap-v3-mlx-parity.json` | `dcd29ab0e68dceebffaeaae80e49072b7c86df34960b1cf469c34066f30424eb` |
| `gemma4-q4-mlx-v3-stock-parity.json` | `fbb6f3db2019b9093a0b21556764bf83c835d937366a2b13cd5d5aaabf95e5cd` |
| `gemma4-q4-mlx-v3-hf-opmath-parity.json` | `9b5c0e3836d5595c73c1eb2d7de97a2a257022f56532fca1749fe5ea957b9076` |
| `gemma4-actual-primitives-mlx-gpu-v2/receipt.json` | `d189aef341626b75362ae259a16957ea8c4eb68cd02fe87aaeb90a890016fea6` |
| `gemma4-gelu-source-metal-v6.log` | `e1d00bcfe300ba29f9bbec5de4f4404b1b7a39c14f7723f9ce0e76eb2c4c0420` |
| `olmo3-quad-sum-check.log` | `f8a4a87c140124906e6ef6e1402b9b362110c40d3a1e1e7430e567f1c0ada282` |
| `q4-repair/cold-tile-sweep-v2.json` | `6d677516274f4d53cd3c9022350259db5592210f0ecb184403d2daeadafa934b` |
| `olmo3-q4-source-generic-proof-v1.json` | `23e9efb6a4ddf8da1ac49b3d139b383c5bf30c8b6211da9d17c088d70bc173ad` |
| `gemma4-q4-bos-trace-fold-comparison-v2.json` | `13d78088d7aa2a0d12029bd6a15acf0641c87920936866c7ae2abc39b272748a` |
| `gemma4-q4-source-f32-partials-v1-parity5.json` | `4259659f60a2083559301bc24c70d85f573b887cce0e4c4395101cc3ae8baff2` |
| `gemma4-q4-source-f32-partials-v1-parity9.json` | `adc84c9c4c16fa1f021c38dc35c9555c1bba454c56d8f401eca027a1a6e133cc` |
| `gemma4-q4-prefill128-source-partials-v2/mlx-gpu-reduction-v1.json` | `fd79abc77017a585ba1e21c527fe3b6907f0198e1621f7c4f8e8874387d562e9` |
| `gemma4-q4-prefill128-source-partials-v2/cpu-small-reduction-v1.json` | `ecff41380bd655902ee2e356f56c9ea64cd31d04698fcb88274adb9e9cbee08a` |
| `q4-repair/cold-tile-sweep-source-v4.json` | `f095defec431a2b6db1a9d0b6ad5eced2406bcfc3c8c34ae374b9c635075c048` |
| `olmo3-q4-source-bf16-v5-artifact-check.json` | `50b8695e2012bd1f1e876d72d4e7c338dddfb141aaaa4581e88555ca40c5c315` |
| `olmo3-q4-source-bf16-v5-conversion.json` | `34b05ebe4ce0aaf9d452becd238323066f07da98fac374ae10e28073e56e9ffb` |
| `q4-repair/gemma4-source-packed-rms-v5-identity.json` | `00e4c063e87aded61cf34f683e80a38ee859d29ca2b76c60c21542ceecbe6291` |
| `q4-repair/gemma4-source-packed-rms-v5-conversion.json` | `1362c2458c6c9f0b10bc9e1fc29607f83732048c2ab94c9220b4d86ea890e7a5` |
| `q4-repair/olmo3-qk-safe-serialized-qualification-v2.json` | `0d12992875a2ed984e1a12465d9dee74f0432abea83b04aa2f3e06b33a88ab8e` |
| `q4-repair/olmo3-qk-safe-original5-v2.json` | `c4e64a9e9cec92700d9331b63418cec16719b58e196891beba7e2259b1835966` |
| `q4-repair/olmo3-qk-safe-sequence9-v2.json` | `52a98c999cdd6e4f7c66dbc56549ebcb5be8f7ba907ff0d4c9e8ca5c1850ee6f` |
| `q4-repair/gemma4-source-qk-safe-v1-parity5.json` | `a4dd273fad555f5cf7ff1c846e166feccefb95c8a99ae3c705b3a92102f1a876` |
| `q4-repair/gemma4-source-qk-safe-v1-parity9-v3.json` | `c8bb4a71a42d5d9e85bb42b2735a96c1ba4748478a23218f0cffe8d3e0015989` |
| `q4-repair/olmo3-native-attention49-layer11-safe-finish-v1.json` | `6c509dfc5b1fd58c3cc0631364ca2a0546446864ddb8d958e6dc3255aa3403e7` |
| `q4-repair/olmo3-cached50-trace-source-safe-comparison-v1.json` | `67d0384f111efd9e4fcfa3a1031695249b3625f9c48460e5a036ef6cc6bc3778` |
| `q4-repair/olmo3-native-cached50-layer2-attention-legacy-safe-v1.json` | `fe6633180f840b9fdd8f2e176bda170abd8cd8cf7966c33ef24bcddb9d9cafe8` |
| `q4-repair/olmo3-source-safe-vector-public-guard-v1.log` | `bd40c072324b56d3d282e7b3049ec88d4dba610847ce78a3e71ce9f5719b79fa` |
| `q4-repair/olmo3-source-safe-vector-prefill-guards-v1.log` | `1555134052e67ebf84bb66a650fe5e8e24531b54b3cb8f1c392af957f68aaa70` |
| `q4-repair/gemma4-native-cached-step512-source-safe-comparison-v3.json` | `009fc9a4238865f24ff28dc29d7366f8fe9b2df5fa6feeced21352edaa2f2a90` |
| `q4-repair/gemma4-native-prefill22-layer24-production-qk-v1/report.json` | `9f6f22833c30aa031d36e7452aa67dd5b549f2901d8b5a9e152d3ba0e6885dd0` |
| `q4-repair/gemma4-prefill22-layer24-qk-f64-control-v1.json` | `10eb92c271a6e6f861883977c9deabe5c91dab3acb8e3d716db9d1af50d457ea` |
| `q4-repair/source-qk-safe-vector-v2-qk-guards.log` | `cae975169101be6caaa8e621a50083971267fa0e340faa0ee75e92e426181983` |

## Gemma cached-trajectory completion

The source Ring traversal and dense-wide QK repairs have a separate completed numerical run. The earlier table remains bound to its earlier binaries. The new Metal comparison binary is `source-cached-qk-wide-v1-dense-parity`, SHA256 `85eb60f8113de59c7462de949bb0ab7c4f58df85f68ae073bf7980ed91b802fd`. It uses the unchanged v5 artifact identities above and the same pinned stock-MLX references and `gemma4-canonical-bf16-v4` acceptance policy.

| Gemma suite | Strict cases passed | Greedy agreement | Full-vocabulary bit-exact cases | Maximum centered RMSE | Maximum centered absolute error | Maximum reference-to-native KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original lengths 1, 19, 23, 129 and 517 | 5/5 | 5/5 | 5/5 | 0 | 0 | 0 |
| Fresh-prefix sequence positions 512 through 640, step 16 | 9/9 | 9/9 | 9/9 | 0 | 0 | 0 |

The cached-trajectory check is separate from those fresh-prefix scores. After a 511-token cache prime, it compares 129 single-token forwards at absolute input positions 511 through 639, including the final lookahead. Every forward matches the source at embedding, all 35 layer outputs, final hidden state and all 262,144 logits: 4,902 of 4,902 component hashes agree, with no differing forward. The source reference with detailed step-8 capture independently preserves all 4,902 component hashes from the earlier source reference.

The trajectory test first validates the public stream and then requires its manual unforced decoder sequence to match that stream. Both, and the teacher-forced trajectory, match stock MLX across all 128 returned choices with SHA256 `03f17cff595a19aba064fb7ddf00547b52fd0cdf3c908b77db96f2727aba52a1`. The executed trajectory binary is `source-cached-qk-wide-v1-test`, SHA256 `13f551f7b368585eb81f2d2c730f57a6e2341b87048a2748c4fb9722f6f82a23`. Its receipt binds the executable, native weights/configuration, private input identities and compiled harness/Ring/cache-host identities. The aggregate receipt also records the frozen QK and fallback source identities. Later helper revisions add those two source identities as embedded fields and default to the current source output hash; historical immutable variants retain an explicit expected-hash override.

The public stream driver, SHA256 `90daa697db02c3ec7077e2e00d6361538ff4df6dd34d85c52a07e5b9e1ae05ca`, repeats the 512-input/128-return request four times. Every run returns the source hash, performs 128 decode forwards and finishes at context 640. These runs establish output identity; their timings remain excluded because concurrent CPU work was active.

These results qualify the tested E2B affine W4/group-64 artifact with BF16 activations on the recorded M5 Max host. They do not establish other checkpoints, architectures, arbitrary contexts or task quality. Worker, canonical BF16/image, complete native-suite and quiet performance checks retain their own source-bound receipts.

All receipt paths below are relative to ignored `runtime/olmo3-gemma4/`. The aggregate contains counts and identities only; input vectors, generated sequences and tensors remain private.

| Completion receipt | SHA256 |
| --- | --- |
| `q4-repair/gemma4-final-numerical-qualification-v1.json` | `87165d907d3b89c558bbf3ca78964ee589e474beafba276e1a9cab832fbe4848` |
| `q4-repair/gemma4-source-ring-qk-wide-v1-parity5.json` | `deb70e576865c4123ae6ff13ca022653173c17f3e70db37627b11efa11ba5b5c` |
| `q4-repair/gemma4-source-ring-qk-wide-v1-parity9-v3.json` | `c5cf10ab3487a34855260b8f9232e0f9052f42b6df7ac3e125739eef7af46806` |
| `q4-repair/gemma4-cached-trajectory-source-v2/trajectory-reference.json` | `207f935247b48f7b6b3836be22959720589c17c49944b421eab833197551dae0` |
| `q4-repair/gemma4-cached-trajectory-native-qk-wide-v4/trajectory-native.json` | `5a7af4fc943f45894cccdfc6badf6c9ad901fb4c30e5cd4d3334c547caf78012` |
| `q4-repair/gemma4-qk-wide-trajectory-comparison-v4.json` | `086678f61ba67394ed47a558ecd06a22a123bfedae5eddc96386207f9f02a715` |
| `q4-repair/gemma4-qk-wide-source128-profile-v1.json` | `e4e8c9ad4003dfb3475e7575e96421f52bf974d972244efd236181eb07272c94` |
| `q4-repair/gemma4-step8-global4-qk-wide-v1/report.json` | `461e45de14227146b1be3e57d6b4dbbc9a76a57fcb8cddcd1edeb7d39d427a0c` |

## Cached-attention regression checks

The corrected native suite passes 709 tests with zero failures and 54 ignored tests, using binary SHA256 `8eca6ab13c368f76de50021b40915562207e782d633b1860541565913c645c30`. The CPU/grammar configuration passes 472 unit tests, seven integration tests and three doctests, with zero failures and 21 ignored tests. Repository policy, Ruff and pinned-nightly formatting checks pass. An earlier full native run passed 708 tests and failed one newly added dispatch test because its fixture retained ring/window settings while asserting the flat global route. That test setup was corrected; the failed run remains preserved separately with log SHA256 `4fdaad737fe4799dbe29db88c9f7184c512455fb186abaed4b93c1df170350f6`.

A fresh OLMo run with comparison binary `85eb60f8113de59c7462de949bb0ab7c4f58df85f68ae073bf7980ed91b802fd` passes all 14 unchanged gates, with 13 full-vocabulary exact cases and all 14 greedy choices matching. The 4,101-token errors remain identical to the initial repaired run. Gemma's later qualification above passes all 14 cases exactly.

Fresh canonical regressions pass all five BF16 text cases for each family and all three HF Gemma image cases. All ten full-vocabulary text hashes and the complete image metrics bytes equal their preserved earlier runs. Complete weight files for the three native artifacts were freshly hashed against their conversion receipts. These checks bind the actual comparison and image-test binaries; worker and quiet performance qualification remain separate.

| New aggregate receipt, relative to ignored `runtime/olmo3-gemma4/` | SHA256 |
| --- | --- |
| `q4-repair/cached-attention-v3-numerical-regression-receipt.json` | `2b5c2a4c6fdef2b11bf755335a56581905b7f77ebfc9fb5ab444fd94b0ef6e4b` |
| `q4-repair/olmo3-cached-attention-v3-parity5.json` | `4d7c8646b948cf77d6bde7080d206fd38049522e2291380037aad510c551e781` |
| `q4-repair/olmo3-cached-attention-v3-parity9.json` | `2562ecc02dd6a58afd902d13f72df90d51d71bfb831970073f74f36647e12274` |
| `q4-repair/final-canonical-regressions-v2-receipt.json` | `a1b939674286c032753a94e8ba125075f2bd4ee7dae264f595ec0e96f755bd65` |

## C2 worker qualification

Fresh real-worker checks bind numerical implementation revision `2e96ec8672398fae1887f6160cea5f598151a010`. Generation and scoring workers were built separately with locked Rust 1.98.1. Every validation process completed successfully, with GPU work serialized through the OS lock. The earlier worker and numerical receipts remain unchanged.

The generation worker SHA256 is `77358fd17074bf76ecc69b0d95db8f59cd66daa1ab925609c8d1d96e0ccfd2d8`; the scoring worker SHA256 is `7ca13f3c7ea57849759670c9274a17ddc1e9016ba2ca14bc3e8ae49c78f97e13`.

| Validation | Cases passed | Recorded lifecycle checks |
| --- | ---: | --- |
| OLMo Q4 generation | 2/2 | Exact replay, invalid-request recovery and restart replay |
| OLMo Q4 scoring | 2/2 | Exact cache replay, clearing after failure and invalidation by an intervening request |
| Gemma Q4 generation | 2/2 | Exact replay, invalid-request recovery and restart replay |
| Gemma Q4 scoring | 2/2 | Exact cache replay, clearing after failure and invalidation by an intervening request |
| Gemma canonical BF16 image worker | 3/3 | Reference agreement, prepared-input single use, fresh/restart replay, malformed-image recovery, counterfactual image response and staging cleanup |
| Gemma image client | 2/2 | Cold and cached checks with staging cleanup |

The image checks use the separately qualified BF16 image artifact, with weight SHA256 `5c26289c1536f6afda47e62616c02ae63df489dc5c37d6034624413b2fd75993` and configuration SHA256 `9acc540130bd576271650c3805467141136570ffc4111c186e317442cc3dc762`. They do not extend the Q4 text-artifact qualification to quantized vision weights.

Matched performance was pending at the C2 worker handoff. The following C3 qualification and its separate matched comparison record the subsequent normalization optimization and measured remaining gap.

| Worker receipt, relative to ignored `runtime/olmo3-gemma4/` | SHA256 |
| --- | --- |
| `q4-repair/cached-qk-v2-workers-receipt.json` | `26cb2b17a3e12e445dd0229431cf2d23b4d58dd63913a5211fa655e20b098d2f` |
| `q4-repair/cached-qk-v2-olmo3-workers.json` | `fa5bb727a25a7712c5e6eb45f01ed64b11c78dec2d390d0e4e8a6691e5090df2` |
| `q4-repair/cached-qk-v2-gemma4-workers.json` | `a12bec1780dc07d5d6419acfb4e7ef19fa7fac051f41ca9b1f0917047380f517` |
| `q4-repair/cached-qk-v2-gemma4-image-workers.json` | `175de3ea6a825f89ae175f8b590011b46fbae042f320b787f0ec308ca7098d76` |
| `q4-repair/cached-qk-v2-gemma4-image-client.json` | `7af813bf6a0c785bbef724337d9dd8d3db0c377fc56cf2e540e080bb07f734be` |

## Production normalization optimization and final qualification

Production revision `598b3f6eaddf2b0b87a08032b92d2e26e4288358` reuses the existing source RMS kernel for contiguous single-token Q/K buffer ranges. OLMo's packed 4,096-element normalization previously emulated 32 logical reduction groups in one hardware SIMD group. The ordinary normalization kernel runs those groups in parallel while retaining the source four-value reduction and BF16 materialization. Gemma batches its eight query-head rows or single key-head row at width 256 or 512. Shared-KV consumers normalize their query rows only.

The route remains restricted to the explicit BF16 source policy, learned scales, `OnlyNormalization`, zero scale offset and the qualified shapes. Multi-token rows, weightless V, canonical BF16 profiles and other shapes retain their previous routing. Model-free CPU and Metal controls cover original-versus-parallel output bits, separate gain vectors, owner/shared layouts, key-only and skipped-query operations, legacy exclusions, padding and untouched channels. No private performance switch or timestamp instrumentation remains in the production route.

The frozen stable comparison binary has SHA256 `be192964f8765cdca31711397140c760651b569789f2b2929c9d71e6fd6a9c2c`; the stable timing binary is `ca1ad5a405afb30c88a6a29b8cef34d213c11a039ae8c08267f880f969b35e51`. The nightly physical-test binary is `503475f5e9698dffdec7c86afcf8ce012246a643d63eff3dc39ad461d3727b67`. Rust 1.98.1 generation and optimized scoring workers were built separately: `b20bbc8b337f64b2d1d46ee279fd2e858a8e45f193ab05432090486530586f71` and `8761ea9557757297eed4413ee6c03ab89fb71e1ccbc3169b8e64ab1279f88953` respectively.

All GPU validation processes used the OS lock and completed before their next dependent step. The quiet timing campaigns began after model qualification and CPU/build work released their resources. Their final source and artifacts remained unchanged.

| Final evidence | OLMo 3 7B Q4 | Gemma 4 E2B Q4 |
| --- | ---: | ---: |
| Strict numerical cases, original gates | 14/14 | 14/14 |
| Next-token greedy agreement | 14/14 | 14/14 |
| Full-vocabulary bit-exact cases | 13/14 | 14/14 |
| pp512/tg128 output agreement with stock MLX | All runs | All runs |
| Generation worker cases with recovery/replay/restart | 2/2 | 2/2 |
| Scoring worker cases with consecutive-cache replay, invalidation and failure clearing | 2/2 | 2/2 |
| Canonical BF16 text regressions | 5/5 | 5/5 |

The non-exact OLMo case remains the 4,101-token input: raw RMSE `0.0971913874`, centered RMSE `0.0358055215`, centered maximum absolute error `0.184105578` and KL `0.00000172058`. It passes the existing gates. Gemma's complete 129-forward trajectory matches all 4,902 component hashes, including embedding, every layer, final hidden state and logits. Public stream, manual generation and teacher-forced execution all retain output hash `03f17cff595a19aba064fb7ddf00547b52fd0cdf3c908b77db96f2727aba52a1`. The matched OLMo stream retains `a71d52d60eff63c4060c047d2d0b756655facee0cc6892ec3cfaa6d3a9462561`.

All ten canonical BF16 text output hashes equal the preserved earlier runs. The three canonical HF image cases pass, with complete aggregate metrics bytes unchanged. Fresh BF16 image workers pass all three lifecycle cases; cold and cached image-client checks pass both cases. This image evidence remains bound to the separate BF16 vision artifact. Maximum scoring-worker absolute error is `0.1875` for OLMo and zero for Gemma under the unchanged `0.75` bound.

The full native unit binary passes 715 tests, with 54 ignored. The CPU/grammar configuration passes 474 unit tests, seven integration tests and three doctests, with 21 ignored. Repository policy, staged-file policy, nightly format, Ruff and diff checks pass. The benchmark guard suite passes all 15 tests. The unchanged converter suite retains its earlier 5,065-pass/90-skip evidence.

One final worker attempt selected `gemma4-workers-final-fixture` instead of the Q4 reference fixture and failed `score_reference_error_exceeds_tolerance`. Its failed aggregate remains preserved. The correct `gemma4-q4-reference` rerun passes without changing production source, reference values or tolerances. The consolidated numerical summary also corrects an obsolete private-switch scope label copied into its first summary; the underlying production-binary reports remain unchanged.

### Measured performance and experiment disposition

The final quiet comparison includes every accepted block from four independent loads per engine, one warmup and three measured requests per load. Every request has 512 input tokens, 128 returned tokens, 128 decode forwards and a final lookahead drain. Loading is excluded. The rate counts all 640 work tokens per complete request and is not a standalone decode rate.

| Model | C2 precision-matched baseline, s | C3 production, s | MLX, s | Paired C3/baseline work-rate ratio | Paired C3/MLX work-rate ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| OLMo 3 7B | 1.567938 | 1.259285 | 1.277698 | 1.241573 | 1.015542 |
| Gemma 4 E2B | 0.646835 | 0.627277 | 0.614090 | 1.031643 | 0.978486 |

Times are medians of process medians; ratios are medians of paired ratios. Paired request-latency reductions are 19.46% for OLMo and 3.07% for Gemma. The earlier C2 baseline's MLX rate ratios were 0.8157 and 0.9489 in these same final campaigns. These measurements qualify the recorded M5 Max, checkpoints and text workload. Gemma remains 2.15% behind MLX by paired work rate.

| Investigation | Disposition |
| --- | --- |
| Parallel contiguous Q/K RMS | Retained after bitwise controls, full numerical suites and matched end-to-end improvement |
| Scale-plane transpose, direct-nibble dequantization and full pointer QMV adaptation | Preserved privately; no material measured end-to-end improvement, not promoted |
| Model-owned source RoPE frequency cache | Bitwise controls pass; OLMo screen improves only about 0.36%, not promoted |
| Removing two identity scatter copies from cached Gemma D512 attention | Same-input scores, probabilities and outputs exact; end-to-end gain only 0.63–0.72%, not promoted |
| Eliminating supposedly redundant KV acceptance copies | Hypothesis rejected: the benchmark's existing state acceptance already avoids those copies |
| Private Metal 4 timestamp instrumentation | Diagnostic only, removed; counter observation adds substantial request cost |

Private timestamp controls attribute roughly 2.52 ms per decode forward to the earlier OLMo Q/K normalization and roughly 0.13 ms after parallelization. Every measured normalization span is valid; incomplete entries correspond to empty acceptance spans. Instrumented request times are excluded from the performance comparison. The source-pointer experiment's CPU subnormal disagreement was separately classified against actual stock MLX; that source control agrees with the candidate's flush-to-zero pattern. It was still rejected on end-to-end performance evidence.

### Final aggregate identities

Paths below remain private under ignored `runtime/olmo3-gemma4/`. The public [benchmark aggregate dataset](../../benchmarks/olmo3-gemma4-q4-repaired-2026-10-29.json) publishes the final counts, timings, source/artifact identities and output hashes without runtime payloads.

| Receipt | SHA256 |
| --- | --- |
| `q4-repair/final-normalization-v1/build-receipt.json` | `7d0724731f2aa5509bb1dc0a6d6ddbf130645b58545254d8fd883d4f25b10f52` |
| `q4-repair/final-normalization-v1/qualification-agent-v1/qualification-summary-v2.json` | `b4dcaa80a2d5cfb65a69296a5a4baaf90961a233751f30a481cb0ee17ee8a435` |
| `q4-repair/final-normalization-v1/root-qualification-final-receipt.json` | `6ce31d1067c585577d9c711618a2ef2c00479743cf4cae3d0fd88aaa00508750` |
| `q4-repair/final-normalization-v1/matched-campaigns-v2-receipt.json` | `141807393c4f2cae5bad5d6fa5472b26a48dbef07ab86e42d1e3b1bfb30d7862` |
| Final OLMo campaign aggregate | `99dfb0f37d615e085ae0db2271ad0453120148da9bc8a2937b9e11d72e44bdad` |
| Final Gemma campaign aggregate | `ab1d0ceb5997ab0479da20763d5bd62ac0d184c3625c032d65f30c0afb189919` |
| `q4-repair/stage-timestamps-v5/olmo3-inline-precise-v1.json`, instrumented diagnostic | `daada24c96755f3d797ce651cbb1465261bd2953d08cdf5af968d55128f21d5f` |
| `q4-repair/stage-timestamps-v6/olmo3-wide-precise-v1.json`, instrumented diagnostic | `eb8bbc41d5c792c0009d46181081ea51e82acbc07db4d76c48b53df31f3735e9` |
| `q4-repair/gemma4-small-head-qkv-v1/final-receipt-v1.json`, preliminary screen | `6ac8e7fa896e0b1dd133e6cb3422c4d520b9200d1552048752486b60f9da34a0` |
| `q4-repair/direct-cached-sdpa-v1/rejected-screen-v1.json`, rejected screen | `83b5bc10f94371919ab63b44cf109a79f2114d44ccb997a8db3d00109b234a83` |
| Public benchmark dataset | `082113e5ed0f2608f938eb0999d7ff427ac49c9b8ee172bf4b1436b938d305d1` |
