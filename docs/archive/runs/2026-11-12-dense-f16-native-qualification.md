# Native F16 dense execution and MLX Qwen2.5 and Mistral requalification

Date is 2026-11-12 US local time (runs between 00:17 and 00:45 UTC on
2026-11-13). Attempts `20261112-DENSE-MLX-04` (Qwen2.5 0.5B Instruct 4-bit,
F16), `-05` (Mistral 7B Instruct v0.3 4-bit diagnostic copy, F16) and `-06`
(BF16 regression control) in the
[experiment ledger](../../model-experiments/2026-11-12-dense-mlx-qwen2-mistral.md).
They follow the BF16 attempts `-02` and `-03` in the
[earlier qualification record](2026-11-12-dense-mlx-qwen2-mistral-qualification.md),
which are unchanged.

Summary:

- The native worker refused F16 artifacts because the engine fixed the dense
  activation dtype to BF16, so every F16 tensor failed load validation.
- The engine now runs an F16 profile for the plain full-attention dense graph.
  Every BF16 artifact keeps its existing policy.
- On the same quantized weights and the same stock MLX-LM reference, the F16
  Qwen2.5 artifact passes all three preregistered gates.
- The F16 Mistral diagnostic copy passes the logit, greedy and scorer gates.
  Its prefix-reuse rows are all exact, but the aggregate fails on the same
  template-driven follow-up rule as `-03`.
- BF16 logits and generations are bit-identical between the baseline and
  candidate builds.

The published Mistral revision still fails conversion because of its named
chat-template list. Neither MLX export becomes a qualified catalogue artifact
through this record: the Qwen2.5 result covers the exact F16 artifact and build
below, and the Mistral result covers a diagnostic copy only.

## Why F16 was refused

`Engine::language_model_from_config` set `let data_type = DataType::BF16`, and
`ParameterLeaf::validate` requires each tensor's stored dtype to equal it. The
first tensor validated for the earlier agent's F16 Qwen2.5 artifact (output
`e227c073...`) was the tied embedding's affine scale plane:
`Invalid tensor: got [151936, 14] @ F16, expected [151936, 14] @ BF16`. The
workers map every load error to `model_load_failed`.

Downstream of that check, the Metal kernels and Rust routes on the dense path
were also not instantiated or admitted for F16:

- the GEMV/GEMM templates (`AT`, `BT`, `DT`);
- input embedding lookup;
- RoPE/QKV preparation (`AttentionPrepare`);
- attention single-pass, two-pass and tiled kernels;
- KV-cache update;
- the gated SiLU product;
- the activation transform used by the matmul epilogue;
- the sampler;
- the scoring readback, which read logits as BF16.

`vendor/lily-core` is the Qwen3.8 Flash Next core and is not on this path.

## Precision policy (F16 dense profile)

| Element | Policy |
| --- | --- |
| Selection | F16 only when the decoder subtree stores at least one F16 tensor and no BF16 tensor (a `--dtype float16` conversion); otherwise BF16, as before. F32 tensors and tensors outside `decoder.` never select F16 |
| Admission | Plain dense graph only: attention plus dense SiLU MLP layers with pre-norms; Unscaled or Llama 3 RoPE; no query/key norms, sliding window, soft caps, sinks, gates, KV sharing, PLE, image embeddings, MTP head or speculator, and no BF16-reference flags. Anything else returns `UnsupportedPrecision(F16)` before decoder weights load |
| Weight storage | Packed MLX affine codes unchanged; scales, affine biases, embeddings, norms and Q/K/V biases stay F16 exactly as stored in the source |
| Dequantization | `scale * code + bias` in F32 inside GEMV, GEMM and the embedding lookup, then rounded once to F16 at the output boundary (the BF16 path's arithmetic with an F16 boundary) |
| Activations | F16 between operators |
| Accumulation | F32 in matrix products, RMS normalization, RoPE (F32 tables) and attention scores and softmax |
| KV cache | F16 |
| Readout | F16 logits, widened to F32 for scoring and in the sampler |
| Not used for F16 | BF16-only fast and source paths: INT8 activation quantization, MXU attention prefill, materialized affine input sums, packed 5/6-bit matrices and source BF16 attention |

## Build and environment

| Item | Identity |
| --- | --- |
| Baseline | `e1e90517`; `stark-generate` `ec741a1ede1a4a1907d70e23e5d39b5272ea4d59b175e701571631941543e5aa`, `stark-score --features optimized` `e83e3c84fed45958351c2d3166b638fa2b038c8724b1738276ec5ea2624fc542`, `dense_parity` `aed90b00077bbc8ce7355f26b353e696f8986387f57379abda7023fbcd6497fa`, load probe `076e045d27fabfe33ef91779b51447606d239ad57bbe83dc80d86dcde89b71f5` |
| Candidate (gate runs) | Engine change on `e1e90517`, built before commit: `stark-generate` `2fa9f4df4a760906ead453525e4e701e210c4d105c0bf9c0f528747257e5ac29`, `stark-score --features optimized` `c6205b4058d2b0530f42a5742edee77b5c9bf51dac4a325b39cf186e5567303b`, `stark-template` `886769cea0c05abf9dac35cd21272353f5dd0c0ee0faa5f97c4c43088b1182ac`, `dense_parity` `26f85d0e409704e2dfe720f240c2087b8e05b4050310056b2e1d777637e87407`, load probe `2ee9dd5745f446083eb332ef85eac0eaf8a0fc6ee4c700e273a124929023f2c1` |
| Candidate (commit recheck) | Engine commit `f203f004` on branch `claude/engine-f16-weights`. Its source differs from the gate-run build only by rustfmt line moves, tests and the CPU reference kernel, which the Metal-only workers do not compile. Rebuilt `stark-generate` `2adf7b0d1e6c3b1504233f0fe906b548d21630db60bc04f009d1daad950771dc`, `stark-score --features optimized` `b73e1fea98d9811d4ceb5204805e67734e1553a414b29477870e5180f35ab99f`, `dense_parity` `224d9571e70545f43e993aa7334b1bd6c342adc86418f90bcf51db9ac082b7b4`. The executables differ byte-wise, but they reproduce the gate runs exactly: all 20 native logit hashes (F16 and BF16, both families), the F16 worker generation hashes and scorer results, and every F16 prefix-harness row (outputs and reuse counts) |
| Toolchains and locks | Rust 1.98.1 release, locked, workers built separately; `Cargo.lock` `f836a99b...`, `rust-toolchain.toml` `72da9f9e...`, `tools/converter/uv.lock` `df771172...` |
| Validation tooling | `scripts/dense_native_validation.py` `172e177b...` and `scripts/validate_generation_prefix_cache.py` `58ad8a04...` (both unchanged from `-02`/`-03`); private runner `184083a2...`, generation-hash driver `6c347b5f...`, load/logit-hash probe source `13812520...` (kept outside the repository) |
| Reference | Unchanged stock MLX-LM `1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7`, MLX 0.32.3, Transformers 5.7.0, recreated lock `a98ad41b...`, run from the earlier environment's interpreter without re-syncing it |
| Host | Apple M5 Max, 128 GiB, macOS 26.7.2 (25G332), Metal 32023.883. Another agent's Phi workload may have shared the GPU; no timing or memory is reported |

## Preregistration and sources

The private preregistration (SHA-256
`2a947504b84f5eb806093f11c671369ade0de19ce1cbca8260d7bd7b32262529`, 00:17:54
UTC) fixed the gates before any numerical comparison: the same three gates as
`6a443af7...`, with no tolerance change. Only a functional smoke preceded it:
the candidate loaded the F16 Qwen2.5 artifact and returned finite logits for
synthetic token IDs. That smoke retained no logits and made no reference
comparison.

No file was downloaded. The pinned sources, fixtures and reference environment
of `-02`/`-03` were read in place and not modified.

| | Qwen2.5 0.5B Instruct 4-bit | Mistral 7B Instruct v0.3 4-bit (diagnostic copy) |
| --- | --- | --- |
| Source | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` @ `a5339a41...` (weights `ddffab9c...`) | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` @ `a4b8f870...` (weights `d7182d88...`) with its `default` template substituted, as in `-03` |
| Conversion | `--architecture qwen2 --metadata-profile hf-generic --dtype float16 --offline`; 364 tensors; output `e227c07354092ee96211a0068e279f81ee26cc553c6f73265247cc904967be04` (identical to the earlier F16 conversion), config `bb3b6784...`, tokenizer `9c5ae00e...` | `--architecture mistral`, same flags; 455 tensors; output `968cf950b4d7cba8ab2f0d1c97a708f0a1bfe2dc9f8f209aa6d8492e671d1a5a`, config `a8ced577...`, tokenizer `3e017710...`. The `tool_use` copy for the prefix harness has the same output hash and config `ca215e8a...` |
| Reference rerun | All five reference logit files byte-identical to the fixture (cases `1f17530d...`); generation tokens identical (only timing fields differ) | All five byte-identical (cases `d90d661d...`); generation tokens identical |

The F16 conversions were made with the converter before its capability
annotation change, so their receipts record `native_loading:
"runtime_blocked"`. The change affects only `conversion.json`; weights, config
and tokenizer are unaffected.

## 20261112-DENSE-MLX-04: Qwen2.5 0.5B Instruct 4-bit, F16

Logit gate (`dense_parity`, budget unchanged) against the stock source profile:

| Input tokens | RMSE | Mean error | Centered max | Centered rel. L2 | KL | Greedy | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 0.0258 | 0.0190 | 0.0825 | 0.61% | 0.00014 | agree | pass |
| 39 | 0.0153 | -0.0087 | 0.0656 | 0.48% | 0.00001 | agree | pass |
| 43 | 0.0140 | 0.0045 | 0.0756 | 0.47% | 0.00003 | agree | pass |
| 129 | 0.0142 | -0.0021 | 0.0682 | 0.60% | 0.0000005 | agree | pass |
| 1,029 | 0.0105 | -0.0018 | 0.0539 | 0.37% | 0.00000002 | agree | pass |

For comparison, the BF16 artifact in `-02` measured 2.86-4.84% on the same cases.

Worker checks: **passed**.

- Native tokenization equals the reference.
- Prompt A generates 13 tokens to EOS, greedy-exact.
- Prompt B generates 32 of 32 tokens, greedy-exact (the BF16 artifact diverged
  at token 20).
- Replay, invalid-request recovery and restart replay are exact.
- Scorer max error is 0.031 on both cases (39 and 1,029 tokens), and the
  consecutive-cache, intervening-request and failure-clearing controls pass.

Prefix reuse (same flags as `-02`): **passed**.

- 32/32 requests are identical to the worker without `--prefix-cache` and to a
  fresh run, and every reuse count matches the rule.
- Follow-ups reuse 2,048 tokens, and the maximum reuse is 3,072.
- The error reply, unscoped request, reset, image operation and cancellation
  each reuse 0 afterwards. Tokenize/decode keeps reuse, as documented.

Verdict: **passed** all three preregistered gates for this exact F16 artifact
and candidate build.

## 20261112-DENSE-MLX-05: Mistral 7B Instruct v0.3 4-bit diagnostic copy, F16

| Input tokens | RMSE | Mean error | Centered max | Centered rel. L2 | KL | Greedy | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 0.0020 | 0.0007 | 0.0071 | 0.08% | 0.0000002 | agree | pass |
| 16 | 0.0056 | 0.0028 | 0.0340 | 0.22% | 0.0000002 | agree | pass |
| 20 | 0.0061 | -0.0023 | 0.0336 | 0.34% | 0.00001 | agree | pass |
| 129 | 0.0030 | 0.0012 | 0.0188 | 0.09% | 0.000002 | agree | pass |
| 1,029 | 0.0051 | -0.0019 | 0.0331 | 0.19% | 0.000002 | agree | pass |

In `-03`, the BF16 copy's 1,029-token case failed at 4.02%.

Worker checks: **passed**.

- Tokenization matches.
- Both generations are greedy-exact: 14 tokens to EOS, and 32 of 32.
- Replay, recovery and restart replay are exact.
- Scorer max error is 0.023 and 0.031.

Prefix reuse (`tool_use` F16 copy, same flags):

- All 32/32 rows are identical and match the rule.
- Follow-ups reuse 0, 2,048, 2,048 and 0 tokens. The aggregate therefore
  reports `passed: false` (`follow_ups_reuse: false`), exactly as the BF16
  `tool_use` copy did in `-03`.
- The five invalidations reuse 0, and tokenize/decode keeps reuse.

Verdict on the diagnostic copy: logit, worker and prefix-row exactness
**passed**; the prefix aggregate **failed** on the template-driven follow-up
rule. The published revision still fails conversion (`UnsupportedMetadata`,
named chat-template list) and is **not qualified**.

## 20261112-DENSE-MLX-06: BF16 regression control

Each comparison below gave identical results on both builds.

| Artifact | Comparison | Result |
| --- | --- | --- |
| BF16 Qwen2.5 MLX artifact `bd822bbb...` (from `-02`) | `dense_parity` baseline and candidate, source fixture | All 5 native logit hashes identical to each other and to the `-02` receipt (built separately as `7fcdcb2a...`) |
| Same | Worker checks, candidate binaries; baseline binaries | Generation hashes, token counts, finish reasons, replay/recovery flags and scorer errors (0.547, 0.172) identical to the `-02` receipt and between builds. The gate still fails exactly as recorded |
| BF16 Mistral diagnostic copy `4283ec25...` (from `-03`) | `dense_parity` baseline and candidate; worker checks, candidate | All 5 logit hashes identical to each other and to the `-03` receipt; generation hashes and scorer errors (0.250, 0.398) identical to `-03` |
| `Qwen/Qwen3-0.6B` @ `c1899de2...` BF16, output `87c3344a...` (the project's reference dense BF16 artifact) | Load probe: next-token logits for three synthetic inputs (1, 40 and 1,029 tokens) | All three logit hashes identical across builds |
| Same | `stark-generate` readiness reply and two 32-token requests with the artifact's default options | All three reply hashes identical across builds (timing fields removed) |

Negative control: an F16 conversion of the same Qwen3-0.6B source (output
`92fa34d8...`; query/key norms) is refused by the candidate with
`Unsupported activation precision F16 for this model graph`. The baseline
refused it with the dtype-mismatch error.

## Tests

- Engine unit tests (pinned nightly, `cpu,metal`, Apple M5 Max): 21 new tests
  pass.
  - F16 KV-cache update: 4.
  - Full-precision F16 GEMV/GEMM against the CPU reference: 7.
  - F16 MLX affine GEMV: 4.
  - F16 MLX affine GEMM, Simdgroup and automatic routing: 3.
  - F16 admission: 2.
  - Loader dtype selection: 1.
- The full native Metal suite: 1,051 passed, 0 failed, 152 ignored (lib), plus
  12 in the other targets.
- Converter: 17 new preflight and capability tests for F16 output. The full
  suite has 7,620 passed and 94 skipped; the same base recorded 7,603 passed
  and 94 skipped.

## Private receipts

All files are under ignored `runtime/engine-f16/`.

| Receipt | SHA-256 |
| --- | --- |
| Qwen F16 parity / workers / prefix / reference rerun | `ac8d718a36414b29d0db7bf212ca34394af4c5b3a68924b0eda120d36c96dd33` / `d158307918c136d4cdf2f408445238a0790a20946b9b765bd8332bf66374c4eb` / `fb92d979a9961a27599e5f5d8e18705022897098721e1ab5c8fb8699d9639966` / `743ae66ac93ab62eeed25ab4bd4d07339fd59ddfba9396a34bea509a50b1baee` |
| Mistral F16 parity / workers / prefix / reference rerun | `8e4dee1ef57252c74e13c545299863de9bcdfff4478f2ed9971182bda5318931` / `dd76cc50037db2571548065af884cc1abd79fb43443b9cc4a06219f681930506` / `6e72161dcba52e2d05e93112f18da91e8614ce9aec305ceb6778d2443a29116a` / `aee18ddf6def976bf855d583ba5ac644ffe217fb8b6ce4d312823394736938ad` |
| BF16 Qwen parity baseline / candidate | `1888c8b7010d5f4a51c570b5c6f11c6cc84c957580017177aa011990c5c47e99` / `d67a0c54e3196ca9386907a8130bc34a61c526fd8ddefccf13f8b525bc809674` |
| BF16 Mistral parity baseline / candidate | `47f1ed4db5ebff849ec86a245ecde1365d5eb8ce0e2474fcf7f228ed4485e19c` / `456d56c059ee1d76e02cffa285e50a97833df6de722febc9b7d4910afe656047` |
| BF16 workers: Qwen baseline / Qwen candidate / Mistral candidate | `d519deb458bd3d14f191e6ede03ce3470426bd30d232d9b60834245a670e0d10` / `cccfb94955e9306bf5ef6f25e0ad61215eab03f6c3f0bbdf17bf2a7862a5146a` / `b323ccfacff3c6f9947d1b890d95b489e97dab4292f63e3012ef5c86c1a46555` |
| Qwen3 BF16 logit hashes (baseline = candidate) / generation hashes | `3a5a325c1782518fd369ca276759839ceeece6825abc38cbb58c4ecb036605fd` / `f071a8d2dc9c7019556118d58dc27501351aa1a122069fe50de272fc990ae285` |
| Conversion receipts: Qwen F16 / Mistral F16 / Mistral `tool_use` F16 / Qwen3 F16 negative | `04ebe957...` / `37f31aed...` / `427594ca...` / `0ee9669c...` |
| Engine test logs: F16 subset / full Metal suite | `fdd7583e...` / `05f96d43...` |
| Commit recheck: Qwen F16 parity / workers / prefix | `868d5c87...` / `267c3484...` / `f3cae5e0...` |
| Commit recheck: Mistral F16 parity / workers / prefix (`tool_use`) | `15ba6307...` / `fdcea842...` / `7e149d4d...` |
| Commit recheck: BF16 Qwen / Mistral parity | `604b53e0...` / `92eba40d...` |

## Not measured or not verified

- **Performance and memory:** not measured, because another agent's workload
  may have shared the GPU. Worker rate fields in the receipts are not a timing
  result.
- **Other F16 inputs:** no full-precision (unquantized) F16 dense checkpoint was
  available without a download. That path has kernel-level tests only.
- **Other F16 configurations:** sliding-window F16 artifacts are refused by
  design. Llama 3 RoPE F16 has kernel and admission coverage but no model run.
- **Other exports:** 8-bit exports, the other seven MLX repositories, and the
  published Mistral revision.
- **App and tools:** the catalogue and app install routes, tool calling, and
  the generation worker with sampling (only greedy and the artifact's default
  options were run).
- **Image parity and MTP:** N/A. Both sources are text-only, with no
  `vision_config` and no draft-head tensors.
