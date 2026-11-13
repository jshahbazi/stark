# MLX affine Qwen2.5 and Mistral numerical qualification

Date is 2026-11-12 (runs between 23:12 and 23:34 UTC). Attempts
`20261112-DENSE-MLX-02` (Qwen2.5 0.5B Instruct 4-bit) and
`20261112-DENSE-MLX-03` (Mistral 7B Instruct v0.3 4-bit) in the
[experiment ledger](../../model-experiments/2026-11-12-dense-mlx-qwen2-mistral.md).
The structural record for `20261112-DENSE-MLX-01` is unchanged in
[its run record](2026-11-12-dense-mlx-qwen2-mistral.md).

Neither artifact qualifies. Qwen2.5 0.5B fails the preregistered logit and
greedy gates against stock MLX-LM. The published Mistral export does not
convert, because of its chat-template metadata. A Mistral copy with a
substituted template (weights unchanged) fails one of five logit cases and
passes the worker checks. In both families, stock MLX disagrees with itself by
the same amount when its F16 parameters are rounded to BF16. Stark's BF16
execution stays inside that envelope. No Stark arithmetic defect was found,
and no tolerance was changed.

## Build and environment

| Item | Identity |
| --- | --- |
| Stark source | Branch `claude/dense-mlx-qwen2-mistral` merge commit `7b395605a745bf1ce2a408c92117329e8bab3586` (`55d42081` merged with `origin/main` `0366d566`); no Rust differs from `0366d566` |
| Validation tooling | `scripts/dense_native_validation.py` SHA-256 `172e177b6476ab50064155617fc7cc09ec523ac1163ee458e747531be6d41ef6` (adds the full-attention long case and the `--reference-dtype` diagnostic, see below); `scripts/validate_generation_prefix_cache.py` `58ad8a0435183fc12d481afda3cdb4142cca529efd8a17cd54914ef166e7a82b` (unchanged) |
| Native builds | Rust 1.98.1 release, locked, built separately. `stark-generate` `6eea83b4135c5219d05cb96445d892b1806b4c176538a55d66ead30f49017c64`; `stark-score --features optimized` `467a33c040a0c3777755fa5573a21b65d39e439c19c75410f54584b96cb53282`; `stark-template` `886769cea0c05abf9dac35cd21272353f5dd0c0ee0faa5f97c4c43088b1182ac`; `dense_parity` example (`--features metal`) `7fcdcb2a955e978af11d4d0c24852d20fd1bfec24d8fce9862189dd5e58f39bd`. `Cargo.lock` `f836a99ba1c6bd017fb829cd45f108fd97f9f1fb40c1497564898b3eac625f5f`, `rust-toolchain.toml` `72da9f9e503464ba7393560f6f51a69fb06be72633d52ae7725fa0ba59dfda22` |
| Converter | `tools/converter/uv.lock` `df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5`; converter sources `0e246b4976ba16a4ae7b65b3180e7170ebdbdce4f8fc5789a3aca2f4e7ae78f4`; `--metadata-profile hf-generic --dtype bfloat16` from local pinned snapshots |
| Reference | Stock MLX-LM `1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7` (git), MLX and `mlx-metal` 0.32.3, Transformers 5.7.0, Python 3.13.13. This is a recreated environment: its lock SHA-256 is `a98ad41bad7da9741362671feea8e587c87910244728e456f751446bd3ccb4a7`, not the historical frozen lock `caa51ccf...` used by the OLMo/Gemma records. Same pinned versions; transitive dependencies may differ |
| Host | Apple M5 Max, 128 GiB, macOS 26.7.2 (25G332). Another agent's inference workload may have shared the GPU; no timing is reported |

## Preregistration

Written and hashed before any numerical run (private file SHA-256
`6a443af776efd7df97a378d8f6f99726162d2fc7b254d3dd7d452445d9ad2ca4`, 23:22:23
UTC). The reference is stock MLX-LM executing the exact downloaded MLX
checkpoint, which has the same quantized weights as the candidate. The forward
profile is `prefix-cache-1024-then-last-token-v2`. Gates are unchanged from
earlier dense records:

1. `dense_parity` `bf16-centered-logits-v3` on five full-vocabulary cases:
   centered max abs error at most 0.75, absolute mean error at most 0.75, KL at
   most 0.01, RMSE at most 0.1, and centered RMSE at most 1/128 + 0.02 times
   the reference centered RMS.
2. `dense_native_validation.py workers`:
   - native tokenization equals the reference;
   - both 32-token greedy generations exactly equal the reference;
   - replay, invalid-request recovery and restart replay are exact;
   - scorer absolute error is at most 0.75, with consecutive-cache controls.
3. `validate_generation_prefix_cache.py --reuse rewind --mode exactness`:
   - outputs are identical cached and uncached;
   - reuse counts match the rule;
   - every invalidation reuses nothing.

Greedy agreement inside `dense_parity` is diagnostic only.

The cases are 1 token, chat prompt A, chat prompt B, 129 repeated tokens, and
1,029 repeated tokens.

## Validation tooling change

`dense_native_validation.py reference` previously required an active sliding
window of at most 4,096 and refused both sources. Qwen2.5 declares
`sliding_window` 32,768 with `use_sliding_window: false`; Mistral v0.3
declares `null`. The new `long_case_tokens` helper keeps `window + 5` for an
active window. For full attention it uses 1,029 tokens, which crosses the
1,024-token prefill chunk. `scripts/test_dense_native_validation.py` covers
both branches and the refusals (3 tests).

The added `--reference-dtype bfloat16` option is a diagnostic profile. It
casts every floating MLX parameter, including affine scales and biases, to
BF16. The default `source` profile is unchanged. Rerunning the Qwen source
profile with the final script reproduced all five reference logit hashes and
both reference generations.

## Live header re-check (metadata only)

The re-check ran the real `preflight_model`, `checkpoint_headers` and
`plan_from_headers` at the recorded revision and the current head of all nine
repositories. Requests were unauthenticated and spaced 2.5 seconds apart.

- **Nine recorded repositories:** every recorded revision is still the head.
  All 9/9 are `conditional` and plan, with zero full-precision matrix specs.
  Every config and header SHA-256 equals the
  [structural record](2026-11-12-dense-mlx-qwen2-mistral.md).
- **Qwen2.5 0.5B candidate:** `mlx-community/Qwen2.5-0.5B-Instruct-4bit` at
  `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` also plans: 24 layers, 628
  tensors, 97 MLX specs, 24 Q/K/V bias fusions, tied head. Config
  `b045e57ea90b8f1b35f89f954b176a5c1faa02bd0af2c89bcec191239d66cef4`, header
  `25c85cd25d23159fc575b1d929bf6bdb2b77ff365748a88ec80633eb7538a25e`.
- **Private receipts:** harness `81e9ef46e6e5aed9b952b449772b6aa98f7bd8086ec55cae0a54412b7fdbac78`,
  output `0f75eaa9d5d43f3bd752dd5d7e9bd532d9591b1cb0077a8b7971b938eada4596`.

## Sources and artifacts

Downloads were unauthenticated, pinned and public. They are stored under the
ignored `runtime/`, about 4.36 GB in total. No BF16 parent was downloaded: the
reference executes the same quantized weights, so the parents are not needed.

- **Shared source layout:** every floating tensor in both sources is F16:
  affine scales, affine biases, norms and the Qwen2 Q/K/V bias vectors.
  Packed weights are U32.
- **Shared conversion result:** the default BF16 conversion rounds every F16
  value to BF16 and runs BF16 activations and KV cache.

| | Qwen2.5 0.5B Instruct 4-bit | Mistral 7B Instruct v0.3 4-bit |
| --- | --- | --- |
| Repository / revision | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` @ `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` @ `a4b8f870474b0eb527f466a03fbc187830d271f5` |
| Source weights SHA-256 | `ddffab9cbc7bf6dde941c6724841eeca8981fcfa81ca20ff8efff1396326d153` | `d7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d` |
| Source `tokenizer_config.json` | `f7c61e32b7a17d19bf8e7037dcb74079a833e53ea9801f24008cac68458f03b7` | `e8fbcfef34d855dba2c7f1d91a2c5470cf493d00ecfd06488ce21e45325bc8ae`; `chat_template` is a named list (`default`, `tool_use`) |
| Group / bits / head | 64 / 4 / tied | 64 / 4 / untied |
| Conversion | `ok`, 364 tensors, output `bd822bbbe7fdaf7cb8e4045f334096a41b63fbed47a4cac6343ae77596b89d81`, config `bb3b67849496c3e0ce0581afb40ec8812fa0439d0ca0b18f557a733d5678b076`, tokenizer `9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa`, manifest `0e23197c5452b8493d230726317aa473e8726d4e5819f8f2efb6d53e7212bb09` | Failed. CLI category `conversion_failed`; in-process cause `UnsupportedMetadata: Unsupported chat-template representation.` (`metadata._chat_template`). No artifact |
| Native readiness | Ready; vocabulary 151,936; context 32,768; MTP unsupported; `fresh-per-request` without `--prefix-cache` | Not applicable to the published route |

**Diagnostic Mistral copy (not a qualification artifact).** All files are APFS
clones of the download, so the weights are byte-identical. The only change is
`tokenizer_config.json`, whose `chat_template` holds the source's own `default`
entry as a string.

- Config `20099d110e3b5318671fa6aed54b1db8fc2d52be420a517e28dc16bac7af8229`;
  template `a6fc0e6bc936bd3aecf4b1e4c1208778b2f916f7e55e01bccd3b232f976b9216`.
- Converted: 455 tensors, output
  `4283ec256c031f767e16139bb50e231292dc7f7fc34c3e451cbbe89ce286f4d4`, config
  `a8ced577ef9bb30f3f0c8d753d241b658faf7c3ee786135ebbb7eb7d40d833c9`.
- Tokenizer `3e01771050ce4fba855c59ddb474c72ca1a3f2113a04727990155e3e47390764`,
  equal to the tokenizer of the qualified BF16 `mistralai/Mistral-7B-Instruct-v0.3`
  native record.

A second copy uses the `tool_use` entry (template
`4c8a246880581b094e13db2699544ded1332ffc0e6ff006b581b89c82e2f1851`, same
weights output hash) for the prefix-cache harness only.

## 20261112-DENSE-MLX-02: Qwen2.5 0.5B Instruct 4-bit

Logit gate against the stock source profile (MLX F16 parameters and
activations) versus Stark BF16:

| Input tokens | RMSE | Mean error | Centered max | Centered rel. L2 | KL | Greedy | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 0.1637 | -0.0868 | 0.7626 | 4.84% | 0.00763 | agree | fail |
| 39 | 0.0965 | -0.0252 | 0.5217 | 3.54% | 0.00284 | agree | fail |
| 43 | 0.0931 | 0.0091 | 0.4714 | 3.32% | 0.00049 | agree | fail |
| 129 | 0.0942 | -0.0173 | 0.4358 | 3.92% | 0.00013 | agree | fail |
| 1,029 | 0.0813 | 0.0177 | 0.3739 | 2.86% | 0.0000005 | agree | fail |

Every case misses the relative centered-RMSE bound. The 1-token case also
misses the RMSE and centered-max limits.

Worker checks:

- Tokenization matches.
- Prompt A generates 13 tokens to EOS, greedy-exact.
- Prompt B diverges at token 20 of 32 (`finish_reason` `length`).
- Replay, invalid-request recovery and restart replay are exact.
- Scorer max error is 0.547 (39 tokens) and 0.172 (1,029 tokens), with
  consecutive-cache, intervening-request and failure-clearing controls passing.

Verdict on gates 1 and 2: **failed**.

Prefix reuse (gate 3), on the BF16 artifact with `--context 8192
--first-tokens 3000 --tools 1 --instruction-paragraphs 2
--follow-up-sentences 8 8 40 --max-new 64`: **passed**.

- 32/32 requests identical to the worker without `--prefix-cache` and to a
  fresh run, and every reuse count matches the rule.
- Follow-ups reuse 2,048 tokens and the maximum is 3,072.
- All six invalidations behave as documented (error reply, unscoped request,
  reset, image operation, cancellation, tokenize/decode), as does restart.
- Baseline and candidate were the same sealed worker: no Rust differs from
  `origin/main`, and the flag is the only difference.

Diagnostics (these do not change the verdict):

| Comparison | Cases passing the same gate | Centered rel. L2 range | Max KL | Greedy, 32-token prompt B |
| --- | ---: | --- | ---: | --- |
| Stark BF16 versus MLX F16 (gate run) | 0/5 | 2.86-4.84% | 0.0076 | diverges at token 20 |
| Stark BF16 versus MLX with BF16-cast parameters | 0/5 | 3.00-7.15% | 0.0101 | diverges at token 19 |
| MLX BF16-cast versus MLX F16 (same weights) | 0/5 | 3.42-6.41% | 0.0129 | diverges at token 19 |

What the diagnostics show:

- Stock MLX rounding its own F16 parameters to BF16 fails the gate by more
  than Stark does, on every case.
- All three executions agree on prompt A and on next-token argmax in every
  case.

An F16 Stark conversion of the same source would match the source profile
directly. It converts (output
`e227c07354092ee96211a0068e279f81ee26cc553c6f73265247cc904967be04`), but the
native worker refuses it with `model_load_failed`. F16 execution is
runtime-blocked, so a precision-matched candidate needs engine work.

## 20261112-DENSE-MLX-03: Mistral 7B Instruct v0.3 4-bit

The published artifact at this revision **fails conversion** with
`UnsupportedMetadata` because of its named chat-template list. It has no
qualification.

The following results come from the diagnostic copy (identical weights,
`default` template). Logit gate against the stock source profile:

| Input tokens | RMSE | Mean error | Centered max | Centered rel. L2 | KL | Greedy | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 0.0217 | 0.0158 | 0.0861 | 0.60% | 0.00032 | agree | pass |
| 16 | 0.0340 | -0.0038 | 0.2577 | 1.53% | 0.00004 | agree | pass |
| 20 | 0.0361 | -0.0089 | 0.1885 | 2.11% | 0.00003 | agree | pass |
| 129 | 0.0335 | 0.0192 | 0.3942 | 0.87% | 0.00209 | agree | pass |
| 1,029 | 0.1072 | 0.0424 | 0.4691 | 4.02% | 0.00158 | agree | fail |

Worker checks: **passed**.

- Tokenization matches.
- Both generations are greedy-exact: 14 tokens to EOS, and 32 of 32.
- Replay, recovery and restart replay are exact.
- Scorer max error is 0.250 and 0.398.

Prefix reuse:

- **`default` template:** the harness could not render its synthetic
  tool-call history (`ChatError`), so the run did not complete.
- **`tool_use` copy, same flags:** 32/32 requests identical and rule-matching,
  and all six invalidations and restart as documented. The aggregate reports
  `passed: false` only because `follow_up_3` reused 0 tokens, which the rule
  allows. This template renders its tool block before the latest user
  message, so a growing session changes early tokens. That is an
  interpretation of the template, not a reproduced engine fault.

Diagnostics:

| Comparison | Cases passing the same gate | 1,029-token case |
| --- | ---: | --- |
| Stark BF16 versus MLX F16 (gate run) | 4/5 | RMSE 0.107, rel. L2 4.02%, fail |
| Stark BF16 versus MLX with BF16-cast parameters | 4/5 (20-token case fails at 2.80%) | RMSE 0.028, rel. L2 1.11%, pass |
| MLX BF16-cast versus MLX F16 | 3/5 (20- and 1,029-token cases fail) | RMSE 0.108, rel. L2 4.18%, fail |

Worker checks against the BF16-cast reference also pass: both greedy-exact,
scorer 0.188 and 0.125.

## Private receipts

All files are under ignored `runtime/dense-mlx-qwen2-mistral/receipts/`.

| Receipt | SHA-256 |
| --- | --- |
| Qwen reference (source profile) | `233e5f09ea3eeabc4840c84dc1acacaceeffb0f0f0f51f477c6b666f701a6849`; cases `1f17530d5ab5e8e2fc4ee56fd40085b8acbf93a1ac4bb3a82357678df464b689` |
| Qwen parity / workers / prefix | `1cbfd6b3c9f6e72a2040ccefaed54cecf78351f6453b5b7ff00c9aa0e43e095b` / `4cacc95fc4c405f4a8f6bb46d9bde7a1bdeffac26bf2fdb2bb01d78fcf241af7` / `ad7e0daba1e292c7bdd668127432e931996b4397fb689360231b3474b78c9884` |
| Qwen BF16-cast reference / parity / workers | `980b9f54cf7806a2774b9e1c0d94b1f76d7573e9dd42b20f72d0049b351d447b` / `b8ba55a53468a3144f545b78d72dbeef40548a6873881b3dd3c94a0ab88bc173` / `7df2401d051857a345e0e3c999e68e03b765961e53ca5813749d08269e67a0a3` |
| Mistral diagnostic reference (source profile) | `3d6aa352e808e2cf238fedea244daff7f5980f25592c0ee69ad6b140991c05d9`; cases `d90d661d3a8ccc65cdd4a0ce8ecd6f1959b990732c78f1280573f811838b16ae` |
| Mistral parity / workers / prefix (`default`, failed to render) | `fbc275ea87473b0f3a90e7ca108dbf45fa80aee4b53f522519d2c6f58d3a4762` / `e19e6106242e7b601becd074829f5986ad5d63322e07d21b5a658c5625e822ab` / `245eed6f2545d9d0795bcaa3f510d56f4873d08fa045bdabdf403e32ffee81fe` |
| Mistral prefix (`tool_use` copy) | `aa1e4e0ab20bb41cf97fd41ebd126ca380b1d001e1488f2c01776b50ab049f9e` |
| Mistral BF16-cast reference / parity / workers | `9e4525a5a528b27b717d0b6eaa6cf7f56235baf03db46b8058f90624b3690e77` / `ab60e509ff6b155f6181dab50b1d7069b6b8259631badf207ae6557abcb6dc28` / `d81a5646d135794a35de8f1b71abc5e5eb195787c6f8a2e0ab6ee68fd50446dd` |
| Harness scripts: per-artifact runner / BF16 diagnostic / reference comparison | `057191a1c24a27822c968520b0605fa5eecb9646a746e64d405a407f6768ed1c` / `f3a2842cbbfb227850a997cb9b848f89b7f6f820f6a1ed99ea7b21facde7cdc5` / `c85e0dc3dcb7e82264c425eb749c7281501e7652c02df8f039a152f0ef9aa98a` |

## Findings

1. **No Stark arithmetic defect was found.** For both families, stock MLX's own
   BF16 rounding of these all-F16 sources moves logits as much as or more than
   Stark does. Stark against MLX F16 sits inside that envelope. The
   `bf16-centered-logits-v3` budget was calibrated on BF16-parameter sources;
   it cannot be met by any BF16 execution of the 0.5B Qwen source. This is
   evidence about precision policy, not a pass.
2. **Precision-matched qualification of F16-parameter MLX sources needs F16
   (or F32) execution in the native runtime.** The F16 Qwen artifact converts
   but fails native load. This is engine work and was not attempted.
3. **The pinned Mistral MLX export cannot be imported.** Its named
   chat-template list is refused by `metadata._chat_template`. By code
   reading, the catalogue still offers such a row for download: `downloadable`
   depends only on `unavailable_reason`, and the named template only marks the
   separate `planning` block unavailable. So conversion would fail after the
   4 GB download. This was not exercised through the catalogue. Fixing it
   needs a decision: either fail closed in the catalogue, or support named
   templates through the tool-template review in
   [model onboarding](../../MODEL_ONBOARDING.md).
4. Prefix reuse on the dense MLX path is exact for every request on both
   weight sets. Qwen passes the full harness. The Mistral aggregate fails only
   on the template-driven follow-up rule.

## Not measured or not verified

- **Performance:** speed and memory not measured, because another workload may
  have been using the GPU. Worker rate fields in the private receipts are not a
  timing result.
- **Other inputs:** 8-bit exports, the other seven planned repositories, BF16
  parents, active sliding windows.
- **Tool-calling:** function-tool behaviour (`validate_tool_calling.py` not run).
- **App route:** the catalogue route (`--repo` catalogue identity) and the
  app's install path.
- **Image parity and MTP:** N/A. Both sources are text-only
  `Qwen2ForCausalLM` / `MistralForCausalLM` checkpoints with no
  `vision_config` and no draft-head tensors (complete inventories planned
  without extra tensors).
