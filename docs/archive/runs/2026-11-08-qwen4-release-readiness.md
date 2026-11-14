# Qwen 4 release preparation

Date: 2026-11-08. Source baseline: Stark
`e39bf792a9e215d02abab61cdc38c2aec5eb536f`.

Status: preparation started. Qwen 4 execution, quality, memory use and performance
remain unqualified. This change adds admission regression coverage and a rehearsed
intake procedure. It changes no runtime dispatch, catalogue entries or capabilities.

## Release evidence

The [linked tester post](https://x.com/follistatindev/status/2119129611510383014),
published November 7, claims early access from Alibaba, identifies a checkpoint
as Qwen-4-27B and reports GameFeel results. Its text was recovered through
[FxTwitter](https://api.fxtwitter.com/follistatindev/status/2119129611510383014)
because X blocked direct retrieval. The identity and benchmark claims remain
attributed to the tester. The post supplies no public weights, licence,
architecture contract or release date.

First-party preparation evidence:

- [Alibaba's September 22 roadmap](https://www.alibabacloud.com/en/press-room/alibaba-unveils-roadmap-on-full-stack-ai-strategy)
  says Qwen 4 is in training and gives no launch date.
- [Alibaba's August 27 Flash-Next article](https://www.alibabacloud.com/blog/qwen3-8-flash-next-a-new-architecture-towardsultimate-cost-efficiency_603501)
  describes Qwen3.8-Flash-Next as an early preview of the Qwen4 architecture.
- The [official preview config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/config.json)
  declares `qwen4_exp` with nested `qwen4_exp_text`. Its SHA-256 is
  `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b`.
- The [preview licence](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/LICENSE)
  is Qwen Community License 1.0. Capture each released checkpoint's own licence
  and redistribution terms during intake.

The preview motivates reviewing QSA, Gated DeltaNet, gated residual streams and
n-gram embeddings. The released config, implementation and tensor inventory must
establish which operations each Qwen 4 checkpoint actually uses. Model sizes,
public availability, context limits, quantization and tool formats remain open.

## Current Stark starting point

| Area | Existing implementation | Release work |
| --- | --- | --- |
| Family identity | `model_support.py` and `registry.py` explicitly register `qwen4_exp` as the Qwen3.8 preview. | Read the released root/nested model types and config classes. Add an explicit adapter or extend a validated contract only after reviewing the actual graph. |
| Preview conversion | `adapters/qwen38.py` requires the nested text config and an explicit `qwen4_exp-affine-v1` Lily source contract. It preserves the prepared mixed quantization. | Resolve the floating-source preparation gap; check availability of a compatible prepared artifact or implement and validate the required preparation route. |
| Dense Qwen route | `adapters/qwen35.py` and the registered `Jundot/Qwen3.8-27B-oQ4e-mtp` profile provide a separate dense hybrid implementation. | A 27B label supplies no architecture evidence. Compare its graph before selecting this route. |
| QSA, GDN, gated residual and PLE | Specialized Qwen3.8 hybrid execution already exists. | Compare dimensions, layer schedules, normalizations, gates, sparse selection, PLE hash/lookup/convolution semantics and tensor packing. |
| Prefix reuse | The Qwen3.8 Lily Metal backend has a state snapshot and loaded-worker capability contract. The separate scalar `Forward::Qwen` route has no prefix snapshot hooks. | Inventory all release state, prove exact restoration and measure snapshot memory for the actual artifact. Follow the onboarding gate below. |
| Templates and tools | Native template evaluation and reviewed parser registrations exist independently of converter support. | Hash the released tokenizer/template, test history rendering and assess the actual tool wire format and reasoning controls. |
| Images and MTP | The typed preview text converter omits vision/MTP tensors; retained native sources and dense Qwen have separate paths. | Qualify each selected path and option independently. Advertise only capabilities admitted by the loaded worker. |

Code and current contracts are indexed by
[model-family support](../../MODEL_SUPPORT.md),
[native hybrid models](../../NATIVE_HYBRID_MODELS.md),
[tool-calling coverage](../../TOOL_CALLING.md) and
[model onboarding](../../MODEL_ONBOARDING.md).

The prepared 27B profile above is the registered dense converter route. The
official `Qwen/Qwen3.8-27B` catalogue row still has no adapter and marks its
conversion as planned. Check the exact source and format when choosing a route.

## Metadata-only intake rehearsal

Six live preflight reports were collected through the existing `preflight_source`
API, one for each source/mode pair below, using the locked converter environment
and Python 3.13.13. Both sources resolved to the requested immutable revisions.
The reports consumed `config.json` only and recorded `qualification.status` as
`not_assessed`. No conversion, model execution or weight download was performed.
Private report files are under `runtime/qwen4-readiness-2026-11-08/`.

| Source | Immutable revision | Text / Decision | Vision-language |
| --- | --- | --- | --- |
| `Qwen/Qwen3.8-Flash-Next` | `812d96e10a0dd82b69d749e68693698cf6c36daa` | `unsupported`: `unsupported_qwen38_metadata` | `unsupported`: `unsupported_mode` |
| `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` | `0f94842e3682b49acace4daeb4cf2e65a6f94f83` | `conditional`: `checkpoint_validation_required` | `unsupported`: `unsupported_mode` |

The Lily config is 6,568 bytes, SHA-256
`66f265fed0cfbafc41a9f35e06cef80fab47d26fe154c5ea8134818037657dda`.
The official config is 4,745 bytes, with the hash recorded above. The floating
official source lacks the prepared Lily source contract required by the current
preview converter. The prepared source's conditional result leaves tensor,
native, template, tools and model-bound checks pending. These generic converter
results do not assess the separate retained native image path.

### Commands ready for release intake

Obtain the actual repository and full 40-character commit from the official model
publication. Replace both placeholders before running these commands from the
repository root:

```sh
repo='Qwen/REPLACE_WITH_RELEASED_REPOSITORY'
revision='REPLACE_WITH_IMMUTABLE_40_CHARACTER_COMMIT'
mkdir -p runtime/qwen4-intake

uv run --project tools/converter --locked --python 3.13.13 stark-convert "$repo" \
  --revision "$revision" --preflight --support-mode text \
  > runtime/qwen4-intake/text.json

uv run --project tools/converter --locked --python 3.13.13 stark-convert "$repo" \
  --revision "$revision" --preflight --support-mode decision \
  > runtime/qwen4-intake/decision.json

uv run --project tools/converter --locked --python 3.13.13 stark-convert "$repo" \
  --revision "$revision" --preflight --support-mode vision-language \
  > runtime/qwen4-intake/vision-language.json
```

Read each report's `status`, `reasons`, `modes`, `pending_checks` and pinned
`source`. Preflight verdicts use exit status zero even for unsupported or
inconclusive results. The command downloads bounded metadata and executes no
checkpoint code. An unknown architecture requires implementation work before
conversion can proceed.

Once public metadata exists, separately capture its model card/licence, config,
generation config, tokenizer files, template, processor metadata where applicable,
and Safetensors index. Record immutable source identities and hashes. Compare
the declared graph with the existing adapters and reference implementation;
inspect tensor names, shapes and dtypes before requesting full weight payloads.
The current preflight reports do not replace that complete inventory.

## Implementation queue and dependencies

All release-specific tasks below are pending the published checkpoint contract.
Start with an official text artifact whose complete memory estimate fits the
target Mac. Select a 27B artifact if one is published and meets that requirement.

| ID | Work | Dependencies and completion evidence |
| --- | --- | --- |
| Q4-01 | Pin the source, licences, metadata and tensor layout; compare the graph with existing routes. | Public checkpoint. Produce a source/hash manifest and an explicit operation/layout delta. |
| Q4-02 | Implement missing config, preparation, tensor mapping and native admission. | Q4-01. Preserve original packing where implemented; reject unimplemented operations, formats and tensor layouts. Test conversion and corruption rejection, then load the exact artifact. |
| Q4-03 | Assess tokenizer, native template rendering, reasoning and function tools. | Q4-01; can proceed alongside Q4-02. Cover definitions, parallel calls, call/result history, split streaming markers and ordinary text. Record an explicit parser disposition. |
| Q4-04 | Inventory state, checked memory admission and exact prefix restoration. | Q4-01 graph; can proceed alongside Q4-02/03. Account for context, scratch and snapshot peak memory; implement reuse or record a concrete blocker with support disabled. |
| Q4-05 | Qualify the exact artifact and precision path against a pinned reference. | Q4-02 through Q4-04. Compare prefill/readout and deterministic trajectories, tools, error recovery, cancellation and fresh-versus-reused execution. Preserve F32 reference evidence and record new mixed-precision evidence separately. |
| Q4-06 | Measure performance and enable the qualified catalogue/application path. | Q4-05. Record cold load, first token, sustained decode, long prefill, peak memory and consecutive agent-turn reuse with the same source, precision, context and sampling settings. Update current support documentation. |

Vision, Decision and MTP each need their own artifact/option-specific acceptance
evidence before exposure. Host application integrations remain in their owning
repositories.

### Prefix restoration acceptance

The existing Lily
[`PrefixSnapshot`](../../../crates/stark-engine/src/engine/hybrid/lily.rs)
copies GDN state/windows, PLE history/window, position and optional MTP hidden
state. Attention, QSA indexer and draft-head cache rows remain resident below
the retained boundary. The
[`hybrid` dispatch](../../../crates/stark-engine/src/engine/hybrid/mod.rs)
selects these hooks only for implemented backends; the scalar Qwen route has
no snapshot implementation. The
[prior Lily run](2026-11-02-hybrid-prefix-reuse.md) reported about 110 MiB of
additional recurrent snapshot memory for its named artifact and precision.
Recalculate that cost for a release using the config/layout-aware estimate in
[`model.rs`](../../../vendor/lily-core/src/qwen4exp/model.rs).

Use [the state inventory](../../MODEL_ONBOARDING.md#inventory-state-before-implementing-reuse)
for the actual release. For a preview-like graph it must cover attention KV and
positions; QSA index keys, partial micro-blocks and pooled state; GDN recurrence
and convolution windows; PLE token history, hash constants and dilated-convolution
history; and any MTP/draft state. Preserve chunk boundaries, reduction order and
precision when resuming a suffix.

The pinned [Transformers preview documentation](https://github.com/huggingface/transformers/blob/c5bd76ec51a70c303be110cc04be3d78cf755299/docs/source/en/model_doc/qwen4_exp.md)
describes separate PLE/QSA cache state and disallows cache cropping with those
features. Compare the actual Stark snapshot with the release state inventory;
that upstream limitation alone does not establish a Stark blocker. Reuse must
remain scoped to the immediately preceding successful request and be cleared
under the [worker invalidation rules](../../GENERATION.md#prefix-reuse).

Extend the existing private
[`validate_generation_prefix_cache.py`](../../../scripts/validate_generation_prefix_cache.py)
harness for the admitted backend where needed. Include boundary-adjacent prompt
lengths and changed suffixes, fresh-versus-cached equality, scope/context/MTP
changes, reset, cancellation, failure, image requests and reload. Publish only
aggregate outcomes, timings and hashes.

## Validation of this preparation change

Regression tests use synthetic example identities `qwen4`, `qwen4_text` and
`qwen4_moe`; these names make no prediction about the release. They verify
registry rejection, unsupported preflight and source rejection after config
retrieval, before requesting tensor metadata or weights. Existing `qwen4_exp`
identity remains explicit.

- Registry and model-support tests: 126 passed.
- Combined registry, model-support, Qwen3.8 conversion, hybrid preparation,
  native catalogue and tool-family protocol suites: 186 passed, with
  `HF_HUB_OFFLINE=1` and `TOKENIZERS_PARALLELISM=false`.
- The expanded suite's first run had 17 failures because the new worktree had
  no native template helper. The passing rerun used this checkout's helper,
  built with Rust 1.98.1 and locked dependencies in the isolated
  `runtime/qwen4-readiness-build` target. Its executable SHA-256 is
  `26f8cf72b41cfc7e92b9010315fadffbd29a5337a979311f4bb92558fe797e0e`.
- Ruff lint and format checks passed for both changed test files.
- Repository policy and whitespace checks passed.

No Qwen 4 model was loaded or benchmarked. This preparation does not establish
day-one availability or compatibility.
