# App installs select native F16 for all-F16 MLX Qwen2 and Mistral sources

Date: 2026-11-12. Branch `claude/install-native-f16`, based on `103a8296`
(`claude/engine-f16-weights`). The change described here is the commit that
adds this record. It follows attempts `20261112-DENSE-MLX-04/05/06` in the
[experiment ledger](../../model-experiments/2026-11-12-dense-mlx-qwen2-mistral.md)
and the [F16 qualification record](2026-11-12-dense-f16-native-qualification.md),
which are unchanged.

This record covers an install-route check of an artifact that was already
qualified. It reruns the worker gate but not the logit gate or the
prefix-reuse gate. It is not a new qualification, so it has no ledger attempt.

## Problem

The desktop backend built `stark_convert.cli` install commands without
`--dtype`, so every app install used the CLI default `bfloat16`. The
`mlx-community/Qwen2.5-0.5B-Instruct-4bit` export therefore installed as the
BF16 artifact that failed the logit gate (`-02`), not the F16 artifact that
passed (`-04`).

## Selection rule

`stark-convert --dtype auto` resolves the output dtype after the source is
resolved and planned. It plans BF16 first. It then selects `float16` only
when all of the following hold (`model_support.auto_f16_refusal`,
`cli._select_auto_dtype`):

1. `--require-native-support` is absent. Stored qualifications are BF16 only.
2. The request is a text artifact for a language model in `text` or
   `decision` mode.
3. The family is in `AUTO_F16_FAMILIES` = `qwen2`, `mistral`.
4. The repository has no record in `native_qualifications.json`.
5. Every floating tensor the plan reads (Safetensors header dtypes,
   including norms and tied aliases) is F16. The config's declared dtype is
   ignored: this Qwen2.5 export declares `bfloat16` but stores F16.
6. Every planned matrix is an MLX affine plane set (`MLXSpec`), the format
   the F16 profile was qualified with.
7. The existing native F16 admission (`preflight_model` with `float16`)
   accepts the metadata. This requires full attention and no sliding window.
8. The F16 re-plan is not `runtime_blocked`.

Any failure keeps the BF16 plan unchanged. The receipt records
`dtype_selection` with the requested value, the selected dtype and the reason.

Pre-download checks cannot see tensor dtypes. Preflight, `--capabilities` and
source admission therefore treat `auto` as `bfloat16`. F16 is chosen only when
its own admission also holds after download. The CLI default stays
`bfloat16`, and explicit `--dtype` values are unchanged.

## Families and kept BF16 paths

- **Qwen2 and Mistral** get automatic F16. They are the families with a
  recorded F16 pass (`-04`; `-05` on the Mistral weights).
- **Llama** stays BF16. No F16 Llama artifact has a model run; the run record
  lists Llama 3 RoPE F16 as kernel and admission coverage only.
  `mlx-community/Llama-3.2-1B-Instruct-4bit` and `-8bit` have stored BF16
  load qualifications. Their scale dtypes could not be read without a
  download: only their metadata is cached locally, and both declare
  `bfloat16`, as the all-F16 Qwen2.5 export does.
- **Qualified BF16 Qwen2 and Mistral repositories** stay BF16 through the
  qualification-record check: `Qwen/Qwen2.5-Coder-0.5B-Instruct` and
  `mistralai/Mistral-7B-Instruct-v0.3`. They are also unquantized, so rule 6
  excludes them. Their local cache entries hold metadata only, which declares
  `bfloat16`.
- **Unquantized F16 checkpoints** stay BF16. The F16 profile has no model run
  on an unquantized F16 source, only kernel tests. The catalogue and the
  guided recipes also label unquantized installs `BF16`.

## Consumers checked

| Consumer | Disposition |
| --- | --- |
| `desktop.py` install command | Adds `--dtype auto` for text artifacts only. Image installs, the metadata preflight command and native preparation are unchanged. No install uses `--require-native-support` |
| Install support check (`check_install_support`) | Unchanged. Its BF16 metadata admission accepts every source that `auto` can turn into F16 |
| Guided recipes (`matches_recipe`) | `_artifact_precision_matches` required BF16 parameters and would have refused an F16 install as a source mismatch. It now accepts F16 parameters only when the receipt records an automatic F16 selection for a Qwen2 or Mistral recipe. It refuses BF16 storage under such a record, and the record for any other family |
| Catalogue rows, `native_catalog.source_choice`, `catalog_installed_ids` | Not dtype-bound. MLX rows are labelled by bit width, and installed matching uses repository and revision |
| Guided memory plan | Unchanged: F16 and BF16 have the same width |

## Install-route check

The desktop backend's real install method ran on the pinned source,
`mlx-community/Qwen2.5-0.5B-Instruct-4bit` @ `a5339a41...` (read in place).
It used the local-folder route, which runs the real preflight and converter
subprocesses offline. A fresh temporary data directory outside the user's
library held the output, and nothing was downloaded. The commands, with paths
replaced:

```text
<PYTHON> -I -B -m stark_convert.cli <SOURCE> --preflight --support-mode text --artifact-mode text --metadata-profile hf-generic --offline
<PYTHON> -I -B -m stark_convert.cli <SOURCE> --output <STAGING>/model --support-mode text --metadata-profile hf-generic --offline --dtype auto
```

| Check | Result |
| --- | --- |
| Preflight | `conditional`, family `qwen2`; text and Decision `conditional`, images `unsupported` |
| Converter exit / installed models | 0 / 1 |
| `dtype_selection` | `float16`, `all_f16_mlx_affine_source` |
| Stored tensors | 218 F16, 49 F32 (norm scales), 97 U8 (packed codes); 364 tensors, 278,150,520 bytes |
| Output `model.safetensors` | `e227c07354092ee96211a0068e279f81ee26cc553c6f73265247cc904967be04`, equal to the qualified `-04` artifact |
| Config / tokenizer / source manifest | `bb3b6784...` / `9c5ae00e...` / `0e23197c...`, each equal to `-04` |
| Capability | `native_loading: unverified` (no stored qualification) |
| Source tree | Unchanged (names, sizes, modification times and content hashes) |

The installed artifact was then checked with `scripts/dense_native_validation.py
workers` (`172e177b...`, unchanged). It used this branch's workers and the
original `-02` reference fixture, the same check as `-04`:

| Check | Result |
| --- | --- |
| Generation, 39-token prompt | 13 tokens to EOS, reference greedy-exact |
| Generation, 43-token prompt | 32 of 32 tokens, reference greedy-exact |
| Replay, invalid-request recovery, restart replay | Exact |
| Scorer | Max error 0.031 on both cases; cache controls pass |
| Against the `-04` receipt and its commit recheck | Native output hashes, reference hashes, token counts, scorer results and verdicts identical |

The logit gate (`dense_parity`) and the prefix-reuse harness were not rerun.
The artifact is byte-identical to `-04`, which passed both.

Before the named-list change below, the published
`mlx-community/Mistral-7B-Instruct-v0.3-4bit` source (`a4b8f870...`, read in
place) went through the same route. Preflight was `conditional`, then the
`--dtype auto` conversion failed with `conversion_failed`, because the named
chat-template list blocked it. Nothing was installed, and the source was
unchanged. The catalogue marked this row downloadable: a template blocker only
removed it from guided setup.

## Published Mistral export (named template list)

This section is attempt `20261112-DENSE-MLX-07`, made by the second commit on
this branch.

The published export's `tokenizer_config.json` (`e8fbcfef...`) gives
`chat_template` as a Hugging Face named list with a `default` entry
(`a6fc0e6b...`) and a `tool_use` entry (`4c8a2468...`). Neither entry maps to
a reviewed tool parser. The `-03`/`-05` diagnostic copy substituted the
`default` string by hand.

### Rule

- **Accepted form:** a named list converts with its `default` entry,
  matching Transformers when no name is given
  (`template_compat.select_named_chat_template`).
- **Validation:** the entry is validated like any plain string.
- **Refused:** a missing `default`, duplicate names, keys other than `name`
  and `template`, non-string fields, empty or oversized lists, and
  name-to-template mappings. Image and classifier conversions and the pinned
  GGUF import also keep refusing named lists.
- **Provenance:** `conversion.json` records `chat_template_selection` (form,
  selected entry, names, each entry's SHA-256, `tool_templates: unreviewed`).
  `config.json` cannot carry it, because the native loader rejects unknown
  codec fields.
- **Tool gate:** `chat.load_template` and `model_inventory.model_tool_support`
  add a Python-only codec annotation from that receipt, and `tool_parser_for`
  returns no parser for an annotated codec. An unreadable receipt fails
  closed; an absent one adds nothing.
- **Catalogue:** offers the `default` entry as the row's template, with
  `supports_tools: false`.

### Results

| Check | Result |
| --- | --- |
| Desktop install (local-folder route, temporary data directory) | Preflight `conditional`; `--dtype auto` conversion exit 0; one model installed |
| Selections | Template `default` of `default`, `tool_use`; dtype `float16` (`all_f16_mlx_affine_source`) |
| Stored tensors | 260 F16, 65 F32, 130 U8; 455 tensors, 4,078,014,360 bytes |
| Output `model.safetensors` | `968cf950b4d7cba8ab2f0d1c97a708f0a1bfe2dc9f8f209aa6d8492e671d1a5a`, equal to `-05` |
| Config / tokenizer | `a8ced577...` / `3e017710...`, equal to `-05`. The whole codec, including the effective template, equals the `-05` copy's |
| Source manifest | `8df43010...`, not `-05`'s `f38f3991...`. The difference is expected: the published `tokenizer_config.json` differs from the substituted copy |
| Tool support | `unavailable` (installed-row inspection and loaded codec) |
| Worker checks (`-03` fixture) | 14 tokens to EOS and 32/32, reference greedy-exact; replay, recovery and restart exact; scorer 0.023 / 0.031. Native output hashes, scorer results and verdicts identical to `-05` and its commit recheck |
| Source tree | Unchanged |

### Prefix reuse

The prefix-reuse gate did not run. With the `-05` flags, the harness stops
with `ChatError` in its template preflight, before any worker loads.

A render probe, which reports only error types, shows why:

- The `default` template refuses a conversation that starts with a system
  message, even without tools.
- It also refuses the harness's tool-call steps.
- User/assistant-only turns render.

The harness always opens with a system message and grows through tool steps
(`--tools` must be positive), so it has no sequence this template accepts.
Completing the gate needs a user/assistant-only session mode in the harness.
That is a validation-tooling change and should be recorded separately. The
`-05` rows on the `tool_use` copy remain the only prefix evidence for these
weights.

The same system-message refusal applies in use: a chat or API request with a
system prompt fails to render on this artifact.

### Receipts

These files are private and kept outside the repository.

| Receipt | SHA-256 |
| --- | --- |
| Installed `conversion.json` | `4053c114b291482a0f98c1ea8173b703314fe862839086c411ec42573ea00a3a` |
| Worker receipt | `5688d1fbed53660a5b2bf89a33610ae5a03ff90e5d967fb2158b4f4c3fa3a05f` |
| Prefix harness output (both runs) | `245eed6f2545d9d0795bcaa3f510d56f4873d08fa045bdabdf403e32ffee81fe` |
| Install driver / render probe / template probe | `43db9326...` / `0e702ec7...` / `af523523...` |

`scripts/validate_generation_prefix_cache.py` (`58ad8a04...`) and
`scripts/dense_native_validation.py` (`172e177b...`) are unchanged.

### Tests

The named-list change adds `tools/converter/tests/test_named_chat_templates.py`
(23 tests) and one classifier refusal test. The tests cover:

- selection of the `default` entry and its hash-only record;
- every refused form;
- a conversion byte-identical to the plain `default` string;
- tool support `unavailable` from the installed row and the loaded codec, for
  a `default` program that would otherwise map to `mistral-json-v1`;
- fail-closed receipt reading;
- catalogue rows without tools.

One existing metadata case now uses a list without `default`. The full
converter suite has 7,677 passed and 94 skipped. The 16 CI ruff commands and
the repository policy check pass.

## System-role fold

This section is attempt `20261112-DENSE-MLX-08`, run on 2026-11-13. It is the
third change on this branch, after merging `origin/main` (`b88b5e6b`).

The published export's `default` entry raises on role order:
`(role == 'user') != (index % 2 == 0)` triggers `raise_exception`, and any
other role reaches a second `raise_exception`. A leading system message
therefore fails, as `-07` found. Mistral's own v0.3 template
(`mistral_v03.jinja`, Apache-2.0) accepts one leading system message and joins
it to the last user message with `"\n\n"`. Transformers adds no system
handling of its own.

### Rule

- **Detection** (`template_runtime.probe_system_role`). Conversion
  test-renders every text artifact's template through the native helper, with
  fixed synthetic messages. It records `chat_template_system_role` =
  `{"policy": "fold_into_first_user", "separator": "\n\n"}` only when:
  - a user-only turn renders;
  - a leading system message raises;
  - the folded single-turn and multi-turn conversations render.

  Otherwise nothing is recorded. A missing helper fails conversion.
- **Results on real templates:**
  - the published `default` (`a6fc0e6b...`) records the fold;
  - its `tool_use` entry (`4c8a2468...`), the Mistral v0.3 fixture
    (`e16746b4...`), the Qwen2.5 fixture (`cd8e9439...`) and the Llama 3.2
    fixture (`5816fce1...`) accept the system role and record nothing.
- **Fold** (`template_runtime.fold_system_message`). One leading system
  message is joined to the first user message's text with `"\n\n"`. Folding
  into the first message keeps earlier turns unchanged.

  It refuses, with fixed reasons and clear chat errors:
  - more than one system message;
  - a system message that is not first;
  - non-text system content;
  - no text user message immediately after the system message.

  Conversations without a system message pass through unchanged.
- **Wiring.** `IsolatedTemplate(system_role=...)` applies the fold before the
  helper evaluates the template. `chat.load_template` passes it only for the
  exact recorded policy, and any other recorded value fails to load. The Rust
  workers and the `stark-template` helper are unchanged, because they do not
  build chat messages.
- **Tools** stay `unavailable` under the named-list rule.

### Results

| Check | Result |
| --- | --- |
| Desktop reinstall (local-folder route, temporary data directory) | Installed; `float16`; template `default`; `chat_template_system_role` recorded |
| Weights / config / tokenizer | `968cf950...` / `a8ced577...` / `3e017710...`, unchanged |
| Synthetic system render | A system-plus-user render succeeds, equals the manually folded render, and differs from the user-only render (hash comparison) |
| Tool support | `unavailable` |
| Worker checks (`-03` fixture, rebuilt workers) | 14 tokens to EOS and 32/32, reference greedy-exact; scorer 0.023 / 0.031; identical to `-05` and `-07` |
| Prefix reuse (`-05` flags plus `--session-kind user-assistant`) | **Passed** (details below) |

Prefix-reuse details:

- The context was 8,192 tokens with a reuse chunk of 1,024. The session's turns
  had 2,652, 2,810, 2,986 and 3,628 tokens.
- All 32 requests were identical to a fresh worker and to the baseline worker,
  and every reuse count matched the rewind rule.
- Each follow-up reused 2,048 tokens (`follow_ups_reuse: true`), and the
  maximum reuse was 3,072.
- Five invalidations reused 0 afterwards: the error reply, an unscoped request,
  reset, an image operation and cancellation. Tokenize/decode kept reuse.
- Every request started with the folded system message, so the passing
  follow-ups also show the fold is a stable prefix on this artifact.

### Harness option

`scripts/validate_generation_prefix_cache.py` gains an opt-in
`--session-kind user-assistant`:

- It keeps the same system and first-user text.
- It declares no tools, and grows through user and assistant turns where the
  default grows through tool calls and results.
- It records `session_kind`.

The default session, its records and its gates are unchanged. This option does
not test tool histories, which this template cannot render.

### Receipts

These files are private and kept outside the repository.

| Receipt | SHA-256 |
| --- | --- |
| Prefix-reuse record / stdout | `87a453482668d2a2ca4299308a5c97d5d3e59065142e61604b7063abb5d6df2f` / `7fc064a001709ace39f31cc7153feeed225f4526329241d110180a5c80e8f1e8` |
| Worker receipt | `1794d6f707d2cdf1d9dfc33b95343d6cc5ce8106918e557b7eda2347e438ee27` |
| Harness after this change | `ffa987ede4a944a8d0f6a88c32d20e02df5fd07e374999586eb07e782cc0cec2` |
| Rebuilt workers (`b88b5e6b` source) | `stark-generate` `041ee11b00cf415897d1b1fadfdad0b7daf1602a8084006c2e44f3ead5ce0cf4`, `stark-score` `95159c2818178b9d22294795eaf9867e4191415e8521921091c9d31cca69813c`, `stark-template` `886769ce...` (unchanged) |

### Tests

- `tools/converter/tests/test_system_role_fold.py` (16 tests). It covers fold
  targets and every refusal, detection on synthetic templates, the published
  `default` entry (read from the local snapshot and referenced by hash),
  prefix stability against a fold into the last message, unchanged rendering
  without the policy, exact-policy loading, and conversion receipts.
- `scripts/test_validate_generation_prefix_cache.py` gains one test of the
  opt-in kind and of default equivalence (19 pass).
- The full converter suite after merge `b88b5e6b` has 7,708 passed and 94
  skipped. The 16 CI ruff commands and the repository policy check pass. No
  Rust or `stark-template` source changed.

## Build and environment

| Item | Identity |
| --- | --- |
| Engine source | `103a8296` (engine identical to `f203f004`) |
| `stark-generate` | `8a2b700cf86b630d928179178e4c68172ca7acf223cb43514cf48529c6aa9e2c`. It differs byte-wise from the `-04` commit recheck (`2adf7b0d...`), which was built in another checkout; outputs are identical |
| `stark-score --features optimized` | `74dd50b32d5796c7697c38337527c8893ef814b02f1c38bc5779bad320db6316` |
| `stark-template` | `886769cea0c05abf9dac35cd21272353f5dd0c0ee0faa5f97c4c43088b1182ac` |
| Toolchains and locks | Rust 1.98.1 release, locked; `Cargo.lock` `f836a99b...`, `rust-toolchain.toml` `72da9f9e...`, `tools/converter/uv.lock` `df771172...`; Python 3.13.13 |
| Host | Apple M5 Max, 128 GiB, macOS 26.7.2. No timing or memory is reported |
| Private receipts | Worker receipt `9ca3146499c0166f5816844357ab81350083163ebd0621fdfdd5c6a26947fa88`; install receipt `conversion.json` `ea8f1d53f8fe6abbd42b9b1fa4f38ac8365370c241168c0ef03e4bc350c5794e`; install driver `e95418ec...`, comparison script `29ac3708...` (kept outside the repository) |

## Tests

- `tools/converter/tests/test_auto_dtype.py` (19 tests). An all-F16 Qwen2 or
  Mistral MLX source selects F16, byte-identical to `--dtype float16`. BF16,
  mixed (F32 norm), sliding-window, Qwen3, Llama and unquantized F16 sources
  stay BF16, byte-identical to `--dtype bfloat16`. Also covered: explicit
  dtypes win, recorded qualifications keep BF16, Decision mode,
  `--require-native-support`, and CLI preflight and conversion with `auto`.
- `tools/converter/tests/test_desktop_auto_dtype.py` (10 tests). Real desktop
  local installs through the preflight and converter subprocesses, plus the
  Hugging Face command shape for text, Decision and image modes.
- `test_guided_models.py`: one parametrized test (4 cases) for F16 guided
  recipe matching.
- The full converter suite has 7,653 passed and 94 skipped. The base recorded
  7,620 passed and 94 skipped; the difference is the 33 new cases. The CI ruff
  commands and the repository policy check pass. Rust is unchanged.

## Not verified

- The Hugging Face download route. Only the local-folder route ran, because
  downloads were out of scope. Both routes build the same converter command.
- An end-to-end guided-setup install. Recipe matching is covered by model-free
  tests.
- 8-bit and larger Qwen2.5/QwQ exports and other Mistral exports. Automatic
  selection reaches them, but each artifact still needs its own qualification.
- Timing, memory, tool calling and sampled generation.
