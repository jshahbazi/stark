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

The published `mlx-community/Mistral-7B-Instruct-v0.3-4bit` source
(`a4b8f870...`, read in place) went through the same route. Preflight is
`conditional`, then the `--dtype auto` conversion fails with
`conversion_failed`, so the named chat-template list still blocks it. Nothing
was installed, and the source was unchanged. The catalogue still marks this
row downloadable: a template blocker only removes it from guided setup.

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
