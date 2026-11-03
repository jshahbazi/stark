# GLM native model catalog integration — 2026-11-03

Source commit: `bb5a3338eb2b07307794b9a66a62840eec54e3e0`.
Branch: `codex/glm-model-catalog`.

## Cause and implementation

The native GLM loader and pinned preparation helper existed, but the model
catalog enumerated generic converter families and Safetensors sources. GLM's
`glm5_next` GGUF path had no catalog family or installation route.

The shared inventory now includes a separate GLM native family. Its catalog
offers only the existing pinned `glm53-q2` choice, with text chat support.
Generic converter architecture recognition and preflight remain unchanged.
Catalog installation uses the isolated native preparation helper, validates
the prepared identity, and registers the completed managed folder. Private
source-bound staging retains interrupted downloads and completed artifacts
after registry failure; a per-source lock serializes installers.

The selected source remains `unsloth/GLM-5.3-Flash-GGUF` at
`a38483c8cd5df544f53d70fb281afe97369d5ab6`, using four `UD-Q2_K_XL` shards
totaling 108,720,071,552 bytes. Metadata remains `zai-org/GLM-5.3-Flash` at
`04c4e9e95c5da8862dced7e5056455116f83a7e0`.

## Validation

- Built `stark-template` with locked dependencies and Rust 1.98.1 for local tests.
- Full converter suite on the source commit: 5,946 passed and 89 skipped in
  230.93 seconds with `HF_HUB_OFFLINE=1` and `TOKENIZERS_PARALLELISM=false`,
  using locked uv dependencies and Python 3.13.13.
- Focused native preparation and desktop browser suite: 53 passed, covering
  pinned routing, real child-process progress pipes, identity mismatch,
  cancellation, partial-download retry, completed-download reuse, registry
  rollback, concurrent installer rejection and symlink rejection.
- Swift controller harness: 119 checks, zero failures and zero skips. Its live
  registry checks include the GLM family, GGUF format and text-only modes.
- Converter Ruff lint and format checks, repository policy and diff checks passed.
- Hugging Face metadata queries confirmed both pinned revisions are accessible,
  all four selected GGUF shards have the expected total bytes, and the metadata
  repository contains the required configuration and tokenizer files.
  No weight payload was requested.

## Installed app

Built a Developer ID signed development app using the generation, scoring and
template executables from the previously installed app as build inputs. This
change does not alter Rust/Metal execution. The builder's model-free executable
protocol probes passed, and `codesign --verify --deep --strict` passed before
and after installation into `~/Applications/Stark.app`.

Development build: `20261103190601511231`.
Installed app manifest SHA-256:
`ca0604ef932e7e337fdda564a71dcd9eac4a8695d0b1109c97a9ee72dc12b689`.

The installed `desktop.py`, `hybrid_prepare.py`, `model_catalog.py`,
`model_support.py` and `native_catalog.py` hashes match the source commit.
The packaged and installed metadata-only catalog checks returned the one
expected GLM entry. A live UI check confirmed **Manage Models → Model Catalog →
GLM-5.3 Flash** displays `UD-Q2_K_XL`, 108.72 GB and an enabled Download button.
The app was left on that screen.

Installed signed executable SHA-256 values:

| Executable | SHA-256 |
| --- | --- |
| `stark-generate` | `17e444dd554497f57dd1fac14d8d8e2ca817b3cbc2f83f1072faf81307e7332a` |
| `stark-score` | `dc57624d5f8decf9389060220591bd9976e975c9a8577b2b536c220938514ee1` |
| `stark-template` | `b262a160d27ece09fbd5807a95e99b179db6ac8b0551195c7d378decf0b2ed84` |

No real model download, load, inference, numerical comparison or performance
benchmark was performed. Synthetic tests and metadata checks qualify the catalog
and installation integration; they do not establish fresh GLM execution evidence.
