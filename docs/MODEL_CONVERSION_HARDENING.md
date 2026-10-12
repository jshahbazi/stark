# Converter hardening

Further architecture and checkpoint expansion is deferred at the owner's direction. This packet hardens installation, local-file conversion, provenance and benchmark failure handling. Adapter coverage and numerical transformations remain at milestone 6.

## Acceptance checklist

- [x] Build an sdist and wheel with explicit upstream license inclusion and a pinned build backend.
- [x] Bundle required checkout inputs under their original logical paths for installed provenance.
- [x] Reject missing, extra or ambiguous provenance inputs.
- [x] Test relocation identity, source-versus-installed identity and installed commands outside the checkout.
- [x] Convert self-contained downloaded folders offline through catalog and anonymous `hf-generic` modes.
- [x] Cover paths containing spaces, sharded weights, missing metadata and existing-output refusal.
- [x] Explain local metadata policy in help and provide a fixed hint when it is absent.
- [x] Validate benchmark receipt status and finite, correctly typed metrics.
- [x] Retain completed measurements and the failed repeat after errors or interruption.
- [x] Write private reports atomically and sanitize report-write errors.
- [x] Exercise installed benchmark repetitions and parity without a checkout or `uv` launcher.
- [x] Add clean-install verification to existing repository CI.
- [x] Complete local checks and independent technical review; see the [validation record](MODEL_CONVERSION_HARDENING_REVIEW.md).

## Installation and provenance

Version 0.9.1 embeds the converter's project metadata, Python pin, upstream license, build hook, manifest, catalog exporter and dependency lock. Installed provenance uses the original logical filenames, so a wheel and its checkout have the same converter source and lock identity. Package Python code and catalog resources remain hashed at runtime. Missing inputs fail closed. Editable installations remain bound to their checkout.

The bundled lock records the intended dependency set; a wheel installer does not enforce it automatically. Use the documented hash-checked lock export and install the wheel with `--no-deps`. Publication, signing, hosting and model redistribution remain separate scoped work.

## Local downloads

A self-contained folder needs a supported `config.json`, `tokenizer.json`, `tokenizer_config.json`, and safetensors weights or an index with every referenced shard. Generic language conversion additionally needs a supported chat template and explicit EOS metadata. Optional generation/template files are consumed and verified when present. Anonymous local `hf-generic` conversion requires no account, remote repository name or source manifest.

Catalog profiles retain checkpoint-specific metadata routing. Profiles using external metadata need a complete private source manifest for offline replay. The converter does not guess profiles from folder names or synthesize missing metadata. Generic outputs retain their existing unqualified native status.

## Benchmark report contract

Schema 3 starts each fresh conversion process with the current environment's Python. Reports identify `measurement.launcher=current_python`. Original schema-2 measurements using `uv` keep their original protocol and hashes. This packet makes no new Lalamo latency claim or qualification.

Preflight checks directories, validates any reference and hashes selected files and tooling before creating output. Once the benchmark owns the new output directory, it saves owner-only `report.json` before each repeat, after each accepted measurement and at completion. `status=running` means incomplete. A completed failure records `status=failed`, a fixed error category and the active repeat; successful earlier measurements remain available. Ctrl-C returns 130, other failures return 1 and passing runs return 0.

Atomic replacement preserves the last complete report when a write fails. The command then emits `report_write_failed` and exits unsuccessfully; any retained running report is not qualification evidence. Benchmark directories are never reused. Partial outputs remain private investigation artifacts.

## Validation

Run from the repository root:

```sh
HF_HUB_OFFLINE=1 uv run --project tools/converter --locked --python 3.13.13 pytest tools/converter/tests -q
uv run --project tools/converter --locked --python 3.13.13 ruff check tools/converter
uv run --project tools/converter --locked --python 3.13.13 ruff format --check tools/converter
uv lock --check --directory tools/converter
uv run --locked --python 3.12.13 --script scripts/ci/check_repository.py
uv run --project tools/converter --locked --python 3.13.13 python tools/converter/scripts/check_installed_package.py
```

The installed-package check builds through an sdist, installs the wheel with locked runtime dependencies in a temporary environment outside the checkout, compares provenance and exercises local conversions plus fresh-process benchmark parity. Children have an empty Hub cache and no Hub credentials. Tiny synthetic model files, environments and distributions are removed on exit; no real model downloads are required.

Later hardening candidates remain separate: a supported total-process memory ceiling, stronger malformed-header/receipt resource limits, crash/power-loss recovery guarantees, and engine compatibility policy for eventual publication. Existing workspace bounds, no-replace publication, source revalidation and exact numerical fixtures continue to apply.
