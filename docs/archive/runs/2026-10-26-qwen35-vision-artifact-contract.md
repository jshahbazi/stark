# Qwen3.5 vision artifact conversion boundary

Date: 2026-10-26. Scope: converter implementation and model-free checks on
`codex/qwen35-vision-artifact-contract`, based on repository revision `6499d5c2`.
This is a new validation record; it does not replace historical text conversion
or native qualification evidence.

## Source binding

| Input | Identity |
| --- | --- |
| Converter package sources | SHA256 `0ec0834988b5d426da24cef2102babba5e510b5774ccbea5b3640e94393e56f9` |
| Converter dependency lock | SHA256 `df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5` |
| Official model/processor/template | `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17` |
| Vision architecture reference | Transformers revision `2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397` |
| Preserved text conversion reference | Lalamo revision `831accb8252a6b6606c378048caeb44178f1a050` |

The converter identities come from `stark_convert.provenance.toolchain_provenance`.
The package hash binds the conversion implementation and conversion-affecting
resources, independently of this documentation file.

## Evidence

- Synthetic complete conversions retain vision tensors and source metadata,
  publish the experimental model tag and M-RoPE contract, and remain
  `runtime_blocked` with no native qualification. Both inline and standalone
  template selection paths pass.
- Boundary cases cover missing/extra/malformed visual tensors, wrong dtypes,
  metadata and weight drift, reserved-token conflicts, changed metadata copies,
  manifest mode/schema substitution, wrong source revisions, failure cleanup,
  and rejection of unsupported tuples before source resolution.
- The validator accepts all five official metadata files at their hardcoded
  hashes, using the actual Tokenizers parser and special-token metadata. This
  check supplied weight/index manifest identities from the immutable source API;
  it did not read or verify the weight payload locally.
- Only the safetensors prefix and 59,840-byte JSON header were fetched to compare
  the visual inventory. All 153 official visual tensors match the planner's
  names, shapes and BF16 dtypes, with exactly 201,185,792 visual payload bytes
  represented by the descriptors. No tensor payload was fetched.

The official [source tree metadata](https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17?recursive=true&expand=false)
provides the LFS payload identity. The index hash comes from its small downloaded
JSON file:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `model.safetensors-00001-of-00001.safetensors` | 1,746,942,600 | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| `model.safetensors.index.json` | 50,900 | `d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e` |

## Local checks

Commands, run from the repository root:

```sh
uv run --locked --python 3.12.13 --script scripts/ci/check_repository.py
uv run --project tools/converter --locked --python 3.13.13 ruff check tools/converter/stark_convert tools/converter/tests
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false uv run --project tools/converter --locked --python 3.13.13 pytest tools/converter/tests -q
git diff --check
```

Final suite result: **4,178 passed, 88 skipped** in 108.10 seconds. Repository
policy, Ruff lint and formatting checks, and `git diff --check` passed. The
focused progress and vision conversion checks also passed after restoring the
existing source helper export and adding checked metadata copies.

## Limits

No model weights were downloaded, no full official-weight conversion ran, and no
native model execution or reference parity experiment ran. This evidence does
not qualify the artifact for loading, preprocessing, numerical correctness,
visual question answering, OCR, latency, memory budgets, cancellation, or app
packaging. No Rust code or dependency lock changed. Native image support remains
unimplemented; its dependencies are in the
[implementation contract](../../MULTIMODAL_IMPLEMENTATION.md).
