# Qwen3.5 vision artifact follow-up review

Date: 2026-10-26. Reviewed baseline: `fe936b7e007311b23acefb4e636c2215ed7836bc`
on `codex/qwen35-vision-artifact-contract`. The scope is the Stage 1 converter
milestone from the first-image implementation brief: exact checkpoint and
processor identity, complete visual tensor preservation, explicit text mode,
versioned artifacts, and a closed runtime boundary.
The governing brief is "Stark image support: first implementation milestone,"
research dated 2026-10-26, based on source
`913010c2d8e4e9e2d058c17f0852d98a09ccf415`.

**Verdict: conforms for the converter milestone after the correction below.**
Native image implementation and qualification remain pending. This record adds
real-checkpoint conversion evidence to the
[initial validation](2026-10-26-qwen35-vision-artifact-contract.md); that earlier
record retains its original source binding and limits.

## Requirement map

| Requirement | Governing brief section | Inspected implementation and evidence | Status |
| --- | --- | --- | --- |
| Exact official BF16 model, processor and template | First model; Stage 1 | `vision_contract.py` pins metadata, source payload and index hashes. The complete cached official checkpoint passes the CLI conversion and final source checks. | Conforms |
| Every required visual tensor retained, invalid inputs rejected | Native artifact changes 1; Stage 1 exit | `qwen35_vision.py` enumerates all 153 source vision tensors. Every converted visual shape, dtype and byte hash matches. Boundary tests reject missing/extra tensors, invalid shapes and dtypes. | Conforms |
| Processor, template, special IDs and positional metadata bound to the artifact | Native artifact changes 1; versioning | Raw metadata is preserved and hashed; normalised tokenizer IDs agree with all four reserved IDs. All six full-attention layer descriptions retain the expected M-RoPE sections, interleaving and rotary width. | Conforms |
| Versioned manifest and explicit text mode | Stage 1; versioning | Schema 3 binds vision mode. Real offline replay reproduces all artifact hashes. Existing text conversion and generic replay regression tests pass. | Conforms |
| Unsupported native execution fails closed | Qualification gate; versioning | Receipt remains `runtime_blocked`, with no qualification. CLI rejects native vision support requests before source resolution. Native config deserialisation is strict and rejects the unknown model tag before opening weights. | Conforms; native rejection inspected in source |
| Worker transport and service capabilities stay bounded | Bounded transport; versioning | Generation/scoring request lines remain 1 MiB. Python and Swift still advertise images as unavailable. Rust configs, worker protocols and dependency locks did not change. | Conforms |
| Native image execution, preprocessing, transport and app UI | Stages 2–5 | These are explicitly unfinished and cannot be established by conversion evidence. | Pending for the broader milestone |

The governing source is the user's first-image implementation brief, sections
listed above. The repository's [implementation contract](../../MULTIMODAL_IMPLEMENTATION.md)
describes the current implementation boundary; it does not establish native
acceptance or supersede the brief's qualification gates.

## Corrected finding

`--source-manifest --artifact-mode vision-language --require-native-support`
previously resolved and rehashed the source before rejecting unsupported native
execution. The CLI's replay path deferred request validation to infer source
identity. This contradicted its documented early-rejection behavior and could
read the full 1.75 GB checkpoint for a request that must fail.

[`cli.main`](../../../tools/converter/stark_convert/cli.py#L555) now validates
that unsupported combination before resolution. New
entrypoint tests cover ordinary and replay requests, and a successful replay
test preserves repository inference from a local manifest. No other reportable
correctness defect remained after independent source and artifact reviews.

The native boundary follows from the config macro's
[exact type and strict deserialisation](../../../crates/stark-engine-macros/src/stark_config.rs#L152)
and [configuration parsing before weight-file access](../../../crates/stark-engine/src/engine/language_model/mod.rs#L69).
That boundary was inspected in source; no native execution was performed.

## Real checkpoint conversion and replay

The official weights were already cached. Only missing processor/template
metadata was downloaded. The cache uses additional links into a shared blob
pool, which the existing source-root policy rejects. Validation used a private,
ignored source directory containing hard links to the resolved regular files.
The actual converter then verified all pinned payload and metadata hashes. The
source authorization policy was unchanged.

The CLI converted that source in BF16 with the default 8 MiB transform workspace.
An independent pass compared all 153 source/converted visual payload hashes,
covering 201,185,792 bytes of vision weights. A second CLI invocation replayed
the private schema-3 manifest offline; weights, config, normalised tokenizer,
source-manifest identity, and artifact contract were identical. The duplicate
replay output was removed after comparison. Model files, private manifests and
receipts remain under ignored runtime storage.

| Binding or result | Value |
| --- | --- |
| Official checkpoint revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| Converter source SHA256 after correction | `53797018c8b663e16c438e93b2bb2ae302c57372075a49243b8a4a82a40279e6` |
| Converter lock SHA256 | `df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5` |
| Source manifest SHA256 | `4f435c7834e8732cdaba4bea881b0573cce48da197200096fb0da1777c6cb598` |
| Converted weights SHA256 | `7c5960e7225b3b36844fa397a85c5e20d281d732e382475062f552c0737c79d4` |
| Converted config SHA256 | `26583eb8a5d2a86aac80d9205eceda6133f0e63ac9ff7085364f8be65fd6139b` |
| Normalised tokenizer SHA256 | `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4` |
| Output tensors | 383, including all 153 vision tensors |
| Output bytes | 1,707,029,400 |
| Reported transform peak | 8,388,608 bytes |
| Capability | Source validation complete; native loading blocked; execution unverified |

## Regression checks

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false uv run --project tools/converter --locked --python 3.13.13 pytest tools/converter/tests -q
uv run --locked --python 3.12.13 --script scripts/ci/check_repository.py
uv run --project tools/converter --locked --python 3.13.13 ruff check tools/converter
uv run --project tools/converter --locked --python 3.13.13 ruff format --check tools/converter
uv lock --check --directory tools/converter
git diff --check
```

Final suite: **4,181 passed, 88 skipped** in 110.15 seconds. Policy, lint,
formatting and dependency-lock checks passed.

## Remaining limits

No native image inference, image decoding/preprocessing, reference intermediate
or logit parity, image token accounting during execution, performance or hardware
budget qualification, cancellation experiment, or app packaging check ran.
The runtime remains blocked. This conversion is a basis for implementing the
native path, and establishes no image-serving capability.
