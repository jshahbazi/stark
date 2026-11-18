# 20261117-QWEN38-CATALOG-01: official dense catalog admission

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | `20261117-QWEN38-CATALOG-01`, registered and executed 2026-11-17 |
| Kind | Compatibility and catalog admission |
| Attempt state | Completed |
| Result and acceptance criterion | Passed for bounded source metadata, discovery and model-free conversion/catalog tests; full-model execution not run |
| Evidence scope | Structural and catalog metadata |
| Promotion decision | Adopt catalog and family metadata; artifact qualification remains pending |
| Disposition | Adopted on branch |
| Predecessor / related attempts | Dense Qwen3.5 family conversion and Qwen3.8 oQ4e MTP records |

## Hypothesis and change

The official Qwen3.8-27B dense image-language checkpoint uses the same
Transformers `qwen3_5` architecture and tensor conventions as Stark's existing
dense Qwen3.5 path. It should appear as a pinned source in that family without
creating a duplicate backend or selecting the separate Flash Next MoE path.

The catalog now searches for `Qwen3.8`, pins the immutable official revision,
accepts the exact reviewed vision template, and selects the existing
`qwen38-xml-v1` tool protocol for the normalized template.

## Artifact, source and environment

| Field | Candidate |
| --- | --- |
| Exact checkpoint | `Qwen/Qwen3.8-27B` revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0` |
| License | Apache-2.0 |
| Config SHA-256 | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| Raw template SHA-256 | `af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375` |
| Normalized template SHA-256 | `61c5db6e90356e85fb2ccfd0466fddf3b2a5c6058b20b195a37d7b01de97c8b8` |
| Normalized program fingerprint | `f15832d2404398fe7cfcd5639938be4884b663b0b3921d80765af4a5578150de` |
| Tokenizer SHA-256 | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| Processor SHA-256 | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| Source tokenizer config SHA-256 | `207c03c41a06a9c3b0cc527d8a56115159c1eac6b59a4ffae85b646e773d074e` |
| Backend | Existing `qwen3_5` dense/VL converter and worker path |
| Base implementation revision | `82f01f8edf37367c06aade6ddeba0387cefc5a96` |
| Host | Apple silicon, macOS 26.7.2, arm64 |
| Precision policy | Candidate source is BF16; conversion writes BF16. Exact activation, KV, recurrent-state and full-model memory qualification is pending. |

Weight hashes and native executable identity are not recorded because the full
54 GB class source was not downloaded or converted in this attempt.

## Workload and controls

- Catalog pin and search identity checked against live source metadata.
- Vision processing, tokenizer, tool template and prefix-reuse inventory use the
  existing Qwen3.5-family implementation.
- Bounded header admission read the metadata and tensor headers for all 18
  source shards; text and image layout planning passed for 55,563,006,776
  weight bytes and 27,781,427,952 reported parameters.
- Exact repository search returns the official source as downloadable under
  `qwen3_5`, with `vision-language` as its default artifact mode.
- Full payload conversion, native load, output parity, tool behavior, image
  execution, MTP, cache exactness and performance were not run.
- Prefix reuse is implemented for the admitted dense backend, but this exact
  checkpoint remains qualification pending.

## Results and limits

The source is compatible at the metadata and configuration-contract level. The
catalog pin is independent of family ranking and keeps the official revision
visible as a Qwen3.5/Qwen3.8 dense source.

This record does not qualify another Qwen3.8 checkpoint or merged/fine-tuned
artifact. Native execution, exact prefix restoration, tools, default image
behavior, memory and speed require a separate source-bound run. No prompts,
outputs, token vectors or private model validation artifacts were retained.
