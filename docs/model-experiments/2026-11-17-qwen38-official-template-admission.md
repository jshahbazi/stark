# 20261117-Q38OFF-01/02: official dense Qwen3.8 template admission

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | 20261117-Q38OFF-01/02 / 2026-11-17 / 2026-11-17 / local development |
| Kind | Compatibility and end-to-end image installation |
| Attempt state | Completed |
| Result and acceptance criterion | Baseline failed native model loading; candidate passed the automatic local image qualification recipe for the exact converted artifact |
| Evidence scope | Exact conversion, native readiness, text execution, invalid-image recovery, two fresh image executions, token accounting and staging cleanup |
| Promotion decision | Promoted on branch for this exact checkpoint/template identity only |
| Disposition | Adopted on branch; not released or installed |
| Predecessor / related attempts / governing issue | [2026-11-08 Qwen images](2026-11-08-qwen-images.md); [image-default parity plan](../IMAGE_DEFAULT_PARITY_PLAN.md) |

## Hypothesis and change

- The official `Qwen/Qwen3.8-27B` source has a reviewed chat template that
  contains Transformers-only generation delimiters. The converter correctly
  removes those delimiters from the runtime prompt, but the native family
  admission and Python chat loader compared the normalized prompt to the raw
  sidecar as if they had to be byte-identical.
- Baseline accepted the exact official source in conversion, then the native
  worker returned `model_load_failed`. The candidate binds the reviewed raw and
  normalized template hashes, admitting only the intended raw/normalized pair.
- Acceptance required native load, the existing greedy text check, invalid-image
  rejection, two valid image requests with exact token accounting, fresh-request
  equality and cleanup. No response quality, numerical parity or speed claim is
  attached to this admission repair.

## Artifact, source and environment

| Field | Baseline | Candidate |
| --- | --- | --- |
| Exact source | `Qwen/Qwen3.8-27B@37561816e66b5b9153b1b64c10968c18c1e6f4a0` | Same |
| Source and output format | 18-shard official BF16 Safetensors converted to exact-family BF16 `qwen35-dense-bf16-merged-v2` | Same |
| Artifact hashes | model `0bf350d1722bb1086073ef7964e3cb68f43a49c586d216fa7d953b7a968834f8`; config `9ddeb7ad7dc5bf05a5e9bc5abd27440e21b146a0df6dd4e9a67f27be3c0f19cb` | Same |
| Processing/template hashes | tokenizer `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4`; processor `6a970fd06f30e6943b3e2c14d5d3b42d49b06cf99b99103d56689bef462d90f8`; raw template `af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375`; normalized runtime template `61c5db6e90356e85fb2ccfd0466fddf3b2a5c6058b20b195a37d7b01de97c8b8` | Same |
| Stark identity | Source `6d8a911ff88559335e9c1e3bdc0ff9f80ce0aaa5`; installed baseline worker `0737cc59bdb26d6ee150f368520f2f2efc02b2448b59194a85fda2dfecb6e355` | Same base plus the branch patch; Rust 1.98.1 locked release worker `fc5c0e31cf7ed668b05165ee7d3a97056c8675bb0d66e4c8019c6a72843f9e71`; template helper `32cfe829f33cb0c5cc5eb5947d7623f0bb96a70337011c2c35a8513e2b881667` |
| Precision | BF16 weights/activations, existing F32 accumulation and state policies | Unchanged |
| Hardware/environment | Apple M5 Max, 128 GB unified memory, macOS 26.7.2 | Same |

The generated artifact and model files remain private ignored runtime data.
Validation did not copy or publish model content, prompts, outputs or token
vectors.

## Workload and controls

- The catalog install default was vision-language conversion because complete
  supported BF16 image metadata was present.
- The source's MTP configuration was retained by conversion; this automatic
  qualification used the existing greedy MTP-off text/image checks.
- The local checker used at most 2,048 context tokens and 8 output tokens, the
  fixed synthetic image, one invalid PNG, and two fresh valid-image requests.
- Prefix reuse, matched text-only versus image-loaded speed, sustained contexts,
  image-to-text cache recovery, quality and numerical source parity were not
  part of this run.

## Attempt history and failures

| Trial / source | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| Q38OFF-01 | Official image conversion and installed native worker | Conversion completed, but native loading returned a content-free `model_load_failed`; diagnosis identified template admission and raw/normalized template mismatch | Valid reproduced installation failure; corrected |
| Q38OFF-02 | Raw/normalized official-template pair added to Python and Rust admission | Rebuilt worker loaded the exact 27B artifact, completed eight text decode steps and the complete automatic local image checker | Passed functional admission; retained with broader qualification limits |

## Correctness and lifecycle

- The ignored native text diagnostic completed load, state creation, prefill and
  eight greedy decode steps without publishing tokens.
- The Python worker reported `ready=true`, `mode=text-generation-v1`,
  `images=true` and `image_verification=locally_tested` with a fresh check.
- The checker passed text generation, invalid-image rejection, two valid image
  executions, expected image-token deltas, fresh reply accounting, and staging
  cleanup. No private completion content was recorded.
- Python regressions: 247 tests passed across `test_image_checks.py`,
  `test_model_support.py` and `test_desktop_family_checks.py`.
- Native regressions: 7 family tests passed; 2 unrelated private-artifact tests
  remained ignored. The locked Rust 1.98.1 release worker build passed, as did
  Rust formatting.

## Performance and memory

Functional validation only. No timing, throughput, peak-memory or paging claim
was measured. The model retained its existing BF16 precision policy and
image-resident text path.

## Evidence, decision and limits

- This attempt fixes the installation blocker, not model quality or numerical
  parity. The resulting evidence is bound to the exact official 27B BF16 source,
  conversion hashes and candidate executable above.
- The raw template hash and normalized runtime hash are both required. Unknown
  templates, alternate normalization and other checkpoints do not inherit this
  result.
- Default catalog support may proceed to the existing local-execution gate, but
  matched image-loaded performance, full context/resource behavior, MTP/cache
  restoration and installed-app release qualification remain separate.
