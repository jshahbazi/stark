# Shared load-time generation context sizing

## Identity and policy

| Field | Value |
| --- | --- |
| Attempt | `20261125-CTX-01` · 2026-11-25 · Codex |
| Kind | Memory admission and generation worker lifecycle |
| Predecessor | [Inkling 2 Mini request-sized allocation](2026-11-24-inkling-mini-request-context.md), `20261124-INKCTX-02` |
| Source | `ee58b284f6c307b40f709c292b184e5ed4382ec3` |
| Baseline | `b92055b983b3c5d6887abf63422d04849544af27`, packaged worker SHA-256 `594c68d74304fe29847eeb1ba6d4ee9d524167f46acfc9e30651116d03d54f60`, raw worker SHA-256 `745b1cb943e7d49785c8abf5e07c359e4f30399226f43b95dffaff16f246e6e7` |
| Candidate | Rust 1.98.1 release generation worker with default Metal features, SHA-256 `ff54b88ad9d80768970011d5b47559f43a650a4af184890fb1da6db54f7b7f84` |
| Cargo lock | `07b6d42497825a651719685a28a107bfc0a32119b8e8b8adeab2d6c01a96695b` |
| Host | M5 Max, 128 GiB unified memory, macOS 26.7.2 (25G332) |
| Disposition | Implemented on the feature branch |

The [engine policy](../GENERATION.md#load-time-context-sizing) computes a stable advertised window from loaded model geometry and memory at startup. Inkling supplies its existing cold cache and workspace estimate. Native and hybrid backends with documented finite windows price full cache capacity with up to 2,048 prefill tokens. The native decoder uses its higher ordinary-text or dense-MTP estimate when it has a predictor. Image-capable models size text with their loaded towers resident, and each image request still accounts for its expanded tokens and transient image workspace. The optional GLM EXL3 bridge admits its configured context internally. Undeclared contexts retain request-level sizing. Every backend retains its own execution allocator and prefix restoration rules.

## Matched workers on Metal

The first load probe set the input to the candidate window minus one. On the installed Qwen3 4B 4-bit artifact that crossed a prefill scratch-estimator ceiling at 32,769 tokens, even with 109,951,352,832 bytes of sizing budget. It reduced the public window from 40,960 to 27,648. That probe was rejected. The capped prefill sample still prices the full cache, and per-request planning checks the actual prefill geometry.

| Model | Baseline window | Candidate window | Control |
| --- | ---: | ---: | --- |
| Qwen3 4B 4-bit, native text | 40,960 | 40,960 | On a 2,481-token input, the two workers sampled the same 12 greedy tokens, reused 2,048 on an exact repeat, rejected a request above the reported maximum, and reproduced the output on the next valid request. |
| Qwen3.8 Flash Next Lily Q4-XL, image tower loaded | 262,144 | 262,144 | Both loaded the same vision artifact, tokenized 323 input tokens, sampled EOS at 512 logical context and rejected a request that exceeded that request's context. |
| Inkling 2 Mini, Stark affine-8 | 570,368 | 528,384 | Both workers sampled the same 12 tokens when requested with zero and one DSpark drafts, used a physical capacity of 1,024 within a 4,096-token request, and rejected a request above the advertised maximum. |

The Qwen3 4B artifact was a local conversion. Its config SHA-256 was `5b505594bf00ef9682b373c6a292b88f94f1e1ceab449784c12accb835806c7f`. The fixed request used 4,096 context tokens and the candidate sized its advertised window with a budget of 110,164,180,992 bytes.

The Lily artifact was `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, config SHA-256 `2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899` and vision identity `18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4`. Its candidate sizing budget was 39,652,327,424 bytes. The private paired smoke receipt SHA-256 is `3ab0fe53c4d3d126a0fced072d6683dc30c9d1f3e0eb20cf2b85c0c774f2ebbe`. A candidate-only synthetic image request still used the 512-token logical window after an overflow rejection. It admitted 83 expanded input tokens and generated one greedy token with a `length` finish. The image receipt SHA-256 is `5f73e8aeab17452722f983badf1f5729d6cecdef35a0e68cd368933bceb5ba51`.

Inkling used `thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, converted artifact `c9e14848ad017467dc0deb91c6aa67022e181dbcdfe02a1b04024434bf9acddd` and config SHA-256 `a63529b36d0ce7fac8b13705c54463e4b1cbb34936d27eefa6a756ab2bd7d4fd`. The paired short requests used 321 input tokens. Sizing budget differed between the two starts at 55,388,078,080 and 51,302,563,840 bytes, respectively. The window difference reflects the memory samples and the same sizing formula.

## Local checks

- Engine context and memory inventory filters passed 18 tests. Synthetic controls covered two different budgets for a fixed model, engine and checkpoint limits, unavailable estimates under low memory, and a rounded window that must shrink to fit a change in scratch size classes.
- Generation worker tests passed 175 tests with six skipped for private artifacts. The process boundary check passed six cases against the release worker.
- CLI and desktop host tests passed 496 with one skipped. They cover model-specific defaults, custom limits, and the unknown-capacity contract. Clients discover the selected window through the existing context field.

## Scope

Load sizing samples headroom once. A larger window requires a reload. Per-request admission can reject an allocation when available memory falls. The planning check uses the backend's cold text inventory, and physical memory is allocated when requests execute. This run covers correctness and lifecycle at the tested lengths.
