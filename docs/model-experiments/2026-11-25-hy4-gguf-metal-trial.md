# 20261125-HY4METAL-01 · AngelSlim Q3_K on the published Metal runtimes

## Identity and state

| Field | Value |
| --- | --- |
| Registered and executed | 2026-11-25, Codex |
| Kind | External runtime build, artifact admission and functional compatibility |
| Scope | AngelSlim Q3_K on Apple Silicon. Native Stark qualification remains a separate attempt. |
| State | Artifact verified and full-model smoke completed. Both engines passed text with built-in drafting after the Strata setup repair. |
| Acceptance | Load the two hash-verified shards and BF16 projector. Complete greedy text with exactly 4,096 input tokens on the same fixture, using one loaded runtime at a time. Check the short known-answer case first. |
| Use | External reference for native implementation and differential tests |
| Related work | [Native Stark plan](../MODEL_RESEARCH.md#research-proposals-with-no-stark-timing) · [Experiment ledger](README.md) |

## Artifact, source and environment

Both arms use [AngelSlim Hy4-flash-GGUF at revision `c50d4dd30be42f2b91138e5731de0e2f32ad56b2`](https://huggingface.co/AngelSlim/Hy4-flash-GGUF/tree/c50d4dd30be42f2b91138e5731de0e2f32ad56b2). The three published digests match the earlier planned revision. The main shard contains mixed Q3_K and Q4_K expert weights and the NextN block. The shared Engram shard contains the FP8 E4M3 table. The payload is 156,415,403,072 bytes including the projector.

| File | Published SHA-256 |
| --- | --- |
| `Hy4-flash-Q3_K-00001-of-00002.gguf` | `987b2902813ad94919e3e46e375bd8144d36113cced62d25f13013ccc88a3abb` |
| `Hy4-flash-Q3_K-00002-of-00002.gguf` | `ecd6a0ecd7f258b1ee9211bf566261ae33efc1743ad40f58cb5004cf7b051839` |
| `mmproj-Hy4-flash-bf16.gguf` | `8c33a48ef57480abf4854126964058a2015b5da3b773b14c847c995cf8899bda` |

| Field | llama.cpp reference | Strata candidate |
| --- | --- | --- |
| Source | llama.cpp `2cbca389f515df455412dafa07e2b1359cc72a6d` plus [AngelSlim Mac patch](https://huggingface.co/AngelSlim/Hy4-flash-GGUF/blob/c50d4dd30be42f2b91138e5731de0e2f32ad56b2/patch/hy4-flash-mac-minimal.patch), SHA-256 `27810270f635fa70c1c8a665bc1a35b4352e6f3c5846ddb793098e6706f9db35` | [Strata PR #7120](https://github.com/Niko1221/Strata/pull/7120) at [`20610a5442f713be26625b0f587bd52252225e4c`](https://github.com/Niko1221/Strata/commit/20610a5442f713be26625b0f587bd52252225e4c) |
| Build | CMake Release, Metal with embedded kernels, embedded web UI disabled | CMake Release, Metal engine with local Apple metal-cpp headers |
| Executable SHA-256 | `b9aad7086abc83bb351abdfa3a41e3f8fde73529fbe8c4a9006d2ee23de0b9bf` | `4c8edd347b1c66624f8f73834b241152def0dfacd23d0f9b667df887df8e3cdd` |
| Host | Apple M5 Max, 128 GiB, macOS 26.7.2 (25G332), AC power | Same machine |
| Device policy | System default Metal wired limit | Automatic arena sizing |

Stark source base for this record is `e0e6916d8e966597ed904c36f01068ad70bac28d`. Runtime sources, model payloads and trial scripts are in ignored local storage.

## Workload and controls

- One runtime and one request at a time, 8,192 context positions. Greedy decode at temperature zero and seed seven, with drafting off for the first pass.
- A short synthetic known-answer completion and a fixed 4,096-token text fixture. Both arms receive the same token IDs. The 4K fixture exercises sparse attention past the 2,048-position selection threshold. The requested decode lengths are eight and 64 tokens.
- Use an image-loaded worker. Hash output token IDs in memory. Retain only aggregate counts, hashes, timings and typed errors. No prefix snapshots from a prior request enter the comparison.
- A separate built-in NextN trial uses up to two drafts per round. The llama.cpp image screen uses two synthetic color swatches. This is a smoke workload with a single observation per case.

## Attempt history

| Trial | Observation | Disposition |
| --- | --- | --- |
| llama.cpp Release build with default prebuilt web UI | Asset provisioning failed because the fetched bundle lacked its hashed JS, CSS and worker files. | Rebuilt with embedded web UI disabled. The Metal bench and API server targets compiled. |
| Strata Metal build | The server and standalone Metal driver compiled at the pinned commit. | The C++ source includes the exact mixed Q3_K/Q4_K expert formats and FP8 Engram rows. Runtime admission requires a prepared pack. |
| Strata pack path | The publisher guide names `strata_pack.py`, whose pinned implementation targets the Qwen Q2_0 pack. The pinned launcher invokes `iq_pack.py` and limits its Hy4 model selection to the Q4_K_M route. | Use the native server protocol with the Engram token map extracted by the pinned packer. The engine reads weight tensors directly from GGUF. |
| Strata model-free tests and header scan | The Hy4 geometry fixture and Metal host-side expert tile test passed. The pinned parser accepted the mixed Q3_K/Q4_K expert layout, the BF16 draft layer and the 120,832-entry Engram map from the shard headers. | Structural evidence. |
| Shared artifact transfer | The original Xet process stopped advancing when the host slept. Two sample reads at the end of the retained prefixes matched the pinned publisher URLs. | Continue by HTTP range from the retained prefixes. Require all three full SHA-256 digests before running inference. |
| Download recovery | The resumed Xet stream and a fresh current-client Xet download failed to advance in the observed retry windows. HTTP range recovery advanced after restart, with approximately 45 GiB received out of 145.67 GiB at the checkpoint. | The local runner proceeds from transfer to full digest verification and then to the serialized smoke tests. |
| Final transfer restart | The range worker stopped with 7.69 GiB left. A restart reused all completed ranges and finished the transfer at 16:22 EST. All three files matched the published sizes and digests. | One verified payload shared by both runners. |
| Strata with built-in drafting | The initial process exited with status 2 before readiness. An independent startup probe reproduced the failure to create `derived/model/mtp_experts_q8_0.bin.part` because its parent directory was missing. | The direct launcher needs the writable output directory. See the follow-up below. |

## Correctness, performance and decision

With the projector loaded, llama.cpp and Strata generated identical token sequences on both greedy text fixtures with drafting off. llama.cpp reproduced both token sequences with drafting on, accepted 41 of 42 drafts on the 4K request and passed both image color checks. Strata image input was outside this trial.

| Runtime and draft mode | 4K prompt time | 64-token decode time | Greedy output |
| --- | ---: | ---: | --- |
| llama.cpp, off | 7.344 s | 1.692 s | 64 tokens |
| llama.cpp, depth two | 7.428 s | 1.012 s | Exact token hash match |
| Strata, off | 22.410 s | 4.411 s | Exact token hash match |

These are smoke timings from one run per arm with a repeated synthetic fixture. The llama.cpp server used a 256-row microbatch and Strata used a 512-row prefill chunk. The run supplies no production throughput comparison, memory envelope, prefix reuse, tool, sampled generation or native Stark qualification. The low-level llama.cpp stderr classifier matched the word `truncated`; both responses passed the harness check for full requested output without the API truncation flag.

Private artifact and smoke aggregate receipt SHA-256 values are `53b25c7b5bebe4362fac82ee7f5bc0df2776913138fb4133f2c33a4594ea8135` and `7c7313ca58345fe779de989f057f785e1e247dbe2ccc3f9a5f6b5e449ef33285`. The trial harness SHA-256 is `5d9373b16222a553c64395a14f9c3ec035542bb628041e6cf1c0aa50c116030b`.

## 20261125-HY4METAL-02 · Strata draft startup repair

The direct launcher computes the derived expert path from the GGUF directory. Created its parent `derived/model` and reran the four arms with the same executables, payload, context and harness. The Strata draft worker generated its 3,850,375,168-byte Q8_0 expert file and reached readiness with the projector loaded. Its derived file SHA-256 is `9c56e5ca221668726c7dda47da71389b5637dfb21aeca6c6a7653ab56de21b68`.

All four workers exited cleanly. An independent check of the receipt found matching input and output token hashes across all eight text cases, with positive draft activity on both engines when enabled. Both llama.cpp image checks passed again. Strata accepted 41 of 43 proposals on the 4K fixture. llama.cpp accepted 41 of 42.

| Runtime and draft mode | 4K prompt time | 64-token decode time |
| --- | ---: | ---: |
| llama.cpp, off | 7.047 s | 1.721 s |
| llama.cpp, depth two | 7.332 s | 1.049 s |
| Strata, off | 23.028 s | 4.588 s |
| Strata, depth two | 23.432 s | 2.477 s |

This establishes external greedy text smoke parity for the pinned Q3_K artifact, with the projector loaded and drafting off or on. Scope and timing limitations follow the first attempt. Strata image requests, sampled behavior and native Stark support remain unqualified. The rerun aggregate receipt SHA-256 is `990b4eb5ee10f21e9f59dae1b9221a47920640600ac93428c5927f497af00e09`.
