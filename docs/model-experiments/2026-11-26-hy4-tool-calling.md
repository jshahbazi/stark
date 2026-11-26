# 20261126-HY4TOOLS-01 · Hy4-flash tool calling

| Field | Value |
| --- | --- |
| Scope | Native template compatibility and bounded Chat Completions tool loop |
| Artifact | `AngelSlim/Hy4-flash-GGUF` at `c50d4dd30be42f2b91138e5731de0e2f32ad56b2`, Q3_K split GGUF, BF16 projector and original NextN |
| Template | `tencent/Hy4-flash` at `eccd3123eec73f2040f4c9ed0de1883593470510`, SHA-256 `11b12531403da5726b7e1f6003f479043b05999e6746d39af949cbcdfd715c35` |
| Host | M5 Max, 128 GiB, macOS 26.7.2 |
| Baseline | Stark 0.24.0 at `cce80236e232481325e62c2009045219829f38a2` |
| Candidate | Developer ID–signed local build from the baseline plus the converter patch, patch SHA-256 `d3d6d5b5e25e6057d18ad463049b38f87ce69e70e1fb2bea432064a91aaeace8` |
| Build identity | App manifest SHA-256 `9d84882cb5a07adfa148cb6661c32dec5fe6adfd2a9568de2e641fef38e1c46b`; reused the release's native workers and verified the signed install |
| Disposition | Adopted on branch and installed locally |

Stark 0.24.0 admits ordinary Hy4 chat. On a request with tools, it returns
`tools_unsupported` before dispatch because it has no parser for the pinned
template. The template renders one outer tool-call block with individual calls
encoded as argument-key/value pairs. The parser uses the reviewed GLM value
decoder and enforces Hy4's own wrapper, delimiters and reasoning boundaries.
Admission is bound to the exact template hash. [Pinned Tencent template](https://huggingface.co/tencent/Hy4-flash/blob/eccd3123eec73f2040f4c9ed0de1883593470510/chat_template.jinja) · [vLLM Hy4 recipe](https://recipes.vllm.ai/tencent/Hy4-flash)

On the signed candidate the existing catalog artifact loaded with the ordinary
desktop settings. The native renderer rendered typed arguments and client tool
results in the pinned format. Two preliminary HTTP harness runs stopped at
incorrect assertions about discovery response shape. The corrected harness
verified discovery and completed three greedy requests.

| Check | Result |
| --- | --- |
| Discovery | `/v1/models` and `/v1/capabilities` advertised tool support and the expected parser |
| Buffered required call, 160-token limit | One call to the requested synthetic tool with the expected integer argument |
| Streamed continuation, 160-token limit | Used the client-supplied tool result in the answer |
| Streamed automatic choice with high reasoning, 512-token limit | Returned the expected structured call, separated reasoning and withheld wire markers |

Parser regression tests cover parallel calls, typed and nested values, partial
frames, disallowed choices, quoted examples and streamed marker boundaries.
The installed app reached Ready and showed Tool use available in Connections.
Long multi-tool agent workflows and model choice quality remain outside this
smoke test. Native performance, prefix restoration and image qualification
are recorded in the [Hy4 performance write-up](speed-hy4.md).
