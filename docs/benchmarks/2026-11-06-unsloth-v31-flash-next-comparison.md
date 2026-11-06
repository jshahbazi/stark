# Unsloth Dynamic V3.1 Qwen3.8-Flash-Next comparison

Date: 2026-11-06. Status: **Stark baseline recorded; Unsloth GGUF benchmark
pending download and execution.** This report does not name a cross-engine
winner.

The proposed comparison is between the Unsloth Dynamic V3.1 Qwen3.8-Flash-Next
UD-Q4_K_XL GGUF with its shared Q8_0 MTP module and Stark's installed Lily Q4-XL
artifact. These are different weight
formats and quantization profiles, so an eventual result will compare these
two delivered configurations; it will not isolate engine efficiency. The
Unsloth download was still pending when this baseline was prepared. No llama.cpp
model run is included.

Unsloth's post claims 50% fewer tokens and 15% higher accuracy against
same-size quantizations. Those claims have not been tested here. See the
[announcement](https://x.com/unslothai/status/2118707871139696911?s=46),
[Qwen3.8 Next guide](https://unsloth.ai/docs/models/qwen3.8-next), and
[Unsloth GGUF repository](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).
The pinned llama.cpp fork is
[`danielhanchen/llama.cpp` at `6fcaa16`](https://github.com/danielhanchen/llama.cpp/commit/6fcaa16f4b360649933a54d1f91ad40ed35c0e11).

## Stark throughput baseline

Both routes used the same fresh isolated Stark generator binary and the same
4,096-token synthetic prompt, with a 512-token generation budget and 16,384
context. Sampling was temperature 1, top-p 0.95, top-k 20, seed 42, with
`xhigh` reasoning enabled. Prefix reuse was disabled by omitting `cache_scope`.
Each route had one warmup followed by three measured requests. The off route
ran before the on route; requests were serialized, not randomized or paired.
The off route's initial, separate three-token screen is excluded from these
aggregates.

| Route | Measured requests | Pooled decode tok/s | Range across requests | MTP acknowledgement |
| --- | ---: | ---: | ---: | --- |
| MTP off | 3 | 70.21 | 69.85–70.86 | disabled, 0 drafts |
| MTP on | 3 | 82.96 | 82.70–83.17 | enabled, 2 drafts |

The pooled difference is **+18.16%** for MTP on in this short timing window.
One additional same-workload MTP activity record reported 198 rounds, 394
proposed tokens, 313 accepted tokens and 592 verified rows. That record confirms
draft activity for that request; it does not add another throughput sample.

This short run is not the source of the previously qualified 82.85 tok/s
Qwen3.8 MTP result. The October 31 benchmark used 4,096 input and 256 sampled
output tokens across 16 measured pairs with randomized balanced ABBA order.
Its setup and conditions differ; that record remains unchanged. See the
[October 31 MTP report](QWEN38_MTP_COMBINATION_2026_10_31.md).

## Separate task pilot

The separate native task pilot used six single-seed tasks per route, `xhigh`
reasoning and a 4,096-token output cap. One task completed with the expected
answer on each route; the other five reached the cap before completion. The
off and on aggregates came from separate pilot reports, not a paired quality
study. This small pilot does not establish answer-quality equivalence or test
Unsloth's accuracy claim. A benchmark report's successful run status means the
harness completed; it does not mean every task was answered correctly.

## Artifact and build identity

The Stark checkpoint is `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`, Lily Q4-XL, with 55 weight shards
totaling 106,905,155,192 bytes. A private full-weight verification receipt
passed for all 55 shards. Its SHA-256 is
`cc274c12d40e0869cf1bdf326ae1a5650dcd7e9629269d360ab2f2f987a4dc8a`.
The public aggregate retains the receipt hash and summary, not the per-shard
hashes.

The Stark engine source is commit
`e77f2f00bea7c7984aa35b7cca48f4460b79ac75`. The fresh isolated generator was
built with Rust 1.98.1 (`48a229cea`, 2026-09-01) and locked dependencies. Its
SHA-256 is
`0e61789afa9aa4e12c7064414f75ca02d5bd3b211a34b4ce483c2ced3595063d`.
The shared harness SHA-256 is
`064be4bd1231d2259fa3af0af54010ca338a155407e41e5ccfbb58665355d42b` for the
throughput reports. No diagnostic selector overrides were deliberately set;
this record does not claim a complete environment-selector audit.

The planned Unsloth artifact is pinned to revision
`d96201194f103c906994f283f504209dc8d2a10c`, with a reported five-file,
114,121,223,034-byte inventory: 111,334,654,778 bytes for the four base shards
and 2,786,568,256 bytes for the shared Q8_0 MTP module. The payload download remained pending, so that
inventory has not been locally verified and no Unsloth binary or throughput
result is reported.

## Host and limits

The host was an Apple M5 Max with 40 GPU cores and 128 GiB memory, on AC power
with `pmset powermode 2`. No thermal warning was recorded. GPU tests were
serialized. Transfers were stopped during the corrected throughput and MTP
activity measurements. Transfer probes overlapped the excluded initial screen;
task-pilot timings are not used for performance comparisons here. The request counts
and single timing window are too small to characterize run-to-run or sustained
performance.

The converter currently does not support importing this Qwen3.8 GGUF into
Stark. The planned comparison therefore uses llama.cpp for Unsloth and Stark
for Lily. Even after download and execution, this would remain a comparison of
two different artifacts and quantizations, not proof of whole-model quality or
numerical equivalence. The top-level HTTP API switch is `"mtp": true` or
`false`; direct Stark worker benchmark requests use `"mtp_drafts": 2` or `0`.

Machine-readable public aggregates are in
[`unsloth-v31-flash-next-2026-11-06.json`](data/unsloth-v31-flash-next-2026-11-06.json).
