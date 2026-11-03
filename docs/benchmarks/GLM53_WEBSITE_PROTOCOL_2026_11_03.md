# GLM-5.3 website comparison protocol

Date: 2026-11-03. This is a protocol recorded before measurement, not a result.

## Objective and identities

Compare the opt-in Stark GLM profile with the installed MLX-Serve using the
same GLM-5.3-Flash UD-Q2_K_XL GGUF payloads, input token IDs, context budget,
and requested greedy output count. Report prefill, useful decode, and complete
request wall time separately. This study is separate from PR #71's 96/8
MMA-off/on experiment.

The native worker is the sealed Rust 1.98.1 generator built from
`cbb66f481c75e16ed8c076290392bcad35df29be`, SHA-256
`29163c3a12d88cc78003b252903c906c7408ee8b1817bc74d0735d6b5c5e769b`.
PR #71 head `148792c0bc0b6321f7580c031052d8e61906e712` has identical engine,
worker, converter runtime, dependency, and wrapper source. The new benchmark
helpers are separately source-bound; they do not change that binary's identity.
The complete opt-in wrapper is used with T8 MMA enabled, F32 activations and
accumulation, and its existing memory admission and scratch caps.

The installed external executable reports MLX-Serve 26.10.2, MLX 0.32.3,
llama.cpp b11429, and SHA-256
`0c7029c3748cd1c86939483903249e3a35ade14eaaa7d41dfcec4ad9823b9b62`.
It will run the embedded `llama` backend. This is not native MLX execution.
It differs from the historical 26.10.3 GLM control, whose identity and results
remain unchanged. No installed application is upgraded for this comparison.

Weights come from `unsloth/GLM-5.3-Flash-GGUF` revision
`a38483c8cd5df544f53d70fb281afe97369d5ab6`, with metadata/tokenizer from
`zai-org/GLM-5.3-Flash` revision
`04c4e9e95c5da8862dced7e5056455116f83a7e0`. The existing full-hash receipts
and current size, modification-time, and inode guards bind the four payloads
before and after each block. Payloads are not requantized or downloaded again.

## Workloads and order

| Case | Input tokens | Requested output tokens | Context tokens |
| --- | ---: | ---: | ---: |
| Decode-focused | 128 | 256 | 384 |
| Balanced | 1024 | 256 | 1280 |
| Long prompt | 3840 | 256 | 4096 |

The context limit is 4096 for this profile. The long case reserves room for
all outputs within that limit; it does not use the earlier Qwen 4096-input
workload.

Use frozen bytes of `docs/NATIVE_HYBRID_MODELS.md` from PR #71 head
`148792c0`. Encode without added special tokens, repeat the token sequence if
needed, and take the exact requested prefix. Both engines receive direct token
IDs. Corpus, tokenizer, and input hashes must match before comparing rates.
Measurement receipts contain corpus and input hashes and counts only.

Run cases in the table's order. For each case, run four independent load blocks
in Stark / MLX-Serve / MLX-Serve / Stark order. Each block has one excluded
warmup and five accepted measured requests. This gives ten measured requests
per engine per case, with two independent model loads per engine. Model loading,
warmup, and quiet-admission time are excluded from request metrics.

Admission pilots may check feasibility and fixed output counts before the
study. Their results stay separately labeled and never enter study statistics.
Any change to a workload or protocol after a pilot is recorded before the
replacement measurements; failed pilots and exclusions are retained.

## Controls and acceptance

- Before timing, extend the existing next-token scorer check to exact corpus
  prefixes of 128, 1024, and 3840 tokens at scoring context 4096, retaining the
  eight short rendered controls and sixteen candidates per context. Compare
  the sealed F32 reference and candidate sequentially, with the existing
  maximum absolute normalized log-probability delta threshold of 0.001. This
  checks 176 points; it does not establish whole-model or answer-quality
  equivalence.
- Greedy sampling, temperature zero, no speculation, one request at a time.
- No prefix reuse: native requests omit cache scope and reject reported reused
  input tokens; external prefix and tokenize caches are disabled and cached
  prompt tokens must be zero.
- Each measured request must emit the full requested output count. Early EOS
  is a feasibility failure for this fixed-output comparison. Record actual
  counts and finish reasons, including failures.
- Use the existing unchanged quiet-admission and per-request host-screen
  policy. Each accepted measured request requires zero pageouts and swapouts,
  bounded background CPU/model activity, valid monitor samples, acceptable
  thermal state, and no compiler activity. Warmups do not qualify as measured
  requests and may be screen-ineligible.
- Bound each block at eighteen attempts, including its warmup. Retain excluded
  attempts. A block that cannot supply five eligible measured requests is
  incomplete; it supplies no headline comparison.
- Require exact repeated output hashes within each engine and case, matching
  source/artifact/configuration pins before and after, and verified native
  process shutdown. Cross-engine output or quality equivalence is not assumed.
- Serialize all native and external model runs. Do not compile, run unrelated
  GPU checks, alter resource caps, or manipulate user windows during timing.

## Statistics and publication

Primary statistics are medians over the ten accepted measured requests per
engine and case. Report prefill tokens/s, useful post-first-token decode
tokens/s, and request wall seconds separately. For full 256-token outputs,
useful decode has numerator 255. Preserve both engines' timer definitions;
native stdio and external HTTP wall times include their respective transport.
Do not subtract estimated unused forwards or other unmeasured overhead.

Retain per-block medians, ranges, maximum decode rates, and pooled rates as
secondary statistics. Do not select a maximum as the primary result after
seeing timings. Do not combine cases into a weighted overall score unless a
separate weighting is recorded before measurement.

Publish source-bound aggregate results and hashes only. Include hardware,
versions, quantization, precision policies, effective external settings,
accepted/excluded counts, output-count/finish checks, and identity guards.
Preserve earlier GLM and Qwen results separately. A throughput result does not
establish answer quality, long-context stability, or a universal performance
floor. Website copy must identify the opt-in profile and embedded llama.cpp
backend explicitly.
