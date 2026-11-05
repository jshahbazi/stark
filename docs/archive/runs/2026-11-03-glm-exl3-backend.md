# GLM EXL3 backend implementation and qualification

Date: 2026-11-03

Source parent: `10ec1f31` plus the source bundle below.

Status: optional backend implemented; model-bound qualification and throughput
measurement are pending the pinned artifact download. Prefix reuse remains off
in ordinary builds. No result in this record establishes a prefill rate.

## Source and artifact

The implementation uses the matched public mlx-serve/mlx-stream source pair
recorded in [the bridge provenance](../../../vendor/glm-exl3-bridge/UPSTREAM.md)
and its source lock. The direct module bridge includes serial execution,
complete F32 readout, bounded phase admission, drain/cancellation and an
experimental 16,384-token checkpoint/restore path. It does not run the upstream
server or its sampler wrappers.

The selected artifact is
[`OpensourceWTF/GLM-5.3-Flash-streaming-repack-exl3-3.05bpw`](https://huggingface.co/OpensourceWTF/GLM-5.3-Flash-streaming-repack-exl3-3.05bpw/tree/50237d5c26ceb98bfd85f779990b0f5e6d7af144),
revision `50237d5c26ceb98bfd85f779990b0f5e6d7af144`. The existing GGUF artifact
and its numerical results are preserved separately. The precision policy uses
the pack's EXL3 experts and FP8/BF16 residents, F32 KDA checkpoint state, BF16 DSA
open tails and full F32 host readouts. This does not qualify another pack or
precision path.

## Implementation checks

Both final native stages passed the model-free C ABI admission checks. The Rust
feature checks passed five admission/ABI/selection tests. The generation worker
passed 112 tests with two ignored, and six content-free process checks passed.
Clippy passed with warnings denied for the worker's targets and the validation
example. Repository policy and formatting checks passed.

Independent source review found and resolved four integration problems: the
worker's optional MTP preload hint incorrectly blocked serial loads; staged
upstream changes could escape source binding; the final worker needed its own
runtime library path; and outer token admission bypassed snapshot invalidation.
Native request rejection was also changed to invalidate retained state. Review
did not execute a model or qualify numerical equality.

The sealed workers launch without `DYLD_LIBRARY_PATH`; the final binary's
runtime path points to the validated MLX stage. The private build receipt
SHA-256 is
`8baac06e2cef30775941755fcd4a8ba6981395429820a629dc4d8f2842e40fc0`.
It binds source bundle
`97f1070bd6f9d197f3111c740a1245a938959f845b9f7e7fcf2642c0501f3f4c`
and these Rust 1.98.1 release executables:

| Executable | SHA-256 |
| --- | --- |
| serial worker | `a3a0b2d713cc079725614644dada57a149c270eeb60584e9724acfb2279fb1e0` |
| prefix-validation worker | `370c609919190dbb69a211df4a69a9a9847c95edcaee22d1fb276eb51f18aabf` |
| state-validation example | `c68c0cb86cbee7ee82d7481c60189dd4eb212b579cac12da8fb6a041ac44453b` |

The experimental GGUF MLA admission helper also replaces four checked-multiply
and maximum-value fallbacks with equivalent saturating multiplication to pass
Clippy. Its prior model-bound receipts remain bound to their original source.

Final native stages v5 remove one extra C benchmark EOF newline. Their build
receipt is
`865ee15a8a319215949f181a38b0ea6ba3f5a2c381684fc4f1bd64d86ce85750`,
binding source bundle
`919d61ff301e15f30faf049c685b7f08bb9a33426cabacab35f14d2f0bd7de96`.
Both stages passed the C ABI checks again. All three sealed Rust executables
are byte-identical to the table above; the prior build receipt and stages are
preserved. Artifact qualification uses these final stages.

## Remaining artifact checks

The prefix validator now supports the raw pack through an explicit verified
download receipt and an in-memory native template adapter. Its schema v4 binds
both expert banks, all resident shards, manifests and other flat-pack metadata;
the existing prepared-model route remains available. Twelve focused tests
passed. A separate metadata-only component check verified the five available
chat/tokenizer files and rendered synthetic sessions with lengths 18,185,
18,808, 19,200 and 20,423 tokens. This check loaded no inference model and did
not declare the incomplete pack verified. The shipped template SHA-256 is
`34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891`;
the native helper is
`ec61d32f636d3e7a4a679f48a8dbdb6d2330801f876f6d2f73c03b0b34cb230b`.

Verify all pack files against the pinned revision before loading them. Then
compare full F32 readouts and complete logical KDA/DSA state digests between
fresh and restored changed-suffix requests, including continuation and failed
request invalidation. Run the shared worker validator against the serial and
prefix-validation workers for scope/context/options changes, consecutive
requests, cancellation and restart.

Measure uncached 16K prefill separately with one excluded warm-up and at least
two screened runs. Record effective admitted prompt/decode expert rows, actual
input/output lengths, precision policy, MTP state, source/executable/model
hashes and host observations. The public historical binary and prompt fixture
are unavailable, so the run must identify its own workload and source tuple.

See [the backend reference](../../validation/GLM_EXL3_BACKEND.md) for the current
build, admission and reuse contract. Private artifacts, prompts, token vectors,
responses and state payloads remain outside this record.
