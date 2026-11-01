# Qwen3 long-prompt prefill candidate — 2026-11-01

Status: **qualification incomplete**. This branch implements a bounded weight
staging candidate. It has no accepted speed result and has not replaced the
installed application. The user deferred a reserved timing window while other
work continued. This record does not supersede the existing
[OpenWork first-text measurement](../archive/runs/2026-10-31-openwork-first-text-latency.md).

## Source, model and workload

Baseline: `36e855ee5bca4ce8c1bd585173bad44cac04b7a1`. Candidate:
`codex/openwork-long-prompt-prefill`, native source-tree SHA-256
`fe9c4736fa37c40a5b09359a05c1ae97368f067ae0474805aa5efed650f5151b`.
The measurement controller hashes tracked native sources, the lockfile,
toolchain and vendor tree; build records separately bind private probes.

Host: Apple M5 Max, 128 GB, macOS 26.7.1 (25G241). Builds use Rust 1.98.1 and
locked Cargo dependencies. Engine tests and formatting use
`nightly-2026-10-01`; Python uses locked uv dependencies and Python 3.13.13.

| Artifact | Identity / SHA-256 |
| --- | --- |
| Model | `mlx-community/Qwen3-4B-4bit`, revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` |
| Quantization | MLX affine unsigned 4-bit, group 64, BF16 coefficients |
| Weights | `7c2af9a45e5b0d196f06a5cbfd6fc5cc95adf29df79885d0540f2206fb3d40d8` |
| Configuration | `5b505594bf00ef9682b373c6a292b88f94f1e1ceab449784c12accb835806c7f` |
| Tokenizer | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| Baseline generation worker | `facebd20b6a9b1242b16e2fb4b5c64e1971f99a98d5b80f9a14f8b83601cad81` |
| Candidate generation worker | `e4f679005ecc1a78f319fd413abc5ecd461be7e9a43092fd1fac89cba9aae83b` |
| Baseline private logit probe | `ff8b4bc486c6b5835fec2b864dd8628ca96d68e50376d148f15fab6484753193` |
| Candidate private logit probe | `e48a66d0d1269be2525f1be2538f1e06c7ee1d6dd278d770b7da3a522d2bcd24` |

The full synthetic workload contains **18,544 tokens, eight complete function
definitions, five messages and one linked tool result**. The pinned model's real
chat template renders it. Its rendered SHA-256 is
`319ac0c792c5c10920563d8f9a2cc7a8dfebe902dabe50b47919440669d077a4`.
A 49-token chat is a separate control. No tool definition or history message is
removed, summarized or compressed. Every request starts fresh native state;
this change adds no prompt, prefix or conversation cache.

## Candidate and precision policy

For selected large BF16-activation, unsigned affine Q4/G64 matrix products on
MXU-capable Metal devices, the engine dequantizes the current immutable weight
matrix into a temporary BF16 buffer, then uses the existing dense GEMM. The
buffer is overwritten for the next matrix. Stored weights and quantization are
unchanged. Existing A8 activation routes, including the eligible MLP down
projection, retain their original implementation.

The route preserves the original 128×128 tile, split-K count, partition
boundaries and ordered K32/F32 accumulation. It requires whole K256 partitions,
at least 512 input rows, contiguous transposed weights, and an expanded matrix
of at most 128 MiB. Activation offsets must fit the dense kernel's 32-bit index
arithmetic. Other shapes, dtypes, quantization formats, gathered operations and
RHT operations retain their existing routes.

One fixed 128 MiB scratch bucket is reused within a request. Metal scoring
shares its request pool across batches; its existing device-visible queue
barriers order GPU-only scratch reuse. Per-command-buffer CPU constants stay
separate. A safe-default backend capability leaves CPU and other backends on
independent batch pools because CPU scratch allocation clears pages on the
host. Submitted work is still drained on failure before state is released.

Independent review found and resolved excess per-batch workspace growth, a
large activation-offset overflow, and unsafe generic CPU pool sharing before
qualification. Tests cover actual target dimensions, both coefficient layouts,
ragged dimensions, workspace overwrites within/across submissions, bounds and
CPU pool isolation.

## Evidence and remaining concern

- Full-vocabulary comparisons match **all 151,936 vocabulary logits bit-for-bit
  in each of seven cases**, including complete 18,544/18,545-token inputs and diagnostic
  1,023/1,024/1,025-token boundaries. Repeated scores match within each build.
  Boundary prefixes are numerical diagnostics; the complete workload remains
  intact for generation and timing.
- Focused GPU comparisons match every output bit for target QKV, attention
  output and MLP matrix dimensions. Same-submission and queued cross-submission
  workspace reuse comparisons pass. An initial pure shape test had incorrect
  hard-coded partition expectations; it now checks the required boundary and
  passes without changing runtime behavior.
- One initial complete generation run reported differences in long greedy,
  text-stop and sampled continuations. Short cases, token stops, cancellation,
  context overflow and recovery passed. That controller retained booleans,
  without the differing output or its first mismatch position.
- Fresh-worker repeat controls, the complete short-request prefix followed by
  repeated long requests, and a full rerun with component diagnostics all
  match. A baseline-versus-baseline full control also matches. These reruns do
  **not** explain or erase the initial failure.
- A final bounded CPU-load replay matches all ten generation/stop cases, two
  within-build long-generation repeat comparisons, and both workers' overflow,
  cancellation and recovery controls. Every comparison overlaps all eight
  compute-only load children. Load lasts 116.58 seconds; all eight children are
  terminated and reaped, with no forced kill. This is correctness stress,
  without accepted timing or an explanation for the earlier mismatch.

Native generation is therefore not unconditionally qualified. Do not install
this candidate or describe it as a verified speedup while that concern remains.
Successful numerical comparisons do not establish general agent competence.

The earlier grouped-attention and compact-tile trials remain exploratory;
startup drift dominated their small apparent gains. Their source-bound private
artifacts are preserved. A guarded run rejected its host with the sanitized
category `quiet_window_timeout`; no failed or unguarded timing is accepted as
a speed result.

## Completed local checks

| Check | Result |
| --- | --- |
| Full model-free converter suite | 5,299 passed, 88 skipped |
| Measurement/numerical controller tests | 32 passed; Ruff check and formatting pass |
| Native Metal engine suite | 682 passed, 48 ignored; seven C++ string tests and three doctests pass |
| CPU/grammar engine suite, no default features | 432 passed, 20 ignored; seven C++ string tests and three doctests pass |
| Generation worker | Release build, all-target Clippy with warnings denied, 92 tests passed / two ignored; no-default configuration 56 passed |
| Scoring worker | Separate default and optimized release builds, all-target Clippy with warnings denied, both synthetic test configurations pass |
| Worker process harnesses | Six generation cases and eight scoring cleanup/isolation checks pass |
| App builder/updater/version model-free tests | 82 passed |
| Repository policy / dependency gate | Policy passes; 41 gate tests pass; converter lock check passes |
| Rust formatting | Pinned nightly check passes |

These local checks do not replace the unresolved native-generation
qualification or a real client run. Hosted CI/security gates were not run;
the branch has no pull request, and pushing to it does not trigger the
repository's push checks.

## Reproduction and privacy

Run from the isolated checkout with a frozen model and separately sealed
baseline/candidate workers. Both loaded workers process requests serially;
timing uses repeated ABBA order. The controller rejects changed binaries,
model/render sources or helper implementations, background model work, thermal
or CPU contamination, paging, monitor failures and sleep. Exploratory or
parity-only runs keep `timing_accepted=false`.

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false uv run --project tools/converter --locked --python 3.13.13 python scripts/benchmark_native_prefill.py --model "$PREFILL_MODEL" --baseline "$PREFILL_BASELINE" --candidate "$PREFILL_CANDIDATE" --output runtime/prefill-guarded.json --mode all --cycles 3 --tokens 32
uv run --project tools/converter --locked --python 3.13.13 python scripts/validate_native_prefill_logits.py compare --model "$PREFILL_MODEL" --baseline "$PREFILL_BASELINE_PROBE" --candidate "$PREFILL_CANDIDATE_PROBE" --output runtime/prefill-logits.json --steps 2 --prefix-lengths 1023 1024 1025 --check-repeats
```

Controllers exclusive-create primary aggregate records with mode 0600 under
this checkout's ignored `runtime/`. Reports contain counts, hashes, timings,
booleans and sanitized failure categories. Full logit vectors travel only
through private pipes and stay in RAM. No real prompts, replies, token vectors,
credentials or raw protocol dumps are retained. API authentication, Keychain
storage and the managed OpenWork connector are unchanged.

The installed OpenWork 0.18.63 / OpenCode 2.0.26 qualification belongs to its
[earlier build](../archive/runs/2026-10-31-openwork-managed-connection.md).
The normal builder produced an uninstalled development app, build
`20261101042536158240`, with the candidate worker above. Swift compilation and
deep/strict ad-hoc signature verification pass; host executable SHA-256 is
`d8034ef3ab9987d6c4b2db7a0bf904369719261269211dae3b7792a684404266`.
It was neither launched nor installed. Developer ID signing, packaged upgrade
credential reuse and real OpenWork tool/restart validation remain pending for
this candidate. No compatibility claim transfers to the new worker merely
because protocol tests or logits pass.
