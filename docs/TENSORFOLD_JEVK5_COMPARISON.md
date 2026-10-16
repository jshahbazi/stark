# TensorFold and Stark on JevK5

## Result — 2026-10-16

TensorFold 1.0.12 was installed locally and passed its archive checksum check.
It detects this Apple M5 Max's Metal backend, but cannot load the local
JevK5 Qwen3.5-4B BF16 checkpoint. The existing Stark scoring binary successfully
loaded the converted BF16 checkpoint and exited cleanly on stdin EOF.

This is a compatibility comparison. No comparative inference latency,
throughput, decision quality, memory benchmark, or speculative speedup was
measured. TensorFold's refusal is not a performance result.

| Check | TensorFold 1.0.12 | Existing Stark scoring binary |
| --- | --- | --- |
| Local executable starts | Passed | Passed |
| Local JevK5 BF16 loads | Refused | Readiness reported `ready: true` |
| Exit after load probe | 1, unsupported model type | 0, clean EOF |
| Inference comparison | Blocked by model support | No requests issued |

## Installed setup and provenance

- Host: Apple M5 Max, 128 GB unified memory.
- Executable: `/Users/johnshahbazian/development/local/tensorfold-1.0.12-macos-arm64/bin/tensorfold-native`.
- Archive: `tensorfold-1.0.12-macos-arm64.tar.gz`, downloaded from the
  [upstream v1.0.12 release](https://github.com/ashhart/TensorFold/releases/tag/v1.0.12).
- Archive SHA256: `a45c3775de12c218dab74906b074e0021860ef61c3501a1bad24337bbb32894a`;
  matched the release's companion checksum.
- Executable SHA256: `97abc60e6b5bb4c48f5f58bfbe49f214a86bd44df25e064186a46aff4e0b8ff3`.
- Source checkout: `/Users/johnshahbazian/development/local/TensorFold`,
  inspected at `b090e3b19014bfa1eb2b7f38ea3cc4d39abec9be`.
  The release binary was downloaded, not built from that checkout.
- Older Python source inspected at tag `v0.6.6`,
  commit `cb2ebf0540f42604e2759b2ddef497861e928248`; it was not installed.
- TensorFold's checked source declares Apache-2.0. The linked review's
  MIT/Python setup description does not describe the current release.

Verify the installation without loading weights:

```sh
/Users/johnshahbazian/development/local/tensorfold-1.0.12-macos-arm64/bin/tensorfold-native --version
/Users/johnshahbazian/development/local/tensorfold-1.0.12-macos-arm64/bin/tensorfold-native capabilities --json
```

## Exact model and observed refusals

The local source is `alibiserikbay/JevK5`, revision
`c4f7fdb3aeab5582336406e78d3bef11bf98833d`, merged BF16 safetensors.
Its config declares `Qwen3_5ForCausalLM`, `model_type: qwen3_5_text`,
hidden size 2560 and 32 layers. It is Qwen3.5-4B, not standard Qwen3.
This result is scoped to that checkpoint, not every JevK5 release or quantization.

The original config SHA256 is
`63f47812d0f11118e4d252d2b3ad488707eb9287a11589f4fd382a1d31182724`.
The converted Stark config SHA256 is
`7cd671fd90e9513487f8c6f14e62af6af638add926f55531d19073a381b66561`.
Source revision and model provenance were read from MeeCP's existing
`experiments/uzu-rust/jev-scoring/MODEL_PROVENANCE.json`; model weights
were not rehashed during this check.

The direct load command was:

```sh
/Users/johnshahbazian/development/local/tensorfold-1.0.12-macos-arm64/bin/tensorfold-native serve \
  /Users/johnshahbazian/development/meecp/models/uzu-jevk5/source \
  --name jevk5-comparison --parallel 1 --context 2048 --temperature 0 \
  --no-drafts --prompt-cache-gib 0 --keep-warm 0 \
  --snapshot-dir none --max-snapshots 0
```

It exited with code 1 and reported no native backend for `qwen3_5_text`.

A separate, ignored metadata-only probe wrapped the source config in
`model_type: qwen3_5` with the original fields under `text_config` and linked
the existing tokenizer metadata. No weights were linked or copied, and the
original model was not edited. The same native loader then exited with code 1
and `UnsupportedQwenGeometry`. This probe demonstrates that changing the model
type alone is insufficient; it does not qualify an adapted checkpoint.

The [native config parser](https://github.com/ashhart/TensorFold/blob/b090e3b19014bfa1eb2b7f38ea3cc4d39abec9be/zig/src/families/qwen3_5/config.zig)
pins the 2B route to hidden size 2048 and 24 layers. The separate dense route
targets Qwen3.8-27B. Neither qualifies this 4B checkpoint.

The older Python version also lacks a `qwen3_5_text` family registration.
Its [Qwen loader](https://github.com/ashhart/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/src/tensorfold/families/qwen3_5/__init__.py)
requires MLX affine quantization metadata and rejects unquantized weights.
It is not an as-is fallback for this BF16 checkpoint. This conclusion is from
source inspection, not a Python runtime trial.

## Stark probe and comparison limits

The existing `target/release/stark-score` was run with
`--score-model-path /Users/johnshahbazian/development/meecp/models/uzu-jevk5/bf16`
and immediate stdin EOF. It produced one readiness record, `ready: true`,
no stderr, and exit code 0. Binary SHA256:
`d246d8e5718673a9101c9bfdc151d28be8b47ffa3193353a666cd2a50f7b4095`.
It was not rebuilt or attributed to the current source revision by this probe.
Existing [JevK5 parity evidence](VALIDATION.md) remains a separate,
source-bound validation run.

A meaningful follow-up requires TensorFold support for this architecture,
geometry and storage format. For JevK5 decision scoring, it also needs access
to the same candidate logits and full-vocabulary normalization as Stark;
generated-text throughput alone does not establish equivalent decisions.
Changing quantization would require a separately labelled quality comparison.
No prompts, generated responses, token vectors or decision probabilities were
logged. No model files, binaries or private runtime artifacts are committed.
