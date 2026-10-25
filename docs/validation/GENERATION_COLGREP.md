# ColGrep generation validation

This record covers the functional acceptance of `stark-generate` on the `lightonai/colgrep-agent-2B` checkpoint and explains how to reproduce it. It is bound to the exact checkpoint revision, converted artifact and worker build named in the linked reports; a different build needs its own run.

## Result

- All four reference cases passed exact tokenization, decoding, EOS handling and valid XML tool calls on the named Apple M5 Max build.
- Ten native controls passed: seed repeat, output and context stops, custom stops, invalid IDs, recovery, restart and artifact stability.
- The generated agent loop completed search, read and spontaneous finish in three turns, with verified calculation and application lines.
- Three of four greedy outputs match the BF16 PyTorch/Transformers reference token for token. The unforced finish case diverges after 116 shared tokens (171 native tokens versus 173 reference tokens) with the same valid finish locations and line coverage. The strict [reference report](../../provenance/COLGREP_GENERATION_VALIDATION.json) therefore records `passed: false`; the [functional disposition](../../provenance/COLGREP_GENERATION_ACCEPTANCE.json), [generated loop](../../provenance/COLGREP_AGENT_FIXTURE.json) and [numerical diagnosis](../../provenance/COLGREP_GENERATION_DIAGNOSIS.json) hold the separate results.
- At the divergence, a fresh native run chooses the same next token as the reference, and native scoring agrees with the teacher-forced reference for the two compared candidates. Cached versus full-prefix execution changes the reference logits, and the differing choice has a small margin. The difference depends on execution history; the exact kernel cause is unresolved.

Native validation used 4,096-token contexts with output limits of 256 (reference cases) and 512 (generated loop). Longer contexts, stochastic sampling, other checkpoints and the live ColGrep index were not exercised.

## Prepare the checkpoint

Use an Apple silicon Mac, Rust 1.98.1 and Xcode's Metal tools. Build the worker:

```sh
cargo build --release --locked -p stark-generate
```

Download the pinned Hugging Face source, keeping the `.cache/huggingface/download` metadata that the harness uses to verify each file:

```sh
uv run --project tools/converter --locked python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('lightonai/colgrep-agent-2B', revision='f798a61129c6decb5166864e0a3c98373e62c764', local_dir='runtime/colgrep-agent-2B-source', allow_patterns=['*.json', '*.jinja', '*.safetensors', 'README.md'])"
```

The checkpoint is Apache-2.0 licensed; preserve its attribution when distributing weights ([model card](https://huggingface.co/lightonai/colgrep-agent-2B/blob/f798a61129c6decb5166864e0a3c98373e62c764/README.md)).

Convert it:

```sh
uv run --project tools/converter --locked stark-convert runtime/colgrep-agent-2B-source \
  --metadata-profile hf-generic --offline \
  --write-source-manifest runtime/converter-sources/colgrep-agent-2B.json \
  --output models/colgrep-agent-2B-stark
```

The application must render the exact upstream MiniCPM5 XML chat template with the upstream tools and `enable_thinking=False`, then pass the complete rendered string, including the generation prefix, as `rendered_prompt`. The checkpoint declares two EOS IDs, `1` and `130073`; the worker honours both.

## Replay

The replay needs the source snapshot, the converted model and the upstream agent checkout at the harness revision:

```sh
git clone https://github.com/lightonai/next-plaid.git runtime/colgrep-agent-upstream
git -C runtime/colgrep-agent-upstream checkout --detach c6c091d7ee14c81206bb28af90d8ff506fd0235e
```

Strict reference comparison (exits 1 for the documented finish-token divergence):

```sh
uv run --project tools --locked python scripts/validate_colgrep_generation.py \
  --source runtime/colgrep-agent-2B-source \
  --model models/colgrep-agent-2B-stark \
  --worker target/release/stark-generate \
  --upstream runtime/colgrep-agent-upstream \
  --device mps \
  --report provenance/COLGREP_GENERATION_VALIDATION.json
```

Generated agent loop:

```sh
uv run --project tools --locked python scripts/validate_colgrep_agent_fixture.py \
  --source runtime/colgrep-agent-2B-source \
  --model models/colgrep-agent-2B-stark \
  --worker target/release/stark-generate \
  --upstream runtime/colgrep-agent-upstream \
  --reference-report provenance/COLGREP_GENERATION_VALIDATION.json \
  --report provenance/COLGREP_AGENT_FIXTURE.json
```

Numerical diagnosis, using the default scoring build its receipt names:

```sh
cargo build --release --locked -p stark-score
uv run --project tools --locked python scripts/diagnose_colgrep_generation.py \
  --source runtime/colgrep-agent-2B-source \
  --model models/colgrep-agent-2B-stark \
  --worker target/release/stark-generate \
  --score-worker target/release/stark-score \
  --upstream runtime/colgrep-agent-upstream \
  --reference-report provenance/COLGREP_GENERATION_VALIDATION.json \
  --report provenance/COLGREP_GENERATION_DIAGNOSIS.json
```

The harness compares BF16 greedy generation against the PyTorch/Transformers reference and checks tokenization, decoded text, stops, XML calls and file and line-range output. Tool observations are controlled fixtures and model-proposed shell commands are never executed. The reports hold aggregate results and hashes only; prompts, generated text, token vectors and weights stay private.
