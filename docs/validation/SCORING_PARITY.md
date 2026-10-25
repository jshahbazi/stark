# Scoring parity validation

This record covers the native scoring worker's parity against frozen Uzu scorer references on the JevK5 evaluation set. It is bound to source commit `97f43755dd833b9c33f35db19bf859057ee0830c`, built on 2026-10-11 with Rust 1.98.1 and the `optimized` feature; a different build needs its own run.

## Result

- All 243 JevK5 BF16 evaluation cases passed native parity: the 486 cold and cache-hit replies and their typed decisions matched the references exactly, including raw logits, log probabilities and calibrated candidate probabilities. All 243 prompt tokenizations matched.
- All 48 calibration boundary requests (96 replies) and 534 full-vocabulary digest requests (1,068 replies) matched bit for bit. The worker made 291 native scoring calls and 291 cache hits; the digest probe made 534 and 534.
- Controls passed for candidate, order, temperature and context changes, cache-off scoring, errors, clean EOF, and restart with a cold cache. Both binaries passed their Rust tests. Eight model-free process-ownership checks passed, covering cancellation, concurrent launch rejection, execution failure and ownership-loss reporting.

The [aggregate report](../../provenance/SCORING_VALIDATION.json) and [build receipt](../../provenance/BUILD_RECEIPT.json) identify the source commit, dependency lock, executable hashes, tooling, portable fixture and private result seals. No timing was measured in this run.

## Reproduce

Model-free regression tests:

```sh
cargo +1.98.1 test --release --locked --offline -p stark-score --features optimized -- --test-threads=1
```

The parity validator needs a private reference directory, the process reaper and a converted model. Reference fixtures and raw results stay private and out of Git. Compile the reaper:

```sh
mkdir -p runtime
clang -std=c11 -O2 -Wall -Wextra -Werror scripts/stark-validation-reap.c -o runtime/stark-validation-reap
```

Check process ownership without a model:

```sh
uv run --python 3.12.13 python scripts/test_scoring_process.py --reaper runtime/stark-validation-reap
```

Export a portable fixture from a sealed scoring run (the source directory supplies `configuration.json`, `dataset.json`, pre- and post-run seals, four complete scoring records and two vocabulary digest records; the exporter checks their seals and model hashes and writes `REFERENCE.json` and `MANIFEST.json` with private permissions):

```sh
uv run --python 3.12.13 python scripts/export_scoring_reference.py \
  --source-directory /absolute/path/to/private/sealed-run \
  --output-directory /absolute/path/to/private/reference \
  --model /absolute/path/to/model
```

Run native parity:

```sh
uv run --python 3.12.13 python scripts/validate_scoring_rename.py \
  --reference-directory /absolute/path/to/private/reference \
  --reaper /absolute/path/to/stark/runtime/stark-validation-reap \
  --worker /absolute/path/to/stark/target/release/stark-score \
  --probe /absolute/path/to/stark/target/release/stark-score-digest \
  --model /absolute/path/to/model \
  --output-root /absolute/path/to/stark/test-artifacts/parity-check
```

These checks establish scoring parity for this checkpoint and fixture set. Generation acceptance is recorded separately in [ColGrep generation validation](GENERATION_COLGREP.md).
