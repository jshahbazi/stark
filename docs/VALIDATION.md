# Validation status

The rewritten standalone source was rebuilt on 2026-10-11 with Rust 1.98.1 and the `optimized` feature. All 243 JevK5 BF16 evaluation cases passed native parity against the frozen accepted Uzu scorer references. The 486 cold/cache-hit replies and their typed decisions matched exactly, including raw logits, log probabilities, and calibrated candidate probabilities. All 243 prompt tokenizations also matched.

All 48 calibration boundary requests (96 replies) and 534 complete-vocabulary digest requests (1,068 replies) passed bit-for-bit. The worker performed 291 native scoring calls and 291 cache hits; the probe performed 534 calls and 534 hits. Separate controls passed candidate/order/temperature/context changes, default-off scoring, errors, clean EOF, and restart with a cold cache. Both binaries passed 19 Rust tests each. Eight model-free process ownership checks passed, covering cancellation, concurrent launch rejection, execution failure, and retained ownership-loss reporting.

The [aggregate validation report](../provenance/SCORING_VALIDATION.json) and [build receipt](../provenance/BUILD_RECEIPT.json) identify source commit `97f43755dd833b9c33f35db19bf859057ee0830c`, dependency lock, executable hashes, tooling, portable fixture, and private result seals. Older receipts were removed during the history rewrite because their source and feature identities describe the previous history. These fresh checks establish correctness; no timing estimate was computed.

## Regression tests

```sh
cargo +1.98.1 test --release --locked --offline -p stark-score --features optimized -- --test-threads=1
```

## Native parity

The validator uses a private, self-contained reference directory, an explicit native process reaper, and a converted model directory. Reference fixtures and raw results are private and are excluded from Git. The standard-library Python tools can run through `uv` without installing project dependencies.

Compile the standalone ownership helper on macOS:

```sh
mkdir -p runtime
clang -std=c11 -O2 -Wall -Wextra -Werror scripts/stark-validation-reap.c -o runtime/stark-validation-reap
```

Check ownership without loading a model:

```sh
uv run --python 3.12.13 python scripts/test_scoring_process.py --reaper runtime/stark-validation-reap
```

To make a portable fixture from an existing sealed scoring run:

```sh
uv run --python 3.12.13 python scripts/export_scoring_reference.py --source-directory /absolute/path/to/private/sealed-run --output-directory /absolute/path/to/private/reference --model /absolute/path/to/model
```

The source directory supplies `configuration.json`, `dataset.json`, pre/post run seals, four complete scoring records, and two vocabulary digest records. The exporter checks their seals and model hashes. It writes `REFERENCE.json` and `MANIFEST.json` with private permissions.

Run native parity against a prepared private fixture:

```sh
uv run --python 3.12.13 python scripts/validate_scoring_rename.py --reference-directory /absolute/path/to/private/reference --reaper /absolute/path/to/stark/runtime/stark-validation-reap --worker /absolute/path/to/stark/target/release/stark-score --probe /absolute/path/to/stark/target/release/stark-score-digest --model /absolute/path/to/model --output-root /absolute/path/to/stark/test-artifacts/parity-check
```

These checks qualify probability scoring parity for this checkpoint and fixture set. A separate [ColGrep generation record](GENERATION.md) provides controlled functional acceptance for its exact checkpoint, artifact and named worker build, including a documented token divergence. Other generation profiles, model families, Swift/Python/TypeScript binary bindings and precise latency measurement remain unqualified.

## Firebase configuration history cleanup (2026-10-22)

The inherited playground Firebase configuration was removed from the history
of all 34 published branches and all 3 tags. The rewrite was published atomically
with leases on every ref. All 37 branch and tag tip trees remained identical.
A fresh fetch of those refs contained 30,996 Git objects; an in-memory comparison
against the flagged key found zero matches. No key value was logged or included
in validation artifacts.

The validated main revision after the rewrite was
`04a5fd1a896362849cc4219540e5b6afa63a5ace`. Commit and tag IDs changed, and
the rewrite stripped original commit and tag signatures. Checked-in upstream
revision references and provenance text continue to identify their original
sources. The original branch-protection settings were restored and verified
after publication.

GitHub still retained 22 read-only PR head refs at verification time, and the
secret-scanning alert remained open. Those refs and cached historical views
require separate cleanup through
[GitHub Support](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository#fully-removing-the-data-from-github).
This validation covers the published branches and tags; it does not establish
removal from PR refs, caches, existing local clones, or upstream Uzu.

Existing clones must migrate unpublished work onto the rewritten history before
merging or pushing. Merging old ancestry can restore the removed configuration.
Preserve uncommitted work before replacing or synchronizing any clone. The
playground ignore rule excludes its local Firebase configuration from ordinary
`git add` operations.
