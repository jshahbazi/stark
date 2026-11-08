# Drex Max v1 and JevK5 local comparison

This research harness compares the available Drex Max v1 affine-Q4 release on
its upstream MLX scorer with JevK5 BF16 on Stark. Drex Max v1.1 weights were
unreleased when the comparison was prepared on 2026-11-08. The models differ
in parameter count, precision, attention, readout, prompt and runtime. Results
describe these two local stacks and the selected cases.

The shared evaluation contains 231 public JevBench cases and 12 synthetic
controls. It is a reused development set, not a fresh deployment test. Its
sealed dataset, tokenization and original readouts remain private. The harness
validates all reference seals through `stock_compare_cases.load_reference`.

The candidate order comes from `scoring_fixture.decision_options`, checked
against the sealed typed decisions. The dataset's `labels` field is an answer
set whose order can differ. Boolean gold labels `yes/no` are mapped to
`true/false`. Both models' probabilities are aligned by answer identifier before
evaluation. This check is necessary before reusing any older aggregate quality
result from a positional mapping.

## Setup

Use an isolated Stark worktree and its own build target. Build the scorer with
Rust 1.98.1 and locked dependencies:

```sh
CARGO_TARGET_DIR="$PWD/target" cargo +1.98.1 build --release --locked -p stark-score --features optimized
uv sync --project scripts/drex_jevk5 --locked --python 3.13.13
git clone https://github.com/nace-ai/drex-decision-models.git runtime/drex-source
git -C runtime/drex-source checkout d2fd26cdf211f87b9a6b1261995528629c9e1401
uv run --project scripts/drex_jevk5 --locked hf download nace-ai/drex-max-v1-affine-q4 \
  --revision 23f21ac6e664d3edb32faa7b55b0c8a7fb09134a \
  --local-dir runtime/drex-max-v1-affine-q4
```

Copy `config.template.json` into ignored `runtime/` and provide the local
paths. JevK5's conversion receipt must identify checkpoint revision
`c4f7fdb3aeab5582336406e78d3bef11bf98833d` and match the current weight hash.
The harness also verifies Drex's exact weight/readout hashes, a clean pinned
source checkout, disabled Jev score caching and untruncated replies. Model
artifacts retain their upstream licenses; this comparison does not redistribute
their weights or upstream implementation.

## Run

Run a four-case admission probe before the complete two-block comparison:

```sh
uv run --project scripts/drex_jevk5 --locked python scripts/drex_jevk5/compare.py \
  --config runtime/drex-jevk5-config.json --output runtime/drex-jevk5-probe.json --probe
uv run --project scripts/drex_jevk5 --locked python scripts/drex_jevk5/compare.py \
  --config runtime/drex-jevk5-config.json --output runtime/drex-jevk5-full.json
```

The order is Jev/Drex, then Drex/Jev. Each block starts a fresh process, loads
one model, completes a first warmup and three further warmups, then scores all
cases. Both paths use fresh complete forward passes without score or prefix
caching. Drex's full attention is noncausal: earlier states depend on the entire
request, preventing ordinary causal prefix reuse. This is an external research
runner, not a new Stark backend or production model qualification.

Tokenization and loading are excluded from latency. Jev uses the exact sealed
input and candidate token IDs, its full vocabulary readout and temperature 1.22.
Drex uses upstream `prepare` and `decide`, its question prompt, noncausal
attention, trained 255-way readout and embedded temperature. The harness captures
full precision probabilities in memory before upstream four-decimal response
formatting. It preserves that formatting work in the timed path. Forward work
is synchronized before each reply. Single-stream timings include scoring,
readout, response formatting and private pipe exchange.

The shared `/tmp/kapitsa-gpu.lock` is advisory. The harness records sampled
background CPU/model activity, pageout/swapout deltas and whole-window sleep
events. It does not enforce GPU exclusivity or continuously measure clocks,
thermal state or power. Examine every block's screens before interpreting
latency. RSS is sampled every 50 ms from startup and includes the harness and
descendant processes. Shared pages may be counted more than once, and RSS is
not a direct measurement of Metal allocated or resident bytes.

Quality includes accuracy, multiclass Brier sum and ECE in ten bins, separately
for public and synthetic cases. Paired quality reports discordant correct
answers, a deterministic 10,000-resample percentile interval for accuracy
difference, and an exact two-sided McNemar test. Repeated passes check numerical
repeatability and do not increase the independent case count. The public-set
result is not the Artificial Analysis Decision Index and is not held-out
evidence of a replacement's reliability or calibration.

Only aggregates and artifact hashes are written. Requests, token vectors,
individual decisions and probabilities stay in process memory/private pipes;
child stderr is discarded and errors use fixed content-free diagnostics.

## Model-free verification

```sh
uv run --project scripts/drex_jevk5 --locked pytest scripts/drex_jevk5/test_compare.py -q
uv run --project scripts/drex_jevk5 --locked ruff check scripts/drex_jevk5
uv run --project scripts/drex_jevk5 --locked ruff format --check scripts/drex_jevk5
```
