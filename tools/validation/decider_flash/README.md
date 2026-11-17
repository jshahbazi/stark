# Decider Flash v1.3 local reference runner

This directory provides a locked CPU/MPS environment and a JSONL bridge to the
official `pplx_decider` reference implementation. The bridge uses
`source/src/pplx_decider` from a caller-supplied local model snapshot; it does
not copy, patch, or install that source. It does not download checkpoints.

The dependency versions follow the upstream `source/pyproject.toml` and
`requirements.txt` at revision
`d06fb3f2d97e0a52325a0d463c450cf0338f44fe`, with one security override:
Pillow is pinned to **13.0.0** instead of upstream's vulnerable 12.3.0. The
official model source and snapshot remain unchanged. Historical qualifications
using 12.3.0 keep their original environment identity; new runs record the
updated lockfile separately. Use the project environment with
`uv sync --locked`. Start a child process with:

```sh
uv run --project tools/validation/decider_flash --locked \
  python tools/validation/decider_flash/runner.py \
  --snapshot /path/to/local/pplx-decider-flash-v1.3 \
  --device cpu
```

Use `--device mps` to evaluate the same FP32 reference path on Apple GPU. The
default is CPU, with eight PyTorch CPU threads. `fast=False` disables the
upstream CUDA compilation and CUDA-graph path; `locate=False` keeps optional
Locate adapter loading out of a baseline decision run. Upstream initialization
and inference stdout/stderr are redirected to `/dev/null`, and exceptions are
translated to fixed error codes so private prompts, outputs and exceptions do
not enter logs.

`--encoding-only` loads the upstream `AutoProcessor` and `Encoder` without
loading a backbone. It returns token IDs, readout positions and image grids on
the protocol pipe. Compare the official and Stark formatters with:

```sh
uv run --project tools/validation/decider_flash --locked \
  python tools/validation/decider_flash/compare_encodings.py
```

That command runs separate Python 3.12 and 3.13 children and compares all
fixture encodings in memory. For an image sequence it expands Stark's single
image placeholder using the official processor's grid before comparing token
sequences. Its output contains aggregate counts and hashes only.

The process reads one JSON object per input line, each shaped as
`{"request": {"state": ..., "questions": ..., "images": ...}}`. It first
emits a `ready` record, then one `response` or fixed `error` record per input.
Reference answer probabilities are returned only in the JSONL response for the
caller to keep in memory or write to a protected validation pipe. Do not save
raw protocol traffic as a repository artifact.

This runner is for reference collection only. It does not establish that Stark's
native implementation supports the checkpoint or that CPU/MPS FP32 results
match the upstream CUDA BF16 execution path.
