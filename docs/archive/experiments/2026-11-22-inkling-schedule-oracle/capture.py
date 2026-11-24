"""Drive the pinned stock-schedule capture runner over private request cases.

Run `make_requests.py` first. For each selected case this checks the runner
hash, runs one capture into the case directory with the case's context as
`--ctx-size`, then checks the four teacher rows, the seed position and the
token count. It writes `capture.json` with SHA-256 values beside the private
files and prints only aggregates. Raw captures, logs and requests stay in the
ignored private directory.

The runner is invoked at its own path because it loads `mlx_metal_v4` from
its executable directory.
"""

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

from schedule import CASES, TEACHER_ROWS, request_fields

RUNNER_SHA256 = "64293fa71651074e08703bdcd33d66c870b1ea23967f04f9da17bd6f095693b5"
MANIFEST_SHA256 = "bd9c92c3d4c4c1a6c6904dea6a51e8503cceb0f77f8073c2c9e5c63976e0aa4d"
MODEL = "inkling-m1-original"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor(path, name):
    """Return (dtype, shape, bytes) of one tensor without loading the rest."""
    with path.open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(length))
        field = header[name]
        start, end = field["data_offsets"]
        stream.seek(8 + length + start)
        return field["dtype"], field["shape"], stream.read(end - start)


def capture(runner, store, directory, prompt, context):
    request = json.loads((directory / "request.json").read_bytes())
    fields = request_fields(prompt)
    if any(request.get(key) != value for key, value in fields.items()):
        raise ValueError("request does not match the case schedule")
    if (directory / "row-0.safetensors").exists():
        raise FileExistsError("case already holds a capture; use a fresh private directory")
    environment = dict(
        os.environ,
        OLLAMA_MODELS=str(store),
        OLLAMA_LLM_LIBRARY="mlx_metal_v4",
        OLLAMA_NUM_PARALLEL="1",
        INKLING_ORACLE_INPUT=str(directory / "request.json"),
        INKLING_ORACLE_OUTPUT=str(directory),
    )
    started = time.monotonic()
    with (directory / "run.log").open("wb") as log:
        status = subprocess.run(
            [str(runner), "--model", MODEL, "--ctx-size", str(context)],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    seconds = time.monotonic() - started
    rows = [directory / f"row-{row}.safetensors" for row in range(TEACHER_ROWS + 1)]
    if status != 0 or not all(row.exists() for row in rows):
        raise RuntimeError(f"capture exited with status {status} without four rows")
    dtype, shape, data = tensor(rows[0], "start")
    if dtype != "U32" or shape != [1] or struct.unpack("<I", data)[0] != prompt - 1:
        raise ValueError("captured seed position differs from the case")
    dtype, shape, tokens = tensor(directory / "tokens.safetensors", "tokens")
    if dtype != "U32" or shape != [prompt + TEACHER_ROWS]:
        raise ValueError("captured token count differs from the case")
    expected = json.loads((directory / "expected.json").read_bytes())
    token_sha = hashlib.sha256(tokens).hexdigest()
    receipt = {
        "prompt_tokens": prompt,
        "context": context,
        "start": prompt - 1,
        "chunks": fields["Chunks"],
        "runner_sha256": RUNNER_SHA256,
        "seconds": seconds,
        "token_sha256": token_sha,
        "matches_release_tokenizer": token_sha == expected["token_sha256"],
        "rows_sha256": [sha256_file(row) for row in rows],
    }
    (directory / "capture.json").write_text(json.dumps(receipt, indent=1))
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runner", type=Path, help="Pinned stock-schedule runner beside its mlx_metal_v4 directory")
    parser.add_argument("store", type=Path, help="Original Ollama model store")
    parser.add_argument("private", type=Path, help="Directory written by make_requests.py")
    parser.add_argument("--cases", default=",".join(case for case, (_, _, wanted) in CASES.items() if wanted))
    args = parser.parse_args()
    os.umask(0o077)
    if sha256_file(args.runner) != RUNNER_SHA256:
        raise ValueError("unexpected capture runner hash")
    manifest = args.store / "manifests-v2/ollama.com/library" / MODEL / "latest"
    if sha256_file(manifest) != MANIFEST_SHA256:
        raise ValueError("unexpected imported manifest hash")
    failed = False
    for case in filter(None, args.cases.split(",")):
        prompt, context, _ = CASES[case]
        try:
            receipt = capture(args.runner.resolve(), args.store.resolve(), (args.private / case).resolve(), prompt, context)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"{case}: failed: {type(error).__name__}: {error}")
            failed = True
            continue
        print(
            f"{case}: N={prompt} context={context} seconds={receipt['seconds']:.1f} "
            f"tokenizer_match={receipt['matches_release_tokenizer']} tokens_sha256={receipt['token_sha256']}"
        )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
