"""Recover four exact prompt fixtures from block zero of the 2026-11-23 native receipt."""

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tools/converter")]


def prompt_bodies(probe, calibration, rung_512):
    # Ask the pinned benchmark's own TypeScript corpus builder to reconstruct
    # the four bodies. The tag is searched against the saved native ID hash.
    bench = probe / "src/bench/index.ts"
    code = bench.read_text()
    prose = re.search(r'"([^"\n]*stream-of-consciousness prose[^"\n]*)"', code)
    if not prose:
        raise ValueError("unrecognized llmprobe prose probe")
    call = """
const calibration = {bytes: CALIBRATION_FILLER_BYTES, tokens: CAL};
const first = {bytes: fillerBytesFor(512, [calibration], RUNG_FIXED_CHARS), tokens: RUNG};
const bytes = fillerBytesFor(4096, [calibration, first], RUNG_FIXED_CHARS);
const contextBody = buildCodeContextWithConstant(bytes) + "\\n\\n" + CODE_INSTRUCTION;
process.stdout.write(JSON.stringify([DECODE_PROMPT, PROSE, contextBody]));
""".replace("tokens: CAL", f"tokens: {calibration}").replace("tokens: RUNG", f"tokens: {rung_512}")
    call = call.replace("PROSE,", json.dumps(prose.group(1)) + ",")
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory) / "recover.cjs"
        subprocess.run(
            [
                probe / "node_modules/.bin/esbuild",
                "--bundle",
                "--platform=node",
                "--loader=ts",
                "--format=cjs",
                "--log-level=error",
                f"--outfile={bundle}",
            ],
            cwd=bench.parent,
            input=code + call,
            text=True,
            check=True,
        )
        return json.loads(subprocess.check_output(["node", bundle]))


def write_input(directory, tokens, eos, teacher=b""):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    size = len(tokens)
    tensors = {
        "tokens": {"dtype": "U32", "shape": [size // 4], "data_offsets": [0, size]},
        "eos_token_id": {"dtype": "U32", "shape": [1], "data_offsets": [size, size + 4]},
        "prompt_len": {"dtype": "U32", "shape": [1], "data_offsets": [size + 4, size + 8]},
    }
    if teacher:
        tensors["reference_teacher"] = {
            "dtype": "U32",
            "shape": [len(teacher) // 4],
            "data_offsets": [size + 8, size + 8 + len(teacher)],
        }
    header = json.dumps(tensors).encode()
    header += b" " * (-len(header) % 8)
    with (directory / "tokens.safetensors").open("xb") as file:
        file.write(struct.pack("<Q", len(header)) + header + tokens + struct.pack("<II", eos, size // 4))
        file.write(teacher)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "native_receipt", "target", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--worker", type=Path, help="reproduce and hash-check the recorded greedy continuations")
    args = parser.parse_args()
    os.umask(0o077)
    with ExitStack() as cleanup:
        worker = None
        if args.worker:
            from stark_convert.chat import Worker

            worker = Worker(args.worker, args.target, 240)
            cleanup.callback(worker.close)
            if not worker.read().get("ready"):
                raise RuntimeError("historical worker failed its readiness check")
        recover(args, worker)


def recover(args, worker):
    from benchmark_standard import chat_adapters

    receipt = [json.loads(row) for row in args.native_receipt.read_text().splitlines()]
    coding, prose, context = prompt_bodies(
        args.probe.resolve(), receipt[48]["input_tokens"], receipt[49]["input_tokens"]
    )
    renderer, *_ = chat_adapters(args.target, "inkling2-json-v1")
    tokenizer = Tokenizer.from_file(str(args.target / "tokenizer.json"))
    eos = json.loads((args.target / "generation_config.json").read_text())["eos_token_id"]
    for name, row, body in (
        ("coding", 5, coding),
        ("sustained", 57, coding),
        ("prose", 20, prose),
        ("context4k", 51, context),
    ):
        ref = receipt[row]
        for sequence in range(100):
            tag = np.base_repr(sequence, base=36).lower()
            messages = [{"role": "user", "content": f"[probe inkbench00-{tag}] {body}"}]
            rendered = renderer(messages, thinking=False, effort=None)
            ids = tokenizer.encode(rendered, add_special_tokens=False).ids
            packed = np.array(ids, dtype="<u4").tobytes()
            if hashlib.sha256(packed).hexdigest() == ref["input_ids_le_u32_sha256"]:
                teacher = b""
                if worker:
                    reply = worker.request(
                        {
                            "operation": "generate",
                            "input_token_ids": ids,
                            "context_tokens": ref["context"],
                            "max_new_tokens": ref["cap"],
                            "sampling": {"mode": "greedy"},
                            "seed": 0,
                            "stream": True,
                        },
                        on_delta=lambda _: None,
                    )
                    teacher = np.asarray(reply.get("generated_token_ids", []), dtype="<u4").tobytes()
                    if hashlib.sha256(teacher).hexdigest() != ref["output_ids_le_u32_sha256"]:
                        raise ValueError(f"{name}: historical continuation hash differs")
                write_input(args.out / name, packed, eos, teacher)
                print(name, len(ids), len(teacher) // 4, ref["input_ids_le_u32_sha256"], flush=True)
                break
        else:
            raise ValueError(f"{name}: no matching historical input")


if __name__ == "__main__":
    main()
