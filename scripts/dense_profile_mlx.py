"""Prepare private matched pp512/tg128 fixtures and profile pinned MLX-LM.

Run ``prepare`` with the locked converter environment (no inference). Run ``run``
with the frozen MLX reference environment after reserving the GPU. The native
counterpart is the dense_profile Rust example. Both return 128 greedy tokens,
execute 128 decode forwards including the last-token lookahead, ignore EOS,
create fresh request state and drain that final forward before stopping time.

The MLX loop uses the stock model/cache and greedy argmax. Its cache-only prefix
and queued lookahead follow MLX-LM generate.py at
1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7 (MIT, Copyright 2023 Apple Inc.).
Log-probability vectors are unnecessary for greedy decoding and are not produced.
Only metadata, counts, hashes and timings leave private files/memory.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import statistics
import struct
import subprocess
import time
from pathlib import Path

MLX_LM_REVISION = "1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7"
ROOT = Path(__file__).resolve().parents[1]
FIXTURE_KEYS = {
    "schema_version",
    "input_token_ids",
    "decode_tokens",
    "context_tokens",
    "warmup_runs",
    "measured_runs",
    "source_repo",
    "source_revision",
    "source_config_sha256",
    "source_weights",
    "quantization",
}


def require(value: bool, code: str) -> None:
    if not value:
        raise ValueError(code)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    after = path.stat()
    require(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        "component_changed",
    )
    return hasher.hexdigest()


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(
        b"".join(struct.pack("<I", value) for value in tokens)
    ).hexdigest()


def weights_digest(weights: list[dict]) -> str:
    hasher = hashlib.sha256()
    for weight in sorted(weights, key=lambda item: item["filename"]):
        hasher.update(weight["filename"].encode())
        hasher.update(b"\0")
        hasher.update(struct.pack("<Q", weight["bytes"]))
        hasher.update(weight["sha256"].encode())
        hasher.update(b"\n")
    return hasher.hexdigest()


def read_json(path: Path, limit: int = 1024 * 1024):
    require(0 < path.stat().st_size <= limit, "metadata_size")
    return json.loads(path.read_bytes())


def weight_records(source: Path) -> list[dict]:
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        names = sorted(
            set(read_json(index_path, 32 * 1024 * 1024)["weight_map"].values())
        )
    else:
        names = ["model.safetensors"]
    require(0 < len(names) <= 64, "weight_file_count")
    records = []
    for name in names:
        require(
            isinstance(name, str)
            and name.endswith(".safetensors")
            and Path(name).name == name,
            "weight_name",
        )
        path = source / name
        size = path.stat().st_size
        require(0 < size <= 128 * 1024**3, "weight_size")
        records.append({"filename": name, "bytes": size, "sha256": digest(path)})
    return records


def quantization(config: dict) -> dict:
    declarations = [
        config.get(key)
        for key in ("quantization", "quantization_config")
        if config.get(key) is not None
    ]
    if not declarations:
        return {
            "mode": "none",
            "bits": None,
            "group_size": None,
            "overrides_sha256": None,
        }
    value = declarations[0]
    require(isinstance(value, dict), "quantization_object")
    mode, bits, group = (
        value.get("mode", "affine"),
        value.get("bits"),
        value.get("group_size"),
    )
    require(
        mode == "affine" and bits in (4, 8) and group in (16, 32, 64, 128),
        "quantization_format",
    )
    for other in declarations[1:]:
        require(isinstance(other, dict), "quantization_object")
        require(
            other.get("bits") == bits and other.get("group_size") == group,
            "quantization_conflict",
        )
    overrides = {
        key: item
        for key, item in value.items()
        if key not in {"mode", "bits", "group_size"}
    }
    override_hash = (
        hashlib.sha256(
            json.dumps(overrides, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if overrides
        else None
    )
    return {
        "mode": mode,
        "bits": bits,
        "group_size": group,
        "overrides_sha256": override_hash,
    }


def private_json(path: Path, value) -> None:
    require(
        subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(path)], cwd=ROOT, check=False
        ).returncode
        == 0,
        "private_fixture_must_be_ignored",
    )
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")


def prepare(args) -> dict:
    source = args.source.resolve(strict=True)
    require(
        source.name == args.revision
        and len(args.revision) == 40
        and all(char in "0123456789abcdef" for char in args.revision),
        "immutable_snapshot_required",
    )
    config = read_json(source / "config.json")
    vocabulary = config.get("text_config", config)["vocab_size"]
    cases = read_json(args.tokens_from)["cases"]
    base = cases[args.case]["tokens"]
    require(
        isinstance(base, list)
        and bool(base)
        and all(type(value) is int and 0 <= value < vocabulary for value in base),
        "private_input_tokens",
    )
    tokens = (base * (512 // len(base) + 1))[:512]
    fixture = {
        "schema_version": 1,
        "input_token_ids": tokens,
        "decode_tokens": 128,
        "context_tokens": 1024,
        "warmup_runs": 1,
        "measured_runs": 3,
        "source_repo": args.repo,
        "source_revision": args.revision,
        "source_config_sha256": digest(source / "config.json"),
        "source_weights": weight_records(source),
        "quantization": quantization(config),
    }
    private_json(args.output.resolve(), fixture)
    return {
        "status": "ready",
        "scope": "matched-pp512-tg128-greedy-v1",
        "input_tokens": 512,
        "returned_tokens": 128,
        "decode_forward_steps": 128,
        "input_sha256": token_digest(tokens),
        "source_repo": args.repo,
        "source_revision": args.revision,
        "source_weights_digest": weights_digest(fixture["source_weights"]),
        "quantization": fixture["quantization"],
        "fixture_sha256": digest(args.output),
    }


def validate_fixture(fixture: dict) -> None:
    require(
        isinstance(fixture, dict) and set(fixture) == FIXTURE_KEYS, "fixture_fields"
    )
    require(
        fixture["schema_version"] == 1
        and fixture["decode_tokens"] == 128
        and fixture["warmup_runs"] == 1
        and fixture["measured_runs"] == 3
        and 641 <= fixture["context_tokens"] <= 8192,
        "fixture_geometry",
    )
    tokens = fixture["input_token_ids"]
    require(
        isinstance(tokens, list)
        and len(tokens) == 512
        and all(type(value) is int and 0 <= value <= 0xFFFFFFFF for value in tokens),
        "fixture_tokens",
    )


def profile(args) -> dict:
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_PROGRESS_BARS",
        "HF_HUB_DISABLE_TELEMETRY",
    ):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    direct = json.loads(
        importlib.metadata.distribution("mlx-lm").read_text("direct_url.json")
    )
    require(
        direct.get("vcs_info", {}).get("commit_id") == MLX_LM_REVISION,
        "reference_revision_changed",
    )
    require(importlib.metadata.version("mlx") == "0.32.3", "mlx_version_changed")
    fixture = read_json(args.fixture)
    validate_fixture(fixture)
    source = args.source.resolve(strict=True)
    require(source.name == fixture["source_revision"], "source_revision_changed")
    require(
        digest(source / "config.json") == fixture["source_config_sha256"],
        "source_config_changed",
    )
    records = weight_records(source)
    require(records == fixture["source_weights"], "source_weights_changed")
    config = read_json(source / "config.json")
    require(quantization(config) == fixture["quantization"], "quantization_changed")
    vocabulary = config.get("text_config", config)["vocab_size"]
    require(
        all(token < vocabulary for token in fixture["input_token_ids"]),
        "input_outside_vocabulary",
    )
    mx.set_default_device(mx.gpu)
    mx.reset_peak_memory()
    load_started = time.perf_counter()
    model, _ = load_model(source, lazy=True, strict=True, trust_remote_code=False)
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - load_started
    stream = mx.new_stream(mx.gpu)
    parameter_dtypes = sorted(
        {
            str(array.dtype)
            for _, array in tree_flatten(model.parameters())
            if array.dtype in (mx.bfloat16, mx.float16, mx.float32)
        }
    )
    runs = []
    for run_index in range(fixture["warmup_runs"] + fixture["measured_runs"]):
        mx.synchronize(stream)
        started = time.perf_counter()
        with mx.stream(stream):
            cache = make_prompt_cache(model)
            prompt = mx.array(fixture["input_token_ids"], dtype=mx.int32)
            # Stock MLX-LM's cache-only prefix prunes unobserved readout and
            # trailing shared-KV work. Its last prompt token starts generation.
            model(prompt[:-1][None], cache=cache)
            mx.eval([entry.state for entry in cache])
            mx.clear_cache()
            logits = model(prompt[-1:][None], cache=cache)
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.async_eval(token)
            setup_seconds = time.perf_counter() - started
            generated = []
            first_return_seconds = 0.0
            for token_index in range(128):
                logits = model(token[None], cache=cache)
                following = mx.argmax(logits[:, -1, :], axis=-1)
                mx.async_eval(following)
                generated.append(int(token.item()))
                if token_index == 0:
                    first_return_seconds = time.perf_counter() - started
                token = following
            last_return_seconds = time.perf_counter() - started
            mx.synchronize(stream)
            total_seconds = time.perf_counter() - started
            require(
                all(
                    int(entry.offset) == 640
                    for entry in cache
                    if hasattr(entry, "offset")
                ),
                "unexpected_mlx_cache_length",
            )
            runs.append(
                {
                    "run": run_index,
                    "measured": run_index >= fixture["warmup_runs"],
                    "input_tokens": 512,
                    "returned_tokens": 128,
                    "decode_forward_steps": 128,
                    "context_tokens_after": 640,
                    "request_total_seconds": total_seconds,
                    "stream_setup_seconds": setup_seconds,
                    "first_token_return_seconds": first_return_seconds,
                    "remaining_token_returns_seconds": last_return_seconds
                    - first_return_seconds,
                    "final_drain_seconds": total_seconds - last_return_seconds,
                    "work_tokens_per_second": 640 / total_seconds,
                    "output_sha256": token_digest(generated),
                }
            )
            del cache, logits, token, following, prompt
    measured = [run for run in runs if run["measured"]]
    median = statistics.median(run["request_total_seconds"] for run in measured)
    repeated = len({run["output_sha256"] for run in runs}) == 1
    info = mx.device_info()
    return {
        "schema_version": 1,
        "passed": repeated,
        "backend": "mlx-lm-direct-greedy",
        "scope": "matched-pp512-tg128-greedy-v1",
        "timing_boundary": "fresh cache, 512-token prefill, 128 returned tokens, 128 decode forwards, final lookahead drain; first-return timing includes lookahead scheduling",
        "input_sha256": token_digest(fixture["input_token_ids"]),
        "input_tokens": 512,
        "decode_forward_steps": 128,
        "returned_tokens": 128,
        "context_tokens": fixture["context_tokens"],
        "warmup_runs": 1,
        "measured_runs": 3,
        "source_repo": fixture["source_repo"],
        "source_revision": fixture["source_revision"],
        "source_config_sha256": fixture["source_config_sha256"],
        "source_weights_digest": weights_digest(records),
        "source_weights": records,
        "quantization": fixture["quantization"],
        "floating_parameter_dtypes": parameter_dtypes,
        "mlx_lm_revision": MLX_LM_REVISION,
        "mlx_version": importlib.metadata.version("mlx"),
        "transformers_version": importlib.metadata.version("transformers"),
        "harness_sha256": digest(Path(__file__)),
        "device": {
            key: info[key]
            for key in (
                "device_name",
                "architecture",
                "max_recommended_working_set_size",
            )
            if key in info
        },
        "load_seconds": load_seconds,
        "peak_memory_bytes": mx.get_peak_memory(),
        "median_request_total_seconds": median,
        "median_work_tokens_per_second": 640 / median,
        "repeated_outputs_identical": repeated,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source", type=Path, required=True)
    prepare_parser.add_argument("--repo", required=True)
    prepare_parser.add_argument("--revision", required=True)
    prepare_parser.add_argument("--tokens-from", type=Path, required=True)
    prepare_parser.add_argument("--case", type=int, default=1)
    prepare_parser.add_argument("--output", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--source", type=Path, required=True)
    run_parser.add_argument("--fixture", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        # Model frameworks may emit metadata warnings. This harness exposes
        # its fixed aggregate report only, never framework exception dumps.
        with (
            open(os.devnull, "w") as null,
            contextlib.redirect_stdout(null),
            contextlib.redirect_stderr(null),
        ):
            report = prepare(args) if args.command == "prepare" else profile(args)
        if args.command == "run":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                os.chmod(args.output, 0o600)
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write("\n")
    except Exception:  # noqa: BLE001 - preserve private model/input data on every failure.
        print(json.dumps({"passed": False, "error": "dense_profile_failed"}))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
