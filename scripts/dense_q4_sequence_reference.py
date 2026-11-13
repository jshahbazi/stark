#!/usr/bin/env python3
"""Write private MLX teacher-forced Q4 references across a fixed sequence.

The input is the existing matched 512-token/128-step benchmark fixture. The
output directory contains nine logits references at context lengths 512, 528,
..., 640 plus the private token cases consumed by ``dense_parity``. Stdout
contains aggregate hashes and pinned source/runtime identity only.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from dense_profile_mlx import (
    MLX_LM_REVISION,
    ROOT,
    digest,
    quantization,
    read_json,
    token_digest,
    validate_fixture,
    weight_records,
    weights_digest,
)

CONTEXT_LENGTHS = tuple(range(512, 641, 16))
REFERENCE_ENV = ROOT / "runtime/olmo3-gemma4/reference-env"
REFERENCE_PROTOCOL = "dense-q4-sequence-v3-cache-state-only-prefill"


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def batch_tokens(mx, tokens: list[int]):
    return mx.array(tokens, dtype=mx.int32)[None, :]


def batch_next_token(token):
    return token.reshape((1, 1))


def prefill_cache_only(mx, model, token_batch, cache) -> None:
    """Match benchmark prefill: evaluate cache state and discard the readout."""
    model(token_batch, cache=cache)
    mx.eval([entry.state for entry in cache])
    mx.clear_cache()


def private_path(path: Path) -> bool:
    return (
        subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(path)],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )


def create_private_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        stream.write(payload)


@contextlib.contextmanager
def quiet_process_output():
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        with (
            open(os.devnull, "w") as null,
            contextlib.redirect_stdout(null),
            contextlib.redirect_stderr(null),
        ):
            yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(null_fd)


def run(args: argparse.Namespace) -> dict:
    source = args.source.resolve(strict=True)
    fixture_path = args.fixture.resolve(strict=True)
    output = args.output.resolve()
    require(private_path(output), "output_must_be_ignored")
    require(not output.exists(), "output_must_be_new")

    fixture = read_json(fixture_path)
    validate_fixture(fixture)
    require(
        fixture["quantization"]
        == {
            "mode": "affine",
            "bits": 4,
            "group_size": 64,
            "overrides_sha256": fixture["quantization"].get("overrides_sha256"),
        },
        "unexpected_quantization",
    )
    require(source.name == fixture["source_revision"], "source_revision_changed")
    config_path = source / "config.json"
    config_hash = digest(config_path)
    require(config_hash == fixture["source_config_sha256"], "source_config_changed")
    records = weight_records(source)
    require(records == fixture["source_weights"], "source_weights_changed")
    config = read_json(config_path)
    require(quantization(config) == fixture["quantization"], "quantization_changed")
    vocabulary = config.get("text_config", config)["vocab_size"]
    prefix = fixture["input_token_ids"]
    require(all(token < vocabulary for token in prefix), "input_outside_vocabulary")

    direct = json.loads(
        importlib.metadata.distribution("mlx-lm").read_text("direct_url.json")
    )
    require(
        direct.get("vcs_info", {}).get("commit_id") == MLX_LM_REVISION,
        "reference_revision_changed",
    )
    profile = read_json(REFERENCE_ENV / "environment.json")
    lock_hash = digest(REFERENCE_ENV / "uv.lock")
    require(lock_hash == profile["lock_sha256"], "reference_lock_changed")
    require(platform.python_version() == profile["python"], "python_version_changed")
    for package, expected in profile["packages"].items():
        require(
            importlib.metadata.version(package) == expected,
            "reference_package_changed",
        )
    mlx_version = profile["packages"]["mlx"]

    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    mx.set_default_device(mx.gpu)
    model, _ = load_model(source, lazy=True, strict=True, trust_remote_code=False)
    mx.eval(model.parameters())
    cache = make_prompt_cache(model)
    tokens = list(prefix)
    mx.set_default_device(mx.gpu)
    with mx.stream(mx.new_stream(mx.gpu)):
        prefix_array = batch_tokens(mx, tokens[:-1])
        prefill_cache_only(mx, model, prefix_array, cache)
        logits = model(batch_tokens(mx, tokens[-1:]), cache=cache)
        mx.eval(logits)
        logits = logits[0, -1]
        references: dict[int, bytes] = {}
        references[512] = np.asarray(logits.astype(mx.float32), dtype="<f4").tobytes()

        token = mx.argmax(logits, axis=-1)
        mx.eval(token)
        generated: list[int] = []
        target_lengths = set(CONTEXT_LENGTHS[1:])
        for generated_count in range(1, 129):
            token_value = int(token.item())
            generated.append(token_value)
            tokens.append(token_value)
            logits = model(batch_next_token(token), cache=cache)[0, -1]
            mx.eval(logits)
            current_length = 512 + generated_count
            if args.reference_profile == "cached" and current_length in target_lengths:
                references[current_length] = np.asarray(
                    logits.astype(mx.float32), dtype="<f4"
                ).tobytes()
            if generated_count < 128:
                token = mx.argmax(logits, axis=-1)
                mx.eval(token)

        if args.reference_profile == "recomputed":
            for context_length in CONTEXT_LENGTHS:
                case_tokens = tokens[:context_length]
                case_cache = make_prompt_cache(model)
                prefill_cache_only(
                    mx, model, batch_tokens(mx, case_tokens[:-1]), case_cache
                )
                logits = model(batch_tokens(mx, case_tokens[-1:]), cache=case_cache)[
                    0, -1
                ]
                mx.eval(logits)
                require(
                    all(
                        int(entry.offset) == context_length
                        for entry in case_cache
                        if hasattr(entry, "offset")
                    ),
                    "unexpected_recomputed_cache_length",
                )
                references[context_length] = np.asarray(
                    logits.astype(mx.float32), dtype="<f4"
                ).tobytes()
                del case_cache, logits

    require(
        set(references) == set(CONTEXT_LENGTHS)
        and len(generated) == 128
        and len(tokens) == 640,
        "reference_case_generation_failed",
    )
    for raw in references.values():
        require(
            0 < len(raw) <= 16 * 1024**2 and len(raw) % 4 == 0,
            "reference_size_invalid",
        )

    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(output, 0o700)
    cases = []
    reference_hashes = []
    for index, context_length in enumerate(CONTEXT_LENGTHS):
        case_tokens = tokens[:context_length]
        raw = references[context_length]
        reference_path = output / f"reference-{index}.f32"
        create_private_file(reference_path, raw)
        reference_hashes.append(hashlib.sha256(raw).hexdigest())
        cases.append({"tokens": case_tokens, "context_tokens": 1024})
    cases_path = output / "cases.json"
    create_private_file(
        cases_path,
        (json.dumps({"cases": cases}, sort_keys=True) + "\n").encode(),
    )
    report = {
        "status": "ready",
        "reference_protocol": REFERENCE_PROTOCOL,
        "case_count": len(cases),
        "context_lengths": list(CONTEXT_LENGTHS),
        "reference_forward_profile": (
            "cached-generation-trajectory-v3-cache-state-only-prefix"
            if args.reference_profile == "cached"
            else "fresh-cache-prefix-n-minus-1-cache-state-eval-final-token-logits-v3"
        ),
        "generation_profile": "stock-mlx-prefix-cache-only-511-then-final-token-and-128-greedy-forwards-ignore-eos-v3",
        "input_tokens": 512,
        "generated_tokens": len(generated),
        "generated128_sha256": token_digest(generated),
        "token_sequence_sha256": token_digest(tokens),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "reference_sha256": reference_hashes,
        "source_repo": fixture["source_repo"],
        "source_revision": fixture["source_revision"],
        "source_config_sha256": config_hash,
        "source_weights_digest": weights_digest(records),
        "quantization": fixture["quantization"],
        "mlx_version": mlx_version,
        "mlx_lm_revision": MLX_LM_REVISION,
        "reference_lock_sha256": lock_hash,
        "harness_sha256": digest(Path(__file__)),
    }
    create_private_file(
        output / "sequence-reference.json",
        (json.dumps(report, sort_keys=True) + "\n").encode(),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-profile",
        choices=("cached", "recomputed"),
        default="cached",
        help="Reference policy; cached preserves the original v1 generation-trajectory behavior",
    )
    args = parser.parse_args()
    try:
        with quiet_process_output():
            report = run(args)
    except Exception:  # noqa: BLE001 - never expose private model/input diagnostics.
        print(json.dumps({"status": "failed", "error": "sequence_reference_failed"}))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
