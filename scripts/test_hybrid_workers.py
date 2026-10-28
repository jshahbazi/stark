#!/usr/bin/env python3
"""Check native hybrid worker integration using a private, bounded tiny fixture.

Run through the locked converter environment. The source fixture must contain
config.json, weights, tokens.json and reference.f32. This creates a temporary
copy with a synthetic WordLevel tokenizer and supplemental EOS metadata. Only
aggregate checks, numeric error metrics and hashes are emitted.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers

MAX_WEIGHT_BYTES = 128 * 1024 * 1024
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_PROTOCOL_BYTES = 1024 * 1024
ABSOLUTE_TOLERANCE = 1e-4
RELATIVE_TOLERANCE = 1e-4


class CheckFailed(RuntimeError):
    """Contains an authored check identifier without private worker content."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise CheckFailed(code)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def bounded_json(path: Path, limit: int = MAX_METADATA_BYTES):
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
        "fixture_metadata_invalid",
    )
    return json.loads(path.read_bytes())


def token_hash(tokens: list[int]) -> str:
    return hashlib.sha256(
        b"".join(struct.pack("<I", token) for token in tokens)
    ).hexdigest()


def prepare_fixture(source: Path, destination: Path) -> dict:
    config_path = source / "config.json"
    config = bounded_json(config_path)
    require(isinstance(config, dict), "fixture_config_invalid")
    family = config.get("model_type")
    require(family in {"glm5_next", "qwen4_exp"}, "fixture_family_invalid")
    text = config.get("text_config", config)
    vocab_size = text.get("vocab_size")
    max_context = text.get("max_position_embeddings")
    require(
        type(vocab_size) is int and 4 <= vocab_size <= 65536,
        "fixture_vocabulary_invalid",
    )
    require(type(max_context) is int and max_context >= 5, "fixture_context_invalid")
    tokens = bounded_json(source / "tokens.json")
    require(
        isinstance(tokens, list)
        and 3 <= len(tokens) <= 4096
        and all(type(token) is int and 0 <= token < vocab_size for token in tokens),
        "fixture_tokens_invalid",
    )
    reference_path = source / "reference.f32"
    require(
        reference_path.is_file()
        and not reference_path.is_symlink()
        and 3 * vocab_size * 4 <= reference_path.stat().st_size <= MAX_WEIGHT_BYTES
        and reference_path.stat().st_size % (vocab_size * 4) == 0,
        "fixture_reference_invalid",
    )
    # The first three source-bound positions suffice for teacher-forced A and B
    # requests. Some source fixtures contain further independent cases.
    with reference_path.open("rb") as file:
        reference = list(
            struct.unpack("<" + "f" * (3 * vocab_size), file.read(3 * vocab_size * 4))
        )
    require(
        all(math.isfinite(value) for value in reference), "fixture_reference_nonfinite"
    )
    rows = [
        reference[index * vocab_size : (index + 1) * vocab_size] for index in range(3)
    ]
    weights = sorted(
        path for path in source.iterdir() if path.suffix in {".gguf", ".safetensors"}
    )
    require(bool(weights), "fixture_weights_missing")
    require(
        all(path.is_file() and not path.is_symlink() for path in weights),
        "fixture_weights_invalid",
    )
    total_bytes = sum(path.stat().st_size for path in weights)
    require(0 < total_bytes <= MAX_WEIGHT_BYTES, "fixture_weights_exceed_tiny_limit")
    destination.mkdir(mode=0o700)
    source_weights = hashlib.sha256()
    for path in weights:
        source_hash = digest(path)
        source_weights.update(path.name.encode() + b"\0" + bytes.fromhex(source_hash))
        copied = destination / path.name
        shutil.copyfile(path, copied)
        require(digest(copied) == source_hash, "fixture_weight_copy_changed")
    shutil.copyfile(config_path, destination / "config.json")
    config_hash = digest(config_path)
    require(
        digest(destination / "config.json") == config_hash,
        "fixture_model_config_changed",
    )

    base_eos = text.get("eos_token_id")
    base_eos = [base_eos] if type(base_eos) is int else base_eos
    require(
        isinstance(base_eos, list)
        and bool(base_eos)
        and all(type(value) is int and 0 <= value < vocab_size for value in base_eos),
        "fixture_eos_invalid",
    )
    supplements = [value for value in range(vocab_size) if value not in base_eos][:2]
    require(len(supplements) == 2, "fixture_eos_supplements_unavailable")
    generation_eos, tokenizer_eos = supplements
    eos_union = sorted({*base_eos, generation_eos, tokenizer_eos})
    vocabulary = {f"unit{index}": index for index in range(vocab_size)}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="unit0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(
        [
            AddedToken(f"unit{index}", special=True, single_word=True)
            for index in eos_union
        ]
    )
    tokenizer.save(str(destination / "tokenizer.json"))
    (destination / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [base_eos[0], generation_eos]}) + "\n"
    )
    (destination / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": {"content": f"unit{tokenizer_eos}"},
                "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
            }
        )
        + "\n"
    )
    return {
        "family": family,
        "vocab_size": vocab_size,
        "context": min(16, max_context),
        "tokens": tokens[:3],
        "reference": rows,
        "eos": eos_union,
        "weight_bytes": total_bytes,
        "source_config_sha256": config_hash,
        "source_weights_sha256": source_weights.hexdigest(),
        "reference_sha256": digest(reference_path),
        "prepared_tokenizer_sha256": digest(destination / "tokenizer.json"),
    }


class Worker:
    def __init__(self, process):
        self.process = process

    async def reply(self, timeout: float = 60.0) -> dict:
        line = await asyncio.wait_for(self.process.stdout.readline(), timeout)
        require(
            bool(line) and len(line) <= MAX_PROTOCOL_BYTES and line.endswith(b"\n"),
            "worker_reply_boundary_invalid",
        )
        value = json.loads(line)
        require(isinstance(value, dict), "worker_reply_invalid")
        return value

    async def request(self, value: dict) -> dict:
        data = (
            json.dumps(value, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        )
        require(len(data) <= MAX_PROTOCOL_BYTES, "test_request_exceeds_limit")
        self.process.stdin.write(data)
        await self.process.stdin.drain()
        return await self.reply()


@asynccontextmanager
async def worker_process(binary: Path, arguments: list[str]):
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("STARK_", "UZU_"))
    }
    process = await asyncio.create_subprocess_exec(
        str(binary),
        *arguments,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=environment,
        limit=MAX_PROTOCOL_BYTES + 1,
    )
    clean = False
    try:
        yield Worker(process)
        clean = True
    finally:
        if process.stdin is not None:
            process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), 10)
        except TimeoutError:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                process.kill()
                await process.wait()
        stderr = await process.stderr.read(MAX_PROTOCOL_BYTES + 1)
        extra = await process.stdout.read(MAX_PROTOCOL_BYTES + 1)
        if clean:
            require(
                process.returncode == 0 and not stderr and not extra,
                "worker_shutdown_boundary_invalid",
            )


def generated_identity(reply: dict) -> tuple:
    require("error" not in reply, "generation_failed")
    return tuple(
        json.dumps(reply.get(name), sort_keys=True)
        for name in [
            "generated_token_ids",
            "text",
            "input_tokens",
            "output_tokens",
            "sampled_tokens",
            "finish_reason",
            "stop_token_id",
        ]
    )


def check_generation(reply: dict, info: dict, length: int) -> None:
    require("error" not in reply, "generation_failed")
    reference = info["reference"][length - 1]
    best = max(range(len(reference)), key=lambda index: reference[index])
    require(
        reply.get("input_tokens") == length and reply.get("sampled_tokens") == 1,
        "generation_accounting_invalid",
    )
    if best in info["eos"]:
        require(
            reply.get("generated_token_ids") == []
            and reply.get("stop_token_id") == best
            and reply.get("finish_reason") == "eos"
            and reply.get("text") == "",
            "generation_eos_reference_mismatch",
        )
    else:
        require(
            reply.get("generated_token_ids") == [best]
            and reply.get("stop_token_id") is None
            and reply.get("finish_reason") == "length"
            and reply.get("text") == f"unit{best}",
            "generation_greedy_reference_mismatch",
        )


async def check_generator(binary: Path, path: Path, info: dict) -> dict:
    async with worker_process(binary, ["--model-path", str(path)]) as worker:
        ready = await worker.reply()
        require(
            ready.get("ready") is True
            and ready.get("vocab_size") == info["vocab_size"]
            and ready.get("state_policy") == "fresh-per-request"
            and sorted(ready.get("eos_token_ids", [])) == info["eos"],
            "generation_readiness_or_eos_union_invalid",
        )
        words = " ".join(f"unit{token}" for token in info["tokens"])
        tokenized = await worker.request(
            {"operation": "tokenize", "text": words, "add_special_tokens": False}
        )
        require(
            tokenized.get("token_ids") == info["tokens"],
            "generation_tokenizer_mismatch",
        )
        decoded = await worker.request(
            {
                "operation": "decode",
                "token_ids": info["tokens"],
                "skip_special_tokens": False,
            }
        )
        require(decoded.get("text") == words, "generation_decoder_mismatch")

        def request(length):
            return {
                "operation": "generate",
                "input_token_ids": info["tokens"][:length],
                "context_tokens": info["context"],
                "max_new_tokens": 1,
                "sampling": {"mode": "greedy"},
            }

        first = await worker.request(request(3))
        other = await worker.request(request(2))
        replay = await worker.request(request(3))
        for reply, length in [(first, 3), (other, 2), (replay, 3)]:
            check_generation(reply, info, length)
        require(
            generated_identity(first) == generated_identity(replay),
            "generation_request_state_leaked",
        )
        invalid = {**request(3), "input_token_ids": [info["vocab_size"]]}
        failed = await worker.request(invalid)
        require(
            failed.get("error") == "invalid_generation_request",
            "generation_invalid_request_accepted",
        )
        recovered = await worker.request(request(3))
        check_generation(recovered, info, 3)
        require(
            generated_identity(first) == generated_identity(recovered),
            "generation_failure_recovery_changed",
        )
    return {
        "eos_union": True,
        "tokenize_decode": True,
        "reference_greedy_agreement": 4,
        "request_isolation": True,
        "invalid_request_recovery": True,
    }


def score_values(reply: dict, reference: list[float], candidates: list[int]) -> float:
    require(
        "error" not in reply and reply.get("candidate_token_ids") == candidates,
        "score_reply_invalid",
    )
    actual = reply.get("logits")
    require(
        isinstance(actual, list) and len(actual) == len(candidates),
        "score_readout_invalid",
    )
    maximum_error = 0.0
    for value, candidate in zip(actual, candidates, strict=True):
        expected = reference[candidate]
        require(
            type(value) in {int, float} and math.isfinite(value),
            "score_readout_nonfinite",
        )
        error = abs(value - expected)
        maximum_error = max(maximum_error, error)
        require(
            error <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * abs(expected),
            "score_reference_logit_mismatch",
        )
    probabilities = reply.get("candidate_probabilities")
    require(
        isinstance(probabilities, list)
        and len(probabilities) == len(candidates)
        and all(
            type(value) in {int, float} and math.isfinite(value) and 0 <= value <= 1
            for value in probabilities
        )
        and abs(sum(probabilities) - 1.0) <= 1e-12,
        "score_probability_normalization_invalid",
    )
    return maximum_error


async def check_scorer(binary: Path, path: Path, info: dict) -> dict:
    candidates = list(range(min(16, info["vocab_size"])))
    expected_hash = token_hash(info["tokens"])
    maximum_error = 0.0
    async with worker_process(
        binary, ["--score-model-path", str(path), "--consecutive-cache"]
    ) as worker:
        ready = await worker.reply()
        require(
            ready.get("ready") is True
            and ready.get("cache_enabled") is True
            and ready.get("vocab_size") == info["vocab_size"],
            "score_readiness_invalid",
        )
        tokenized = await worker.request(
            {
                "operation": "tokenize",
                "text": " ".join(f"unit{token}" for token in info["tokens"]),
            }
        )
        require(
            tokenized.get("token_ids") == info["tokens"]
            and tokenized.get("token_ids_sha256") == expected_hash,
            "score_tokenizer_mismatch",
        )

        def request(length):
            return {
                "operation": "score",
                "input_token_ids": info["tokens"][:length],
                "candidate_token_ids": candidates,
                "context_tokens": info["context"],
                "temperature": 1.0,
            }

        responses = []
        for length, expected_hit in [(3, False), (3, True), (2, False), (3, False)]:
            response = await worker.request(request(length))
            require(
                response.get("cache_hit") is expected_hit,
                "score_consecutive_cache_policy_invalid",
            )
            require(
                response.get("prompt_tokens_sha256")
                == token_hash(info["tokens"][:length]),
                "score_prompt_identity_invalid",
            )
            maximum_error = max(
                maximum_error,
                score_values(response, info["reference"][length - 1], candidates),
            )
            responses.append(response)
        require(
            responses[0]["logits"] == responses[1]["logits"] == responses[3]["logits"],
            "score_replay_changed",
        )
        bad = {**request(3), "candidate_token_ids": [candidates[0], candidates[0]]}
        failed = await worker.request(bad)
        require(
            failed.get("error") == "invalid_scoring_request",
            "score_invalid_request_accepted",
        )
        recovered = await worker.request(request(3))
        require(
            recovered.get("cache_hit") is False, "score_failure_did_not_clear_cache"
        )
        maximum_error = max(
            maximum_error, score_values(recovered, info["reference"][2], candidates)
        )
        require(
            recovered["logits"] == responses[0]["logits"],
            "score_failure_replay_changed",
        )
    return {
        "tokenize": True,
        "reference_logits_close": True,
        "maximum_absolute_logit_error": maximum_error,
        "consecutive_repeat_hits": True,
        "intervening_request_invalidates": True,
        "invalid_request_clears_cache": True,
    }


async def run(args) -> dict:
    source = args.fixture.resolve(strict=True)
    require(source.is_dir() and not source.is_symlink(), "fixture_directory_invalid")
    generate = args.generate_worker.resolve(strict=True)
    score = args.score_worker.resolve(strict=True)
    require(generate.is_file() and score.is_file(), "worker_binary_invalid")
    # The repository's ignored runtime directory keeps the temporary synthetic
    # tokenizer and model copy separate from each source-bound oracle fixture.
    runtime = Path(__file__).resolve().parents[1] / "runtime"
    runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="hybrid-worker-check-", dir=runtime
    ) as temporary:
        os.chmod(temporary, 0o700)
        destination = Path(temporary) / "model"
        info = prepare_fixture(source, destination)
        generation = await check_generator(generate, destination, info)
        scoring = await check_scorer(score, destination, info)
    return {
        "passed": True,
        "family": info["family"],
        "private_synthetic_fixture": True,
        "weight_bytes": info["weight_bytes"],
        "model_config_preserved": True,
        "supplemental_tokenizer_metadata": True,
        "source_config_sha256": info["source_config_sha256"],
        "source_weights_sha256": info["source_weights_sha256"],
        "reference_sha256": info["reference_sha256"],
        "prepared_tokenizer_sha256": info["prepared_tokenizer_sha256"],
        "generate_worker_sha256": digest(generate),
        "score_worker_sha256": digest(score),
        "generation": generation,
        "scoring": scoring,
        "workers_exited_cleanly": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--generate-worker", type=Path, required=True)
    parser.add_argument("--score-worker", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = asyncio.run(asyncio.wait_for(run(args), 240))
        print(json.dumps(result, sort_keys=True))
        return 0
    except CheckFailed as failure:
        print(json.dumps({"passed": False, "check": str(failure)}, sort_keys=True))
        return 1
    except (Exception, KeyboardInterrupt):  # noqa: BLE001 - never expose worker payloads or exception details
        print(
            json.dumps(
                {"passed": False, "check": "hybrid_worker_integration_failed"},
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
