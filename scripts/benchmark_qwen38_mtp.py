"""Paired same-worker benchmark for Qwen3.8 MTP exact-sampling routes.

The report stores hashes, counts, route acknowledgements and timings only.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
import time
from pathlib import Path

from stark_convert.chat import ChatError, Worker

EXACT_VERIFIERS = frozenset({"serial-decode-exact-v1", "batched-decode-exact-v1"})
OFFICIAL_REPOSITORY = "ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit"
OFFICIAL_REVISION = "788c07a8725642bccf13a0a3b8a21954ab84b4f2"
OFFICIAL_MANIFEST_SHA256 = (
    "11246ebbea8a24a988517ebeb59623b857df89120e5ae3becd48807194542ef6"
)
OFFICIAL_RECEIPT_SHA256 = (
    "e112599b932abf197011afe5eb91bc1f8a2bb55716eda470b6ee4ef0fcb275e8"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def exact_prefix(ids: list[int], count: int) -> list[int]:
    if not ids or count < 1:
        raise ValueError("corpus_tokenization_empty")
    return (ids * ((count + len(ids) - 1) // len(ids)))[:count]


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.input_tokens < 1
        or not 2 <= args.tokens <= 4096
        or not 4 <= args.blocks <= 16
        or args.blocks % 4 != 0
        or args.drafts not in (1, 2, 3)
    ):
        raise ValueError("invalid_benchmark_limits")
    if args.input_tokens + args.tokens > 16384:
        raise ValueError("context_limit_exceeded")
    if (
        args.top_k < 0
        or not 0 < args.top_p <= 1
        or not math.isfinite(args.top_p)
        or not 0 <= args.seed <= 2**64 - 1
    ):
        raise ValueError("invalid_sampling_parameters")


def sampling_options(args: argparse.Namespace) -> dict:
    value = {"mode": "stochastic", "temperature": 1.0, "top_p": args.top_p}
    if args.top_k:
        value["top_k"] = args.top_k
    return value


def checkpoint_metadata(args: argparse.Namespace) -> dict:
    """Bind a supported checkpoint without inferring provenance from its path.

    The official snapshot reuses a pinned complete payload-hash receipt only
    while every file retains its recorded size, nanosecond mtime and inode.
    This is local receipt revalidation, not a new hash of the weight payloads.
    Configuration and tensor-index bytes are additionally hashed on this run.
    """
    try:
        config = json.loads((args.model / "config.json").read_text())
    except (OSError, ValueError):
        raise ValueError("model_metadata_unavailable") from None
    if not isinstance(config, dict) or config.get("model_type") != "qwen4_exp":
        raise ValueError("unexpected_qwen38_checkpoint")
    receipt_path = getattr(args, "checkpoint_receipt", None)
    if receipt_path is None:
        try:
            receipt_path = args.model / "native-source.json"
            receipt = json.loads(receipt_path.read_text())
        except (OSError, ValueError):
            raise ValueError("model_metadata_unavailable") from None
        source = receipt.get("weights_source") if isinstance(receipt, dict) else None
        if (
            not isinstance(config.get("lily"), dict)
            or not isinstance(config["lily"].get("mtp"), dict)
            or not isinstance(source, dict)
            or source.get("repo") != "fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl"
            or source.get("revision") != "0f94842e3682b49acace4daeb4cf2e65a6f94f83"
        ):
            raise ValueError("unexpected_qwen38_checkpoint")
        return {
            "source_receipt_sha256": sha256_file(receipt_path),
            "checkpoint": {"repo": source["repo"], "revision": source["revision"]},
        }

    if sha256_file(receipt_path) != OFFICIAL_RECEIPT_SHA256:
        raise ValueError("checkpoint_receipt_hash_mismatch")
    try:
        receipt = json.loads(receipt_path.read_text())
        index = json.loads((args.model / "model.safetensors.index.json").read_text())
    except (OSError, ValueError):
        raise ValueError("model_metadata_unavailable") from None
    text_config = config.get("text_config")
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != "stark-shared-checkpoint-payload-hashes-v1"
        or receipt.get("completed") is not True
        or receipt.get("all_manifest_lfs_payloads_verified") is not True
        or receipt.get("repository") != OFFICIAL_REPOSITORY
        or receipt.get("revision") != OFFICIAL_REVISION
        or receipt.get("manifest_sha256") != OFFICIAL_MANIFEST_SHA256
        or not isinstance(text_config, dict)
        or not isinstance(text_config.get("mtp"), dict)
        or not isinstance(index, dict)
        or not isinstance(index.get("weight_map"), dict)
        or sum(name.startswith("language_model.mtp.") for name in index["weight_map"])
        != 72
    ):
        raise ValueError("unexpected_qwen38_checkpoint")
    snapshot = receipt.get("checkpoint_snapshot")
    hashes = receipt.get("all_file_sha256")
    if (
        not isinstance(snapshot, dict)
        or not isinstance(hashes, dict)
        or len(snapshot) != 113
        or snapshot.keys() != hashes.keys()
    ):
        raise ValueError("checkpoint_receipt_inventory_invalid")
    for name, identity in snapshot.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("checkpoint_receipt_inventory_invalid")
        stat = (args.model / relative).stat()
        if [stat.st_size, stat.st_mtime_ns, stat.st_ino] != identity:
            raise ValueError("checkpoint_snapshot_identity_changed")
    for name in ("config.json", "model.safetensors.index.json"):
        if sha256_file(args.model / name) != hashes.get(name):
            raise ValueError("checkpoint_metadata_hash_mismatch")
    return {
        "checkpoint_receipt_sha256": OFFICIAL_RECEIPT_SHA256,
        "checkpoint": {
            "repo": OFFICIAL_REPOSITORY,
            "revision": OFFICIAL_REVISION,
            "manifest_sha256": OFFICIAL_MANIFEST_SHA256,
            "files": len(snapshot),
            "identity_policy": "prior_complete_hash_receipt_matching_size_mtime_inode_v1",
            "payloads_rehashed_this_run": False,
            "metadata_rehashed_this_run": [
                "config.json",
                "model.safetensors.index.json",
            ],
        },
    }


def _valid_ids(value) -> bool:
    return isinstance(value, list) and all(
        type(token) is int and token >= 0 for token in value
    )


def _validate_result(
    result: dict,
    ready: dict,
    args: argparse.Namespace,
    route: str,
    drafts: int,
    vocab_size: int,
) -> dict:
    if not isinstance(result, dict) or "error" in result:
        raise ValueError("generation_failed")
    generated = result.get("generated_token_ids")
    sampled = result.get("sampled_tokens")
    stop = result.get("stop_token_id")
    stopped = type(stop) is int and stop in ready.get("eos_token_ids", [])
    finish = result.get("finish_reason")
    if (
        not _valid_ids(generated)
        or any(token >= vocab_size for token in generated)
        or type(sampled) is not int
        or sampled != len(generated) + int(stopped)
        or sampled != args.tokens
        or finish not in {"length", "eos"}
        or (finish == "eos") != stopped
        or (not stopped and stop is not None)
    ):
        raise ValueError("generation_count_mismatch")
    ack = result.get("mtp")
    if not isinstance(ack, dict) or ack.get("sampling") != "exact-rejection-v1":
        raise ValueError("mtp_acknowledgement_unavailable")
    verifier = ack.get("verification")
    if (
        not isinstance(verifier, str)
        or verifier not in EXACT_VERIFIERS
        or verifier != ready.get("mtp", {}).get("verification")
    ):
        raise ValueError("mtp_verifier_acknowledgement_mismatch")
    expected = 0 if route == "off" else drafts
    if ack.get("drafts") != expected or ack.get("enabled") is not (expected > 0):
        raise ValueError("mtp_route_acknowledgement_mismatch")
    stats = ack.get("stats")
    if not isinstance(stats, dict):
        raise TypeError("invalid_mtp_statistics")
    counts = {}
    for key in ("rounds", "proposed", "accepted", "verified_rows"):
        value = stats.get(key)
        if type(value) is not int or value < 0:
            raise ValueError("invalid_mtp_statistics")
        counts[key] = value
    if (
        counts["accepted"] > counts["proposed"]
        or counts["verified_rows"] < counts["accepted"]
    ):
        raise ValueError("invalid_mtp_statistics")
    timings = {
        key: stats.get(key)
        for key in ("draft_seconds", "verify_seconds", "rollback_seconds")
    }
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        for key, value in timings.items()
    ):
        raise ValueError("invalid_mtp_timings")
    if route == "off" and any((*counts.values(), *timings.values())):
        raise ValueError("disabled_mtp_has_activity")
    if (
        route == "on"
        and args.tokens > drafts + 1
        and (counts["rounds"] == 0 or counts["proposed"] == 0)
    ):
        raise ValueError("enabled_mtp_has_no_draft_activity")
    decode_seconds = result.get("decode_seconds")
    if (
        not isinstance(decode_seconds, (int, float))
        or isinstance(decode_seconds, bool)
        or not math.isfinite(decode_seconds)
        or decode_seconds <= 0
    ):
        raise ValueError("decode_timing_unavailable")
    item = {
        "route": route,
        "requested_drafts": expected,
        "sampled_tokens": sampled,
        "output_tokens": len(generated),
        "output_sha256": ids_sha256(generated),
        "decode_seconds": decode_seconds,
        "mtp": {
            **counts,
            "enabled": ack["enabled"],
            "drafts": ack["drafts"],
            "sampling": ack["sampling"],
            "verification": ack["verification"],
            "timings_seconds": timings,
        },
    }
    for key in ("prefill_seconds", "output_tokens_per_second", "generation_seconds"):
        value = result.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ):
            item[key] = value
    return item


def _atomic_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link creation is atomic and fails if the destination exists.
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


@contextlib.contextmanager
def _profile_env_removed():
    previous = os.environ.pop("STARK_QWEN38_PROFILE_OUTPUT", None)
    try:
        yield
    finally:
        if previous is not None:
            os.environ["STARK_QWEN38_PROFILE_OUTPUT"] = previous


def run(args: argparse.Namespace, worker_factory=Worker, *, rng=None) -> dict:
    validate_args(args)
    if args.output.exists():
        raise ValueError("output_exists")
    corpus_bytes = args.corpus.read_bytes()
    try:
        corpus_text = corpus_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("corpus_not_utf8") from None
    if not corpus_text:
        raise ValueError("corpus_empty")

    report = {
        "schema_version": 3,
        "scope": "qwen38_same_worker_paired_mtp_sampling",
        "sampling": sampling_options(args),
        "seed": args.seed,
        "drafts": args.drafts,
        "worker_sha256": sha256_file(args.worker),
        "configuration_sha256": sha256_file(args.model / "config.json"),
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "requested_input_tokens": args.input_tokens,
        "requested_output_tokens": args.tokens,
        "context_tokens": args.input_tokens + args.tokens,
        "blocks": args.blocks,
        "block_order": "randomized_abba",
        "warmups_per_route": 1,
        "requests": [],
    }
    report.update(checkpoint_metadata(args))
    env = dict(os.environ)
    env.pop("STARK_QWEN38_PROFILE_OUTPUT", None)
    report["environment_policy"] = {"profile_output_removed": True}
    if worker_factory is Worker:
        # Worker inherits the parent environment internally. Remove profiling
        # only while it starts its subprocess, then restore the caller state.
        with _profile_env_removed():
            worker = worker_factory(args.worker.resolve(), args.model.resolve(), 3600)
    else:
        worker = worker_factory(
            args.worker.resolve(), args.model.resolve(), 3600, env=env
        )
    try:
        ready = worker.read()
        if ready.get("ready") is not True or ready.get("mode") != "text-generation-v1":
            raise ValueError("worker_not_ready")
        mtp = ready.get("mtp")
        if not isinstance(mtp, dict) or mtp.get("supported") is not True:
            raise ValueError("mtp_not_supported")
        if mtp.get("sampling") != "exact-rejection-v1":
            raise ValueError("mtp_sampling_unavailable")
        if (
            not isinstance(mtp.get("verification"), str)
            or mtp["verification"] not in EXACT_VERIFIERS
        ):
            raise ValueError("mtp_verifier_unavailable")
        report["verification"] = mtp["verification"]
        max_drafts = mtp.get("max_drafts")
        if type(max_drafts) is not int or max_drafts < args.drafts:
            raise ValueError("mtp_drafts_unavailable")
        vocab_size = ready.get("vocab_size")
        if type(vocab_size) is not int or vocab_size < 1:
            raise ValueError("worker_vocabulary_unavailable")
        worker_context = ready.get("max_context_tokens", 16384)
        if (
            type(worker_context) is not int
            or args.input_tokens + args.tokens > worker_context
        ):
            raise ValueError("worker_context_limit_exceeded")
        tokenized = worker.request(
            {"operation": "tokenize", "text": corpus_text, "add_special_tokens": False}
        )
        tokenized_ids = tokenized.get("token_ids")
        if (
            "error" in tokenized
            or not _valid_ids(tokenized_ids)
            or any(token >= vocab_size for token in tokenized_ids)
        ):
            raise ValueError("corpus_tokenization_failed")
        ids = exact_prefix(tokenized_ids, args.input_tokens)
        report["input_tokens"] = len(ids)
        report["corpus_token_count"] = len(tokenized_ids)
        report["input_sha256"] = ids_sha256(ids)
        request_id = 0

        def request(route: str) -> dict:
            nonlocal request_id
            request_id += 1
            requested = 0 if route == "off" else args.drafts
            payload = {
                "operation": "generate",
                "input_token_ids": ids,
                "context_tokens": args.input_tokens + args.tokens,
                "max_new_tokens": args.tokens,
                "sampling": report["sampling"],
                "seed": args.seed,
                "mtp_drafts": requested,
                "request_id": request_id,
            }
            started = time.monotonic()
            result = worker.request(payload)
            elapsed = time.monotonic() - started
            item = _validate_result(result, ready, args, route, args.drafts, vocab_size)
            item["wall_seconds"] = elapsed
            item["request_id"] = request_id
            return item

        for route in ("off", "on"):
            warmup = request(route)
            report.setdefault("warmups", []).append(
                {
                    key: warmup[key]
                    for key in (
                        "route",
                        "sampled_tokens",
                        "output_tokens",
                        "output_sha256",
                        "mtp",
                    )
                }
            )
        order_rng = rng or random.Random(args.seed)
        first = order_rng.choice(("off", "on"))
        other = "on" if first == "off" else "off"
        abba = (first, other, other, first)
        for block in range(args.blocks):
            leading = abba[block % 4]
            order = [leading, "on" if leading == "off" else "off"]
            pair = {"block": block, "order": order, "routes": {}}
            for route in order:
                item = request(route)
                item["block"] = block
                pair["routes"][route] = item
                report["requests"].append(item)
            if any(
                pair["routes"][route]["sampled_tokens"] != args.tokens
                for route in order
            ):
                raise ValueError("unequal_output_counts")
            report.setdefault("pairs", []).append(pair)

        report["route_summary"] = {}
        for route in ("off", "on"):
            items = [item for item in report["requests"] if item["route"] == route]
            route_hashes = [item["output_sha256"] for item in items]
            route_hashes.extend(
                item["output_sha256"]
                for item in report["warmups"]
                if item["route"] == route
            )
            seconds = sum(item["decode_seconds"] for item in items)
            steps = sum(item["sampled_tokens"] - 1 for item in items)
            report["route_summary"][route] = {
                "requests": len(items),
                "pooled_decode_steps": steps,
                "pooled_decode_seconds": seconds,
                "pooled_decode_tokens_per_second": steps / seconds if seconds else None,
                "median_wall_seconds": statistics.median(
                    item["wall_seconds"] for item in items
                ),
                "deterministic_within_route": len(set(route_hashes)) == 1,
                "mtp_totals": {
                    key: sum(item["mtp"][key] for item in items)
                    for key in ("rounds", "proposed", "accepted", "verified_rows")
                },
            }
        report["paired_block_rates"] = [
            {
                "block": pair["block"],
                "off_tok_s": (pair["routes"]["off"]["sampled_tokens"] - 1)
                / pair["routes"]["off"]["decode_seconds"],
                "on_tok_s": (pair["routes"]["on"]["sampled_tokens"] - 1)
                / pair["routes"]["on"]["decode_seconds"],
            }
            for pair in report["pairs"]
        ]
        report["cross_route_hashes_equal"] = (
            report["route_summary"]["off"]["deterministic_within_route"]
            and report["route_summary"]["on"]["deterministic_within_route"]
            and len(
                {
                    item["output_sha256"]
                    for item in report["requests"]
                    if item["route"] == "off"
                }
                | {
                    item["output_sha256"]
                    for item in report["requests"]
                    if item["route"] == "on"
                }
            )
            == 1
        )
        report["passed"] = all(
            value["deterministic_within_route"]
            for value in report["route_summary"].values()
        )
        _atomic_report(args.output, report)
        return report
    finally:
        worker.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-receipt",
        type=Path,
        help="Pinned complete payload-hash receipt for the official mixed-affine snapshot; its local file identities must still match.",
    )
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-tokens", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--drafts", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--top-k", type=int, default=20, help="Zero disables top-k filtering."
    )
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (OSError, ValueError, KeyError, TypeError, ChatError):
        print(json.dumps({"passed": False, "stage": "mtp_benchmark_failed"}))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
