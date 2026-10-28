#!/usr/bin/env python3
"""Validate private Gemma image fixtures through the real native worker protocol.

Only aggregate checks, timings and hashes are emitted. Images, prompts,
generated text and token vectors remain in private fixtures and pipes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

from stark_convert.media import ImageData, PrivateMediaStore
from stark_convert.vision_runtime import GEMMA4_FAMILY_PROFILE, image_input_contract
from test_hybrid_workers import CheckFailed, digest, require, worker_process


def token_hash(tokens):
    return hashlib.sha256(
        b"".join(struct.pack("<I", value) for value in tokens)
    ).hexdigest()


async def run(args):
    fixture = args.fixture.resolve(strict=True)
    manifest_path = fixture / "manifest.json"
    require(
        manifest_path.stat().st_size <= 4 * 1024 * 1024,
        "fixture_manifest_exceeds_limit",
    )
    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"]
    require(isinstance(cases, list) and 2 <= len(cases) <= 8, "invalid_case_count")
    model = args.model.resolve(strict=True)
    executable = args.worker.resolve(strict=True)
    results = []
    observed = []
    with PrivateMediaStore() as media:
        arguments = [
            "--model-path",
            str(model),
            "--vision-staging-root",
            str(media.root),
        ]

        async def generate(worker, case, *, repeat=False):
            name = case["image"]
            require(
                isinstance(name, str) and Path(name).name == name,
                "invalid_image_fixture_name",
            )
            path = (args.images_root or fixture) / name
            require(
                digest(path) == case["image_sha256"], "image_fixture_identity_changed"
            )
            options = {
                "context_tokens": case["context_tokens"],
                "max_new_tokens": case["max_new_tokens"],
            }
            with media.stage(ImageData(path.read_bytes(), "image/png")) as reference:
                prepared = await worker.request(
                    {
                        "operation": "prepare_image",
                        "image": reference,
                        "rendered_prompt": case["rendered_prompt"],
                        **options,
                    }
                )
                require("error" not in prepared, "native_image_preparation_failed")
                require(
                    prepared["image_tokens"] == case["image_tokens"],
                    "image_token_count_mismatch",
                )
                require(prepared["grid"] == case["grid"], "image_grid_mismatch")
                require(
                    prepared["processed_dimensions"] == case["processed_dimensions"],
                    "image_resize_mismatch",
                )
                require(
                    prepared["input_tokens"] == len(case["tokens"]),
                    "expanded_prompt_count_mismatch",
                )
                identifier = prepared["prepared_id"]
                reply = await worker.request(
                    {
                        "operation": "generate_image",
                        "prepared_id": identifier,
                        **options,
                    }
                )
                require(
                    "error" not in reply
                    and reply.get("finish_reason") in {"eos", "length", "context"},
                    "native_image_generation_failed",
                )
                require(
                    reply["input_tokens"] == prepared["input_tokens"],
                    "image_generation_count_changed",
                )
                invalid = await worker.request(
                    {
                        "operation": "generate_image",
                        "prepared_id": identifier,
                        **options,
                    }
                )
                require("error" in invalid, "prepared_image_was_reused")
                cleared = await worker.request({"operation": "discard_image"})
                require(type(cleared.get("discarded")) is bool, "image_discard_failed")
            require(not list(media.root.iterdir()), "image_staging_not_cleaned")
            if not repeat:
                actual = reply["generated_token_ids"]
                expected = case["reference_tokens"]
                results.append(
                    {
                        "case": case["name"],
                        "input_tokens": reply["input_tokens"],
                        "image_tokens": prepared["image_tokens"],
                        "output_tokens": len(actual),
                        "reference_output_tokens": len(expected),
                        "reference_greedy_exact": actual == expected,
                        "output_sha256": token_hash(actual),
                        "reference_sha256": token_hash(expected),
                        "finish_reason": reply["finish_reason"],
                        "prepared_single_use": True,
                        "staging_cleanup": True,
                        "prefill_seconds": reply.get("prefill_seconds"),
                        "decode_tokens_per_second": reply.get(
                            "output_tokens_per_second"
                        ),
                        "dominant_color_answer": case["name"] not in {"red", "blue"}
                        or case["name"] in reply["text"].lower(),
                    }
                )
            return reply["generated_token_ids"]

        async with worker_process(executable, arguments) as worker:
            ready = await worker.reply(timeout=180)
            require(ready.get("ready") is True, "image_worker_not_ready")
            capabilities = image_input_contract(ready.get("image_input"))
            require(
                capabilities is not None
                and capabilities["family_profile"] == GEMMA4_FAMILY_PROFILE,
                "gemma_image_capability_missing",
            )
            for case in cases:
                first = await generate(worker, case)
                require(
                    first == await generate(worker, case, repeat=True),
                    "fresh_image_replay_changed",
                )
                observed.append(first)
            with media.stage(
                ImageData(b"\x89PNG\r\n\x1a\n\0", "image/png")
            ) as reference:
                failed = await worker.request(
                    {
                        "operation": "prepare_image",
                        "image": reference,
                        "rendered_prompt": cases[0]["rendered_prompt"],
                        "context_tokens": 1024,
                        "max_new_tokens": 16,
                    }
                )
                require(
                    failed.get("error") == "invalid_image_request",
                    "malformed_image_not_recoverable",
                )
            require(
                observed[0] == await generate(worker, cases[0], repeat=True),
                "invalid_image_recovery_changed",
            )
        async with worker_process(executable, arguments) as worker:
            require(
                (await worker.reply(timeout=180)).get("ready") is True,
                "image_restart_not_ready",
            )
            require(
                observed[0] == await generate(worker, cases[0], repeat=True),
                "image_restart_replay_changed",
            )
        require(not list(media.root.iterdir()), "final_image_staging_not_cleaned")
    counterfactual_changed = observed[0] != observed[1]
    return {
        "passed": all(
            row["reference_greedy_exact"] and row["dominant_color_answer"]
            for row in results
        )
        and counterfactual_changed,
        "cases": results,
        "red_blue_counterfactual_changes_output": counterfactual_changed,
        "fresh_replay_exact": True,
        "restart_replay_exact": True,
        "malformed_image_recovery": True,
        "staging_cleanup": True,
        "reference_manifest_sha256": digest(manifest_path),
        "worker_sha256": digest(executable),
        "model_config_sha256": digest(model / "config.json"),
        "artifact_sha256": capabilities["artifact_sha256"],
        "runtime_identity": capabilities["runtime_identity"],
    }


def client_check(args):
    """Exercise the shipped client and its cold/cached local image checks."""
    from stark_convert.chat import Worker

    observations = []
    with tempfile.TemporaryDirectory(
        prefix="gemma4-client-check-", dir=args.fixture.parent
    ) as temporary:
        receipts = Path(temporary) / "receipts"
        for expected_cached in (False, True):
            worker = Worker(
                args.worker.resolve(strict=True), args.model.resolve(strict=True), 180
            )
            worker.image_check_dir = receipts
            try:
                ready = worker.read()
                require(
                    ready.get("ready") is True and worker.images,
                    "client_image_capability_missing",
                )
                require(
                    worker.image_verification == "locally_tested",
                    "client_local_image_check_failed",
                )
                require(
                    worker.image_check_cached is expected_cached,
                    "client_check_cache_binding_failed",
                )
                require(
                    worker.media is not None and not list(worker.media.root.iterdir()),
                    "client_staging_not_clean",
                )
                observations.append(
                    {
                        "locally_tested": True,
                        "cached": worker.image_check_cached,
                        "family_profile": worker.image_input["family_profile"],
                    }
                )
            finally:
                worker.close()
    return {
        "passed": True,
        "client_checks": observations,
        "staging_cleanup": True,
        "worker_sha256": digest(args.worker),
        "model_config_sha256": digest(args.model / "config.json"),
    }


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--images-root", type=Path)
    parser.add_argument("--client-check", action="store_true")
    args = parser.parse_args()
    try:
        report = (
            client_check(args)
            if args.client_check
            else asyncio.run(asyncio.wait_for(run(args), 1800))
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if report["passed"] else 1
    except CheckFailed as failure:
        print(json.dumps({"passed": False, "check": str(failure)}))
    except (Exception, KeyboardInterrupt) as failure:  # noqa: BLE001 - no private exception messages or dumps
        print(
            json.dumps(
                {
                    "passed": False,
                    "check": "image_worker_validation_failed",
                    "exception_type": type(failure).__name__,
                }
            )
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
