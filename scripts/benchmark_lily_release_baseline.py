"""Record the fixed busy-host Lily matrix for the llama.cpp 1.0.0 assessment.

Reuses the paired MTP harness. Persists aggregate data, identities and hashes
only. Pass the same frozen corpus to baseline and candidate workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import benchmark_qwen38_mtp as paired
from stark_convert.chat import Worker


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def identity(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ino]


def describe(values):
    median = statistics.median(values)
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    relative_range = (max(values) - min(values)) / median
    return {
        "n": len(values), "min": min(values), "median": median,
        "max": max(values), "iqr": quartiles[2] - quartiles[0],
        "mad": statistics.median(abs(value - median) for value in values),
        "relative_range": relative_range,
        "future_screening_hurdle": max(0.10, relative_range),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cell", help="Run one declared matrix cell by its recorded name.")
    args = parser.parse_args()
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    removed = []
    for name in tuple(os.environ):
        if name.startswith(("STARK_", "UZU_", "LILY_", "GGML_")):
            removed.append(name)
            os.environ.pop(name)
    frozen = args.output_dir / "frozen-corpus.rs"
    shutil.copyfile(args.corpus, frozen)
    repo = Path(__file__).resolve().parents[1]
    files = sorted(args.model.glob("*.safetensors"))
    files.extend(args.model / name for name in (
        "config.json", "model.safetensors.index.json", "tokenizer.json",
        "tokenizer_config.json", "preprocessor_config.json", "processor_config.json",
        "native-source.json", "chat_template.jinja", "stark-image-artifact.json",
    ) if (args.model / name).is_file())
    initial = {path.name: identity(path) for path in files}
    manifest = {
        "schema": "stark-lily-release-baseline-v1",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "worker_sha256": paired.sha256_file(args.worker),
        "runner_sha256": paired.sha256_file(Path(__file__)),
        "paired_harness_sha256": paired.sha256_file(Path(paired.__file__)),
        "cargo_lock_sha256": paired.sha256_file(repo / "Cargo.lock"),
        "corpus_sha256": paired.sha256_file(frozen),
        "environment_names_removed": sorted(removed),
        "host_policy": "observe all requests under existing load; no quiet admission or host exclusions",
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "hw.model", "hw.memsize", "hw.ncpu", "machdep.cpu.brand_string"],
            text=True).splitlines(),
        "os": subprocess.check_output(["sw_vers"], text=True).splitlines(),
        "model_file_identities": initial,
        "model_file_sha256": {}, "cells": [],
    }
    write_json(args.output_dir / "identity.json", manifest)
    print(json.dumps({"stage": "hashing_artifact", "files": len(files)}), flush=True)
    for path in files:
        manifest["model_file_sha256"][path.name] = paired.sha256_file(path)
        if identity(path) != initial[path.name]:
            raise ValueError("artifact_changed_during_hash")
    manifest["model_content_manifest_sha256"] = hashlib.sha256(json.dumps(
        manifest["model_file_sha256"], sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    manifest["artifact_bytes"] = sum(item[0] for item in initial.values())
    manifest["weight_shards"] = len(list(args.model.glob("*.safetensors")))
    write_json(args.output_dir / "identity.json", manifest)
    # Counterbalance short/long cells and avoid monotonic draft-depth order.
    matrix = [(512, 20, 2), (4096, 20, 1), (512, 20, 3), (4096, 0, 2),
              (512, 20, 1), (4096, 20, 3), (512, 0, 2), (4096, 20, 2)]
    if args.cell:
        matrix = [item for item in matrix if f"input{item[0]}-topk{item[1]}-drafts{item[2]}" == args.cell]
        if not matrix:
            raise ValueError("unknown_matrix_cell")
    manifest["declared_cells"] = [f"input{n}-topk{k}-drafts{d}" for n, k, d in matrix]

    def best_effort_screen(*screen_args):
        from benchmark_glm_mlx_serve import screen_attempt

        try:
            return screen_attempt(*screen_args)
        except ValueError as error:
            if str(error) != "host_observations_missing":
                raise
            return {"eligible": False, "observation_status": "missing", "samples": 0}

    for length, top_k, drafts in matrix:
        cell = f"input{length}-topk{top_k}-drafts{drafts}"
        telemetry = {"requests": []}

        class ObservedWorker(Worker):
            def __init__(self, executable, model, timeout, *, env=None):
                started = time.monotonic()
                self._baseline_launch_at = started
                super().__init__(executable, model, timeout)
                telemetry["worker_constructor_seconds"] = time.monotonic() - started
                self.image_check_dir = args.output_dir / f"{cell}-image-checks"

            def read(self, *read_args, **read_kwargs):
                # The native launch already received --vision-staging-root.
                # Keep this text baseline independent of Python's additional
                # image-execution qualification (which runs image requests).
                # Hiding the host media handle only during initial readiness
                # changes no native load option and makes no image quality claim.
                media = self.media
                initial_read = not self._ready_seen
                if initial_read:
                    self.media = None
                try:
                    reply = super().read(*read_args, **read_kwargs)
                finally:
                    if initial_read:
                        self.media = media
                if reply.get("ready") is True:
                    telemetry["native_ready_seconds"] = time.monotonic() - self._baseline_launch_at
                    telemetry["ready"] = {key: reply.get(key) for key in (
                        "mode", "image_input", "mtp", "prefix_cache", "max_context_tokens",
                    )}
                    telemetry["image_execution_qualification"] = "not_run_text_baseline"
                    image = reply.get("image_input")
                    if not isinstance(image, dict) or any(image.get(key) != value for key, value in {
                        "version": "mediaref-v1",
                        "family_profile": "qwen38-lily-retained-bf16-vision-v1",
                        "model_source": "fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl",
                        "model_revision": "0f94842e3682b49acace4daeb4cf2e65a6f94f83",
                    }.items()):
                        raise ValueError("required_image_ready_missing")
                return reply

            def request(self, payload, *request_args, **request_kwargs):
                reply = super().request(payload, *request_args, **request_kwargs)
                if payload.get("operation") == "generate" and "input_token_ids" in payload:
                    if reply.get("reused_input_tokens", 0) != 0:
                        raise ValueError("unexpected_prefix_reuse")
                    if reply.get("input_tokens") != len(payload["input_token_ids"]):
                        raise ValueError("unexpected_input_count")
                    telemetry["requests"].append({key: reply.get(key) for key in (
                        "input_tokens", "reused_input_tokens", "finish_reason",
                    )})
                return reply

        options = argparse.Namespace(
            model=args.model, worker=args.worker, corpus=frozen,
            output=args.output_dir / f"{cell}.json", input_tokens=length,
            tokens=256, blocks=8, drafts=drafts, top_k=top_k, top_p=0.95,
            seed=42, screened=False, observe_host=True, quiet_scope="request",
            checkpoint_receipt=None,
        )
        print(json.dumps({"stage": "starting_cell", "cell": cell}), flush=True)
        try:
            report = paired.run(options, worker_factory=ObservedWorker, screen_fn=best_effort_screen)
            summary = {"cell": cell, "passed": report["passed"], "routes": {}}
            for route in ("off", "on"):
                items = [item for item in report["requests"] if item["route"] == route]
                summary["routes"][route] = {
                    "decode_tokens_per_second": describe([
                        (item["sampled_tokens"] - 1) / item["decode_seconds"] for item in items]),
                    "useful_output_tokens_per_second": describe([
                        max(0, item["output_tokens"] - 1) / item["decode_seconds"] for item in items]),
                    "wall_seconds": describe([item["wall_seconds"] for item in items]),
                    "prefill_seconds": describe([item["prefill_seconds"] for item in items]),
                    "pooled_decode_tokens_per_second": report["route_summary"][route]["pooled_decode_tokens_per_second"],
                }
            summary["host_screen_passes"] = sum(item["host_screen"]["eligible"] for item in report["requests"])
            summary["requests"] = len(report["requests"])
        except Exception as error:
            # Exception messages can contain native or model details; persist only type.
            summary = {"cell": cell, "passed": False, "failure_type": type(error).__name__}
        write_json(args.output_dir / f"{cell}-telemetry.json", telemetry)
        summary["artifact_identity_unchanged"] = all(identity(path) == initial[path.name] for path in files)
        manifest["cells"].append(summary)
        write_json(args.output_dir / "identity.json", manifest)
        print(json.dumps({"stage": "completed_cell", **summary}), flush=True)
        if not summary["passed"] or not summary["artifact_identity_unchanged"]:
            break
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["worker_unchanged"] = paired.sha256_file(args.worker) == manifest["worker_sha256"]
    manifest["completed"] = len(manifest["cells"]) == len(matrix) and all(
        cell["passed"] and cell["artifact_identity_unchanged"] for cell in manifest["cells"])
    write_json(args.output_dir / "identity.json", manifest)
    return 0 if manifest["completed"] and manifest["worker_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
