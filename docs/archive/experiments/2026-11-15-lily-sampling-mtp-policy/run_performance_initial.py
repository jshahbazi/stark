"""Run the sealed private 64-request Lily top-k MTP confirmation matrix.

Only hashes, counts, timings and sanitized host metrics are persisted. Prompt
text, token IDs, generated text and logits remain in worker memory.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_generation_prefix_cache import Worker
from benchmark_qwen38_mtp import exact_prefix, ids_sha256, sha256_file
from benchmark_clean_qwen import Monitor, paging

OUT = ROOT / "runtime/lily-topk-policy/performance-private"
MODEL = Path("/Users/johnshahbazian/Library/Application Support/Stark/models/qwen38-flash-next-lily-q4-xl-0f94842e")
BASE = Path("/Users/johnshahbazian/development/stark-llama-v1-study/runtime/llama-v1-baseline-target/release/stark-generate")
CANDIDATE = ROOT / "runtime/build-topk-candidate/release/stark-generate"
IDENTITY = Path("/Users/johnshahbazian/development/stark-llama-v1-study/runtime/llama-v1-lily-baseline-20261115-v2/identity.json")
BUILD_IDENTITY = ROOT / "runtime/lily-topk-policy/build-identity.json"
NATIVE_PATCH = ROOT / "runtime/lily-topk-policy/native.patch"
PYTHON_PATCH = ROOT / "runtime/lily-topk-policy/python.patch"
CORPUS = ROOT / "runtime/lily-one-draft/results/model-corpus.rs"
HELDOUT = ROOT / "runtime/lily-one-draft/results/held-corpus.rs"
REFERENCE_DIR = ROOT / "runtime/lily-one-draft/results"
REFERENCE_SUMMARY = REFERENCE_DIR / "summary.json"
BASE_SHA256 = "61b9c8cc724407573169eb2abad7f908e4943eead7ac3122eeec3ef841a48640"
BUILD_IDENTITY_SHA256 = "273c56d4470b36bf7257e62b5431b8b55081dbbc1e88a2d08aaeb6e79e93a691"
NATIVE_PATCH_SHA256 = "24bc87a033386e9150a7d264afb546e83cf5dd86b1aee9e9f0bce8d3aa3797c0"
PYTHON_PATCH_SHA256 = "e947c9919583e68d09678c684b7de80ed400a472ff8873150465d7efc6efd307"
REFERENCE_FILES = {
    "primary-input512-top20-out256-seed42": "input512-top20-output256-seed42.json",
    "heldout-input512-top20-out128-seed7": "input512-top20-output128-seed7.json",
    "guard-input512-greedy-out256-seed42": "input512-greedy-output256-seed42.json",
    "guard-input4096-nucleus-out256-seed42": "input4096-nucleus-output256-seed42.json",
}
REFERENCE_FILE_SHA256 = {
    "summary.json": "1a7c6d2f193f4bd13165da50b1ff3f5533f289887a26bed7ee66fcad436a87a7",
    "input512-top20-output256-seed42.json": "c56252ec8996bdb5a845052c2f2cbd45038c1105ea2137fb26976f724a979dbd",
    "input512-top20-output128-seed7.json": "249297c0309cd8793987dfb8c9d37e81ccbb2bbd7c6e865e86b29a063e1fdc52",
    "input512-greedy-output256-seed42.json": "126d1dd4b026aa3f45fc95e2ebd94027c6f34b21f7417f21eb2e6b40ebbe1ee2",
    "input4096-nucleus-output256-seed42.json": "e751a3bda024b48f78283f583b87725dd21529c2acc1b5eef30966129c7131b9",
}
IMAGE_PROFILE = "qwen38-lily-retained-bf16-vision-v1"
REFERENCE_OUTPUTS: dict[str, dict[str, str]] = {}
REFERENCE_INPUTS: dict[str, str] = {}

CELLS = (
    {"name": "primary-input512-top20-out256-seed42", "source": CORPUS,
     "input_tokens": 512, "output_budget": 256, "seed": 42, "kind": "top20"},
    {"name": "heldout-input512-top20-out128-seed7", "source": HELDOUT,
     "input_tokens": 512, "output_budget": 128, "seed": 7, "kind": "top20"},
    {"name": "guard-input512-greedy-out256-seed42", "source": CORPUS,
     "input_tokens": 512, "output_budget": 256, "seed": 42, "kind": "greedy"},
    {"name": "guard-input4096-nucleus-out256-seed42", "source": CORPUS,
     "input_tokens": 4096, "output_budget": 256, "seed": 42, "kind": "nucleus"},
)
SESSIONS = ("before", "candidate", "candidate", "before")


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sampling(kind: str) -> dict:
    if kind == "greedy":
        return {"mode": "greedy"}
    value = {"mode": "stochastic", "temperature": 1.0, "top_p": 0.95}
    if kind == "top20":
        value["top_k"] = 20
    elif kind != "nucleus":
        raise ValueError("unknown_sampling_profile")
    return value


def cell_orders() -> list[list[str]]:
    # Each profile occupies each ordinal position once. The schedule is fixed.
    return [list(order) for order in (
        (CELLS[0]["name"], CELLS[1]["name"], CELLS[2]["name"], CELLS[3]["name"]),
        (CELLS[1]["name"], CELLS[3]["name"], CELLS[0]["name"], CELLS[2]["name"]),
        (CELLS[2]["name"], CELLS[0]["name"], CELLS[3]["name"], CELLS[1]["name"]),
        (CELLS[3]["name"], CELLS[2]["name"], CELLS[1]["name"], CELLS[0]["name"]),
    )]


def current_model_identity(expected: dict) -> bool:
    rows = expected.get("model_file_identities")
    if not isinstance(rows, dict) or not rows:
        return False
    try:
        current_names = {item.name for item in MODEL.iterdir() if item.name != ".cache"}
        if current_names != set(rows):
            return False
        for name, identity in rows.items():
            path = MODEL / name
            if path.is_symlink():
                return False
            stat = path.stat()
            if [stat.st_size, stat.st_mtime_ns, stat.st_ino] != identity:
                return False
        return True
    except OSError:
        return False


def load_reference_hashes() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    if sha256_file(REFERENCE_SUMMARY) != REFERENCE_FILE_SHA256["summary.json"]:
        raise ValueError("reference_summary_seal_mismatch")
    summary = json.loads(REFERENCE_SUMMARY.read_text())
    if (summary.get("completed") is not True or summary.get("worker_sha256") != BASE_SHA256
            or summary.get("model_content_manifest_sha256") !=
            "97c11dd6f7ee5a69e23759b9774acb21a786c48552ca2ede9dac8b6be4b5bc98"
            or summary.get("corpus_sha256") != sha256_file(CORPUS)
            or summary.get("held_corpus_sha256") != sha256_file(HELDOUT)):
        raise ValueError("reference_summary_identity_mismatch")
    outputs: dict[str, dict[str, str]] = {}
    inputs: dict[str, str] = {}
    for cell in CELLS:
        filename = REFERENCE_FILES[cell["name"]]
        path = REFERENCE_DIR / filename
        if sha256_file(path) != REFERENCE_FILE_SHA256[filename]:
            raise ValueError("reference_cell_seal_mismatch")
        prior = json.loads(path.read_text())
        if (prior.get("input_tokens") != cell["input_tokens"]
                or prior.get("output_budget") != cell["output_budget"]
                or prior.get("seed") != cell["seed"]
                or prior.get("sampling") != sampling(cell["kind"])):
            raise ValueError("reference_cell_profile_mismatch")
        summary_cell = next((item for item in summary["cells"]
                             if item.get("name") == prior.get("name")), None)
        if summary_cell is None or summary_cell.get("input_sha256") != prior.get("input_sha256"):
            raise ValueError("reference_input_identity_mismatch")
        hashes = {}
        for depth in (1, 2):
            rows = [row for row in prior.get("requests", [])
                    if row.get("requested_drafts") == depth and row.get("output_sha256")]
            unique = {row["output_sha256"] for row in rows}
            if not rows or len(unique) != 1:
                raise ValueError("reference_depth_hash_unavailable")
            hashes[str(depth)] = unique.pop()
        outputs[cell["name"]] = hashes
        inputs[cell["name"]] = prior["input_sha256"]
    return outputs, inputs


def finite_nonnegative(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def validate_reply(reply: dict, ready: dict, cell: dict, build: str) -> dict:
    if not isinstance(reply, dict) or "error" in reply:
        raise ValueError("generation_failed")
    generated = reply.get("generated_token_ids")
    sampled = reply.get("sampled_tokens")
    stop = reply.get("stop_token_id")
    stopped = type(stop) is int and stop in ready.get("eos_token_ids", [])
    if (not isinstance(generated, list) or len(generated) < 2
            or any(type(x) is not int for x in generated)
            or type(sampled) is not int or sampled != len(generated) + int(stopped)
            or sampled != cell["output_budget"]
            or reply.get("finish_reason") not in {"length", "eos"}
            or (reply["finish_reason"] == "eos") != stopped
            or (not stopped and stop is not None)):
        raise ValueError("generation_count_mismatch")
    ack = reply.get("mtp")
    if (not isinstance(ack, dict) or ack.get("sampling") != "exact-rejection-v1"
            or ack.get("enabled") is not True
            or not isinstance(ack.get("verification"), str)
            or ack.get("verification") != (ready.get("mtp") or {}).get("verification")):
        raise ValueError("mtp_acknowledgement_unavailable")
    depth = ack.get("drafts")
    expected = (2 if build == "before" or cell["kind"] != "top20" else 1)
    if type(depth) is not int or depth != expected:
        raise ValueError("auto_mtp_depth_mismatch")
    stats = ack.get("stats")
    if not isinstance(stats, dict):
        raise ValueError("mtp_statistics_unavailable")
    counts = {key: stats.get(key) for key in ("rounds", "proposed", "accepted", "verified_rows")}
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("invalid_mtp_statistics")
    if (counts["accepted"] > counts["proposed"]
            or counts["verified_rows"] < counts["accepted"]
            or (sampled > depth + 1 and (counts["rounds"] == 0 or counts["proposed"] == 0))):
        raise ValueError("invalid_mtp_statistics")
    timings = {key: stats.get(key) for key in ("draft_seconds", "verify_seconds", "rollback_seconds")}
    if any(not finite_nonnegative(value) for value in timings.values()):
        raise ValueError("invalid_mtp_timings")
    decode = reply.get("decode_seconds")
    generation = reply.get("generation_seconds")
    prefill = reply.get("prefill_seconds")
    if (not finite_nonnegative(decode) or decode <= 0
            or (generation is not None and not finite_nonnegative(generation))):
        raise ValueError("invalid_timing")
    if prefill is not None and not finite_nonnegative(prefill):
        raise ValueError("invalid_timing")
    vocab = ready.get("vocab_size")
    if type(vocab) is not int or any(token < 0 or token >= vocab for token in generated):
        raise ValueError("invalid_generated_ids")
    return {
        "actual_drafts": depth,
        "sampled_tokens": sampled,
        "output_tokens": len(generated),
        "output_sha256": ids_sha256(generated),
        "decode_seconds": decode,
        "generation_seconds": generation,
        "prefill_seconds": prefill,
        "useful_decode_tokens_per_second": (len(generated) - 1) / decode,
        "mtp": {**counts, "timings_seconds": timings,
                "verification": ack.get("verification"), "sampling": ack["sampling"]},
    }


def host_metrics(monitor: Monitor, phase: str, before: dict, after: dict) -> dict:
    rows = [row for row in monitor.rows if phase in row["phases"]]
    if not rows:
        return {"samples": 0, "observation_missing": True,
                "monitor_failures_so_far": monitor.failures,
                "paging_delta": {key: after.get(key, 0) - before.get(key, 0) for key in before}}
    return {
        "samples": len(rows), "observation_missing": False,
        "external_cpu_max_percent": max(row["external_cpu_percent"] for row in rows),
        "max_process_cpu_percent": max(row["max_process_cpu_percent"] for row in rows),
        "max_model_cpu_percent": max(row["max_model_cpu_percent"] for row in rows),
        "thermal_max_state": max(row["thermal_state"] for row in rows),
        "sampling_gap_exceeded": any(row["interval_seconds"] > 2 for row in rows),
        "monitor_failures_so_far": monitor.failures,
        "paging_delta": {key: after[key] - before[key] for key in before},
    }


def invoke(worker: Worker, ids: list[int], cell: dict, *, context: int, maximum: int,
           seed: int, monitor: Monitor, sequence: int, phase: str, build: str,
           session: int, role: str) -> dict:
    monitor.set_phase(phase)
    paging_before = paging()
    started = time.monotonic()
    reply = worker.request({
        "operation": "generate", "input_token_ids": ids,
        "context_tokens": context, "max_new_tokens": maximum,
        "seed": seed, "sampling": sampling(cell["kind"]), "mtp": True,
    })
    wall = time.monotonic() - started
    paging_after = paging()
    if reply.get("input_tokens") != len(ids) or reply.get("reused_input_tokens", 0) != 0:
        raise ValueError("input_or_prefix_reuse_mismatch")
    row = validate_reply(reply, worker.ready, cell, build)
    if role != "separate_max_context_setup":
        expected_hash = REFERENCE_OUTPUTS[cell["name"]][str(row["actual_drafts"])]
        if row["output_sha256"] != expected_hash:
            raise ValueError("per_depth_reference_output_mismatch")
    row.update({"session": session, "build": build, "cell": cell["name"],
                "role": role, "ordinal": sequence, "wall_seconds": wall,
                "mtp_selector": "auto",
                "input_tokens": len(ids), "context_tokens": context,
                "output_budget": maximum, "seed": seed,
                "host": host_metrics(monitor, phase, paging_before, paging_after)})
    return row


def run() -> int:
    os.umask(0o077)
    if OUT.exists():
        raise SystemExit("private_results_already_exist")
    required = (BASE, CANDIDATE, IDENTITY, BUILD_IDENTITY, NATIVE_PATCH,
                PYTHON_PATCH, CORPUS, HELDOUT, REFERENCE_SUMMARY)
    if not all(path.is_file() for path in required):
        raise SystemExit("sealed_input_missing")
    for name in tuple(os.environ):
        if name.startswith(("STARK_", "UZU_", "LILY_", "GGML_")):
            os.environ.pop(name)
    os.environ["STARK_REQUEST_BUDGET_MIB"] = "16"
    expected_identity = json.loads(IDENTITY.read_text())
    if not current_model_identity(expected_identity):
        raise SystemExit("model_identity_mismatch")
    baseline_hash = sha256_file(BASE)
    if baseline_hash != BASE_SHA256:
        raise SystemExit("baseline_binary_hash_mismatch")
    candidate_hash = sha256_file(CANDIDATE)
    if (sha256_file(BUILD_IDENTITY) != BUILD_IDENTITY_SHA256
            or sha256_file(NATIVE_PATCH) != NATIVE_PATCH_SHA256
            or sha256_file(PYTHON_PATCH) != PYTHON_PATCH_SHA256):
        raise SystemExit("candidate_source_seal_mismatch")
    build_identity = json.loads(BUILD_IDENTITY.read_text())
    if (candidate_hash != "e1e432fcd5d490c8ab900be23c7cbc3855fc4515d75a38089bac672a85e9be22"
            or build_identity.get("candidate_sha256") != candidate_hash
            or build_identity.get("before_sha256") != baseline_hash
            or build_identity.get("native_patch_sha256") != NATIVE_PATCH_SHA256
            or build_identity.get("python_patch_sha256") != PYTHON_PATCH_SHA256
            or build_identity.get("source_base") != "f331c648"):
        raise SystemExit("candidate_build_identity_mismatch")
    global REFERENCE_OUTPUTS, REFERENCE_INPUTS
    REFERENCE_OUTPUTS, REFERENCE_INPUTS = load_reference_hashes()
    inputs = {str(path): path.read_text() for path in (CORPUS, HELDOUT)}
    report = {
        "schema": "stark-lily-topk-policy-performance-v1",
        "attempt": "20261115-LILY-MTP1-05",
        "worker_sha256": {"before": baseline_hash, "candidate": candidate_hash},
        "tool_sha256": {
            name: sha256_file(ROOT / "scripts" / name) for name in (
                "validate_generation_prefix_cache.py", "benchmark_qwen38_mtp.py",
                "benchmark_clean_qwen.py",
            )
        },
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "model_content_manifest_sha256": expected_identity["model_content_manifest_sha256"],
        "model_identity_receipt_sha256": sha256_file(IDENTITY),
        "build_identity_sha256": sha256_file(BUILD_IDENTITY),
        "native_patch_sha256": NATIVE_PATCH_SHA256,
        "python_patch_sha256": PYTHON_PATCH_SHA256,
        "reference_file_sha256": REFERENCE_FILE_SHA256,
        "reference_output_sha256": REFERENCE_OUTPUTS,
        "reference_input_sha256": REFERENCE_INPUTS,
        "model_identity_method": "prior verified payload hashes guarded by current size/mtime/inode; no weight rehash",
        "harness_sha256": sha256_file(Path(__file__)),
        "corpora_sha256": {path.name: sha256_file(path) for path in (CORPUS, HELDOUT)},
        "request_budget_mib": 16,
        "prefix_cache": False,
        "vision_worker": True,
        "image_profile_required": IMAGE_PROFILE,
        "session_build_order": list(SESSIONS),
        "session_cell_orders": cell_orders(),
        "session_file_sha256": {},
        "cell_definitions": [{k: v for k, v in cell.items() if k != "source"} for cell in CELLS],
        "sampling_policy": "top20=(stochastic,temp=1,top_k=20,top_p=.95); nucleus=(stochastic,temp=1,top_p=.95; top_k omitted); greedy=(mode=greedy); min_p omitted",
        "mtp_policy": "send mtp=true AUTO in both builds; require before=2, candidate=1 for top20 and 2 for greedy/nucleus",
        "environment_policy": "remove inherited STARK_/UZU_/LILY_/GGML_ variables, then set STARK_REQUEST_BUDGET_MIB=16",
        "host_observation_policy": "all requests retained; monitor and paging observations are descriptive and never gate/exclude",
        "endpoint": {"heldout_paired_wall_reduction_at_least": 0.10,
                     "heldout_positional_pair_wins_at_least": 6,
                     "heldout_positional_pair_count": 8,
                     "primary_direction_must_be_positive": True,
                     "cross_session_noise_policy": "report inconclusive; do not rerun to chase a favorable result"},
        "sessions": [],
    }
    sealed_paths = {
        str(IDENTITY): report["model_identity_receipt_sha256"],
        str(BUILD_IDENTITY): report["build_identity_sha256"],
        str(NATIVE_PATCH): report["native_patch_sha256"],
        str(PYTHON_PATCH): report["python_patch_sha256"],
        str(Path(__file__)): report["harness_sha256"],
        str(CORPUS): report["corpora_sha256"][CORPUS.name],
        str(HELDOUT): report["corpora_sha256"][HELDOUT.name],
        **{str(ROOT / "scripts" / name): digest
           for name, digest in report["tool_sha256"].items()},
        **{str(REFERENCE_DIR / name): digest
           for name, digest in REFERENCE_FILE_SHA256.items()},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT / "manifest.json", report)
    token_hashes: dict[str, str] = {}
    corpus_by_path = {str(path): text for path, text in ((CORPUS, inputs[str(CORPUS)]), (HELDOUT, inputs[str(HELDOUT)]))}
    for session, build in enumerate(SESSIONS, start=1):
        binary = BASE if build == "before" else CANDIDATE
        if sha256_file(binary) != report["worker_sha256"][build]:
            raise SystemExit("worker_changed_before_session")
        session_row = {"session": session, "build": build, "cell_order": cell_orders()[session - 1],
                       "setup": None, "cells": []}
        report["sessions"].append(session_row)
        worker = None
        monitor = None
        try:
            worker = Worker(str(binary), str(MODEL), prefix_cache=False, vision=True, timeout=900)
            ready = worker.ready
            mtp = ready.get("mtp") or {}
            image = ready.get("image_input") or {}
            if (mtp.get("supported") is not True or type(mtp.get("max_drafts")) is not int
                    or mtp["max_drafts"] < 2):
                raise ValueError("mtp_readiness_mismatch")
            if image.get("family_profile") != IMAGE_PROFILE:
                raise ValueError("image_profile_mismatch")
            session_row["ready"] = {"mtp": mtp, "image_input": image}
            tokenized: dict[str, list[int]] = {}
            for cell in CELLS:
                source = str(cell["source"])
                if source not in tokenized:
                    tokenized[source] = worker.tokenize(corpus_by_path[source])
                ids = exact_prefix(tokenized[source], cell["input_tokens"])
                fingerprint = ids_sha256(ids)
                previous = token_hashes.setdefault(cell["name"], fingerprint)
                if previous != fingerprint:
                    raise ValueError("cross_build_input_token_mismatch")
                if fingerprint != REFERENCE_INPUTS[cell["name"]]:
                    raise ValueError("prior_reference_input_mismatch")
                cell["_ids"] = ids
                cell["_input_sha256"] = fingerprint
            monitor = Monitor().__enter__()
            setup_cell = {"name": "warmmax4352statefirst", "kind": "nucleus", "output_budget": 2}
            setup_ids = exact_prefix(tokenized[str(CORPUS)], 4096)
            setup_fp = ids_sha256(setup_ids)
            if setup_fp != REFERENCE_INPUTS["guard-input4096-nucleus-out256-seed42"]:
                raise ValueError("prior_reference_setup_input_mismatch")
            previous_setup = token_hashes.setdefault("warmmax4352statefirst", setup_fp)
            if previous_setup != setup_fp:
                raise ValueError("cross_build_setup_input_mismatch")
            setup = invoke(worker, setup_ids, setup_cell, context=4352, maximum=2, seed=42,
                           monitor=monitor, sequence=0, phase=f"session{session}-setup",
                           build=build, session=session, role="separate_max_context_setup")
            session_row["setup"] = setup
            for ordinal, cell_name in enumerate(session_row["cell_order"], start=1):
                cell = next(item for item in CELLS if item["name"] == cell_name)
                ids = cell["_ids"]
                cell_row = {"name": cell["name"], "input_sha256": cell["_input_sha256"],
                            "warmup": None, "measurements": []}
                session_row["cells"].append(cell_row)
                warm = invoke(worker, ids, cell, context=cell["input_tokens"] + cell["output_budget"],
                              maximum=cell["output_budget"], seed=cell["seed"], monitor=monitor,
                              sequence=0, phase=f"session{session}-cell{ordinal}-warmup",
                              build=build, session=session, role="fixed_cell_warmup")
                cell_row["warmup"] = warm
                write(OUT / f"session-{session}.json", session_row)
                for rep in range(1, 5):
                    measured = invoke(worker, ids, cell,
                                      context=cell["input_tokens"] + cell["output_budget"],
                                      maximum=cell["output_budget"], seed=cell["seed"], monitor=monitor,
                                      sequence=rep, phase=f"session{session}-cell{ordinal}-rep{rep}",
                                      build=build, session=session, role="measured")
                    cell_row["measurements"].append(measured)
                    write(OUT / f"session-{session}.json", session_row)
        finally:
            if monitor is not None:
                monitor.__exit__(None, None, None)
            if worker is not None:
                worker.close()
        if not current_model_identity(expected_identity):
            raise SystemExit("model_changed_during_run")
        if sha256_file(binary) != report["worker_sha256"][build]:
            raise SystemExit("worker_changed_during_run")
        if any(sha256_file(Path(path)) != digest for path, digest in sealed_paths.items()):
            raise SystemExit("sealed_source_changed_during_run")
        session_row["monitor_summary"] = {"samples": len(monitor.rows),
                                           "sampling_failures": monitor.failures}
        session_path = OUT / f"session-{session}.json"
        write(session_path, session_row)
        report["session_file_sha256"][str(session)] = sha256_file(session_path)
        write(OUT / "manifest.json", report)
    report["input_token_sha256"] = token_hashes
    report["completed"] = True
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write(OUT / "manifest.json", report)
    return 0


if __name__ == "__main__":
    status = 0
    try:
        status = run()
    except SystemExit as error:
        if error.code not in (None, 0):
            print(json.dumps({"status": "failed", "failure_type": "sealed_precondition"}), flush=True)
            status = 1
    except BaseException as error:
        print(json.dumps({"status": "failed", "failure_type": type(error).__name__}), flush=True)
        status = 1
    raise SystemExit(status)
