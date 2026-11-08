"""Replay hash-bound Human100 requests locally; persist only aggregates/hashes."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from compare import (
    DREX_SOURCE,
    DREX_REVISION,
    HostMonitor,
    MemoryMonitor,
    Pipe,
    align,
    clean_environment,
    read,
    save,
    sha,
    sleep_events,
    stop,
    vm_counters,
)

OPTIONS = ["correct", "incorrect", "partial"]
EXPECTED_INPUTS = {
    "human_cases": "c3cdb14e94db8db5364693cff3ffbeca68697a3fb5497160997a40984f72690e",
    "human_evidence": "b6c5188b5b5c9cbf569d592adda1d7c530733503bfc7a06622e5a6c1d3d295d0",
    "human_labels": "84aa9cd6ed3891e1c61b1881b317a0abafd1d98e5717df5476a0d443e9dc9a95",
}


def require(condition, code):
    if not condition:
        raise ValueError(code)


def load_reference(config):
    root = Path(config["reference"])
    seal_path = root / "artifacts/SEAL.json"
    seal = read(seal_path)
    completed = read(root / "subq-snap/COMPLETE.json")
    verdict_path = root / "subq-snap/verdicts.jsonl"
    require(sha(seal_path) == completed["seal_sha256"], "reference_seal_changed")
    require(
        sha(verdict_path) == completed["verdicts_sha256"], "reference_verdicts_changed"
    )
    require(
        completed["completed"] == 400 and completed["errors"] == 0,
        "reference_incomplete",
    )
    for name, expected in EXPECTED_INPUTS.items():
        entry = seal["inputs"][name]
        require(
            entry["sha256"] == expected and sha(entry["path"]) == expected,
            "reference_input_changed",
        )
    labels = read(seal["inputs"]["human_labels"]["path"])
    ids = sorted(read(seal["inputs"]["human_evidence"]["path"]))
    require(labels["revision"] == 121 and len(ids) == 100, "human_population_changed")
    gold = {cid: labels["decisions"][cid]["label"] == "accepted" for cid in ids}
    require(
        sum(gold.values()) == 67
        and all(
            labels["decisions"][cid]["label"] in {"accepted", "rejected"} for cid in ids
        ),
        "human_labels_invalid",
    )
    inventory = {(r["id"], r["arm"]): r for r in seal["inventory"]}
    require(len(inventory) == 200, "inventory_invalid")
    rows, seen, bodies, wire_hashes = [], set(), {}, []
    for line in verdict_path.read_text().splitlines():
        row = json.loads(line)
        key = (row["id"], row["arm"], row["repeat"])
        require(
            row["id"] in gold
            and row["arm"] in {"J0", "E1"}
            and row["repeat"] in {1, 2}
            and key not in seen,
            "reference_row_invalid",
        )
        seen.add(key)
        wire_path = root / "subq-snap/wire" / Path(row["wire_file"]).name
        require(sha(wire_path) == row["wire_sha256"], "reference_wire_changed")
        wire_hashes.append(row["wire_sha256"])
        wire = read(wire_path)
        require(wire["http_status"] == 200, "reference_wire_failed")
        body = wire["request"]
        require(
            set(body["questions"]) == {"verdict"}, "reference_question_scope_invalid"
        )
        q = body["questions"]["verdict"]
        require(
            q["type"] == "choice" and list(q["criteria"]) == OPTIONS,
            "reference_options_changed",
        )
        prior = inventory[row["id"], row["arm"]]
        require(
            all(
                row[k] == prior[k] for k in ("request_sha256", "ordered_request_sha256")
            ),
            "reference_request_identity_changed",
        )
        pair = (row["id"], row["arm"])
        require(
            pair not in bodies or bodies[pair] == body,
            "reference_repeat_request_changed",
        )
        bodies[pair] = body
        rows.append(
            {
                "id": row["id"],
                "arm": row["arm"],
                "repeat": row["repeat"],
                "request": body,
            }
        )
    require(len(seen) == 400 and len(bodies) == 200, "reference_coverage_invalid")
    return (
        rows,
        gold,
        {
            "cases": 100,
            "accepted": 67,
            "rejected": 33,
            "label_revision": 121,
            "input_sha256": EXPECTED_INPUTS,
            "reference_seal_sha256": sha(seal_path),
            "reference_verdicts_sha256": sha(verdict_path),
            "wire_inventory_sha256": hashlib.sha256(
                "".join(wire_hashes).encode()
            ).hexdigest(),
        },
    )


def drex_module(config):
    source = Path(config["drex_source"])
    require(
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        == DREX_SOURCE,
        "drex_source_changed",
    )
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=source),
        "drex_source_dirty",
    )
    spec = importlib.util.spec_from_file_location(
        "pinned_drex", source / "drex_max_mlx.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_identity(config):
    model = Path(config["drex_model"])
    hashes = {
        name: sha(model / name)
        for name in (
            "model.safetensors",
            "pointer.bin",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    }
    require(
        hashes["model.safetensors"]
        == "e9b0439e8c0cd994b54487c114b5d8fe683f10d2d05cb0f9407b2d869cfe51ac",
        "drex_weights_changed",
    )
    require(
        hashes["pointer.bin"]
        == "a89619d4ee1a61e6eec8580ed6792ca609b8972c240730dc3d193dfa00f069dd",
        "drex_readout_changed",
    )
    return {
        "model": "nace-ai/drex-max-v1-affine-q4",
        "revision": DREX_REVISION,
        "source_commit": DREX_SOURCE,
        "artifacts_sha256": hashes,
        "scorer_sha256": sha(Path(config["drex_source"]) / "drex_max_mlx.py"),
    }


def prepare(config, rows):
    module = drex_module(config)
    runner = module.DrexMax(config["drex_model"], max_context=16384)
    bodies, plans = {}, {}
    for row in rows:
        key = (row["id"], row["arm"])
        if key not in plans:
            body = {k: v for k, v in row["request"].items() if k != "model"}
            bodies[key] = body
            plans[key] = runner.prepare(body)
    return module, runner, bodies, plans


def worker(config):
    rows, _, _ = load_reference(config)
    module, runner, bodies, plans = prepare(config, rows)
    # Public upstream example warms lazy model loading; no Human100 readout is discarded.
    public = read(Path(config["drex_source"]) / "examples/request.json")
    runner.decide(public, runner.prepare(public))
    captured = []
    original = module.answer

    def capture(question, p):
        captured[:] = p
        return original(question, p)

    module.answer = capture
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        index = json.loads(line)["index"]
        row = rows[index]
        key = (row["id"], row["arm"])
        reply = runner.decide(bodies[key], plans[key])
        keys, _ = module.options(bodies[key]["questions"]["verdict"])
        p = align(OPTIONS, keys, captured)
        print(
            json.dumps(
                {"p": p, "input_tokens": reply["usage"]["input_tokens"]},
                allow_nan=False,
            ),
            flush=True,
        )


def run(config, config_path):
    root = Path(config["output_root"])
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    rows, gold, dataset = load_reference(config)
    report = {
        "schema": 1,
        "status": "preparing",
        "dataset": dataset,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "harness_sha256": sha(__file__),
        "config_sha256": sha(config_path),
        "auditor_sha256": sha(Path(__file__).with_name("human100_audit.py")),
        "registration_sha256": sha(config["registration"]),
        "lock_sha256": sha(Path(__file__).with_name("uv.lock")),
        "protocol": {
            "contexts": ["J0", "E1"],
            "repeats": 2,
            "expected_judgments": 400,
            "acceptance": "native argmax correct only",
            "cache": "none; fresh noncausal forward",
            "max_context_tokens": 16384,
            "truncation": False,
            "bootstrap_draws": 10000,
            "seed": 20260929,
        },
        "progress": {
            arm: {"judgments": 0, "matches": 0, "false_accepts": 0, "false_rejects": 0}
            for arm in ("J0", "E1")
        },
    }
    path = root / "aggregate.json"
    save(path, report)
    started = time.time()
    try:
        report["model"] = model_identity(config)
        _, _, _, plans = prepare(config, rows)
        report["token_lengths"] = {
            arm: {"min": min(v), "median": statistics.median(v), "max": max(v)}
            for arm in ("J0", "E1")
            for v in [
                [len(ids[0]) for (_, context), ids in plans.items() if context == arm]
            ]
        }
        # Freeze all identities and request coverage before starting the inference child.
        report["status"] = "sealed"
        save(root / "preflight-seal.json", report)
        save(path, report)
        awake = subprocess.Popen(
            ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with contextlib.ExitStack() as stack, HostMonitor() as host:
            stack.callback(stop, awake)
            memory = stack.enter_context(
                MemoryMonitor(__import__("psutil").Process(os.getpid()))
            )
            before = vm_counters()
            pipe = Pipe(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    str(config_path.resolve()),
                    "--worker",
                ],
                stack,
                Path.cwd(),
                {
                    **clean_environment(),
                    "HF_HUB_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                },
            )
            report["status"] = "running"
            records = []
            for index, row in enumerate(rows):
                t0 = time.perf_counter()
                reply = pipe.request({"index": index})
                elapsed = time.perf_counter() - t0
                p = align(OPTIONS, OPTIONS, reply["p"])
                accepted = OPTIONS[max(range(3), key=p.__getitem__)] == "correct"
                progress = report["progress"][row["arm"]]
                progress["judgments"] += 1
                progress["matches"] += accepted == gold[row["id"]]
                progress["false_accepts"] += accepted and not gold[row["id"]]
                progress["false_rejects"] += not accepted and gold[row["id"]]
                records.append(
                    {
                        "id": row["id"],
                        "arm": row["arm"],
                        "repeat": row["repeat"],
                        "p": p,
                        "seconds": elapsed,
                        "input_tokens": reply["input_tokens"],
                    }
                )
                save(path, report)
                if (index + 1) % 10 == 0:
                    print(
                        json.dumps({"completed": index + 1, "expected": 400}),
                        flush=True,
                    )
            report["sampled_peak_process_tree_rss_bytes"] = memory.peak
            report["vm_counter_delta"] = {
                k: v - before[k] for k, v in vm_counters().items()
            }
        report["host_screen"] = host.result()
        report["sleep_screen"] = sleep_events(started, time.time())
        audit = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("human100_audit.py")),
                "--config",
                str(config_path),
            ],
            input=json.dumps(records, allow_nan=False).encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        require(audit.returncode == 0, "independent_analysis_failed")
        metrics = json.loads(audit.stdout)
        require(metrics["status"] == "passed", "independent_audit_failed")
        for arm in ("J0", "E1"):
            require(
                all(
                    metrics["contexts"][arm][k] == v
                    for k, v in report["progress"][arm].items()
                ),
                "independent_counts_disagree",
            )
        report["analysis"] = metrics
        report["status"] = "complete"
        save(root / "audit.json", metrics)
    except BaseException as exc:
        report["status"] = "failed"
        report["failure_type"] = type(exc).__name__
        raise
    finally:
        save(path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    try:
        config = read(args.config)
        if args.worker:
            worker(config)
        elif args.prepare_only:
            rows, _, dataset = load_reference(config)
            _, _, _, plans = prepare(config, rows)
            print(
                json.dumps(
                    {
                        "verified": True,
                        "cases": dataset["cases"],
                        "requests": len(plans),
                        "maximum_input_tokens": max(len(p[0]) for p in plans.values()),
                    }
                )
            )
        else:
            with open("/tmp/kapitsa-gpu.lock", "a+b") as lock:
                __import__("fcntl").flock(
                    lock, __import__("fcntl").LOCK_EX | __import__("fcntl").LOCK_NB
                )
                result = run(config, args.config)
            print(json.dumps({"status": result["status"], "judgments": 400}))
    except Exception as exc:
        print(
            json.dumps({"status": "failed", "error_type": type(exc).__name__}),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
