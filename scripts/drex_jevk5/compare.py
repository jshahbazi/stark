"""Compare two local decision stacks on sealed JevBench; persist aggregates only."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import threading
import time

import psutil

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from benchmark_stock_engines import HostMonitor, Pipe, save, sha, sleep_events, stop  # noqa: E402
from benchmark_stock_scoring import quality  # noqa: E402
from scoring_fixture import decision_options  # noqa: E402
from stock_compare_cases import load_reference  # noqa: E402

DREX_SOURCE = "d2fd26cdf211f87b9a6b1261995528629c9e1401"
DREX_REVISION = "23f21ac6e664d3edb32faa7b55b0c8a7fb09134a"
JEV_REVISION = "c4f7fdb3aeab5582336406e78d3bef11bf98833d"


def read(path):
    return json.loads(Path(path).read_bytes())


def load_cases(reference):
    sealed, provenance = load_reference(reference)
    raw = read(Path(reference) / "dataset.json")["cases"]
    baseline = {
        row["case_id"]: row
        for row in read(Path(reference) / "round-1-baseline.json")["rows"]
        if row["condition"] == "fresh"
    }
    by_id = {case["id"]: case for case in raw}
    for case in sealed:
        case.update(
            state=by_id[case["id"]]["state"], question=by_id[case["id"]]["question"]
        )
        if case["question"]["type"] == "noul":
            aliases = {"yes": "true", "no": "false"}
            case["labels"] = [aliases[key] for key in case["labels"]]
            case["expected"] = aliases[case["expected"]]
        case["candidate_labels"] = [
            str(key) for key, _ in decision_options(case["question"])
        ]
        raw_p = baseline[case["id"]]["probabilities"]
        typed = baseline[case["id"]]["decision"]
        if case["question"]["type"] == "noul":
            if raw_p[case["candidate_labels"].index("true")] != typed["noul"]:
                raise ValueError("sealed_boolean_mapping_invalid")
        elif any(
            raw_p[i] != typed["probabilities"][key]
            for i, key in enumerate(case["candidate_labels"])
        ):
            raise ValueError("sealed_choice_mapping_invalid")
        case["reference_probabilities"] = align(
            case["labels"], case["candidate_labels"], raw_p
        )
    provenance.pop("reference_path")
    return sealed, provenance


def drex_request(case):
    question = dict(case["question"])
    if question["type"] == "choice" and isinstance(question["criteria"], list):
        question["criteria"] = dict.fromkeys(question["criteria"])
    return {"state": case["state"], "questions": {"decision": question}}


def align(labels, keys, values):
    labels, keys = list(map(str, labels)), list(map(str, keys))
    if len(keys) != len(set(keys)) or set(keys) != set(labels):
        raise ValueError("answer_labels_mismatch")
    if len(values) != len(keys) or any(
        not math.isfinite(p) or not 0 <= p <= 1 for p in values
    ):
        raise ValueError("probabilities_invalid")
    if abs(math.fsum(values) - 1) > 1e-5:
        raise ValueError("normalization_invalid")
    mapping = dict(zip(keys, values, strict=True))
    return [mapping[key] for key in labels]


def paired(cases, left, right):
    result = {}
    for tier in ("public", "synthetic"):
        cohort = [
            c for c in cases if (c["tier"] == "synthetic") == (tier == "synthetic")
        ]
        differences, left_only, right_only = [], 0, 0
        for case in cohort:

            def correct(probabilities):
                p = probabilities[case["id"]]
                return str(
                    case["labels"][max(range(len(p)), key=p.__getitem__)]
                ) == str(case["expected"])

            a, b = correct(left), correct(right)
            differences.append(int(b) - int(a))
            left_only += a and not b
            right_only += b and not a
        discordant = left_only + right_only
        p_value = min(
            1.0,
            2
            * sum(
                math.comb(discordant, i) for i in range(min(left_only, right_only) + 1)
            )
            / 2**discordant,
        )
        rng = random.Random(20261108)
        samples = sorted(
            statistics.mean(rng.choices(differences, k=len(differences)))
            for _ in range(10000)
        )
        result[tier] = {
            "cases": len(cohort),
            "jev_only_correct": left_only,
            "drex_only_correct": right_only,
            "drex_minus_jev_accuracy": statistics.mean(differences),
            "paired_bootstrap_95_percentile_interval": [samples[250], samples[9749]],
            "mcnemar_exact_two_sided_p": p_value,
        }
    return result


class MemoryMonitor:
    def __init__(self, process):
        self.process = psutil.Process(process.pid)
        self.peak = 0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.done.is_set():
            try:
                processes = [self.process, *self.process.children(recursive=True)]
                self.peak = max(self.peak, sum(p.memory_info().rss for p in processes))
            except psutil.Error:
                pass
            self.done.wait(0.05)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join()


def vm_counters():
    value = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
    return {
        k.strip(): int(v.strip().rstrip("."))
        for line in value.splitlines()[1:]
        if ":" in line
        for k, v in [line.split(":", 1)]
        if k.strip() in {"Pageouts", "Swapouts"}
    }


def worker(config, kind, probe):
    cases, _ = load_cases(config["reference"])
    cases = select_cases(cases, probe)
    context = config.get("context_tokens", 8192)
    with contextlib.ExitStack() as stack:
        if kind == "jev":
            pipe = Pipe(
                [config["jev_worker"], "--score-model-path", config["jev_model"]],
                stack,
                Path.cwd(),
                os.environ,
            )
            if pipe.ready.get("cache_enabled") is not False:
                raise ValueError("jev_cache_must_be_disabled")

            def score(case):
                reply = pipe.request(
                    {
                        "operation": "score",
                        "input_token_ids": case["input_token_ids"],
                        "candidate_token_ids": case["candidate_ids"],
                        "context_tokens": context,
                        "temperature": 1.22,
                    }
                )
                if (
                    reply.get("input_tokens") != len(case["input_token_ids"])
                    or reply.get("truncated") is not False
                    or reply.get("cache_hit") is not False
                ):
                    raise ValueError("jev_request_contract_invalid")
                return align(
                    case["labels"],
                    case["candidate_labels"],
                    reply["candidate_probabilities"],
                ), reply["input_tokens"]
        else:
            source = Path(config["drex_source"]) / "drex_max_mlx.py"
            spec = importlib.util.spec_from_file_location("pinned_drex", source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            runner = module.DrexMax(config["drex_model"], max_context=context)
            requests = {c["id"]: drex_request(c) for c in cases}
            plans = {key: runner.prepare(request) for key, request in requests.items()}
            captured = []
            upstream_answer = module.answer

            def capture(question, probabilities):
                captured[:] = probabilities
                return upstream_answer(question, probabilities)

            module.answer = capture

            def score(case):
                result = runner.decide(requests[case["id"]], plans[case["id"]])
                keys, _ = module.options(requests[case["id"]]["questions"]["decision"])
                return align(case["labels"], keys, captured), result["usage"][
                    "input_tokens"
                ]

        score(cases[0])  # Complete lazy model load and one warmup before readiness.
        print(json.dumps({"ready": True}), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            p, tokens = score(cases[request["index"]])
            # Readouts stay in this private pipe and the parent's memory.
            print(
                json.dumps(
                    {"probabilities": p, "input_tokens": tokens}, allow_nan=False
                ),
                flush=True,
            )


def select_cases(cases, probe):
    return [cases[i] for i in (0, 60, 120, 180)] if probe else cases


def clean_environment():
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("STARK_", "UZU_"))
        and key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT"}
    }


def provenance(config):
    source = Path(config["drex_source"])
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        != DREX_SOURCE
    ):
        raise ValueError("drex_source_revision_mismatch")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source):
        raise ValueError("drex_source_dirty")
    jev = Path(config["jev_model"])
    conversion = read(jev / "conversion.json")
    if conversion.get("source_revision") != JEV_REVISION:
        raise ValueError("jev_source_revision_mismatch")
    jev_hash = sha(jev / "model.safetensors")
    if jev_hash != conversion.get("output_sha256"):
        raise ValueError("jev_weight_hash_mismatch")
    drex = Path(config["drex_model"])
    expected = {
        "model.safetensors": "e9b0439e8c0cd994b54487c114b5d8fe683f10d2d05cb0f9407b2d869cfe51ac",
        "pointer.bin": "a89619d4ee1a61e6eec8580ed6792ca609b8972c240730dc3d193dfa00f069dd",
    }
    for name, digest in expected.items():
        if sha(drex / name) != digest:
            raise ValueError("drex_artifact_hash_mismatch")
    return {
        "jev": {
            "model": "alibiserikbay/JevK5",
            "revision": JEV_REVISION,
            "parameters": "4B",
            "weights": "BF16 native repack",
            "weight_sha256": jev_hash,
            "temperature": 1.22,
            "worker_sha256": sha(config["jev_worker"]),
            "config_sha256": sha(jev / "config.json"),
            "tokenizer_sha256": sha(jev / "tokenizer.json"),
        },
        "drex": {
            "model": "nace-ai/drex-max-v1-affine-q4",
            "revision": DREX_REVISION,
            "parameters": "27B",
            "weights": "affine Q4 group 64; mixed BF16 control tensors",
            "artifacts_sha256": {
                name: sha(drex / name)
                for name in (
                    "model.safetensors",
                    "pointer.bin",
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                )
            },
            "source_commit": DREX_SOURCE,
            "scorer_sha256": sha(source / "drex_max_mlx.py"),
        },
    }


def run(config, config_path, output, probe):
    cases, dataset = load_cases(config["reference"])
    selected = select_cases(cases, probe)
    report = {
        "schema": 1,
        "status": "preparing",
        "probe": probe,
        "dataset": dataset,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "harness_sha256": sha(__file__),
        "dependency_lock_sha256": sha(Path(__file__).with_name("uv.lock")),
        "host": {
            "system": platform.platform(),
            "chip": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip(),
            "memory_bytes": psutil.virtual_memory().total,
        },
        "protocol": {
            "blocks": 1 if probe else config.get("blocks", 2),
            "order": "Jev,Drex then Drex,Jev",
            "context_tokens": config.get("context_tokens", 8192),
            "cache": "disabled; fresh complete forward per question",
            "timing": "prepared inputs through model-native scoring, readout and private pipe; excludes tokenization/load/warmup",
            "memory": "50ms sampled harness and child process tree RSS, including startup; shared pages may be counted twice; not Metal allocation",
            "quality": "identical semantic cases; each model's own prompt/tokenization/readout/calibration",
            "limits": "Different sizes, precision and runtimes. Reused public development set. Does not reproduce AA index or Drex v1.1.",
        },
        "blocks": [],
    }
    save(output, report)
    first, repeatability = {}, {}
    started = time.time()
    awake = subprocess.Popen(
        ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        report["models"] = provenance(config)
        report["status"] = "running"
        for block in range(report["protocol"]["blocks"]):
            for kind in ("jev", "drex") if block % 2 == 0 else ("drex", "jev"):
                row = {"block": block, "model": kind, "status": "running"}
                report["blocks"].append(row)
                save(output, report)
                with contextlib.ExitStack() as stack, HostMonitor() as host:
                    memory = stack.enter_context(
                        MemoryMonitor(psutil.Process(os.getpid()))
                    )
                    before = vm_counters()
                    t0 = time.perf_counter()
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--config",
                        str(config_path.resolve()),
                        "--worker",
                        kind,
                    ]
                    if probe:
                        command.append("--probe")
                    pipe = Pipe(
                        command,
                        stack,
                        Path.cwd(),
                        {
                            **clean_environment(),
                            "HF_HUB_OFFLINE": "1",
                            "TOKENIZERS_PARALLELISM": "false",
                        },
                    )
                    row["startup_and_first_warmup_seconds"] = time.perf_counter() - t0
                    with contextlib.nullcontext():
                        for index in range(min(3, len(selected))):
                            pipe.request({"index": index})
                        probabilities, timings, lengths = {}, [], []
                        for index, case in enumerate(selected):
                            t0 = time.perf_counter()
                            reply = pipe.request({"index": index})
                            timings.append(time.perf_counter() - t0)
                            probabilities[case["id"]] = align(
                                case["labels"], case["labels"], reply["probabilities"]
                            )
                            lengths.append(reply["input_tokens"])
                            if (index + 1) % 25 == 0:
                                print(
                                    json.dumps(
                                        {
                                            "block": block,
                                            "model": kind,
                                            "completed": index + 1,
                                        }
                                    ),
                                    flush=True,
                                )
                        row["sampled_peak_process_tree_rss_bytes"] = memory.peak
                    row.update(
                        status="complete",
                        cases=len(selected),
                        median_seconds=statistics.median(timings),
                        p95_seconds=sorted(timings)[math.ceil(0.95 * len(timings)) - 1],
                        total_seconds=math.fsum(timings),
                        input_tokens={
                            "min": min(lengths),
                            "median": statistics.median(lengths),
                            "max": max(lengths),
                        },
                    )
                    if not probe:
                        row["quality"] = quality(selected, probabilities)
                    if kind == "jev":
                        row["sealed_jev_reference_max_probability_delta"] = max(
                            abs(a - b)
                            for case in selected
                            for a, b in zip(
                                case["reference_probabilities"],
                                probabilities[case["id"]],
                                strict=True,
                            )
                        )
                    if kind in first:
                        repeatability[kind] = max(
                            abs(a - b)
                            for case in selected
                            for a, b in zip(
                                first[kind][case["id"]],
                                probabilities[case["id"]],
                                strict=True,
                            )
                        )
                    else:
                        first[kind] = probabilities
                    after = vm_counters()
                    row["vm_counter_delta"] = {k: after[k] - before[k] for k in before}
                row["host_screen"] = host.result()
                row["timing_screen_passed"] = (
                    not row["host_screen"]["cpu_guard_exceeded"]
                    and not row["host_screen"]["background_model_active"]
                    and row["host_screen"]["sampling_failures"] == 0
                    and all(v == 0 for v in row["vm_counter_delta"].values())
                )
                save(output, report)
        if not probe:
            report["paired_quality"] = paired(selected, first["jev"], first["drex"])
        report["repeatability_max_probability_delta"] = repeatability
        report["sleep_screen"] = sleep_events(started, time.time())
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["failure_type"] = type(exc).__name__
        if report["blocks"] and report["blocks"][-1]["status"] == "running":
            report["blocks"][-1]["status"] = "failed"
        raise
    finally:
        stop(awake)
        save(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--worker", choices=("jev", "drex"))
    args = parser.parse_args()
    try:
        config = read(args.config)
        if args.worker:
            worker(config, args.worker, args.probe)
        else:
            if args.output is None:
                parser.error("--output is required")
            # Same advisory lock as the reference Drex scorer; retained through child reaping.
            with open(
                os.environ.get("BENCH_GPU_LOCK", "/tmp/kapitsa-gpu.lock"), "a+b"
            ) as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = run(config, args.config, args.output, args.probe)
            print(
                json.dumps(
                    {"status": result["status"], "blocks": len(result["blocks"])}
                )
            )
    except Exception:
        # Exception messages may contain private content; emit a fixed diagnostic.
        print("comparison_failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
