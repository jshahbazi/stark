"""Fixed one-vs-two MTP depth study; persists hashes/counts/timings only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_qwen38_mtp as bench
from benchmark_clean_qwen import Monitor, paging
from benchmark_glm_mlx_serve import screen_attempt
from stark_convert.chat import Worker

OUT = ROOT / "runtime/lily-one-draft/results"
MODEL = Path("/Users/johnshahbazian/Library/Application Support/Stark/models/qwen38-flash-next-lily-q4-xl-0f94842e")
PRIOR = Path("/Users/johnshahbazian/development/stark-llama-v1-study/runtime/llama-v1-lily-baseline-20261115-v2")
WORKER = PRIOR.parent / "llama-v1-baseline-target/release/stark-generate"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stats(values):
    median = statistics.median(values)
    return {"min": min(values), "median": median, "max": max(values),
            "mad": statistics.median(abs(v-median) for v in values),
            "relative_range": (max(values)-min(values))/median}


class ResidentWorker(Worker):
    def read(self, *args, **kwargs):
        media = self.media
        initial = not self._ready_seen
        if initial:
            self.media = None
        try:
            return super().read(*args, **kwargs)
        finally:
            if initial:
                self.media = media


def main():
    os.umask(0o077)
    OUT.mkdir(parents=True, exist_ok=False)
    for name in tuple(os.environ):
        if name.startswith(("STARK_", "UZU_", "LILY_", "GGML_")):
            os.environ.pop(name)
    identity = json.loads((PRIOR / "identity.json").read_text())
    assert bench.sha256_file(WORKER) == identity["worker_sha256"]
    model_guard = identity["model_file_identities"]

    def unchanged():
        return all([s.st_size, s.st_mtime_ns, s.st_ino] == expected
                   for name, expected in model_guard.items()
                   for s in [(MODEL/name).stat()])

    assert unchanged()
    corpus = OUT / "model-corpus.rs"
    held = OUT / "held-corpus.rs"
    shutil.copyfile(PRIOR/"frozen-corpus.rs", corpus)
    shutil.copyfile(ROOT/"vendor/lily-core/src/qwen4exp/config.rs", held)
    report = {"attempt": "20261115-LILY-MTP1-01", "source_commit": "f9c9720c23ee848fcd64c002450f04b727e32e35",
              "engine_source_commit": identity["source_commit"], "worker_sha256": identity["worker_sha256"],
              "model_content_manifest_sha256": identity["model_content_manifest_sha256"],
              "harness_sha256": bench.sha256_file(Path(__file__)),
              "corpus_sha256": bench.sha256_file(corpus), "held_corpus_sha256": bench.sha256_file(held),
              "host_policy": "all observations retained; no quiet admission or timing retries",
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cells": []}
    # Balance prompt length and sampling mode across consecutive loaded sessions.
    matrix = [(512, "top20", 256, 42, corpus), (4096, "greedy", 256, 42, corpus),
              (512, "nucleus", 256, 42, corpus), (4096, "top20", 256, 42, corpus),
              (512, "greedy", 256, 42, corpus), (4096, "nucleus", 256, 42, corpus),
              (32, "greedy", 64, 42, corpus), (512, "top20", 128, 7, held)]
    for length, mode, budget, seed, source in matrix:
        name = f"input{length}-{mode}-output{budget}-seed{seed}"
        cell = {"name": name, "input_tokens": length, "output_budget": budget, "seed": seed,
                "sampling": {"mode": "greedy"} if mode == "greedy" else
                    {"mode": "stochastic", "temperature": 1, "top_p": .95, **({"top_k": 20} if mode == "top20" else {})},
                "requests": [], "warmups": []}
        print(json.dumps({"stage": "starting_cell", "cell": name}), flush=True)
        worker = None
        monitor = None
        try:
            worker = ResidentWorker(WORKER, MODEL, 3600)
            ready = worker.read()
            assert ready.get("ready") is True
            assert ready["image_input"]["family_profile"] == "qwen38-lily-retained-bf16-vision-v1"
            assert ready["mtp"]["max_drafts"] >= 3
            cell["ready"] = {k: ready[k] for k in ("mtp", "image_input", "prefix_cache")}
            ids = bench.exact_prefix(worker.request({"operation": "tokenize", "text": source.read_text(),
                                                    "add_special_tokens": False})["token_ids"], length)
            cell["input_sha256"] = bench.ids_sha256(ids)
            monitor = Monitor()
            monitor.__enter__()
            request_id = 0

            def request(depth):
                nonlocal request_id
                request_id += 1
                phase = f"request_{request_id}"
                monitor.set_phase(phase)
                before = paging()
                started = time.monotonic()
                reply = worker.request({"operation": "generate", "input_token_ids": ids,
                    "context_tokens": length+budget, "max_new_tokens": budget, "sampling": cell["sampling"],
                    "seed": seed, "mtp_drafts": depth})
                wall = time.monotonic()-started
                assert reply.get("input_tokens") == length and reply.get("reused_input_tokens", 0) == 0
                item = bench._validate_result(reply, ready, argparse.Namespace(tokens=budget), "on", depth, ready["vocab_size"])
                item.update(depth=depth, wall_seconds=wall, reused_input_tokens=reply.get("reused_input_tokens", 0))
                try:
                    item["host_screen"] = screen_attempt(monitor, phase, before, paging())
                except ValueError as error:
                    if str(error) != "host_observations_missing":
                        raise
                    item["host_screen"] = {"eligible": False, "observation_status": "missing"}
                return item

            for depth in (1, 2):
                cell["warmups"].append(request(depth))
            first = random.Random(seed).choice((1, 2))
            pattern = (first, 3-first, 3-first, first)
            pairs = []
            for block in range(8):
                pair = {}
                for depth in (pattern[block%4], 3-pattern[block%4]):
                    item = request(depth)
                    item["block"] = block
                    pair[depth] = item
                    cell["requests"].append(item)
                pairs.append(pair)
            cell["summary"] = {}
            for depth in (1, 2):
                measured = [r for r in cell["requests"] if r["depth"] == depth]
                all_depth = measured + [r for r in cell["warmups"] if r["depth"] == depth]
                assert len({r["output_sha256"] for r in all_depth}) == 1
                cell["summary"][str(depth)] = {"useful_decode_tok_s": stats([(r["output_tokens"]-1)/r["decode_seconds"] for r in measured]),
                    "wall_seconds": stats([r["wall_seconds"] for r in measured])}
            cell["paired_wall_change_one_vs_two"] = statistics.median(p[1]["wall_seconds"]/p[2]["wall_seconds"]-1 for p in pairs)
            cell["paired_decode_change_one_vs_two"] = statistics.median(
                ((p[1]["output_tokens"]-1)/p[1]["decode_seconds"])/((p[2]["output_tokens"]-1)/p[2]["decode_seconds"])-1 for p in pairs)
            cell["one_draft_wall_wins"] = sum(p[1]["wall_seconds"] < p[2]["wall_seconds"] for p in pairs)
            cell["greedy_cross_depth_hashes_equal"] = len({r["output_sha256"] for r in cell["requests"]}) == 1 if mode == "greedy" else None
            cell["passed"] = unchanged()
        except Exception as error:
            cell["passed"] = False
            cell["failure_type"] = type(error).__name__
            cell["failure_code"] = str(error) if str(error) in bench.SAFE_FAILURE_CODES else "depth_study_failed"
        finally:
            if worker is not None:
                worker.close()
            if monitor is not None:
                monitor.__exit__(None, None, None)
        save(OUT/(name+".json"), cell)
        report["cells"].append({k:v for k,v in cell.items() if k not in ("requests", "warmups", "ready")})
        save(OUT/"summary.json", report)
        print(json.dumps({"stage": "completed_cell", "cell": name, "passed": cell["passed"],
                          "paired_wall_change_one_vs_two": cell.get("paired_wall_change_one_vs_two")}), flush=True)
        if not cell["passed"]:
            break
    report["completed"] = len(report["cells"]) == len(matrix) and all(c["passed"] for c in report["cells"])
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(OUT/"summary.json", report)
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
