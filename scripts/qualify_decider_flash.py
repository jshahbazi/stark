"""Compare native Decider with the unmodified upstream implementation in memory.

Fixtures and responses stay private. Only aggregate errors, timings and identity
hashes are written. Reference and native models are loaded sequentially.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import selectors
import statistics
import subprocess
import time
from pathlib import Path


class QualificationError(Exception):
    pass


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class Pipe:
    def __init__(self, command, timeout=1200):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, bufsize=0, start_new_session=True)
        self.timeout, self.buffer = timeout, bytearray()
        try:
            self.ready = self.read()
            if self.ready.get("ready") is not True:
                raise QualificationError("worker_not_ready")
        except BaseException:
            self.close()
            raise

    def read(self):
        deadline = time.monotonic() + self.timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while True:
                if b"\n" in self.buffer:
                    line, _, self.buffer = self.buffer.partition(b"\n")
                    return json.loads(line)
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise QualificationError("worker_timeout")
                part = os.read(self.process.stdout.fileno(), 65536)
                if not part:
                    raise QualificationError("worker_stopped")
                self.buffer.extend(part)
                if len(self.buffer) > 8 * 1024 * 1024:
                    raise QualificationError("reply_too_large")

    def call(self, value):
        self.process.stdin.write(json.dumps(value, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        self.process.stdin.flush()
        return self.read()

    def close(self):
        self.process.kill()
        self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()


def distribution(answer):
    if answer["type"] == "noul":
        value = float(answer["noul"])
        values = [1 - value, value]
    else:
        values = [float(x) for x in answer["probabilities"].values()]
    if not values or any(not math.isfinite(x) or x < 0 or x > 1 for x in values):
        raise QualificationError("invalid_distribution")
    if abs(sum(values) - 1) > 1e-6:
        raise QualificationError("invalid_distribution")
    return values


def compare(reference, candidate):
    if reference["answers"].keys() != candidate["answers"].keys():
        raise QualificationError("answer_keys_differ")
    errors, matches = [], []
    for key, expected in reference["answers"].items():
        actual = candidate["answers"][key]
        if actual["type"] != expected["type"]:
            raise QualificationError("answer_type_differs")
        if expected["type"] != "noul":
            labels = list(expected["probabilities"])
            if set(labels) != set(actual["probabilities"]):
                raise QualificationError("option_labels_differ")
            actual = {**actual, "probabilities": {key: actual["probabilities"][key] for key in labels}}
            if expected["type"] == "choice":
                for answer in (expected, actual):
                    best = max(answer["probabilities"], key=answer["probabilities"].__getitem__)
                    if answer["choice"] != best:
                        raise QualificationError("answer_choice_differs")
                if abs(expected["confidence"] - actual["confidence"]) > 0.04:
                    raise QualificationError("answer_confidence_differs")
            elif expected["type"] == "score":
                if expected["legend"] != actual["legend"]:
                    raise QualificationError("score_legend_differs")
                for answer in (expected, actual):
                    score = sum(i * float(p) for i, p in enumerate(answer["probabilities"].values()))
                    if abs(float(answer["score"]) - score) > 1e-6:
                        raise QualificationError("invalid_ordinal_score")
        left, right = distribution(expected), distribution(actual)
        if len(left) != len(right):
            raise QualificationError("option_counts_differ")
        errors.extend(abs(x - y) for x, y in zip(left, right, strict=True))
        matches.append(max(range(len(left)), key=left.__getitem__) == max(range(len(right)), key=right.__getitem__))
    return errors, matches


def private_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def text_residency_timing(args, request):
    """Same text graph, with only tower residency changed; no reference speed claim."""
    from stark_convert.decider_protocol import Encoder
    from stark_convert.template_runtime import IsolatedTemplate
    from tokenizers import Tokenizer

    config = json.loads((args.model / "config.json").read_text())
    encoder = Encoder(Tokenizer.from_file(str(args.model / "tokenizer.json")),
                      IsolatedTemplate((args.model / "chat_template.jinja").read_text()),
                      config["decision_config"]["codes"])
    encoded = encoder.plan(request, 4096)[0]
    message = {"operation": "decide", "input_token_ids": list(encoded.input_ids),
               "readout_positions": [len(encoded.input_ids) - 1],
               "option_counts": [encoded.option_count], "context_tokens": 4096}
    timings, digests = {False: [], True: []}, set()
    for images_loaded in (False, True, True, False):
        command = [str(args.worker), "--model-path", str(args.model)]
        if not images_loaded:
            command.append("--text-only-validation")
        worker = Pipe(command)
        try:
            if worker.ready.get("images") != images_loaded:
                raise QualificationError("timing_residency_mismatch")
            for repetition in range(4):
                started = time.monotonic()
                reply = worker.call(message)
                elapsed = time.monotonic() - started
                if reply.get("ok") is not True:
                    raise QualificationError("timing_request_failed")
                digests.add(hashlib.sha256(json.dumps(reply["logits"]).encode()).hexdigest())
                if repetition:
                    timings[images_loaded].append(elapsed)
        finally:
            worker.close()
    baseline, candidate = timings[False], timings[True]
    variability = (max(baseline) - min(baseline)) / statistics.median(baseline)
    ratio = statistics.median(candidate) / statistics.median(baseline)
    return {"order": ["tower_off", "tower_on", "tower_on", "tower_off"],
            "warmups_per_load": 1, "observations_per_arm": len(baseline),
            "tower_off_seconds": baseline, "tower_on_seconds": candidate,
            "baseline_relative_range": variability, "median_ratio": ratio,
            "logits_exact_across_modes": len(digests) == 1,
            "gate": "inconclusive" if variability > 0.1 else "passed" if ratio <= 1.1 else "failed",
            "scope": "native warmed text worker latency, same branch/weights/graph; tower residency only"}


def qualify(args):
    from stark_convert.decider import Decider
    from stark_convert.decider_protocol import DeciderError

    fixtures = json.loads(args.fixtures.read_text())
    cases = fixtures["cases"]
    if not cases:
        raise QualificationError("empty_corpus")
    record = {
        "attempt_id": "20261116-DF13-01", "source_revision": "d06fb3f2d97e0a52325a0d463c450cf0338f44fe",
        "fixture_sha256": file_hash(args.fixtures), "worker_sha256": file_hash(args.worker),
        "native_config_sha256": file_hash(args.model / "config.json"),
        "native_weights_sha256": file_hash(args.model / "model.safetensors"),
        "reference_lock_sha256": file_hash(args.root / "tools/validation/decider_flash/uv.lock"),
        "reference_source_sha256": file_hash(args.snapshot / "source/src/pplx_decider/model.py"),
        "reference_device": args.device, "native_precision": "BF16", "cases": len(cases),
        "criteria": {"max_absolute_probability_error": 0.02, "top_choice_agreement": 1.0},
        "image_sha256": sorted({file_hash(p) for c in cases for p in c["request"].get("images", [])}),
    }
    references = []
    command = ["uv", "run", "--project", str(args.root / "tools/validation/decider_flash"), "--locked",
               "python", str(args.root / "tools/validation/decider_flash/runner.py"),
               "--snapshot", str(args.snapshot), "--device", args.device]
    reference = Pipe(command)
    record["reference_readiness"] = reference.ready
    try:
        for index, case in enumerate(cases):
            started = time.monotonic()
            result = reference.call({"request": case["request"]})
            if "response" not in result:
                raise QualificationError("reference_case_failed")
            references.append(result["response"])
            print(json.dumps({"phase": "reference", "completed": index + 1,
                              "seconds": round(time.monotonic() - started, 3)}), flush=True)
    finally:
        reference.close()
    private_write(args.output.with_suffix(".progress.json"), {**record, "reference_completed": len(references)})
    errors, matches, rows, outputs = [], [], [], []
    with Decider(args.model, args.worker, timeout=1200) as native:
        record["native_readiness"] = native.ready
        for index, (case, expected) in enumerate(zip(cases, references, strict=True)):
            started = time.monotonic()
            result = native.decide(case["request"])
            outputs.append(result)
            delta, agree = compare(expected, result)
            errors.extend(delta)
            matches.extend(agree)
            rows.append({"image": bool(case["request"].get("images")), "questions": len(agree),
                         "max_absolute_error": max(delta), "top_matches": sum(agree),
                         "usage_equal": expected["usage"] == result["usage"],
                         "input_tokens": result["usage"]["input_tokens"],
                         "native_seconds": time.monotonic() - started})
            print(json.dumps({"phase": "native", "completed": index + 1, **rows[-1]}), flush=True)
        # Text -> image -> text has occurred above; requests always start fresh.
        repeated = native.decide(cases[0]["request"])
        exact_repeat = repeated == outputs[0]
        try:
            native.decide({"state": None, "questions": {}})
            raise QualificationError("invalid_request_accepted")
        except DeciderError:
            pass
        recovered = native.decide(cases[0]["request"]) == outputs[0]
        reset = native._request({"operation": "reset_cache"}).get("cache_cleared") is True
        image_failure_closed = None
        image_case = next((case for case in cases if case["request"].get("images")), None)
        if image_case is not None:
            bad_image = args.output.with_suffix(".invalid-image.png")
            descriptor = os.open(bad_image, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(b"invalid-image-control")
            try:
                try:
                    native.decide({**image_case["request"], "images": [str(bad_image.absolute())]})
                    raise QualificationError("invalid_image_accepted")
                except DeciderError:
                    image_failure_closed = native._process is None
            finally:
                bad_image.unlink()
        # Interrupt an owned worker with a complete long request pending. No
        # command logs or response dumps are used for the cancellation control.
        text_plans = [item for case in cases if not case["request"].get("images")
                      for item in native.encoder.plan(case["request"], 4096)]
        longest = max(text_plans, key=lambda item: len(item.input_ids))
    interrupted = Pipe([str(args.worker), "--model-path", str(args.model)])
    try:
        message = {"operation": "decide", "input_token_ids": list(longest.input_ids),
                   "readout_positions": [len(longest.input_ids) - 1],
                   "option_counts": [longest.option_count], "context_tokens": 4096}
        interrupted.process.stdin.write(json.dumps(message).encode() + b"\n")
        interrupted.process.stdin.flush()
        time.sleep(0.02)
        with selectors.DefaultSelector() as selector:
            selector.register(interrupted.process.stdout, selectors.EVENT_READ)
            cancelled_while_pending = not selector.select(0)
    finally:
        interrupted.close()
    with Decider(args.model, args.worker, timeout=1200) as restarted:
        restart_exact = restarted.decide(cases[0]["request"]) == outputs[0]
    record.update({"rows": rows, "questions": len(matches), "top_matches": sum(matches),
                   "max_absolute_probability_error": max(errors),
                   "mean_absolute_probability_error": sum(errors) / len(errors),
                   "exact_repeat": exact_repeat, "validation_error_recovery": recovered,
                   "reset": reset, "restart_exact": restart_exact,
                   "invalid_image_closes_client_worker": image_failure_closed,
                   "cancelled_while_pending": cancelled_while_pending,
                   "usage_all_equal": all(r["usage_equal"] for r in rows)})
    record["passed"] = (all(matches) and max(errors) <= 0.02 and exact_repeat and recovered
                        and reset and restart_exact and record["usage_all_equal"]
                        and cancelled_while_pending and image_failure_closed is not False)
    if args.timing:
        record["text_residency_timing"] = text_residency_timing(args, cases[0]["request"])
    private_write(args.output, record)
    print(json.dumps({k: record[k] for k in ("passed", "cases", "questions", "top_matches",
                                           "max_absolute_probability_error", "mean_absolute_probability_error",
                                           "exact_repeat", "validation_error_recovery", "restart_exact")}), flush=True)
    return 0 if record["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "model", "worker", "fixtures", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--timing", action="store_true", help="also compare native text with tower residency off/on")
    args = parser.parse_args()
    args.root = Path(__file__).resolve().parents[1]
    try:
        return qualify(args)
    except Exception as error:  # noqa: BLE001 - redact native exceptions at the privacy boundary
        code = str(error) if isinstance(error, QualificationError) else "qualification_failed"
        print(json.dumps({"passed": False, "error": code}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
