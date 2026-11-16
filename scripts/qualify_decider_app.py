"""Opt-in packaged-app Decider check; private traffic stays in memory and pipes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from qualify_decider_flash import Pipe, file_hash, private_write
from qualify_image_app import Backend, QualificationError, inventory, require


def fixture_cases(path):
    original = json.loads(path.read_text())["cases"]
    # Existing private cases, kept distinct from the earlier native qualification.
    selected = [original[index]["request"] for index in (1, 3, 7, 8, 9)]
    result = []
    for index, request in enumerate(selected):
        question = next(iter(request["questions"].values()))
        labels = list(question["criteria"])
        state = "" if index == 2 else request["state"]
        result.append(
            {
                "state": state,
                "question": question["instructions"],
                "answers": labels,
                "images": request.get("images", []),
            }
        )
    return result


def reference_results(root, snapshot, cases):
    reference = Pipe(
        [
            "uv",
            "run",
            "--project",
            str(root / "tools/validation/decider_flash"),
            "--locked",
            "python",
            str(root / "tools/validation/decider_flash/runner.py"),
            "--snapshot",
            str(snapshot),
            "--device",
            "mps",
        ]
    )
    try:
        answers = []
        for case in cases:
            request = {
                "state": case["state"],
                "questions": {
                    "decision": {
                        "type": "choice",
                        "instructions": case["question"],
                        "criteria": dict.fromkeys(case["answers"]),
                    }
                },
            }
            if case["images"]:
                request["images"] = case["images"]
            reply = reference.call({"request": request})
            require("response" in reply, "reference_failed")
            answers.append(reply["response"]["answers"]["decision"])
        return reference.ready, answers
    finally:
        reference.close()


def qualify(args):
    root = Path(__file__).resolve().parents[1]
    cases = fixture_cases(args.fixtures)
    readiness, expected = reference_results(root, args.snapshot, cases)
    print(json.dumps({"phase": "reference", "completed": len(cases)}), flush=True)
    app_hashes = inventory(args.app)
    report = {
        "attempt_id": "20261116-DF13APP-01",
        "reference_readiness": readiness,
        "fixtures_sha256": file_hash(args.fixtures),
        "script_sha256": file_hash(__file__),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "model_weights_sha256": file_hash(args.model / "model.safetensors"),
        "bundle_inventory": app_hashes,
        "cases": len(cases),
        "images": 3,
    }
    with tempfile.TemporaryDirectory(
        prefix="decider-app-control-", dir=args.output.parent
    ) as temporary:
        work = Path(temporary).resolve()
        work.chmod(0o700)
        copied = work / "relocated app/Stark.app"
        copied.parent.mkdir()
        shutil.copytree(args.app, copied)
        require(inventory(copied) == app_hashes, "relocated_bundle_changed")
        runtime = copied / "Contents/Resources/runtime"
        home, attachments, temp = work / "home", work / "attachments", work / "tmp"
        for directory in (home, attachments, temp):
            directory.mkdir(mode=0o700)
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
            "TMPDIR": str(temp),
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": "/invalid-checkout-python",
            "PYTHONHOME": "/invalid-checkout-python",
            "LANG": "C",
        }
        signature = subprocess.run(
            ["/usr/bin/codesign", "--verify", "--deep", "--strict", str(copied)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
            check=False,
        )
        require(signature.returncode == 0, "app_signature_failed")
        backend = Backend(
            [
                runtime / "libexec/python/bin/python3.13",
                "-I",
                "-B",
                "-m",
                "stark_convert.desktop",
                "--worker",
                runtime / "libexec/stark-generate",
                "--score-worker",
                runtime / "libexec/stark-score",
                "--decide-worker",
                runtime / "libexec/stark-decide",
                "--data-dir",
                home / "library",
                "--attachment-root",
                attachments,
            ],
            environment,
            work,
        )
        gpu = []

        def observe(event):
            if event.get("event") == "gpu_memory":
                gpu.append(event)

        def until(predicate, **options):
            return backend.until(predicate, observe=observe, timeout=180, **options)

        try:
            until(lambda event: event.get("event") == "models")
            backend.send(
                {
                    "command": "install",
                    "kind": "local",
                    "source": str(args.model),
                    "artifact_mode": "decision",
                    "support_mode": "decision",
                    "request_id": "import",
                }
            )
            library = until(
                lambda event: event.get("event") == "models" and event.get("models")
            )
            until(
                lambda event: (
                    event.get("event") == "status"
                    and event.get("message") == "Model installed"
                )
            )
            model = library["models"][0]
            identifier = model["id"]
            require(model.get("artifact_mode") == "decision", "import_not_decision")
            backend.send({"command": "run", "model_id": identifier, "mode": "chat"})
            rejected = until(
                lambda event: event.get("event") == "error", allow_error=True
            )
            require("Decision" in rejected.get("message", ""), "chat_not_rejected")
            backend.send({"command": "run", "model_id": identifier, "mode": "decision"})
            ready = until(lambda event: event.get("event") == "ready")
            require(
                ready.get("mode") == "decision"
                and ready.get("decision_model") is True
                and ready.get("decision_images") is True
                and ready.get("images") is True
                and ready.get("image_attachment_transport") == "private-file-v1",
                "ready_capabilities_invalid",
            )
            report["ready"] = {
                key: ready.get(key)
                for key in (
                    "mode",
                    "backend",
                    "images",
                    "decision_images",
                    "decision_model",
                    "image_limits",
                    "max_context_tokens",
                    "checkpoint_temperature",
                )
            }
            errors, matches, results = [], [], []

            def post(case, request_id):
                message = {
                    "command": "decide",
                    "text": case["state"],
                    "question": case["question"],
                    "answers": case["answers"],
                    "request_id": request_id,
                }
                if case["images"]:
                    path = Path(case["images"][0])
                    data = path.read_bytes()
                    name = uuid.uuid4().hex
                    target = attachments / name
                    descriptor = os.open(
                        target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                    )
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(data)
                    message["image"] = {
                        "id": name,
                        "media_type": "image/png"
                        if data.startswith(b"\x89PNG")
                        else "image/jpeg",
                        "byte_length": len(data),
                    }
                backend.send(message)
                reply = until(
                    lambda event: (
                        event.get("event") == "decision"
                        and event.get("request_id") == request_id
                    )
                )
                until(
                    lambda event: (
                        event.get("event") == "status" and event.get("state") == "ready"
                    )
                )
                require(not list(attachments.iterdir()), "attachment_not_consumed")
                return reply

            for index, (case, reference) in enumerate(
                zip(cases, expected, strict=True)
            ):
                result = post(case, "decision_" + str(index))
                require(
                    result.get("answers") == case["answers"], "answer_order_changed"
                )
                values = result["probabilities"]
                reference_values = [
                    reference["probabilities"][label] for label in case["answers"]
                ]
                require(len(values) == len(reference_values), "answer_count_changed")
                delta = [
                    abs(left - right)
                    for left, right in zip(values, reference_values, strict=True)
                ]
                errors.extend(delta)
                matches.append(
                    case["answers"][result["choice_index"]] == reference["choice"]
                )
                require(
                    result.get("calibration") == "checkpoint_temperature",
                    "calibration_missing",
                )
                results.append(result)
                print(
                    json.dumps(
                        {
                            "phase": "packaged_app",
                            "completed": index + 1,
                            "image": bool(case["images"]),
                            "max_error": max(delta),
                            "choice_match": matches[-1],
                        }
                    ),
                    flush=True,
                )
            repeated = post(cases[0], "repeat_after_images")
            report["repeat_exact"] = all(
                repeated[key] == results[0][key]
                for key in ("choice_index", "answers", "probabilities", "input_tokens")
            )
            backend.send({"command": "reset"})
            after_reset = post(cases[0], "repeat_after_reset")
            report["reset_exact"] = all(
                after_reset[key] == results[0][key]
                for key in ("choice_index", "answers", "probabilities", "input_tokens")
            )
            report["reset_attachment_cleanup"] = not list(attachments.iterdir())
            backend.send({"command": "stop"})
            until(
                lambda event: (
                    event.get("event") == "status" and event.get("message") == "Stopped"
                )
            )
            report["stop_attachment_cleanup"] = not list(attachments.iterdir())
            report["gpu_memory_events"] = len(gpu)
            report["top_matches"] = sum(matches)
            report["max_absolute_probability_error"] = max(errors)
            report["mean_absolute_probability_error"] = sum(errors) / len(errors)
            report["passed"] = (
                all(matches)
                and max(errors) <= 0.02
                and report["repeat_exact"]
                and report["reset_exact"]
                and report["reset_attachment_cleanup"]
                and report["stop_attachment_cleanup"]
                and bool(gpu)
            )
        finally:
            backend.close()
        require(not list(attachments.iterdir()), "shutdown_attachment_cleanup_failed")
    private_write(args.output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "passed",
                    "cases",
                    "images",
                    "top_matches",
                    "max_absolute_probability_error",
                    "repeat_exact",
                    "gpu_memory_events",
                )
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("app", "model", "snapshot", "fixtures", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    for key in ("app", "model", "snapshot", "fixtures", "output"):
        setattr(args, key, getattr(args, key).resolve())
    try:
        return qualify(args)
    except QualificationError as error:
        print(json.dumps({"passed": False, "error": str(error)}), flush=True)
        return 1
    except Exception:  # noqa: BLE001 - runtime exceptions never enter diagnostics
        print(
            json.dumps({"passed": False, "error": "decider_app_qualification_failed"}),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
