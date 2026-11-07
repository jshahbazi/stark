#!/usr/bin/env python3
"""Qualify Gemma 3 through shipped Worker, Desktop and loopback HTTP paths.

Run with the sealed host package on PYTHONPATH and its stark-template on PATH.
Only aggregate receipts leave private model/reference pipes and memory.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import io
import json
import os
import re
import secrets
import socket
import threading
import time
from pathlib import Path

import stark_convert.chat as chat_module
from stark_convert.chat import ChatCancelled, ChatImageError, Worker, load_template, render_template
from stark_convert.desktop import Desktop
from stark_convert.local_api import LocalAPI
from stark_convert.media import ImageData
from gemma3_image_reference import REFERENCE_HASHES, REVISION

PHASE = "admission"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(value):
    if not value:
        raise ValueError("qualification_gate_failed")


def expected_generated(tokens, eos):
    """The canonical fixed-length trajectory includes EOS; worker output omits it."""
    values = list(tokens)
    for index, token in enumerate(values):
        if token in eos:
            return values[:index], token
    return values, None


def write_receipt(path, receipt):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def run(args):
    global PHASE
    model, fixture, executable = (path.resolve(strict=True) for path in (args.model, args.fixture, args.worker))
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {"passed": False, "worker_sha256": digest(executable),
               "fixture_sha256": digest(fixture / "manifest.json"), "checks": {}}
    receipt["qualifier_sha256"] = digest(Path(__file__).resolve())
    package = Path(chat_module.__file__).resolve().parent
    receipt["host_module_hashes"] = {name: digest(package / name) for name in (
        "chat.py", "desktop.py", "local_api.py", "image_checks.py", "vision_runtime.py", "media.py", "template_runtime.py")}
    manifest = json.loads((fixture / "manifest.json").read_text())
    require(manifest["source_repo"] == "google/gemma-3-4b-it" and len(manifest["cases"]) == 3)
    require(manifest["source_revision"] == REVISION and manifest["reference_hashes"] == REFERENCE_HASHES)
    require(manifest["torch"] == "2.14.1" and manifest["transformers"] == "5.15.1")
    require(manifest["tokens_per_case"] == 8 and [case["name"] for case in manifest["cases"]] == ["red", "blue", "pattern"])
    config = json.loads((model / "config.json").read_text())
    require(config["artifact_contract"]["family"] == {"profile": "gemma3-siglip-bf16-v1", "version": 1})
    source = config["artifact_contract"]["source"]
    require(source["repo"] == manifest["source_repo"] and source["revision"] == manifest["source_revision"])
    require(source["hashes"]["config.json"] == manifest["source_hashes"]["config.json"])
    require(source["weight_files"]["model.safetensors.index.json"]["sha256"] == manifest["source_hashes"]["model.safetensors.index.json"])
    require(digest(fixture / "resume-inputs.json") == manifest["resume_inputs_sha256"])
    resume = json.loads((fixture / "resume-inputs.json").read_text())
    require(all(resume["verified_source_files"][name] == record["sha256"] for name, record in source["weight_files"].items()))
    receipt["source_manifest_sha256"] = resume["source_manifest_sha256"]
    receipt["resume_inputs_sha256"] = manifest["resume_inputs_sha256"]
    receipt["reference_tool_sha256"] = digest(Path(__file__).with_name("gemma3_image_reference.py"))
    for case in manifest["cases"]:
        for name, checksum in case["hashes"].items():
            require(Path(name).name == name and digest(fixture / name) == checksum)
    receipt["artifact_sha256"] = digest(model / "model.safetensors")
    receipt["config_sha256"] = digest(model / "config.json")
    codec, template = load_template(model)
    receipt["template_helper_sha256"] = digest(template._executable)
    messages = [{"role": "user", "content": [{"type": "image"},
                 {"type": "text", "text": "Describe the image briefly."}]}]
    prompt = render_template(template, messages=messages, tools=None, add_generation_prompt=True,
                             enable_thinking=False, bos_token=codec.get("bos_token") or "",
                             eos_token=codec.get("eos_token") or "")
    options = {"context_tokens": 2048, "max_new_tokens": manifest["tokens_per_case"],
               "sampling": {"mode": "greedy"}}
    checks = args.image_check_directory or args.output / "image-checks"
    worker = None

    class ObservedWorker(Worker):
        def read(self, *arguments, **keywords):
            reply = super().read(*arguments, **keywords)
            code = reply.get("error")
            if isinstance(code, str) and re.fullmatch(r"[a-z_]{1,80}", code):
                errors = receipt.setdefault("native_error_codes", [])
                if len(errors) < 32:
                    errors.append(code)
                    write_receipt(args.output / "receipt.json", receipt)
            return reply

    def checkpoint(name, value=True):
        receipt["checks"][name] = value
        receipt["last_completed_phase"] = PHASE
        write_receipt(args.output / "receipt.json", receipt)
        print(json.dumps({"phase": PHASE, "completed": True}), flush=True)

    def clean():
        require(worker.process.poll() is None and not list(worker.media.root.iterdir()))

    def recover_text():
        result = worker.request({"operation": "generate", "rendered_prompt": "Hello.",
                                 "context_tokens": 2048, "max_new_tokens": 2})
        require("error" not in result and isinstance(result.get("text"), str))
        clean()

    try:
        PHASE = "cold_local_checks"
        PHASE = "cold_worker_spawn"
        worker = ObservedWorker(executable, model, timeout=180)
        worker.image_check_dir = checks
        PHASE = "cold_worker_read"
        ready = worker.read()
        worker.streaming = ready.get("streaming") == "text-delta-v1"
        worker.stop_sequences = ready.get("stop_sequences") is True
        PHASE = "cold_worker_admission"
        require(ready.get("ready") is True and worker.images)
        require(worker.image_check_cached is (args.image_check_directory is not None))
        require(worker.image_verification == "locally_tested" and worker.cancellation and not worker.prefix_cache)
        receipt["local_check_identity"] = worker._image_check_identity
        checkpoint("cached_local_checks" if worker.image_check_cached else "cold_local_checks")
        eos = config["generation_config"]["stop_token_ids"]
        require(isinstance(eos, list) and eos == [1, 106])
        outputs = []
        for index, case in enumerate(manifest["cases"]):
            PHASE = "canonical_worker_" + case["name"] + "_prompt"
            tokens = json.loads((fixture / case["tokens"]).read_text())
            raw = worker.request({"operation": "tokenize", "text": prompt, "add_special_tokens": False})
            require(raw.get("token_ids") == tokens["raw"])
            image = ImageData((fixture / case["image"]).read_bytes(), "image/png")
            if index == 0:
                PHASE = "prepared_single_use"
                with worker.media.stage(image) as reference:
                    prepared = worker.request({"operation": "prepare_image", "image": reference,
                                              "rendered_prompt": prompt,
                                              "context_tokens": options["context_tokens"],
                                              "max_new_tokens": options["max_new_tokens"]})
                    require("error" not in prepared and prepared["image_tokens"] == 256)
                    require(prepared["input_tokens"] == len(tokens["expanded"]))
                    reply = worker.request({"operation": "generate_image", "prepared_id": prepared["prepared_id"], **options})
                    require("error" not in reply)
                    reused = worker.request({"operation": "generate_image", "prepared_id": prepared["prepared_id"], **options})
                    require(reused.get("error") == "invalid_prepared_image")
                    require(type(worker.request({"operation": "discard_image"}).get("discarded")) is bool)
                checkpoint("prepared_single_use")
            else:
                PHASE = "canonical_worker_" + case["name"] + "_generation"
                reply = worker.request_image(prompt, image, options)
            PHASE = "canonical_worker_" + case["name"] + "_comparison"
            expected, stop = expected_generated(tokens["generated"], eos)
            require(reply.get("generated_token_ids") == expected)
            require(stop is None or reply.get("stop_token_id") == stop)
            require(reply["input_tokens"] == len(tokens["expanded"]) == len(tokens["raw"]) + 258)
            clean()
            outputs.append(reply["text"])
            checkpoint(case["name"] + "_canonical_greedy", {"exact": True, "input_tokens": reply["input_tokens"],
                       "output_tokens": len(expected), "image_tokens": 256})
        PHASE = "fresh_replay"
        first = manifest["cases"][0]
        image = ImageData((fixture / first["image"]).read_bytes(), "image/png")
        replay = worker.request_image(prompt, image, options)
        require(replay["text"] == outputs[0])
        clean()
        checkpoint("fresh_replay")
        PHASE = "malformed_recovery"
        try:
            worker.request_image(prompt, ImageData(b"\x89PNG\r\n\x1a\n\0", "image/png"), options)
        except ChatImageError:
            pass
        else:
            raise ValueError("malformed_image_accepted")
        recover_text()
        checkpoint("malformed_image_text_recovery")
        PHASE = "image_request_cancellation"
        cancelled = threading.Event()
        timer = threading.Timer(3.0, cancelled.set)
        timer.start()
        try:
            try:
                worker.request_image(prompt, image, options, cancelled=cancelled)
            except ChatCancelled:
                pass
            else:
                raise ValueError("image_not_cancelled")
        finally:
            timer.cancel()
            timer.join()
        recover_text()
        checkpoint("same_instance_cancellation_text_recovery")
        PHASE = "http_api_image"
        desktop = Desktop(executable, args.output / "desktop", output=io.StringIO())
        desktop.worker, desktop.codec, desktop.template = worker, codec, template
        desktop.active_model, desktop.active_mode = "qualification-gemma3", "chat"
        desktop.models = [{"id": desktop.active_model, "name": "Gemma3 qualification", "path": str(model),
                           "source": "local:" + str(model)}]
        desktop.context = desktop.api_context = 2048
        desktop.context_capacity_known = True
        desktop.api_maximum = 8
        desktop.api_vocab = 262208
        api = LocalAPI(desktop.api_models, desktop.api_complete, api_key=secrets.token_hex(32), port=0,
                       status=desktop.api_status, cancellable=True, request_timeout=180)
        info = api.start()
        try:
            port = int(info["base_url"].rsplit(":", 1)[1].split("/")[0])
            content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                       base64.b64encode(image.data).decode("ascii")}},
                       {"type": "text", "text": "Describe the image briefly."}]
            body = {"messages": [{"role": "user", "content": content}], "max_tokens": 8, "temperature": 0}
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=180)
            try:
                connection.request("POST", "/v1/chat/completions", body=json.dumps(body),
                                   headers={"Authorization": "Bearer " + info["api_key"], "Content-Type": "application/json"})
                response = connection.getresponse()
                payload = json.loads(response.read())
                status = response.status
            finally:
                connection.close()
            PHASE = "http_api_status"
            checkpoint("http_status", status)
            require(status == 200)
            PHASE = "http_api_content"
            require(payload["choices"][0]["message"]["content"] == outputs[0])
            PHASE = "http_api_accounting"
            checkpoint("http_usage", payload["usage"])
            require(payload["usage"]["prompt_tokens"] == 273)
            PHASE = "http_api_alignment"
            require(desktop.worker is worker and desktop.job is None and not desktop.api_queue)
            require(desktop.api_status()["capabilities"]["images"])
            clean()
            logs = desktop.output.getvalue()
            require(prompt not in logs and content[0]["image_url"]["url"] not in logs)
            require(not outputs[0] or outputs[0] not in logs)
            checkpoint("loopback_http_desktop_worker_image")
            PHASE = "http_disconnect_cancellation"
            before = worker.request_sequence
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=180)
            try:
                connection.request("POST", "/v1/chat/completions", body=json.dumps({**body, "stream": True}),
                                   headers={"Authorization": "Bearer " + info["api_key"], "Content-Type": "application/json"})
                deadline = time.monotonic() + 15
                while worker.request_sequence < before + 2 and time.monotonic() < deadline:
                    time.sleep(0.02)
                require(worker.request_sequence >= before + 2 and desktop.job is not None)
                connection.sock.shutdown(socket.SHUT_RDWR)
            finally:
                connection.close()
            deadline = time.monotonic() + 30
            while desktop.job is not None and time.monotonic() < deadline:
                time.sleep(0.02)
            require(desktop.job is None and desktop.worker is worker and not desktop.api_queue)
            recover_text()
            checkpoint("http_disconnect_same_instance_text_recovery")
        finally:
            api.close()
        recover_text()
        checkpoint("http_image_text_recovery")
        PHASE = "cached_restart"
        worker.close()
        worker = ObservedWorker(executable, model, timeout=180)
        worker.image_check_dir = checks
        require(worker.read().get("ready") is True and worker.images and worker.image_check_cached)
        recover_text()
        cached = worker.request_image(prompt, image, options)
        require(cached["text"] == outputs[0])
        clean()
        checkpoint("restart_cached_local_checks")
        receipt["passed"] = True
        write_receipt(args.output / "receipt.json", receipt)
        return receipt
    finally:
        if worker is not None:
            worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worker", "model", "fixture", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--image-check-directory", type=Path,
                        help="Reuse source-bound local checks from a preserved earlier run.")
    args = parser.parse_args()
    os.umask(0o077)
    try:
        result = run(args)
    except Exception as error:
        causes = []
        current = error.__context__
        while current is not None and len(causes) < 8:
            causes.append({"class": type(current).__name__, "message_sha256": hashlib.sha256(str(current).encode()).hexdigest()})
            current = current.__context__
        failure = {"passed": False, "phase": PHASE, "error_class": type(error).__name__,
                   "error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(), "causes": causes}
        if args.output.is_dir():
            write_receipt(args.output / "failure.json", failure)
        print(json.dumps(failure), flush=True)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
