"""Measure one isolated Unsloth Inkling GGUF session. Only aggregate responses leave memory."""

import argparse
import base64
import binascii
import ctypes
import hashlib
import json
import os
import re
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

SERVER_SHA = "9c99f4b9182aa90830c9bd15d2635707afeab55e7e64dbe3f37de97eaf4115a8"
FIXTURE_SHA = "fade74992579dc76ff0f5319eec5b70c9e07be7c54338ebae7f43c1097fe98bd"
PROC = ctypes.CDLL("/usr/lib/libproc.dylib")
PROC.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]


def memory(pid):
    # Darwin rusage_info_v4 begins with a UUID and then 64-bit counters.
    buffer = ctypes.create_string_buffer(1024)
    if PROC.proc_pid_rusage(pid, 4, buffer) != 0:
        return {}
    counters = struct.unpack_from("=35Q", buffer, 16)
    return {
        "resident_bytes": counters[6],
        "phys_footprint_bytes": counters[7],
        "lifetime_max_phys_footprint_bytes": counters[28],
    }


def host_memory():
    pressure = subprocess.check_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True)
    stats = subprocess.check_output(["vm_stat"], text=True)
    result = {"pressure_level": int(pressure)}
    for counter in ("Swapins", "Swapouts"):
        match = re.search(rf"^{counter}:\s+(\d+)", stats, re.MULTILINE)
        if match:
            result[counter.lower()] = int(match.group(1))
    return result


def request(url, payload=None, timeout=600):
    encoded = json.dumps(payload).encode() if payload is not None else None
    return urllib.request.urlopen(
        urllib.request.Request(url, encoded, headers={"Content-Type": "application/json"}),
        timeout=timeout,
    )


def capture_diagnostics(stream, diagnostics):
    # Keep fixed fields only. The server can emit arbitrary content in diagnostics.
    for row in stream:
        if re.search(rb"using device (MTL\d*|Metal|GPU)\b", row, re.IGNORECASE):
            diagnostics["metal_selected"] = True
        match = re.search(rb"offloaded (\d+)/(\d+) layers to GPU", row)
        if match:
            diagnostics["offloaded_layers"] = list(map(int, match.groups()))
        if re.search(rb"(clip_model_loader|load_hparams|mtmd).*inkling2v", row, re.IGNORECASE):
            diagnostics["projector_architecture_inkling"] = True
        match = re.search(
            rb"(Metal|MTL\d+|CPU)[^\n]*?\b(model|compute|KV) buffer size\s*=\s*([\d.]+) MiB", row, re.IGNORECASE
        )
        if match:
            device, kind, size = match.groups()
            diagnostics.setdefault("buffers_mib", []).append([device.decode(), kind.decode(), float(size)])


def completion(base_url, case, n_predict=256):
    prompt = case["prompt"]
    if hashlib.sha256(prompt.encode()).hexdigest() != case["prompt_sha256"]:
        raise RuntimeError("fixture hash mismatch")
    payload = {
        "prompt": prompt,
        "stream": True,
        "return_tokens": True,
        "n_predict": n_predict,
        "temperature": 0.0 if case["mode"] == "greedy" else 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "seed": case["seed"],
        "cache_prompt": False,
        "id_slot": 0,
        "ignore_eos": False,
    }
    epoch_start = time.time_ns()
    t0 = time.perf_counter_ns()
    first = None
    first_text = None
    final = None
    output_digest = hashlib.sha256()
    token_digest = hashlib.sha256()
    count = 0
    output_bytes = 0
    replacement_seen = False
    with request(base_url + "/completion", payload) as stream:
        for line in stream:
            if not line.startswith(b"data:"):
                continue
            data = line.removeprefix(b"data:").strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("stop"):
                final = event
                continue
            if "error" in event:
                raise RuntimeError("stream contains an error")
            if first is None and event.get("tokens_predicted", 0) > 0:
                first = time.perf_counter_ns()
            if first_text is None and event.get("content"):
                first_text = time.perf_counter_ns()
            fragment = event.get("content", "").encode()
            replacement_seen |= b"\xef\xbf\xbd" in fragment
            output_bytes += len(fragment)
            output_digest.update(fragment)
            for token in event.get("tokens", []):
                token_digest.update(struct.pack("<I", token))
                count += 1
    end = time.perf_counter_ns()
    epoch_end = time.time_ns()
    if final is None or first is None:
        raise RuntimeError("missing completion boundary")
    timings = final["timings"]
    measured = {k: case[k] for k in ("id", "window", "input_tokens", "mode", "seed", "prompt_sha256") if k in case}
    measured.update(
        {
            "wanted_output": n_predict,
            "tokens_evaluated": final["tokens_evaluated"],
            "stop_type": final["stop_type"],
            "truncated": bool(final["truncated"]),
            "stream_token_ids": count,
            "stream_output_sha256": output_digest.hexdigest(),
            "stream_output_bytes": output_bytes,
            "replacement_seen": replacement_seen,
            "stream_token_ids_sha256": token_digest.hexdigest(),
            "first_token_seconds": (first - t0) / 1e9,
            "ttft_seconds": (first_text - t0) / 1e9 if first_text is not None else None,
            "epoch_start_ns": epoch_start,
            "epoch_end_ns": epoch_end,
            "elapsed_seconds": (end - t0) / 1e9,
            "client_decode_seconds": (end - first) / 1e9,
        }
    )
    measured["timings"] = {
        k: timings[k]
        for k in ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "draft_n", "draft_n_accepted")
        if k in timings
    }
    settings = final.get("generation_settings", {})
    measured["effective_sampling"] = {
        key: settings[key]
        for key in (
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "repeat_penalty",
            "seed",
            "ignore_eos",
            "n_predict",
            "n_ctx",
        )
        if key in settings
    }
    measured["eligible"] = (
        timings.get("cache_n") == 0
        and timings["prompt_n"] == case["input_tokens"]
        and timings["predicted_n"] == n_predict
        and not measured["truncated"]
        and final["stop_type"] == "limit"
        and timings.get("draft_n", 0) == 0
    )
    return measured


def image_smoke(base_url):
    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
        )

    width = height = 80
    scanline = b"\x00" + (b"\xff\x00\x00" * width)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    image = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanline * height))
        + chunk(b"IEND", b"")
    )
    part = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode("ascii")},
    }
    text = {"type": "text", "text": "Name the color of the square in one word."}
    summary = {}
    for name, parts in (("text", [text]), ("image", [part, text])):
        payload = {
            "messages": [{"role": "user", "content": parts}],
            "temperature": 0,
            "max_tokens": 4,
            "stream": False,
            "cache_prompt": False,
        }
        with request(base_url + "/v1/chat/completions", payload) as stream:
            reply = json.load(stream)
        summary[name] = {key: reply["usage"][key] for key in ("prompt_tokens", "completion_tokens")}
        summary[name]["finish_reason"] = reply["choices"][0]["finish_reason"]
        summary[name]["cache_n"] = reply.get("timings", {}).get("cache_n")
    summary["image_width"] = width
    summary["image_height"] = height
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        required=True,
        type=Path,
        help="Directory containing verified UD-Q4_K_XL and mmproj-F16 GGUF files",
    )
    parser.add_argument("--server", required=True, type=Path, help="Pinned ARM64 llama-server executable")
    parser.add_argument("--fixtures", required=True, type=Path, help="Private workload JSON from workloads.py")
    parser.add_argument("--output", required=True, type=Path, help="New private aggregate JSONL")
    parser.add_argument("--windows", default="0,1,2,3")
    parser.add_argument("--cases", default="")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--image-smoke", action="store_true", help="Run text/image controls after timed requests")
    parser.add_argument(
        "--token-controls", type=Path, help="Private one-token greedy fixtures run after timed requests"
    )
    args = parser.parse_args()
    os.umask(0o077)
    raw = args.fixtures.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA:
        raise ValueError("unexpected workload hash")
    if hashlib.sha256(args.server.read_bytes()).hexdigest() != SERVER_SHA:
        raise ValueError("unexpected server executable hash")
    cases = json.loads(raw)["cases"]
    windows = {int(x) for x in args.windows.split(",")}
    ids = set(filter(None, args.cases.split(",")))
    selected = [x for x in cases if x["id"] in ids or not ids and x["window"] in windows]
    if not selected:
        raise ValueError("no workloads selected")
    workload = args.output.open("x", buffering=1)

    def publish(measurement):
        workload.write(json.dumps(measurement, sort_keys=True) + "\n")

    with socket.socket() as channel:
        channel.bind(("127.0.0.1", 0))
        port = channel.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    diagnostics = {}
    server = None
    options = [
        "--model",
        str(args.artifacts / "Inkling-2-Mini-UD-Q4_K_XL.gguf"),
        "--mmproj",
        str(args.artifacts / "mmproj-F16.gguf"),
        "--jinja",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        "8192",
        "--parallel",
        "1",
        "--spec-type",
        "none",
        "--n-gpu-layers",
        "99",
        "--no-cache-prompt",
        "--batch-size",
        "2048",
        "--ubatch-size",
        "512",
    ]
    try:
        publish({"kind": "host_before_load", "memory": host_memory(), "epoch_ns": time.time_ns()})
        start = time.perf_counter()
        server = subprocess.Popen([str(args.server), *options], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        pid = server.pid
        thread = threading.Thread(target=capture_diagnostics, args=(server.stderr, diagnostics), daemon=True)
        thread.start()
        while time.perf_counter() - start < 360:
            if server is not None and server.poll() is not None:
                thread.join(timeout=1)
                publish({"kind": "load_failed", "exit_code": server.returncode, "diagnostics": diagnostics})
                raise RuntimeError("server exited before readiness")
            try:
                with request(base_url + "/health", timeout=3) as response:
                    if json.load(response).get("status") == "ok":
                        break
            except (OSError, urllib.error.HTTPError):
                pass
            time.sleep(0.25)
        else:
            raise RuntimeError("server initialization timed out")
        ready_latency = time.perf_counter() - start
        with request(base_url + "/props") as response:
            props = json.load(response)
        publish(
            {
                "kind": "ready",
                "process_start_to_ready_s": ready_latency,
                "pid": pid,
                "port": port,
                "diagnostics": diagnostics,
                "memory": memory(pid),
                "modalities": props.get("modalities", {}),
                "actual_context": props.get("default_generation_settings", {}).get("n_ctx"),
                "server_sha256": SERVER_SHA,
                "workload_sha256": FIXTURE_SHA,
            }
        )
        if not props.get("modalities", {}).get("vision"):
            raise RuntimeError("image-loaded reference does not advertise vision")
        warmup = next(x for x in cases if x["id"] == "g512-00")
        warm = completion(base_url, warmup, n_predict=4)
        publish({"kind": "warm", "result": warm, "memory": memory(pid)})
        if warm["timings"]["prompt_n"] != 512 or warm["timings"].get("cache_n") != 0:
            raise RuntimeError("invalid warm prompt token count or cache")
        for case in [] if args.smoke_only else selected:
            result = completion(base_url, case)
            publish({"kind": "sample", "result": result, "memory": memory(pid)})
            print(
                f"{case['id']} pp={result['timings']['prompt_n']} tg={result['timings']['predicted_n']}"
                f" eligible={result['eligible']}",
                flush=True,
            )
        if args.token_controls:
            control_bytes = args.token_controls.read_bytes()
            publish({"kind": "control_manifest", "sha256": hashlib.sha256(control_bytes).hexdigest()})
            for case in json.loads(control_bytes)["cases"]:
                if case["mode"] != "greedy":
                    raise ValueError("expected greedy controls")
                publish({"kind": "greedy_token_control", "result": completion(base_url, case, n_predict=1)})
        if args.image_smoke:
            publish({"kind": "image_control", "result": image_smoke(base_url), "memory": memory(pid)})
        publish(
            {
                "kind": "complete",
                "memory": memory(pid),
                "diagnostics": diagnostics,
                "host_memory": host_memory(),
                "epoch_ns": time.time_ns(),
            }
        )
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        workload.close()


if __name__ == "__main__":
    main()
