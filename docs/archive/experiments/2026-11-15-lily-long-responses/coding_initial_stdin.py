"""Private, bounded long-completion comparison for the Lily top-k policy.

The exact user prompt arrives on stdin and is never written. Generated text is
kept in memory until all timed requests and worker sessions finish, then sent
once per unique output hash to the private program checker. Artifacts contain
only source hashes, counts, timings, hashes, and sanitized checker/host results.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path("/Users/johnshahbazian/Library/Application Support/Stark/models/qwen38-flash-next-lily-q4-xl-0f94842e")
BASE = Path("/Users/johnshahbazian/development/stark-llama-v1-study/runtime/llama-v1-baseline-target/release/stark-generate")
CANDIDATE = ROOT / "runtime/build-topk-candidate/release/stark-generate"
TEMPLATE_HELPER = ROOT / "runtime/build-candidate/release/stark-template"
IDENTITY = Path("/Users/johnshahbazian/development/stark-llama-v1-study/runtime/llama-v1-lily-baseline-20261115-v2/identity.json")
OLD_RUNNER = ROOT / "runtime/lily-topk-policy/run_performance.py"
CHECKER = ROOT / "runtime/lily-ttt-long/check_program.py"
OUT = ROOT / "runtime/lily-ttt-long/private-results-20261115"

BASE_SHA256 = "61b9c8cc724407573169eb2abad7f908e4943eead7ac3122eeec3ef841a48640"
CANDIDATE_SHA256 = "e1e432fcd5d490c8ab900be23c7cbc3855fc4515d75a38089bac672a85e9be22"
BUILD_IDENTITY_SHA256 = "273c56d4470b36bf7257e62b5431b8b55081dbbc1e88a2d08aaeb6e79e93a691"
NATIVE_PATCH_SHA256 = "24bc87a033386e9150a7d264afb546e83cf5dd86b1aee9e9f0bce8d3aa3797c0"
PYTHON_PATCH_SHA256 = "e947c9919583e68d09678c684b7de80ed400a472ff8873150465d7efc6efd307"
CONFIG_SHA256 = "2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899"
BUILD_IDENTITY = ROOT / "runtime/lily-topk-policy/build-identity.json"
NATIVE_PATCH = ROOT / "runtime/lily-topk-policy/native.patch"
PYTHON_PATCH = ROOT / "runtime/lily-topk-policy/python.patch"

IMAGE_PROFILE = "qwen38-lily-retained-bf16-vision-v1"
POLICY = "lily-topk20-one-draft-v1"
CONTEXT = 8192
MAX_NEW = 2048
WARMUP_MAX_NEW = 64
SESSIONS = ("before", "candidate", "candidate", "before")
SAMPLING = {
    "mode": "stochastic",
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_previous_runner():
    spec = importlib.util.spec_from_file_location("_lily_topk_previous_runner", OLD_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("previous_runner_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sanitize_host_screen(value: dict) -> dict:
    allowed = (
        "samples", "external_cpu_max", "external_model_cpu_max", "thermal_max",
        "compiler_active", "sampling_gap", "sampling_failures", "paging_delta",
        "eligible",
    )
    return {key: value[key] for key in allowed if key in value}


def validate_generation(reply: dict, ready: dict, build: str, input_tokens: int,
                        max_new_tokens: int) -> tuple[dict, str | None]:
    """Return content-free metrics and transient decoded text, if present."""
    if not isinstance(reply, dict) or "error" in reply:
        return {"request_status": "worker_error"}, None
    generated = reply.get("generated_token_ids")
    text = reply.get("text")
    if (not isinstance(generated, list) or any(type(token) is not int for token in generated)
            or not isinstance(text, str)):
        return {"request_status": "malformed_reply"}, None
    sampled = reply.get("sampled_tokens")
    stop = reply.get("stop_token_id")
    eos_ids = ready.get("eos_token_ids", [])
    stopped = type(stop) is int and stop in eos_ids
    finish = reply.get("finish_reason")
    count_valid = (
        type(sampled) is int
        and sampled == len(generated) + int(stopped)
        and finish in {"length", "eos"}
        and ((finish == "eos") == stopped)
        and (finish != "length" or stop is None)
        and sampled <= max_new_tokens
        and (finish != "length" or sampled == max_new_tokens)
    )
    ack = reply.get("mtp")
    expected_depth = 2 if build == "before" else 1
    stats = ack.get("stats") if isinstance(ack, dict) else None
    counter_keys = ("rounds", "proposed", "accepted", "verified_rows")
    counters = {key: stats.get(key) for key in counter_keys} if isinstance(stats, dict) else {}
    counters_valid = all(type(counters.get(key)) is int and counters[key] >= 0 for key in counter_keys)
    ack_valid = (
        isinstance(ack, dict)
        and ack.get("enabled") is True
        and ack.get("sampling") == "exact-rejection-v1"
        and type(ack.get("drafts")) is int
        and ack.get("drafts") == expected_depth
        and counters_valid
    )
    decode_seconds = reply.get("decode_seconds")
    generation_seconds = reply.get("generation_seconds")
    prefill_seconds = reply.get("prefill_seconds")
    timing_valid = (
        isinstance(decode_seconds, (int, float)) and not isinstance(decode_seconds, bool)
        and math.isfinite(decode_seconds) and decode_seconds > 0
        and isinstance(generation_seconds, (int, float)) and not isinstance(generation_seconds, bool)
        and math.isfinite(generation_seconds) and generation_seconds >= 0
        and (prefill_seconds is None or (
            isinstance(prefill_seconds, (int, float)) and not isinstance(prefill_seconds, bool)
            and math.isfinite(prefill_seconds) and prefill_seconds >= 0
        ))
    )
    token_count = reply.get("input_tokens")
    token_count_valid = type(token_count) is int and token_count == input_tokens
    raw_ids = b"".join(token.to_bytes(4, "little", signed=False) for token in generated)
    token_hash = hashlib.sha256(raw_ids).hexdigest()
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    row = {
        "request_status": "ok",
        "input_tokens": token_count,
        "prefix_reused_tokens": reply.get("reused_input_tokens", 0),
        "prefix_reuse_valid": reply.get("reused_input_tokens", 0) == 0,
        "output_tokens": len(generated),
        "sampled_tokens": sampled,
        "finish_reason": finish if finish in {"length", "eos"} else "unknown",
        "natural_eos": stopped,
        "output_token_ids_sha256": token_hash,
        "text_sha256": text_hash,
        "counts_valid": bool(count_valid and token_count_valid),
        "automatic_depth_valid": bool(ack_valid),
        "actual_drafts": ack.get("drafts") if isinstance(ack, dict) and type(ack.get("drafts")) is int else None,
        "mtp_counters": counters,
        "mtp_verification": ack.get("verification") if isinstance(ack, dict) else None,
        "decode_seconds": decode_seconds,
        "generation_seconds": generation_seconds,
        "prefill_seconds": prefill_seconds,
        "useful_decode_tokens_per_second": (
            (len(generated) - 1) / decode_seconds if len(generated) > 1 and timing_valid else None
        ),
        "timings_valid": bool(timing_valid),
    }
    return row, text


def render_and_tokenize(prompt: str, worker, codec, template, chat, reasoning):
    system = codec.get("default_system_prompt")
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    rendered = chat.render_template(
        template,
        messages=messages,
        tools=None,
        add_generation_prompt=True,
        **reasoning.reasoning_options(codec, thinking=False),
        bos_token=codec.get("bos_token") or "",
        eos_token=codec.get("eos_token") or "",
    )
    ids = worker.tokenize(rendered)
    if (not isinstance(ids, list)
            or any(type(token) is not int or token < 0 for token in ids)):
        raise RuntimeError("prompt_tokenization_failed")
    codec_json = json.dumps(codec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return rendered, ids, hashlib.sha256(codec_json.encode("utf-8")).hexdigest()


def monitor_request(worker, request: dict, monitor, phase: str, before_paging, paging, screen_attempt):
    monitor.set_phase(phase)
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.monotonic()
    failure_type = None
    try:
        reply = worker.request(request)
    except BaseException as error:
        reply = None
        failure_type = type(error).__name__
    wall_seconds = time.monotonic() - started
    finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    after_paging = paging()
    try:
        screen = sanitize_host_screen(screen_attempt(monitor, phase, before_paging, after_paging))
    except Exception as error:  # Persist only a type label, never exception text.
        screen = {"screen_failure_type": type(error).__name__}
    return reply, wall_seconds, started_utc, finished_utc, screen, failure_type


def run() -> int:
    os.umask(0o077)
    if OUT.exists():
        raise RuntimeError("private_results_already_exist")
    required = (MODEL / "config.json", MODEL / "native-source.json", BASE, CANDIDATE,
                TEMPLATE_HELPER, IDENTITY, OLD_RUNNER, CHECKER, BUILD_IDENTITY,
                NATIVE_PATCH, PYTHON_PATCH)
    if not all(path.is_file() for path in required):
        raise RuntimeError("sealed_input_missing")

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "tools/converter"))
    from stark_convert import chat, reasoning, template_runtime
    from benchmark_clean_qwen import Monitor, paging
    from benchmark_glm_mlx_serve import screen_attempt

    for name in tuple(os.environ):
        if name.startswith(("STARK_", "UZU_", "LILY_", "GGML_")):
            os.environ.pop(name)
    os.environ["STARK_REQUEST_BUDGET_MIB"] = "16"
    os.environ["PATH"] = str(TEMPLATE_HELPER.parent) + os.pathsep + os.environ.get("PATH", "")
    if shutil.which("stark-template") != str(TEMPLATE_HELPER):
        raise RuntimeError("template_helper_path_mismatch")
    template_runtime._template_executable = lambda: TEMPLATE_HELPER.resolve()

    old = load_previous_runner()
    expected_identity = json.loads(IDENTITY.read_text())
    if not old.current_model_identity(expected_identity):
        raise RuntimeError("model_identity_mismatch")
    if sha256_file(BASE) != BASE_SHA256:
        raise RuntimeError("before_worker_hash_mismatch")
    if sha256_file(CANDIDATE) != CANDIDATE_SHA256:
        raise RuntimeError("candidate_worker_hash_mismatch")
    if (sha256_file(BUILD_IDENTITY) != BUILD_IDENTITY_SHA256
            or sha256_file(NATIVE_PATCH) != NATIVE_PATCH_SHA256
            or sha256_file(PYTHON_PATCH) != PYTHON_PATCH_SHA256
            or sha256_file(MODEL / "config.json") != CONFIG_SHA256):
        raise RuntimeError("candidate_source_seal_mismatch")
    native_receipt = json.loads((MODEL / "native-source.json").read_text())
    weights_source = native_receipt.get("weights_source") or {}
    if (native_receipt.get("config_sha256") != CONFIG_SHA256
            or weights_source.get("repo") != "fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl"
            or weights_source.get("revision") != "0f94842e3682b49acace4daeb4cf2e65a6f94f83"):
        raise RuntimeError("source_receipt_mismatch")

    template_sha256 = sha256_file(MODEL / "chat_template.jinja")
    codec, template = chat.load_template(
        MODEL,
        expected_hashes={"config_sha256": CONFIG_SHA256, "template_sha256": template_sha256},
    )
    if Path(template._executable).resolve() != TEMPLATE_HELPER.resolve():
        raise RuntimeError("template_helper_identity_mismatch")

    print("ready_for_prompt", flush=True)
    line = sys.stdin.readline()
    try:
        supplied = json.loads(line)
    except (ValueError, TypeError):
        raise RuntimeError("invalid_prompt_pipe") from None
    prompt = supplied.get("prompt") if isinstance(supplied, dict) else None
    if not isinstance(prompt, str) or not prompt.strip() or set(supplied) != {"prompt"}:
        raise RuntimeError("invalid_prompt_pipe")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    OUT.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "stark-lily-ttt-long-completion-v1",
        "attempt": "20261115-LILY-LONG-01",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_sha256": prompt_sha256,
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "prompt_policy": "one exact natural-language prompt supplied on stdin; text not persisted",
        "sampling": SAMPLING,
        "sampling_note": "min_p omitted; seed 42; different MTP depths can produce different token sequences",
        "context_tokens": CONTEXT,
        "max_new_tokens": MAX_NEW,
        "warmup_max_new_tokens": WARMUP_MAX_NEW,
        "warmup_count": 4,
        "measured_count": 8,
        "session_build_order": list(SESSIONS),
        "measurements_per_session": 2,
        "prefix_cache": False,
        "vision_worker": True,
        "image_profile_required": IMAGE_PROFILE,
        "mtp_policy": "mtp=true AUTO in both builds; require before=2, candidate=1",
        "request_budget_mib": 16,
        "host_observation_policy": "descriptive only; no quiet gate, filtering, exclusions or retries",
        "quality_policy": "hold generated strings in RAM until all measured requests and workers finish; check once per unique text hash; persist only booleans/counts",
        "worker_sha256": {"before": sha256_file(BASE), "candidate": sha256_file(CANDIDATE)},
        "template_helper_sha256": sha256_file(TEMPLATE_HELPER),
        "template_source_sha256": template_sha256,
        "codec_policy_sha256": hashlib.sha256(
            json.dumps(codec, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "model_content_manifest_sha256": expected_identity.get("model_content_manifest_sha256"),
        "model_identity_receipt_sha256": sha256_file(IDENTITY),
        "native_source_receipt_sha256": sha256_file(MODEL / "native-source.json"),
        "model_config_sha256": CONFIG_SHA256,
        "build_identity_sha256": sha256_file(BUILD_IDENTITY),
        "native_patch_sha256": sha256_file(NATIVE_PATCH),
        "python_patch_sha256": sha256_file(PYTHON_PATCH),
        "harness_sha256": sha256_file(Path(__file__)),
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "sessions": [],
    }
    write(OUT / "manifest.json", manifest)

    prompt_codec_hashes: list[str] = []
    rendered_hashes: list[str] = []
    token_hashes: list[str] = []
    response_texts: dict[str, str] = {}
    all_requests: list[dict] = []
    for session_index, build in enumerate(SESSIONS, start=1):
        binary = BASE if build == "before" else CANDIDATE
        if sha256_file(binary) != manifest["worker_sha256"][build]:
            raise RuntimeError("worker_changed_before_session")
        session = {"session": session_index, "build": build, "setup_failure_type": None,
                   "ready": None, "warmup": None, "measurements": []}
        manifest["sessions"].append(session)
        worker = None
        monitor = None
        try:
            from validate_generation_prefix_cache import Worker
            worker = Worker(str(binary), str(MODEL), prefix_cache=False, timeout=900, vision=True)
            ready = worker.ready
            mtp = ready.get("mtp") or {}
            image = ready.get("image_input") or {}
            if (mtp.get("supported") is not True or type(mtp.get("max_drafts")) is not int
                    or mtp["max_drafts"] < 2 or image.get("family_profile") != IMAGE_PROFILE):
                raise RuntimeError("worker_readiness_mismatch")
            session["ready"] = {
                "mtp_supported": mtp["supported"],
                "mtp_max_drafts": mtp["max_drafts"],
                "mtp_recommended_drafts": mtp.get("recommended_drafts"),
                "mtp_sampling": mtp.get("sampling"),
                "image_family_profile": image.get("family_profile"),
                "image_verification": image.get("verification"),
                "prefix_cache_enabled": (ready.get("prefix_cache") or {}).get("enabled"),
            }
            rendered, ids, codec_hash = render_and_tokenize(prompt, worker, codec, template, chat, reasoning)
            rendered_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            token_hash = hashlib.sha256(b"".join(x.to_bytes(4, "little") for x in ids)).hexdigest()
            prompt_codec_hashes.append(codec_hash)
            rendered_hashes.append(rendered_hash)
            token_hashes.append(token_hash)
            if len(ids) + MAX_NEW > CONTEXT:
                raise RuntimeError("prompt_exceeds_fixed_context")
            session["input_tokens"] = len(ids)
            session["rendered_prompt_sha256"] = rendered_hash
            session["input_token_ids_sha256"] = token_hash
            session["template_source_sha256"] = template_sha256
            session["template_helper_sha256"] = sha256_file(TEMPLATE_HELPER)
            monitor = Monitor().__enter__()
            before_paging = paging()
            warm_payload = {
                "operation": "generate", "rendered_prompt": rendered,
                "context_tokens": CONTEXT, "max_new_tokens": WARMUP_MAX_NEW,
                "seed": 42, "sampling": SAMPLING, "mtp": True,
            }
            reply, wall, started, finished, host, transport_failure = monitor_request(
                worker, warm_payload, monitor, f"session{session_index}-warmup",
                before_paging, paging, screen_attempt,
            )
            warm, _warm_text = validate_generation(reply, ready, build, len(ids), WARMUP_MAX_NEW)
            if transport_failure:
                warm["transport_failure_type"] = transport_failure
            warm.update({"wall_seconds": wall, "started_utc": started,
                         "finished_utc": finished, "host": host})
            session["warmup"] = warm
            write(OUT / f"session-{session_index}.json", session)

            for repetition in range(1, 3):
                phase = f"session{session_index}-measurement{repetition}"
                before_paging = paging()
                payload = {
                    "operation": "generate", "rendered_prompt": rendered,
                    "context_tokens": CONTEXT, "max_new_tokens": MAX_NEW,
                    "seed": 42, "sampling": SAMPLING, "mtp": True,
                }
                reply, wall, started, finished, host, transport_failure = monitor_request(
                    worker, payload, monitor, phase, before_paging, paging, screen_attempt,
                )
                row, response_text = validate_generation(reply, ready, build, len(ids), MAX_NEW)
                if transport_failure:
                    row["transport_failure_type"] = transport_failure
                row.update({"session": session_index, "build": build,
                            "repetition": repetition, "wall_seconds": wall,
                            "started_utc": started, "finished_utc": finished,
                            "host": host, "mtp_selector": "auto"})
                if response_text is not None and row.get("request_status") == "ok":
                    response_texts.setdefault(row["text_sha256"], response_text)
                session["measurements"].append(row)
                all_requests.append(row)
                write(OUT / f"session-{session_index}.json", session)
        except BaseException as error:
            # The exception payload is intentionally discarded; no prompt data
            # may enter a saved diagnostic.
            session["setup_failure_type"] = type(error).__name__
            write(OUT / f"session-{session_index}.json", session)
        finally:
            if monitor is not None:
                monitor.__exit__(None, None, None)
                session["monitor_summary"] = {
                    "samples": len(monitor.rows),
                    "sampling_failures": monitor.failures,
                }
            if worker is not None:
                worker.close()
        write(OUT / f"session-{session_index}.json", session)
        write(OUT / "manifest.json", manifest)
        if session.get("setup_failure_type") is not None:
            break

    manifest["prompt_codec_sha256"] = prompt_codec_hashes
    manifest["rendered_prompt_sha256"] = rendered_hashes
    manifest["input_token_ids_sha256"] = token_hashes
    manifest["prompt_tokenization_match"] = (
        len(set(prompt_codec_hashes)) == 1
        and len(set(rendered_hashes)) == 1
        and len(set(token_hashes)) == 1
    )
    manifest["measured_rows_written"] = len(all_requests)
    for build, depth in (("before", 2), ("candidate", 1)):
        rows = [row for row in all_requests if row.get("build") == build]
        successful = [row for row in rows if row.get("request_status") == "ok"]
        manifest.setdefault("same_arm_repeatability", {})[build] = {
            "expected_depth": depth,
            "row_count": len(rows),
            "successful_row_count": len(successful),
            "same_output_token_hash": bool(successful) and len({row.get("output_token_ids_sha256") for row in successful}) == 1,
            "same_text_hash": bool(successful) and len({row.get("text_sha256") for row in successful}) == 1,
            "same_output_length": bool(successful) and len({row.get("output_tokens") for row in successful}) == 1,
            "same_mtp_counters": len({json.dumps(row.get("mtp_counters"), sort_keys=True)
                                      for row in successful}) == 1 if successful else False,
        }

    # Quality work is deliberately deferred until every measured generation
    # has completed and all native workers have closed.
    checker_spec = importlib.util.spec_from_file_location("_lily_ttt_checker", CHECKER)
    if checker_spec is None or checker_spec.loader is None:
        raise RuntimeError("quality_checker_unavailable")
    checker = importlib.util.module_from_spec(checker_spec)
    checker_spec.loader.exec_module(checker)
    manifest["quality_checker_sha256"] = sha256_file(CHECKER)
    quality = {}
    for digest, text in response_texts.items():
        try:
            result = checker.check_response(text)
            quality[digest] = {
                key: result.get(key) for key in (
                    "syntax_valid", "normal_exit", "game_end_signal", "input_calls",
                    "gui_import_detected", "unsupported_import", "sandbox_or_runtime_failure",
                    "replays",
                ) if isinstance(result, dict) and key in result
            }
        except BaseException as error:
            quality[digest] = {"quality_check_failure_type": type(error).__name__}
    manifest["quality_by_text_sha256"] = quality
    manifest["quality_unique_output_count"] = len(quality)
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["completed"] = len(all_requests) == 8
    write(OUT / "manifest.json", manifest)
    response_texts.clear()
    return 0 if manifest["completed"] else 2


if __name__ == "__main__":
    try:
        code = run()
    except BaseException as error:
        # Type only: never echo exception text, which could contain request data.
        print(json.dumps({"status": "failed", "failure_type": type(error).__name__}), flush=True)
        code = 2
    raise SystemExit(code)
