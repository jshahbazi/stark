"""Model-free tests for scripts/benchmark_inkling_engine_matrix.py.

A fake worker stands in for stark-generate, a fake HTTP server for MLX-Serve
and a fake Node script for llmprobe. Nothing here loads a model, starts an
engine or touches the GPU.
"""

from __future__ import annotations

import io
import json
import os
import stat
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import benchmark_inkling_engine_matrix as bench
import pytest

SENTINEL = "PRIVATE-OUTPUT-SENTINEL"
PROMPT_SENTINEL = "PRIVATE-PROMPT-SENTINEL"

BATT_AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged; 0:00 remaining present: true\n"
BATT_BATTERY = (
    "Now drawing from 'Battery Power'\n"
    " -InternalBattery-0 (id=22610019)\t77%; discharging; 1:35 remaining present: true\n"
)
THERM_NONE = (
    "Note: No thermal warning level has been recorded\n"
    "Note: No performance warning level has been recorded\n"
    "Note: No CPU power status has been recorded\n"
)
THERM_LIMITED = (
    "CPU_Scheduler_Limit \t= 100\nCPU_Available_CPUs \t= 18\nCPU_Speed_Limit \t= 80\n"
)


def host_with(batt=BATT_AC, low_power="0", competing=""):
    outputs = {
        ("/usr/bin/pmset", "-g", "batt"): batt,
        (
            "/usr/bin/pmset",
            "-g",
        ): f"System-wide power settings:\n lowpowermode         {low_power}\n",
        ("/usr/bin/pmset", "-g", "therm"): THERM_NONE,
        ("/usr/bin/sw_vers", "-productVersion"): "26.7.2\n",
        ("/usr/bin/sw_vers", "-buildVersion"): "25H200\n",
        ("/usr/sbin/sysctl", "-n", "hw.model"): "Mac17,7\n",
        ("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"): "Apple M5 Max\n",
        ("/usr/sbin/sysctl", "-n", "hw.memsize"): "137438953472\n",
        ("/bin/ps", "-axo", "pid=,comm="): "  1 /sbin/launchd\n" + competing,
    }

    def runner(command, timeout=30):
        return outputs.get(tuple(command))

    return lambda: bench.host_snapshot(runner)


def make_executable(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def layout(tmp_path):
    """Fake binaries, model and llmprobe checkout with the shape the harness checks."""
    binary = make_executable(tmp_path / "stark-generate", b"\x00current build")
    baseline = make_executable(
        tmp_path / "stark-generate-baseline", b"\x00earlier build"
    )
    mlx = make_executable(tmp_path / "mlx-serve", b"\x00mlx")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"vocab_size": 201024}))
    (model / "model-00001-of-00001.safetensors").write_bytes(b"\x00" * 16)
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}})
    )
    probe = tmp_path / "llmprobe"
    (probe / "bin/dist").mkdir(parents=True)
    (probe / "src/bench").mkdir(parents=True)
    (probe / "node_modules").mkdir()
    (probe / "package.json").write_text(
        json.dumps({"name": "llmprobe", "version": "0.6.18"})
    )
    (probe / "bin/dist/llmprobe.mjs").write_text(
        "// bundle\nprocess.stderr.write('[inkling-bench-sample] ')\n"
    )
    (probe / "src/bench/index.ts").write_text("// index\n")
    (probe / "src/bench/corpus.ts").write_text("// corpus\n")
    return SimpleNamespace(
        root=tmp_path,
        binary=binary,
        baseline=baseline,
        mlx=mlx,
        model=model,
        probe=probe,
    )


def stark_args(layout, *extra):
    return [
        "stark",
        "--stark-binary",
        str(layout.binary),
        "--stark-model",
        str(layout.model),
        "--llmprobe",
        str(layout.probe),
        "--output",
        str(layout.root / "out"),
        *extra,
    ]


# --------------------------------------------------------------------------
# Power guard and host facts
# --------------------------------------------------------------------------


def test_pmset_batt_parses_ac_and_battery():
    ac = bench.parse_pmset_batt(BATT_AC)
    assert ac["source"] == "AC Power" and ac["on_ac"] is True
    battery = bench.parse_pmset_batt(BATT_BATTERY)
    assert battery == {
        "source": "Battery Power",
        "on_ac": False,
        "batteries": [{"percent": 77, "state": "discharging"}],
    }
    assert bench.parse_pmset_batt("Now drawing from 'UPS Power'\n")["on_ac"] is False
    assert bench.parse_pmset_batt(None)["on_ac"] is False
    assert bench.parse_pmset_batt("garbage")["source"] is None


def test_pmset_therm_and_low_power_mode():
    quiet = bench.parse_pmset_therm(THERM_NONE)
    assert quiet["thermal_warning"] == "none_recorded" and quiet["available"] is True
    limited = bench.parse_pmset_therm(THERM_LIMITED)
    assert limited["limits"] == {
        "CPU_Scheduler_Limit": 100,
        "CPU_Available_CPUs": 18,
        "CPU_Speed_Limit": 80,
    }
    assert bench.parse_pmset_therm(None)["available"] is False
    assert bench.parse_low_power_mode(" lowpowermode         1\n") is True
    assert bench.parse_low_power_mode(" powermode 0\n") is False
    assert bench.parse_low_power_mode(" powermode            1\n") is True
    assert bench.parse_low_power_mode(" powermode            2\n") is False
    assert bench.parse_low_power_mode(" lowpowermode 0\n powermode 2\n") is False
    assert bench.parse_low_power_mode("nothing") is None


def test_power_guard_refuses_battery_low_power_and_busy_host():
    bench.require_timing_host(host_with()())
    with pytest.raises(bench.BenchError, match="not_on_ac_power"):
        bench.require_timing_host(host_with(BATT_BATTERY)())
    with pytest.raises(bench.BenchError, match="low_power_mode_enabled"):
        bench.require_timing_host(host_with(low_power="1")())
    busy = host_with(
        competing="  99 /Users/x/target/release/stark-generate\n 100 cargo\n"
    )()
    assert busy["competing_processes"] == ["cargo", "stark-generate"]
    with pytest.raises(
        bench.BenchError, match="competing_processes_running:cargo,stark-generate"
    ):
        bench.require_timing_host(busy)
    bench.require_timing_host(busy, allow_busy_host=True)


def test_host_snapshot_records_identity():
    snapshot = host_with()()
    assert (
        snapshot["macos_version"] == "26.7.2"
        and snapshot["hardware_model"] == "Mac17,7"
    )
    assert (
        snapshot["chip"] == "Apple M5 Max" and snapshot["memory_bytes"] == 137438953472
    )
    assert snapshot["thermal"]["thermal_warning"] == "none_recorded"


def test_sleep_events_counts_only_the_window():
    log = (
        "2026-11-22 13:40:00 -0800 Sleep               \tEntering Sleep state due to 'Idle Sleep'\n"
        "2026-11-22 15:40:00 -0800 Sleep               \tEntering Sleep state due to 'Idle Sleep'\n"
        "2026-11-22 13:41:00 -0800 Wake                \tDarkWake from Deep Idle\n"
    )
    start = bench.dt.datetime.strptime(
        "2026-11-22 13:00:00 -0800", "%Y-%m-%d %H:%M:%S %z"
    ).timestamp()
    end = start + 3600
    assert bench.sleep_events(start, end, runner=lambda *a, **k: log) == {
        "checked": True,
        "sleep_events": 1,
    }
    assert (
        bench.sleep_events(start, end, runner=lambda *a, **k: None)["checked"] is False
    )


# --------------------------------------------------------------------------
# Configuration: arms are binaries
# --------------------------------------------------------------------------


def parse(argv):
    args = bench.parser().parse_args(argv)
    args.engine = args.command
    return args


def test_stark_arms_are_binaries_with_optional_diagnostic_env(layout):
    args = parse(
        stark_args(
            layout,
            "--stark-arm",
            f"baseline={layout.baseline}",
            "--stark-env",
            "baseline=STARK_DIAGNOSTIC=1",
        )
    )
    config = bench.build_config(args)
    assert [arm.name for arm in config.arms] == ["stark-current", "stark-baseline"]
    assert (
        config.arms[0].binary == layout.binary
        and config.arms[1].binary == layout.baseline
    )
    assert config.arms[0].env == {} and config.arms[1].env == {"STARK_DIAGNOSTIC": "1"}
    assert config.reasoning == "off" and config.stark_drafts == "engine"
    assert not hasattr(bench, "RETAINED_SELECTORS")
    for bad in (
        ["--stark-arm", "current=/x"],
        ["--stark-arm", "Bad Name=/x"],
        ["--stark-env", "nope=K=V"],
        ["--stark-env", "current=lower=1"],
    ):
        with pytest.raises(bench.BenchError):
            bench.build_config(parse(stark_args(layout, *bad)))


def test_stark_arm_model_replaces_the_shared_model_for_one_arm(layout):
    other = layout.root / "other-artifact"
    other.mkdir()
    args = parse(
        stark_args(
            layout,
            "--stark-arm",
            f"affine8={layout.binary}",
            "--stark-arm-model",
            f"affine8={other}",
        )
    )
    config = bench.build_config(args)
    assert [arm.model for arm in config.arms] == [layout.model, other]
    assert "model_label" not in config.arms[0].identity()
    assert config.arms[1].identity()["model_label"] == "other-artifact"
    for bad in (["--stark-arm-model", "nope=/x"], ["--stark-arm-model", "current="]):
        with pytest.raises(bench.BenchError):
            bench.build_config(parse(stark_args(layout, *bad)))


def test_both_command_alternates_engines(layout):
    args = parse(
        [
            "both",
            "--stark-binary",
            str(layout.binary),
            "--stark-model",
            str(layout.model),
            "--stark-arm",
            f"baseline={layout.baseline}",
            "--mlx-serve-binary",
            str(layout.mlx),
            "--mlx-original-model",
            str(layout.model),
            "--mlx-author-8bit-model",
            str(layout.model),
            "--mlx-community-4bit-model",
            str(layout.model),
        ]
    )
    names = [arm.name for arm in bench.build_config(args).arms]
    assert names == [
        "stark-current",
        "mlx-original",
        "stark-baseline",
        "mlx-author-8bit",
        "mlx-community-4bit",
    ]
    assert bench.interleave([1], [2, 3, 4]) == [1, 2, 3, 4]
    with pytest.raises(bench.BenchError, match="no_arms"):
        bench.build_config(
            parse(
                [
                    "both",
                    "--stark-binary",
                    str(layout.binary),
                    "--mlx-original-model",
                    "",
                ]
            )
        )


def test_build_config_rejects_bad_values(layout):
    with pytest.raises(bench.BenchError, match="invalid_repeat_count"):
        bench.build_config(parse(stark_args(layout, "--repeats", "0")))
    with pytest.raises(bench.BenchError, match="invalid_sessions"):
        bench.build_config(parse(stark_args(layout, "--sessions", "probe,other")))
    with pytest.raises(bench.BenchError, match="invalid_effort_alias"):
        bench.build_config(
            parse(stark_args(layout, "--stark-effort-alias", "Medium=high"))
        )


# --------------------------------------------------------------------------
# Dry run
# --------------------------------------------------------------------------


def test_dry_run_prints_matrix_on_battery_and_starts_nothing(layout, monkeypatch):
    monkeypatch.setattr(bench, "node_version", lambda node: (24, 22, 3))
    monkeypatch.setattr(
        bench, "template_helper", lambda: layout.root / "stark-template"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run started a process")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    out = io.StringIO()
    code = bench.main(
        stark_args(
            layout,
            "--allow-unpinned-llmprobe",
            "--stark-arm",
            f"baseline={layout.baseline}",
            "--dry-run",
        ),
        host=host_with(BATT_BATTERY),
        out=out,
    )
    text = out.getvalue()
    assert code == 0, text
    assert "Configuration valid." in text
    assert "Timing run would be refused: not on AC power" in text
    assert "Dry run: no server, worker or client was started." in text
    assert "Reasoning mode: llmprobe --reasoning off" in text
    assert (
        "Drafting: Stark engine default; MLX-Serve engine default; burst forced off on both"
        in text
    )
    assert (
        "stark-current" in text
        and "stark-baseline" in text
        and "selector" not in text.lower()
    )
    assert "sustained_decode       n=9" in text
    assert "context_16k            n=3" in text
    assert "prefill_8800_native    n=3" in text
    assert "burst_16x300           n=3" in text
    assert not (layout.root / "out").exists()


def test_dry_run_reports_configuration_problems(layout, monkeypatch):
    monkeypatch.setattr(bench, "node_version", lambda node: (20, 1, 0))
    monkeypatch.setattr(bench, "template_helper", lambda: None)
    out = io.StringIO()
    code = bench.main(
        stark_args(
            layout,
            "--stark-arm",
            f"same={layout.binary}",
            "--stark-arm",
            f"gone={layout.root / 'missing'}",
            "--llmprobe-reasoning",
            "client-default",
            "--dry-run",
        ),
        host=host_with(),
        out=out,
    )
    text = out.getvalue()
    assert code == 2
    assert "two Stark arms point at byte-identical binaries" in text
    assert "stark-gone: engine binary missing or not executable" in text
    assert "differs from the pinned build" in text
    assert "Node.js 22.19 or newer required" in text
    assert "native chat-template helper missing" in text
    assert "Reasoning mode: client default (reasoning_effort medium)" in text
    assert "Timing run would be allowed" in text


def test_dry_run_for_mlx_serve_arms(layout, monkeypatch):
    monkeypatch.setattr(bench, "node_version", lambda node: (24, 0, 0))
    out = io.StringIO()
    code = bench.main(
        [
            "mlx-serve",
            "--mlx-serve-binary",
            str(layout.mlx),
            "--mlx-original-model",
            str(layout.model),
            "--mlx-community-4bit-model",
            str(layout.model),
            "--llmprobe",
            str(layout.probe),
            "--allow-unpinned-llmprobe",
            "--repeats",
            "5",
            "--sessions",
            "probe,burst",
            "--output",
            str(layout.root / "out"),
            "--dry-run",
        ],
        host=host_with(),
        out=out,
    )
    text = out.getvalue()
    assert code == 0, text
    assert (
        "mlx-original" in text
        and "mlx-community-4bit" in text
        and "mlx-author-8bit" not in text
    )
    assert "decode                 n=15" in text and "prefill_8800" not in text


def test_timing_run_refuses_battery_before_creating_output(layout, monkeypatch, capsys):
    monkeypatch.setattr(bench, "node_version", lambda node: (24, 0, 0))
    monkeypatch.setattr(
        bench, "template_helper", lambda: layout.root / "stark-template"
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("started")),
    )
    code = bench.main(
        stark_args(
            layout,
            "--allow-unpinned-llmprobe",
            "--lock-file",
            str(layout.root / "lock"),
        ),
        host=host_with(BATT_BATTERY),
    )
    assert code == 1
    assert "not_on_ac_power" in capsys.readouterr().err
    assert not (layout.root / "out").exists() and not (layout.root / "lock").exists()


# --------------------------------------------------------------------------
# Drafting and context planning
# --------------------------------------------------------------------------


def test_choose_context_is_sizing_only():
    assert bench.choose_context(True, 10) == 2048
    assert bench.choose_context(False, 300) == 4352
    assert bench.choose_context(False, 4352) == 4352
    assert bench.choose_context(False, 8808) == 9216
    assert bench.choose_context(False, 40000) == 32768


def test_draft_fields_forward_the_client_and_force_burst_off():
    assert bench.draft_fields("engine", burst_mode=False, enable_mtp=None) == (
        {},
        "engine_default",
    )
    assert bench.draft_fields("engine", burst_mode=False, enable_mtp=True) == (
        {"mtp": True},
        "on",
    )
    assert bench.draft_fields("engine", burst_mode=False, enable_mtp=False) == (
        {"mtp_drafts": 0},
        0,
    )
    assert bench.draft_fields("engine", burst_mode=True, enable_mtp=True) == (
        {"mtp_drafts": 0},
        0,
    )
    assert bench.draft_fields("off", burst_mode=False, enable_mtp=None) == (
        {"mtp_drafts": 0},
        0,
    )
    with pytest.raises(bench.BenchError):
        bench.draft_fields("admission", burst_mode=False, enable_mtp=None)


def test_engine_mtp_report_keeps_counts_only():
    reply = {
        "mtp": {
            "enabled": True,
            "drafts": 3,
            "verification": "short-v1",
            "sampling": "x",
            "stats": {
                "proposed": 30,
                "accepted": 21,
                "speculative": {"rounds": [1, 2, 3, 4], "draft_seconds": 0.1},
            },
        }
    }
    assert bench.engine_mtp_report(reply) == {
        "enabled": True,
        "drafts": 3,
        "requested_drafts": None,
        "fallback_reason": None,
        "verification": "short-v1",
        "proposed": 30,
        "accepted": 21,
        "rounds": [1, 2, 3, 4],
    }
    assert bench.engine_mtp_report({}) is None
    assert bench.numeric_tree({"text": "free text " + SENTINEL}) == {"text": None}


# --------------------------------------------------------------------------
# Stark gateway with a fake worker
# --------------------------------------------------------------------------


class FakeWorker:
    prefix_cache = True
    images = True

    def __init__(self, default_drafts=3, error=None):
        self.default_drafts = default_drafts
        self.mtp = {"supported": True, "max_drafts": 3, "recommended_drafts": 1}
        self.seen = []
        self.error = error
        self.closed = False

    def read(self):
        return {
            "ready": True,
            "mtp": {
                "supported": True,
                "max_drafts": 3,
                "recommended_drafts": 1,
                "default_drafts": self.default_drafts,
                "verification": "short-v1",
            },
        }

    def close(self):
        self.closed = True

    def request(self, body, on_delta=None):
        if body["operation"] == "tokenize":
            return {"token_ids": [1] * (600 if "long" in body["text"] else 4)}
        self.seen.append(body)
        if self.error:
            return {"error": self.error}
        if on_delta is not None:
            on_delta(SENTINEL)
        cap = body["max_new_tokens"]
        drafts = (
            0
            if body.get("mtp_drafts") == 0
            else 1
            if body.get("mtp")
            else self.default_drafts
        )
        return {
            "generated_token_ids": [2] * cap,
            "text": SENTINEL,
            "input_tokens": len(body["input_token_ids"]),
            "output_tokens": cap,
            "sampled_tokens": cap,
            "finish_reason": "length",
            "prefill_seconds": 0.02,
            "decode_seconds": 0.03,
            "reused_input_tokens": 0,
            "mtp": {
                "enabled": drafts > 0,
                "drafts": drafts,
                "verification": "short-v1",
                "stats": {"proposed": 10 * drafts, "accepted": 7 * drafts},
            },
        }


class FakeStream:
    def __init__(self, content, reasoning):
        self.content = content

    def feed(self, text):
        self.content(text)

    def finish(self, message):
        return None


def fake_renderer(messages, *, thinking, effort):
    if effort not in (None, "low", "high", "max"):
        raise ValueError("unsupported effort")
    text = " ".join(message["content"] for message in messages)
    return (
        "long " if "long" in text else ""
    ) + f"rendered thinking={thinking} effort={effort}"


@contextmanager
def gateway(
    tmp_path, worker=None, *, burst_mode=False, cache=True, mode="engine", aliases=None
):
    worker = worker or FakeWorker()
    receipts = bench.Receipts(tmp_path / "receipts")
    server = bench.Gateway(
        worker,
        receipts=receipts,
        burst_mode=burst_mode,
        cache=cache,
        renderer=fake_renderer,
        stream_factory=FakeStream,
        parse_reply=lambda text: {"role": "assistant", "content": text},
        draft_mode=mode,
        effort_aliases=aliases,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, worker, receipts
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def post(port, route, body):
    try:
        with bench.http_request(port, route, body) as response:
            return response.status, response.read().decode()
    except bench.HTTPError as error:
        return error.code, error.read().decode()


def chat_body(text="Fixture user.", **extra):
    return {
        "model": bench.STARK_MODEL_ID,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "max_completion_tokens": 8,
        **extra,
    }


def test_gateway_leaves_drafting_to_the_worker(tmp_path):
    with gateway(tmp_path) as (port, worker, _receipts):
        assert bench.fetch(port, "/v1/models")["data"][0]["id"] == bench.STARK_MODEL_ID
        assert bench.fetch(port, "/tokenize", {"content": "fixture"}) == {
            "tokens": [1] * 4
        }
        # No client field: no worker field, for any prompt length or context.
        for ids in ([1] * 4, [1] * 512, [1] * 8800):
            result = bench.completion(port, bench.STARK_MODEL_ID, ids, 8, None, "salt")
            assert "mtp_drafts" not in worker.seen[-1] and "mtp" not in worker.seen[-1]
            assert result["drafts_requested"] == "engine_default"
            assert result["engine_mtp"]["requested"] == "engine_default"
            assert (
                result["engine_mtp"]["drafts"] == 3
                and result["engine_mtp"]["accepted"] == 21
            )
        assert worker.seen[-1]["context_tokens"] == 9216
        bench.completion(port, bench.STARK_MODEL_ID, [1] * 4, 8, True, "salt")
        assert worker.seen[-1]["mtp"] is True and "mtp_drafts" not in worker.seen[-1]
        result = bench.completion(port, bench.STARK_MODEL_ID, [1] * 4, 8, False, "salt")
        assert (
            worker.seen[-1]["mtp_drafts"] == 0 and result["engine_mtp"]["drafts"] == 0
        )
        assert bench.native_metrics(port) == {
            bench.PREFILL_METRIC: pytest.approx(0.1),
            bench.DECODE_METRIC: pytest.approx(0.15),
        }
    with gateway(tmp_path, mode="off") as (port, worker, _receipts):
        bench.completion(port, bench.STARK_MODEL_ID, [1] * 600, 8, None, "salt")
        assert worker.seen[-1]["mtp_drafts"] == 0


def test_gateway_chat_render_cache_scope_and_reasoning(tmp_path):
    with gateway(tmp_path) as (port, worker, _receipts):
        status, text = post(
            port, "/v1/chat/completions", chat_body(reasoning_effort="none")
        )
        reply = json.loads(text)
        assert status == 200 and reply["usage"]["completion_tokens"] == 8
        assert reply["usage"]["stark_mtp"]["requested"] == "engine_default"
        assert worker.seen[-1]["cache_scope"] == bench.STARK_CACHE_SCOPE
        assert "mtp_drafts" not in worker.seen[-1]
        # The Inkling template rejects "medium", so llmprobe would fall back to no effort.
        status, text = post(
            port, "/v1/chat/completions", chat_body(reasoning_effort="medium")
        )
        assert status == 400 and "invalid_benchmark_request" in text
        status, text = post(
            port, "/v1/chat/completions", chat_body(ignore_eos=True, min_tokens=192)
        )
        assert status == 400 and "unsupported_stop_override" in text
    with gateway(tmp_path, aliases={"medium": "high"}) as (port, _worker, _receipts):
        status, _ = post(
            port, "/v1/chat/completions", chat_body(reasoning_effort="medium")
        )
        assert status == 200


def test_gateway_burst_mode_forces_drafting_off(tmp_path):
    with gateway(tmp_path, burst_mode=True, cache=False) as (port, worker, _receipts):
        status, text = post(port, "/v1/chat/completions", chat_body(stream=True))
        assert status == 200 and text.rstrip().endswith("data: [DONE]")
        assert (
            worker.seen[-1]["context_tokens"] == 2048
            and worker.seen[-1]["mtp_drafts"] == 0
        )
        assert "cache_scope" not in worker.seen[-1]
        result = bench.completion(
            port, bench.STARK_MODEL_ID, "long raw prompt", 16, None, "salt"
        )
        assert result["generated_tokens"] == 16 and worker.seen[-1]["mtp_drafts"] == 0


def test_gateway_rejects_bad_requests_and_maps_worker_errors(tmp_path):
    with gateway(tmp_path) as (port, _worker, _receipts):
        assert (
            post(port, "/v1/completions", {"prompt": [1] * 32760, "max_tokens": 64})[0]
            == 400
        )
        assert (
            post(port, "/v1/completions", {"prompt": [-1], "max_tokens": 8})[0] == 400
        )
        assert post(port, "/v1/completions", {"prompt": [1], "max_tokens": 0})[0] == 400
        assert post(port, "/v1/unknown", {"prompt": [1]})[0] == 404
    with gateway(tmp_path, FakeWorker(error="mtp_not_supported")) as (
        port,
        _worker,
        _receipts,
    ):
        status, text = post(
            port, "/v1/completions", {"prompt": [1] * 600, "max_tokens": 8}
        )
        assert status == 400 and "mtp_not_supported" in text
    with gateway(tmp_path, FakeWorker(error="Some free text " + SENTINEL)) as (
        port,
        _worker,
        _receipts,
    ):
        status, text = post(
            port, "/v1/completions", {"prompt": [1] * 600, "max_tokens": 8}
        )
        assert status == 400 and SENTINEL not in text and "worker_error" in text


def test_gateway_receipts_hold_no_text(tmp_path):
    with gateway(tmp_path) as (port, _worker, _receipts):
        post(
            port,
            "/v1/chat/completions",
            chat_body(PROMPT_SENTINEL + " long", stream=True),
        )
        bench.completion(port, bench.STARK_MODEL_ID, [5] * 600, 8, True, "salt")
    native = bench.read_jsonl(tmp_path / "receipts/native.jsonl")
    assert [row["drafts_requested"] for row in native] == ["engine_default", "on"]
    assert [row["engine_mtp"]["drafts"] for row in native] == [3, 1]
    assert native[1]["input_ids_le_u32_sha256"] == bench.ids_digest([5] * 600)
    for path in (tmp_path / "receipts").iterdir():
        content = path.read_text()
        assert SENTINEL not in content and PROMPT_SENTINEL not in content


def session_config(**overrides):
    values = {
        "stark_worker_timeout": 1.0,
        "stark_image_check_dir": None,
        "stark_effort_aliases": {},
        "stark_min_context": 4352,
        "stark_drafts": "engine",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_stark_session_uses_the_arm_binary_and_scoped_environment(
    tmp_path, layout, monkeypatch
):
    monkeypatch.setenv("STARK_INKLING_EXPERIMENT_STALE", "1")
    monkeypatch.setenv("UZU_UNRELATED", "1")
    seen = {}
    workers = []

    def factory(binary, model, timeout, image_dir):
        seen["binary"] = binary
        seen["env"] = {
            k: v for k, v in os.environ.items() if k.startswith(("STARK_", "UZU_"))
        }
        worker = FakeWorker()
        workers.append(worker)
        return worker

    adapters = lambda model: (fake_renderer, FakeStream, lambda text: {"content": text})
    arm = bench.Arm(
        "stark-baseline",
        "stark",
        layout.baseline,
        layout.model,
        env={"STARK_DIAGNOSTIC": "1"},
    )
    receipts = bench.Receipts(tmp_path / "arm")
    with bench.stark_session(
        session_config(),
        arm,
        receipts,
        "probe",
        worker_factory=factory,
        adapters=adapters,
    ) as (port, model_id):
        assert model_id == bench.STARK_MODEL_ID
        assert bench.fetch(port, "/v1/models")["data"][0]["id"] == model_id
    assert workers[0].closed
    assert seen == {"binary": layout.baseline, "env": {"STARK_DIAGNOSTIC": "1"}}
    assert os.environ["STARK_INKLING_EXPERIMENT_STALE"] == "1"  # restored afterwards
    ready = bench.read_jsonl(tmp_path / "arm/native.jsonl")[0]
    assert (
        ready["event"] == "worker_ready"
        and ready["readiness_mtp"]["default_drafts"] == 3
    )
    assert ready["draft_mode"] == "engine" and ready["diagnostic_env"] == {
        "STARK_DIAGNOSTIC": "1"
    }
    assert ready["removed_environment_names"] == [
        "STARK_INKLING_EXPERIMENT_STALE",
        "UZU_UNRELATED",
    ]

    def no_images(*args, **kwargs):
        worker = FakeWorker()
        worker.images = False
        workers.append(worker)
        return worker

    with (
        pytest.raises(bench.BenchError, match="stark_images_not_loaded"),
        bench.stark_session(
            session_config(),
            arm,
            receipts,
            "probe",
            worker_factory=no_images,
            adapters=adapters,
        ),
    ):
        pass
    assert workers[-1].closed


# --------------------------------------------------------------------------
# Fake MLX-Serve server: prefill control and burst
# --------------------------------------------------------------------------


class FakeEngine(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *, cached=0, short_slot=None, tokens=9000):
        self.cached = cached
        self.short_slot = short_slot
        self.tokens = tokens
        self.lock = threading.Lock()
        self.prefill = 0.0
        self.decode = 0.0
        self.bodies = []
        super().__init__((bench.HOST, 0), FakeEngineHandler)


class FakeEngineHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        return

    def do_GET(self):
        if self.path == "/v1/models":
            data = json.dumps({"data": [{"id": "fake"}]}).encode()
        else:
            data = (
                f"{bench.PREFILL_METRIC} {self.server.prefill}\n{bench.DECODE_METRIC} {self.server.decode}\n"
                "other_metric 1\n"
            ).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            data = json.dumps(
                {"tokens": [i % 1000 for i in range(self.server.tokens)]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        prompt = body["prompt"]
        with self.server.lock:
            self.server.bodies.append(body)
            self.server.prefill += 0.5
            self.server.decode += 0.25
        count = body["max_tokens"]
        short = self.server.short_slot
        if (
            isinstance(prompt, str)
            and short is not None
            and f"request {short:02d}]" in prompt
        ):
            count -= 1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for _ in range(count):
            self.wfile.write(
                b'data: {"choices":[{"text":"x","finish_reason":null}]}\n\n'
            )
        finish = "length" if count == body["max_tokens"] else "stop"
        self.wfile.write(
            f'data: {{"choices":[{{"text":"","finish_reason":"{finish}"}}]}}\n\n'.encode()
        )
        prompt_tokens = len(prompt) if isinstance(prompt, list) else 40
        usage = {
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": count,
                "prompt_tokens_details": {"cached_tokens": self.server.cached},
            },
            "choices": [],
        }
        self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


@contextmanager
def fake_engine(**kwargs):
    server = FakeEngine(**kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_exact_prefill_records_hashes_validity_and_no_draft_field(tmp_path):
    seeds = []
    receipts = bench.Receipts(tmp_path)
    with fake_engine() as server:
        bench.exact_prefill(
            server.server_port,
            "fake",
            receipts,
            corpus=lambda seed: seeds.append(seed) or "words",
            warmups=1,
            repeats=2,
        )
        assert all("enable_mtp" not in body for body in server.bodies)
    rows = bench.read_jsonl(tmp_path / "prefill-8800.jsonl")
    assert [row["warmup"] for row in rows] == [True, False, False]
    assert seeds == [f"{bench.EXACT_PREFILL_SEED}-{i}" for i in range(3)]
    expected = bench.ids_digest([i % 1000 for i in range(8800)])
    assert all(
        row["valid"]
        and row["prompt_tokens"] == 8800
        and row["input_ids_le_u32_sha256"] == expected
        for row in rows
    )
    assert all(
        row["drafts_requested"] == "engine_default" and row["engine_mtp"] is None
        for row in rows
    )
    assert rows[1]["native_prefill_s"] == pytest.approx(0.5)
    assert rows[1]["native_prefill_tok_s"] == pytest.approx(17600.0)
    with fake_engine(cached=64) as server:
        bench.exact_prefill(
            server.server_port,
            "fake",
            bench.Receipts(tmp_path / "cached"),
            corpus=lambda seed: "words",
            warmups=0,
            repeats=1,
        )
    assert bench.read_jsonl(tmp_path / "cached/prefill-8800.jsonl")[0]["valid"] is False
    with (
        fake_engine(tokens=100) as server,
        pytest.raises(bench.BenchError, match="unseen_corpus_too_short"),
    ):
        bench.exact_prefill(
            server.server_port,
            "fake",
            receipts,
            corpus=lambda seed: "words",
            warmups=0,
            repeats=1,
        )


def test_burst_aggregates_forces_drafting_off_and_flags_short_streams(tmp_path):
    with fake_engine() as server:
        bench.burst(
            server.server_port,
            "fake",
            bench.Receipts(tmp_path),
            width=4,
            cap=10,
            warmups=1,
            repeats=1,
        )
        assert len(server.bodies) == 8 and all(
            body["enable_mtp"] is False for body in server.bodies
        )
    rows = bench.read_jsonl(tmp_path / "burst.jsonl")
    measured = rows[1]
    assert (
        measured["valid"]
        and measured["total_generated_tokens"] == 40
        and measured["length_finished"] == 4
    )
    assert measured["aggregate_tok_s"] == pytest.approx(40 / measured["wall_s"])
    assert measured["native_prefill_request_seconds_sum"] == pytest.approx(2.0)
    assert {req["input_utf8_sha256"] for req in measured["requests"]} == {
        bench.text_digest(prompt) for prompt in bench.burst_prompts(4)
    }
    assert {req["drafts_requested"] for req in measured["requests"]} == {0}
    with fake_engine(short_slot=2) as server:
        bench.burst(
            server.server_port,
            "fake",
            bench.Receipts(tmp_path / "short"),
            width=4,
            cap=10,
            warmups=0,
            repeats=1,
        )
    short = bench.read_jsonl(tmp_path / "short/burst.jsonl")[0]
    assert short["valid"] is False and short["total_generated_tokens"] == 39


def test_burst_deck_matches_the_recorded_comparison():
    prompts = bench.burst_prompts()
    assert len(prompts) == 16 and len(set(prompts)) == 16
    assert prompts[6].startswith("[load trial request 06] Write the complete list")
    assert prompts[7].startswith(
        "[load trial request 07] Begin now with the numbers. Count in increasing order from 3247."
    )


def test_mlx_command_leaves_drafting_to_the_engine_except_burst():
    probe = bench.mlx_command(Path("/bin/mlx-serve"), Path("/m"), 1234, "probe")
    assert (
        "--mtp" not in probe
        and "--no-mtp" not in probe
        and "--prefix-cache-entries" not in probe
    )
    assert "32768" in probe and "--no-pld" in probe
    forced = bench.mlx_command(
        Path("/bin/mlx-serve"), Path("/m"), 1234, "probe", force_mtp=True
    )
    assert "--mtp" in forced
    burst = bench.mlx_command(
        Path("/bin/mlx-serve"), Path("/m"), 1234, "burst", force_mtp=True
    )
    assert (
        burst[burst.index("--max-concurrent") + 1] == "16"
        and "--no-mtp" in burst
        and "--mtp" not in burst
    )
    assert burst[burst.index("--prefix-cache-entries") + 1] == "0"
    prefill = bench.mlx_command(Path("/bin/mlx-serve"), Path("/m"), 1234, "prefill")
    assert (
        prefill[prefill.index("--prefix-cache-entries") + 1] == "0"
        and "--mtp" not in prefill
    )


def test_mlx_server_records_keep_numbers_only(tmp_path):
    receipts = bench.Receipts(tmp_path)
    bench.mlx_server_records(
        [
            f"user said {PROMPT_SENTINEL}\n",
            "[spec-stats] mode=dspark attempts=10 accepts=7 per_draft_pct=70%\n",
            "req 1 <- 75+192 tokens streamed (2100ms) [prefill 0.2s, decode 91.0 t/s] [length]\n",
            "Inkling 2 draft head loaded\n",
        ],
        receipts,
    )
    rows = bench.read_jsonl(tmp_path / "native.jsonl")
    assert rows[0] == {
        "event": "spec",
        "mode": "dspark",
        "fields": {"attempts": 10.0, "accepts": 7.0, "per_draft_pct": 70.0},
    }
    assert (
        rows[1]["prompt"] == 75
        and rows[1]["generated"] == 192
        and rows[1]["wall_ms"] == 2100
    )
    assert rows[2] == {"event": "inkling_draft_head", "state": "loaded"}
    assert PROMPT_SENTINEL not in (tmp_path / "native.jsonl").read_text()


# --------------------------------------------------------------------------
# llmprobe runner with a fake Node
# --------------------------------------------------------------------------

FAKE_NODE = """#!{python}
import json, os, sys
args = sys.argv[1:]
if "--input-type=module" in args:
    sys.stdout.write("word " * 20000)
    raise SystemExit(0)
assert args[0].endswith("bin/dist/llmprobe.mjs") and "--bench-only" in args and "--no-save" in args
assert args[args.index("--reasoning") + 1] == "off"
assert os.environ["HOME"] == os.getcwd()
assert os.environ["LLMPROBE_INKLING_LOCAL_TELEMETRY"] == "1"
assert "LLMPROBE_UPLOAD_URL" not in os.environ and "OPENAI_API_KEY" not in os.environ
tag = os.environ["LLMPROBE_INKLING_LOCAL_CACHE_TAG"]
print("progress line with " + {prompt!r})
def receipt(label, **fields):
    base = dict(label=label, warmup=False, failed=False, inputTokens=74, outputTokens=192, cachedInputTokens=0,
                ttftMs=300, wallMs=2000, decodeTokPerSec=100.0, prefillTokPerSec=None, tokensPerStep=3.1,
                streamCoalesced=False, finishReason="length")
    base.update(fields)
    sys.stderr.write("[inkling-bench-sample] " + json.dumps(base) + "\\n")
receipt("decode warmup", warmup=True)
receipt("decode 1/3", text={prompt!r})
receipt("context ~4.1k", inputTokens=4020, decodeTokPerSec=90.0)
receipt("context ~4.1k ceiling", inputTokens=3970, cachedInputTokens=2048)
receipt("sustained decode 1/3", decodeTokPerSec=110.0, inputUtf8Sha256="zz")
receipt("unexpected label")
sys.stderr.write("[inkling-bench-sample] not json\\n")
save = args[args.index("--save") + 1]
report = {{"bench": {{"decodeLengthNote": "forced", "reasoningNote": None, "loadDrift": {{"driftPct": 1}}}},
          "target": {{"engine": "fake", "model": "m", "engineSettings": {{"mtp": {{"default_on": True}}}}}},
          "conformance": {{"results": [{prompt!r}]}},
          "run": {{"depth": "default", "mode": "bench-only", "startedAt": "x"}}, "machine": {{"cpu": "fake"}}}}
open(save, "w").write(json.dumps(report))
print("cache tag " + tag)
"""


@pytest.fixture
def fake_node(tmp_path):
    return make_executable(
        tmp_path / "fake-node",
        FAKE_NODE.format(python=sys.executable, prompt=PROMPT_SENTINEL).encode(),
    )


def probe_config(layout, node, **overrides):
    values = {
        "engine": "mlx-serve",
        "arms": [],
        "llmprobe": layout.probe,
        "node": str(node),
        "output": layout.root / "out",
        "lock_file": layout.root / "lock",
        "repeats": 1,
        "warmups": 0,
        "sessions": ("probe",),
        "probe_runs": None,
        "reasoning": "off",
        "cache_tag_base": "inkbench",
        "allow_busy_host": False,
        "allow_unpinned_llmprobe": True,
    }
    values.update(overrides)
    return bench.Config(**values)


def test_run_llmprobe_keeps_sanitized_receipts_only(layout, fake_node, monkeypatch):
    monkeypatch.setenv("LLMPROBE_UPLOAD_URL", "http://example.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    directory = layout.root / "probe-00"
    result = bench.run_llmprobe(
        probe_config(layout, fake_node), 9, "model", directory, "inkbench00"
    )
    assert result == {"samples": 6, "cache_tag": "inkbench00"}
    samples = bench.read_jsonl(directory / "samples.jsonl")
    assert [s["scenario"] for s in samples] == [
        "decode",
        "decode",
        "context_4k",
        "context_4k_ceiling",
        "sustained_decode",
        None,
    ]
    assert samples[-1]["label"] == "unrecognized"
    assert samples[4]["inputUtf8Sha256"] is None
    assert all("text" not in s for s in samples)
    saved = json.loads((directory / "bench.json").read_text())
    assert saved["bench"]["decodeLengthNote"] == "forced" and "conformance" not in saved
    assert sorted(p.name for p in directory.iterdir()) == [
        "bench.json",
        "samples.jsonl",
    ]
    for path in directory.iterdir():
        assert PROMPT_SENTINEL not in path.read_text()


def test_llmprobe_command_reasoning_modes(layout):
    command = bench.llmprobe_command(
        probe_config(layout, "node", probe_runs=3), 4321, "m", Path("/tmp/r.json")
    )
    assert (
        command[2] == "http://127.0.0.1:4321"
        and command[command.index("--runs") + 1] == "3"
    )
    assert command[command.index("--reasoning") + 1] == "off" and "--no-save" in command
    client = bench.llmprobe_command(
        probe_config(layout, "node", reasoning="client-default"), 1, "m", Path("/r")
    )
    assert "--reasoning" not in client


def test_label_mapping_covers_every_receipt_label():
    cases = {
        "decode length probe": "decode_length_probe",
        "decode warmup": "decode",
        "decode 3/3": "decode",
        "prefill 1/3": "prefill_repeated",
        "unseen prefill 2/3": "unseen_prefill",
        "spec:predictable included 2": "spec_predictable",
        "spec:predictable 1/3": "spec_predictable_all",
        "spec:novel warmup": "spec_novel",
        "context ~512": "context_512",
        "context ~512 ceiling 2/3": "context_512_ceiling",
        "context ~8.2k 1/3": "context_8k",
        "context ~16.4k ceiling": "context_16k_ceiling",
        "sustained decode 3/3": "sustained_decode",
        "context ~128k": "context_128k",
    }
    for label, key in cases.items():
        assert bench.scenario_for_label(label) == key, label
    assert bench.scenario_for_label("batching x4") is None


# --------------------------------------------------------------------------
# Orchestration with fake sessions
# --------------------------------------------------------------------------


def test_run_matrix_alternates_engines_and_writes_manifest(layout, fake_node):
    order = []

    @contextmanager
    def session_factory(config, arm, receipts, kind):
        order.append((arm.name, kind))
        with fake_engine() as server:
            yield server.server_port, "fake"

    arms = [
        bench.Arm("stark-current", "stark", layout.binary, layout.model),
        bench.Arm("mlx-original", "mlx-serve", layout.mlx, layout.model),
    ]
    config = probe_config(
        layout,
        fake_node,
        engine="both",
        arms=arms,
        repeats=2,
        warmups=1,
        sessions=("probe", "prefill", "burst"),
    )
    run_dir = bench.run_matrix(
        config,
        {"llmprobe": {"version": "0.6.18"}},
        session_factory=session_factory,
        host=host_with(),
        sleep_check=lambda start, end: {"checked": True, "sleep_events": 0},
    )
    assert order == [
        ("stark-current", "probe"),
        ("mlx-original", "probe"),
        ("stark-current", "probe"),
        ("mlx-original", "probe"),
        ("stark-current", "prefill"),
        ("mlx-original", "prefill"),
        ("stark-current", "burst"),
        ("mlx-original", "burst"),
    ]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete" and len(manifest["power_checks"]) == 8
    assert manifest["config"]["reasoning"] == "off"
    assert manifest["arms"][0]["binary_sha256"] == bench.sha256_file(layout.binary)
    assert manifest["sleep"] == {"checked": True, "sleep_events": 0}
    assert (run_dir / "mlx-original/probe-01/samples.jsonl").is_file()
    assert len(bench.read_jsonl(run_dir / "stark-current/burst.jsonl")) == 3
    summary = bench.summarize([run_dir])
    assert summary["cells"]["decode"]["mlx-original"]["stats"]["n"] == 2
    assert summary["cells"]["burst_16x300"]["stark-current"]["stats"]["n"] == 2
    assert summary["prefill_input_ids_match_across_arms"] is True
    assert (
        summary["reasoning_modes"] == ["off"]
        and summary["mixed_reasoning_modes"] is False
    )


def test_run_matrix_aborts_when_power_source_changes(layout, fake_node):
    snapshots = iter(
        [
            host_with()(),
            host_with()(),
            host_with(BATT_BATTERY)(),
            host_with(BATT_BATTERY)(),
        ]
    )

    @contextmanager
    def session_factory(config, arm, receipts, kind):
        with fake_engine() as server:
            yield server.server_port, "fake"

    config = probe_config(
        layout,
        fake_node,
        arms=[bench.Arm("mlx-original", "mlx-serve", layout.mlx, layout.model)],
        repeats=2,
        sessions=("probe",),
    )
    with pytest.raises(bench.BenchError, match="power_source_changed"):
        bench.run_matrix(
            config,
            {},
            session_factory=session_factory,
            host=lambda: next(snapshots),
            sleep_check=lambda start, end: {"checked": False, "sleep_events": None},
        )
    run_dir = next((layout.root / "out").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert (
        manifest["status"] == "failed" and manifest["failure"] == "power_source_changed"
    )


# --------------------------------------------------------------------------
# Summarizer arithmetic
# --------------------------------------------------------------------------


def sample(
    label,
    value,
    *,
    wall=2000.0,
    output=192,
    cached=0,
    inputs=74,
    warmup=False,
    failed=False,
    steps=3.0,
):
    return {
        "label": label,
        "scenario": bench.scenario_for_label(label),
        "warmup": warmup,
        "failed": failed,
        "inputTokens": inputs,
        "outputTokens": output,
        "cachedInputTokens": cached,
        "ttftMs": 300.0,
        "wallMs": wall,
        "decodeTokPerSec": value,
        "prefillTokPerSec": None,
        "tokensPerStep": steps,
        "streamCoalesced": False,
        "finishReason": "length",
        "inputUtf8Sha256": None,
    }


def write_run(
    root: Path,
    name: str,
    engine: str,
    arm: str,
    samples,
    *,
    prefill=(),
    bursts=(),
    sha="a" * 64,
    reasoning_note=None,
    config=None,
    natives=(),
):
    run_dir = root / name
    (run_dir / arm / "probe-00").mkdir(parents=True)
    manifest = {
        "schema": bench.SCHEMA,
        "engine": engine,
        "status": "complete",
        "config": {"reasoning": "off"} if config is None else config,
        "host_start": host_with()(),
        "sleep": {"sleep_events": 0},
        "arms": [
            {
                "name": arm,
                "engine": engine,
                "binary_sha256": sha,
                "description": arm,
                "diagnostic_env": {},
                "model": {"files": {}},
            }
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    receipts = bench.Receipts(run_dir / arm / "probe-00")
    for row in samples:
        receipts.write("samples", row)
    for row in natives:
        receipts.write("native", row)
    (run_dir / arm / "probe-00/bench.json").write_text(
        json.dumps(
            {
                "bench": {
                    "decodeLengthNote": "note-" + engine,
                    "reasoningNote": reasoning_note,
                },
                "target": {
                    "engineSettings": {"mtp": {"default_on": True}}
                    if engine == "mlx-serve"
                    else None
                },
            }
        )
    )
    arm_receipts = bench.Receipts(run_dir / arm)
    for row in prefill:
        arm_receipts.write("prefill-8800", row)
    for row in bursts:
        arm_receipts.write("burst", row)
    return run_dir


def prefill_row(rate, seed=1, *, warmup=False, valid=True, digest=None):
    digest = digest or bench.text_digest(f"seed-{seed}")
    return {
        "warmup": warmup,
        "valid": valid,
        "seed": f"seed-{seed}",
        "prompt_tokens": 8800,
        "cached_tokens": 0,
        "native_prefill_tok_s": rate,
        "client_prompt_tok_s": rate * 0.99,
        "client_ttft_s": 8800 / rate,
        "input_ids_le_u32_sha256": digest,
        "drafts_requested": "engine_default",
        "engine_mtp": None,
    }


def burst_row(rate, wall, *, warmup=False, valid=True):
    return {
        "warmup": warmup,
        "valid": valid,
        "aggregate_tok_s": rate,
        "wall_s": wall,
        "total_generated_tokens": 4800,
        "requests": [
            {
                "prompt_tokens": 40,
                "cached_tokens": 0,
                "drafts_requested": 0,
                "engine_mtp": {
                    "requested": 0,
                    "drafts": 0,
                    "proposed": 0,
                    "accepted": 0,
                },
            }
        ],
    }


def native(
    inputs, output, drafts, *, proposed=0, accepted=0, requested="engine_default"
):
    return {
        "event": "native_generation",
        "input_tokens": inputs,
        "output_tokens": output,
        "drafts_requested": requested,
        "engine_mtp": {"drafts": drafts, "proposed": proposed, "accepted": accepted},
    }


def test_join_native_matches_samples_in_order():
    samples = [
        sample("decode 1/3", 50.0, inputs=73),
        sample("decode 2/3", None, failed=True),
        sample("context ~4.1k", 96.0, inputs=4024),
    ]
    natives = [
        {"event": "worker_ready"},
        native(30, 1, 0),
        native(73, 192, 0),
        native(500, 8, 3),
        native(4024, 192, 3, proposed=100, accepted=60),
    ]
    bench.join_native(samples, natives)
    assert samples[0]["engine_mtp"]["drafts"] == 0 and "engine_mtp" not in samples[1]
    assert samples[2]["engine_mtp"] == {"drafts": 3, "proposed": 100, "accepted": 60}
    assert samples[2]["drafts_requested"] == "engine_default"


def test_summary_arithmetic_exclusions_outliers_ratios_and_drafting(tmp_path):
    mlx = write_run(
        tmp_path,
        "r1-mlx",
        "mlx-serve",
        "mlx-original",
        [
            sample("decode warmup", 50.0, warmup=True),
            sample("decode 1/3", 100.0),
            sample("decode 2/3", 90.0),
            sample("decode 3/3", 110.0),
            sample("sustained decode 1/3", 120.0),
            sample("sustained decode 2/3", 104.0),
            sample("context ~4.1k", 80.0, inputs=4018),
            sample("context ~4.1k ceiling", 70.0, inputs=3971, cached=0),
        ],
        prefill=[
            prefill_row(1000.0, 0, warmup=True),
            prefill_row(1900.0, 1),
            prefill_row(1880.0, 2),
            prefill_row(1910.0, 3),
        ],
        bursts=[
            burst_row(170.0, 28.0, warmup=True),
            burst_row(172.0, 27.9),
            burst_row(176.0, 27.2),
            burst_row(168.0, 28.5),
        ],
    )
    stark_samples = [
        sample("decode 1/3", 46.6, wall=4421.0),
        sample("decode 2/3", 0.4, wall=468330.0),
        sample("decode 3/3", 36.1, wall=5633.0),
        sample("decode 4/3", 30.0, output=150),
        sample("decode 5/3", None, failed=True),
        sample("sustained decode 1/3", 37.0),
        sample("sustained decode 2/3", 37.0),
        sample("context ~4.1k", 96.0, inputs=4024, steps=3.15),
        sample("context ~4.1k ceiling", 78.6, inputs=3977, cached=2048),
    ]
    stark_natives = [
        {
            "event": "worker_ready",
            "session": "probe",
            "draft_mode": "engine",
            "readiness_mtp": {"default_drafts": 3},
        }
    ]
    stark_natives += [
        native(s["inputTokens"], s["outputTokens"], 3, proposed=30, accepted=15)
        for s in stark_samples
        if not s["failed"]
    ]
    stark = write_run(
        tmp_path,
        "r2-stark",
        "stark",
        "stark-current",
        stark_samples,
        prefill=[
            prefill_row(389.0, 1),
            prefill_row(341.6, 2),
            prefill_row(389.0, 3, valid=False),
        ],
        bursts=[burst_row(34.0, 141.0), burst_row(33.8, 141.9)],
        sha="b" * 64,
        reasoning_note="engine rejected the reasoning effort param",
        natives=stark_natives,
    )
    summary = bench.summarize([stark, mlx])
    decode = summary["cells"]["decode"]
    assert decode["mlx-original"]["stats"] == {
        "median": 100.0,
        "min": 90.0,
        "max": 110.0,
        "n": 3,
    }
    current = decode["stark-current"]
    assert current["stats"] == {"median": 36.1, "min": 0.4, "max": 46.6, "n": 3}
    assert current["excluded"] == {"failed": 1, "short_output": 1, "no_rate": 0}
    assert current["wall_outliers"] == 1
    assert current["median_without_outliers"] == pytest.approx((46.6 + 36.1) / 2)
    assert current["speed_vs_reference"] == pytest.approx(0.361)
    assert current["drafting"] == {
        "requested": ["engine_default"],
        "engine_drafts": ["3"],
        "engine_reports": 3,
        "proposed": 90,
        "accepted": 45,
        "acceptance": 0.5,
    }
    assert decode["mlx-original"]["drafting"]["engine_reports"] == 0
    assert summary["cells"]["context_4k"]["stark-current"][
        "speed_vs_reference"
    ] == pytest.approx(1.2)
    assert summary["cells"]["context_4k_ceiling"]["stark-current"]["cached_tokens"] == [
        2048
    ]
    assert (
        summary["cells"]["context_4k"]["stark-current"]["tokens_per_step_median"]
        == 3.15
    )
    assert summary["cells"]["sustained_decode"]["mlx-original"]["stats"][
        "median"
    ] == pytest.approx(112.0)
    prefill = summary["cells"]["prefill_8800_native"]
    assert (
        prefill["mlx-original"]["stats"]["n"] == 3
        and prefill["mlx-original"]["stats"]["median"] == 1900.0
    )
    assert prefill["stark-current"]["excluded"] == {"invalid": 1}
    assert prefill["stark-current"]["stats"]["median"] == pytest.approx(
        (389.0 + 341.6) / 2
    )
    assert summary["cells"]["burst_16x300"]["stark-current"]["drafting"][
        "requested"
    ] == ["0"]
    makespan = summary["cells"]["burst_16x300_makespan"]
    assert makespan["mlx-original"]["stats"]["median"] == pytest.approx(27.9)
    # Lower-is-better rows invert so that above 1.00 still means faster.
    assert makespan["stark-current"]["speed_vs_reference"] == pytest.approx(
        27.9 / ((141.0 + 141.9) / 2)
    )
    assert summary["best_single_user_decode"]["mlx-original"]["any"] == (
        112.0,
        "sustained_decode",
    )
    assert summary["best_single_user_decode"]["stark-current"]["cold_prefix"] == (
        96.0,
        "context_4k",
    )
    assert summary["prefill_input_ids_match_across_arms"] is True
    assert (
        summary["arm_notes"]["stark-current"]["drafting"]
        == "engine; worker default_drafts 3"
    )
    assert summary["arm_notes"]["mlx-original"]["drafting"].startswith(
        "server mtp default_on True"
    )
    markdown = bench.render_markdown(summary)
    assert (
        "| Initial coding (llmprobe decode) | stark-current | 36.1 tok/s | 0.4-46.6 | 3 |"
        in markdown
    )
    assert "failed 1, short_output 1, wall outliers 1 (kept)" in markdown
    assert "req engine_default; engine 3, accept 0.50" in markdown
    assert "| 0.36 |" in markdown
    assert "| 4018 |" in markdown
    assert "Reasoning mode: off." in markdown and "llmprobe --reasoning off" in markdown
    assert "Warning" not in markdown
    assert "Exact 8,800-ID prefill inputs identical across arms: True." in markdown
    json.dumps(summary)


def test_summary_warns_on_mixed_reasoning_modes_and_splits_arms(tmp_path):
    off = write_run(
        tmp_path, "a", "mlx-serve", "mlx-original", [sample("decode 1/3", 90.0)]
    )
    client = write_run(
        tmp_path,
        "b",
        "mlx-serve",
        "mlx-original",
        [sample("decode 1/3", 110.0)],
        config={"reasoning": "client-default"},
    )
    summary = bench.summarize([off, client])
    assert set(summary["cells"]["decode"]) == {"mlx-original", "mlx-original@b"}
    assert summary["mixed_reasoning_modes"] is True
    markdown = bench.render_markdown(summary)
    assert "Warning: this table mixes reasoning modes (client-default, off)" in markdown
    assert "client default (reasoning_effort medium)" in markdown


def test_summary_detects_prefill_input_mismatch_and_merges_identical_arms(tmp_path):
    first = write_run(
        tmp_path,
        "a",
        "mlx-serve",
        "mlx-original",
        [sample("decode 1/3", 90.0)],
        prefill=[prefill_row(1900.0)],
    )
    second = write_run(
        tmp_path,
        "b",
        "mlx-serve",
        "mlx-original",
        [sample("decode 1/3", 110.0)],
        prefill=[prefill_row(1880.0)],
        config={"reasoning": "off", "repeats": 7},
    )
    other = write_run(
        tmp_path,
        "c",
        "stark",
        "stark-current",
        [sample("decode 1/3", 35.0)],
        prefill=[prefill_row(380.0, 1, digest="e" * 64)],
        sha="c" * 64,
    )
    summary = bench.summarize([first, second, other])
    assert summary["cells"]["decode"]["mlx-original"]["stats"]["n"] == 2
    assert summary["cells"]["decode"]["mlx-original"]["stats"][
        "median"
    ] == pytest.approx(100.0)
    assert summary["prefill_input_ids_match_across_arms"] is False
    rebuilt = write_run(
        tmp_path,
        "d",
        "stark",
        "stark-current",
        [sample("decode 1/3", 50.0)],
        sha="f" * 64,
    )
    split = bench.summarize([other, rebuilt])
    assert set(split["cells"]["decode"]) == {"stark-current", "stark-current@d"}


def test_summarize_command_writes_markdown_and_json(tmp_path):
    run = write_run(
        tmp_path, "run", "mlx-serve", "mlx-original", [sample("decode 1/3", 90.0)]
    )
    out = io.StringIO()
    assert bench.main(["summarize", str(run)], out=out) == 0
    assert (
        (tmp_path / "summary.md")
        .read_text()
        .startswith("# Inkling 2 Mini engine matrix")
    )
    assert (
        json.loads((tmp_path / "summary.json").read_text())["schema"]
        == bench.SUMMARY_SCHEMA
    )


def test_scenario_catalog_and_docstring():
    keys = [scenario.key for scenario in bench.SCENARIOS]
    assert len(keys) == len(set(keys))
    for rung in ("512", "4k", "8k", "16k"):
        assert f"context_{rung}" in keys and f"context_{rung}_ceiling" in keys
    sustained = bench.SCENARIO_BY_KEY["sustained_decode"]
    assert (
        "not multi-turn" in sustained.description
        and "not a cold single-turn number" in sustained.comparability
    )
    assert all(s.cap == 192 for s in bench.SCENARIOS if s.metric == "decodeTokPerSec")
    for phrase in (
        "--dry-run",
        "summarize",
        "--stark-arm",
        " both ",
        "--reasoning off",
        "client-default",
    ):
        assert phrase in bench.__doc__
    assert (
        "admission" not in bench.__doc__.lower()
        and "selector" not in bench.__doc__.lower()
    )
    assert "--arms" in bench.__doc__ and "arm_failures" in bench.__doc__


# --------------------------------------------------------------------------
# Failed arms: preflight, startup diagnostics, isolation and merging
# --------------------------------------------------------------------------


def test_weight_problem_detects_removed_shards(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "mtp.safetensors").write_bytes(b"\x00" * 8)
    assert "no weight shards" in bench.weight_problem(pack)
    names = [f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)]
    (pack / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"w{i}": name for i, name in enumerate(names)}})
    )
    (pack / names[1]).write_bytes(b"\x00" * 8)
    assert bench.weight_problem(pack) == (
        "3 of 4 weight shards named in model.safetensors.index.json are missing "
        "(first: model-00001-of-00004.safetensors)"
    )
    for name in names:
        (pack / name).write_bytes(b"\x00" * 8)
    assert bench.weight_problem(pack) is None
    (pack / "model.safetensors.index.json").write_text("{")
    assert "unreadable" in bench.weight_problem(pack)


def test_dry_run_reports_a_pack_with_missing_shards(layout, monkeypatch):
    monkeypatch.setattr(bench, "node_version", lambda node: (24, 0, 0))
    pack = layout.root / "author-8bit"
    pack.mkdir()
    (pack / "config.json").write_text("{}")
    (pack / "mtp.safetensors").write_bytes(b"\x00")
    (pack / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00004.safetensors"}})
    )
    out = io.StringIO()
    code = bench.main(
        [
            "mlx-serve",
            "--mlx-serve-binary",
            str(layout.mlx),
            "--mlx-original-model",
            str(layout.model),
            "--mlx-author-8bit-model",
            str(pack),
            "--llmprobe",
            str(layout.probe),
            "--allow-unpinned-llmprobe",
            "--output",
            str(layout.root / "out"),
            "--dry-run",
        ],
        host=host_with(),
        out=out,
    )
    assert code == 2
    assert (
        "mlx-author-8bit: 1 of 1 weight shards named in model.safetensors.index.json are missing"
        in out.getvalue()
    )


def test_arms_filter_selects_one_arm(layout):
    args = parse(
        [
            "mlx-serve",
            "--mlx-serve-binary",
            str(layout.mlx),
            "--mlx-author-8bit-model",
            str(layout.model),
            "--arms",
            "mlx-author-8bit",
        ]
    )
    assert [arm.name for arm in bench.build_config(args).arms] == ["mlx-author-8bit"]
    with pytest.raises(bench.BenchError, match="unknown_arms:mlx-nope"):
        bench.build_config(parse(["mlx-serve", "--arms", "mlx-nope"]))
    both = parse(
        [
            "both",
            "--stark-binary",
            str(layout.binary),
            "--mlx-original-model",
            "",
            "--arms",
            "stark-current",
        ]
    )
    assert [arm.name for arm in bench.build_config(both).arms] == ["stark-current"]


FAKE_MLX = """#!{python}
import sys
print("mlx-serve 26.11.4-pre-release.1 (MLX 0.32.5)")
print("model.safetensors.index.json names no shard in this directory; loading every *.safetensors instead")
print("[plugin] inkling2: load failed: Inkling2PackLayoutUnknown")
print("error: LoadFailed" + "x" * 1000)
sys.exit(1)
"""


def test_mlx_start_failure_keeps_a_startup_tail(layout, tmp_path):
    binary = make_executable(
        tmp_path / "fake-mlx", FAKE_MLX.format(python=sys.executable).encode()
    )
    arm = bench.Arm("mlx-author-8bit", "mlx-serve", binary, layout.model)
    config = SimpleNamespace(mlx_force_mtp=False)
    receipts = bench.Receipts(tmp_path / "arm" / "probe-00")
    with (
        pytest.raises(bench.BenchError, match="mlx_serve_exited:1"),
        bench.mlx_session(config, arm, receipts, "probe"),
    ):
        pass
    tail = (
        (tmp_path / "arm/probe-00/probe-server-startup-tail.txt")
        .read_text()
        .splitlines()
    )
    assert (
        tail[0].startswith("mlx-serve 26.11.4")
        and "Inkling2PackLayoutUnknown" in tail[2]
    )
    assert len(tail[3]) == bench.StartupTail.WIDTH
    events = [
        row["event"] for row in bench.read_jsonl(tmp_path / "arm/probe-00/native.jsonl")
    ]
    assert events == ["server_start", "server_start_failed", "server_exit"]
    failed = bench.read_jsonl(tmp_path / "arm/probe-00/native.jsonl")[1]
    assert (
        failed["tail_file"] == "probe-server-startup-tail.txt"
        and failed["failure"] == "mlx_serve_exited:1"
    )
    assert any("LoadFailed" in line for line in failed["error_lines"])


def test_startup_tail_stops_collecting_once_ready():
    tail = bench.StartupTail()
    for index in range(100):
        tail.add(f"line {index}\n")
    assert (
        len(tail.snapshot()) == bench.StartupTail.LIMIT
        and tail.snapshot()[-1] == "line 99"
    )
    tail.stop()
    tail.add(PROMPT_SENTINEL)
    assert tail.snapshot() == []


def test_stark_start_failure_is_recorded(tmp_path, layout):
    def broken(*args, **kwargs):
        raise bench.BenchError("stark_worker_not_ready")

    receipts = bench.Receipts(tmp_path / "arm")
    arm = bench.Arm("stark-current", "stark", layout.binary, layout.model)
    with (
        pytest.raises(bench.BenchError),
        bench.stark_session(
            session_config(), arm, receipts, "prefill", worker_factory=broken
        ),
    ):
        pass
    row = bench.read_jsonl(tmp_path / "arm/native.jsonl")[0]
    assert (
        row["event"] == "server_start_failed"
        and row["failure"] == "stark_worker_not_ready"
    )


def test_one_failed_arm_does_not_stop_the_others(layout, fake_node):
    order = []

    @contextmanager
    def session_factory(config, arm, receipts, kind):
        order.append((arm.name, kind))
        if arm.name == "mlx-author-8bit":
            raise bench.BenchError("mlx_serve_exited:1")
        with fake_engine() as server:
            yield server.server_port, "fake"

    arms = [
        bench.Arm("stark-current", "stark", layout.binary, layout.model),
        bench.Arm("mlx-author-8bit", "mlx-serve", layout.mlx, layout.model),
        bench.Arm("mlx-original", "mlx-serve", layout.mlx, layout.model),
    ]
    config = probe_config(
        layout,
        fake_node,
        engine="both",
        arms=arms,
        repeats=2,
        sessions=("probe", "burst"),
    )
    run_dir = bench.run_matrix(
        config,
        {},
        session_factory=session_factory,
        host=host_with(),
        sleep_check=lambda start, end: {"checked": True, "sleep_events": 0},
    )
    assert (
        order.count(("mlx-author-8bit", "probe")) == 1
        and ("mlx-author-8bit", "burst") not in order
    )
    assert (
        order.count(("mlx-original", "probe")) == 2
        and ("mlx-original", "burst") in order
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "partial"
    assert manifest["arm_failures"] == [
        {
            "arm": "mlx-author-8bit",
            "kind": "probe",
            "directory": "mlx-author-8bit/probe-00",
            "failure": "mlx_serve_exited:1",
            "seconds": manifest["arm_failures"][0]["seconds"],
        }
    ]
    summary = bench.summarize([run_dir])
    assert "mlx-author-8bit" not in summary["arms"]
    assert summary["failed_arms"] == [
        {
            "run": run_dir.name,
            "arm": "mlx-author-8bit",
            "failure": "mlx_serve_exited:1",
            "measured_in_another_run": False,
        }
    ]
    assert "Failed arm: `mlx-author-8bit`" in bench.render_markdown(summary)


def test_a_later_run_fills_in_a_failed_arm(tmp_path):
    old = write_run(
        tmp_path, "old", "mlx-serve", "mlx-original", [sample("decode 1/3", 90.0)]
    )
    manifest = json.loads((old / "manifest.json").read_text())
    manifest["arms"].append(
        {
            "name": "mlx-author-8bit",
            "engine": "mlx-serve",
            "binary_sha256": "a" * 64,
            "description": "x",
            "diagnostic_env": {},
            "model": {"files": {}},
        }
    )
    manifest["status"], manifest["failure"] = "failed", "mlx_serve_exited:1"
    (old / "manifest.json").write_text(json.dumps(manifest))
    (old / "mlx-author-8bit/probe-00").mkdir(parents=True)
    new = write_run(
        tmp_path, "new", "mlx-serve", "mlx-author-8bit", [sample("decode 1/3", 95.0)]
    )
    summary = bench.summarize([old, new])
    assert set(summary["cells"]["decode"]) == {"mlx-original", "mlx-author-8bit"}
    assert summary["cells"]["decode"]["mlx-author-8bit"][
        "speed_vs_reference"
    ] == pytest.approx(95 / 90)
    assert summary["failed_arms"][0]["measured_in_another_run"] is True
