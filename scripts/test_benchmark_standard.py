"""Model-free tests for scripts/benchmark_standard.py.

A synthetic profile, a fake worker, fake HTTP servers and a fake Node script
stand in for a model, stark-generate, a reference engine and the client.
Nothing here loads a model, starts an engine or touches the GPU. The original
single-model command line is covered by test_benchmark_inkling_engine_matrix.py,
which runs through the same code.
"""

from __future__ import annotations

import io
import json
import stat
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import benchmark_standard as bench
import pytest

TEMPLATE = "{{ fixture chat template }}"
PROMPT_SENTINEL = "PRIVATE-PROMPT-SENTINEL"
BATT_AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged; 0:00 remaining present: true\n"
BATT_CHARGING = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t84%; charging; (no estimate) present: true\n"
BATT_BATTERY = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t77%; discharging; 1:35 remaining present: true\n"
ADAPTER_LIMITED = (
    " Wattage = 94W\n Current = 4690mA\n Adapter Name = 140W USB-C Power Adapter\n"
)

PROFILE = """
schema = "stark-benchmark-profile-v1"
key = "{key}"
title = "Test Model"
status = "{status}"
reference_arm = "ref-base"

[source]
repo = "example/test-model"
revision = "0123456789abcdef0123456789abcdef01234567"

[client]
reasoning = "off"
cache_tag_base = "tm"

[stark]
served_model_id = "test-model-stark"
model_type = "test_family"
vocab_size = 1000
chat_adapter = "{adapter}"
template_sha256 = "{template}"
image_capable = {images}
drafting = "engine-default"

[stark.context]
minimum = 1024
maximum = {maximum}
burst = 512

[stark.artifacts.base]
description = "Base artifact"
quantization = "bf16"
match = "base"
path = "{base}"

[stark.artifacts.q8]
description = "8-bit artifact"
quantization = "8-bit"
match = "q8"

[[reference]]
engine = "{engine}"
label = "Reference"
arm_prefix = "ref"
binary = "{binary}"
flags = ["--flag"]
draft_off_flags = ["--no-draft"]
draft_force_flags = ["--draft"]

[[reference.log_event]]
event = "draft_head"
all = ["draft head"]
states = ["loaded", "disabled"]

[reference.artifacts.base]
description = "Base pack"
quantization = "bf16"
match = "base"
path = "{base}"

[reference.artifacts.q8]
description = "8-bit pack"
quantization = "8-bit"
match = "q8"

[quality]
reference_artifact = "base"
reference_path = "exact"
probe = "fixture-probe"

[notes]
decode = "Fixture note."
"""


def host_with(batt=BATT_AC, adapter=None, competing=""):
    outputs = {
        ("/usr/bin/pmset", "-g", "batt"): batt,
        (
            "/usr/bin/pmset",
            "-g",
        ): "System-wide power settings:\n powermode            2\n",
        ("/usr/bin/pmset", "-g", "ac"): adapter,
        (
            "/usr/bin/pmset",
            "-g",
            "therm",
        ): "Note: No thermal warning level has been recorded\n",
        ("/usr/bin/sw_vers", "-productVersion"): "26.7.2\n",
        ("/usr/bin/sw_vers", "-buildVersion"): "25H200\n",
        ("/usr/sbin/sysctl", "-n", "hw.model"): "Mac17,7\n",
        ("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"): "Apple M5 Max\n",
        ("/usr/sbin/sysctl", "-n", "hw.memsize"): "137438953472\n",
        ("/bin/ps", "-axo", "pid=,comm="): "  1 /sbin/launchd\n" + competing,
    }
    return lambda: bench.host_snapshot(
        lambda command, timeout=30: outputs.get(tuple(command))
    )


def make_executable(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def make_model(
    path: Path, *, weights=16, model_type="test_family", template=TEMPLATE
) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "vocab_size": 1000,
                "token_codec_config": {"prompt_template": template},
            }
        )
    )
    (path / "model-00001-of-00001.safetensors").write_bytes(b"\x00" * weights)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}})
    )
    return path


def write_profile(directory: Path, layout=None, **overrides) -> Path:
    values = {
        "key": "test-model",
        "status": "complete",
        "adapter": "inkling2-json-v1",
        "template": bench.text_digest(TEMPLATE),
        "images": "true",
        "maximum": 16384,
        "engine": "mlx-serve",
        "base": str(layout.base) if layout else "runtime/base",
        "binary": str(layout.reference) if layout else "runtime/reference",
    }
    values.update(overrides)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{values['key']}.toml"
    path.write_text(PROFILE.format(**values))
    return path


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """Fake binaries, artifacts, client checkout and profile."""
    probe = tmp_path / "llmprobe"
    (probe / "bin/dist").mkdir(parents=True)
    (probe / "src/bench").mkdir(parents=True)
    (probe / "node_modules").mkdir()
    (probe / "package.json").write_text(
        json.dumps({"name": "llmprobe", "version": "0.6.18"})
    )
    (probe / "bin/dist/llmprobe.mjs").write_text("// bundle [inkling-bench-sample]\n")
    (probe / "src/bench/index.ts").write_text("// index\n")
    (probe / "src/bench/corpus.ts").write_text("// corpus\n")
    result = SimpleNamespace(
        root=tmp_path,
        binary=make_executable(tmp_path / "stark-generate", b"\x00current build"),
        baseline=make_executable(tmp_path / "stark-generate-baseline", b"\x00earlier"),
        reference=make_executable(tmp_path / "reference-serve", b"\x00reference"),
        base=make_model(tmp_path / "base"),
        q8=make_model(tmp_path / "q8", weights=8),
        probe=probe,
    )
    result.profile = write_profile(tmp_path / "profiles", result)
    monkeypatch.setattr(bench, "node_version", lambda node: (24, 0, 0))
    helper = make_executable(tmp_path / "stark-template", b"\x00template helper")
    monkeypatch.setattr(bench, "template_helper", lambda: helper)
    monkeypatch.setattr(
        bench,
        "LLMPROBE_PINNED",
        {name: bench.sha256_file(probe / name) for name in bench.LLMPROBE_PINNED},
    )
    return result


def arguments(layout, *extra, command="plan"):
    return [
        command,
        "--profile",
        str(layout.profile),
        "--stark-binary",
        str(layout.binary),
        "--llmprobe",
        str(layout.probe),
        "--output",
        str(layout.root / "out"),
        "--lock-file",
        str(layout.root / "lock"),
        *extra,
    ]


def config_for(layout, *extra):
    return bench.build_config(bench.parser().parse_args(arguments(layout, *extra)))


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------


def test_inkling_profile_is_complete_and_declares_matched_pairs():
    profile = bench.load_profile("inkling-2-mini")
    assert profile.status == "complete" and profile.title == "Inkling 2 Mini"
    assert (profile.min_context, profile.max_context, profile.burst_context) == (
        4352,
        32768,
        2048,
    )
    assert (
        profile.served_model_id == "inkling-2-mini-stark"
        and profile.vocab_size == 201024
    )
    assert profile.chat_adapter == "inkling2-json-v1" and profile.image_capable is True
    assert profile.reasoning == "off" and profile.cache_tag_base == "inkbench"
    assert [item.key for item in profile.artifacts] == ["original", "affine8"]
    reference = profile.reference("mlx-serve")
    assert [item.key for item in reference.artifacts] == [
        "original",
        "author-8bit",
        "community-4bit",
    ]
    assert reference.flags == (
        "--no-pld",
        "--metrics",
        "--warmup-full",
        "--log-file",
        "off",
    )
    matches = {
        item.key: item.match for item in (*profile.artifacts, *reference.artifacts)
    }
    assert matches["affine8"] == matches["author-8bit"] != matches["original"]
    assert profile.reference_arm == "mlx-original" and len(profile.sha256) == 64
    assert profile.facts()["context"] == {
        "minimum": 4352,
        "maximum": 32768,
        "burst": 2048,
    }
    with pytest.raises(bench.BenchError, match="profile_has_no_reference"):
        profile.reference("other-engine")


def test_inkling_default_stark_artifact_is_affine8_and_the_original_stays_an_arm():
    profile = bench.load_profile("inkling-2-mini")
    assert profile.default_artifact == "affine8"
    assert [item.key for item in profile.default_stark_artifacts()] == ["affine8"]
    assert profile.facts()["default_artifact"] == "affine8"
    assert profile.quality["reference_artifact"] == "original"
    args = SimpleNamespace(
        engines="stark",
        stark_binary="target/release/stark-generate",
        stark_build=None,
        stark_artifact=None,
        reference_binary=None,
        reference_artifact=None,
    )
    [default] = bench.profile_arms(profile, args)
    assert (default.name, default.artifact, default.match) == (
        "stark-affine8",
        "affine8",
        "affine8",
    )
    assert default.model == bench.resolve_path("runtime/inkling-affine8/artifact")
    args.stark_artifact = ["original", "affine8"]
    args.engines = None
    args.reference_artifact = ["original", "author-8bit=/models/author-8bit"]
    arms = bench.profile_arms(profile, args)
    assert bench.matched_counterparts(arms) == {
        "stark-original": "mlx-original",
        "stark-affine8": "mlx-author-8bit",
    }


def test_default_artifact_falls_back_to_the_first_and_is_recorded(tmp_path):
    profile = bench.load_profile(str(write_profile(tmp_path)))
    assert profile.default_artifact == ""
    assert [item.key for item in profile.default_stark_artifacts()] == ["base"]
    assert profile.facts()["default_artifact"] == "base"
    path = write_profile(tmp_path)
    path.write_text(
        path.read_text().replace(
            'drafting = "engine-default"',
            'drafting = "engine-default"\ndefault_artifact = "q8"',
            1,
        )
    )
    chosen = bench.load_profile(str(path))
    assert [item.key for item in chosen.default_stark_artifacts()] == ["q8"]
    # q8 declares no path, so a run that relies on the default must name one.
    with pytest.raises(bench.BenchError, match="artifact_path_required:q8"):
        bench.artifact_choices(
            chosen.artifacts,
            None,
            default=chosen.default_stark_artifacts(),
            code="invalid_stark_artifact",
        )


def test_every_checked_in_profile_loads():
    listed = bench.available_profiles()
    assert listed and all(isinstance(profile, bench.Profile) for _, profile in listed)
    assert all(key == profile.key == profile.path.stem for key, profile in listed)


def test_hy4_profile_admits_split_gguf_and_passes_projector_to_reference(tmp_path):
    profile = bench.load_profile("hy4-flash")
    assert profile.image_capable and profile.status == "draft"
    assert profile.reference_arm == "llama-q3-k"
    first = tmp_path / "weights-00001-of-00002.gguf"
    first.write_bytes(b"first")
    second = tmp_path / "weights-00002-of-00002.gguf"
    assert bench.weight_problem(tmp_path) == f"missing GGUF weight shard: {second.name}"
    second.write_bytes(b"second")
    projector = tmp_path / "mmproj-weights.gguf"
    projector.write_bytes(b"projector")
    assert bench.weight_problem(tmp_path) is None
    assert set(bench.model_identity(tmp_path)["weights"]) == {
        first.name,
        second.name,
        projector.name,
    }
    config = SimpleNamespace(profile=profile, mlx_force_mtp=True)
    engine = bench.ENGINES["llama-cpp"]
    arm = bench.Arm("llama-q3-k", engine.key, tmp_path / "llama-server", tmp_path)
    argv = engine.command(config, arm, 1234, "probe")
    assert argv[argv.index("--model") + 1] == str(first)
    assert argv[argv.index("--mmproj") + 1] == str(projector)
    assert argv[argv.index("--spec-type") + 1] == "draft-mtp"
    assert "--no-cache-prompt" in argv
    assert "--log-disable" in argv
    argv = engine.command(config, arm, 1234, "burst")
    assert argv[argv.index("--spec-type") + 1] == "none"


def test_llama_cpp_metrics_name_the_same_phase_boundaries(monkeypatch):
    source = b"# HELP text\nllamacpp:prompt_seconds_total 1.2\nllamacpp:tokens_predicted_seconds_total 0.4\n"
    monkeypatch.setattr(bench, "http_request", lambda *_: io.BytesIO(source))
    assert bench.native_metrics(1234) == {
        bench.PREFILL_METRIC: 1.2,
        bench.DECODE_METRIC: 0.4,
    }


@pytest.mark.parametrize(
    ("old", "new", "error"),
    [
        ('status = "complete"', 'status = "finished"', "profile.status"),
        ("vocab_size = 1000", "vocab_size = 0", "stark.vocab_size"),
        ("maximum = 16384", "maximum = 512", "stark.context"),
        (
            'chat_adapter = "inkling2-json-v1"',
            'chat_adapter = "other"',
            "stark.chat_adapter",
        ),
        ('engine = "mlx-serve"', 'engine = "unknown-serve"', "engine"),
        ("image_capable = true", 'image_capable = "yes"', "stark.image_capable"),
        (
            'reference_arm = "ref-base"',
            'reference_arm = "ref-none"',
            "profile.reference_arm",
        ),
        # The reference arm is a reference engine's arm, never one of Stark's.
        (
            'reference_arm = "ref-base"',
            'reference_arm = "stark-base"',
            "profile.reference_arm",
        ),
        ('drafting = "engine-default"', 'drafting = "always-on"', "stark.drafting"),
        (
            'drafting = "engine-default"',
            'drafting = "engine-default"\ndefault_artifact = "absent"',
            "stark.default_artifact",
        ),
        ('match = "base"\npath', "path", "artifacts.base.match"),
        ("[notes]\ndecode", "[notes]\nnot_a_scenario", "notes"),
        ('reasoning = "off"', 'reasoning = "off"\ntypo_key = 1', "client.typo_key"),
        (
            'reference_artifact = "base"',
            'reference_artifact = "absent"',
            "quality.reference_artifact",
        ),
    ],
)
def test_profile_rejects_bad_values_and_unknown_keys(tmp_path, old, new, error):
    path = write_profile(tmp_path)
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))
    with pytest.raises(
        bench.BenchError, match="invalid_profile:.*" + error.replace(".", r"\.")
    ):
        bench.load_profile(str(path))


def test_profile_lookup_errors_and_listing(tmp_path):
    good = write_profile(tmp_path)
    (tmp_path / "broken.toml").write_text("schema = [")
    write_profile(tmp_path, key="renamed").rename(tmp_path / "other.toml")
    listed = dict(bench.available_profiles(tmp_path))
    assert isinstance(listed["test-model"], bench.Profile)
    assert "profile_unreadable" in str(listed["broken"])
    assert "key_differs_from_file_name" in str(listed["other"])
    assert bench.load_profile("test-model", directory=tmp_path).path == good
    with pytest.raises(bench.BenchError, match="profile_not_found"):
        bench.load_profile("absent", directory=tmp_path)
    with pytest.raises(bench.BenchError, match="invalid_profile_name"):
        bench.load_profile("Bad Name")
    out = io.StringIO()
    assert bench.print_profiles(out, tmp_path) == 2
    text = out.getvalue()
    assert "test-model: Test Model [complete]" in text and "broken: invalid" in text
    assert "ref-base (match base), ref-q8 (match q8)" in text
    out = io.StringIO()
    assert bench.main(["profiles"], out=out) == 0 and "inkling-2-mini" in out.getvalue()


def test_the_tool_itself_names_no_model_limits():
    source = Path(bench.__file__).read_text()
    for literal in (
        "4352",
        "32768",
        "201024",
        "inkling-2-mini-stark",
        "--no-pld",
        "INKBENCH",
    ):
        assert literal not in source, literal


# --------------------------------------------------------------------------
# Arms: (engine build, artifact) pairs from the profile
# --------------------------------------------------------------------------


def test_default_arms_are_the_first_artifact_and_every_pack_with_a_path(layout):
    config = config_for(layout)
    assert [arm.name for arm in config.arms] == ["stark-base", "ref-base"]
    stark, reference = config.arms
    assert (stark.engine, stark.binary, stark.model) == (
        "stark",
        layout.binary,
        layout.base,
    )
    assert (stark.artifact, stark.match, stark.build) == ("base", "base", "current")
    assert (reference.engine, reference.binary) == ("mlx-serve", layout.reference)
    assert config.profile.key == "test-model" and config.layout == "blocks"
    assert config.reasoning == "off" and config.cache_tag_base == "tm"
    assert config.stark_min_context == 1024 and config.run_label == "test-model"
    assert (
        stark.identity()["quantization"] == "bf16"
        and "declared" not in stark.identity()
    )


def test_builds_and_artifacts_form_pairs_and_matched_arms_sit_together(layout):
    config = config_for(
        layout,
        "--stark-build",
        f"baseline={layout.baseline}",
        "--stark-artifact",
        "base",
        "--stark-artifact",
        f"q8={layout.q8}",
        "--reference-artifact",
        "base",
        "--reference-artifact",
        f"mlx-serve:q8={layout.q8}",
    )
    assert [arm.name for arm in config.arms] == [
        "stark-base",
        "ref-base",
        "stark-q8",
        "ref-q8",
        "stark-base-baseline",
        "stark-q8-baseline",
    ]
    assert (
        config.arms[4].binary == layout.baseline and config.arms[5].model == layout.q8
    )
    assert bench.matched_counterparts(config.arms) == {
        "stark-base": "ref-base",
        "stark-q8": "ref-q8",
        "stark-base-baseline": "ref-base",
        "stark-q8-baseline": "ref-q8",
    }
    # The same build on two artifacts is two arms; the same pair twice is not.
    problems, _ = bench.validate(config, strict=True)
    assert problems == []
    config.arms[4].binary = layout.binary
    problems, _ = bench.validate(config, strict=True)
    assert problems == ["two Stark arms point at byte-identical binaries"]
    only = config_for(
        layout, "--engines", "stark", "--stark-artifact", f"q8={layout.q8}"
    )
    assert [arm.name for arm in only.arms] == ["stark-q8"]
    picked = config_for(layout, "--arms", "ref-base")
    assert [arm.name for arm in picked.arms] == ["ref-base"]


def test_an_undeclared_artifact_needs_a_path_and_stays_off_the_scoreboard(layout):
    other = make_model(layout.root / "other")
    config = config_for(layout, "--stark-artifact", f"trial={other}")
    arm = config.arms[0]
    assert arm.name == "stark-trial" and arm.declared is False and arm.match == ""
    assert bench.arm_disqualifications(arm) == ["artifact not declared in the profile"]
    assert arm.identity()["declared"] is False
    assert bench.matched_counterparts(config.arms) == {}
    diagnostic = config_for(layout, "--stark-env", "stark-base=STARK_EXAMPLE=1")
    assert bench.arm_disqualifications(diagnostic.arms[0]) == [
        "diagnostic environment: STARK_EXAMPLE"
    ]


@pytest.mark.parametrize(
    ("extra", "error"),
    [
        (["--stark-artifact", "trial"], "invalid_stark_artifact"),
        (["--stark-artifact", "q8"], "artifact_path_required:q8"),
        (["--reference-artifact", "q8"], "artifact_path_required:q8"),
        (
            ["--stark-artifact", "base", "--stark-artifact", "base"],
            "invalid_stark_artifact",
        ),
        (["--stark-build", "current=/x"], "invalid_stark_build"),
        (["--stark-build", "Bad Name=/x"], "invalid_stark_build"),
        (["--engines", "stark,other"], "unknown_engine"),
        (["--reference-binary", "other=/x"], "invalid_reference_binary"),
        (["--reference-artifact", "other:base"], "invalid_reference_artifact"),
        (["--stark-env", "ref-base=STARK_X=1"], "invalid_stark_env"),
        (["--stark-env", "stark-base=lower=1"], "invalid_stark_env"),
        (["--arms", "stark-none"], "unknown_arms:stark-none"),
        (["--layout", "random"], "invalid_layout"),
        (["--repeats", "0"], "invalid_repeat_count"),
        (["--sessions", "probe,other"], "invalid_sessions"),
        (["--stark-min-context", "99999"], "invalid_min_context"),
        (["--llmprobe-reasoning", "Bad Mode"], "invalid_reasoning_mode"),
    ],
)
def test_arm_and_run_argument_errors(layout, extra, error):
    with pytest.raises(bench.BenchError, match=error):
        config_for(layout, *extra)


# --------------------------------------------------------------------------
# Plan: everything a run would start, without starting it
# --------------------------------------------------------------------------


def run_main(layout, *extra, command="plan", host=None, monkeypatch=None):
    if monkeypatch is not None:

        def forbidden(*args, **kwargs):
            raise AssertionError("the plan started a process")

        monkeypatch.setattr(subprocess, "Popen", forbidden)
    out = io.StringIO()
    code = bench.main(
        arguments(layout, *extra, command=command), host=host or host_with(), out=out
    )
    return code, out.getvalue()


def test_plan_prints_exact_launches_in_order_and_starts_nothing(layout, monkeypatch):
    code, text = run_main(
        layout,
        "--stark-artifact",
        "base",
        "--stark-artifact",
        f"q8={layout.q8}",
        "--reference-artifact",
        "base",
        "--reference-artifact",
        f"q8={layout.q8}",
        host=host_with(BATT_BATTERY),
        monkeypatch=monkeypatch,
    )
    assert code == 0, text
    assert "Stark standard benchmark, profile test-model" in text
    assert "Configuration valid." in text and "Standard run: yes." in text
    assert "Timing run would be refused: not on AC power" in text
    assert "Dry run: no server, worker or client was started." in text
    assert (
        "Drafting: Stark engine default; MLX-Serve engine default; burst forced off on both"
        in text
    )
    assert "prefill: 3 sessions, each 1 warmup + 1 exact 8,800-ID trial" in text
    assert "sustained_decode       n=9" in text and "burst_16x300           n=3" in text
    assert "stark-q8: artifact q8 [8-bit], match class q8" in text
    assert "matched reference arm: ref-q8" in text
    # Block layout: every repeat is one block, arm order reversed on odd repeats.
    order = [line.split()[1:] for line in text.splitlines() if ". repeat " in line]
    assert len(order) == 3 * 3 * 4
    assert order[0] == ["repeat", "0", "probe", "stark-base"]
    assert [row[3] for row in order[:4]] == [
        "stark-base",
        "ref-base",
        "stark-q8",
        "ref-q8",
    ]
    assert [row[2] for row in order[:12:4]] == ["probe", "prefill", "burst"]
    assert [row[3] for row in order[12:16]] == [
        "ref-q8",
        "stark-q8",
        "ref-base",
        "stark-base",
    ]
    assert [row[3] for row in order[24:28]] == [
        "stark-base",
        "ref-base",
        "stark-q8",
        "ref-q8",
    ]
    # Launches come from the profile: limits, flags and the served id.
    worker = f"{layout.binary} --model-path {layout.q8} --vision-staging-root <private temporary directory> --prefix-cache"
    assert worker in text
    assert "served id test-model-stark, context 1,024 to 16,384 by request size" in text
    assert "context 512 per stream, drafting forced off" in text
    probe = f"{layout.reference} --model {layout.base} --serve --host 127.0.0.1 --port <port> --ctx-size 16384 --max-concurrent 1 --flag"
    assert probe + "\n" in text and probe + " --prefix-cache-entries 0\n" in text
    burst = "--port <port> --ctx-size 512 --max-concurrent 16 --flag --no-draft --prefix-cache-entries 0"
    assert burst in text and "--draft\n" not in text
    assert (
        f"{layout.probe}/bin/dist/llmprobe.mjs http://127.0.0.1:<port> --model <model-id> --bench-only"
        in text
    )
    assert (
        "--reasoning off" in text
        and "LLMPROBE_INKLING_LOCAL_CACHE_TAG=tm<repeat" in text
    )
    assert "Quality slot (engine diagnostic, attached after the run):" in text
    assert (
        "required for: stark-base, stark-q8" in text and "probe fixture-probe" in text
    )
    # The plan states the bar a quality result is judged against.
    assert (
        "pass: next-token accuracy not more than 2.0 points below the reference's on the "
        "pooled rows, and top-1 agreement at least 98% where the reference's top "
        "probability is at least 0.5" in text
    )
    assert "the reference predicts under 10% of next tokens" in text
    assert "Without a pass an arm is speed only and holds no standing" in text
    assert "--report <file> [--report <file> ...]" in text
    assert not (layout.root / "out").exists() and not (layout.root / "lock").exists()
    again, same = run_main(layout, "--dry-run", command="run", monkeypatch=monkeypatch)
    assert again == 0 and "Standard run: yes." in same


def test_plan_reports_what_a_standard_run_must_not_have(layout, monkeypatch):
    make_model(
        layout.root / "wrong", model_type="other_family", template="other template"
    )
    code, text = run_main(
        layout,
        "--stark-artifact",
        f"base={layout.root / 'wrong'}",
        "--stark-build",
        f"gone={layout.root / 'missing'}",
        monkeypatch=monkeypatch,
    )
    assert code == 2
    assert (
        "stark-base: config.json model_type is other_family, the profile expects test_family"
        in text
    )
    assert "stark-base: chat template differs from the profile's" in text
    assert "stark-base-gone: engine binary missing or not executable" in text
    write_profile(layout.root / "profiles", layout, status="draft", maximum=8192)
    code, text = run_main(layout, monkeypatch=monkeypatch)
    assert code == 2
    assert "profile test-model is marked draft; only a complete profile is run" in text
    assert (
        "session prefill cannot run for this profile: 8,800 input tokens exceed the 8,192-token"
        in text
    )


def test_plan_marks_every_override_as_diagnostic(layout, monkeypatch):
    code, text = run_main(
        layout,
        "--repeats",
        "2",
        "--layout",
        "legacy",
        "--probe-runs",
        "2",
        "--stark-drafts",
        "off",
        "--reference-force-drafts",
        "--llmprobe-reasoning",
        "client-default",
        "--stark-effort-alias",
        "medium=high",
        "--stark-min-context",
        "2048",
        "--cache-tag-base",
        "other",
        "--warmups",
        "2",
        "--allow-busy-host",
        "--stark-env",
        "stark-base=STARK_EXAMPLE=1",
        monkeypatch=monkeypatch,
    )
    assert code == 0, text
    assert (
        "Standard run: no. This run is diagnostic and stays off the scoreboard:" in text
    )
    for reason in (
        "reasoning mode client-default differs from the profile's off",
        "cache tag base differs from the profile's",
        "client run count overridden",
        "Stark drafting forced off",
        "reference-engine drafting forced on",
        "reasoning effort alias in use",
        "worker minimum context overridden",
        "session layout legacy is not the block layout",
        "fewer than 3 repeats",
        "warmup count differs from 1",
        "busy host allowed",
    ):
        assert "  - " + reason in text, reason
    assert "off the scoreboard: diagnostic environment: STARK_EXAMPLE" in text
    assert "prefill: one session, 2 warmup + 2 exact 8,800-ID trials" in text
    assert "MLX-Serve --mtp forced" in text and "--flag --draft\n" in text
    assert "environment: STARK_*, UZU_* removed; set STARK_EXAMPLE=1" in text
    assert "context 2,048 to 16,384" in text
    # An unpinned client build is a deviation too, and a problem unless allowed.
    monkeypatch.setattr(bench, "LLMPROBE_PINNED", {"package.json": "0" * 64})
    code, text = run_main(layout)
    assert code == 2 and "differs from the pinned build" in text
    code, text = run_main(layout, "--allow-unpinned-llmprobe")
    assert code == 0 and "  - client build differs from the pinned build" in text


def test_timing_run_refuses_a_bad_host_before_creating_anything(
    layout, monkeypatch, capsys
):
    code, _ = run_main(
        layout, command="run", host=host_with(BATT_BATTERY), monkeypatch=monkeypatch
    )
    assert code == 1 and "not_on_ac_power" in capsys.readouterr().err
    busy = host_with(competing="  99 /x/target/release/stark-generate\n")
    code, _ = run_main(layout, command="run", host=busy, monkeypatch=monkeypatch)
    assert (
        code == 1
        and "competing_processes_running:stark-generate" in capsys.readouterr().err
    )
    assert not (layout.root / "out").exists() and not (layout.root / "lock").exists()


def test_receipts_must_stay_private():
    inside = bench.REPOSITORY / "docs/benchmarks/run-output"
    assert "not ignored by Git" in bench.output_problem(
        inside, runner=lambda command: ""
    )
    ignored = bench.REPOSITORY / "runtime/benchmark"
    assert (
        bench.output_problem(ignored, runner=lambda command: str(ignored) + "\n")
        is None
    )
    assert (
        bench.output_problem(Path("/private/tmp/elsewhere"), runner=lambda command: "")
        is None
    )


def test_host_notes_record_a_power_limited_machine():
    snapshot = host_with(BATT_CHARGING, ADAPTER_LIMITED)()
    assert snapshot["adapter"] == {"watts": 94, "rated_watts": 140}
    snapshot["process_thermal_state"] = 1
    assert bench.host_warnings(snapshot) == [
        "battery charging at 84 percent, which lowers the power available to the GPU",
        "power adapter negotiated 94 W of a rated 140 W",
        "process thermal state 1 at start (0 is nominal)",
    ]
    bench.require_timing_host(snapshot)  # recorded, not refused
    quiet = host_with()()
    quiet["process_thermal_state"] = 0
    assert bench.host_warnings(quiet) == [] and quiet["adapter"]["watts"] is None
    assert bench.parse_low_power_mode(" powermode            2\n") is False
    assert bench.parse_low_power_mode(" powermode            1\n") is True


# --------------------------------------------------------------------------
# Engines: the Stark session and the reference-engine interface
# --------------------------------------------------------------------------


class FakeWorker:
    prefix_cache = True

    def __init__(self, images=True):
        self.images = images
        self.mtp = {"supported": True}
        self.seen = []
        self.closed = False

    def read(self):
        return {"ready": True, "mtp": {"supported": True, "default_drafts": 3}}

    def close(self):
        self.closed = True

    def request(self, body, on_delta=None):
        if body["operation"] == "tokenize":
            return {"token_ids": [1] * 4}
        self.seen.append(body)
        if on_delta is not None:
            on_delta("reply")
        cap = body["max_new_tokens"]
        return {
            "generated_token_ids": [2] * cap,
            "text": "reply",
            "input_tokens": len(body["input_token_ids"]),
            "output_tokens": cap,
            "finish_reason": "length",
            "prefill_seconds": 0.02,
            "decode_seconds": 0.03,
        }


class RecordingStream:
    seen: tuple | list = ()  # reset to a fresh list by each test that uses it

    def __init__(self, content, reasoning, **options):
        self.content = content
        RecordingStream.seen.append(("open", options))

    def feed(self, text):
        self.content(text)

    def finish(self, message, **options):
        RecordingStream.seen.append(("finish", options))


def renderer(messages, *, thinking, effort):
    return "rendered " + PROMPT_SENTINEL


def session_config(profile):
    return SimpleNamespace(
        profile=profile,
        stark_worker_timeout=1.0,
        stark_image_check_dir=None,
        stark_effort_aliases={},
        stark_min_context=None,
        stark_drafts="engine",
        mlx_force_mtp=False,
    )


def test_stark_session_takes_limits_identity_and_image_rule_from_the_profile(layout):
    profile = bench.load_profile(str(layout.profile))
    arm = bench.Arm("stark-base", "stark", layout.binary, layout.base)
    worker = FakeWorker()
    RecordingStream.seen = []
    adapters = lambda model: (
        renderer,
        RecordingStream,
        lambda text: {"role": "assistant", "content": text},
    )
    receipts = bench.Receipts(layout.root / "arm")
    with bench.stark_session(
        session_config(profile),
        arm,
        receipts,
        "probe",
        worker_factory=lambda *a: worker,
        adapters=adapters,
    ) as (port, model_id):
        assert model_id == "test-model-stark"
        assert bench.fetch(port, "/v1/models")["data"][0]["id"] == "test-model-stark"
        bench.completion(port, model_id, [1] * 4, 8, None, "salt")
        assert (
            worker.seen[-1]["context_tokens"] == 1024
            and "mtp_drafts" not in worker.seen[-1]
        )
        bench.completion(port, model_id, [1] * 3000, 8, None, "salt")
        assert worker.seen[-1]["context_tokens"] == 3072
        try:
            bench.completion(port, model_id, [1] * 16380, 8, None, "salt")
        except bench.HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("a request beyond the profile's context was accepted")
        # A token outside the artifact's vocabulary is refused.
        with pytest.raises(bench.HTTPError):
            bench.completion(port, model_id, [1000], 8, None, "salt")
    assert worker.closed
    # An image-capable model is measured with images resident, or not at all.
    blind = FakeWorker(images=False)
    with (
        pytest.raises(bench.BenchError, match="stark_images_not_loaded"),
        bench.stark_session(
            session_config(profile),
            arm,
            receipts,
            "probe",
            worker_factory=lambda *a: blind,
            adapters=adapters,
        ),
    ):
        pass
    assert blind.closed
    text_only = bench.load_profile(
        str(
            write_profile(
                layout.root / "text", layout, images="false", adapter="think-tags-v1"
            )
        )
    )
    with bench.stark_session(
        session_config(text_only),
        arm,
        receipts,
        "burst",
        worker_factory=lambda *a: FakeWorker(False),
        adapters=adapters,
    ) as (port, model_id):
        bench.completion(port, model_id, [1] * 4, 8, None, "salt")
    assert "--vision-staging-root" not in " ".join(
        bench.ENGINES["stark"].launch_plan(session_config(text_only), arm, "probe")
    )
    with (
        pytest.raises(bench.BenchError, match="profile_required"),
        bench.stark_session(SimpleNamespace(), arm, receipts, "probe"),
    ):
        pass
    for path in (layout.root / "arm").iterdir():
        assert PROMPT_SENTINEL not in path.read_text()


def test_a_prompt_aware_adapter_sees_the_prompt_in_memory_only(layout):
    profile = bench.load_profile(str(layout.profile))
    arm = bench.Arm("stark-base", "stark", layout.binary, layout.base)
    RecordingStream.seen = []
    parsed = []

    def parse_reply(text, *, prompt=""):
        parsed.append(prompt)
        return {"role": "assistant", "content": text}

    adapters = lambda model: (renderer, RecordingStream, parse_reply, True)
    receipts = bench.Receipts(layout.root / "arm")
    with bench.stark_session(
        session_config(profile),
        arm,
        receipts,
        "probe",
        worker_factory=lambda *a: FakeWorker(),
        adapters=adapters,
    ) as (port, model_id):
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
        }
        with bench.http_request(port, "/v1/chat/completions", body) as response:
            assert response.read().rstrip().endswith(b"data: [DONE]")
    prompt = "rendered " + PROMPT_SENTINEL
    assert RecordingStream.seen == [
        ("open", {"prompt": prompt}),
        ("finish", {"text": "reply"}),
    ]
    assert parsed == [prompt]
    for path in (layout.root / "arm").iterdir():
        assert PROMPT_SENTINEL not in path.read_text()


def test_think_tag_stream_splits_reasoning_like_the_buffered_parser():
    content, reasoning = [], []
    stream = bench.ThinkTagStream(content.append, reasoning.append, prompt="question")
    for piece in ("<thi", "nk>plan</th", "ink>answer"):
        stream.feed(piece)
    stream.finish({}, text="<think>plan</think>answer")
    assert "".join(reasoning) == "plan" and "".join(content) == "answer"
    content, reasoning = [], []
    opened = bench.ThinkTagStream(
        content.append, reasoning.append, prompt="question<think>"
    )
    opened.feed("plan</think>answer")
    opened.finish({}, text="plan</think>answer")
    assert "".join(reasoning) == "plan" and "".join(content) == "answer"


FAKE_SERVER = """#!{python}
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
args = sys.argv[1:]
if "--fail" in args:
    print("fixture-serve 1.0")
    print("error: LoadFailed")
    raise SystemExit(1)
print("draft head loaded", flush=True)
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        return
    def do_GET(self):
        data = json.dumps({{"data": [{{"id": "fixture-model"}}]}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
HTTPServer(("127.0.0.1", int(args[args.index("--port") + 1])), Handler).serve_forever()
"""


class FixtureEngine(bench.ServerEngine):
    """A second reference engine: the whole adapter is a command line."""

    key = "fixture-serve"
    label = "Fixture"
    failure_prefix = "fixture"

    def command(self, config, arm, port, kind):
        reference = bench.require_profile(config).reference(self.key)
        return [
            str(arm.binary),
            "--model",
            str(arm.model),
            "--port",
            str(port),
            *reference.flags,
        ]


def test_a_new_reference_engine_is_an_adapter_and_a_profile_block(layout, monkeypatch):
    with pytest.raises(bench.BenchError, match="invalid_profile:reference"):
        bench.load_profile(
            str(write_profile(layout.root / "early", layout, engine="fixture-serve"))
        )
    monkeypatch.setitem(bench.ENGINES, "fixture-serve", FixtureEngine())
    server = make_executable(
        layout.root / "fixture-serve",
        FAKE_SERVER.format(python=sys.executable).encode(),
    )
    layout.profile = write_profile(
        layout.root / "profiles", layout, engine="fixture-serve", binary=str(server)
    )
    code, text = run_main(layout)
    assert code == 0, text
    assert (
        "Drafting: Stark engine default; Fixture engine default; burst forced off on both"
        in text
    )
    assert f"{server} --model {layout.base} --port <port> --flag" in text
    config = config_for(layout)
    arm = config.arms[1]
    assert (arm.name, arm.engine) == ("ref-base", "fixture-serve")
    receipts = bench.Receipts(layout.root / "ref")
    with bench.engine_session(config, arm, receipts, "probe") as (port, model_id):
        assert model_id == "fixture-model" and port > 0
    events = [
        row["event"] for row in bench.read_jsonl(layout.root / "ref/native.jsonl")
    ]
    assert events == ["server_start", "server_exit"]
    assert sorted(path.name for path in (layout.root / "ref").iterdir()) == [
        "native.jsonl"
    ]
    # A failed start keeps the server's own startup lines and a fixed code.
    layout.profile.write_text(
        layout.profile.read_text().replace('flags = ["--flag"]', 'flags = ["--fail"]')
    )
    failing = config_for(layout)
    with (
        pytest.raises(bench.BenchError, match="fixture_exited:1"),
        bench.engine_session(failing, failing.arms[1], receipts, "prefill"),
    ):
        pass
    assert (
        "LoadFailed"
        in (layout.root / "ref/prefill-server-startup-tail.txt").read_text()
    )
    with pytest.raises(bench.BenchError, match="unknown_engine"):
        bench.engine_session(
            config, bench.Arm("x", "absent", server, layout.base), receipts, "probe"
        )


def test_mlx_serve_launch_and_console_events_come_from_the_profile(layout):
    config = config_for(layout)
    engine = bench.ENGINES["mlx-serve"]
    probe = engine.command(config, config.arms[1], 1234, "probe")
    assert probe[probe.index("--ctx-size") + 1] == "16384" and probe[-1] == "--flag"
    burst = engine.command(config, config.arms[1], 1234, "burst")
    assert burst[burst.index("--ctx-size") + 1] == "512" and "--no-draft" in burst
    assert burst[burst.index("--max-concurrent") + 1] == str(bench.BURST_WIDTH)
    receipts = bench.Receipts(layout.root / "console")
    engine.console(
        config,
        [f"user said {PROMPT_SENTINEL}\n", "model draft head disabled by the index\n"],
        receipts,
        bench.StartupTail(),
    )
    assert bench.read_jsonl(layout.root / "console/native.jsonl") == [
        {"event": "draft_head", "state": "disabled"}
    ]


# --------------------------------------------------------------------------
# Orchestration: the block layout with fake sessions
# --------------------------------------------------------------------------


class FakeEngine(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        self.lock = threading.Lock()
        self.phase = 0.0
        super().__init__((bench.HOST, 0), FakeEngineHandler)


class FakeEngineHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        return

    def reply(self, data: bytes):
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self.reply(json.dumps({"data": [{"id": "fake"}]}).encode())
        else:
            self.reply(
                f"{bench.PREFILL_METRIC} {self.server.phase}\n{bench.DECODE_METRIC} {self.server.phase}\n".encode()
            )

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            self.reply(json.dumps({"tokens": [i % 1000 for i in range(9000)]}).encode())
            return
        with self.server.lock:
            self.server.phase += 0.5
        count, prompt = body["max_tokens"], body["prompt"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for _ in range(count):
            self.wfile.write(
                b'data: {"choices":[{"text":"x","finish_reason":null}]}\n\n'
            )
        self.wfile.write(
            b'data: {"choices":[{"text":"","finish_reason":"length"}]}\n\n'
        )
        usage = {
            "usage": {
                "prompt_tokens": len(prompt) if isinstance(prompt, list) else 40,
                "completion_tokens": count,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "choices": [],
        }
        self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())


FAKE_NODE = """#!{python}
import json, os, sys
args = sys.argv[1:]
if "--input-type=module" in args:
    sys.stdout.write("word " * 20000)
    raise SystemExit(0)
assert os.environ["LLMPROBE_INKLING_LOCAL_CACHE_TAG"].startswith("tm")
def receipt(label, **fields):
    base = dict(label=label, warmup=False, failed=False, inputTokens=74, outputTokens=192, cachedInputTokens=0,
                ttftMs=300, wallMs=2000, decodeTokPerSec=100.0, prefillTokPerSec=None, tokensPerStep=3.1,
                streamCoalesced=False, finishReason="length")
    base.update(fields)
    sys.stderr.write("[inkling-bench-sample] " + json.dumps(base) + "\\n")
receipt("decode 1/3", text={prompt!r})
receipt("sustained decode 1/3", decodeTokPerSec=110.0)
report = {{"bench": {{"decodeLengthNote": "forced"}}, "target": {{"engine": "fake", "model": "m"}},
          "conformance": {{"results": [{prompt!r}]}}}}
open(args[args.index("--save") + 1], "w").write(json.dumps(report))
"""


def test_block_layout_runs_blocks_numbers_trials_and_records_the_standing(layout):
    node = make_executable(
        layout.root / "fake-node",
        FAKE_NODE.format(python=sys.executable, prompt=PROMPT_SENTINEL).encode(),
    )
    config = config_for(
        layout, "--node", str(node), "--stark-env", "stark-base=STARK_EXAMPLE=1"
    )
    problems, facts = bench.validate(config, strict=True)
    assert problems == []
    order = []

    @contextmanager
    def session_factory(config, arm, receipts, kind):
        order.append((kind, arm.name))
        server = FakeEngine()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_port, "fake"
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    run_dir = bench.run_matrix(
        config,
        facts,
        session_factory=session_factory,
        host=host_with(BATT_CHARGING, ADAPTER_LIMITED),
        sleep_check=lambda start, end: {"checked": True, "sleep_events": 0},
    )
    assert run_dir.name.endswith("-test-model")
    assert order == [
        (kind, arm.name) for kind, _, arm in bench.planned_sessions(config)
    ]
    assert order[:6] == [
        ("probe", "stark-base"),
        ("probe", "ref-base"),
        ("prefill", "stark-base"),
        ("prefill", "ref-base"),
        ("burst", "stark-base"),
        ("burst", "ref-base"),
    ]
    assert order[6:8] == [("probe", "ref-base"), ("probe", "stark-base")]
    # One warmup and one measured trial per session; the trial number is the
    # block, so every arm sees the same seed in the same block.
    for arm in ("stark-base", "ref-base"):
        rows = bench.read_jsonl(run_dir / arm / "prefill-8800.jsonl")
        assert [(row["trial"], row["warmup"]) for row in rows] == [
            (0, True),
            (1, False),
            (0, True),
            (2, False),
            (0, True),
            (3, False),
        ]
        assert [row["seed"][-2:] for row in rows[1::2]] == ["-1", "-2", "-3"]
        bursts = bench.read_jsonl(run_dir / arm / "burst.jsonl")
        assert [row["trial"] for row in bursts if not row["warmup"]] == [1, 2, 3]
        assert (run_dir / arm / "probe-02/samples.jsonl").is_file()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["schema"] == bench.SCHEMA and manifest["status"] == "complete"
    assert (
        manifest["profile"]["key"] == "test-model"
        and manifest["profile"]["sha256"] == config.profile.sha256
    )
    assert (
        manifest["config"]["layout"] == "blocks"
        and manifest["config"]["stark_min_context"] == 1024
    )
    assert manifest["standard"] == {"eligible": True, "deviations": []}
    assert set(manifest["harness_files"]) == {
        "benchmark_standard.py",
        "benchmark_profiles/test-model.toml",
    }
    # The thermal state is read from the real host and may add a third note.
    assert [note.split()[0] for note in manifest["host_warnings"][:2]] == [
        "battery",
        "power",
    ]
    stark, reference = manifest["arms"]
    assert stark["scoreboard"] == {
        "eligible": False,
        "reasons": ["diagnostic environment: STARK_EXAMPLE"],
    }
    assert reference["scoreboard"] == {"eligible": True, "reasons": []}
    assert stark["artifact_fingerprint"] == bench.fingerprint(
        bench.model_identity(layout.base)
    )
    assert (stark["artifact"], stark["match"], stark["build"]) == (
        "base",
        "base",
        "current",
    )
    for path in run_dir.rglob("*"):
        if path.is_file():
            assert PROMPT_SENTINEL not in path.read_text()
    summary = bench.summarize([run_dir])
    assert summary["title"] == "Stark standard benchmark: Test Model"
    assert summary["reference_arm"] == "ref-base"
    decode = summary["cells"]["decode"]["stark-base"]
    assert (
        decode["paired_vs_reference"]["n"] == 3
        and decode["level"] == bench.LEVEL_CLIENT
    )
    assert (
        summary["cells"]["prefill_8800_native"]["stark-base"]["level"]
        == bench.LEVEL_PHASE
    )
    assert (
        summary["cells"]["prefill_8800_client"]["stark-base"]["paired_vs_reference"][
            "n"
        ]
        == 3
    )
    assert summary["scenarios"]["decode"]["profile_note"] == "Fixture note."
    assert summary["scoreboard"]["stark-base"]["eligible"] is False
    out = io.StringIO()
    assert bench.report_run(run_dir, out) == 0 and "Run complete" in out.getvalue()


# --------------------------------------------------------------------------
# Summary: paired blocks, standing, validity, older run directories
# --------------------------------------------------------------------------


def sample(label, value, *, inputs=74, output=192, cached=0):
    return {
        "label": label,
        "scenario": bench.scenario_for_label(label),
        "warmup": False,
        "failed": False,
        "inputTokens": inputs,
        "outputTokens": output,
        "cachedInputTokens": cached,
        "ttftMs": 300.0,
        "wallMs": 2000.0,
        "decodeTokPerSec": value,
        "prefillTokPerSec": None,
        "tokensPerStep": 3.0,
        "streamCoalesced": False,
        "finishReason": "length",
        "inputUtf8Sha256": None,
    }


def prefill_row(trial, rate, *, warmup=False):
    return {
        "trial": trial,
        "warmup": warmup,
        "valid": True,
        "seed": f"seed-{trial}",
        "prompt_tokens": 8800,
        "cached_tokens": 0,
        "native_prefill_tok_s": rate,
        "client_prompt_tok_s": rate,
        "client_ttft_s": 8800 / rate,
        "input_ids_le_u32_sha256": bench.text_digest(f"seed-{trial}"),
    }


def burst_row(trial, rate, *, warmup=False):
    return {
        "trial": trial,
        "warmup": warmup,
        "valid": True,
        "aggregate_tok_s": rate,
        "wall_s": 4800 / rate,
        "total_generated_tokens": 4800,
        "requests": [{"prompt_tokens": 40, "cached_tokens": 0}],
    }


def quality_result(
    *, accuracy=(0.42, 0.41), confident=(1500, 0.99), kind="natural-text", **corpus
):
    """A slot-schema quality result; it passes the bar unless told otherwise."""
    rows, agreement = confident
    return {
        "schema": bench.QUALITY_SCHEMA,
        "corpus": {"rows": 2580, "inputs": 10, "kind": kind, **corpus},
        "reference": {"path": "exact"},
        "candidate": {"path": "default"},
        "metrics": {
            "next_token_accuracy": {"reference": accuracy[0], "candidate": accuracy[1]},
            "top1_agreement": {
                "all": 0.97,
                "strata": [
                    {
                        "reference_top1_probability_min": 0.5,
                        "rows": rows,
                        "agreement": agreement,
                    }
                ],
            },
            "kl_reference_to_candidate": {"mean": 0.004, "max": 0.31},
        },
    }


def write_run(root: Path, name: str, arms: dict, **manifest_fields) -> Path:
    """arms: name -> {engine, match, repeats: [[samples]], prefill, burst, model, quality}.

    quality is the keyword arguments of quality_result; the result is bound to
    the arm's own artifact unless "fingerprint" says otherwise.
    """
    run_dir = root / name
    manifest = {
        "schema": bench.SCHEMA,
        "engine": "all",
        "status": "complete",
        "config": {"reasoning": "off", "layout": "blocks"},
        "profile": {
            "key": "test-model",
            "title": "Test Model",
            "reference_arm": "ref-base",
        },
        "standard": {"eligible": True, "deviations": []},
        "host_start": host_with()(),
        "host_warnings": [],
        "sleep": {"sleep_events": 0},
        "arms": [],
    }
    manifest.update(manifest_fields)
    for arm, spec in arms.items():
        identity = (
            bench.model_identity(spec["model"]) if spec.get("model") else {"files": {}}
        )
        entry = {
            "name": arm,
            "engine": spec.get("engine", "stark"),
            "binary_sha256": bench.text_digest(arm),
            "description": arm,
            "diagnostic_env": {},
            "model": identity,
            "artifact_fingerprint": bench.fingerprint(identity),
            "scoreboard": spec.get("scoreboard", {"eligible": True, "reasons": []}),
        }
        if spec.get("match"):
            entry.update(
                artifact=spec["match"], match=spec["match"], quantization=spec["match"]
            )
        manifest["arms"].append(entry)
        for repeat, samples in enumerate(spec.get("repeats", [])):
            receipts = bench.Receipts(run_dir / arm / f"probe-{repeat:02d}")
            for row in samples:
                receipts.write("samples", row)
        for row in spec.get("prefill", []):
            bench.Receipts(run_dir / arm).write("prefill-8800", row)
        for row in spec.get("burst", []):
            bench.Receipts(run_dir / arm).write("burst", row)
        if "quality" in spec:
            options = dict(spec["quality"])
            bound = options.pop("fingerprint", entry["artifact_fingerprint"])
            result = quality_result(**options)
            result["candidate"]["artifact_fingerprint"] = bound
            (run_dir / arm).mkdir(parents=True, exist_ok=True)
            (run_dir / arm / "quality.json").write_text(json.dumps(result))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


def decode_repeats(values, **fields):
    return [[sample("decode 1/3", value, **fields)] for value in values]


def test_summary_reads_a_standing_only_from_blocks_that_all_agree(tmp_path):
    run = write_run(
        tmp_path,
        "r1",
        {
            "stark-base": {
                "match": "base",
                "quality": {},
                "burst": [burst_row(1, 60.0), burst_row(2, 61.0), burst_row(3, 62.0)],
                "repeats": [
                    [
                        sample("decode 1/3", 110.0),
                        sample("decode 2/3", 114.0),
                        sample("spec:novel 1/3", 90.0),
                    ],
                    [
                        sample("decode 1/3", 99.0),
                        sample("decode 2/3", 101.0),
                        sample("spec:novel 1/3", 105.0),
                    ],
                    [sample("decode 1/3", 121.0), sample("spec:novel 1/3", 95.0)],
                ],
                "prefill": [
                    prefill_row(1, 1000.0),
                    prefill_row(2, 900.0),
                    prefill_row(3, 1100.0),
                ],
            },
            "ref-base": {
                "engine": "mlx-serve",
                "match": "base",
                "burst": [
                    burst_row(1, 170.0),
                    burst_row(2, 171.0),
                    burst_row(3, 172.0),
                ],
                "repeats": [
                    [sample("decode 1/3", 100.0), sample("spec:novel 1/3", 100.0)],
                    [sample("decode 1/3", 80.0), sample("spec:novel 1/3", 100.0)],
                    [sample("decode 1/3", 110.0), sample("spec:novel 1/3", 100.0)],
                ],
                "prefill": [
                    prefill_row(1, 2000.0),
                    prefill_row(2, 2000.0),
                    prefill_row(3, 2000.0),
                ],
            },
            "stark-q8": {"match": "q8", "repeats": decode_repeats([130.0, 131.0])},
            "ref-q8": {
                "engine": "mlx-serve",
                "match": "q8",
                "repeats": decode_repeats([120.0, 125.0]),
            },
        },
    )
    summary = bench.summarize([run])
    decode = summary["cells"]["decode"]
    paired = decode["stark-base"]["paired_vs_reference"]
    # Block medians 112/100, 100/80 and 121/110: every block ahead.
    assert paired["standing"] == "ahead" and paired["n"] == 3 and paired["wins"] == 3
    assert paired["median"] == pytest.approx(1.12) and paired["min"] == pytest.approx(
        1.1
    )
    assert paired["max"] == pytest.approx(1.25)
    # The ratio of medians is kept, and is a different number.
    assert decode["stark-base"]["speed_vs_reference"] == pytest.approx(110.0 / 100.0)
    assert decode["ref-base"]["paired_vs_reference"] is None
    novel = summary["cells"]["spec_novel"]["stark-base"]["paired_vs_reference"]
    assert novel["standing"] == "unresolved" and novel["wins"] == 1
    prefill = summary["cells"]["prefill_8800_native"]["stark-base"][
        "paired_vs_reference"
    ]
    assert prefill["standing"] == "behind" and prefill["max"] == pytest.approx(0.55)
    # Matched artifacts are compared with each other; two blocks resolve nothing.
    assert summary["matched_arms"] == {"stark-base": "ref-base", "stark-q8": "ref-q8"}
    matched = decode["stark-q8"]["paired_vs_matched"]
    assert (
        matched["arm"] == "ref-q8"
        and matched["standing"] == "too few pairs"
        and matched["n"] == 2
    )
    assert decode["stark-base"]["paired_vs_matched"]["standing"] == "ahead"
    assert decode["stark-base"]["tier"] == "standard"
    assert bench.scenario_tier("spec_predictable") == "supplementary"
    # The burst is reported and decides nothing: Stark serializes one worker.
    burst = summary["cells"]["burst_16x300"]["stark-base"]
    assert burst["tier"] == "supplementary" and "burst_16x300" not in (
        bench.STANDARD_SCENARIOS
    )
    assert burst["paired_vs_reference"]["standing"] == "reported only"
    assert burst["paired_vs_reference"]["n"] == 3
    assert burst["paired_vs_reference"]["max"] == pytest.approx(62.0 / 172.0)
    assert burst["paired_vs_matched"]["standing"] == "reported only"
    # A passing quality result makes the figures a standard result; without one
    # a Stark arm is speed only. A reference arm has no slot.
    assert summary["quality"]["stark-base"]["status"] == "pass"
    assert summary["quality"]["stark-q8"]["status"] == "missing"
    assert summary["quality"]["ref-base"]["status"] == "not applicable"
    assert summary["scoreboard"]["stark-base"] == {"eligible": True, "reasons": []}
    assert summary["scoreboard"]["stark-q8"] == {
        "eligible": False,
        "reasons": ["speed only: quality missing"],
    }
    assert summary["scoreboard"]["ref-base"] == {"eligible": True, "reasons": []}
    markdown = bench.render_markdown(summary)
    assert markdown.startswith("# Stark standard benchmark: Test Model")
    assert (
        "| server, client-timed | standard | 1.12 (1.10-1.25, n=3) ahead | ref-base: 1.12 (1.10-1.25, n=3) ahead |"
        in markdown
    )
    assert (
        "| server, engine phase time | standard | 0.50 (0.45-0.55, n=3) behind |"
        in markdown
    )
    assert (
        "| server, client-timed | supplementary | 0.36 (0.35-0.36, n=3) reported only |"
        in markdown
    )
    assert "| stark-q8 | q8 | q8 | ref-q8 | missing (no result attached) |" in markdown
    assert "no: speed only: quality missing" in markdown
    assert "every block agrees and there are at least 3 blocks" in markdown
    json.dumps(summary)


@pytest.mark.parametrize(
    ("quality", "status"),
    [
        (None, "missing"),
        ({"accuracy": (0.42, 0.39)}, "fail"),
        ({"confident": (1500, 0.97)}, "fail"),
        ({"accuracy": (0.05, 0.05)}, "not assessed"),
        ({"kind": "synthetic"}, "not assessed"),
        ({"fingerprint": "f" * 64}, "not assessed"),
    ],
)
def test_a_stark_arm_without_a_passing_quality_result_holds_no_standing(
    tmp_path, quality, status
):
    stark = {
        "match": "base",
        "repeats": decode_repeats([110.0, 111.0, 112.0]),
        "prefill": [
            prefill_row(1, 1000.0),
            prefill_row(2, 900.0),
            prefill_row(3, 1100.0),
        ],
    }
    if quality is not None:
        stark["quality"] = quality
    reference = {
        "engine": "mlx-serve",
        "match": "base",
        "repeats": decode_repeats([100.0, 100.0, 100.0]),
        "prefill": [
            prefill_row(1, 2000.0),
            prefill_row(2, 2000.0),
            prefill_row(3, 2000.0),
        ],
    }
    run = write_run(tmp_path, "r1", {"stark-base": stark, "ref-base": reference})
    summary = bench.summarize([run])
    assert summary["quality"]["stark-base"]["status"] == status
    assert summary["scoreboard"]["stark-base"] == {
        "eligible": False,
        "reasons": [f"speed only: quality {status}"],
    }
    # The ratio and its range are still reported; the standing is withheld,
    # whichever way the speed went.
    for scenario, low in (("decode", 1.10), ("prefill_8800_native", 0.45)):
        cell = summary["cells"][scenario]["stark-base"]
        for key in ("paired_vs_reference", "paired_vs_matched"):
            assert cell[key]["standing"] == "speed only", (scenario, key)
            assert cell[key]["n"] == 3 and cell[key]["min"] == pytest.approx(low)
    # The reference engine's own arms are unaffected.
    assert summary["scoreboard"]["ref-base"]["eligible"] is True
    markdown = bench.render_markdown(summary)
    assert "1.11 (1.10-1.12, n=3) speed only" in markdown
    assert f"no: speed only: quality {status}" in markdown
    if status == "not assessed" and quality.get("fingerprint"):
        assert "not assessed (the result is bound to another artifact)" in markdown


def test_summary_marks_unmatched_work_and_diagnostic_arms_not_comparable(tmp_path):
    run = write_run(
        tmp_path,
        "r1",
        {
            "stark-base": {
                "match": "base",
                "repeats": [
                    [
                        sample("decode 1/3", 110.0, inputs=80),
                        sample("decode 2/3", 60.0, output=150),
                    ],
                    [sample("spec:novel 1/3", 70.0, cached=9)],
                ],
                "prefill": [prefill_row(1, 1000.0)],
                "scoreboard": {
                    "eligible": False,
                    "reasons": ["diagnostic environment: STARK_EXAMPLE"],
                },
            },
            "stark-q8": {"match": "q8", "repeats": decode_repeats([100.0])},
            "ref-base": {
                "engine": "mlx-serve",
                "match": "base",
                "repeats": decode_repeats([100.0]),
                "prefill": [
                    {**prefill_row(1, 2000.0), "input_ids_le_u32_sha256": "e" * 64}
                ],
            },
        },
        standard={"eligible": False, "deviations": ["Stark drafting forced off"]},
        host_warnings=["power adapter negotiated 94 W of a rated 140 W"],
    )
    summary = bench.summarize([run])
    findings = {item["code"]: item for item in summary["validity"]}
    assert findings["rendered_inputs_differ"]["blocking"] is True
    assert findings["rendered_inputs_differ"]["arms"] == [
        "ref-base",
        "stark-base",
        "stark-q8",
    ]
    assert "stark-base 80" in findings["rendered_inputs_differ"]["detail"]
    assert findings["prefill_inputs_differ"]["blocking"] is True
    assert findings["short_outputs_excluded"]["arms"] == ["stark-base"]
    assert (
        "reused 9 cached input tokens"
        in findings["prefix_reuse_in_cold_cell"]["detail"]
    )
    assert findings["host_condition"]["arms"] == ["ref-base", "stark-base", "stark-q8"]
    assert findings["artifact_not_matched"]["arms"] == ["stark-q8"]
    standing = summary["scoreboard"]["stark-base"]
    assert standing["eligible"] is False
    assert "Stark drafting forced off" in standing["reasons"]
    assert "diagnostic environment: STARK_EXAMPLE" in standing["reasons"]
    assert any(
        "did not render the same prompt" in reason for reason in standing["reasons"]
    )
    assert (
        summary["cells"]["prefill_8800_native"]["stark-base"]["paired_vs_reference"][
            "standing"
        ]
        == "too few pairs"
    )
    markdown = bench.render_markdown(summary)
    assert "Not comparable: decode: input token counts differ between arms" in markdown
    assert "Note: run r1: power adapter negotiated 94 W of a rated 140 W." in markdown
    # Subsets are a match: an arm measured over fewer repeats saw fewer inputs.
    more = write_run(
        tmp_path,
        "r2",
        {
            "stark-base": {
                "match": "base",
                "repeats": [[sample("unseen prefill 1/3", 800.0, inputs=7514)]],
            },
            "ref-base": {
                "engine": "mlx-serve",
                "match": "base",
                "repeats": [
                    [sample("unseen prefill 1/3", 2000.0, inputs=7514)],
                    [sample("unseen prefill 1/3", 2000.0, inputs=7586)],
                ],
            },
        },
    )
    assert "rendered_inputs_differ" not in {
        item["code"] for item in bench.summarize([more])["validity"]
    }


def test_summary_reads_run_directories_from_before_the_standard(tmp_path):
    legacy = write_run(
        tmp_path,
        "20261122-175344-both",
        {
            "stark-current": {
                "repeats": decode_repeats([79.7]),
                "prefill": [
                    prefill_row(1, 1194.0),
                    prefill_row(2, 1200.0),
                    prefill_row(3, 1190.0),
                ],
            },
            "mlx-original": {
                "engine": "mlx-serve",
                "repeats": decode_repeats([79.3]),
                "prefill": [
                    prefill_row(1, 2235.0),
                    prefill_row(2, 2230.0),
                    prefill_row(3, 2240.0),
                ],
            },
        },
        schema="stark-inkling-engine-matrix-v2",
        config={"reasoning": "off"},
        profile=None,
        standard=None,
    )
    summary = bench.summarize([legacy])
    assert summary["title"] == "Stark standard benchmark: Inkling 2 Mini"
    assert summary["reference_arm"] == "mlx-original"
    assert summary["arms"]["stark-current"]["runs"][0]["layout"] == "legacy"
    assert summary["scoreboard"]["stark-current"]["eligible"] is None
    assert (
        summary["scoreboard"]["stark-current"]["reasons"][0]
        == "recorded before the standard was defined"
    )
    cell = summary["cells"]["prefill_8800_native"]["stark-current"]
    assert cell["speed_vs_reference"] == pytest.approx(1194.0 / 2235.0)
    # One session per arm there: trials pair by index, not in time.
    assert cell["paired_vs_reference"]["standing"] == "not time-paired"
    assert (
        "not judged: recorded before the standard was defined"
        in bench.render_markdown(summary)
    )
    assert "Sustained" not in summary["scenarios"]["decode"]["profile_note"]
    # The same arm measured under the block layout is a different condition.
    blocks = write_run(
        tmp_path,
        "later",
        {"stark-current": {"repeats": decode_repeats([81.0])}},
        profile={
            "key": "inkling-2-mini",
            "title": "Inkling 2 Mini",
            "reference_arm": "mlx-original",
        },
    )
    joined = bench.summarize([legacy, blocks])
    assert set(joined["cells"]["decode"]) == {
        "stark-current",
        "mlx-original",
        "stark-current@later",
    }
    with pytest.raises(bench.BenchError, match="unknown_manifest_schema"):
        bench.summarize([write_run(tmp_path, "bad", {}, schema="other-schema")])


# --------------------------------------------------------------------------
# Quality slot: one bar, judged on the pooled rows of the corpus
# --------------------------------------------------------------------------

PROBE_OUTPUT = {
    "prompt_tokens": 4096,
    "context": 8192,
    "window": 258,
    "rows": 258,
    "reference": {"target": PROMPT_SENTINEL, "affine8": False, "exact_prefill": True},
    "candidate": {
        "target": "model.safetensors",
        "affine8": False,
        "exact_prefill": False,
    },
    "next_token_accuracy_reference": 0.5,
    "next_token_accuracy_candidate": 0.5,
    "top1_agreement": 0.97,
    "confident_rows_p50": 150,
    "top1_agreement_p50": 0.99,
    "confident_rows_p90": 0,
    "top1_agreement_p90": None,
    "kl_reference_to_candidate_mean": 0.004,
    "kl_reference_to_candidate_max": 0.31,
    "next_token_logprob_abs_change_mean": 0.02,
}
PROBE_FORMAT = "inkling-slice-quality-compare"


def probe_output(**changes):
    return {**PROBE_OUTPUT, **changes}


def test_quality_report_is_reduced_to_numbers_and_fixed_codes():
    clean = bench.normalize_quality(PROBE_OUTPUT, PROBE_FORMAT)
    assert (
        clean["schema"] == bench.QUALITY_SCHEMA
        and clean["level"] == bench.LEVEL_DIAGNOSTIC
    )
    assert clean["corpus"] == {
        "rows": 258,
        "inputs": 1,
        "prompt_tokens": 4096,
        "context": 8192,
        "window": 258,
    }
    assert clean["reference"] == {"path": "exact"} and clean["candidate"] == {
        "path": "default"
    }
    assert clean["metrics"]["next_token_accuracy"] == {
        "reference": 0.5,
        "candidate": 0.5,
    }
    # A stratum with no rows has no rate and is dropped, not reported as zero.
    assert clean["metrics"]["top1_agreement"] == {
        "all": 0.97,
        "strata": [
            {"reference_top1_probability_min": 0.5, "rows": 150, "agreement": 0.99}
        ],
    }
    assert clean["metrics"]["kl_reference_to_candidate"] == {
        "unit": "nats",
        "mean": 0.004,
        "max": 0.31,
    }
    assert PROMPT_SENTINEL not in json.dumps(clean)
    assert bench.normalize_quality(clean) == clean
    example = bench.normalize_quality(bench.QUALITY_SCHEMA_EXAMPLE)
    assert (
        example["metrics"]["top1_agreement"]["strata"] == []
        and example["corpus"]["inputs"] == 1
    )
    for field, value in (
        ("top1_agreement", 1.2),
        ("next_token_accuracy_candidate", None),
        ("kl_reference_to_candidate_mean", -0.1),
        ("rows", 0),
    ):
        with pytest.raises(bench.BenchError, match="invalid_quality_report"):
            bench.normalize_quality(probe_output(**{field: value}), PROBE_FORMAT)
    with pytest.raises(bench.BenchError, match="unknown_quality_format"):
        bench.normalize_quality(PROBE_OUTPUT, "other-format")
    with pytest.raises(bench.BenchError, match="invalid_quality_report:schema"):
        bench.normalize_quality(PROBE_OUTPUT)
    with pytest.raises(bench.BenchError, match="invalid_quality_report:corpus.inputs"):
        bench.normalize_quality(quality_result(inputs=0))


@pytest.mark.parametrize(
    ("options", "verdict", "reason"),
    [
        # Not more than 2.0 points below the reference, and at least 98%
        # agreement where the reference's top probability is at least 0.5:
        # pass. The limits themselves pass.
        ({}, "pass", None),
        ({"accuracy": (0.42, 0.40)}, "pass", None),
        # One-sided: scoring higher than the reference is never a failure.
        ({"accuracy": (0.42, 0.44)}, "pass", None),
        ({"accuracy": (0.42, 0.45)}, "pass", None),
        ({"accuracy": (0.42, 0.70)}, "pass", None),
        ({"confident": (1500, 0.98)}, "pass", None),
        ({"accuracy": (0.10, 0.10)}, "pass", None),
        # Either criterion alone fails the result, and both are named.
        (
            {"accuracy": (0.42, 0.399)},
            "fail",
            "2.1 points below the reference's on the pooled rows (limit 2.0)",
        ),
        ({"accuracy": (0.42, 0.39)}, "fail", "3.0 points below the reference's"),
        # Higher accuracy does not excuse disagreement on confident rows.
        (
            {"accuracy": (0.42, 0.50), "confident": (1500, 0.9)},
            "fail",
            "top-1 agreement is 90.0%",
        ),
        (
            {"confident": (1500, 0.979)},
            "fail",
            "top-1 agreement is 97.9% where the reference's top probability is at least 0.5 (floor 98%)",
        ),
        (
            {"accuracy": (0.42, 0.30), "confident": (1500, 0.9)},
            "fail",
            "limit 2.0); top-1 agreement is 90.0%",
        ),
        # A corpus that cannot tell is never a pass, however good the figures.
        (
            {"accuracy": (0.09, 0.09)},
            "not assessed",
            "the reference predicts 9% of next tokens on this corpus",
        ),
        (
            {"accuracy": (0.0, 0.0), "confident": (1500, 1.0)},
            "not assessed",
            "the reference predicts 0%",
        ),
        ({"kind": "synthetic"}, "not assessed", "corpus is not natural text"),
        (
            {"confident": (0, 0.0)},
            "not assessed",
            "no rows where the reference's top probability is at least 0.5",
        ),
        # Not assessed comes first: a weak corpus is not reported as a failure.
        (
            {"accuracy": (0.05, 0.01), "confident": (10, 0.5)},
            "not assessed",
            "the reference predicts 5%",
        ),
    ],
)
def test_quality_bar_is_pass_fail_or_not_assessed(options, verdict, reason):
    result = bench.normalize_quality(quality_result(**options))
    result["corpus"]["kind"] = options.get("kind", "natural-text")
    found, why, criteria = bench.quality_verdict(result)
    assert found == verdict and found in bench.QUALITY_VERDICTS
    assert (why is None) if reason is None else (reason in why), why
    assert criteria["accuracy_shortfall_limit_points"] == 2.0
    assert criteria["confident_agreement_floor"] == 0.98
    assert criteria["confident_probability"] == 0.5
    assert criteria["reference_accuracy_floor"] == 0.10
    assert criteria["pooled_rows"] == 2580 and criteria["pooled_inputs"] == 10
    reference, arm = options.get("accuracy", (0.42, 0.41))
    # Signed: negative when the arm scores higher than the reference.
    assert criteria["accuracy_shortfall_points"] == pytest.approx(
        (reference - arm) * 100
    )
    if arm > reference:
        assert "next-token accuracy" not in (why or "")
    judged = bench.judge_quality(result)
    assert (judged["verdict"], judged["verdict_reason"]) == (found, why)


def test_quality_is_judged_on_pooled_rows_not_on_single_inputs():
    # Ten inputs of 258 rows. Input 3 is 2.3 points below the reference on its
    # own, which would fail alone; the pooled corpus is 0.23 points apart.
    per_input = [
        bench.normalize_quality(
            probe_output(
                next_token_accuracy_candidate=0.477 if index == 3 else 0.5,
                confident_rows_p50=150 + index,
                top1_agreement_p50=1.0 if index else 0.98,
                confident_rows_p90=40 if index < 2 else 0,
                top1_agreement_p90=1.0 if index < 2 else None,
                kl_reference_to_candidate_max=0.1 * (index + 1),
            ),
            PROBE_FORMAT,
        )
        for index in range(10)
    ]
    alone = dict(
        per_input[3], corpus={**per_input[3]["corpus"], "kind": "natural-text"}
    )
    assert bench.quality_verdict(alone)[0] == "fail"
    pooled = bench.pool_quality(per_input)
    assert pooled["corpus"] == {
        "rows": 2580,
        "inputs": 10,
        "prompt_tokens": 4096,
        "context": 8192,
        "window": 258,
    }
    accuracy = pooled["metrics"]["next_token_accuracy"]
    assert accuracy["reference"] == pytest.approx(0.5)
    assert accuracy["candidate"] == pytest.approx(0.5 - 0.023 / 10)
    confident, strong = pooled["metrics"]["top1_agreement"]["strata"]
    assert confident["rows"] == sum(150 + index for index in range(10))
    assert confident["agreement"] == pytest.approx(1 - 0.02 * 150 / confident["rows"])
    assert strong == {
        "reference_top1_probability_min": 0.9,
        "rows": 80,
        "agreement": 1.0,
    }
    assert pooled["metrics"]["kl_reference_to_candidate"]["max"] == pytest.approx(1.0)
    assert pooled["metrics"]["kl_reference_to_candidate"]["mean"] == pytest.approx(
        0.004
    )
    pooled["corpus"]["kind"] = "natural-text"
    verdict, reason, criteria = bench.quality_verdict(pooled)
    assert (verdict, reason) == ("pass", None)
    assert criteria["accuracy_shortfall_points"] == pytest.approx(0.23)
    assert criteria["pooled_rows"] == 2580 and criteria["pooled_inputs"] == 10
    # One report is its own pool; different execution paths cannot be pooled.
    assert bench.pool_quality(per_input[:1]) is per_input[0]
    other = bench.normalize_quality(
        probe_output(candidate={"exact_prefill": True}), PROBE_FORMAT
    )
    with pytest.raises(bench.BenchError, match="quality_reports_differ:candidate.path"):
        bench.pool_quality([per_input[0], other])
    with pytest.raises(bench.BenchError, match="no_quality_reports"):
        bench.pool_quality([])
    # A missing log-probability figure in one input leaves the pooled one unset.
    partial = bench.normalize_quality(
        probe_output(next_token_logprob_abs_change_mean=None), PROBE_FORMAT
    )
    assert (
        bench.pool_quality([per_input[0], partial])["metrics"][
            "next_token_logprob_abs_change_mean"
        ]
        is None
    )


def attach_arguments(run, layout, reports, arm, artifact):
    arguments = ["quality", "attach", "--run", str(run), "--arm", arm]
    for report in reports:
        arguments += ["--report", str(report)]
    return [
        *arguments,
        "--candidate-artifact",
        str(artifact),
        "--format",
        PROBE_FORMAT,
        "--corpus-kind",
        "natural-text",
        "--corpus-sha256",
        "c" * 64,
        "--probe-binary",
        str(layout.binary),
        "--probe-name",
        "fixture-probe",
        "--reference-artifact",
        str(layout.base),
    ]


def test_quality_attach_pools_judges_and_binds_to_the_timed_artifact(layout, capsys):
    blocks = decode_repeats([110.0, 111.0, 112.0])
    run = write_run(
        layout.root,
        "r1",
        {
            "stark-base": {"match": "base", "model": layout.base, "repeats": blocks},
            "stark-q8": {"match": "q8", "model": layout.q8, "repeats": blocks},
            "stark-low": {"match": "low", "model": layout.q8, "repeats": blocks},
            "ref-base": {
                "engine": "mlx-serve",
                "match": "base",
                "repeats": decode_repeats([100.0, 100.0, 100.0]),
            },
        },
    )

    def reports(name, changes_by_input):
        paths = []
        for index, changes in enumerate(changes_by_input):
            path = layout.root / f"{name}-{index}.json"
            path.write_text(json.dumps(probe_output(**changes)))
            paths.append(path)
        return paths

    # Input 1 alone is 2.2 points off; the pooled corpus of four is 0.55.
    passing = reports("pass", [{}, {"next_token_accuracy_candidate": 0.478}, {}, {}])
    good = attach_arguments(run, layout, passing, "stark-base", layout.base)
    # A result measured on another artifact cannot be attached to this arm.
    wrong = attach_arguments(run, layout, passing, "stark-base", layout.q8)
    assert (
        bench.main(wrong) == 1
        and "quality_artifact_mismatch" in capsys.readouterr().err
    )
    assert (
        bench.main(attach_arguments(run, layout, passing, "ref-base", layout.base)) == 1
    )
    assert "quality_slot_is_for_stark_arms" in capsys.readouterr().err
    out = io.StringIO()
    assert bench.main(good, out=out) == 0
    assert (
        out.getvalue()
        == "Quality result attached: pass; pooled over 1032 rows of 4 input(s)\n"
    )
    assert (
        bench.main(good) == 1
        and "quality_report_already_attached" in capsys.readouterr().err
    )
    assert bench.main([*good, "--replace"], out=io.StringIO()) == 0
    stored = json.loads((run / "stark-base/quality.json").read_text())
    assert stored["verdict"] == "pass" and stored["verdict_reason"] is None
    assert stored["corpus"]["rows"] == 1032 and stored["corpus"]["inputs"] == 4
    assert stored["criteria"]["accuracy_shortfall_points"] == pytest.approx(0.55)
    assert stored["candidate"]["artifact_fingerprint"] == bench.fingerprint(
        bench.model_identity(layout.base)
    )
    assert stored["probe"]["binary_sha256"] == bench.sha256_file(layout.binary)
    assert (
        stored["bound_to"]["arm"] == "stark-base"
        and stored["corpus"]["sha256"] == "c" * 64
    )
    assert stored["bound_to"]["source_report_sha256"] == [
        bench.sha256_file(path) for path in passing
    ]
    assert PROMPT_SENTINEL not in (run / "stark-base/quality.json").read_text()
    # Fail: every input is 3 points below the reference.
    failing = reports("fail", [{"next_token_accuracy_candidate": 0.47}] * 2)
    out = io.StringIO()
    assert (
        bench.main(
            attach_arguments(run, layout, failing, "stark-q8", layout.q8), out=out
        )
        == 0
    )
    assert out.getvalue().startswith(
        "Quality result attached: fail (next-token accuracy is 3.0 points below the reference's on the pooled rows"
    )
    # Not assessed: the reference predicts almost nothing on this corpus.
    weak = reports(
        "weak",
        [{"next_token_accuracy_reference": 0.0, "next_token_accuracy_candidate": 0.0}],
    )
    out = io.StringIO()
    assert (
        bench.main(attach_arguments(run, layout, weak, "stark-low", layout.q8), out=out)
        == 0
    )
    assert out.getvalue().startswith(
        "Quality result attached: not assessed (the reference predicts 0% of next tokens on this corpus)"
    )
    unreadable = layout.root / "not-json.json"
    unreadable.write_text("{")
    again = attach_arguments(run, layout, [unreadable], "stark-low", layout.q8)
    assert bench.main([*again, "--replace"]) == 1
    assert "quality_report_unreadable" in capsys.readouterr().err
    summary = bench.summarize([run])
    assert {name: item["status"] for name, item in summary["quality"].items()} == {
        "stark-base": "pass",
        "stark-q8": "fail",
        "stark-low": "not assessed",
        "ref-base": "not applicable",
    }
    assert summary["scoreboard"]["stark-base"] == {"eligible": True, "reasons": []}
    assert summary["scoreboard"]["stark-q8"]["reasons"] == ["speed only: quality fail"]
    assert summary["scoreboard"]["stark-low"]["reasons"] == [
        "speed only: quality not assessed"
    ]
    decode = summary["cells"]["decode"]
    assert decode["stark-base"]["paired_vs_reference"]["standing"] == "ahead"
    assert decode["stark-q8"]["paired_vs_reference"]["standing"] == "speed only"
    assert decode["stark-low"]["paired_vs_reference"]["standing"] == "speed only"
    assert decode["stark-q8"]["paired_vs_reference"]["median"] == pytest.approx(1.11)
    markdown = bench.render_markdown(summary)
    assert (
        "| pass | 0.5000 / 0.4945 | 0.9700; p>=0.5: 0.9900 of 600 | 0.00400 / 0.31000 | yes |"
        in markdown
    )
    assert (
        "| fail (next-token accuracy is 3.0 points below the reference's on the pooled rows (limit 2.0)) |"
        in markdown
    )
    assert (
        "next-token accuracy is not more than 2.0 points below the reference's"
        in markdown
    )
    assert (
        "at least 98% where the reference's top probability is at least 0.5" in markdown
    )
    # A failing result in any run of an arm fails the arm.
    later = write_run(
        layout.root,
        "r2",
        {
            "stark-base": {
                "match": "base",
                "model": layout.base,
                "repeats": blocks,
                "quality": {"accuracy": (0.42, 0.3)},
            }
        },
    )
    assert bench.summarize([run, later])["quality"]["stark-base"]["status"] == "fail"
    out = io.StringIO()
    assert bench.main(["quality", "schema"], out=out) == 0
    assert json.loads(out.getvalue())["schema"] == bench.QUALITY_SCHEMA
