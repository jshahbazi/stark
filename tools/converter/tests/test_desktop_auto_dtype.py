"""The app install route converts with `--dtype auto` and keeps BF16 elsewhere.

The local cases run the real preflight and converter subprocesses on synthetic
sources under a temporary data directory. No model is loaded.
"""

import io
import json
from pathlib import Path

import pytest
from test_auto_dtype import source
from test_desktop_model_support import support_report, text_artifact

from stark_convert.desktop import Desktop
from stark_convert.streaming import read_safetensors_header


@pytest.fixture
def host(tmp_path):
    controller = Desktop(tmp_path / "worker", tmp_path / "data", io.StringIO(), timeout=2)
    yield controller
    controller.stop(quiet=True)


def recording(host):
    """Run the real subprocesses, keeping each command and preflight report."""
    calls = []
    run_support_check, run_converter = host.run_support_check, host.run_converter

    def support(command, environment, epoch):
        report = run_support_check(command, environment, epoch)
        calls.append(("preflight", command, report))
        return report

    def convert(command, environment, epoch):
        result = run_converter(command, environment, epoch)
        calls.append(("convert", command, result))
        return result

    host.run_support_check, host.run_converter = support, convert
    return calls


def option(command, name):
    return command[command.index(name) + 1] if name in command else None


@pytest.mark.parametrize(
    "case,mode,selected,reason",
    [
        ("qwen2-f16", "text", "float16", "all_f16_mlx_affine_source"),
        ("mistral-f16", "decision", "float16", "all_f16_mlx_affine_source"),
        ("qwen2-bf16", "text", "bfloat16", "source_not_all_f16"),
        ("qwen2-mixed", "text", "bfloat16", "source_not_all_f16"),
        ("qwen2-window", "text", "bfloat16", "native_f16_not_admitted"),
        ("qwen3-f16", "text", "bfloat16", "family_not_qualified_for_f16"),
        ("llama-f16", "text", "bfloat16", "family_not_qualified_for_f16"),
    ],
)
def test_local_install_selects_dtype_from_downloaded_tensors(tmp_path, host, case, mode, selected, reason):
    path = source(tmp_path, case)
    calls = recording(host)
    host.install(host.epoch, "local", str(path), None, mode, "fixture-request")
    (_, preflight, report), (_, command, result) = calls
    # Metadata preflight is unchanged: no dtype, BF16 admission.
    assert option(preflight, "--dtype") is None and report["status"] == "conditional"
    assert option(command, "--dtype") == "auto" and option(command, "--support-mode") == mode
    assert result[0] == 0 and len(host.models) == 1
    installed = Path(host.models[0]["path"])
    assert installed.parent == host.model_dir and str(installed).startswith(str(tmp_path / "data"))
    receipt = json.loads((installed / "conversion.json").read_text())
    assert receipt["dtype_selection"] == {"requested": "auto", "selected": selected, "reason": reason}
    _, tensors = read_safetensors_header(installed / "model.safetensors")
    floating = {info.dtype for info in tensors.values() if info.dtype not in {"U8", "I8", "U32", "I32"}}
    assert floating == ({"F16", "F32"} if selected == "float16" else {"BF16", "F32"})


@pytest.mark.parametrize("mode", ["text", "decision", "vision-language"])
def test_hugging_face_install_command_requests_auto_only_for_text_artifacts(host, mode):
    commands = []

    def preflight(command, _environment, _epoch):
        commands.append(command)
        return support_report(mode=mode)

    def convert(command, _environment, _epoch):
        commands.append(command)
        text_artifact(Path(option(command, "--output")))
        return 0

    host.run_support_check, host.run_converter = preflight, convert
    host.check_family_install = lambda *_args: None
    host.install(
        host.epoch, "huggingface", "example/model", "vision-language" if mode == "vision-language" else None, mode, "r"
    )
    preflight_command, convert_command = commands
    assert "--dtype" not in preflight_command
    assert option(convert_command, "--dtype") == (None if mode == "vision-language" else "auto")
    assert "--require-native-support" not in convert_command
