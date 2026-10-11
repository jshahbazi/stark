"""Exercise backend selection, private failure handling, and atomic publication."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stark_convert import cli, metadata, qwen3, streaming
from stark_convert.streaming import ConversionError, UnsupportedConversion


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source"
    path.mkdir()
    (path / "config.json").write_text("{}")
    return path


def write_outputs(output: Path):
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (output / name).write_bytes(b"{}")
    return {"status": "ok", "tensor_count": 2}


def run(source, output, backend="auto"):
    return cli.convert(source, output, repo="Qwen/Qwen3-0.6B", dtype="bfloat16", backend=backend, chunk_bytes=4096)


def test_unsupported_plan_uses_lalamo_and_publishes_receipt(source, tmp_path, monkeypatch):
    def unsupported(*args, **kwargs):
        raise UnsupportedConversion("unsupported")

    monkeypatch.setattr(qwen3, "build_qwen3_plan", unsupported)
    monkeypatch.setattr(cli, "_lalamo", lambda source, output, repo, dtype: write_outputs(output))
    output = tmp_path / "converted"
    result = run(source, output)
    assert result["backend"] == "lalamo" and result["fallback"] is True
    assert result["tensor_count"] == 2
    assert json.loads((output / "conversion.json").read_text())["output_sha256"] == cli.file_hash(
        output / "model.safetensors"
    )
    assert not list(tmp_path.glob(".stark-convert-*"))


def test_malformed_input_never_falls_back(source, tmp_path, monkeypatch):
    def malformed(*args, **kwargs):
        raise ConversionError("malformed")

    monkeypatch.setattr(qwen3, "build_qwen3_plan", malformed)
    monkeypatch.setattr(cli, "_lalamo", lambda *args: pytest.fail("invalid input fell back"))
    output = tmp_path / "converted"
    with pytest.raises(ConversionError):
        run(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".stark-convert-*"))


def test_unsupported_metadata_cleans_staging_before_fallback(source, tmp_path, monkeypatch):
    monkeypatch.setattr(qwen3, "build_qwen3_plan", lambda *args, **kwargs: SimpleNamespace(decoder_config={}))

    def unsupported(source, output, config):
        (output / "partial.json").write_text("{}")
        raise metadata.UnsupportedMetadata("unsupported")

    def fallback(source, output, repo, dtype):
        assert list(output.iterdir()) == []
        return write_outputs(output)

    monkeypatch.setattr(metadata, "write_metadata", unsupported)
    monkeypatch.setattr(cli, "_lalamo", fallback)
    result = run(source, tmp_path / "converted")
    assert result["backend"] == "lalamo"


def test_explicit_streaming_rejects_unsupported_without_fallback(source, tmp_path, monkeypatch):
    def unsupported(*args, **kwargs):
        raise UnsupportedConversion("unsupported")

    monkeypatch.setattr(qwen3, "build_qwen3_plan", unsupported)
    monkeypatch.setattr(cli, "_lalamo", lambda *args: pytest.fail("explicit streaming fell back"))
    with pytest.raises(UnsupportedConversion):
        run(source, tmp_path / "converted", "streaming")


def test_failure_keeps_existing_output_and_cleans_partial(source, tmp_path, monkeypatch):
    output = tmp_path / "converted"
    output.mkdir()
    (output / "existing").write_text("sentinel")
    with pytest.raises(FileExistsError):
        run(source, output)
    assert (output / "existing").read_text() == "sentinel"
    output = tmp_path / "failed"

    def fail(source, output, repo, dtype):
        (output / "model.safetensors").write_bytes(b"partial")
        raise RuntimeError("sensitive exception details")

    monkeypatch.setattr(cli, "_lalamo", fail)
    with pytest.raises(RuntimeError):
        run(source, output, "lalamo")
    assert not output.exists()
    assert not list(tmp_path.glob(".stark-convert-*"))


def test_cli_suppresses_exception_details(source, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["stark-convert", str(source), "--output", str(tmp_path / "output")])

    def fail(*args, **kwargs):
        raise RuntimeError("sensitive exception details")

    monkeypatch.setattr(cli, "convert", fail)
    assert cli.main() == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {"status": "error", "error_category": "RuntimeError"}


def test_atomic_publish_refuses_empty_existing_directory(tmp_path):
    staging, target = tmp_path / "staging", tmp_path / "target"
    staging.mkdir()
    target.mkdir()
    (staging / "complete").write_text("{}")
    with pytest.raises(FileExistsError):
        cli._publish(staging, target)
    assert staging.exists() and target.exists()
    assert not list(target.iterdir())


def test_changed_metadata_prevents_publication(source, tmp_path, monkeypatch):
    def fallback(source, output, repo, dtype):
        (source / "config.json").write_text('{"changed":true}')
        return write_outputs(output)

    monkeypatch.setattr(cli, "_lalamo", fallback)
    output = tmp_path / "converted"
    with pytest.raises(ConversionError):
        run(source, output, "lalamo")
    assert not output.exists()
    assert not list(tmp_path.glob(".stark-convert-*"))


def test_streaming_uses_the_same_plan_for_config_and_weights(source, tmp_path, monkeypatch):
    plan = SimpleNamespace(decoder_config={"fixture": True})
    planned = []

    def build(*args, **kwargs):
        planned.append(plan)
        return plan

    def codec(source, output, config):
        (output / "tokenizer.json").write_text("{}")
        return {"token_codec_config": {}, "generation_config": {}}

    def write(actual_plan, path, **kwargs):
        assert actual_plan is plan
        path.write_bytes(b"{}")
        return {"tensors": 2}

    monkeypatch.setattr(qwen3, "build_qwen3_plan", build)
    monkeypatch.setattr(metadata, "write_metadata", codec)
    monkeypatch.setattr(streaming, "write_plan", write)
    output = tmp_path / "converted"
    result = run(source, output)
    assert planned == [plan]
    assert result["backend"] == "streaming" and result["fallback"] is False
    assert json.loads((output / "config.json").read_text())["decoder_config"] == plan.decoder_config


@pytest.mark.parametrize(
    "reported,expected",
    [
        ("external_source_unavailable_offline", "external_source_unavailable_offline"),
        ("sensitive untrusted exception text", "conversion_failed"),
    ],
)
def test_fallback_errors_expose_only_known_categories(source, tmp_path, monkeypatch, reported, expected):
    result = SimpleNamespace(returncode=1, stdout=json.dumps({"error_category": reported}).encode())
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(cli.FallbackFailure) as caught:
        cli._lalamo(source, tmp_path / "output", "Qwen/Qwen3-0.6B", "bfloat16")
    assert caught.value.category == expected


def test_interrupt_has_no_exception_dump(source, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["stark-convert", str(source), "--output", str(tmp_path / "output")])

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "convert", interrupted)
    assert cli.main() == 130
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {"status": "error", "error_category": "interrupted"}
