"""End-to-end synthetic conversion and fail-closed orchestration contracts."""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest
from test_dense_adapters import dense_fixture
from test_gemma import gemma_fixture
from test_granite_smollm import fixture as scalar_family_fixture
from test_mlx import mlx_qwen_fixture
from test_mlx_dense import mlx_dense_fixture
from test_mlx_gemma import mlx_gemma_fixture
from test_phi import phi_fixture
from test_qualifications import evidence_record
from test_sources import Hub
from test_streaming import qwen_fixture
from tokenizers import Tokenizer, models

from stark_convert import cli, metadata, qualifications
from stark_convert.contracts import ConversionRequest
from stark_convert.provenance import toolchain_provenance
from stark_convert.registry import get_model
from stark_convert.registry import models as catalog_models
from stark_convert.sources import resolve_source
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header

REPOS = {
    "qwen3": "Qwen/Qwen3-0.6B",
    "qwen2": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2": "google/gemma-2-2b-it",
    "gemma3_text": "google/gemma-3-1b-it",
    "phi3": "microsoft/Phi-4-mini-instruct",
    "granite": "ibm-granite/granite-3.3-2b-instruct",
    "smollm3": "HuggingFaceTB/SmolLM3-3B",
}


def source_fixture(path: Path, family: str = "qwen3", *, sharded=False):
    if family == "qwen3":
        qwen_fixture(path, sharded=sharded)
    elif family in {"gemma2", "gemma3_text"}:
        gemma_fixture(path, family, sharded=sharded)
    elif family == "phi3":
        phi_fixture(path, sharded=sharded)
    elif family in {"granite", "smollm3"}:
        scalar_family_fixture(path, case=family, sharded=sharded)
    else:
        dense_fixture(path, family, sharded=sharded)
    tokenizer = Tokenizer(
        models.WordLevel({f"[SYNTHETIC_{index}]": index for index in range(16)}, unk_token="[SYNTHETIC_0]")
    )
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": None if family == "smollm3" else "{{ messages }}"})
    )
    if family == "smollm3":
        (path / "chat_template.jinja").write_text("{{ messages }}")
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": 1, "temperature": 0.6}))
    return path


@pytest.fixture
def source(tmp_path):
    return source_fixture(tmp_path / "source")


@pytest.fixture(autouse=True)
def source_tooling(tmp_path, monkeypatch):
    # Bind orchestration to a real, isolated source inventory so concurrent
    # development cannot invalidate this synthetic test's receipt.
    project = tmp_path / "converter-tooling"
    (project / "stark_convert" / "adapters").mkdir(parents=True)
    (project / "stark_convert" / "adapters" / "fixture.py").write_text("# synthetic source\n")
    (project / "uv.lock").write_text("# synthetic lock\n")
    (project / "pyproject.toml").write_text("# synthetic project\n")
    (project / "stark_convert" / "native_qualifications.json").write_text(
        json.dumps({"schema_version": 1, "records": []})
    )
    monkeypatch.setattr(cli, "toolchain_provenance", lambda: toolchain_provenance(project))
    monkeypatch.setattr(qualifications, "files", lambda package: project / "stark_convert")
    return project


def run(source, output, *, backend="auto", repo=REPOS["qwen3"], **kwargs):
    return cli.convert(source, output, repo=repo, dtype="bfloat16", backend=backend, chunk_bytes=4096, **kwargs)


@pytest.mark.parametrize("backend", ["auto", "streaming"])
@pytest.mark.parametrize("family", REPOS)
def test_complete_conversion_publishes_all_outputs_and_bound_receipt(tmp_path, family, backend):
    source = source_fixture(tmp_path / "source", family, sharded=True)
    output = tmp_path / "converted"
    receipt = run(source, output, repo=REPOS[family], backend=backend)
    assert {path.name for path in output.iterdir()} == {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "conversion.json",
    }
    assert json.loads((output / "conversion.json").read_text()) == receipt
    _, tensors = read_safetensors_header(output / "model.safetensors")
    assert len(tensors) == receipt["tensor_count"] > 0
    assert receipt["backend"] == "streaming"
    assert "fallback" not in receipt
    assert receipt["capability"]["source_validation"] == "complete"
    assert receipt["capability"]["execution"] == "unverified"
    assert receipt["capability"]["native_loading"] == "supported_layout_unverified_tuple"
    assert receipt["native_qualification"] is None
    assert receipt["transform_peak_bytes"] <= 4096
    assert receipt["output_sha256"] == cli.file_hash(output / "model.safetensors")
    assert receipt["config_sha256"] == cli.file_hash(output / "config.json")
    assert receipt["tokenizer_sha256"] == cli.file_hash(output / "tokenizer.json")
    assert str(tmp_path) not in json.dumps(receipt)
    assert not list(tmp_path.glob(".stark-convert-*"))
    assert output.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("backend", ["auto", "streaming"])
def test_unsupported_layout_never_invokes_an_external_backend(source, tmp_path, backend):
    config = json.loads((source / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "unsupported"}
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(UnsupportedConversion):
        run(source, tmp_path / "output", backend=backend)
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".stark-convert-*"))
    assert not hasattr(cli, "_lalamo")


@pytest.mark.parametrize("repo", ["unknown/model", "Qwen/Qwen3.5-35B-A3B"])
def test_unsupported_catalog_tuple_is_rejected_before_resolution(tmp_path, monkeypatch, repo):
    monkeypatch.setattr(cli, "resolve_source", lambda *args, **kwargs: pytest.fail("source resolution occurred"))
    with pytest.raises(UnsupportedConversion):
        run(Path("does-not-exist"), tmp_path / "output", repo=repo)
    assert not (tmp_path / "output").exists()


MLX_PROFILES = {
    "qwen3": "Qwen/Qwen3-0.6B-MLX",
    "llama": "mlx-community/Llama-3.2-1B-Instruct",
    "smollm3": "mlx-community/SmolLM3-3B",
    "gemma3_text": "mlx-community/gemma-3-1b-it",
}


def mlx_snapshot(tmp_path, *, bits=4, profile_bits=None, family="qwen3"):
    primary = tmp_path / "primary"
    if family == "qwen3":
        mlx_qwen_fixture(primary, bits=bits, sharded=True)
    elif family == "gemma3_text":
        mlx_gemma_fixture(primary, bits=bits, sharded=True)
    else:
        mlx_dense_fixture(primary, family, bits=bits, sharded=True)
    external = source_fixture(tmp_path / "external")
    (external / "chat_template.jinja").write_text("{{ messages }}")
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        shutil.copyfile(external / name, primary / name)
    if family == "smollm3":
        (primary / "tokenizer_config.json").write_text(json.dumps({"chat_template": None}))
    repo = f"{MLX_PROFILES[family]}-{bits if profile_bits is None else profile_bits}bit"
    repositories = {rule["repo"]: external for rule in get_model(repo)["files"].values() if rule["repo"] != repo}
    hub = Hub({**repositories, repo: primary})
    return resolve_source(repo, hub=hub), hub


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("backend", ["auto", "streaming"])
@pytest.mark.parametrize("family", MLX_PROFILES)
def test_mlx_profile_complete_conversion_and_offline_external_replay(tmp_path, bits, backend, family):
    snapshot, hub = mlx_snapshot(tmp_path, bits=bits, family=family)
    assert {event[1] for event in hub.events if event[0] == "pin"} == {snapshot.repo} | {
        rule["repo"] for rule in get_model(snapshot.repo)["files"].values()
    }
    manifest = tmp_path / "source-manifest.json"
    snapshot.write_manifest(manifest)
    replay = resolve_source(snapshot.repo, source_manifest=manifest, offline=True, hub=object())
    receipt = run(replay, tmp_path / "output", repo=replay.repo, backend=backend)
    assert receipt["backend"] == "streaming" and receipt["native_qualification"] is None
    assert receipt["capability"]["source_validation"] == "complete"
    assert receipt["capability"]["native_loading"] == "supported_layout_unverified_tuple"
    assert receipt["transform_peak_bytes"] <= 4096
    specs, tensors = read_safetensors_header(tmp_path / "output" / "model.safetensors")
    assert any(info.dtype == "U8" for info in tensors.values())
    assert {json.loads(value)["bits"] for value in specs.values() if "MLXSpec" in value} == {bits}
    assert str(tmp_path) not in json.dumps(receipt)


@pytest.mark.parametrize("bits,profile_bits", [(4, 8), (8, 4)])
@pytest.mark.parametrize("family", MLX_PROFILES)
def test_mlx_catalog_bit_mismatch_fails_before_staging(tmp_path, bits, profile_bits, family):
    snapshot, _ = mlx_snapshot(tmp_path, bits=bits, profile_bits=profile_bits, family=family)
    with pytest.raises(ConversionError, match="packing disagrees"):
        run(snapshot, tmp_path / "output", repo=snapshot.repo)
    assert not (tmp_path / "output").exists() and not list(tmp_path.glob(".stark-convert-*"))


def test_dense_catalog_profile_rejects_compressed_checkpoint(tmp_path):
    snapshot, _ = mlx_snapshot(tmp_path)
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        shutil.copyfile(tmp_path / "external" / name, snapshot.root / name)
    with pytest.raises(UnsupportedConversion, match="registered compressed profile"):
        run(snapshot.root, tmp_path / "output", repo=REPOS["qwen3"])
    assert not (tmp_path / "output").exists() and not list(tmp_path.glob(".stark-convert-*"))


def test_mlx_parameter_writer_failure_removes_private_staging(tmp_path, monkeypatch):
    from stark_convert import compressed

    snapshot, _ = mlx_snapshot(tmp_path)

    def fail(op, output, chunk_bytes):
        assert output.tell() > 0
        raise ConversionError("synthetic_plane_failure")

    monkeypatch.setattr(compressed, "write_mlx_params", fail)
    with pytest.raises(ConversionError, match="synthetic_plane_failure"):
        run(snapshot, tmp_path / "output", repo=snapshot.repo)
    assert not (tmp_path / "output").exists() and not list(tmp_path.glob(".stark-convert-*"))


def test_capability_probe_never_reads_source_or_claims_validation(monkeypatch, capsys):
    monkeypatch.setattr(cli, "resolve_source", lambda *args, **kwargs: pytest.fail("capability read a source"))
    monkeypatch.setattr("sys.argv", ["stark-convert", "--repo", REPOS["qwen3"], "--capabilities"])
    assert cli.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["capability"]["conversion"] == "implemented"
    assert payload["capability"]["source_validation"] == "not_performed"
    assert payload["capability"]["execution"] == "unverified"
    assert payload["capability"]["native_loading"] != "qualified_tuple"


def test_capability_reports_actual_unsupported_model_kind():
    repo = next(entry["repo"] for entry in catalog_models() if entry["model_kind"] == "ClassifierModelSpec")
    result = asdict(cli.capability(ConversionRequest(repo)))
    assert result["model_kind"] == "classifier"
    assert result["conversion"] == "unsupported" and result["source_validation"] == "not_performed"


def test_native_requirement_does_not_accept_architecture_probe(source, tmp_path):
    with pytest.raises(UnsupportedConversion):
        run(source, tmp_path / "output", require_native_support=True)
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".stark-convert-*"))


def store_qualification(source, tmp_path, source_tooling):
    baseline = run(source, tmp_path / "baseline")
    record = evidence_record()
    for field in ("source_manifest_sha256", "output_sha256", "config_sha256", "tokenizer_sha256"):
        record[field] = baseline[field]
    path = source_tooling / "stark_convert" / "native_qualifications.json"
    path.write_text(json.dumps({"schema_version": 1, "records": [record]}))
    return record, path


@pytest.mark.parametrize("strict", [False, True])
def test_exact_named_build_qualification_is_in_published_receipt(source, tmp_path, source_tooling, strict):
    record, _ = store_qualification(source, tmp_path, source_tooling)
    output = tmp_path / "qualified"
    receipt = run(source, output, require_native_support=strict)
    assert receipt["native_qualification"] == record
    assert receipt["capability"]["native_loading"] == "qualified_tuple"
    assert receipt["capability"]["source_validation"] == "complete"
    assert receipt["capability"]["execution"] == "unverified"
    assert receipt["native_qualification"]["native_build"]["native_score_calls"] == 0
    assert json.loads((output / "conversion.json").read_text()) == receipt
    assert receipt["output_sha256"] == cli.file_hash(output / "model.safetensors")
    assert str(tmp_path) not in json.dumps(receipt)
    # The source-free catalog probe still cannot claim this qualification.
    assert cli.capability(ConversionRequest(REPOS["qwen3"])).native_loading != "qualified_tuple"


@pytest.mark.parametrize("field", ["source_manifest_sha256", "output_sha256", "config_sha256", "tokenizer_sha256"])
def test_strict_native_requirement_rejects_any_artifact_hash_mismatch(source, tmp_path, source_tooling, field):
    record, path = store_qualification(source, tmp_path, source_tooling)
    record[field] = "9" * 64
    path.write_text(json.dumps({"schema_version": 1, "records": [record]}))
    output = tmp_path / "unqualified"
    with pytest.raises(UnsupportedConversion, match="No recorded named-build native qualification"):
        run(source, output, require_native_support=True)
    assert not output.exists()
    assert not list(tmp_path.glob(".stark-convert-*"))


def test_qualification_mutation_during_lookup_invalidates_publication(source, tmp_path, source_tooling, monkeypatch):
    _, path = store_qualification(source, tmp_path, source_tooling)
    lookup = qualifications.qualification_for

    def changed(repo, dtype, hashes):
        result = lookup(repo, dtype, hashes)
        path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(qualifications, "qualification_for", changed)
    output = tmp_path / "output"
    with pytest.raises(ConversionError, match="^Converter tooling changed during qualification$"):
        run(source, output, require_native_support=True)
    assert not output.exists()
    assert not list(tmp_path.glob(".stark-convert-*"))


@pytest.mark.parametrize("strict", [False, True])
def test_corrupt_qualification_resource_fails_closed(source, tmp_path, source_tooling, strict):
    (source_tooling / "stark_convert" / "native_qualifications.json").write_text("{}")
    with pytest.raises(ConversionError, match="Invalid native qualification catalog"):
        run(source, tmp_path / "output", require_native_support=strict)
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".stark-convert-*"))


def test_malformed_checkpoint_does_not_publish(source, tmp_path):
    (source / "model.safetensors").write_bytes(b"invalid")
    with pytest.raises(ConversionError):
        run(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".stark-convert-*"))


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_existing_output_is_preserved(source, tmp_path, kind):
    output = tmp_path / "converted"
    if kind == "directory":
        output.mkdir()
        (output / "sentinel").write_bytes(b"keep")
    elif kind == "file":
        output.write_bytes(b"keep")
    else:
        output.symlink_to(source / "config.json")
    with pytest.raises(FileExistsError):
        run(source, output)
    if kind == "directory":
        assert (output / "sentinel").read_bytes() == b"keep"
    elif kind == "file":
        assert output.read_bytes() == b"keep"
    else:
        assert output.is_symlink()
    assert not list(tmp_path.glob(".stark-convert-*"))


def test_publication_race_cannot_replace_even_an_empty_directory(source, tmp_path, monkeypatch):
    publish = cli._publish
    output = tmp_path / "output"

    def occupied(staging, target):
        target.mkdir()
        publish(staging, target)

    monkeypatch.setattr(cli, "_publish", occupied)
    with pytest.raises(FileExistsError):
        run(source, output)
    assert output.is_dir() and not list(output.iterdir())
    assert not list(tmp_path.glob(".stark-convert-*"))


def test_partial_writer_failure_is_cleaned(source, tmp_path, monkeypatch):
    def fail(plan, path, **kwargs):
        path.write_bytes(b"partial")
        raise RuntimeError("private checkpoint data")

    monkeypatch.setattr(cli, "write_plan", fail)
    with pytest.raises(RuntimeError):
        run(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".stark-convert-*"))


def test_initial_provenance_failure_cleans_staging(source, tmp_path, monkeypatch):
    def failed_provenance():
        raise ValueError("unaccounted_package_resource")

    monkeypatch.setattr(cli, "toolchain_provenance", failed_provenance)
    with pytest.raises(ValueError):
        run(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".stark-convert-*"))


def test_missing_output_file_is_not_published(source, tmp_path, monkeypatch):
    monkeypatch.setattr(metadata, "write_metadata", lambda *args, **kwargs: {})
    with pytest.raises(ConversionError, match="incomplete_output"):
        run(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("mutation", ["metadata", "weights", "tooling"])
def test_source_or_tool_mutation_prevents_publication(source, tmp_path, source_tooling, monkeypatch, mutation):
    writer = cli.write_plan

    def changed(plan, path, **kwargs):
        stats = writer(plan, path, **kwargs)
        if mutation == "tooling":
            (source_tooling / "stark_convert" / "adapters" / "new_adapter.py").write_text("# change\n")
        elif mutation == "metadata":
            (source / "tokenizer_config.json").write_text("{}")
        else:
            (source / "model.safetensors").write_bytes(b"changed")
        return stats

    monkeypatch.setattr(cli, "write_plan", changed)
    with pytest.raises(ConversionError):
        run(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".stark-convert-*"))


@pytest.mark.parametrize(
    "error,code,category",
    [
        (RuntimeError("sensitive exception details"), 1, "RuntimeError"),
        (KeyboardInterrupt("private interruption"), 130, "interrupted"),
    ],
)
def test_cli_emits_only_fixed_error_category(source, tmp_path, monkeypatch, capsys, error, code, category):
    monkeypatch.setattr(
        "sys.argv", ["stark-convert", str(source), "--repo", REPOS["qwen3"], "--output", str(tmp_path / "output")]
    )

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "convert", fail)
    assert cli.main() == code
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"status": "error", "error_category": category}
    assert "private" not in captured.err and "sensitive" not in captured.err and str(source) not in captured.err


def test_removed_external_backend_argument_does_not_echo_private_values(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["stark-convert", "private/source", "--backend", "lalamo"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"status": "error", "error_category": "invalid_arguments"}


def test_offline_replay_publishes_identical_model_artifacts(source, tmp_path):
    snapshot = resolve_source(source, repo=REPOS["qwen3"])
    manifest = tmp_path / "private-manifest.json"
    snapshot.write_manifest(manifest)
    first = run(snapshot, tmp_path / "first")
    replay = resolve_source(REPOS["qwen3"], source_manifest=manifest, offline=True, hub=object())
    second = run(replay, tmp_path / "second")
    for field in ("output_sha256", "config_sha256", "tokenizer_sha256", "source_manifest_sha256"):
        assert first[field] == second[field]
    assert get_model(replay.repo)["metadata_profile"] == "qwen3"
