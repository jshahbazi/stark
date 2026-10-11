"""Fallback protocol and local checkpoint selection require no model load."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from stark_convert import lalamo_runner


def test_checkpoint_index_selects_only_referenced_shards(tmp_path):
    for name in ("model-01.safetensors", "model-02.safetensors", "unrelated.safetensors"):
        (tmp_path / name).touch()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-02.safetensors",
                    "b": "model-01.safetensors",
                    "c": "model-02.safetensors",
                }
            }
        )
    )
    assert lalamo_runner._weight_files(tmp_path) == (("model-01.safetensors", "model-02.safetensors"), ".safetensors")


@pytest.mark.parametrize("name", ["../outside.safetensors", "/absolute.safetensors", "model.bin"])
def test_checkpoint_index_rejects_traversal_and_wrong_weight_format(tmp_path, name):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": name}}))
    with pytest.raises(ValueError):
        lalamo_runner._weight_files(tmp_path)


def test_checkpoint_index_requires_all_referenced_shards(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "missing.safetensors",
                }
            }
        )
    )
    with pytest.raises(FileNotFoundError):
        lalamo_runner._weight_files(tmp_path)


@pytest.mark.parametrize("extension", [".bin", ".pth", ".pt"])
def test_fallback_can_discover_registered_torch_checkpoint_formats(tmp_path, extension):
    (tmp_path / f"model{extension}").touch()
    assert lalamo_runner._weight_files(tmp_path) == ((f"model{extension}",), extension)


def test_native_and_python_diagnostics_are_suppressed():
    child = (
        "import os,sys; from stark_convert.lalamo_runner import _quiet_native_output; "
        "guard=_quiet_native_output(); guard.__enter__(); "
        "print('python private detail'); print('stderr private detail',file=sys.stderr); "
        "os.write(1,b'native private detail'); os.write(2,b'native stderr private detail'); "
        "guard.__exit__(None,None,None); print('aggregate result')"
    )
    result = subprocess.run([sys.executable, "-c", child], capture_output=True, check=True)
    assert result.stdout == b"aggregate result\n"
    assert result.stderr == b""


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (ValueError("private detail"), "invalid_model"),
        (ImportError("private detail"), "dependency_error"),
        (RuntimeError("private detail"), "conversion_failed"),
        (FileNotFoundError("private detail"), "missing_source_file"),
        (lalamo_runner.UnsupportedModelError("private detail"), "unsupported_model"),
    ],
)
def test_error_categories_never_contain_exception_text(error, category):
    assert lalamo_runner._error_category(error) == category


def fake_hub(tmp_path):
    calls = []
    sha = "a" * 40

    def model_info(*, repo_id):
        calls.append(("info", repo_id))
        return SimpleNamespace(sha=sha)

    def download(*, repo_id, filename, revision, local_files_only):
        calls.append(("download", repo_id, filename, revision, local_files_only))
        destination = tmp_path / "external" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("external repository content")
        return destination

    return SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=model_info), hf_hub_download=download), calls, sha


def test_cross_repository_files_use_their_pinned_repository_even_when_primary_name_exists(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    (primary / "tokenizer.json").write_text("wrong primary content")
    hub, calls, sha = fake_hub(tmp_path)
    resolver = lalamo_runner.PinnedSnapshotResolver("primary/repo", primary, allow_downloads=True, hub=hub)
    external = resolver.resolve_file("primary/repo", "tokenizer.json", file_repo="external/repo")
    assert external.read_text() == "external repository content"
    assert external != primary / "tokenizer.json"
    resolver.resolve_file("primary/repo", "tokenizer_config.json", file_repo="external/repo")
    assert calls == [
        ("info", "external/repo"),
        ("download", "external/repo", "tokenizer.json", sha, False),
        ("download", "external/repo", "tokenizer_config.json", sha, False),
    ]
    assert resolver.provenance() == {
        "sources": [
            {"repo": "external/repo", "revision": sha},
            {"repo": "primary/repo", "revision": None},
        ]
    }


def test_local_primary_files_remain_authoritative_and_never_download_missing_files(tmp_path):
    (tmp_path / "config.json").write_text("primary content")
    hub, calls, _ = fake_hub(tmp_path)
    resolver = lalamo_runner.PinnedSnapshotResolver("primary/repo", tmp_path, allow_downloads=True, hub=hub)
    assert resolver.resolve_file("primary/repo", "config.json") == tmp_path / "config.json"
    with pytest.raises(FileNotFoundError):
        resolver.resolve_file("primary/repo", "missing.json")
    assert calls == []


def test_offline_external_dependencies_cannot_use_same_named_primary_file(tmp_path):
    (tmp_path / "tokenizer.json").write_text("wrong primary content")
    hub, calls, _ = fake_hub(tmp_path)
    resolver = lalamo_runner.PinnedSnapshotResolver("primary/repo", tmp_path, hub=hub)
    with pytest.raises(lalamo_runner.OfflineDependencyError):
        resolver.resolve_file("primary/repo", "tokenizer.json", file_repo="external/repo")
    assert calls == []
    assert (
        lalamo_runner._error_category(lalamo_runner.OfflineDependencyError()) == "external_source_unavailable_offline"
    )


def test_origin_helper_adapter_routes_dependencies_and_restores_upstream_helpers(tmp_path):
    hub, calls, sha = fake_hub(tmp_path)
    resolver = lalamo_runner.PinnedSnapshotResolver("primary/repo", tmp_path, allow_downloads=True, hub=hub)
    old_file, old_weights = object(), object()
    origins = SimpleNamespace(
        hf_resolve_file=old_file,
        hf_resolve_weights=old_weights,
        report_status=lambda callback, event: callback(event) if callback else None,
        DownloadingFileEvent=lambda spec: ("started", spec.filename),
        FinishedDownloadingFileEvent=lambda spec: ("finished", spec.filename),
        FileSpec=lambda filename: SimpleNamespace(filename=filename, repo=None),
    )
    events = []
    with lalamo_runner._patched_origins(origins, resolver):
        result = origins.hf_resolve_file(
            "primary/repo",
            SimpleNamespace(filename="tokenizer.json", repo="external/repo"),
            events.append,
        )
        assert result == tmp_path / "external" / "tokenizer.json"
        assert origins.hf_resolve_file is not old_file
    assert origins.hf_resolve_file is old_file
    assert origins.hf_resolve_weights is old_weights
    assert events == [("started", "tokenizer.json"), ("finished", "tokenizer.json")]
    assert calls[-1] == ("download", "external/repo", "tokenizer.json", sha, False)


def test_registered_nemo_archive_resolution_stays_local(tmp_path):
    nested = tmp_path / "artifacts"
    nested.mkdir()
    (nested / "model.nemo").touch()
    hub, calls, _ = fake_hub(tmp_path)
    resolver = lalamo_runner.PinnedSnapshotResolver("primary/repo", tmp_path, hub=hub)
    assert resolver.weight_filenames("primary/repo", ".nemo") == ("artifacts/model.nemo",)
    assert resolver.resolve_file("primary/repo", "artifacts/model.nemo") == nested / "model.nemo"
    assert calls == []


def test_primary_weight_selection_keeps_registered_container_format(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "model.pth").touch()
    hub, _, _ = fake_hub(tmp_path)
    resolver = lalamo_runner.PinnedSnapshotResolver("primary/repo", tmp_path, hub=hub)
    assert resolver.weight_filenames("primary/repo", ".safetensors") == ("model.safetensors",)
    assert resolver.weight_filenames("primary/repo", ".pth") == ("model.pth",)


def test_primary_revision_must_be_immutable(tmp_path):
    hub, _, _ = fake_hub(tmp_path)
    with pytest.raises(ValueError):
        lalamo_runner.PinnedSnapshotResolver("primary/repo", tmp_path, source_revision="main", hub=hub)
