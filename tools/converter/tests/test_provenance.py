"""Comprehensive inventories invalidate evidence for relevant tooling changes."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from stark_convert.provenance import SOURCE_INPUTS, toolchain_provenance


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "converter"
    for filename in (
        "stark_convert/__init__.py",
        "stark_convert/adapters/dense.py",
        "stark_convert/profiles/schema.json",
        "stark_convert/catalog.json",
        "scripts/export_conversion_catalog.py",
        "pyproject.toml",
        ".python-version",
        "LICENSE.lalamo",
        "uv.lock",
        "setup.py",
        "MANIFEST.in",
    ):
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic provenance fixture\n")
    return root


@pytest.mark.parametrize(
    "filename",
    [
        "stark_convert/adapters/dense.py",
        "stark_convert/profiles/schema.json",
        "stark_convert/catalog.json",
        "scripts/export_conversion_catalog.py",
        "pyproject.toml",
        ".python-version",
        "LICENSE.lalamo",
    ],
)
def test_nested_adapter_profile_catalog_and_exporter_changes_invalidate_sources(project, filename):
    baseline = toolchain_provenance(project)
    (project / filename).write_text("changed fixture\n")
    changed = toolchain_provenance(project)
    assert changed["converter_sources_sha256"] != baseline["converter_sources_sha256"]
    assert changed["converter_lock_sha256"] == baseline["converter_lock_sha256"]


def test_nested_addition_and_removal_change_manifest_without_filename_collisions(project):
    baseline = toolchain_provenance(project)
    first = project / "stark_convert" / "profiles" / "dense.py"
    second = project / "stark_convert" / "adapters" / "dense.py"
    first.write_bytes(second.read_bytes())
    addition = toolchain_provenance(project)
    assert addition["converter_sources_sha256"] != baseline["converter_sources_sha256"]
    second.unlink()
    removal = toolchain_provenance(project)
    assert removal["converter_sources_sha256"] != addition["converter_sources_sha256"]
    assert removal["converter_sources_sha256"] != baseline["converter_sources_sha256"]


@pytest.mark.parametrize("name", SOURCE_INPUTS)
def test_missing_build_input_fails_closed(project, name):
    (project / name).unlink()
    with pytest.raises(ValueError, match="^missing_provenance_input$"):
        toolchain_provenance(project)


def test_relocated_package_has_the_same_identity_and_rejects_capsule_changes(project, tmp_path):
    baseline = toolchain_provenance(project)
    installed = tmp_path / "site-packages"
    shutil.copytree(project / "stark_convert", installed / "stark_convert")
    capsule = installed / "stark_convert" / "_build_inputs"
    for name in (*SOURCE_INPUTS, "uv.lock"):
        path = capsule / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / name, path)
    assert toolchain_provenance(installed) == baseline
    cache = capsule / "__pycache__"
    cache.mkdir()
    (cache / "setup.cpython-313.pyc").write_bytes(b"interpreter cache")
    assert toolchain_provenance(installed) == baseline
    (capsule / "setup.py").write_text("changed package build\n")
    assert toolchain_provenance(installed) != baseline
    (capsule / "unexpected.txt").write_text("unaccounted input\n")
    with pytest.raises(ValueError, match="^invalid_packaged_provenance$"):
        toolchain_provenance(installed)


def test_dependency_lock_hash_is_explicit_and_invalidates_evidence(project):
    baseline = toolchain_provenance(project)
    (project / "uv.lock").write_text("changed dependency set\n")
    changed = toolchain_provenance(project)
    assert changed != baseline
    assert changed["converter_lock_sha256"] != baseline["converter_lock_sha256"]


def test_stale_capsule_cannot_hide_changed_checkout_inputs(project):
    (project / "stark_convert" / "_build_inputs").mkdir()
    with pytest.raises(ValueError, match="^ambiguous_packaged_provenance$"):
        toolchain_provenance(project)


def test_runtime_python_caches_are_not_conversion_inputs(project):
    baseline = toolchain_provenance(project)
    cache = project / "stark_convert" / "adapters" / "__pycache__"
    cache.mkdir()
    (cache / "dense.pyc").write_bytes(b"synthetic interpreter cache")
    assert toolchain_provenance(project) == baseline


def test_unaccounted_resource_fails_instead_of_omitting_input(project):
    (project / "stark_convert" / "new_resource.dat").write_bytes(b"unaccounted resource")
    with pytest.raises(ValueError, match="^unaccounted_package_resource$"):
        toolchain_provenance(project)


def test_all_conversion_imports_are_independent_of_heavy_reference_frameworks():
    modules = (
        "stark_convert.cli",
        "stark_convert.metadata",
        "stark_convert.qwen3",
        "stark_convert.adapters.llama",
        "stark_convert.adapters.mistral",
        "stark_convert.adapters.qwen2",
        "stark_convert.sources",
        "stark_convert.registry",
        "stark_convert.benchmark",
    )
    code = (
        "import importlib, sys; "
        f"[importlib.import_module(name) for name in {modules!r}]; "
        "assert not any(name.split('.')[0] in {'lalamo','jax','torch'} for name in sys.modules)"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["HF_HUB_OFFLINE"] = "1"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env)
    assert result.returncode == 0
