"""Model-free checks of the Inkling 2 Mini default: the derived 8-bit artifact.

A synthetic original with the real tensor set stands in for the pinned
checkpoint; the pinned hashes and sizes are patched to match it. The real
preparation, derive, verification, publication and upgrade code runs.
"""

import errno
import hashlib
import json
from collections import namedtuple
from pathlib import Path

import pytest
from test_inkling_affine8_artifact import _synthetic

from stark_convert import hybrid_prepare as hybrid
from stark_convert import image_checks, inkling_affine8_artifact, inkling_prepare, native_catalog
from stark_convert.progress import converter_failure_category
from stark_convert.streaming import ConversionError

DRAFT = b"synthetic draft head bytes"
METADATA = {
    "config.json": json.dumps({"model_type": inkling_prepare.FAMILY, "text_config": {"vocab_size": 64}}).encode(),
    "chat_template.jinja": b"{{ messages }}",
    "processor_config.json": b"{}",
    "tokenizer.json": b'{"model": {}}',
    "tokenizer_config.json": b"{}",
    "generation_config.json": b"{}",
    "hf_quant_config.json": b"{}\n",
    inkling_prepare.INDEX: b'{"weight_map": {}}',
    "README.md": b"synthetic card",
}
Pinned = namedtuple("Pinned", "target original derived_target_bytes calls")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_original(root: Path, target: bytes) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / inkling_prepare.TARGET).write_bytes(target)
    (root / inkling_prepare.DRAFT).write_bytes(DRAFT)
    for name, data in METADATA.items():
        (root / name).write_bytes(data)


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    """Patch the pinned identities to a synthetic original and its derived artifact."""
    upstream, _ = _synthetic(tmp_path)
    target = upstream.read_bytes()
    upstream.unlink()
    for attribute, name in (
        ("CONFIG_SHA", "config.json"),
        ("TEMPLATE_SHA", "chat_template.jinja"),
        ("PROCESSOR_SHA", "processor_config.json"),
        ("TOKENIZER_SHA", "tokenizer.json"),
        ("QUANT_SHA", "hf_quant_config.json"),
        ("INDEX_SHA", inkling_prepare.INDEX),
    ):
        monkeypatch.setattr(inkling_prepare, attribute, _sha(METADATA[name]))
    monkeypatch.setattr(
        inkling_prepare,
        "WEIGHTS",
        {inkling_prepare.TARGET: (len(target), _sha(target)), inkling_prepare.DRAFT: (len(DRAFT), _sha(DRAFT))},
    )
    original = {**inkling_prepare.SOURCE, "weight_bytes": len(target) + len(DRAFT)}
    monkeypatch.setattr(inkling_prepare, "SOURCE", original)
    monkeypatch.setitem(hybrid.SOURCES, inkling_prepare.ORIGINAL_CHOICE, original)
    # Learn the derived target identity from the real builder, then pin it.
    probe = tmp_path / "probe"
    _write_original(probe / "original", target)
    result = inkling_affine8_artifact.build(probe / "original", probe / "derived")
    hybrid.shutil.rmtree(probe)
    monkeypatch.setattr(inkling_prepare, "AFFINE8_TARGET_SHA", result["target_sha256"])
    monkeypatch.setattr(inkling_prepare, "AFFINE8_TARGET_BYTES", result["target_bytes"])
    derived = {
        **original,
        "weight_bytes": result["target_bytes"] + len(DRAFT),
        "quantization": inkling_prepare.AFFINE8_QUANTIZATION,
        "derived_from": inkling_prepare.ORIGINAL_CHOICE,
    }
    monkeypatch.setattr(inkling_prepare, "AFFINE8_SOURCE", derived)
    monkeypatch.setitem(hybrid.SOURCES, inkling_prepare.CHOICE, derived)
    calls = []

    def download(repo, **options):
        calls.append((repo, options["revision"], tuple(options["allow_patterns"])))
        root = Path(options["local_dir"])
        if options["allow_patterns"] == inkling_prepare.METADATA:
            for name, data in METADATA.items():
                (root / name).write_bytes(data)
        else:
            (root / inkling_prepare.TARGET).write_bytes(target)
            (root / inkling_prepare.DRAFT).write_bytes(DRAFT)

    monkeypatch.setattr(hybrid, "snapshot_download", download)
    return Pinned(target, original, result["target_bytes"], calls)


def _identity(source: dict) -> dict:
    keys = ("weights_repo", "weights_revision", "architecture", "weight_bytes", "quantization")
    return dict(
        zip(("repo", "revision", "architecture", "weight_bytes", "quantization"), map(source.get, keys), strict=True)
    )


def _siblings(output: Path) -> list[str]:
    return sorted(path.name for path in output.parent.iterdir() if path.name.startswith(output.name))


def test_default_choice_downloads_the_original_and_publishes_the_verified_8bit_artifact(tmp_path, pinned):
    from stark_convert.model_inventory import model_display
    from stark_convert.vision_runtime import model_image_family

    output = tmp_path / "model"
    progress = []
    result = hybrid.prepare(inkling_prepare.CHOICE, output, progress=progress.append, artifact_mode="vision-language")
    assert result["status"] == "prepared" and result["layout"] == inkling_prepare.AFFINE8_LAYOUT
    assert result["weight_bytes"] == hybrid.SOURCES[inkling_prepare.CHOICE]["weight_bytes"]
    # The download is the pinned original at its revision, metadata and weights only.
    assert [(repo, revision) for repo, revision, _ in pinned.calls] == [
        (inkling_prepare.REPO, inkling_prepare.REVISION)
    ] * 2
    assert pinned.calls[0][2] == tuple(inkling_prepare.SOURCE["weights_patterns"])
    # Only the verified derived folder is published; the download and derive folders are gone.
    assert _siblings(output) == ["model"]
    assert not (output / "stark-download.json").exists() and not (output / "stark-derive.json").exists()
    assert (output / inkling_prepare.TARGET).stat().st_size == pinned.derived_target_bytes
    assert (output / inkling_prepare.DRAFT).read_bytes() == DRAFT
    assert (output / inkling_affine8_artifact.PROVENANCE).is_file()
    # The desktop's pinned check compares exactly these facts.
    assert hybrid.prepared_identity(output) == _identity(hybrid.SOURCES[inkling_prepare.CHOICE])
    display = model_display({"id": "test", "name": "Inkling", "path": str(output)})
    assert display["quantization"] == inkling_prepare.AFFINE8_QUANTIZATION
    assert display["weight_bytes"] == pinned.derived_target_bytes + len(DRAFT)
    config = json.loads((output / "config.json").read_text())
    assert model_image_family(config, output) == {"profile": inkling_prepare.VISION_PROFILE, "version": 1}
    assert [(event["phase"], event["state"]) for event in progress[-3:]] == [
        ("download", "complete"),
        ("conversion", "running"),
        ("conversion", "complete"),
    ]
    with pytest.raises(ValueError, match="output already exists"):
        hybrid.prepare(inkling_prepare.CHOICE, output, offline=True)


def test_original_choice_is_prepared_exactly_as_before(tmp_path, pinned):
    output = tmp_path / "model"
    result = hybrid.prepare(inkling_prepare.ORIGINAL_CHOICE, output, artifact_mode="vision-language")
    assert result["layout"] == inkling_prepare.LAYOUT
    assert result["weight_bytes"] == pinned.original["weight_bytes"]
    assert (output / inkling_prepare.TARGET).read_bytes() == pinned.target
    assert hybrid.prepared_identity(output) == _identity(pinned.original)
    assert _siblings(output) == ["model"]


def test_an_existing_download_from_before_the_change_resumes_into_the_default(tmp_path, pinned):
    # The default's staging marker names the original download, as before.
    output = tmp_path / "model"
    stage = tmp_path / "model.native-download"
    stage.mkdir()
    marker = {"choice": inkling_prepare.CHOICE, "source": hybrid.SOURCES[inkling_prepare.ORIGINAL_CHOICE]}
    (stage / "stark-download.json").write_text(json.dumps(marker, sort_keys=True))
    _write_original(stage, pinned.target)
    hybrid.prepare(inkling_prepare.CHOICE, output, offline=True)
    assert hybrid.prepared_identity(output)["quantization"] == inkling_prepare.AFFINE8_QUANTIZATION


@pytest.mark.parametrize("field", ["AFFINE8_TARGET_SHA", "AFFINE8_TARGET_BYTES"])
def test_a_derived_identity_mismatch_publishes_nothing_and_keeps_the_download(tmp_path, pinned, monkeypatch, field):
    output = tmp_path / "model"
    good = getattr(inkling_prepare, field)
    monkeypatch.setattr(inkling_prepare, field, "0" * 64 if field.endswith("SHA") else good + 1)
    with pytest.raises(ConversionError):
        hybrid.prepare(inkling_prepare.CHOICE, output)
    assert not output.exists()
    assert (tmp_path / "model.native-download" / inkling_prepare.TARGET).read_bytes() == pinned.target
    assert (tmp_path / "model.native-derive" / "stark-derive.json").is_file()
    # A retry reuses the download and replaces its own unpublished derive folder.
    monkeypatch.setattr(inkling_prepare, field, good)
    hybrid.prepare(inkling_prepare.CHOICE, output, offline=True)
    assert hybrid.prepared_identity(output) == _identity(hybrid.SOURCES[inkling_prepare.CHOICE])
    assert _siblings(output) == ["model"]


def test_an_interrupted_derive_leaves_no_published_folder(tmp_path, pinned, monkeypatch):
    output = tmp_path / "model"
    build = inkling_affine8_artifact.build

    def interrupted(original, work, tensors_only=False):
        (work / inkling_prepare.TARGET).write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(inkling_affine8_artifact, "build", interrupted)
    with pytest.raises(KeyboardInterrupt):
        hybrid.prepare(inkling_prepare.CHOICE, output)
    assert not output.exists()
    assert sorted(_siblings(output)) == ["model.native-derive", "model.native-download"]
    monkeypatch.setattr(inkling_affine8_artifact, "build", build)
    hybrid.prepare(inkling_prepare.CHOICE, output, offline=True)
    assert hybrid.prepared_identity(output)["weight_bytes"] == pinned.derived_target_bytes + len(DRAFT)


def test_a_derive_folder_not_owned_by_the_operation_is_refused(tmp_path, pinned):
    output = tmp_path / "model"
    (tmp_path / "model.native-derive").mkdir()
    (tmp_path / "model.native-derive" / "keep.txt").write_text("not ours")
    with pytest.raises(ValueError, match="not owned"):
        hybrid.prepare(inkling_prepare.CHOICE, output)
    assert (tmp_path / "model.native-derive" / "keep.txt").read_text() == "not ours"
    assert not output.exists()


def test_disk_space_covers_download_and_derived_folder(tmp_path, pinned, monkeypatch):
    usage = namedtuple("usage", "total used free")
    download = pinned.original["weight_bytes"]
    margin = 4 * 1024**3
    # Enough for the download alone, not for both folders at once.
    monkeypatch.setattr(hybrid.shutil, "disk_usage", lambda _path: usage(0, 0, download + margin))
    with pytest.raises(ValueError, match="insufficient disk space for the pinned checkpoint"):
        hybrid.prepare(inkling_prepare.CHOICE, tmp_path / "model")
    assert pinned.calls == []
    # Offline, the download check is skipped; the derive check still refuses.
    monkeypatch.setattr(hybrid.shutil, "disk_usage", lambda _path: usage(0, 0, 1))
    stage = tmp_path / "model.native-download"
    _write_original(stage, pinned.target)
    with pytest.raises(OSError) as raised:
        hybrid.prepare(inkling_prepare.CHOICE, tmp_path / "model", offline=True)
    assert raised.value.errno == errno.ENOSPC and "derived 8-bit artifact" in str(raised.value)
    assert converter_failure_category(raised.value) == "disk_full"
    assert sorted(_siblings(tmp_path / "model")) == ["model.native-download"]


def test_upgrade_replaces_a_prepared_original_in_place(tmp_path, pinned, capsys):
    output = tmp_path / "model"
    hybrid.prepare(inkling_prepare.ORIGINAL_CHOICE, output)
    assert hybrid.main([inkling_prepare.CHOICE, "--output", str(output), "--upgrade-8bit"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "upgraded" and receipt["layout"] == inkling_prepare.AFFINE8_LAYOUT
    assert hybrid.prepared_identity(output) == _identity(hybrid.SOURCES[inkling_prepare.CHOICE])
    assert _siblings(output) == ["model"] and not (output / "stark-derive.json").exists()
    # Repeating it changes nothing.
    assert hybrid.upgrade_inkling_affine8(output)["status"] == "unchanged"


@pytest.mark.parametrize("failure", ["identity", "space"])
def test_a_failed_upgrade_leaves_the_original_untouched(tmp_path, pinned, monkeypatch, failure):
    output = tmp_path / "model"
    hybrid.prepare(inkling_prepare.ORIGINAL_CHOICE, output)
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    if failure == "identity":
        monkeypatch.setattr(inkling_prepare, "AFFINE8_TARGET_SHA", "0" * 64)
        error = ConversionError
    else:
        usage = namedtuple("usage", "total used free")
        monkeypatch.setattr(hybrid.shutil, "disk_usage", lambda _path: usage(0, 0, 1))
        error = OSError
    with pytest.raises(error):
        hybrid.upgrade_inkling_affine8(output)
    assert {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()} == before
    assert hybrid.prepared_identity(output) == _identity(pinned.original)


def test_upgrade_refuses_other_folders_and_choices(tmp_path, pinned, capsys):
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="prepared original"):
        hybrid.upgrade_inkling_affine8(other)
    with pytest.raises(ValueError, match="prepared Inkling"):
        hybrid.upgrade_inkling_affine8(tmp_path / "missing")
    for argv in (
        [inkling_prepare.ORIGINAL_CHOICE, "--output", str(other), "--upgrade-8bit"],
        [inkling_prepare.CHOICE, "--output", str(other), "--upgrade-8bit", "--upgrade-images"],
    ):
        with pytest.raises(SystemExit):
            hybrid.main(argv)
    assert hybrid.main([inkling_prepare.CHOICE, "--output", str(other), "--upgrade-8bit"]) == 1
    assert "did not complete" in capsys.readouterr().err


def test_catalog_offers_the_8bit_default_with_its_own_facts():
    item = native_catalog.catalog_item(inkling_prepare.CHOICE)
    source = hybrid.SOURCES[inkling_prepare.CHOICE]
    assert source is inkling_prepare.AFFINE8_SOURCE and source["derived_from"] == inkling_prepare.ORIGINAL_CHOICE
    assert hybrid.SOURCES[inkling_prepare.ORIGINAL_CHOICE] is inkling_prepare.SOURCE
    assert item["weight_bytes"] == 18_052_013_648 + 635_801_416
    assert item["quantization"] == inkling_prepare.AFFINE8_QUANTIZATION == inkling_affine8_artifact.QUANTIZATION
    assert item["artifact_modes"] == ["vision-language"] and item["default_artifact_mode"] == "vision-language"
    # The item keeps the identifier existing installations were matched by.
    revision = inkling_prepare.REPO + "@" + inkling_prepare.REVISION
    assert item["id"] == hashlib.sha256(revision.encode()).hexdigest()[:32]
    assert native_catalog.catalog_items(native_catalog.INKLING_FAMILY_ID) == [item]
    assert not native_catalog.allowed_choice(inkling_prepare.ORIGINAL_CHOICE)
    definition = native_catalog.family_definition(native_catalog.INKLING_FAMILY_ID)
    assert definition["conversion"]["source_formats"] == [inkling_prepare.AFFINE8_QUANTIZATION]
    assert inkling_affine8_artifact.LAYOUT == inkling_prepare.AFFINE8_LAYOUT
    assert not hasattr(inkling_affine8_artifact, "CHOICE")


def test_image_check_memory_uses_the_loaded_layout(monkeypatch):
    original = inkling_prepare.SOURCE["weight_bytes"] + 1152 * 1024**2
    affine8 = original + inkling_prepare.AFFINE8_EXTRA_RESIDENT_BYTES
    config = {"model_type": inkling_prepare.FAMILY}
    bound = {
        **config,
        "stark_inkling_affine8": {"layout": inkling_prepare.AFFINE8_LAYOUT, "group_size": 64, "bits": 8},
    }
    memory = namedtuple("memory", "available")
    monkeypatch.setattr(image_checks.psutil, "virtual_memory", lambda: memory(affine8 - 1))
    image_checks.preflight_family_memory(Path("unused"), config)
    with pytest.raises(image_checks.ImageCheckError, match=f"{affine8 / 1e9:.1f} GB"):
        image_checks.preflight_family_memory(Path("unused"), bound)
    monkeypatch.setattr(image_checks.psutil, "virtual_memory", lambda: memory(affine8))
    image_checks.preflight_family_memory(Path("unused"), bound)
