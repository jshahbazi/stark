"""Synthetic regression cases for converter metadata trust boundaries."""

import importlib
import json
import os
import struct
from dataclasses import replace

import pytest
from test_dense_adapters import dense_fixture
from test_gemma import gemma_fixture
from test_gguf_import import gguf_fixture
from test_granite_smollm import fixture as granite_fixture
from test_lfm2 import lfm2_fixture
from test_sources import PRIMARY, Hub, files
from test_streaming import qwen_fixture

from stark_convert import benchmark, gguf_import, provenance, sources, streaming
from stark_convert.sources import resolve_source
from stark_convert.streaming import ConversionError


def test_selected_directory_alias_preserves_json_reads_without_allowing_escaped_sidecars(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text('{"synthetic": true}')
    alias = tmp_path / "selected-alias"
    alias.symlink_to(root, target_is_directory=True)
    assert streaming.read_json(alias / "config.json") == {"synthetic": True}
    outside = tmp_path / "private.json"
    outside.write_text('{"synthetic_outside": true}')
    (root / "escaped.json").symlink_to(outside)
    with pytest.raises(ConversionError):
        streaming.read_json(alias / "escaped.json")


def test_supported_planning_budget_boundary_remains_usable():
    streaming.validate_planning_layers(
        streaming.MAX_PLANNING_LAYERS, dict.fromkeys(range(streaming.MAX_PLANNING_LAYERS))
    )


def planner_fixture(path, family):
    if family == "qwen3":
        qwen_fixture(path)
        module, function, kwargs = "qwen3", "build_qwen3_plan", {}
    elif family in {"llama", "qwen2", "mistral"}:
        dense_fixture(path, family)
        module, function, kwargs = "adapters.dense", "build_dense_plan", {"family": family}
    elif family in {"gemma2", "gemma3_text"}:
        gemma_fixture(path, family)
        module, function, kwargs = "adapters.gemma", "build_gemma_plan", {"family": family}
    elif family in {"granite", "smollm3"}:
        granite_fixture(path, family)
        module, function, kwargs = "adapters.granite", "_build_plan", {"family": family, "dtype": "bfloat16"}
    else:
        lfm2_fixture(path)
        module, function, kwargs = "adapters.lfm2", "build_lfm2_plan", {}
    module = importlib.import_module("stark_convert." + module)
    return module, lambda: getattr(module, function)(path, **kwargs)


@pytest.mark.parametrize(
    "family", ["qwen3", "llama", "qwen2", "mistral", "gemma2", "gemma3_text", "granite", "smollm3", "lfm2"]
)
@pytest.mark.parametrize("layers", [2**60, streaming.MAX_PLANNING_LAYERS + 1, 4000])
def test_topology_budget_precedes_expansion(tmp_path, monkeypatch, family, layers):
    path = tmp_path / "source"
    module, build = planner_fixture(path, family)
    assert build().operations
    config = json.loads((path / "config.json").read_text())
    config["num_hidden_layers"] = layers
    (path / "config.json").write_text(json.dumps(config))
    if layers > streaming.MAX_PLANNING_LAYERS:

        def unexpected_load(*args, **kwargs):
            pytest.fail("oversized topology reached tensor processing")

        monkeypatch.setattr(module, "load_checkpoint", unexpected_load)
    else:
        monkeypatch.setattr(module, "load_checkpoint", lambda *_: {})
    with pytest.raises(ConversionError, match="budget|inventory"):
        build()


def test_gguf_rejects_matching_oversized_layers_before_name_expansion(tmp_path, monkeypatch):
    path, hf, config, _ = gguf_fixture(tmp_path)
    source = gguf_import.read_gguf(path)
    assert gguf_import.build_import_plan(source, hf)[0].operations
    for layers in (2**60, 4000):
        config["num_hidden_layers"] = layers
        (hf / "config.json").write_text(json.dumps(config))
        malicious = replace(source, metadata={**source.metadata, "qwen3.block_count": layers})
        monkeypatch.setattr(gguf_import, "_hf_names", lambda *_: pytest.fail("name expansion reached"))
        with pytest.raises(ConversionError, match="budget|inventory"):
            gguf_import.build_import_plan(malicious, hf)


def test_lfm2_index_topology_matches_explicit_topology_without_list_membership(tmp_path, monkeypatch):
    path = tmp_path / "source"
    lfm2_fixture(path)
    from stark_convert.adapters import lfm2

    config = json.loads((path / "config.json").read_text())
    baseline = lfm2.build_lfm2_plan(path)
    layer_types = config.pop("layer_types", None)
    if layer_types is not None:
        config["full_attn_idxs"] = [i for i, kind in enumerate(layer_types) if kind == "full_attention"]

    class NoMembershipList(list):
        def __contains__(self, _):
            pytest.fail("linear list membership used for topology")

    config["full_attn_idxs"] = NoMembershipList(config["full_attn_idxs"])
    monkeypatch.setattr(lfm2, "read_json", lambda *_: config)
    assert lfm2.build_lfm2_plan(path).decoder_config == baseline.decoder_config


def write_header(path, shape, payload=b""):
    header = json.dumps({"tensor": {"dtype": "U8", "shape": shape, "data_offsets": [0, len(payload)]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


@pytest.mark.parametrize("shape", [[2] * 10000, [2**63, 2**63], [1] * 33, [False]])
def test_tensor_shape_work_is_bounded_in_both_parsers(tmp_path, shape):
    path = tmp_path / "model.safetensors"
    write_header(path, shape)
    with pytest.raises(ConversionError):
        streaming.read_safetensors_header(path)
    with pytest.raises(benchmark.BenchmarkError):
        benchmark._tensor_manifest(path)
    with pytest.raises(benchmark.BenchmarkError):
        benchmark.compare_outputs(tmp_path, tmp_path)


@pytest.mark.parametrize("shape,payload", [([], b"x"), ([3], b"abc"), ([2, 2], b"abcd"), ([2**63, 0], b"")])
def test_scalar_vector_matrix_and_zero_shapes_remain_valid(tmp_path, shape, payload):
    path = tmp_path / "model.safetensors"
    write_header(path, shape, payload)
    assert streaming.read_safetensors_header(path)[1]["tensor"].nbytes == len(payload)
    assert benchmark._tensor_manifest(path)[0]


@pytest.mark.parametrize(
    "filename,repo",
    [
        ("tokenizer_config.json", PRIMARY),
        ("SYSTEM_PROMPT.txt", "mistralai/Devstral-Small-2505"),
        ("chat_template.jinja", None),
    ],
)
def test_local_sidecars_cannot_escape_source_root(tmp_path, filename, repo):
    root = files(tmp_path / "source")
    if repo is None:
        (root / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    external = files(tmp_path / "external")
    (external / "chat_template.json").write_text("{}")
    hub = Hub(
        {
            "mistralai/Mistral-Small-3.1-24B-Base-2503": external,
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503": external,
        }
    )
    arguments = {"repo": repo, "metadata_profile": "hf-generic" if repo is None else None, "hub": hub}
    sidecar = root / filename
    if not sidecar.exists():
        sidecar.write_text("synthetic sidecar")
    assert resolve_source(root, **arguments).resolve_file(filename) == sidecar
    private = tmp_path / "outside.txt"
    private.write_text("synthetic outside text")
    sidecar.unlink()
    sidecar.symlink_to(private)
    with pytest.raises(ConversionError, match="authorized"):
        resolve_source(root, **arguments)


def test_canonical_hf_blob_metadata_links_remain_supported(tmp_path):
    model = tmp_path / "models--Qwen--Qwen3-0.6B"
    root = files(model / "snapshots" / ("1" * 40))
    blobs = model / "blobs"
    blobs.mkdir()
    for sidecar in root.iterdir():
        target = blobs / sidecar.name
        sidecar.rename(target)
        sidecar.symlink_to(target)
    snapshot = resolve_source(root, repo=PRIMARY)
    snapshot.assert_unchanged()
    assert snapshot.resolve_file("tokenizer_config.json") == blobs / "tokenizer_config.json"
    # Earlier manifests retained the snapshot link pathname instead of its target.
    for record in snapshot.manifest["files"]:
        record["path"] = str(root / record["filename"])
    manifest = tmp_path / "private-manifest.json"
    snapshot.write_manifest(manifest)
    replay = resolve_source(PRIMARY, source_manifest=manifest, offline=True)
    replay.assert_unchanged()


@pytest.mark.parametrize(
    "relative",
    [
        ("blobs", "aa", "a" * 64),
        ("blobs", "bb", "a" * 64),
        ("blobs", "aa", "a" * 63),
        ("blobs", "AA", "A" * 64),
        ("blobs", "aa", "a" * 64, "nested"),
        ("other-blobs", "aa", "a" * 64),
        ("unrelated-cache", "blobs", "aa", "a" * 64),
    ],
)
def test_shared_hf_blob_authority_requires_current_cache_and_exact_digest_layout(tmp_path, relative):
    cache = tmp_path / "hub"
    root = files(cache / "models--Qwen--Qwen3-0.6B" / "snapshots" / ("1" * 40))
    target = cache.joinpath(*relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("synthetic shared blob")
    sidecar = root / "tokenizer_config.json"
    sidecar.unlink()
    sidecar.symlink_to(target)
    if relative == ("blobs", "aa", "a" * 64):
        assert streaming.authorize_source_path(root, sidecar) == target
    else:
        with pytest.raises(ConversionError, match="authorized"):
            streaming.authorize_source_path(root, sidecar)


def test_shared_hf_blob_authority_rejects_a_link_escaping_the_selected_cache(tmp_path):
    cache = tmp_path / "hub"
    root = files(cache / "models--Qwen--Qwen3-0.6B" / "snapshots" / ("1" * 40))
    outside = tmp_path / "outside"
    outside.write_text("synthetic outside text")
    target = cache / "blobs" / "aa" / ("a" * 64)
    target.parent.mkdir(parents=True)
    target.symlink_to(outside)
    sidecar = root / "tokenizer_config.json"
    sidecar.unlink()
    sidecar.symlink_to(target)
    with pytest.raises(ConversionError, match="authorized"):
        streaming.authorize_source_path(root, sidecar)


@pytest.mark.parametrize("kind", ["fifo", "device"])
def test_special_sidecars_are_rejected_without_blocking(tmp_path, kind):
    root = files(tmp_path / "source")
    sidecar = root / "tokenizer_config.json"
    sidecar.unlink()
    if kind == "fifo":
        os.mkfifo(sidecar)
    else:
        sidecar.symlink_to("/dev/zero")
    with pytest.raises(ConversionError):
        resolve_source(root, repo=PRIMARY)
    # The separately reachable fixed HF/GGUF hashing path must also reject.
    with pytest.raises(ConversionError):
        provenance.file_hash(sidecar)
    with pytest.raises(ConversionError):
        sources._hash(sidecar.resolve())


def test_authorized_path_cannot_be_replaced_with_an_escaping_link(tmp_path):
    root = files(tmp_path / "source")
    path = streaming.authorize_source_path(root, root / "tokenizer_config.json")
    outside = tmp_path / "outside"
    outside.write_text("synthetic outside text")
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(OSError):
        streaming.read_regular_bytes(path)


def test_replaced_source_directory_cannot_authorize_an_outside_tree(tmp_path):
    root = files(tmp_path / "source")
    outside = files(tmp_path / "outside")
    root.rename(tmp_path / "original")
    root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ConversionError, match="authorized"):
        streaming.authorize_source_path(root, root / "tokenizer_config.json")
