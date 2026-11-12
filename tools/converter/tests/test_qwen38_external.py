"""External-table Qwen3.8 discovery and import use actual tensor headers."""

import hashlib
import json
import math
import struct
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from test_model_catalog import PIN, Hub

from stark_convert import model_catalog, qwen38_external
from stark_convert.adapters.qwen38 import describe_text_weights
from stark_convert.cli import convert
from stark_convert.desktop import model_valid
from stark_convert.hybrid_prepare import prepared_identity
from stark_convert.model_inventory import model_display
from stark_convert.model_support import preflight_source
from stark_convert.sources import resolve_source
from stark_convert.streaming import UnsupportedConversion

EXPERT = "model.language_model.layers.0.mlp.experts.gate_proj"


@pytest.fixture
def external_config(monkeypatch):
    geometry = {
        "hidden_size": 128,
        "vocab_size": 16,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 128,
        "shared_expert_intermediate_size": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "hc_count": 2,
        "hc_lowrank": 64,
        "indexer_n_heads": 1,
        "indexer_head_dim": 16,
        "indexer_budget": 4,
        "indexer_compress_ratio": 2,
        "ple_embed_dim": 128,
        "ple_conv_kernel_size": 4,
        "heads_per_ngram": 2,
    }
    monkeypatch.setattr(qwen38_external, "_NATIVE_TEXT_GEOMETRY", geometry)
    return {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {
            **geometry,
            "model_type": "qwen4_exp_text",
            "hidden_act": "silu",
            "eos_token_id": 1,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 1024,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "indexer_kv_heads": 1,
            "ple_layer_ids": [2],
            "ngram_vocab_size_base": 5,
            "rope_parameters": {"rope_theta": 10_000, "partial_rotary_factor": 0.25},
        },
        "quantization": {"mode": "affine", "bits": 4, "group_size": 64, "calibration": "imatrix-weighted affine"},
        "ngram_table": {"file": "ngram_table.bin", "bits": 4, "group_size": 32},
    }


def tensor_files(config, *, expert_bits=4):
    tensors, table = {}, {}
    # 5 + 7 + 11 + 13 is padded to 128 rows for this synthetic table.
    architecture = {
        **config["text_config"],
        "ngram_size": 3,
        "split_ngram_parts": 1,
        "ple": {"padded_vocab_size": 128},
    }

    def control(name, shape, integers=False):
        tensors[name] = ("I64" if integers else "BF16", shape)

    def matrix(stem, shape, *, group=None, fixed_q8=False, embedding=False, ngram=False):
        if stem.endswith((".block_inject_weight", ".mlp.shared_expert_gate")):
            control(stem + ".weight", shape)
            return
        bits = 8 if fixed_q8 or group in {"attn", "gdn", "shared", "head"} else 4
        group_size = 32 if ngram else 64
        if stem == EXPERT and expert_bits == 3:
            bits, group_size = 3, 128
        leading, columns = shape[:-1], shape[-1]
        for plane, dtype, width in (
            ("weight", "U32", columns * bits // 32),
            ("scales", "BF16", columns // group_size),
            ("biases", "BF16", columns // group_size),
        ):
            target, name = (table, plane) if ngram else (tensors, stem + "." + plane)
            target[name] = (dtype, (*leading, width))

    describe_text_weights(architecture, {}, matrix, control)
    return tensors, table


def tensor_bytes(tensors):
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        width = math.prod(shape) * {"U32": 4, "BF16": 2, "F32": 4, "I64": 8}[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + width]}
        offset += width
    payload = json.dumps(header).encode()
    return struct.pack("<Q", len(payload)) + payload + bytes(offset)


@pytest.mark.parametrize("theta", [1e-50, 1e50])
def test_external_draft_theta_must_remain_positive_finite_in_native_f32(external_config, theta):
    external_config["text_config"]["mtp"] = {
        "hybrid": True,
        "num_hidden_layers": 1,
        "layer_types": ["full_attention"],
        "rope_theta": theta,
    }
    tensors, table = tensor_files(external_config)
    # The metadata gate runs before visiting any draft planes or payloads.
    with pytest.raises(UnsupportedConversion, match="draft head"):
        qwen38_external.validate_layout(external_config, tensors, table)


@pytest.mark.parametrize("theta", [1e-45, 1.0, 10_000_000, 3.4028234663852886e38])
def test_native_mtp_theta_accepts_positive_finite_f32_values(theta):
    assert qwen38_external._valid_native_mtp_theta(theta)


class RangeHub(Hub):
    def __init__(self):
        super().__init__()
        self.utils = SimpleNamespace(build_hf_headers=dict, get_session=lambda: self)
        self.ignores_ranges = False

    def list_repo_files(self, *, repo_id, revision):
        assert revision == PIN
        return [path.name for path in self.repos[repo_id][0].iterdir()]

    def hf_hub_download(self, *, repo_id, filename, revision, **kwargs):
        assert revision == PIN
        self.events.append(("download", repo_id, filename))
        return str(self.repos[repo_id][0] / filename)

    @contextmanager
    def stream(self, method, url, *, headers, timeout, follow_redirects):
        repo, filename, revision = url
        assert method == "GET" and revision == PIN
        assert headers["Accept-Encoding"] == "identity" and follow_redirects
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        self.events.append(("range", repo, filename, start, end))
        path = self.repos[repo][0] / filename
        with path.open("rb") as stream:
            boundary = 8 + int.from_bytes(stream.read(8), "little")
            assert (start, end) in {(0, 7), (8, boundary - 1)}
            stream.seek(start)

            def chunks(chunk_size):
                assert not self.ignores_ranges, "A full-body response must be closed without being read"
                yield stream.read(end - start + 1)

            yield SimpleNamespace(
                status_code=200 if self.ignores_ranges else 206,
                headers={"content-range": f"bytes {start}-{end}/{path.stat().st_size}"},
                iter_raw=chunks,
            )


@pytest.mark.parametrize(
    "case", ["compatible", "three_bit", "missing_table", "ignored_range", "wrong_index", "bad_vision", "bad_seed"]
)
def test_catalog_checks_pinned_tensor_headers(tmp_path, monkeypatch, external_config, case):
    monkeypatch.setattr(model_catalog, "_catalog_seeds", lambda _family: [])
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    hub = RangeHub()
    repo = "example/weighted-export"
    tensors, table = tensor_files(external_config, expert_bits=3 if case == "three_bit" else 4)
    if case == "compatible":
        tensors["model.language_model.layers.0.linear_attn.A_log"] = ("F32", (2,))
        del external_config["text_config"]["ple_embed_dim"]
        del external_config["text_config"]["ple_conv_kernel_size"]
    elif case == "bad_seed":
        external_config["text_config"]["ngram_seed"] = 0
    elif case == "bad_vision":
        external_config["vision_config"] = {"depth": 0}
    files = {"model.safetensors": tensor_bytes(tensors), "tokenizer.json": {}, "tokenizer_config.json": {}}
    if case != "missing_table":
        files["ngram_table.bin"] = tensor_bytes(table)
    if case == "wrong_index":
        files["model.safetensors.index.json"] = {
            "weight_map": {**dict.fromkeys(tensors, "model.safetensors"), "phantom": "model.safetensors"}
        }
    names = {name: len(data) if isinstance(data, bytes) else 20 for name, data in files.items()}
    hub.add(tmp_path, repo, external_config, names={**names, "config.json": 100}, sidecars=files)
    hub.discovered["qwen4_exp"] = [repo]
    hub.ignores_ranges = case == "ignored_range"

    report = model_catalog.family_catalog("qwen4_exp", hub=hub)
    assert report["status"] == "ready"
    if case == "missing_table":
        assert report["items"] == []
        assert not any(event[0] == "range" for event in hub.events)
        return
    (item,) = report["items"]
    assert item["repo"] == repo and item["revision"] == PIN
    assert item["weight_bytes"] == names["model.safetensors"] + names["ngram_table.bin"]
    assert item["artifact_modes"] == ["text"]
    assert item["downloadable"] is (case == "compatible")
    if case == "three_bit":
        assert "3-bit packing with groups of 128" in item["unavailable_reason"]
        assert "4-bit packing with groups of 64" in item["unavailable_reason"]
    if case == "ignored_range":
        assert item["unavailable_reason"] == qwen38_external.UNVERIFIED
    if case == "wrong_index":
        assert "shard index" in item["unavailable_reason"]
    if case == "bad_vision":
        assert "declared vision configuration" in item["unavailable_reason"]
    if case == "bad_seed":
        assert "n-gram table configuration" in item["unavailable_reason"]
    assert not any(event[0] == "download" and event[-1].endswith((".bin", ".safetensors")) for event in hub.events)


def test_invalid_tensor_header_does_not_displace_valid_candidate(tmp_path, monkeypatch, external_config):
    monkeypatch.setattr(model_catalog, "_catalog_seeds", lambda _family: [])
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    hub = RangeHub()
    tensors, table = tensor_files(external_config)
    valid_shard = tensor_bytes(tensors)
    invalid_shard = valid_shard[:9] + b"!" + valid_shard[10:]
    for name, shard in (("valid", valid_shard), ("invalid", invalid_shard)):
        repo = f"example/{name}"
        sidecars = {
            "model.safetensors": shard,
            "ngram_table.bin": tensor_bytes(table),
            "tokenizer.json": {},
            "tokenizer_config.json": {},
        }
        sizes = {name: len(data) if isinstance(data, bytes) else 20 for name, data in sidecars.items()}
        hub.add(tmp_path, repo, external_config, names={**sizes, "config.json": 100}, sidecars=sidecars)
    hub.discovered["qwen4_exp"] = ["example/invalid", "example/valid"]

    report = model_catalog.family_catalog("qwen4_exp", hub=hub)
    assert report["status"] == "ready"
    items = {item["repo"]: item for item in report["items"]}
    assert items["example/valid"]["downloadable"]
    assert not items["example/invalid"]["downloadable"]
    assert items["example/invalid"]["unavailable_reason"] == qwen38_external.UNSUPPORTED


@pytest.mark.parametrize("remote", [False, True])
def test_generic_external_qwen_import_preserves_indexed_source_and_binds_table(tmp_path, external_config, remote):
    root = tmp_path / "checkpoint"
    root.mkdir()
    config_path = root / "config.json"
    config_path.write_text(json.dumps(external_config))
    (root / "tokenizer.json").write_text("{}")
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages[0].content }}"}))
    (root / "LICENSE").write_text("Synthetic license fixture")
    tensors, planes = tensor_files(external_config)
    shard = root / "model-00001-of-00001.safetensors"
    shard.write_bytes(tensor_bytes(tensors))
    table = root / "ngram_table.bin"
    table.write_bytes(tensor_bytes(planes))
    index = root / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": dict.fromkeys(tensors, shard.name)}))

    assert preflight_source(root)["status"] == "conditional"
    assert preflight_source(root, mode="decision")["status"] == "unsupported"
    assert preflight_source(root, mode="vision-language")["status"] == "unsupported"
    if remote:
        repo = "example/external-export"
        hub = RangeHub()
        hub.repos[repo] = (root, SimpleNamespace(sha=PIN))
        assert preflight_source(repo, hub=hub)["source"]["revision"] == PIN
        snapshot = resolve_source(repo, revision=PIN, metadata_profile="hf-generic", hub=hub)
        first_bulk = hub.events.index(("download", repo, shard.name))
        assert any(event[0] == "range" and event[2] == shard.name for event in hub.events[:first_bulk])
    else:
        repo = None
        snapshot = resolve_source(root, metadata_profile="hf-generic", offline=True)
    manifest = tmp_path / "source.json"
    snapshot.write_manifest(manifest)
    replay = resolve_source(root, source_manifest=manifest, offline=True)
    output = tmp_path / "prepared"
    receipt = convert(replay, output, repo=repo)
    assert receipt["status"] == "prepared"
    assert receipt["native_loading"] == receipt["execution"] == "unverified"
    assert receipt["weight_bytes"] == table.stat().st_size + shard.stat().st_size
    assert "lily" not in json.loads((output / "config.json").read_text())
    for file in (shard, table, index, root / "LICENSE"):
        assert (output / file.name).read_bytes() == file.read_bytes()
    assert receipt["weight_files"][table.name]["sha256"] == hashlib.sha256(table.read_bytes()).hexdigest()
    identity = prepared_identity(output)
    assert (identity["repo"], identity["revision"]) == (repo, PIN if remote else None)
    assert identity["architecture"] == "qwen4_exp" and identity["weight_bytes"] == receipt["weight_bytes"]
    assert model_valid(output)
    display = model_display({"id": "test", "name": "External Qwen", "path": str(output)})
    assert display["source_revision"] == identity["revision"]
    assert display["weight_bytes"] == identity["weight_bytes"]
    (output / index.name).write_bytes(index.read_bytes().replace(b"embed_tokens", b"other_tokens"))
    assert prepared_identity(output) is None

    del planes["biases"]
    table.write_bytes(tensor_bytes(planes))
    invalid = tmp_path / "invalid"
    with pytest.raises(UnsupportedConversion, match="n-gram table must contain"):
        convert(root, invalid, repo=None, metadata_profile="hf-generic")
    assert not invalid.exists()
    if remote:
        hub.events.clear()
        with pytest.raises(UnsupportedConversion, match="n-gram table must contain"):
            resolve_source(repo, revision=PIN, metadata_profile="hf-generic", hub=hub)
        assert not any(event[0] == "download" and event[-1].endswith((".bin", ".safetensors")) for event in hub.events)

    external_config["quantization"]["bits"] = 3
    config_path.write_text(json.dumps(external_config))
    assert preflight_source(root)["status"] == "unsupported"
    with pytest.raises(UnsupportedConversion):
        resolve_source(root, metadata_profile="hf-generic", offline=True)
