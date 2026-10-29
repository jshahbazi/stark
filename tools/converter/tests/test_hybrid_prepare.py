import json
from pathlib import Path

import pytest

from stark_convert import hybrid_prepare as hybrid


def metadata(root: Path, architecture="glm5_next"):
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": architecture,
                "text_config": {"vocab_size": 8, "eos_token_id": 1},
            }
        )
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": "{{ messages[0]['content'] }}",
            }
        )
    )
    (root / "tokenizer.json").write_text("{}")
    (root / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 2]}))


def test_metadata_keeps_architecture_and_unions_checkpoint_stops(tmp_path):
    metadata(tmp_path)
    receipt = hybrid.finish_metadata(tmp_path, "glm53-q2")
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["model_type"] == "glm5_next"
    assert config["text_config"]["eos_token_id"] == [1, 2]
    assert config["token_codec_config"]["prompt_template"] == "{{ messages[0]['content'] }}"
    assert receipt["native_loading"] == receipt["execution"] == "unverified"
    assert receipt["weights_source"]["revision"] == hybrid.SOURCES["glm53-q2"]["weights_revision"]


def test_metadata_rejects_changed_architecture_and_invalid_stops(tmp_path):
    metadata(tmp_path, "qwen3_5")
    with pytest.raises(ValueError, match="architecture"):
        hybrid.finish_metadata(tmp_path, "glm53-q2")
    metadata(tmp_path)
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": [99]}')
    with pytest.raises(ValueError, match="EOS"):
        hybrid.finish_metadata(tmp_path, "glm53-q2")


def test_metadata_preserves_declared_special_tokens_without_inventing_a_bos(tmp_path):
    metadata(tmp_path)
    tokenizer_path = tmp_path / "tokenizer_config.json"
    tokenizer = json.loads(tokenizer_path.read_text())
    tokenizer.update(eos_token={"content": "<|endoftext|>"})
    tokenizer_path.write_text(json.dumps(tokenizer))
    hybrid.finish_metadata(tmp_path, "glm53-q2")
    codec = json.loads((tmp_path / "config.json").read_text())["token_codec_config"]
    assert codec["eos_token"] == "<|endoftext|>"
    assert "bos_token" not in codec


def test_pinned_download_publishes_only_complete_original_format(tmp_path, monkeypatch):
    source = {**hybrid.SOURCES["glm53-q2"], "weight_bytes": 4}
    monkeypatch.setitem(hybrid.SOURCES, "glm53-q2", source)
    calls = []

    def download(repo, **options):
        calls.append((repo, options))
        root = Path(options["local_dir"])
        if repo == source["weights_repo"]:
            sub = root / "UD-Q2_K_XL"
            sub.mkdir(exist_ok=True)
            for index in range(4):
                (sub / f"model-{index}.gguf").write_bytes(b"x")
        else:
            metadata(root)

    monkeypatch.setattr(hybrid, "snapshot_download", download)
    output = tmp_path / "model"
    result = hybrid.prepare("glm53-q2", output)
    assert result["status"] == "prepared"
    assert len(list(output.glob("*.gguf"))) == 4
    assert not (output / "stark-download.json").exists()
    assert calls[0][1]["revision"] == source["weights_revision"]
    assert calls[1][1]["revision"] == source["metadata_revision"]
    assert all(not any(pattern.endswith(".py") for pattern in call[1]["allow_patterns"]) for call in calls)
    identity = hybrid.prepared_identity(output)
    assert identity["architecture"] == "glm5_next"
    from stark_convert.desktop import model_valid
    from stark_convert.model_inventory import model_display

    assert model_valid(output)
    display = model_display({"id": "test", "name": "Native GLM", "path": str(output)})
    assert display["weight_bytes"] == 4
    assert display["quantization"] == "UD-Q2_K_XL"
    with pytest.raises(ValueError, match="output already exists"):
        hybrid.prepare("glm53-q2", output)
    config = json.loads((output / "config.json").read_text())
    config["text_config"]["vocab_size"] = 9
    (output / "config.json").write_text(json.dumps(config))
    assert hybrid.prepared_identity(output) is None
    assert not model_valid(output)


def test_incomplete_download_keeps_owned_resumable_stage(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("private upstream exception")

    monkeypatch.setattr(hybrid, "snapshot_download", fail)
    with pytest.raises(RuntimeError):
        hybrid.prepare("glm53-q2", tmp_path / "model", offline=True)
    assert not (tmp_path / "model").exists()
    marker = tmp_path / "model.native-download" / "stark-download.json"
    assert json.loads(marker.read_text())["choice"] == "glm53-q2"
    marker.write_text('{"choice":"different"}')
    with pytest.raises(ValueError, match="staging source mismatch"):
        hybrid.prepare("glm53-q2", tmp_path / "model", offline=True)


def test_flint_preparation_preserves_packed_bytes_and_imports_local_folder(tmp_path, monkeypatch):
    source = {**hybrid.SOURCES["flint-int4"], "weight_bytes": 20}
    monkeypatch.setitem(hybrid.SOURCES, "flint-int4", source)
    calls = []

    def download(repo, **options):
        assert repo == "RecursiveSI/flint-int4"
        calls.append(options)
        root = Path(options["local_dir"])
        if "*.safetensors" in options["allow_patterns"]:
            for index in range(5):
                (root / f"model-{index:05}.safetensors").write_bytes(bytes([index, 2, 3, 4]))
        else:
            metadata(root, "flint")
            config = {"model_type": "flint", "vocab_size": 8, "eos_token_id": [1, 3]}
            (root / "config.json").write_text(json.dumps(config))
            (root / "chat_template.jinja").write_text("<|user|>{{ messages[0]['content'] }}<|assistant|>")

    monkeypatch.setattr(hybrid, "snapshot_download", download)
    output = tmp_path / "flint"
    receipt = hybrid.prepare("flint-int4", output, offline=True)
    assert receipt["architecture"] == "flint"
    assert receipt["native_loading"] == receipt["execution"] == "unverified"
    assert all(call["revision"] == "3b93f9053b530488405ae9fca51d4900d6ce21f0" for call in calls)
    assert all(call["local_files_only"] for call in calls)
    assert not any(pattern.endswith(".py") for call in calls for pattern in call["allow_patterns"])
    assert (output / "model-00004.safetensors").read_bytes() == bytes([4, 2, 3, 4])
    config = json.loads((output / "config.json").read_text())
    assert config["eos_token_id"] == [1, 3, 2]
    assert config["token_codec_config"]["prompt_template"].startswith("<|user|>")
    from stark_convert.desktop import model_valid
    from stark_convert.model_inventory import model_display

    assert model_valid(output)
    assert hybrid.prepared_identity(output)["architecture"] == "flint"
    display = model_display({"id": "flint", "name": "Flint", "path": str(output)})
    assert display["weight_bytes"] == 20
    assert display["quantization"] == source["quantization"]
    # Equal total bytes cannot substitute for a complete pinned shard set.
    (output / "model-00004.safetensors").unlink()
    (output / "model-00003.safetensors").write_bytes(b"12345678")
    assert hybrid.prepared_identity(output) is None
    assert not model_valid(output)


def test_lily_q4_xl_recipe_is_pinned_and_quantization_label_is_distinct():
    source = hybrid.SOURCES["qwen38-lily-q4-xl"]
    assert source["weights_repo"] == "fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl"
    assert source["weights_revision"] == "0f94842e3682b49acace4daeb4cf2e65a6f94f83"
    assert source["metadata_repo"] == source["weights_repo"]
    assert source["metadata_revision"] == source["weights_revision"]
    assert source["architecture"] == "qwen4_exp"
    assert source["weight_bytes"] == 106_905_155_192
    assert source["weight_shards"] == 55
    assert source["quantization"] != hybrid.SOURCES["qwen38-mixed"]["quantization"]
    assert source["required_files"] == ["LICENSE", "README.md"]


def test_lily_q4_xl_preparation_preserves_config_metadata_and_import_identity(tmp_path, monkeypatch):
    source = {**hybrid.SOURCES["qwen38-lily-q4-xl"], "weight_bytes": 55}
    monkeypatch.setitem(hybrid.SOURCES, "qwen38-lily-q4-xl", source)
    calls = []
    lily_config = {
        "format": "qwen4_exp-affine-v1",
        "quantization": {
            "default": {"bits": 4, "group_size": 64},
            "q8_groups": ["attn", "shared", "head", "embed"],
            "ngram_embedding": {"bits": 4, "group_size": 32},
        },
        "ple": {"layer_multipliers": [3, 5], "ngram_heads_vocab_sizes": [101], "ngram_heads_offsets": [0]},
        "vision": None,
        "mtp": None,
    }

    def download(repo, **options):
        calls.append((repo, options))
        root = Path(options["local_dir"])
        if "model-*-of-*.safetensors" in options["allow_patterns"]:
            for index in range(1, 56):
                name = f"model-{index:05}-of-00055.safetensors"
                (root / name).write_bytes(bytes([index % 256]))
            (root / "model.safetensors.index.json").write_text("{}")
            (root / "README.md").write_text("Lily checkpoint README\n")
            (root / "LICENSE").write_text("upstream checkpoint license\n")
            return
        metadata(root, "qwen4_exp")
        config = json.loads((root / "config.json").read_text())
        config["text_config"]["model_type"] = "qwen4_exp_text"
        config["lily"] = lily_config
        (root / "config.json").write_text(json.dumps(config))

    monkeypatch.setattr(hybrid, "snapshot_download", download)
    output = tmp_path / "qwen38-lily-q4-xl"
    receipt = hybrid.prepare("qwen38-lily-q4-xl", output, offline=True)

    assert len(calls) == 2
    assert all(repo == source["weights_repo"] for repo, _ in calls)
    assert all(options["revision"] == source["weights_revision"] for _, options in calls)
    assert all(options["local_files_only"] for _, options in calls)
    assert calls[0][1]["allow_patterns"] == source["weights_patterns"]
    assert receipt["weight_bytes"] == 55
    assert receipt["native_loading"] == receipt["execution"] == "unverified"
    assert (output / "LICENSE").read_text() == "upstream checkpoint license\n"
    assert (output / "README.md").read_text() == "Lily checkpoint README\n"
    assert (output / "model-00055-of-00055.safetensors").read_bytes() == bytes([55])

    config = json.loads((output / "config.json").read_text())
    assert config["lily"] == lily_config
    assert config["text_config"]["eos_token_id"] == [1, 2]
    assert config["token_codec_config"]["prompt_template"] == "{{ messages[0]['content'] }}"
    assert config["stark_native_hybrid"]["source"] == source["weights_repo"]

    from stark_convert.desktop import model_valid
    from stark_convert.model_inventory import model_display

    assert hybrid.prepared_identity(output)["quantization"] == source["quantization"]
    assert model_valid(output)
    display = model_display({"id": "lily", "name": "Lily Q4 XL", "path": str(output)})
    assert display["quantization"] == source["quantization"]

    (output / "README.md").unlink()
    assert hybrid.prepared_identity(output) is None


def test_interrupted_gguf_flatten_resumes_without_downloading_moved_shards(tmp_path, monkeypatch):
    source = {**hybrid.SOURCES["glm53-q2"], "weight_bytes": 4}
    monkeypatch.setitem(hybrid.SOURCES, "glm53-q2", source)
    downloads = []

    def download(repo, **options):
        root = Path(options["local_dir"])
        if repo != source["weights_repo"]:
            metadata(root)
            return
        sub = root / "UD-Q2_K_XL"
        sub.mkdir(exist_ok=True)
        missing = []
        for index in range(4):
            path = sub / f"model-{index}.gguf"
            if not path.exists():
                missing.append(index)
                path.write_bytes(b"x")
        downloads.append(missing)

    original = Path.rename
    moves = 0

    def interrupt(path, target):
        nonlocal moves
        if path.parent.name == "UD-Q2_K_XL":
            moves += 1
            if moves == 2:
                raise InterruptedError()
        return original(path, target)

    monkeypatch.setattr(hybrid, "snapshot_download", download)
    monkeypatch.setattr(Path, "rename", interrupt)
    output = tmp_path / "model"
    with pytest.raises(InterruptedError):
        hybrid.prepare("glm53-q2", output)
    monkeypatch.setattr(Path, "rename", original)
    assert hybrid.prepare("glm53-q2", output)["status"] == "prepared"
    assert downloads == [[0, 1, 2, 3], []]


def test_preparation_cli_never_prints_exception_payload(tmp_path, monkeypatch, capsys):
    def fail(*_args, **_kwargs):
        raise RuntimeError("private upstream exception")

    monkeypatch.setattr(hybrid, "prepare", fail)
    assert hybrid.main(["qwen38-mixed", "--output", str(tmp_path / "model")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "private upstream exception" not in captured.err
    assert "did not complete" in captured.err
