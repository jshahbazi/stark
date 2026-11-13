import hashlib
import io
import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

from stark_convert import hybrid_prepare as hybrid


def test_completed_model_bytes_excludes_partial_stale_and_unreceipted_files(tmp_path):
    source = {"weights_revision": "a" * 40, "weight_bytes": 100}
    records = tmp_path / ".cache/huggingface/download"
    records.mkdir(parents=True)

    def complete(name, size, revision=source["weights_revision"]):
        path = tmp_path / name
        path.write_bytes(b"x" * size)
        (records / (name + ".metadata")).write_text(f"{revision}\netag\n{path.stat().st_mtime + 1}\n")

    complete("a.safetensors", 25)
    complete("b.safetensors", 30, "b" * 40)
    (tmp_path / "c.safetensors").write_bytes(b"x" * 40)
    (tmp_path / "d.incomplete").write_bytes(b"x" * 200)
    assert hybrid.completed_model_bytes(tmp_path, source) == 25
    complete("b.safetensors", 30)
    assert hybrid.completed_model_bytes(tmp_path, source) == 55


def test_completed_bytes_matches_real_hub_receipt_timestamp_tolerance(tmp_path):
    from huggingface_hub._local_folder import read_download_metadata, write_download_metadata

    revision = "a" * 40
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"complete")
    write_download_metadata(tmp_path, path.name, revision, "etag")
    receipt = read_download_metadata(tmp_path, path.name)
    assert receipt is not None
    source = {"weights_revision": revision, "weight_bytes": path.stat().st_size}
    os.utime(path, (receipt.timestamp + 0.5, receipt.timestamp + 0.5))
    assert read_download_metadata(tmp_path, path.name) is not None
    assert hybrid.completed_model_bytes(tmp_path, source) == source["weight_bytes"]
    os.utime(path, (receipt.timestamp + 2, receipt.timestamp + 2))
    assert read_download_metadata(tmp_path, path.name) is None
    assert hybrid.completed_model_bytes(tmp_path, source) == 0


@pytest.fixture(autouse=True)
def synthetic_download_disk_headroom(monkeypatch, tmp_path):
    # Download tests use byte-sized local stand-ins. Their admission must not
    # depend on how much disk happens to be free on the test host. Individual
    # disk-rejection cases can override this fixture's probe.
    usage = hybrid.shutil.disk_usage(tmp_path)
    monkeypatch.setattr(hybrid.shutil, "disk_usage", lambda _: usage._replace(free=16 * 1024**3))


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


def test_edge0_recipe_is_pinned_and_prepare_only():
    source = hybrid.SOURCES["edge0-145b-q3k"]
    assert source["weights_repo"] == "Edge0/Edge0-145B-q3k"
    assert source["weights_revision"] == "d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74"
    assert source["preparatory_only"] is True
    assert source["weight_bytes"] == 103_374_417_600
    assert len(source["weight_files"]) == 8
    assert sum(source["file_manifest"][name]["size"] for name in source["weight_files"]) == source["weight_bytes"]
    assert set(source["weights_patterns"]) == set(source["file_manifest"])
    assert not {"profile.bin", "mtp/draft_vocab.bin", "latent_dense_q4.bin", "start.sh"} & set(
        source["weights_patterns"]
    )
    assert not any(name.startswith("assets/") or "*" in name for name in source["weights_patterns"])


@pytest.fixture
def edge0_preparation(monkeypatch):
    from stark_convert import edge0_source, edge0_tokenizer

    content = {"experts.bin": b"1234", "mtp/dense.bin": b"56", "LICENSE": b"terms", "tokenizer/vocab.json": b"{}"}
    manifest = {
        name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in content.items()
    }
    source = {
        **hybrid.SOURCES["edge0-145b-q3k"],
        "file_manifest": manifest,
        "weights_patterns": list(manifest),
        "weight_files": ["experts.bin", "mtp/dense.bin"],
        "weight_bytes": 6,
    }
    monkeypatch.setitem(hybrid.SOURCES, "edge0-145b-q3k", source)
    calls = []

    def download(repo, **options):
        assert repo == source["weights_repo"] and options["revision"] == source["weights_revision"]
        assert options["local_files_only"]
        calls.append(list(options["allow_patterns"]))
        for name in options["allow_patterns"]:
            target = Path(options["local_dir"]) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content[name])

    def tokenizer(directory, metadata, *, offline=False):
        assert offline
        assert metadata == {"checked": True}
        derived = {}
        for name in ("tokenizer.json", "tokenizer_config.json"):
            data = b"{}"
            (directory / name).write_bytes(data)
            derived[name] = {"sha256": hashlib.sha256(data).hexdigest(), "derivation": "synthetic"}
        return {"derived_metadata": derived, "tokenizer_equivalence": {"equivalent": True}}

    monkeypatch.setattr(hybrid, "snapshot_download", download)
    monkeypatch.setattr(edge0_source, "fetch_skeleton_metadata", lambda *_args, **_kw: {"checked": True})
    monkeypatch.setattr(
        edge0_source,
        "validate_source",
        lambda _directory: {"gguf_metadata": {"checked": True}, "source_layout": {"validated": True}},
    )
    monkeypatch.setattr(edge0_tokenizer, "prepare_tokenizer", tokenizer)
    monkeypatch.setattr(edge0_tokenizer, "preflight_tokenizer", lambda *_args, **_kw: {"equivalent": True})
    return source, calls, content


def test_edge0_preparation_binds_every_file_and_never_grants_runtime_admission(tmp_path, edge0_preparation):
    from stark_convert import qwen38_external, qwen38_retained
    from stark_convert.desktop import model_valid

    source, calls, content = edge0_preparation
    output = tmp_path / "edge0"
    result = hybrid.prepare("edge0-145b-q3k", output, offline=True)
    assert calls == [["LICENSE", "tokenizer/vocab.json"], ["experts.bin", "mtp/dense.bin"]]
    assert result["status"] == "prepared"
    assert result["native_loading"] == result["execution"] == "unverified"
    assert result["usage_scope"] == "local_noncommercial_evaluation"
    assert set(result["files"]) == {*content, "config.json", "tokenizer.json", "tokenizer_config.json"}
    assert result == json.loads((output / "native-source.json").read_text())
    config = json.loads((output / "config.json").read_text())
    assert config["model_type"] == "qwen4_exp"
    assert config["storage"]["layout"] == "edge0-strata"
    assert not qwen38_external.is_external_affine(config)
    assert not qwen38_retained.is_retained_affine(config)
    assert hybrid.preparation_identity(output)["loadable"] is False
    assert hybrid.prepared_identity(output) is None
    assert not model_valid(output)
    assert not output.with_name(output.name + ".native-download").exists()
    assert not (output / "stark-download.json").exists()
    (output / "mtp/dense.bin").write_bytes(b"78")
    assert hybrid.preparation_identity(output) is None


@pytest.mark.parametrize("mutation", ["source", "derived", "extra", "receipt", "config"])
def test_edge0_preparation_identity_rejects_mutations(tmp_path, edge0_preparation, mutation):
    output = tmp_path / "edge0"
    hybrid.prepare("edge0-145b-q3k", output, offline=True)
    if mutation == "source":
        (output / "LICENSE").write_bytes(b"other")
    elif mutation == "derived":
        (output / "tokenizer.json").write_bytes(b"[]")
    elif mutation == "extra":
        (output / "unexpected.bin").write_bytes(b"extra")
    elif mutation == "config":
        config = json.loads((output / "config.json").read_text())
        config["lily"] = {"format": "qwen4_exp-affine-v1"}
        (output / "config.json").write_text(json.dumps(config))
    else:
        receipt = json.loads((output / "native-source.json").read_text())
        receipt["native_loading"] = "verified"
        (output / "native-source.json").write_text(json.dumps(receipt))
    assert hybrid.preparation_identity(output) is None
    assert hybrid.prepared_identity(output) is None


def test_edge0_tokenizer_preflight_rejects_before_any_payload_download(
    tmp_path, monkeypatch, edge0_preparation, capsys
):
    from stark_convert import edge0_tokenizer

    _, calls, _ = edge0_preparation

    def reject(*_args, **_kw):
        raise ValueError("private source details")

    monkeypatch.setattr(edge0_tokenizer, "preflight_tokenizer", reject)
    output = tmp_path / "edge0"
    assert hybrid.main(["edge0-145b-q3k", "--output", str(output), "--offline", "--preflight-only"]) == 1
    assert calls == [["LICENSE", "tokenizer/vocab.json"]]
    assert not output.exists()
    assert output.with_name(output.name + ".native-download").is_dir()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"status": "rejected", "category": "tokenizer_equivalence_rejected"}
    assert "private source details" not in captured.err


def test_edge0_metadata_only_preflight_does_not_publish(tmp_path, edge0_preparation):
    _, calls, _ = edge0_preparation
    output = tmp_path / "edge0"
    result = hybrid.prepare("edge0-145b-q3k", output, offline=True, preflight_only=True)
    assert result["status"] == "preflight_verified"
    assert calls == [["LICENSE", "tokenizer/vocab.json"]]
    assert not output.exists()
    assert not (output.with_name(output.name + ".native-download") / "experts.bin").exists()


@pytest.mark.parametrize("failure", ["hash", "symlink", "validation", "mutation", "publication"])
def test_edge0_failed_preparation_keeps_owned_retryable_stage(tmp_path, monkeypatch, edge0_preparation, failure):
    from stark_convert import edge0_source

    _, _, content = edge0_preparation
    output = tmp_path / "edge0"
    if failure == "hash":
        content["experts.bin"] = b"4321"
    elif failure == "symlink":
        original = hybrid.snapshot_download

        def download(*args, **options):
            original(*args, **options)
            if "experts.bin" in options["allow_patterns"]:
                target = Path(options["local_dir"]) / "experts.bin"
                target.unlink()
                target.symlink_to(Path(options["local_dir"]) / "LICENSE")

        monkeypatch.setattr(hybrid, "snapshot_download", download)
    elif failure == "validation":
        monkeypatch.setattr(
            edge0_source, "validate_source", lambda _directory: (_ for _ in ()).throw(ValueError("invalid source"))
        )
    elif failure == "mutation":

        def mutate(directory):
            (directory / "experts.bin").write_bytes(b"4321")
            return {"gguf_metadata": {"checked": True}, "source_layout": {"validated": True}}

        monkeypatch.setattr(edge0_source, "validate_source", mutate)
    else:
        monkeypatch.setattr(hybrid, "_publish", lambda *_args: (_ for _ in ()).throw(InterruptedError()))
    with pytest.raises((ValueError, InterruptedError)):
        hybrid.prepare("edge0-145b-q3k", output, offline=True)
    assert not output.exists()
    stage = output.with_name(output.name + ".native-download")
    assert json.loads((stage / "stark-download.json").read_text())["choice"] == "edge0-145b-q3k"


def test_edge0_publication_retry_preserves_owned_staging(tmp_path, monkeypatch, edge0_preparation):
    output = tmp_path / "edge0"
    publish = hybrid._publish
    monkeypatch.setattr(hybrid, "_publish", lambda *_args: (_ for _ in ()).throw(InterruptedError()))
    with pytest.raises(InterruptedError):
        hybrid.prepare("edge0-145b-q3k", output, offline=True)
    monkeypatch.setattr(hybrid, "_publish", publish)
    assert hybrid.prepare("edge0-145b-q3k", output, offline=True)["status"] == "prepared"
    assert hybrid.preparation_identity(output)["loadable"] is False


def test_edge0_generic_metadata_finisher_cannot_issue_a_native_receipt(tmp_path):
    with pytest.raises(ValueError, match="require pinned source preparation"):
        hybrid.finish_metadata(tmp_path, "edge0-145b-q3k")
    assert not (tmp_path / "native-source.json").exists()


def test_edge0_progress_counts_nested_payloads_from_verified_hub_download_records(tmp_path):
    source = hybrid.SOURCES["edge0-145b-q3k"]
    path = tmp_path / "mtp/dense.bin"
    path.parent.mkdir()
    path.write_bytes(b"1234")
    records = tmp_path / ".cache/huggingface/download/mtp"
    records.mkdir(parents=True)
    (records / "dense.bin.metadata").write_text(f"{source['weights_revision']}\netag\n{path.stat().st_mtime + 1}\n")
    assert hybrid.completed_model_bytes(tmp_path, source) == 4
    (tmp_path / "unlisted.gguf").write_bytes(b"ignored")
    assert hybrid.completed_model_bytes(tmp_path, source) == 4


def test_edge0_hash_batch_detects_a_file_changed_after_its_hash(tmp_path, monkeypatch):
    (tmp_path / "a").write_bytes(b"A")
    (tmp_path / "b").write_bytes(b"B")
    original = hashlib.sha256

    class Digest:
        def __init__(self):
            self.digest = original()

        def update(self, data):
            self.digest.update(data)
            if data == b"B":
                (tmp_path / "a").write_bytes(b"C")

        def hexdigest(self):
            return self.digest.hexdigest()

    monkeypatch.setattr(hashlib, "sha256", Digest)
    with pytest.raises(ValueError, match="changed"):
        hybrid._edge0_files(tmp_path, ["a", "b"])


@pytest.mark.parametrize("name", ["tokenizer", ".cache", "config.json", "native-source.json"])
def test_edge0_resume_rejects_symlinks_before_any_source_writes(tmp_path, edge0_preparation, name):
    source, calls, _ = edge0_preparation
    output = tmp_path / "edge0"
    stage = output.with_name(output.name + ".native-download")
    stage.mkdir()
    (stage / "stark-download.json").write_text(
        json.dumps({"choice": "edge0-145b-q3k", "source": source, "tokenizer_policy": "borrowed-qwen"})
    )
    target = tmp_path / "untouched"
    target.mkdir()
    (stage / name).symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="staging path"):
        hybrid.prepare("edge0-145b-q3k", output, offline=True)
    assert calls == []
    assert list(target.iterdir()) == []


def test_edge0_progress_callback_mutation_cannot_publish(tmp_path, edge0_preparation):
    output = tmp_path / "edge0"
    stage = output.with_name(output.name + ".native-download")

    def progress(event):
        if event["phase"] == "conversion" and event["state"] == "running":
            (stage / "experts.bin").write_bytes(b"4321")

    with pytest.raises(ValueError, match="changed before publication"):
        hybrid.prepare("edge0-145b-q3k", output, offline=True, progress=progress)
    assert not output.exists()


def test_edge0_receipt_cannot_claim_preparation_with_failed_tokenizer_equivalence(tmp_path, edge0_preparation):
    output = tmp_path / "edge0"
    hybrid.prepare("edge0-145b-q3k", output, offline=True)
    receipt = json.loads((output / "native-source.json").read_text())
    receipt["tokenizer_equivalence"]["equivalent"] = False
    (output / "native-source.json").write_text(json.dumps(receipt))
    assert hybrid.preparation_identity(output) is None


@pytest.fixture
def qualified_native_tokenizer(tmp_path, monkeypatch, edge0_preparation):
    """Synthetic acceptance exercises publication; it is not a real proof."""
    from stark_convert import edge0_tokenizer_admission as admission

    asset = {"properties": {"L": [[97, 122]], "M": [], "N": [[48, 57]], "s": [[32, 32]]}}
    asset_bytes = admission._encoded(asset)
    metadata = {
        "tokens": [
            "a",
            "b",
            "ab",
            "ba",
            "<|endoftext|>",
            "<|im_end|>",
            "<|fim_pad|>",
            "<|repo_name|>",
            "<|file_sep|>",
        ],
        "types": [1, 1, 1, 1, 3, 3, 3, 3, 3],
        "vocab": {
            "a": 0,
            "b": 1,
            "ab": 2,
            "ba": 3,
            "<|endoftext|>": 4,
            "<|im_end|>": 5,
            "<|fim_pad|>": 6,
            "<|repo_name|>": 7,
            "<|file_sep|>": 8,
        },
        "merges": ["a b", "b a"],
        "descriptor": {
            "pre_pattern": r"\p{L}+",
            "special_ids": {f"tokenizer.ggml.{name}_token_id": 4 for name in ("bos", "eos", "padding")},
        },
    }
    monkeypatch.setattr(admission.candidate, "sealed_metadata", lambda *_args: metadata)
    monkeypatch.setattr(admission, "_source_bytes", lambda *_args: b"{{ messages[0].content }}")
    package = {"edge0_unicode.json": asset_bytes, "edge0_native_tokenizer.py": b"synthetic-candidate"}
    monkeypatch.setattr(admission, "_package_bytes", lambda name: package[name])
    derived = admission.inspect_derived_metadata(tmp_path, {})
    profile = {
        "policy": "edge0-native-v2",
        "profile": "synthetic-native-policy",
        "candidate_revision": admission.candidate.UNICODE_CANDIDATE_REVISION,
        "source": {"repo": admission.EDGE0_REPO_ID, "revision": admission.EDGE0_REVISION},
        "tokenizers_version": admission.tokenizers.__version__,
        "candidate_module_sha256": hashlib.sha256(package["edge0_native_tokenizer.py"]).hexdigest(),
        "unicode_asset_sha256": hashlib.sha256(asset_bytes).hexdigest(),
        "tokenizer_json_sha256": derived["tokenizer.json"]["sha256"],
        "tokenizer_config_sha256": derived["tokenizer_config.json"]["sha256"],
        "proof": {"attempt": "20261112-EDGE0-03B.1", "sha256": "a" * 64},
        "primary_eos_token_id": 4,
        "eog_token_ids": [4, 5, 6, 7, 8],
    }
    monkeypatch.setattr(admission, "ACCEPTED_PROFILE", profile)
    return admission, package, profile


def test_edge0_native_policy_is_disabled_without_registered_profile(tmp_path, monkeypatch, edge0_preparation, capsys):
    from stark_convert import edge0_tokenizer_admission as admission

    monkeypatch.setattr(admission, "ACCEPTED_PROFILE", None)
    output = tmp_path / "edge0"
    assert (
        hybrid.main(
            [
                "edge0-145b-q3k",
                "--output",
                str(output),
                "--offline",
                "--preflight-only",
                "--tokenizer-policy",
                "edge0-native-v2",
            ]
        )
        == 1
    )
    assert edge0_preparation[1] == []
    assert not output.exists() and not output.with_name(output.name + ".native-download").exists()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"status": "rejected", "category": "native_tokenizer_unqualified"}


def test_edge0_native_policy_uses_separate_qualified_receipt(tmp_path, edge0_preparation, qualified_native_tokenizer):
    admission, _, profile = qualified_native_tokenizer
    output = tmp_path / "edge0-native"
    receipt = hybrid.prepare("edge0-145b-q3k", output, offline=True, tokenizer_policy="edge0-native-v2")
    assert receipt["tokenizer_policy"] == "edge0-native-v2"
    assert "tokenizer_equivalence" not in receipt
    assert receipt["native_tokenizer_validation"] == admission.validation_identity()
    assert receipt["native_tokenizer_validation"]["qualification"] == profile
    assert receipt["native_loading"] == receipt["execution"] == "unverified"
    assert receipt["files"]["tokenizer.json"]["sha256"] == profile["tokenizer_json_sha256"]
    assert receipt["files"]["tokenizer_config.json"]["sha256"] == profile["tokenizer_config_sha256"]
    assert hybrid.preparation_identity(output)["tokenizer_policy"] == "edge0-native-v2"
    assert hybrid.prepared_identity(output) is None
    from stark_convert.desktop import model_valid

    assert not model_valid(output)


@pytest.mark.parametrize("change", ["program", "asset", "profile", "proof", "equivalence"])
def test_edge0_native_identity_rejects_changed_qualification(
    tmp_path, qualified_native_tokenizer, edge0_preparation, change
):
    _, package, profile = qualified_native_tokenizer
    output = tmp_path / "edge0-native"
    hybrid.prepare("edge0-145b-q3k", output, offline=True, tokenizer_policy="edge0-native-v2")
    if change == "program":
        package["edge0_native_tokenizer.py"] = b"changed"
    elif change == "asset":
        package["edge0_unicode.json"] = b"{}"
    elif change == "profile":
        profile["tokenizer_json_sha256"] = "b" * 64
    else:
        receipt = json.loads((output / "native-source.json").read_text())
        if change == "proof":
            receipt["native_tokenizer_validation"]["qualification"]["proof"]["sha256"] = "b" * 64
        else:
            receipt["tokenizer_equivalence"] = {"equivalent": True}
        (output / "native-source.json").write_text(json.dumps(receipt))
    assert hybrid.preparation_identity(output) is None
    assert hybrid.prepared_identity(output) is None


def test_edge0_native_policy_cannot_reuse_borrowed_policy_stage(
    tmp_path, edge0_preparation, qualified_native_tokenizer
):
    output = tmp_path / "edge0"
    hybrid.prepare("edge0-145b-q3k", output, offline=True, preflight_only=True)
    stage = output.with_name(output.name + ".native-download")
    assert json.loads((stage / "stark-download.json").read_text())["tokenizer_policy"] == "borrowed-qwen"
    with pytest.raises(ValueError, match="staging source mismatch"):
        hybrid.prepare("edge0-145b-q3k", output, offline=True, tokenizer_policy="edge0-native-v2")


def test_edge0_native_preflight_has_no_payload_download_or_derived_writes(
    tmp_path, edge0_preparation, qualified_native_tokenizer
):
    output = tmp_path / "edge0-native"
    report = hybrid.prepare(
        "edge0-145b-q3k", output, offline=True, tokenizer_policy="edge0-native-v2", preflight_only=True
    )
    assert report["status"] == "preflight_verified" and "tokenizer_equivalence" not in report
    assert report["native_tokenizer_validation"]["source_metadata_validated"] is True
    assert edge0_preparation[1] == [["LICENSE", "tokenizer/vocab.json"]]
    stage = output.with_name(output.name + ".native-download")
    assert not (stage / "tokenizer.json").exists()
    assert not (stage / "tokenizer_config.json").exists()
    assert not output.exists()


@pytest.mark.parametrize("change", ["self_attestation", "version", "source", "eog"])
def test_edge0_native_policy_rejects_incomplete_or_drifted_registration_before_download(
    tmp_path, edge0_preparation, qualified_native_tokenizer, change
):
    _, _, profile = qualified_native_tokenizer
    if change == "self_attestation":
        profile.clear()
        profile["passed"] = True
    elif change == "version":
        profile["tokenizers_version"] = "different"
    elif change == "source":
        profile["source"]["revision"] = "b" * 40
    else:
        profile["eog_token_ids"] = [4]
    with pytest.raises(hybrid.Edge0NativeUnqualifiedError):
        hybrid.prepare("edge0-145b-q3k", tmp_path / "edge0", offline=True, tokenizer_policy="edge0-native-v2")
    assert edge0_preparation[1] == []


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
    progress = []

    def download(repo, **options):
        assert progress[0]["state"] == "running"
        assert progress[0]["total_bytes"] is None
        assert not any(event["state"] == "complete" for event in progress)
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
    result = hybrid.prepare("glm53-q2", output, progress=progress.append)
    assert result["status"] == "prepared"
    assert progress[-3] == {"phase": "download", "state": "complete", "completed_bytes": 4, "total_bytes": 4}
    assert progress[-2]["phase"] == "conversion" and progress[-2]["state"] == "running"
    assert progress[-1]["phase"] == "conversion" and progress[-1]["state"] == "complete"
    assert all(
        set(event)
        in (
            {"phase", "state", "completed_bytes", "total_bytes"},
            {"phase", "state", "completed_bytes", "total_bytes", "transferred_bytes"},
        )
        for event in progress
    )
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


def test_incomplete_weight_download_cannot_report_completion(tmp_path, monkeypatch):
    source = {**hybrid.SOURCES["glm53-q2"], "weight_bytes": 5}
    monkeypatch.setitem(hybrid.SOURCES, "glm53-q2", source)

    def download(repo, **options):
        root = Path(options["local_dir"])
        if repo == source["weights_repo"]:
            nested = root / "UD-Q2_K_XL"
            nested.mkdir()
            for index in range(4):
                (nested / f"model-{index}.gguf").write_bytes(b"x")
        else:
            metadata(root)

    monkeypatch.setattr(hybrid, "snapshot_download", download)
    events = []
    output = tmp_path / "model"
    with pytest.raises(ValueError, match="weight size mismatch"):
        hybrid.prepare("glm53-q2", output, progress=events.append)
    assert not output.exists()
    assert not any(event["state"] == "complete" for event in events)


@pytest.mark.parametrize("failure", [False, True])
def test_preparation_cli_uses_and_closes_private_progress_pipe(tmp_path, monkeypatch, capsys, failure):
    read_descriptor, write_descriptor = os.pipe()
    sinks = []
    sink_type = hybrid.FDProgressSink

    def sink_factory(descriptor):
        sink = sink_type(descriptor)
        sinks.append(sink)
        return sink

    def prepare(choice, output, *, offline, progress):
        assert choice == "glm53-q2" and output == tmp_path / "model" and offline
        progress({"phase": "download", "state": "running", "completed_bytes": 0, "total_bytes": None})
        if failure:
            raise RuntimeError("private upstream exception")
        return {"status": "prepared"}

    monkeypatch.setattr(hybrid, "FDProgressSink", sink_factory)
    monkeypatch.setattr(hybrid, "prepare", prepare)
    try:
        result = hybrid.main(
            [
                "glm53-q2",
                "--output",
                str(tmp_path / "model"),
                "--offline",
                "--progress-fd",
                str(write_descriptor),
            ]
        )
        assert result == (1 if failure else 0)
        assert len(sinks) == 1 and sinks[0].descriptor is None
        os.close(write_descriptor)
        write_descriptor = None
        event = json.loads(os.read(read_descriptor, 65536))
        assert event == {"phase": "download", "state": "running", "completed_bytes": 0, "total_bytes": None}
        captured = capsys.readouterr()
        assert captured.out == ("" if failure else '{"status": "prepared"}\n')
        assert "private upstream exception" not in captured.err
    finally:
        os.close(read_descriptor)
        if write_descriptor is not None:
            os.close(write_descriptor)


@pytest.mark.parametrize("weight_bytes", [4, 5])
def test_real_native_preparation_child_reports_progress_and_rejects_incomplete_weights(tmp_path, weight_bytes):
    from stark_convert.desktop import Desktop

    # The child runs the actual CLI and preparation code. Only the network
    # boundary and pinned byte total are replaced with a four-byte checkpoint.
    bootstrap = textwrap.dedent("""\
        import json
        import sys
        from pathlib import Path
        from stark_convert import hybrid_prepare as hybrid

        source = {**hybrid.SOURCES["glm53-q2"], "weight_bytes": int(sys.argv[2])}
        hybrid.SOURCES["glm53-q2"] = source

        def snapshot(repo, **options):
            assert options["local_files_only"] is True
            root = Path(options["local_dir"])
            if repo == source["weights_repo"]:
                assert options["revision"] == source["weights_revision"]
                nested = root / "UD-Q2_K_XL"
                nested.mkdir()
                for index in range(4):
                    (nested / f"model-{index}.gguf").write_bytes(b"x")
                with options["tqdm_class"](
                    name="huggingface_hub.snapshot_download", unit="B", total=4
                ) as bar:
                    bar.update(2)
                with options["tqdm_class"](
                    name="huggingface_hub.snapshot_download.transfer", unit="B", total=4
                ) as transfer:
                    transfer.update(8)
            else:
                assert repo == source["metadata_repo"]
                assert options["revision"] == source["metadata_revision"]
                config = {"model_type": "glm5_next", "text_config": {"vocab_size": 8, "eos_token_id": 1}}
                (root / "config.json").write_text(json.dumps(config))
                (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages[0].content }}"}))
                (root / "tokenizer.json").write_text("{}")

        hybrid.snapshot_download = snapshot
        raise SystemExit(hybrid.main(["glm53-q2", "--output", sys.argv[1], "--offline", *sys.argv[3:]]))
    """)
    controller = Desktop(tmp_path / "unused-worker", tmp_path / "library", io.StringIO(), timeout=2)
    output = tmp_path / "prepared"
    try:
        controller.begin_install_progress(controller.epoch, "native")
        result, failure_category = controller.run_converter(
            [sys.executable, "-I", "-B", "-c", bootstrap, str(output), str(weight_bytes)],
            dict(os.environ, HF_HUB_OFFLINE="1"),
            controller.epoch,
        )
        assert controller.converter is None
        assert failure_category is None
        assert controller.models == [] and controller.worker is None
        events = [json.loads(line) for line in controller.output.getvalue().splitlines()]
        assert all(event["event"] in {"progress", "status"} for event in events)
        messages = [event["message"] for event in events if event["event"] == "status"]
        assert messages[0] == "Downloading model files…"
        assert any(
            event["event"] == "progress"
            and event.get("phase") == "download"
            and event.get("state") == "running"
            and event.get("completed_bytes") == 0
            and event.get("transferred_bytes") == 8
            for event in events
        )
        assert str(output) not in controller.output.getvalue()
        if weight_bytes == 4:
            assert messages[-1] == "Preparing model…"
            assert result == 0 and output.is_dir()
            assert len(list(output.glob("*.gguf"))) == 4
            assert json.loads((output / "native-source.json").read_bytes())["weight_bytes"] == 4
            assert controller.install_progress["download"] == {
                "state": "complete",
                "completed_bytes": 4,
                "total_bytes": 4,
                "transferred_bytes": 8,
            }
            # Desktop reserves completion until its own publication checks pass.
            assert controller.install_progress["conversion"]["state"] == "running"
        else:
            assert result == 1 and not output.exists()
            assert output.with_name(output.name + ".native-download").is_dir()
            assert all(event["state"] != "complete" for event in events)
    finally:
        controller.stop(quiet=True)
