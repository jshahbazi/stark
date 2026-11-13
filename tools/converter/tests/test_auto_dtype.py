"""`--dtype auto`: F16 only for all-F16 MLX affine Qwen2/Mistral sources.

Model-free fixtures. They check the selection rule, its receipts and its
equivalence with explicit dtypes; they do not qualify any checkpoint.
"""

import json
import sys

import pytest
from test_dense_adapters import dense_fixture
from test_mlx import _float_values, mlx_qwen_fixture, save_mlx_fixture
from test_mlx_dense import mlx_dense_fixture
from test_mlx_qwen2_mistral import mlx_family_fixture
from tokenizers import Tokenizer, models

from stark_convert import cli
from stark_convert.model_support import AUTO_F16_FAMILIES, NATIVE_F16_FAMILIES, auto_f16_refusal
from stark_convert.sources import resolve_source
from stark_convert.streaming import UnsupportedConversion, read_safetensors_header

FUTURE = "future/checkpoint"


def with_metadata(path):
    tokenizer = Tokenizer(models.WordLevel({f"[SYNTHETIC_{i}]": i for i in range(32)}, unk_token="[SYNTHETIC_0]"))
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    return path


def rewrite_config(path, **changes):
    config = json.loads((path / "config.json").read_text())
    config.update(changes)
    (path / "config.json").write_text(json.dumps(config))


def source(tmp_path, case):
    """Build one named synthetic source; every MLX case declares `torch_dtype: bfloat16`."""
    path = tmp_path / ("source-" + case)
    family = case.split("-")[0]
    if case in {"qwen2-f16", "mistral-f16", "qwen2-bf16", "qwen2-mixed", "qwen2-window", "mistral-window"}:
        dtype = "BF16" if case.endswith("bf16") else "F16"
        _, tensors = mlx_family_fixture(
            path, family, dtype=dtype, tied=family == "qwen2", sliding_window=4 if "window" in case else None
        )
        if case == "qwen2-mixed":
            tensors["model.norm.weight"] = "F32", _float_values((128,), "F32")
            save_mlx_fixture(path, tensors)
    elif case == "llama-f16":
        mlx_dense_fixture(path, "llama", dtype="F16")
    elif case == "qwen3-f16":
        mlx_qwen_fixture(path, dtype="F16")
    elif case == "qwen2-dense-f16":
        dense_fixture(path, "qwen2", dtype="F16")
        rewrite_config(path, use_sliding_window=False)
    else:
        raise AssertionError(case)
    if case != "qwen2-dense-f16":
        # The declared dtype is never consulted; only tensor headers decide.
        rewrite_config(path, torch_dtype="bfloat16")
    return with_metadata(path)


def convert(tmp_path, path, name, **kwargs):
    snapshot = resolve_source(path, repo=FUTURE, metadata_profile="hf-generic")
    return cli.convert(snapshot, tmp_path / name, repo=FUTURE, chunk_bytes=4096, **kwargs)


def floating_dtypes(path):
    _, tensors = read_safetensors_header(path / "model.safetensors")
    return {info.dtype for info in tensors.values() if info.dtype not in {"U8", "I8", "U32", "I32"}}


def test_auto_f16_families_are_a_subset_of_the_native_f16_profile():
    assert AUTO_F16_FAMILIES == {"qwen2", "mistral"}
    assert AUTO_F16_FAMILIES < NATIVE_F16_FAMILIES and "llama" not in AUTO_F16_FAMILIES


@pytest.mark.parametrize("case", ["qwen2-f16", "mistral-f16"])
def test_all_f16_mlx_source_selects_f16_identical_to_explicit_float16(tmp_path, case):
    path = source(tmp_path, case)
    receipt = convert(tmp_path, path, "auto", dtype="auto")
    assert receipt["dtype_selection"] == {
        "requested": "auto",
        "selected": "float16",
        "reason": "all_f16_mlx_affine_source",
    }
    assert receipt["capability"]["native_loading"] == "unverified"
    assert receipt["native_qualification"] is None
    # Norm scales stay F32 as in every dense output; nothing is BF16.
    assert floating_dtypes(tmp_path / "auto") == {"F16", "F32"}
    explicit = convert(tmp_path, path, "explicit", dtype="float16")
    assert "dtype_selection" not in explicit
    for key in ("output_sha256", "config_sha256", "tokenizer_sha256", "source_manifest_sha256"):
        assert receipt[key] == explicit[key]


@pytest.mark.parametrize(
    "case,reason",
    [
        ("qwen2-bf16", "source_not_all_f16"),
        ("qwen2-mixed", "source_not_all_f16"),
        ("qwen2-window", "native_f16_not_admitted"),
        ("mistral-window", "native_f16_not_admitted"),
        ("qwen3-f16", "family_not_qualified_for_f16"),
        ("llama-f16", "family_not_qualified_for_f16"),
        ("qwen2-dense-f16", "source_not_mlx_affine"),
    ],
)
def test_every_other_source_keeps_the_bf16_output_byte_for_byte(tmp_path, case, reason):
    path = source(tmp_path, case)
    receipt = convert(tmp_path, path, "auto", dtype="auto")
    assert receipt["dtype_selection"] == {"requested": "auto", "selected": "bfloat16", "reason": reason}
    assert "F16" not in floating_dtypes(tmp_path / "auto")
    explicit = convert(tmp_path, path, "explicit", dtype="bfloat16")
    assert "dtype_selection" not in explicit
    for key in ("output_sha256", "config_sha256", "tokenizer_sha256", "capability"):
        assert receipt[key] == explicit[key]


def test_explicit_dtype_always_wins(tmp_path):
    f16 = source(tmp_path, "qwen2-f16")
    convert(tmp_path, f16, "bf16", dtype="bfloat16")
    assert floating_dtypes(tmp_path / "bf16") == {"BF16", "F32"}
    bf16 = source(tmp_path, "qwen2-bf16")
    receipt = convert(tmp_path, bf16, "f16", dtype="float16")
    assert "dtype_selection" not in receipt
    assert floating_dtypes(tmp_path / "f16") == {"F16", "F32"}


def test_recorded_qualification_keeps_bf16(tmp_path, monkeypatch):
    monkeypatch.setattr("stark_convert.qualifications.has_recorded_qualification", lambda repo: repo == FUTURE)
    receipt = convert(tmp_path, source(tmp_path, "qwen2-f16"), "auto", dtype="auto")
    assert receipt["dtype_selection"]["selected"] == "bfloat16"
    assert receipt["dtype_selection"]["reason"] == "qualified_bf16_artifact"


def test_every_bf16_qualified_repository_is_recognized():
    from stark_convert.qualifications import _load_qualifications, has_recorded_qualification

    repositories = {record["repo"] for record in _load_qualifications()}
    assert {
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "mlx-community/Llama-3.2-1B-Instruct-8bit",
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    } <= repositories
    assert all(has_recorded_qualification(repo) for repo in repositories)
    assert not has_recorded_qualification(FUTURE) and not has_recorded_qualification(None)


@pytest.mark.parametrize("mode", ["text", "decision"])
def test_decision_mode_shares_the_text_f16_admission(tmp_path, mode):
    receipt = convert(tmp_path, source(tmp_path, "qwen2-f16"), "auto", dtype="auto", support_mode=mode)
    assert receipt["dtype_selection"]["selected"] == "float16"


def test_refusal_rule_order_and_scope():
    config = {"model_type": "qwen2"}
    arguments = dict(mode="text", artifact_mode="text", model_kind="language", require_native_support=False)
    f16, specs = ("U32", "F16", "F16"), ("MLXSpec", "MLXSpec")
    assert auto_f16_refusal("qwen2", config, f16, specs, qualified=False, **arguments) is None
    assert (
        auto_f16_refusal("qwen2", config, f16, specs, qualified=False, **{**arguments, "require_native_support": True})
        == "native_qualification_requires_bf16"
    )
    assert (
        auto_f16_refusal("qwen2", config, f16, specs, qualified=False, **{**arguments, "mode": "vision-language"})
        == "unsupported_mode"
    )
    assert (
        auto_f16_refusal("qwen2", config, f16, specs, qualified=False, **{**arguments, "model_kind": "classifier"})
        == "unsupported_mode"
    )
    assert auto_f16_refusal("qwen2", config, f16, specs, qualified=True, **arguments) == "qualified_bf16_artifact"
    assert auto_f16_refusal("qwen2", config, ("U32",), specs, qualified=False, **arguments) == "source_not_all_f16"
    assert (
        auto_f16_refusal("qwen2", config, ("F16", "BF16"), specs, qualified=False, **arguments) == "source_not_all_f16"
    )
    assert (
        auto_f16_refusal("qwen2", config, f16, ("MLXSpec", "FullPrecisionSpec"), qualified=False, **arguments)
        == "source_not_mlx_affine"
    )
    assert auto_f16_refusal("qwen2", config, f16, (), qualified=False, **arguments) == "source_not_mlx_affine"
    windowed = {"model_type": "mistral", "sliding_window": 4096}
    assert auto_f16_refusal("mistral", windowed, f16, specs, qualified=False, **arguments) == "native_f16_not_admitted"
    assert (
        auto_f16_refusal("llama", {"model_type": "llama"}, f16, specs, qualified=False, **arguments)
        == "family_not_qualified_for_f16"
    )


def test_auto_with_required_native_support_is_refused_like_bf16(tmp_path):
    path = source(tmp_path, "qwen2-f16")
    snapshot = resolve_source(path, repo=FUTURE, metadata_profile="hf-generic")
    for dtype in ("auto", "bfloat16"):
        with pytest.raises(UnsupportedConversion, match="Generic conversion has no recorded native qualification"):
            cli.convert(snapshot, tmp_path / dtype, repo=FUTURE, dtype=dtype, require_native_support=True)


def run_cli(monkeypatch, capsys, *arguments):
    monkeypatch.setattr(sys, "argv", ["stark-convert", *arguments])
    code = cli.main()
    captured = capsys.readouterr()
    return code, json.loads(captured.out) if code == 0 else json.loads(captured.err)


@pytest.mark.parametrize("case,selected", [("qwen2-f16", "float16"), ("qwen2-bf16", "bfloat16")])
def test_cli_auto_preflight_and_conversion(tmp_path, monkeypatch, capsys, case, selected):
    path = source(tmp_path, case)
    common = ("--metadata-profile", "hf-generic", "--support-mode", "text", "--offline")
    code, auto = run_cli(monkeypatch, capsys, str(path), "--preflight", "--dtype", "auto", *common[2:])
    assert code == 0 and auto["status"] == "conditional"
    code, default = run_cli(monkeypatch, capsys, str(path), "--preflight", *common[2:])
    assert auto == default
    code, receipt = run_cli(
        monkeypatch, capsys, str(path), "--output", str(tmp_path / "output"), "--dtype", "auto", *common
    )
    assert code == 0 and receipt["dtype_selection"]["selected"] == selected
    assert json.loads((tmp_path / "output/conversion.json").read_text())["dtype_selection"]["selected"] == selected
