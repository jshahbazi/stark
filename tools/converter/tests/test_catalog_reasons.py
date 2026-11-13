"""Unavailable-row wording: foreign quantization, gated sources and importer messages."""

from types import SimpleNamespace

import pytest
from test_model_catalog import Hub

from stark_convert import catalog_reasons, hub_headers, planning, qwen38_external
from stark_convert import model_catalog as catalog
from stark_convert.streaming import ConversionError, UnsupportedConversion

FOREIGN = {
    "awq": ("AWQ", {"bits": 4, "group_size": 128, "zero_point": True, "version": "gemm"}),
    "gptq": ("GPTQ", {"bits": 4, "group_size": 128, "desc_act": False}),
    "bitsandbytes": ("bitsandbytes", {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}),
    "fp8": ("FP8", {"activation_scheme": "dynamic", "weight_block_size": [128, 128]}),
    "compressed-tensors": ("compressed-tensors", {"format": "float-quantized", "config_groups": {}}),
    "modelopt": ("ModelOpt", {"quant_algo": "NVFP4"}),
    "auto-round": ("AutoRound", {"bits": 4, "group_size": 128}),
    "torchao": ("torchao", {"quant_type": {}}),
    "quark": ("Quark", {"global_quant_config": {}}),
    "glq": ("glq", {"bits": 3}),
}


@pytest.fixture(autouse=True)
def isolated_catalog(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(catalog, "_catalog_seeds", lambda _family: [])
    monkeypatch.setattr(catalog, "get_model", lambda _repo: None)


@pytest.fixture
def headers(monkeypatch):
    """Record header reads and answer them with an empty, readable inventory."""
    calls = []
    state = {"error": None}

    def read(hub, repo, *args):
        calls.append(repo)
        if state["error"] is not None:
            raise state["error"]
        return {}

    monkeypatch.setattr(hub_headers, "checkpoint_headers", read)
    return SimpleNamespace(calls=calls, state=state)


@pytest.fixture
def planner(monkeypatch):
    """Replace the tensor planner with one that accepts or raises ``state['error']``."""
    state = {"error": None}

    def plan(*args, **kwargs):
        if state["error"] is not None:
            raise state["error"]

    monkeypatch.setattr(planning, "plan_from_headers", plan)
    return state


def single(hub, tmp_path, config, *, repo="someone/Model", gated=None, family="qwen3", cache_dir=None):
    hub.add(tmp_path / repo.replace("/", "--"), repo, config)
    if gated is not None:
        hub.repos[repo][1].gated = gated
    hub.discovered[family] = [repo]
    report = catalog.family_catalog(family, hub=hub, cache_dir=cache_dir)
    assert report["status"] == "ready", report["message"]
    (item,) = report["items"]
    return item


def assert_displayable(reason):
    assert isinstance(reason, str)
    assert 0 < len(reason.encode("utf-8")) <= 240
    assert not any(ord(char) < 32 or ord(char) == 127 for char in reason)


@pytest.mark.parametrize("method", sorted(FOREIGN))
def test_foreign_quantization_is_named_and_never_reads_headers(tmp_path, headers, planner, method):
    badge, fields = FOREIGN[method]
    config = {
        "model_type": "qwen3",
        "torch_dtype": "bfloat16",
        "quantization_config": {"quant_method": method, **fields},
    }
    item = single(Hub(), tmp_path, config)
    assert item["downloadable"] is False
    assert item["quantization"] == badge
    assert item["unavailable_reason"] == (
        f"This checkpoint is quantized with {badge} for GPU inference servers. "
        "Stark imports this model's original weights or its MLX exports."
    )
    assert_displayable(item["unavailable_reason"])
    assert headers.calls == []


def test_foreign_reason_replaces_an_unrelated_preflight_message(tmp_path, headers, planner):
    # Preflight would otherwise report the custom-code rule first.
    config = {
        "model_type": "qwen3",
        "auto_map": {"AutoModelForCausalLM": "modeling.Model"},
        "quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128},
    }
    item = single(Hub(), tmp_path, config)
    assert item["quantization"] == "AWQ" and "AWQ" in item["unavailable_reason"]
    assert headers.calls == []


def test_foreign_rule_reads_text_config_and_names_unknown_methods_safely():
    nested = {"text_config": {"quantization_config": {"quant_method": "hqq"}}}
    assert catalog_reasons.foreign_format(nested)[0] == "HQQ"
    assert catalog_reasons.foreign_format({"quantization_config": {"quant_method": "newq_v2"}})[0] == "newq_v2"
    badge, reason = catalog_reasons.foreign_format({"quantization_config": {"quant_method": "bad\nmethod"}})
    assert badge == "Other format" and "another method" in reason
    assert_displayable(reason)
    assert catalog_reasons.foreign_format({"quantization_config": {"quant_method": 4}}) is None


@pytest.mark.parametrize(
    "quantization",
    [
        {"quantization": {"bits": 4, "group_size": 64}},
        {"quantization_config": {"bits": 4, "group_size": 64}},
        {"quantization": {"bits": 4, "group_size": 64}, "quantization_config": {"bits": 4, "group_size": 64}},
        {"quantization_config": {"bits": 4, "group_size": 64, "quant_method": "mlx"}},
        {
            "quantization": {"bits": 4, "group_size": 64, "quant_method": "mlx"},
            "quantization_config": {"bits": 4, "group_size": 64, "quant_method": "mlx"},
        },
    ],
)
def test_mlx_exports_are_not_foreign(quantization):
    assert catalog_reasons.foreign_format({"model_type": "qwen3", **quantization}) is None


def test_mlx_export_row_keeps_its_bit_width_badge(tmp_path, headers, planner):
    config = {
        "model_type": "qwen3",
        "quantization": {"bits": 4, "group_size": 64},
        "quantization_config": {"bits": 4, "group_size": 64},
    }
    item = single(Hub(), tmp_path, config)
    assert item["quantization"] == "4-bit" and item["downloadable"] is True
    assert headers.calls == ["someone/Model"]


@pytest.mark.parametrize("gated", ["manual", "auto", True])
def test_gated_header_failure_asks_for_access_and_is_checked_again(tmp_path, headers, planner, gated):
    headers.state["error"] = hub_headers.HeaderAccessError()
    hub = Hub()
    for attempt in range(2):
        if attempt:
            hub.repos.clear()
        item = single(hub, tmp_path / str(attempt), {"model_type": "qwen3"}, gated=gated, cache_dir=tmp_path / "cache")
        assert item["downloadable"] is False
        assert item["unavailable_reason"] == catalog_reasons.GATED
        assert_displayable(item["unavailable_reason"])
    # The access verdict is never stored, so the second page checks again.
    assert headers.calls == ["someone/Model", "someone/Model"]


def test_gated_source_with_granted_access_is_unchanged(tmp_path, headers, planner):
    item = single(Hub(), tmp_path, {"model_type": "qwen3"}, gated="manual")
    assert item["downloadable"] is True and "unavailable_reason" not in item


def test_gated_header_budget_keeps_its_own_reason(tmp_path, headers, planner):
    headers.state["error"] = hub_headers.HeaderBudgetError()
    item = single(Hub(), tmp_path, {"model_type": "qwen3"}, gated="manual")
    assert "verification limit" in item["unavailable_reason"]


@pytest.mark.parametrize("error", [hub_headers.HeaderAccessError(), OSError("network")])
def test_non_gated_header_failure_is_unchanged(tmp_path, headers, planner, error):
    headers.state["error"] = error
    hub = Hub()
    for attempt in range(2):
        if attempt:
            hub.repos.clear()
        item = single(hub, tmp_path / str(attempt), {"model_type": "qwen3"}, gated=False, cache_dir=tmp_path / "cache")
        assert item["unavailable_reason"] == qwen38_external.UNVERIFIED
        assert type(item["unavailable_reason"]) is str
    assert headers.calls == ["someone/Model", "someone/Model"]


def test_gated_planner_fault_is_not_described_as_an_access_problem(tmp_path, headers, planner):
    planner["error"] = RuntimeError("unexpected")
    item = single(Hub(), tmp_path, {"model_type": "qwen3"}, gated="manual")
    assert item["unavailable_reason"] == qwen38_external.UNVERIFIED


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (
            UnsupportedConversion("RoPE scaling is outside qualified dense coverage"),
            "RoPE scaling is outside qualified dense coverage.",
        ),
        (ConversionError("Granite requires tied word embeddings"), "Granite requires tied word embeddings."),
        (ConversionError("Dense checkpoint has unexpected tensors."), "Dense checkpoint has unexpected tensors."),
        (ConversionError("bad\x07message"), catalog_reasons.UNSUPPORTED_LAYOUT),
        (ConversionError(""), catalog_reasons.UNSUPPORTED_LAYOUT),
        (ConversionError("two", "args"), catalog_reasons.UNSUPPORTED_LAYOUT),
        (KeyError("model.layers.0.self_attn.q_proj.weight"), catalog_reasons.UNSUPPORTED_LAYOUT),
        (ValueError("not a planner message"), catalog_reasons.UNSUPPORTED_LAYOUT),
        (qwen38_external.LayoutError("subclass"), catalog_reasons.UNSUPPORTED_LAYOUT),
    ],
)
def test_planner_messages_pass_through_only_for_importer_errors(tmp_path, headers, planner, error, reason):
    planner["error"] = error
    item = single(Hub(), tmp_path, {"model_type": "qwen3"})
    assert item["downloadable"] is False
    assert item["unavailable_reason"] == reason
    assert_displayable(item["unavailable_reason"])


def test_header_inventory_messages_pass_through(tmp_path, headers, planner):
    headers.state["error"] = ConversionError("Tensor occurs in more than one shard")
    item = single(Hub(), tmp_path, {"model_type": "qwen3"})
    assert item["unavailable_reason"] == "Tensor occurs in more than one shard."


def test_long_importer_messages_are_cut_to_the_backend_limit(tmp_path, headers, planner):
    planner["error"] = UnsupportedConversion("Gemma ö rule " * 40)
    item = single(Hub(), tmp_path, {"model_type": "qwen3"})
    reason = item["unavailable_reason"]
    assert_displayable(reason)
    assert reason.startswith("Gemma ö rule") and reason.endswith("…")
    assert catalog_reasons.bounded_reason("x" * 1000).encode("utf-8") == b"x" * 237 + "…".encode()


FIXTURES = {
    "bf16": ({"model_type": "qwen3", "torch_dtype": "bfloat16"}, True),
    "mlx-4bit": (
        {
            "model_type": "qwen3",
            "quantization": {"bits": 4, "group_size": 64},
            "quantization_config": {"bits": 4, "group_size": 64},
        },
        True,
    ),
    "mlx-8bit-config-only": ({"model_type": "qwen3", "quantization_config": {"bits": 8, "group_size": 64}}, True),
    **{
        method: ({"model_type": "qwen3", "quantization_config": {"quant_method": method, **fields}}, False)
        for method, (_badge, fields) in FOREIGN.items()
    },
}


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_downloadable_set_is_unchanged(tmp_path, headers, planner, name):
    config, downloadable = FIXTURES[name]
    item = single(Hub(), tmp_path, config)
    assert item["downloadable"] is downloadable
    if catalog_reasons.foreign_format(config) is not None:
        # The foreign rule applies only where the existing import-format check
        # already rejects the source, so it can never change availability.
        info = SimpleNamespace(safetensors={"parameters": {"BF16": 1}})
        assert catalog._quantization(config, info, None, family="qwen3") is None
