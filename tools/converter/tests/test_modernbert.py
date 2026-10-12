"""Pinned full classifier exports and fail-closed ModernBERT planning."""

import hashlib
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from test_streaming import save_tensors, tensor_bytes

from stark_convert.adapters.modernbert import build_modernbert_plan
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header, write_plan

DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}
CASES = {
    "default": {},
    "attention_bias": {"attention_bias": True},
    "classifier_bias": {"classifier_bias": True},
    "both_biases": {"attention_bias": True, "classifier_bias": True},
    "unused_mlm_decoder_bias": {"decoder_bias": True},
    "global_every_layer": {"global_attn_every_n_layers": 1},
    "alternating_global": {"global_attn_every_n_layers": 2, "num_hidden_layers": 3},
    "single_layer": {"num_hidden_layers": 1},
    "numeric_label_order": {"labels": 12},
    "norm_float32": {"norm_dtype": "F32", "norm_eps": 2e-4, "layer_norm_eps": 3e-5},
    "dropout_annotations": {
        "attention_dropout": 0.1,
        "classifier_dropout": 0.2,
        "embedding_dropout": 0.3,
        "mlp_dropout": 0.4,
    },
}
REFERENCE_PATH = Path(__file__).parent / "fixtures" / "modernbert_exports.json"
REFERENCE = json.loads(REFERENCE_PATH.read_text())


def modernbert_fixture(source, case="default", *, dtype="BF16", sharded=False):
    source.mkdir()
    flags = CASES[case]
    config = {
        "architectures": ["ModernBertForSequenceClassification"],
        "model_type": "modernbert",
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "vocab_size": 16,
        "max_position_embeddings": 128,
        "global_attn_every_n_layers": 3,
        "global_rope_theta": 160000.0,
        "local_rope_theta": 10000.0,
        "local_attention": 16,
        "norm_eps": 1e-5,
        "layer_norm_eps": 1e-5,
        "attention_bias": False,
        "classifier_bias": False,
        "mlp_bias": False,
        "norm_bias": False,
        "decoder_bias": False,
        "attention_dropout": 0.0,
        "classifier_dropout": 0.0,
        "embedding_dropout": 0.0,
        "mlp_dropout": 0.0,
        "hidden_activation": "gelu",
        "classifier_activation": "gelu",
        "classifier_pooling": "mean",
        "position_embedding_type": "absolute",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "cls_token_id": 1,
        "sep_token_id": 2,
        "pad_token_id": 0,
        "deterministic_flash_attn": False,
        "gradient_checkpointing": False,
        "initializer_cutoff_factor": 2.0,
        "initializer_range": 0.02,
        "torch_dtype": DTYPES[dtype],
        "transformers_version": "4.51.3",
    }
    config.update({key: value for key, value in flags.items() if key not in {"norm_dtype", "labels"}})
    labels = flags.get("labels", 2)
    # Deliberately reverse insertion order; row order follows numeric IDs.
    config["id2label"] = {str(index): f"label_{index}" for index in reversed(range(labels))}
    config["label2id"] = {f"label_{index}": index for index in reversed(range(labels))}
    tensors = {}

    def add(name, shape, *, norm=False):
        ty = flags.get("norm_dtype", dtype) if norm else dtype
        values = (np.arange(np.prod(shape), dtype="<f4").reshape(shape) % 31 - 15) / 8 + len(tensors) / 16
        raw = {
            "BF16": lambda x: x.astype(ml_dtypes.bfloat16).view("<u2"),
            "F16": lambda x: x.astype("<f2"),
            "F32": lambda x: x,
        }[ty](values)
        tensors[name] = (ty, raw)

    model, hidden = config["hidden_size"], config["intermediate_size"]
    add("model.embeddings.tok_embeddings.weight", (config["vocab_size"], model))
    add("model.embeddings.norm.weight", (model,), norm=True)
    for index in range(config["num_hidden_layers"]):
        src = f"model.layers.{index}"
        if index:
            add(src + ".attn_norm.weight", (model,), norm=True)
        add(src + ".mlp_norm.weight", (model,), norm=True)
        add(src + ".attn.Wqkv.weight", (3 * model, model))
        if config["attention_bias"]:
            add(src + ".attn.Wqkv.bias", (3 * model,))
        add(src + ".attn.Wo.weight", (model, model))
        add(src + ".mlp.Wi.weight", (2 * hidden, model))
        add(src + ".mlp.Wo.weight", (model, hidden))
    add("model.final_norm.weight", (model,), norm=True)
    add("head.dense.weight", (model, model))
    if config["classifier_bias"]:
        add("head.dense.bias", (model,))
    add("head.norm.weight", (model,), norm=True)
    add("classifier.weight", (labels, model))
    add("classifier.bias", (labels,))
    (source / "config.json").write_text(json.dumps(config))
    if sharded:
        mapping = {}
        for index in range(2):
            filename = f"model-{index}.safetensors"
            shard = dict(list(tensors.items())[index::2])
            save_tensors(source / filename, shard)
            mapping.update(dict.fromkeys(shard, filename))
        (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    else:
        save_tensors(source / "model.safetensors", tensors)
    return config, tensors


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("source_dtype", DTYPES)
@pytest.mark.parametrize("target_dtype", DTYPES)
@pytest.mark.parametrize("sharded", [False, True])
def test_modernbert_actual_classifier_exports(tmp_path, case, source_dtype, target_dtype, sharded):
    source = tmp_path / "source"
    _, inputs = modernbert_fixture(source, case, dtype=source_dtype, sharded=sharded)
    frozen = REFERENCE["cases"][f"{case}_{source_dtype}_{target_dtype}"]
    plan = build_modernbert_plan(source, dtype=DTYPES[target_dtype])
    assert plan.decoder_config == frozen["classifier_config"]
    for name, (ty, raw) in inputs.items():
        assert {"dtype": ty, "shape": list(raw.shape), "sha256": hashlib.sha256(raw.tobytes()).hexdigest()} == frozen[
            "inputs"
        ][name]
    output = tmp_path / "converted.safetensors"
    stats = write_plan(plan, output, chunk_bytes=128)
    metadata, tensors = read_safetensors_header(output)
    assert set(tensors) == set(frozen["arrays"])
    for name, info in tensors.items():
        assert {
            "dtype": info.dtype,
            "shape": list(info.shape),
            "sha256": hashlib.sha256(tensor_bytes(info)).hexdigest(),
        } == frozen["arrays"][name]
    assert {name: json.loads(value) for name, value in metadata.items()} == frozen["metadata"]
    assert plan.source_tensors == len(inputs) == len(plan.input_tensors)
    assert stats["peak_buffer_bytes"] <= 128
    assert stats["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def rewrite(source, config, tensors=None):
    (source / "config.json").write_text(json.dumps(config))
    if tensors is not None:
        save_tensors(source / "model.safetensors", tensors)


@pytest.mark.parametrize(
    "corruption", ["missing", "shape", "extra", "norm_bias", "layer0_norm", "mlp_bias", "readout_bias", "mlm_decoder"]
)
def test_modernbert_tensor_contract_precedes_unknown_semantics(tmp_path, corruption):
    source = tmp_path / "source"
    config, tensors = modernbert_fixture(source)
    config["unknown_semantics"] = True
    if corruption == "missing":
        tensors.pop("model.layers.0.mlp.Wi.weight")
    elif corruption == "shape":
        ty, raw = tensors["model.layers.0.mlp.Wi.weight"]
        tensors["model.layers.0.mlp.Wi.weight"] = (ty, raw[:-1])
    elif corruption == "extra":
        tensors["unconsumed.weight"] = tensors["head.dense.weight"]
    elif corruption == "readout_bias":
        tensors.pop("classifier.bias")
    else:
        key = {
            "norm_bias": "head.norm.bias",
            "layer0_norm": "model.layers.0.attn_norm.weight",
            "mlp_bias": "model.layers.0.mlp.Wi.bias",
            "mlm_decoder": "decoder.bias",
        }[corruption]
        tensors[key] = tensors["model.final_norm.weight"]
    rewrite(source, config, tensors)
    with pytest.raises(ConversionError):
        build_modernbert_plan(source)


@pytest.mark.parametrize(
    "key,value",
    [
        ("hidden_size", True),
        ("hidden_size", 0),
        ("num_attention_heads", 3),
        ("num_attention_heads", 8),
        ("num_hidden_layers", 0),
        ("intermediate_size", -1),
        ("global_attn_every_n_layers", 0),
        ("local_attention", False),
        ("norm_eps", 0.0),
        ("norm_eps", float("inf")),
        ("global_rope_theta", True),
        ("attention_bias", 1),
        ("classifier_bias", None),
        ("decoder_bias", 1),
        ("attention_dropout", -0.1),
        ("mlp_dropout", True),
        ("embedding_dropout", 1.1),
        ("classifier_dropout", "0"),
        ("num_labels", True),
        ("head_dim", True),
        ("num_key_value_heads", True),
    ],
)
def test_modernbert_malformed_config(tmp_path, key, value):
    source = tmp_path / "source"
    config, _ = modernbert_fixture(source)
    config[key] = value
    rewrite(source, config)
    with pytest.raises(ConversionError):
        build_modernbert_plan(source)


@pytest.mark.parametrize(
    "corruption", ["missing", "gap", "noncanonical", "duplicate", "inverse", "bool", "extra", "count"]
)
def test_modernbert_labels_fail_closed(tmp_path, corruption):
    source = tmp_path / "source"
    config, _ = modernbert_fixture(source)
    if corruption == "missing":
        config.pop("label2id")
    elif corruption == "gap":
        config["id2label"]["2"] = config["id2label"].pop("1")
    elif corruption == "noncanonical":
        config["id2label"]["01"] = config["id2label"].pop("1")
    elif corruption == "duplicate":
        config["id2label"]["1"] = config["id2label"]["0"]
    elif corruption == "inverse":
        config["label2id"]["label_0"] = 1
    elif corruption == "bool":
        config["label2id"]["label_0"] = False
    elif corruption == "extra":
        config["label2id"]["unknown"] = 3
    else:
        config["num_labels"] = 3
    rewrite(source, config)
    with pytest.raises(ConversionError):
        build_modernbert_plan(source)


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_type", "bert"),
        ("architectures", ["ModernBertForMaskedLM"]),
        ("hidden_activation", "gelu_new"),
        ("classifier_activation", "relu"),
        ("classifier_pooling", "cls"),
        ("position_embedding_type", "relative"),
        ("norm_bias", True),
        ("mlp_bias", True),
        ("is_decoder", True),
        ("add_cross_attention", True),
        ("quantization", {"bits": 4, "group_size": 16}),
        ("quantization_config", {"quant_method": "awq"}),
        ("rope_scaling", {"type": "linear", "factor": 2.0}),
        ("num_key_value_heads", 1),
        ("head_dim", 8),
        ("pruned_heads", {"0": [0]}),
        ("unknown_semantics", True),
    ],
)
def test_modernbert_unsupported_config(tmp_path, key, value):
    source = tmp_path / "source"
    config, _ = modernbert_fixture(source)
    config[key] = value
    rewrite(source, config)
    with pytest.raises(UnsupportedConversion):
        build_modernbert_plan(source)


def test_modernbert_packed_tensor_dtype_rejected(tmp_path):
    source = tmp_path / "source"
    config, tensors = modernbert_fixture(source)
    _, raw = tensors["model.layers.0.mlp.Wi.weight"]
    tensors["model.layers.0.mlp.Wi.weight"] = ("U32", raw.astype("<u4"))
    rewrite(source, config, tensors)
    with pytest.raises(UnsupportedConversion):
        build_modernbert_plan(source)


def test_modernbert_missing_pooling_is_not_inferred_as_mean(tmp_path):
    source = tmp_path / "source"
    config, _ = modernbert_fixture(source)
    config.pop("classifier_pooling")
    rewrite(source, config)
    with pytest.raises(UnsupportedConversion):
        build_modernbert_plan(source)


@pytest.mark.parametrize(
    "case,key",
    [("attention_bias", "model.layers.0.attn.Wqkv.bias"), ("classifier_bias", "head.dense.bias")],
)
def test_modernbert_declared_bias_requires_complete_tensor(tmp_path, case, key):
    source = tmp_path / "source"
    config, tensors = modernbert_fixture(source, case)
    tensors.pop(key)
    rewrite(source, config, tensors)
    with pytest.raises(ConversionError):
        build_modernbert_plan(source)


def test_modernbert_body_semantics_and_original_signatures(tmp_path):
    source = tmp_path / "source"
    _, tensors = modernbert_fixture(source, "alternating_global")
    plan = build_modernbert_plan(source)
    config = plan.decoder_config
    layers = config["transformer_config"]["layer_configs"]
    assert layers[0]["pre_mixer_norm_config"] is None
    assert [layer["mixer_config"]["sliding_window_size"] for layer in layers] == [None, 16, None]
    assert [layer["rope_config"]["base"] for layer in layers] == [160000.0, 10000.0, 160000.0]
    assert all(not layer["mixer_config"]["is_causal"] for layer in layers)
    assert all(layer["mlp_config"]["activation"] == {"type": "GELU", "approximate": False} for layer in layers)
    assert config["embedding_norm_config"]["subtract_mean"] is True
    assert config["classifier_pooling"] == "mean"
    assert config["output_labels"] == ["label_0", "label_1"]
    original = next(
        info for info in plan.input_tensors if info.shape == tensors["model.layers.0.mlp.Wi.weight"][1].shape
    )
    op = next(
        op for op in plan.operations if op.name == "classifier.transformer.layers.0.mlp.up_projection.weights.weights"
    )
    assert sum(info.nbytes for info in op.sources) == original.nbytes
    assert op.sources[0].offset > op.sources[1].offset
    assert all(info.file_signature == original.file_signature for info in op.sources)


def test_modernbert_invalid_output_dtype(tmp_path):
    source = tmp_path / "source"
    modernbert_fixture(source)
    with pytest.raises(UnsupportedConversion):
        build_modernbert_plan(source, dtype="float64")
