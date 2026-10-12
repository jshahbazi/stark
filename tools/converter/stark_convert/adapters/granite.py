"""Independent dense Granite planner and shared explicit tensor layout.

Config and tensor conventions are adapted from Lalamo revision
831accb8252a6b6606c378048caeb44178f1a050, MIT License.
Copyright (c) 2025 Mirai Tech Inc. See LICENSE.lalamo and source provenance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..compressed import parse_mlx_quantization, plan_mlx_matrix, read_mlx_matrix
from ..streaming import (
    ConversionError,
    ConversionPlan,
    TensorOperation,
    UnsupportedConversion,
    load_checkpoint,
    read_json,
    tensor_sha256,
)
from .dense import _COMMON_KEYS, _DTYPES, _bool, _positive_float, _positive_int

_GRANITE_KEYS = {"embedding_multiplier", "residual_multiplier", "attention_multiplier", "logits_scaling"}
_SMOLLM3_KEYS = {
    "no_rope_layers",
    "no_rope_layer_interval",
    "max_window_layers",
    "sliding_window",
    "use_sliding_window",
}


def _build_plan(source: Path, *, family: str, dtype: str) -> ConversionPlan:
    config = read_json(source / "config.json")
    if config.get("model_type") != family:
        raise UnsupportedConversion("Checkpoint model_type does not match selected adapter")
    if dtype not in _DTYPES:
        raise UnsupportedConversion("Dense adapters support BF16, F16, and F32")
    target_dtype = _DTYPES[dtype]
    unsupported: list[str] = []
    model_dim = _positive_int(config, "hidden_size")
    hidden_dim = _positive_int(config, "intermediate_size")
    layers = _positive_int(config, "num_hidden_layers")
    heads = _positive_int(config, "num_attention_heads")
    groups = _positive_int(config, "num_key_value_heads", heads)
    if heads % groups:
        raise ConversionError("Dense attention groups must divide the head count")
    if config.get("head_dim") is None:
        if model_dim % heads:
            raise ConversionError("Dense attention head dimension cannot be inferred")
        head_dim = model_dim // heads
    else:
        head_dim = _positive_int(config, "head_dim")
        if family == "smollm3" and head_dim * heads != model_dim:
            unsupported.append("SmolLM3 requires the inferred attention head dimension")
    if head_dim % 2:
        raise ConversionError("Dense rotary head dimension must be even")
    vocab = _positive_int(config, "vocab_size")
    epsilon = _positive_float(config, "rms_norm_eps", 1e-6)
    sequence_length = _positive_int(config, "max_position_embeddings", 32768)
    rope = {
        "type": "UnscaledRoPEConfig",
        "base": _positive_float(config, "rope_theta", 10000.0),
        "max_sequence_length": sequence_length,
        "head_dim": head_dim,
    }
    tied = _bool(config, "tie_word_embeddings", True)
    attention_bias = _bool(config, "attention_bias", False)
    mlp_bias = _bool(config, "mlp_bias", False)
    dropout = config.get("attention_dropout", 0.0)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout):
        raise ConversionError("Dense config requires finite attention_dropout")
    if not 0 <= dropout <= 1:
        raise ConversionError("Dense attention_dropout is outside its valid range")
    if dropout != 0:
        unsupported.append("Attention dropout variant is outside qualified dense coverage")
    # HFSmolLM3Config carries pretraining_tp as training metadata and does not
    # change its decoder or checkpoint load path for values above one.
    if _positive_int(config, "pretraining_tp", 1) != 1 and family == "granite":
        unsupported.append("Tensor-parallel pretraining config is outside qualified dense coverage")
    architecture = "GraniteForCausalLM" if family == "granite" else "SmolLM3ForCausalLM"
    if config.get("architectures", [architecture]) != [architecture]:
        unsupported.append("Checkpoint architecture is outside qualified dense coverage")
    activations = {"silu"} if family == "granite" else {"silu", "silu_glu"}
    if config.get("hidden_act", "silu") not in activations:
        unsupported.append("Dense adapter requires SiLU")
    allowed = _COMMON_KEYS | (_GRANITE_KEYS if family == "granite" else _SMOLLM3_KEYS)
    if set(config) - allowed:
        unsupported.append("Config options are outside qualified dense coverage")
    for flag in ("is_encoder_decoder", "is_decoder", "add_cross_attention", "tie_encoder_decoder"):
        if _bool(config, flag, False):
            unsupported.append("Decoder topology is outside qualified dense coverage")
    if config.get("pruned_heads") not in (None, {}):
        unsupported.append("Pruned attention heads are outside qualified dense coverage")
    if config.get("rope_scaling") is not None or config.get("rope_parameters") is not None:
        unsupported.append("RoPE scaling is outside qualified dense coverage")
    if family == "granite" and config.get("layer_types") is not None:
        unsupported.append("Explicit layer types are outside qualified dense coverage")
    embedding_scale, residual_scale, attention_scale, logit_divisor = None, None, None, None
    uses_rope = [True] * layers
    if family == "granite":
        if not _GRANITE_KEYS <= set(config):
            raise ConversionError("Granite config is missing a required scale")
        if not tied:
            unsupported.append("Granite requires tied word embeddings")
        embedding_scale = _positive_float(config, "embedding_multiplier", 1.0)
        residual_scale = _positive_float(config, "residual_multiplier", 1.0)
        attention_scale = _positive_float(config, "attention_multiplier", 1.0)
        logit_divisor = _positive_float(config, "logits_scaling", 1.0)
    else:
        mask = config.get("no_rope_layers")
        if not isinstance(mask, list) or len(mask) < layers or any(type(flag) is not int for flag in mask):
            raise ConversionError("SmolLM3 requires an integer per-layer no_rope_layers mask")
        # The pinned importer treats a nonzero flag as USES RoPE, despite the
        # field's name; unused trailing entries are intentionally ignored.
        uses_rope = [bool(flag) for flag in mask[:layers]]
        interval = config.get("no_rope_layer_interval")
        if interval is not None:
            interval = _positive_int(config, "no_rope_layer_interval")
            expected_rope = [(index + 1) % interval != 0 for index in range(layers)]
            if uses_rope != expected_rope:
                unsupported.append("SmolLM3 RoPE interval disagrees with its per-layer mask")
        layer_types = config.get("layer_types")
        if layer_types is not None:
            if (
                not isinstance(layer_types, list)
                or len(layer_types) != layers
                or any(not isinstance(layer, str) for layer in layer_types)
            ):
                raise ConversionError("SmolLM3 config has malformed layer_types")
            if layer_types != ["full_attention"] * layers:
                unsupported.append("SmolLM3 requires full attention in every layer")
        use_sliding = _bool(config, "use_sliding_window", False)
        window = config.get("sliding_window")
        if window is not None:
            _positive_int(config, "sliding_window")
        boundary = config.get("max_window_layers", layers)
        if type(boundary) is not int or not 0 <= boundary <= layers:
            raise ConversionError("SmolLM3 sliding-window layer boundary is invalid")
        if use_sliding or window is not None:
            unsupported.append("SmolLM3 sliding attention is outside qualified dense coverage")
    quantization = parse_mlx_quantization(config) if family == "smollm3" else None
    tensors = load_checkpoint(source)
    if family != "smollm3":
        for key in ("quantization", "quantization_config"):
            if config.get(key) is not None:
                if not isinstance(config[key], dict) or not config[key]:
                    raise ConversionError("Dense config has malformed quantization parameters")
                raise UnsupportedConversion("Quantized checkpoint is outside qualified dense coverage")
    expected: set[str] = set()
    operations: list[TensorOperation] = []
    metadata: dict[str, str] = {}
    used_mlx = False

    def tensor(name: str, shape: tuple[int, ...]):
        expected.add(name)
        if name not in tensors:
            raise ConversionError("Dense checkpoint is missing a required tensor")
        info = tensors[name]
        if info.shape != shape:
            raise ConversionError("Dense checkpoint tensor shape disagrees with its config")
        if info.dtype not in _DTYPES.values():
            unsupported.append("Checkpoint dtype is outside qualified dense coverage")
        return info

    def parameter(out, names, shapes, shape, *, scalar=None, divide=False):
        inputs = tuple(tensor(name, required) for name, required in zip(names, shapes, strict=True))
        if len({info.dtype for info in inputs}) > 1:
            unsupported.append("Mixed-dtype fusion promotion is outside qualified dense coverage")
        kind = "fuse" if len(inputs) > 1 else "copy"
        if any(info.dtype != target_dtype for info in inputs):
            kind = "fuse_cast" if len(inputs) > 1 else "cast"
        if scalar is not None:
            if len(inputs) != 1:
                raise ConversionError("Scalar operation requires one source tensor")
            kind = "scalar_divide" if divide else "scalar_multiply"
        operations.append(TensorOperation(out, inputs, target_dtype, shape, kind, scalar))

    def matrix(out, names, shapes, shape, *, layout="output_input", scalar=None):
        nonlocal used_mlx
        representations = [
            any(name.removesuffix(".weight") + suffix in tensors for suffix in (".scales", ".biases"))
            or (name in tensors and tensors[name].dtype in {"I32", "U32"})
            for name in names
        ]
        if family == "smollm3" and any(representations):
            if quantization is None:
                raise UnsupportedConversion("MLX checkpoint requires explicit quantization configuration")
            if not all(representations):
                raise UnsupportedConversion("Mixed full-precision/MLX fusion is outside implemented coverage")
            if scalar is not None:
                raise UnsupportedConversion("Scalar transformations of MLX matrices are outside implemented coverage")
            sources = tuple(
                read_mlx_matrix(tensors, name, source_shape, quantization)
                for name, source_shape in zip(names, shapes, strict=True)
            )
            for name in names:
                prefix = name.removesuffix(".weight")
                expected.update((name, prefix + ".scales", prefix + ".biases"))
            operations.extend(
                plan_mlx_matrix(out, sources, dtype=target_dtype, layout=layout, quantization=quantization)
            )
            metadata[out + ".spec"] = json.dumps(
                {
                    "type": "MLXSpec",
                    "bits": quantization.bits,
                    "group_size": quantization.group_size,
                    "layout": layout,
                }
            )
            used_mlx = True
            return
        parameter(out + ".weights", names, shapes, shape, scalar=scalar)
        metadata[out + ".spec"] = json.dumps({"type": "FullPrecisionSpec", "layout": layout})

    def norm(out, name, *, divisor=None):
        info = tensor(name, (model_dim,))
        kind = "cast_f32" if divisor is None else "scalar_divide"
        operations.append(TensorOperation(out + ".scales", (info,), "F32", (model_dim,), kind, divisor))

    embedding = "decoder.embedding.embedding" if tied else "decoder.embedding.input_embedding"
    matrix(embedding, ("model.embed_tokens.weight",), ((vocab, model_dim),), (vocab, model_dim), layout="input_output")
    if tied:
        if any("lm_head" + suffix in tensors for suffix in (".weight", ".scales", ".biases")):
            if json.loads(metadata[embedding + ".spec"])["type"] == "MLXSpec":
                head = read_mlx_matrix(tensors, "lm_head.weight", (vocab, model_dim), quantization)
                input_embedding = read_mlx_matrix(
                    tensors, "model.embed_tokens.weight", (vocab, model_dim), quantization
                )
                expected.update(("lm_head.weight", "lm_head.scales", "lm_head.biases"))
                if any(
                    left.dtype != right.dtype or tensor_sha256(left) != tensor_sha256(right)
                    for left, right in zip(head.planes, input_embedding.planes, strict=True)
                ):
                    raise ConversionError("Tied MLX lm_head differs from the input embedding")
            else:
                head = tensor("lm_head.weight", (vocab, model_dim))
                input_embedding = tensors["model.embed_tokens.weight"]
                if head.dtype != input_embedding.dtype or tensor_sha256(head) != tensor_sha256(input_embedding):
                    raise ConversionError("Tied lm_head differs from the input embedding")
    else:
        matrix("decoder.embedding.output_embedding", ("lm_head.weight",), ((vocab, model_dim),), (vocab, model_dim))
    q_width, kv_width = heads * head_dim, groups * head_dim
    for layer in range(layers):
        src, out = f"model.layers.{layer}", f"decoder.transformer.layers.{layer}"
        norm(out + ".pre_mixer_norm", src + ".input_layernorm.weight")
        norm(out + ".pre_mlp_norm", src + ".post_attention_layernorm.weight")
        matrix(
            out + ".mixer.qkvg_projection.weights",
            tuple(src + f".self_attn.{name}_proj.weight" for name in ("q", "k", "v")),
            ((q_width, model_dim), (kv_width, model_dim), (kv_width, model_dim)),
            (q_width + 2 * kv_width, model_dim),
        )
        matrix(
            out + ".mixer.out_projection.weights",
            (src + ".self_attn.o_proj.weight",),
            ((model_dim, q_width),),
            (model_dim, q_width),
            scalar=residual_scale,
        )
        matrix(
            out + ".mlp.up_projection.weights",
            (src + ".mlp.up_proj.weight", src + ".mlp.gate_proj.weight"),
            ((hidden_dim, model_dim), (hidden_dim, model_dim)),
            (2 * hidden_dim, model_dim),
        )
        matrix(
            out + ".mlp.down_projection.weights",
            (src + ".mlp.down_proj.weight",),
            ((model_dim, hidden_dim),),
            (model_dim, hidden_dim),
            scalar=residual_scale,
        )
        if attention_bias:
            parameter(
                out + ".mixer.qkvg_projection.biases",
                tuple(src + f".self_attn.{name}_proj.bias" for name in ("q", "k", "v")),
                ((q_width,), (kv_width,), (kv_width,)),
                (q_width + 2 * kv_width,),
            )
            if family == "smollm3":
                parameter(
                    out + ".mixer.out_projection.biases",
                    (src + ".self_attn.o_proj.bias",),
                    ((model_dim,),),
                    (model_dim,),
                )
        if mlp_bias:
            parameter(
                out + ".mlp.up_projection.biases",
                (src + ".mlp.up_proj.bias", src + ".mlp.gate_proj.bias"),
                ((hidden_dim,), (hidden_dim,)),
                (2 * hidden_dim,),
            )
            parameter(
                out + ".mlp.down_projection.biases",
                (src + ".mlp.down_proj.bias",),
                ((model_dim,),),
                (model_dim,),
                scalar=residual_scale,
            )
    norm("decoder.transformer.output_norm", "model.norm.weight", divisor=logit_divisor)
    if set(tensors) != expected:
        raise ConversionError("Dense checkpoint has unexpected tensors")
    if quantization is not None and not used_mlx:
        raise ConversionError("MLX quantization configuration has no packed checkpoint matrices")
    if unsupported:
        raise UnsupportedConversion(unsupported[0])
    norm_config = {
        "epsilon": epsilon,
        "scale_offset": None,
        "upcast_mode": "only_normalization",
        "subtract_mean": False,
        "has_scale": True,
        "has_biases": False,
    }
    layer_configs = []
    for use_rope in uses_rope:
        layer_configs.append(
            {
                "pre_mixer_norm_config": norm_config,
                "mixer_conv_config": None,
                "mixer_config": {
                    "type": "AttentionConfig",
                    "qkvg_projection_config": {},
                    "out_projection_config": {},
                    "query_norm_config": None,
                    "key_norm_config": None,
                    "num_heads": heads,
                    "num_groups": groups,
                    "head_dim": head_dim,
                    "is_causal": True,
                    "scale": attention_scale,
                    "sliding_window_size": None,
                    "logit_soft_cap": None,
                    "has_sinks": False,
                    "has_qkvg_biases": attention_bias,
                    "has_out_biases": attention_bias and family == "smollm3",
                    "has_gate": False,
                    "normalize_values": False,
                    "is_kv_sharing": False,
                },
                "post_mixer_norm_config": None,
                "pre_mlp_norm_config": norm_config,
                "mlp_conv_config": None,
                "mlp_config": {
                    "type": "DenseMLPConfig",
                    "linear_config": {},
                    "activation": {"type": "SiLU", "alpha": 1.0},
                    "has_up_biases": mlp_bias,
                    "has_down_biases": mlp_bias,
                    "gate_clipping": None,
                    "up_clipping": None,
                },
                "post_mlp_norm_config": None,
                "hidden_dim": None,
                "ple_config": None,
                "has_post_layer_scalar": False,
                "kv_source_layer_index": None,
                "rope_config": rope if use_rope else None,
            }
        )
    decoder = {
        "embedding_config": {
            "type": "TiedEmbeddingConfig" if tied else "UntiedEmbeddingConfig",
            "input_scale": embedding_scale,
            "logit_soft_cap": None,
            "logit_scale": None,
        },
        "transformer_config": {
            "layer_configs": layer_configs,
            "output_norm_config": norm_config,
            "model_dim": model_dim,
            "hidden_dim": hidden_dim,
        },
        "vocab_size": vocab,
        "ple_model_config": None,
        "embedding_norm_config": None,
    }
    return ConversionPlan(
        decoder,
        tuple(operations),
        metadata,
        sum(t.nbytes for t in tensors.values()),
        len(tensors),
        tuple(tensors.values()),
    )


def build_granite_plan(source: Path, *, dtype: str = "bfloat16") -> ConversionPlan:
    """Validate Granite's tied embedding, source-precision scaling, and layout."""
    return _build_plan(source, family="granite", dtype=dtype)
