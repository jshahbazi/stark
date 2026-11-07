"""Preserve the one-layer dense Lightning MTP graph in a converted artifact.

Graph provenance: Apache-2.0 jundot/omlx qwen35_model.py at
2165e75726ba83d9651dbaced61c97728da77bba. Auxiliary tensors never grant
runtime capability until the native loader has validated the complete graph.
"""

from copy import deepcopy

from ..streaming import ConversionError, UnsupportedConversion


def plan_qwen35_mtp(
    *,
    tensors,
    declaration,
    dim,
    hidden,
    heads,
    groups,
    head_dim,
    target_layers,
    norm_config,
    layer_configs,
    matrix_source,
    matrix,
    norm,
    row_range,
    include_vision,
    quantization,
):
    candidates = [
        stem for stem in ("mtp", "language_model.mtp", "model.language_model.mtp") if stem + ".fc.weight" in tensors
    ]
    if not candidates:
        # Legacy uniform text exports may declare bare auxiliary MTP tensors
        # which were always omitted. Per-module exports and language-model
        # predictor namespaces require a complete identifiable graph.
        if any(
            name.startswith(("language_model.mtp.", "model.language_model.mtp."))
            or (quantization is not None and quantization.overrides and name.startswith("mtp."))
            for name in tensors
        ):
            raise ConversionError("Qwen3.5 MTP predictor is missing its fusion matrix")
        return None
    if len(candidates) != 1 or declaration != 1:
        raise ConversionError("Qwen3.5 MTP requires one explicitly declared predictor")
    if quantization is None:
        # Preserve the existing raw text-only conversion contract. Its declared
        # auxiliary weights remain omitted and cannot grant MTP capability.
        return None
    if include_vision:
        # Image artifacts deliberately omit the separately declared predictor.
        # Image generation uses the target graph until multimodal MTP is
        # qualified; auxiliary weights cannot enable that capability.
        return None
    source = candidates[0]
    attention = next((layer for layer in layer_configs if layer["mixer_config"]["type"] == "AttentionConfig"), None)
    if attention is None or attention["mixer_config"]["has_qkvg_biases"]:
        raise UnsupportedConversion("Qwen3.5 MTP requires bias-free full attention")
    norm("mtp.embedding_norm", source + ".pre_fc_norm_embedding.weight", dim)
    norm("mtp.hidden_norm", source + ".pre_fc_norm_hidden.weight", dim)
    matrix("mtp.fusion.weights", (matrix_source(source + ".fc.weight", (dim, 2 * dim)),), (dim, 2 * dim))
    src, out = source + ".layers.0", "mtp.transformer.layers.0"
    norm(out + ".pre_mixer_norm", src + ".input_layernorm.weight", dim)
    norm(out + ".pre_mlp_norm", src + ".post_attention_layernorm.weight", dim)
    qwidth, kvwidth = heads * head_dim, groups * head_dim
    attn = src + ".self_attn"
    q = matrix_source(attn + ".q_proj.weight", (2 * qwidth, dim))
    sources = (
        *(row_range(q, h * 2 * head_dim, head_dim) for h in range(heads)),
        matrix_source(attn + ".k_proj.weight", (kvwidth, dim)),
        matrix_source(attn + ".v_proj.weight", (kvwidth, dim)),
        *(row_range(q, h * 2 * head_dim + head_dim, head_dim) for h in range(heads)),
    )
    matrix(out + ".mixer.qkvg_projection.weights", sources, (2 * qwidth + 2 * kvwidth, dim))
    matrix(
        out + ".mixer.out_projection.weights", (matrix_source(attn + ".o_proj.weight", (dim, qwidth)),), (dim, qwidth)
    )
    norm(out + ".mixer.query_norm", attn + ".q_norm.weight", head_dim)
    norm(out + ".mixer.key_norm", attn + ".k_norm.weight", head_dim)
    matrix(
        out + ".mlp.up_projection.weights",
        tuple(matrix_source(src + f".mlp.{branch}_proj.weight", (hidden, dim)) for branch in ("up", "gate")),
        (2 * hidden, dim),
    )
    matrix(
        out + ".mlp.down_projection.weights",
        (matrix_source(src + ".mlp.down_proj.weight", (dim, hidden)),),
        (dim, hidden),
    )
    norm("mtp.transformer.output_norm", source + ".norm.weight", dim)
    return {
        "model_dim": dim,
        "target_layer": target_layers - 1,
        "input_norm_config": deepcopy(norm_config),
        "transformer_config": {
            "model_dim": dim,
            "hidden_dim": hidden,
            "layer_configs": [deepcopy(attention)],
            "output_norm_config": deepcopy(norm_config),
        },
    }
