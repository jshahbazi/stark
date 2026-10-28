#!/usr/bin/env python3
"""Create a private tiny GLM fixture using pinned upstream forward modules.

Run with the locked tools environment. The resulting synthetic weight files,
token IDs and logits stay in an ignored runtime directory. Only aggregate
counts and source identities are printed.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import json
import math
import os
import struct
import sys
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional

REVISION = "95bafdcccfffe17ec383999fc39273867b11385e"
SOURCE_SHA256 = "7870e7ce6ca477d26de3ec79b31d41553b75a09a726c661f04c2e26e790bdf36"
MINIMUM_INDEXER_CUTOFF_MARGIN = 1e-5
SOURCE_URL = (
    "https://raw.githubusercontent.com/huggingface/transformers/"
    f"{REVISION}/src/transformers/models/glm5_next/modeling_glm5_next.py"
)


def upstream_namespace(source: bytes) -> dict:
    """Load the actual upstream math/classes without optional integrations.

    This removes the kernel-dispatch decorators and supplies nn.Module for the
    checkpointing-only base class. The forward method bodies are unchanged.
    The oracle performs full-sequence prefill, independently of Stark's
    incremental implementation and its absorbed MLA projection.
    """
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("upstream_source_hash_mismatch")
    names = {
        "Glm5NextTextRMSNorm",
        "Glm5NextTextMLP",
        "Glm5NextTextExperts",
        "Glm5NextTextTopkRouter",
        "Glm5NextTextMoE",
        "Glm5NextTextUnweightedRMSNorm",
        "Glm5NextTextHyperConnection",
        "Glm5NextTextHyperHead",
        "Glm5NextTextForgetGate",
        "Glm5NextTextRMSNormGated",
        "apply_mask_to_padding_states",
        "causal_conv1d_update",
        "causal_conv1d_fn",
        "l2norm",
        "recurrent_kimi_delta_attention",
        "chunk_kimi_delta_attention",
        "Glm5NextTextLinearAttention",
        "Glm5NextTextIndexer",
        "repeat_kv",
        "eager_attention_forward",
        "Glm5NextTextAttention",
        "Glm5NextTextDecoderLayer",
    }
    tree = ast.parse(source, filename="pinned_glm5_next.py")
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in nodes} != names:
        raise RuntimeError("upstream_reference_definition_set_changed")
    for node in nodes:
        for definition in ast.walk(node):
            if isinstance(definition, ast.ClassDef | ast.FunctionDef):
                definition.decorator_list = []
    program = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(program)
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": functional,
        "math": math,
        "ACT2FN": {"silu": functional.silu, "sigmoid": torch.sigmoid},
        "GradientCheckpointingLayer": nn.Module,
        "ALL_ATTENTION_FUNCTIONS": SimpleNamespace(
            get_interface=lambda _name, fallback: fallback
        ),
    }
    # Only the hash-verified upstream function bodies enter this namespace.
    exec(compile(program, "pinned_glm5_next.py", "exec"), namespace)  # noqa: S102
    return namespace


@contextmanager
def audit_indexer_cutoffs():
    """Observe cutoff margins from the unchanged upstream indexer frame.

    The source forward already calculates its masked scores and candidate
    validity. Inspecting those completed local tensors avoids replacing its
    top-k implementation or recomputing its scoring equations in the oracle.
    """
    statistics = []
    previous = sys.getprofile()

    def observe(frame, event, _result):
        if (
            event != "return"
            or frame.f_code.co_filename != "pinned_glm5_next.py"
            or frame.f_code.co_name != "forward"
            or type(frame.f_locals.get("self")).__name__ != "Glm5NextTextIndexer"
        ):
            return
        scores = frame.f_locals["index_scores"]
        valid = frame.f_locals["valid_candidates"]
        keep = frame.f_locals["select_k"]
        cutoffs = ties = 0
        minimum = None
        for row, mask in zip(scores.flatten(0, -2), valid.flatten(0, -2), strict=True):
            candidates = row[mask]
            if candidates.numel() <= keep:
                continue
            ordered = candidates.sort(descending=True).values
            margin = float(ordered[keep - 1] - ordered[keep])
            cutoffs += 1
            ties += int(margin == 0.0)
            minimum = margin if minimum is None else min(minimum, margin)
        statistics.append(
            {
                "queries_with_cutoff": cutoffs,
                "tied_cutoffs": ties,
                "minimum_margin": minimum,
            }
        )

    sys.setprofile(observe)
    try:
        yield statistics
    finally:
        sys.setprofile(previous)


def tiny_config(*, quantized: bool = False) -> dict:
    result = {
        "model_type": "glm5_next_text",
        "hidden_size": 16,
        "num_hidden_layers": 3,
        "layer_types": [
            "linear_attention",
            "deepseek_sparse_attention",
            "linear_attention",
        ],
        "mlp_layer_types": ["dense", "sparse", "sparse"],
        "indexer_types": ["full", "full", "full"],
        "first_k_dense_replace": 1,
        "intermediate_size": 12,
        "moe_intermediate_size": 8,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "q_lora_rank": 8,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 0,
        "v_head_dim": 4,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "rms_norm_eps": 1e-5,
        "linear_attn_config": {
            "num_heads": 2,
            "head_dim": 4,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_kpool": 2,
        "index_topk": 4,
        "index_kpool_always_select_tail": True,
        "index_kpool_compress": True,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "swiglu_limit": 10.0,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mhc": True,
        "mla_use_nope": True,
        "vocab_size": 16,
        "max_position_embeddings": 64,
        "eos_token_id": [15],
    }
    if quantized:
        result.update(
            hidden_size=256,
            intermediate_size=256,
            moe_intermediate_size=256,
            q_lora_rank=256,
            kv_lora_rank=256,
            qk_nope_head_dim=128,
            v_head_dim=128,
            index_head_dim=128,
            index_n_heads=32,
            linear_attn_config={
                "num_heads": 2,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
            },
        )
    return result


def reference_modules(source: bytes, *, quantized: bool = False):
    ns = upstream_namespace(source)
    config = tiny_config(quantized=quantized)
    normalized = {
        **config,
        "num_local_experts": config["n_routed_experts"],
        "linear_num_heads": config["linear_attn_config"]["num_heads"],
        "linear_head_dim": config["linear_attn_config"]["head_dim"],
        "linear_conv_kernel_dim": config["linear_attn_config"][
            "short_conv_kernel_size"
        ],
        "linear_lower_bound": config["linear_attn_config"]["gate_lower_bound"],
        "_attn_implementation": "eager",
    }
    cfg = SimpleNamespace(**normalized)
    torch.manual_seed(20261028)
    layers = nn.ModuleList(
        ns["Glm5NextTextDecoderLayer"](cfg, i)
        for i in range(config["num_hidden_layers"])
    )
    embedding = nn.Embedding(config["vocab_size"], config["hidden_size"])
    head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
    norm = ns["Glm5NextTextRMSNorm"](config["hidden_size"], config["rms_norm_eps"])
    modules = nn.ModuleList([layers, embedding, head, norm]).eval()
    with torch.no_grad():
        for name, parameter in modules.named_parameters():
            parameter.normal_(mean=0.0, std=0.15)
            if "norm.weight" in name or name == "3.weight":
                parameter.add_(1.0)
            if name.endswith("A_log"):
                parameter.sub_(1.0)
            if ".indexer.weights_proj.weight" in name:
                # Positive head mixing plus the production head count avoids
                # synthetic all-zero cutoff ties. Torch top-k does not define
                # which equal-score pool to retain; Stark orders equal scores
                # by their original index, so a tied oracle is not unique.
                parameter.abs_().add_(0.01)
        for layer in layers:
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.e_score_correction_bias.copy_(
                    torch.linspace(-0.04, 0.05, config["n_routed_experts"])
                )
    return config, layers, embedding, head, norm


def gguf_tensors(
    config: dict, layers, embedding, head, norm
) -> dict[str, torch.Tensor]:
    tensors = {
        "token_embd.weight": embedding.weight,
        "output.weight": head.weight,
        "output_norm.weight": norm.weight,
    }
    for index, layer in enumerate(layers):
        prefix = f"blk.{index}."
        tensors[prefix + "attn_norm.weight"] = layer.input_layernorm.weight
        tensors[prefix + "ffn_norm.weight"] = layer.post_attention_layernorm.weight
        for part in ["attn", "ffn"]:
            connection = getattr(layer, part + "_hc")
            for target, source in [("fn", "fn"), ("base", "base"), ("scale", "scale")]:
                tensors[prefix + f"hc_{part}_{target}.weight"] = getattr(
                    connection, source
                )
        attention = layer.self_attn
        if config["layer_types"][index] == "linear_attention":
            for part in ["q", "k", "v"]:
                tensors[prefix + f"attn_{part}.weight"] = getattr(
                    attention, part + "_proj"
                ).weight
            for part, conv in zip(
                ["q", "k", "v"], attention.conv1d.weight.chunk(3, dim=0), strict=True
            ):
                tensors[prefix + f"ssm_conv1d_{part}.weight"] = conv
            tensors[prefix + "ssm_a"] = -attention.forget_gate.A_log.exp()
            tensors[prefix + "ssm_dt.bias"] = attention.forget_gate.dt_bias
            tensors[prefix + "ssm_beta.weight"] = attention.b_proj.weight
            for part in ["f_a", "f_b"]:
                tensors[prefix + f"ssm_{part}.weight"] = getattr(
                    attention.forget_gate, part + "_proj"
                ).weight
            for part in ["g_a", "g_b"]:
                tensors[prefix + f"ssm_{part}.weight"] = getattr(
                    attention, part + "_proj"
                ).weight
            tensors[prefix + "ssm_norm.weight"] = attention.o_norm.weight
        else:
            for gguf, hf in [
                ("attn_q_a.weight", "q_a_proj"),
                ("attn_q_a_norm.weight", "q_a_layernorm"),
                ("attn_q_b.weight", "q_b_proj"),
                ("attn_kv_a_mqa.weight", "kv_a_proj_with_mqa"),
                ("attn_kv_a_norm.weight", "kv_a_layernorm"),
            ]:
                tensors[prefix + gguf] = getattr(attention, hf).weight
            expanded = attention.kv_b_proj.weight.view(
                config["num_attention_heads"], -1, config["kv_lora_rank"]
            )
            tensors[prefix + "attn_k_b.weight"] = expanded[
                :, : config["qk_nope_head_dim"]
            ].transpose(-1, -2)
            tensors[prefix + "attn_v_b.weight"] = expanded[
                :, config["qk_nope_head_dim"] :
            ]
            for gguf, hf in [
                ("indexer.attn_q_b.weight", "wq_b"),
                ("indexer.attn_k.weight", "wk"),
                ("indexer.k_norm.weight", "k_norm"),
                ("indexer.proj.weight", "weights_proj"),
            ]:
                tensors[prefix + gguf] = getattr(attention.indexer, hf).weight
            tensors[prefix + "indexer.k_norm.bias"] = attention.indexer.k_norm.bias
            tensors[prefix + "indexer_compressor_gate.weight"] = (
                attention.indexer.index_kpool_compress_gate
            )
            tensors[prefix + "indexer_compressor_ape.weight"] = (
                attention.indexer.index_kpool_compress_ape
            )
        tensors[prefix + "attn_output.weight"] = attention.o_proj.weight
        if config["mlp_layer_types"][index] == "dense":
            for part in ["gate", "up", "down"]:
                tensors[prefix + f"ffn_{part}.weight"] = getattr(
                    layer.mlp, part + "_proj"
                ).weight
        else:
            tensors[prefix + "ffn_gate_inp.weight"] = layer.mlp.gate.weight
            tensors[prefix + "exp_probs_b.bias"] = (
                layer.mlp.gate.e_score_correction_bias
            )
            gate, up = layer.mlp.experts.gate_up_proj.chunk(2, dim=1)
            tensors[prefix + "ffn_gate_exps.weight"] = gate
            tensors[prefix + "ffn_up_exps.weight"] = up
            tensors[prefix + "ffn_down_exps.weight"] = layer.mlp.experts.down_proj
            for part in ["gate", "up", "down"]:
                tensors[prefix + f"ffn_{part}_shexp.weight"] = getattr(
                    layer.mlp.shared_experts, part + "_proj"
                ).weight
    return tensors


def gguf_metadata(c: dict) -> dict:
    prefix = "glm5-next."
    metadata = {
        "general.architecture": "glm5-next",
        "general.alignment": 32,
    }
    values = {
        "block_count": c["num_hidden_layers"],
        "nextn_predict_layers": 0,
        "embedding_length": c["hidden_size"],
        "context_length": c["max_position_embeddings"],
        "feed_forward_length": c["intermediate_size"],
        "attention.head_count": c["num_attention_heads"],
        "attention.head_count_kv": [
            0 if v == "linear_attention" else 1 for v in c["layer_types"]
        ],
        "attention.q_lora_rank": c["q_lora_rank"],
        "attention.kv_lora_rank": c["kv_lora_rank"],
        "attention.key_length": c["kv_lora_rank"],
        "attention.value_length": c["kv_lora_rank"],
        "attention.key_length_mla": c["qk_nope_head_dim"],
        "attention.value_length_mla": c["v_head_dim"],
        "rope.dimension_count": 0,
        "expert_count": c["n_routed_experts"],
        "expert_used_count": c["num_experts_per_tok"],
        "expert_group_count": 1,
        "expert_group_used_count": 1,
        "expert_gating_func": 2,
        "expert_feed_forward_length": c["moe_intermediate_size"],
        "expert_shared_feed_forward_length": c["moe_intermediate_size"]
        * c["n_shared_experts"],
        "expert_shared_count": c["n_shared_experts"],
        "leading_dense_block_count": c["first_k_dense_replace"],
        "vocab_size": c["vocab_size"],
        "ssm.conv_kernel": c["linear_attn_config"]["short_conv_kernel_size"],
        "kda.head_dim": c["linear_attn_config"]["head_dim"],
        "kda.gate_lower_bound": c["linear_attn_config"]["gate_lower_bound"],
        "attention.indexer.head_count": c["index_n_heads"],
        "attention.indexer.key_length": c["index_head_dim"],
        "attention.indexer.top_k": c["index_topk"],
        "attention.indexer.kpool": c["index_kpool"],
        "hyper_connection.count": c["hc_mult"],
        "hyper_connection.sinkhorn_iterations": c["hc_sinkhorn_iters"],
        "attention.layer_norm_rms_epsilon": c["rms_norm_eps"],
        "attention.layer_norm_epsilon": 1e-6,
        "hyper_connection.epsilon": c["hc_eps"],
        "expert_weights_scale": c["routed_scaling_factor"],
        "expert_weights_norm": True,
        "swiglu_clamp_exp": [c["swiglu_limit"]] * c["num_hidden_layers"],
        "swiglu_clamp_shexp": [c["swiglu_limit"]] * c["num_hidden_layers"],
    }
    metadata.update({prefix + k: v for k, v in values.items()})
    return metadata


def text(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def metadata_value(value) -> tuple[int, bytes]:
    if isinstance(value, bool):
        return 7, struct.pack("<?", value)
    if isinstance(value, int):
        return 10, struct.pack("<Q", value)
    if isinstance(value, float):
        return 6, struct.pack("<f", value)
    if isinstance(value, str):
        return 8, text(value)
    if isinstance(value, list) and value:
        kind, _ = metadata_value(value[0])
        members = [metadata_value(item) for item in value]
        if any(subtype != kind for subtype, _ in members):
            raise RuntimeError("heterogeneous_gguf_metadata")
        return 9, struct.pack("<IQ", kind, len(value)) + b"".join(
            data for _, data in members
        )
    raise RuntimeError("unsupported_gguf_metadata")


def write_gguf(
    path: Path, metadata: dict, tensors: dict, encoded: dict | None = None
) -> None:
    records = bytearray()
    for name, value in metadata.items():
        kind, data = metadata_value(value)
        records += text(name) + struct.pack("<I", kind) + data
    tensor_records = bytearray()
    payload = bytearray()
    for name, tensor in sorted(tensors.items()):
        tensor = tensor.detach().float().contiguous().cpu()
        while len(payload) % 32:
            payload.append(0)
        tensor_records += text(name) + struct.pack("<I", tensor.dim())
        tensor_records += struct.pack("<" + "Q" * tensor.dim(), *reversed(tensor.shape))
        kind, raw = (encoded or {}).get(
            name, (0, tensor.numpy().astype("<f4").tobytes())
        )
        tensor_records += struct.pack("<IQ", kind, len(payload))
        payload += raw
    header = (
        b"GGUF"
        + struct.pack("<IQQ", 3, len(tensors), len(metadata))
        + records
        + tensor_records
    )
    header += b"\0" * (-len(header) % 32)
    path.write_bytes(header + payload)


def encode_quantized(tensors: dict, reference_library: Path) -> dict:
    """Exercise actual IQ2 expert layouts with an independent GGML decoder.

    The supplied private library is compiled from the pinned ggml dequantizers.
    All source-module parameters are replaced with the decoded values before
    the Torch oracle runs. Matrix products are still Torch operations.
    """
    import numpy as np

    library = ctypes.CDLL(str(reference_library.resolve(strict=True)))
    rng = np.random.default_rng(20261028)
    encoded = {}
    with torch.no_grad():
        for name, tensor in tensors.items():
            if tensor.dim() < 2:
                continue
            if "_exps.weight" in name:
                kind = 18 if "ffn_down_exps" in name else 17
                block_values, block_bytes = 256, 74 if kind == 18 else 66
                raw = bytearray(
                    rng.integers(
                        0,
                        256,
                        tensor.numel() // block_values * block_bytes,
                        dtype=np.uint8,
                    )
                )
                for offset in range(0, len(raw), block_bytes):
                    raw[offset : offset + 2] = struct.pack("<e", 0.0005)
                function = getattr(
                    library,
                    "dequantize_row_iq2_xs" if kind == 18 else "dequantize_row_iq2_xxs",
                )
            elif (
                tensor.shape[-1] % 32 == 0
                and "ffn_gate_inp" not in name
                and "indexer.proj" not in name
            ):
                kind, block_values, block_bytes = 8, 32, 34
                values = (
                    tensor.detach().float().contiguous().cpu().numpy().reshape(-1, 32)
                )
                raw = bytearray()
                for block in values:
                    scale = np.float16(np.abs(block).max() / 127.0)
                    raw += struct.pack("<e", float(scale))
                    quantized = (
                        np.rint(block / scale).clip(-127, 127).astype(np.int8)
                        if scale
                        else np.zeros(32, dtype=np.int8)
                    )
                    raw += quantized.tobytes()
                function = library.dequantize_row_q8_0
            else:
                continue
            if tensor.shape[-1] % block_values:
                raise RuntimeError("quantized_fixture_row_alignment")
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            packed = ctypes.create_string_buffer(bytes(raw))
            values = (ctypes.c_float * tensor.numel())()
            function(packed, values, tensor.numel())
            decoded = torch.from_numpy(np.ctypeslib.as_array(values).copy()).view(
                tensor.shape
            )
            tensor.copy_(decoded)
            encoded[name] = kind, bytes(raw)
    return encoded


def create_fixture(
    output: Path, source: bytes, reference_library: Path | None = None
) -> dict:
    if output.exists():
        raise RuntimeError("output_already_exists")
    config, layers, embedding, head, norm = reference_modules(
        source, quantized=reference_library is not None
    )
    output.mkdir(parents=True, mode=0o700)
    tensors = gguf_tensors(config, layers, embedding, head, norm)
    encoded = (
        encode_quantized(tensors, reference_library) if reference_library else None
    )
    write_gguf(output / "model.gguf", gguf_metadata(config), tensors, encoded)
    outer = {"model_type": "glm5_next", "text_config": config}
    (output / "config.json").write_text(json.dumps(outer, indent=2) + "\n")
    sequences = [
        [(i * 7 + 3) % 16 for i in range(19)],
        [(i * 7 + 3) % 16 for i in range(19)],
        list(range(9)),
    ]
    sequences[1][-1] ^= 1
    cases = []
    torch.set_num_threads(1)
    with torch.inference_mode(), audit_indexer_cutoffs() as cutoff_statistics:
        for sequence in sequences:
            tokens = torch.tensor([sequence], dtype=torch.long)
            hidden = (
                embedding(tokens)
                .unsqueeze(2)
                .expand(-1, -1, config["hc_mult"], -1)
                .contiguous()
            )
            attention_mask = torch.ones_like(tokens, dtype=torch.bool)
            layer_outputs = []
            for layer in layers:
                hidden, _ = layer(hidden, attention_mask=attention_mask)
                layer_outputs.append(hidden[0].flatten(start_dim=1).tolist())
            logits = head(norm(hidden.mean(dim=2)))[0]
            if not torch.isfinite(logits).all():
                raise RuntimeError("reference_nonfinite")
            cases.append(
                {
                    "input_token_ids": sequence,
                    "logits": logits.tolist(),
                    "layer_outputs": layer_outputs,
                }
            )
    # Changing the final input may not affect any earlier output position.
    causality_error = max(
        abs(a - b)
        for row_a, row_b in zip(
            cases[0]["logits"][:-1], cases[1]["logits"][:-1], strict=True
        )
        for a, b in zip(row_a, row_b, strict=True)
    )
    if causality_error > 1e-5:
        raise RuntimeError("reference_causality_failed")
    if not cutoff_statistics or not any(
        case["queries_with_cutoff"] for case in cutoff_statistics
    ):
        raise RuntimeError("reference_sparse_cutoff_not_exercised")
    if any(
        case["tied_cutoffs"]
        or (
            case["queries_with_cutoff"]
            and (
                not math.isfinite(case["minimum_margin"])
                or case["minimum_margin"] <= MINIMUM_INDEXER_CUTOFF_MARGIN
            )
        )
        for case in cutoff_statistics
    ):
        raise RuntimeError("reference_sparse_cutoff_is_ambiguous")
    reference = {
        "source": {"revision": REVISION, "sha256": SOURCE_SHA256, "url": SOURCE_URL},
        "cases": cases,
        "indexer_cutoffs": cutoff_statistics,
        "minimum_required_cutoff_margin": MINIMUM_INDEXER_CUTOFF_MARGIN,
    }
    (output / "reference.json").write_text(json.dumps(reference) + "\n")
    os.chmod(output / "reference.json", 0o600)
    (output / "tokens.json").write_text(json.dumps(sequences[0]) + "\n")
    (output / "reference.f32").write_bytes(
        b"".join(
            struct.pack("<f", value) for row in cases[0]["logits"] for value in row
        )
    )
    return {
        "created": True,
        "cases": len(cases),
        "positions": sum(map(len, sequences)),
        "source_sha256": SOURCE_SHA256,
        "causality_maximum_error": causality_error,
        "quantized_reference": reference_library is not None,
        "quantized_tensors": len(encoded or {}),
        "indexer_cutoffs": cutoff_statistics,
        "minimum_required_cutoff_margin": MINIMUM_INDEXER_CUTOFF_MARGIN,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--quantized-reference-library", type=Path)
    args = parser.parse_args()
    source = (
        args.source.read_bytes()
        if args.source
        else urllib.request.urlopen(SOURCE_URL, timeout=30).read(2_000_000)
    )
    result = create_fixture(args.output, source, args.quantized_reference_library)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - keep private fixture failures content-free
        print(
            json.dumps(
                {"created": False, "error_type": type(error).__name__}, sort_keys=True
            )
        )
        raise SystemExit(1) from None
