"""Private primitive controls for distinct Gemma 4 numerical profiles.

Run once in each frozen reference environment. Inputs and outputs remain in an
ignored directory; stdout contains operation hashes and dependency identities.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import math
import subprocess
from pathlib import Path

from gemma4_hf_reference import digest

MLX_LM_REVISION = "1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7"


def run(
    engine: str,
    output: Path,
    lock: Path,
    device: str = "cpu",
    checkpoint: Path | None = None,
    cases: Path | None = None,
    case_index: int = 0,
    prefill_projection: bool = False,
    layer_index: int = 0,
    cached_last: bool = False,
) -> dict:
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    if (
        output.exists()
        or subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(output)], cwd=root, check=False
        ).returncode
        != 0
    ):
        raise ValueError("output_must_be_new_and_ignored")
    grid = (np.arange(65536, dtype=np.float32) - 32768) / 2048
    other = np.cos(np.arange(65536, dtype=np.float32) * np.float32(0.31))
    if engine == "mlx":
        import mlx.core as mx
        from mlx import nn
        from mlx_lm.models import gemma4_text
        from mlx_lm.models.rope_utils import ProportionalRoPE

        direct = json.loads(
            importlib.metadata.distribution("mlx-lm").read_text("direct_url.json")
        )
        if (
            direct.get("vcs_info", {}).get("commit_id") != MLX_LM_REVISION
            or importlib.metadata.version("mlx") != "0.32.3"
        ):
            raise ValueError("mlx_reference_identity_changed")
        mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
        x, y = mx.array(grid).astype(mx.bfloat16), mx.array(other).astype(mx.bfloat16)
        values = {
            "input": x,
            "other": y,
            "gelu": nn.gelu_approx(x),
            "geglu": gemma4_text.geglu(x, y),
            "geglu_uncompiled": nn.gelu_approx(x) * y,
            "softcap": gemma4_text.logit_softcap(30.0, x),
            "softcap_uncompiled": mx.tanh(x / 30.0) * 30.0,
            "projection_scale": x * (1536**-0.5),
            "input_scale": x * (2**-0.5),
            "global_proportional_frequencies": ProportionalRoPE(
                512, 128, base=1_000_000.0
            )._freqs,
        }
        if checkpoint is not None:
            from mlx_lm.utils import load_model

            selected = json.loads(cases.read_text())["cases"][case_index]["tokens"]
            prefix = selected[:-1] if cached_last else []
            selected = (
                selected[-1:]
                if cached_last
                else (selected[:-1] if prefill_projection else selected[:1])
            )
            if not selected:
                raise ValueError("actual_input_control_requires_a_token")
            if len(selected) > 1024:
                raise ValueError("actual_projection_control_exceeds_row_bound")
            model, _ = load_model(
                checkpoint, lazy=True, strict=True, trust_remote_code=False
            )
            mx.eval(model.parameters())
            text = getattr(model, "language_model", model).model
            if not 0 <= layer_index < len(text.layers):
                raise ValueError("layer_index_out_of_bounds")
            first = text.layers[layer_index]
            if hasattr(first.self_attn.rope, "_freqs"):
                values["actual_rope_frequencies"] = first.self_attn.rope._freqs
            hidden = (
                text.embed_tokens(mx.array([selected], dtype=mx.int32))
                * text.embed_scale
            )
            if prefill_projection:
                from mlx_lm.models.cache import make_prompt_cache

                modules = {
                    id(first.input_layernorm): "actual_input_norm",
                    id(first.self_attn.q_proj): "actual_q_projection",
                    id(first.self_attn.q_norm): "actual_q_norm",
                    id(first.self_attn.o_proj): "actual_o_projection",
                    id(first.post_attention_layernorm): "actual_post_attention_norm",
                    id(first.pre_feedforward_layernorm): "actual_mlp_norm",
                    id(first.mlp.up_proj): "actual_mlp_up",
                    id(first.mlp.gate_proj): "actual_mlp_gate",
                    id(first.mlp.down_proj): "actual_mlp_down",
                    id(first.post_feedforward_layernorm): "actual_post_mlp_norm",
                    id(first.per_layer_input_gate): "actual_ple_gate",
                    id(first.per_layer_projection): "actual_ple_projection",
                    id(first.post_per_layer_input_norm): "actual_ple_norm",
                }
                if first.self_attn.has_kv:
                    modules.update(
                        {
                            id(first.self_attn.k_proj): "actual_k_projection",
                            id(first.self_attn.k_norm): "actual_k_norm",
                            id(first.self_attn.v_norm): "actual_v_norm",
                        }
                    )
                if hasattr(first.self_attn, "v_proj"):
                    modules[id(first.self_attn.v_proj)] = "actual_v_projection"
                module_types = {
                    type(first.input_layernorm),
                    type(first.self_attn.q_proj),
                }
                originals = {kind: kind.__call__ for kind in module_types}
                cache = make_prompt_cache(model)
                if cached_last:
                    prefix_result = model(
                        mx.array([prefix], dtype=mx.int32), cache=cache
                    )
                    mx.eval(prefix_result)
                original_layer = gemma4_text.DecoderLayer.__call__
                original_sdpa = gemma4_text.scaled_dot_product_attention
                active_layer = None

                def capture_layer(this, x, *parts, **kwargs):
                    nonlocal active_layer
                    active_layer = this.layer_idx
                    if this is first:
                        values["actual_layer_input"] = x
                        values["actual_ple_input"] = kwargs["per_layer_input"]
                    result = original_layer(this, x, *parts, **kwargs)
                    if this is first:
                        values["actual_layer_output"] = result[0]
                    return result

                def capture_sdpa(q, k, v, *parts, **kwargs):
                    result = original_sdpa(q, k, v, *parts, **kwargs)
                    if active_layer == layer_index:
                        values["actual_q_rope"] = q
                        values["actual_attention_keys_native"] = k.transpose(0, 2, 1, 3)
                        values["actual_attention_values_native"] = v.transpose(
                            0, 2, 1, 3
                        )
                        values["actual_k_rope_native"] = k[
                            :, :, -len(selected) :, :
                        ].transpose(0, 2, 1, 3)
                        values["actual_v_norm"] = v[
                            :, :, -len(selected) :, :
                        ].transpose(0, 2, 1, 3)
                        values["actual_attention_output"] = result.transpose(0, 2, 1, 3)
                        if not cached_last:
                            # Exact upstream fast.cpp fallback construction. These
                            # nodes diagnose QK/softmax/PV separately; their combined
                            # result must match the unchanged stock SDPA output.
                            repeats = q.shape[1] // k.shape[1]
                            scaled_q = q * mx.array(
                                kwargs.get("scale", 1.0), dtype=q.dtype
                            )
                            grouped_q = mx.unflatten(scaled_q, 1, (k.shape[1], repeats))
                            grouped_k = mx.expand_dims(k, 2)
                            grouped_v = mx.expand_dims(v, 2)
                            scores = mx.matmul(grouped_q, grouped_k.swapaxes(-1, -2))
                            mask = kwargs.get("mask")
                            if isinstance(mask, str):
                                if mask != "causal":
                                    raise ValueError("unsupported_source_mask")
                                length = q.shape[-2]
                                offset = k.shape[-2] - length
                                mask = (mx.arange(length) + offset)[
                                    :, None
                                ] >= mx.arange(k.shape[-2])[None, :]
                            if mask is not None:
                                scores = (
                                    mx.where(
                                        mask,
                                        scores,
                                        mx.array(
                                            mx.finfo(scores.dtype).min, scores.dtype
                                        ),
                                    )
                                    if mask.dtype == mx.bool_
                                    else scores + mask
                                )
                            probabilities = mx.softmax(scores, axis=-1, precise=True)
                            reconstructed = mx.matmul(probabilities, grouped_v)
                            values["actual_attention_scores"] = scores
                            values["actual_attention_probabilities"] = probabilities
                            values["actual_attention_reconstructed"] = reconstructed
                    return result

                def wrapper(original):
                    def capture(this, x, *parts, **kwargs):
                        result = original(this, x, *parts, **kwargs)
                        if id(this) in modules:
                            name = modules[id(this)]
                            values[name + "_input"] = x
                            values[name] = result
                        return result

                    return capture

                try:
                    for kind, original in originals.items():
                        kind.__call__ = wrapper(original)
                    gemma4_text.DecoderLayer.__call__ = capture_layer
                    gemma4_text.scaled_dot_product_attention = capture_sdpa
                    result = model(
                        mx.array([selected], dtype=mx.int32),
                        cache=cache,
                    )
                    mx.eval(result)
                finally:
                    for kind, original in originals.items():
                        kind.__call__ = original
                    gemma4_text.DecoderLayer.__call__ = original_layer
                    gemma4_text.scaled_dot_product_attention = original_sdpa
                values["actual_qkv_input"] = values["actual_q_projection_input"]
                if first.self_attn.has_kv and not hasattr(first.self_attn, "v_proj"):
                    # Global K=V aliases the raw K projection before K RMS/RoPE.
                    values["actual_v_projection"] = values["actual_k_projection"]
                    values["actual_v_projection_input"] = values[
                        "actual_k_projection_input"
                    ]
                values["actual_qkv"] = (
                    mx.concatenate(
                        [
                            values["actual_q_projection"],
                            values["actual_k_projection"],
                            values["actual_v_projection"],
                        ],
                        axis=-1,
                    )
                    if first.self_attn.has_kv
                    else values["actual_q_projection"]
                )
            else:
                attention, _, _ = first.self_attn(
                    first.input_layernorm(hidden), mask=None, cache=None
                )
                residual = hidden + first.post_attention_layernorm(attention)
                normalized = first.pre_feedforward_layernorm(residual)
                gate, up = (
                    first.mlp.gate_proj(normalized),
                    first.mlp.up_proj(normalized),
                )
                cubic = gate**3
                term = 0.044715 * cubic
                inner = gate + term
                argument = math.sqrt(2 / math.pi) * inner
                values.update(
                    {
                        "actual_mlp_gate_input": gate,
                        "actual_mlp_gelu": nn.gelu_approx(gate),
                        "actual_mlp_geglu": gemma4_text.geglu(gate, up),
                        "actual_mlp_norm_input": residual,
                        "actual_mlp_norm": normalized,
                        "actual_mlp_norm_weight": first.pre_feedforward_layernorm.weight,
                        "actual_mlp_cube": cubic,
                        "actual_mlp_term": term,
                        "actual_mlp_inner": inner,
                        "actual_mlp_argument": argument,
                        "actual_mlp_tanh": mx.tanh(argument),
                        "actual_mlp_increment": 1 + mx.tanh(argument),
                        "actual_mlp_half": 0.5 * gate,
                    }
                )
                after_mlp = residual + first.post_feedforward_layernorm(
                    first.mlp.down_proj(gemma4_text.geglu(gate, up))
                )
                ple_gate = first.per_layer_input_gate(after_mlp)
                values.update(
                    {
                        "actual_ple_gate_input": ple_gate,
                        "actual_ple_gelu": nn.gelu_approx(ple_gate),
                    }
                )
        mx.eval(*values.values())
        arrays = {
            name: np.array(value.astype(mx.float32)).astype("<f4")
            for name, value in values.items()
        }
        if (
            prefill_projection
            and not cached_last
            and not np.array_equal(
                arrays["actual_attention_reconstructed"]
                .reshape(arrays["actual_q_rope"].shape)
                .transpose(0, 2, 1, 3),
                arrays["actual_attention_output"],
            )
        ):
            raise ValueError("explicit_fallback_does_not_match_stock_attention")
        identity = {
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm_revision": MLX_LM_REVISION,
            "activation_source_sha256": digest(Path(inspect.getfile(nn.gelu_approx))),
            "quantized_layers_source_sha256": digest(
                Path(inspect.getfile(nn.QuantizedLinear))
            ),
            "rope_utils_source_sha256": digest(Path(inspect.getfile(ProportionalRoPE))),
        }
        if checkpoint is not None:
            identity["weights_sha256"] = digest(checkpoint / "model.safetensors")
            identity["cases_sha256"] = digest(cases)
            identity["actual_case_index"] = case_index
            identity["actual_layer_index"] = layer_index
            identity["actual_has_kv"] = first.self_attn.has_kv
            identity["actual_position_offset"] = len(prefix)
            identity["cached_last"] = cached_last
            identity["rope_utils_source_sha256"] = digest(
                Path(inspect.getfile(type(first.self_attn.rope)))
            )
            identity["actual_projection_rows"] = len(selected)
        source = Path(inspect.getfile(gemma4_text))
    else:
        if device != "cpu":
            raise ValueError("canonical_primitive_controls_are_cpu_only")
        import torch
        import transformers
        from transformers.models.gemma4 import modeling_gemma4

        torch.set_num_threads(8)
        x, y = (
            torch.from_numpy(grid).to(torch.bfloat16),
            torch.from_numpy(other).to(torch.bfloat16),
        )
        gelu = torch.nn.functional.gelu(x, approximate="tanh")
        values = {
            "input": x,
            "other": y,
            "gelu": gelu,
            "geglu": gelu * y,
            "softcap": torch.tanh(x / 30.0) * 30.0,
            "projection_scale": x * (1536**-0.5),
            "input_scale": x * (2**-0.5),
        }
        arrays = {
            name: value.float().numpy().astype("<f4") for name, value in values.items()
        }
        identity = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
        source = Path(inspect.getfile(modeling_gemma4))
    output.mkdir(parents=True, mode=0o700)
    rows = []
    for name, values in arrays.items():
        path = output / (name + ".f32")
        path.write_bytes(values.tobytes())
        path.chmod(0o600)
        rows.append(
            {"operation": name, "elements": values.size, "sha256": digest(path)}
        )
    receipt = {
        "engine": engine,
        "device": device,
        "identity": identity,
        "source_sha256": digest(source),
        "harness_sha256": digest(Path(__file__)),
        "lock_sha256": digest(lock),
        "operations": rows,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("mlx", "hf"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--prefill-projection", action="store_true")
    parser.add_argument("--cached-last", action="store_true")
    args = parser.parse_args()
    if bool(args.source) != bool(args.cases) or (
        args.source is not None and args.engine != "mlx"
    ):
        parser.error(
            "--source and --cases must be selected together for MLX actual-input controls"
        )
    if args.cached_last and not args.prefill_projection:
        parser.error("--cached-last requires --prefill-projection")
    if args.prefill_projection and args.source is None:
        parser.error("--prefill-projection requires --source and --cases")
    try:
        receipt = run(
            args.engine,
            args.output.resolve(),
            args.lock.resolve(strict=True),
            args.device,
            args.source,
            args.cases,
            args.case_index,
            args.prefill_projection,
            args.layer_index,
            args.cached_last,
        )
    except Exception:  # noqa: BLE001 - reference data and exceptions remain private
        print(json.dumps({"passed": False, "error": "primitive_reference_failed"}))
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
