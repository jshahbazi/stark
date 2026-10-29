"""Write private layer and attention traces from unchanged pinned MLX-LM OLMo3."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
from pathlib import Path

REVISION = "1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7"
MLX_VERSION = "0.32.3"
MAX_PREFIX_TOKENS = 1024


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def tree_digest(paths: list[Path]) -> str:
    value = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        value.update(path.name.encode("utf-8"))
        value.update(bytes.fromhex(digest(path)))
    return value.hexdigest()


def private_output(path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    if path.exists() or path.is_symlink():
        raise ValueError("output_must_be_new")
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(path)],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise ValueError("output_must_be_ignored")


def run(args: argparse.Namespace) -> dict:
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models import olmo3
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    direct = json.loads(
        importlib.metadata.distribution("mlx-lm").read_text("direct_url.json")
    )
    if (
        direct.get("vcs_info", {}).get("commit_id") != REVISION
        or importlib.metadata.version("mlx") != MLX_VERSION
    ):
        raise ValueError("reference_identity_changed")

    output = args.output.resolve()
    private_output(output)
    cases_document = json.loads(args.cases.read_text())
    cases = cases_document.get("cases")
    if not isinstance(cases, list) or not 0 <= args.case_index < len(cases):
        raise ValueError("invalid_case_selection")
    tokens = cases[args.case_index].get("tokens")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(not isinstance(token, int) or token < 0 for token in tokens)
        or len(tokens) < 2
    ):
        raise ValueError("invalid_case_tokens")
    prefix = tokens[:-1]
    if len(prefix) > MAX_PREFIX_TOKENS:
        raise ValueError("prefix_exceeds_limit")

    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    model, _ = load_model(args.source, lazy=True, strict=True, trust_remote_code=False)
    mx.eval(model.parameters())
    text = getattr(model, "model", None)
    if not isinstance(text, olmo3.Olmo3Model):
        raise TypeError("unexpected_olmo3_model")

    layer_outputs: dict[int, object] = {}
    layer_inputs: dict[int, object] = {}
    captures: dict[str, object] = {}
    active_layer: int | None = None
    capture_enabled = not args.cached_step
    capture_layers = set(range(len(text.layers))) if args.cached_step else {0, 1, 3, 11}
    layer_ids = {id(layer): index for index, layer in enumerate(text.layers)}
    original_layer_call = olmo3.Olmo3DecoderLayer.__call__
    original_attention_call = olmo3.Olmo3Attention.__call__
    original_sdpa = olmo3.scaled_dot_product_attention
    original_mlp_call = olmo3.Olmo3MLP.__call__
    mlp_ids = {id(layer.mlp): index for index, layer in enumerate(text.layers)}

    def capture_layer(this, x, *parts, **kwargs):
        if not capture_enabled:
            return original_layer_call(this, x, *parts, **kwargs)
        index = layer_ids[id(this)]
        layer_inputs[index] = x
        result = original_layer_call(this, x, *parts, **kwargs)
        layer_outputs[index] = result
        return result

    def capture_attention(this, x, *parts, **kwargs):
        nonlocal active_layer
        active_layer = this.layer_idx
        result = original_attention_call(this, x, *parts, **kwargs)
        if capture_enabled and this.layer_idx in capture_layers:
            # Re-evaluate deterministic source operators on the exact input;
            # the original forward and its attention operands stay unchanged.
            index = this.layer_idx
            q_projected = this.q_proj(x)
            k_projected = this.k_proj(x)
            q_normalized = this.q_norm(q_projected)
            k_normalized = this.k_norm(k_projected)
            captures[f"layer-{index}-attention-output"] = result
            captures[f"layer-{index}-post-attention-normalized"] = text.layers[
                index
            ].post_attention_layernorm(result)
            captures[f"layer-{index}-attention-projection-input"] = x
            captures[f"layer-{index}-q-projection"] = q_projected
            captures[f"layer-{index}-k-projection"] = k_projected
            captures[f"layer-{index}-v-projection"] = this.v_proj(x)
            captures[f"layer-{index}-q-normalized"] = q_normalized
            captures[f"layer-{index}-k-normalized"] = k_normalized
            if hasattr(this.rope, "_freqs") and hasattr(this.rope, "mscale"):
                if not hasattr(this.rope, "_freqs") or not hasattr(this.rope, "mscale"):
                    raise ValueError("expected_yarn_layer")
                captures[f"layer-{index}-yarn-frequencies"] = this.rope._freqs
                q_shaped = q_normalized.reshape(
                    1, x.shape[1], this.num_attention_heads, this.head_dim
                ).transpose(0, 2, 1, 3)
                k_shaped = k_normalized.reshape(
                    1, x.shape[1], this.num_key_value_heads, this.head_dim
                ).transpose(0, 2, 1, 3)
                # This is the exact source YaRN multiply operator before RoPE.
                captures[f"layer-{index}-q-yarn-prescaled"] = q_shaped.at[
                    ..., : this.rope.dims
                ].multiply(this.rope.mscale)[0]
                captures[f"layer-{index}-k-yarn-prescaled-native"] = (
                    k_shaped.at[..., : this.rope.dims]
                    .multiply(this.rope.mscale)
                    .transpose(0, 2, 1, 3)[0]
                )
        return result

    def capture_mlp(this, x):
        result = original_mlp_call(this, x)
        index = mlp_ids[id(this)]
        if capture_enabled and (args.cached_step or index == 11):
            gate = this.gate_proj(x)
            up = this.up_proj(x)
            captures[f"layer-{index}-mlp-input"] = x
            captures[f"layer-{index}-gate-projection"] = gate
            captures[f"layer-{index}-up-projection"] = up
            captures[f"layer-{index}-swiglu-output"] = olmo3.swiglu(gate, up)
            captures[f"layer-{index}-down-projection"] = result
            captures[f"layer-{index}-post-feedforward-normalized"] = text.layers[
                index
            ].post_feedforward_layernorm(result)
        return result

    def capture_sdpa(q, k, v, *parts, **kwargs):
        result = original_sdpa(q, k, v, *parts, **kwargs)
        if capture_enabled and active_layer in capture_layers:
            # The engine packs token-major [L, heads, D] attention buffers.
            index = active_layer
            captures.update(
                {
                    f"layer-{index}-sdpa-query": q,
                    f"layer-{index}-sdpa-key": k,
                    f"layer-{index}-sdpa-value": v,
                    f"layer-{index}-sdpa-output": result,
                    f"attention-query-{index}": q[0],
                    f"attention-key-native-{index}": k.transpose(0, 2, 1, 3)[0],
                    f"attention-value-native-{index}": v.transpose(0, 2, 1, 3)[0],
                    f"attention-output-native-{index}": result.transpose(0, 2, 1, 3)[0],
                    f"prepared-key-native-{index}": k.transpose(0, 2, 1, 3)[
                        0, -q.shape[2] :
                    ],
                    f"prepared-value-native-{index}": v.transpose(0, 2, 1, 3)[
                        0, -q.shape[2] :
                    ],
                    f"layer-{index}-key-native-packed": k.transpose(0, 2, 1, 3),
                    f"layer-{index}-value-native-packed": v.transpose(0, 2, 1, 3),
                    f"layer-{index}-sdpa-output-native-packed": result.transpose(
                        0, 2, 1, 3
                    ),
                }
            )
        return result

    try:
        olmo3.Olmo3DecoderLayer.__call__ = capture_layer
        olmo3.Olmo3Attention.__call__ = capture_attention
        olmo3.scaled_dot_product_attention = capture_sdpa
        olmo3.Olmo3MLP.__call__ = capture_mlp
        cache = make_prompt_cache(model)
        trace_tokens = tokens[-1:] if args.cached_step else prefix
        if args.cached_step:
            model(mx.array([prefix], dtype=mx.int32), cache=cache)
            mx.eval([entry.state for entry in cache])
            capture_enabled = True
            active_layer = None
        model(mx.array([trace_tokens], dtype=mx.int32), cache=cache)
        embedding = text.embed_tokens(mx.array([trace_tokens], dtype=mx.int32))
        final_hidden = text.norm(layer_outputs[len(text.layers) - 1][:, -1:])
        logits = model.lm_head(final_hidden)
        mx.eval(
            logits, embedding, final_hidden, *captures.values(), *layer_outputs.values()
        )
    finally:
        olmo3.Olmo3DecoderLayer.__call__ = original_layer_call
        olmo3.Olmo3Attention.__call__ = original_attention_call
        olmo3.scaled_dot_product_attention = original_sdpa
        olmo3.Olmo3MLP.__call__ = original_mlp_call

    arrays = {f"layer-{index}": value for index, value in layer_outputs.items()}
    arrays.update(captures)
    arrays.update(
        {
            "embedding": embedding,
            "final-hidden": final_hidden,
            "logits": logits,
        }
    )
    output.mkdir(parents=True, mode=0o700)
    trace_output = output / "trace-prefill"
    trace_output.mkdir(mode=0o700)
    component_rows = []
    for name, value in arrays.items():
        data = np.asarray(value.astype(mx.float32)).astype("<f4")
        if not np.isfinite(data).all():
            raise ValueError("nonfinite_trace")
        path = trace_output / f"{name}.f32"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data.tobytes())
        component_rows.append(
            {
                "component": name,
                "shape": list(data.shape),
                "elements": int(data.size),
                "sha256": digest(path),
            }
        )

    source_file = Path(olmo3.__file__).resolve()
    weight_files = sorted(args.source.glob("model*.safetensors"))
    if not weight_files:
        raise ValueError("model_weights_not_found")
    attention = text.layers[0].self_attn
    receipt = {
        "device": args.device,
        "harness_sha256": digest(Path(__file__)),
        "source_revision": args.source.name,
        "source_config_sha256": digest(args.source / "config.json"),
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_revision": REVISION,
        "model_source_sha256": digest(source_file),
        "lock_sha256": digest(args.lock),
        "weights_sha256": tree_digest(weight_files),
        "cases_sha256": digest(args.cases),
        "selected_case_index": args.case_index,
        "input_tokens": len(trace_tokens),
        "cached_step": args.cached_step,
        "prefix_tokens": len(prefix) if args.cached_step else 0,
        "position_offset": len(prefix) if args.cached_step else 0,
        "input_token_sha256": hashlib.sha256(
            np.asarray(trace_tokens, dtype="<u4").tobytes()
        ).hexdigest(),
        "geometry": {
            "hidden_size": text.layers[0].hidden_size,
            "layers": len(text.layers),
            "attention_heads": attention.num_attention_heads,
            "attention_groups": attention.num_key_value_heads,
            "head_dim": attention.head_dim,
            "attention_scale": attention.scale,
            "layer_types": list(text.layer_types),
        },
        "prepared_rope_controls": {
            "layers": sorted(capture_layers),
            "yarn_layer": 3,
            "yarn_dims": text.layers[3].self_attn.rope.dims,
            "yarn_input_mscale": text.layers[3].self_attn.rope.mscale,
            "yarn_input_mscale_bf16": float(
                mx.array(text.layers[3].self_attn.rope.mscale, dtype=mx.bfloat16).item()
            ),
            "yarn_frequencies_component": "layer-3-yarn-frequencies",
            "yarn_frequencies_dtype": str(text.layers[3].self_attn.rope._freqs.dtype),
            "yarn_input_scale_boundary": "bf16-array-multiply-before-fast-rope",
            "yarn_by_layer": {
                str(index): {
                    "dims": text.layers[index].self_attn.rope.dims,
                    "input_mscale": text.layers[index].self_attn.rope.mscale,
                    "frequencies_component": f"layer-{index}-yarn-frequencies",
                }
                for index in capture_layers
                if hasattr(text.layers[index].self_attn.rope, "mscale")
            },
        },
        "components": component_rows,
        "bos_trace_components": component_rows,
        "trace_geometry": {
            "input_tokens": len(trace_tokens),
            "kv_tokens": len(tokens) if args.cached_step else len(prefix),
            "position_offset": len(prefix) if args.cached_step else 0,
            "attention_heads": attention.num_attention_heads,
            "attention_groups": attention.num_key_value_heads,
            "attention_head_dim": attention.head_dim,
            "attention_scale": attention.scale,
            "sliding_window": text.sliding_window,
        },
    }
    receipt_path = output / "receipt.json"
    descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--case-index", type=int, default=2)
    parser.add_argument("--cached-step", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            receipt = run(args)
    except (Exception, KeyboardInterrupt):  # noqa: BLE001 - keep private details off stdout
        print(json.dumps({"passed": False, "error": "olmo3_prefill_reference_failed"}))
        return 1
    print(
        json.dumps(
            {
                "passed": True,
                "components": len(receipt["components"]),
                "input_tokens": receipt["input_tokens"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
