"""Private paired Gemma 4 source controls with frozen MLX weights and caching.

The two optional controls replace only scalar opmath and/or approximate GELU
with canonical FP32 opmath followed by a BF16 result. The stock profile remains
unchanged. PLE structure follows MLX-LM 1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7,
Copyright 2025 Apple Inc., MIT; its license is preserved in the distribution.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import subprocess
import types
from pathlib import Path

from gemma4_hf_reference import digest, metrics

REVISION = "1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7"


def run(args) -> dict:
    import mlx.core as mx
    import numpy as np
    from mlx import nn
    from mlx_lm.models import gemma4_text
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    direct = json.loads(
        importlib.metadata.distribution("mlx-lm").read_text("direct_url.json")
    )
    if (
        direct.get("vcs_info", {}).get("commit_id") != REVISION
        or importlib.metadata.version("mlx") != "0.32.3"
    ):
        raise ValueError("reference_identity_changed")
    output = args.output.resolve()
    root = Path(__file__).resolve().parents[1]
    if (
        output.exists()
        or subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(output)], cwd=root, check=False
        ).returncode
    ):
        raise ValueError("output_must_be_new_and_ignored")
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    cases = json.loads(args.cases.read_text())["cases"]
    if (
        not cases
        or len(cases) > 32
        or any(not c["tokens"] or len(c["tokens"]) > 8192 for c in cases)
    ):
        raise ValueError("invalid_case_geometry")
    model, _ = load_model(args.source, lazy=True, strict=True, trust_remote_code=False)
    mx.eval(model.parameters())
    language_model = getattr(model, "language_model", model)
    text = language_model.model
    if not isinstance(text, gemma4_text.Gemma4TextModel):
        raise TypeError("unexpected_text_module")
    original_projection = text._project_per_layer_inputs
    original_gelu, original_geglu = nn.gelu_approx, gemma4_text.geglu
    activation_source = Path(inspect.getfile(original_gelu))
    original_rms = nn.RMSNorm.__call__

    def canonical_gelu(x):
        f = x.astype(mx.float32)
        return (
            0.5
            * f
            * (1.0 + mx.tanh(math.sqrt(2 / math.pi) * (f + 0.044715 * f * f * f)))
        ).astype(x.dtype)

    def canonical_projection(this, input_embeddings, per_layer_inputs=None):
        projected = this.per_layer_model_projection(input_embeddings)
        projected = (
            projected.astype(mx.float32)
            * mx.array(this.per_layer_projection_scale, dtype=mx.float32)
        ).astype(projected.dtype)
        projected = mx.unflatten(
            projected,
            -1,
            (this.config.num_hidden_layers, this.hidden_size_per_layer_input),
        )
        projected = this.per_layer_projection_norm(projected)
        if per_layer_inputs is None:
            return projected
        summed = projected + per_layer_inputs
        return (
            summed.astype(mx.float32)
            * mx.array(this.per_layer_input_scale, dtype=mx.float32)
        ).astype(summed.dtype)

    def canonical_rms(this, x):
        normalized = mx.fast.rms_norm(x.astype(mx.float32), None, this.eps)
        return (normalized * this.weight.astype(mx.float32)).astype(x.dtype)

    def prefill(tokens):
        cache = make_prompt_cache(model)
        batches = [tokens[i : i + 1024] for i in range(0, len(tokens), 1024)]
        result = None
        for index, batch in enumerate(batches):
            parts = (
                [batch[:-1], batch[-1:]]
                if index == len(batches) - 1 and len(batch) > 1
                else [batch]
            )
            for part in parts:
                result = model(mx.array([part], dtype=mx.int32), cache=cache)
                mx.eval(result)
        return np.array(result[0, -1].astype(mx.float32)).astype("<f4")

    output.mkdir(parents=True, mode=0o700)
    trace_rows = []
    trace_kind = None
    trace_geometry = None
    if args.trace_bos or args.trace_prefill_case_index is not None:
        if args.trace_bos and args.trace_prefill_case_index is not None:
            raise ValueError("select_one_trace_geometry")
        trace_case = 0 if args.trace_bos else args.trace_prefill_case_index
        if not 0 <= trace_case < len(cases):
            raise ValueError("trace_case_out_of_bounds")
        trace_tokens = cases[trace_case]["tokens"]
        source_prefix = trace_tokens[:-1]
        trace_tokens = (
            trace_tokens
            if (args.trace_bos or args.trace_cached_last)
            else source_prefix
        )
        if not trace_tokens or len(trace_tokens) > 1024:
            raise ValueError("trace_requires_bounded_prefix")
        if args.trace_bos and len(trace_tokens) != 1:
            raise ValueError("bos_trace_requires_one_token_first_case")
        layer_indices = {id(layer): index for index, layer in enumerate(text.layers)}
        captured = {}
        attention_values = {}
        active_layer = None
        original_layer_call = gemma4_text.DecoderLayer.__call__
        original_sdpa = gemma4_text.scaled_dot_product_attention

        def capture_layer(this, *parts, **kwargs):
            nonlocal active_layer
            active_layer = layer_indices[id(this)]
            if active_layer == 0 and kwargs.get("per_layer_input") is not None:
                attention_values["ple-input-0"] = kwargs["per_layer_input"]
            result = original_layer_call(this, *parts, **kwargs)
            captured[layer_indices[id(this)]] = result[0]
            return result

        def capture_sdpa(q, k, v, *parts, **kwargs):
            result = original_sdpa(q, k, v, *parts, **kwargs)
            if active_layer == 0:
                attention_values.update(
                    {
                        "attention-query-0": q[0],
                        "attention-key-native-0": k.transpose(0, 2, 1, 3)[0],
                        "attention-value-native-0": v.transpose(0, 2, 1, 3)[0],
                        "attention-output-native-0": result.transpose(0, 2, 1, 3)[0],
                    }
                )
            return result

        cache = make_prompt_cache(model)
        if args.trace_cached_last:
            prefix_result = model(
                mx.array([source_prefix], dtype=mx.int32), cache=cache
            )
            mx.eval(prefix_result)
            trace_tokens = trace_tokens[-1:]
        try:
            gemma4_text.DecoderLayer.__call__ = capture_layer
            gemma4_text.scaled_dot_product_attention = capture_sdpa
            if args.trace_bos:
                logits = prefill(trace_tokens)
            else:
                result = model(
                    mx.array([trace_tokens], dtype=mx.int32),
                    cache=cache,
                )
                mx.eval(result)
                logits = np.array(result[0, -1].astype(mx.float32)).astype("<f4")
        finally:
            gemma4_text.DecoderLayer.__call__ = original_layer_call
            gemma4_text.scaled_dot_product_attention = original_sdpa
        values = {f"layer-{index}": value for index, value in captured.items()}
        values.update(attention_values)
        values["embedding"] = (
            text.embed_tokens(mx.array([trace_tokens], dtype=mx.int32))
            * text.embed_scale
        )
        values["final-hidden"] = text.norm(captured[len(text.layers) - 1][:, -1:])
        mx.eval(*values.values())
        trace_kind = (
            "cached-last-token"
            if args.trace_cached_last
            else ("bos" if args.trace_bos else "prefill-prefix-N-1")
        )
        trace_geometry = {
            "case_index": trace_case,
            "input_tokens": len(trace_tokens),
            "prefix_tokens": len(source_prefix) if args.trace_cached_last else 0,
            "input_sha256": hashlib.sha256(
                np.asarray(trace_tokens, dtype="<u4").tobytes()
            ).hexdigest(),
            "attention_heads": text.config.num_attention_heads,
            "attention_groups": text.config.num_key_value_heads,
            "attention_head_dim": text.config.head_dim,
            "attention_scale": 1.0,
            "sliding_window": text.config.sliding_window,
        }
        directory = output / (
            "trace-cached-last"
            if args.trace_cached_last
            else ("trace-bos" if args.trace_bos else "trace-prefill")
        )
        directory.mkdir(mode=0o700)
        arrays = {
            name: np.array(value.astype(mx.float32)).astype("<f4")
            for name, value in values.items()
        }
        arrays["logits"] = logits
        for name, values in arrays.items():
            path = directory / (name + ".f32")
            path.write_bytes(values.tobytes())
            path.chmod(0o600)
            trace_rows.append(
                {
                    "component": name,
                    "elements": values.size,
                    "shape": list(values.shape),
                    "sha256": digest(path),
                }
            )
    references, profiles = {}, []
    selected_profiles = args.profiles.split(",")
    allowed = {
        "stock",
        "canonical-scalars",
        "canonical-gelu",
        "canonical-both",
        "canonical-rms",
        "canonical-all",
        "float32-opmath",
    }
    if (
        not selected_profiles
        or selected_profiles[0] != "stock"
        or len(set(selected_profiles)) != len(selected_profiles)
        or set(selected_profiles) - allowed
    ):
        raise ValueError("invalid_precision_profile_selection")
    if (
        "float32-opmath" in selected_profiles
        and selected_profiles[-1] != "float32-opmath"
    ):
        raise ValueError("float32_control_must_run_last")
    if args.trace_only:
        if trace_kind is None:
            raise ValueError("trace_only_requires_trace_geometry")
        selected_profiles = []
    for profile in selected_profiles:
        text._project_per_layer_inputs = (
            types.MethodType(canonical_projection, text)
            if profile in ("canonical-scalars", "canonical-both", "canonical-all")
            else original_projection
        )
        canonical_activation = profile in (
            "canonical-gelu",
            "canonical-both",
            "canonical-all",
        )
        nn.RMSNorm.__call__ = (
            canonical_rms
            if profile in ("canonical-rms", "canonical-all")
            else original_rms
        )
        nn.gelu_approx = canonical_gelu if canonical_activation else original_gelu
        gemma4_text.geglu = (
            mx.compile(lambda gate, x: canonical_gelu(gate) * x, shapeless=True)
            if canonical_activation
            else original_geglu
        )
        if profile == "float32-opmath":
            model.apply(
                lambda x: (
                    x.astype(mx.float32)
                    if x.dtype in (mx.bfloat16, mx.float16, mx.float32)
                    else x
                )
            )
            mx.eval(model.parameters())
        directory = output / profile
        directory.mkdir(mode=0o700)
        (directory / "cases.json").write_text(json.dumps({"cases": cases}))
        values, rows = [], []
        for index, case in enumerate(cases):
            logits = prefill(case["tokens"])
            if not np.isfinite(logits).all():
                raise ValueError("nonfinite_control")
            path = directory / f"reference-{index}.f32"
            path.write_bytes(logits.tobytes())
            path.chmod(0o600)
            values.append(logits)
            rows.append(
                {"input_tokens": len(case["tokens"]), "reference_sha256": digest(path)}
            )
        references[profile] = values
        profiles.append({"profile": profile, "cases": rows})
    receipt = {
        "device": args.device,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_revision": REVISION,
        "source_sha256": digest(Path(inspect.getfile(gemma4_text))),
        "activation_source_sha256": digest(activation_source),
        "quantized_layers_source_sha256": digest(
            Path(inspect.getfile(nn.QuantizedLinear))
        ),
        "lock_sha256": digest(args.lock),
        "weights_sha256": digest(args.source / "model.safetensors"),
        "cases_sha256": digest(args.cases),
        "profiles": profiles,
        "bos_trace_components": trace_rows,
        "trace_kind": trace_kind,
        "trace_geometry": trace_geometry,
        "stock_to_control": {
            name: [
                {"input_tokens": len(case["tokens"]), **metrics(a, b)}
                for case, a, b in zip(cases, references["stock"], values, strict=True)
            ]
            for name, values in references.items()
            if name != "stock"
        },
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--profiles", default="stock,canonical-scalars,canonical-gelu,canonical-both"
    )
    parser.add_argument("--trace-bos", action="store_true")
    parser.add_argument("--trace-cached-last", action="store_true")
    parser.add_argument("--trace-prefill-case-index", type=int)
    parser.add_argument("--trace-only", action="store_true")
    args = parser.parse_args()
    if args.trace_cached_last and (
        args.trace_bos or args.trace_prefill_case_index is None
    ):
        parser.error("--trace-cached-last requires --trace-prefill-case-index")
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            result = run(args)
    except Exception:  # noqa: BLE001 - model inputs and exception data stay private
        print(
            json.dumps({"passed": False, "error": "gemma_precision_reference_failed"})
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
