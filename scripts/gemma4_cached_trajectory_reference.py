#!/usr/bin/env python3
"""Capture private stock MLX Gemma 4 cached decode trajectories.

The report contains hashes and counts only. Optional component arrays are
written to a new ignored directory with private permissions.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from dense_profile_mlx import (
    MLX_LM_REVISION,
    ROOT,
    digest,
    quantization,
    read_json,
    token_digest,
    validate_fixture,
    weight_records,
    weights_digest,
)

EXPECTED_GREEDY_SHA256 = "03f17cff595a19aba064fb7ddf00547b52fd0cdf3c908b77db96f2727aba52a1"
REFERENCE_ENV = ROOT / "runtime/olmo3-gemma4/reference-env"


def require(value: bool, code: str) -> None:
    if not value:
        raise ValueError(code)


def ignored(path: Path) -> bool:
    return subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(path)],
        cwd=ROOT,
        check=False,
    ).returncode == 0


@contextlib.contextmanager
def quiet_process_output():
    sys.stdout.flush()
    sys.stderr.flush()
    stdout, stderr = os.dup(1), os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        with (
            open(os.devnull, "w") as null,
            contextlib.redirect_stdout(null),
            contextlib.redirect_stderr(null),
        ):
            yield
    finally:
        os.dup2(stdout, 1)
        os.dup2(stderr, 2)
        os.close(stdout)
        os.close(stderr)
        os.close(null_fd)


def write_private(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


def run(args: argparse.Namespace) -> dict:
    source = args.source.resolve(strict=True)
    cases_path = args.cases.resolve(strict=True)
    fixture_path = args.fixture.resolve(strict=True)
    lock_path = args.lock.resolve(strict=True)
    output = args.output.resolve()
    require(source.name == "238767527555cb75a05732a84dff5d6ba0dd6809", "source_revision")
    require(not output.exists() and ignored(output), "output_must_be_new_and_ignored")
    require(args.capture_step is None or 0 <= args.capture_step <= 128, "capture_step")
    require(args.capture_layer is None or (args.capture_step is not None and 0 <= args.capture_layer < 35), "capture_layer")

    config_path = source / "config.json"
    config = read_json(config_path)
    records = weight_records(source)
    fixture = read_json(fixture_path)
    validate_fixture(fixture)
    fixture_tokens = fixture.get("input_token_ids")
    require(isinstance(fixture_tokens, list) and len(fixture_tokens) == 512, "fixture_tokens")
    require(fixture.get("source_revision") == source.name, "fixture_source_revision")
    require(fixture.get("source_config_sha256") == digest(config_path), "fixture_config")
    require(fixture.get("source_weights") == records, "fixture_weights")
    require(fixture.get("quantization") == quantization(config), "fixture_quantization")
    require(fixture.get("decode_tokens") == 128, "fixture_decode_length")

    case_doc = read_json(cases_path, 16 * 1024 * 1024)
    entries = case_doc.get("cases") if isinstance(case_doc, dict) else None
    require(isinstance(entries, list), "cases_format")
    candidates = [entry for entry in entries if isinstance(entry, dict) and isinstance(entry.get("tokens"), list) and len(entry["tokens"]) >= 640]
    require(bool(candidates), "640_token_case_required")
    selected = max(candidates, key=lambda entry: len(entry["tokens"]))
    full_tokens = selected["tokens"][-640:]
    expected = full_tokens[512:]
    prompt = full_tokens[:512]
    require(prompt == fixture_tokens, "fixture_prefix_mismatch")
    require(all(type(token) is int and 0 <= token < config.get("text_config", config)["vocab_size"] for token in full_tokens), "token_vocabulary")
    require(token_digest(expected) == EXPECTED_GREEDY_SHA256, "expected_sequence_digest")

    direct = json.loads(importlib.metadata.distribution("mlx-lm").read_text("direct_url.json"))
    require(direct.get("vcs_info", {}).get("commit_id") == MLX_LM_REVISION, "mlx_lm_revision")
    environment = read_json(REFERENCE_ENV / "environment.json")
    require(digest(lock_path) == environment["lock_sha256"], "lock_hash")
    require(lock_path == (REFERENCE_ENV / "uv.lock").resolve(), "lock_path")
    require(platform.python_version() == environment["python"], "python_version")
    for package, version in environment["packages"].items():
        require(importlib.metadata.version(package) == version, "package_version")

    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models import gemma4_text
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    mx.set_default_device(mx.gpu)
    model, _ = load_model(source, lazy=True, strict=True, trust_remote_code=False)
    mx.eval(model.parameters())
    language_model = getattr(model, "language_model", model)
    text = language_model.model
    require(isinstance(text, gemma4_text.Gemma4TextModel), "unexpected_text_module")
    layer_indices = {id(layer): index for index, layer in enumerate(text.layers)}
    original_layer = gemma4_text.DecoderLayer.__call__
    original_sdpa = gemma4_text.scaled_dot_product_attention
    layer_outputs: dict[int, object] = {}
    operations: dict[str, object] = {}
    current_step = -1
    active_layer = None
    selected_layer = text.layers[args.capture_layer] if args.capture_layer is not None else None
    modules = {}
    if selected_layer is not None:
        a = selected_layer.self_attn
        for attribute, label in [("input_layernorm", "actual_input_norm"), ("post_attention_layernorm", "actual_post_attention_norm"), ("pre_feedforward_layernorm", "actual_mlp_norm"), ("post_feedforward_layernorm", "actual_post_mlp_norm"), ("per_layer_input_gate", "actual_ple_gate"), ("per_layer_projection", "actual_ple_projection"), ("post_per_layer_input_norm", "actual_ple_norm")]:
            modules[id(getattr(selected_layer, attribute))] = label
        for attribute, label in [("q_proj", "actual_q_projection"), ("q_norm", "actual_q_norm"), ("o_proj", "actual_o_projection"), ("k_proj", "actual_k_projection"), ("v_proj", "actual_v_projection"), ("k_norm", "actual_k_norm"), ("v_norm", "actual_v_norm")]:
            if hasattr(a, attribute):
                modules[id(getattr(a, attribute))] = label
        for attribute, label in [("up_proj", "actual_mlp_up"), ("gate_proj", "actual_mlp_gate"), ("down_proj", "actual_mlp_down")]:
            modules[id(getattr(selected_layer.mlp, attribute))] = label
    module_types = set() if selected_layer is None else {type(selected_layer.input_layernorm), type(selected_layer.self_attn.q_proj), type(selected_layer.self_attn.rope)}
    if selected_layer is not None and selected_layer.self_attn.has_kv:
        module_types.add(type(selected_layer.self_attn.v_norm))
    originals = {kind: kind.__call__ for kind in module_types}

    def capture_module(original):
        def wrapper(this, x, *parts, **kwargs):
            result = original(this, x, *parts, **kwargs)
            if current_step == args.capture_step and selected_layer is not None:
                if id(this) in modules:
                    name = modules[id(this)]
                    operations[name + "_input"] = x
                    operations[name] = result
                if this is selected_layer.self_attn.rope:
                    if x.shape[1] == selected_layer.self_attn.n_heads:
                        operations["actual_q_rope"] = result
                    else:
                        operations["actual_k_rope_native"] = result.transpose(0, 2, 1, 3)
            return result
        return wrapper

    def capture_layer(this, *parts, **kwargs):
        nonlocal active_layer
        active_layer = layer_indices[id(this)]
        if current_step == args.capture_step and this is selected_layer:
            operations["actual_layer_input"] = parts[0]
            operations["actual_ple_input"] = kwargs["per_layer_input"]
        result = original_layer(this, *parts, **kwargs)
        layer_outputs[layer_indices[id(this)]] = result[0]
        if current_step == args.capture_step and this is selected_layer:
            operations["actual_layer_output"] = result[0]
        return result

    def capture_sdpa(q, k, v, *parts, **kwargs):
        result = original_sdpa(q, k, v, *parts, **kwargs)
        if current_step == args.capture_step and active_layer == args.capture_layer:
            operations["actual_q_rope"] = q
            operations["actual_attention_keys_native"] = k.transpose(0, 2, 1, 3)
            operations["actual_attention_values_native"] = v.transpose(0, 2, 1, 3)
            operations["actual_attention_output"] = result.transpose(0, 2, 1, 3)
            if q.shape[-1] == 512 and k.shape[-2] < 1024:
                repeats = q.shape[1] // k.shape[1]
                scaled = q * mx.array(kwargs.get("scale", 1.0), dtype=q.dtype)
                grouped_q = mx.unflatten(scaled, 1, (k.shape[1], repeats))
                scores = mx.matmul(grouped_q, mx.expand_dims(k, 2).swapaxes(-1, -2))
                mask = kwargs.get("mask")
                if isinstance(mask, str):
                    require(mask == "causal", "source_mask")
                    mask = (mx.arange(q.shape[-2]) + k.shape[-2] - q.shape[-2])[:, None] >= mx.arange(k.shape[-2])[None, :]
                if mask is not None:
                    scores = mx.where(mask, scores, mx.array(mx.finfo(scores.dtype).min, scores.dtype)) if mask.dtype == mx.bool_ else scores + mask
                probabilities = mx.softmax(scores, axis=-1, precise=True)
                reconstructed = mx.matmul(probabilities, mx.expand_dims(v, 2))
                operations["actual_attention_scores"] = scores
                operations["actual_attention_probabilities"] = probabilities
                operations["actual_attention_reconstructed"] = reconstructed
        return result

    cache = make_prompt_cache(model)
    prefix = mx.array([prompt[:-1]], dtype=mx.int32)
    model(prefix, cache=cache)
    mx.eval([entry.state for entry in cache])
    mx.clear_cache()

    per_step = []
    selected_arrays = None
    operation_arrays = None
    native_greedy: list[int] = []
    gemma4_text.DecoderLayer.__call__ = capture_layer
    gemma4_text.scaled_dot_product_attention = capture_sdpa
    for kind, original in originals.items():
        kind.__call__ = capture_module(original)
    try:
        with mx.stream(mx.new_stream(mx.gpu)):
            for step in range(129):
                current_step = step
                token_id = prompt[-1] if step == 0 else expected[step - 1]
                layer_outputs.clear()
                token = mx.array([[token_id]], dtype=mx.int32)
                logits_batch = model(token, cache=cache)
                require(len(layer_outputs) == len(text.layers), "layer_capture_incomplete")
                captured = {f"layer_{index:02d}": layer_outputs[index] for index in range(len(text.layers))}
                captured["embedding"] = text.embed_tokens(token) * text.embed_scale
                captured["final_hidden"] = text.norm(layer_outputs[len(text.layers) - 1][:, -1:])
                captured["logits"] = logits_batch[:, -1, :]
                mx.eval(*captured.values())
                arrays = {
                    name: np.asarray(value.astype(mx.float32), dtype="<f4")
                    for name, value in captured.items()
                }
                if step < 128:
                    token_choice = mx.argmax(logits_batch[0, -1], axis=-1)
                    mx.eval(token_choice)
                    native_greedy.append(int(token_choice.item()))
                rows = []
                for name, array in sorted(arrays.items()):
                    raw = array.tobytes(order="C")
                    rows.append({"component": name, "elements": int(array.size), "shape": list(array.shape), "sha256": hashlib.sha256(raw).hexdigest()})
                per_step.append({"step": step, "components": rows})
                if step == args.capture_step:
                    selected_arrays = arrays
                    if selected_layer is not None:
                        a = selected_layer.self_attn
                        operations["actual_qkv_input"] = operations["actual_q_projection_input"]
                        if a.has_kv:
                            if not hasattr(a, "v_proj"):
                                operations["actual_v_projection"] = operations["actual_k_projection"]
                                operations["actual_v_projection_input"] = operations["actual_k_projection_input"]
                            operations["actual_qkv"] = mx.concatenate([operations["actual_q_projection"], operations["actual_k_projection"], operations["actual_v_projection"]], axis=-1)
                        else:
                            operations["actual_qkv"] = operations["actual_q_projection"]
                        mx.eval(*operations.values())
                        operation_arrays = {name: np.array(value.astype(mx.float32), dtype="<f4", copy=True) for name, value in operations.items()}
                        if "actual_attention_reconstructed" in operation_arrays:
                            require(np.array_equal(operation_arrays["actual_attention_reconstructed"].reshape(operation_arrays["actual_q_rope"].shape).transpose(0, 2, 1, 3), operation_arrays["actual_attention_output"]), "fallback_reconstruction_changed_stock")
    finally:
        gemma4_text.DecoderLayer.__call__ = original_layer
        gemma4_text.scaled_dot_product_attention = original_sdpa
        for kind, original in originals.items():
            kind.__call__ = original

    greedy_bytes = b"".join(int(value).to_bytes(4, "little") for value in native_greedy)
    greedy_hash = hashlib.sha256(greedy_bytes).hexdigest()
    require(greedy_hash == EXPECTED_GREEDY_SHA256, "source_greedy_mismatch")
    output.mkdir(parents=True, mode=0o700)
    os.chmod(output, 0o700)
    capture_records = []
    if selected_layer is not None:
        a = selected_layer.self_attn
        require(operation_arrays is not None, "operation_capture_missing")
        control_dir = output / "controls"
        control_dir.mkdir(mode=0o700)
        operation_records = []
        for name, array in sorted(operation_arrays.items()):
            payload = array.tobytes(order="C")
            write_private(control_dir / (name + ".f32"), payload)
            operation_records.append({"operation": name, "elements": int(array.size), "sha256": hashlib.sha256(payload).hexdigest()})
        identity = {"mlx_lm_revision": MLX_LM_REVISION, "actual_layer_index": args.capture_layer, "actual_projection_rows": 1, "actual_position_offset": 511 + args.capture_step, "actual_has_kv": a.has_kv, "actual_kv_rows": int(operation_arrays["actual_attention_keys_native"].shape[1]), "cached_last": True, "cached_trajectory": True, "trajectory_step": args.capture_step, "cases_sha256": digest(cases_path)}
        write_private(control_dir / "receipt.json", json.dumps({"device": "gpu", "identity": identity, "operations": operation_records, "harness_sha256": digest(Path(__file__).resolve()), "source_config_sha256": digest(config_path), "source_weights": records, "lock_sha256": digest(lock_path)}, sort_keys=True).encode())
    if selected_arrays is not None:
        capture_dir = output / f"step-{args.capture_step:03d}"
        capture_dir.mkdir(mode=0o700)
        for name, array in sorted(selected_arrays.items()):
            path = capture_dir / f"{name}.f32"
            payload = array.tobytes(order="C")
            write_private(path, payload)
            capture_records.append({"component": name, "elements": int(array.size), "shape": list(array.shape), "sha256": hashlib.sha256(payload).hexdigest(), "file": path.name})
    report = {
        "schema_version": 1,
        "status": "ready",
        "source_revision": source.name,
        "source_config_sha256": digest(config_path),
        "source_weights": records,
        "source_weights_sha256": weights_digest(records),
        "quantization": quantization(config),
        "mlx_lm_revision": MLX_LM_REVISION,
        "mlx_version": importlib.metadata.version("mlx"),
        "lock_sha256": digest(lock_path),
        "cases_sha256": digest(cases_path),
        "case_selection": "longest-at-least-640-first-tie-last-640",
        "fixture_sha256": digest(fixture_path),
        "harness_sha256": digest(Path(__file__).resolve()),
        "input_tokens": 512,
        "returned_tokens": 128,
        "logit_steps": 129,
        "lookahead_step": 128,
        "source_greedy_sha256": greedy_hash,
        "source_greedy_matches_expected": native_greedy == expected,
        "steps": per_step,
        "captured_step": args.capture_step,
        "captured_components": capture_records,
    }
    report_path = output / "trajectory-reference.json"
    write_private(report_path, (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode())
    return {"status": "ready", "report": str(report_path), "source_greedy_sha256": greedy_hash, "source_greedy_matches_expected": native_greedy == expected, "logit_steps": 129, "captured_step": args.capture_step}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--capture-step", type=int)
    parser.add_argument("--capture-layer", type=int)
    args = parser.parse_args()
    try:
        with quiet_process_output():
            result = run(args)
    except Exception:  # noqa: BLE001 - framework diagnostics may contain private data.
        print(json.dumps({"status": "error", "error": "reference_failed"}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
