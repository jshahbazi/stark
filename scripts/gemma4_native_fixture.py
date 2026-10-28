"""Create private Gemma 4 text fixtures from the pinned independent MLX-LM engine.

The reference source is Apple MLX-LM (MIT), commit
1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7. Native Stark never imports it.
Outputs contain private synthetic tokens and logits; stdout contains aggregate
counts and hashes only. Use the isolated MLX benchmark environment through uv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

SOURCE_REVISION = "1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7"
SOURCE_MODULES = (
    "mlx_lm/models/gemma4_text.py",
    "mlx_lm/models/rope_utils.py",
    "mlx_lm/models/cache.py",
    "mlx_lm/models/base.py",
    "mlx_lm/models/switch_layers.py",
)


def generate(
    output: Path,
    reference_checkout: Path,
    *,
    device: str,
    sliding_window: int = 16,
    weights_from: Path | None = None,
) -> dict:
    root = Path(__file__).resolve().parents[1]
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(output)], cwd=root, check=False
    )
    if ignored.returncode != 0 or output.exists():
        raise ValueError("fixture_destination_must_be_new_and_git_ignored")
    if not 1 <= sliding_window <= 128:
        raise ValueError("invalid_fixture_window")
    revision = subprocess.check_output(
        ["git", "-C", str(reference_checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    clean = subprocess.run(
        [
            "git",
            "-C",
            str(reference_checkout),
            "diff",
            "--quiet",
            SOURCE_REVISION,
            "--",
            "mlx_lm",
        ],
        check=False,
    )
    if revision != SOURCE_REVISION or clean.returncode != 0:
        raise ValueError("reference_source_identity_mismatch")
    hashes = {
        name: hashlib.sha256((reference_checkout / name).read_bytes()).hexdigest()
        for name in SOURCE_MODULES
    }
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.models import gemma4_text
    from tokenizers import Tokenizer, models

    package_root = Path(gemma4_text.__file__).resolve().parents[2]
    if any(
        hashlib.sha256((package_root / name).read_bytes()).hexdigest() != expected
        for name, expected in hashes.items()
    ):
        raise ValueError("installed_reference_does_not_match_pinned_source")
    mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
    mx.random.seed(20261028)
    config = {
        "model_type": "gemma4_text",
        "architectures": ["Gemma4ForCausalLM"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 6,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        # Use the actual E2B/E4B head geometry. Metal's attention kernels are
        # specialized for these widths; smaller mathematical toys cannot exercise
        # the production global-head fallback or normal generation dispatch.
        "head_dim": 256,
        "global_head_dim": 512,
        "hidden_size_per_layer_input": 32,
        "vocab_size": 128,
        "vocab_size_per_layer_input": 128,
        "num_kv_shared_layers": 2,
        "use_double_wide_mlp": True,
        "hidden_activation": "gelu_pytorch_tanh",
        "rms_norm_eps": 1e-6,
        "sliding_window": sliding_window,
        "max_position_embeddings": 128,
        "layer_types": ["sliding_attention", "full_attention"] * 3,
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        },
        "tie_word_embeddings": True,
        "final_logit_softcapping": 30.0,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "dtype": "bfloat16",
    }
    model = gemma4_text.Model(gemma4_text.ModelArgs.from_dict(config))
    for index, layer in enumerate(model.layers):
        layer.layer_scalar = mx.array([0.8 + 0.05 * index])
        if layer.self_attn.has_kv:
            # The pinned real E2B checkpoint has K-norm RMS gains .060–.132
            # (mean .112), while Q-norm gains are approximately one. Untrained
            # unit K gains produce an ill-conditioned sharp-attention toy whose
            # MLX CPU/GPU references disagree by >.5 RMSE. Use a representative
            # power-of-two K gain so the fixture tests architecture, not that
            # source engine's own numerically unstable random attention.
            layer.self_attn.k_norm.weight = layer.self_attn.k_norm.weight * 0.125
    model.set_dtype(mx.bfloat16)
    if weights_from is not None:
        if not weights_from.is_file() or weights_from.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("invalid_fixture_control_weights")
        model.load_weights(list(mx.load(str(weights_from)).items()))
    weights = dict(tree_flatten(model.parameters()))
    mx.eval(weights)
    key_gains = [
        float(np.sqrt(np.mean(np.asarray(value.astype(mx.float32)).astype("<f4") ** 2)))
        for name, value in weights.items()
        if name.endswith(".self_attn.k_norm.weight")
    ]
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config, sort_keys=True))
    mx.save_safetensors(
        str(source / "model.safetensors"), weights, metadata={"format": "mlx"}
    )
    tokenizer = Tokenizer(
        models.WordLevel({f"t{index}": index for index in range(128)}, unk_token="t3")
    )
    tokenizer.save(str(source / "tokenizer.json"))
    (source / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "t2",
                "eos_token": "t1",
                "pad_token": "t0",
                "unk_token": "t3",
                "chat_template": "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}",
            }
        )
    )
    tokens = [(index * 29 + 7) % 125 + 3 for index in range(65)]
    lengths = [1, 4, 16, 17, 31, 33, 65]
    cases = []
    references = []
    for index, length in enumerate(lengths):
        case_tokens = tokens[:length]
        cache = model.make_cache()
        # Native scoring splits the last token from the final prefill batch for
        # trailing shared-KV layers. Match those BF16 GEMM/GEMV boundaries in the
        # independent engine rather than comparing against a different schedule.
        if length > 1:
            prefix_logits = model(
                mx.array([case_tokens[:-1]], dtype=mx.int32), cache=cache
            )
            mx.eval(prefix_logits)
        logits = model(mx.array([case_tokens[-1:]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        values = np.asarray(logits[0, -1].astype(mx.float32)).astype("<f4")
        if not np.isfinite(values).all():
            raise ValueError("reference_produced_nonfinite_logits")
        payload = values.tobytes()
        (output / f"reference-{index}.f32").write_bytes(payload)
        references.append(hashlib.sha256(payload).hexdigest())
        cases.append({"tokens": case_tokens, "context_tokens": 128})
    (output / "cases.json").write_text(json.dumps({"cases": cases}))
    cache = model.make_cache()
    cached_rows = []
    for token in tokens:
        logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        cached_rows.append(np.asarray(logits[0, -1].astype(mx.float32)).astype("<f4"))
    cached = np.stack(cached_rows)
    (output / "cached-reference.f32").write_bytes(cached.tobytes())
    receipt = {
        "schema_version": 1,
        "architecture": "gemma4_text",
        "reference_engine": "mlx-lm",
        "reference_repository": "https://github.com/ml-explore/mlx-lm",
        "reference_revision": SOURCE_REVISION,
        "reference_module_sha256": hashes,
        "reference_license": "MIT",
        "reference_device": device,
        "source_dtype": "bfloat16",
        "execution_dtype": "bfloat16",
        "reference_batching": "native-scoring-trailing-shared-layer-split",
        "positions": len(tokens),
        "vocabulary": 128,
        "case_count": len(cases),
        "sliding_window": sliding_window,
        "local_head_dim": config["head_dim"],
        "global_head_dim": config["global_head_dim"],
        "per_layer_embedding_dim": config["hidden_size_per_layer_input"],
        "shared_kv_layers": config["num_kv_shared_layers"],
        "double_wide_shared_mlp": config["use_double_wide_mlp"],
        "key_norm_rms_mean": float(np.mean(key_gains)),
        "weights_sha256": hashlib.sha256(
            (source / "model.safetensors").read_bytes()
        ).hexdigest(),
        "cases_sha256": hashlib.sha256(
            (output / "cases.json").read_bytes()
        ).hexdigest(),
        "reference_sha256": references,
        "cached_reference_sha256": hashlib.sha256(cached.tobytes()).hexdigest(),
    }
    (output / "fixture.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-checkout", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--sliding-window", type=int, default=16)
    parser.add_argument("--weights-from", type=Path)
    args = parser.parse_args()
    try:
        receipt = generate(
            args.output.resolve(),
            args.reference_checkout.resolve(),
            device=args.device,
            sliding_window=args.sliding_window,
            weights_from=args.weights_from.resolve()
            if args.weights_from is not None
            else None,
        )
    except Exception:  # noqa: BLE001 - validation boundary must never print exception dumps
        print(
            json.dumps({"passed": False, "error": "gemma4_fixture_generation_failed"})
        )
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
