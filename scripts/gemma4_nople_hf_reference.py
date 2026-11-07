"""Create private canonical BF16 no-PLE Gemma decoder controls on CPU.

The bounded synthetic decoder preserves production head widths and exercises
negative/nonunit layer scalars. Optional MoE controls use source router and
expert branches. This does not qualify a full released checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

SOURCE = "4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    import torch
    import transformers
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models
    from transformers.models.gemma4 import modeling_gemma4
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    if (
        torch.__version__.split("+")[0] != "2.14.1"
        or transformers.__version__ != "5.15.1"
    ):
        raise ValueError("reference_versions_changed")
    if digest(modeling_gemma4.__file__) != SOURCE:
        raise ValueError("reference_source_changed")
    output = args.output.resolve()
    root = Path(__file__).resolve().parents[1]
    if (
        output.exists()
        or subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(output)], cwd=root, check=False
        ).returncode
    ):
        raise ValueError("output_must_be_new_and_ignored")
    output.mkdir(parents=True, mode=0o700)
    source = output / "source"
    source.mkdir(mode=0o700)
    config = {
        "model_type": "gemma4_text",
        "architectures": ["Gemma4ForCausalLM"],
        "dtype": "bfloat16",
        "hidden_size": args.model_dim,
        "intermediate_size": args.model_dim * 2,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 1,
        "head_dim": 256,
        "global_head_dim": 512,
        "vocab_size": 128,
        "vocab_size_per_layer_input": 128,
        "hidden_size_per_layer_input": 0,
        "num_kv_shared_layers": 0,
        "attention_k_eq_v": True,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 1024,
        "sliding_window": 16,
        "tie_word_embeddings": True,
        "hidden_activation": "gelu_pytorch_tanh",
        "final_logit_softcapping": 30.0,
        "use_double_wide_mlp": False,
        "use_bidirectional_attention": "vision",
        "eos_token_id": 1,
        "bos_token_id": 2,
        "pad_token_id": 0,
        "layer_types": ["sliding_attention", "full_attention"] * 3,
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        },
        "enable_moe_block": args.moe,
    }
    if args.moe:
        config.update(num_experts=4, top_k_experts=2, moe_intermediate_size=64)
    if (
        args.layers not in (1, 2, 6)
        or args.query_key_norm_scale not in (0.125, 1.0)
        or args.model_dim not in (64, 512)
    ):
        raise ValueError("unsupported_control_geometry")
    config["num_hidden_layers"] = args.layers
    config["layer_types"] = (
        ["full_attention"] if args.layers == 1 else config["layer_types"][: args.layers]
    )
    torch.set_num_threads(4)
    torch.manual_seed(20261107)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        hf_config = Gemma4TextConfig(**copy.deepcopy(config))
        hf_config._attn_implementation = "sdpa"
        model = modeling_gemma4.Gemma4ForCausalLM(hf_config).eval()
    finally:
        torch.set_default_dtype(previous)
    scalars = [-0.5, 0.625, -0.75, 0.875, 1.125, -0.25][: args.layers]
    with torch.no_grad():
        for index, layer in enumerate(model.model.layers):
            layer.layer_scalar.fill_(scalars[index])
            # Keep random synthetic attention away from BF16-sensitive one-hot
            # transitions while preserving production head widths and masks.
            layer.self_attn.q_norm.weight.fill_(args.query_key_norm_scale)
            layer.self_attn.k_norm.weight.fill_(args.query_key_norm_scale)
            if args.moe:
                layer.router.scale.copy_(
                    torch.linspace(0.5, 1.5, args.model_dim).bfloat16()
                )
                layer.router.per_expert_scale.copy_(
                    torch.tensor([0.75, 1.25, -0.5, 1.5]).bfloat16()
                )
                layer.post_feedforward_layernorm_1.weight.fill_(1.25)
                layer.post_feedforward_layernorm_2.weight.fill_(0.75)
                layer.pre_feedforward_layernorm_2.weight.copy_(
                    torch.linspace(0.75, 1.25, args.model_dim).bfloat16()
                )
    state = {
        name: value.detach().contiguous().clone()
        for name, value in model.state_dict().items()
        if name != "lm_head.weight"
    }
    save_file(state, source / "model.safetensors")
    (source / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")
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
                "chat_template": "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}",
            }
        )
    )
    cases = []
    with torch.inference_mode():
        for index, length in enumerate([1, 9, 31, 63]):
            tokens = [2] + [
                (position * 17 + index * 3) % 124 + 4 for position in range(length - 1)
            ]
            result = model(
                input_ids=torch.tensor([tokens]), use_cache=False, logits_to_keep=1
            ).logits[0, -1]
            path = output / f"case-{index}-logits.f32"
            path.write_bytes(result.float().numpy().astype("<f4").tobytes())
            cases.append(
                {
                    "tokens": tokens,
                    "context_tokens": 128,
                    "logits": path.name,
                    "logits_sha256": digest(path),
                    "greedy_token": int(result.argmax()),
                }
            )
    manifest = {
        "scope": "synthetic-nople-moe" if args.moe else "synthetic-nople-dense",
        "reference_profile": "transformers-canonical",
        "reference_version": transformers.__version__,
        "source_sha256": SOURCE,
        "source_weights_sha256": digest(source / "model.safetensors"),
        "layer_scalars": scalars,
        "model_dim": args.model_dim,
        "query_key_norm_scale": args.query_key_norm_scale,
        "cases": cases,
        "complete": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return {
        "passed": True,
        "scope": manifest["scope"],
        "cases": len(cases),
        "manifest_sha256": digest(output / "manifest.json"),
    }


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--moe", action="store_true")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--query-key-norm-scale", type=float, default=0.125)
    args = parser.parse_args()
    try:
        with (
            open(os.devnull, "w") as quiet,
            contextlib.redirect_stdout(quiet),
            contextlib.redirect_stderr(quiet),
        ):
            report = run(args)
    except Exception as error:  # noqa: BLE001 - keep private framework failures content-free
        print(json.dumps({"passed": False, "error_type": type(error).__name__}))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
