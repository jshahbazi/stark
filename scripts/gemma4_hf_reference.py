"""Write private canonical Transformers Gemma 4 text references.

Run through the repository's locked tools environment. Only the text decoder is
loaded from the selected local safetensors snapshot; modality towers are never
constructed. Tokens and full-vocabulary logits stay in the ignored destination.
Stdout reports aggregate errors, source hashes and dependency identities only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def metrics(left, right) -> dict:
    import numpy as np

    a, b = left.astype("<f8"), right.astype("<f8")
    error = a - b
    centered = error - error.mean()
    reference_centered = a - a.mean()
    log_p = a - (a.max() + np.log(np.exp(a - a.max()).sum()))
    log_q = b - (b.max() + np.log(np.exp(b - b.max()).sum()))
    return {
        "max_abs_error": float(np.abs(error).max()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "centered_relative_l2": float(
            np.linalg.norm(centered) / max(np.linalg.norm(reference_centered), 1e-30)
        ),
        "reference_to_candidate_kl": float(
            max(0.0, np.sum(np.exp(log_p) * (log_p - log_q)))
        ),
        "greedy_agreement": bool(np.argmax(a) == np.argmax(b)),
    }


def generate(
    source: Path, cases_path: Path, output: Path, indices: list[int], *, attention: str
) -> dict:
    import numpy as np
    import torch
    import transformers
    from safetensors import safe_open
    from transformers.initialization import no_init_weights
    from transformers.models.gemma4 import modeling_gemma4
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    root = Path(__file__).resolve().parents[1]
    if (
        output.exists()
        or subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(output)], cwd=root, check=False
        ).returncode
        != 0
    ):
        raise ValueError("output_must_be_new_and_git_ignored")
    torch.set_num_threads(8)
    outer = json.loads((source / "config.json").read_text())
    config_dict = outer.get("text_config", outer)
    if config_dict.get("model_type") != "gemma4_text":
        raise ValueError("source_is_not_gemma4_text")
    cases = json.loads(cases_path.read_text())["cases"]
    if (
        not indices
        or len(indices) > 8
        or any(index < 0 or index >= len(cases) for index in indices)
    ):
        raise ValueError("invalid_reference_case_selection")
    selected = [cases[index] for index in indices]
    if any(not case["tokens"] or len(case["tokens"]) > 8192 for case in selected):
        raise ValueError("invalid_reference_case_geometry")
    config = Gemma4TextConfig(**config_dict)
    config._attn_implementation = attention
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with no_init_weights():
            model = modeling_gemma4.Gemma4ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    expected = set(model.state_dict())
    source_prefix = (
        "model.language_model." if outer.get("model_type") == "gemma4" else "model."
    )
    state = {}
    weight_path = source / "model.safetensors"
    with safe_open(str(weight_path), framework="pt", device="cpu") as weights:
        for name in weights.keys():  # noqa: SIM118 - safetensors exposes keys(), not mapping iteration
            if name.startswith(source_prefix):
                destination = "model." + name.removeprefix(source_prefix)
                if destination in expected:
                    state[destination] = weights.get_tensor(name)
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    if set(state) != expected:
        raise ValueError("canonical_text_weight_inventory_mismatch")
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.eval()
    output.mkdir(parents=True)
    reports = []
    references = {}

    def prefill(tokens):
        cache = None
        result = None
        # Match the native scorer's 1024-token batches and final shared-layer
        # split. Source and candidate still own independent cache implementations.
        batches = [
            tokens[index : index + 1024] for index in range(0, len(tokens), 1024)
        ]
        for index, batch in enumerate(batches):
            parts = [batch]
            if (
                index == len(batches) - 1
                and len(batch) > 1
                and config.num_kv_shared_layers
            ):
                parts = [batch[:-1], batch[-1:]]
            for part in parts:
                result = model(
                    torch.tensor([part], dtype=torch.long),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = result.past_key_values
        return result.logits[0, -1].float().cpu().numpy().astype("<f4")

    with torch.inference_mode():
        for dtype_name, dtype in (
            ("bfloat16", torch.bfloat16),
            ("float32", torch.float32),
        ):
            # Parameters were loaded in BF16 while the canonical rotary
            # constructor explicitly created FP32 frequencies. A blanket BF16
            # .to() would incorrectly round those nonpersistent constants.
            if dtype_name != "bfloat16":
                model.to(dtype=dtype)
                for module in model.modules():
                    if isinstance(
                        module, modeling_gemma4.Gemma4TextScaledWordEmbedding
                    ):
                        module.embed_scale.fill_(module.scalar_embed_scale)
            directory = output / dtype_name
            directory.mkdir()
            (directory / "cases.json").write_text(json.dumps({"cases": selected}))
            rows = []
            values = []
            for index, case in enumerate(selected):
                started = time.perf_counter()
                logits = prefill(case["tokens"])
                if not np.isfinite(logits).all():
                    raise ValueError("nonfinite_canonical_reference")
                payload = logits.tobytes()
                (directory / f"reference-{index}.f32").write_bytes(payload)
                values.append(logits)
                rows.append(
                    {
                        "input_tokens": len(case["tokens"]),
                        "elapsed_seconds": time.perf_counter() - started,
                        "reference_sha256": hashlib.sha256(payload).hexdigest(),
                        "vocabulary": len(logits),
                    }
                )
            references[dtype_name] = values
            report = {
                "dtype": dtype_name,
                "device": "cpu",
                "attention": attention,
                "cases": rows,
            }
            if dtype_name == "bfloat16":
                for index, case in enumerate(selected):
                    if len(case["tokens"]) > 1:
                        reset = (
                            model(
                                torch.tensor([case["tokens"][-1:]], dtype=torch.long),
                                use_cache=False,
                                logits_to_keep=1,
                            )
                            .logits[0, -1]
                            .float()
                            .cpu()
                            .numpy()
                        )
                        report["context_reset_negative_control"] = {
                            "input_tokens": len(case["tokens"]),
                            **metrics(values[index], reset),
                        }
                        break
            reports.append(report)
    receipt = {
        "schema_version": 1,
        "reference_engine": "huggingface-transformers",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "tools_lock_sha256": digest(root / "tools/uv.lock"),
        "modeling_gemma4_sha256": digest(Path(modeling_gemma4.__file__)),
        "source_config_sha256": digest(source / "config.json"),
        "weights_sha256": digest(weight_path),
        "input_cases_sha256": digest(cases_path),
        "selected_case_indices": indices,
        "batching": "native-score-1024-trailing-shared-layer-split",
        "profiles": reports,
        "bf16_to_float32_control": [
            {"input_tokens": len(case["tokens"]), **metrics(a, b)}
            for case, a, b in zip(
                selected, references["bfloat16"], references["float32"], strict=True
            )
        ],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-indices", default="0,3")
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="sdpa")
    args = parser.parse_args()
    try:
        result = generate(
            args.source.resolve(),
            args.cases.resolve(),
            args.output.resolve(),
            [int(value) for value in args.case_indices.split(",")],
            attention=args.attention,
        )
    except Exception:  # noqa: BLE001 - validation must never emit prompts or exception dumps
        print(json.dumps({"passed": False, "error": "gemma4_hf_reference_failed"}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
