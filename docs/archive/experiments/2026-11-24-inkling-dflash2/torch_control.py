"""Check sampled draft blocks using z-lab/dflash's independent PyTorch model."""

import argparse
import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import Qwen3Config


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "config", "draft", "target", "trace"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--offsets", type=int, nargs="+", default=[0, 31, 95, 159])
    parser.add_argument(
        "--all-anchors", action="store_true", help="walk every width independently on the CPU reference"
    )
    parser.add_argument("--candidates", type=Path, help="compare MLX candidate rows without logging token IDs")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("official_dflash", args.reference)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    config = Qwen3Config.from_json_file(str(args.config))
    config._attn_implementation = "sdpa"
    # Load the unmodified PyTorch block, strictly, without materializing another
    # initialized copy of the 0.5B parameters.
    with torch.device("meta"):
        model = ref.DFlash2DraftModel(config)
    weights = load_file(args.draft)
    for suffix in ("predecessor_codebook", "successor_codebook"):
        k = "candidate_selector." + suffix
        weights[k + ".weight"] = weights.pop(k)
    model.load_state_dict(weights, strict=True, assign=True)
    # Rotary inv_freq is a nonpersistent buffer and needs a real CPU allocation.
    model.rotary_emb = ref.Qwen3RotaryEmbedding(config)
    model.eval()
    with safe_open(args.target, framework="pt") as f:
        embed = f.get_tensor("model.llm.embed.weight")
        norm = f.get_tensor("model.llm.embed_norm.weight")
        head = f.get_tensor("model.llm.unembed.weight")
    with safe_open(args.trace, framework="pt") as f:
        tokens = f.get_tensor("tokens").long()
        prompt = int(f.get_tensor("prompt_len").item())
        features = f.get_tensor("features").flatten(1)
    labels = tokens[prompt:].tolist()
    other = load_file(args.candidates)["proposals"].long().tolist() if args.candidates else None
    checked, prefixes = {}, {}
    equal_rows = equal_slots = compared = slots = 0
    target_config = json.loads((args.target.parent / "source-config.json").read_text())["text_config"]
    offsets = range(len(labels) - 1) if args.all_anchors else args.offsets
    for offset in offsets:
        at = prompt + offset
        if at + 1 >= len(tokens):
            continue
        start = max(0, at - (config.sliding_window - 1))
        ids = torch.full((1, 8), config.dflash_config["mask_token_id"])
        ids[0, 0] = tokens[at]
        noise = torch.nn.functional.rms_norm(embed[ids].float(), (config.hidden_size,), norm.float(), 1e-6)
        h = model(
            position_ids=torch.arange(start, at + 8)[None],
            noise_embedding=noise.bfloat16(),
            target_hidden=features[start:at].unsqueeze(0),
        )[:, 1:]
        logits = torch.nn.functional.linear(h, head[: target_config["unpadded_vocab_size"]])
        # Inkling applies its cap after the neighboring-candidate correction.
        # Evaluate that final score in F32, preserving the greedy ordering.
        selector = model.candidate_selector
        unary, candidates = logits.float().topk(selector.top_k, sorted=False)
        projection = selector.hidden_projection(h).float()
        pred = ids[:, 0]
        proposals = []
        for i in range(7):
            edge = torch.einsum(
                "br,bkr->bk",
                selector.predecessor_codebook(pred).float() * projection[:, i],
                selector.successor_codebook(candidates[:, i]).float(),
            )
            score = 30 * torch.tanh((unary[:, i] + edge) / 30)
            pred = candidates[:, i].gather(1, score.argmax(dim=-1, keepdim=True))[:, 0]
            proposals.append(pred.item())
        if offset in args.offsets:
            checked[str(offset)] = hashlib.sha256(struct.pack("<7I", *proposals)).hexdigest()
        prefix = 0
        for predicted, truth in zip(proposals, labels[offset + 1 :], strict=False):
            if predicted != truth:
                break
            prefix += 1
        prefixes[offset] = prefix
        if other is not None:
            compared += 1
            equal_rows += int(proposals == other[offset])
            remaining = min(7, len(labels) - offset - 1)
            slots += remaining
            equal_slots += sum(proposals[i] == other[offset][i] for i in range(remaining))
    result = {"proposal_block_hashes": checked}
    if args.all_anchors:
        rounds = []
        for width in range(1, 8):
            position = count = 0
            while position < len(labels):
                position += 1 + min(width, prefixes.get(position, 0))
                count += 1
            rounds.append(count)
        result["rounds_at_widths_1_to_7"] = rounds
    if other is not None:
        result["mlx_agreement"] = {
            "blocks": equal_rows,
            "tested_blocks": compared,
            "slots": equal_slots,
            "tested_slots": slots,
        }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
