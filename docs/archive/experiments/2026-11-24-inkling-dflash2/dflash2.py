"""Offline teacher-forced evaluator for the Inkling DFlash2 draft head.

The draft forward and dynamic convolutions come from z-lab/dflash
``dflash/model_mlx.py`` (revision 07ebd93db9f472af339b644bb70221ad8428328a).
The round protocol follows ddalcu/mlx-serve's DFlash implementation (revision
2201ec9a838246b7a361d96a20cd3fb39b749ac8): target context ends before the
anchor, the input is ``[anchor, mask x 7]``, the anchor output row is dropped,
and selector choices chain greedily. The evaluator does not run the target.

Expected private SafeTensors fixture keys:
``features`` BF16/F32 [sequence, 4, 2304] (flattened last two dims also
accepted), ``tokens`` U32 [sequence], ``prompt_len`` U32 [1], and
``teacher_len`` U32 [1]. Features are post-consumption rows: for anchor at
position p only features [0:p] are supplied. For every fixed width 1..7,
teacher forcing advances to the next anchor by accepted_prefix + one target
token; each width walks its own anchors.

Requires local MLX, NumPy, safetensors, and a local checkout of z-lab/dflash.
It is intentionally an evaluator, not a capture or benchmark runner. It emits
only aggregate counters and input digests; never token IDs, logits, or text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

_DTYPES = {
    "BF16": np.dtype("<u2"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "U32": np.dtype("<u4"),
    "I32": np.dtype("<i4"),
}


def _read_safetensors(path: Path, names: set[str]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Read only selected tensors, including BF16 without a NumPy BF16 plugin."""
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError("invalid SafeTensors header")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = json.loads(f.read(header_len))
        missing = names.difference(header)
        if missing:
            raise ValueError(f"SafeTensors missing required tensors: {sorted(missing)}")
        data_start = 8 + header_len
        tensors: dict[str, np.ndarray] = {}
        dtypes: dict[str, str] = {}
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as view:
            for name in names:
                info = header[name]
                dtype = info["dtype"]
                if dtype not in _DTYPES:
                    raise ValueError(f"unsupported SafeTensors dtype for {name}: {dtype}")
                start, end = info["data_offsets"]
                shape = tuple(info["shape"])
                values = np.frombuffer(view, dtype=_DTYPES[dtype], count=(end - start) // _DTYPES[dtype].itemsize,
                                       offset=data_start + start).reshape(shape).copy()
                tensors[name] = values
                dtypes[name] = dtype
        return tensors, dtypes


def _to_float32(values: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return (values.astype(np.uint32) << 16).view(np.float32)
    return values.astype(np.float32, copy=False)


def _read_fixture(path: Path) -> tuple[np.ndarray, np.ndarray, int, int, str]:
    names = {"features", "tokens", "prompt_len", "teacher_len"}
    tensors, dtypes = _read_safetensors(path, names)
    features = _to_float32(tensors["features"], dtypes["features"])
    tokens = tensors["tokens"]
    prompt_len = int(tensors["prompt_len"].reshape(-1)[0])
    teacher_len = int(tensors["teacher_len"].reshape(-1)[0])

    if features.ndim == 3 and features.shape[1:] == (4, 2304):
        features = features.reshape(features.shape[0], 4 * 2304)
    if features.ndim != 2 or features.shape[1] != 4 * 2304:
        raise ValueError("features must have shape [sequence, 4, 2304] or [sequence, 9216]")
    if dtypes["features"] not in ("BF16", "F32"):
        raise ValueError("features must be BF16 or F32")
    if tokens.ndim != 1 or dtypes["tokens"] != "U32":
        raise ValueError("tokens must be a one-dimensional U32 tensor")
    if dtypes["prompt_len"] != "U32" or dtypes["teacher_len"] != "U32":
        raise ValueError("prompt_len and teacher_len must be U32")
    tokens = tokens.astype(np.uint32, copy=False)
    sequence_end = prompt_len + teacher_len
    if prompt_len < 1 or teacher_len < 1 or sequence_end > len(tokens) or sequence_end > len(features):
        raise ValueError("prompt_len/teacher_len exceed available trace rows")
    return features[:sequence_end], tokens[:sequence_end], prompt_len, teacher_len, dtypes["features"]


def _load_draft(draft_dir: Path):
    import mlx.core as mx
    from dflash.model_mlx import DFlash2DraftModel, DFlashConfig

    config_path = draft_dir / "config.json"
    weights_path = draft_dir / "draft.safetensors"
    cfg = json.loads(config_path.read_text())
    dc = cfg["dflash_config"]
    rope = cfg.get("rope_parameters") or cfg.get("rope_scaling")
    layer_types = tuple(cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"])
    if len(layer_types) != cfg["num_hidden_layers"]:
        raise ValueError("draft layer_types length differs from num_hidden_layers")
    config = DFlashConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=(rope or {}).get("rope_theta", cfg.get("rope_theta", 10000.0)),
        max_position_embeddings=cfg["max_position_embeddings"],
        block_size=dc["block_size"],
        target_layer_ids=tuple(dc["target_layer_ids"]),
        num_target_layers=cfg["num_target_layers"],
        mask_token_id=dc["mask_token_id"],
        rope_scaling=rope,
        layer_types=layer_types,
        sliding_window=cfg.get("sliding_window"),
        final_logit_softcapping=dc.get("final_logit_softcapping"),
        conv_kernel_size=dc["conv_kernel_size"],
        conv_group_size=dc["conv_group_size"],
        selector_rank=dc["selector_rank"],
        selector_top_k=dc["selector_top_k"],
        is_causal=cfg.get("is_causal"),
    )
    if config.hidden_size != 2304 or len(config.target_layer_ids) != 4 or config.block_size != 8:
        raise ValueError("draft geometry does not match the Inkling DFlash2 trace contract")

    weights = dict(mx.load(str(weights_path)))
    for name in ("predecessor_codebook", "successor_codebook"):
        key = f"candidate_selector.{name}"
        weights[f"{key}.weight"] = weights.pop(key)
    draft = DFlash2DraftModel(config)
    draft.eval()
    draft.load_weights(list(weights.items()))
    # Keep upstream MLX math, but borrow the Inkling target embedding/head.
    return draft


def _target_settings(weights_path: Path, default_eps: float, physical_vocab: int) -> tuple[float, int]:
    """Read source config first so padded vocabulary rows never enter top-k."""
    for name in ("source-config.json", "config.json"):
        path = weights_path.parent / name
        if not path.is_file():
            continue
        cfg = json.loads(path.read_text())
        text = cfg.get("text_config", cfg)
        eps = float(text.get("rms_norm_eps", default_eps))
        valid_vocab = int(text.get("unpadded_vocab_size", text.get("vocab_size", physical_vocab)))
        if not 0 < valid_vocab <= physical_vocab:
            raise ValueError("target config declares an invalid unpadded vocabulary size")
        return eps, valid_vocab
    return default_eps, physical_vocab


def _bind_target_weights(draft: Any, weights_path: Path, embedding_key: str,
                         embedding_norm_key: str, head_key: str,
                         target_eps: float) -> None:
    import mlx.core as mx
    from mlx import nn
    names = {embedding_key, embedding_norm_key, head_key}
    tensors, dtypes = _read_safetensors(weights_path, names)
    embedding = tensors[embedding_key]
    embedding_norm = tensors[embedding_norm_key]
    head = tensors[head_key]
    if embedding.shape != (201024, 2304) or head.shape != (201024, 2304) or embedding_norm.shape != (2304,):
        raise ValueError("target embedding/unembedding must be [201024, 2304], embedding norm [2304]")

    class InklingDraftEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.tokens = nn.Embedding(201024, 2304)
            self.tokens.weight = _mx_array(mx, embedding, dtypes[embedding_key])
            self.norm = nn.RMSNorm(2304, eps=target_eps)
            self.norm.weight = _mx_array(mx, embedding_norm, dtypes[embedding_norm_key])

        def __call__(self, ids):
            return self.norm(self.tokens(ids))

    draft.embed_tokens = InklingDraftEmbedding()
    draft.lm_head = nn.Linear(2304, 201024, bias=False)
    draft.lm_head.weight = _mx_array(mx, head, dtypes[head_key])
    draft.embed_scale = 1.0


def _mx_array(mx: Any, values: np.ndarray, dtype: str):
    array = mx.array(values)
    return array.view(mx.bfloat16) if dtype == "BF16" else array


def _selector_path(draft: Any, hidden: Any, anchor_id: Any,
                   selector_softcap_mode: str, valid_vocab: int):
    """Apply the trained candidate lattice with the Inkling softcap policy."""
    import mlx.core as mx

    selector = draft.candidate_selector
    logits = draft.lm_head(hidden).astype(mx.float32)[..., :valid_vocab]
    logits = logits * float(draft.config.output_multiplier)
    top_k = selector.top_k
    candidates = mx.argpartition(logits, -top_k, axis=-1)[..., -top_k:]
    unary = mx.take_along_axis(logits, candidates, axis=-1)
    projected = selector.hidden_projection(hidden).astype(mx.float32)
    predecessor = anchor_id
    path = []
    cap = float(draft.config.final_logit_softcapping or 0.0)
    for position in range(projected.shape[1]):
        pred = selector.predecessor_codebook(predecessor).astype(mx.float32)[:, None, :]
        succ = selector.successor_codebook(candidates[:, position]).astype(mx.float32)
        edge = mx.sum(pred * projected[:, position, None, :] * succ, axis=-1)
        if selector_softcap_mode == "before_correction" and cap > 0:
            scores = cap * mx.tanh(unary[:, position] / cap) + edge
        else:
            scores = unary[:, position] + edge
            if selector_softcap_mode == "after_correction" and cap > 0:
                scores = cap * mx.tanh(scores / cap)
        chosen = mx.argmax(scores, axis=-1)
        predecessor = mx.take_along_axis(candidates[:, position], chosen[:, None], axis=-1)[:, 0]
        path.append(predecessor)
    return mx.stack(path, axis=1)


def _draft_block(draft: Any, features: np.ndarray, tokens: np.ndarray, anchor: int,
                 mask_id: int, selector_softcap_mode: str, valid_vocab: int):
    import mlx.core as mx

    context_start = max(0, anchor - (draft.config.sliding_window - 1))
    context_np = features[context_start:anchor].astype(np.float32, copy=False)
    context = mx.array(context_np[None]).astype(mx.bfloat16)
    block = mx.array([[int(tokens[anchor])] + [int(mask_id)] * 7], dtype=mx.int32)
    cache = draft.make_cache()
    # The context slice begins at an absolute target position; preserve it for RoPE.
    for layer_cache in cache:
        layer_cache.offset = context_start
    hidden = draft.hidden_states(block, context, cache)
    proposal_hidden = hidden[:, 1:]
    path = _selector_path(draft, proposal_hidden, block[:, 0], selector_softcap_mode, valid_vocab)

    # Independent parity control against the pinned MLX selector implementation.
    # Its unsoftcapped greedy path must equal the trained post-correction path;
    # for before-correction checkpoints, soften unary logits before invoking it.
    raw_logits = draft.lm_head(proposal_hidden)[..., :valid_vocab] * float(draft.config.output_multiplier)
    cap = float(draft.config.final_logit_softcapping or 0.0)
    if selector_softcap_mode == "before_correction" and cap > 0:
        raw_logits = cap * mx.tanh(raw_logits / cap)
    reference_path, _, _ = draft.candidate_selector.select(
        proposal_hidden, raw_logits, block[:, 0], temperature=0.0
    )
    return path, reference_path


def evaluate(fixture: Path, draft_dir: Path, dflash_repo: Path, target_weights: Path,
             output: Path, embedding_key: str, embedding_norm_key: str,
             head_key: str, candidate_fixture: Path | None) -> dict[str, Any]:

    features, tokens, prompt_len, teacher_len, feature_dtype_in = _read_fixture(fixture)
    if not (dflash_repo / "dflash" / "model_mlx.py").is_file():
        raise ValueError("--dflash-repo must point to a local z-lab/dflash checkout")
    revision = subprocess.run(
        ["git", "-C", str(dflash_repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != "07ebd93db9f472af339b644bb70221ad8428328a":
        raise ValueError("z-lab/dflash must be at the recorded reference revision")
    sys.path.insert(0, str(dflash_repo))
    draft = _load_draft(draft_dir)
    target_eps, valid_vocab = _target_settings(
        target_weights, draft.config.rms_norm_eps, draft.config.vocab_size
    )
    _bind_target_weights(draft, target_weights, embedding_key, embedding_norm_key, head_key, target_eps)
    dc = json.loads((draft_dir / "config.json").read_text())["dflash_config"]
    softcap_mode = dc.get("selector_softcap_mode", "before_correction")
    if softcap_mode not in ("before_correction", "after_correction"):
        raise ValueError(f"unsupported selector_softcap_mode: {softcap_mode}")

    sequence_end = prompt_len + teacher_len
    width_results = []
    blocks: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def get_block(anchor: int) -> tuple[np.ndarray, np.ndarray]:
        if anchor not in blocks:
            path, reference = _draft_block(
                draft, features, tokens, anchor, dc["mask_token_id"], softcap_mode, valid_vocab
            )
            blocks[anchor] = (
                np.asarray(path[0].tolist(), dtype=np.uint32),
                np.asarray(reference[0].tolist(), dtype=np.uint32),
            )
        return blocks[anchor]

    determinism_checked = False
    determinism_equal = None
    selector_reference_checked = False
    selector_reference_matches = None
    proposal_block_hashes: dict[str, str] = {}

    # These fixed anchors are shared with the independent PyTorch check. Hash
    # all seven proposals in little-endian U32 form; keep the token values out
    # of both stdout and the JSON receipt.
    for offset in (0, 31, 95, 159):
        anchor = prompt_len + offset
        if teacher_len - offset - 1 < 7:
            continue
        proposal = get_block(anchor)[0].astype("<u4", copy=False)
        if proposal.shape != (7,):
            raise ValueError("DFlash selector did not return seven proposal IDs")
        proposal_block_hashes[str(offset)] = hashlib.sha256(proposal.tobytes()).hexdigest()

    for width in range(1, 8):
        anchor = prompt_len
        rounds = proposed = matches = accepted = 0
        per_position = [0] * width
        position_opportunities = [0] * width
        while anchor < sequence_end:
            rounds += 1
            available = sequence_end - anchor - 1
            if available <= 0:
                break
            active_width = min(width, available)
            proposal, reference_proposal = get_block(anchor)
            truth = tokens[anchor + 1:anchor + 1 + active_width]
            same = proposal[:active_width] == truth
            for j, value in enumerate(same):
                position_opportunities[j] += 1
                per_position[j] += int(value)
            proposed += active_width
            matches += int(np.count_nonzero(same))
            prefix = 0
            for value in same:
                if not value:
                    break
                prefix += 1
            accepted += prefix
            if not determinism_checked:
                repeated, _ = _draft_block(
                    draft, features, tokens, anchor, dc["mask_token_id"], softcap_mode, valid_vocab
                )
                determinism_equal = bool(
                    np.array_equal(proposal, np.asarray(repeated[0].tolist(), dtype=np.uint32))
                )
                determinism_checked = True
            if not selector_reference_checked:
                selector_reference_matches = int(np.count_nonzero(proposal == reference_proposal))
                selector_reference_checked = True

            anchor += prefix + 1

        if rounds + accepted != teacher_len:
            raise ValueError("teacher-forced rounds did not consume the complete continuation")
        width_results.append({
            "width": width,
            "rounds": rounds,
            "proposals": proposed,
            "tokenwise_matches": matches,
            "accepted_prefix_tokens": accepted,
            "position_opportunities": position_opportunities,
            "position_matches": per_position,
        })

    candidate_written = False
    if candidate_fixture is not None:
        proposal_rows = np.zeros((teacher_len, 7), dtype="<u4")
        for offset in range(teacher_len):
            proposal_rows[offset] = get_block(prompt_len + offset)[0]
        from safetensors.numpy import save

        encoded = save({"proposals": np.ascontiguousarray(proposal_rows)})
        candidate_fixture.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(candidate_fixture, flags, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(encoded)
        os.chmod(candidate_fixture, 0o600)
        candidate_written = True

    token_digest = hashlib.sha256(tokens.astype("<u4", copy=False).tobytes()).hexdigest()
    feature_digest = hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()
    result = {
        "schema": "inkling-dflash2-teacher-replay-v1",
        "draft_device_execution": "mlx",
        "draft_model": "DFlash2",
        "reference_sources": {
            "z_lab_dflash": "07ebd93db9f472af339b644bb70221ad8428328a",
            "mlx_serve_protocol": "2201ec9a838246b7a361d96a20cd3fb39b749ac8",
        },
        "sequence_rows": int(sequence_end),
        "prompt_rows": int(prompt_len),
        "teacher_rows": int(teacher_len),
        "feature_shape": [int(sequence_end), 4, 2304],
        "feature_dtype_in": feature_dtype_in,
        "feature_dtype_draft": "bfloat16",
        "draft_precision_policy": "BF16 draft weights and activations; BF16 target features and embedded block inputs",
        "selector_precision_policy": (
            "F32 unary logits, projected hidden and codebooks before edge products, "
            "F32 edge reduction and score softcap"
        ),
        "selector_reference_precision": "upstream MLX selector using checkpoint BF16 hidden projection and codebooks",
        "target_feature_ids": list(draft.config.target_layer_ids),
        "target_feature_convention": "partial residual inputs at visits 5,23,41; final norm output at id 48",
        "draft_block_size": 8,
        "draft_rows_per_round": 7,
        "context_policy": "all preceding rows within sliding_window-1; exclude current anchor row",
        "rope_offset_policy": "absolute offset of retained context slice",
        "selector_softcap_mode": softcap_mode,
        "target_embedding_norm_epsilon": target_eps,
        "target_valid_vocab_size": valid_vocab,
        "input_token_ids_sha256": token_digest,
        "canonical_feature_f32_sha256": feature_digest,
        "parity_controls": {
            "repeat_first_anchor_path_exact": determinism_equal,
            "upstream_bf16_selector_matching_positions": selector_reference_matches,
            "upstream_bf16_selector_positions": 7 if selector_reference_checked else 0,
        },
        "proposal_block_hashes": proposal_block_hashes,
        "candidate_fixture_written": candidate_written,
        "widths": width_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(output, flags, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    os.chmod(output, 0o600)
    return result


def main() -> None:
    here = Path(__file__).resolve()
    repo_root = next((parent for parent in here.parents if (parent / "Cargo.toml").is_file()), None)
    if repo_root is None:
        raise RuntimeError("could not locate the Stark repository root")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True, help="private trace SafeTensors")
    parser.add_argument("--target-weights", type=Path, required=True,
                        help="local target SafeTensors containing the shared input embedding and output head")
    parser.add_argument("--draft-dir", type=Path, default=repo_root / "runtime/dflash2")
    parser.add_argument("--dflash-repo", type=Path, required=True,
                        help="local checkout of z-lab/dflash with dflash/model_mlx.py")
    parser.add_argument("--output", type=Path, required=True, help="private aggregate JSON output")
    parser.add_argument("--candidate-fixture", type=Path,
                        help="optional mode-0600 private SafeTensors file with proposals[teacher_len,7]")
    parser.add_argument("--embedding-key", default="model.llm.embed.weight")
    parser.add_argument("--embedding-norm-key", default="model.llm.embed_norm.weight")
    parser.add_argument("--head-key", default="model.llm.unembed.weight")
    args = parser.parse_args()
    result = evaluate(args.fixture, args.draft_dir, args.dflash_repo,
                      args.target_weights, args.output, args.embedding_key,
                      args.embedding_norm_key, args.head_key, args.candidate_fixture)
    print(json.dumps({"output": str(args.output), "widths": result["widths"]}, separators=(",", ":")))


if __name__ == "__main__":
    main()
