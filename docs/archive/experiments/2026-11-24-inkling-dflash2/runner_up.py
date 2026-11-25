"""Extract private conditional DFlash2 runner-up candidates for policy replay.

This uses the archived Nov 24 evaluator's fixture/model loading and repeats its
exact F32 selector score path. The sidecar contains candidate token IDs and
score gaps; keep it private. Stdout contains only aggregate parity metadata.
No target-model forward pass is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from dflash2 import (
    _bind_target_weights,
    _load_draft,
    _read_fixture,
    _read_safetensors,
    _target_settings,
)


def _path_with_runners(draft: Any, hidden: Any, anchor_id: Any,
                       selector_softcap_mode: str, valid_vocab: int):
    """Return the ordinary seven-token path plus conditional first-two runners.

    Candidate filtering, unary logits, edge correction, and softcap order are
    intentionally identical to dflash2._selector_path. At position one the
    predecessor is the anchor; at position two it is the ordinary primary at
    position one. The runner-up never conditions later positions.
    """
    import mlx.core as mx

    selector = draft.candidate_selector
    logits = draft.lm_head(hidden).astype(mx.float32)[..., :valid_vocab]
    logits = logits * float(draft.config.output_multiplier)
    top_k = selector.top_k
    if top_k < 2:
        raise ValueError("selector top_k must be at least two to extract runner-up")
    candidates = mx.argpartition(logits, -top_k, axis=-1)[..., -top_k:]
    unary = mx.take_along_axis(logits, candidates, axis=-1)
    projected = selector.hidden_projection(hidden).astype(mx.float32)
    cap = float(draft.config.final_logit_softcapping or 0.0)

    primary_ids = []
    alternate_ids = []
    gaps = []
    predecessor = anchor_id
    selector_columns = mx.arange(top_k)[None, :]
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
        primary = mx.take_along_axis(candidates[:, position], chosen[:, None], axis=-1)[:, 0]
        primary_ids.append(primary)
        if position < 2:
            chosen_score = mx.take_along_axis(scores, chosen[:, None], axis=-1)[:, 0]
            without_primary = mx.where(selector_columns == chosen[:, None], -float("inf"), scores)
            runner_index = mx.argmax(without_primary, axis=-1)
            runner = mx.take_along_axis(candidates[:, position], runner_index[:, None], axis=-1)[:, 0]
            runner_score = mx.take_along_axis(scores, runner_index[:, None], axis=-1)[:, 0]
            alternate_ids.append(runner)
            gaps.append(chosen_score - runner_score)
        predecessor = primary

    path = mx.stack(primary_ids, axis=1)
    alternates = mx.stack(alternate_ids, axis=1)
    score_gaps = mx.stack(gaps, axis=1)
    return path, alternates, score_gaps


def extract(fixture: Path, draft_dir: Path, dflash_repo: Path, target_weights: Path,
            output: Path, embedding_key: str, embedding_norm_key: str,
            head_key: str) -> dict[str, Any]:
    if output.resolve() in (fixture.resolve(), target_weights.resolve()):
        raise ValueError("output must be a distinct private sidecar path")
    if not (dflash_repo / "dflash" / "model_mlx.py").is_file():
        raise ValueError("--dflash-repo must point to a local z-lab/dflash checkout")
    import subprocess

    revision = subprocess.run(
        ["git", "-C", str(dflash_repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != "07ebd93db9f472af339b644bb70221ad8428328a":
        raise ValueError("z-lab/dflash must be at the recorded reference revision")
    sys.path.insert(0, str(dflash_repo))
    features, tokens, prompt_len, teacher_len, _ = _read_fixture(fixture)
    draft = _load_draft(draft_dir)
    target_eps, valid_vocab = _target_settings(
        target_weights, draft.config.rms_norm_eps, draft.config.vocab_size
    )
    _bind_target_weights(draft, target_weights, embedding_key, embedding_norm_key, head_key, target_eps)
    dc = json.loads((draft_dir / "config.json").read_text())["dflash_config"]
    softcap_mode = dc.get("selector_softcap_mode", "before_correction")
    if softcap_mode not in ("before_correction", "after_correction"):
        raise ValueError("unsupported selector_softcap_mode")

    import mlx.core as mx

    rows = np.empty((teacher_len, 3), dtype="<u4")
    alternatives = np.empty((teacher_len, 2), dtype="<u4")
    gaps = np.empty((teacher_len, 2), dtype="<f4")
    exact_primary_rows = 0
    for offset in range(teacher_len):
        anchor = prompt_len + offset
        context_start = max(0, anchor - (draft.config.sliding_window - 1))
        context_np = features[context_start:anchor].astype(np.float32, copy=False)
        context = mx.array(context_np[None]).astype(mx.bfloat16)
        block = mx.array([[int(tokens[anchor])] + [int(dc["mask_token_id"])] * 7], dtype=mx.int32)
        cache = draft.make_cache()
        for layer_cache in cache:
            layer_cache.offset = context_start
        hidden = draft.hidden_states(block, context, cache)[:, 1:]

        # Reuse the archived evaluator's ordinary path as the parity oracle.
        from dflash2 import _selector_path
        ordinary = _selector_path(draft, hidden, block[:, 0], softcap_mode, valid_vocab)
        path, alternate, gap = _path_with_runners(
            draft, hidden, block[:, 0], softcap_mode, valid_vocab
        )
        ordinary_ids = np.asarray(ordinary[0].tolist(), dtype=np.uint32)
        path_ids = np.asarray(path[0].tolist(), dtype=np.uint32)
        if not np.array_equal(ordinary_ids, path_ids):
            raise ValueError("runner-up selector changed the ordinary primary path")
        exact_primary_rows += int(np.count_nonzero(ordinary_ids == path_ids))
        rows[offset] = path_ids[:3]
        alternatives[offset] = np.asarray(alternate[0].tolist(), dtype=np.uint32)
        gap_values = np.asarray(gap[0].tolist(), dtype=np.float32)
        if not np.isfinite(gap_values).all() or np.any(gap_values < 0):
            raise ValueError("selector returned a non-finite or negative top1-top2 gap")
        gaps[offset] = gap_values

    historical_path = fixture.parent / "dflash-f32.safetensors"
    historical_primary_matches = None
    historical_primary_count = 0
    if historical_path.is_file():
        historical, historical_dtypes = _read_safetensors(historical_path, {"proposals"})
        if historical_dtypes["proposals"] != "U32" or historical["proposals"].shape != (teacher_len, 7):
            raise ValueError("historical candidate fixture has an unexpected schema")
        historical_primary_matches = int(np.count_nonzero(
            rows == historical["proposals"][:, :3]
        ))
        historical_primary_count = int(teacher_len * 3)
        if historical_primary_matches != historical_primary_count:
            raise ValueError("runner-up extraction disagrees with historical MLX primary IDs")

    trace_digest = hashlib.sha256()
    with fixture.open("rb") as trace_file:
        for chunk in iter(lambda: trace_file.read(4 * 1024 * 1024), b""):
            trace_digest.update(chunk)

    sidecar = {
        "schema": "inkling-dflash2-runner-up-v1",
        "case": fixture.parent.name,
        "trace_sha256": trace_digest.hexdigest(),
        "rows": int(teacher_len),
        "teacher": tokens[prompt_len:prompt_len + teacher_len].astype(np.uint32).tolist(),
        "primary": rows.tolist(),
        "alternate": alternatives.tolist(),
        "gap": gaps.tolist(),
        "semantics": {
            "primary": "first three greedy selector IDs from the ordinary seven-row path",
            "alternate": "conditional runner-up at proposal positions 1 and 2; position 2 conditions on primary position 1",
            "gap": "corrected top1 minus top2 selector score, F32, same position conditioning as alternate",
            "near_terminal": "one row per teacher anchor, including anchors with fewer than three remaining continuation tokens",
        },
        "selector_softcap_mode": softcap_mode,
        "target_valid_vocab_size": int(valid_vocab),
        "ordinary_primary_ids_equal": int(exact_primary_rows),
        "ordinary_primary_id_count": int(teacher_len * 7),
        "historical_primary_ids_equal": historical_primary_matches,
        "historical_primary_id_count": historical_primary_count,
    }
    encoded = (json.dumps(sidecar, separators=(",", ":")) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(encoded)
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return {
        "rows": sidecar["rows"],
        "ordinary_primary_ids_equal": sidecar["ordinary_primary_ids_equal"],
        "ordinary_primary_id_count": sidecar["ordinary_primary_id_count"],
        "historical_primary_ids_equal": sidecar["historical_primary_ids_equal"],
        "historical_primary_id_count": sidecar["historical_primary_id_count"],
    }


def main() -> None:
    here = Path(__file__).resolve()
    repo_root = next((parent for parent in here.parents if (parent / "Cargo.toml").is_file()), None)
    if repo_root is None:
        raise RuntimeError("could not locate the Stark repository root")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True, help="private trace SafeTensors")
    parser.add_argument("--target-weights", type=Path, required=True,
                        help="local target SafeTensors containing shared embedding and output head")
    parser.add_argument("--draft-dir", type=Path, default=repo_root / "runtime/dflash2")
    parser.add_argument("--dflash-repo", type=Path, required=True,
                        help="local checkout of z-lab/dflash at the recorded revision")
    parser.add_argument("--output", type=Path, required=True, help="private mode-0600 JSON sidecar")
    parser.add_argument("--embedding-key", default="model.llm.embed.weight")
    parser.add_argument("--embedding-norm-key", default="model.llm.embed_norm.weight")
    parser.add_argument("--head-key", default="model.llm.unembed.weight")
    args = parser.parse_args()
    result = extract(args.fixture, args.draft_dir, args.dflash_repo, args.target_weights,
                     args.output, args.embedding_key, args.embedding_norm_key, args.head_key)
    print(json.dumps({"output": str(args.output), "rows": result["rows"],
                      "ordinary_primary_ids_equal": result["ordinary_primary_ids_equal"],
                      "ordinary_primary_id_count": result["ordinary_primary_id_count"],
                      "historical_primary_ids_equal": result["historical_primary_ids_equal"],
                      "historical_primary_id_count": result["historical_primary_id_count"]}, separators=(",", ":")))


if __name__ == "__main__":
    main()
