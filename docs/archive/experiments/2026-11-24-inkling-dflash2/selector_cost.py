#!/usr/bin/env python3
"""Price ordinary DFlash2 selection against conditional runner-up extraction.

The target-derived feature fixture and each draft hidden block are read/built
once. Only selector execution and readback are timed. Output is aggregate-only.
Run behind the repository's shared GPU lock; this script does not take a lock.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]

from dflash2 import (  # noqa: E402
    _bind_target_weights,
    _load_draft,
    _read_fixture,
    _target_settings,
    _selector_path,
)
from runner_up import _path_with_runners  # noqa: E402

CASES = ("prose01", "coding01", "prose07", "coding07")
DEFAULT_OFFSETS = (16, 64, 128)


def _read_eos(target_weights: Path) -> int:
    config_path = target_weights.parent / "generation_config.json"
    value = json.loads(config_path.read_text())
    eos_id = value.get("eos_token_id")
    if type(eos_id) is not int or eos_id < 0:
        raise ValueError("invalid_target_eos")
    return eos_id


def _method_a(draft: Any, hidden: Any, anchor_id: Any, mode: str, valid_vocab: int):
    import mlx.core as mx

    start = time.perf_counter_ns()
    path = _selector_path(draft, hidden, anchor_id, mode, valid_vocab)
    mx.eval(path)
    ids = path[0].tolist()
    return time.perf_counter_ns() - start, ids


def _method_b(draft: Any, hidden: Any, anchor_id: Any, mode: str, valid_vocab: int):
    import mlx.core as mx

    start = time.perf_counter_ns()
    path, alternate, gaps = _path_with_runners(draft, hidden, anchor_id, mode, valid_vocab)
    mx.eval(path, alternate, gaps)
    ids = path[0].tolist()
    alternate.tolist()
    gaps.tolist()
    return time.perf_counter_ns() - start, ids


def _schedule(pair_count: int) -> list[str]:
    # Adjacent pairs alternate order; concatenated blocks are ABBA, BAAB, ABBA…
    order: list[str] = []
    for index in range(pair_count):
        if index % 4 in (0, 3):
            order.extend(("A", "B"))
        else:
            order.extend(("B", "A"))
    return order


def _hidden_for_anchor(mx: Any, draft: Any, features: Any, tokens: Any,
                       anchor: int, mask_id: int):
    context_start = max(0, anchor - (draft.config.sliding_window - 1))
    context_np = features[context_start:anchor].astype("float32", copy=False)
    context = mx.array(context_np[None]).astype(mx.bfloat16)
    block = mx.array(
        [[int(tokens[anchor])] + [int(mask_id)] * 7],
        dtype=mx.int32,
    )
    cache = draft.make_cache()
    for layer_cache in cache:
        layer_cache.offset = context_start
    hidden = draft.hidden_states(block, context, cache)[:, 1:]
    mx.eval(hidden)
    return hidden, block[:, 0]


def _benchmark_anchor(mx: Any, draft: Any, hidden: Any, anchor_id: Any,
                      mode: str, valid_vocab: int, pair_count: int, reverse: bool) -> dict[str, Any]:
    # Warm each implementation once and force all requested results to host.
    _, reference = _method_a(draft, hidden, anchor_id, mode, valid_vocab)
    _, candidate = _method_b(draft, hidden, anchor_id, mode, valid_vocab)
    if reference != candidate or len(reference) != 7:
        raise ValueError("primary_path_parity_failed")

    times_a: list[int] = []
    times_b: list[int] = []
    paired_delta: list[int] = []
    schedule = _schedule(pair_count)
    for pair_index in range(pair_count):
        order = schedule[2 * pair_index:2 * pair_index + 2]
        if reverse:
            order.reverse()
        pair: dict[str, int] = {}
        for method in order:
            if method == "A":
                elapsed, ids = _method_a(draft, hidden, anchor_id, mode, valid_vocab)
                times_a.append(elapsed)
            else:
                elapsed, ids = _method_b(draft, hidden, anchor_id, mode, valid_vocab)
                times_b.append(elapsed)
            if ids != reference:
                raise ValueError("primary_path_parity_failed")
            pair[method] = elapsed
        paired_delta.append(pair["B"] - pair["A"])

    return {
        "pairs": pair_count,
        "ordinary_primary_median_ms": statistics.median(times_a) / 1_000_000,
        "runner_up_primary_median_ms": statistics.median(times_b) / 1_000_000,
        "runner_up_extra_median_ms": statistics.median(paired_delta) / 1_000_000,
        "primary_equal": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, default=ROOT / "runtime/dflash2/self-correction-20261125/cases")
    parser.add_argument("--target-weights", type=Path, required=True)
    parser.add_argument("--draft-dir", type=Path, default=ROOT / "runtime/dflash2")
    parser.add_argument("--dflash-repo", type=Path, default=ROOT / "runtime/dflash2/zlab")
    parser.add_argument("--offsets", type=int, nargs="+", default=list(DEFAULT_OFFSETS))
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--reverse", action="store_true", help="reverse cases, offsets and paired arm order")
    args = parser.parse_args()

    try:
        if args.pairs < 6:
            raise ValueError("minimum_pair_count")
        if not args.offsets or any(offset < 0 for offset in args.offsets):
            raise ValueError("invalid_offsets")
        revision = subprocess.run(
            ["git", "-C", str(args.dflash_repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != "07ebd93db9f472af339b644bb70221ad8428328a":
            raise ValueError("dflash_reference_revision_mismatch")

        sys.path.insert(0, str(args.dflash_repo.resolve()))
        draft = _load_draft(args.draft_dir)
        target_eps, valid_vocab = _target_settings(
            args.target_weights, draft.config.rms_norm_eps, draft.config.vocab_size
        )
        _bind_target_weights(
            draft,
            args.target_weights,
            "model.llm.embed.weight",
            "model.llm.embed_norm.weight",
            "model.llm.unembed.weight",
            target_eps,
        )
        draft_config = json.loads((args.draft_dir / "config.json").read_text())["dflash_config"]
        mode = draft_config.get("selector_softcap_mode", "before_correction")
        if mode not in ("before_correction", "after_correction"):
            raise ValueError("unsupported_selector_softcap_mode")

        eos_id = _read_eos(args.target_weights)
        import mlx.core as mx

        anchors: list[dict[str, Any]] = []
        for case in (reversed(CASES) if args.reverse else CASES):
            fixture = args.case_root / case / "teacher.safetensors"
            features, tokens, prompt_len, teacher_len, _ = _read_fixture(fixture)
            if teacher_len <= max(args.offsets):
                raise ValueError("offset_outside_teacher")
            teacher = tokens[prompt_len:prompt_len + teacher_len]
            eos_positions = [i for i, token in enumerate(teacher.tolist()) if token == eos_id]
            first_eos = min(eos_positions) if eos_positions else teacher_len
            if first_eos <= max(args.offsets):
                raise ValueError("offset_at_or_after_eos")

            for offset in (reversed(args.offsets) if args.reverse else args.offsets):
                anchor = prompt_len + offset
                hidden, anchor_id = _hidden_for_anchor(
                    mx, draft, features, tokens, anchor, draft_config["mask_token_id"]
                )
                result = _benchmark_anchor(
                    mx, draft, hidden, anchor_id, mode, valid_vocab, args.pairs, args.reverse
                )
                anchors.append({"case": case, "offset": offset, **result})
                del hidden, anchor_id

        grouped: dict[str, list[dict[str, Any]]] = {"prose": [], "coding": []}
        for row in anchors:
            grouped["prose" if row["case"].startswith("prose") else "coding"].append(row)
        output = {
            "schema": "inkling-selector-cost-v1",
            "methods": {
                "ordinary": "_selector_path; seven primary argmax outputs read back",
                "runner_up": "_path_with_runners; same seven primary outputs plus two runner-up and gap outputs read back",
            },
            "timing": {
                "clock": "perf_counter_ns wall time",
                "warmups_per_method_per_anchor": 1,
                "interleaving": "ABBA/BAAB alternating pairs",
                "pairs_per_anchor": args.pairs,
                "reverse": args.reverse,
                "hidden_built_once_per_anchor": True,
            },
            "anchors": anchors,
            "groups": {
                domain: {
                    "anchors": len(rows),
                    "ordinary_primary_median_ms": statistics.median(
                        row["ordinary_primary_median_ms"] for row in rows
                    ),
                    "runner_up_primary_median_ms": statistics.median(
                        row["runner_up_primary_median_ms"] for row in rows
                    ),
                    "runner_up_extra_median_ms": statistics.median(
                        row["runner_up_extra_median_ms"] for row in rows
                    ),
                    "all_primary_equal": all(row["primary_equal"] for row in rows),
                }
                for domain, rows in grouped.items()
            },
        }
        print(json.dumps(output, separators=(",", ":")))
        return 0
    except Exception:
        # Fail closed without exposing fixture data, IDs, logits or score gaps.
        print('{"error":"selector_microbench_failed"}', file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
