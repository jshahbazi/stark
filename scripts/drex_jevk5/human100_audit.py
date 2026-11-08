"""Independent aggregate analysis; receives private readouts only over stdin."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

from human100 import OPTIONS, load_reference, read, require, sha


def interval(values):
    values = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(20260929)
    estimates = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def analyze(records, gold, baseline=None):
    ids = sorted(gold)
    expected = {
        (cid, arm, repeat) for cid in ids for arm in ("J0", "E1") for repeat in (1, 2)
    }
    indexed = {(row["id"], row["arm"], row["repeat"]): row for row in records}
    require(
        len(indexed) == len(records) == 400 and set(indexed) == expected,
        "analysis_coverage_invalid",
    )
    checks = 1
    for row in records:
        p = row["p"]
        require(
            len(p) == 3
            and all(
                type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in p
            ),
            "analysis_readout_invalid",
        )
        require(abs(sum(p) - 1) <= 1e-5, "analysis_mass_invalid")
        checks += 2
    contexts = {}
    for arm in ("J0", "E1"):
        rows = [indexed[cid, arm, repeat] for repeat in (1, 2) for cid in ids]
        chosen = {
            key: OPTIONS[int(np.argmax(row["p"]))]
            for key, row in indexed.items()
            if key[1] == arm
        }
        accepts = {key: value == "correct" for key, value in chosen.items()}
        matches = {key: value == gold[key[0]] for key, value in accepts.items()}
        accepted_p = [float(row["p"][0]) for row in rows]
        labels = [int(gold[row["id"]]) for row in rows]
        ece = 0.0
        for bucket in range(10):
            group = [
                i for i, p in enumerate(accepted_p) if min(9, int(p * 10)) == bucket
            ]
            if group:
                ece += (
                    len(group)
                    / 200
                    * abs(
                        statistics.mean(accepted_p[i] for i in group)
                        - statistics.mean(labels[i] for i in group)
                    )
                )
        result = {
            "judgments": 200,
            "matches": sum(matches.values()),
            "agreement": sum(matches.values()) / 200,
            "false_accepts": sum(a and not gold[key[0]] for key, a in accepts.items()),
            "false_rejects": sum(not a and gold[key[0]] for key, a in accepts.items()),
            "verdict_counts": dict(collections.Counter(chosen.values())),
            "per_repeat": {
                str(repeat): {
                    "judgments": 100,
                    "matches": sum(matches[cid, arm, repeat] for cid in ids),
                    "false_accepts": sum(
                        accepts[cid, arm, repeat] and not gold[cid] for cid in ids
                    ),
                    "false_rejects": sum(
                        not accepts[cid, arm, repeat] and gold[cid] for cid in ids
                    ),
                }
                for repeat in (1, 2)
            },
            "repeat_binary_flips": sum(
                accepts[cid, arm, 1] != accepts[cid, arm, 2] for cid in ids
            ),
            "repeat_verdict_flips": sum(
                chosen[cid, arm, 1] != chosen[cid, arm, 2] for cid in ids
            ),
            "repeat_max_probability_delta": max(
                abs(a - b)
                for cid in ids
                for a, b in zip(
                    indexed[cid, arm, 1]["p"], indexed[cid, arm, 2]["p"], strict=True
                )
            ),
            "binary_brier": statistics.mean(
                (p - label) ** 2 for p, label in zip(accepted_p, labels, strict=True)
            ),
            "binary_log_loss": statistics.mean(
                -math.log(max(1e-15, p if label else 1 - p))
                for p, label in zip(accepted_p, labels, strict=True)
            ),
            "ece_10_bins": ece,
            "agreement_case_cluster_95_interval": interval(
                [
                    (int(matches[cid, arm, 1]) + int(matches[cid, arm, 2])) / 2
                    for cid in ids
                ]
            ),
            "median_seconds": statistics.median(row["seconds"] for row in rows),
            "p95_seconds": sorted(row["seconds"] for row in rows)[189],
        }
        if baseline is not None:
            base_accept = {
                (r["id"], r["arm"], r["repeat"]): r["choice"] == "correct"
                for r in baseline
            }
            require(set(base_accept) == expected, "baseline_coverage_invalid")
            base_matches = {
                key: value == gold[key[0]] for key, value in base_accept.items()
            }
            differences = [
                sum(
                    int(matches[cid, arm, p]) - int(base_matches[cid, arm, p])
                    for p in (1, 2)
                )
                / 2
                for cid in ids
            ]
            only_drex = sum(
                matches[cid, arm, 1] and not base_matches[cid, arm, 1] for cid in ids
            )
            only_baseline = sum(
                not matches[cid, arm, 1] and base_matches[cid, arm, 1] for cid in ids
            )
            discordant = only_drex + only_baseline
            result["versus_frozen_jevk5_mlx"] = {
                "baseline_matches": sum(
                    base_matches[cid, arm, p] for cid in ids for p in (1, 2)
                ),
                "agreement_difference": statistics.mean(differences),
                "paired_case_cluster_95_interval": interval(differences),
                "first_repeat_drex_only_correct": only_drex,
                "first_repeat_baseline_only_correct": only_baseline,
                "first_repeat_mcnemar_exact_two_sided_p": min(
                    1.0,
                    2
                    * sum(
                        math.comb(discordant, i)
                        for i in range(min(only_drex, only_baseline) + 1)
                    )
                    / 2**discordant,
                ),
            }
        contexts[arm] = result
        require(
            result["matches"] + result["false_accepts"] + result["false_rejects"]
            == 200,
            "analysis_confusion_mismatch",
        )
        require(
            sum(result["verdict_counts"].values()) == 200,
            "analysis_verdict_count_mismatch",
        )
        require(
            all(
                r["matches"] + r["false_accepts"] + r["false_rejects"] == 100
                for r in result["per_repeat"].values()
            ),
            "analysis_repeat_count_mismatch",
        )
        require(0 <= result["agreement"] <= 1, "analysis_agreement_invalid")
        checks += 4
    return {
        "schema": 1,
        "status": "passed",
        "checks": checks,
        "errors": 0,
        "contexts": contexts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = read(args.config)
        _, gold, _ = load_reference(config)
        baseline = Path(config["baseline_mlx"])
        require(
            sha(baseline)
            == "874aa7a59d98afbeef71f317f43f657f5ff09741ed14b675639cc1510d23c0f7",
            "baseline_hash_changed",
        )
        baseline_rows = [json.loads(line) for line in baseline.read_text().splitlines()]
        records = json.load(sys.stdin)
        result = analyze(records, gold, baseline_rows)
        result["baseline_verdicts_sha256"] = sha(baseline)
        print(json.dumps(result, allow_nan=False))
    except Exception:
        print('{"status":"failed"}', file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
