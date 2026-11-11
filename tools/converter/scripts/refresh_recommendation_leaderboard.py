#!/usr/bin/env python3
"""Build Stark's compact whatmodel.sh leaderboard snapshot.

Default operation is offline and consumes saved API responses plus a pinned
whatmodel research checkout. Network access requires the explicit --fetch flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

RAM_BUCKETS = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512)
TASKS = ("general", "coding")
BASE_URL = "https://whatmodel.sh/?gpu=apple&max=0&ram={ram}&format=json&n=1000"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def safe_fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Stark recommendation snapshot maintainer"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise ValueError(f"HTTP status {response.status}")
            return response.read(16 * 1024 * 1024 + 1)
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        # Do not expose response bodies, request headers, or exception dumps.
        raise ValueError(f"fetch failed for {url}: {type(error).__name__}") from None


def source_models(path: Path) -> dict[str, str]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError("source model catalogue must be a JSON list")
    result = {}
    pattern = re.compile(r"https://huggingface\.co/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/?$")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            continue
        match = pattern.fullmatch(row.get("model_repo", ""))
        if match:
            result[row["id"]] = "/".join(match.groups())
    return result


def git_output(source_dir: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_dir), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def sync_notices(source_dir: Path, notices_dir: Path, source_revision: str) -> None:
    paths = git_output(source_dir, "ls-files", "--", "DATA-LICENSE.md", "LICENSE", "licenses").splitlines()
    required = {"DATA-LICENSE.md", "LICENSE"}
    if not required.issubset(paths) or not any(path.startswith("licenses/") for path in paths):
        raise ValueError("source checkout has no tracked full data license and notice bundle")
    dirty = git_output(
        source_dir,
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        "data/models.json",
        "DATA-LICENSE.md",
        "LICENSE",
        "licenses",
    )
    if dirty:
        raise ValueError("source model catalogue or license notices have tracked local changes")

    expected = set(paths)
    for existing in notices_dir.rglob("*") if notices_dir.exists() else ():
        if existing.is_file() and existing.relative_to(notices_dir).as_posix() not in expected | {"README.md"}:
            existing.unlink()
    for relative in paths:
        source = source_dir / relative
        if not source.is_file():
            raise ValueError(f"tracked source notice is missing: {relative}")
        destination = notices_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    notice_text = "\n".join(
        [
            "# whatmodel.sh snapshot data notices",
            "",
            f"Source checkout revision: `{source_revision}`.",
            (
                "Source model catalogue: `data/models.json` "
                f"(SHA-256 `{sha256(source_dir / 'data/models.json')}`)."
            ),
            "",
            "## License scope",
            "",
            (
                "The numerical compilation, adaptations, and derived scores originate from whatmodel.sh "
                "and are distributed under CC BY-SA 4.0, as described in `DATA-LICENSE.md`. Underlying "
                "source grants remain applicable to their included material; the full source license texts "
                "and notices are copied unchanged from the pinned checkout under `licenses/`. The repository "
                "`LICENSE` covers software code, not the numerical data compilation. Model weights, trademarks, "
                "and application code are not included or relicensed by this snapshot."
            ),
            "",
            "## Snapshot transformation",
            "",
            (
                "The refresh script consumes saved whatmodel.sh JSON responses (or downloads them only with "
                "explicit `--fetch`). For each task and RAM bucket it retains source-order rows whose `where` "
                "value is `local` or `local-slow`, including rows with null scores. It copies only model ID, "
                "display name, score, uncertainty, estimated memory, quantization, and the source speed object; "
                "commands, code, prompts, provider prose, and model-card text are omitted. Exact Hugging Face "
                "repository mappings come only from `model_repo` URLs in the pinned `data/models.json`; IDs "
                "are never matched by name. Raw response hashes, query specs, score metadata, and source "
                "attributions are preserved in the generated snapshot."
            ),
            "",
        ]
    )
    (notices_dir / "README.md").write_text(notice_text, encoding="utf-8")


def compact_model(row: dict) -> dict:
    # Deliberately omit commands, prompts, provider annotations and model-card
    # prose; retain only ranking data and source-provided speed context.
    return {
        "id": row["id"],
        "name": row["name"],
        "score": row.get("score"),
        "uncertainty": row.get("pm"),
        "memory_gb": row.get("mem_gb"),
        "quantization": row.get("quant"),
        "speed": row.get("speed"),
    }


def build(args) -> dict:
    raw_dir, source_dir = args.raw_dir, args.source_dir
    expected_commit = args.source_revision
    actual_commit = git_output(source_dir, "rev-parse", "HEAD")
    if expected_commit and actual_commit != expected_commit:
        raise ValueError(f"source checkout revision mismatch: expected {expected_commit}, got {actual_commit}")
    expected_commit = expected_commit or actual_commit
    model_path = source_dir / "data" / "models.json"
    manifest_path = raw_dir / "manifest.json"
    license_path = source_dir / "DATA-LICENSE.md"
    source_license_path = source_dir / "LICENSE"
    if not model_path.is_file() or not license_path.is_file() or not source_license_path.is_file():
        raise ValueError("source checkout is missing required model or license files")
    dirty_sources = git_output(
        source_dir,
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        "data/models.json",
        "DATA-LICENSE.md",
        "LICENSE",
        "licenses",
    )
    if dirty_sources:
        raise ValueError("source model catalogue or license notices have tracked local changes")
    expected_inputs = {}
    if not args.fetch:
        if not manifest_path.is_file():
            raise ValueError(f"missing offline input manifest: {manifest_path}")
        manifest = load_json(manifest_path)
        if not isinstance(manifest, list):
            raise ValueError("offline input manifest must be a JSON list")
        for entry in manifest:
            if not isinstance(entry, dict) or type(entry.get("ram")) is not int or entry.get("task") not in TASKS:
                raise ValueError("invalid offline input manifest entry")
            key = f"{entry['task']}-{entry['ram']}gb.json"
            if key in expected_inputs or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
                raise ValueError("duplicate or invalid offline input manifest hash")
            expected_inputs[key] = entry["sha256"]

    buckets = []
    input_hashes = {}
    attributions = None
    data_license = None
    updated_values = set()
    refreshed_manifest = []
    for task in TASKS:
        for ram in RAM_BUCKETS:
            filename = f"{task}-{ram}gb.json"
            path = raw_dir / filename
            url = BASE_URL.format(ram=ram) + ("&task=coding" if task == "coding" else "")
            if args.fetch:
                payload = safe_fetch(url)
                if len(payload) > 16 * 1024 * 1024:
                    raise ValueError(f"response too large for {filename}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            if not path.is_file():
                raise ValueError(f"missing offline input: {path}; use --fetch to download it")
            input_hashes[filename] = sha256(path)
            if expected_inputs and expected_inputs.get(filename) != input_hashes[filename]:
                raise ValueError(f"offline input hash does not match manifest: {filename}")
            document = load_json(path)
            if not isinstance(document, dict) or not isinstance(document.get("models"), list):
                raise ValueError(f"invalid leaderboard document: {filename}")
            if (
                document.get("updated") is None
                or document.get("score_mode") is None
                or document.get("score_set") is None
            ):
                raise ValueError(f"leaderboard metadata missing in {filename}")
            specs = document.get("specs")
            expected_task = "coding" if task == "coding" else None
            if (
                not isinstance(specs, dict)
                or specs.get("ram") != ram
                or specs.get("gpu") != "apple"
                or specs.get("vram") != 0
                or document.get("max_price") != 0
                or document.get("task") != expected_task
            ):
                raise ValueError(f"query specs do not match requested {task} {ram} GB bucket: {filename}")
            updated_values.add(document["updated"])
            if attributions is None:
                attributions = document.get("attributions")
                data_license = document.get("data_license")
                if not isinstance(attributions, list) or not isinstance(data_license, dict):
                    raise ValueError("leaderboard attribution or data license missing")
            elif document.get("attributions") != attributions or document.get("data_license") != data_license:
                raise ValueError(f"attribution or license changed between snapshots: {filename}")
            models = []
            for row in document["models"]:
                if not isinstance(row, dict) or row.get("where") not in {"local", "local-slow"}:
                    continue
                if not isinstance(row.get("id"), str) or not isinstance(row.get("name"), str):
                    raise ValueError(f"malformed model identity in {filename}")
                models.append(compact_model(row))
            buckets.append(
                {
                    "task": task,
                    "ram_gb": ram,
                    "url": url,
                    "sha256": input_hashes[filename],
                    "updated": document["updated"],
                    "score_mode": document["score_mode"],
                    "score_set": document["score_set"],
                    "models": models,
                }
            )
            scored_top = [row for row in document["models"] if isinstance(row, dict) and row.get("score") is not None]
            scored_top.sort(key=lambda row: row["score"], reverse=True)
            refreshed_manifest.append(
                {
                    "ram": ram,
                    "task": task,
                    "url": url,
                    "path": str(path),
                    "sha256": input_hashes[filename],
                    "updated": document["updated"],
                    "count": len(document["models"]),
                    "top": [[row.get("id"), row.get("score"), row.get("mem_gb")] for row in scored_top[:5]],
                }
            )
    if len(updated_values) != 1:
        raise ValueError("the 22 leaderboard queries do not share one update timestamp")
    if expected_inputs and set(expected_inputs) != set(input_hashes):
        raise ValueError("offline input manifest must contain exactly the 22 expected task/RAM files")
    model_repositories = source_models(model_path)
    sync_notices(source_dir, args.notices_dir, expected_commit)
    if args.fetch:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(refreshed_manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "captured": args.captured_date or datetime.now(UTC).date().isoformat(),
        "source_revision": expected_commit,
        "source_url": "https://whatmodel.sh/",
        "data_license": data_license,
        "attributions": attributions,
        "source_model_catalog_sha256": sha256(model_path),
        "model_repositories": model_repositories,
        "input_sha256": input_hashes,
        "buckets": buckets,
    }


def _report_row(bucket: dict, rank: int, model: dict) -> str:
    columns = (
        f"{bucket['ram_gb']} GB",
        str(rank),
        model["name"],
        str(model["score"]),
        str(model["uncertainty"]),
        f"{model['memory_gb']} GB",
        str(model["quantization"]),
    )
    return "| " + " | ".join(columns) + " |"


def write_report(snapshot: dict, path: Path):
    rows = [
        "# whatmodel.sh recommendations snapshot",
        "",
        f"Captured {snapshot['captured']} from whatmodel.sh source revision `{snapshot['source_revision']}`.",
        "The tables show source ranking scores, not Stark measurements or compatibility claims.",
        "",
        "## General",
        "",
        "| Mac RAM | Rank | Model | Score | Uncertainty | Estimated memory | Quantization |",
        "|---:|---:|---|---:|---:|---:|---|",
    ]
    for bucket in snapshot["buckets"]:
        if bucket["task"] != "general":
            continue
        scored = [m for m in bucket["models"] if isinstance(m["score"], (int, float))]
        for rank, model in enumerate(sorted(scored, key=lambda m: m["score"], reverse=True)[:3], 1):
            rows.append(_report_row(bucket, rank, model))
    rows += [
        "",
        "## Coding",
        "",
        "| Mac RAM | Rank | Model | Score | Uncertainty | Estimated memory | Quantization |",
        "|---:|---:|---|---:|---:|---:|---|",
    ]
    for bucket in snapshot["buckets"]:
        if bucket["task"] != "coding":
            continue
        scored = [m for m in bucket["models"] if isinstance(m["score"], (int, float))]
        for rank, model in enumerate(sorted(scored, key=lambda m: m["score"], reverse=True)[:3], 1):
            rows.append(_report_row(bucket, rank, model))
    rows += [
        "",
        "## Scope and limitations",
        "",
        (
            "- Source scores and estimated memory come from dated whatmodel.sh API snapshots. They may be "
            "incomplete or outdated and are not Stark artifact measurements."
        ),
        (
            "- Source memory estimates allow an 8K token conversation; Stark beginner recommendations plan "
            "around a 32K context. Memory estimates do not establish Stark fit."
        ),
        (
            "- Speed entries retain source hardware and `how` fields; speeds measured elsewhere are not "
            "Stark performance claims."
        ),
        (
            "- Model repository identities come only from exact Hugging Face URLs in the pinned source "
            "catalogue. Name similarity is not identity evidence."
        ),
        (
            "- This snapshot adds recommendation evidence only. It adds no conversion, backend, or native "
            "model compatibility."
        ),
        (
            f"- Source model catalogue SHA-256: `{snapshot['source_model_catalog_sha256']}`. "
            "Raw API response hashes are in the snapshot JSON."
        ),
        (
            "- whatmodel.sh numerical compilation and derived scores are CC BY-SA 4.0. Source-specific "
            "grants and full notices are preserved under "
            "`tools/converter/stark_convert/_third_party/whatmodel/`."
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("/tmp/stark-whatmodel-snapshots"))
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/stark-whatmodel-research"))
    parser.add_argument("--source-revision", default="2769f5ba536d5d78fc70a772048512471a499802")
    parser.add_argument(
        "--notices-dir", type=Path, default=Path("tools/converter/stark_convert/_third_party/whatmodel")
    )
    parser.add_argument("--captured-date")
    parser.add_argument(
        "--output", type=Path, default=Path("tools/converter/stark_convert/recommendation_leaderboard.json")
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--fetch", action="store_true", help="download API snapshots; otherwise require saved offline inputs"
    )
    args = parser.parse_args()
    if args.report is None:
        report_date = args.captured_date or datetime.now(UTC).date().isoformat()
        args.report = Path(f"docs/archive/runs/{report_date}-whatmodel-recommendations.md")
    try:
        snapshot = build(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        write_report(snapshot, args.report)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        print(f"snapshot refresh failed: {error}", file=sys.stderr)
        return 2
    print(
        f"wrote {args.output} and {args.report}; {len(snapshot['buckets'])} buckets, "
        f"{len(snapshot['model_repositories'])} exact repository identities"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
