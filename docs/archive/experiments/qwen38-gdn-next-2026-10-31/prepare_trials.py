#!/usr/bin/env python3
"""Archive frozen sources and apply isolated, fixed research routes; no builds/GPU."""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUNTIME = ROOT / "runtime/gdn-next-gpu-20261031"
BASES = {
    "single": "645ad5562f44926341b3a74001ed4243e40db3e9",
    "mtp": "959ec981354acab313a9a8d709933fb2f2649b17",
}


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


SHARED_PATH = HERE.parent / "qwen38-gdn-fusion-2026-10-31/prepare_model_trial.py"
shared = module(SHARED_PATH, "fusion_preparation")


def patch_1024(destination: Path) -> None:
    shader_rel = Path(shared.METAL_REL)
    (destination / shader_rel).write_bytes((HERE / "fusion1024.metal").read_bytes())
    gdn = destination / shared.GDN_REL
    shared.write_text(gdn, shared.read_text(gdn) + shared.rust_wrapper("qwen38_gdn_fused1024_research", 1024))
    shared.patch_model(destination)


def make_patch(baseline: Path, candidate: Path, changes: list[str]) -> None:
    output = []
    for name in changes:
        old = baseline / name
        new = candidate / name
        before = old.read_text() if old.is_file() else ""
        after = new.read_text() if new.is_file() else ""
        output.extend(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=name if old.is_file() else "/dev/null", tofile=name,
        ))
    shared.write_text(candidate / "research.patch", "".join(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(BASES))
    parser.add_argument("--arms", nargs="+", required=True)
    args = parser.parse_args()
    allowed = {"single": {"q8conv", "fused1024"}, "mtp": {"q8reuse"}}[args.mode]
    arms = tuple(args.arms)
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= allowed:
        shared.fail("invalid_variant_set")
    trial = RUNTIME / args.mode
    if (trial / "sources").exists() or (trial / "preparation.json").exists():
        shared.fail("preparation_already_exists")
    shared.BASE_COMMIT = BASES[args.mode]
    shared.ROOT = ROOT
    resolved = shared.subprocess.run(
        ["git", "rev-parse", "--verify", shared.BASE_COMMIT + "^{commit}"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if resolved != shared.BASE_COMMIT:
        shared.fail("base_commit_identity")
    os.umask(0o077)
    sources = trial / "sources"
    sources.mkdir(parents=True)
    baseline = sources / "baseline"
    baseline.mkdir()
    shared.extract_git_archive(baseline)
    base_files = shared.source_file_map(baseline)
    manifests = {}
    tooling = {"prepare_trials.py": shared.sha256_file(Path(__file__)),
               "shared_preparation": shared.sha256_file(SHARED_PATH)}
    for arm in ("baseline", *arms):
        source = sources / arm
        changes = []
        prototype_sha = None
        if arm != "baseline":
            shutil.copytree(baseline, source, symlinks=True)
            if arm == "fused1024":
                patch_1024(source)
                prototype_sha = shared.sha256_file(HERE / "fusion1024.metal")
            else:
                helper_name = {"q8conv": "q8_conv_integration", "q8reuse": "q8_rows_integration"}[arm]
                helper = HERE / (helper_name + ".py")
                tooling[helper.name] = shared.sha256_file(helper)
                integration = module(helper, helper_name)
                integration.patch_source(source)
                shader_name = "q8_conv.metal" if arm == "q8conv" else "q8_rows.metal"
                prototype_sha = shared.sha256_file(HERE / shader_name)
            files = shared.source_file_map(source)
            changes = sorted(name for name in base_files.keys() | files.keys()
                             if base_files.get(name) != files.get(name))
            make_patch(baseline, source, changes)
            changes.append("research.patch")
        files = shared.source_file_map(source)
        manifest = {
            "schema": f"qwen38-gdn-next-{args.mode}-source-manifest-v1",
            "variant": arm, "base_commit": resolved, "prototype_sha256": prototype_sha,
            "files": files, "file_count": len(files), "changed_files": changes,
            "source_tree_manifest_sha256": shared.tree_digest(files),
        }
        name = f"manifests/{arm}.source-manifest.json"
        path = trial / name
        path.parent.mkdir(exist_ok=True)
        raw = shared.save_json(path, manifest)
        manifests[arm] = {
            "source_root": f"sources/{arm}", "manifest_path": name,
            "source_manifest_sha256": shared.sha256_bytes(raw),
            "source_tree_manifest_sha256": manifest["source_tree_manifest_sha256"],
            "file_count": len(files), "files": files, "changed_files": changes,
        }
    preparation = {
        "schema": f"qwen38-gdn-next-{args.mode}-preparation-v1",
        "base_commit": resolved, "source_manifests": manifests,
        "tooling_sha256": tooling, "mtp_drafts": 2 if args.mode == "mtp" else 0,
        "build_receipt_schema": f"qwen38-gdn-next-{args.mode}-build-receipt-v1",
        "policy": {"models_archived": False, "production_dispatch_modified": False,
                   "isolated_fixed_routes_only": True, "builds_or_gpu_performed": False},
    }
    shared.save_json(trial / "preparation.json", preparation)
    print(json.dumps({"prepared": True, "mode": args.mode, "arms": list(manifests),
                      "base_commit": resolved, "preparation_sha256": shared.sha256_file(trial / "preparation.json")}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - emit fixed categories without exception dumps.
        shared.fail("source_preparation_failed")
