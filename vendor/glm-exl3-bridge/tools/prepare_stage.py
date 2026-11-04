#!/usr/bin/env python3
"""Build a local pinned bridge stage explicitly. No network or model operations."""
from __future__ import annotations
import argparse
import hashlib
import json
import io
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def require(condition: bool, category: str) -> None:
    if not condition:
        raise SystemExit(category)

def command(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    # Never echo subprocess exception dumps, config paths, or arbitrary output.
    require(result.returncode == 0, "source/build command failed")
    return result.stdout.strip()

def archive(source: Path, revision: str, destination: Path) -> None:
    packed = subprocess.run(["git", "-C", str(source), "archive", "--format=tar", revision], capture_output=True)
    require(packed.returncode == 0, "pinned source archive failed")
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(packed.stdout), mode="r:") as bundle:
        bundle.extractall(destination, filter="data")

def mlxc_diff(source: Path) -> bytes:
    result = subprocess.run(["git", "-C", str(source), "--no-pager", "diff", "--no-ext-diff", "--no-textconv",
        "--binary", "--src-prefix=a/", "--dst-prefix=b/", "HEAD", "--"], capture_output=True)
    require(result.returncode == 0, "MLX-C source check failed")
    return result.stdout

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", type=Path, required=True)
    p.add_argument("--plugin", type=Path, required=True)
    p.add_argument("--mlx-stage", type=Path, required=True)
    p.add_argument("--zig", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--experimental-prefix", action="store_true")
    args = p.parse_args()
    host, plugin, stage, zig, out = (x.resolve() for x in
        (args.host, args.plugin, args.mlx_stage, args.zig, args.output))
    lock = json.loads((ROOT / "lock.json").read_text())
    for entry in lock["owned_sources"]:
        require(sha(ROOT / entry["path"]) == entry["sha256"], "bridge source hash mismatch")
    for directory, revision in ((host, lock["host_revision"]), (plugin, lock["plugin_revision"]),
            (host / "lib/mlx-src", lock["mlx_revision"]), (host / "lib/mlxc-src", lock["mlxc_revision"]),
            (host / "lib/sushi", lock["sushi_revision"])):
        require(command("git", "-C", str(directory), "rev-parse", "HEAD") == revision,
                "upstream source revision mismatch")
    # Compile immutable archives, never copy a worktree with staged/untracked edits.
    mlxc = host / "lib/mlxc-src"
    require(command("git", "-C", str(mlxc), "diff", "HEAD", "--name-only") == "mlx/c/ops.cpp",
            "MLX-C unexpected modified source")
    require(hashlib.sha256(mlxc_diff(mlxc)).hexdigest() == lock["mlxc_worktree_diff_sha256"],
            "MLX-C actual source patch mismatch")
    require(command("git", "-C", str(mlxc), "ls-files", "--others", "--exclude-standard", "mlx/c") == "",
            "MLX-C has untracked source imports")
    require(command(str(zig), "version") == lock["zig_version"] and sha(zig) == lock["zig_sha256"],
            "compiler pin mismatch")
    require(sha(host / "patches/mlxc-gather-qmm-global-scale.patch") == lock["mlxc_patch_sha256"],
            "MLX-C compatibility patch mismatch")
    dependency_paths = {"mlx": "libmlx.dylib", "mlxc": "libmlxc.dylib", "jaccl": "libjaccl.dylib", "metallib": "mlx.metallib"}
    for name, filename in dependency_paths.items():
        require(sha(stage / "lib" / filename) == lock["dependency_artifacts"][name], "MLX artifact hash mismatch")
    actual_headers = sorted(
        ["mlx-stage/" + x.relative_to(stage).as_posix() for x in (stage / "include").rglob("*")
            if x.is_file() and x.suffix in (".h", ".hpp", ".hxx", ".metal")]
        + ["mlxc/" + x.relative_to(mlxc).as_posix() for x in (mlxc / "mlx/c/private").rglob("*")
            if x.is_file() and x.suffix in (".h", ".hpp")])
    require(actual_headers == [x["path"] for x in lock["dependency_headers"]], "MLX header inventory mismatch")
    for entry in lock["dependency_headers"]:
        prefix, relative = entry["path"].split("/", 1)
        base = stage if prefix == "mlx-stage" else mlxc
        require(sha(base / relative) == entry["sha256"], "MLX header content mismatch")
    require(not out.exists(), "output already exists; choose a new ignored stage")
    # No writes may land in tracked source or any shared upstream checkout.
    repo = Path(command("git", "-C", str(ROOT), "rev-parse", "--show-toplevel"))
    check = subprocess.run(["git", "-C", str(repo), "check-ignore", "--quiet", str(out)], capture_output=True)
    require(check.returncode == 0, "output must be ignored by the repository")
    out.mkdir(parents=True)
    copied = out / "source/mlx-stream"
    archive(plugin, lock["plugin_revision"], copied)
    sushi = out / "source/sushi"
    archive(host / "lib/sushi", lock["sushi_revision"], sushi)
    patch = ROOT / "patches/mlx-stream-prefix.patch"
    require(sha(patch) == lock["plugin_patch_sha256"], "plugin patch hash mismatch")
    command("patch", "-p1", "--batch", "--forward", "-i", str(patch), cwd=copied)
    command(str(zig), "build", "test", "-Doptimize=ReleaseFast",
        "--cache-dir", str(out / "build-cache"),
        f"-Dhost-dir={host}", f"-Dplugin-dir={copied}", f"-Dmlx-stage={stage}",
        f"-Dsushi-dir={sushi}",
        f"-Dexperimental-prefix={'true' if args.experimental_prefix else 'false'}", cwd=ROOT)
    command(str(zig), "build", "-Doptimize=ReleaseFast", "--prefix", str(out),
        "--cache-dir", str(out / "build-cache"),
        f"-Dhost-dir={host}", f"-Dplugin-dir={copied}", f"-Dmlx-stage={stage}",
        f"-Dsushi-dir={sushi}",
        f"-Dexperimental-prefix={'true' if args.experimental_prefix else 'false'}", cwd=ROOT)
    artifacts = {"stark_glm": {"path": "lib/libstark_glm.a", "sha256": sha(out / "lib/libstark_glm.a")}}
    for name, filename in dependency_paths.items():
        dep = stage / "lib" / filename
        artifacts[name] = {"path": str(dep), "sha256": sha(dep)}
    receipt = {"version": 1, "abi_version": 1,
        **{key: lock[key] for key in ("bridge_source_sha256", "host_revision", "plugin_revision", "mlx_revision", "mlxc_revision", "mlxc_patch_sha256", "plugin_patch_sha256", "sushi_revision", "zig_version", "zig_sha256")},
        "experimental_prefix": args.experimental_prefix, "mlx_lib_dir": str(stage / "lib"),
        "mlxc_worktree_diff_sha256": lock["mlxc_worktree_diff_sha256"],
        "dependency_headers_sha256": lock["dependency_headers_sha256"],
        "artifacts": artifacts, "validation": {"c_abi_smoke": "passed", "model_loaded": False,
            "gpu_inference": False, "prefix_restore_qualified": False}}
    (out / "stark-glm-stage.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("bridge_stage_built=1 c_abi_smoke=passed model_loaded=0 gpu_inference=0")

if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError):
        raise SystemExit("source/build preparation failed") from None
