#!/usr/bin/env python3
"""Create immutable baseline/staged/direct source archives for the GDN trial.

This script performs no builds, model reads, tests, or inference. It refuses to
overwrite source snapshots or preparation metadata. Run it from its research copy;
the source archive itself always comes from the fixed Git commit below.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[4]
BASE_COMMIT = "645ad5562f44926341b3a74001ed4243e40db3e9"
TRIAL_ROOT = ROOT / "runtime/gdn-prework-gpu-20261031"
SOURCES = TRIAL_ROOT / "sources"
PREPARED_METAL = ROOT / "docs/archive/experiments/qwen38-gdn-prework-2026-10-31/prepared.metal"
METAL_REL = "vendor/lily-core/src/kernels/metal/qwen38_gdn_prework_research.metal"
GDN_REL = "vendor/lily-core/src/kernels/gdn.rs"
MODEL_REL = "vendor/lily-core/src/qwen4exp/model.rs"
RECURRENCE_KERNELS = {
    "staged": "qwen38_gdn_recurrence_staged_research",
    "direct": "qwen38_gdn_recurrence_research",
}

ARCHIVE_FORBIDDEN_SUFFIXES = {
    ".safetensors", ".gguf", ".ggml", ".pt", ".pth", ".ckpt", ".onnx", ".tflite", ".bin", ".model",
}
ARCHIVE_FORBIDDEN_PARTS = {"models", "model_weights", "checkpoints", "weights"}


def fail(category: str) -> None:
    raise SystemExit(category)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checked_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        fail("unsafe_archive_path")
    if ".git" in path.parts:
        fail("git_metadata_in_archive")
    if path.suffix.lower() in ARCHIVE_FORBIDDEN_SUFFIXES:
        fail("model_payload_in_archive")
    if any(part.lower() in ARCHIVE_FORBIDDEN_PARTS for part in path.parts):
        fail("model_directory_in_archive")
    return path


def extract_git_archive(destination: Path) -> None:
    process = subprocess.Popen(
        ["git", "archive", "--format=tar", BASE_COMMIT],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        for member in archive:
            relative = checked_relative(member.name)
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.issym():
                link = PurePosixPath(member.linkname)
                resolved = PurePosixPath(os.path.normpath(str(relative.parent / link)))
                if link.is_absolute() or ".." in resolved.parts:
                    fail("unsafe_archive_symlink")
                target.symlink_to(member.linkname)
                continue
            if not member.isfile():
                fail("unsupported_archive_entry")
            if member.size > 64 * 1024 * 1024:
                fail("oversized_tracked_artifact")
            source = archive.extractfile(member)
            if source is None:
                fail("archive_entry_unreadable")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(member.mode & 0o777)
    return_code = process.wait()
    stderr = process.stderr.read() if process.stderr is not None else b""
    if return_code != 0:
        fail("git_archive_failed" if not stderr else "git_archive_failed_with_error")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        fail(f"source_anchor_{label}_{count}")
    return source.replace(old, new, 1)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def rust_wrapper(recurrent_kernel: str) -> str:
    return f'''

// Research-only official ValueKey decode prework. The entrypoint is fixed in
// each archived source copy; there is no runtime or environment selector.
const QWEN38_GDN_PREWORK_SOURCE: &str = include_str!("metal/qwen38_gdn_prework_research.metal");
const QWEN38_GDN_RESEARCH_RECURRENCE: &str = "{recurrent_kernel}";

#[allow(clippy::too_many_arguments)]
pub fn qwen38_gdn_prework_research(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    window: &Tensor,
    qkv: &Tensor,
    conv_w: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    prepared: &Tensor,
    gate_pairs: &Tensor,
    state: &Tensor,
    z: &Tensor,
    norm_w: &Tensor,
    raw: &Tensor,
    out: &Tensor,
    scale: f32,
    eps: f32,
) -> Result<()> {{
    const C: usize = 10240;
    const H: usize = 48;
    const D: usize = 128;
    ensure!(window.numel() == C * 3 && window.dtype() == DType::BF16, "research GDN window geometry");
    ensure!(qkv.numel() == C && qkv.dtype() == DType::BF16, "research GDN QKV geometry");
    ensure!(conv_w.numel() == C * 4 && conv_w.dtype() == DType::BF16, "research GDN conv weights");
    ensure!(a.numel() == H && a.dtype() == DType::BF16, "research GDN a geometry");
    ensure!(b.numel() == H && b.dtype() == DType::BF16, "research GDN b geometry");
    ensure!(a_log.numel() == H && a_log.dtype() == DType::F32, "research GDN A_log geometry");
    ensure!(dt_bias.numel() == H && dt_bias.dtype() == DType::BF16, "research GDN dt bias geometry");
    ensure!(prepared.numel() == C && prepared.dtype() == DType::BF16, "research GDN prepared scratch");
    ensure!(gate_pairs.numel() == H * 2 && gate_pairs.dtype() == DType::F32, "research GDN gate scratch");
    ensure!(state.shape() == [H, D, D] && state.dtype() == DType::F32, "research GDN state geometry");
    ensure!(z.numel() == H * D && z.dtype() == DType::BF16, "research GDN z geometry");
    ensure!(norm_w.numel() == D && norm_w.dtype() == DType::F32, "research GDN norm weight geometry");
    ensure!(raw.numel() == H * D && raw.dtype() == DType::BF16, "research GDN raw geometry");
    ensure!(out.numel() == H * D && out.dtype() == DType::BF16, "research GDN output geometry");
    ensure!(scale.to_bits() == (1.0f32 / (D as f32).sqrt()).to_bits(), "research GDN K scale");
    ensure!(eps.is_finite() && eps > 0.0, "research GDN norm epsilon");

    pass.profile_stage(crate::metal::stage_profile::Phase::GdnConv)?;
    let prepare = ctx.pipeline_with_math_mode(
        "qwen38_gdn_conv_prepare_research",
        QWEN38_GDN_PREWORK_SOURCE,
        MslVersion::V3_1,
        crate::metal::MslMathMode::Safe,
    )?;
    pass.dispatch_at(
        &prepare,
        &[
            window.binding(), qkv.binding(), conv_w.binding(), prepared.binding(),
            a.binding(), b.binding(), a_log.binding(), dt_bias.binding(), gate_pairs.binding(),
        ],
        &[&scale.to_ne_bytes()],
        Grid::Threadgroups {{ groups: (80, 1, 1), threadgroup: (32, 1, 1) }},
    )?;
    pass.level_barrier(&[window, prepared, gate_pairs])?;

    pass.profile_stage(crate::metal::stage_profile::Phase::GdnRecurrence)?;
    let recurrence = ctx.pipeline_with_math_mode(
        QWEN38_GDN_RESEARCH_RECURRENCE,
        QWEN38_GDN_PREWORK_SOURCE,
        MslVersion::V3_1,
        crate::metal::MslMathMode::Safe,
    )?;
    pass.dispatch_at(
        &recurrence,
        &[prepared.binding(), gate_pairs.binding(), state.binding(), raw.binding()],
        &[],
        Grid::Threadgroups {{ groups: (192, 1, 1), threadgroup: (256, 1, 1) }},
    )?;
    pass.level_barrier(&[raw, state])?;
    pass.profile_stage(crate::metal::stage_profile::Phase::GdnNorm)?;
    gated_rmsnorm_value_key_decode(ctx, pass, raw, z, norm_w, out, eps)
}}
'''


def patch_gdn(source_root: Path, variant: str) -> None:
    gdn = source_root / GDN_REL
    source = read_text(gdn)
    source = replace_once(
        source,
        'const SOURCE: &str = include_str!("metal/gdn.metal");',
        'const SOURCE: &str = include_str!("metal/gdn.metal");\n',
        "gdn_source_const",
    )
    source += rust_wrapper(RECURRENCE_KERNELS[variant])
    write_text(gdn, source)


def patch_model(source_root: Path) -> None:
    model = source_root / MODEL_REL
    source = read_text(model)
    source = replace_once(
        source,
        "    gdn_step_gated_fused_with_layout as gdn_step_gated_fused,",
        "    gdn_step_gated_fused_with_layout as gdn_step_gated_fused,\n"
        "    qwen38_gdn_prework_research,",
        "model_gdn_import",
    )
    source = replace_once(
        source,
        "    gdn_raw: Tensor,",
        "    gdn_raw: Tensor,\n"
        "    /// Experimental F32 ValueKey gates; prepared Q/K/V reuses qkv_conv.\n"
        "    gdn_research_gates: Tensor,",
        "scratch_gdn_fields",
    )
    source = replace_once(
        source,
        "            gdn_raw: Tensor::zeros(ctx, &[dim_v], bf)?,",
        "            gdn_raw: Tensor::zeros(ctx, &[dim_v], bf)?,\n"
        "            gdn_research_gates: Tensor::zeros(ctx, &[96], DType::F32)?,",
        "scratch_gdn_allocations",
    )
    anchor = """        if crate::ablate::on(crate::ablate::Group::GdnStep) {
            return Ok(());
        }
"""
    addition = anchor + """        // The archived official ValueKey route uses a fixed research kernel
        // variant. The baseline copy retains this original path unchanged.
        if self.config.official_mlx_rounding()
            && gdn_layout == GdnStateLayout::ValueKey
            && !crate::ablate::on(crate::ablate::Group::GdnProj)
        {
            qwen38_gdn_prework_research(
                ctx,
                pass,
                conv_window,
                &s.qkv,
                &w.conv_w,
                &s.a,
                &s.b,
                &w.a_log,
                &w.dt_bias,
                &s.qkv_conv,
                &s.gdn_research_gates,
                gdn_state,
                &s.z,
                &w.norm_w,
                &s.gdn_raw,
                &s.gdn_gated,
                self.gdn_scale,
                self.config.rms_norm_eps,
            )?;
            pass.level_barrier(&[&s.gdn_gated, gdn_state])?;
            pass.profile_stage(crate::metal::stage_profile::Phase::GdnOutProjection)?;
            return crate::kernels::official_q8_projection::gemv_gdn_decode(
                ctx,
                pass,
                &w.out_proj,
                &s.gdn_gated,
                &s.branch_out,
                self.config.official_mlx_rounding(),
            );
        }
"""
    source = replace_once(source, anchor, addition, "model_gdn_route")
    write_text(model, source)


def unified_diff(old: str, new: str, old_name: str, new_name: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=old_name,
            tofile=new_name,
        )
    )


def apply_variant(baseline: Path, destination: Path, variant: str, metal_bytes: bytes) -> list[str]:
    shutil.copytree(baseline, destination, symlinks=True)
    gdn_path, model_path = destination / GDN_REL, destination / MODEL_REL
    baseline_gdn, baseline_model = read_text(baseline / GDN_REL), read_text(baseline / MODEL_REL)
    patch_gdn(destination, variant)
    patch_model(destination)
    metal_path = destination / METAL_REL
    metal_path.parent.mkdir(parents=True, exist_ok=True)
    metal_path.write_bytes(metal_bytes)
    changes = [
        unified_diff(baseline_gdn, read_text(gdn_path), GDN_REL, GDN_REL),
        unified_diff(baseline_model, read_text(model_path), MODEL_REL, MODEL_REL),
        unified_diff("", metal_bytes.decode("utf-8"), "/dev/null", METAL_REL),
    ]
    patch_path = destination / "research.patch"
    patch_path.write_text("".join(changes), encoding="utf-8", newline="\n")
    return [GDN_REL, MODEL_REL, METAL_REL, "research.patch"]


def source_file_map(source_root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(source_root.rglob("*")):
        relative = path.relative_to(source_root).as_posix()
        if ".git" in PurePosixPath(relative).parts:
            fail("git_metadata_created")
        if path.is_symlink():
            files[relative] = sha256_bytes(os.readlink(path).encode("utf-8"))
        elif path.is_file():
            files[relative] = sha256_file(path)
    return files


def tree_digest(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, value in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def save_json(path: Path, value: object) -> bytes:
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return raw


def main() -> None:
    if not ROOT.is_dir():
        fail("repository_root_missing")
    if SOURCES.exists() or (TRIAL_ROOT / "preparation.json").exists() or (TRIAL_ROOT / "manifests").exists():
        fail("trial_output_exists")
    try:
        resolved = subprocess.run(
            ["git", "rev-parse", "--verify", f"{BASE_COMMIT}^{{commit}}"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        fail("base_commit_unavailable")
    if not resolved.startswith(BASE_COMMIT[:8]):
        fail("base_commit_identity")
    if not PREPARED_METAL.is_file():
        fail("prepared_metal_missing")
    metal_bytes = PREPARED_METAL.read_bytes()
    metal_text = metal_bytes.decode("utf-8")
    for name in (
        "qwen38_gdn_conv_prepare_research",
        "qwen38_gdn_recurrence_research",
        "qwen38_gdn_recurrence_staged_research",
    ):
        if f"kernel void {name}(" not in metal_text:
            fail("prepared_metal_entry_missing")
    prototype_sha = sha256_bytes(metal_bytes)

    os.umask(0o077)
    TRIAL_ROOT.mkdir(parents=True, exist_ok=True)
    SOURCES.mkdir()
    baseline = SOURCES / "baseline"
    baseline.mkdir()
    extract_git_archive(baseline)
    source_manifests: dict[str, dict[str, object]] = {}

    for variant in ("baseline", "staged", "direct"):
        root = SOURCES / variant
        changed: list[str]
        if variant == "baseline":
            changed = []
        else:
            changed = apply_variant(baseline, root, variant, metal_bytes)
        file_map = source_file_map(root)
        manifest = {
            "schema": "qwen38-gdn-prework-source-manifest-v1",
            "variant": variant,
            "base_commit": resolved,
            "prototype_sha256": prototype_sha if variant != "baseline" else None,
            "recurrence_kernel": RECURRENCE_KERNELS.get(variant),
            "file_count": len(file_map),
            "source_tree_manifest_sha256": tree_digest(file_map),
            "files": file_map,
            "changed_files": changed,
        }
        manifest_path = TRIAL_ROOT / "manifests" / f"{variant}.source-manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_bytes = save_json(manifest_path, manifest)
        source_manifests[variant] = {
            "source_root": f"sources/{variant}",
            "manifest_path": f"manifests/{variant}.source-manifest.json",
            "source_manifest_sha256": sha256_bytes(manifest_bytes),
            "source_tree_manifest_sha256": manifest["source_tree_manifest_sha256"],
            "file_count": len(file_map),
            "files": file_map,
            "changed_files": changed,
            "recurrence_kernel": RECURRENCE_KERNELS.get(variant),
        }

    preparation = {
        "schema": "qwen38-gdn-prework-model-trial-preparation-v1",
        "base_commit": resolved,
        "prototype_sha256": prototype_sha,
        "build_directory_template": "builds/{variant}/release/stark-generate",
        "build_receipt_contract": {
            "schema": "qwen38-gdn-prework-build-receipt-v1",
            "path_template": "builds/{variant}/build-receipt.json",
            "required_fields": [
                "worker_sha256", "preparation_sha256", "source_manifest_sha256",
                "source_tree_manifest_sha256", "locked_toolchain", "command", "completed",
            ],
            "locked_toolchain": "1.98.1",
            "preparation_sha256_scope": "SHA-256 of the completed preparation.json bytes",
        },
        "policy": {
            "baseline_source_unmodified": True,
            "candidate_routes": "fixed official ValueKey source guards; no environment or runtime selector",
            "models_archived": False,
            "git_metadata_archived": False,
            "build_or_inference_performed": False,
        },
        "source_manifests": source_manifests,
    }
    save_json(TRIAL_ROOT / "preparation.json", preparation)
    print(json.dumps({
        "completed": True,
        "trial_root": str(TRIAL_ROOT),
        "base_commit": resolved,
        "prototype_sha256": prototype_sha,
        "source_tree_manifest_sha256": {
            name: item["source_tree_manifest_sha256"] for name, item in source_manifests.items()
        },
        "file_count": {name: item["file_count"] for name, item in source_manifests.items()},
        "build_commands": {
            name: f"cd {SOURCES / name} && CARGO_TARGET_DIR={TRIAL_ROOT / 'builds' / name} cargo +1.98.1 build --locked --release -p stark-generate"
            for name in source_manifests
        },
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, UnicodeError, tarfile.TarError):
        fail("preparation_failed")
