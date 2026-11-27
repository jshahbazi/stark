// SPDX-License-Identifier: Apache-2.0
use std::{path::Path, process::Command};

use anyhow::{Context, ensure};

#[path = "../native/hy4/embed_msl.rs"]
mod embed;

pub(super) fn build(out_dir: &Path) -> anyhow::Result<()> {
    let native = Path::new("native/hy4");
    let vendor = native.join("vendor/strata/engine");
    println!("cargo::rerun-if-changed={}", native.display());
    embed::generate(native, out_dir)?;

    let headers = [
        "metal/src",
        "metal/src/engine",
        "metal/src/hy4flash_gguf",
        "metal/src/hy4flash_gguf/msl",
        "include",
        "third_party/ggml",
    ];
    let sources = [
        "src/hy4flash_gguf/geometry.cpp",
        "src/hy4flash_gguf/engram.cpp",
        "src/hy4flash_gguf/engram_io.cpp",
        "src/hy4flash_gguf/sampler.cpp",
        "src/kernels/ngram.cpp",
        "src/ngram/ple_reader.cpp",
        "src/platform/direct_file.cpp",
        "src/platform/memory.cpp",
        "metal/src/engine/stream.cpp",
        "metal/src/hy4flash_gguf/dispatch.cpp",
        "metal/src/hy4flash_gguf/model.cpp",
        "metal/src/hy4flash_gguf/experts.cpp",
        "metal/src/hy4flash_gguf/engram.cpp",
        "metal/src/hy4flash_gguf/forward.cpp",
        "metal/src/hy4flash_gguf/dump.cpp",
        "metal/src/hy4flash_gguf/mtp.cpp",
        "metal/src/hy4flash_gguf/prefill_snap.cpp",
        "metal/src/hy4flash_gguf/vision.cpp",
        "metal/src/hy4flash_gguf/vis_tower.cpp",
        "metal/src/hy4flash_gguf/vis_pos.cpp",
        "metal/src/mtl_rt.mm",
        "metal/src/ref_quant.cpp",
        "metal/src/gguf_lite.cpp",
    ];
    let sources = sources.into_iter().map(|path| vendor.join(path)).chain(
        [
            "stark_hy4.cpp",
            "stark_hy4_vision.cpp",
            "stq_codec.cpp",
            "stq_attention.cpp",
            "stq_engram.cpp",
            "stq_embedding.cpp",
            "stq_expert.cpp",
            "stq_geometry.cpp",
            "stq_model.cpp",
        ]
        .map(|path| native.join(path)),
    );
    let mut jobs = Vec::new();
    for (index, source) in sources.enumerate() {
        let object = out_dir.join(format!("hy4_{index}.o"));
        let mut clang = Command::new("xcrun");
        clang
            .args(["-sdk", "macosx", "clang++", "-arch", "arm64", "-std=c++20", "-O2", "-Wno-deprecated-declarations"])
            .arg("-I")
            .arg(out_dir)
            .arg("-I")
            .arg(native);
        for header in headers {
            clang.arg("-I").arg(vendor.join(header));
        }
        if source.extension().is_some_and(|ext| ext == "mm") {
            clang.arg("-fobjc-arc");
        }
        let child = clang
            .arg("-c")
            .arg(&source)
            .arg("-o")
            .arg(&object)
            .spawn()
            .with_context(|| format!("starting the compiler for {}", source.display()))?;
        jobs.push((child, source, object));
    }
    let mut failure = None;
    let mut objects = Vec::with_capacity(jobs.len());
    for (mut child, source, object) in jobs {
        if !child.wait()?.success() {
            failure = Some(source);
        }
        objects.push(object);
    }
    ensure!(failure.is_none(), "failed to compile Hy4 Metal source: {failure:?}");
    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "libtool", "-static", "-o"])
        .arg(out_dir.join("libstark_hy4.a"))
        .args(objects)
        .status()
        .context("archiving the Hy4 Metal graph")?;
    ensure!(status.success(), "failed to archive the Hy4 Metal graph");
    println!("cargo::rustc-link-search=native={}", out_dir.display());
    println!("cargo::rustc-link-lib=static=stark_hy4");
    println!("cargo::rustc-link-lib=c++");
    println!("cargo::rustc-link-lib=framework=Metal");
    println!("cargo::rustc-link-lib=framework=Foundation");
    Ok(())
}
