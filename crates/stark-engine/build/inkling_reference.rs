// SPDX-License-Identifier: Apache-2.0
use std::{path::Path, process::Command};

use anyhow::{Context, ensure};

pub(super) fn build(out_dir: &Path) -> anyhow::Result<()> {
    let source = Path::new("src/engine/inkling_slice/metal_reference");
    println!("cargo::rerun-if-changed={}", source.display());

    let mut objects = Vec::new();
    for name in ["attention", "thin", "elementwise"] {
        let output = out_dir.join(format!("inkling_{name}.air"));
        let status = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-std=metal4.0", "-fno-fast-math", "-I"])
            .arg(source)
            .arg("-c")
            .arg(source.join(format!("{name}.metal")))
            .arg("-o")
            .arg(&output)
            .status()
            .with_context(|| format!("running the Metal compiler for Inkling {name}"))?;
        ensure!(status.success(), "Metal compiler failed for Inkling {name}");
        objects.push(output);
    }
    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .args(objects)
        .arg("-o")
        .arg(out_dir.join("inkling_source.metallib"))
        .status()
        .context("linking the Inkling Metal source reference")?;
    ensure!(status.success(), "Metal linker failed for the Inkling source reference");
    Ok(())
}
