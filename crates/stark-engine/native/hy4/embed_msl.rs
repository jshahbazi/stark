use std::{
    fs::{self, File},
    io::{self, Write},
    path::Path,
};

pub fn generate(
    native: &Path,
    out: &Path,
) -> io::Result<()> {
    let source = native.join("vendor/strata/engine");
    let mut paths = vec![source.join("include/strata/quant/strata-quant-common.h")];
    for kernel in ["quant.metal", "units.metal", "gemv.metal"] {
        paths.push(source.join("metal/kernels").join(kernel));
    }
    for entry in fs::read_dir(source.join("metal/src/hy4flash_gguf/msl"))? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("metal") {
            paths.push(path);
        }
    }
    paths.extend(
        [
            "pair1_k4_grouped.metal",
            "pair1_k4_decode.metal",
            "pair1_k4_io.metal",
            "spec1_dense.metal",
            "stq_mla_register.metal",
            "stq_embedding.metal",
        ]
        .map(|name| native.join("stq_metal").join(name)),
    );
    paths.sort();

    let mut header = File::create(out.join("stark_hy4_msl.h"))?;
    writeln!(header, "#pragma once\n#include <string_view>\n")?;
    writeln!(header, "namespace stark::hy4 {{")?;
    writeln!(header, "inline std::string_view msl(std::string_view name) {{")?;
    for path in paths {
        println!("cargo:rerun-if-changed={}", path.display());
        let name = path.file_name().and_then(|name| name.to_str()).unwrap();
        let code = fs::read_to_string(&path)?;
        if code.contains(")stark_msl\"") {
            return Err(io::Error::new(io::ErrorKind::InvalidData, path.display().to_string()));
        }
        writeln!(header, "    if (name == \"{name}\") return R\"stark_msl({code})stark_msl\";")?;
    }
    writeln!(header, "    return {{}};\n}}\n}}")?;
    Ok(())
}
