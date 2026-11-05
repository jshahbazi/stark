fn main() {
    println!("cargo::rerun-if-env-changed=DEP_STARK_ENGINE_GLM_EXL3_LIB_DIR");
    if std::env::var_os("CARGO_FEATURE_GLM_EXL3").is_some() {
        let path = std::env::var("DEP_STARK_ENGINE_GLM_EXL3_LIB_DIR")
            .expect("the GLM EXL3 engine must supply its validated MLX library directory");
        assert!(!path.contains(['\n', '\r']), "invalid GLM EXL3 library directory");
        println!("cargo::rustc-link-arg=-Wl,-rpath,{path}");
    }
}
