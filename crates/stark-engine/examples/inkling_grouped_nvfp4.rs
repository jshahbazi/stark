// SPDX-License-Identifier: Apache-2.0
//! Real-weight sorted-expert projection pilot. Emits aggregates only.

#[cfg(target_os = "macos")]
fn main() -> std::process::ExitCode {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [checkpoint, short, long] = args.as_slice() else {
        eprintln!("expected original checkpoint and private short and 4K oracle fixtures");
        return std::process::ExitCode::FAILURE;
    };
    use std::path::Path;
    match stark_engine::engine::inkling_slice::measure_grouped_nvfp4(
        Path::new(checkpoint),
        Path::new(short),
        Path::new(long),
    ) {
        Ok(report) => {
            println!("{report}");
            std::process::ExitCode::SUCCESS
        },
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        },
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("native Inkling grouped NVFP4 diagnostic requires macOS Metal");
    std::process::exit(1);
}
