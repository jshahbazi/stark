// SPDX-License-Identifier: Apache-2.0
//! Content-free original Inkling vision diagnostics for local fixtures.

#[cfg(all(target_os = "macos", feature = "inkling-slice"))]
fn main() -> std::process::ExitCode {
    use std::{env::args_os, fs, path::Path, process::ExitCode};

    let args: Vec<_> = args_os().skip(1).collect();
    if args.len() != 3 {
        eprintln!("usage: inkling_vision <checkpoint> <image> <oracle>");
        return ExitCode::FAILURE;
    }
    let result = fs::read(&args[1]).map_err(|_| "cannot read image").and_then(|image| {
        stark_engine::engine::inkling_slice::measure_vision(Path::new(&args[0]), &image, Path::new(&args[2]))
    });
    match result {
        Ok(summary) => {
            println!("{summary}");
            ExitCode::SUCCESS
        },
        Err(reason) => {
            eprintln!("{reason}");
            ExitCode::FAILURE
        },
    }
}

#[cfg(not(all(target_os = "macos", feature = "inkling-slice")))]
fn main() {
    eprintln!("Inkling vision diagnostics require the native Inkling feature on macOS");
    std::process::exit(1);
}
