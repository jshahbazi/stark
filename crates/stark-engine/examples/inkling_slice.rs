// SPDX-License-Identifier: Apache-2.0
//! Aggregate-only native diagnostic for the original Inkling 2 Mini.

#[cfg(target_os = "macos")]
fn main() {
    use std::path::Path;

    fn run() -> Result<i32, &'static str> {
        let args: Vec<_> = std::env::args_os().skip(1).collect();
        if !(2..=3).contains(&args.len()) || (args.len() == 3 && args[2] != "--parity-only") {
            return Err("expected original SafeTensors, private reference SafeTensors and optional --parity-only");
        }
        let report =
            stark_engine::engine::inkling_slice::measure(Path::new(&args[0]), Path::new(&args[1]), args.len() == 3)?;
        println!("{report}");
        Ok(if report["parity"]["passed"] == true {
            0
        } else {
            1
        })
    }

    std::panic::set_hook(Box::new(|_| {}));
    let code = match std::panic::catch_unwind(run) {
        Ok(Ok(code)) => code,
        Ok(Err(label)) => {
            eprintln!("native Inkling slice failed: {label}");
            1
        },
        Err(_) => {
            eprintln!("native Inkling slice failed");
            1
        },
    };
    std::process::exit(code);
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("native Inkling slice requires macOS Metal");
    std::process::exit(1);
}
