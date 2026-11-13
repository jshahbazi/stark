//! Disposable native evaluation of untrusted model-provided chat templates.

mod engine;
mod limits;

use std::io::{Read, Write};

#[global_allocator]
static ALLOCATOR: limits::BoundedAllocator = limits::BoundedAllocator;

fn main() {
    if limits::configure().is_err() || !limits::install_process_limits() {
        limits::terminate();
    }
    std::panic::set_hook(Box::new(|_| limits::terminate()));
    let mut arguments = std::env::args_os().skip(1);
    if let Some(argument) = arguments.next() {
        if argument != "--self-test" || arguments.next().is_some() {
            limits::terminate();
        }
        let request = br#"{"source":"{{ value }}","arguments":{"value":"synthetic control"}}"#;
        if engine::evaluate(request).is_ok_and(|value| value == b"synthetic control") {
            let _ = std::io::stdout().write_all(b"{\"status\":\"ok\",\"protocol\":\"stark-template-v1\"}\n");
            return;
        }
        limits::terminate();
    }
    let mut data = Vec::new();
    if std::io::stdin().take((limits::wire_bytes() + 1) as u64).read_to_end(&mut data).is_err() {
        limits::terminate();
    }
    let (status, output) = match engine::evaluate(&data) {
        Ok(output) => (b'O', output),
        Err(error) => {
            let _ = std::io::stdout().write_all(&[error.status()]);
            std::process::exit(1);
        },
    };
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    if stdout.write_all(&[status]).and_then(|_| stdout.write_all(&output)).is_err() {
        std::process::exit(1);
    }
}
