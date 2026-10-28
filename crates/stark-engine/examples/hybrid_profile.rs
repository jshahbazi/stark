//! Aggregate native full-checkpoint profile. Never prints model inputs/outputs.

use std::{path::Path, time::Instant};

use sha2::{Digest, Sha256};
use stark_engine::engine::hybrid::{Error, Model, profile};

// Original benchmark-only activity guard. Algorithmic reference: Lily
// e22d58fa7cc488493fb1b5d810a198bd29730aeb/src/activity.rs. Options and ABI
// follow the local Foundation NSProcessInfo.h declarations; no dependency added.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}

#[cfg(all(target_os = "macos", feature = "metal"))]
struct ActivityAssertion {
    process: objc2::rc::Retained<objc2::runtime::AnyObject>,
    token: objc2::rc::Retained<objc2::runtime::AnyObject>,
    // The assertion starts and ends on this replay's calling thread.
    same_thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
struct ActivityAssertion;

impl ActivityAssertion {
    fn begin(enabled: bool) -> Result<Option<Self>, &'static str> {
        if !enabled {
            return Ok(None);
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            use objc2::{
                msg_send,
                rc::Retained,
                runtime::{AnyClass, AnyObject},
            };
            let process_class = AnyClass::get(c"NSProcessInfo").ok_or("activity process class unavailable")?;
            let string_class = AnyClass::get(c"NSString").ok_or("activity string class unavailable")?;
            // The Foundation class methods return autoreleased object pointers;
            // Retained applies the matching retain while handling nil explicitly.
            let process: Option<Retained<AnyObject>> = unsafe { msg_send![process_class, processInfo] };
            let reason: Option<Retained<AnyObject>> =
                unsafe { msg_send![string_class, stringWithUTF8String: c"Stark native benchmark".as_ptr()] };
            let process = process.ok_or("activity process unavailable")?;
            let reason = reason.ok_or("activity reason unavailable")?;
            // NSActivityUserInitiated | NSActivityLatencyCritical, both u64.
            let options = 0x00ff_ffffu64 | 0xff_0000_0000u64;
            let token: Option<Retained<AnyObject>> =
                unsafe { msg_send![&*process, beginActivityWithOptions: options, reason: &*reason] };
            Ok(Some(Self {
                process,
                token: token.ok_or("activity assertion unavailable")?,
                same_thread: std::marker::PhantomData,
            }))
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        Err("activity assertion requires macOS Metal")
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl Drop for ActivityAssertion {
    fn drop(&mut self) {
        // This is the exact retained token returned by this process's begin;
        // the guard cannot be sent to another thread.
        unsafe {
            let _: () = objc2::msg_send![&*self.process, endActivity: &*self.token];
        }
    }
}

// These variants carry fixed source labels only. Never format I/O exceptions,
// model values or metadata into diagnostic output.
fn native_error(error: Error) -> &'static str {
    match error {
        Error::Invalid(label) | Error::Unsupported(label) => label,
        _ => "native execution failed",
    }
}

fn run() -> Result<(), &'static str> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(3..=4).contains(&args.len()) {
        return Err("expected model directory, decode steps, replays and optional synthetic prefix length");
    }
    let steps: usize = args[1].parse().map_err(|_| "invalid step count")?;
    let replays: usize = args[2].parse().map_err(|_| "invalid replay count")?;
    if !(2..=256).contains(&steps) || !(1..=8).contains(&replays) {
        return Err("invalid profile bounds");
    }
    let mut model = Model::load(Path::new(&args[0])).map_err(|_| "model admission failed")?;
    let mut input = model.tokenizer().encode("Hello.", false).map_err(|_| "tokenization failed")?.get_ids().to_vec();
    if input.is_empty() {
        return Err("empty profile input");
    }
    if let Some(length) = args.get(3) {
        let length: usize = length.parse().map_err(|_| "invalid prefix length")?;
        if !(2..=65536).contains(&length) {
            return Err("invalid prefix length");
        }
        // Synthetic repeated prefix: performance coverage, not an answer-
        // quality evaluation. IDs remain private and are never persisted.
        input = input.iter().copied().cycle().take(length).collect();
    }
    let mut input_hash = Sha256::new();
    for &token in &input {
        input_hash.update(token.to_le_bytes());
    }
    let input_sha256 = format!("{:x}", input_hash.finalize());
    let activity_requested = std::env::var_os("STARK_PROFILE_ACTIVITY").is_some_and(|value| value == "1");
    for replay in 0..replays {
        let activity = ActivityAssertion::begin(activity_requested)?;
        model.reset();
        profile::reset();
        profile::enable(true);
        let started = Instant::now();
        let mut logits = Vec::new();
        for (index, &token) in input.iter().enumerate() {
            if index + 1 == input.len() {
                logits = model.step(token).map_err(native_error)?;
            } else {
                model.ingest(token).map_err(native_error)?;
            }
        }
        let prefill_seconds = started.elapsed().as_secs_f64();
        let prefill_profile = profile::snapshot();
        profile::reset();
        let started = Instant::now();
        let mut hash = Sha256::new();
        for _ in 0..steps {
            let mut best = 0;
            for i in 1..logits.len() {
                if logits[i] > logits[best] {
                    best = i;
                }
            }
            // Hash only: inputs, outputs, probabilities and logits stay private.
            hash.update((best as u32).to_le_bytes());
            logits = model.step(best as u32).map_err(native_error)?;
        }
        let decode_seconds = started.elapsed().as_secs_f64();
        profile::enable(false);
        println!(
            "{}",
            serde_json::json!({"replay": replay, "input_tokens": input.len(), "input_sha256": input_sha256,
            "activity_assertion": activity.is_some(),
            "synthetic_repeated_prefix": args.len() == 4, "decode_steps": steps,
            "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
            "decode_tokens_per_second": steps as f64 / decode_seconds,
            "output_sha256": format!("{:x}",hash.finalize()),
            "prefill_profile": prefill_profile, "decode_profile": profile::snapshot()})
        );
    }
    model.reset();
    Ok(())
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    match std::panic::catch_unwind(run) {
        Ok(Ok(())) => {},
        Ok(Err(message)) => {
            eprintln!("native aggregate profile failed: {message}");
            std::process::exit(1);
        },
        Err(_) => {
            eprintln!("native aggregate profile failed");
            std::process::exit(1);
        },
    }
}
