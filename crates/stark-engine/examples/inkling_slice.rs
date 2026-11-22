// SPDX-License-Identifier: Apache-2.0
//! Aggregate-only native diagnostic for the original Inkling 2 Mini.

#[cfg(target_os = "macos")]
fn main() {
    use std::path::Path;

    fn run() -> Result<i32, &'static str> {
        let args: Vec<_> = std::env::args_os().skip(1).collect();
        use stark_engine::engine::inkling_slice::PrefillTraceMode;
        if args.first().is_some_and(|arg| arg == "--profile-stock-4k" || arg == "--probe-stock-nvfp4") {
            if args.len() < 4 {
                return Err(
                    "expected --profile-stock-4k, original SafeTensors, draft SafeTensors and stock oracle directories",
                );
            }
            let oracles = args[3..].iter().map(std::path::PathBuf::from).collect::<Vec<_>>();
            let profile = if args[0] == "--probe-stock-nvfp4" {
                stark_engine::engine::inkling_slice::probe_stock_nvfp4
            } else {
                stark_engine::engine::inkling_slice::profile_stock_4k
            };
            let report = profile(Path::new(&args[1]), Path::new(&args[2]), &oracles)?;
            println!("{report}");
            return Ok(0);
        }

        if args.first().is_some_and(|arg| arg == "--measure-dspark-depth" || arg == "--measure-source-dspark-depth") {
            if args.len() != 5 {
                return Err(
                    "expected --measure-dspark-depth, original and draft SafeTensors, private oracle directory and prefix",
                );
            }
            let prefix =
                args[4].to_str().and_then(|value| value.parse().ok()).ok_or("invalid Inkling prefix length")?;
            let measure = if args[0] == "--measure-source-dspark-depth" {
                stark_engine::engine::inkling_slice::measure_source_dspark_depth
            } else {
                stark_engine::engine::inkling_slice::measure_dspark_depth
            };
            let report = measure(Path::new(&args[1]), Path::new(&args[2]), Path::new(&args[3]), prefix)?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| arg == "--measure-decode-verify") {
            if args.len() != 4 {
                return Err(
                    "expected --measure-decode-verify, original SafeTensors, private oracle directory and prefix length",
                );
            }
            let prefix =
                args[3].to_str().and_then(|value| value.parse().ok()).ok_or("invalid Inkling prefix length")?;
            let report = stark_engine::engine::inkling_slice::measure_decode_verify(
                Path::new(&args[1]),
                Path::new(&args[2]),
                prefix,
            )?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| arg == "--prefill-speed") {
            if args.len() != 5 {
                return Err("expected --prefill-speed, original SafeTensors, private oracle, input length and variant");
            }
            let count = args[3].to_str().and_then(|value| value.parse().ok()).ok_or("invalid Inkling input length")?;
            let variant = args[4].to_str().ok_or("invalid Inkling prefill candidate")?;
            let report = stark_engine::engine::inkling_slice::measure_prefill_speed(
                Path::new(&args[1]),
                Path::new(&args[2]),
                count,
                variant,
            )?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| arg == "--trace-prefill-packed") {
            if args.len() != 4 {
                return Err("expected --trace-prefill-packed, original SafeTensors, private oracle and variant");
            }
            let variant = args[3].to_str().ok_or("invalid packed prefill variant")?;
            let report = stark_engine::engine::inkling_slice::trace_packed_prefill(
                Path::new(&args[1]),
                Path::new(&args[2]),
                variant,
            )?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| arg == "--probe-prefill-expert-weights") {
            if args.len() != 3 {
                return Err(
                    "expected --probe-prefill-expert-weights, original SafeTensors and private oracle directory",
                );
            }
            let report = stark_engine::engine::inkling_slice::probe_prefill_expert_weights(
                Path::new(&args[1]),
                Path::new(&args[2]),
            )?;
            println!("{report}");
            return Ok(0);
        }
        let trace_mode = match args.first().and_then(|arg| arg.to_str()) {
            Some("--trace-prefill-512") => Some(PrefillTraceMode::Baseline),
            Some("--trace-prefill-512-dense") => Some(PrefillTraceMode::Dense),
            Some("--trace-prefill-512-local") => Some(PrefillTraceMode::Local),
            Some("--trace-prefill-512-routed") => Some(PrefillTraceMode::Routed),
            _ => None,
        };
        if let Some(mode) = trace_mode {
            if args.len() != 3 {
                return Err("expected --trace-prefill-512, original SafeTensors and private oracle directory");
            }
            let report =
                stark_engine::engine::inkling_slice::trace_prefill(Path::new(&args[1]), Path::new(&args[2]), mode)?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| {
            arg == "--profile-prefill" || arg == "--profile-prefill-source" || arg == "--profile-prefill-packed"
        }) {
            let packed = args[0] == "--profile-prefill-packed";
            if args.len()
                != if packed {
                    5
                } else {
                    4
                }
            {
                return Err(
                    "expected --profile-prefill, original SafeTensors, private oracle directory and input length",
                );
            }
            let count = args[3].to_str().and_then(|value| value.parse().ok()).ok_or("invalid Inkling input length")?;
            let checkpoint = Path::new(&args[1]);
            let oracle = Path::new(&args[2]);
            let report = if packed {
                let variant = args[4].to_str().ok_or("invalid Inkling prefill candidate")?;
                stark_engine::engine::inkling_slice::profile_packed_prefill(checkpoint, oracle, count, variant)
            } else {
                stark_engine::engine::inkling_slice::profile_prefill(
                    checkpoint,
                    oracle,
                    count,
                    args[0] == "--profile-prefill-source",
                )
            }?;
            println!("{report}");
            return Ok(i32::from(report["exact_continuation_logits"] != true));
        }
        if args.first().is_some_and(|arg| arg == "--prefill-order") {
            if args.len() != 4 {
                return Err(
                    "expected --prefill-order, original SafeTensors, private oracle directory and input length",
                );
            }
            let count = args[3].to_str().and_then(|value| value.parse().ok()).ok_or("invalid Inkling input length")?;
            let report =
                stark_engine::engine::inkling_slice::measure_prefill(Path::new(&args[1]), Path::new(&args[2]), count)?;
            println!("{report}");
            return Ok(i32::from(
                [
                    "exact_target_state",
                    "exact_target_taps",
                    "exact_continuation_logits",
                    "exact_snapshot_replay",
                    "exact_snapshot_tile_replay",
                ]
                .iter()
                .any(|key| report[*key] != true),
            ));
        }
        if args.first().is_some_and(|arg| arg == "--dspark") {
            if args.len() != 6 {
                return Err("expected --dspark, target, draft, private fixture, prefix rows and decode rows");
            }
            let positive = |index: usize| {
                args[index].to_str().and_then(|value| value.parse::<usize>().ok()).ok_or("invalid row count")
            };
            let steps = positive(5)?;
            let report = stark_engine::engine::inkling_slice::measure_dspark(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                positive(4)?,
                steps,
            )?;
            println!("{report}");
            return Ok(i32::from(
                report["target_logit_matches"] != steps
                    || report["target_token_matches"] != steps
                    || report["source_ingress_proposal_matches"] != report["source_ingress_proposals"]
                    || report["draft_restore_exact"] != true
                    || report["rejects_other_mode_snapshot"] != true
                    || report["serial_recovery"] != true,
            ));
        }
        if args.first().is_some_and(|arg| arg == "--full-budget") {
            if args.len() != 3 {
                return Err("expected --full-budget, original SafeTensors and a private 4K cache seed");
            }
            let report =
                stark_engine::engine::inkling_slice::measure_full_budget(Path::new(&args[1]), Path::new(&args[2]))?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| arg == "--full-target" || arg == "--full-target-batched") {
            if args.len() != 4 {
                return Err("expected --full-target, original SafeTensors, private oracle directory and context");
            }
            let context = args[3].to_str().and_then(|value| value.parse().ok()).ok_or("invalid Inkling context")?;
            let report = stark_engine::engine::inkling_slice::measure_full(
                Path::new(&args[1]),
                Path::new(&args[2]),
                context,
                args[0] == "--full-target-batched",
            )?;
            println!("{report}");
            return Ok(if report["exact_snapshot_replay"] == true {
                0
            } else {
                1
            });
        }
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
