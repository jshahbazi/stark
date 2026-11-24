// SPDX-License-Identifier: Apache-2.0
//! Aggregate-only native diagnostic for the original Inkling 2 Mini.

#[cfg(target_os = "macos")]
fn main() {
    use std::path::Path;

    fn run() -> Result<i32, &'static str> {
        let mut args: Vec<_> = std::env::args_os().skip(1).collect();
        if args.first().is_some_and(|arg| arg == "--capture-dspark-trace") {
            if args.len() != 5 {
                return Err(
                    "expected --capture-dspark-trace, original SafeTensors, DSpark SafeTensors, prompt SafeTensors and private output SafeTensors",
                );
            }
            let report = stark_engine::engine::inkling_slice::capture_dspark_trace(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
            )?;
            println!("{report}");
            return Ok(0);
        }
        // Diagnostic only: run the parity and transaction checks on the
        // bit-exact reference prefill kernels instead of the default path.
        let exact_prefill = match args.iter().position(|arg| arg == "--exact-prefill") {
            Some(index) => {
                args.remove(index);
                true
            },
            None => false,
        };
        use stark_engine::engine::inkling_slice::PrefillTraceMode;
        if args.first().is_some_and(|arg| arg == "--measure-trace-verify-widths") {
            if args.len() != 4 {
                return Err(
                    "expected --measure-trace-verify-widths, target SafeTensors, draft SafeTensors and private teacher trace",
                );
            }
            let report = stark_engine::engine::inkling_slice::measure_trace_verify_widths(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
            )?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| arg == "--measure-trace-candidate-rounds") {
            if !(5..=6).contains(&args.len()) || (args.len() == 6 && args[5] != "--seven-first") {
                return Err(
                    "expected --measure-trace-candidate-rounds, target SafeTensors, MTP SafeTensors, teacher trace, candidate proposals and optional --seven-first",
                );
            }
            let report = stark_engine::engine::inkling_slice::measure_candidate_rounds(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                args.len() == 6,
            )?;
            println!("{report}");
            return Ok(0);
        }
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
        if args.first().is_some_and(|arg| arg == "--measure-draft-overlap") {
            if args.len() != 5 {
                return Err(
                    "expected --measure-draft-overlap, original and draft SafeTensors, private oracle and seed",
                );
            }
            let seed = args[4].to_str().and_then(|value| value.parse().ok()).ok_or("invalid Inkling sampling seed")?;
            let report = stark_engine::engine::inkling_slice::measure_draft_overlap(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                seed,
            )?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| arg == "--measure-verify-transaction") {
            if args.len() != 5 {
                return Err(
                    "expected --measure-verify-transaction, original and draft SafeTensors, private oracle and prefix",
                );
            }
            let prefix =
                args[4].to_str().and_then(|value| value.parse().ok()).ok_or("invalid Inkling prefix length")?;
            let report = stark_engine::engine::inkling_slice::measure_verify_transaction(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                prefix,
                exact_prefill,
            )?;
            println!("{report}");
            return Ok(0);
        }
        // Affine 8-bit dense and readout kernels against BF16 on restored
        // real state: alternating GPU times and numeric differences.
        if args.first().is_some_and(|arg| arg == "--measure-affine8-dense") {
            if args.len() != 8 {
                return Err(
                    "expected --measure-affine8-dense, original and draft SafeTensors, affine tensors, private token directory, prompt, context and pairs",
                );
            }
            let number = |index: usize| {
                args[index].to_str().and_then(|value| value.parse::<usize>().ok()).ok_or("invalid number")
            };
            let report = stark_engine::engine::inkling_slice::measure_affine8_dense(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                number(5)?,
                number(6)?,
                number(7)?,
            )?;
            println!("{report}");
            return Ok(0);
        }
        // Affine8 artifact against the original artifact: teacher-forced
        // logit quality, and paired step, verify-round and prefill times.
        if args.first().is_some_and(|arg| arg == "--affine8-quality") {
            if args.len() != 6 {
                return Err(
                    "expected --affine8-quality, original and affine8 SafeTensors, private token directory, prompt and context",
                );
            }
            let number = |index: usize| {
                args[index].to_str().and_then(|value| value.parse::<usize>().ok()).ok_or("invalid number")
            };
            let report = stark_engine::engine::inkling_slice::measure_affine8_quality(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                number(4)?,
                number(5)?,
            )?;
            println!("{report}");
            return Ok(0);
        }
        // Teacher-forced quality comparison of two sides, each `default:PATH`
        // (default prefill) or `exact:PATH` (bit-exact reference prefill).
        if args.first().is_some_and(|arg| arg == "--quality-compare") {
            if args.len() != 7 {
                return Err(
                    "expected --quality-compare, reference and candidate sides (default:PATH or exact:PATH), private token directory, prompt, context and window",
                );
            }
            let number = |index: usize| {
                args[index].to_str().and_then(|value| value.parse::<usize>().ok()).ok_or("invalid number")
            };
            let side = |index: usize| -> Result<stark_engine::engine::inkling_slice::QualitySide<'_>, &'static str> {
                let value = args[index].to_str().ok_or("invalid side")?;
                let (control_off, value) = match value.strip_prefix("off-").and_then(|rest| rest.split_once(':')) {
                    Some((control, rest)) => (Some(control), rest),
                    None => (None, value),
                };
                let (exact, path) = if let Some(path) = value.strip_prefix("exact:") {
                    (true, path)
                } else {
                    (false, value.strip_prefix("default:").unwrap_or(value))
                };
                Ok(stark_engine::engine::inkling_slice::QualitySide {
                    checkpoint: Path::new(path),
                    exact_prefill: exact,
                    control_off,
                })
            };
            let report = stark_engine::engine::inkling_slice::measure_quality_compare(
                side(1)?,
                side(2)?,
                Path::new(&args[3]),
                number(4)?,
                number(5)?,
                number(6)?,
            )?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| arg == "--paired-control") {
            if args.len() != 8 {
                return Err(
                    "expected --paired-control, target and draft SafeTensors, private token directory, prompt, context, pairs and control",
                );
            }
            let number = |index: usize| {
                args[index].to_str().and_then(|value| value.parse::<usize>().ok()).ok_or("invalid number")
            };
            let report = stark_engine::engine::inkling_slice::measure_paired_control(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                number(4)?,
                number(5)?,
                number(6)?,
                args[7].to_str().ok_or("invalid control")?,
            )?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| arg == "--depth-policies") {
            if args.len() != 8 {
                return Err(
                    "expected --depth-policies, target and draft SafeTensors, private token directory, prompt, context, outputs and repeats",
                );
            }
            let number = |index: usize| {
                args[index].to_str().and_then(|value| value.parse::<usize>().ok()).ok_or("invalid number")
            };
            let report = stark_engine::engine::inkling_slice::measure_depth_policies(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                number(4)?,
                number(5)?,
                number(6)?,
                number(7)?,
            )?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| arg == "--draft-trace") {
            if args.len() != 7 {
                return Err(
                    "expected --draft-trace, target and draft SafeTensors, private token directory, prompt, context and outputs",
                );
            }
            let number = |index: usize| {
                args[index].to_str().and_then(|value| value.parse::<usize>().ok()).ok_or("invalid number")
            };
            let report = stark_engine::engine::inkling_slice::measure_draft_trace(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                number(4)?,
                number(5)?,
                number(6)?,
            )?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| arg == "--round-budget") {
            if args.len() != 7 {
                return Err(
                    "expected --round-budget, target and draft SafeTensors, private token directory, prompt, context and outputs",
                );
            }
            let number = |index: usize| {
                args[index].to_str().and_then(|value| value.parse::<usize>().ok()).ok_or("invalid number")
            };
            let report = stark_engine::engine::inkling_slice::measure_round_budget(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                number(4)?,
                number(5)?,
                number(6)?,
            )?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| arg == "--decode-experts") {
            if args.len() != 7 {
                return Err(
                    "expected --decode-experts, target and draft SafeTensors, private token directory, prompt, context and pairs",
                );
            }
            let number = |index: usize| {
                args[index].to_str().and_then(|value| value.parse::<usize>().ok()).ok_or("invalid number")
            };
            let report = stark_engine::engine::inkling_slice::measure_decode_experts(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                number(4)?,
                number(5)?,
                number(6)?,
            )?;
            println!("{report}");
            return Ok(0);
        }
        // Decode stage table: per-stage GPU time, launches, bytes and read ceilings.
        if args.first().is_some_and(|arg| arg == "--decode-stages") {
            if args.len() != 7 {
                return Err(
                    "expected --decode-stages, target and draft SafeTensors, private token directory, prompt, context and repeats",
                );
            }
            let number = |index: usize| {
                args[index].to_str().and_then(|value| value.parse::<usize>().ok()).ok_or("invalid number")
            };
            let report = stark_engine::engine::inkling_slice::measure_decode_stages(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                number(4)?,
                number(5)?,
                number(6)?,
            )?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| arg == "--affine8-timing") {
            if args.len() != 8 {
                return Err(
                    "expected --affine8-timing, original, affine8 and draft SafeTensors, private token directory, prompt, context and pairs",
                );
            }
            let number = |index: usize| {
                args[index].to_str().and_then(|value| value.parse::<usize>().ok()).ok_or("invalid number")
            };
            let report = stark_engine::engine::inkling_slice::measure_affine8_timing(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                number(5)?,
                number(6)?,
                number(7)?,
            )?;
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
        // Caller-chosen prompt length and context under the worker's cold
        // schedule. These subcommands leave every fixed-shape command above
        // and below unchanged.
        let count = |index: usize, label: &'static str| {
            args.get(index).and_then(|value| value.to_str()).and_then(|value| value.parse::<usize>().ok()).ok_or(label)
        };
        if args.first().is_some_and(|arg| arg == "--full-target-schedule") {
            if args.len() != 4 {
                return Err(
                    "expected --full-target-schedule, original SafeTensors, private oracle directory and context",
                );
            }
            let report = stark_engine::engine::inkling_slice::measure_full_schedule(
                Path::new(&args[1]),
                Path::new(&args[2]),
                count(3, "invalid Inkling context")?,
                exact_prefill,
            )?;
            println!("{report}");
            return Ok(i32::from(
                ["all_logits_exact", "exact_snapshot_replay", "exact_boundary_replays"]
                    .iter()
                    .any(|key| report[*key] != true),
            ));
        }
        // Tolerance report: default tensor prefill against the pinned
        // captures and against Stark's bit-exact reference kernels.
        // The draft variant takes the draft SafeTensors after the
        // target and profiles prefill with drafting enabled.
        if args.first().is_some_and(|arg| arg == "--profile-schedule-draft") {
            if !(7..=8).contains(&args.len()) {
                return Err(
                    "expected --profile-schedule-draft, target, draft, token directory, prompt, context, repeats and optional command rows",
                );
            }
            let report = stark_engine::engine::inkling_slice::profile_schedule(
                Path::new(&args[1]),
                Path::new(&args[3]),
                count(4, "invalid Inkling prompt length")?,
                count(5, "invalid Inkling context")?,
                count(6, "invalid repeat count")?,
                Some(Path::new(&args[2])),
                args.get(7).and_then(|value| value.to_str()?.parse().ok()).unwrap_or(2048),
            )?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| arg == "--dense-ceiling") {
            if args.len() != 6 {
                return Err("expected --dense-ceiling, target, rows, width, columns and repeats");
            }
            let report = stark_engine::engine::inkling_slice::measure_dense_ceiling(
                Path::new(&args[1]),
                count(2, "invalid rows")?,
                count(3, "invalid width")?,
                count(4, "invalid columns")?,
                count(5, "invalid repeat count")?,
            )?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| arg == "--expert-ceiling") {
            if !(7..=9).contains(&args.len()) {
                return Err(
                    "expected --expert-ceiling, target, token directory, prompt, context, visit, repeats, optional command rows and optional MLX-Serve kernel directory",
                );
            }
            let report = stark_engine::engine::inkling_slice::measure_expert_ceiling(
                Path::new(&args[1]),
                Path::new(&args[2]),
                count(3, "invalid Inkling prompt length")?,
                count(4, "invalid Inkling context")?,
                count(5, "invalid visit")?,
                count(6, "invalid repeat count")?,
                args.get(7).and_then(|value| value.to_str()?.parse().ok()).unwrap_or(2048),
                args.get(8).map(Path::new),
            )?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| arg == "--profile-worker-sequence") {
            if args.len() != 9 {
                return Err(
                    "expected --profile-worker-sequence, target, draft, token directory, prompt, context, repeats, idle ms and setup",
                );
            }
            let report = stark_engine::engine::inkling_slice::profile_worker_sequence(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                count(4, "invalid Inkling prompt length")?,
                count(5, "invalid Inkling context")?,
                count(6, "invalid repeat count")?,
                count(7, "invalid idle milliseconds")? as u64,
                args[8].to_str().ok_or("invalid setup")?,
            )?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| arg == "--profile-schedule") {
            if !(6..=7).contains(&args.len()) {
                return Err(
                    "expected --profile-schedule, original SafeTensors, private token directory, prompt length, context and repeats",
                );
            }
            let report = stark_engine::engine::inkling_slice::profile_schedule(
                Path::new(&args[1]),
                Path::new(&args[2]),
                count(3, "invalid Inkling prompt length")?,
                count(4, "invalid Inkling context")?,
                count(5, "invalid repeat count")?,
                None,
                args.get(6).and_then(|value| value.to_str()?.parse().ok()).unwrap_or(2048),
            )?;
            println!("{report}");
            return Ok(0);
        }
        if args.first().is_some_and(|arg| arg == "--split-consistency") {
            if args.len() != 5 {
                return Err(
                    "expected --split-consistency, original SafeTensors, private token directory, prompt length and context",
                );
            }
            let report = stark_engine::engine::inkling_slice::measure_split_consistency(
                Path::new(&args[1]),
                Path::new(&args[2]),
                count(3, "invalid Inkling prompt length")?,
                count(4, "invalid Inkling context")?,
            )?;
            println!("{report}");
            return Ok(i32::from(report["verdicts"]["default_identical"] != true));
        }
        // The default path is judged against a noise-floor control of
        // one-unit BF16 residual flips on the exact kernels.
        if args.first().is_some_and(|arg| arg == "--prefill-tolerance") {
            if args.len() < 5 {
                return Err(
                    "expected --prefill-tolerance, original SafeTensors, context, control trials and capture directories",
                );
            }
            let oracles = args[4..].iter().map(std::path::PathBuf::from).collect::<Vec<_>>();
            let report = stark_engine::engine::inkling_slice::measure_prefill_tolerance(
                Path::new(&args[1]),
                &oracles,
                count(2, "invalid Inkling context")?,
                count(3, "invalid control trial count")?,
            )?;
            println!("{report}");
            return Ok(i32::from(report["noise_floor_pass"] != true));
        }
        if args.first().is_some_and(|arg| arg == "--schedule-replay") {
            if !(5..=6).contains(&args.len()) {
                return Err(
                    "expected --schedule-replay, original SafeTensors, private token directory, prompt length, context and optional command rows",
                );
            }
            let report = stark_engine::engine::inkling_slice::measure_schedule_replay(
                Path::new(&args[1]),
                Path::new(&args[2]),
                count(3, "invalid Inkling prompt length")?,
                count(4, "invalid Inkling context")?,
                exact_prefill,
                args.get(5).and_then(|value| value.to_str()?.parse().ok()).unwrap_or(2048),
            )?;
            println!("{report}");
            return Ok(i32::from(
                ["exact_snapshot_replay", "exact_boundary_replays"].iter().any(|key| report[*key] != true),
            ));
        }
        if args.first().is_some_and(|arg| {
            arg == "--measure-verify-transaction-schedule" || arg == "--measure-source-dspark-depth-schedule"
        }) {
            if args.len() != 6 {
                return Err(
                    "expected the schedule check, original and draft SafeTensors, private directory, prompt length and context",
                );
            }
            let prompt = count(4, "invalid Inkling prompt length")?;
            let context = count(5, "invalid Inkling context")?;
            let (target, draft, oracle) = (Path::new(&args[1]), Path::new(&args[2]), Path::new(&args[3]));
            if args[0] == "--measure-source-dspark-depth-schedule" {
                let report = stark_engine::engine::inkling_slice::measure_source_dspark_depth_schedule(
                    target,
                    draft,
                    oracle,
                    prompt,
                    context,
                    exact_prefill,
                )?;
                println!("{report}");
                return Ok(0);
            }
            let report = stark_engine::engine::inkling_slice::measure_verify_transaction_schedule(
                target,
                draft,
                oracle,
                prompt,
                context,
                exact_prefill,
            )?;
            println!("{report}");
            return Ok(i32::from(report["seed_matches_pinned"] == false || report["exact_boundary_replays"] != true));
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
                exact_prefill,
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
