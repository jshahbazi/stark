//! Edge0 uses a complete callback-driven request so the target/head lease and
//! checkpoint never escape a serialized generation request.

use std::cell::RefCell;

use stark_engine::engine::hybrid::{Error, MtpStats, Result as EngineResult};

use super::*;
use crate::generation::TokenCollector;

/// Bridge a synchronous native producer to the worker's ordinary output
/// collector. The producer must honor false and invoke control during prefill,
/// drafts, target verification and replay, including before long native work.
fn collect_callback_tokens(
    input_tokens: usize,
    options: &GenerationOptions,
    vocab_size: usize,
    started: Instant,
    on_token: &mut dyn FnMut(u32) -> Result<bool, &'static str>,
    on_progress: Option<&mut ProgressCallback<'_>>,
    cancelled: &mut dyn FnMut() -> Result<bool, &'static str>,
    produce: impl FnOnce(
        &mut dyn FnMut(u32) -> EngineResult<bool>,
        &mut dyn FnMut() -> EngineResult<()>,
    ) -> EngineResult<MtpStats>,
) -> Result<(GeneratedTokens, MtpStats), &'static str> {
    let collector = RefCell::new(TokenCollector::new(input_tokens, options));
    let progress = RefCell::new(on_progress);
    let cancelled = RefCell::new(cancelled);
    let error = RefCell::new(None);
    let first_output = std::cell::Cell::new(false);
    let last_prefill_progress = RefCell::new(Duration::ZERO);
    let check_cancel = || {
        if (*cancelled.borrow_mut())()? {
            Err("generation_cancelled")
        } else {
            Ok(())
        }
    };
    let authored_error = |value| {
        *error.borrow_mut() = Some(value);
        Error::Cancelled
    };
    let stats = produce(
        &mut |token| {
            let result = (|| {
                check_cancel()?;
                first_output.set(true);
                collector.borrow_mut().push(
                    u64::from(token),
                    vocab_size,
                    started.elapsed(),
                    on_token,
                    progress.borrow_mut().as_deref_mut(),
                )
            })();
            result.map_err(&authored_error)
        },
        &mut || {
            let result = (|| {
                check_cancel()?;
                if !first_output.get() {
                    report_prefill_progress(
                        started.elapsed(),
                        &mut last_prefill_progress.borrow_mut(),
                        progress.borrow_mut().as_deref_mut(),
                    )?;
                }
                Ok(())
            })();
            result.map_err(&authored_error)
        },
    );
    let stats = stats.map_err(|_| error.borrow().unwrap_or("native_generation_failed"))?;
    if !collector.borrow().is_complete() {
        return Err("invalid_generation_output");
    }
    Ok((collector.into_inner().finish(), stats))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::generation::FinishReason;

    fn options(budget: u32) -> GenerationOptions {
        GenerationOptions {
            context_tokens: 512,
            max_new_tokens: budget,
            seed: 17,
            sampling: Sampling::Greedy {},
            mtp_drafts: 3,
            eos_token_ids: vec![9],
            extra_stop_token_ids: vec![8],
            stop: Vec::new(),
            request_id: 42,
            cache_scope: None,
        }
    }

    #[test]
    fn callback_producer_obeys_every_tail_and_terminal_output_boundary() {
        for budget in 1..=7 {
            for terminal in [None, Some(8), Some(9)] {
                for terminal_at in 0..budget {
                    let options = options(budget);
                    let calls = Cell::new(0);
                    let mut visible = Vec::new();
                    let mut progress = Vec::new();
                    let (result, _) = collect_callback_tokens(
                        2,
                        &options,
                        10,
                        Instant::now(),
                        &mut |token| {
                            visible.push(token);
                            Ok(true)
                        },
                        Some(&mut |value| {
                            progress.push(value);
                            Ok(())
                        }),
                        &mut || Ok(false),
                        |output, control| {
                            for position in 0..budget {
                                control()?;
                                calls.set(calls.get() + 1);
                                let token = terminal.filter(|_| position == terminal_at).unwrap_or(1);
                                if !output(token)? {
                                    break;
                                }
                            }
                            Ok(MtpStats::default())
                        },
                    )
                    .unwrap();
                    let sampled = if terminal.is_some() {
                        terminal_at + 1
                    } else {
                        budget
                    } as usize;
                    assert_eq!(result.sampled_tokens, sampled);
                    assert_eq!(calls.get(), sampled);
                    assert_eq!(progress.len(), sampled);
                    assert_eq!(visible, result.generated_token_ids);
                    assert_eq!(visible.len(), sampled - usize::from(terminal.is_some()));
                    assert_eq!(
                        result.finish_reason,
                        match terminal {
                            Some(8) => FinishReason::StopToken,
                            Some(9) => FinishReason::Eos,
                            _ => FinishReason::Length,
                        }
                    );
                    assert!(progress.iter().all(|value| value.prefill_seconds.is_some()));
                }
            }
        }
    }

    #[test]
    fn callback_text_stop_and_failures_stop_the_producer_immediately() {
        for callback_error in [false, true] {
            let options = options(7);
            let calls = Cell::new(0);
            let result = collect_callback_tokens(
                2,
                &options,
                10,
                Instant::now(),
                &mut |_| {
                    if callback_error {
                        Err("stream_write_failed")
                    } else {
                        Ok(false)
                    }
                },
                None,
                &mut || Ok(false),
                |output, control| {
                    control()?;
                    calls.set(calls.get() + 1);
                    if output(1)? {
                        panic!("producer crossed text stop");
                    }
                    Ok(MtpStats::default())
                },
            );
            assert_eq!(calls.get(), 1);
            if callback_error {
                assert_eq!(result.unwrap_err(), "stream_write_failed");
            } else {
                let (result, _) = result.unwrap();
                assert_eq!(result.finish_reason, FinishReason::StopSequence);
                assert_eq!(result.sampled_tokens, 1);
            }
        }
    }

    #[test]
    fn callback_control_cancels_prompt_draft_verify_and_replay_without_more_output() {
        // These independent control points represent prompt work, drafting,
        // target verification, and checkpoint replay within one native round.
        for cancelled_at in 1..=4 {
            let options = options(7);
            let checks = Cell::new(0);
            let outputs = Cell::new(0);
            let result = collect_callback_tokens(
                2,
                &options,
                10,
                Instant::now(),
                &mut |_| {
                    outputs.set(outputs.get() + 1);
                    Ok(true)
                },
                None,
                &mut || {
                    checks.set(checks.get() + 1);
                    Ok(checks.get() == cancelled_at)
                },
                |output, control| {
                    for _ in 0..4 {
                        control()?;
                    }
                    output(1)?;
                    Ok(MtpStats::default())
                },
            );
            assert_eq!(result.unwrap_err(), "generation_cancelled");
            assert_eq!(checks.get(), cancelled_at);
            assert_eq!(outputs.get(), 0);
        }
    }

    #[test]
    fn callback_prompt_progress_is_content_free_and_failure_is_authored() {
        let options = options(2);
        let mut observations = Vec::new();
        let result = collect_callback_tokens(
            2,
            &options,
            10,
            Instant::now() - Duration::from_secs(2),
            &mut |_| Ok(true),
            Some(&mut |value| {
                observations.push(value);
                Err("progress_write_failed")
            }),
            &mut || Ok(false),
            |_, control| {
                control()?;
                panic!("producer ignored prompt callback failure")
            },
        );
        assert_eq!(result.unwrap_err(), "progress_write_failed");
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].sampled_tokens, 0);
        assert_eq!(observations[0].prefill_seconds, None);
    }

    #[test]
    fn callback_rejects_invalid_or_short_output_and_native_failures() {
        let options = options(2);
        for mode in 0..3 {
            let result = collect_callback_tokens(
                2,
                &options,
                10,
                Instant::now(),
                &mut |_| Ok(true),
                None,
                &mut || Ok(false),
                |output, _| {
                    if mode == 0 {
                        output(10)?;
                    }
                    if mode == 2 {
                        return Err(Error::Invalid("synthetic native failure"));
                    }
                    Ok(MtpStats::default())
                },
            );
            assert_eq!(
                result.unwrap_err(),
                if mode == 2 {
                    "native_generation_failed"
                } else {
                    "invalid_generation_output"
                }
            );
        }
    }
}

impl HybridBackend {
    pub(super) fn generate_edge0_inner(
        &mut self,
        input: &[u32],
        options: &GenerationOptions,
        mut on_text: Option<&mut TextCallback<'_>>,
        on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        let started = Instant::now();
        let tokenizer = self.model.tokenizer().clone();
        let mut decoder = tokenizer.decode_stream(false);
        let mut filter = StopFilter::new(options.stop.clone());
        let mut raw = String::new();
        let needs_text = on_text.is_some() || !options.stop.is_empty();
        let vocab_size = self.model.vocab_size();
        let budget = (options.context_tokens as usize - input.len()).min(options.max_new_tokens as usize);
        let cancellation = &mut self.cancellation;
        let mut cancelled = || match cancellation.as_mut() {
            Some(signal) => signal.check(options.request_id),
            None => Ok(false),
        };
        let mut rng = options.seed;
        // Edge0 verifies the complete vocabulary with the ordinary worker
        // filtering policy; the dense GPU sampler is a different contract.
        let mut sampler = mtp::ExactMtpSampler::new(&options.sampling, &mut rng, &mut self.probability_mapper);
        let model = &mut self.model;
        let (mut result, _) = collect_callback_tokens(
            input.len(),
            options,
            vocab_size,
            started,
            &mut |token| {
                if needs_text && let Some(text) = decoder.step(token).map_err(|_| "decoding_failed")? {
                    raw.push_str(&text);
                    let (text, stopped) = filter.push(&text);
                    if let Some(callback) = on_text.as_deref_mut() {
                        callback(&text)?;
                    }
                    return Ok(!stopped);
                }
                Ok(true)
            },
            on_progress,
            &mut cancelled,
            |on_output, control| {
                model.generate_edge0_mtp_controlled(
                    input,
                    budget,
                    options.mtp_drafts as usize,
                    &mut sampler,
                    on_output,
                    control,
                )
            },
        )?;
        if needs_text {
            let full = tokenizer.decode(&result.generated_token_ids, false).map_err(|_| "decoding_failed")?;
            let suffix = full.strip_prefix(&raw).ok_or("decoding_failed")?;
            let (tail, stopped) = filter.push(suffix);
            if stopped {
                result.finish_reason = crate::generation::FinishReason::StopSequence;
            }
            let tail = tail + &filter.finish();
            if let Some(callback) = on_text {
                callback(&tail)?;
            }
            result.text = Some(filter.text().to_owned());
        }
        Ok(result)
    }
}
