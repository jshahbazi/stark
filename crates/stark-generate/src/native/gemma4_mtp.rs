//! Staged Gemma assistant integration through the ordinary worker protocol.
use std::cell::Cell;

use super::*;

impl NativeBackend {
    pub(super) fn generate_gemma4_assistant(
        &mut self,
        input: &[u32],
        options: &GenerationOptions,
        mut on_text: Option<&mut TextCallback<'_>>,
        on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        let reused = self.prefix_cache_plan(input, options);
        let retained = self.gemma_prefix_cache.take().filter(|_| reused > 0);
        self.clear_prefix_cache();
        if input.is_empty() || options.mtp_drafts == 0 || options.mtp_drafts > 3 || options.context_tokens > 2048 {
            return Err("mtp_context_not_qualified");
        }
        let budget = (options.context_tokens as usize).saturating_sub(input.len()).min(options.max_new_tokens as usize);
        if budget == 0 {
            return Err("context_overflow");
        }
        // No silent target-only fallback beyond the staged head-position proof.
        // Qualification of the context's upper end remains a separate gate.
        if budget >= 3 && input.len().saturating_add(budget - 2) > 1792 {
            return Err("mtp_context_not_qualified");
        }
        let head = self.gemma4_assistant.as_ref().ok_or("mtp_not_supported")?;
        let NativeModel::Gemma4Vision(model) = &mut self.model else {
            return Err("mtp_not_supported");
        };
        let input_rows = u32::try_from(input.len()).map_err(|_| "context_overflow")?;
        admit_memory_bytes(model.assistant_generation_memory_bytes(
            head,
            input_rows,
            options.context_tokens,
            options.max_new_tokens,
        ))?;
        let started = Instant::now();
        let control_error = Cell::new(None);
        let cancellation = &mut self.cancellation;
        let mut control = || match cancellation.as_mut().map(|signal| signal.check(options.request_id)) {
            None | Some(Ok(false)) => false,
            Some(Ok(true)) => {
                control_error.set(Some("generation_cancelled"));
                true
            },
            Some(Err(error)) => {
                control_error.set(Some(error));
                true
            },
        };
        if control() {
            return Err(control_error.get().unwrap_or("generation_cancelled"));
        }
        let capture = if options.cache_scope.is_some() {
            prefix_cache::reusable_prefill_rows(input.len())
        } else {
            0
        };
        let frame = if let Some(entry) = retained {
            model.resume_assistant_prompt_prefix(head, entry.state.0, &input[reused..], capture, &mut control)
        } else {
            model.prefill_assistant_target(head, input, options.context_tokens, capture, &mut control)
        }
        .map_err(|_| control_error.get().unwrap_or("native_generation_failed"))?;
        if frame.position() as usize != input.len() {
            return Err("native_generation_failed");
        }
        let tokenizer = model.tokenizer().clone();
        let mut rng = options.seed;
        let mut mapper = crate::hybrid::ProbabilityMapper::default();
        let mut sampler = crate::hybrid::mtp::ExactMtpSampler::new(&options.sampling, &mut rng, &mut mapper);
        let first = sampler.sample_initial(frame.logits())?;
        let mut frame = Some(frame);
        let mut root = first;
        let mut pending = VecDeque::from([first]);
        let mut remaining = budget;
        let mut rounds = 0usize;
        let mut proposed = 0usize;
        let mut accepted = 0usize;
        let mut rounds_by_accepted_drafts = [0u64; 4];
        let mut verified_rows = 0usize;
        let mut repaired_rows = 0usize;
        let mut draft_seconds = 0.0;
        let mut verify_seconds = 0.0;
        let mut failed = false;
        let greedy = matches!(options.sampling, Sampling::Greedy {});
        let tokens = std::iter::from_fn(|| {
            if remaining == 0 || failed {
                return None;
            }
            if control() {
                failed = true;
                return Some(Err(control_error.get().unwrap_or("generation_cancelled")));
            }
            if pending.is_empty() {
                let drafts = (options.mtp_drafts as usize).min(remaining.saturating_sub(1));
                let Some(current) = frame.take() else {
                    failed = true;
                    return Some(Err("native_generation_failed"));
                };
                let round = if greedy {
                    // Keep exact ordinary target arithmetic across all rounds.
                    model.assistant_greedy_round(head, current, drafts, &mut control)
                } else {
                    // Exact-rejection sampling owns complete q distributions;
                    // ordinary target verification stays serial initially.
                    model.assistant_sampled_round(head, current, root, drafts, &mut sampler, &mut control)
                };
                let round = match round {
                    Ok(round) => round,
                    Err(_) => {
                        failed = true;
                        return Some(Err(control_error.get().unwrap_or("native_generation_failed")));
                    },
                };
                if round.verified.committed_tokens.first() != Some(&root)
                    || !record_accepted_prefix_round(&mut rounds_by_accepted_drafts, round.verified.accepted_drafts)
                {
                    failed = true;
                    return Some(Err("native_generation_failed"));
                }
                rounds += 1;
                proposed += round.proposed;
                accepted += round.verified.accepted_drafts;
                verified_rows += round.verified.verified_rows;
                repaired_rows += round.verified.repaired_rows;
                draft_seconds += round.draft_seconds;
                verify_seconds += round.verify_seconds;
                // The root was emitted before this round. Only accepted new
                // rows and the separate unprocessed carry enter the queue.
                pending.extend(round.verified.committed_tokens.into_iter().skip(1));
                root = round.verified.next_token;
                pending.push_back(root);
                frame = Some(round.verified.frame);
            }
            if control() {
                failed = true;
                return Some(Err(control_error.get().unwrap_or("generation_cancelled")));
            }
            remaining -= 1;
            pending.pop_front().map(|token| Ok(u64::from(token)))
        });
        let mut decoder = tokenizer.decode_stream(false);
        let mut filter = StopFilter::new(options.stop.clone());
        let mut raw = String::new();
        let needs_text = on_text.is_some() || !options.stop.is_empty();
        let result = collect_tokens_controlled(
            tokens,
            input.len(),
            options,
            self.vocab_size,
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
            &mut || Ok(false),
        );
        let mut result = native_result_after_cleanup(result, model.is_fenced())?;
        if control() {
            return Err(control_error.get().unwrap_or("generation_cancelled"));
        }
        result.reused_input_tokens = reused;
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
        // Never publish state advanced past EOS, stop strings or output cap.
        drop(pending);
        let retained = model
            .retain_assistant_prompt_prefix(head, frame.take().ok_or("native_generation_failed")?, &mut control)
            .map_err(|_| control_error.get().unwrap_or("native_generation_failed"))?;
        self.last_mtp = Some(serde_json::json!({
            "enabled":true,"drafts":options.mtp_drafts,"rounds":rounds,"proposed":proposed,
            "accepted":accepted,"rounds_by_accepted_drafts":rounds_by_accepted_drafts,
            "verified_rows":verified_rows,"repaired_rows":repaired_rows,
            "draft_seconds":draft_seconds,"verify_seconds":verify_seconds,
            "verification":GEMMA4_ASSISTANT_VERIFICATION_POLICY,
            "verification_method":if greedy {"greedy-serial-target"} else {"exact-rejection-serial"},
            "qualified_context_max":2048,"qualified_draft_position_max":1792,
            "resident_target_and_vision_bytes":model.resident_parameter_bytes(),
            "image_tower_resident":model.image_tower_resident(),
            "resident_assistant_bytes":head.resident_storage_bytes(),
            "assistant_peak_loading_bytes":head.peak_loading_bytes(),
        }));
        if let (Some(scope), Some(prefix)) = (options.cache_scope, retained)
            && prefix.position() == capture
            && capture > 0
        {
            self.gemma_pending_prefix = Some(prefix_cache::BoundaryEntry::new(
                (prefix, options.mtp_drafts, head.numerical_policy()),
                scope,
                options.context_tokens,
                &input[..capture],
            ));
        }
        Ok(result)
    }
}
