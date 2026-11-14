//! Private toy-artifact qualification. Never prints prompts, tokens or logits.
use stark_engine_macros::stark_test;

use super::{LanguageModel, dense_mtp::DenseMtpSession};
use crate::{
    backends::{
        common::{BufferRef, Context},
        metal::Metal,
    },
    engine::{
        Engine,
        hybrid::{MtpSampler, Result},
    },
};

struct Sampler {
    seed: u64,
    stochastic: bool,
}

fn argmax(values: &[f32]) -> u32 {
    values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0))).unwrap().0 as u32
}

impl MtpSampler for Sampler {
    fn begin_round(&mut self) {}
    fn draft(
        &mut self,
        values: &[f32],
    ) -> Result<u32> {
        Ok(argmax(values))
    }
    fn target(
        &mut self,
        values: &[f32],
        _: Option<u32>,
    ) -> Result<u32> {
        if !self.stochastic {
            return Ok(argmax(values));
        }
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 7;
        self.seed ^= self.seed << 17;
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights = values.iter().map(|&v| f64::from(v - max).exp()).collect::<Vec<_>>();
        let mut draw = (self.seed >> 11) as f64 / ((1u64 << 53) as f64) * weights.iter().sum::<f64>();
        for (index, weight) in weights.iter().enumerate() {
            draw -= weight;
            if draw < 0.0 {
                return Ok(index as u32);
            }
        }
        Ok(values.len() as u32 - 1)
    }
}

fn collect(
    model: &LanguageModel<Metal>,
    session: &mut DenseMtpSession<Metal>,
    drafts: usize,
    stochastic: bool,
) -> Vec<u32> {
    let mut sampler = Sampler {
        seed: 17,
        stochastic,
    };
    let mut current = sampler.target(session.logits(), None).unwrap();
    let mut output = Vec::new();
    while output.len() < 16 {
        output.push(current);
        if output.len() == 16 {
            break;
        }
        let round = model.dense_mtp_round(session, current, drafts.min(15 - output.len()), &mut sampler).unwrap();
        output.extend(round.accepted_drafts);
        current = round.next_token;
    }
    output
}

fn collect_ordinary(
    model: &LanguageModel<Metal>,
    state: &mut super::state::LanguageModelState<Metal>,
    input: &[u32],
    stochastic: bool,
) -> Vec<u32> {
    let mut options = model.default_stream_options();
    options.sampling_method = if stochastic {
        super::stream::SamplingMethod::Stochastic {
            temperature: Some(0.8),
            top_k: Some(32),
            top_p: Some(0.9),
            min_p: None,
            repetition_penalty: None,
            suffix_repetition_length: None,
        }
    } else {
        super::stream::SamplingMethod::Greedy
    };
    let input = input.iter().copied().map(u64::from).collect::<Vec<_>>();
    model.stream_text(&input, state, options).unwrap().take(16).map(|token| token.unwrap() as u32).collect()
}

#[stark_test]
// Requires explicitly generated private dense-MTP toy artifact.
#[ignore]
fn dense_mtp_toy_greedy_and_prefix_metal() {
    let path = std::path::PathBuf::from(std::env::var_os("STARK_DENSE_MTP_TOY").expect("set private toy path"));
    let engine = Engine::<Metal>::new().unwrap();
    let model = engine.load_language_model(&path).unwrap();
    assert!(model.has_dense_mtp());
    let ordinary_bytes = model.ordinary_text_generation_memory_bytes(1100, 4096, 16).unwrap();
    let predictor_bytes = model.generation_memory_bytes(1100, 4096, 16).unwrap();
    assert!(ordinary_bytes < predictor_bytes, "ordinary requests must not allocate inactive predictor state");
    assert!(model.generation_memory_bytes_with_text_prefix(1100, 4096, 16).unwrap() > ordinary_bytes);
    let short = (0..32).map(|i| 1 + i % 31).collect::<Vec<u32>>();
    let baseline = collect(&model, &mut model.prefill_dense_mtp(&short, 4096).unwrap(), 0, false);
    // One prefill row also fixes readout GEMV geometry on both routes.
    let initial = model.prefill_dense_mtp(&short[..1], 4096).unwrap();
    let initial_logits = initial.logits().to_vec();
    let mut ordinary_state = model.create_empty_state(Some(4096), 17).unwrap();
    ordinary_state.vision_prefill_logits_readback = Some(engine.context.create_buffer(256 * 2).unwrap());
    let mut ordinary_options = model.default_stream_options();
    ordinary_options.sampling_method = super::stream::SamplingMethod::Greedy;
    let host_ids = short[..1].iter().copied().map(u64::from).collect::<Vec<_>>();
    // The first output qualifies the shared embedding-input contract. Later
    // ordinary decode uses fused full-accept recurrence; this session uses
    // TreeVerify/StateAdvance, whose floating-point ordering differs.
    let ordinary =
        model.stream(&host_ids, &mut ordinary_state, ordinary_options).unwrap().next().unwrap().unwrap() as u32;
    assert!(ordinary == argmax(&initial_logits), "ordinary stream and dense target device-input contracts differ");
    let ordinary_logits = ordinary_state.vision_prefill_logits_readback.take().unwrap().copyout::<half::bf16>();
    assert!(
        ordinary_logits.iter().copied().map(f32::from).eq(initial_logits),
        "ordinary and dense full-vocabulary prefill logits differ"
    );
    for drafts in 1..=3 {
        let result = collect(&model, &mut model.prefill_dense_mtp(&short, 4096).unwrap(), drafts, false);
        assert!(result == baseline, "greedy target and MTP token hashes differ");
    }
    let prompt = (0..1100).map(|i| 1 + i % 31).collect::<Vec<u32>>();
    let mut uncached = model.prefill_dense_mtp_with_prefix_controlled(&prompt, 4096, false, &mut || false).unwrap();
    assert_eq!(uncached.prefix_position(), None);
    assert_eq!(uncached.prefix_payload_bytes(), 0);
    let mut cached = model.prefill_dense_mtp_with_prefix_controlled(&prompt, 4096, true, &mut || false).unwrap();
    assert_eq!(cached.prefix_position(), Some(1024));
    assert!(uncached.logits() == cached.logits(), "snapshot control changed prefill logits");
    assert!(
        collect(&model, &mut uncached, 2, false) == collect(&model, &mut cached, 2, false),
        "snapshot control changed dense MTP continuation"
    );
    let mut followup = prompt.clone();
    followup.extend([3, 5, 7]);
    for stochastic in [false, true] {
        for drafts in 0..=3 {
            let mut retained = model.prefill_dense_mtp(&prompt, 4096).unwrap();
            assert_eq!(retained.prefix_position(), Some(1024));
            assert!(retained.prefix_payload_bytes() > 0);
            collect(&model, &mut retained, drafts, stochastic);
            model.restore_dense_mtp_prefix(&mut retained, 1024).unwrap();
            assert_eq!(retained.position(), 1024);
            model.continue_prefill_dense_mtp_controlled(&mut retained, &followup[1024..], 1024, &mut || false).unwrap();
            let reused = collect(&model, &mut retained, drafts, stochastic);
            let fresh = collect(&model, &mut model.prefill_dense_mtp(&followup, 4096).unwrap(), drafts, stochastic);
            assert!(reused == fresh, "restored and fresh target token hashes differ");
        }
        // MTP0 now uses the ordinary stream, rather than a zero-draft
        // predictor session. Exercise multi-token decode and its own snapshot.
        let mut ordinary = model.create_empty_state(Some(4096), 17).unwrap();
        model.prepare_text_prefix_capture(&mut ordinary, 1024).unwrap();
        collect_ordinary(&model, &mut ordinary, &prompt, stochastic);
        assert_eq!(ordinary.text_prefix_position(), Some(1024));
        model.restore_text_prefix(&mut ordinary, 1024, Some(4096), 17).unwrap();
        model.prepare_text_prefix_capture(&mut ordinary, 1024).unwrap();
        let reused = collect_ordinary(&model, &mut ordinary, &followup[1024..], stochastic);
        let mut fresh = model.create_empty_state(Some(4096), 17).unwrap();
        let expected = collect_ordinary(&model, &mut fresh, &followup, stochastic);
        assert!(reused == expected, "headed ordinary stream fresh/restored output differs");
    }
    let other = engine.load_language_model(&path).unwrap();
    let mut session = model.prefill_dense_mtp(&prompt, 4096).unwrap();
    assert!(other.restore_dense_mtp_prefix(&mut session, 1024).is_err());
    let mut called = false;
    let cancelled = model.continue_prefill_dense_mtp_controlled(&mut session, &short, 0, &mut || {
        called = true;
        true
    });
    // This session is not aligned after the original prefill, so it is refused
    // before callback or GPU dispatch; restoring its boundary then tests cancel.
    assert!(cancelled.is_err());
    assert!(!called);
    model.restore_dense_mtp_prefix(&mut session, 1024).unwrap();
    assert!(model.continue_prefill_dense_mtp_controlled(&mut session, &short, 0, &mut || true).is_err());
    assert!(model.restore_dense_mtp_prefix(&mut session, 1024).is_err());
}

#[stark_test]
#[ignore]
fn dense_mtp_toy_admission_selects_active_path_and_credits_retained_owners() {
    let path = std::path::PathBuf::from(std::env::var_os("STARK_DENSE_MTP_TOY").expect("set private toy path"));
    let engine = Engine::<Metal>::new().unwrap();
    let model = engine.load_language_model(&path).unwrap();
    assert!(model.has_dense_mtp());
    let plain = model.generation_memory_bytes_for_request(1101, 4096, 16, false).unwrap();
    let enabled = model.generation_memory_bytes_for_request(1101, 4096, 16, true).unwrap();
    assert!(plain < enabled);
    let target_limit = model.decoder.persistent_state_bytes(2050, 1025, 2049, false).unwrap();
    assert_eq!(target_limit, model.decoder.memory_estimate_with_prefix(2050, 1025, 2049, false).unwrap().state_bytes);
    let head = model.dense_mtp.as_ref().unwrap();
    assert_eq!(
        head.persistent_state_bytes(2050, 1025, 2049).unwrap(),
        head.memory_estimate_with_prefix(2050, 1025, 2049, false).unwrap().state_bytes
    );
    assert_eq!(
        head.persistent_state_bytes(2050, 1, 2049).unwrap(),
        head.memory_estimate_with_prefix(2050, 1, 2049, false).unwrap().state_bytes
    );
    let prompt = vec![5; 1100];
    let mut retained = model.prefill_dense_mtp(&prompt, 4096).unwrap();
    assert_eq!(retained.prefix_position(), Some(1024));
    let warm = model.generation_memory_bytes_reusing_dense_mtp(&retained, 1101, 4096, 16, 1024).unwrap();
    assert!(warm < enabled);
    assert!(model.generation_memory_bytes_reusing_dense_mtp(&retained, 1101, 2048, 16, 1024).is_none());
    let foreign = engine.load_language_model(&path).unwrap();
    assert!(foreign.generation_memory_bytes_reusing_dense_mtp(&retained, 1101, 4096, 16, 1024).is_none());
    model.restore_dense_mtp_prefix(&mut retained, 1024).unwrap();
    assert!(model.continue_prefill_dense_mtp_controlled(&mut retained, &[5, 5], 0, &mut || true).is_err());
    assert!(model.generation_memory_bytes_reusing_dense_mtp(&retained, 1101, 4096, 16, 1024).is_none());
}
