//! Tiny state/schedule control with an explicit dense synthetic target override.
//! The independent routed affine operator control covers expert splitting.
use std::io::Write;

use half::bf16;
use serde_json::json;
use stark_engine_macros::stark_test;

use crate::{
    backends::{
        common::{BufferRef, Context},
        metal::Metal,
    },
    encodable_block::sampling::SamplingMethod,
    engine::{
        Engine,
        language_model::{LanguageModel, state::LanguageModelState},
    },
};

fn collect(
    model: &LanguageModel<Metal>,
    state: &mut LanguageModelState<Metal>,
    input: &[u64],
    stochastic: bool,
) -> (Vec<u64>, Vec<f32>, usize) {
    let mut options = model.default_stream_options();
    options.sampling_method = if stochastic {
        SamplingMethod::Stochastic {
            temperature: Some(0.8),
            top_k: Some(32),
            top_p: Some(0.9),
            min_p: None,
            repetition_penalty: None,
            suffix_repetition_length: None,
        }
    } else {
        SamplingMethod::Greedy
    };
    let vocab = model.decoder.embedding().vocab_size() as usize;
    state.vision_prefill_logits_readback = Some(model.engine.context.create_buffer(vocab * 2).unwrap());
    let mut stream = model.stream_text(input, state, options).unwrap();
    let output = stream.by_ref().take(8).map(|token| token.unwrap()).collect::<Vec<_>>();
    assert_eq!(output.len(), 8);
    let passes = stream.metrics().num_prefill_forward_passes;
    drop(stream);
    let logits =
        state.vision_prefill_logits_readback.take().unwrap().copyout::<bf16>().into_iter().map(f32::from).collect();
    (output, logits, passes)
}

#[ignore]
#[stark_test]
fn bounded_prefill_toy_numerical_and_exact_prefix_state() {
    let artifact = std::env::var_os("STARK_DENSE_MTP_TOY").expect("set private tiny artifact");
    let output = std::env::var_os("STARK_TEST_BOUNDED_PREFILL_RECEIPT").expect("set private aggregate receipt");
    let engine = Engine::<Metal>::new().unwrap();
    let mut model = engine.load_language_model(std::path::Path::new(&artifact)).unwrap();
    assert_eq!(model.decoder.embedding().model_dim(), 128);
    assert_eq!(model.decoder.embedding().vocab_size(), 256);
    assert_eq!(model.decoder.ordinary_prefill_execution_cap(), None, "fixture must remain synthetically dense");
    let prompt = (0..1792u64)
        .map(|index| {
            let mut value = index + 0x9e3779b97f4a7c15;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
            1 + (value ^ (value >> 31)) % 255
        })
        .collect::<Vec<_>>();
    let mut cases = Vec::new();
    for split_last in [false, true] {
        model.decoder.validation_set_separate_prefill_last_token(split_last);
        for length in [1025usize, 1026, 1280, 1792] {
            for stochastic in [false, true] {
                model.decoder.validation_set_qwen_routed_prefill(false);
                let mut previous = model.create_empty_state(Some(4096), 0).unwrap();
                let (old_output, old_logits, _) = collect(&model, &mut previous, &prompt[..length], stochastic);
                model.decoder.validation_set_qwen_routed_prefill(true);
                let mut retained = model.create_empty_state(Some(4096), 0).unwrap();
                model.prepare_text_prefix_capture(&mut retained, 1024).unwrap();
                let (new_output, new_logits, passes) = collect(&model, &mut retained, &prompt[..length], stochastic);
                let difference = old_logits.iter().zip(&new_logits).map(|(a, b)| (a - b).abs()).collect::<Vec<_>>();
                let max_abs = difference.iter().copied().fold(0.0f32, f32::max);
                let squared: f64 = difference.iter().map(|value| f64::from(*value).powi(2)).sum();
                let reference: f64 = old_logits.iter().map(|value| f64::from(*value).powi(2)).sum();
                let relative_rms = (squared / reference.max(1e-30)).sqrt();
                assert!(old_logits.iter().chain(&new_logits).all(|value| value.is_finite()));
                assert!(max_abs <= 0.03125 && relative_rms <= 0.01, "tiny target batch-shape numerical gate failed");
                assert!(old_output == new_output, "tiny target batch-shape output changed");
                assert_eq!(retained.text_prefix_position(), Some(1024));
                assert!(passes >= length.div_ceil(256));
                if length == 1026 && split_last {
                    assert_eq!(passes, 6);
                }
                model.restore_text_prefix(&mut retained, 1024, Some(4096), 0).unwrap();
                let (reused, reused_logits, _) = collect(&model, &mut retained, &prompt[1024..length], stochastic);
                let mut fresh = model.create_empty_state(Some(4096), 0).unwrap();
                let (expected, expected_logits, _) = collect(&model, &mut fresh, &prompt[..length], stochastic);
                assert!(reused == expected, "bounded restored tokens differ from fresh");
                assert!(reused_logits == expected_logits, "bounded restored logits differ from fresh");
                cases.push(json!({"length":length,"split_last":split_last,"seed":0,"stochastic":stochastic,
                    "uncapped_output_match":true,"max_abs":max_abs,"relative_rms":relative_rms,
                    "snapshot":1024,"fresh_restored_exact":true,"prefill_submissions":passes}));
            }
        }
    }
    let mut failed = model.create_empty_state(Some(4096), 0).unwrap();
    model.prepare_text_prefix_capture(&mut failed, 1024).unwrap();
    failed.validation_fail_after_completed_prefill = Some(1024);
    assert!(model.stream_text(&prompt, &mut failed, model.default_stream_options()).is_err());
    assert!(failed.multimodal_failed && failed.text_prefix.is_none() && failed.text_prefix_capture_position.is_none());
    assert!(model.restore_text_prefix(&mut failed, 1024, Some(4096), 0).is_err());
    assert!(model.stream(&prompt[..2], &mut failed, model.default_stream_options()).is_err());
    let mut recovered = model.create_empty_state(Some(4096), 0).unwrap();
    collect(&model, &mut recovered, &prompt[..1026], false);
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut receipt = options.open(output).unwrap();
    serde_json::to_writer_pretty(
        &mut receipt,
        &json!({"completed":true,"synthetic_dense_override":true,
        "full_moe_qualified":false,"performance_eligible":false,"case_count":cases.len(),"cases":cases,
        "partial_failure_never_published":true,"failed_state_fenced":true,"fresh_state_recovers":true}),
    )
    .unwrap();
    receipt.write_all(b"\n").unwrap();
}
