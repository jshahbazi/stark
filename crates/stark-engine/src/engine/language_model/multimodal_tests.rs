//! Decoder integration fixtures are generated locally and removed when each test ends.
//! The feature oracle is an independent embedding-table lookup of control tokens.

use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    ops::Range,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use half::bf16;
use serde_json::{Value, json};
use stark_engine_macros::stark_test;

use super::{
    LanguageModel,
    multimodal::{ImageGrid, ImageTokenIds, MultimodalPrefill, ProjectedImage},
    stream::{LanguageModelStreamOptions, SamplingMethod},
};
use crate::{
    backends::common::{
        Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        gpu_types::trie::TrieNode,
    },
    data_type::DataType,
    encodable_block::{batch_topology::BatchTopology, decoder::EmbeddingReplacement, transformer::TransformerState},
    engine::Engine,
};

const VOCAB: u32 = 16;
const CAPACITY: u32 = 4096;
const SPECIAL_IDS: ImageTokenIds = ImageTokenIds {
    image: 1,
    vision_start: 2,
    vision_end: 3,
    video: 4,
};

struct TinyModel {
    directory: PathBuf,
    model_dim: u32,
    embeddings: Vec<bf16>,
}

impl Drop for TinyModel {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.directory).expect("remove private multimodal fixture");
    }
}

impl TinyModel {
    fn new(model_dim: u32) -> Self {
        Self::with_rope_mix(model_dim, false)
    }

    fn with_rope_mix(
        model_dim: u32,
        mixed_scalar_rope: bool,
    ) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "stark-native-fusion-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir(&directory).expect("create private multimodal fixture");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&directory, fs::Permissions::from_mode(0o700)).unwrap();
        }

        let norm = json!({
            "epsilon": 0.000001, "scale_offset": null, "upcast_mode": "full_layer",
            "subtract_mean": false, "has_scale": false, "has_biases": false,
        });
        let rope = json!({
            "type": "Qwen35MRoPEConfig", "base": 100.0, "head_dim": 4,
            "max_sequence_length": CAPACITY, "mrope_section": [1, 1, 0], "interleaved": true,
        });
        let mixer = json!({
            "type": "AttentionConfig", "qkvg_projection_config": {}, "out_projection_config": {},
            "query_norm_config": null, "key_norm_config": null, "num_heads": 2,
            "num_groups": 2, "head_dim": model_dim / 2, "is_causal": true,
            "scale": null, "sliding_window_size": null, "logit_soft_cap": null,
            "has_sinks": false, "has_qkvg_biases": false, "has_out_biases": false,
            "has_gate": false, "normalize_values": false, "is_kv_sharing": false,
        });
        let mlp = json!({
            "type": "DenseMLPConfig", "linear_config": {}, "activation": {"type": "SiLU", "alpha": 1.0},
            "has_up_biases": false, "has_down_biases": false,
            "gate_clipping": null, "up_clipping": null,
        });
        let layer = json!({
            "pre_mixer_norm_config": norm, "post_mixer_norm_config": null,
            "mixer_conv_config": null, "pre_mlp_norm_config": norm,
            "post_mlp_norm_config": null, "mlp_conv_config": null,
            "hidden_dim": null, "ple_config": null, "has_post_layer_scalar": false,
            "kv_source_layer_index": null, "rope_config": rope, "mixer_config": mixer, "mlp_config": mlp,
        });
        let transformer = json!({
            "model_dim": model_dim, "hidden_dim": model_dim,
            "output_norm_config": norm, "layer_configs": [layer],
        });
        let decoder = json!({
            "embedding_config": {
                "type": "TiedEmbeddingConfig", "input_scale": null,
                "logit_soft_cap": null, "logit_scale": null,
            },
            "vocab_size": VOCAB, "ple_model_config": null, "embedding_norm_config": null,
            "transformer_config": transformer,
        });
        let generation = json!({
            "stop_token_ids": [], "temperature": null, "top_k": null, "top_p": null,
            "min_p": null, "banned_tokens": null, "repetition_penalty": null,
            "presence_penalty": null, "frequency_penalty": null, "suffix_repetition_length": null,
        });
        let mut config = json!({
            "type": "LanguageModelConfig", "token_codec_config": {"type": "RawTextCodecConfig"},
            "generation_config": generation, "decoder_config": decoder,
        });
        if mixed_scalar_rope {
            let layers = config["decoder_config"]["transformer_config"]["layer_configs"].as_array_mut().unwrap();
            let mut scalar_layer = layers[0].clone();
            scalar_layer["rope_config"] = json!({
                "type": "UnscaledRoPEConfig", "base": 100.0, "head_dim": 4,
                "max_sequence_length": CAPACITY,
            });
            layers.push(scalar_layer);
        }
        fs::write(directory.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
        let vocab = (0..VOCAB).map(|id| (format!("t{id}"), id)).collect();
        let tokenizer = tokenizers::Tokenizer::new(
            tokenizers::models::wordlevel::WordLevel::builder().vocab(vocab).build().unwrap(),
        );
        tokenizer.save(directory.join("tokenizer.json"), false).unwrap();

        // Distinct signed rows prevent placeholder lookup and a missing copy from
        // accidentally satisfying the control comparison. Every weight is finite.
        let embeddings = (0..VOCAB)
            .flat_map(|row| {
                (0..model_dim).map(move |column| {
                    bf16::from_f32((((row * 13 + column * 7 + row * column * 3) % 29) as f32 - 14.0) / 16.0)
                })
            })
            .collect::<Vec<_>>();
        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        Self::matrix(
            &mut tensors,
            &mut metadata,
            "decoder.embedding.embedding",
            &[VOCAB, model_dim],
            embeddings.clone(),
            "input_output",
        );
        let projection = |rows: u32, scale: f32| {
            (0..rows)
                .flat_map(|row| {
                    (0..model_dim).map(move |column| {
                        bf16::from_f32(if row % model_dim == column {
                            scale
                        } else if (row + 1) % model_dim == column {
                            scale / 8.0
                        } else {
                            0.0
                        })
                    })
                })
                .collect::<Vec<_>>()
        };
        for layer in 0..if mixed_scalar_rope {
            2
        } else {
            1
        } {
            for (name, rows, scale) in [
                ("mixer.qkvg_projection", 3 * model_dim, 0.5),
                ("mixer.out_projection", model_dim, 0.75),
                ("mlp.up_projection", 2 * model_dim, 0.125),
                ("mlp.down_projection", model_dim, 0.125),
            ] {
                Self::matrix(
                    &mut tensors,
                    &mut metadata,
                    &format!("decoder.transformer.layers.{layer}.{name}.weights"),
                    &[rows, model_dim],
                    projection(rows, scale),
                    "output_input",
                );
            }
        }

        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), serde_json::to_value(metadata).unwrap());
        let mut payload = Vec::new();
        for (name, (shape, values)) in tensors {
            let start = payload.len();
            for value in values {
                payload.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            header.insert(name, json!({"dtype": "BF16", "shape": shape, "data_offsets": [start, payload.len()]}));
        }
        let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut weights = fs::File::create(directory.join("model.safetensors")).unwrap();
        weights.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        weights.write_all(&header).unwrap();
        weights.write_all(&payload).unwrap();
        Self {
            directory,
            model_dim,
            embeddings,
        }
    }

    fn matrix(
        tensors: &mut BTreeMap<String, (Vec<u32>, Vec<bf16>)>,
        metadata: &mut BTreeMap<String, String>,
        tree: &str,
        shape: &[u32],
        values: Vec<bf16>,
        layout: &str,
    ) {
        tensors.insert(format!("{tree}.weights"), (shape.to_vec(), values));
        metadata.insert(format!("{tree}.spec"), json!({"type": "FullPrecisionSpec", "layout": layout}).to_string());
    }

    fn prefill<B: Backend>(
        &self,
        model: &LanguageModel<B>,
        prefix_len: usize,
    ) -> MultimodalPrefill {
        let mut tokens = vec![5; prefix_len];
        tokens.extend([2, 1, 1, 1, 1, 3, 6, 7]);
        let features = (8usize..12)
            .flat_map(|id| {
                self.embeddings[id * self.model_dim as usize..(id + 1) * self.model_dim as usize].iter().copied()
            })
            .collect::<Box<[_]>>();
        MultimodalPrefill::try_new(
            tokens.into_boxed_slice(),
            ProjectedImage {
                model_identity: model.model_identity.clone(),
                model_dim: self.model_dim,
                data_type: DataType::BF16,
                grid: ImageGrid {
                    temporal: 1,
                    height: 4,
                    width: 4,
                    merge_size: 2,
                },
                token_start: prefix_len + 1,
                features,
            },
            SPECIAL_IDS,
            &model.model_identity,
            self.model_dim,
            DataType::BF16,
            VOCAB,
            CAPACITY,
        )
        .unwrap()
    }
}

fn control_tokens(input: &MultimodalPrefill) -> Vec<u64> {
    let mut tokens = input.tokens.to_vec();
    tokens[input.image.token_start..input.image.token_start + 4].copy_from_slice(&[8, 9, 10, 11]);
    tokens
}

fn encode<B: Backend>(
    model: &LanguageModel<B>,
    state: &mut TransformerState<B>,
    tokens: &[u64],
    positions: Option<&[[u32; 3]]>,
    replacement: Option<EmbeddingReplacement<'_>>,
) -> Vec<f32> {
    state.prepare(state.context_length(), tokens.len() as u32).unwrap();
    let mut commands = model.engine.context.create_command_buffer(None, None).unwrap();
    let ids = commands.allocate_constant_from_slice(&tokens.iter().map(|&id| id as u32).collect::<Vec<_>>()).unwrap();
    let nodes = (0..tokens.len() as u32)
        .map(|height| TrieNode {
            trie_start: height,
            trie_end: tokens.len() as u32,
            height,
        })
        .collect::<Vec<_>>();
    let topology = if let Some(positions) = positions {
        BatchTopology::with_rotary_positions(&nodes, true, positions).unwrap()
    } else {
        BatchTopology::new(&nodes, true)
    };
    let output = model
        .decoder
        .encode_with_features(
            &ids,
            replacement,
            &topology,
            Some((0..tokens.len() as u32).into()),
            None,
            state,
            &mut commands,
        )
        .unwrap();
    let logits = output.logits.unwrap();
    let mut readback = model.engine.context.create_buffer((&logits).size()).unwrap();
    commands.encode_copy(&logits, &mut readback);
    state.encode_accept(&(0..tokens.len() as u32).collect::<Vec<_>>(), &mut commands).unwrap();
    // CPU command finalization requires constant handles to have returned their
    // allocation ownership, just as the production stream scopes token IDs.
    drop(ids);
    commands.end_encoding().submit().wait_until_completed().unwrap();
    (&readback).as_slice::<bf16>().iter().map(|value| value.to_f32()).collect()
}

fn error_aggregates(
    actual: &[f32],
    expected: &[f32],
) -> (f32, f32, f32) {
    assert_eq!(actual.len(), expected.len());
    let mut maximum_absolute_error = 0.0f32;
    let mut squared_error = 0.0f32;
    let mut maximum_tolerance_ratio = 0.0f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite() && expected.is_finite(), "nonfinite decoder output at element {index}");
        // Different batch shapes can change FP32 reduction order before BF16
        // rounding. Permit one BF16 relative unit and a small cancellation floor.
        let tolerance = 0.003_906_25 + expected.abs() / 128.0;
        let absolute_error = (actual - expected).abs();
        maximum_absolute_error = maximum_absolute_error.max(absolute_error);
        squared_error += absolute_error * absolute_error;
        maximum_tolerance_ratio = maximum_tolerance_ratio.max(absolute_error / tolerance);
    }
    let rms_error = (squared_error / actual.len() as f32).sqrt();
    (maximum_absolute_error, rms_error, maximum_tolerance_ratio)
}

fn assert_close(
    actual: &[f32],
    expected: &[f32],
    context: &str,
) {
    let (maximum_absolute_error, rms_error, maximum_tolerance_ratio) = error_aggregates(actual, expected);
    assert!(
        maximum_tolerance_ratio <= 1.0,
        "{context}: maximum absolute error={maximum_absolute_error}, RMS error={rms_error}, maximum tolerance ratio={maximum_tolerance_ratio}"
    );
}

fn assert_identical(
    actual: &[f32],
    expected: &[f32],
) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite() && expected.is_finite(), "nonfinite decoder output at element {index}");
        assert!(actual.to_bits() == expected.to_bits(), "identical decoder input differs at element {index}");
    }
}

fn image_replacement(
    input: &MultimodalPrefill,
    range: Range<usize>,
) -> Option<EmbeddingReplacement<'_>> {
    let start = range.start.max(input.image.token_start);
    let end = range.end.min(input.image.token_start + 4);
    if start >= end {
        return None;
    }
    let width = input.image.model_dim as usize;
    Some(EmbeddingReplacement {
        token_start: (start - range.start) as u32,
        features: &input.image.features
            [(start - input.image.token_start) * width..(end - input.image.token_start) * width],
    })
}

fn fusion_and_cache_parity<B: Backend>(model_dim: u32) {
    let fixture = TinyModel::new(model_dim);
    let engine = Engine::<B>::new().unwrap();
    let model = engine.load_language_model(&fixture.directory).unwrap();
    // Eight rows keep both whole and chunked attention on single-pass kernels.
    // Nine rows select GEMM attention on Metal; that inherited dispatch change
    // is characterized separately below, without loosening fusion assertions.
    let input = fixture.prefill(&model, 0);
    let controls = control_tokens(&input);
    let empty = || model.decoder.create_empty_state(Some(64), &model.engine.context).unwrap();

    let mut fused = empty();
    let fused_logits =
        encode(&model, &mut fused, &input.tokens, Some(&input.positions), image_replacement(&input, 0..8));
    let mut control = empty();
    let control_logits = encode(&model, &mut control, &controls, Some(&input.positions), None);
    assert_identical(&fused_logits, &control_logits);

    let mut scalar_positions = empty();
    let scalar_logits = encode(&model, &mut scalar_positions, &controls, None, None);
    assert!(
        control_logits.iter().zip(&scalar_logits).any(|(a, b)| a.to_bits() != b.to_bits()),
        "fixture must detect ignored multimodal rotary coordinates"
    );

    let mut placeholders = empty();
    let placeholder_logits = encode(&model, &mut placeholders, &input.tokens, Some(&input.positions), None);
    assert!(
        fused_logits.iter().zip(&placeholder_logits).any(|(a, b)| (a - b).abs() > 0.05),
        "fixture must detect a missing image feature replacement"
    );

    let mut split = empty();
    let mut split_logits = Vec::new();
    let mut split_control = empty();
    let mut split_control_logits = Vec::new();
    // Both boundaries cut the image span so each invocation needs a different
    // feature offset. No runtime chunk-range helper is used by this oracle.
    for range in [0..2, 2..4, 4..8] {
        split_control_logits.extend(encode(
            &model,
            &mut split_control,
            &controls[range.clone()],
            Some(&input.positions[range.clone()]),
            None,
        ));
        split_logits.extend(encode(
            &model,
            &mut split,
            &input.tokens[range.clone()],
            Some(&input.positions[range.clone()]),
            image_replacement(&input, range),
        ));
    }
    assert_identical(&split_logits, &split_control_logits);
    assert_close(&split_control_logits, &control_logits, "chunked lookup versus whole lookup");
    assert_eq!(fused.context_length(), 8);
    assert_eq!(split.context_length(), 8);
    assert_eq!(input.position_delta, -2);
    assert!(input.positions.iter().flatten().all(|&position| position < fused.context_length()));

    fused.set_rotary_position_delta(Some(input.position_delta));
    split.set_rotary_position_delta(Some(input.position_delta));
    let continuation = [[6; 3]];
    let expected = encode(&model, &mut control, &[12], Some(&continuation), None);
    let split_expected = encode(&model, &mut split_control, &[12], Some(&continuation), None);
    let actual = encode(&model, &mut fused, &[12], None, None);
    let split_actual = encode(&model, &mut split, &[12], None, None);
    assert_identical(&actual, &expected);
    assert_identical(&split_actual, &split_expected);
    assert_close(&split_expected, &expected, "continuation after chunked lookup versus whole lookup");
    assert_eq!(fused.context_length(), 9);
    assert_eq!(split.context_length(), 9);

    // Characterize the existing nine-row GEMM versus single-pass numerical
    // difference using lookup-only controls. Image fusion must still produce
    // bit-identical output for every matching chunk shape and cached prefix.
    let input = fixture.prefill(&model, 1);
    let controls = control_tokens(&input);
    let mut whole_control = empty();
    let whole_logits = encode(&model, &mut whole_control, &controls, Some(&input.positions), None);
    let mut split_control = empty();
    let mut split_fused = empty();
    let mut chunked_logits = Vec::new();
    for range in [0..3, 3..5, 5..9] {
        let expected =
            encode(&model, &mut split_control, &controls[range.clone()], Some(&input.positions[range.clone()]), None);
        let actual = encode(
            &model,
            &mut split_fused,
            &input.tokens[range.clone()],
            Some(&input.positions[range.clone()]),
            image_replacement(&input, range),
        );
        assert_identical(&actual, &expected);
        chunked_logits.extend(expected);
    }
    let (maximum_absolute_error, rms_error, maximum_tolerance_ratio) = error_aggregates(&chunked_logits, &whole_logits);
    eprintln!(
        "{} lookup-only nine-row dispatch characterization: maximum absolute error={maximum_absolute_error}, RMS error={rms_error}, maximum strict tolerance ratio={maximum_tolerance_ratio}",
        B::NAME
    );

    // Ordinary scalar text positions reproduce the dispatch difference without
    // image replacement or explicit temporal/height/width coordinate input.
    let mut scalar_whole = empty();
    let scalar_whole_logits = encode(&model, &mut scalar_whole, &controls, None, None);
    let mut scalar_split = empty();
    let mut scalar_chunked_logits = Vec::new();
    for range in [0..3, 3..5, 5..9] {
        scalar_chunked_logits.extend(encode(&model, &mut scalar_split, &controls[range], None, None));
    }
    let (maximum_absolute_error, rms_error, maximum_tolerance_ratio) =
        error_aggregates(&scalar_chunked_logits, &scalar_whole_logits);
    eprintln!(
        "{} scalar-position lookup-only nine-row dispatch characterization: maximum absolute error={maximum_absolute_error}, RMS error={rms_error}, maximum strict tolerance ratio={maximum_tolerance_ratio}",
        B::NAME
    );
}

fn greedy_options<B: Backend>(model: &LanguageModel<B>) -> LanguageModelStreamOptions {
    let mut options = model.default_stream_options();
    options.sampling_method = SamplingMethod::Greedy;
    options
}

fn greedy(logits: &[f32]) -> u64 {
    let final_row = &logits[logits.len() - VOCAB as usize..];
    final_row
        .iter()
        .enumerate()
        .max_by(|(a_index, a), (b_index, b)| a.total_cmp(b).then(b_index.cmp(a_index)))
        .unwrap()
        .0 as u64
}

fn stream_crosses_image_boundary<B: Backend>(model_dim: u32) {
    let fixture = TinyModel::new(model_dim);
    let engine = Engine::<B>::new().unwrap();
    let model = engine.load_language_model(&fixture.directory).unwrap();
    let input = fixture.prefill(&model, 1021);
    assert_eq!(input.image.token_start, 1022);
    assert_eq!(input.tokens.len(), 1029);
    let controls = control_tokens(&input);
    let mut control = model.decoder.create_empty_state(Some(1100), &model.engine.context).unwrap();
    let mut control_logits = Vec::new();
    for range in [0..1024, 1024..1029] {
        control_logits = encode(&model, &mut control, &controls[range.clone()], Some(&input.positions[range]), None);
    }
    let expected_first = greedy(&control_logits);
    // Explicit arithmetic is the continuation oracle; it never consults the
    // runtime delta helper or multimodal chunk selection.
    let next_position = (input.tokens.len() as i64 + input.position_delta) as u32;
    let next_logits = encode(&model, &mut control, &[expected_first], Some(&[[next_position; 3]]), None);
    let expected_second = greedy(&next_logits);

    let mut state = model.create_empty_state(Some(1100), 0).unwrap();
    {
        let mut stream = model.stream_multimodal(&input, &mut state, greedy_options(&model)).unwrap();
        // Both physical chunks are encoded into the existing single prefill
        // submission; this metric counts submissions rather than decoder calls.
        assert_eq!(stream.metrics().num_prefill_forward_passes, 1);
        assert_eq!(stream.metrics().num_tokens_prefilled, 1029);
        assert_eq!(
            stream.next().unwrap().unwrap(),
            expected_first,
            "first multimodal output differs from lookup control"
        );
        assert_eq!(
            stream.next().unwrap().unwrap(),
            expected_second,
            "multimodal continuation differs from explicit-position control"
        );
    }
    // The stream queues one accepted input per returned output. Its pending
    // next token remains a token ID and does not affect the physical cache.
    assert_eq!(state.tokens().len(), 1031);
    assert_eq!(state.transformer_state.context_length(), 1031);
    assert_eq!(state.transformer_state.rotary_position_delta(), Some(-2));
}

#[cfg(backend = "cpu")]
#[stark_test]
fn native_image_stream_crosses_1024_token_feature_boundary_cpu() {
    stream_crosses_image_boundary::<crate::backends::cpu::Cpu>(128);
}

#[cfg(backend = "metal")]
#[stark_test]
fn native_image_stream_crosses_1024_token_feature_boundary_metal() {
    stream_crosses_image_boundary::<crate::backends::metal::Metal>(128);
}

#[cfg(backend = "cpu")]
#[stark_test]
fn native_image_preflight_rejects_foreign_and_nonfresh_state_before_mutation() {
    use crate::backends::cpu::Cpu;

    let fixture = TinyModel::new(128);
    let engine = Engine::<Cpu>::new().unwrap();
    let model = engine.load_language_model(&fixture.directory).unwrap();
    let foreign = engine.load_language_model(&fixture.directory).unwrap();
    let foreign_input = fixture.prefill(&foreign, 1);
    let input = fixture.prefill(&model, 1);
    let mut state = model.create_empty_state(Some(64), 0).unwrap();
    assert!(model.stream_multimodal(&foreign_input, &mut state, greedy_options(&model)).is_err());
    assert_eq!(state.tokens().len(), 0);
    assert_eq!(state.transformer_state.context_length(), 0);
    assert_eq!(state.transformer_state.rotary_position_delta(), None);

    drop(model.stream(&[5], &mut state, greedy_options(&model)).unwrap());
    let before = (state.tokens.len(), state.transformer_state.context_length(), state.last_output_token);
    assert!(model.stream_multimodal(&input, &mut state, greedy_options(&model)).is_err());
    assert_eq!((state.tokens.len(), state.transformer_state.context_length(), state.last_output_token), before);
    assert_eq!(state.transformer_state.rotary_position_delta(), None);

    let mut foreign_state = foreign.create_empty_state(Some(64), 0).unwrap();
    assert!(model.stream_multimodal(&input, &mut foreign_state, greedy_options(&model)).is_err());
    assert_eq!(foreign_state.tokens().len(), 0);
    assert_eq!(foreign_state.transformer_state.context_length(), 0);
}

#[cfg(backend = "cpu")]
#[stark_test]
fn native_image_preflight_rejects_mixed_scalar_and_multimodal_rope() {
    use crate::backends::cpu::Cpu;

    let fixture = TinyModel::with_rope_mix(128, true);
    let engine = Engine::<Cpu>::new().unwrap();
    let model = engine.load_language_model(&fixture.directory).unwrap();
    let input = fixture.prefill(&model, 1);
    let mut state = model.create_empty_state(Some(64), 0).unwrap();
    assert!(model.stream_multimodal(&input, &mut state, greedy_options(&model)).is_err());
    assert_eq!(state.tokens().len(), 0);
    assert_eq!(state.transformer_state.context_length(), 0);
    assert_eq!(state.transformer_state.rotary_position_delta(), None);
}

#[cfg(backend = "cpu")]
#[stark_test]
fn native_image_stream_reserves_output_headroom_and_stops_at_context_capacity() {
    use super::stream::LanguageModelStreamError;
    use crate::backends::cpu::Cpu;

    let fixture = TinyModel::new(128);
    let engine = Engine::<Cpu>::new().unwrap();
    let model = engine.load_language_model(&fixture.directory).unwrap();
    let input = fixture.prefill(&model, 1);
    assert_eq!(input.tokens.len(), 9);

    // A prompt filling the complete budget has no legal output slot. Reject
    // before mutating or fencing the request, rather than sampling past it.
    let mut full = model.create_empty_state(Some(9), 0).unwrap();
    assert!(matches!(
        model.stream_multimodal(&input, &mut full, greedy_options(&model)),
        Err(LanguageModelStreamError::ContextOverflow)
    ));
    assert!(full.tokens().is_empty());
    assert_eq!(full.transformer_state.context_length(), 0);
    assert_eq!(full.transformer_state.rotary_position_delta(), None);
    assert_eq!(full.last_output_token, None);
    assert!(!full.multimodal_failed);

    // The native stream encodes one lookahead step. Its extra sampled token
    // must stay unreturned and be discarded once prompt + output reaches ten.
    let mut bounded = model.create_empty_state(Some(10), 0).unwrap();
    {
        let mut stream = model.stream_multimodal(&input, &mut bounded, greedy_options(&model)).unwrap();
        assert!(stream.next().unwrap().unwrap() < u64::from(VOCAB));
        assert!(stream.next().is_none());
        assert!(stream.next().is_none());
        assert_eq!(stream.metrics().num_tokens_returned, 1);
    }
    assert_eq!(bounded.tokens().len(), 10);
    assert_eq!(bounded.transformer_state.context_length(), 10);
    assert_eq!(bounded.transformer_state.rotary_position_delta(), Some(-2));
    assert_eq!(bounded.last_output_token, None);
    assert!(!bounded.multimodal_failed);
}

#[cfg(backend = "cpu")]
#[stark_test]
fn native_image_setup_failure_fences_state_and_fresh_request_recovers() {
    use super::stream::LanguageModelStreamError;
    use crate::{backends::cpu::Cpu, encodable_block::decoder::DecoderError};

    let fixture = TinyModel::new(128);
    let engine = Engine::<Cpu>::new().unwrap();
    let model = engine.load_language_model(&fixture.directory).unwrap();
    let valid = fixture.prefill(&model, 1);
    let mut faulty = fixture.prefill(&model, 1);
    // Inject an internal fault after the typed constructor has validated the
    // features. External inputs cannot normally bypass that finite-value check.
    // Stream preflight accepts the valid ownership/shape; decoder validation
    // then fails after setup has marked this request state as in progress.
    faulty.image.features[0] = bf16::from_f32(f32::NAN);
    let mut failed = model.create_empty_state(Some(64), 0).unwrap();
    assert!(matches!(
        model.stream_multimodal(&faulty, &mut failed, greedy_options(&model)),
        Err(LanguageModelStreamError::Decoder(DecoderError::InvalidInput(_)))
    ));
    assert!(failed.multimodal_failed);
    assert_eq!(failed.transformer_state.rotary_position_delta(), None);
    assert_eq!(failed.last_output_token, None);
    assert_eq!(failed.tokens().len(), 0);
    assert_eq!(failed.transformer_state.context_length(), 0);

    assert!(matches!(
        model.stream_multimodal(&valid, &mut failed, greedy_options(&model)),
        Err(LanguageModelStreamError::InvalidMultimodal)
    ));
    assert!(matches!(
        model.stream(&[5], &mut failed, greedy_options(&model)),
        Err(LanguageModelStreamError::InvalidMultimodal)
    ));
    assert!(failed.multimodal_failed);
    assert_eq!(failed.transformer_state.rotary_position_delta(), None);
    assert_eq!(failed.last_output_token, None);

    let mut fresh = model.create_empty_state(Some(64), 0).unwrap();
    {
        let mut stream = model.stream_multimodal(&valid, &mut fresh, greedy_options(&model)).unwrap();
        assert!(stream.next().unwrap().unwrap() < u64::from(VOCAB));
    }
    assert!(!fresh.multimodal_failed);
    assert_eq!(fresh.tokens().len(), 10);
    assert_eq!(fresh.transformer_state.context_length(), 10);
    assert_eq!(fresh.transformer_state.rotary_position_delta(), Some(-2));
    assert!(fresh.last_output_token.is_some());
}

#[cfg(backend = "cpu")]
#[stark_test]
fn native_image_fusion_matches_lookup_and_chunked_decoder_cpu() {
    fusion_and_cache_parity::<crate::backends::cpu::Cpu>(128);
}

#[cfg(backend = "metal")]
#[stark_test]
fn native_image_fusion_matches_lookup_and_chunked_decoder_metal() {
    // Metal attention instantiates head widths 64 and larger. Rotate the same
    // four-channel prefix while exercising its actual GPU feature copy path.
    fusion_and_cache_parity::<crate::backends::metal::Metal>(128);
}
