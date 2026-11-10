//! Source-local fixtures verify plain-text snapshots on spatial-capable decoders.
//! Only counts and invariant labels appear in failures; token/logit payloads stay private.

use super::*;
use crate::engine::language_model::state::LanguageModelState;

fn prompt(length: usize) -> Vec<u64> {
    (0..length).map(|index| 5 + (index * 7 % 11) as u64).collect()
}

fn hybrid_fixture() -> TinyModel {
    let fixture = TinyModel::with_rope_mix(128, true);
    let config_path = fixture.directory.join("config.json");
    let mut config: Value = serde_json::from_slice(&fs::read(&config_path).unwrap()).unwrap();
    let layers = config["decoder_config"]["transformer_config"]["layer_configs"].as_array_mut().unwrap();
    // Retain spatial-capable full attention after a recurrent mixer. A
    // complete prefix must restore both its old KV and the DeltaNet state.
    layers[1]["rope_config"] = layers[0]["rope_config"].clone();
    layers[0]["rope_config"] = Value::Null;
    let mut norm = layers[0]["pre_mixer_norm_config"].clone();
    norm["has_scale"] = json!(true);
    layers[0]["mixer_config"] = json!({
        "type": "DeltaNetConfig", "in_proj_config": {}, "out_proj_config": {},
        "conv_config": {"has_biases": false}, "norm_config": norm,
        "reference_bf16": true, "num_heads": 1, "num_groups": 1,
        "head_dim": 128, "value_head_dim": 128, "kernel_size": 4,
    });
    fs::write(config_path, serde_json::to_vec(&config).unwrap()).unwrap();

    // Reuse the independent fixture's embedding/MLP/attention tensors. Replace
    // only the first mixer's weights with deterministic nonzero projections,
    // convolution, decay and normalization tensors in their native dtypes.
    let weights_path = fixture.directory.join("model.safetensors");
    let original = fs::read(&weights_path).unwrap();
    let header_len = u64::from_le_bytes(original[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&original[8..8 + header_len]).unwrap();
    let payload = &original[8 + header_len..];
    let stem = "decoder.transformer.layers.0.mixer.";
    let mut metadata = header["__metadata__"].as_object().unwrap().clone();
    metadata.retain(|name, _| !name.starts_with(stem));
    let mut tensors = BTreeMap::<String, (String, Value, Vec<u8>)>::new();
    for (name, info) in header.as_object().unwrap() {
        if name == "__metadata__" || name.starts_with(stem) {
            continue;
        }
        let start = info["data_offsets"][0].as_u64().unwrap() as usize;
        let end = info["data_offsets"][1].as_u64().unwrap() as usize;
        tensors.insert(
            name.clone(),
            (info["dtype"].as_str().unwrap().into(), info["shape"].clone(), payload[start..end].to_vec()),
        );
    }
    for (name, rows, scale) in [("in_proj", 514, 0.125f32), ("out_proj", 128, 0.25)] {
        let values = (0..rows)
            .flat_map(|row| {
                (0..128).flat_map(move |column| {
                    bf16::from_f32(if row % 128 == column {
                        scale
                    } else {
                        0.0
                    })
                    .to_bits()
                    .to_le_bytes()
                })
            })
            .collect();
        let tree = format!("{stem}{name}.weights");
        tensors.insert(format!("{tree}.weights"), ("BF16".into(), json!([rows, 128]), values));
        metadata.insert(
            format!("{tree}.spec"),
            json!(json!({"type": "FullPrecisionSpec", "layout": "output_input"}).to_string()),
        );
    }
    let floats = |values: Vec<f32>| values.into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>();
    for (name, shape, values) in [
        ("conv.weights", json!([384, 4]), (0..384).flat_map(|_| [0.05, 0.1, 0.2, 0.65]).collect()),
        ("a_log", json!([1]), vec![0.0]),
        ("dt_bias", json!([1]), vec![-1.0]),
        ("norm.scales", json!([128]), vec![1.0; 128]),
    ] {
        tensors.insert(format!("{stem}{name}"), ("F32".into(), shape, floats(values)));
    }
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), Value::Object(metadata));
    let mut payload = Vec::new();
    for (name, (dtype, shape, bytes)) in tensors {
        let start = payload.len();
        payload.extend(bytes);
        header.insert(name, json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]}));
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut weights = fs::File::create(weights_path).unwrap();
    weights.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    weights.write_all(&header).unwrap();
    weights.write_all(&payload).unwrap();
    fixture
}

fn continuation<B: Backend>(
    model: &LanguageModel<B>,
    state: &mut LanguageModelState<B>,
    input: &[u64],
    expected_prefill: usize,
    stochastic: bool,
) -> (Vec<u64>, Vec<f32>) {
    state.vision_prefill_logits_readback = Some(model.engine.context.create_buffer(VOCAB as usize * 2).unwrap());
    let mut output = Vec::new();
    {
        let mut options = greedy_options(model);
        if stochastic {
            options.sampling_method = SamplingMethod::Stochastic {
                temperature: Some(0.8),
                top_k: Some(8),
                top_p: None,
                min_p: None,
                repetition_penalty: None,
                suffix_repetition_length: None,
            };
        }
        let mut stream = model.stream_text(input, state, options).unwrap();
        assert_eq!(stream.metrics().num_tokens_prefilled, expected_prefill);
        for _ in 0..4 {
            output.push(stream.next().unwrap().unwrap());
        }
    }
    let readback = state.vision_prefill_logits_readback.take().unwrap();
    let logits = (&readback).as_slice::<bf16>().iter().map(|value| value.to_f32()).collect();
    (output, logits)
}

fn text_snapshot_matches_fresh<B: Backend>(
    hybrid: bool,
    ring: bool,
) {
    let fixture = if hybrid {
        hybrid_fixture()
    } else {
        TinyModel::with_rope_mix(128, ring)
    };
    if ring {
        let path = fixture.directory.join("config.json");
        let mut config: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let layers = config["decoder_config"]["transformer_config"]["layer_configs"].as_array_mut().unwrap();
        for layer in layers.iter_mut() {
            layer["mixer_config"]["sliding_window_size"] = json!(256);
        }
        layers[1]["kv_source_layer_index"] = json!(0);
        layers[1]["mixer_config"]["is_kv_sharing"] = json!(true);
        layers[1]["rope_config"] = layers[0]["rope_config"].clone();
        fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();

        // A shared-KV consumer projects only Q. Retain the original synthetic
        // Q rows and every other payload byte; the custom loader admits holes.
        let path = fixture.directory.join("model.safetensors");
        let mut weights = fs::read(&path).unwrap();
        let header_len = u64::from_le_bytes(weights[..8].try_into().unwrap()) as usize;
        let mut header: Value = serde_json::from_slice(&weights[8..8 + header_len]).unwrap();
        let q = &mut header["decoder.transformer.layers.1.mixer.qkvg_projection.weights.weights"];
        let begin = q["data_offsets"][0].as_u64().unwrap();
        q["shape"] = json!([128, 128]);
        q["data_offsets"][1] = json!(begin + 128 * 128 * 2);
        let replacement = serde_json::to_vec(&header).unwrap();
        assert!(replacement.len() <= header_len);
        weights[8..8 + header_len].fill(b' ');
        weights[8..8 + replacement.len()].copy_from_slice(&replacement);
        fs::write(path, weights).unwrap();
    }
    let engine = Engine::<B>::new().unwrap();
    let model = engine.load_language_model(&fixture.directory).unwrap();
    // Spatial rotary capability and rings block simple rewind. A snapshot
    // restores overwritten ring payload as well as ordinary text positions.
    assert!(!model.prefix_reuse_supported());
    assert!(model.text_prefix_snapshot_supported());
    if ring {
        assert_eq!(model.context_memory_estimate(CAPACITY, 1280).unwrap().prefix_snapshot_bytes, 256 * 128 * 2 * 2);
    }
    let original = prompt(1280);
    let mut follow_up = prompt(1400);
    for (index, token) in follow_up[1024..].iter_mut().enumerate() {
        *token = 5 + (index * 3 % 11) as u64;
    }
    let mut cached = model.create_empty_state(Some(CAPACITY), 23).unwrap();
    assert!(cached.text_prefix_position().is_none());
    model.prepare_text_prefix_capture(&mut cached, 1024).unwrap();
    continuation(&model, &mut cached, &original, original.len(), false);
    assert_eq!(cached.text_prefix_position(), Some(1024));

    // Mismatched guards must leave a valid saved prefix usable afterwards.
    assert!(matches!(
        model.restore_text_prefix(&mut cached, 1024, Some(CAPACITY - 1), 71),
        Err(crate::engine::language_model::state::TextPrefixRestoreError::Invalid)
    ));
    assert!(matches!(
        model.restore_text_prefix(&mut cached, 1023, Some(CAPACITY), 71),
        Err(crate::engine::language_model::state::TextPrefixRestoreError::Invalid)
    ));
    let foreign = engine.load_language_model(&fixture.directory).unwrap();
    assert!(matches!(
        foreign.restore_text_prefix(&mut cached, 1024, Some(CAPACITY), 71),
        Err(crate::engine::language_model::state::TextPrefixRestoreError::Invalid)
    ));
    assert_eq!(cached.text_prefix_position(), Some(1024));
    model.restore_text_prefix(&mut cached, 1024, Some(CAPACITY), 71).unwrap();
    model.prepare_text_prefix_capture(&mut cached, 1024).unwrap();
    assert_eq!(cached.text_prefix_position(), Some(1024));
    model.preflight_restored_text_stream(&follow_up[1024..], &cached, &greedy_options(&model)).unwrap();
    assert_eq!(cached.tokens().len(), 1024);
    let (actual, actual_logits) = continuation(&model, &mut cached, &follow_up[1024..], follow_up.len() - 1024, false);
    let mut fresh = model.create_empty_state(Some(CAPACITY), 71).unwrap();
    let (expected, expected_logits) = continuation(&model, &mut fresh, &follow_up, follow_up.len(), false);
    assert!(actual == expected, "restored text continuation differs from fresh state");
    assert_identical(&actual_logits, &expected_logits);

    // Another completed suffix must not overwrite the retained boundary.
    model.restore_text_prefix(&mut cached, 1024, Some(CAPACITY), 71).unwrap();
    let (repeated, repeated_logits) =
        continuation(&model, &mut cached, &follow_up[1024..], follow_up.len() - 1024, false);
    assert!(repeated == expected, "repeated snapshot restoration differs");
    assert_identical(&repeated_logits, &expected_logits);

    // Restoration reseeds the per-request sampler at its absolute token
    // position; a previous greedy request must not affect stochastic draws.
    model.restore_text_prefix(&mut cached, 1024, Some(CAPACITY), 97).unwrap();
    let (actual_sampled, _) = continuation(&model, &mut cached, &follow_up[1024..], follow_up.len() - 1024, true);
    let mut fresh_sampled = model.create_empty_state(Some(CAPACITY), 97).unwrap();
    let (expected_sampled, _) = continuation(&model, &mut fresh_sampled, &follow_up, follow_up.len(), true);
    assert!(actual_sampled == expected_sampled, "restored seeded sampling differs from fresh state");

    // The following request can replace the saved chunk with a later complete
    // boundary. Restoring that chunk must carry forward both layers' new KV.
    let longer = prompt(2300);
    model.restore_text_prefix(&mut cached, 1024, Some(CAPACITY), 31).unwrap();
    model.prepare_text_prefix_capture(&mut cached, 2048).unwrap();
    // Arming the next boundary must preserve the one-shot authorization to
    // enter a suffix stream on the already-restored current prefix.
    model.preflight_restored_text_stream(&longer[1024..], &cached, &greedy_options(&model)).unwrap();
    continuation(&model, &mut cached, &longer[1024..], longer.len() - 1024, false);
    assert_eq!(cached.text_prefix_position(), Some(2048));
    model.restore_text_prefix(&mut cached, 2048, Some(CAPACITY), 31).unwrap();
    let (later, later_logits) = continuation(&model, &mut cached, &longer[2048..], longer.len() - 2048, false);
    let mut fresh_later = model.create_empty_state(Some(CAPACITY), 31).unwrap();
    let (expected_later, expected_later_logits) = continuation(&model, &mut fresh_later, &longer, longer.len(), false);
    assert!(later == expected_later, "replacement snapshot continuation differs from fresh state");
    assert_identical(&later_logits, &expected_later_logits);
}

fn text_snapshot_guards<B: Backend>() {
    let fixture = TinyModel::new(128);
    let engine = Engine::<B>::new().unwrap();
    let model = engine.load_language_model(&fixture.directory).unwrap();
    let mut empty = model.create_empty_state(Some(CAPACITY), 0).unwrap();
    model.prepare_text_prefix_capture(&mut empty, 0).unwrap();
    assert!(empty.text_prefix_position().is_none());
    assert!(model.prepare_text_prefix_capture(&mut empty, 1023).is_err());
    assert!(model.restore_text_prefix(&mut empty, 1024, Some(CAPACITY), 0).is_err());
    let foreign = engine.load_language_model(&fixture.directory).unwrap();
    assert!(foreign.prepare_text_prefix_capture(&mut empty, 1024).is_err());
    empty.multimodal_failed = true;
    assert!(model.prepare_text_prefix_capture(&mut empty, 1024).is_err());

    // A real image-prefilled state cannot be offered for ordinary text reuse,
    // even when the decoder's immutable layout supports text snapshots.
    let mut image_state = model.create_empty_state(Some(CAPACITY), 0).unwrap();
    let image = fixture.prefill(&model, 0);
    {
        let mut stream = model.stream_multimodal(&image, &mut image_state, greedy_options(&model)).unwrap();
        assert!(stream.next().unwrap().is_ok());
    }
    assert!(image_state.has_multimodal_context());
    assert!(image_state.text_prefix_position().is_none());
    assert!(model.prepare_text_prefix_capture(&mut image_state, 1024).is_err());
    assert!(model.restore_text_prefix(&mut image_state, 1024, Some(CAPACITY), 0).is_err());

    // Ring caches overwrite old rows; they need owned snapshots rather than
    // admitting the cheaper scalar-only rewind operation.
    let config_path = fixture.directory.join("config.json");
    let mut config: Value = serde_json::from_slice(&fs::read(&config_path).unwrap()).unwrap();
    config["decoder_config"]["transformer_config"]["layer_configs"][0]["mixer_config"]["sliding_window_size"] =
        json!(128);
    fs::write(config_path, serde_json::to_vec(&config).unwrap()).unwrap();
    let ring = engine.load_language_model(&fixture.directory).unwrap();
    assert!(ring.text_prefix_snapshot_supported());
    assert!(!ring.prefix_reuse_supported());
    let mut ring_state = ring.create_empty_state(Some(CAPACITY), 0).unwrap();
    assert!(ring.prepare_text_prefix_capture(&mut ring_state, 1024).is_ok());
}

#[cfg(backend = "cpu")]
#[stark_test]
fn spatial_decoder_text_prefix_matches_fresh_cpu() {
    text_snapshot_matches_fresh::<crate::backends::cpu::Cpu>(false, false);
}

#[cfg(backend = "cpu")]
#[stark_test]
fn spatial_decoder_text_prefix_guards_cpu() {
    text_snapshot_guards::<crate::backends::cpu::Cpu>();
}

#[cfg(backend = "metal")]
#[stark_test]
fn spatial_decoder_text_prefix_matches_fresh_metal() {
    text_snapshot_matches_fresh::<crate::backends::metal::Metal>(false, false);
}

#[cfg(backend = "metal")]
#[stark_test]
fn spatial_decoder_text_prefix_guards_metal() {
    text_snapshot_guards::<crate::backends::metal::Metal>();
}

#[cfg(backend = "cpu")]
#[stark_test]
fn hybrid_spatial_decoder_text_prefix_matches_fresh_cpu() {
    text_snapshot_matches_fresh::<crate::backends::cpu::Cpu>(true, false);
}

#[cfg(backend = "metal")]
#[stark_test]
fn hybrid_spatial_decoder_text_prefix_matches_fresh_metal() {
    text_snapshot_matches_fresh::<crate::backends::metal::Metal>(true, false);
}

#[cfg(backend = "cpu")]
#[stark_test]
fn overwritten_ring_shared_kv_text_prefix_matches_fresh_cpu() {
    text_snapshot_matches_fresh::<crate::backends::cpu::Cpu>(false, true);
}

#[cfg(backend = "metal")]
#[stark_test]
fn overwritten_ring_shared_kv_text_prefix_matches_fresh_metal() {
    text_snapshot_matches_fresh::<crate::backends::metal::Metal>(false, true);
}
