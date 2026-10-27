use stark_engine_macros::stark_test;

use super::*;

fn fixture(
    model: u32,
    hidden: u32,
    layers: usize,
    attention_heads: u32,
    attention_groups: u32,
    delta_heads: u32,
    delta_groups: u32,
    tied: bool,
    depth: u32,
    width: u32,
    intermediate: u32,
    vision_heads: u32,
) -> (Value, Value) {
    let mut decoder = decoder_profile();
    decoder["transformer_config"]["model_dim"] = json!(model);
    decoder["transformer_config"]["hidden_dim"] = json!(hidden);
    decoder["embedding_config"]["type"] = json!(if tied {
        "TiedEmbeddingConfig"
    } else {
        "UntiedEmbeddingConfig"
    });
    let baseline = decoder["transformer_config"]["layer_configs"].as_array().unwrap().clone();
    decoder["transformer_config"]["layer_configs"] = json!(
        (0..layers)
            .map(|index| {
                let attention = index % 4 == 3;
                let mut layer = baseline[if attention {
                    3
                } else {
                    0
                }]
                .clone();
                layer["mixer_config"]["num_heads"] = json!(if attention {
                    attention_heads
                } else {
                    delta_heads
                });
                layer["mixer_config"]["num_groups"] = json!(if attention {
                    attention_groups
                } else {
                    delta_groups
                });
                layer
            })
            .collect::<Vec<_>>()
    );
    let mut vision = vision_profile();
    for (key, number) in [
        ("out_hidden_size", model),
        ("depth", depth),
        ("hidden_size", width),
        ("intermediate_size", intermediate),
        ("num_heads", vision_heads),
    ] {
        vision[key] = json!(number);
    }
    (decoder, vision)
}

#[stark_test]
fn dynamic_geometry_covers_dense_size_shapes_without_a_size_dispatch_table() {
    for parameters in [
        (1024, 3584, 24, 8, 2, 16, 16, true, 12, 768, 3072, 12),
        (2048, 6144, 24, 8, 2, 16, 16, true, 24, 1024, 4096, 16),
        (2560, 9216, 32, 16, 4, 32, 16, true, 24, 1024, 4096, 16),
        (4096, 12288, 32, 16, 4, 32, 16, false, 27, 1152, 4304, 16),
        (5120, 17408, 64, 24, 4, 48, 16, false, 27, 1152, 4304, 16),
        // Deliberately differs from named released sizes.
        (1568, 6304, 8, 12, 3, 48, 16, false, 3, 1152, 4304, 16),
    ] {
        let (m, h, l, q, g, d, dg, t, vd, vw, vi, vh) = parameters;
        let (decoder, vision) = fixture(m, h, l, q, g, d, dg, t, vd, vw, vi, vh);
        let admitted = geometry(&decoder, &vision).unwrap();
        assert_eq!(admitted.tower.out_hidden_size, m);
        assert_eq!(admitted.tower.hidden_size / admitted.tower.num_heads, vw / vh);
        let name = if t {
            "decoder.embedding.embedding.weights"
        } else {
            "decoder.embedding.output_embedding.weights"
        };
        assert_eq!(admitted.tensors.tensors[name], ("BF16", vec![248320, m]));
        assert_eq!(
            admitted.tensors.tensors["decoder.transformer.layers.0.mixer.in_proj.weights.weights"].1,
            vec![2 * dg * 128 + 2 * d * 128 + 2 * d, m]
        );
    }
    let baseline = geometry(&decoder_profile(), &vision_profile()).unwrap();
    assert_eq!(baseline.tensors.tensors, super::super::expected_tensors());
}

#[stark_test]
fn dynamic_geometry_rejects_unsupported_operations_and_unsafe_allocations() {
    let (decoder, vision) = fixture(4096, 12288, 32, 16, 4, 32, 16, false, 27, 1152, 4304, 16);
    for (pointer, value) in [
        ("/transformer_config/model_dim", json!(4097)),
        ("/transformer_config/hidden_dim", json!(12304)),
        ("/vocab_size", json!(1048577)),
        ("/embedding_config/input_scale", json!(2.0)),
        ("/transformer_config/layer_configs/0/mixer_config/num_groups", json!(17)),
        ("/transformer_config/layer_configs/0/mixer_config/head_dim", json!(64)),
        ("/transformer_config/layer_configs/3/mixer_config/reference_bf16", json!(false)),
        ("/transformer_config/layer_configs/3/rope_config/max_sequence_length", json!(1048577)),
        ("/transformer_config/layer_configs/3/rope_config/base", json!(10000.0)),
        ("/transformer_config/output_norm_config/epsilon", json!(1e-60)),
    ] {
        let mut changed = decoder.clone();
        *changed.pointer_mut(pointer).unwrap() = value;
        assert!(geometry(&changed, &vision).is_err(), "accepted {pointer}");
    }
    for (key, value) in [
        ("num_heads", json!(17)),
        ("depth", json!(65)),
        ("intermediate_size", json!(32769)),
        ("num_position_embeddings", json!(2305)),
        ("out_hidden_size", json!(1024)),
        ("patch_size", json!(14)),
    ] {
        let mut changed = vision.clone();
        changed[key] = value;
        assert!(geometry(&decoder, &changed).is_err(), "accepted vision {key}");
    }
    // Each individual matrix index and the aggregate payload have separate limits.
    let (mut huge, huge_vision) = fixture(16384, 131072, 4, 16, 4, 32, 16, false, 27, 1152, 4304, 16);
    huge["vocab_size"] = json!(1048576);
    assert!(geometry(&huge, &huge_vision).is_err());
    let (huge, huge_vision) = fixture(8192, 32768, 256, 16, 4, 32, 16, false, 27, 1152, 4304, 16);
    assert!(geometry(&huge, &huge_vision).is_err());
}

#[stark_test]
fn dynamic_schedule_and_norms_are_consumed_as_one_validated_graph() {
    let (mut decoder, vision) = fixture(2560, 9216, 8, 16, 4, 32, 16, true, 24, 1024, 4096, 16);
    decoder["transformer_config"]["layer_configs"].as_array_mut().unwrap().swap(0, 3);
    let epsilon = json!(1e-5);
    decoder["transformer_config"]["output_norm_config"]["epsilon"] = epsilon.clone();
    for layer in decoder["transformer_config"]["layer_configs"].as_array_mut().unwrap() {
        layer["pre_mixer_norm_config"]["epsilon"] = epsilon.clone();
        layer["pre_mlp_norm_config"]["epsilon"] = epsilon.clone();
        let mixer = &mut layer["mixer_config"];
        if mixer["type"] == "AttentionConfig" {
            mixer["query_norm_config"]["epsilon"] = epsilon.clone();
            mixer["key_norm_config"]["epsilon"] = epsilon.clone();
        } else {
            mixer["norm_config"]["epsilon"] = epsilon.clone();
        }
    }
    let admitted = geometry(&decoder, &vision).unwrap();
    assert!(
        admitted.tensors.tensors.contains_key("decoder.transformer.layers.0.mixer.qkvg_projection.weights.weights")
    );
    assert!(admitted.tensors.tensors.contains_key("decoder.transformer.layers.3.mixer.in_proj.weights.weights"));
    decoder["transformer_config"]["layer_configs"][0]["mixer_config"]["query_norm_config"]["epsilon"] = json!(1e-6);
    assert!(geometry(&decoder, &vision).is_err());
}
