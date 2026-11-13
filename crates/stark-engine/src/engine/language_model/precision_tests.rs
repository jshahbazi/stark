use serde_json::{Value, json};
use stark_engine_macros::stark_test;

use super::precision::{activation_precision_admitted, f16_dense_admitted};
use crate::{config::model::language_model::LanguageModelConfig, data_type::DataType};

fn norm() -> Value {
    json!({"epsilon": 1e-6, "scale_offset": null, "upcast_mode": "only_normalization",
        "subtract_mean": false, "has_scale": true, "has_biases": false})
}

/// Plain two-layer dense graph as produced for Qwen2 (Q/K/V biases, tied head).
fn dense() -> Value {
    let layer = json!({
        "pre_mixer_norm_config": norm(),
        "mixer_conv_config": null,
        "mixer_config": {"type": "AttentionConfig", "qkvg_projection_config": {}, "out_projection_config": {},
            "query_norm_config": null, "key_norm_config": null, "num_heads": 4, "num_groups": 2,
            "head_dim": 64, "is_causal": true, "scale": null, "sliding_window_size": null,
            "logit_soft_cap": null, "has_sinks": false, "has_qkvg_biases": true, "has_out_biases": false,
            "has_gate": false, "normalize_values": false, "is_kv_sharing": false},
        "post_mixer_norm_config": null,
        "pre_mlp_norm_config": norm(),
        "mlp_conv_config": null,
        "mlp_config": {"type": "DenseMLPConfig", "linear_config": {}, "activation": {"type": "SiLU", "alpha": 1.0},
            "has_up_biases": false, "has_down_biases": false, "gate_clipping": null, "up_clipping": null},
        "post_mlp_norm_config": null,
        "hidden_dim": null,
        "ple_config": null,
        "has_post_layer_scalar": false,
        "kv_source_layer_index": null,
        "rope_config": {"type": "UnscaledRoPEConfig", "base": 1000000.0, "max_sequence_length": 4096, "head_dim": 64}
    });
    json!({
        "type": "LanguageModelConfig",
        "decoder_config": {
            "embedding_config": {"type": "TiedEmbeddingConfig", "input_scale": null, "logit_soft_cap": null,
                "logit_scale": null},
            "transformer_config": {"layer_configs": [layer.clone(), layer], "output_norm_config": norm(),
                "model_dim": 256, "hidden_dim": 512},
            "vocab_size": 1024,
            "ple_model_config": null,
            "embedding_norm_config": null
        },
        "token_codec_config": {"type": "RawTextCodecConfig"},
        "generation_config": {"stop_token_ids": [], "temperature": null, "top_k": null, "top_p": null,
            "min_p": null, "banned_tokens": null, "repetition_penalty": null, "presence_penalty": null,
            "frequency_penalty": null, "suffix_repetition_length": null}
    })
}

fn parse(value: Value) -> LanguageModelConfig {
    serde_json::from_value(value).unwrap()
}

#[stark_test]
fn plain_dense_graph_admits_f16_and_bf16_is_unconditional() {
    let config = parse(dense());
    assert!(f16_dense_admitted(&config));
    assert!(activation_precision_admitted(DataType::F16, &config, false));
    assert!(activation_precision_admitted(DataType::BF16, &config, false));
    // A speculator has no F16 path; BF16 admission never depends on it.
    assert!(!activation_precision_admitted(DataType::F16, &config, true));
    assert!(activation_precision_admitted(DataType::BF16, &config, true));
    assert!(!activation_precision_admitted(DataType::F32, &config, false));

    let mut llama = dense();
    llama["decoder_config"]["embedding_config"]["type"] = json!("UntiedEmbeddingConfig");
    for layer in llama["decoder_config"]["transformer_config"]["layer_configs"].as_array_mut().unwrap() {
        layer["mixer_config"]["has_qkvg_biases"] = json!(false);
        layer["rope_config"] = json!({"type": "LlamaRoPEConfig", "base": 500000.0, "max_sequence_length": 8192,
            "head_dim": 64, "scaling_factor": 8.0, "original_context_length": 8192,
            "low_frequency_factor": 1.0, "high_frequency_factor": 4.0});
    }
    assert!(f16_dense_admitted(&parse(llama)));
}

#[stark_test]
fn f16_fails_closed_outside_the_plain_dense_graph() {
    let layer_mutations: [(&str, &str, Value); 11] = [
        ("mixer_config", "query_norm_config", norm()),
        ("mixer_config", "key_norm_config", norm()),
        ("mixer_config", "sliding_window_size", json!(4096)),
        ("mixer_config", "logit_soft_cap", json!(50.0)),
        ("mixer_config", "has_sinks", json!(true)),
        ("mixer_config", "has_gate", json!(true)),
        ("mixer_config", "is_kv_sharing", json!(true)),
        ("mixer_config", "is_causal", json!(false)),
        ("mixer_config", "reference_bf16", json!(true)),
        ("mlp_config", "activation", json!({"type": "GELU", "approximate": true})),
        ("mlp_config", "gate_clipping", json!([-10.0, 10.0])),
    ];
    for (block, field, value) in layer_mutations {
        let mut config = dense();
        config["decoder_config"]["transformer_config"]["layer_configs"][1][block][field] = value;
        let config = parse(config);
        assert!(!f16_dense_admitted(&config), "{block}.{field}");
        assert!(activation_precision_admitted(DataType::BF16, &config, false));
    }
    let layer_fields: [(&str, Value); 4] = [
        ("post_mlp_norm_config", norm()),
        ("post_mixer_norm_config", norm()),
        ("has_post_layer_scalar", json!(true)),
        (
            "rope_config",
            json!({"type": "YARNRoPEConfig", "base": 10000.0, "max_sequence_length": 4096, "head_dim": 64,
            "scaling_factor": 4.0, "original_context_length": 4096, "beta_fast": 32.0, "beta_slow": 1.0, "truncate": true}),
        ),
    ];
    for (field, value) in layer_fields {
        let mut config = dense();
        config["decoder_config"]["transformer_config"]["layer_configs"][0][field] = value;
        let config = parse(config);
        assert!(!f16_dense_admitted(&config), "{field}");
    }
    for (field, value) in [("logit_soft_cap", json!(30.0)), ("input_scale", json!(16.0)), ("logit_scale", json!(0.5))] {
        let mut config = dense();
        config["decoder_config"]["embedding_config"][field] = value;
        assert!(!f16_dense_admitted(&parse(config)), "{field}");
    }
    let mut config = dense();
    config["decoder_config"]["embedding_norm_config"] = norm();
    assert!(!f16_dense_admitted(&parse(config)));
    let mut config = dense();
    config["decoder_config"]["transformer_config"]["output_norm_config"]["mlx_bf16_reduction"] = json!(true);
    assert!(!f16_dense_admitted(&parse(config)));
    let mut config = dense();
    config["decoder_config"]["transformer_config"]["layer_configs"][0]["pre_mixer_norm_config"] = Value::Null;
    assert!(!f16_dense_admitted(&parse(config)));
}
