//! Runtime-only configuration for the pinned, explicitly selected Edge0 reference.
//!
//! GGUF geometry authority: Edge0's llama.cpp pin 3cf03257f219afbe7334045ff7c6a06ac68c627d,
//! conversion/qwen4exp.py and src/models/qwen4exp.cpp. The prepared receipt and
//! source files remain unchanged. This diagnostic path has a 512-token limit.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::{Error, Result};

pub(super) fn selected(config: &Value) -> bool {
    config.pointer("/storage/layout").and_then(Value::as_str) == Some("edge0-strata")
}

/// Exact header facts from the pinned artifact, shared with preparation's
/// EXPECTED_GEOMETRY. Reject extra family keys as well as changed values.
pub(super) fn validate_metadata(metadata: &BTreeMap<String, Value>) -> Result<()> {
    let expected: BTreeMap<String, Value> = serde_json::from_str(include_str!("edge0_geometry.json"))?;
    let actual: BTreeMap<String, Value> = metadata
        .iter()
        .filter(|(key, _)| key.starts_with("qwen4exp."))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    if metadata.get("general.architecture").and_then(Value::as_str) != Some("qwen4exp") || actual != expected {
        return Err(Error::Unsupported("Edge0 pinned skeleton geometry"));
    }
    Ok(())
}

pub(super) fn runtime_config(
    prepared: &Value,
    metadata: &BTreeMap<String, Value>,
) -> Result<Value> {
    validate_metadata(metadata)?;
    if prepared.get("model_type").and_then(Value::as_str) != Some("qwen4_exp")
        || prepared.pointer("/storage/schema_version").and_then(Value::as_u64) != Some(1)
        || prepared.pointer("/stark_preparation/schema_version").and_then(Value::as_u64) != Some(1)
        || prepared.pointer("/stark_preparation/source").and_then(Value::as_str) != Some("Edge0/Edge0-145B-q3k")
        || prepared.pointer("/stark_preparation/revision").and_then(Value::as_str)
            != Some("d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74")
        || prepared.pointer("/stark_preparation/preparatory_only").and_then(Value::as_bool) != Some(true)
        || prepared.pointer("/stark_preparation/usage_scope").and_then(Value::as_str)
            != Some("local_noncommercial_evaluation")
        || prepared.pointer("/stark_preparation/tokenizer_policy").and_then(Value::as_str) != Some("edge0-native-v2")
        || metadata.get("general.architecture").and_then(Value::as_str) != Some("qwen4exp")
    {
        return Err(Error::Unsupported("Edge0 reference source/configuration"));
    }
    let integer = |key: &str| -> Result<u64> {
        metadata
            .get(key)
            .and_then(Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or(Error::Invalid("Edge0 reference geometry"))
    };
    let scalar = |key: &str| -> Result<f64> {
        metadata
            .get(key)
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or(Error::Invalid("Edge0 reference scalar"))
    };
    let count = integer("qwen4exp.block_count")?;
    let recurrent = metadata
        .get("qwen4exp.attention.recurrent_layers")
        .and_then(Value::as_array)
        .filter(|values| values.len() as u64 == count)
        .ok_or(Error::Invalid("Edge0 reference schedule"))?;
    let layers = recurrent
        .iter()
        .map(|value| match value.as_bool() {
            Some(true) => Ok("linear_attention"),
            Some(false) => Ok("full_attention"),
            None => Err(Error::Invalid("Edge0 reference schedule")),
        })
        .collect::<Result<Vec<_>>>()?;
    let ple_layers = metadata
        .get("qwen4exp.ple.layers")
        .and_then(Value::as_array)
        .ok_or(Error::Invalid("Edge0 reference PLE schedule"))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .filter(|id| *id < count)
                .map(|id| id + 1)
                .ok_or(Error::Invalid("Edge0 reference PLE schedule"))
        })
        .collect::<Result<Vec<_>>>()?;
    let value_heads = integer("qwen4exp.ssm.time_step_rank")?;
    let inner = integer("qwen4exp.ssm.inner_size")?;
    let head_dim = integer("qwen4exp.attention.key_length")?;
    let rotary = integer("qwen4exp.rope.dimension_count")?;
    let ngram = integer("qwen4exp.ple.ngram_size")?;
    let heads_per_ngram = integer("qwen4exp.ple.heads_per_ngram")?;
    if !inner.is_multiple_of(value_heads) || rotary > head_dim || ngram < 2 {
        return Err(Error::Invalid("Edge0 reference derived geometry"));
    }
    let ple_dim = (ngram - 1)
        .checked_mul(heads_per_ngram)
        .and_then(|heads| heads.checked_mul(integer("qwen4exp.embedding_length_per_layer_input").ok()?))
        .ok_or(Error::Invalid("Edge0 reference PLE width"))?;
    let mut config = prepared.clone();
    config["quantization"] =
        json!({"format": "edge0-native-reference", "normalization": "gguf-centered-gamma-converted"});
    config["text_config"] = json!({
        "model_type": "qwen4_exp_text", "hidden_act": "silu", "attention_bias": false,
        "hidden_size": integer("qwen4exp.embedding_length")?, "vocab_size": 248320,
        "num_hidden_layers": count, "layer_types": layers,
        "max_position_embeddings": 512, "source_max_position_embeddings": integer("qwen4exp.context_length")?,
        "hc_count": integer("qwen4exp.hyper_connection.count")?,
        "hc_lowrank": integer("qwen4exp.hyper_connection.low_rank")?,
        "num_attention_heads": integer("qwen4exp.attention.head_count")?,
        "num_key_value_heads": integer("qwen4exp.attention.head_count_kv")?, "head_dim": head_dim,
        "rope_parameters": {"rope_type": "default", "rope_theta": scalar("qwen4exp.rope.freq_base")?,
            "partial_rotary_factor": rotary as f64 / head_dim as f64},
        "linear_num_key_heads": integer("qwen4exp.ssm.group_count")?,
        "linear_num_value_heads": value_heads, "linear_key_head_dim": integer("qwen4exp.ssm.state_size")?,
        "linear_value_head_dim": inner / value_heads, "linear_conv_kernel_dim": integer("qwen4exp.ssm.conv_kernel")?,
        "num_experts": integer("qwen4exp.expert_count")?,
        "num_experts_per_tok": integer("qwen4exp.expert_used_count")?,
        "moe_intermediate_size": integer("qwen4exp.expert_feed_forward_length")?,
        "shared_expert_intermediate_size": integer("qwen4exp.expert_shared_feed_forward_length")?,
        "indexer_n_heads": integer("qwen4exp.attention.indexer.head_count")?,
        "indexer_head_dim": integer("qwen4exp.attention.indexer.key_length")?,
        "indexer_budget": integer("qwen4exp.attention.indexer.top_k")?,
        "indexer_compress_ratio": 4, "indexer_kv_heads": 1,
        "rms_norm_eps": scalar("qwen4exp.attention.layer_norm_rms_epsilon")?,
        "output_gate_type": "sigmoid", "tie_word_embeddings": false,
        "ngram_size": ngram, "heads_per_ngram": heads_per_ngram, "ple_embed_dim": ple_dim,
        "ple_conv_kernel_size": integer("qwen4exp.ple.conv_kernel")?, "ple_layer_ids": ple_layers,
        "split_ngram_parts": 1,
        // Qwen's first EOS is also its PLE history reset. The source PLE reset
        // (248044) differs from its primary generation EOS (248046). Retain
        // all five source EOG IDs while preserving that independent boundary.
        "eos_token_id": [248044, 248046, 248063, 248064, 248065]
    });
    Ok(config)
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    fn inputs() -> (Value, BTreeMap<String, Value>) {
        let mut metadata: BTreeMap<String, Value> = serde_json::from_str(include_str!("edge0_geometry.json")).unwrap();
        metadata.insert("general.architecture".into(), json!("qwen4exp"));
        let prepared = json!({
            "model_type":"qwen4_exp", "storage":{"schema_version":1,"layout":"edge0-strata"},
            "stark_preparation":{"schema_version":1,"source":"Edge0/Edge0-145B-q3k",
                "revision":"d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74","preparatory_only":true,
                "usage_scope":"local_noncommercial_evaluation","tokenizer_policy":"edge0-native-v2"}
        });
        (prepared, metadata)
    }

    #[stark_test]
    fn edge0_runtime_geometry_preserves_source_conventions_and_receipt() {
        let (prepared, metadata) = inputs();
        let original = prepared.clone();
        let config = runtime_config(&prepared, &metadata).unwrap();
        assert_eq!(prepared, original);
        let text = &config["text_config"];
        assert_eq!(text["ple_layer_ids"], json!([2]));
        assert_eq!(text["ple_embed_dim"], 2560);
        assert_eq!(text["linear_num_key_heads"], 16);
        assert_eq!(text["linear_num_value_heads"], 48);
        assert_eq!(text["linear_value_head_dim"], 128);
        assert_eq!(text["max_position_embeddings"], 512);
        assert_eq!(text["source_max_position_embeddings"], 262144);
        assert_eq!(text["rope_parameters"]["partial_rotary_factor"], 0.25);
        assert_eq!(text["eos_token_id"], json!([248044, 248046, 248063, 248064, 248065]));
        let layers = text["layer_types"].as_array().unwrap();
        assert_eq!(layers.iter().filter(|kind| **kind == "linear_attention").count(), 36);
        assert_eq!(layers.iter().filter(|kind| **kind == "full_attention").count(), 12);
    }

    #[stark_test]
    fn edge0_reference_rejects_geometry_or_preparation_drift() {
        let (prepared, metadata) = inputs();
        for key in ["qwen4exp.ple.layers", "qwen4exp.ssm.group_count", "qwen4exp.rope.freq_base"] {
            let mut changed = metadata.clone();
            changed.remove(key);
            assert!(runtime_config(&prepared, &changed).is_err());
        }
        let mut changed = metadata.clone();
        changed.insert("qwen4exp.new_feature".into(), json!(1));
        assert!(runtime_config(&prepared, &changed).is_err());
        for (pointer, value) in [
            ("/stark_preparation/revision", json!("unqualified")),
            ("/stark_preparation/preparatory_only", json!(false)),
            ("/stark_preparation/usage_scope", json!("product")),
            ("/stark_preparation/tokenizer_policy", json!("borrowed-qwen")),
        ] {
            let mut changed = prepared.clone();
            *changed.pointer_mut(pointer).unwrap() = value;
            assert!(runtime_config(&changed, &metadata).is_err());
        }
    }
}
