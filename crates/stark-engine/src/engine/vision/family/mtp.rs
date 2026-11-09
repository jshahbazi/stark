//! The optional dense predictor is bound to the admitted target graph. A
//! declared head cannot broaden the image family's operation or tensor set.

use serde_json::{Value, json};

use super::{FamilyProfile, TensorContract, invalid};
use crate::backends::common::InvalidInput;

pub(super) fn extend_contract(
    config: &Value,
    profile: FamilyProfile,
    tensors: &mut TensorContract,
) -> Result<(), InvalidInput> {
    let Some(head) = config.get("mtp_config").filter(|value| !value.is_null()) else {
        return Ok(());
    };
    // Only the pinned BF16 expert layout extends the dense contract below.
    // Affine expert predictors require independent quantized-head admission.
    if !matches!(
        profile,
        FamilyProfile::FixedV1 | FamilyProfile::DynamicV2 | FamilyProfile::AffineV3 | FamilyProfile::Bf16MoeV5
    ) {
        return Err(invalid());
    }
    let target = &config["decoder_config"]["transformer_config"];
    let layers = target["layer_configs"].as_array().ok_or_else(invalid)?;
    let attention =
        layers.iter().find(|layer| layer["mixer_config"]["type"] == "AttentionConfig").ok_or_else(invalid)?;
    // The predictor only consumes text. Its ordinary rotary positions equal
    // the text diagonal of the target's multimodal position contract.
    let mut predictor = attention.clone();
    let rope = &attention["rope_config"];
    predictor["rope_config"] = json!({"type":"UnscaledRoPEConfig", "base":rope["base"],
        "max_sequence_length":rope["max_sequence_length"],"head_dim":rope["head_dim"]});
    let model = target["model_dim"].as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(invalid)?;
    let hidden = target["hidden_dim"].as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(invalid)?;
    let drafts = head.get("draft_readout_bits").cloned().unwrap_or(Value::Null);
    if !matches!(drafts.as_u64(), None | Some(3 | 4)) || !drafts.is_null() && !drafts.is_u64() {
        return Err(invalid());
    }
    let mut normalized = head.clone();
    normalized.as_object_mut().ok_or_else(invalid)?.entry("draft_readout_bits").or_insert(Value::Null);
    if normalized
        != json!({"model_dim":model,"target_layer":layers.len()-1,
            "input_norm_config":target["output_norm_config"],
            "transformer_config":{"model_dim":model,"hidden_dim":hidden,"layer_configs":[predictor],
                "output_norm_config":target["output_norm_config"]},
            "draft_readout_bits":drafts})
    {
        return Err(invalid());
    }
    let mixer = &attention["mixer_config"];
    let count = |name| mixer[name].as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(invalid);
    let (heads, groups, head_dim) = (count("num_heads")?, count("num_groups")?, count("head_dim")?);
    let query = heads.checked_mul(head_dim).ok_or_else(invalid)?;
    let kv = groups.checked_mul(head_dim).ok_or_else(invalid)?;
    let qkvg = query.checked_add(kv).and_then(|n| n.checked_mul(2)).ok_or_else(invalid)?;
    let doubled_model = model.checked_mul(2).ok_or_else(invalid)?;
    let doubled_hidden = hidden.checked_mul(2).ok_or_else(invalid)?;
    let mut expected = vec![
        ("embedding_norm.scales", "F32", vec![model]),
        ("hidden_norm.scales", "F32", vec![model]),
        ("fusion.weights.weights", "BF16", vec![model, doubled_model]),
        ("transformer.output_norm.scales", "F32", vec![model]),
        ("transformer.layers.0.pre_mixer_norm.scales", "F32", vec![model]),
        ("transformer.layers.0.pre_mlp_norm.scales", "F32", vec![model]),
        ("transformer.layers.0.mixer.query_norm.scales", "F32", vec![head_dim]),
        ("transformer.layers.0.mixer.key_norm.scales", "F32", vec![head_dim]),
        ("transformer.layers.0.mixer.qkvg_projection.weights.weights", "BF16", vec![qkvg, model]),
        ("transformer.layers.0.mixer.out_projection.weights.weights", "BF16", vec![model, query]),
    ];
    if matches!(profile, FamilyProfile::Bf16MoeV5) {
        let mlp = &attention["mlp_config"];
        let count = |name| mlp[name].as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(invalid);
        let experts = count("num_routed_experts")?;
        let width = count("expert_hidden_dim")?;
        if !(1..=1024).contains(&experts) || width == 0 || width > 131072 {
            return Err(invalid());
        }
        let up = experts.checked_mul(width).and_then(|n| n.checked_mul(2)).ok_or_else(invalid)?;
        let down = experts.checked_mul(model).ok_or_else(invalid)?;
        expected.extend([
            ("transformer.layers.0.mlp.experts.up_projection.weights.weights", "BF16", vec![up, model]),
            ("transformer.layers.0.mlp.experts.down_projection.weights.weights", "BF16", vec![down, width]),
            ("transformer.layers.0.mlp.router.weights.weights", "BF16", vec![experts, model]),
            ("transformer.layers.0.mlp.shared_gate.weights.weights", "BF16", vec![1, model]),
            (
                "transformer.layers.0.mlp.shared_expert.up_projection.weights.weights",
                "BF16",
                vec![doubled_hidden, model],
            ),
            ("transformer.layers.0.mlp.shared_expert.down_projection.weights.weights", "BF16", vec![model, hidden]),
        ]);
    } else {
        expected.extend([
            ("transformer.layers.0.mlp.up_projection.weights.weights", "BF16", vec![doubled_hidden, model]),
            ("transformer.layers.0.mlp.down_projection.weights.weights", "BF16", vec![model, hidden]),
        ]);
    }
    for (name, dtype, shape) in expected {
        shape.iter().try_fold(1u32, |n, width| n.checked_mul(*width)).ok_or_else(invalid)?;
        if tensors.tensors.insert(format!("mtp.{name}"), (dtype, shape)).is_some() {
            return Err(invalid());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    fn fixture() -> Value {
        fixture_with_norm(false)
    }

    fn fixture_with_norm(raw_bf16: bool) -> Value {
        let mut decoder = super::super::decoder_profile();
        fn folded(value: &mut Value) {
            match value {
                Value::Object(map) => {
                    if let Some(offset) = map.get_mut("scale_offset") {
                        *offset = Value::Null;
                    }
                    for value in map.values_mut() {
                        folded(value);
                    }
                },
                Value::Array(values) => values.iter_mut().for_each(folded),
                _ => {},
            }
        }
        if !raw_bf16 {
            folded(&mut decoder);
        }
        let target = &decoder["transformer_config"];
        let mut predictor = target["layer_configs"][3].clone();
        let rope = &predictor["rope_config"];
        predictor["rope_config"] = json!({"type":"UnscaledRoPEConfig","base":rope["base"],
            "max_sequence_length":rope["max_sequence_length"],"head_dim":rope["head_dim"]});
        let head = json!({"model_dim":target["model_dim"],
            "target_layer":target["layer_configs"].as_array().unwrap().len()-1,
            "input_norm_config":target["output_norm_config"],
            "transformer_config":{"model_dim":target["model_dim"],"hidden_dim":target["hidden_dim"],
                "layer_configs":[predictor],"output_norm_config":target["output_norm_config"]}});
        json!({"decoder_config":decoder,"mtp_config":head})
    }

    fn contract() -> TensorContract {
        TensorContract {
            tensors: Default::default(),
            affine: true,
        }
    }

    #[stark_test]
    fn image_dense_mtp_contract_binds_complete_head_and_target_geometry() {
        let config = fixture();
        let mut tensors = contract();
        extend_contract(&config, FamilyProfile::AffineV3, &mut tensors).unwrap();
        assert_eq!(tensors.tensors.len(), 12);
        assert_eq!(tensors.tensors["mtp.fusion.weights.weights"], ("BF16", vec![1024, 2048]));
        assert_eq!(tensors.tensors["mtp.transformer.layers.0.mixer.query_norm.scales"], ("F32", vec![256]));
        for bits in [Value::Null, json!(3), json!(4)] {
            let mut config = config.clone();
            config["mtp_config"]["draft_readout_bits"] = bits;
            extend_contract(&config, FamilyProfile::AffineV3, &mut contract()).unwrap();
        }
    }

    #[stark_test]
    fn image_bf16_mtp_contract_preserves_raw_norm_offsets_and_full_precision_head() {
        let config = fixture_with_norm(true);
        assert_eq!(config["mtp_config"]["input_norm_config"]["scale_offset"], json!(1.0));
        for profile in [FamilyProfile::FixedV1, FamilyProfile::DynamicV2] {
            let mut tensors = contract();
            tensors.affine = false;
            extend_contract(&config, profile, &mut tensors).unwrap();
            assert_eq!(tensors.tensors.len(), 12);
            assert_eq!(tensors.tensors["mtp.transformer.layers.0.mlp.up_projection.weights.weights"].0, "BF16");
        }
    }

    #[stark_test]
    fn image_bf16_expert_mtp_contract_binds_routed_shared_and_gate_payloads() {
        let mut config = fixture_with_norm(true);
        let dense = config["decoder_config"]["transformer_config"]["layer_configs"][0]["mlp_config"].clone();
        let expert = json!({"type":"MixtureOfExpertsConfig","expert_config":dense,"router_config":{},
            "routing_function":{"type":"SoftmaxRouting"},"num_routed_experts":4,"num_active_routed_experts":2,
            "router_has_biases":false,"num_shared_experts":1,"expert_hidden_dim":256,"gate_config":{}});
        for layer in config["decoder_config"]["transformer_config"]["layer_configs"].as_array_mut().unwrap() {
            layer["mlp_config"] = expert.clone();
        }
        config["mtp_config"]["transformer_config"]["layer_configs"][0]["mlp_config"] = expert;
        let mut tensors = contract();
        tensors.affine = false;
        extend_contract(&config, FamilyProfile::Bf16MoeV5, &mut tensors).unwrap();
        assert_eq!(tensors.tensors.len(), 16);
        assert_eq!(
            tensors.tensors["mtp.transformer.layers.0.mlp.experts.up_projection.weights.weights"],
            ("BF16", vec![2048, 1024])
        );
        assert_eq!(
            tensors.tensors["mtp.transformer.layers.0.mlp.experts.down_projection.weights.weights"],
            ("BF16", vec![4096, 256])
        );
        let mut changed = config;
        changed["mtp_config"]["transformer_config"]["layer_configs"][0]["mlp_config"]["num_active_routed_experts"] =
            json!(1);
        assert!(extend_contract(&changed, FamilyProfile::Bf16MoeV5, &mut contract()).is_err());
    }

    #[stark_test]
    fn image_dense_mtp_contract_rejects_new_operations_and_unsupported_profiles() {
        let config = fixture();
        for profile in [FamilyProfile::MoeV4, FamilyProfile::Bf16MoeV5] {
            assert!(extend_contract(&config, profile, &mut contract()).is_err());
        }
        for (pointer, value) in [
            ("/mtp_config/target_layer", json!(0)),
            ("/mtp_config/model_dim", json!(2048)),
            ("/mtp_config/extra", json!(true)),
            ("/mtp_config/draft_readout_bits", json!(8)),
            ("/mtp_config/draft_readout_bits", json!("3")),
            ("/mtp_config/transformer_config/layer_configs/0/mixer_config/sliding_window_size", json!(128)),
            ("/mtp_config/transformer_config/layer_configs/0/mixer_config/is_kv_sharing", json!(true)),
            ("/mtp_config/transformer_config/layer_configs/0/mlp_config/type", json!("MixtureOfExpertsConfig")),
        ] {
            let mut changed = config.clone();
            let (parent, field) = pointer.rsplit_once('/').unwrap();
            changed.pointer_mut(parent).unwrap().as_object_mut().unwrap().insert(field.into(), value);
            assert!(extend_contract(&changed, FamilyProfile::AffineV3, &mut contract()).is_err(), "accepted {pointer}");
        }
    }

    #[stark_test]
    fn image_dense_mtp_contract_keeps_headless_legacy_exports_headless() {
        for config in [json!({}), json!({"mtp_config":null})] {
            let mut tensors = contract();
            extend_contract(&config, FamilyProfile::AffineV3, &mut tensors).unwrap();
            assert!(tensors.tensors.is_empty());
        }
    }
}
