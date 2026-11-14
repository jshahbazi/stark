//! Exact canonical decoder semantics for the five pinned image checkpoints.
//! Geometry follows their source config.json files at the revisions in PROFILES;
//! offline source fixtures and hashes live in the converter's image-profile tests.

use serde_json::{Value, json};

use super::ImageProfile;
use crate::{backends::common::InvalidInput, config::decoder::DecoderConfig};

pub(super) fn expected_decoder(profile: ImageProfile) -> Result<DecoderConfig, InvalidInput> {
    let (count, hidden, heads, groups, global_groups, shared, period, ple, double_wide, moe) = match profile.model_dim {
        1536 => (35, 6144, 8, 1, 1, 20, 5, 256, true, false),
        2560 => (42, 10240, 8, 2, 2, 18, 6, 256, false, false),
        3840 => (48, 15360, 16, 8, 1, 0, 6, 0, false, false),
        2816 => (30, 2112, 16, 8, 2, 0, 6, 0, false, true),
        5376 => (60, 21504, 32, 16, 4, 0, 6, 0, false, false),
        _ => return Err(InvalidInput("unsupported Gemma image decoder profile")),
    };
    let model = profile.model_dim;
    let context = if ple > 0 {
        131072
    } else {
        262144
    };
    let window = if ple > 0 {
        512
    } else {
        1024
    };
    let norm = json!({"epsilon":1e-6,"scale_offset":null,"upcast_mode":"full_layer",
        "subtract_mean":false,"has_scale":true,"has_biases":false});
    let activation = json!({"type":"GELU","approximate":true});
    let dense = json!({"type":"DenseMLPConfig","linear_config":{},"activation":activation,
        "has_up_biases":false,"has_down_biases":false,"gate_clipping":null,"up_clipping":null});
    let mut owners = [None, None];
    let mut layers = Vec::with_capacity(count);
    for index in 0..count {
        let global = (index + 1) % period == 0;
        let slot = usize::from(global);
        let owner = if index < count - shared {
            owners[slot] = Some(index);
            None
        } else {
            owners[slot]
        };
        let width = if global {
            512
        } else {
            256
        };
        let mut mixer = json!({"type":"AttentionConfig","qkvg_projection_config":{},"out_projection_config":{},
            "query_norm_config":norm,"key_norm_config":if owner.is_some() { Value::Null } else { norm.clone() },
            "num_heads":heads,"num_groups":if global { global_groups } else { groups },"head_dim":width,
            "is_causal":true,"scale":1.0,"sliding_window_size":if global { Value::Null } else { json!(window) },
            "logit_soft_cap":null,"has_sinks":false,"has_qkvg_biases":false,"has_out_biases":false,
            "has_gate":false,"normalize_values":true,"is_kv_sharing":owner.is_some()});
        if ple == 0 {
            mixer["materialize_rope_bf16"] = true.into();
            mixer["image_attention_policy"] = if global {
                "causal"
            } else {
                "windowed_same_image"
            }
            .into();
        }
        let mut rope = json!({"type":if global {"ProportionalRoPEConfig"} else {"UnscaledRoPEConfig"},
            "base":if global {1000000.0} else {10000.0},"head_dim":width,"max_sequence_length":context});
        if global {
            rope["rotary_dim"] = 128.into();
        }
        let mlp = if moe {
            json!({"type":"Gemma4MoeConfig","expert_config":dense,"norm_config":norm,
                "num_experts":128,"active_experts":8,"expert_hidden_dim":704,
                "router_root_scale":f64::from(model).sqrt().recip()})
        } else {
            dense.clone()
        };
        let ple_config = if ple > 0 {
            json!({"linear_config":{},"norm_config":norm,"ple_dim":ple,"activation":activation,"reference_bf16":true})
        } else {
            Value::Null
        };
        layers.push(json!({"pre_mixer_norm_config":norm,"mixer_conv_config":null,"mixer_config":mixer,
            "post_mixer_norm_config":norm,"pre_mlp_norm_config":norm,"mlp_conv_config":null,"mlp_config":mlp,
            "post_mlp_norm_config":norm,"hidden_dim":if double_wide && owner.is_some() {json!(hidden*2)} else {Value::Null},
            "ple_config":ple_config,"has_post_layer_scalar":true,"materialize_post_layer_scalar":ple==0,
            "kv_source_layer_index":owner,"rope_config":rope}));
    }
    let ple_config = if ple > 0 {
        json!({"ple_dim":ple,"num_layers":count,"ple_vocab_size":262144,"ple_embed_scale":16.0,
            "model_projection_scale":f64::from(model).sqrt().recip(),"input_scale":2.0f64.sqrt().recip(),
            "linear_config":{},"norm_config":norm,"reference_bf16":true})
    } else {
        Value::Null
    };
    let expected = json!({"gemma4_image_embeddings":ple>0,"scalar_image_embeddings":ple==0,
        "bidirectional_image_attention":ple==0,"vocab_size":262144,"embedding_norm_config":null,
        "embedding_config":{"type":"TiedEmbeddingConfig",
            "input_scale":half::bf16::from_f64(f64::from(model).sqrt()).to_f32(),
            "logit_soft_cap":30.0,"logit_scale":null,"materialize_dequant":true,"materialize_soft_cap":true},
        "ple_model_config":ple_config,"transformer_config":{"layer_configs":layers,
            "output_norm_config":norm,"model_dim":model,"hidden_dim":hidden}});
    serde_json::from_value(expected).map_err(|_| InvalidInput("invalid canonical Gemma image decoder profile"))
}

pub(super) fn validate_decoder(
    raw: &Value,
    profile: ImageProfile,
) -> Result<(), InvalidInput> {
    // Compare typed configs so omitted serde defaults and explicit default
    // values remain equivalent. All behavior-affecting fields participate,
    // including future typed precision selectors, PLE and MoE geometry.
    let actual: DecoderConfig =
        serde_json::from_value(raw.clone()).map_err(|_| InvalidInput("invalid Gemma image decoder configuration"))?;
    let expected = expected_decoder(profile)?;
    if actual == expected {
        return Ok(());
    }
    if profile.model_dim == 1536 {
        // The originally qualified E2B artifact predates both embedding
        // materialization selectors. Retain that exact legacy policy as a
        // separate accepted contract; its artifact identity/evidence stay distinct.
        let mut legacy =
            serde_json::to_value(expected).map_err(|_| InvalidInput("invalid legacy Gemma image decoder profile"))?;
        legacy["embedding_config"]["materialize_dequant"] = false.into();
        legacy["embedding_config"]["materialize_soft_cap"] = false.into();
        let legacy: DecoderConfig =
            serde_json::from_value(legacy).map_err(|_| InvalidInput("invalid legacy Gemma image decoder profile"))?;
        if actual == legacy {
            return Ok(());
        }
    }
    Err(InvalidInput("Gemma image decoder differs from its pinned canonical profile"))
}
