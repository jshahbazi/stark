//! Bounded config-derived geometry for merged dense Qwen3.5 BF16 artifacts.

use serde_json::{Value, json};

use super::{TensorContract, TensorMap, decoder_profile, invalid, vision_profile};
use crate::{backends::common::InvalidInput, engine::vision::VisionTowerConfig};

pub(super) const MAX_WEIGHT_BYTES: u64 = 128 * 1024 * 1024 * 1024;

pub(super) struct Geometry {
    pub tower: VisionTowerConfig,
    pub vocab: u32,
    pub tensors: TensorContract,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mixer {
    Attention {
        heads: u32,
        groups: u32,
    },
    Delta {
        heads: u32,
        groups: u32,
    },
}

fn number(
    value: &Value,
    maximum: u32,
    alignment: u32,
) -> Result<u32, InvalidInput> {
    value
        .as_u64()
        .and_then(|number| u32::try_from(number).ok())
        .filter(|number| *number > 0 && *number <= maximum && number.is_multiple_of(alignment))
        .ok_or_else(invalid)
}

fn heads(value: &Value) -> Result<(u32, u32), InvalidInput> {
    let heads = number(&value["num_heads"], 256, 1)?;
    let groups = number(&value["num_groups"], 256, 1)?;
    if !heads.is_multiple_of(groups) {
        return Err(invalid());
    }
    Ok((heads, groups))
}

/// Reconstruct the supported operation graph from bounded scalar dimensions,
/// then require exact equality. Unknown flags cannot select extra operations.
pub(super) fn geometry(
    decoder: &Value,
    vision: &Value,
) -> Result<Geometry, InvalidInput> {
    let transformer = &decoder["transformer_config"];
    let model = number(&transformer["model_dim"], 16384, 32)?;
    let hidden = number(&transformer["hidden_dim"], 131072, 32)?;
    let vocab = number(&decoder["vocab_size"], 1048576, 1)?;
    let layers = transformer["layer_configs"]
        .as_array()
        .filter(|layers| !layers.is_empty() && layers.len() <= 256)
        .ok_or_else(invalid)?;
    let epsilon = &transformer["output_norm_config"]["epsilon"];
    let epsilon_number =
        epsilon.as_f64().filter(|number| number.is_finite() && *number > 0.0 && *number <= 1e-2).ok_or_else(invalid)?;
    if !(epsilon_number as f32).is_finite() || epsilon_number as f32 == 0.0 {
        return Err(invalid());
    }
    let tied = match decoder["embedding_config"]["type"].as_str() {
        Some("TiedEmbeddingConfig") => true,
        Some("UntiedEmbeddingConfig") => false,
        _ => return Err(invalid()),
    };
    let mut expected = decoder_profile();
    expected["vocab_size"] = json!(vocab);
    expected["embedding_config"]["type"] = json!(if tied {
        "TiedEmbeddingConfig"
    } else {
        "UntiedEmbeddingConfig"
    });
    expected["transformer_config"]["model_dim"] = json!(model);
    expected["transformer_config"]["hidden_dim"] = json!(hidden);
    expected["transformer_config"]["output_norm_config"]["epsilon"] = epsilon.clone();
    let baseline_layers = decoder_profile()["transformer_config"]["layer_configs"].as_array().unwrap().clone();
    let mut expected_layers = Vec::with_capacity(layers.len());
    let mut mixers = Vec::with_capacity(layers.len());
    let (mut attention_geometry, mut delta_geometry, mut context_geometry) = (None, None, None);
    for layer in layers {
        let mixer = &layer["mixer_config"];
        let (head_count, group_count) = heads(mixer)?;
        let attention = match mixer["type"].as_str() {
            Some("AttentionConfig") => true,
            Some("DeltaNetConfig") => false,
            _ => return Err(invalid()),
        };
        let geometry = (head_count, group_count);
        let previous = if attention {
            &mut attention_geometry
        } else {
            &mut delta_geometry
        };
        if previous.is_some_and(|previous| previous != geometry) {
            return Err(invalid());
        }
        *previous = Some(geometry);
        let mut expected_layer = baseline_layers[if attention {
            3
        } else {
            0
        }]
        .clone();
        for key in ["pre_mixer_norm_config", "pre_mlp_norm_config"] {
            expected_layer[key]["epsilon"] = epsilon.clone();
        }
        expected_layer["mixer_config"]["num_heads"] = json!(head_count);
        expected_layer["mixer_config"]["num_groups"] = json!(group_count);
        if attention {
            for key in ["query_norm_config", "key_norm_config"] {
                expected_layer["mixer_config"][key]["epsilon"] = epsilon.clone();
            }
            let context = number(&layer["rope_config"]["max_sequence_length"], 1048576, 1)?;
            if context_geometry.is_some_and(|previous| previous != context) {
                return Err(invalid());
            }
            context_geometry = Some(context);
            expected_layer["rope_config"]["max_sequence_length"] = json!(context);
            mixers.push(Mixer::Attention {
                heads: head_count,
                groups: group_count,
            });
        } else {
            expected_layer["mixer_config"]["norm_config"]["epsilon"] = epsilon.clone();
            mixers.push(Mixer::Delta {
                heads: head_count,
                groups: group_count,
            });
        }
        expected_layers.push(expected_layer);
    }
    if attention_geometry.is_none() || delta_geometry.is_none() {
        return Err(invalid());
    }
    expected["transformer_config"]["layer_configs"] = json!(expected_layers);
    if *decoder != expected {
        return Err(invalid());
    }

    let mut expected_vision = vision_profile();
    let mut actual_vision = vision.clone();
    if let Some(initializer) = actual_vision.as_object_mut().ok_or_else(invalid)?.remove("initializer_range") {
        if !initializer.as_f64().is_some_and(|number| number.is_finite() && number >= 0.0) {
            return Err(invalid());
        }
    }
    expected_vision.as_object_mut().unwrap().remove("initializer_range");
    let depth = number(&vision["depth"], 64, 1)?;
    let width = number(&vision["hidden_size"], 4096, 1)?;
    let intermediate = number(&vision["intermediate_size"], 32768, 1)?;
    let vision_heads = number(&vision["num_heads"], 64, 1)?;
    let positions = number(&vision["num_position_embeddings"], 16384, 1)?;
    for (name, value) in [
        ("depth", depth),
        ("hidden_size", width),
        ("intermediate_size", intermediate),
        ("num_heads", vision_heads),
        ("num_position_embeddings", positions),
        ("out_hidden_size", model),
    ] {
        expected_vision[name] = json!(value);
    }
    if actual_vision != expected_vision {
        return Err(invalid());
    }
    let tower = VisionTowerConfig {
        depth,
        hidden_size: width,
        intermediate_size: intermediate,
        num_heads: vision_heads,
        num_position_embeddings: positions,
        out_hidden_size: model,
        patch_size: 16,
        spatial_merge_size: 2,
        temporal_patch_size: 2,
        in_channels: 3,
        layer_norm_eps: 1e-6,
    };
    tower.validate()?;
    let tensors = tensor_contract(model, hidden, vocab, tied, &mixers, &tower)?;
    Ok(Geometry {
        tower,
        vocab,
        tensors,
    })
}

fn tensor_contract(
    model: u32,
    hidden: u32,
    vocab: u32,
    tied: bool,
    mixers: &[Mixer],
    tower: &VisionTowerConfig,
) -> Result<TensorContract, InvalidInput> {
    let mut tensors = TensorMap::new();
    let mut bytes = 0u64;
    let mut add = |name: String, dtype, shape: &[u32]| -> Result<(), InvalidInput> {
        let count = shape.iter().try_fold(1u32, |count, width| count.checked_mul(*width)).ok_or_else(invalid)?;
        bytes = bytes
            .checked_add(
                u64::from(count)
                    * if dtype == "F32" {
                        4
                    } else {
                        2
                    },
            )
            .ok_or_else(invalid)?;
        if bytes > MAX_WEIGHT_BYTES || count == 0 || tensors.insert(name, (dtype, shape.to_vec())).is_some() {
            return Err(invalid());
        }
        Ok(())
    };
    if tied {
        add("decoder.embedding.embedding.weights".into(), "BF16", &[vocab, model])?;
    } else {
        add("decoder.embedding.input_embedding.weights".into(), "BF16", &[vocab, model])?;
        add("decoder.embedding.output_embedding.weights".into(), "BF16", &[vocab, model])?;
    }
    add("decoder.transformer.output_norm.scales".into(), "F32", &[model])?;
    for (index, mixer) in mixers.iter().enumerate() {
        let prefix = format!("decoder.transformer.layers.{index}");
        for name in ["pre_mixer_norm.scales", "pre_mlp_norm.scales"] {
            add(format!("{prefix}.{name}"), "F32", &[model])?;
        }
        add(format!("{prefix}.mlp.up_projection.weights.weights"), "BF16", &[2 * hidden, model])?;
        add(format!("{prefix}.mlp.down_projection.weights.weights"), "BF16", &[model, hidden])?;
        match *mixer {
            Mixer::Attention {
                heads,
                groups,
            } => {
                for name in ["query_norm", "key_norm"] {
                    add(format!("{prefix}.mixer.{name}.scales"), "F32", &[256])?;
                }
                add(
                    format!("{prefix}.mixer.qkvg_projection.weights.weights"),
                    "BF16",
                    &[2 * (heads + groups) * 256, model],
                )?;
                add(format!("{prefix}.mixer.out_projection.weights.weights"), "BF16", &[model, heads * 256])?;
            },
            Mixer::Delta {
                heads,
                groups,
            } => {
                let keys = groups * 128;
                let values = heads * 128;
                for name in ["a_log", "dt_bias"] {
                    add(format!("{prefix}.mixer.{name}"), "F32", &[heads])?;
                }
                add(format!("{prefix}.mixer.conv.weights"), "F32", &[2 * keys + values, 4])?;
                add(format!("{prefix}.mixer.norm.scales"), "F32", &[128])?;
                add(
                    format!("{prefix}.mixer.in_proj.weights.weights"),
                    "BF16",
                    &[2 * keys + 2 * values + 2 * heads, model],
                )?;
                add(format!("{prefix}.mixer.out_proj.weights.weights"), "BF16", &[model, values])?;
            },
        }
    }
    let width = tower.hidden_size;
    add("vision.patch_embed.proj.weight".into(), "BF16", &[width, 3, 2, 16, 16])?;
    add("vision.patch_embed.proj.bias".into(), "BF16", &[width])?;
    add("vision.pos_embed.weight".into(), "BF16", &[tower.num_position_embeddings, width])?;
    for index in 0..tower.depth {
        let prefix = format!("vision.blocks.{index}");
        for norm in ["norm1", "norm2"] {
            for suffix in ["weight", "bias"] {
                add(format!("{prefix}.{norm}.{suffix}"), "BF16", &[width])?;
            }
        }
        for (name, output, input) in [
            ("attn.qkv", 3 * width, width),
            ("attn.proj", width, width),
            ("mlp.linear_fc1", tower.intermediate_size, width),
            ("mlp.linear_fc2", width, tower.intermediate_size),
        ] {
            add(format!("{prefix}.{name}.weight"), "BF16", &[output, input])?;
            add(format!("{prefix}.{name}.bias"), "BF16", &[output])?;
        }
    }
    for suffix in ["weight", "bias"] {
        add(format!("vision.merger.norm.{suffix}"), "BF16", &[width])?;
    }
    for (name, output, input) in [("linear_fc1", 4 * width, 4 * width), ("linear_fc2", model, 4 * width)] {
        add(format!("vision.merger.{name}.weight"), "BF16", &[output, input])?;
        add(format!("vision.merger.{name}.bias"), "BF16", &[output])?;
    }
    Ok(TensorContract {
        tensors,
    })
}

#[cfg(test)]
#[path = "dynamic_tests.rs"]
mod tests;
