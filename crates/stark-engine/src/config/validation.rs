use super::{rope::AnyRoPEConfig, token_mixer::attention::AttentionConfig, transformer::TransformerConfig};
use crate::backends::common::InvalidInput;

pub(crate) fn validate_mamba_state_width(
    backend: &str,
    width: u32,
) -> Result<(), InvalidInput> {
    if width == 0 || (backend == "metal" && width > 256) {
        return Err(InvalidInput("unsupported backend state dimension"));
    }
    Ok(())
}

pub(crate) fn validate_pooling_width(
    output: u32,
    pooling: u32,
) -> Result<(), InvalidInput> {
    if output == 0 || output != pooling {
        return Err(InvalidInput("pooling width must match transformer output"));
    }
    Ok(())
}

pub(crate) fn validate_draft_geometry(config: &super::dflash::DFlashDraftConfig) -> Result<(), InvalidInput> {
    use super::token_mixer::AnyTokenMixerConfig;
    let first = config.layer_configs.first().ok_or(InvalidInput("DFlash requires attention layers"))?;
    let AnyTokenMixerConfig::AttentionConfig(owner) = &first.mixer_config else {
        return Err(InvalidInput("DFlash layers must use attention mixers"));
    };
    for layer in &config.layer_configs {
        let AnyTokenMixerConfig::AttentionConfig(attention) = &layer.mixer_config else {
            return Err(InvalidInput("DFlash layers must use attention mixers"));
        };
        validate_attention(attention, Some(&config.rope_config))?;
        if attention.num_groups != owner.num_groups
            || attention.head_dim != owner.head_dim
            || attention.is_kv_sharing
            || layer.kv_source_layer_index.is_some()
        {
            return Err(InvalidInput("DFlash projected KV geometry differs across layers"));
        }
    }
    let layers = u32::try_from(config.layer_configs.len()).map_err(|_| InvalidInput("too many draft layers"))?;
    checked_product(&[2, owner.num_groups, owner.head_dim, layers])?;
    checked_product(&[
        config.model_dim,
        u32::try_from(config.target_layer_ids.len()).map_err(|_| InvalidInput("too many target features"))?,
    ])?;
    if config.block_size == 0
        || config.block_size > crate::encodable_block::mixer::attention::ATTENTION_SUFFIX_CAPACITY
        || config.mask_token_id >= u64::from(config.vocab_size)
    {
        return Err(InvalidInput("invalid draft token or block size"));
    }
    Ok(())
}

pub(crate) fn validate_target_geometry(
    config: &super::speculator::dflash::DFlashSpeculatorConfig,
    vocabulary: u32,
    width: u32,
) -> Result<(), InvalidInput> {
    if vocabulary == 0
        || width == 0
        || vocabulary != config.draft_config.vocab_size
        || width != config.draft_config.model_dim
        || config
            .weaver_config
            .as_ref()
            .is_some_and(|weaver| weaver.target_embedding_dim != width || weaver.target_model_dim != width)
    {
        return Err(InvalidInput("draft/target vocabulary or embedding geometry differs"));
    }
    Ok(())
}

pub(crate) fn validate_ple_layout(
    producer: Option<&super::per_layer_embedding::PLEModelConfig>,
    layers: &[super::transformer_layer::TransformerLayerConfig],
) -> Result<(), InvalidInput> {
    if let Some(producer) = producer {
        if producer.ple_dim == 0 || producer.num_layers as usize != layers.len() {
            return Err(InvalidInput("invalid per-layer embedding layout"));
        }
        checked_product(&[producer.num_layers, producer.ple_dim])?;
    }
    for layer in layers {
        if let Some(consumer) = &layer.ple_config
            && producer.is_none_or(|producer| consumer.ple_dim == 0 || consumer.ple_dim != producer.ple_dim)
        {
            return Err(InvalidInput("per-layer embedding has no compatible producer"));
        }
    }
    Ok(())
}

pub(crate) fn checked_product(dimensions: &[u32]) -> Result<u32, InvalidInput> {
    dimensions.iter().try_fold(1u32, |size, &dimension| {
        if dimension == 0 {
            return Err(InvalidInput("zero dimension"));
        }
        size.checked_mul(dimension).ok_or(InvalidInput("dimension product overflow"))
    })
}

pub(crate) fn validate_heads(
    heads: u32,
    groups: u32,
    head_dim: u32,
) -> Result<(), InvalidInput> {
    if heads == 0 || groups == 0 || head_dim == 0 || heads < groups || !heads.is_multiple_of(groups) {
        return Err(InvalidInput("head counts must be positive and exactly divisible"));
    }
    checked_product(&[heads, head_dim])?;
    checked_product(&[groups, head_dim])?;
    Ok(())
}

pub(crate) fn validate_attention(
    config: &AttentionConfig,
    rope: Option<&AnyRoPEConfig>,
) -> Result<(), InvalidInput> {
    validate_heads(config.num_heads, config.num_groups, config.head_dim)?;
    if let Some(rope) = rope {
        rope.validate()?;
        let width = *rope.head_dim();
        if width == 0 || !width.is_multiple_of(2) || width > config.head_dim || *rope.max_sequence_length() == 0 {
            return Err(InvalidInput("rotary width must be positive, even and fit the attention head"));
        }
    }
    if let Some(window) = config.sliding_window_size
        && (window == 0
            || window.checked_add(crate::encodable_block::mixer::attention::ATTENTION_SUFFIX_CAPACITY).is_none())
    {
        return Err(InvalidInput("invalid sliding window capacity"));
    }
    Ok(())
}

pub(crate) fn validate_transformer(config: &TransformerConfig) -> Result<(), InvalidInput> {
    if config.layer_configs.is_empty() || u32::try_from(config.layer_configs.len()).is_err() {
        return Err(InvalidInput("transformer must contain a representable nonempty layer array"));
    }
    checked_product(&[config.model_dim, config.hidden_dim])?;
    for (index, layer) in config.layer_configs.iter().enumerate() {
        use super::token_mixer::AnyTokenMixerConfig;
        if layer.post_norm
            && (layer.pre_mixer_norm_config.is_some()
                || layer.pre_mlp_norm_config.has_scale
                || layer.pre_mlp_norm_config.has_biases
                || layer.post_mixer_norm_config.is_none()
                || layer.post_mlp_norm_config.is_none()
                || layer.mixer_conv_config.is_some()
                || layer.mlp_conv_config.is_some()
                || layer.ple_config.is_some()
                || layer.has_post_layer_scalar)
        {
            return Err(InvalidInput("post-norm layers require branch norms and no pre-norm transforms"));
        }
        // Transformer table collection includes every declared RoPE config,
        // including metadata attached to a non-attention mixer.
        if let Some(rope) = &layer.rope_config {
            rope.validate()?;
        }
        if let AnyTokenMixerConfig::AttentionConfig(attention) = &layer.mixer_config {
            validate_attention(attention, layer.rope_config.as_ref())?;
            if attention.is_kv_sharing != layer.kv_source_layer_index.is_some() {
                return Err(InvalidInput("shared attention must name an owned cache"));
            }
        } else if layer.kv_source_layer_index.is_some() {
            return Err(InvalidInput("only attention layers may share caches"));
        }
        if let Some(source_index) = layer.kv_source_layer_index {
            let source = config
                .layer_configs
                .get(source_index as usize)
                .filter(|_| (source_index as usize) < index)
                .ok_or(InvalidInput("shared cache owner must precede its consumer"))?;
            let (AnyTokenMixerConfig::AttentionConfig(owner), AnyTokenMixerConfig::AttentionConfig(consumer)) =
                (&source.mixer_config, &layer.mixer_config)
            else {
                return Err(InvalidInput("shared cache owner must use attention"));
            };
            if source.kv_source_layer_index.is_some()
                || owner.is_kv_sharing
                || owner.num_groups != consumer.num_groups
                || owner.head_dim != consumer.head_dim
                || owner.sliding_window_size != consumer.sliding_window_size
                || source.rope_config != layer.rope_config
                || owner.is_causal != consumer.is_causal
            {
                return Err(InvalidInput("incompatible shared cache geometry or positional semantics"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    fn normalization() -> super::super::normalization::NormalizationConfig {
        super::super::normalization::NormalizationConfig {
            epsilon: 1e-6,
            scale_offset: None,
            upcast_mode: super::super::normalization::UpcastMode::FullLayer,
            subtract_mean: false,
            has_scale: false,
            has_biases: false,
        }
    }

    fn attention() -> AttentionConfig {
        serde_json::from_value(serde_json::json!({
            "type":"AttentionConfig", "qkvg_projection_config":{}, "out_projection_config":{},
            "query_norm_config":null, "key_norm_config":null, "num_heads":4, "num_groups":2,
            "head_dim":64, "is_causal":true, "scale":null, "sliding_window_size":null,
            "logit_soft_cap":null, "has_sinks":false, "has_qkvg_biases":false,
            "has_out_biases":false, "has_gate":false, "normalize_values":false, "is_kv_sharing":false
        }))
        .unwrap()
    }

    fn rope(width: u32) -> AnyRoPEConfig {
        serde_json::from_value(serde_json::json!({
            "type":"UnscaledRoPEConfig", "base":10000.0, "max_sequence_length":1024, "head_dim":width
        }))
        .unwrap()
    }

    fn layer(attention: AttentionConfig) -> super::super::transformer_layer::TransformerLayerConfig {
        use super::super::{
            linear::LinearConfig,
            mlp::{AnyMLPConfig, dense_mlp::DenseMLPConfig},
            token_mixer::AnyTokenMixerConfig,
            transformer_layer::TransformerLayerConfig,
        };
        TransformerLayerConfig {
            post_norm: false,
            pre_mixer_norm_config: Some(normalization()),
            mixer_config: AnyTokenMixerConfig::AttentionConfig(attention),
            mixer_conv_config: None,
            post_mixer_norm_config: None,
            pre_mlp_norm_config: normalization(),
            mlp_config: AnyMLPConfig::DenseMLPConfig(DenseMLPConfig::unclipped(
                LinearConfig {},
                serde_json::from_value(serde_json::json!({"type":"Identity"})).unwrap(),
                false,
                false,
            )),
            mlp_conv_config: None,
            post_mlp_norm_config: None,
            hidden_dim: None,
            ple_config: None,
            has_post_layer_scalar: false,
            kv_source_layer_index: None,
            rope_config: Some(rope(64)),
        }
    }

    fn transformer() -> TransformerConfig {
        TransformerConfig {
            layer_configs: vec![layer(attention())].into_boxed_slice(),
            output_norm_config: normalization(),
            model_dim: 128,
            hidden_dim: 256,
        }
    }

    fn draft() -> super::super::dflash::DFlashDraftConfig {
        super::super::dflash::DFlashDraftConfig {
            model_dim: 128,
            hidden_dim: 256,
            block_size: 16,
            mask_token_id: 0,
            target_layer_ids: vec![0].into_boxed_slice(),
            num_target_layers: 1,
            vocab_size: 128,
            context_projection_config: super::super::linear::LinearConfig {},
            context_norm_config: normalization(),
            rope_config: rope(64),
            layer_configs: vec![layer(attention()), layer(attention())].into_boxed_slice(),
            output_norm_config: normalization(),
        }
    }

    #[stark_test]
    fn grouped_heads_reject_truncated_ratios_and_overflow() {
        for (heads, groups, width) in [(3, 2, 64), (1, 2, 64), (0, 1, 64), (1, 0, 64), (1, 1, 0), (u32::MAX, 1, 64)] {
            assert!(validate_heads(heads, groups, width).is_err());
        }
        assert!(validate_heads(32, 8, 128).is_ok());
        assert!(validate_heads(8, 8, 64).is_ok());
    }

    #[stark_test]
    fn mamba_state_width_preserves_cpu_support_and_rejects_metal_no_write_dispatch() {
        for width in [1, 64, 128, 256] {
            assert!(validate_mamba_state_width("metal", width).is_ok());
        }
        for width in [257, 512, u32::MAX] {
            assert!(validate_mamba_state_width("metal", width).is_err());
            assert!(validate_mamba_state_width("cpu", width).is_ok());
        }
        assert!(validate_mamba_state_width("metal", 0).is_err());
        assert!(validate_mamba_state_width("cpu", 0).is_err());
    }

    #[stark_test]
    fn products_reject_wrap_in_release_as_well_as_debug() {
        assert!(checked_product(&[u32::MAX, 2]).is_err());
        assert!(checked_product(&[10, 0]).is_err());
        assert_eq!(checked_product(&[16, 128]).unwrap(), 2048);
    }

    #[stark_test]
    fn rotary_width_is_checked_before_native_projection() {
        for width in [0, 1, 65, 128, 32768] {
            assert!(validate_attention(&attention(), Some(&rope(width))).is_err());
        }
        for width in [2, 32, 64] {
            assert!(validate_attention(&attention(), Some(&rope(width))).is_ok());
        }
        assert!(validate_attention(&attention(), None).is_ok());
    }

    #[stark_test]
    fn qwen35_partial_rotary_width_validates_sections_before_native_projection() {
        let mut config = attention();
        config.head_dim = 256;
        let mut rope: AnyRoPEConfig = serde_json::from_value(serde_json::json!({
            "type":"Qwen35MRoPEConfig", "base":10000000.0, "max_sequence_length":262144,
            "head_dim":64, "mrope_section":[11,11,10], "interleaved":true
        }))
        .unwrap();
        assert!(validate_attention(&config, Some(&rope)).is_ok());
        let AnyRoPEConfig::Qwen35MRoPEConfig(mrope) = &mut rope else {
            unreachable!()
        };
        mrope.mrope_section = [10, 12, 10];
        assert!(validate_attention(&config, Some(&rope)).is_err());
        let AnyRoPEConfig::Qwen35MRoPEConfig(mrope) = &mut rope else {
            unreachable!()
        };
        mrope.mrope_section = [11, 11, 10];
        config.head_dim = 32;
        assert!(validate_attention(&config, Some(&rope)).is_err());
    }

    #[stark_test]
    fn qwen35_non_attention_rotary_metadata_is_validated_before_table_collection() {
        let mut config = transformer();
        config.layer_configs[0].mixer_config = serde_json::from_value(serde_json::json!({
            "type":"DeltaNetConfig", "in_proj_config":{}, "conv_config":{"has_biases":false},
            "out_proj_config":{}, "norm_config":normalization(), "num_heads":4,
            "num_groups":2, "head_dim":64, "value_head_dim":64, "kernel_size":4
        }))
        .unwrap();
        let valid = serde_json::json!({
            "type":"Qwen35MRoPEConfig", "base":10000000.0, "max_sequence_length":262144,
            "head_dim":64, "mrope_section":[11,11,10], "interleaved":true
        });
        config.layer_configs[0].rope_config = Some(serde_json::from_value(valid.clone()).unwrap());
        assert!(validate_transformer(&config).is_ok());
        for (field, value) in [
            ("head_dim", serde_json::json!(0)),
            ("base", serde_json::json!(0.0)),
            ("interleaved", serde_json::json!(false)),
            ("mrope_section", serde_json::json!([10, 12, 10])),
        ] {
            let mut malformed = valid.clone();
            malformed[field] = value;
            config.layer_configs[0].rope_config = Some(serde_json::from_value(malformed).unwrap());
            assert!(validate_transformer(&config).is_err(), "non-attention RoPE accepted malformed {field}");
        }
    }

    #[stark_test]
    fn sliding_window_reserves_suffix_without_wrap() {
        let mut config = attention();
        for window in [0, u32::MAX, u32::MAX - 511] {
            config.sliding_window_size = Some(window);
            assert!(validate_attention(&config, None).is_err());
        }
        config.sliding_window_size = Some(1024);
        assert!(validate_attention(&config, None).is_ok());
    }

    #[stark_test]
    fn empty_transformer_cannot_reach_uninitialized_shortcut() {
        let mut config = transformer();
        assert!(validate_transformer(&config).is_ok());
        config.layer_configs = Box::new([]);
        assert!(validate_transformer(&config).is_err());
    }

    #[stark_test]
    fn olmo_post_norm_requires_explicit_branch_norms_and_no_pre_norm_parameters() {
        let mut config = transformer();
        let layer = &mut config.layer_configs[0];
        layer.post_norm = true;
        layer.pre_mixer_norm_config = None;
        layer.post_mixer_norm_config = Some(normalization());
        layer.post_mlp_norm_config = Some(normalization());
        assert!(validate_transformer(&config).is_ok());
        config.layer_configs[0].pre_mixer_norm_config = Some(normalization());
        assert!(validate_transformer(&config).is_err());
        config.layer_configs[0].pre_mixer_norm_config = None;
        config.layer_configs[0].pre_mlp_norm_config.has_scale = true;
        assert!(validate_transformer(&config).is_err());
        config.layer_configs[0].pre_mlp_norm_config.has_scale = false;
        config.layer_configs[0].post_mixer_norm_config = None;
        assert!(validate_transformer(&config).is_err());
        config.layer_configs[0].post_mixer_norm_config = Some(normalization());
        config.layer_configs[0].post_mlp_norm_config = None;
        assert!(validate_transformer(&config).is_err());
    }

    #[stark_test]
    fn attention_projection_norm_scope_defaults_to_per_head_and_roundtrips_explicit_scope() {
        let mut config = attention();
        assert!(!config.qk_norm_full_width);
        config.qk_norm_full_width = true;
        let serialized = serde_json::to_value(&config).unwrap();
        assert_eq!(serialized["qk_norm_full_width"], true);
        let decoded: AttentionConfig = serde_json::from_value(serialized).unwrap();
        assert!(decoded.qk_norm_full_width);
    }

    #[stark_test]
    fn shared_cache_requires_compatible_earlier_owned_layer() {
        let mut config = transformer();
        let mut consumer = attention();
        consumer.is_kv_sharing = true;
        let mut shared = layer(consumer.clone());
        shared.kv_source_layer_index = Some(0);
        config.layer_configs = vec![layer(attention()), shared.clone()].into_boxed_slice();
        assert!(validate_transformer(&config).is_ok());
        consumer.num_groups = 1;
        config.layer_configs[1].mixer_config =
            super::super::token_mixer::AnyTokenMixerConfig::AttentionConfig(consumer.clone());
        assert!(validate_transformer(&config).is_err());
        consumer.num_groups = 2;
        consumer.head_dim = 128;
        config.layer_configs[1].mixer_config =
            super::super::token_mixer::AnyTokenMixerConfig::AttentionConfig(consumer);
        assert!(validate_transformer(&config).is_err());
        config.layer_configs[1] = shared.clone();
        config.layer_configs[1].kv_source_layer_index = Some(1);
        assert!(validate_transformer(&config).is_err());
        config.layer_configs[1] = shared;
        config.layer_configs[0].kv_source_layer_index = Some(0);
        assert!(validate_transformer(&config).is_err());
    }

    #[stark_test]
    fn classifier_pooling_rejects_independently_wider_reads() {
        for pooling in [0, 128, 1024] {
            assert!(validate_pooling_width(64, pooling).is_err());
        }
        assert!(validate_pooling_width(64, 64).is_ok());
        assert!(validate_pooling_width(128, 128).is_ok());
    }

    #[stark_test]
    fn per_layer_embedding_requires_the_same_producer_stride() {
        use super::super::{
            linear::LinearConfig,
            per_layer_embedding::{PLELayerConfig, PLEModelConfig},
        };
        let mut layers = vec![layer(attention())];
        layers[0].ple_config = Some(PLELayerConfig {
            linear_config: LinearConfig {},
            norm_config: normalization(),
            ple_dim: 64,
            activation: serde_json::from_value(serde_json::json!({"type":"Identity"})).unwrap(),
            reference_bf16: false,
        });
        let mut producer = PLEModelConfig {
            ple_dim: 32,
            num_layers: 1,
            ple_vocab_size: 128,
            ple_embed_scale: 1.0,
            model_projection_scale: 1.0,
            input_scale: 1.0,
            linear_config: LinearConfig {},
            norm_config: normalization(),
            reference_bf16: false,
        };
        assert!(validate_ple_layout(None, &layers).is_err());
        assert!(validate_ple_layout(Some(&producer), &layers).is_err());
        producer.ple_dim = 64;
        assert!(validate_ple_layout(Some(&producer), &layers).is_ok());
        producer.num_layers = 2;
        assert!(validate_ple_layout(Some(&producer), &layers).is_err());
    }

    #[stark_test]
    fn draft_projection_rejects_mixed_kv_geometry() {
        let mut config = draft();
        assert!(validate_draft_geometry(&config).is_ok());
        let super::super::token_mixer::AnyTokenMixerConfig::AttentionConfig(attention) =
            &mut config.layer_configs[1].mixer_config
        else {
            unreachable!()
        };
        attention.num_groups = 4;
        assert!(validate_draft_geometry(&config).is_err());
        config.layer_configs = Box::new([]);
        assert!(validate_draft_geometry(&config).is_err());
    }

    #[stark_test]
    fn speculator_is_bound_to_actual_target_vocabulary_and_embedding() {
        use super::super::{linear::LinearConfig, speculator::dflash::DFlashSpeculatorConfig, weaver::WeaverConfig};
        let weaver = WeaverConfig {
            model_dim: 128,
            target_model_dim: 128,
            target_embedding_dim: 128,
            num_layers: 1,
            num_heads: 2,
            hidden_dim: 256,
            max_depth: 8,
            candidate_pool_size: 16,
            linear_config: LinearConfig {},
            norm_config: normalization(),
            rope_config: rope(64),
        };
        let mut config: DFlashSpeculatorConfig = serde_json::from_value(serde_json::json!({
            "type":"DFlashSpeculatorConfig", "draft_config":draft(), "weaver_config":weaver
        }))
        .unwrap();
        assert!(validate_target_geometry(&config, 128, 128).is_ok());
        assert!(validate_target_geometry(&config, 64, 128).is_err());
        assert!(validate_target_geometry(&config, 128, 64).is_err());
        config.weaver_config.as_mut().unwrap().target_embedding_dim = 256;
        assert!(validate_target_geometry(&config, 128, 128).is_err());
        config.weaver_config = None;
        assert!(validate_target_geometry(&config, 128, 128).is_ok());
    }
}
