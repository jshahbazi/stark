use crate::{
    backends::common::{Backend, CommandBuffer, CommandBufferEncoding, InvalidInput},
    config::rope::AnyRoPEConfig,
};

pub struct PrecalculatedRoPE<B: Backend> {
    pub cosines: B::ConstantBuffer,
    pub sines: B::ConstantBuffer,
    pub dim: u32,
}

/// Model-owned CPU constants only: no tokens, sequence identity or GPU storage.
/// Each instance is bound to a clone of one complete immutable RoPE config.
#[cfg(feature = "rope-cache")]
pub struct RoPETableCache {
    config: AnyRoPEConfig,
    max_cached_positions: usize,
    tables: parking_lot::Mutex<(Vec<f32>, Vec<f32>)>,
}

#[cfg(feature = "rope-cache")]
impl RoPETableCache {
    pub fn new(config: &AnyRoPEConfig) -> Self {
        let head_dim = *config.head_dim() as usize;
        assert!(head_dim > 0 && head_dim.is_multiple_of(2), "RoPE head_dim must be positive and even");
        // At most 8 MiB of retained f32 values per distinct configuration.
        // Dense rows represent mathematical absolute positions, not requests.
        let max_cached_positions = (*config.max_sequence_length() as usize).min(8192).min(1_048_576 / head_dim);
        Self {
            config: config.clone(),
            max_cached_positions,
            tables: parking_lot::Mutex::new((Vec::new(), Vec::new())),
        }
    }

    pub fn precalculate<B: Backend>(
        &self,
        token_positions: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<PrecalculatedRoPE<B>, B::Error> {
        self.with_tables(token_positions, |cosines, sines| {
            Ok(PrecalculatedRoPE {
                cosines: command_buffer.allocate_constant_from_slice(cosines)?,
                sines: command_buffer.allocate_constant_from_slice(sines)?,
                dim: *self.config.head_dim(),
            })
        })
    }

    /// Request-specific spatial rows must never enter the model-owned scalar cache.
    pub fn precalculate_multimodal<B: Backend>(
        &self,
        positions: &[[u32; 3]],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<PrecalculatedRoPE<B>, B::Error> {
        PrecalculatedRoPE::<B>::precalculate_multimodal(&self.config, positions, command_buffer)
    }

    fn with_tables<R>(
        &self,
        token_positions: &[u32],
        use_tables: impl FnOnce(&[f32], &[f32]) -> R,
    ) -> R {
        let head_dim = *self.config.head_dim() as usize;
        let required_positions = token_positions
            .iter()
            .map(|position| *position as usize)
            .filter(|position| *position < self.max_cached_positions)
            .max()
            .map_or(0, |position| position + 1);
        let mut tables = self.tables.lock();
        let cached_positions = tables.0.len() / head_dim;
        if required_positions > cached_positions {
            let positions = (cached_positions as u32..required_positions as u32).collect::<Vec<_>>();
            let (cosines, sines) = precalculate_cpu_tables(&self.config, &positions);
            // Rebuild into exact-length vectors so the retained allocation stays
            // within the configured cap instead of Vec's geometric spare capacity.
            let mut new_cosines = Vec::with_capacity(required_positions * head_dim);
            let mut new_sines = Vec::with_capacity(required_positions * head_dim);
            new_cosines.extend_from_slice(&tables.0);
            new_sines.extend_from_slice(&tables.1);
            new_cosines.extend_from_slice(&cosines);
            new_sines.extend_from_slice(&sines);
            *tables = (new_cosines, new_sines);
        }

        // Flat scoring chunks can upload the retained rows without temporary
        // gather vectors. Tree/reordered/out-of-cap positions keep caller order.
        if let (Some(first), Some(last)) = (token_positions.first(), token_positions.last()) {
            let contiguous = token_positions.windows(2).all(|pair| pair[0].checked_add(1) == Some(pair[1]));
            if contiguous && (*last as usize) < required_positions {
                let start = *first as usize * head_dim;
                let end = (*last as usize + 1) * head_dim;
                return use_tables(&tables.0[start..end], &tables.1[start..end]);
            }
        }

        let mut cosines = Vec::with_capacity(token_positions.len() * head_dim);
        let mut sines = Vec::with_capacity(token_positions.len() * head_dim);
        for position in token_positions {
            let position_index = *position as usize;
            if position_index < required_positions {
                let start = position_index * head_dim;
                cosines.extend_from_slice(&tables.0[start..start + head_dim]);
                sines.extend_from_slice(&tables.1[start..start + head_dim]);
            } else {
                let (row_cosines, row_sines) = precalculate_cpu_tables(&self.config, &[*position]);
                cosines.extend_from_slice(&row_cosines);
                sines.extend_from_slice(&row_sines);
            }
        }
        use_tables(&cosines, &sines)
    }
}

impl<B: Backend> PrecalculatedRoPE<B> {
    /// Explicit image/text positions are request-owned and bypass the scalar table cache.
    pub fn precalculate_multimodal(
        rope_config: &AnyRoPEConfig,
        positions: &[[u32; 3]],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Self, B::Error> {
        let (cosines, sines) = precalculate_multimodal_cpu_tables(rope_config, positions)?;
        Ok(Self {
            cosines: command_buffer.allocate_constant_from_slice(&cosines)?,
            sines: command_buffer.allocate_constant_from_slice(&sines)?,
            dim: *rope_config.head_dim(),
        })
    }

    pub fn precalculate(
        rope_config: &AnyRoPEConfig,
        token_positions: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Self, B::Error> {
        let (cosines, sines) = precalculate_cpu_tables(rope_config, token_positions);

        Ok(Self {
            cosines: command_buffer.allocate_constant_from_slice(&cosines)?,
            sines: command_buffer.allocate_constant_from_slice(&sines)?,
            dim: *rope_config.head_dim(),
        })
    }
}

// Keep every formula, cast and operation order identical to the uncached path.
fn precalculate_cpu_tables(
    rope_config: &AnyRoPEConfig,
    token_positions: &[u32],
) -> (Vec<f32>, Vec<f32>) {
    let head_dim = *rope_config.head_dim();
    assert!(head_dim > 0 && head_dim.is_multiple_of(2), "RoPE head_dim must be positive and even");
    let half_dim = head_dim / 2;
    let attention_scaling_factor = match rope_config {
        AnyRoPEConfig::YARNRoPEConfig(config) => 0.1 * config.scaling_factor.ln() + 1.0,
        AnyRoPEConfig::LongRoPEConfig(config) if config.scaling_factor > 1.0 => {
            (1.0 + config.scaling_factor.ln() / (config.original_context_length as f32).ln()).sqrt()
        },
        _ => 1.0,
    };

    let element_count = token_positions.len() * head_dim as usize;
    let half_offset = half_dim as usize;
    let mut sines = vec![0.0; element_count];
    let mut cosines = vec![0.0; element_count];
    for pair_index in 0..half_dim {
        let channel_index = pair_index * 2;
        let inverse_frequency = 1.0 / rope_config.base().powf(channel_index as f32 / head_dim as f32);
        let inverse_frequency = match rope_config {
            AnyRoPEConfig::UnscaledRoPEConfig(_) | AnyRoPEConfig::Qwen35MRoPEConfig(_) => inverse_frequency,
            AnyRoPEConfig::LinearScalingRoPEConfig(config) => inverse_frequency / config.scaling_factor,
            AnyRoPEConfig::LlamaRoPEConfig(config) => {
                let low_frequency_wavelength = config.original_context_length as f32 / config.low_frequency_factor;
                let high_frequency_wavelength = config.original_context_length as f32 / config.high_frequency_factor;
                let wavelength = 2.0 * std::f32::consts::PI / inverse_frequency;
                let scaled_frequency = inverse_frequency / config.scaling_factor;

                if wavelength < high_frequency_wavelength {
                    inverse_frequency
                } else if wavelength > low_frequency_wavelength {
                    scaled_frequency
                } else {
                    let smoothing_factor =
                        config.original_context_length as f32 / wavelength - config.low_frequency_factor;
                    let smoothing_factor =
                        smoothing_factor / (config.high_frequency_factor - config.low_frequency_factor);
                    smoothing_factor * inverse_frequency + (1.0 - smoothing_factor) * scaled_frequency
                }
            },
            AnyRoPEConfig::YARNRoPEConfig(config) => {
                let dim = config.head_dim as f64;
                let base = config.base as f64;
                let original_context_length = config.original_context_length as f64;
                let beta_fast = config.beta_fast as f64;
                let beta_slow = config.beta_slow as f64;
                let mut low =
                    dim * (original_context_length / (beta_fast * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
                let mut high =
                    dim * (original_context_length / (beta_slow * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
                if config.truncate {
                    low = low.floor();
                    high = high.ceil();
                }
                let low = low.max(0.0) as f32;
                let mut high = high.min((config.head_dim - 1) as f64) as f32;
                if low == high {
                    high += 0.001;
                }
                let ramp = ((pair_index as f32 - low) / (high - low)).clamp(0.0, 1.0);
                let smoothing_factor = 1.0 - ramp;
                let scaled_frequency = inverse_frequency / config.scaling_factor;
                scaled_frequency * (1.0 - smoothing_factor) + inverse_frequency * smoothing_factor
            },
            AnyRoPEConfig::LongRoPEConfig(config) => {
                let factors = if config.max_sequence_length > config.original_context_length {
                    &config.long_factor
                } else {
                    &config.short_factor
                };
                inverse_frequency / factors[pair_index as usize]
            },
        };

        for (token_index, token_position) in token_positions.iter().enumerate() {
            let embedding = *token_position as f32 * inverse_frequency;
            let sine = embedding.sin() * attention_scaling_factor;
            let cosine = embedding.cos() * attention_scaling_factor;
            let pair_offset = token_index * head_dim as usize + pair_index as usize;
            sines[pair_offset] = sine;
            sines[pair_offset + half_offset] = sine;
            cosines[pair_offset] = cosine;
            cosines[pair_offset + half_offset] = cosine;
        }
    }

    (cosines, sines)
}

/// Independently implemented from the Qwen3.5 rotary contract in Transformers
/// revision 2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397, modeling_qwen3_5.py,
/// Qwen3_5TextRotaryEmbedding (Qwen/Hugging Face, Apache-2.0).
/// Each axis selects coefficients before duplicating the complete half row.
pub(crate) fn precalculate_multimodal_cpu_tables(
    rope_config: &AnyRoPEConfig,
    positions: &[[u32; 3]],
) -> Result<(Vec<f32>, Vec<f32>), InvalidInput> {
    rope_config.validate()?;
    let AnyRoPEConfig::Qwen35MRoPEConfig(config) = rope_config else {
        return Err(InvalidInput("three-axis positions require Qwen3.5 M-RoPE"));
    };
    let width = config.head_dim as usize;
    let half_width = width / 2;
    let element_count = positions.len().checked_mul(width).ok_or(InvalidInput("M-RoPE table size overflow"))?;
    let mut cosines = vec![0.0; element_count];
    let mut sines = vec![0.0; element_count];
    for channel in 0..half_width {
        // The H/W streams replace stride-three slots of the temporal stream.
        let candidate_axis = channel % 3;
        let axis = if candidate_axis != 0 && channel / 3 < config.mrope_section[candidate_axis] as usize {
            candidate_axis
        } else {
            0
        };
        let inverse_frequency = 1.0 / config.base.powf((channel * 2) as f32 / config.head_dim as f32);
        for (row, position) in positions.iter().enumerate() {
            let embedding = position[axis] as f32 * inverse_frequency;
            let offset = row * width + channel;
            let cosine = embedding.cos();
            let sine = embedding.sin();
            cosines[offset] = cosine;
            cosines[offset + half_width] = cosine;
            sines[offset] = sine;
            sines[offset + half_width] = sine;
        }
    }
    Ok((cosines, sines))
}

#[cfg(test)]
mod multimodal_tests {
    use stark_engine_macros::stark_test;

    use super::*;

    fn config() -> AnyRoPEConfig {
        serde_json::from_value(serde_json::json!({
            "type":"Qwen35MRoPEConfig", "base":10000000.0, "max_sequence_length":262144,
            "head_dim":64, "mrope_section":[11,11,10], "interleaved":true
        }))
        .unwrap()
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[stark_test]
    fn qwen35_mrope_matches_independent_real_layout_with_distinct_axes() {
        // Explicit axis oracle for the production [11,11,10]/64 layout.
        // This catches contiguous sections and a too-short final H section.
        let axes = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1];
        let positions = [[2, 5, 11], [7, 13, 3], [0, 4, 9]];
        let (cosines, sines) = precalculate_multimodal_cpu_tables(&config(), &positions).unwrap();
        assert_eq!(cosines.len(), positions.len() * 64);
        assert_eq!(sines.len(), cosines.len());
        for (row, position) in positions.iter().enumerate() {
            for (channel, axis) in axes.iter().enumerate() {
                // Compute in f64 with an independent exponent expression.
                let angle = f64::from(position[*axis]) * 10_000_000f64.powf(-(channel as f64) / 32.0);
                let offset = row * 64 + channel;
                assert!((f64::from(cosines[offset]) - angle.cos()).abs() < 2e-6);
                assert!((f64::from(sines[offset]) - angle.sin()).abs() < 2e-6);
                assert_eq!(cosines[offset].to_bits(), cosines[offset + 32].to_bits());
                assert_eq!(sines[offset].to_bits(), sines[offset + 32].to_bits());
            }
        }
    }

    #[stark_test]
    fn qwen35_mrope_scalar_positions_preserve_unscaled_bits() {
        let mrope = config();
        let scalar: AnyRoPEConfig = serde_json::from_value(serde_json::json!({
            "type":"UnscaledRoPEConfig", "base":10000000.0, "max_sequence_length":262144, "head_dim":64
        }))
        .unwrap();
        let positions = [0, 1, 67, 1023, 16_777_217, u32::MAX];
        let diagonal = positions.map(|position| [position; 3]);
        let (reference_cosines, reference_sines) = precalculate_cpu_tables(&scalar, &positions);
        let (scalar_cosines, scalar_sines) = precalculate_cpu_tables(&mrope, &positions);
        let (multimodal_cosines, multimodal_sines) = precalculate_multimodal_cpu_tables(&mrope, &diagonal).unwrap();
        assert_eq!(bits(&reference_cosines), bits(&scalar_cosines));
        assert_eq!(bits(&reference_sines), bits(&scalar_sines));
        assert_eq!(bits(&reference_cosines), bits(&multimodal_cosines));
        assert_eq!(bits(&reference_sines), bits(&multimodal_sines));
    }

    #[stark_test]
    fn qwen35_mrope_chunk_slices_preserve_rows_and_axis_order() {
        // Unequal H/W ranges represent a rectangular image grid.
        let positions = (0..1031).map(|row| [2, 7 + row / 17, 5 + row % 17]).collect::<Vec<_>>();
        let config = config();
        let (whole_cosines, whole_sines) = precalculate_multimodal_cpu_tables(&config, &positions).unwrap();
        let mut chunk_cosines = Vec::new();
        let mut chunk_sines = Vec::new();
        for chunk in positions.chunks(1024) {
            let (cosines, sines) = precalculate_multimodal_cpu_tables(&config, chunk).unwrap();
            chunk_cosines.extend(cosines);
            chunk_sines.extend(sines);
        }
        assert_eq!(bits(&whole_cosines), bits(&chunk_cosines));
        assert_eq!(bits(&whole_sines), bits(&chunk_sines));
        let reordered = [positions[1030], positions[0], positions[1024], positions[1030]];
        let (cosines, sines) = precalculate_multimodal_cpu_tables(&config, &reordered).unwrap();
        for (row, source) in [1030, 0, 1024, 1030].into_iter().enumerate() {
            assert_eq!(bits(&cosines[row * 64..(row + 1) * 64]), bits(&whole_cosines[source * 64..(source + 1) * 64]));
            assert_eq!(bits(&sines[row * 64..(row + 1) * 64]), bits(&whole_sines[source * 64..(source + 1) * 64]));
        }
    }

    #[stark_test]
    fn qwen35_mrope_explicit_tables_reject_scalar_configs_and_invalid_layouts() {
        let scalar: AnyRoPEConfig = serde_json::from_value(serde_json::json!({
            "type":"UnscaledRoPEConfig", "base":10000.0, "max_sequence_length":1024, "head_dim":64
        }))
        .unwrap();
        assert!(precalculate_multimodal_cpu_tables(&scalar, &[[0, 1, 2]]).is_err());
        assert!(precalculate_multimodal_cpu_tables(&scalar, &[]).is_err());
        let mut invalid = config();
        let AnyRoPEConfig::Qwen35MRoPEConfig(inner) = &mut invalid else {
            unreachable!()
        };
        inner.mrope_section = [10, 12, 10];
        assert!(precalculate_multimodal_cpu_tables(&invalid, &[[0, 1, 2]]).is_err());
        let empty = precalculate_multimodal_cpu_tables(&config(), &[]).unwrap();
        assert!(empty.0.is_empty() && empty.1.is_empty());
    }

    #[cfg(feature = "cpu")]
    fn run_partial_rotation<B: Backend>() {
        use half::bf16;

        use crate::{
            backends::common::{
                CommandBufferExecutable, CommandBufferPending, Context, Kernels, kernel::AttentionPrepareKernel,
            },
            data_type::DataType,
            tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data},
        };

        let context = B::Context::new().unwrap();
        let kernel = <B::Kernels as Kernels>::AttentionPrepareKernel::new(
            &context,
            DataType::BF16,
            DataType::F32,
            true,
            true,
            false,
        )
        .unwrap();
        let input = (0..3 * 256).map(|index| bf16::from_f32((index % 31) as f32 / 32.0 - 0.5)).collect::<Vec<_>>();
        let packed = create_buffer_with_data::<B, bf16>(&context, &input);
        let mut queries = create_buffer::<B, bf16>(&context, 256);
        let mut keys = create_buffer::<B, bf16>(&context, 256);
        let mut values = create_buffer::<B, bf16>(&context, 256);
        let mut command = context.create_command_buffer(None, None).unwrap();
        let coefficients =
            PrecalculatedRoPE::<B>::precalculate_multimodal(&config(), &[[2, 5, 11]], &mut command).unwrap();
        assert_eq!(coefficients.dim, 64);
        kernel.encode(
            &packed,
            &mut queries,
            Some(&mut keys),
            Some(&mut values),
            Some(&coefficients.cosines),
            Some(&coefficients.sines),
            1,
            Some(1),
            256,
            Some(coefficients.dim),
            Some(0),
            3 * 256,
            1,
            &mut command,
        );
        drop(coefficients);
        command.end_encoding().submit().wait_until_completed().unwrap();
        let actual_queries = buffer_to_vec::<B, bf16>(&queries);
        let actual_keys = buffer_to_vec::<B, bf16>(&keys);
        let actual_values = buffer_to_vec::<B, bf16>(&values);
        let axes = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1];
        for (head_index, actual) in [&actual_queries, &actual_keys].into_iter().enumerate() {
            let original = &input[head_index * 256..(head_index + 1) * 256];
            for channel in 0..64 {
                let pair = channel % 32;
                let angle = [2f64, 5.0, 11.0][axes[pair]] * 10_000_000f64.powf(-(pair as f64) / 32.0);
                let paired = if channel < 32 {
                    channel + 32
                } else {
                    channel - 32
                };
                let sign = if channel < 32 {
                    -1.0
                } else {
                    1.0
                };
                let expected = f64::from(original[channel].to_f32()) * angle.cos()
                    + sign * f64::from(original[paired].to_f32()) * angle.sin();
                assert!((f64::from(actual[channel].to_f32()) - expected).abs() < 0.004);
            }
            assert_eq!(&actual[64..], &original[64..], "partial rotation changed the unrotated head suffix");
        }
        assert_eq!(&actual_values, &input[512..], "rotary coefficients changed value vectors");
    }

    #[cfg(feature = "cpu")]
    #[stark_test]
    fn qwen35_mrope_native_prepare_rotates_only_the_declared_prefix() {
        use crate::{backends::cpu::Cpu, tests::helpers::for_each_non_cpu_backend};

        run_partial_rotation::<Cpu>();
        for_each_non_cpu_backend!(|B| run_partial_rotation::<B>());
    }
}

#[cfg(all(test, feature = "rope-cache"))]
mod tests {
    use serde_json::json;
    use stark_engine_macros::stark_test;

    use super::*;

    fn configs() -> Vec<AnyRoPEConfig> {
        let variants = [
            json!({"type": "UnscaledRoPEConfig"}),
            json!({"type": "Qwen35MRoPEConfig", "mrope_section": [2,1,1], "interleaved": true}),
            json!({"type": "LinearScalingRoPEConfig", "scaling_factor": 4.0}),
            json!({"type": "LlamaRoPEConfig", "scaling_factor": 8.0, "original_context_length": 2048,
                "low_frequency_factor": 1.0, "high_frequency_factor": 4.0}),
            json!({"type": "YARNRoPEConfig", "scaling_factor": 4.0, "original_context_length": 2048,
                "beta_fast": 32.0, "beta_slow": 1.0, "truncate": true}),
            json!({"type": "YARNRoPEConfig", "scaling_factor": 2.0, "original_context_length": 2048,
                "beta_fast": 32.0, "beta_slow": 1.0, "truncate": false}),
            json!({"type": "LongRoPEConfig", "scaling_factor": 4.0, "original_context_length": 2048,
                "short_factor": [1.0, 1.25, 1.5, 2.0], "long_factor": [2.0, 4.0, 6.0, 8.0]}),
            json!({"type": "LongRoPEConfig", "scaling_factor": 1.0, "original_context_length": 16384,
                "short_factor": [1.0, 1.25, 1.5, 2.0], "long_factor": [2.0, 4.0, 6.0, 8.0]}),
        ];
        variants
            .into_iter()
            .map(|mut variant| {
                let fields = variant.as_object_mut().unwrap();
                fields.insert("base".into(), json!(10000.0));
                fields.insert("max_sequence_length".into(), json!(8192));
                fields.insert("head_dim".into(), json!(8));
                serde_json::from_value(variant).unwrap()
            })
            .collect()
    }

    fn assert_bit_equal(
        cache: &RoPETableCache,
        positions: &[u32],
    ) {
        let (expected_cosines, expected_sines) = precalculate_cpu_tables(&cache.config, positions);
        cache.with_tables(positions, |cosines, sines| {
            assert_eq!(cosines.len(), expected_cosines.len());
            assert_eq!(sines.len(), expected_sines.len());
            for (actual, expected) in cosines.iter().zip(&expected_cosines).chain(sines.iter().zip(&expected_sines)) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        });
    }

    #[stark_test]
    fn rope_cache_preserves_bits_for_growth_reorder_duplicates_and_fallback() {
        for config in configs() {
            let cache = RoPETableCache::new(&config);
            assert_bit_equal(&cache, &[]);
            for length in [1, 1023, 1024, 1025, 2048] {
                assert_bit_equal(&cache, &(0..length).collect::<Vec<_>>());
            }
            assert_bit_equal(&cache, &(1024..2048).collect::<Vec<_>>());
            assert_bit_equal(&cache, &[8, 0, 1024, 8, 1023, 1]);
            assert_bit_equal(&cache, &[8191, 8192, 8193, 0, 16_777_217, u32::MAX]);
            assert_bit_equal(&cache, &(0..1024).collect::<Vec<_>>());
            let tables = cache.tables.lock();
            assert_eq!(tables.0.len(), cache.max_cached_positions * *config.head_dim() as usize);
            assert_eq!(tables.1.len(), tables.0.len());
            assert!(tables.0.capacity() <= 1_048_576);
            assert!(tables.1.capacity() <= 1_048_576);
        }
    }

    #[stark_test]
    fn rope_cache_is_bound_to_complete_config_and_respects_small_caps() {
        let configs = configs();
        let first = RoPETableCache::new(&configs[0]);
        let mut changed = serde_json::to_value(&configs[0]).unwrap();
        changed["base"] = json!(500000.0);
        changed["max_sequence_length"] = json!(4);
        let changed: AnyRoPEConfig = serde_json::from_value(changed).unwrap();
        let second = RoPETableCache::new(&changed);
        for positions in [&[0, 1, 2, 3][..], &[3, 4, 5, 2][..]] {
            assert_bit_equal(&first, positions);
            assert_bit_equal(&second, positions);
        }
        assert_eq!(second.tables.lock().0.len(), 4 * *changed.head_dim() as usize);
        let first_bits =
            first.with_tables(&[3], |cosines, _| cosines.iter().map(|value| value.to_bits()).collect::<Vec<_>>());
        let second_bits =
            second.with_tables(&[3], |cosines, _| cosines.iter().map(|value| value.to_bits()).collect::<Vec<_>>());
        assert_ne!(first_bits, second_bits);
    }
}
