use crate::{
    backends::common::{Backend, BufferRef, CommandBuffer, InvalidInput, Kernels},
    data_type::DataType,
    encodable_block::mixer::attention::KVCacheView,
};

pub trait AttentionKernel: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<AttentionKernel = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        config: AttentionKernelConfig,
    ) -> Result<Self, <Self::Backend as Backend>::Error>;

    fn encode(
        &self,
        arguments: AttentionArguments<
            '_,
            Self::Backend,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
        >,
        command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<<Self::Backend as Backend>::ScratchBuffer, <Self::Backend as Backend>::Error>;
}

#[derive(Clone, Copy)]
pub struct AttentionKernelConfig {
    pub head_dim: u32,
    pub num_groups: u32,
    pub num_q_heads: u32,
    pub has_sinks: bool,
    pub is_kv_cache_ring: bool,
    pub is_causal: bool,
    pub sliding_window_size: Option<u32>,
    pub scale: Option<f32>,
    pub data_type: DataType,
    pub materialize_prefill_scores: bool,
    pub source_bf16_prefill: bool,
}

impl AttentionKernelConfig {
    /// Qualified MLX NAX QK split: BF16 operands, F32 partitions/reduction,
    /// followed by one BF16 score store. Small decode vectors retain their
    /// earlier dense-wide route; multiple KV groups cannot collapse to one batch.
    pub fn source_nax_qk_split_parts(
        self,
        rows: u32,
        keys: u32,
        supports_mxu: bool,
    ) -> Option<u32> {
        if !supports_mxu
            || !self.materialize_prefill_scores
            || self.validate_prefill_materialization().is_err()
            || self.data_type != DataType::BF16
            || self.head_dim != 512
            || self.num_q_heads != 8
            || self.num_groups != 1
            || !(9..1024).contains(&rows)
            || keys < rows
        {
            return None;
        }
        let m = self.num_q_heads * rows;
        let largest = u64::from(m.max(keys));
        let inner = u64::from(self.head_dim);
        (inner >= 3 * largest || (largest <= 1024 && inner > 2 * largest)).then_some(2)
    }

    pub(crate) fn uses_source_mxu_prefill(
        &self,
        rows: u32,
        supports_mxu: bool,
    ) -> bool {
        self.source_bf16_prefill
            && rows > 8
            && supports_mxu
            && self.data_type == DataType::BF16
            && matches!(self.head_dim, 64 | 128)
            && self.is_causal
            && !self.has_sinks
    }

    pub(crate) fn materializes_scores_for_rows(
        &self,
        rows: u32,
    ) -> bool {
        self.materialize_prefill_scores && (9..1024).contains(&rows)
    }

    pub(crate) fn materializes_scores_for_geometry(
        &self,
        rows: u32,
        kv_length: u32,
    ) -> bool {
        self.materializes_scores_for_rows(rows)
            || (self.materialize_prefill_scores && self.head_dim == 512 && rows > 0 && kv_length < 1024)
    }

    pub(crate) fn validate_prefill_materialization(&self) -> Result<(), InvalidInput> {
        if self.materialize_prefill_scores
            && (self.data_type != DataType::BF16
                || !matches!(self.head_dim, 256 | 512)
                || !self.is_causal
                || self.has_sinks
                || self.scale != Some(1.0))
        {
            return Err(InvalidInput(
                "BF16 prefill scores require causal D256/D512 unit-scale attention without sinks",
            ));
        }
        Ok(())
    }
}

pub struct AttentionArguments<
    'a,
    B: Backend,
    QT: BufferRef<Backend = B>,
    TT: BufferRef<Backend = B>,
    KT: BufferRef<Backend = B>,
    VT: BufferRef<Backend = B>,
> {
    pub queries: QT,
    pub keys: KT,
    pub values: VT,
    pub suffix_length: u32,
    pub trie: Option<TT>,
    pub sinks: Option<&'a B::GlobalBuffer>,
    pub cache: KVCacheView,
}

#[cfg(test)]
mod materialized_tests {
    use half::bf16;
    use stark_engine_macros::stark_test;

    use super::*;
    use crate::{
        backends::common::{CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context},
        tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_backend},
    };

    fn config(enabled: bool) -> AttentionKernelConfig {
        AttentionKernelConfig {
            head_dim: 256,
            num_groups: 1,
            num_q_heads: 2,
            has_sinks: false,
            is_kv_cache_ring: true,
            is_causal: true,
            sliding_window_size: Some(512),
            scale: Some(1.0),
            data_type: DataType::BF16,
            materialize_prefill_scores: enabled,
            source_bf16_prefill: false,
        }
    }

    #[stark_test]
    fn source_prefill_score_scaling_is_explicit_bf16_mxu_and_preserves_decode() {
        let source = AttentionKernelConfig {
            head_dim: 128,
            materialize_prefill_scores: false,
            source_bf16_prefill: true,
            ..config(false)
        };
        for rows in [1, 8] {
            assert!(!source.uses_source_mxu_prefill(rows, true));
        }
        for rows in [9, 63, 64, 1024, 4096] {
            assert!(source.uses_source_mxu_prefill(rows, true));
            assert!(!source.uses_source_mxu_prefill(rows, false));
            assert!(
                !AttentionKernelConfig {
                    source_bf16_prefill: false,
                    ..source
                }
                .uses_source_mxu_prefill(rows, true)
            );
        }
        for unsupported in [
            AttentionKernelConfig {
                head_dim: 256,
                ..source
            },
            AttentionKernelConfig {
                data_type: DataType::F32,
                ..source
            },
            AttentionKernelConfig {
                is_causal: false,
                ..source
            },
            AttentionKernelConfig {
                has_sinks: true,
                ..source
            },
        ] {
            assert!(!unsupported.uses_source_mxu_prefill(49, true));
        }
    }

    #[stark_test]
    fn materialized_prefill_policy_is_explicit_bounded_and_rejects_unsupported_arithmetic() {
        let baseline = config(false);
        let source = config(true);
        assert!(source.validate_prefill_materialization().is_ok());
        for rows in [1, 8, 1024, 2048] {
            assert!(!source.materializes_scores_for_rows(rows));
        }
        for rows in [9, 511, 1023] {
            assert!(source.materializes_scores_for_rows(rows));
            assert!(!baseline.materializes_scores_for_rows(rows));
        }
        for unsupported in [
            AttentionKernelConfig {
                head_dim: 128,
                ..source
            },
            AttentionKernelConfig {
                data_type: DataType::F32,
                ..source
            },
            AttentionKernelConfig {
                scale: None,
                ..source
            },
            AttentionKernelConfig {
                has_sinks: true,
                ..source
            },
            AttentionKernelConfig {
                is_causal: false,
                ..source
            },
        ] {
            assert!(unsupported.validate_prefill_materialization().is_err());
        }
    }

    #[stark_test]
    fn source_d512_short_cache_materializes_decode_and_preserves_other_vector_routes() {
        let source = AttentionKernelConfig {
            head_dim: 512,
            ..config(true)
        };
        let legacy = AttentionKernelConfig {
            head_dim: 512,
            ..config(false)
        };
        for (rows, keys, expected) in
            [(1, 512, true), (8, 1023, true), (1, 1024, false), (8, 1024, false), (9, 1024, true), (1024, 1024, false)]
        {
            assert_eq!(source.materializes_scores_for_geometry(rows, keys), expected);
            assert!(!legacy.materializes_scores_for_geometry(rows, keys));
        }
        assert!(!config(true).materializes_scores_for_geometry(1, 512));
    }

    #[stark_test]
    fn source_qk_split_policy_is_scoped_after_wide_and_preserves_legacy_shapes() {
        let source = AttentionKernelConfig {
            head_dim: 512,
            num_q_heads: 8,
            ..config(true)
        };
        for rows in [9, 22, 31] {
            assert_eq!(source.source_nax_qk_split_parts(rows, rows, true), Some(2));
        }
        for (rows, keys) in [(1, 23), (8, 8), (32, 32), (22, 1024), (1024, 1024), (22, 21)] {
            assert_eq!(source.source_nax_qk_split_parts(rows, keys, true), None);
        }
        assert_eq!(source.source_nax_qk_split_parts(22, 22, false), None);
        for other in [
            AttentionKernelConfig {
                materialize_prefill_scores: false,
                ..source
            },
            AttentionKernelConfig {
                head_dim: 256,
                ..source
            },
            AttentionKernelConfig {
                num_groups: 2,
                ..source
            },
            AttentionKernelConfig {
                num_q_heads: 2,
                ..source
            },
            AttentionKernelConfig {
                data_type: DataType::F32,
                ..source
            },
            AttentionKernelConfig {
                scale: Some(0.5),
                ..source
            },
        ] {
            assert_eq!(other.source_nax_qk_split_parts(22, 22, true), None);
        }
        assert_eq!(source.source_nax_qk_split_parts(22, u32::MAX, true), None);
    }

    #[stark_test]
    fn source_d512_decode_has_independent_bf16_score_probability_oracle() {
        for_each_backend!(|B| {
            let context = <B as Backend>::Context::new().unwrap();
            let mut q = vec![bf16::ZERO; 512];
            let mut k = vec![bf16::ZERO; 1024];
            let mut v = vec![bf16::ZERO; 1024];
            q[0] = bf16::from_f32(1.0078125);
            k[0] = bf16::from_f32(0.99609375);
            k[512] = -k[0];
            v[0] = bf16::ONE;
            let q = create_buffer_with_data::<B, bf16>(&context, &q);
            let k = create_buffer_with_data::<B, bf16>(&context, &k);
            let v = create_buffer_with_data::<B, bf16>(&context, &v);
            for enabled in [false, true, false] {
                let kernel = <<B as Backend>::Kernels as Kernels>::AttentionKernel::new(
                    &context,
                    AttentionKernelConfig {
                        head_dim: 512,
                        num_q_heads: 1,
                        num_groups: 1,
                        is_kv_cache_ring: false,
                        sliding_window_size: None,
                        ..config(enabled)
                    },
                )
                .unwrap();
                let mut commands = context.create_command_buffer(None, None).unwrap();
                let result = kernel
                    .encode(
                        AttentionArguments {
                            queries: &q,
                            keys: &k,
                            values: &v,
                            suffix_length: 1,
                            trie: None::<&<B as Backend>::GlobalBuffer>,
                            sinks: None,
                            cache: KVCacheView::full(1),
                        },
                        &mut commands,
                    )
                    .unwrap();
                let mut readback = create_buffer::<B, bf16>(&context, 512);
                commands.encode_copy(&result, &mut readback);
                commands.end_encoding().submit().wait_until_completed().unwrap();
                // Exact product is +/-1.003875732421875. Source score storage
                // rounds it to +/-1 before softmax, crossing the final BF16 midpoint.
                let score = if enabled {
                    1.0f64
                } else {
                    1.003875732421875
                };
                let expected = bf16::from_f32((1.0 / (1.0 + (-2.0 * score).exp())) as f32);
                let actual = buffer_to_vec::<B, bf16>(&readback);
                assert_eq!(actual[0], expected);
                assert!(actual[1..].iter().all(|x| *x == bf16::ZERO));
            }
        });
    }

    fn run<B: Backend>(
        rows: usize,
        enabled: bool,
    ) -> Vec<bf16> {
        let context = B::Context::new().unwrap();
        let mut query = vec![bf16::ZERO; 2 * rows * 256];
        for head in 0..2 {
            for row in 0..rows {
                query[(head * rows + row) * 256] = bf16::from_f32(if head == 0 {
                    1.0078125
                } else {
                    -1.0078125
                });
            }
        }
        let mut key = vec![bf16::ZERO; rows * 256];
        key[0] = bf16::from_f32(0.99609375);
        let mut value = vec![bf16::ZERO; rows * 256];
        value[0] = bf16::ONE;
        let query = create_buffer_with_data::<B, bf16>(&context, &query);
        let key = create_buffer_with_data::<B, bf16>(&context, &key);
        let value = create_buffer_with_data::<B, bf16>(&context, &value);
        let kernel = <B::Kernels as Kernels>::AttentionKernel::new(&context, config(enabled)).unwrap();
        let mut commands = context.create_command_buffer(None, None).unwrap();
        let result = kernel
            .encode(
                AttentionArguments {
                    queries: &query,
                    keys: &key,
                    values: &value,
                    suffix_length: rows as u32,
                    trie: None::<&B::GlobalBuffer>,
                    sinks: None,
                    cache: KVCacheView::ring(0, 0),
                },
                &mut commands,
            )
            .unwrap();
        let mut readback = create_buffer::<B, bf16>(&context, rows * 2 * 256);
        commands.encode_copy(&result, &mut readback);
        commands.end_encoding().submit().wait_until_completed().unwrap();
        buffer_to_vec::<B, bf16>(&readback)
    }

    #[stark_test]
    fn materialized_prefill_scores_have_independent_causal_probability_oracle_and_preserve_decode() {
        // BF16 operands multiply to +/-1.003875732421875; materialized scores
        // round to +/-1. Only the first value row is nonzero. The first output
        // component is therefore BF16(1 / (1 + row * exp(-/+1))).
        for_each_backend!(|B| {
            let output = run::<B>(9, true);
            for row in 0..9 {
                for head in 0..2 {
                    let sign = if head == 0 {
                        -1.0f64
                    } else {
                        1.0f64
                    };
                    let expected = bf16::from_f32((1.0 / (1.0 + row as f64 * sign.exp())) as f32);
                    assert_eq!(output[(row * 2 + head) * 256], expected);
                    assert!(
                        output[(row * 2 + head) * 256 + 1..(row * 2 + head + 1) * 256]
                            .iter()
                            .all(|value| *value == bf16::ZERO)
                    );
                }
            }
            assert_eq!(run::<B>(1, false), run::<B>(1, true));
        });
    }
}
