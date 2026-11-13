#[cfg(feature = "attention-kv-only")]
use crate::backends::common::kernel::AttentionPrepareKVOnlyKernel;
use crate::{
    backends::common::{
        Backend, Buffer, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding,
        kernel::{AttentionArguments, AttentionKernel, AttentionPrepareKernel, SigmoidGateKernel},
    },
    encodable_block::{
        batch_topology::BatchTopology,
        linear::Linear,
        mixer::{
            MixerState,
            attention::{Attention, KVCacheView, qkv_norm::QKVNorm, rope::PrecalculatedRoPE, state::AttentionState},
        },
    },
    utils::maybe_mut::MaybeMut,
};

pub(super) struct LinearProjection<B: Backend> {
    pub(super) lin: Box<dyn Linear<B>>,
    pub(super) norm: Option<QKVNorm<B>>,
}

impl<B: Backend> LinearProjection<B> {
    fn project(
        &self,
        hidden: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut projected = self.lin.encode(hidden, batch_dim, command_buffer)?;
        if let Some(norm) = &self.norm {
            norm.encode(&mut projected, batch_dim, command_buffer)?;
        }
        Ok(projected)
    }
}

impl<B: Backend> Attention<B> {
    #[cfg(test)]
    pub(crate) fn assistant_query_trace(
        &self,
        hidden: B::ScratchBuffer,
        rope: &PrecalculatedRoPE<B>,
        target: &super::CommittedAttentionKV<'_, B>,
        reference: &super::reference::ReferenceBF16Attention<B>,
        command: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Vec<(String, B::ScratchBuffer)>, B::Error> {
        let mut projected = self.projection.lin.encode(hidden, 1, command)?;
        let mut raw = command.allocate_scratch(projected.size())?;
        command.encode_copy(&projected, &mut raw);
        if let Some(norm) = &self.projection.norm {
            norm.encode(&mut projected, 1, command)?;
        }
        let mut normalized = command.allocate_scratch(projected.size())?;
        command.encode_copy(&projected, &mut normalized);
        let queries = self.prepare_queries(&projected, Some(rope), 1, command)?;
        let (keys, values) = target.validation_copy(command)?;
        let mut stages = reference.trace_single_query(&queries, &keys, &values, target.retained_rows(), command)?;
        stages.insert(0, ("q_projection".into(), raw));
        stages.insert(1, ("q_norm".into(), normalized));
        stages.insert(2, ("q_rope".into(), queries));
        Ok(stages)
    }

    pub(crate) fn attend_committed_target_eager(
        &self,
        hidden: B::ScratchBuffer,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        target: &super::CommittedAttentionKV<'_, B>,
        reference: &super::reference::ReferenceBF16Attention<B>,
        command: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        if self.data_type != crate::data_type::DataType::BF16
            || !reference.matches_geometry(self.num_q_heads, self.num_groups, self.head_dim)
        {
            return Err(crate::backends::common::InvalidInput("committed BF16 reference geometry mismatch").into());
        }
        self.attend_committed_target_inner(hidden, precalculated_rope, target, Some(reference), command)
    }

    fn attend_committed_target_inner(
        &self,
        hidden: B::ScratchBuffer,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        target: &super::CommittedAttentionKV<'_, B>,
        reference: Option<&super::reference::ReferenceBF16Attention<B>>,
        command: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let invalid = || crate::backends::common::InvalidInput("invalid committed target KV query");
        let required = self.num_groups.checked_mul(self.head_dim).ok_or_else(invalid)?;
        if self.num_kv_heads.is_some()
            || self.gate_kernel.is_some()
            || self.sinks.is_some()
            || target.state.element_dim != required
            || target.state.data_type != self.data_type
            || hidden.size() != self.input_dim as usize * self.data_type.size_in_bytes()
            || self.max_rope_length.is_some() != precalculated_rope.is_some()
            || self.max_rope_length.is_some_and(|maximum| target.position() >= maximum)
            || self.ring_capacity.is_some() != target.view.ring_params().is_some()
            || self.ring_capacity.is_some_and(|window| target.retained_rows() > window)
        {
            return Err(invalid().into());
        }
        let projected = self.projection.project(hidden, 1, command)?;
        let queries = self.prepare_queries(&projected, precalculated_rope, 1, command)?;
        let cache = target.query_cache_view();
        let temporal = target.view.ring_params().is_some().then(|| target.linearize_ring(command)).transpose()?;
        let output = if let Some((keys, values)) = &temporal {
            if let Some(reference) = reference {
                reference.encode(&queries, keys, values, 1, target.retained_rows() - 1, command)?
            } else {
                // The assistant's current position is fixed. Copy at most one
                // local window to temporal order; full target context stays shared.
                self.encode_core(
                    AttentionArguments {
                        queries: &queries,
                        keys,
                        values,
                        suffix_length: 1,
                        trie: None::<&B::ConstantBuffer>,
                        sinks: None,
                        cache,
                        image_span: None,
                    },
                    command,
                )?
            }
        } else {
            if let Some(reference) = reference {
                reference.encode(
                    &queries,
                    target.state.keys.as_ref(),
                    target.state.values.as_ref(),
                    1,
                    target.retained_rows() - 1,
                    command,
                )?
            } else {
                self.encode_core(
                    AttentionArguments {
                        queries: &queries,
                        keys: target.state.keys.as_ref(),
                        values: target.state.values.as_ref(),
                        suffix_length: 1,
                        trie: None::<&B::ConstantBuffer>,
                        sinks: None,
                        cache,
                        image_span: None,
                    },
                    command,
                )?
            }
        };
        self.out_projection.encode(output, 1, command)
    }

    /// Re-encode a small flat query with the actual projection/core/gate paths.
    /// Persisted buffers belong only to the private test exporter, never KV state.
    #[cfg(test)]
    pub(crate) fn validation_stateless_stages(
        &self,
        hidden: B::ScratchBuffer,
        rope: Option<&PrecalculatedRoPE<B>>,
        topology: &BatchTopology,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Vec<(&'static str, B::ScratchBuffer, Vec<usize>)>, B::Error> {
        use crate::{backends::common::Buffer, data_type::DataType};
        let rows = topology.size();
        if rows == 0
            || rows > 16
            || !topology.is_flat()
            || self.ring_capacity.is_some()
            || self.sinks.is_some()
            || self.data_type != DataType::BF16
            || rope.is_some() != self.max_rope_length.is_some()
        {
            return Err(crate::backends::common::InvalidInput("small stateless BF16 attention trace required").into());
        }
        let kv = self.num_kv_heads.ok_or(crate::backends::common::InvalidInput("owned KV trace required"))?;
        let copy = |buffer: &B::ScratchBuffer, commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding| {
            let mut output = commands.allocate_scratch(buffer.size())?;
            commands.encode_copy(buffer, &mut output);
            Ok::<_, B::Error>(output)
        };
        let mut result = Vec::new();
        let mut qkvg = self.projection.lin.encode(hidden, rows, commands)?;
        let packed_shape = vec![1, rows as usize, self.projection_dim as usize];
        result.push(("attention_raw_qkvg", copy(&qkvg, commands)?, packed_shape.clone()));
        if let Some(norm) = &self.projection.norm {
            norm.encode(&mut qkvg, rows, commands)?;
        }
        result.push(("attention_normalized_qkvg", copy(&qkvg, commands)?, packed_shape));
        let mut keys = commands.allocate_scratch_for_shape(&[rows, kv, self.head_dim], self.data_type)?;
        let mut values = commands.allocate_scratch_for_shape(&[rows, kv, self.head_dim], self.data_type)?;
        let queries =
            self.prepare_kv_and_queries(&qkvg, &mut keys, &mut values, 0, self.num_q_heads, rope, rows, commands)?;
        result.push((
            "attention_queries",
            copy(&queries, commands)?,
            vec![self.num_q_heads as usize, rows as usize, self.head_dim as usize],
        ));
        result.push((
            "attention_keys",
            copy(&keys, commands)?,
            vec![rows as usize, kv as usize, self.head_dim as usize],
        ));
        result.push((
            "attention_values",
            copy(&values, commands)?,
            vec![rows as usize, kv as usize, self.head_dim as usize],
        ));
        let mut output = self.encode_core(
            AttentionArguments {
                image_span: None,
                queries: &queries,
                keys: &keys,
                values: &values,
                suffix_length: rows,
                trie: None::<&B::ConstantBuffer>,
                sinks: None::<&B::GlobalBuffer>,
                cache: KVCacheView::full(0),
            },
            commands,
        )?;
        let core_shape = vec![1, rows as usize, (self.num_q_heads * self.head_dim) as usize];
        result.push(("attention_core", copy(&output, commands)?, core_shape.clone()));
        if let Some(gate) = &self.gate_kernel {
            let width = self.num_q_heads * self.head_dim;
            let offset = self.projection_dim - width;
            let ones = vec![half::bf16::from_f32(1.0); (rows * width) as usize];
            let ones = commands.allocate_constant_from_slice(&ones)?;
            let mut sigmoid = commands.allocate_scratch(rows as usize * width as usize * 2)?;
            commands.encode_copy(&ones, &mut sigmoid);
            gate.encode(qkvg.subrange(offset as usize * 2..), &mut sigmoid, width, rows, self.projection_dim, commands);
            result.push(("attention_sigmoid", sigmoid, core_shape.clone()));
            gate.encode(qkvg.subrange(offset as usize * 2..), &mut output, width, rows, self.projection_dim, commands);
        }
        result.push(("attention_gated", copy(&output, commands)?, core_shape));
        let output = self.out_projection.encode(output, rows, commands)?;
        let output_dim = output.size() / (rows as usize * 2);
        result.push(("attention_projected", output, vec![1, rows as usize, output_dim]));
        Ok(result)
    }

    pub(crate) fn supports_cache_only(&self) -> bool {
        self.num_kv_heads.is_some() && self.ring_capacity.is_none()
    }

    fn source_rope_arguments<'a>(
        &self,
        rope: Option<&'a PrecalculatedRoPE<B>>,
    ) -> Result<(Option<&'a B::ConstantBuffer>, Option<f32>, Option<u32>, Option<&'a B::ConstantBuffer>), B::Error>
    {
        if !self.source_rope {
            return Ok((None, None, None, None));
        }
        let rope = rope.ok_or(crate::backends::common::InvalidInput("source RoPE requires scalar tables"))?;
        let parameters = rope
            .source_parameters
            .ok_or(crate::backends::common::InvalidInput("unsupported source RoPE configuration"))?;
        let positions = rope
            .source_positions
            .as_ref()
            .ok_or(crate::backends::common::InvalidInput("source RoPE requires scalar positions"))?;
        if parameters.proportional != self.source_proportional || parameters.yarn != self.source_yarn {
            return Err(crate::backends::common::InvalidInput("source RoPE profile mismatch").into());
        }
        if rope.source_yarn_parameters.is_some() != self.source_yarn {
            return Err(crate::backends::common::InvalidInput("source YaRN parameters mismatch").into());
        }
        Ok((Some(positions), Some(parameters.base), Some(parameters.rotary_dim), rope.source_yarn_parameters.as_ref()))
    }
    #[cfg(feature = "scoring-last-layer-pruning")]
    pub(crate) fn supports_scoring_cache_only(&self) -> bool {
        self.supports_cache_only()
    }

    /// Preserve full-batch projection and all KV arithmetic/storage. The optional
    /// KV-only candidate omits dead query normalization and preparation while
    /// retaining the original packed offsets and prepare dispatch coordinates.
    pub(crate) fn encode_cache_only(
        &self,
        hidden: B::ScratchBuffer,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: &mut AttentionState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        assert!(self.supports_cache_only(), "unsupported cache-only attention");
        assert!(batch_dim.is_flat(), "cache-only requires flat topology");
        assert_eq!(precalculated_rope.is_some(), self.max_rope_length.is_some(), "precalculated rope mismatch");
        let qkvg = self.projection.project(hidden, batch_dim.size(), command_buffer)?;
        self.prepare_cache_only(&qkvg, state, precalculated_rope, batch_dim, command_buffer)?;
        // The caller still accepts every row before encoding the next batch.
        Ok(())
    }

    fn prepare_cache_only(
        &self,
        qkvg: impl BufferRef<Backend = B>,
        state: &mut AttentionState<B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let prefix_len = state.view().prefix_len();
        let _queries = self.prepare_kv_and_queries(
            qkvg,
            state.keys.as_mut(),
            state.values.as_mut(),
            prefix_len,
            self.num_q_heads,
            precalculated_rope,
            batch_dim.size(),
            command_buffer,
        )?;
        Ok(())
    }

    #[cfg(feature = "scoring-last-layer-pruning")]
    pub(crate) fn encode_scoring_cache_only(
        &self,
        hidden: B::ScratchBuffer,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: &mut AttentionState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        assert!(self.supports_scoring_cache_only(), "unsupported scoring cache-only attention");
        assert!(batch_dim.is_flat(), "scoring cache-only requires flat topology");
        assert_eq!(precalculated_rope.is_some(), self.max_rope_length.is_some(), "precalculated rope mismatch");
        #[cfg(feature = "attention-kv-only")]
        let qkvg = {
            let mut projected = self.projection.lin.encode(hidden, batch_dim.size(), command_buffer)?;
            if let Some(norm) = &self.projection.norm {
                norm.encode_projected_key_value(&mut projected, batch_dim.size(), command_buffer)?;
            }
            projected
        };
        #[cfg(not(feature = "attention-kv-only"))]
        let qkvg = self.projection.project(hidden, batch_dim.size(), command_buffer)?;
        #[cfg(feature = "attention-kv-only")]
        let prefix_len = state.view().prefix_len();
        #[cfg(feature = "attention-kv-only")]
        let (source_positions, source_base, source_rotary_dim, source_yarn_parameters) =
            self.source_rope_arguments(precalculated_rope)?;
        #[cfg(feature = "attention-kv-only")]
        self.prepare_kv_only.as_ref().expect("KV-only prepare requires owned full attention").encode(
            &qkvg,
            state.keys.as_mut(),
            state.values.as_mut(),
            precalculated_rope.map(|rope| &rope.cosines),
            precalculated_rope.map(|rope| &rope.sines),
            source_positions,
            source_base,
            source_rotary_dim,
            source_yarn_parameters,
            self.num_q_heads,
            self.num_kv_heads.expect("KV-only prepare requires KV heads"),
            self.head_dim,
            precalculated_rope.map(|rope| rope.dim),
            prefix_len,
            self.projection_dim,
            batch_dim.size(),
            command_buffer,
        );
        #[cfg(not(feature = "attention-kv-only"))]
        self.prepare_cache_only(&qkvg, state, precalculated_rope, batch_dim, command_buffer)?;
        Ok(())
    }

    pub(super) fn attend(
        &self,
        hidden: B::ScratchBuffer,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<AttentionState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let qkvg = self.projection.project(hidden, batch_dim.size(), command_buffer)?;

        let mut attention_output = match state {
            Some(MaybeMut::Mut(state)) => {
                let prefix_len = state.view().prefix_len();
                let queries = self.prepare_kv_and_queries(
                    &qkvg,
                    state.keys.as_mut(),
                    state.values.as_mut(),
                    prefix_len,
                    self.num_q_heads,
                    precalculated_rope,
                    batch_dim.size(),
                    command_buffer,
                )?;
                self.run_core(&queries, batch_dim, state, command_buffer)?
            },
            Some(MaybeMut::Const(state)) => {
                // KV sharing: QKVG contains queries and an optional gate only.
                let queries = self.prepare_queries(&qkvg, precalculated_rope, batch_dim.size(), command_buffer)?;
                self.run_core(&queries, batch_dim, state, command_buffer)?
            },
            None => {
                let Some(num_kv_heads) = self.num_kv_heads else {
                    panic!("stateless attention doesn't support query-only projection");
                };
                assert!(batch_dim.is_flat(), "stateless attention doesn't support trie");

                let mut keys = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim.size(), num_kv_heads, self.head_dim], self.data_type)?;
                let mut values = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim.size(), num_kv_heads, self.head_dim], self.data_type)?;

                let queries = self.prepare_kv_and_queries(
                    &qkvg,
                    &mut keys,
                    &mut values,
                    0,
                    self.num_q_heads,
                    precalculated_rope,
                    batch_dim.size(),
                    command_buffer,
                )?;

                let cache = self.ring_capacity.map_or_else(|| KVCacheView::full(0), |_| KVCacheView::ring(0, 0));

                self.encode_core(
                    AttentionArguments {
                        image_span: batch_dim.image_span().map(Into::into),
                        queries: &queries,
                        keys: &keys,
                        values: &values,
                        suffix_length: batch_dim.size(),
                        trie: None::<&B::ConstantBuffer>,
                        sinks: self.sinks.as_ref(),
                        cache,
                    },
                    command_buffer,
                )?
            },
        };

        if let Some(gate_kernel) = &self.gate_kernel {
            let gate_dim = self.num_q_heads * self.head_dim;
            let gate_offset = self.projection_dim - gate_dim;
            gate_kernel.encode(
                qkvg.subrange(gate_offset as usize * self.data_type.size_in_bytes()..),
                &mut attention_output,
                gate_dim,
                batch_dim.size(),
                self.projection_dim,
                command_buffer,
            );
        }
        self.out_projection.encode(attention_output, batch_dim.size(), command_buffer)
    }

    pub fn append_projected_kv(
        &self,
        mut key_value: impl BufferMut<Backend = B>,
        precalculated_rope: &PrecalculatedRoPE<B>,
        batch_dim: u32,
        state: &mut AttentionState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        if let Some(norm) = &self.projection.norm {
            norm.encode_key_value(key_value.reborrow(), batch_dim, command_buffer)?;
        }
        let prefix_len = state.view().prefix_len();
        self.prepare_kv_and_queries(
            key_value.as_ref(),
            state.keys.as_mut(),
            state.values.as_mut(),
            prefix_len,
            0,
            Some(precalculated_rope),
            batch_dim,
            command_buffer,
        )?;
        state.encode_accept(&(0..batch_dim).collect::<Box<[u32]>>(), command_buffer)?;
        Ok(())
    }

    fn run_core(
        &self,
        queries: impl BufferRef<Backend = B>,
        batch_dim: &BatchTopology,
        state: &AttentionState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let trie = if batch_dim.is_flat() {
            None
        } else {
            Some(command_buffer.allocate_constant_from_slice(batch_dim.nodes())?)
        };

        self.encode_core(
            AttentionArguments {
                image_span: batch_dim.image_span().map(Into::into),
                queries,
                keys: state.keys.as_ref(),
                values: state.values.as_ref(),
                suffix_length: batch_dim.size(),
                trie: trie.as_ref(),
                sinks: self.sinks.as_ref(),
                cache: state.view(),
            },
            command_buffer,
        )
    }

    fn encode_core(
        &self,
        mut arguments: AttentionArguments<
            '_,
            B,
            impl BufferRef<Backend = B>,
            impl BufferRef<Backend = B>,
            impl BufferRef<Backend = B>,
            impl BufferRef<Backend = B>,
        >,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        use crate::config::token_mixer::attention::ImageAttentionPolicy;
        arguments.image_span = arguments.image_span.and_then(|mut span| {
            if self.image_attention_policy == ImageAttentionPolicy::Causal {
                None
            } else {
                span.respects_window = self.image_attention_policy == ImageAttentionPolicy::WindowedSameImage;
                Some(span)
            }
        });
        if let Some(reference) = &self.reference {
            if arguments.image_span.is_some() {
                return Err(crate::backends::common::InvalidInput(
                    "reference BF16 causal score materialization does not implement bidirectional image attention",
                )
                .into());
            }
            if arguments.trie.is_some() || arguments.sinks.is_some() || arguments.cache.ring_params().is_some() {
                return Err(
                    crate::backends::common::InvalidInput("reference BF16 attention requires flat full cache").into()
                );
            }
            reference.encode(
                arguments.queries,
                arguments.keys,
                arguments.values,
                arguments.suffix_length,
                arguments.cache.prefix_len(),
                command_buffer,
            )
        } else {
            self.kernel.encode(arguments, command_buffer)
        }
    }

    fn prepare_kv_and_queries(
        &self,
        input: impl BufferRef<Backend = B>,
        keys: impl BufferMut<Backend = B>,
        values: impl BufferMut<Backend = B>,
        kv_token_offset: u32,
        num_q_heads: u32,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let num_kv_heads = self.num_kv_heads.expect("KV prepare requires KV heads");
        // Appended KV is tightly packed; attention projections may have a trailing gate segment.
        let input_row_stride = if num_q_heads == 0 {
            2 * num_kv_heads * self.head_dim
        } else {
            self.projection_dim
        };
        let mut queries = if num_q_heads == 0 {
            command_buffer.allocate_scratch(self.data_type.size_in_bytes())?
        } else {
            command_buffer.allocate_scratch_for_shape(&[self.num_q_heads, batch_dim, self.head_dim], self.data_type)?
        };
        let (source_positions, source_base, source_rotary_dim, source_yarn_parameters) =
            self.source_rope_arguments(precalculated_rope)?;
        self.prepare.encode(
            input,
            &mut queries,
            Some(keys),
            Some(values),
            precalculated_rope.map(|precalculated_rope| &precalculated_rope.cosines),
            precalculated_rope.map(|precalculated_rope| &precalculated_rope.sines),
            source_positions,
            source_base,
            source_rotary_dim,
            source_yarn_parameters,
            num_q_heads,
            Some(num_kv_heads),
            self.head_dim,
            precalculated_rope.map(|precalculated_rope| precalculated_rope.dim),
            Some(kv_token_offset),
            input_row_stride,
            batch_dim,
            command_buffer,
        );
        Ok(queries)
    }

    fn prepare_queries(
        &self,
        qkvg: impl BufferRef<Backend = B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut queries =
            command_buffer.allocate_scratch_for_shape(&[self.num_q_heads, batch_dim, self.head_dim], self.data_type)?;
        let (source_positions, source_base, source_rotary_dim, source_yarn_parameters) =
            self.source_rope_arguments(precalculated_rope)?;
        self.prepare.encode(
            qkvg,
            &mut queries,
            None::<&mut B::ScratchBuffer>,
            None::<&mut B::ScratchBuffer>,
            precalculated_rope.map(|rope| &rope.cosines),
            precalculated_rope.map(|rope| &rope.sines),
            source_positions,
            source_base,
            source_rotary_dim,
            source_yarn_parameters,
            self.num_q_heads,
            None,
            self.head_dim,
            precalculated_rope.map(|rope| rope.dim),
            None,
            self.projection_dim,
            batch_dim,
            command_buffer,
        );
        Ok(queries)
    }
}
