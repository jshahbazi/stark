// SPDX-License-Identifier: Apache-2.0
//! Deferred native decode commands. Device vectors carry checked element spans;
//! intermediates remain owned until the single explicit completion boundary.

use super::*;

#[path = "decode_graph_counters.rs"]
mod counters;
use counters::EncoderTiming;

#[derive(Clone)]
pub(crate) struct DeviceVector {
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub offset: usize,
    pub len: usize,
}

impl DeviceVector {
    pub fn slice(
        &self,
        start: usize,
        len: usize,
    ) -> Result<Self> {
        self.validate()?;
        if len == 0 || start.checked_add(len).is_none_or(|end| end > self.len) {
            return Err(Error::Invalid("decode vector slice exceeds allocation"));
        }
        let offset = start
            .checked_mul(4)
            .and_then(|n| self.offset.checked_add(n))
            .ok_or(Error::Invalid("decode vector slice overflow"))?;
        Ok(Self {
            buffer: self.buffer.clone(),
            offset,
            len,
        })
    }

    fn validate(&self) -> Result<()> {
        if self.len == 0
            || self.len > u32::MAX as usize
            || !self.offset.is_multiple_of(4)
            || self.len.checked_mul(4).and_then(|n| self.offset.checked_add(n)).is_none_or(|n| n > self.buffer.length())
        {
            return Err(Error::Invalid("invalid decode vector span"));
        }
        Ok(())
    }
}

pub(crate) struct DecodeGraph {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoding: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    encoding_dispatch: MTLDispatchType,
    encoder_timing: Option<EncoderTiming>,
    failure: DeviceVector,
    // Explicit ownership complements Metal's retained command references,
    // including immutable mappings with deallocators retaining their files.
    resources: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    encoding_ended: bool,
    encode_scope: Option<profile::Scope>,
    scratch: Vec<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>,
    scratch_bytes: usize,
    single_encoder: bool,
    #[cfg(test)]
    serial_projection_levels: bool,
    byte_loads: bool,
    vector8: bool,
    scalar_uint_loads: bool,
    factored_affine: bool,
    projection_level: bool,
    projection_checks: Vec<DeviceVector>,
    separate_finite: bool,
    #[cfg(test)]
    phase_timing: bool,
}

impl DecodeGraph {
    pub fn new() -> Result<Self> {
        Self::new_with_encoder(false)
    }

    /// A Qwen-only experimental transport. Every dependent dispatch inserts
    /// an explicit buffer barrier; projection levels admit independent work.
    #[cfg(test)]
    pub fn new_parallel() -> Result<Self> {
        Self::new_with_encoder(true)
    }

    fn new_with_encoder(single_encoder: bool) -> Result<Self> {
        let phase_timing = std::env::var_os("STARK_QWEN38_PHASE_TIMING").is_some_and(|value| value == "1");
        let encoder_timing = std::env::var_os("STARK_QWEN38_ENCODER_TIMING").is_some_and(|value| value == "1");
        Self::new_with_timing(single_encoder, phase_timing, encoder_timing)
    }

    fn new_with_timing(
        single_encoder: bool,
        phase_timing: bool,
        encoder_timing: bool,
    ) -> Result<Self> {
        if phase_timing && encoder_timing {
            return Err(Error::Invalid("native timing modes cannot be combined"));
        }
        objc2::rc::autoreleasepool(|_| {
            let executor = executor()?;
            let command = executor.queue.command_buffer().ok_or(Error::Invalid("decode command allocation failed"))?;
            let dispatch = if single_encoder {
                MTLDispatchType::CONCURRENT
            } else {
                MTLDispatchType::SERIAL
            };
            let mut encoder_timing = encoder_timing.then(|| EncoderTiming::new(executor.device.clone())).transpose()?;
            let encoding = if let Some(timing) = &mut encoder_timing {
                Some(timing.encoder(&command, dispatch)?)
            } else if single_encoder {
                command.compute_command_encoder_with_dispatch_type(dispatch)
            } else {
                command.compute_command_encoder()
            }
            .ok_or(Error::Invalid("decode encoder allocation failed"))?;
            let scratch = executor
                .device
                .new_buffer(1024 * 1024, MTLResourceOptions::STORAGE_MODE_SHARED)
                .ok_or(Error::Invalid("decode scratch allocation failed"))?;
            // Status owns the first aligned region; every other allocation in
            // this command is disjoint and written before its first GPU read.
            unsafe {
                scratch.contents().as_ptr().cast::<u32>().write(0);
            }
            profile::record("native.metal.graph.scratch", 0.0, 1024 * 1024);
            Ok(Self {
                command,
                encoding,
                encoding_dispatch: dispatch,
                encoder_timing,
                failure: DeviceVector {
                    buffer: scratch.clone(),
                    offset: 0,
                    len: 1,
                },
                resources: Vec::new(),
                encoding_ended: false,
                encode_scope: Some(profile::Scope::new("native.metal.encode")),
                scratch: vec![(scratch, 256)],
                scratch_bytes: 1024 * 1024,
                single_encoder,
                #[cfg(test)]
                serial_projection_levels: std::env::var_os("STARK_QWEN38_SERIAL_LEVELS")
                    .is_some_and(|value| value == "1"),
                byte_loads: std::env::var_os("STARK_HYBRID_BYTE_LOADS").is_some_and(|value| value == "1"),
                vector8: std::env::var_os("STARK_HYBRID_VECTOR8").is_some_and(|value| value == "1"),
                scalar_uint_loads: std::env::var_os("STARK_HYBRID_SCALAR_UINT_LOADS").is_some_and(|value| value == "1"),
                factored_affine: std::env::var_os("STARK_HYBRID_FACTORED_AFFINE").is_some_and(|value| value == "1"),
                projection_level: false,
                projection_checks: Vec::new(),
                separate_finite: std::env::var_os("STARK_HYBRID_SEPARATE_FINITE").is_some_and(|value| value == "1"),
                #[cfg(test)]
                phase_timing,
            })
        })
    }

    pub fn upload(values: &[f32]) -> Result<DeviceVector> {
        if values.is_empty() || values.len() > 8_000_000 || values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("invalid decode upload"));
        }
        let _scope = profile::Scope::bytes("native.metal.upload", values.len() * 4);
        Ok(DeviceVector {
            buffer: buffer(executor()?, bytemuck::cast_slice(values))?,
            offset: 0,
            len: values.len(),
        })
    }

    pub fn upload_into(
        &mut self,
        values: &[f32],
    ) -> Result<DeviceVector> {
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("invalid decode upload"));
        }
        let output = self.scratch(values.len())?;
        let _scope = profile::Scope::bytes("native.metal.upload", values.len() * 4);
        // This fresh region is disjoint from every encoded operand. The graph
        // has not been committed; host writes finish before GPU execution.
        unsafe {
            output
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(output.offset)
                .cast::<f32>()
                .copy_from_nonoverlapping(values.as_ptr(), values.len());
        }
        Ok(output)
    }

    pub fn zeros(len: usize) -> Result<DeviceVector> {
        let bytes = len
            .checked_mul(4)
            .filter(|&n| n <= executor().map_or(0, |e| e.device.max_buffer_length()))
            .ok_or(Error::Invalid("decode allocation bound exceeded"))?;
        if len == 0 || len > u32::MAX as usize {
            return Err(Error::Invalid("decode allocation bound exceeded"));
        }
        let buffer = executor()?
            .device
            .new_buffer(bytes, MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or(Error::Invalid("decode vector allocation failed"))?;
        // Newly allocated owned shared storage has no pending GPU users.
        unsafe {
            buffer.contents().as_ptr().cast::<u8>().write_bytes(0, len * 4);
        }
        Ok(DeviceVector {
            buffer,
            offset: 0,
            len,
        })
    }

    pub fn encoder(&self) -> &ProtocolObject<dyn MTLComputeCommandEncoder> {
        &self.encoding
    }

    /// The legacy transport gets dependency ordering from serial encoders and
    /// encoder transitions. The single concurrent encoder requires a barrier
    /// before every dependent dispatch. Callers inside a projection level own
    /// the contract that their projections have no dependencies on each other.
    fn order_dispatch(&self) {
        if let Some(timing) = &self.encoder_timing {
            timing.mark_dispatch();
        }
        if self.single_encoder && !self.projection_level {
            self.encoding.memory_barrier_with_scope(MTLBarrierScope::Buffers);
        }
    }

    /// Graph-owned outputs share aligned backing allocations. Persistent model
    /// caches use `zeros` separately, so a subsequent token cannot overwrite
    /// any recurrent/cache state. Returned views retain their backing buffer
    /// when a GLM routing boundary carries them into the next command.
    pub fn scratch(
        &mut self,
        len: usize,
    ) -> Result<DeviceVector> {
        if len == 0 || len > 8_000_000 {
            return Err(Error::Invalid("decode scratch bound exceeded"));
        }
        let bytes = len * 4;
        let aligned =
            bytes.checked_add(255).map(|n| n / 256 * 256).ok_or(Error::Invalid("decode scratch size overflow"))?;
        if self
            .scratch
            .last()
            .is_none_or(|(buffer, used)| used.checked_add(aligned).is_none_or(|n| n > buffer.length()))
        {
            let size = aligned.max(1024 * 1024);
            let total = self
                .scratch_bytes
                .checked_add(size)
                .filter(|&n| n <= 512 * 1024 * 1024)
                .ok_or(Error::Invalid("decode command workspace exceeds 512 MiB"))?;
            let buffer = executor()?
                .device
                .new_buffer(size, MTLResourceOptions::STORAGE_MODE_SHARED)
                .ok_or(Error::Invalid("decode scratch allocation failed"))?;
            self.scratch.push((buffer, 0));
            self.scratch_bytes = total;
            profile::record("native.metal.graph.scratch", 0.0, size);
        }
        let (buffer, used) = self.scratch.last_mut().ok_or(Error::Invalid("decode scratch unavailable"))?;
        let output = DeviceVector {
            buffer: buffer.clone(),
            offset: *used,
            len,
        };
        *used += aligned;
        Ok(output)
    }
    pub fn failure(&self) -> &DeviceVector {
        &self.failure
    }

    pub fn kernel(
        &mut self,
        name: &'static str,
        vectors: &[&DeviceVector],
        parameters: &[u32],
        grid: usize,
        threads: usize,
    ) -> Result<()> {
        if vectors.len() >= 30 || parameters.is_empty() || parameters.len() > 64 || grid == 0 || threads == 0 {
            return Err(Error::Invalid("invalid decode kernel geometry"));
        }
        let executor = executor()?;
        let pipeline = {
            let mut cache =
                executor.graph_pipelines.lock().map_err(|_| Error::Invalid("decode pipeline cache failed"))?;
            if let Some(pipeline) = cache.get(name) {
                pipeline.clone()
            } else {
                let function = executor
                    .library
                    .new_function_with_name(name)
                    .ok_or(Error::Invalid("decode function unavailable"))?;
                let pipeline = executor
                    .device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|_| Error::Invalid("decode pipeline creation failed"))?;
                cache.insert(name, pipeline.clone());
                pipeline
            }
        };
        if threads > pipeline.max_total_threads_per_threadgroup() {
            return Err(Error::Invalid("decode kernel thread limit"));
        }
        self.encoding.set_compute_pipeline_state(&pipeline);
        for (index, vector) in vectors.iter().enumerate() {
            vector.validate()?;
            self.encoding.set_buffer(Some(&vector.buffer), vector.offset, index);
            self.resources.push(vector.buffer.clone());
        }
        self.encoding.set_bytes(
            NonNull::new(parameters.as_ptr().cast_mut()).unwrap().cast(),
            std::mem::size_of_val(parameters),
            vectors.len(),
        );
        self.encoding.set_buffer(Some(&self.failure.buffer), self.failure.offset, 30);
        self.order_dispatch();
        self.encoding.dispatch_threads(
            MTLSize {
                width: grid,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    pub fn check_finite(
        &mut self,
        input: &DeviceVector,
    ) -> Result<()> {
        if self.projection_level {
            input.validate()?;
            self.projection_checks.push(input.clone());
            return Ok(());
        }
        self.kernel("hybrid_graph_finite", &[input], &[input.len as u32], input.len, 256)
    }

    /// Encode independent projections together. Inputs must have been produced
    /// before this level; no projection may consume another projection's output.
    /// The dependency-level schedule follows Lily's design at e22d58f, expressed
    /// with Stark's tracked Metal encoder and unchanged F32 operators.
    #[cfg(test)]
    pub fn begin_projection_level(&mut self) -> Result<()> {
        if self.projection_level {
            return Err(Error::Invalid("nested decode projection level"));
        }
        if !self.serial_projection_levels {
            if self.single_encoder {
                self.order_dispatch();
            } else {
                self.switch_dispatch(MTLDispatchType::CONCURRENT)?;
            }
            self.projection_level = true;
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn end_projection_level(&mut self) -> Result<()> {
        if self.serial_projection_levels {
            return Ok(());
        }
        if !self.projection_level {
            return Err(Error::Invalid("decode projection level not active"));
        }
        // All projection outputs become visible before deferred finite scans.
        // Scans retain the same sticky status and run before dependent work.
        self.encoding.memory_barrier_with_scope(MTLBarrierScope::Buffers);
        self.projection_level = false;
        let checks = std::mem::take(&mut self.projection_checks);
        let has_checks = !checks.is_empty();
        for input in checks {
            self.check_finite(&input)?;
        }
        if has_checks {
            self.encoding.memory_barrier_with_scope(MTLBarrierScope::Buffers);
        }
        if !self.single_encoder {
            self.switch_dispatch(MTLDispatchType::SERIAL)?;
        }
        profile::record("native.metal.graph.projection_levels", 0.0, 1);
        Ok(())
    }

    pub fn copy_into(
        &mut self,
        input: &DeviceVector,
        output: &DeviceVector,
    ) -> Result<()> {
        if input.len != output.len {
            return Err(Error::Invalid("decode copy length mismatch"));
        }
        input.validate()?;
        output.validate()?;
        if std::ptr::eq(&*input.buffer, &*output.buffer) {
            if input.offset == output.offset {
                return self.check_finite(input);
            }
            if input.offset < output.offset + output.len * 4 && output.offset < input.offset + input.len * 4 {
                return Err(Error::Invalid("decode copy has overlapping device spans"));
            }
        }
        self.kernel("hybrid_graph_copy", &[input, output], &[output.len as u32, input.len as u32], output.len, 256)
    }

    pub fn repeat(
        &mut self,
        input: &DeviceVector,
        repeats: usize,
    ) -> Result<DeviceVector> {
        let output = self.scratch(input.len.checked_mul(repeats).ok_or(Error::Invalid("decode repeat overflow"))?)?;
        self.kernel("hybrid_graph_copy", &[input, &output], &[output.len as u32, input.len as u32], output.len, 256)?;
        Ok(output)
    }

    #[cfg(test)]
    pub fn add(
        &mut self,
        a: &DeviceVector,
        b: &DeviceVector,
    ) -> Result<DeviceVector> {
        if a.len != b.len {
            return Err(Error::Invalid("decode addition length mismatch"));
        }
        let output = self.scratch(a.len)?;
        self.kernel("hybrid_graph_add", &[a, b, &output], &[a.len as u32], a.len, 256)?;
        Ok(output)
    }

    pub fn rms_norm(
        &mut self,
        input: &DeviceVector,
        weight: &DeviceVector,
        group: usize,
        eps: f32,
    ) -> Result<DeviceVector> {
        self.rms_norm_kernel(
            input,
            weight,
            group,
            eps,
            if executor()?.ordered_rms {
                "hybrid_graph_rms_ordered"
            } else {
                "hybrid_graph_rms"
            },
        )
    }

    fn rms_norm_kernel(
        &mut self,
        input: &DeviceVector,
        weight: &DeviceVector,
        group: usize,
        eps: f32,
        name: &'static str,
    ) -> Result<DeviceVector> {
        if group == 0
            || !input.len.is_multiple_of(group)
            || ![group, input.len].contains(&weight.len)
            || !eps.is_finite()
            || eps <= 0.0
        {
            return Err(Error::Invalid("decode normalization geometry"));
        }
        let output = self.scratch(input.len)?;
        self.kernel(
            name,
            &[input, weight, &output],
            &[(input.len / group) as u32, group as u32, eps.to_bits(), weight.len as u32],
            (input.len / group) * 256,
            256,
        )?;
        Ok(output)
    }

    pub fn swiglu(
        &mut self,
        gate: &DeviceVector,
        up: &DeviceVector,
        clamp: Option<f32>,
    ) -> Result<DeviceVector> {
        if gate.len != up.len || clamp.is_some_and(|v| !v.is_finite() || v <= 0.0) {
            return Err(Error::Invalid("decode activation geometry"));
        }
        let out = self.scratch(gate.len)?;
        self.kernel(
            "hybrid_graph_swiglu",
            &[gate, up, &out],
            &[gate.len as u32, clamp.unwrap_or(0.0).to_bits()],
            gate.len,
            256,
        )?;
        Ok(out)
    }

    pub fn weighted_sum(
        &mut self,
        experts: &DeviceVector,
        probabilities: &[f32],
        shared: &DeviceVector,
        shared_gate: f32,
    ) -> Result<DeviceVector> {
        if !shared_gate.is_finite() || probabilities.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(Error::Invalid("decode mixture values"));
        }
        let probabilities = self.upload_into(probabilities)?;
        let shared_gate = self.upload_into(&[shared_gate])?;
        self.weighted_sum_device(experts, &probabilities, shared, &shared_gate)
    }

    pub fn weighted_sum_device(
        &mut self,
        experts: &DeviceVector,
        probabilities: &DeviceVector,
        shared: &DeviceVector,
        shared_gate: &DeviceVector,
    ) -> Result<DeviceVector> {
        if shared.len.checked_mul(probabilities.len) != Some(experts.len) || shared_gate.len != 1 {
            return Err(Error::Invalid("decode mixture geometry"));
        }
        let out = self.scratch(shared.len)?;
        self.kernel(
            "hybrid_graph_weighted_sum",
            &[experts, probabilities, shared, shared_gate, &out],
            &[shared.len as u32, probabilities.len as u32],
            shared.len,
            256,
        )?;
        Ok(out)
    }

    pub fn mapped(
        &mut self,
        operation: &MappedOperation<'_>,
        input: &DeviceVector,
    ) -> Result<DeviceVector> {
        if self.single_encoder && operation.matrices.len() > 1 {
            return Err(Error::Invalid("single encoder requires one mapped matrix per dispatch"));
        }
        input.validate()?;
        let parameters = operation.parameters_for_input(input.len)?;
        let output = self.scratch(operation.rows * operation.matrices.len())?;
        let executor = executor()?;
        let checked = matches!(operation.format, MatrixFormat::Affine { .. }) && !self.separate_finite;
        let typed = checked
            && !self.byte_loads
            && operation.matrices.iter().all(|matrix| matrix.aligned_affine_loads(operation.format));
        let chunk32 = typed && !self.vector8 && operation.columns.is_multiple_of(32);
        let uint4 = chunk32
            && !self.scalar_uint_loads
            && operation.matrices.iter().all(|matrix| matrix.codes.offset.is_multiple_of(16));
        let pipeline = if uint4 && self.factored_affine {
            executor.cached_pipeline(
                operation.format,
                "hybrid_graph_affine_matvec_factored",
                &executor.factored_matrix_pipelines,
            )?
        } else if uint4 {
            executor.cached_pipeline(
                operation.format,
                "hybrid_graph_affine_matvec_chunk32_uint4",
                &executor.uint4_matrix_pipelines,
            )?
        } else if chunk32 {
            executor.cached_pipeline(
                operation.format,
                "hybrid_graph_affine_matvec_chunk32",
                &executor.chunk32_matrix_pipelines,
            )?
        } else if typed {
            executor.cached_pipeline(
                operation.format,
                "hybrid_graph_affine_matvec_uint",
                &executor.typed_matrix_pipelines,
            )?
        } else if checked {
            executor.cached_pipeline(
                operation.format,
                "hybrid_graph_affine_matvec_vector8",
                &executor.checked_matrix_pipelines,
            )?
        } else {
            executor.pipeline(operation.format)?
        };
        if !checked {
            self.check_finite(input)?;
        }
        // Routed experts write disjoint row ranges. Preserve their previous
        // concurrent execution inside this command instead of serializing the
        // independent GEMVs when joining surrounding decode stages.
        let concurrent = operation.matrices.len() > 1
            && std::env::var_os("STARK_HYBRID_CONCURRENT_FFN").is_none_or(|value| value != "0");
        if concurrent {
            self.switch_dispatch(MTLDispatchType::CONCURRENT)?;
        }
        if checked {
            self.encoding.set_buffer(Some(&self.failure.buffer), self.failure.offset, 30);
        }
        self.order_dispatch();
        Executor::encode_matrix_at(
            &self.encoding,
            operation,
            &parameters,
            &pipeline,
            (&input.buffer, input.offset),
            (&output.buffer, output.offset),
        );
        if concurrent {
            self.encoding.memory_barrier_with_scope(MTLBarrierScope::Buffers);
            self.switch_dispatch(MTLDispatchType::SERIAL)?;
            profile::record("native.metal.graph.concurrent_experts", 0.0, operation.matrices.len());
        }
        self.resources.push(input.buffer.clone());
        for matrix in &operation.matrices {
            self.resources.push(matrix.codes.buffer.clone());
            if let Some((scales, biases)) = &matrix.planes {
                self.resources.extend([scales.buffer.clone(), biases.buffer.clone()]);
            }
        }
        if !checked {
            self.check_finite(&output)?;
        }
        Ok(output)
    }

    fn switch_dispatch(
        &mut self,
        dispatch: MTLDispatchType,
    ) -> Result<()> {
        if self.single_encoder {
            return Err(Error::Invalid("single decode encoder cannot switch dispatch type"));
        }
        self.encoding.end_encoding();
        self.encoding_ended = true;
        let encoding = if let Some(timing) = &mut self.encoder_timing {
            Some(timing.encoder(&self.command, dispatch)?)
        } else {
            self.command.compute_command_encoder_with_dispatch_type(dispatch)
        }
        .ok_or(Error::Invalid("decode encoder transition failed"))?;
        self.encoding = encoding;
        self.encoding_dispatch = dispatch;
        self.encoding_ended = false;
        Ok(())
    }

    #[cfg(test)]
    pub fn indexed(
        &mut self,
        matrix: &MappedMatrix,
        format: MatrixFormat,
        shape: [usize; 3],
        indices: &DeviceVector,
        selected: usize,
        input: &DeviceVector,
        batched: bool,
    ) -> Result<DeviceVector> {
        let [experts, rows, columns] = shape;
        if experts == 0
            || rows == 0
            || columns == 0
            || selected == 0
            || selected > 64
            || selected > experts
            || indices.len != selected
            || columns.checked_mul(if batched {
                selected
            } else {
                1
            }) != Some(input.len)
        {
            return Err(Error::Invalid("decode indexed matrix geometry"));
        }
        input.validate()?;
        indices.validate()?;
        let output =
            self.scratch(rows.checked_mul(selected).ok_or(Error::Invalid("decode indexed output overflow"))?)?;
        let executor = executor()?;
        MappedOperation {
            matrices: vec![matrix.clone()],
            inputs: &[],
            rows: rows.checked_mul(experts).ok_or(Error::Invalid("indexed bank overflow"))?,
            columns,
            rows_per_batch: 0,
            format,
        }
        .parameters_for_input(columns)?;
        let typed = !self.byte_loads && matrix.aligned_affine_loads(format);
        let chunk32 = typed && !self.vector8 && columns.is_multiple_of(32);
        let uint4 = chunk32 && !self.scalar_uint_loads && matrix.codes.offset.is_multiple_of(16);
        let name = match format {
            MatrixFormat::Packed(_) => "hybrid_graph_packed_indexed",
            MatrixFormat::Affine {
                ..
            } if uint4 && self.factored_affine => "hybrid_graph_affine_indexed_factored",
            MatrixFormat::Affine {
                ..
            } if uint4 => "hybrid_graph_affine_indexed_chunk32_uint4",
            MatrixFormat::Affine {
                ..
            } if chunk32 => "hybrid_graph_affine_indexed_chunk32",
            MatrixFormat::Affine {
                ..
            } if typed => "hybrid_graph_affine_indexed_uint",
            MatrixFormat::Affine {
                ..
            } => "hybrid_graph_affine_indexed",
        };
        let pipeline = executor.cached_pipeline(
            format,
            name,
            if uint4 && self.factored_affine {
                &executor.factored_indexed_pipelines
            } else if uint4 {
                &executor.uint4_indexed_pipelines
            } else if chunk32 {
                &executor.chunk32_indexed_pipelines
            } else if typed {
                &executor.typed_indexed_pipelines
            } else {
                &executor.indexed_pipelines
            },
        )?;
        // Indexed affine GEMV reports its own nonfinite, unclipped reduction.
        // Packed kernels retain the separate scans.
        let checked = matches!(format, MatrixFormat::Affine { .. }) && !self.separate_finite;
        if !checked {
            self.check_finite(input)?;
        }
        self.order_dispatch();
        self.encoding.set_compute_pipeline_state(&pipeline);
        self.encoding.set_buffer(Some(&matrix.codes.buffer), matrix.codes.offset, 0);
        self.encoding.set_buffer(Some(&input.buffer), input.offset, 1);
        self.encoding.set_buffer(Some(&output.buffer), output.offset, 2);
        let p = [rows as u32, columns as u32, experts as u32, u32::from(batched)];
        self.encoding.set_bytes(NonNull::from(&p).cast(), std::mem::size_of_val(&p), 3);
        if let Some((scales, biases)) = &matrix.planes {
            self.encoding.set_buffer(Some(&scales.buffer), scales.offset, 4);
            self.encoding.set_buffer(Some(&biases.buffer), biases.offset, 5);
            self.resources.extend([scales.buffer.clone(), biases.buffer.clone()]);
        }
        self.encoding.set_buffer(Some(&indices.buffer), indices.offset, 6);
        self.encoding.set_buffer(Some(&self.failure.buffer), 0, 30);
        self.encoding.dispatch_threadgroups(
            MTLSize {
                width: if matches!(format, MatrixFormat::Affine { .. }) {
                    rows.div_ceil(8)
                } else {
                    rows
                },
                height: selected,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        self.resources.extend([
            matrix.codes.buffer.clone(),
            input.buffer.clone(),
            output.buffer.clone(),
            indices.buffer.clone(),
        ]);
        if !checked {
            self.check_finite(&output)?;
        }
        Ok(output)
    }

    #[cfg(test)]
    pub fn qwen_hyperconnection(
        &mut self,
        weights: &weights::Weights,
        down_name: &str,
        up_name: &str,
        inject_name: Option<&str>,
        normalized: &DeviceVector,
        streams: usize,
    ) -> Result<(DeviceVector, DeviceVector)> {
        if streams == 0 || streams > 16 || !normalized.len.is_multiple_of(streams) {
            return Err(Error::Invalid("decode hyperconnection geometry"));
        }
        let reduced = weights.encode_matvec(self, down_name, normalized)?;
        self.kernel(
            "hybrid_graph_qwen_hc_activate",
            &[&reduced],
            &[reduced.len as u32, streams as u32],
            reduced.len,
            256,
        )?;
        let gates = weights.encode_matvec(self, up_name, &reduced)?;
        if gates.len != normalized.len {
            return Err(Error::Invalid("decode hyperconnection output shape"));
        }
        let inject = if let Some(name) = inject_name {
            weights.encode_matvec(self, name, normalized)?
        } else {
            Self::zeros(1)?
        };
        if inject_name.is_some() && inject.len != streams {
            return Err(Error::Invalid("decode hyperconnection injection shape"));
        }
        let dim = normalized.len / streams;
        let output = self.scratch(
            dim + if inject_name.is_some() {
                streams
            } else {
                0
            },
        )?;
        self.kernel(
            "hybrid_graph_qwen_hc_mix",
            &[&gates, normalized, &inject, &output],
            &[dim as u32, streams as u32, u32::from(inject_name.is_some())],
            dim.max(streams),
            256,
        )?;
        // The final hyperconnection has no injection users; keep a valid empty-
        // use placeholder rather than construct a zero-length device span.
        Ok((
            output.slice(0, dim)?,
            if inject_name.is_some() {
                output.slice(dim, streams)?
            } else {
                inject
            },
        ))
    }

    #[cfg(test)]
    pub fn qwen_gdn(
        &mut self,
        weights: &weights::Weights,
        prefix: &str,
        input: &DeviceVector,
        state: &mut QwenGdnState,
    ) -> Result<DeviceVector> {
        self.begin_projection_level()?;
        let projections = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"]
            .iter()
            .map(|name| weights.encode_matvec(self, &format!("{prefix}.{name}.weight"), input))
            .collect::<Result<Vec<_>>>()?;
        self.end_projection_level()?;
        let g = &state.geometry;
        let width = 2 * g.key_heads * g.key_dim + g.value_heads * g.value_dim;
        if projections[0].len != width
            || projections[1].len != g.value_heads * g.value_dim
            || projections[2].len != g.value_heads
            || projections[3].len != g.value_heads
        {
            return Err(Error::Invalid("decode GDN projection shape"));
        }
        let (conv, recur) = executor()?.gdn_pipelines()?;
        let threads = g.key_dim.max(g.value_dim).next_power_of_two();
        if threads > recur.max_total_threads_per_threadgroup() {
            return Err(Error::Unsupported("decode GDN thread width"));
        }
        let e = &self.encoding;
        e.set_compute_pipeline_state(&conv);
        for (i, b) in
            [&projections[0].buffer, &state.controls, &state.history, &state.convolved].into_iter().enumerate()
        {
            e.set_buffer(
                Some(b),
                if i == 0 {
                    projections[0].offset
                } else {
                    0
                },
                i,
            );
        }
        e.set_bytes(NonNull::from(&state.parameters).cast(), std::mem::size_of_val(&state.parameters), 4);
        e.set_buffer(Some(&self.failure.buffer), 0, 6);
        self.order_dispatch();
        e.dispatch_threads(
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        e.set_compute_pipeline_state(&recur);
        for (i, b) in [
            &state.convolved,
            &projections[1].buffer,
            &projections[2].buffer,
            &projections[3].buffer,
            &state.controls,
            &state.recurrent,
            &self.failure.buffer,
            &state.result,
        ]
        .into_iter()
        .enumerate()
        {
            e.set_buffer(
                Some(b),
                match i {
                    1..=3 => projections[i].offset,
                    _ => 0,
                },
                i,
            );
        }
        e.set_bytes(NonNull::from(&state.parameters).cast(), std::mem::size_of_val(&state.parameters), 8);
        self.order_dispatch();
        e.dispatch_threadgroups(
            MTLSize {
                width: g.value_heads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        let result = DeviceVector {
            buffer: state.result.clone(),
            offset: 0,
            len: g.value_heads * g.value_dim,
        };
        weights.encode_matvec(self, &format!("{prefix}.out_proj.weight"), &result)
    }

    /// Diagnostic only. Encoder timing labels stage samples in one command;
    /// legacy phase timing splits commands and adds synchronous completions.
    #[cfg(test)]
    pub fn diagnostic_checkpoint(
        &mut self,
        name: &'static str,
    ) -> Result<()> {
        if !self.phase_timing && self.encoder_timing.is_none() {
            return Ok(());
        }
        if self.projection_level {
            return Err(Error::Invalid("phase boundary inside projection level"));
        }
        self.encoding.end_encoding();
        self.encoding_ended = true;
        if let Some(timing) = &mut self.encoder_timing {
            timing.label_completed(name);
            self.encoding = timing.encoder(&self.command, self.encoding_dispatch)?;
            self.encoding_ended = false;
            return Ok(());
        }
        drop(self.encode_scope.take());
        self.command.commit();
        self.command.wait_until_completed();
        if self.command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Invalid("phase command failed"));
        }
        profile::record(name, self.command.gpu_end_time() - self.command.gpu_start_time(), 0);
        let command = executor()?.queue.command_buffer().ok_or(Error::Invalid("phase command allocation failed"))?;
        let encoding = if self.single_encoder {
            command.compute_command_encoder_with_dispatch_type(MTLDispatchType::CONCURRENT)
        } else {
            command.compute_command_encoder()
        }
        .ok_or(Error::Invalid("phase encoder allocation failed"))?;
        self.command = command;
        self.encoding = encoding;
        self.encoding_dispatch = if self.single_encoder {
            MTLDispatchType::CONCURRENT
        } else {
            MTLDispatchType::SERIAL
        };
        self.encoding_ended = false;
        self.encode_scope = Some(profile::Scope::new("native.metal.encode"));
        Ok(())
    }

    pub fn finish(
        mut self,
        outputs: &[DeviceVector],
    ) -> Result<Vec<Vec<f32>>> {
        if self.projection_level {
            return Err(Error::Invalid("unfinished decode projection level"));
        }
        for output in outputs {
            output.validate()?;
        }
        self.encoding.end_encoding();
        self.encoding_ended = true;
        drop(self.encode_scope.take());
        let execute = profile::Scope::new("native.metal.execute");
        if let Some(timing) = &mut self.encoder_timing {
            timing.start_clock();
        }
        {
            let _scope = profile::Scope::new("native.metal.submit");
            self.command.commit();
        }
        {
            let _scope = profile::Scope::new("native.metal.wait");
            self.command.wait_until_completed();
        }
        drop(execute);
        if self.command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Invalid("decode graph execution failed"));
        }
        profile::record("native.metal.gpu", self.command.gpu_end_time() - self.command.gpu_start_time(), 0);
        // All writers have completed. GPU status is initialized to zero and
        // sticky across every operator, including masked intermediate errors.
        if unsafe { *self.failure.buffer.contents().as_ptr().cast::<u32>() } != 0 {
            return Err(Error::Invalid("nonfinite decode graph intermediate"));
        }
        if let Some(timing) = &self.encoder_timing {
            timing.complete(self.command.gpu_end_time() - self.command.gpu_start_time())?;
        }
        let mut results = Vec::with_capacity(outputs.len());
        for output in outputs {
            let _scope = profile::Scope::bytes("native.metal.readback", output.len * 4);
            let values = unsafe {
                std::slice::from_raw_parts(
                    output.buffer.contents().as_ptr().cast::<u8>().add(output.offset).cast::<f32>(),
                    output.len,
                )
            }
            .to_vec();
            if values.iter().any(|v| !v.is_finite()) {
                return Err(Error::Invalid("nonfinite decode graph output"));
            }
            results.push(values);
        }
        Ok(results)
    }
}

impl Drop for DecodeGraph {
    fn drop(&mut self) {
        // An encoding-time error has not submitted any state mutations. End
        // its encoder before releasing retained resources and the command.
        if !self.encoding_ended {
            self.encoding.end_encoding();
        }
    }
}

#[cfg(test)]
#[path = "decode_graph_typed_tests.rs"]
mod typed_load_tests;

#[cfg(test)]
#[path = "decode_graph_parallel_tests.rs"]
mod parallel_transport_tests;

#[cfg(test)]
#[path = "decode_graph_chunk32_tests.rs"]
mod chunk32_tests;

#[cfg(test)]
#[path = "decode_graph_factored_tests.rs"]
mod factored_tests;

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn projection_levels_preserve_packed_outputs_and_order_deferred_checks() {
        let (rows, columns) = (13, 64);
        let values = (0..rows * columns).map(|i| ((i * 13 % 101) as f32 - 50.0) / 51.0).collect::<Vec<_>>();
        let codes = buffer(executor().unwrap(), bytemuck::cast_slice(&values)).unwrap();
        let matrix = MappedMatrix {
            codes: MappedSlice {
                buffer: codes,
                offset: 0,
                length: values.len() * 4,
                registered_length: values.len() * 4,
            },
            planes: None,
        };
        let input = DecodeGraph::upload(&(0..columns).map(|i| (i as f32 - 15.0) / 20.0).collect::<Vec<_>>()).unwrap();
        let mut expected = None;
        for serial in [true, false] {
            let mut graph = DecodeGraph::new().unwrap();
            graph.serial_projection_levels = serial;
            graph.begin_projection_level().unwrap();
            if !serial {
                assert!(graph.begin_projection_level().is_err());
            }
            let mut outputs = Vec::new();
            for _ in 0..4 {
                outputs.push(
                    graph
                        .mapped(
                            &MappedOperation {
                                matrices: vec![matrix.clone()],
                                inputs: &[],
                                rows,
                                columns,
                                rows_per_batch: 0,
                                format: MatrixFormat::Packed(0),
                            },
                            &input,
                        )
                        .unwrap(),
                );
            }
            graph.end_projection_level().unwrap();
            assert!(graph.projection_checks.is_empty());
            let combined = graph.add(&outputs[0], &outputs[3]).unwrap();
            outputs.push(combined);
            let actual = graph.finish(&outputs).unwrap();
            if let Some(expected) = &expected {
                assert_eq!(&actual, expected);
            } else {
                expected = Some(actual);
            }
        }

        let mut invalid = DecodeGraph::new().unwrap();
        invalid.serial_projection_levels = false;
        let input = DecodeGraph::zeros(1).unwrap();
        // Deliberately poison test storage to exercise deferred validation.
        unsafe { input.buffer.contents().as_ptr().cast::<f32>().write(f32::NAN) };
        invalid.begin_projection_level().unwrap();
        invalid.check_finite(&input).unwrap();
        invalid.end_projection_level().unwrap();
        assert!(invalid.finish(&[]).is_err());
        let mut unfinished = DecodeGraph::new().unwrap();
        unfinished.serial_projection_levels = false;
        unfinished.begin_projection_level().unwrap();
        assert!(unfinished.finish(&[]).is_err());
    }

    #[stark_test]
    fn deferred_graph_preserves_grouped_f32_math_and_device_views() {
        let x = [0.2, -0.3, 0.7, 1.1, -2.0, 0.4, 1.5, -0.8];
        let w = [1.2, 0.9, 0.8, 1.1];
        let mut graph = DecodeGraph::new().unwrap();
        let input = DecodeGraph::upload(&x).unwrap();
        let weights = DecodeGraph::upload(&w).unwrap();
        let normalized = graph.rms_norm(&input, &weights, 4, 1e-6).unwrap();
        let doubled = graph.add(&normalized, &normalized).unwrap();
        let repeated = graph.repeat(&doubled.slice(4, 4).unwrap(), 3).unwrap();
        let actual = graph.finish(&[repeated]).unwrap().remove(0);
        let expected = crate::engine::hybrid::math::rms_norm(&x[4..], &w, 1e-6).unwrap();
        assert!(
            actual.chunks_exact(4).all(|chunk| chunk.iter().zip(&expected).all(|(&a, &e)| (a - 2.0 * e).abs() < 2e-6))
        );
        assert!(input.slice(7, 2).is_err());
        assert!(input.slice(0, 0).is_err());
    }

    #[stark_test]
    fn deferred_norm_prefetch_preserves_scalar_bits_across_tiles_and_wide_groups() {
        for width in [1, 7, 1023, 1024, 1025, 2560, 3072, 10240, 12288, 16384] {
            for broadcast in [false, true] {
                let groups = 3;
                let input = (0..groups * width).map(|i| ((i * 31 % 127) as f32 - 63.0) * 0.019).collect::<Vec<_>>();
                let weights = (0..if broadcast {
                    width
                } else {
                    groups * width
                })
                    .map(|i| 0.7 + (i % 17) as f32 * 0.031)
                    .collect::<Vec<_>>();
                let mut graph = DecodeGraph::new().unwrap();
                let input = graph.upload_into(&input).unwrap();
                let weights = graph.upload_into(&weights).unwrap();
                let expected = graph.scratch(groups * width).unwrap();
                graph
                    .kernel(
                        "hybrid_graph_rms_scalar_reference",
                        &[&input, &weights, &expected],
                        &[groups as u32, width as u32, 1e-6f32.to_bits(), weights.len as u32],
                        groups,
                        1,
                    )
                    .unwrap();
                let actual = graph.rms_norm_kernel(&input, &weights, width, 1e-6, "hybrid_graph_rms_ordered").unwrap();
                let values = graph.finish(&[expected, actual]).unwrap();
                assert!(
                    values[0].iter().zip(&values[1]).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "RMS prefetch changed F32 arithmetic at width {width}"
                );
            }
        }
    }

    #[stark_test]
    fn deferred_parallel_norm_matches_f64_at_wide_and_tail_widths() {
        for width in [1, 7, 31, 33, 127, 255, 256, 257, 1023, 1025, 2560, 3072, 4096, 10240, 12288, 16384] {
            for broadcast in [false, true] {
                let groups = 3;
                let input = (0..groups * width).map(|i| ((i * 37 % 173) as f32 - 86.0) * 0.027).collect::<Vec<_>>();
                let weights = (0..if broadcast {
                    width
                } else {
                    groups * width
                })
                    .map(|i| 0.5 + (i % 19) as f32 * 0.043)
                    .collect::<Vec<_>>();
                let mut graph = DecodeGraph::new().unwrap();
                let x = graph.upload_into(&input).unwrap();
                let w = graph.upload_into(&weights).unwrap();
                let output = graph.rms_norm_kernel(&x, &w, width, 1e-6, "hybrid_graph_rms").unwrap();
                let actual = graph.finish(&[output]).unwrap().remove(0);
                for group in 0..groups {
                    let row = &input[group * width..(group + 1) * width];
                    let sum = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>();
                    let scale = (sum / width as f64 + f64::from(1e-6f32)).sqrt().recip();
                    for (d, &v) in row.iter().enumerate() {
                        let weight = weights[if broadcast {
                            d
                        } else {
                            group * width + d
                        }];
                        let expected = f64::from(v) * scale * f64::from(weight);
                        let error = (f64::from(actual[group * width + d]) - expected).abs();
                        assert!(error <= 2e-6 + 2e-6 * expected.abs(), "parallel F32 RMS differs from F64 oracle");
                    }
                }
            }
        }
    }

    #[stark_test]
    fn deferred_graph_rejects_intermediate_overflow_hidden_by_clipping() {
        let mut graph = DecodeGraph::new().unwrap();
        let x = DecodeGraph::upload(&[f32::MAX]).unwrap();
        let overflow = graph.add(&x, &x).unwrap();
        let clipped = graph.swiglu(&overflow, &DecodeGraph::upload(&[1.0]).unwrap(), Some(1.0)).unwrap();
        assert!(graph.finish(&[clipped]).is_err());
        let mut recovered = DecodeGraph::new().unwrap();
        let value =
            recovered.add(&DecodeGraph::upload(&[2.0]).unwrap(), &DecodeGraph::upload(&[3.0]).unwrap()).unwrap();
        assert_eq!(recovered.finish(&[value]).unwrap()[0], [5.0]);
    }

    #[stark_test]
    fn checked_affine_projection_preserves_bits_and_rejects_hidden_nonfinite_values() {
        let (rows, columns) = (9, 32);
        let codes = vec![0xffu8; rows * columns / 2];
        let scales = bytemuck::cast_slice(&vec![0.5f32; rows]).to_vec();
        let biases = bytemuck::cast_slice(&vec![-0.25f32; rows]).to_vec();
        let view = |bytes: &[u8]| MappedSlice {
            buffer: buffer(executor().unwrap(), bytes).unwrap(),
            offset: 0,
            length: bytes.len(),
            registered_length: bytes.len(),
        };
        let matrix = MappedMatrix {
            codes: view(&codes),
            planes: Some((view(&scales), view(&biases))),
        };
        let operation = MappedOperation {
            matrices: vec![matrix.clone()],
            inputs: &[],
            rows,
            columns,
            rows_per_batch: 0,
            format: MatrixFormat::Affine {
                scale_dtype: 0,
                bits: 4,
                group: 32,
            },
        };
        let input = (0..columns).map(|i| (i as f32 - 15.0) / 20.0).collect::<Vec<_>>();
        let expected = mlx_matvec(&codes, &scales, &biases, 0, 4, 32, rows, columns, &input).unwrap();
        let mut graph = DecodeGraph::new().unwrap();
        // Isolate finite-check/load fusion from the separately qualified
        // chunk32 reduction schedule.
        graph.vector8 = true;
        graph.begin_projection_level().unwrap();
        let output = graph.mapped(&operation, &DecodeGraph::upload(&input).unwrap()).unwrap();
        graph.end_projection_level().unwrap();
        assert_eq!(graph.finish(&[output]).unwrap()[0], expected);
        for indexed in [false, true] {
            for poison in [f32::MAX, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut graph = DecodeGraph::new().unwrap();
                let input = DecodeGraph::zeros(columns).unwrap();
                // Owned test storage has no submitted users. Poison one lane
                // without passing through upload's independent host checks.
                unsafe { input.buffer.contents().as_ptr().cast::<f32>().add(31).write(poison) };
                let projected = if indexed {
                    graph
                        .indexed(
                            &matrix,
                            operation.format,
                            [1, rows, columns],
                            &DecodeGraph::upload(&[0.0]).unwrap(),
                            1,
                            &input,
                            false,
                        )
                        .unwrap()
                } else {
                    graph.mapped(&operation, &input).unwrap()
                };
                let clipped =
                    graph.swiglu(&projected, &DecodeGraph::upload(&vec![1.0; rows]).unwrap(), Some(1.0)).unwrap();
                assert!(graph.finish(&[clipped]).is_err());
            }
        }
    }

    #[stark_test]
    fn deferred_arena_views_survive_command_boundaries_and_reject_overlapping_copies() {
        let mut first = DecodeGraph::new().unwrap();
        let input = first.upload_into(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(input.offset > 0);
        assert!(first.copy_into(&input.slice(0, 3).unwrap(), &input.slice(1, 3).unwrap()).is_err());
        let output = first.add(&input, &input).unwrap();
        first.finish(&[]).unwrap();
        let mut second = DecodeGraph::new().unwrap();
        let recovered = second.add(&output, &output).unwrap();
        assert_eq!(second.finish(&[recovered]).unwrap()[0], [4.0, 8.0, 12.0, 16.0]);
        let mut bounded = DecodeGraph::new().unwrap();
        bounded.scratch_bytes = 512 * 1024 * 1024;
        assert!(bounded.scratch(262_144).is_err());
    }

    #[stark_test]
    fn deferred_indexed_bank_matches_selected_projection_and_bounds_bad_ids() {
        let (experts, rows, columns) = (3, 8, 32);
        let values = (0..experts * rows * columns).map(|i| ((i * 13 % 101) as f32 - 50.0) / 51.0).collect::<Vec<_>>();
        let codes = buffer(executor().unwrap(), bytemuck::cast_slice(&values)).unwrap();
        let matrix = MappedMatrix {
            codes: MappedSlice {
                buffer: codes,
                offset: 0,
                length: values.len() * 4,
                registered_length: values.len() * 4,
            },
            planes: None,
        };
        let input = (0..columns).map(|i| (i as f32 - 15.0) / 20.0).collect::<Vec<_>>();
        let selected = [2usize, 0];
        let mut graph = DecodeGraph::new().unwrap();
        let indices = DecodeGraph::upload(&selected.map(|i| f32::from_bits(i as u32))).unwrap();
        let output = graph
            .indexed(
                &matrix,
                MatrixFormat::Packed(0),
                [experts, rows, columns],
                &indices,
                selected.len(),
                &DecodeGraph::upload(&input).unwrap(),
                false,
            )
            .unwrap();
        let actual = graph.finish(&[output]).unwrap().remove(0);
        for (slot, expert) in selected.iter().enumerate() {
            for row in 0..rows {
                let start = (expert * rows + row) * columns;
                let expected = values[start..start + columns].iter().zip(&input).map(|(w, x)| w * x).sum::<f32>();
                assert!((actual[slot * rows + row] - expected).abs() < 2e-6);
            }
        }
        let mut invalid = DecodeGraph::new().unwrap();
        let indices = DecodeGraph::upload(&[f32::from_bits(experts as u32)]).unwrap();
        let output = invalid
            .indexed(
                &matrix,
                MatrixFormat::Packed(0),
                [experts, rows, columns],
                &indices,
                1,
                &DecodeGraph::upload(&input).unwrap(),
                false,
            )
            .unwrap();
        assert!(invalid.finish(&[output]).is_err());
    }

    #[stark_test]
    fn deferred_affine_indexing_preserves_all_planes_groups_and_device_offsets() {
        let span = |bytes: &[u8], offset: usize| {
            let mut storage = vec![0u8; offset];
            storage.extend_from_slice(bytes);
            MappedSlice {
                buffer: buffer(executor().unwrap(), &storage).unwrap(),
                offset,
                length: bytes.len(),
                registered_length: storage.len(),
            }
        };
        for bits in [4, 8] {
            for group in [32, 64, 128] {
                for scale_dtype in [0, 1, 30] {
                    let (experts, rows, columns) = (4, 13, group * 3);
                    let codes =
                        (0..experts * rows * columns * bits / 8).map(|i| (i * 19 + 37) as u8).collect::<Vec<_>>();
                    let scales = (0..experts * rows * columns / group)
                        .map(|i| ((i % 13 + 1) as f32) * 0.013)
                        .collect::<Vec<_>>();
                    let biases = scales.iter().enumerate().map(|(i, _)| -((i % 7) as f32) * 0.071).collect::<Vec<_>>();
                    let encode = |values: &[f32]| {
                        values
                            .iter()
                            .flat_map(|&value| match scale_dtype {
                                0 => value.to_le_bytes().to_vec(),
                                1 => half::f16::from_f32(value).to_bits().to_le_bytes().to_vec(),
                                _ => half::bf16::from_f32(value).to_bits().to_le_bytes().to_vec(),
                            })
                            .collect::<Vec<_>>()
                    };
                    let matrix = MappedMatrix {
                        codes: span(&codes, 64),
                        planes: Some((span(&encode(&scales), 128), span(&encode(&biases), 192))),
                    };
                    let format = MatrixFormat::Affine {
                        scale_dtype,
                        bits,
                        group,
                    };
                    let selected = [3usize, 0, 2];
                    let input = (0..columns).map(|i| ((i % 23) as f32 - 11.0) * 0.043).collect::<Vec<_>>();
                    let matrix_size = codes.len() / experts;
                    let plane_size = matrix.planes.as_ref().unwrap().0.length / experts;
                    let matrices = selected
                        .iter()
                        .map(|&e| MappedMatrix {
                            codes: matrix.codes.subslice(e * matrix_size, matrix_size).unwrap(),
                            planes: Some((
                                matrix.planes.as_ref().unwrap().0.subslice(e * plane_size, plane_size).unwrap(),
                                matrix.planes.as_ref().unwrap().1.subslice(e * plane_size, plane_size).unwrap(),
                            )),
                        })
                        .collect::<Vec<_>>();
                    for batched in [false, true] {
                        let inputs = if batched {
                            (0..selected.len())
                                .flat_map(|e| input.iter().map(move |v| v * (e + 1) as f32))
                                .collect::<Vec<_>>()
                        } else {
                            input.repeat(selected.len())
                        };
                        let operation = MappedOperation {
                            matrices: matrices.clone(),
                            inputs: &inputs,
                            rows,
                            columns,
                            rows_per_batch: 0,
                            format,
                        };
                        let expected = mapped_many(&[operation]).unwrap().remove(0);
                        let mut selected_graph = DecodeGraph::new().unwrap();
                        selected_graph.vector8 = true;
                        let selected_input = selected_graph.upload_into(&inputs).unwrap();
                        let selected_output = selected_graph
                            .mapped(
                                &MappedOperation {
                                    matrices: matrices.clone(),
                                    inputs: &[],
                                    rows,
                                    columns,
                                    rows_per_batch: 0,
                                    format,
                                },
                                &selected_input,
                            )
                            .unwrap();
                        let actual_selected = selected_graph.finish(&[selected_output]).unwrap().remove(0);
                        assert!(
                            actual_selected.iter().zip(&expected).all(|(a, e)| a.to_bits() == e.to_bits()),
                            "concurrent selected projection changed F32 arithmetic"
                        );
                        let mut graph = DecodeGraph::new().unwrap();
                        graph.vector8 = true;
                        let ids = DecodeGraph::upload(&[0.0, f32::from_bits(3), f32::from_bits(0), f32::from_bits(2)])
                            .unwrap();
                        let mut uploaded = vec![0.0; 7];
                        uploaded.extend_from_slice(if batched {
                            &inputs
                        } else {
                            &input
                        });
                        let device_input = DecodeGraph::upload(&uploaded)
                            .unwrap()
                            .slice(
                                7,
                                if batched {
                                    inputs.len()
                                } else {
                                    input.len()
                                },
                            )
                            .unwrap();
                        let output = graph
                            .indexed(
                                &matrix,
                                format,
                                [experts, rows, columns],
                                &ids.slice(1, 3).unwrap(),
                                3,
                                &device_input,
                                batched,
                            )
                            .unwrap();
                        let actual = graph.finish(&[output]).unwrap().remove(0);
                        assert!(
                            actual.iter().zip(expected).all(|(&a, e)| a.to_bits() == e.to_bits()),
                            "indexed affine arithmetic changed"
                        );
                    }
                }
            }
        }
    }
}
