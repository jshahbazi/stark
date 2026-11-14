use std::collections::{HashMap, hash_map::Entry};

use super::{
    super::{MatmulOutputWork, supports_integer_right_operand},
    GemmEngine, GemmPlan,
    selection::{GemmProblem, outer_block_k},
    specialization::GemmSpecialization,
};
use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding,
            gpu_types::{
                GemmParams,
                gemm::{GemmAPrologueKind, GemmAlignment, GemmBPrologueKind, GemmDTransform, GemmTiling},
            },
            kernel::matmul::{
                Int8CodeLayout, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulError, MatmulShape,
                QuantParamsLayout, QuantParamsStrides, QuantizedB,
            },
        },
        metal::{
            Metal,
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            error::MetalError,
            kernel::{GemmMetalKernel, GemmSplitKReduceMetalKernel, SourceBF16SplitReduceMetalKernel},
        },
    },
    data_type::DataType,
};

pub struct GemmKernel {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    kernels: HashMap<GemmSpecialization, GemmMetalKernel>,
    split_k_reduce: HashMap<GemmDTransform, GemmSplitKReduceMetalKernel>,
    source_split_k_reduce: Option<SourceBF16SplitReduceMetalKernel>,
}

#[cfg(test)]
#[path = "../../../../../../unit/backends/metal/kernel/matmul/gemm/precision_test.rs"]
mod precision_tests;

impl GemmKernel {
    pub fn new(
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Self {
        Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            kernels: HashMap::new(),
            split_k_reduce: HashMap::new(),
            source_split_k_reduce: None,
        }
    }

    fn get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: GemmSpecialization,
    ) -> Result<&GemmMetalKernel, MetalError> {
        match self.kernels.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = GemmMetalKernel::new(
                    context,
                    self.input_data_type,
                    self.weights_data_type,
                    self.output_data_type,
                    specialization.tiling,
                    specialization.transpose_b,
                    specialization.use_mxu,
                    specialization.b_prologue,
                    specialization.bits_per_b.unwrap_or(0),
                    specialization.b_group_size.unwrap_or(0),
                    specialization.a_prologue,
                    specialization.a_group_size.unwrap_or(0),
                    specialization.source_f32_fragments,
                    specialization.output_transform,
                    specialization.alignment,
                    specialization.signed_codes,
                )?;
                Ok(entry.insert(kernel))
            },
        }
    }

    fn get_or_create_split_k_reduce(
        &mut self,
        context: &MetalContext,
        output_transform: GemmDTransform,
    ) -> Result<&GemmSplitKReduceMetalKernel, MetalError> {
        match self.split_k_reduce.entry(output_transform) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = GemmSplitKReduceMetalKernel::new(context, self.output_data_type, output_transform)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    fn problem(
        &self,
        shape: MatmulShape,
        context: &MetalContext,
    ) -> GemmProblem {
        GemmProblem::new(
            shape,
            self.weights_data_type,
            self.output_data_type,
            context.supports_mxu,
            context.apple_gpu_family,
        )
    }

    pub(crate) fn encode_source_qmm(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferMut<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        output_work: &MatmulOutputWork,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MetalError> {
        let shape = MatmulShape::from_arguments(&arguments);
        // The declared source QMM uses 32x32x32 tiles and a 2x2 SIMDgroup
        // BlockMMA. Keep its accumulation schedule independently of MXU policy.
        let mut plan = GemmPlan {
            engine: GemmEngine::Simdgroup,
            tiling: GemmTiling::Tile32x32x32_Simdgroups2x2,
            split_k: 1,
        };
        if arguments.m <= 1 {
            return Err(MatmulError::<Metal>::UnsupportedLayout {
                path: "source BF16 QMM requires multiple rows",
            }
            .into());
        }
        // MLX v0.32.3 quantized.cpp qmm_splitk: use original logical
        // projection dimensions, target 512 32x32 tiles, and whole groups.
        let tiles = shape.m.div_ceil(32).saturating_mul(shape.n.div_ceil(32));
        let alignment = shape.b_group_size.unwrap_or(0).max(32);
        let mut partitions = (512 / tiles.max(1)).max(1).min(shape.k / alignment);
        while partitions > 1 && !shape.k.is_multiple_of(partitions * alignment) {
            partitions -= 1;
        }
        if partitions > 256 {
            return Err(MatmulError::<Metal>::UnsupportedLayout {
                path: "source QMM two-pass reduction is outside coverage",
            }
            .into());
        }
        if partitions == 1 && command_buffer.context().supports_mxu {
            // MLX qmm_splitk falls through to qmm_nax when there is only
            // one partition. NAX retains BF16 operands and F32 accumulators.
            // Match its BM32/64, BN64, BK64 logical group schedule.
            plan.engine = GemmEngine::Mxu;
            plan.tiling = if shape.m <= 32 {
                GemmTiling::Tile32x64x256_Simdgroups2x2
            } else {
                GemmTiling::Tile64x64x256_Simdgroups2x2
            };
        }
        plan.split_k = partitions.max(1);
        self.encode_plan(arguments, plan, output_work, command_buffer)
    }

    #[cfg(test)]
    pub fn select_plan_for_engine(
        &self,
        shape: &MatmulShape,
        engine: GemmEngine,
        context: &MetalContext,
    ) -> Result<GemmPlan, MetalError> {
        self.problem(*shape, context)
            .select_plan_for_engine(engine)
            .map_err(|error| MetalError::KernelDispatchFailed(Box::new(error)))
    }

    pub fn encode_plan(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferMut<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        plan: GemmPlan,
        output_work: &MatmulOutputWork,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MetalError> {
        let shape = MatmulShape::from_arguments(&arguments);
        self.problem(shape, command_buffer.context())
            .validate_engine(plan.engine)
            .map_err(|error| MetalError::KernelDispatchFailed(Box::new(error)))?;

        if !matches!(arguments.b, MatmulB::FullPrecision { .. })
            && arguments.d_transform.mask().contains(GemmDTransform::ACCUMULATE)
        {
            return Err(MatmulError::UnsupportedDOp {
                bit: GemmDTransform::ACCUMULATE,
                path: "QuantGemm",
            }
            .into());
        }

        let MatmulArguments {
            a,
            b,
            d,
            d_transform,
            ..
        } = arguments;
        match b {
            MatmulB::FullPrecision {
                b: weights,
            } => self.encode_weights(a, weights, None, d, d_transform, shape, plan, output_work, command_buffer),
            MatmulB::Quantized(quantized) => self.encode_weights(
                a,
                quantized.codes,
                Some(quantized),
                d,
                d_transform,
                shape,
                plan,
                output_work,
                command_buffer,
            ),
        }
    }

    fn encode_weights<WB: BufferRef<Backend = Metal>>(
        &mut self,
        a: MatmulA<impl BufferRef<Backend = Metal>>,
        weights: WB,
        quantized: Option<QuantizedB<WB>>,
        mut d: impl BufferMut<Backend = Metal>,
        d_transform: MatmulDOps<'_, Metal>,
        shape: MatmulShape,
        plan: GemmPlan,
        output_work: &MatmulOutputWork,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MetalError> {
        let (m, n, k) = (shape.m, shape.n, shape.k);
        let ab_scale = d_transform.ab_scale;
        let output_bias = d_transform.bias;
        let rht_factors = d_transform.rht_factors;
        let output_transform = d_transform.mask();
        let (output_bias, bias_after_rht, output_transform) =
            if plan.split_k > 1 && rht_factors.is_some() && output_bias.is_some() {
                (None, output_bias, output_transform.difference(GemmDTransform::BIAS))
            } else {
                (output_bias, None, output_transform)
            };

        let use_mxu = plan.engine == GemmEngine::Mxu;

        let is_quant = quantized.is_some();
        let source_bf16_affine = quantized.as_ref().is_some_and(|q| q.params.materializes_affine_input_sum());
        if source_bf16_affine
            && !(m > 1
                && self.weights_data_type == DataType::BF16
                && self.input_data_type == DataType::BF16
                && self.output_data_type == DataType::BF16
                && matches!(&a, MatmulA::FullPrecision { .. })
                && shape.b_prologue == GemmBPrologueKind::ScaleBiasDequant
                && shape.b_bits == Some(4)
                && shape.b_group_size == Some(64)
                && shape.b_transpose
                && !shape.signed_codes
                && output_transform.is_empty())
        {
            return Err(MatmulError::<Metal>::UnsupportedLayout {
                path: "source BF16 QMM supports unbiased unsigned W4/group64",
            }
            .into());
        }
        if !is_quant && !matches!(&a, MatmulA::FullPrecision { .. }) {
            return Err(MatmulError::IncompatibleA {
                path: "Gemm",
                reason: "int8 activations require quantized weights",
            }
            .into());
        }
        let (scales, biases, zero_points, scale_strides, zero_point_strides) = match quantized.as_ref() {
            None => (None, None, None, Default::default(), Default::default()),
            Some(quantized) => (
                Some(quantized.scales),
                quantized.biases(),
                quantized.zero_points(),
                quantized.params.scale_strides(),
                quantized.zero_point_strides(),
            ),
        };
        let a_prologue = a.prologue_kind();
        let (a_full_precision, a_int8, a_scales, a_group_sums, a_group_size) = match &a {
            MatmulA::FullPrecision {
                values,
                offset,
            } => (Some(values.subrange(*offset..)), None, None, None, None),
            MatmulA::Int8Symmetric {
                values,
                scales: activation_scales,
                group_sums: activation_group_sums,
                scale_group_size,
                code_layout,
            } => {
                validate_int8_left_operand(
                    use_mxu,
                    shape,
                    *scale_group_size,
                    *code_layout,
                    activation_group_sums.is_some(),
                )?;
                if output_transform.contains(GemmDTransform::SOFT_CAP) {
                    return Err(MatmulError::UnsupportedDOp {
                        bit: GemmDTransform::SOFT_CAP,
                        path: "Gemm int8 left operand",
                    }
                    .into());
                }
                (None, Some(*values), Some(*activation_scales), *activation_group_sums, Some(*scale_group_size))
            },
        };

        if plan.split_k > 1 {
            self.encode_split_k(
                a,
                weights,
                scales,
                biases,
                zero_points,
                d.reborrow(),
                ab_scale,
                scale_strides,
                zero_point_strides,
                shape,
                plan,
                output_transform,
                output_bias,
                source_bf16_affine,
                command_buffer,
            )?;
            if let Some(factors) = rht_factors {
                output_work.apply(d.reborrow(), factors, bias_after_rht, m, n, command_buffer);
            }
            return Ok(());
        }

        let tiling = plan.tiling;
        let alignment =
            GemmAlignment::new(m % tiling.block_m() == 0, n % tiling.block_n() == 0, k % tiling.block_k() == 0);
        let mut params = gemm_params(shape, plan, ab_scale, scale_strides, zero_point_strides);
        let threadgroups_per_row = n.div_ceil(tiling.block_n());
        let threadgroups_per_column = m.div_ceil(tiling.block_m());
        let (use_morton, group_count_x, group_count_y) = if !is_quant && use_mxu {
            let max_dim = threadgroups_per_row.max(threadgroups_per_column);
            let min_dim = threadgroups_per_row.min(threadgroups_per_column);
            let morton_dim = max_dim.next_power_of_two();
            let morton_total = morton_dim.saturating_mul(morton_dim);
            let actual_total = threadgroups_per_row.saturating_mul(threadgroups_per_column);
            if min_dim > 1 && morton_total <= 4_u32.saturating_mul(actual_total) {
                (true, morton_total, 1)
            } else {
                (false, threadgroups_per_row, threadgroups_per_column)
            }
        } else {
            (false, threadgroups_per_row, threadgroups_per_column)
        };
        if !is_quant {
            params.leading_dimension_b = shape.b_leading_dimension.unwrap_or(if shape.b_transpose {
                k
            } else {
                n
            });
        }
        params.use_morton = use_morton;

        let mut specialization = GemmSpecialization::from_plan(
            plan,
            shape,
            self.weights_data_type,
            output_transform,
            alignment,
            a_prologue,
            a_group_size,
        )?;
        specialization.source_f32_fragments = source_bf16_affine && !use_mxu;
        let kernel = self.get_or_create(command_buffer.context(), specialization)?;
        kernel.encode(
            a_full_precision,
            weights,
            d.reborrow(),
            scales,
            biases,
            zero_points,
            output_bias,
            rht_factors,
            a_int8,
            a_scales,
            a_group_sums,
            std::slice::from_ref(&params),
            group_count_x,
            group_count_y,
            1,
            command_buffer,
        );

        Ok(())
    }

    fn encode_split_k(
        &mut self,
        a: MatmulA<impl BufferRef<Backend = Metal>>,
        weights: impl BufferRef<Backend = Metal>,
        scales: Option<impl BufferRef<Backend = Metal>>,
        biases: Option<impl BufferRef<Backend = Metal>>,
        zero_points: Option<impl BufferRef<Backend = Metal>>,
        mut d: impl BufferMut<Backend = Metal>,
        ab_scale: f32,
        scale_strides: QuantParamsStrides,
        zero_point_strides: QuantParamsStrides,
        shape: MatmulShape,
        plan: GemmPlan,
        output_transform: GemmDTransform,
        output_bias: Option<impl BufferRef<Backend = Metal>>,
        source_bf16_affine: bool,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MetalError> {
        let (m, n, k) = (shape.m, shape.n, shape.k);
        let (a_full_precision, a_int8, a_scales, a_group_sums, a_prologue, a_group_size) = match a {
            MatmulA::FullPrecision {
                values,
                offset,
            } => (Some(values.subrange(offset..)), None, None, None, GemmAPrologueKind::FullPrecision, None),
            MatmulA::Int8Symmetric {
                values,
                scales,
                group_sums,
                scale_group_size,
                code_layout: _,
            } => {
                (None, Some(values), Some(scales), group_sums, GemmAPrologueKind::Int8Symmetric, Some(scale_group_size))
            },
        };
        let tiling = plan.tiling;
        let split_k = plan.split_k;
        let kp = k / split_k;
        let k_step = outer_block_k(shape, plan.engine, plan.tiling).unwrap_or(1);
        let base_gx = n.div_ceil(tiling.block_n());
        let base_gy = m.div_ceil(tiling.block_m());
        let alignment =
            GemmAlignment::new(m.is_multiple_of(tiling.block_m()), n.is_multiple_of(tiling.block_n()), true);
        let mut part_spec = GemmSpecialization::from_plan(
            plan,
            shape,
            self.weights_data_type,
            GemmDTransform::empty(),
            alignment,
            a_prologue,
            a_group_size,
        )?;
        part_spec.source_f32_fragments = source_bf16_affine && plan.engine == GemmEngine::Simdgroup;

        let elem = (m as usize) * (n as usize);
        let slice_bytes = elem * self.output_data_type.size_in_bytes();
        let mut temp = command_buffer.allocate_scratch(split_k as usize * slice_bytes)?;
        let mut params = gemm_params(shape, plan, 1.0, scale_strides, zero_point_strides);
        params.aligned_inner_iterations = kp / k_step;
        let part_kernel = self.get_or_create(command_buffer.context(), part_spec)?;
        part_kernel.encode(
            a_full_precision,
            weights,
            &mut temp,
            scales,
            biases,
            zero_points,
            None::<&<Metal as Backend>::GlobalBuffer>,
            None::<&<Metal as Backend>::GlobalBuffer>,
            a_int8,
            a_scales,
            a_group_sums,
            std::slice::from_ref(&params),
            base_gx,
            base_gy,
            split_k,
            command_buffer,
        );

        if source_bf16_affine {
            if self.source_split_k_reduce.is_none() {
                self.source_split_k_reduce = Some(SourceBF16SplitReduceMetalKernel::new(command_buffer.context())?);
            }
            self.source_split_k_reduce.as_ref().unwrap().encode(
                &temp,
                d.reborrow(),
                elem as u32,
                split_k,
                command_buffer,
            );
            return Ok(());
        }

        debug_assert_eq!(elem % 4, 0, "split-K reduce requires M*N divisible by 4");
        let group_count = ((elem as u32) / 4).div_ceil(256);
        let reduce_transform =
            output_transform.intersection(GemmDTransform::SCALE | GemmDTransform::ACCUMULATE | GemmDTransform::BIAS);
        let bias_arg = if reduce_transform.contains(GemmDTransform::BIAS) {
            output_bias
        } else {
            None
        };
        let scale_arg = if reduce_transform.contains(GemmDTransform::SCALE) {
            Some(ab_scale)
        } else {
            None
        };
        let reduce = self.get_or_create_split_k_reduce(command_buffer.context(), reduce_transform)?;
        reduce.encode(&temp, d.reborrow(), bias_arg, elem as u32, split_k, group_count, n, scale_arg, command_buffer);

        Ok(())
    }
}

fn validate_int8_left_operand(
    use_mxu: bool,
    shape: MatmulShape,
    a_group_size: u32,
    code_layout: Int8CodeLayout,
    has_group_sums: bool,
) -> Result<(), MetalError> {
    if shape.b_bits.and_then(Int8CodeLayout::for_right_bits) != Some(code_layout) {
        return Err(MatmulError::IncompatibleA {
            path: "Gemm",
            reason: "left code layout is incompatible with the right operand",
        }
        .into());
    }
    let needs_group_sums =
        matches!(shape.b_prologue, GemmBPrologueKind::ScaleBiasDequant | GemmBPrologueKind::ScaleZeroPointDequant);
    if needs_group_sums && !has_group_sums {
        return Err(MatmulError::IncompatibleA {
            path: "Gemm",
            reason: "quantized correction requires left group sums",
        }
        .into());
    }
    let compatible = use_mxu
        && supports_integer_right_operand(&shape)
        && shape.params_layout == Some(QuantParamsLayout::GroupOutput)
        && matches!(
            shape.b_prologue,
            GemmBPrologueKind::ScaleSymmetricDequant
                | GemmBPrologueKind::ScaleBiasDequant
                | GemmBPrologueKind::ScaleZeroPointDequant
        )
        && matches!(a_group_size, 32 | 64 | 128)
        && shape.k.is_multiple_of(a_group_size)
        && shape
            .b_group_size
            .is_some_and(|gs| matches!(gs, 32 | 64 | 128) && shape.k.is_multiple_of(gs) && a_group_size >= gs);
    if !compatible {
        return Err(MatmulError::IncompatibleA {
            path: "Gemm",
            reason: "symmetric int8 left operands require group-major metadata and supported 32/64/128 groups",
        }
        .into());
    }
    Ok(())
}

fn gemm_params(
    shape: MatmulShape,
    plan: GemmPlan,
    ab_scale: f32,
    scale_strides: QuantParamsStrides,
    zero_point_strides: QuantParamsStrides,
) -> GemmParams {
    let (m, n, k) = (shape.m, shape.n, shape.k);
    let tiling = plan.tiling;
    GemmParams {
        M: m,
        N: n,
        K: k,
        leading_dimension_a: k,
        leading_dimension_b: k,
        leading_dimension_d: n,
        threadgroups_per_row: n.div_ceil(tiling.block_n()),
        threadgroups_per_column: m.div_ceil(tiling.block_m()),
        aligned_inner_iterations: outer_block_k(shape, plan.engine, plan.tiling).map_or(0, |step| k / step),
        use_morton: false,
        ab_scale,
        scale_output_stride: scale_strides.output_stride,
        scale_group_stride: scale_strides.group_stride,
        zero_point_output_stride: zero_point_strides.output_stride,
        zero_point_group_stride: zero_point_strides.group_stride,
    }
}
