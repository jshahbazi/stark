//! Source BF16-effective affine dots without a dequantized weight bank.

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, InvalidInput, Kernels,
        kernel::{
            ReferenceAffineLinearBF16Kernel,
            matmul::{MatmulB, QuantParams},
        },
    },
    data_type::DataType,
};

pub(crate) const MAX_REFERENCE_DISPATCHES: u32 = 1024;
const MAX_TERMS_PER_DISPATCH: u64 = 1 << 30;
const MAX_TERMS_PER_ENCODE: u64 = 1 << 38;
const MAX_OUTPUT_ELEMENTS: u64 = 64 * 1024 * 1024;
const MAX_OUTPUTS_PER_DISPATCH: u64 = 65536;
const MAX_BANK_ROWS: u32 = 1 << 28;

/// Explicit source arithmetic markers outrank this profile's default dot.
/// Plain mixed-width row partitions do not carry these numerical markers.
pub(crate) fn has_source_affine_policy<B: Backend>(weights: &MatmulB<impl BufferRef<Backend = B>>) -> bool {
    weights
        .quantized()
        .is_some_and(|q| q.params.materializes_affine_input_sum() || !q.params.source_output_widths().is_empty())
}

#[inline]
// The backend selects a thread-local witness only in unit-test builds.
#[cfg_attr(not(test), allow(clippy::extra_unused_type_parameters))]
pub(crate) fn is_affine_dot_witness_active<B: Backend>() -> bool {
    #[cfg(test)]
    {
        use crate::backends::common::kernel::matmul::MatmulKernel;
        <B::Kernels as Kernels>::MatmulKernel::reference_dot_witness_active()
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// Tile output coordinates only. K is never partitioned or rounded early.
pub(crate) struct ReferenceAffinePlan {
    m: u32,
    n: u32,
    k: u32,
    tile_rows: u32,
    tile_columns: u32,
    pub(crate) dispatches: u32,
}

impl ReferenceAffinePlan {
    pub(crate) fn new(
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Self, InvalidInput> {
        Self::with_limits(m, n, k, MAX_TERMS_PER_DISPATCH, MAX_TERMS_PER_ENCODE, MAX_REFERENCE_DISPATCHES)
    }

    fn with_limits(
        m: u32,
        n: u32,
        k: u32,
        per_dispatch: u64,
        total: u64,
        max_dispatches: u32,
    ) -> Result<Self, InvalidInput> {
        let invalid = || InvalidInput("Qwen affine reference exceeds bounded work geometry");
        let outputs = u64::from(m).checked_mul(u64::from(n)).ok_or_else(invalid)?;
        let terms = outputs.checked_mul(u64::from(k)).ok_or_else(invalid)?;
        if m == 0
            // A native 1024-token prefill can expand to 32 selected experts.
            || m > 32768
            || n == 0
            || n > 1048576
            || k == 0
            || k > 131072
            || !k.is_multiple_of(32)
            || outputs > MAX_OUTPUT_ELEMENTS
            || terms > total
            || u64::from(k) > per_dispatch
        {
            return Err(invalid());
        }
        let output_tile = (per_dispatch / u64::from(k)).min(MAX_OUTPUTS_PER_DISPATCH) as u32;
        let tile_columns = n.min(output_tile);
        let tile_rows = m.min(output_tile / tile_columns);
        let dispatches = m.div_ceil(tile_rows).checked_mul(n.div_ceil(tile_columns)).ok_or_else(invalid)?;
        if dispatches == 0 || dispatches > max_dispatches {
            return Err(invalid());
        }
        Ok(Self {
            m,
            n,
            k,
            tile_rows,
            tile_columns,
            dispatches,
        })
    }
}

pub(crate) struct ReferenceAffineBF16Matmul<B: Backend> {
    plain: <B::Kernels as Kernels>::ReferenceAffineLinearBF16Kernel,
    gathered: <B::Kernels as Kernels>::ReferenceAffineLinearBF16Kernel,
    output_type: DataType,
}

impl<B: Backend> ReferenceAffineBF16Matmul<B> {
    pub(crate) fn new(
        context: &B::Context,
        output_type: DataType,
    ) -> Result<Self, B::Error> {
        if !matches!(output_type, DataType::BF16 | DataType::F32) {
            return Err(InvalidInput("Qwen affine reference requires BF16 or widened BF16 output").into());
        }
        Ok(Self {
            plain: <B::Kernels as Kernels>::ReferenceAffineLinearBF16Kernel::new(
                context,
                false,
                output_type == DataType::F32,
            )?,
            gathered: <B::Kernels as Kernels>::ReferenceAffineLinearBF16Kernel::new(
                context,
                true,
                output_type == DataType::F32,
            )?,
            output_type,
        })
    }

    pub(crate) fn encode(
        &self,
        input: impl BufferRef<Backend = B>,
        input_offset: usize,
        weights: MatmulB<impl BufferRef<Backend = B>>,
        indices: Option<impl BufferRef<Backend = B>>,
        m: u32,
        n: u32,
        k: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let invalid = || InvalidInput("invalid Qwen affine reference buffers or metadata");
        let plan = ReferenceAffinePlan::new(m, n, k)?;
        if has_source_affine_policy::<B>(&weights) {
            return Err(InvalidInput("explicit source affine policies require their existing backend route").into());
        }
        let (weights, scales, biases, bits, strides) = match weights {
            MatmulB::FullPrecision {
                b,
            } => (b, b, b, 0, Default::default()),
            MatmulB::Quantized(q) => {
                if q.group_size != 64
                    || q.signed_codes
                    || !matches!(q.mode.bits(), 4 | 5 | 6 | 8)
                    || !k.is_multiple_of(64)
                {
                    return Err(invalid().into());
                }
                let row_bytes = k as usize * q.mode.bits() as usize / 8;
                if q.codes.size() == 0 || !q.codes.size().is_multiple_of(row_bytes) {
                    return Err(invalid().into());
                }
                let rows = u32::try_from(q.codes.size() / row_bytes).map_err(|_| invalid())?;
                if rows > MAX_BANK_ROWS {
                    return Err(invalid().into());
                }
                let expected = QuantParams::new(q.params.layout(), rows, k / 64);
                if q.params.scale_shape() != expected.scale_shape() {
                    return Err(invalid().into());
                }
                (q.codes, q.scales, q.biases().ok_or_else(invalid)?, q.mode.bits(), q.params.scale_strides())
            },
        };
        let row_bytes = if bits == 0 {
            k as usize * 2
        } else {
            k as usize * bits as usize / 8
        };
        if weights.size() == 0 || !weights.size().is_multiple_of(row_bytes) {
            return Err(invalid().into());
        }
        let bank_rows = u32::try_from(weights.size() / row_bytes).map_err(|_| invalid())?;
        if bank_rows > MAX_BANK_ROWS {
            return Err(invalid().into());
        }
        let input_bytes = (m as usize).checked_mul(k as usize).and_then(|x| x.checked_mul(2)).ok_or_else(invalid)?;
        let input_end = input_offset.checked_add(input_bytes).ok_or_else(invalid)?;
        let outputs = (m as usize).checked_mul(n as usize).ok_or_else(invalid)?;
        let index_bytes = outputs.checked_mul(4).ok_or_else(invalid)?;
        let absolute_input_start = input.parts().1.start.checked_add(input_offset).ok_or_else(invalid)?;
        if !absolute_input_start.is_multiple_of(2)
            || (bits == 0 && !weights.parts().1.start.is_multiple_of(2))
            || (bits != 0 && (!scales.parts().1.start.is_multiple_of(2) || !biases.parts().1.start.is_multiple_of(2)))
            || input_end > input.size()
            || indices
                .as_ref()
                .is_some_and(|index| index.size() < index_bytes || !index.parts().1.start.is_multiple_of(4))
            || (indices.is_none() && n > bank_rows)
        {
            return Err(invalid().into());
        }
        if bits != 0 {
            let metadata_end = (bank_rows as usize - 1)
                .checked_mul(strides.output_stride as usize)
                .and_then(|x| {
                    (k as usize / 64 - 1)
                        .checked_mul(strides.group_stride as usize)
                        .and_then(|group| x.checked_add(group))
                })
                .and_then(|x| x.checked_add(1))
                .and_then(|x| x.checked_mul(2))
                .ok_or_else(invalid)?;
            if metadata_end > scales.size() || metadata_end > biases.size() {
                return Err(invalid().into());
            }
        }
        // All shape, count and extent checks precede the sole allocation.
        let mut output = commands.allocate_scratch_for_shape(&[m, n], self.output_type)?;
        if !(&output).parts().1.start.is_multiple_of(self.output_type.size_in_bytes()) {
            return Err(invalid().into());
        }
        let dot = if indices.is_some() {
            &self.gathered
        } else {
            &self.plain
        };
        for row_start in (0..plan.m).step_by(plan.tile_rows as usize) {
            for column_start in (0..plan.n).step_by(plan.tile_columns as usize) {
                let tile_rows = plan.tile_rows.min(plan.m - row_start);
                let tile_columns = plan.tile_columns.min(plan.n - column_start);
                if self.output_type == DataType::BF16 {
                    dot.encode(
                        input.subrange(input_offset..input_end),
                        weights,
                        scales,
                        biases,
                        indices,
                        Some(&mut output),
                        None::<&mut B::ScratchBuffer>,
                        m,
                        n,
                        plan.k,
                        bank_rows,
                        bits,
                        strides.output_stride,
                        strides.group_stride,
                        row_start,
                        column_start,
                        tile_rows,
                        tile_columns,
                        commands,
                    );
                } else {
                    dot.encode(
                        input.subrange(input_offset..input_end),
                        weights,
                        scales,
                        biases,
                        indices,
                        None::<&mut B::ScratchBuffer>,
                        Some(&mut output),
                        m,
                        n,
                        plan.k,
                        bank_rows,
                        bits,
                        strides.output_stride,
                        strides.group_stride,
                        row_start,
                        column_start,
                        tile_rows,
                        tile_columns,
                        commands,
                    );
                }
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
