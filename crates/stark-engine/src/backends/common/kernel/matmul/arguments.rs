use super::{MatmulError, d_ops::MatmulDOps, matmul_a::MatmulA, matmul_b::MatmulB};
use crate::{
    backends::common::{Backend, BufferMut, BufferRef, gpu_types::QuantizationMode},
    data_type::DataType,
};

pub struct MatmulArguments<
    'd,
    B: Backend,
    TA: BufferRef<Backend = B>,
    TB: BufferRef<Backend = B>,
    TD: BufferMut<Backend = B>,
    TI: BufferRef<Backend = B>,
> {
    pub a: MatmulA<TA>,
    pub b: MatmulB<TB>,
    pub b_leading_dimension: Option<u32>,
    pub b_transpose: bool,
    pub d: TD,
    pub d_transform: MatmulDOps<'d, B>,
    pub gather_indices: Option<TI>,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl<
    'd,
    B: Backend,
    TA: BufferRef<Backend = B>,
    TB: BufferRef<Backend = B>,
    TD: BufferMut<Backend = B>,
    TI: BufferRef<Backend = B>,
> MatmulArguments<'d, B, TA, TB, TD, TI>
{
    pub(crate) fn validate_affine_input_sum_policy(
        &self,
        weights: DataType,
        input: DataType,
        output: DataType,
    ) -> Result<(), MatmulError<B>> {
        let Some(quantized) = self.b.quantized().filter(|q| q.params.materializes_affine_input_sum()) else {
            return Ok(());
        };
        if weights != DataType::BF16
            || input != DataType::BF16
            || output != DataType::BF16
            || quantized.mode != QuantizationMode::U4
            || quantized.biases().is_none()
            || quantized.signed_codes
            || !self.b_transpose
            || !matches!(self.a, MatmulA::FullPrecision { .. })
            || !matches!(quantized.group_size, 16 | 32 | 64 | 128)
            || !self.k.is_multiple_of(4)
        {
            return Err(MatmulError::UnsupportedLayout {
                path: "BF16 W4 affine input-sum policy",
            });
        }
        Ok(())
    }
}
