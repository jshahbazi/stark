use super::*;

impl<B: Backend> LinearMatmul<B> {
    pub(crate) fn coarse_affine(
        &self,
        context: &B::Context,
        bits: u32,
    ) -> Result<Option<Self>, LinearMatmulError<B>> {
        if self.output_data_type != DataType::BF16
            || self.biases.is_some()
            || self.output_hadamard_factors.is_some()
            || self.reference_matmul.is_some()
            || self.reference_affine.is_some()
        {
            return Ok(None);
        }
        let Some(matrix) = self.matrix.coarse_affine(context, self.output_dim, self.input_dim, bits)? else {
            return Ok(None);
        };
        // F32 coarse scores feed the GPU shortlist; exact rescoring keeps the
        // target's normal BF16 readout and complete target distribution.
        let kernel = <B::Kernels as Kernels>::MatmulKernel::new(context, DataType::BF16, DataType::BF16, DataType::F32)
            .map_err(LinearMatmulError::BackendError)?;
        Ok(Some(Self {
            kernel: Mutex::new(kernel),
            matrix,
            biases: None,
            output_hadamard_factors: None,
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            output_data_type: DataType::F32,
            reference_matmul: None,
            reference_affine: None,
        }))
    }
}
