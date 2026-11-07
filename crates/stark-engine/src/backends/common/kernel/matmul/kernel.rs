use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, Kernels,
        kernel::{
            ActivationQuantization,
            matmul::{
                arguments::MatmulArguments,
                routing::{ActivationFormat, MatmulShape},
            },
        },
    },
    data_type::DataType,
};

pub trait MatmulKernel: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<MatmulKernel = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error>;

    /// Preserve the widened affine matrix's row reduction policy after a
    /// storage-only partition. CPU reduction is already independent of N and
    /// packed width; backends with shape-dependent reductions override this.
    fn set_row_arithmetic_reference(
        &mut self,
        _output_dim: u32,
    ) {
    }

    /// Keep one F32 contraction before the declared output cast. Backends
    /// whose optimized reductions round partial outputs override this hook.
    /// This is enabled only by an explicitly admitted numerical profile.
    fn set_reference_accumulation(&mut self) {}

    /// Isolated test-only contraction witness. Production builds omit this
    /// scope; backends without the witness keep their ordinary implementation.
    #[cfg(test)]
    fn with_reference_dot_witness<R>(run: impl FnOnce() -> R) -> R {
        run()
    }

    fn encode(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Self::Backend,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferMut<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
        >,
        command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), <Self::Backend as Backend>::Error>;

    fn select_activation_quantization(
        &self,
        _candidate: &MatmulShape,
        _context: &<Self::Backend as Backend>::Context,
    ) -> Option<ActivationQuantization> {
        None
    }

    fn select_activation_format(
        &self,
        _bf16_shape: &MatmulShape,
        _context: &<Self::Backend as Backend>::Context,
    ) -> ActivationFormat {
        ActivationFormat::Bf16
    }
}
