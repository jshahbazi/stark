use crate::backends::common::{Backend, Buffer};

pub trait SparseBuffer: Buffer<Backend: Backend<SparseBuffer = Self>> {
    /// Successfully mapped backing bytes. Unknown implementations receive no
    /// resident-memory credit; virtual address capacity is not physical memory.
    fn mapped_size(&self) -> usize {
        0
    }

    fn map(
        &mut self,
        until: usize,
    ) -> Result<(), <Self::Backend as Backend>::Error>;
}
