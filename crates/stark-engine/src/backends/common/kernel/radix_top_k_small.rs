use crate::{
    backends::common::{Backend, BufferMut, BufferRef, CommandBuffer, InvalidInput, Kernels},
    data_type::DataType,
};

pub const MAX_K: u32 = 512;

pub trait RadixTopKSmall: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<RadixTopKSmall = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        columns: u32,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        Self::new_with_data_type(context, columns, DataType::BF16)
    }

    fn new_with_data_type(
        context: &<Self::Backend as Backend>::Context,
        columns: u32,
        input_data_type: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error>;

    fn encode(
        &self,
        input: impl BufferRef<Backend = Self::Backend>,
        output_ids: impl BufferMut<Backend = Self::Backend>,
        output_scores: impl BufferMut<Backend = Self::Backend>,
        rows: u32,
        k: u32,
        command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), <Self::Backend as Backend>::Error>;
}

pub(crate) fn validate_input_type(
    columns: u32,
    input_data_type: DataType,
) -> Result<(), InvalidInput> {
    if columns == 0 || !matches!(input_data_type, DataType::BF16 | DataType::F32) {
        return Err(InvalidInput("small radix selection requires nonempty BF16 or F32 columns"));
    }
    Ok(())
}

/// Larger views are allowed because existing lookahead callers select only
/// their leading rows. Every accessed byte and typed alignment is checked.
pub(crate) fn validate_buffers(
    columns: u32,
    rows: u32,
    k: u32,
    input_data_type: DataType,
    input_size: usize,
    input_offset: usize,
    ids_size: usize,
    ids_offset: usize,
    scores_size: usize,
    scores_offset: usize,
) -> Result<(), InvalidInput> {
    validate_input_type(columns, input_data_type)?;
    let item_bytes = input_data_type.size_in_bytes();
    let input_alignment = if columns.is_multiple_of(4) {
        4 * item_bytes
    } else {
        item_bytes
    };
    let input_bytes = (rows as usize).checked_mul(columns as usize).and_then(|n| n.checked_mul(item_bytes));
    let output_bytes = (rows as usize).checked_mul(k as usize).and_then(|n| n.checked_mul(4));
    if rows == 0
        || k == 0
        || k > MAX_K
        || k > columns
        || input_bytes.is_none_or(|n| n > input_size)
        || output_bytes.is_none_or(|n| n > ids_size || n > scores_size)
        || !input_size.is_multiple_of(item_bytes)
        || !input_offset.is_multiple_of(input_alignment)
        || !ids_size.is_multiple_of(4)
        || !ids_offset.is_multiple_of(4)
        || !scores_size.is_multiple_of(4)
        || !scores_offset.is_multiple_of(4)
    {
        return Err(InvalidInput("small radix selection buffer geometry or alignment differs"));
    }
    Ok(())
}
