use super::{GemmEngine, GemmPlan};
use crate::{
    backends::common::{
        gpu_types::gemm::{GemmBPrologueKind, GemmDTransform, GemmTiling},
        kernel::matmul::MatmulShape,
    },
    data_type::DataType,
};

// Every admitted matrix uses this one power-of-two scratch bucket. Each encode
// drops its local handle back into the shared request pool, bounding additional
// workspace capacity across all prefill layers/chunks/command buffers to one
// page. Cross-buffer reuse relies on Metal's Device queue barriers; constants
// have separate command-buffer allocators and never alias this GPU-only page.
pub(super) const MAX_MATERIALIZED_AFFINE_BYTES: usize = 128 * 1024 * 1024;

pub(super) fn materialized_affine_bytes(
    shape: MatmulShape,
    plan: GemmPlan,
    weights_type: DataType,
    input_type: DataType,
    output_type: DataType,
) -> Option<usize> {
    if (weights_type, input_type, output_type) != (DataType::BF16, DataType::BF16, DataType::BF16)
        || plan.engine != GemmEngine::Mxu
        || plan.tiling != GemmTiling::Tile128x128x256_Simdgroups4x4
        || !shape.a_full_precision
        || shape.m < 512
        // DenseSchedule uses uint arithmetic for the activation row offset.
        // Preserve the staged schedule's wider addressing by retaining it for
        // matrices whose activation element span exceeds that representation.
        || u64::from(shape.m) * u64::from(shape.k) > u64::from(u32::MAX)
        || shape.gathered
        || !shape.b_transpose
        || shape.b_leading_dimension.is_some()
        || shape.b_prologue != GemmBPrologueKind::ScaleBiasDequant
        || shape.b_bits != Some(4)
        || shape.b_group_size != Some(64)
        || shape.signed_codes
        || shape.d_transform.contains(GemmDTransform::RHT)
        || plan.split_k == 0
        || !shape.k.is_multiple_of(plan.split_k)
        || !(shape.k / plan.split_k).is_multiple_of(plan.tiling.block_k())
    {
        return None;
    }
    let bytes = (shape.n as usize).checked_mul(shape.k as usize)?.checked_mul(DataType::BF16.size_in_bytes())?;
    (bytes > 0 && bytes <= MAX_MATERIALIZED_AFFINE_BYTES).then_some(bytes)
}

#[cfg(test)]
#[path = "../../../../../../unit/backends/metal/kernel/matmul/gemm/materialize_test.rs"]
mod tests;
