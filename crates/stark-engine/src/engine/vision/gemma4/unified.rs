//! Encoder-free Gemma 4 12B image embedding. Independently implements Google's
//! pinned checkpoint and Transformers 5.15.1 unified equations (Apache-2.0).

use half::bf16;

use super::{
    super::tower::{LayerNorm, RawLinear, VisionTowerError},
    tower::Linear,
};
use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferEncoding, InvalidInput, Kernels,
        kernel::{Gemma4UnifiedPositionBF16Kernel, Gemma4VisionNormBF16Kernel},
    },
    data_type::DataType,
    parameters::ParameterTree,
};

const WIDTH: u32 = 3840;
const PATCH: u32 = 48 * 48 * 3;
const POSITIONS: u32 = 1120;

pub(super) struct UnifiedVisionEmbedder<B: Backend> {
    patch_norm: LayerNorm<B>,
    patch_projection: RawLinear<B>,
    projected_norm: LayerNorm<B>,
    position_norm: LayerNorm<B>,
    positions: B::GlobalBuffer,
    add_positions: <B::Kernels as Kernels>::Gemma4UnifiedPositionBF16Kernel,
    rms: <B::Kernels as Kernels>::Gemma4VisionNormBF16Kernel,
    projection: Linear<B>,
}

impl<B: Backend> UnifiedVisionEmbedder<B> {
    pub(super) fn new(
        context: &B::Context,
        tree: &ParameterTree<B>,
    ) -> Result<Self, VisionTowerError<B>> {
        Ok(Self {
            patch_norm: LayerNorm::new(context, &tree.subtree("patch_ln1"), PATCH, 1e-5)?,
            patch_projection: RawLinear::new(context, &tree.subtree("patch_dense"), PATCH, WIDTH, &[WIDTH, PATCH])?,
            projected_norm: LayerNorm::new(context, &tree.subtree("patch_ln2"), WIDTH, 1e-5)?,
            position_norm: LayerNorm::new(context, &tree.subtree("pos_norm"), WIDTH, 1e-5)?,
            positions: tree.leaf("pos_embedding")?.validate(&[POSITIONS, 2, WIDTH], DataType::BF16)?.read_buffer()?,
            add_positions: <B::Kernels as Kernels>::Gemma4UnifiedPositionBF16Kernel::new(context)
                .map_err(VisionTowerError::Backend)?,
            rms: <B::Kernels as Kernels>::Gemma4VisionNormBF16Kernel::new(context)
                .map_err(VisionTowerError::Backend)?,
            projection: Linear::new(context, &tree.subtree("embed_vision.embedding_projection"), WIDTH, WIDTH, false)?,
        })
    }

    pub(super) fn encode(
        &self,
        patches: &[bf16],
        grid: [u32; 3],
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, VisionTowerError<B>> {
        let [temporal, height, width] = grid;
        let rows = height
            .checked_mul(width)
            .filter(|n| *n > 0 && *n <= 2520)
            .ok_or(InvalidInput("unified image patch budget exceeded"))?
            / 9;
        if temporal != 1
            || !height.is_multiple_of(3)
            || !width.is_multiple_of(3)
            || height / 3 >= POSITIONS
            || width / 3 >= POSITIONS
            || rows == 0
            || patches.len() != (rows * PATCH) as usize
            || patches.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(&v.to_f32()))
        {
            return Err(InvalidInput("invalid unified image patches").into());
        }
        let input = commands.allocate_constant_from_slice(patches).map_err(VisionTowerError::Backend)?;
        let mut hidden =
            commands.allocate_scratch_for_shape(&[rows, PATCH], DataType::BF16).map_err(VisionTowerError::Backend)?;
        commands.encode_copy(&input, &mut hidden);
        hidden = self.patch_norm.encode(&hidden, rows, commands).map_err(VisionTowerError::Backend)?;
        hidden = self.patch_projection.encode(&hidden, rows, commands).map_err(VisionTowerError::Backend)?;
        hidden = self.projected_norm.encode(&hidden, rows, commands).map_err(VisionTowerError::Backend)?;
        let mut positioned =
            commands.allocate_scratch_for_shape(&[rows, WIDTH], DataType::BF16).map_err(VisionTowerError::Backend)?;
        self.add_positions.encode(&hidden, &self.positions, &mut positioned, rows, WIDTH, width / 3, commands);
        hidden = self.position_norm.encode(&positioned, rows, commands).map_err(VisionTowerError::Backend)?;
        let mut normalized =
            commands.allocate_scratch_for_shape(&[rows, WIDTH], DataType::BF16).map_err(VisionTowerError::Backend)?;
        self.rms.encode(&hidden, &self.positions, &mut normalized, rows, WIDTH, 1e-6, 0, commands);
        self.projection.encode(&normalized, rows, commands).map_err(VisionTowerError::Backend)
    }
}
