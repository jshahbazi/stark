//! Native BF16 SigLIP tower and Gemma 3 soft-image projector.
//!
//! Independent equations/layout follow Google's Apache-2.0 Gemma 3 and
//! Transformers v4.53.3 SiglipVisionModel/Gemma3MultiModalProjector. The
//! admitting wrapper selects one original 896px SigLIP profile. Each block is
//! submitted separately to bound scratch memory at the full 4096-patch size.

use half::bf16;

use super::{
    super::tower::{LayerNorm, RawLinear, VisionEagerAttention, VisionTowerError},
    preprocess::{PATCH_ROWS, PATCH_WIDTH},
};
use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, InvalidInput, Kernels,
        kernel::{
            ReferenceRMSBF16Kernel, SiglipPoolBF16Kernel, TensorAddScaleKernel, VisionCastBF16Kernel,
            VisionGeluBF16Kernel, VisionMatmulBF16Kernel, VisionPrepareBF16Kernel,
        },
    },
    data_type::DataType,
    parameters::ParameterTree,
};

const WIDTH: u32 = 1152;
const INNER: u32 = 4304;
const HEADS: u32 = 16;
const DEPTH: u32 = 27;
const TOKENS: u32 = 256;

struct Block<B: Backend> {
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    qkv: RawLinear<B>,
    projection: RawLinear<B>,
    fc1: RawLinear<B>,
    fc2: RawLinear<B>,
}

pub(crate) struct Gemma3VisionTower<B: Backend> {
    model_dim: u32,
    patch: RawLinear<B>,
    positions: B::GlobalBuffer,
    blocks: Box<[Block<B>]>,
    post_norm: LayerNorm<B>,
    projector_norm: B::GlobalBuffer,
    projector: B::GlobalBuffer,
    prepare: <B::Kernels as Kernels>::VisionPrepareBF16Kernel,
    attention: VisionEagerAttention<B>,
    add: <B::Kernels as Kernels>::TensorAddScaleKernel,
    activation: <B::Kernels as Kernels>::VisionGeluBF16Kernel,
    pool: <B::Kernels as Kernels>::SiglipPoolBF16Kernel,
    rms: <B::Kernels as Kernels>::ReferenceRMSBF16Kernel,
    matmul: <B::Kernels as Kernels>::VisionMatmulBF16Kernel,
    cast: <B::Kernels as Kernels>::VisionCastBF16Kernel,
}

fn validate_parameters<B: Backend>(
    tree: &ParameterTree<B>,
    model_dim: u32,
) -> Result<(), VisionTowerError<B>> {
    let bf = |name: &str, shape: &[u32]| tree.leaf(name)?.validate(shape, DataType::BF16).map(|_| ());
    let linear = |prefix: &str, input: u32, output: u32| -> Result<(), VisionTowerError<B>> {
        bf(&format!("{prefix}.weight"), &[output, input])?;
        bf(&format!("{prefix}.bias"), &[output])?;
        Ok(())
    };
    bf("embeddings.patch_embedding.weight", &[WIDTH, 3, 14, 14])?;
    bf("embeddings.patch_embedding.bias", &[WIDTH])?;
    bf("embeddings.position_embedding.weight", &[PATCH_ROWS, WIDTH])?;
    for index in 0..DEPTH {
        let prefix = format!("encoder.layers.{index}");
        for norm in ["layer_norm1", "layer_norm2"] {
            bf(&format!("{prefix}.{norm}.weight"), &[WIDTH])?;
            bf(&format!("{prefix}.{norm}.bias"), &[WIDTH])?;
        }
        linear(&format!("{prefix}.self_attn.qkv"), WIDTH, WIDTH * 3)?;
        linear(&format!("{prefix}.self_attn.out_proj"), WIDTH, WIDTH)?;
        linear(&format!("{prefix}.mlp.fc1"), WIDTH, INNER)?;
        linear(&format!("{prefix}.mlp.fc2"), INNER, WIDTH)?;
    }
    bf("post_layernorm.weight", &[WIDTH])?;
    bf("post_layernorm.bias", &[WIDTH])?;
    tree.leaf("mm_soft_emb_norm.scales")?.validate(&[WIDTH], DataType::F32)?;
    bf("mm_input_projection_weight", &[WIDTH, model_dim])?;
    tree.assert_all_tensors_validated()?;
    Ok(())
}

impl<B: Backend> Gemma3VisionTower<B> {
    pub(crate) fn new(
        context: &B::Context,
        tree: &ParameterTree<B>,
        model_dim: u32,
    ) -> Result<Self, VisionTowerError<B>> {
        if ![2560, 3840, 5376].contains(&model_dim) {
            return Err(InvalidInput("unsupported Gemma 3 image decoder width").into());
        }
        validate_parameters(tree, model_dim)?;
        let linear = |prefix: &str, input, output| {
            RawLinear::new(context, &tree.subtree(prefix), input, output, &[output, input])
        };
        let norm = |prefix: &str| LayerNorm::new(context, &tree.subtree(prefix), WIDTH, 1e-6);
        let mut blocks = Vec::with_capacity(DEPTH as usize);
        for index in 0..DEPTH {
            let prefix = format!("encoder.layers.{index}");
            blocks.push(Block {
                norm1: norm(&format!("{prefix}.layer_norm1"))?,
                norm2: norm(&format!("{prefix}.layer_norm2"))?,
                qkv: linear(&format!("{prefix}.self_attn.qkv"), WIDTH, WIDTH * 3)?,
                projection: linear(&format!("{prefix}.self_attn.out_proj"), WIDTH, WIDTH)?,
                fc1: linear(&format!("{prefix}.mlp.fc1"), WIDTH, INNER)?,
                fc2: linear(&format!("{prefix}.mlp.fc2"), INNER, WIDTH)?,
            });
        }
        Ok(Self {
            model_dim,
            patch: RawLinear::new(
                context,
                &tree.subtree("embeddings.patch_embedding"),
                PATCH_WIDTH,
                WIDTH,
                &[WIDTH, 3, 14, 14],
            )?,
            positions: tree
                .leaf("embeddings.position_embedding.weight")?
                .validate(&[PATCH_ROWS, WIDTH], DataType::BF16)?
                .read_buffer()?,
            blocks: blocks.into_boxed_slice(),
            post_norm: norm("post_layernorm")?,
            projector_norm: tree.leaf("mm_soft_emb_norm.scales")?.validate(&[WIDTH], DataType::F32)?.read_buffer()?,
            projector: tree
                .leaf("mm_input_projection_weight")?
                .validate(&[WIDTH, model_dim], DataType::BF16)?
                .read_buffer()?,
            prepare: <B::Kernels as Kernels>::VisionPrepareBF16Kernel::new(context)
                .map_err(VisionTowerError::Backend)?,
            attention: VisionEagerAttention::new(context, WIDTH, HEADS).map_err(VisionTowerError::Backend)?,
            add: <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, DataType::BF16, false)
                .map_err(VisionTowerError::Backend)?,
            activation: <B::Kernels as Kernels>::VisionGeluBF16Kernel::new(context)
                .map_err(VisionTowerError::Backend)?,
            pool: <B::Kernels as Kernels>::SiglipPoolBF16Kernel::new(context).map_err(VisionTowerError::Backend)?,
            rms: <B::Kernels as Kernels>::ReferenceRMSBF16Kernel::new(context, false, false, false)
                .map_err(VisionTowerError::Backend)?,
            matmul: <B::Kernels as Kernels>::VisionMatmulBF16Kernel::new(context).map_err(VisionTowerError::Backend)?,
            cast: <B::Kernels as Kernels>::VisionCastBF16Kernel::new(context).map_err(VisionTowerError::Backend)?,
        })
    }

    pub(crate) fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub(crate) fn encode_initial(
        &self,
        patches: &[bf16],
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, VisionTowerError<B>> {
        self.encode_initial_with_observer(patches, commands, &mut |_, _, _| Ok(()))
    }

    pub(crate) fn encode_initial_with_observer<F>(
        &self,
        patches: &[bf16],
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        observer: &mut F,
    ) -> Result<B::ScratchBuffer, VisionTowerError<B>>
    where
        F: FnMut(&str, &B::ScratchBuffer, &mut <B::CommandBuffer as CommandBuffer>::Encoding) -> Result<(), B::Error>,
    {
        if patches.len() != (PATCH_ROWS * PATCH_WIDTH) as usize || patches.iter().any(|value| !value.is_finite()) {
            return Err(InvalidInput("Gemma 3 patches have invalid shape or nonfinite values").into());
        }
        let patches = commands.allocate_constant_from_slice(patches).map_err(VisionTowerError::Backend)?;
        let hidden = self.patch.encode(&patches, PATCH_ROWS, commands).map_err(VisionTowerError::Backend)?;
        observer("patch", &hidden, commands).map_err(VisionTowerError::Backend)?;
        let output = self.residual(&hidden, &self.positions, commands).map_err(VisionTowerError::Backend)?;
        observer("embeddings", &output, commands).map_err(VisionTowerError::Backend)?;
        Ok(output)
    }

    pub(crate) fn encode_block(
        &self,
        index: usize,
        hidden: impl BufferRef<Backend = B>,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, VisionTowerError<B>> {
        self.encode_block_with_observer(index, hidden, commands, &mut |_, _, _| Ok(()))
    }

    pub(crate) fn encode_block_with_observer<F>(
        &self,
        index: usize,
        hidden: impl BufferRef<Backend = B>,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        observer: &mut F,
    ) -> Result<B::ScratchBuffer, VisionTowerError<B>>
    where
        F: FnMut(&str, &B::ScratchBuffer, &mut <B::CommandBuffer as CommandBuffer>::Encoding) -> Result<(), B::Error>,
    {
        let block = self.blocks.get(index).ok_or(InvalidInput("Gemma 3 image block index is invalid"))?;
        let mut input = commands
            .allocate_scratch_for_shape(&[PATCH_ROWS, WIDTH], DataType::BF16)
            .map_err(VisionTowerError::Backend)?;
        commands.encode_copy(hidden, &mut input);
        let normalized = block.norm1.encode(&input, PATCH_ROWS, commands).map_err(VisionTowerError::Backend)?;
        observer("norm1", &normalized, commands).map_err(VisionTowerError::Backend)?;
        let qkv = block.qkv.encode(&normalized, PATCH_ROWS, commands).map_err(VisionTowerError::Backend)?;
        observer("qkv", &qkv, commands).map_err(VisionTowerError::Backend)?;
        let mut queries = commands
            .allocate_scratch_for_shape(&[HEADS, PATCH_ROWS, WIDTH / HEADS], DataType::BF16)
            .map_err(VisionTowerError::Backend)?;
        let mut keys = commands
            .allocate_scratch_for_shape(&[PATCH_ROWS, WIDTH], DataType::BF16)
            .map_err(VisionTowerError::Backend)?;
        let mut values = commands
            .allocate_scratch_for_shape(&[PATCH_ROWS, WIDTH], DataType::BF16)
            .map_err(VisionTowerError::Backend)?;
        // Identity rotations reuse the exact Q/K/V layout kernel. SigLIP has
        // learned absolute positions and no rotary transformation.
        let cosines = commands
            .allocate_constant_from_slice(&vec![1.0f32; (PATCH_ROWS * (WIDTH / HEADS)) as usize])
            .map_err(VisionTowerError::Backend)?;
        let sines = commands
            .allocate_constant_from_slice(&vec![0.0f32; (PATCH_ROWS * (WIDTH / HEADS)) as usize])
            .map_err(VisionTowerError::Backend)?;
        self.prepare.encode(
            &qkv,
            &mut queries,
            &mut keys,
            &mut values,
            &cosines,
            &sines,
            PATCH_ROWS,
            HEADS,
            WIDTH / HEADS,
            commands,
        );
        let attention =
            self.attention.encode(&queries, &keys, &values, PATCH_ROWS, commands).map_err(VisionTowerError::Backend)?;
        observer("attention", &attention, commands).map_err(VisionTowerError::Backend)?;
        let attention = block.projection.encode(&attention, PATCH_ROWS, commands).map_err(VisionTowerError::Backend)?;
        observer("projection", &attention, commands).map_err(VisionTowerError::Backend)?;
        let hidden = self.residual(&input, &attention, commands).map_err(VisionTowerError::Backend)?;
        let normalized = block.norm2.encode(&hidden, PATCH_ROWS, commands).map_err(VisionTowerError::Backend)?;
        observer("norm2", &normalized, commands).map_err(VisionTowerError::Backend)?;
        let mut up = block.fc1.encode(&normalized, PATCH_ROWS, commands).map_err(VisionTowerError::Backend)?;
        observer("fc1", &up, commands).map_err(VisionTowerError::Backend)?;
        self.activation.encode(&mut up, PATCH_ROWS * INNER, 1, commands);
        observer("activation", &up, commands).map_err(VisionTowerError::Backend)?;
        let down = block.fc2.encode(&up, PATCH_ROWS, commands).map_err(VisionTowerError::Backend)?;
        observer("fc2", &down, commands).map_err(VisionTowerError::Backend)?;
        let output = self.residual(&hidden, &down, commands).map_err(VisionTowerError::Backend)?;
        observer("block", &output, commands).map_err(VisionTowerError::Backend)?;
        Ok(output)
    }

    pub(crate) fn encode_finish(
        &self,
        hidden: impl BufferRef<Backend = B>,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, VisionTowerError<B>> {
        let mut input = commands
            .allocate_scratch_for_shape(&[PATCH_ROWS, WIDTH], DataType::BF16)
            .map_err(VisionTowerError::Backend)?;
        commands.encode_copy(hidden, &mut input);
        let normalized = self.post_norm.encode(&input, PATCH_ROWS, commands).map_err(VisionTowerError::Backend)?;
        let mut pooled =
            commands.allocate_scratch_for_shape(&[TOKENS, WIDTH], DataType::BF16).map_err(VisionTowerError::Backend)?;
        self.pool.encode(&normalized, &mut pooled, WIDTH, 64, 4, commands);
        let mut normalized =
            commands.allocate_scratch_for_shape(&[TOKENS, WIDTH], DataType::BF16).map_err(VisionTowerError::Backend)?;
        self.rms.encode(
            Some(&pooled),
            &self.projector_norm,
            &mut normalized,
            None::<&mut B::ScratchBuffer>,
            TOKENS,
            WIDTH,
            WIDTH,
            1e-6,
            1.0,
            0,
            1,
            commands,
        );
        let mut projected = commands
            .allocate_scratch_for_shape(&[TOKENS, self.model_dim], DataType::F32)
            .map_err(VisionTowerError::Backend)?;
        self.matmul.encode(
            &normalized,
            &self.projector,
            &mut projected,
            TOKENS,
            self.model_dim,
            WIDTH,
            self.model_dim,
            0,
            4,
            commands,
        );
        let mut output = commands
            .allocate_scratch_for_shape(&[TOKENS, self.model_dim], DataType::BF16)
            .map_err(VisionTowerError::Backend)?;
        self.cast.encode(&projected, &mut output, TOKENS * self.model_dim, commands);
        Ok(output)
    }

    fn residual(
        &self,
        input: &B::ScratchBuffer,
        addition: impl BufferRef<Backend = B>,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut output = commands.allocate_scratch_for_shape(&[PATCH_ROWS, WIDTH], DataType::BF16)?;
        self.add.encode(Some(input), addition, &mut output, PATCH_ROWS * WIDTH, PATCH_ROWS * WIDTH, 1.0, commands);
        Ok(output)
    }
}
