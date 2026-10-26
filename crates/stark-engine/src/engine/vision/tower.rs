//! Native BF16 Qwen3.5 static-image tower.
//!
//! Independently implements the tensor layout and equations in Transformers
//! 2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397, modeling_qwen3_5.py and
//! vision_utils.py (Qwen/Hugging Face, Apache-2.0). No reference runtime is used.
//! The input rows are already normalized, temporally repeated and ordered by
//! spatial merge block; the Conv3d kernel/stride equal its full patch extent.

use half::bf16;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, InvalidInput, Kernels,
        kernel::{
            TensorAddScaleKernel, VisionBiasCastBF16Kernel, VisionCastBF16Kernel, VisionGeluBF16Kernel,
            VisionHeadCopyBF16Kernel, VisionLayerNormBF16Kernel, VisionMatmulBF16Kernel, VisionPrepareBF16Kernel,
            VisionSoftmaxBF16Kernel,
        },
        vision_math::vision_sin_cos,
    },
    data_type::DataType,
    parameters::{ParameterLoaderError, ParameterTree},
};

/// Internal architectural description. The artifact loader additionally requires
/// the exact official 0.8B configuration and checkpoint identity.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VisionTowerConfig {
    pub(crate) depth: u32,
    pub(crate) hidden_size: u32,
    pub(crate) intermediate_size: u32,
    pub(crate) num_heads: u32,
    pub(crate) num_position_embeddings: u32,
    pub(crate) out_hidden_size: u32,
    pub(crate) patch_size: u32,
    pub(crate) spatial_merge_size: u32,
    pub(crate) temporal_patch_size: u32,
    pub(crate) in_channels: u32,
    pub(crate) layer_norm_eps: f32,
}

impl VisionTowerConfig {
    pub(crate) fn validate(&self) -> Result<(), InvalidInput> {
        if self.depth == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_heads == 0
            || self.out_hidden_size == 0
            || self.patch_size == 0
            || self.spatial_merge_size == 0
            || self.temporal_patch_size == 0
            || self.in_channels != 3
            || !self.hidden_size.is_multiple_of(self.num_heads)
            || !(self.hidden_size / self.num_heads).is_multiple_of(4)
            || self.layer_norm_eps != 1e-6
        {
            return Err(InvalidInput("unsupported native vision architecture"));
        }
        let side = self.position_side();
        if side == 0 || side.checked_mul(side) != Some(self.num_position_embeddings) {
            return Err(InvalidInput("vision learned positions must form a square grid"));
        }
        self.patch_width()?;
        self.merged_width()?;
        for width in [self.hidden_size, self.intermediate_size, self.out_hidden_size] {
            checked_elements(4096, width)?;
        }
        self.hidden_size.checked_mul(3).ok_or(InvalidInput("vision QKV width overflow"))?;
        Ok(())
    }

    fn position_side(&self) -> u32 {
        self.num_position_embeddings.isqrt()
    }

    fn patch_width(&self) -> Result<u32, InvalidInput> {
        [self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size]
            .into_iter()
            .try_fold(1u32, |count, width| count.checked_mul(width))
            .ok_or(InvalidInput("vision patch width overflow"))
    }

    fn merged_width(&self) -> Result<u32, InvalidInput> {
        self.hidden_size
            .checked_mul(self.spatial_merge_size)
            .and_then(|width| width.checked_mul(self.spatial_merge_size))
            .ok_or(InvalidInput("vision merger width overflow"))
    }
}

#[derive(Debug, Error)]
pub(crate) enum VisionTowerError<B: Backend> {
    #[error(transparent)]
    InvalidInput(#[from] InvalidInput),
    #[error("Vision parameter loading failed")]
    Parameter(#[from] ParameterLoaderError<B>),
    #[error("Vision backend operation failed")]
    Backend(#[source] B::Error),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VisionStage {
    PatchProjection,
    PositionAdded,
    Block(usize),
    Merger,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VisionBlockStage {
    Norm1,
    Qkv,
    Queries,
    Keys,
    Values,
    Attention,
    Projection,
    AttentionResidual,
    Norm2,
    Fc1,
    Gelu,
    Fc2,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VisionMergerStage {
    Norm,
    Fc1,
    Gelu,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VisionAttentionStage {
    FullPrecisionScores,
    Scores,
    ScaledScores,
    Probabilities,
    FullPrecisionHead,
    Head,
}

fn checked_elements(
    rows: u32,
    width: u32,
) -> Result<u32, InvalidInput> {
    rows.checked_mul(width).ok_or(InvalidInput("vision tensor element count overflow"))
}

fn checked_rows(
    grid: [u32; 3],
    merge: u32,
) -> Result<u32, InvalidInput> {
    let [temporal, height, width] = grid;
    if temporal != 1
        || height == 0
        || width == 0
        || merge == 0
        || !height.is_multiple_of(merge)
        || !width.is_multiple_of(merge)
    {
        return Err(InvalidInput("native vision requires one static divisible image grid"));
    }
    height
        .checked_mul(width)
        .filter(|&rows| rows <= 4096)
        .ok_or(InvalidInput("native vision grid exceeds patch budget"))
}

/// Enumerate the same block-major patch rows as get_vision_position_ids.
fn spatial_positions(
    grid: [u32; 3],
    merge: u32,
) -> Result<Vec<[u32; 2]>, InvalidInput> {
    let rows = checked_rows(grid, merge)?;
    let mut result = Vec::with_capacity(rows as usize);
    for block_row in 0..grid[1] / merge {
        for block_column in 0..grid[2] / merge {
            for inner_row in 0..merge {
                for inner_column in 0..merge {
                    result.push([block_row * merge + inner_row, block_column * merge + inner_column]);
                }
            }
        }
    }
    Ok(result)
}

fn rotary_tables(
    positions: &[[u32; 2]],
    head_dim: u32,
) -> Result<(Vec<f32>, Vec<f32>), InvalidInput> {
    if head_dim == 0 || !head_dim.is_multiple_of(4) {
        return Err(InvalidInput("vision rotary head width must be divisible by four"));
    }
    let count = positions.len().checked_mul(head_dim as usize).ok_or(InvalidInput("vision rotary table overflow"))?;
    let mut angles = Vec::with_capacity(count / 2);
    let quarter = head_dim / 4;
    let half = head_dim / 2;
    for position in positions {
        for axis in 0..2 {
            for pair in 0..quarter {
                let inverse_frequency = 1.0 / 10000f32.powf((2 * pair) as f32 / half as f32);
                angles.push(position[axis] as f32 * inverse_frequency);
            }
        }
    }
    let (cos_half, sin_half) = vision_sin_cos(&angles);
    let mut cosines = Vec::with_capacity(count);
    let mut sines = Vec::with_capacity(count);
    for row in 0..positions.len() {
        let range = row * half as usize..(row + 1) * half as usize;
        cosines.extend_from_slice(&cos_half[range.clone()]);
        cosines.extend_from_slice(&cos_half[range.clone()]);
        sines.extend_from_slice(&sin_half[range.clone()]);
        sines.extend_from_slice(&sin_half[range]);
    }
    Ok((cosines, sines))
}

fn interpolated_positions(
    config: &VisionTowerConfig,
    table: &[bf16],
    positions: &[[u32; 2]],
    grid: [u32; 3],
) -> Result<Vec<bf16>, InvalidInput> {
    let width = config.hidden_size as usize;
    let side = config.position_side();
    let expected = checked_elements(config.num_position_embeddings, config.hidden_size)? as usize;
    if table.len() != expected || table.iter().any(|value| !value.is_finite()) {
        return Err(InvalidInput("invalid learned vision position table"));
    }
    let count = positions.len().checked_mul(width).ok_or(InvalidInput("vision position output overflow"))?;
    let mut output = Vec::with_capacity(count);
    // The pinned model explicitly selects bilinear, align_corners=True.
    for &[row, column] in positions {
        let y = row as f32 * (side - 1) as f32 / grid[1].saturating_sub(1).max(1) as f32;
        let x = column as f32 * (side - 1) as f32 / grid[2].saturating_sub(1).max(1) as f32;
        let y0 = y.floor() as u32;
        let x0 = x.floor() as u32;
        let y1 = (y0 + 1).min(side - 1);
        let x1 = (x0 + 1).min(side - 1);
        let wy = y - y.floor();
        let wx = x - x.floor();
        let taps = [(y0, x0), (y0, x1), (y1, x0), (y1, x1)];
        let weights = [(1.0 - wy) * (1.0 - wx), (1.0 - wy) * wx, wy * (1.0 - wx), wy * wx];
        for channel in 0..width {
            let mut sum = 0.0f32;
            for ((tap_y, tap_x), weight) in taps.into_iter().zip(weights) {
                sum += table[((tap_y * side + tap_x) as usize) * width + channel].to_f32() * weight;
            }
            output.push(bf16::from_f32(sum));
        }
    }
    Ok(output)
}

/// Validate the complete raw-source vision subtree before allocating its tensors.
fn validate_parameters<B: Backend>(
    config: &VisionTowerConfig,
    tree: &ParameterTree<B>,
) -> Result<(), VisionTowerError<B>> {
    let width = config.hidden_size;
    tree.leaf("patch_embed.proj.weight")?.validate(
        &[width, config.in_channels, config.temporal_patch_size, config.patch_size, config.patch_size],
        DataType::BF16,
    )?;
    tree.leaf("patch_embed.proj.bias")?.validate(&[width], DataType::BF16)?;
    tree.leaf("pos_embed.weight")?.validate(&[config.num_position_embeddings, width], DataType::BF16)?;
    let linear = |name: &str, input: u32, output: u32| -> Result<(), VisionTowerError<B>> {
        tree.leaf(&format!("{name}.weight"))?.validate(&[output, input], DataType::BF16)?;
        tree.leaf(&format!("{name}.bias"))?.validate(&[output], DataType::BF16)?;
        Ok(())
    };
    for index in 0..config.depth {
        for norm in ["norm1", "norm2"] {
            tree.leaf(&format!("blocks.{index}.{norm}.weight"))?.validate(&[width], DataType::BF16)?;
            tree.leaf(&format!("blocks.{index}.{norm}.bias"))?.validate(&[width], DataType::BF16)?;
        }
        linear(&format!("blocks.{index}.attn.qkv"), width, 3 * width)?;
        linear(&format!("blocks.{index}.attn.proj"), width, width)?;
        linear(&format!("blocks.{index}.mlp.linear_fc1"), width, config.intermediate_size)?;
        linear(&format!("blocks.{index}.mlp.linear_fc2"), config.intermediate_size, width)?;
    }
    tree.leaf("merger.norm.weight")?.validate(&[width], DataType::BF16)?;
    tree.leaf("merger.norm.bias")?.validate(&[width], DataType::BF16)?;
    let merged = config.merged_width()?;
    linear("merger.linear_fc1", merged, merged)?;
    linear("merger.linear_fc2", merged, config.out_hidden_size)?;
    tree.assert_all_tensors_validated()?;
    Ok(())
}

struct RawLinear<B: Backend> {
    kernel: <B::Kernels as Kernels>::VisionMatmulBF16Kernel,
    cast_bias: <B::Kernels as Kernels>::VisionBiasCastBF16Kernel,
    weights: B::GlobalBuffer,
    biases: B::GlobalBuffer,
    input_width: u32,
    output_width: u32,
    streams: u32,
}

impl<B: Backend> RawLinear<B> {
    fn new(
        context: &B::Context,
        tree: &ParameterTree<B>,
        input_width: u32,
        output_width: u32,
        weight_shape: &[u32],
    ) -> Result<Self, VisionTowerError<B>> {
        Ok(Self {
            kernel: <B::Kernels as Kernels>::VisionMatmulBF16Kernel::new(context).map_err(VisionTowerError::Backend)?,
            cast_bias: <B::Kernels as Kernels>::VisionBiasCastBF16Kernel::new(context)
                .map_err(VisionTowerError::Backend)?,
            weights: tree.leaf("weight")?.validate(weight_shape, DataType::BF16)?.read_buffer()?,
            biases: tree.leaf("bias")?.validate(&[output_width], DataType::BF16)?.read_buffer()?,
            input_width,
            output_width,
            streams: if weight_shape.len() == 5 {
                4
            } else {
                32
            },
        })
    }

    fn encode(
        &self,
        input: impl BufferRef<Backend = B>,
        rows: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        // Preserve the pinned BF16 operation's FP32 reduction, then add its
        // source bias before the sole BF16 cast. Conv3d uses four partial sums;
        // dense transposed GEMM uses an eight-register (32-stream) dot schedule.
        let mut full_precision = commands.allocate_scratch_for_shape(&[rows, self.output_width], DataType::F32)?;
        self.kernel.encode(
            input,
            &self.weights,
            &mut full_precision,
            rows,
            self.output_width,
            self.input_width,
            self.input_width,
            1,
            self.streams,
            commands,
        );
        let mut output = commands.allocate_scratch_for_shape(&[rows, self.output_width], DataType::BF16)?;
        self.cast_bias.encode(
            &full_precision,
            &self.biases,
            &mut output,
            self.output_width,
            rows * self.output_width,
            commands,
        );
        Ok(output)
    }
}

struct LayerNorm<B: Backend> {
    kernel: <B::Kernels as Kernels>::VisionLayerNormBF16Kernel,
    weight: B::GlobalBuffer,
    bias: B::GlobalBuffer,
    width: u32,
    epsilon: f32,
}

impl<B: Backend> LayerNorm<B> {
    fn new(
        context: &B::Context,
        tree: &ParameterTree<B>,
        width: u32,
        epsilon: f32,
    ) -> Result<Self, VisionTowerError<B>> {
        Ok(Self {
            kernel: <B::Kernels as Kernels>::VisionLayerNormBF16Kernel::new(context)
                .map_err(VisionTowerError::Backend)?,
            weight: tree.leaf("weight")?.validate(&[width], DataType::BF16)?.read_buffer()?,
            bias: tree.leaf("bias")?.validate(&[width], DataType::BF16)?.read_buffer()?,
            width,
            epsilon,
        })
    }

    fn encode(
        &self,
        input: &B::ScratchBuffer,
        rows: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut output = commands.allocate_scratch_for_shape(&[rows, self.width], DataType::BF16)?;
        self.kernel.encode(input, &self.weight, &self.bias, &mut output, rows, self.width, self.epsilon, commands);
        Ok(output)
    }
}

/// Materialize the BF16 boundaries of the pinned eager reference. The generic
/// fused language attention path deliberately retains more intermediate FP32
/// precision and is not numerically interchangeable over this vision stack.
struct VisionEagerAttention<B: Backend> {
    matmul: <B::Kernels as Kernels>::VisionMatmulBF16Kernel,
    cast: <B::Kernels as Kernels>::VisionCastBF16Kernel,
    softmax: <B::Kernels as Kernels>::VisionSoftmaxBF16Kernel,
    head_copy: <B::Kernels as Kernels>::VisionHeadCopyBF16Kernel,
    scale_scores: <B::Kernels as Kernels>::TensorAddScaleKernel,
    model_dim: u32,
    head_dim: u32,
    num_heads: u32,
}

impl<B: Backend> VisionEagerAttention<B> {
    fn new(
        context: &B::Context,
        model_dim: u32,
        num_heads: u32,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            matmul: <B::Kernels as Kernels>::VisionMatmulBF16Kernel::new(context)?,
            cast: <B::Kernels as Kernels>::VisionCastBF16Kernel::new(context)?,
            softmax: <B::Kernels as Kernels>::VisionSoftmaxBF16Kernel::new(context)?,
            head_copy: <B::Kernels as Kernels>::VisionHeadCopyBF16Kernel::new(context)?,
            scale_scores: <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, DataType::BF16, true)?,
            model_dim,
            head_dim: model_dim / num_heads,
            num_heads,
        })
    }

    fn encode(
        &self,
        queries: &B::ScratchBuffer,
        keys: &B::ScratchBuffer,
        values: &B::ScratchBuffer,
        rows: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        self.encode_with_observer(queries, keys, values, rows, commands, &mut |_, _, _, _| Ok(()))
    }

    fn encode_with_observer<F>(
        &self,
        queries: &B::ScratchBuffer,
        keys: &B::ScratchBuffer,
        values: &B::ScratchBuffer,
        rows: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        observer: &mut F,
    ) -> Result<B::ScratchBuffer, B::Error>
    where
        F: FnMut(
            u32,
            VisionAttentionStage,
            &B::ScratchBuffer,
            &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        ) -> Result<(), B::Error>,
    {
        let mut output = commands.allocate_scratch_for_shape(&[rows, self.model_dim], DataType::BF16)?;
        let zero = commands.allocate_constant_from_slice(&[bf16::ZERO])?;
        let head_elements = rows * self.head_dim;
        let head_bytes = head_elements as usize * size_of::<bf16>();
        for head in 0..self.num_heads {
            let query_offset = head as usize * head_bytes;
            let kv_offset = head as usize * self.head_dim as usize * size_of::<bf16>();
            let mut full_precision_scores = commands.allocate_scratch_for_shape(&[rows, rows], DataType::F32)?;
            // Q is [head,row,channel]. K/V are [row,head,channel], so B's
            // physical row stride remains the complete model width.
            // The pinned BF16 BMM uses 32 accumulation streams for QK and
            // four streams for AV. Their reduction orders affect BF16 ties.
            self.matmul.encode(
                queries.subrange(query_offset..query_offset + head_bytes),
                keys.subrange(kv_offset..),
                &mut full_precision_scores,
                rows,
                rows,
                self.head_dim,
                self.model_dim,
                1,
                32,
                commands,
            );
            observer(head, VisionAttentionStage::FullPrecisionScores, &full_precision_scores, commands)?;
            let mut scores = commands.allocate_scratch_for_shape(&[rows, rows], DataType::BF16)?;
            self.cast.encode(&full_precision_scores, &mut scores, rows * rows, commands);
            observer(head, VisionAttentionStage::Scores, &scores, commands)?;
            // Keep the QK BF16 rounding before scaling, even when the exact
            // official head64 scale (1/8) could commute in the normal range.
            self.scale_scores.encode(
                None::<&B::ScratchBuffer>,
                &zero,
                &mut scores,
                1,
                rows * rows,
                1.0 / (self.head_dim as f32).sqrt(),
                commands,
            );
            observer(head, VisionAttentionStage::ScaledScores, &scores, commands)?;
            self.softmax.encode(&mut scores, rows, rows, commands);
            observer(head, VisionAttentionStage::Probabilities, &scores, commands)?;
            let mut full_precision_head = commands.allocate_scratch_for_shape(&[rows, self.head_dim], DataType::F32)?;
            self.matmul.encode(
                &scores,
                values.subrange(kv_offset..),
                &mut full_precision_head,
                rows,
                self.head_dim,
                rows,
                self.model_dim,
                0,
                4,
                commands,
            );
            observer(head, VisionAttentionStage::FullPrecisionHead, &full_precision_head, commands)?;
            let mut head_output = commands.allocate_scratch_for_shape(&[rows, self.head_dim], DataType::BF16)?;
            self.cast.encode(&full_precision_head, &mut head_output, head_elements, commands);
            observer(head, VisionAttentionStage::Head, &head_output, commands)?;
            self.head_copy.encode(
                &head_output,
                &mut output,
                head_elements,
                self.head_dim,
                self.model_dim,
                head,
                commands,
            );
        }
        Ok(output)
    }
}

struct VisionBlock<B: Backend> {
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    qkv: RawLinear<B>,
    projection: RawLinear<B>,
    fc1: RawLinear<B>,
    fc2: RawLinear<B>,
    prepare: <B::Kernels as Kernels>::VisionPrepareBF16Kernel,
    attention: VisionEagerAttention<B>,
}

pub(crate) struct VisionTower<B: Backend> {
    config: VisionTowerConfig,
    patch_projection: RawLinear<B>,
    learned_positions: Box<[bf16]>,
    blocks: Box<[VisionBlock<B>]>,
    merger_norm: LayerNorm<B>,
    merger_fc1: RawLinear<B>,
    merger_fc2: RawLinear<B>,
    add: <B::Kernels as Kernels>::TensorAddScaleKernel,
    activation: <B::Kernels as Kernels>::VisionGeluBF16Kernel,
}

impl<B: Backend> VisionTower<B> {
    pub(crate) fn new(
        context: &B::Context,
        config: &VisionTowerConfig,
        tree: &ParameterTree<B>,
    ) -> Result<Self, VisionTowerError<B>> {
        config.validate()?;
        validate_parameters(config, tree)?;
        let width = config.hidden_size;
        let linear = |name: &str, input: u32, output: u32| {
            RawLinear::new(context, &tree.subtree(name), input, output, &[output, input])
        };
        let mut blocks = Vec::with_capacity(config.depth as usize);
        for index in 0..config.depth {
            blocks.push(VisionBlock {
                norm1: LayerNorm::new(
                    context,
                    &tree.subtree(&format!("blocks.{index}.norm1")),
                    width,
                    config.layer_norm_eps,
                )?,
                norm2: LayerNorm::new(
                    context,
                    &tree.subtree(&format!("blocks.{index}.norm2")),
                    width,
                    config.layer_norm_eps,
                )?,
                qkv: linear(&format!("blocks.{index}.attn.qkv"), width, width * 3)?,
                projection: linear(&format!("blocks.{index}.attn.proj"), width, width)?,
                fc1: linear(&format!("blocks.{index}.mlp.linear_fc1"), width, config.intermediate_size)?,
                fc2: linear(&format!("blocks.{index}.mlp.linear_fc2"), config.intermediate_size, width)?,
                prepare: <B::Kernels as Kernels>::VisionPrepareBF16Kernel::new(context)
                    .map_err(VisionTowerError::Backend)?,
                attention: VisionEagerAttention::new(context, width, config.num_heads)
                    .map_err(VisionTowerError::Backend)?,
            });
        }
        let learned_positions = tree
            .leaf("pos_embed.weight")?
            .validate(&[config.num_position_embeddings, width], DataType::BF16)?
            .read_slice::<bf16>()?;
        if learned_positions.iter().any(|value| !value.is_finite()) {
            return Err(InvalidInput("nonfinite learned vision position table").into());
        }
        let merged = config.merged_width()?;
        Ok(Self {
            config: config.clone(),
            patch_projection: RawLinear::new(
                context,
                &tree.subtree("patch_embed.proj"),
                config.patch_width()?,
                width,
                &[width, config.in_channels, config.temporal_patch_size, config.patch_size, config.patch_size],
            )?,
            learned_positions,
            blocks: blocks.into_boxed_slice(),
            merger_norm: LayerNorm::new(context, &tree.subtree("merger.norm"), width, config.layer_norm_eps)?,
            merger_fc1: linear("merger.linear_fc1", merged, merged)?,
            merger_fc2: linear("merger.linear_fc2", merged, config.out_hidden_size)?,
            add: <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, DataType::BF16, false)
                .map_err(VisionTowerError::Backend)?,
            activation: <B::Kernels as Kernels>::VisionGeluBF16Kernel::new(context)
                .map_err(VisionTowerError::Backend)?,
        })
    }

    pub(crate) fn encode(
        &self,
        patches: &[bf16],
        grid: [u32; 3],
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, VisionTowerError<B>> {
        self.encode_with_observer(patches, grid, commands, |_, _, _| Ok(()))
    }

    /// The observer is an internal validation hook. It must retain any owned
    /// readbacks until completion; no feature data is sent to diagnostics.
    pub(crate) fn encode_with_observer<F>(
        &self,
        patches: &[bf16],
        grid: [u32; 3],
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        mut observer: F,
    ) -> Result<B::ScratchBuffer, VisionTowerError<B>>
    where
        F: FnMut(
            VisionStage,
            &B::ScratchBuffer,
            &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        ) -> Result<(), B::Error>,
    {
        let (rows, mut hidden, cosines, sines) = self.prepare_hidden(patches, grid, commands, &mut observer)?;
        for index in 0..self.blocks.len() {
            hidden = self.encode_block(index, hidden, rows, &cosines, &sines, commands, &mut |_, _, _| Ok(()))?;
            observer(VisionStage::Block(index), &hidden, commands).map_err(VisionTowerError::Backend)?;
        }
        let output = self.encode_merger(&hidden, rows, commands, &mut |_, _, _| Ok(()))?;
        observer(VisionStage::Merger, &output, commands).map_err(VisionTowerError::Backend)?;
        Ok(output)
    }

    #[cfg(test)]
    pub(crate) fn encode_first_block_with_observer<F>(
        &self,
        patches: &[bf16],
        grid: [u32; 3],
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        mut observer: F,
    ) -> Result<B::ScratchBuffer, VisionTowerError<B>>
    where
        F: FnMut(
            VisionBlockStage,
            &B::ScratchBuffer,
            &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        ) -> Result<(), B::Error>,
    {
        let (rows, hidden, cosines, sines) = self.prepare_hidden(patches, grid, commands, &mut |_, _, _| Ok(()))?;
        self.encode_block(0, hidden, rows, &cosines, &sines, commands, &mut observer)
    }

    /// Feed exact reference hidden rows to one production block. This isolates
    /// a local primitive discrepancy from errors accumulated in earlier blocks.
    #[cfg(test)]
    pub(crate) fn encode_isolated_block_with_observer<F>(
        &self,
        index: usize,
        hidden: &[bf16],
        grid: [u32; 3],
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        mut observer: F,
    ) -> Result<B::ScratchBuffer, VisionTowerError<B>>
    where
        F: FnMut(
            VisionBlockStage,
            &B::ScratchBuffer,
            &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        ) -> Result<(), B::Error>,
    {
        if index >= self.blocks.len() {
            return Err(InvalidInput("invalid isolated vision block").into());
        }
        let rows = checked_rows(grid, self.config.spatial_merge_size)?;
        let hidden = self.upload_hidden(hidden, rows, commands)?;
        let positions = spatial_positions(grid, self.config.spatial_merge_size)?;
        let (cosines, sines) = rotary_tables(&positions, self.config.hidden_size / self.config.num_heads)?;
        let cosines = commands.allocate_constant_from_slice(&cosines).map_err(VisionTowerError::Backend)?;
        let sines = commands.allocate_constant_from_slice(&sines).map_err(VisionTowerError::Backend)?;
        self.encode_block(index, hidden, rows, &cosines, &sines, commands, &mut observer)
    }

    #[cfg(test)]
    pub(crate) fn encode_isolated_merger_with_observer<F>(
        &self,
        hidden: &[bf16],
        grid: [u32; 3],
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        mut observer: F,
    ) -> Result<B::ScratchBuffer, VisionTowerError<B>>
    where
        F: FnMut(
            VisionMergerStage,
            &B::ScratchBuffer,
            &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        ) -> Result<(), B::Error>,
    {
        let rows = checked_rows(grid, self.config.spatial_merge_size)?;
        let hidden = self.upload_hidden(hidden, rows, commands)?;
        self.encode_merger(&hidden, rows, commands, &mut observer)
    }

    /// Observe exact score/probability boundaries from captured BF16 inputs.
    #[cfg(test)]
    pub(crate) fn encode_isolated_attention_with_observer<F>(
        &self,
        index: usize,
        queries: &[bf16],
        keys: &[bf16],
        values: &[bf16],
        grid: [u32; 3],
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        mut observer: F,
    ) -> Result<B::ScratchBuffer, VisionTowerError<B>>
    where
        F: FnMut(
            u32,
            VisionAttentionStage,
            &B::ScratchBuffer,
            &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        ) -> Result<(), B::Error>,
    {
        let block = self.blocks.get(index).ok_or(InvalidInput("invalid isolated vision attention block"))?;
        let rows = checked_rows(grid, self.config.spatial_merge_size)?;
        let queries = self.upload_hidden(queries, rows, commands)?;
        let keys = self.upload_hidden(keys, rows, commands)?;
        let values = self.upload_hidden(values, rows, commands)?;
        block
            .attention
            .encode_with_observer(&queries, &keys, &values, rows, commands, &mut observer)
            .map_err(VisionTowerError::Backend)
    }

    #[cfg(test)]
    fn upload_hidden(
        &self,
        hidden: &[bf16],
        rows: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, VisionTowerError<B>> {
        if hidden.len() != checked_elements(rows, self.config.hidden_size)? as usize
            || hidden.iter().any(|value| !value.is_finite())
        {
            return Err(InvalidInput("invalid isolated vision hidden rows").into());
        }
        let source = commands.allocate_constant_from_slice(hidden).map_err(VisionTowerError::Backend)?;
        let mut output = commands
            .allocate_scratch_for_shape(&[rows, self.config.hidden_size], DataType::BF16)
            .map_err(VisionTowerError::Backend)?;
        commands.encode_copy(&source, &mut output);
        Ok(output)
    }

    fn encode_merger<F>(
        &self,
        hidden: &B::ScratchBuffer,
        rows: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        observer: &mut F,
    ) -> Result<B::ScratchBuffer, VisionTowerError<B>>
    where
        F: FnMut(
            VisionMergerStage,
            &B::ScratchBuffer,
            &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        ) -> Result<(), B::Error>,
    {
        let config = &self.config;
        // Each merge block's patch rows are consecutive. Normalize individual
        // patches before viewing their same bytes as one wider merger row.
        let normalized = self.merger_norm.encode(hidden, rows, commands).map_err(VisionTowerError::Backend)?;
        observer(VisionMergerStage::Norm, &normalized, commands).map_err(VisionTowerError::Backend)?;
        let merged_rows = rows / config.spatial_merge_size / config.spatial_merge_size;
        let mut merged =
            self.merger_fc1.encode(&normalized, merged_rows, commands).map_err(VisionTowerError::Backend)?;
        observer(VisionMergerStage::Fc1, &merged, commands).map_err(VisionTowerError::Backend)?;
        self.activation.encode(&mut merged, checked_elements(merged_rows, config.merged_width()?)?, 0, commands);
        observer(VisionMergerStage::Gelu, &merged, commands).map_err(VisionTowerError::Backend)?;
        let output = self.merger_fc2.encode(&merged, merged_rows, commands).map_err(VisionTowerError::Backend)?;
        observer(VisionMergerStage::Output, &output, commands).map_err(VisionTowerError::Backend)?;
        Ok(output)
    }

    fn prepare_hidden<F>(
        &self,
        patches: &[bf16],
        grid: [u32; 3],
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        observer: &mut F,
    ) -> Result<(u32, B::ScratchBuffer, B::ConstantBuffer, B::ConstantBuffer), VisionTowerError<B>>
    where
        F: FnMut(
            VisionStage,
            &B::ScratchBuffer,
            &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        ) -> Result<(), B::Error>,
    {
        let config = &self.config;
        let rows = checked_rows(grid, config.spatial_merge_size)?;
        let count = checked_elements(rows, config.patch_width()?)? as usize;
        if patches.len() != count || patches.iter().any(|value| !value.is_finite()) {
            return Err(InvalidInput("invalid normalized vision patch tensor").into());
        }
        let positions = spatial_positions(grid, config.spatial_merge_size)?;
        let learned = interpolated_positions(config, &self.learned_positions, &positions, grid)?;
        let head_dim = config.hidden_size / config.num_heads;
        let (cosines, sines) = rotary_tables(&positions, head_dim)?;
        let source = commands.allocate_constant_from_slice(patches).map_err(VisionTowerError::Backend)?;
        let learned = commands.allocate_constant_from_slice(&learned).map_err(VisionTowerError::Backend)?;
        let cosines = commands.allocate_constant_from_slice(&cosines).map_err(VisionTowerError::Backend)?;
        let sines = commands.allocate_constant_from_slice(&sines).map_err(VisionTowerError::Backend)?;
        let mut hidden = self.patch_projection.encode(&source, rows, commands).map_err(VisionTowerError::Backend)?;
        observer(VisionStage::PatchProjection, &hidden, commands).map_err(VisionTowerError::Backend)?;
        hidden = self.add_buffers(&hidden, &learned, rows, commands).map_err(VisionTowerError::Backend)?;
        observer(VisionStage::PositionAdded, &hidden, commands).map_err(VisionTowerError::Backend)?;
        Ok((rows, hidden, cosines, sines))
    }

    fn encode_block<F>(
        &self,
        index: usize,
        mut hidden: B::ScratchBuffer,
        rows: u32,
        cosines: &B::ConstantBuffer,
        sines: &B::ConstantBuffer,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        observer: &mut F,
    ) -> Result<B::ScratchBuffer, VisionTowerError<B>>
    where
        F: FnMut(
            VisionBlockStage,
            &B::ScratchBuffer,
            &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        ) -> Result<(), B::Error>,
    {
        let config = &self.config;
        let head_dim = config.hidden_size / config.num_heads;
        let block = &self.blocks[index];
        let normalized = block.norm1.encode(&hidden, rows, commands).map_err(VisionTowerError::Backend)?;
        observer(VisionBlockStage::Norm1, &normalized, commands).map_err(VisionTowerError::Backend)?;
        let projected = block.qkv.encode(&normalized, rows, commands).map_err(VisionTowerError::Backend)?;
        observer(VisionBlockStage::Qkv, &projected, commands).map_err(VisionTowerError::Backend)?;
        let mut queries = commands
            .allocate_scratch_for_shape(&[config.num_heads, rows, head_dim], DataType::BF16)
            .map_err(VisionTowerError::Backend)?;
        let mut keys = commands
            .allocate_scratch_for_shape(&[rows, config.num_heads, head_dim], DataType::BF16)
            .map_err(VisionTowerError::Backend)?;
        let mut values = commands
            .allocate_scratch_for_shape(&[rows, config.num_heads, head_dim], DataType::BF16)
            .map_err(VisionTowerError::Backend)?;
        block.prepare.encode(
            &projected,
            &mut queries,
            &mut keys,
            &mut values,
            cosines,
            sines,
            rows,
            config.num_heads,
            head_dim,
            commands,
        );
        observer(VisionBlockStage::Queries, &queries, commands).map_err(VisionTowerError::Backend)?;
        observer(VisionBlockStage::Keys, &keys, commands).map_err(VisionTowerError::Backend)?;
        observer(VisionBlockStage::Values, &values, commands).map_err(VisionTowerError::Backend)?;
        let attended =
            block.attention.encode(&queries, &keys, &values, rows, commands).map_err(VisionTowerError::Backend)?;
        observer(VisionBlockStage::Attention, &attended, commands).map_err(VisionTowerError::Backend)?;
        let attended = block.projection.encode(&attended, rows, commands).map_err(VisionTowerError::Backend)?;
        observer(VisionBlockStage::Projection, &attended, commands).map_err(VisionTowerError::Backend)?;
        hidden = self.add_buffers(&hidden, &attended, rows, commands).map_err(VisionTowerError::Backend)?;
        observer(VisionBlockStage::AttentionResidual, &hidden, commands).map_err(VisionTowerError::Backend)?;
        let normalized = block.norm2.encode(&hidden, rows, commands).map_err(VisionTowerError::Backend)?;
        observer(VisionBlockStage::Norm2, &normalized, commands).map_err(VisionTowerError::Backend)?;
        let mut up = block.fc1.encode(&normalized, rows, commands).map_err(VisionTowerError::Backend)?;
        observer(VisionBlockStage::Fc1, &up, commands).map_err(VisionTowerError::Backend)?;
        self.activation.encode(&mut up, checked_elements(rows, config.intermediate_size)?, 1, commands);
        observer(VisionBlockStage::Gelu, &up, commands).map_err(VisionTowerError::Backend)?;
        let down = block.fc2.encode(&up, rows, commands).map_err(VisionTowerError::Backend)?;
        observer(VisionBlockStage::Fc2, &down, commands).map_err(VisionTowerError::Backend)?;
        hidden = self.add_buffers(&hidden, &down, rows, commands).map_err(VisionTowerError::Backend)?;
        observer(VisionBlockStage::Output, &hidden, commands).map_err(VisionTowerError::Backend)?;
        Ok(hidden)
    }

    fn add_buffers(
        &self,
        input: &B::ScratchBuffer,
        addition: impl BufferRef<Backend = B>,
        rows: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut output = commands.allocate_scratch_for_shape(&[rows, self.config.hidden_size], DataType::BF16)?;
        let elements = rows * self.config.hidden_size;
        self.add.encode(Some(input), addition, &mut output, elements, elements, 1.0, commands);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn vision_spatial_positions_follow_rectangular_merge_blocks() {
        let positions = spatial_positions([1, 4, 6], 2).unwrap();
        assert_eq!(&positions[..8], &[[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]]);
        assert_eq!(&positions[20..], &[[2, 4], [2, 5], [3, 4], [3, 5]]);
        assert!(spatial_positions([2, 4, 6], 2).is_err());
        assert!(spatial_positions([1, 3, 6], 2).is_err());
        assert!(spatial_positions([1, u32::MAX, 2], 1).is_err());
        assert!(spatial_positions([1, 66, 64], 2).is_err());
    }

    #[stark_test]
    fn vision_axial_rotary_tables_match_independent_f64_equations() {
        let positions = [[0, 0], [2, 7], [3, 1]];
        let (cosines, sines) = rotary_tables(&positions, 64).unwrap();
        for (row, position) in positions.iter().enumerate() {
            for channel in 0..64usize {
                let half_channel = channel % 32;
                let axis = half_channel / 16;
                let power = (half_channel % 16) as f64 / 16.0;
                let angle = position[axis] as f64 * 10000f64.powf(-power);
                assert!((cosines[row * 64 + channel] as f64 - angle.cos()).abs() < 2e-6);
                assert!((sines[row * 64 + channel] as f64 - angle.sin()).abs() < 2e-6);
            }
        }
        assert!(rotary_tables(&positions, 6).is_err());
    }

    #[stark_test]
    fn vision_learned_positions_use_align_corners_and_merge_order() {
        let config = VisionTowerConfig {
            depth: 1,
            hidden_size: 8,
            intermediate_size: 16,
            num_heads: 2,
            num_position_embeddings: 4,
            out_hidden_size: 8,
            patch_size: 2,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            in_channels: 3,
            layer_norm_eps: 1e-6,
        };
        assert!(config.validate().is_ok());
        // The source table is an affine plane, so the independent analytic
        // interpolation is exact at these target coordinates.
        let table = (0..2)
            .flat_map(|row| {
                (0..2).flat_map(move |column| {
                    (0..8).map(move |channel| bf16::from_f32((row * 40 + column * 12 + channel) as f32))
                })
            })
            .collect::<Vec<_>>();
        let grid = [1, 2, 4];
        let positions = spatial_positions(grid, 2).unwrap();
        let actual = interpolated_positions(&config, &table, &positions, grid).unwrap();
        let expected = positions
            .iter()
            .flat_map(|[row, column]| {
                (0..8).map(move |channel| bf16::from_f32((row * 40 + column * 4 + channel) as f32))
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert!(interpolated_positions(&config, &table[..4], &positions, grid).is_err());
        let mut invalid = table;
        invalid[0] = bf16::NAN;
        assert!(interpolated_positions(&config, &invalid, &positions, grid).is_err());
        let mut invalid = config;
        invalid.num_position_embeddings = 3;
        assert!(invalid.validate().is_err());
    }

    #[cfg(feature = "cpu")]
    fn native_cast_rounding<B: Backend>() {
        use crate::{
            backends::common::{CommandBufferExecutable, CommandBufferPending, Context},
            tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data},
        };

        let context = B::Context::new().unwrap();
        let kernel = <B::Kernels as Kernels>::VisionCastBF16Kernel::new(&context).unwrap();
        // Include BF16 halfway values with both even and odd mantissas, signs,
        // subnormal/normal transitions and infinities. Feature inputs themselves
        // are finite; these values test the cast primitive's defined conversion.
        let input = [
            0.0f32,
            -0.0,
            1.00390625,
            1.01171875,
            -1.00390625,
            -1.01171875,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        let source = create_buffer_with_data::<B, f32>(&context, &input);
        let mut output = create_buffer::<B, bf16>(&context, input.len());
        let mut commands = context.create_command_buffer(None, None).unwrap();
        kernel.encode(&source, &mut output, input.len() as u32, &mut commands);
        commands.end_encoding().submit().wait_until_completed().unwrap();
        let actual = buffer_to_vec::<B, bf16>(&output);
        assert_eq!(
            actual.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            input.map(|value| bf16::from_f32(value).to_bits())
        );
    }

    #[cfg(feature = "cpu")]
    #[stark_test]
    fn vision_native_final_cast_preserves_bf16_rounding_boundaries() {
        use crate::{backends::cpu::Cpu, tests::helpers::for_each_non_cpu_backend};

        native_cast_rounding::<Cpu>();
        for_each_non_cpu_backend!(|B| native_cast_rounding::<B>());
    }

    #[cfg(feature = "cpu")]
    fn native_layer_norm_moments<B: Backend>() {
        use crate::{
            backends::common::{CommandBufferExecutable, CommandBufferPending, Context},
            tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data},
        };

        let context = B::Context::new().unwrap();
        let kernel = <B::Kernels as Kernels>::VisionLayerNormBF16Kernel::new(&context).unwrap();
        // Exercise scalar tails, full BF16 vector pairs, multiple Welford chunks
        // and the official width. Large offsets with small variance expose the
        // cancellation in E[x^2]-E[x]^2; the oracle computes centered FP64 sums.
        for width in [4usize, 8, 20, 64, 768] {
            let rows = 4usize;
            let inputs = (0..rows * width)
                .map(|index| {
                    let column = index % width;
                    let pattern = (column % 3) as f32 - 1.0;
                    bf16::from_f32(match index / width {
                        0 => 12800.0,
                        1 => 12800.0 + pattern * 64.0,
                        2 => ((column * 13 % 31) as f32 - 15.0) / 16.0,
                        _ => -12800.0 + pattern * 64.0,
                    })
                })
                .collect::<Vec<_>>();
            let weights = (0..width).map(|index| bf16::from_f32(0.5 + (index % 7) as f32 / 8.0)).collect::<Vec<_>>();
            let biases = (0..width).map(|index| bf16::from_f32((index % 9) as f32 / 32.0)).collect::<Vec<_>>();
            let source = create_buffer_with_data::<B, bf16>(&context, &inputs);
            let scales = create_buffer_with_data::<B, bf16>(&context, &weights);
            let offsets = create_buffer_with_data::<B, bf16>(&context, &biases);
            let mut output = create_buffer::<B, bf16>(&context, inputs.len());
            let mut commands = context.create_command_buffer(None, None).unwrap();
            kernel.encode(&source, &scales, &offsets, &mut output, rows as u32, width as u32, 1e-6, &mut commands);
            commands.end_encoding().submit().wait_until_completed().unwrap();
            let actual = buffer_to_vec::<B, bf16>(&output);
            let mut squared_error = 0.0;
            let mut squared_reference = 0.0;
            for row in 0..rows {
                let values = &inputs[row * width..(row + 1) * width];
                let mean = values.iter().map(|value| f64::from(value.to_f32())).sum::<f64>() / width as f64;
                let variance =
                    values.iter().map(|value| (f64::from(value.to_f32()) - mean).powi(2)).sum::<f64>() / width as f64;
                let inverse_std = (variance + 1e-6).sqrt().recip();
                for column in 0..width {
                    let normalized = (f64::from(values[column].to_f32()) - mean) * inverse_std;
                    let expected = bf16::from_f32(
                        (normalized * f64::from(weights[column].to_f32()) + f64::from(biases[column].to_f32())) as f32,
                    );
                    let result = actual[row * width + column].to_f32();
                    assert!(result.is_finite(), "vision normalization produced a non-finite value");
                    squared_error += (f64::from(result) - f64::from(expected.to_f32())).powi(2);
                    squared_reference += f64::from(expected.to_f32()).powi(2);
                }
            }
            assert!(
                (squared_error / squared_reference).sqrt() < 0.0015,
                "vision normalization differs from centered FP64 moments"
            );
        }
    }

    #[cfg(feature = "cpu")]
    #[stark_test]
    fn vision_native_layer_norm_preserves_small_variance_with_large_offsets() {
        use crate::{backends::cpu::Cpu, tests::helpers::for_each_non_cpu_backend};

        native_layer_norm_moments::<Cpu>();
        for_each_non_cpu_backend!(|B| native_layer_norm_moments::<B>());
    }

    #[cfg(feature = "cpu")]
    fn native_eager_attention<B: Backend>() {
        use crate::{
            backends::common::{CommandBufferExecutable, CommandBufferPending, Context},
            tests::helpers::{buffer_to_vec, create_buffer_with_data},
        };

        let context = B::Context::new().unwrap();
        let attention = VisionEagerAttention::<B>::new(&context, 128, 2).unwrap();
        let (rows, heads, width) = (4usize, 2usize, 64usize);
        let queries = (0..heads * rows * width)
            .map(|index| bf16::from_f32(((index * 7 % 23) as f32 - 11.0) / 8.0))
            .collect::<Vec<_>>();
        let keys = (0..rows * heads * width)
            .map(|index| bf16::from_f32(((index * 11 % 29) as f32 - 14.0) / 8.0))
            .collect::<Vec<_>>();
        let values = (0..rows * heads * width)
            .map(|index| bf16::from_f32(((index * 13 % 31) as f32 - 15.0) / 8.0))
            .collect::<Vec<_>>();
        let mut expected = vec![bf16::ZERO; rows * heads * width];
        let mut differs_from_fused = false;
        for head in 0..heads {
            for row in 0..rows {
                let dots = (0..rows)
                    .map(|key_row| {
                        (0..width)
                            .map(|channel| {
                                f64::from(queries[(head * rows + row) * width + channel].to_f32())
                                    * f64::from(keys[(key_row * heads + head) * width + channel].to_f32())
                            })
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                let scores = dots
                    .iter()
                    .map(|&value| bf16::from_f32(bf16::from_f32(value as f32).to_f32() / 8.0).to_f32() as f64)
                    .collect::<Vec<_>>();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let total = scores.iter().map(|score| (score - maximum).exp()).sum::<f64>();
                let probabilities = scores
                    .iter()
                    .map(|score| bf16::from_f32(((score - maximum).exp() / total) as f32).to_f32() as f64)
                    .collect::<Vec<_>>();
                let fused_maximum = dots.iter().map(|score| score / 8.0).fold(f64::NEG_INFINITY, f64::max);
                let fused_total = dots.iter().map(|score| (score / 8.0 - fused_maximum).exp()).sum::<f64>();
                for channel in 0..width {
                    let value = (0..rows)
                        .map(|key_row| {
                            probabilities[key_row]
                                * f64::from(values[(key_row * heads + head) * width + channel].to_f32())
                        })
                        .sum::<f64>();
                    let fused = (0..rows)
                        .map(|key_row| {
                            ((dots[key_row] / 8.0 - fused_maximum).exp() / fused_total)
                                * f64::from(values[(key_row * heads + head) * width + channel].to_f32())
                        })
                        .sum::<f64>();
                    let expected_value = bf16::from_f32(value as f32);
                    expected[(row * heads + head) * width + channel] = expected_value;
                    differs_from_fused |= expected_value.to_bits() != bf16::from_f32(fused as f32).to_bits();
                }
            }
        }
        assert!(differs_from_fused, "eager oracle must detect omitted BF16 attention boundaries");
        let mut commands = context.create_command_buffer(None, None).unwrap();
        let mut query_buffer = commands.allocate_scratch_for_shape(&[2, 4, 64], DataType::BF16).unwrap();
        let mut key_buffer = commands.allocate_scratch_for_shape(&[4, 2, 64], DataType::BF16).unwrap();
        let mut value_buffer = commands.allocate_scratch_for_shape(&[4, 2, 64], DataType::BF16).unwrap();
        let query_source = create_buffer_with_data::<B, bf16>(&context, &queries);
        let key_source = create_buffer_with_data::<B, bf16>(&context, &keys);
        let value_source = create_buffer_with_data::<B, bf16>(&context, &values);
        commands.encode_copy(&query_source, &mut query_buffer);
        commands.encode_copy(&key_source, &mut key_buffer);
        commands.encode_copy(&value_source, &mut value_buffer);
        let output = attention.encode(&query_buffer, &key_buffer, &value_buffer, rows as u32, &mut commands).unwrap();
        let mut readback = context.create_buffer(rows * heads * width * size_of::<bf16>()).unwrap();
        commands.encode_copy(&output, &mut readback);
        commands.end_encoding().submit().wait_until_completed().unwrap();
        let actual = buffer_to_vec::<B, bf16>(&readback);
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                actual.to_bits() == expected.to_bits(),
                "native eager attention differs from independent oracle at {index}"
            );
        }
    }

    #[cfg(feature = "cpu")]
    #[stark_test]
    fn vision_native_eager_attention_preserves_reference_rounding_and_head_layout() {
        use crate::{backends::cpu::Cpu, tests::helpers::for_each_non_cpu_backend};

        native_eager_attention::<Cpu>();
        for_each_non_cpu_backend!(|B| native_eager_attention::<B>());
    }

    #[cfg(feature = "cpu")]
    fn native_affine_rounding<B: Backend>() {
        use std::{
            fs::OpenOptions,
            io::{Seek, Write},
            time::{SystemTime, UNIX_EPOCH},
        };

        use crate::{
            backends::common::{CommandBufferExecutable, CommandBufferPending, Context},
            parameters::ParameterLoader,
            tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data},
        };

        let context = B::Context::new().unwrap();
        let cast = <B::Kernels as Kernels>::VisionBiasCastBF16Kernel::new(&context).unwrap();
        let input = [1.00390625f32, -1.00390625, 1.01171875, -1.01171875];
        let bias = [bf16::from_f32(0.00390625), bf16::from_f32(-0.00390625)];
        let input_buffer = create_buffer_with_data::<B, f32>(&context, &input);
        let bias_buffer = create_buffer_with_data::<B, bf16>(&context, &bias);
        let mut output_buffer = create_buffer::<B, bf16>(&context, input.len());
        let mut commands = context.create_command_buffer(None, None).unwrap();
        cast.encode(&input_buffer, &bias_buffer, &mut output_buffer, 2, 4, &mut commands);
        commands.end_encoding().submit().wait_until_completed().unwrap();
        let actual = buffer_to_vec::<B, bf16>(&output_buffer);
        for index in 0..input.len() {
            let expected = bf16::from_f32(input[index] + bias[index % 2].to_f32());
            assert!(actual[index].to_bits() == expected.to_bits(), "vision affine bias boundary differs");
        }
        assert!(actual[0].to_bits() != bf16::from_f32(bf16::from_f32(input[0]).to_f32() + bias[0].to_f32()).to_bits());

        let (rows, inner, columns) = (256u32, 768u32, 128u32);
        let inputs =
            (0..rows * inner).map(|index| bf16::from_f32(((index * 7 % 31) as f32 - 15.0) / 16.0)).collect::<Vec<_>>();
        let weights = (0..columns * inner)
            .map(|index| bf16::from_f32(((index * 11 % 37) as f32 - 18.0) / 32.0))
            .collect::<Vec<_>>();
        let biases = (0..columns).map(|index| bf16::from_f32(((index % 13) as f32 - 6.0) / 256.0)).collect::<Vec<_>>();
        let path = std::env::temp_dir().join(format!(
            "stark-vision-affine-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let weight_bytes = weights.len() * 2;
        let mut header = serde_json::to_vec(&serde_json::json!({
            "weight": {"dtype":"BF16", "shape":[columns, inner], "data_offsets":[0,weight_bytes]},
            "bias": {"dtype":"BF16", "shape":[columns], "data_offsets":[weight_bytes,weight_bytes+biases.len()*2]},
        }))
        .unwrap();
        header.resize(header.len().next_multiple_of(8), b' ');
        file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&header).unwrap();
        for value in weights.iter().chain(&biases) {
            file.write_all(&value.to_bits().to_le_bytes()).unwrap();
        }
        file.rewind().unwrap();
        let loader = ParameterLoader::<B>::new(&file, &context).unwrap();
        let tree = loader.tree();
        let linear = RawLinear::<B>::new(&context, &tree, inner, columns, &[columns, inner])
            .unwrap_or_else(|_| panic!("native vision affine fixture loading failed"));
        tree.assert_all_tensors_validated().unwrap();
        let source = create_buffer_with_data::<B, bf16>(&context, &inputs);
        let mut commands = context.create_command_buffer(None, None).unwrap();
        let output = linear.encode(&source, rows, &mut commands).unwrap();
        let mut readback = context.create_buffer(rows as usize * columns as usize * 2).unwrap();
        commands.encode_copy(&output, &mut readback);
        commands.end_encoding().submit().wait_until_completed().unwrap();
        let actual = buffer_to_vec::<B, bf16>(&readback);
        for row in 0..rows as usize {
            for column in 0..columns as usize {
                // Power-of-two inputs make these products and complete sums
                // exact in FP32; a BF16 split partition can still lose bits.
                let sum = (0..inner as usize)
                    .map(|index| {
                        f64::from(inputs[row * inner as usize + index].to_f32())
                            * f64::from(weights[column * inner as usize + index].to_f32())
                    })
                    .sum::<f64>();
                let expected = bf16::from_f32((sum + f64::from(biases[column].to_f32())) as f32);
                assert!(
                    actual[row * columns as usize + column].to_bits() == expected.to_bits(),
                    "native vision affine differs from independent exact-sum oracle"
                );
            }
        }
    }

    #[cfg(feature = "cpu")]
    #[stark_test]
    fn vision_native_affine_preserves_full_dot_and_bias_before_bf16_cast() {
        use crate::{backends::cpu::Cpu, tests::helpers::for_each_non_cpu_backend};

        native_affine_rounding::<Cpu>();
        for_each_non_cpu_backend!(|B| native_affine_rounding::<B>());
    }
}
