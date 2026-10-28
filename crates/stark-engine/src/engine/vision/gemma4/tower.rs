//! Gemma 4 E2B native BF16 image encoder and soft-token projector.
//!
//! Equations/layout independently implemented from Google's Apache-2.0 Gemma 4
//! checkpoint and MIT MLX-VLM 53ebd1647cb83549b482a031ea70ba9d9ea60a04.

use half::bf16;

use super::super::tower::VisionTowerError;
use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferEncoding, InvalidInput, Kernels,
        kernel::{
            AttentionSinglePassKernel, Gemma4VisionClampBF16Kernel, Gemma4VisionNormBF16Kernel,
            Gemma4VisionPoolBF16Kernel, Gemma4VisionPrepareBF16Kernel, Gemma4VisionProductBF16Kernel,
            TensorAddScaleKernel, VisionCastBF16Kernel, VisionGeluBF16Kernel, VisionMatmulBF16Kernel,
        },
    },
    data_type::DataType,
    parameters::ParameterTree,
};

const WIDTH: u32 = 768;
const HEADS: u32 = 12;
const HEAD_DIM: u32 = 64;
const MLP: u32 = 3072;
const MODEL: u32 = 1536;
const DEPTH: usize = 16;
const POSITIONS: u32 = 10240;

struct Linear<B: Backend> {
    matrix: <B::Kernels as Kernels>::VisionMatmulBF16Kernel,
    cast: <B::Kernels as Kernels>::VisionCastBF16Kernel,
    clamp: <B::Kernels as Kernels>::Gemma4VisionClampBF16Kernel,
    weights: B::GlobalBuffer,
    input: u32,
    output: u32,
    clip: Option<[f32; 4]>,
}

impl<B: Backend> Linear<B> {
    fn new(
        context: &B::Context,
        tree: &ParameterTree<B>,
        input: u32,
        output: u32,
        clipped: bool,
    ) -> Result<Self, VisionTowerError<B>> {
        let (weights, clip) = if clipped {
            let mut clip = [0.0; 4];
            for (i, name) in ["input_min", "input_max", "output_min", "output_max"].iter().enumerate() {
                let values = tree.leaf(name)?.validate(&[], DataType::BF16)?.read_slice::<bf16>()?;
                if values.len() != 1 || values[0].is_nan() {
                    return Err(InvalidInput("invalid Gemma vision clipping scalar").into());
                }
                clip[i] = values[0].to_f32();
            }
            if clip[0] > clip[1] || clip[2] > clip[3] {
                return Err(InvalidInput("unordered Gemma vision clipping bounds").into());
            }
            (tree.leaf("linear.weight")?.validate(&[output, input], DataType::BF16)?.read_buffer()?, Some(clip))
        } else {
            (tree.leaf("weight")?.validate(&[output, input], DataType::BF16)?.read_buffer()?, None)
        };
        Ok(Self {
            matrix: <B::Kernels as Kernels>::VisionMatmulBF16Kernel::new(context).map_err(VisionTowerError::Backend)?,
            cast: <B::Kernels as Kernels>::VisionCastBF16Kernel::new(context).map_err(VisionTowerError::Backend)?,
            clamp: <B::Kernels as Kernels>::Gemma4VisionClampBF16Kernel::new(context)
                .map_err(VisionTowerError::Backend)?,
            weights,
            input,
            output,
            clip,
        })
    }

    fn encode(
        &self,
        input: &B::ScratchBuffer,
        rows: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let clamped = if let Some(clip) = self.clip {
            let mut values = commands.allocate_scratch_for_shape(&[rows, self.input], DataType::BF16)?;
            self.clamp.encode(input, &mut values, rows * self.input, clip[0], clip[1], commands);
            Some(values)
        } else {
            None
        };
        let input = clamped.as_ref().unwrap_or(input);
        let mut full = commands.allocate_scratch_for_shape(&[rows, self.output], DataType::F32)?;
        self.matrix.encode(input, &self.weights, &mut full, rows, self.output, self.input, self.input, 1, 32, commands);
        let mut result = commands.allocate_scratch_for_shape(&[rows, self.output], DataType::BF16)?;
        self.cast.encode(&full, &mut result, rows * self.output, commands);
        if let Some(clip) = self.clip {
            let mut clipped = commands.allocate_scratch_for_shape(&[rows, self.output], DataType::BF16)?;
            self.clamp.encode(&result, &mut clipped, rows * self.output, clip[2], clip[3], commands);
            Ok(clipped)
        } else {
            Ok(result)
        }
    }
}

struct Norm<B: Backend> {
    kernel: <B::Kernels as Kernels>::Gemma4VisionNormBF16Kernel,
    scales: B::GlobalBuffer,
    width: u32,
}

impl<B: Backend> Norm<B> {
    fn new(
        context: &B::Context,
        tree: &ParameterTree<B>,
        width: u32,
    ) -> Result<Self, VisionTowerError<B>> {
        Ok(Self {
            kernel: <B::Kernels as Kernels>::Gemma4VisionNormBF16Kernel::new(context)
                .map_err(VisionTowerError::Backend)?,
            scales: tree.leaf("weight")?.validate(&[width], DataType::BF16)?.read_buffer()?,
            width,
        })
    }
    fn encode(
        &self,
        input: &B::ScratchBuffer,
        rows: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut output = commands.allocate_scratch_for_shape(&[rows, self.width], DataType::BF16)?;
        self.kernel.encode(input, &self.scales, &mut output, rows, self.width, 1e-6, 1, commands);
        Ok(output)
    }
}

struct Block<B: Backend> {
    input_norm: Norm<B>,
    post_attention: Norm<B>,
    pre_ff: Norm<B>,
    post_ff: Norm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    q_norm: Norm<B>,
    k_norm: Norm<B>,
    gate: Linear<B>,
    up: Linear<B>,
    down: Linear<B>,
}

pub(crate) struct Gemma4VisionTower<B: Backend> {
    patch: Linear<B>,
    positions: Box<[bf16]>,
    blocks: Box<[Block<B>]>,
    projector: Linear<B>,
    norm: <B::Kernels as Kernels>::Gemma4VisionNormBF16Kernel,
    prepare: <B::Kernels as Kernels>::Gemma4VisionPrepareBF16Kernel,
    pool: <B::Kernels as Kernels>::Gemma4VisionPoolBF16Kernel,
    product: <B::Kernels as Kernels>::Gemma4VisionProductBF16Kernel,
    add: <B::Kernels as Kernels>::TensorAddScaleKernel,
    gelu: <B::Kernels as Kernels>::VisionGeluBF16Kernel,
    attention: <B::Kernels as Kernels>::AttentionSinglePassKernel,
}

impl<B: Backend> Gemma4VisionTower<B> {
    pub(crate) fn new(
        context: &B::Context,
        tree: &ParameterTree<B>,
    ) -> Result<Self, VisionTowerError<B>> {
        let mut blocks = Vec::with_capacity(DEPTH);
        for i in 0..DEPTH {
            let t = tree.subtree(&format!("encoder.layers.{i}"));
            let linear = |name, input, output| Linear::new(context, &t.subtree(name), input, output, true);
            blocks.push(Block {
                input_norm: Norm::new(context, &t.subtree("input_layernorm"), WIDTH)?,
                post_attention: Norm::new(context, &t.subtree("post_attention_layernorm"), WIDTH)?,
                pre_ff: Norm::new(context, &t.subtree("pre_feedforward_layernorm"), WIDTH)?,
                post_ff: Norm::new(context, &t.subtree("post_feedforward_layernorm"), WIDTH)?,
                q: linear("self_attn.q_proj", WIDTH, WIDTH)?,
                k: linear("self_attn.k_proj", WIDTH, WIDTH)?,
                v: linear("self_attn.v_proj", WIDTH, WIDTH)?,
                o: linear("self_attn.o_proj", WIDTH, WIDTH)?,
                q_norm: Norm::new(context, &t.subtree("self_attn.q_norm"), HEAD_DIM)?,
                k_norm: Norm::new(context, &t.subtree("self_attn.k_norm"), HEAD_DIM)?,
                gate: linear("mlp.gate_proj", WIDTH, MLP)?,
                up: linear("mlp.up_proj", WIDTH, MLP)?,
                down: linear("mlp.down_proj", MLP, WIDTH)?,
            });
        }
        let positions = tree
            .leaf("patch_embedder.position_embedding_table")?
            .validate(&[2, POSITIONS, WIDTH], DataType::BF16)?
            .read_slice::<bf16>()?;
        if positions.iter().any(|value| !value.is_finite()) {
            return Err(InvalidInput("nonfinite Gemma learned image positions").into());
        }
        Ok(Self {
            patch: Linear::new(context, &tree.subtree("patch_embedder.input_proj"), 768, WIDTH, false)?,
            positions,
            blocks: blocks.into_boxed_slice(),
            projector: Linear::new(context, &tree.subtree("embed_vision.embedding_projection"), WIDTH, MODEL, false)?,
            norm: <B::Kernels as Kernels>::Gemma4VisionNormBF16Kernel::new(context)
                .map_err(VisionTowerError::Backend)?,
            prepare: <B::Kernels as Kernels>::Gemma4VisionPrepareBF16Kernel::new(context)
                .map_err(VisionTowerError::Backend)?,
            pool: <B::Kernels as Kernels>::Gemma4VisionPoolBF16Kernel::new(context)
                .map_err(VisionTowerError::Backend)?,
            product: <B::Kernels as Kernels>::Gemma4VisionProductBF16Kernel::new(context)
                .map_err(VisionTowerError::Backend)?,
            add: <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, DataType::BF16, true)
                .map_err(VisionTowerError::Backend)?,
            gelu: <B::Kernels as Kernels>::VisionGeluBF16Kernel::new(context).map_err(VisionTowerError::Backend)?,
            attention: <B::Kernels as Kernels>::AttentionSinglePassKernel::new(
                context,
                DataType::BF16,
                HEAD_DIM,
                false,
                false,
                false,
                false,
                false,
            )
            .map_err(VisionTowerError::Backend)?,
        })
    }

    fn attention(
        &self,
        q: &B::ScratchBuffer,
        k: &B::ScratchBuffer,
        v: &B::ScratchBuffer,
        rows: u32,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut output = commands.allocate_scratch_for_shape(&[rows, WIDTH], DataType::BF16)?;
        // Reuse the native online-softmax kernel: no quadratic score storage
        // and no intermediate BF16 probability materialization.
        self.attention.encode(
            q,
            k,
            v,
            &mut output,
            1,
            rows,
            HEAD_DIM,
            WIDTH,
            HEAD_DIM,
            WIDTH,
            None,
            1.0,
            None::<&B::ScratchBuffer>,
            None,
            None::<&B::ScratchBuffer>,
            HEADS,
            rows,
            commands,
        );
        Ok(output)
    }

    pub(crate) fn encode(
        &self,
        patches: &[bf16],
        grid: [u32; 3],
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, VisionTowerError<B>> {
        let [temporal, height, width] = grid;
        let rows = height
            .checked_mul(width)
            .filter(|rows| *rows > 0 && *rows <= 2520)
            .ok_or(InvalidInput("Gemma image patch budget exceeded"))?;
        if temporal != 1
            || !height.is_multiple_of(3)
            || !width.is_multiple_of(3)
            || height >= POSITIONS
            || width >= POSITIONS
            || patches.len() != (rows * 768) as usize
            || patches.iter().any(|value| !value.is_finite())
        {
            return Err(InvalidInput("invalid Gemma image patches").into());
        }
        let source = commands.allocate_constant_from_slice(patches).map_err(VisionTowerError::Backend)?;
        let mut uploaded =
            commands.allocate_scratch_for_shape(&[rows, 768], DataType::BF16).map_err(VisionTowerError::Backend)?;
        commands.encode_copy(&source, &mut uploaded);
        let mut hidden = self.patch.encode(&uploaded, rows, commands).map_err(VisionTowerError::Backend)?;
        let mut positions = Vec::with_capacity((rows * WIDTH) as usize);
        let mut cos = Vec::with_capacity((rows * HEAD_DIM) as usize);
        let mut sin = Vec::with_capacity((rows * HEAD_DIM) as usize);
        for y in 0..height {
            for x in 0..width {
                for channel in 0..WIDTH {
                    positions.push(bf16::from_f32(
                        self.positions[(x * WIDTH + channel) as usize].to_f32()
                            + self.positions[((POSITIONS + y) * WIDTH + channel) as usize].to_f32(),
                    ));
                }
                for coordinate in [x, y] {
                    for channel in 0..32 {
                        let pair = channel % 16;
                        let angle = coordinate as f32 / 100f32.powf((2 * pair) as f32 / 32.0);
                        cos.push(bf16::from_f32(angle.cos()));
                        sin.push(bf16::from_f32(angle.sin()));
                    }
                }
            }
        }
        let positions = commands.allocate_constant_from_slice(&positions).map_err(VisionTowerError::Backend)?;
        let cos = commands.allocate_constant_from_slice(&cos).map_err(VisionTowerError::Backend)?;
        let sin = commands.allocate_constant_from_slice(&sin).map_err(VisionTowerError::Backend)?;
        self.add.encode(None::<&B::ScratchBuffer>, &positions, &mut hidden, rows * WIDTH, rows * WIDTH, 1.0, commands);
        let one =
            commands.allocate_constant_from_slice(&[bf16::ONE; WIDTH as usize]).map_err(VisionTowerError::Backend)?;
        for block in &self.blocks {
            let normalized = block.input_norm.encode(&hidden, rows, commands).map_err(VisionTowerError::Backend)?;
            let q = block.q.encode(&normalized, rows, commands).map_err(VisionTowerError::Backend)?;
            let k = block.k.encode(&normalized, rows, commands).map_err(VisionTowerError::Backend)?;
            let v = block.v.encode(&normalized, rows, commands).map_err(VisionTowerError::Backend)?;
            let q = block.q_norm.encode(&q, rows * HEADS, commands).map_err(VisionTowerError::Backend)?;
            let k = block.k_norm.encode(&k, rows * HEADS, commands).map_err(VisionTowerError::Backend)?;
            let mut normalized_v = commands
                .allocate_scratch_for_shape(&[rows, WIDTH], DataType::BF16)
                .map_err(VisionTowerError::Backend)?;
            self.norm.encode(&v, &one, &mut normalized_v, rows * HEADS, HEAD_DIM, 1e-6, 0, commands);
            let mut queries = commands
                .allocate_scratch_for_shape(&[HEADS, rows, HEAD_DIM], DataType::BF16)
                .map_err(VisionTowerError::Backend)?;
            let mut keys = commands
                .allocate_scratch_for_shape(&[rows, WIDTH], DataType::BF16)
                .map_err(VisionTowerError::Backend)?;
            let mut values = commands
                .allocate_scratch_for_shape(&[rows, WIDTH], DataType::BF16)
                .map_err(VisionTowerError::Backend)?;
            self.prepare.encode(
                &q,
                &k,
                &normalized_v,
                &mut queries,
                &mut keys,
                &mut values,
                &cos,
                &sin,
                rows,
                HEADS,
                HEAD_DIM,
                commands,
            );
            let attention =
                self.attention(&queries, &keys, &values, rows, commands).map_err(VisionTowerError::Backend)?;
            let projected = block.o.encode(&attention, rows, commands).map_err(VisionTowerError::Backend)?;
            let residual =
                block.post_attention.encode(&projected, rows, commands).map_err(VisionTowerError::Backend)?;
            self.add.encode(
                None::<&B::ScratchBuffer>,
                &residual,
                &mut hidden,
                rows * WIDTH,
                rows * WIDTH,
                1.0,
                commands,
            );
            let normalized = block.pre_ff.encode(&hidden, rows, commands).map_err(VisionTowerError::Backend)?;
            let mut gate = block.gate.encode(&normalized, rows, commands).map_err(VisionTowerError::Backend)?;
            let up = block.up.encode(&normalized, rows, commands).map_err(VisionTowerError::Backend)?;
            self.gelu.encode(&mut gate, rows * MLP, 1, commands);
            let mut product =
                commands.allocate_scratch_for_shape(&[rows, MLP], DataType::BF16).map_err(VisionTowerError::Backend)?;
            self.product.encode(&gate, &up, &mut product, rows * MLP, commands);
            let down = block.down.encode(&product, rows, commands).map_err(VisionTowerError::Backend)?;
            let residual = block.post_ff.encode(&down, rows, commands).map_err(VisionTowerError::Backend)?;
            self.add.encode(
                None::<&B::ScratchBuffer>,
                &residual,
                &mut hidden,
                rows * WIDTH,
                rows * WIDTH,
                1.0,
                commands,
            );
        }
        let output_rows = rows / 9;
        let mut pooled = commands
            .allocate_scratch_for_shape(&[output_rows, WIDTH], DataType::BF16)
            .map_err(VisionTowerError::Backend)?;
        self.pool.encode(&hidden, &mut pooled, width, height, WIDTH, commands);
        let mut normalized = commands
            .allocate_scratch_for_shape(&[output_rows, WIDTH], DataType::BF16)
            .map_err(VisionTowerError::Backend)?;
        self.norm.encode(&pooled, &one, &mut normalized, output_rows, WIDTH, 1e-6, 0, commands);
        self.projector.encode(&normalized, output_rows, commands).map_err(VisionTowerError::Backend)
    }
}
