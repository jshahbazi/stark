// Modified by Stark: in-process core extraction; see UPSTREAM.md.
//! The vision tower's weights (`model.visual.*`), resident bf16 on the GPU
//! exactly as the converter copied them (`docs/qwen38-flash-next-checkpoint-format.md`).
//! Every shape is checked against `VisionConfig`, so a checkpoint whose
//! `config.json` and tensors disagree fails at load. Loading only: the
//! compute is `docs/architecture.md`, "The vision tower", and the layouts here are
//! the ones it will consume (`tools/reference/VISION.md`, "The tower").

use anyhow::{Result, ensure};

use crate::safetensors::SafetensorsDType;
use crate::tensor::{DType, Tensor};
use crate::weights::{Loader, expect_shape};

use super::config::{VISION_PREFIX, VisionConfig};

/// One transformer block: pre-norm attention and pre-norm MLP, both
/// LayerNorm with bias, every projection with bias.
pub struct VisionBlockWeights {
    /// `[hidden]` each.
    pub norm1_w: Tensor,
    pub norm1_b: Tensor,
    /// `[3 * hidden, hidden]`: q | k | v rows.
    pub qkv_w: Tensor,
    /// `[3 * hidden]`.
    pub qkv_b: Tensor,
    /// `[hidden, hidden]`.
    pub proj_w: Tensor,
    pub proj_b: Tensor,
    pub norm2_w: Tensor,
    pub norm2_b: Tensor,
    /// `[intermediate, hidden]`.
    pub fc1_w: Tensor,
    pub fc1_b: Tensor,
    /// `[hidden, intermediate]`.
    pub fc2_w: Tensor,
    pub fc2_b: Tensor,
}

pub struct VisionWeights {
    /// The patch embedding's `Conv3d` weight `[hidden, C, T, P, P]` viewed as
    /// the `[hidden, C * T * P * P]` linear map it is when the kernel takes one
    /// step per patch. Row-major flattening of the trailing dims is the
    /// (C, T, H, W) order of a preprocessed pixel row. Indexed MLX exports
    /// store the axes as `[hidden, T, P, P, C]`; those BF16 encodings are
    /// transposed once at load without modifying the source checkpoint.
    pub patch_proj_w: Tensor,
    /// `[hidden]`.
    pub patch_proj_b: Tensor,
    /// `[num_position_embeddings, hidden]`: the learned table, resampled to
    /// the image grid at run time.
    pub pos_embed: Tensor,
    pub blocks: Vec<VisionBlockWeights>,
    /// `[hidden]` each: the pre-shuffle LayerNorm of the merger.
    pub merger_norm_w: Tensor,
    pub merger_norm_b: Tensor,
    /// `[merge_dim, merge_dim]` with `merge_dim = hidden * merge * merge`.
    pub merger_fc1_w: Tensor,
    pub merger_fc1_b: Tensor,
    /// `[out_hidden, merge_dim]`: lands in the text model's hidden size.
    pub merger_fc2_w: Tensor,
    pub merger_fc2_b: Tensor,
    bytes: usize,
}

impl VisionWeights {
    /// GPU bytes the tower occupies.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Reorder BF16 encodings from `[out, T, H, W, C]` to `[out, C, T, H, W]`.
/// The fixed native image bounds cap each input/output temporary at 12 MiB.
fn patch_channels_first(input: &[u8], shape: [usize; 5]) -> Result<Vec<u8>> {
    let [out, frames, height, width, channels] = shape;
    ensure!(
        shape.iter().all(|dimension| *dimension > 0)
            && out <= 4096
            && frames <= 2
            && height <= 16
            && width <= 16
            && channels <= 3,
        "patch embedding transpose exceeds native image bounds"
    );
    let elements = out * frames * height * width * channels;
    ensure!(input.len() == elements * 2, "invalid BF16 patch embedding payload length");
    let mut output = vec![0u8; input.len()];
    for o in 0..out {
        for c in 0..channels {
            for t in 0..frames {
                for y in 0..height {
                    for x in 0..width {
                        let source = ((((o * frames + t) * height + y) * width + x)
                            * channels
                            + c)
                            * 2;
                        let target = ((((o * channels + c) * frames + t) * height + y)
                            * width
                            + x)
                            * 2;
                        output[target..target + 2]
                            .copy_from_slice(&input[source..source + 2]);
                    }
                }
            }
        }
    }
    Ok(output)
}

/// Reads every tower tensor into GPU memory, checking dtype and shape.
pub(crate) fn load(
    loader: &Loader<'_>,
    v: &VisionConfig,
    indexed_mlx: bool,
) -> Result<VisionWeights> {
    let h = v.hidden_size;
    let bytes = std::cell::Cell::new(0usize);
    let tensor = |name: &str, shape: &[usize]| -> Result<Tensor> {
        let full = format!("{VISION_PREFIX}{name}");
        let t = loader.tensor(&full)?;
        ensure!(
            t.dtype() == DType::BF16,
            "{full} is {:?}, expected BF16 (the tower is stored unquantized)",
            t.dtype()
        );
        expect_shape(&t, shape, &full)?;
        bytes.set(bytes.get() + t.byte_len());
        Ok(t)
    };

    let patch_name = format!("{VISION_PREFIX}patch_embed.proj.weight");
    let channels_last =
        [h, v.temporal_patch_size, v.patch_size, v.patch_size, v.in_channels];
    let canonical =
        [h, v.in_channels, v.temporal_patch_size, v.patch_size, v.patch_size];
    let conv = if indexed_mlx
        && loader
            .checkpoint()
            .meta(&patch_name)
            .is_some_and(|meta| meta.shape == channels_last)
    {
        ensure!(
            loader
                .checkpoint()
                .meta(&patch_name)
                .is_some_and(|meta| meta.dtype == SafetensorsDType::BF16),
            "indexed MLX patch embedding requires BF16 storage"
        );
        // Read the small patch projection before allocating its final GPU
        // buffer. No redundant source GPU allocation survives model loading.
        let source = loader.checkpoint().read(&patch_name)?;
        let reordered = patch_channels_first(&source, channels_last)?;
        let result =
            Tensor::from_bytes(loader.context(), &reordered, &canonical, DType::BF16)?;
        loader.mark_consumed(&patch_name);
        bytes.set(bytes.get() + result.byte_len());
        result
    } else {
        tensor("patch_embed.proj.weight", &canonical)?
    };
    let patch_proj_w = conv.view(0, &[h, v.patch_dim()])?;
    let patch_proj_b = tensor("patch_embed.proj.bias", &[h])?;
    let pos_embed = tensor("pos_embed.weight", &[v.num_position_embeddings, h])?;

    let mut blocks = Vec::with_capacity(v.depth);
    for i in 0..v.depth {
        let p = format!("blocks.{i}.");
        blocks.push(VisionBlockWeights {
            norm1_w: tensor(&format!("{p}norm1.weight"), &[h])?,
            norm1_b: tensor(&format!("{p}norm1.bias"), &[h])?,
            qkv_w: tensor(&format!("{p}attn.qkv.weight"), &[3 * h, h])?,
            qkv_b: tensor(&format!("{p}attn.qkv.bias"), &[3 * h])?,
            proj_w: tensor(&format!("{p}attn.proj.weight"), &[h, h])?,
            proj_b: tensor(&format!("{p}attn.proj.bias"), &[h])?,
            norm2_w: tensor(&format!("{p}norm2.weight"), &[h])?,
            norm2_b: tensor(&format!("{p}norm2.bias"), &[h])?,
            fc1_w: tensor(
                &format!("{p}mlp.linear_fc1.weight"),
                &[v.intermediate_size, h],
            )?,
            fc1_b: tensor(&format!("{p}mlp.linear_fc1.bias"), &[v.intermediate_size])?,
            fc2_w: tensor(
                &format!("{p}mlp.linear_fc2.weight"),
                &[h, v.intermediate_size],
            )?,
            fc2_b: tensor(&format!("{p}mlp.linear_fc2.bias"), &[h])?,
        });
    }

    let m = v.merge_dim();
    let merger_norm_w = tensor("merger.norm.weight", &[h])?;
    let merger_norm_b = tensor("merger.norm.bias", &[h])?;
    let merger_fc1_w = tensor("merger.linear_fc1.weight", &[m, m])?;
    let merger_fc1_b = tensor("merger.linear_fc1.bias", &[m])?;
    let merger_fc2_w = tensor("merger.linear_fc2.weight", &[v.out_hidden_size, m])?;
    let merger_fc2_b = tensor("merger.linear_fc2.bias", &[v.out_hidden_size])?;

    Ok(VisionWeights {
        patch_proj_w,
        patch_proj_b,
        pos_embed,
        blocks,
        merger_norm_w,
        merger_norm_b,
        merger_fc1_w,
        merger_fc1_b,
        merger_fc2_w,
        merger_fc2_b,
        bytes: bytes.get(),
    })
}

#[cfg(test)]
mod tests {
    use super::patch_channels_first;

    #[test]
    fn indexed_patch_transpose_preserves_bf16_encodings_and_pixel_axis_order() {
        let [out, frames, height, width, channels] = [2, 2, 2, 2, 3];
        let expected: Vec<u16> = (0..out * frames * height * width * channels)
            .map(|index| match index {
                0 => 0,
                1 => 0x8000,
                2 => 0x7f80,
                3 => 0xff80,
                4 => 0x7fc1,
                _ => 0xa000 + index as u16,
            })
            .collect();
        // Construct source bytes from semantic (out, channel, frame, y, x)
        // coordinates, using the export's innermost channel axis.
        let mut source = Vec::new();
        for o in 0..out {
            for t in 0..frames {
                for y in 0..height {
                    for x in 0..width {
                        for c in 0..channels {
                            source.extend_from_slice(
                                &expected[((((o * channels + c) * frames + t)
                                    * height
                                    + y)
                                    * width)
                                    + x]
                                    .to_le_bytes(),
                            );
                        }
                    }
                }
            }
        }
        let original = source.clone();
        let actual =
            patch_channels_first(&source, [out, frames, height, width, channels])
                .unwrap();
        let expected: Vec<u8> =
            expected.iter().flat_map(|encoding| encoding.to_le_bytes()).collect();
        assert_eq!(actual, expected);
        assert_eq!(source, original, "source checkpoint bytes changed");
    }

    #[test]
    fn patch_transpose_rejects_invalid_payloads_and_unbounded_dimensions() {
        assert!(patch_channels_first(&[0; 4], [1, 1, 1, 1, 3]).is_err());
        assert!(patch_channels_first(&[], [usize::MAX, 2, 16, 16, 3]).is_err());
        assert!(patch_channels_first(&[], [1, 0, 16, 16, 3]).is_err());
    }
}
