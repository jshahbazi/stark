//! Native tensor bindings for the pinned Hy4 Sushi STQ artifact.

use std::{io, os::unix::fs::FileExt};

use sha2::{Digest, Sha256};

use super::{
    VOCAB,
    stq_source::{StqDtype, StqSource, StqTensorLocation},
};

const EXPERTS: u32 = 512;
const ROUTED_WIDTH: u32 = 1536;
const ENGRAM_DIM: u32 = 96;
const ENGRAM_TABLES: [(u32, u32); 4] = [(1, 164_994_380), (11, 164_999_008), (21, 165_004_086), (31, 165_008_942)];
const ENGRAM_LOOKUP_LEN: usize = 120_818;
const ENGRAM_LOOKUP_SHA: &str = "87c7b2e1c2fc227ba7fdec4dd7a867533072eb2b73a85cc54189174ccfaf5f3f";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StqFormat {
    Pair1K4,
    Spec1K4,
    Spec1K6,
}

/// The three planes of one projection, bound to descriptors owned by the source.
pub struct StqProjection<'a> {
    pub format: StqFormat,
    pub inputs: u32,
    pub outputs: u32,
    pub banks: u32,
    pub trellis: StqTensorLocation<'a>,
    pub input_scale: StqTensorLocation<'a>,
    pub output_scale: StqTensorLocation<'a>,
}

pub struct StqEngramTable<'a> {
    pub block: u32,
    pub rows: u32,
    pub location: StqTensorLocation<'a>,
}

impl StqSource {
    /// Resolve an STQ projection in place. Its dimensions are the unpadded scale lengths.
    pub fn stq_projection(
        &self,
        name: &str,
    ) -> Option<StqProjection<'_>> {
        let input_name = format!("{name}.suh");
        let output_name = format!("{name}.svh");
        let input = self.tensor(&input_name)?;
        let output = self.tensor(&output_name)?;
        if input.dtype != StqDtype::F16 || output.dtype != StqDtype::F16 {
            return None;
        }
        let (format, inputs, outputs, banks, trellis_name) = match (input.shape.as_slice(), output.shape.as_slice()) {
            ([inputs], [outputs]) if inputs.is_multiple_of(16) && outputs.is_multiple_of(16) => {
                let trellis_name = format!("{name}.trellis");
                let plane = self.tensor(&trellis_name)?;
                let (format, words) = if name.starts_with("model.dfly_layers.") {
                    (StqFormat::Spec1K4, 64)
                } else {
                    (StqFormat::Spec1K6, 96)
                };
                if plane.dtype != StqDtype::U16 || plane.shape != [inputs / 16, outputs / 16, words] {
                    return None;
                }
                (format, *inputs, *outputs, 1, trellis_name)
            },
            ([EXPERTS, ROUTED_WIDTH], [EXPERTS, ROUTED_WIDTH]) => {
                let trellis_name = format!("{name}.trellis_pm");
                let plane = self.tensor(&trellis_name)?;
                if plane.dtype != StqDtype::U16 || plane.shape != [EXPERTS, ROUTED_WIDTH / 32, ROUTED_WIDTH / 16, 2, 64]
                {
                    return None;
                }
                (StqFormat::Pair1K4, ROUTED_WIDTH, ROUTED_WIDTH, EXPERTS, trellis_name)
            },
            _ => return None,
        };
        Some(StqProjection {
            format,
            inputs,
            outputs,
            banks,
            trellis: self.tensor_location(&trellis_name)?,
            input_scale: self.tensor_location(&input_name)?,
            output_scale: self.tensor_location(&output_name)?,
        })
    }

    /// In gate, up, down order to match the native Hy4 MoE dispatcher.
    pub fn stq_routed_experts(
        &self,
        block: u32,
    ) -> Option<[StqProjection<'_>; 3]> {
        let base = format!("model.layers.{block}.mlp.switch_mlp");
        Some([
            self.stq_projection(&format!("{base}.gate_proj.g0"))?,
            self.stq_projection(&format!("{base}.up_proj.g0"))?,
            self.stq_projection(&format!("{base}.down_proj.g0"))?,
        ])
    }

    /// Bind the four local row stores. The dense banks and routed experts have separate residency rules.
    pub fn stq_engram_table(
        &self,
        block: u32,
    ) -> Option<StqEngramTable<'_>> {
        let (_, rows) = ENGRAM_TABLES.into_iter().find(|(layer, _)| *layer == block)?;
        let name = format!("model.layers.{block}.engram_layer.multi_head_embedding.embedding.weight");
        let tensor = self.tensor(&name)?;
        if tensor.dtype != StqDtype::BF16 || tensor.shape != [rows, ENGRAM_DIM] {
            return None;
        }
        Some(StqEngramTable {
            block,
            rows,
            location: self.tensor_location(&name)?,
        })
    }

    /// Read the pinned table's compressed token IDs. Sushi uses the token ID
    /// itself for the final fourteen vocabulary entries beyond the map.
    pub fn stq_engram_token_map(&self) -> io::Result<Vec<i32>> {
        const NAME: &str = "model.layers.1.engram_layer.lookup_table";
        let invalid = || io::Error::new(io::ErrorKind::InvalidData, "invalid pinned STQ Engram token map");
        let info = self.tensor(NAME).ok_or_else(invalid)?;
        let location = self.tensor_location(NAME).ok_or_else(invalid)?;
        if info.dtype != StqDtype::I32
            || info.shape != [ENGRAM_LOOKUP_LEN as u32]
            || location.byte_len != ENGRAM_LOOKUP_LEN * size_of::<i32>()
        {
            return Err(invalid());
        }
        let mut bytes = vec![0; location.byte_len];
        location.file.read_exact_at(&mut bytes, location.offset)?;
        if format!("{:x}", Sha256::digest(&bytes)) != ENGRAM_LOOKUP_SHA {
            return Err(invalid());
        }
        let (words, _) = bytes.as_chunks::<4>();
        Ok(words.iter().copied().map(i32::from_le_bytes).chain(ENGRAM_LOOKUP_LEN as i32..VOCAB as i32).collect())
    }
}
