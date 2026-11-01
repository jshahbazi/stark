// SPDX-License-Identifier: Apache-2.0
//! Explicit in-memory numerical diagnostics. No file or environment interface.

use std::cell::RefCell;

use anyhow::{Result, ensure};
use serde::Serialize;

use crate::{
    kernels::elementwise::copy_words,
    metal::{ComputePass, MetalContext},
    tensor::Tensor,
};

struct Capture {
    layer: usize,
    stage: &'static str,
    rows: usize,
    tensor: Tensor,
}

/// Opaque GPU copies, retained only by a caller's explicit diagnostic scope.
pub struct NumericalTrace {
    layers: usize,
    captures: RefCell<Vec<Capture>>,
}

#[derive(Debug, Serialize)]
pub struct NumericalDifference {
    pub layer: usize,
    pub stage: &'static str,
    pub values: usize,
    pub unequal_values: usize,
    pub max_absolute_error: f32,
    pub reference_fnv1a64: String,
    pub candidate_fnv1a64: String,
}

fn hash(bytes: &[u8]) -> String {
    let value = bytes.iter().fold(0xcbf29ce484222325u64, |h, &byte| {
        (h ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    });
    format!("{value:016x}")
}

impl NumericalTrace {
    pub(super) fn new(layers: usize) -> Self {
        Self { layers, captures: RefCell::new(Vec::new()) }
    }

    pub(super) fn capture(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        layer: usize,
        stage: &'static str,
        tensor: &Tensor,
        rows: usize,
    ) -> Result<()> {
        if layer >= self.layers {
            return Ok(());
        }
        ensure!(rows > 0 && tensor.numel().is_multiple_of(rows), "trace row geometry");
        let mut captures = self.captures.borrow_mut();
        ensure!(captures.len() < 2048, "trace capacity");
        let copy = Tensor::zeros(ctx, tensor.shape(), tensor.dtype())?;
        copy_words(ctx, pass, tensor, &copy)?;
        // Every producer is already ordered by its normal graph barrier;
        // order the copy's read before the normal graph reuses this buffer.
        pass.level_barrier(&[tensor, &copy])?;
        captures.push(Capture { layer, stage, rows, tensor: copy });
        Ok(())
    }

    /// Both traces' GPU work must have completed. Only aggregate differences
    /// and explicitly named non-cryptographic byte hashes leave this scope.
    pub fn compare(
        &self,
        candidate: &Self,
        candidate_row: usize,
    ) -> Result<Vec<NumericalDifference>> {
        let reference = self.captures.borrow();
        let candidate = candidate.captures.borrow();
        let mut differences = Vec::new();
        for source in reference.iter() {
            let other = candidate
                .iter()
                .find(|x| x.layer == source.layer && x.stage == source.stage)
                .ok_or_else(|| anyhow::anyhow!("trace stage missing"))?;
            ensure!(
                source.rows == 1 && candidate_row < other.rows,
                "trace comparison row"
            );
            let values = source.tensor.numel();
            ensure!(
                values * other.rows == other.tensor.numel()
                    && source.tensor.dtype() == other.tensor.dtype(),
                "trace comparison geometry"
            );
            let left = source.tensor.to_f32()?;
            let row = other.tensor.view(candidate_row * values, &[values])?;
            let right = row.to_f32()?;
            let unequal_values = left
                .iter()
                .zip(&right)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let max_absolute_error = left
                .iter()
                .zip(&right)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0, f32::max);
            differences.push(NumericalDifference {
                layer: source.layer,
                stage: source.stage,
                values,
                unequal_values,
                max_absolute_error,
                reference_fnv1a64: hash(source.tensor.contents()),
                candidate_fnv1a64: hash(row.contents()),
            });
        }
        Ok(differences)
    }
}
