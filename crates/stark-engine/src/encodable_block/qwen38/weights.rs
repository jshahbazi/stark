//! Native Uzu parameter construction for the typed Qwen3.8 artifact.
//! Codes and GroupOutput affine planes are consumed in their stored layout.

use std::collections::BTreeMap;

use half::bf16;

use super::Qwen38Error;
use crate::{
    backends::common::{Backend, BufferMut, Context},
    config::{
        model::qwen38_language_model::{
            Qwen38DecoderConfig, Qwen38TensorBinding, Qwen38TensorDType, Qwen38TensorLayout,
        },
        weight_matrix::{AnyWeightMatrixSpec, Layout},
    },
    data_type::DataType,
    encodable_block::{embedding_table::EmbeddingTable, linear::LinearMatmul, weight_matrix::WeightMatrix},
    parameters::{ParameterLoader, ValidatedFileRegion},
};

#[path = "admission.rs"]
mod admission;

#[cfg(test)]
pub(crate) use admission::tests::Fixture as TestFixture;

#[cfg(test)]
#[path = "expert_gather_tests.rs"]
mod expert_gather_tests;

#[cfg(test)]
#[path = "router_tests.rs"]
mod router_tests;

/// Sparse InputOutput embedding storage; each selected row is contiguous in
/// all three planes. Regions own the file descriptor beyond loader lifetime.
pub(crate) struct PagedMlxEmbedding {
    pub rows: u32,
    pub columns: u32,
    pub bits: u32,
    pub group_size: u32,
    pub code_row_bytes: usize,
    pub parameter_row_bytes: usize,
    pub codes: ValidatedFileRegion,
    pub scales: ValidatedFileRegion,
    pub biases: ValidatedFileRegion,
}

pub(crate) struct Qwen38WeightLoader<'a, B: Backend> {
    context: &'a B::Context,
    config: &'a Qwen38DecoderConfig,
    loader: &'a ParameterLoader<'a, B>,
    matrices: BTreeMap<String, admission::Matrix>,
}

impl<'a, B: Backend> Qwen38WeightLoader<'a, B> {
    pub(crate) fn new(
        context: &'a B::Context,
        config: &'a Qwen38DecoderConfig,
        loader: &'a ParameterLoader<'a, B>,
    ) -> Result<Self, Qwen38Error<B>> {
        crate::config::model::qwen38_language_model::validate_qwen38_architecture(&config.architecture)
            .map_err(|_| Qwen38Error::Invalid("invalid Qwen3.8 architecture"))?;
        let matrices = admission::admit(loader, config)?;
        Ok(Self {
            context,
            config,
            loader,
            matrices,
        })
    }

    pub(crate) fn binding(
        &self,
        name: &str,
    ) -> Result<&Qwen38TensorBinding, Qwen38Error<B>> {
        self.config.tensor_bindings.0.get(name).ok_or(Qwen38Error::Invalid("missing Qwen3.8 tensor binding"))
    }

    fn matrix(
        &self,
        name: &str,
        rows: u32,
        columns: u32,
        embedding: bool,
    ) -> Result<&admission::Matrix, Qwen38Error<B>> {
        let matrix = self.matrices.get(name).ok_or(Qwen38Error::Invalid("missing Qwen3.8 matrix role"))?;
        if (matrix.rows, matrix.columns, matrix.embedding) != (rows, columns, embedding) {
            return Err(Qwen38Error::Invalid("Qwen3.8 matrix role geometry mismatch"));
        }
        Ok(matrix)
    }

    pub(crate) fn linear(
        &self,
        name: &str,
        input_dim: u32,
        output_dim: u32,
        input_dtype: DataType,
        output_dtype: DataType,
    ) -> Result<LinearMatmul<B>, Qwen38Error<B>> {
        let matrix = self.matrix(name, output_dim, input_dim, false)?;
        let tree = self.loader.tree().subtree(&matrix.physical);
        let spec = tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        Ok(LinearMatmul::load(
            self.context,
            spec,
            input_dim,
            output_dim,
            DataType::BF16,
            input_dtype,
            output_dtype,
            &tree,
            None,
            None,
        )?)
    }

    pub(crate) fn bank(
        &self,
        name: &str,
        experts: u32,
        output_dim: u32,
        input_dim: u32,
    ) -> Result<WeightMatrix<B>, Qwen38Error<B>> {
        let rows = experts.checked_mul(output_dim).ok_or(Qwen38Error::Invalid("Qwen3.8 expert dimension overflow"))?;
        let matrix = self.matrix(name, rows, input_dim, false)?;
        let binding = self.binding(&format!("{name}.weight"))?;
        if binding.shape.len() != 3 || binding.shape[0] != experts || binding.shape[1] != output_dim {
            return Err(Qwen38Error::Invalid("Qwen3.8 expert bank rank mismatch"));
        }
        let tree = self.loader.tree().subtree(&matrix.physical);
        let spec = tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        Ok(WeightMatrix::load(&tree, spec, Layout::OutputInput, rows, input_dim, DataType::BF16)?)
    }

    pub(crate) fn embedding(
        &self,
        name: &str,
        rows: u32,
        columns: u32,
    ) -> Result<EmbeddingTable<B>, Qwen38Error<B>> {
        let matrix = self.matrix(name, rows, columns, true)?;
        if matrix.ngram {
            return Err(Qwen38Error::Invalid("paged Qwen3.8 table requires file regions"));
        }
        Ok(EmbeddingTable::load(
            self.context,
            &self.loader.tree().subtree(&matrix.physical),
            rows,
            columns,
            DataType::BF16,
        )?)
    }

    /// Preserve all F32 values. BF16 controls are promoted exactly, never
    /// rounded to a narrower type; the artifact's bytes remain unchanged.
    pub(crate) fn control_f32(
        &self,
        name: &str,
        shape: &[u32],
    ) -> Result<B::GlobalBuffer, Qwen38Error<B>> {
        let binding = self.binding(name)?;
        if binding.shape.as_ref() != shape
            || binding.layout != Qwen38TensorLayout::Direct
            || name.ends_with(".scales")
            || name.ends_with(".biases")
        {
            return Err(Qwen38Error::Invalid("Qwen3.8 control role or shape mismatch"));
        }
        let count = shape
            .iter()
            .try_fold(1_usize, |total, &dim| total.checked_mul(dim as usize))
            .ok_or(Qwen38Error::Invalid("Qwen3.8 control size overflow"))?;
        if count > 64 * 1024 * 1024 {
            return Err(Qwen38Error::Invalid("Qwen3.8 control exceeds load budget"));
        }
        let tree = self.loader.tree();
        let leaf = tree.leaf(&binding.name)?;
        match binding.dtype {
            Qwen38TensorDType::F32 => Ok(leaf.validate(shape, DataType::F32)?.read_buffer()?),
            Qwen38TensorDType::BF16 => {
                let values = leaf.validate(shape, DataType::BF16)?.read_slice::<bf16>()?;
                let byte_len =
                    values.len().checked_mul(4).ok_or(Qwen38Error::Invalid("Qwen3.8 control size overflow"))?;
                let mut output = self.context.create_buffer(byte_len).map_err(Qwen38Error::Backend)?;
                for (destination, value) in output.as_slice_mut::<f32>().iter_mut().zip(values.iter()) {
                    *destination = value.to_f32();
                }
                Ok(output)
            },
            _ => Err(Qwen38Error::Invalid("Qwen3.8 floating control has integer storage")),
        }
    }

    pub(crate) fn paged_embedding(
        &self,
        name: &str,
        rows: u32,
        columns: u32,
    ) -> Result<PagedMlxEmbedding, Qwen38Error<B>> {
        let matrix = self.matrix(name, rows, columns, true)?;
        if !matrix.ngram {
            return Err(Qwen38Error::Invalid("Qwen3.8 paged table role mismatch"));
        }
        let code_columns =
            columns.checked_mul(matrix.bits).ok_or(Qwen38Error::Invalid("Qwen3.8 row size overflow"))? / 8;
        let groups = columns / matrix.group_size;
        let tree = self.loader.tree().subtree(&matrix.physical);
        let codes = tree.leaf("weights")?.validate(&[rows, code_columns], DataType::U8)?.file_region()?;
        let scales = tree.leaf("scales")?.validate(&[rows, groups], DataType::BF16)?.file_region()?;
        let biases = tree.leaf("biases")?.validate(&[rows, groups], DataType::BF16)?.file_region()?;
        let code_row_bytes = code_columns as usize;
        let parameter_row_bytes = groups as usize * 2;
        // The PLE consumer computes offsets from these row strides after this
        // loader has gone away. Check the owned spans against that contract
        // before granting sparse access, including their element byte widths.
        for (region, row_columns, dtype, row_bytes) in [
            (&codes, code_columns, DataType::U8, code_row_bytes),
            (&scales, groups, DataType::BF16, parameter_row_bytes),
            (&biases, groups, DataType::BF16, parameter_row_bytes),
        ] {
            let expected_bytes = u64::from(rows)
                .checked_mul(row_bytes as u64)
                .ok_or(Qwen38Error::Invalid("Qwen3.8 paged plane size overflow"))?;
            if region.is_empty()
                || region.shape() != [rows, row_columns]
                || region.data_type() != dtype
                || region.len() != expected_bytes
            {
                return Err(Qwen38Error::Invalid("Qwen3.8 paged plane row layout mismatch"));
            }
        }
        Ok(PagedMlxEmbedding {
            rows,
            columns,
            bits: matrix.bits,
            group_size: matrix.group_size,
            code_row_bytes,
            parameter_row_bytes,
            codes,
            scales,
            biases,
        })
    }

    pub(crate) fn finish(&self) -> Result<(), Qwen38Error<B>> {
        self.loader.tree().assert_all_tensors_validated()?;
        Ok(())
    }
}
