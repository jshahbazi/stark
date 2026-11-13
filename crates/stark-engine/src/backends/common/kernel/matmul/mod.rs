mod arguments;
mod d_ops;
mod error;
mod kernel;
mod matmul_a;
mod matmul_b;
pub mod routing;

pub use arguments::MatmulArguments;
pub use d_ops::MatmulDOps;
pub use error::MatmulError;
pub use kernel::MatmulKernel;
pub use matmul_a::{Int8CodeLayout, MatmulA};
pub use matmul_b::{MatmulB, QuantizedB, QuantizedCorrection};
pub use routing::{ActivationFormat, MatmulShape};

use crate::backends::common::gpu_types::{QUANT_PARAMS_GROUP_OUTPUT_ALIGNMENT, QuantizationMode};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantParamsLayout {
    /// `[N, G]`.
    OutputGroup,
    /// `[G, round_up_4(N)]`.
    GroupOutput,
}

#[derive(Clone, Copy, Default)]
pub struct QuantParamsStrides {
    pub output_stride: u32,
    pub group_stride: u32,
}

#[derive(Clone, Copy)]
pub struct QuantParams {
    layout: QuantParamsLayout,
    row_count: u32,
    group_count: u32,
    materialize_affine_input_sum: bool,
    source_output_widths: [u32; 4],
    source_output_count: u8,
}

impl QuantParams {
    pub const fn new(
        layout: QuantParamsLayout,
        row_count: u32,
        group_count: u32,
    ) -> Self {
        Self {
            layout,
            row_count,
            group_count,
            materialize_affine_input_sum: false,
            source_output_widths: [0; 4],
            source_output_count: 0,
        }
    }

    pub const fn with_materialized_affine_input_sum(
        mut self,
        enabled: bool,
    ) -> Self {
        self.materialize_affine_input_sum = enabled;
        self
    }

    pub const fn materializes_affine_input_sum(self) -> bool {
        self.materialize_affine_input_sum
    }

    /// Original linear output widths represented by consecutive row spans.
    /// An empty slice means one source projection with this matrix's full N.
    /// This describes concatenated rows, never alternating-row interleaving.
    pub fn source_output_widths(&self) -> &[u32] {
        &self.source_output_widths[..self.source_output_count as usize]
    }

    pub fn with_source_output_widths(
        mut self,
        widths: &[u32],
    ) -> Result<Self, &'static str> {
        if widths.is_empty() || widths.len() > self.source_output_widths.len() || widths.contains(&0) {
            return Err("source output widths require one to four nonzero contiguous spans");
        }
        if widths.iter().try_fold(0u32, |sum, width| sum.checked_add(*width)) != Some(self.row_count) {
            return Err("source output widths must exactly cover the weight matrix rows");
        }
        self.source_output_widths.fill(0);
        self.source_output_widths[..widths.len()].copy_from_slice(widths);
        self.source_output_count = widths.len() as u8;
        Ok(self)
    }

    pub const fn layout(self) -> QuantParamsLayout {
        self.layout
    }

    pub const fn scale_shape(self) -> [u32; 2] {
        self.shape(1)
    }

    pub fn zero_point_shape(
        self,
        mode: QuantizationMode,
    ) -> [u32; 2] {
        self.shape(mode.packing_divisor())
    }

    pub const fn scale_strides(self) -> QuantParamsStrides {
        self.strides(1)
    }

    pub fn zero_point_strides(
        self,
        mode: QuantizationMode,
    ) -> QuantParamsStrides {
        self.strides(mode.packing_divisor())
    }

    const fn shape(
        self,
        packing_divisor: u32,
    ) -> [u32; 2] {
        let row_values = self.padded_values_per_row(packing_divisor);
        match self.layout {
            QuantParamsLayout::OutputGroup => [self.row_count, row_values / packing_divisor],
            QuantParamsLayout::GroupOutput => [self.group_count, row_values / packing_divisor],
        }
    }

    const fn strides(
        self,
        packing_divisor: u32,
    ) -> QuantParamsStrides {
        let row_values = self.padded_values_per_row(packing_divisor);
        match self.layout {
            QuantParamsLayout::OutputGroup => QuantParamsStrides {
                output_stride: row_values,
                group_stride: 1,
            },
            QuantParamsLayout::GroupOutput => QuantParamsStrides {
                output_stride: 1,
                group_stride: row_values,
            },
        }
    }

    const fn padded_values_per_row(
        self,
        packing_divisor: u32,
    ) -> u32 {
        match self.layout {
            QuantParamsLayout::OutputGroup => self.group_count.next_multiple_of(packing_divisor),
            QuantParamsLayout::GroupOutput => self.row_count.next_multiple_of(QUANT_PARAMS_GROUP_OUTPUT_ALIGNMENT),
        }
    }
}
