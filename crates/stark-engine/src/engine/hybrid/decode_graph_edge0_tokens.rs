// SPDX-License-Identifier: Apache-2.0
//! Checked bindings for Edge0 token-batched rows. One dispatch decodes each
//! weight of a packed matrix once for up to eight token rows. Every token
//! keeps the one-token kernel's F32 reduction, so the written bits match
//! `edge0_packed_single_into` and `edge0_gate_up_swiglu_single`.

use super::*;

#[cfg(test)]
#[path = "decode_graph_edge0_tokens_tests.rs"]
mod tests;

const TOKENS: usize = 8;

macro_rules! token_kernels {
    ($prefix:literal) => {
        [
            concat!($prefix, "_2"),
            concat!($prefix, "_3"),
            concat!($prefix, "_4"),
            concat!($prefix, "_5"),
            concat!($prefix, "_6"),
            concat!($prefix, "_7"),
            concat!($prefix, "_8"),
        ]
    };
}

const Q8: [&str; 7] = token_kernels!("edge0_tokens_q8");
const Q3: [&str; 7] = token_kernels!("edge0_tokens_q3");
const IQ4: [&str; 7] = token_kernels!("edge0_tokens_iq4");
const Q8_SCALED: [&str; 7] = token_kernels!("edge0_tokens_q8_scaled");
const GATE_UP: [&str; 7] = token_kernels!("edge0_tokens_gate_up");
const F32: [&str; 7] = token_kernels!("edge0_tokens_f32");
const F16: [&str; 7] = token_kernels!("edge0_tokens_f16");
const BF16: [&str; 7] = token_kernels!("edge0_tokens_bf16");

/// Float formats have no older one-token Edge0 kernel, so theirs is the
/// one-token instance of the same template.
fn float_single_kernel(kind: u32) -> Option<&'static str> {
    match kind {
        0 => Some("edge0_tokens_f32_1"),
        1 => Some("edge0_tokens_f16_1"),
        30 => Some("edge0_tokens_bf16_1"),
        _ => None,
    }
}

fn packed_kernel(
    kind: u32,
    scaled: bool,
    tokens: usize,
) -> Option<&'static str> {
    if tokens == 1 && !scaled {
        return float_single_kernel(kind);
    }
    let names = match (kind, scaled) {
        (8, false) => &Q8,
        (11, false) => &Q3,
        (20, false) => &IQ4,
        (8, true) => &Q8_SCALED,
        (0, false) => &F32,
        (1, false) => &F16,
        (30, false) => &BF16,
        _ => return None,
    };
    names.get(tokens.checked_sub(2)?).copied()
}

/// Element offsets of whole rows inside one parent view.
fn row_offsets(
    parent: &DeviceVector,
    width: usize,
    rows: &[usize],
) -> Result<Vec<u32>> {
    parent.validate()?;
    if width == 0 || !parent.len.is_multiple_of(width) {
        return Err(Error::Invalid("Edge0 token row parent shape"));
    }
    rows.iter()
        .map(|&row| {
            if row >= parent.len / width {
                return Err(Error::Invalid("Edge0 token row index"));
            }
            u32::try_from(row * width).map_err(|_| Error::Invalid("Edge0 token row offset"))
        })
        .collect()
}

fn distinct(rows: &[usize]) -> bool {
    let mut sorted = rows.to_vec();
    sorted.sort_unstable();
    sorted.windows(2).all(|pair| pair[0] != pair[1])
}

impl DecodeGraph {
    /// Token-batched schedule for a contiguous token batch, with one scan of
    /// the caller-supplied rows. Formats without a format-constant kernel
    /// keep the original batched kernel.
    pub fn edge0_packed_tokens(
        &mut self,
        matrix: &MappedMatrix,
        format: MatrixFormat,
        rows: usize,
        columns: usize,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        let MatrixFormat::Packed(8 | 11 | 20) = format else {
            return self.edge0_packed_batch_selected(matrix, format, rows, columns, input, tokens, true);
        };
        geometry(rows, columns, tokens, input)?;
        let output = self.scratch(rows * tokens)?;
        self.check_finite(input)?;
        let ordered = (0..tokens).collect::<Vec<_>>();
        self.edge0_packed_tokens_into(matrix, format, rows, columns, input, &ordered, &output, &ordered, None)?;
        Ok(output)
    }

    /// Token-batched form of the generic one-token projection (`mapped` over
    /// one packed matrix) for a contiguous token batch. Each token keeps that
    /// projection's F32 reduction. Results are checked in the kernel, which
    /// replaces the generic path's separate input and output scans.
    pub fn edge0_matvec_tokens(
        &mut self,
        matrix: &MappedMatrix,
        format: MatrixFormat,
        rows: usize,
        columns: usize,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        geometry(rows, columns, tokens, input)?;
        let output = self.scratch(rows * tokens)?;
        let ordered = (0..tokens).collect::<Vec<_>>();
        self.edge0_packed_tokens_into(matrix, format, rows, columns, input, &ordered, &output, &ordered, None)?;
        Ok(output)
    }

    /// Token rows of one packed matrix. Row `i` reads input row
    /// `input_rows[i]` of width `columns` and writes output row
    /// `output_rows[i]` of width `rows`. A stored scale selects the fused
    /// column-scale quotient. Rows are not scanned first: every written value
    /// is checked in the kernel, and a nonfinite input makes every result of
    /// its row nonfinite. More than eight rows encode several dispatches.
    #[expect(clippy::too_many_arguments, reason = "explicit checked operands for token rows of one matrix")]
    pub fn edge0_packed_tokens_into(
        &mut self,
        matrix: &MappedMatrix,
        format: MatrixFormat,
        rows: usize,
        columns: usize,
        input: &DeviceVector,
        input_rows: &[usize],
        output: &DeviceVector,
        output_rows: &[usize],
        scale: Option<&DeviceVector>,
    ) -> Result<()> {
        let MatrixFormat::Packed(kind @ (0 | 1 | 8 | 11 | 20 | 30)) = format else {
            return Err(Error::Unsupported("Edge0 token row format"));
        };
        if scale.is_some() && !matches!(kind, 8 | 11 | 20) {
            return Err(Error::Unsupported("Edge0 scaled token row format"));
        }
        if rows == 0
            || rows > 1_000_000
            || columns == 0
            || columns > 65_536
            || input_rows.is_empty()
            || input_rows.len() != output_rows.len()
            || !distinct(output_rows)
            || overlaps(input, output)
            || scale.is_some_and(|scale| scale.len != rows)
        {
            return Err(Error::Invalid("Edge0 token row shape"));
        }
        let sources = row_offsets(input, columns, input_rows)?;
        let targets = row_offsets(output, rows, output_rows)?;
        let (block, bytes) = weights::block_layout(kind)?;
        if !columns.is_multiple_of(block) {
            return Err(Error::Invalid("Edge0 packed batch block alignment"));
        }
        let length = rows
            .checked_mul(columns / block)
            .and_then(|n| n.checked_mul(bytes))
            .ok_or(Error::Invalid("Edge0 packed batch byte overflow"))?;
        codes(matrix, length)?;
        if let Some(scale) = scale {
            scale.validate()?;
        }
        for (sources, targets) in sources.chunks(TOKENS).zip(targets.chunks(TOKENS)) {
            let Some(name) = packed_kernel(kind, scale.is_some(), sources.len()) else {
                for (&source, &target) in sources.iter().zip(targets) {
                    self.edge0_packed_single_into(
                        matrix,
                        format,
                        rows,
                        columns,
                        &input.slice(source as usize, columns)?,
                        &output.slice(target as usize, rows)?,
                        scale,
                        false,
                    )?;
                }
                continue;
            };
            let mut parameters = vec![rows as u32, columns as u32];
            parameters.extend_from_slice(sources);
            parameters.extend_from_slice(targets);
            if let Some(scale) = scale {
                self.edge0_dispatch(name, &[matrix], &[input, output, scale], &parameters, rows, false)?;
            } else {
                self.edge0_dispatch(name, &[matrix], &[input, output], &parameters, rows, false)?;
            }
        }
        Ok(())
    }

    /// Fused Q3_K gate/up rows of one expert for the given input rows. The
    /// result holds one row of width `rows` per input row, in order.
    pub fn edge0_gate_up_swiglu_tokens(
        &mut self,
        gate: &MappedMatrix,
        up: &MappedMatrix,
        rows: usize,
        columns: usize,
        input: &DeviceVector,
        input_rows: &[usize],
    ) -> Result<DeviceVector> {
        if rows == 0
            || rows > 1_000_000
            || columns == 0
            || columns > 65_536
            || !columns.is_multiple_of(256)
            || input_rows.is_empty()
            || rows.checked_mul(input_rows.len()).is_none_or(|count| count > 8_000_000)
        {
            return Err(Error::Invalid("Edge0 fused token row shape"));
        }
        let sources = row_offsets(input, columns, input_rows)?;
        let bytes = rows
            .checked_mul(columns / 256)
            .and_then(|n| n.checked_mul(110))
            .ok_or(Error::Invalid("Edge0 fused Q3_K byte overflow"))?;
        codes(gate, bytes)?;
        codes(up, bytes)?;
        let output = self.scratch(rows * input_rows.len())?;
        for (chunk, sources) in sources.chunks(TOKENS).enumerate() {
            let first = chunk * TOKENS;
            if let Some(&name) = sources.len().checked_sub(2).and_then(|index| GATE_UP.get(index)) {
                let mut parameters = vec![rows as u32, columns as u32];
                parameters.extend_from_slice(sources);
                parameters.extend((0..sources.len()).map(|index| ((first + index) * rows) as u32));
                self.edge0_dispatch(name, &[gate, up], &[input, &output], &parameters, rows, false)?;
            } else {
                self.edge0_dispatch(
                    "edge0_graph_gate_up_swiglu_single",
                    &[gate, up],
                    &[&input.slice(sources[0] as usize, columns)?, &output.slice(first * rows, rows)?],
                    &[rows as u32, columns as u32, 1],
                    rows,
                    false,
                )?;
            }
        }
        Ok(output)
    }
}
