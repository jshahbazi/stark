use std::mem::size_of;

use super::{RadixTopKSmallCollectMetalKernel, RadixTopKSmallPassMetalKernel};
use crate::{
    backends::{
        common::{
            BufferMut, BufferRef, CommandBufferEncoding, InvalidInput,
            kernel::radix_top_k_small::{RadixTopKSmall, validate_buffers, validate_input_type},
        },
        metal::{Metal, command_buffer::MetalCommandBufferEncoding, context::MetalContext, error::MetalError},
    },
    data_type::DataType,
};

const DEFAULT_PARTITIONS: u32 = 4;
const LARGE_COLUMNS_MIN: u32 = 131_072;
const RADIX_BITS: u32 = 10;
const RADIX_BUCKETS: usize = 1 << RADIX_BITS;

const fn partitions(columns: u32) -> u32 {
    if columns >= LARGE_COLUMNS_MIN {
        8
    } else {
        DEFAULT_PARTITIONS
    }
}

pub struct MetalRadixTopKSmall {
    pass: RadixTopKSmallPassMetalKernel,
    collect: RadixTopKSmallCollectMetalKernel,
    columns: u32,
    partitions: u32,
    passes: u32,
    input_data_type: DataType,
}

impl RadixTopKSmall for MetalRadixTopKSmall {
    type Backend = Metal;

    fn new_with_data_type(
        context: &MetalContext,
        columns: u32,
        input_data_type: DataType,
    ) -> Result<Self, MetalError> {
        validate_input_type(columns, input_data_type)?;
        let index_bits = if columns <= 1 {
            1
        } else {
            u32::BITS - (columns - 1).leading_zeros()
        };
        let partitions = partitions(columns);
        if columns.checked_mul(partitions).is_none() {
            return Err(InvalidInput("small radix partition indexing overflows").into());
        }
        Ok(Self {
            pass: RadixTopKSmallPassMetalKernel::new(context, input_data_type, columns, partitions)?,
            collect: RadixTopKSmallCollectMetalKernel::new(context, input_data_type, columns, partitions)?,
            columns,
            partitions,
            passes: (u32::BITS + index_bits).div_ceil(RADIX_BITS),
            input_data_type,
        })
    }

    fn encode(
        &self,
        input: impl BufferRef<Backend = Metal>,
        output_ids: impl BufferMut<Backend = Metal>,
        output_scores: impl BufferMut<Backend = Metal>,
        rows: u32,
        k: u32,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MetalError> {
        validate_buffers(
            self.columns,
            rows,
            k,
            self.input_data_type,
            input.size(),
            input.parts().1.start,
            output_ids.size(),
            output_ids.as_ref().parts().1.start,
            output_scores.size(),
            output_scores.as_ref().parts().1.start,
        )?;
        let rows = rows as usize;
        let workspace_bytes = |elements: usize, bytes: usize| {
            rows.checked_mul(elements)
                .and_then(|n| n.checked_mul(bytes))
                .filter(|&n| n <= 2 * 1024 * 1024 * 1024)
                .ok_or(InvalidInput("small radix workspace exceeds native allocation limit"))
        };
        let histogram_bytes = workspace_bytes(self.partitions as usize * RADIX_BUCKETS, size_of::<u32>())?;
        let key_bytes = workspace_bytes(k as usize, size_of::<u64>())?;
        let mut histograms = command_buffer.allocate_scratch(histogram_bytes)?;
        let mut prefixes = command_buffer.allocate_scratch(rows * size_of::<u64>())?;
        let mut masks = command_buffer.allocate_scratch(rows * size_of::<u64>())?;
        let mut ranks = command_buffer.allocate_scratch(rows * size_of::<u32>())?;
        let mut counts = command_buffer.allocate_scratch(rows * size_of::<u32>())?;
        let mut done = command_buffer.allocate_scratch(rows * size_of::<u32>())?;
        let mut keys = command_buffer.allocate_scratch(key_bytes)?;
        command_buffer.encode_fill(&mut prefixes, 0);
        command_buffer.encode_fill(&mut masks, 0);
        command_buffer.encode_fill(&mut counts, 0);
        command_buffer.encode_fill(&mut done, 0);

        for pass in 0..self.passes {
            self.pass.encode(
                input,
                &mut histograms,
                &mut prefixes,
                &mut masks,
                &mut ranks,
                &mut done,
                rows as u32,
                k,
                pass,
                command_buffer,
            );
        }
        self.collect.encode(
            input,
            output_ids,
            output_scores,
            &prefixes,
            &mut counts,
            &mut done,
            &mut keys,
            rows as u32,
            k,
            command_buffer,
        );
        Ok(())
    }
}
