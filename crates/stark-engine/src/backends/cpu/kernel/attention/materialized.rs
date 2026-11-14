use half::bf16;

use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, InvalidInput,
            kernel::{AttentionArguments, AttentionKernelConfig},
        },
        cpu::{Cpu, buffer::CpuBufferExt, command_buffer::CpuCommandBufferEncoding, error::CpuError},
    },
    utils::pointers::{SendPtr, SendPtrMut},
};

fn pointer(buffer: impl BufferRef<Backend = Cpu>) -> SendPtr<bf16> {
    let (buffer, range) = buffer.parts();
    SendPtr(buffer.cpu_address().as_ptr().cast::<u8>().wrapping_byte_add(range.start).cast())
}

pub(super) fn encode(
    config: AttentionKernelConfig,
    arguments: AttentionArguments<
        '_,
        Cpu,
        impl BufferRef<Backend = Cpu>,
        impl BufferRef<Backend = Cpu>,
        impl BufferRef<Backend = Cpu>,
        impl BufferRef<Backend = Cpu>,
    >,
    mut output: &mut <Cpu as Backend>::ScratchBuffer,
    commands: &mut CpuCommandBufferEncoding,
) -> Result<(), CpuError> {
    if arguments.trie.is_some() {
        return Err(InvalidInput("materialized prefill attention does not support tries").into());
    }
    let queries = pointer(arguments.queries);
    let keys = pointer(arguments.keys);
    let values = pointer(arguments.values);
    let (buffer, range) = output.reborrow().parts();
    let output = SendPtrMut(buffer.cpu_address().as_ptr().cast::<u8>().wrapping_byte_add(range.start).cast::<bf16>());
    let rows = arguments.suffix_length;
    let prefix = arguments.cache.prefix_len();
    let ring = arguments.cache.ring_params();
    let count = prefix.checked_add(rows).ok_or(InvalidInput("materialized attention length overflow"))?;
    let suffix_position = ring.map_or(prefix, |ring| ring.ring_length);
    commands.push_command(move || {
        for head in 0..config.num_q_heads {
            let group = head / (config.num_q_heads / config.num_groups);
            for row in 0..rows {
                let mut scores = vec![f32::NEG_INFINITY; count as usize];
                for key in 0..count {
                    if super::mask::should_use_key(
                        ring,
                        None,
                        config.sliding_window_size,
                        row,
                        prefix,
                        suffix_position,
                        suffix_position + row,
                        key,
                        true,
                    ) {
                        let mut dot = 0.0f32;
                        for dim in 0..config.head_dim {
                            let q = ((head * rows + row) * config.head_dim + dim) as usize;
                            let k = ((key * config.num_groups + group) * config.head_dim + dim) as usize;
                            dot += unsafe { (*queries.as_ptr().add(q)).to_f32() * (*keys.as_ptr().add(k)).to_f32() };
                        }
                        scores[key as usize] = bf16::from_f32(dot).to_f32();
                    }
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let total: f32 = scores.iter().map(|score| (*score - maximum).exp()).sum();
                let probabilities: Vec<bf16> =
                    scores.iter().map(|score| bf16::from_f32((*score - maximum).exp() / total)).collect();
                for dim in 0..config.head_dim {
                    let mut sum = 0.0f32;
                    for key in 0..count {
                        let v = ((key * config.num_groups + group) * config.head_dim + dim) as usize;
                        sum += probabilities[key as usize].to_f32() * unsafe { (*values.as_ptr().add(v)).to_f32() };
                    }
                    let index = ((row * config.num_q_heads + head) * config.head_dim + dim) as usize;
                    unsafe { *output.as_ptr().add(index) = bf16::from_f32(sum) };
                }
            }
        }
    });
    Ok(())
}
