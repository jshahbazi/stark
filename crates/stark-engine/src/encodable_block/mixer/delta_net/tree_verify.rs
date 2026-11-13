use crate::{
    backends::common::{Backend, Buffer, Subbuffer},
    data_type::DataType,
};

#[derive(Clone, Copy)]
pub struct TreeVerifyNewArguments {
    pub data_type: DataType,
    pub num_k_heads: u32,
    pub num_v_heads: u32,
    pub head_k_dim: u32,
    pub head_v_dim: u32,
}

#[derive(Clone, Copy)]
pub struct TreeVerifyEncodeArguments<'a, B: Backend> {
    pub q: Subbuffer<'a, B::ScratchBuffer, false>,
    pub k: Subbuffer<'a, B::ScratchBuffer, false>,
    pub v: Subbuffer<'a, B::ScratchBuffer, false>,
    pub trie: &'a B::ConstantBuffer,
    pub log_decay: Subbuffer<'a, B::ScratchBuffer, false>,
    pub beta: Subbuffer<'a, B::ScratchBuffer, false>,
    pub h0: &'a dyn Buffer<Backend = B>,
    pub tree_size: u32,
}

#[cfg(test)]
#[path = "../../../../unit/encodable_block/delta_net_tree_verify_bench.rs"]
mod tests;
