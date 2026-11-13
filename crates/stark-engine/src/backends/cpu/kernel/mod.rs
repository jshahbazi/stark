use crate::backends::{
    common::{Kernels, kernel::Unsupported},
    cpu::Cpu,
};

mod activation;
pub(crate) mod activation_transform;
mod attention;
mod convolution;
mod copy_rows;
mod embedding;
mod gated_act_mul;
mod gdn;
mod logit_transform;
mod matmul;
mod moe;
mod normalization;
mod pooling;
mod radix_top_k_small;
mod sampling;
mod short_conv;
mod softmax;
mod ssm;
mod tensor_add_bias;
mod tensor_add_scale;
mod weaver;

#[cfg(test)]
pub(crate) use attention::gemma4_vision::gemma4_vision_pool_standardized_bf16;
#[cfg(test)]
pub(crate) use matmul::vision::vision_matmul_bf16;

include!(concat!(env!("OUT_DIR"), "/cpu.rs"));

pub struct CpuKernels;

impl Kernels for CpuKernels {
    type Backend = Cpu;

    autogen_kernels!();
    type AttentionKernel = attention::AttentionCpuKernel;
    type DeltaNetChunkedPrefill = Unsupported<Cpu>;
    type DeltaNetTreeVerify = Unsupported<Cpu>;
    type MatmulKernel = matmul::MatmulCpuKernel;
    type RadixTopKSmall = radix_top_k_small::CpuRadixTopKSmall;
}
