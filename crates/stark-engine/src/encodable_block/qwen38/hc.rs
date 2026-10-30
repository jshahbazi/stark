//! Hyper-connections composed from standard Uzu projections and kernel DSL operations.
use super::{Encoding, Qwen38Error, project_batch, weights::Qwen38WeightLoader};
use crate::{
    backends::common::{
        Backend, BufferRef, CommandBufferEncoding, Kernels,
        kernel::{Qwen38HcInjectKernel, Qwen38HcMixKernel, Qwen38HcNormKernel, Qwen38SiluScaledKernel},
    },
    config::model::qwen38_language_model::Qwen38ArchitectureConfig,
    data_type::DataType,
    encodable_block::linear::LinearMatmul,
};

pub(super) struct HyperConnection<B: Backend> {
    norm: B::GlobalBuffer,
    down: LinearMatmul<B>,
    up: LinearMatmul<B>,
    injection: Option<LinearMatmul<B>>,
    norm_kernel: <B::Kernels as Kernels>::Qwen38HcNormKernel,
    silu_kernel: <B::Kernels as Kernels>::Qwen38SiluScaledKernel,
    mix_kernel: <B::Kernels as Kernels>::Qwen38HcMixKernel,
    inject_kernel: <B::Kernels as Kernels>::Qwen38HcInjectKernel,
    hidden: u32,
    streams: u32,
    rank: u32,
    epsilon: f32,
}

pub(super) struct Read<B: Backend> {
    pub mixed: B::ScratchBuffer,
    pub injection: Option<B::ScratchBuffer>,
}

impl<B: Backend> HyperConnection<B> {
    pub fn new(
        context: &B::Context,
        cfg: &Qwen38ArchitectureConfig,
        prefix: &str,
        injection: bool,
        loader: &Qwen38WeightLoader<B>,
    ) -> Result<Self, Qwen38Error<B>> {
        let wide = cfg.hidden_size.checked_mul(cfg.hc_count).ok_or(Qwen38Error::Invalid("HC width overflow"))?;
        Ok(Self {
            norm: loader.control_f32(&format!("{prefix}.hc_norm.weight"), &[wide])?,
            down: loader.linear(
                &format!("{prefix}.input_mix_weight_down"),
                wide,
                cfg.hc_lowrank,
                DataType::BF16,
                DataType::BF16,
            )?,
            up: loader.linear(
                &format!("{prefix}.input_mix_weight_up"),
                cfg.hc_lowrank,
                wide,
                DataType::BF16,
                DataType::BF16,
            )?,
            injection: injection
                .then(|| {
                    loader.linear(
                        &format!("{prefix}.block_inject_weight"),
                        wide,
                        cfg.hc_count,
                        DataType::BF16,
                        DataType::BF16,
                    )
                })
                .transpose()?,
            norm_kernel: <B::Kernels as Kernels>::Qwen38HcNormKernel::new(context).map_err(Qwen38Error::Backend)?,
            silu_kernel: <B::Kernels as Kernels>::Qwen38SiluScaledKernel::new(context).map_err(Qwen38Error::Backend)?,
            mix_kernel: <B::Kernels as Kernels>::Qwen38HcMixKernel::new(context).map_err(Qwen38Error::Backend)?,
            inject_kernel: <B::Kernels as Kernels>::Qwen38HcInjectKernel::new(context).map_err(Qwen38Error::Backend)?,
            hidden: cfg.hidden_size,
            streams: cfg.hc_count,
            rank: cfg.hc_lowrank,
            epsilon: cfg.rms_norm_eps,
        })
    }

    pub fn read_batch(
        &self,
        hyper: impl BufferRef<Backend = B>,
        rows: u32,
        commands: &mut Encoding<B>,
    ) -> Result<Read<B>, B::Error> {
        let mut normalized = commands.allocate_scratch_for_shape(&[rows, self.streams, self.hidden], DataType::BF16)?;
        self.norm_kernel.encode(
            hyper,
            &self.norm,
            &mut normalized,
            self.hidden,
            self.streams,
            rows,
            self.epsilon,
            commands,
        );
        let down = project_batch(&self.down, &normalized, rows, commands)?;
        let injection =
            self.injection.as_ref().map(|linear| project_batch(linear, &normalized, rows, commands)).transpose()?;
        let mut activated = commands.allocate_scratch_for_shape(&[rows, self.rank], DataType::BF16)?;
        self.silu_kernel.encode(&down, &mut activated, rows * self.rank, 1.0 / self.streams as f32, commands);
        let up = project_batch(&self.up, &activated, rows, commands)?;
        let mut mixed = commands.allocate_scratch_for_shape(&[rows, self.hidden], DataType::BF16)?;
        self.mix_kernel.encode(&normalized, &up, &mut mixed, self.hidden, self.streams, rows, commands);
        Ok(Read {
            mixed,
            injection,
        })
    }

    pub fn inject_batch(
        &self,
        hyper: impl BufferRef<Backend = B>,
        branch: impl BufferRef<Backend = B>,
        injection: impl BufferRef<Backend = B>,
        rows: u32,
        commands: &mut Encoding<B>,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut output = commands.allocate_scratch_for_shape(&[rows, self.streams, self.hidden], DataType::BF16)?;
        self.inject_kernel.encode(hyper, branch, injection, &mut output, self.hidden, self.streams, rows, commands);
        Ok(output)
    }
}
