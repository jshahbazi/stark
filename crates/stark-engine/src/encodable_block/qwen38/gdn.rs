//! Qwen3.8 GDN: Uzu projections, BF16 convolution/activations and F32 recurrent state.
use super::{Encoding, Qwen38Error, project_batch, weights::Qwen38WeightLoader};
use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBufferEncoding, Context, Kernels,
        kernel::{Qwen38GdnConvKernel, Qwen38GdnStepKernel},
    },
    config::model::qwen38_language_model::Qwen38ArchitectureConfig,
    data_type::DataType,
    encodable_block::linear::LinearMatmul,
};

pub(super) struct Gdn<B: Backend> {
    qkv: LinearMatmul<B>,
    z: LinearMatmul<B>,
    a: LinearMatmul<B>,
    b: LinearMatmul<B>,
    out: LinearMatmul<B>,
    convolution: B::GlobalBuffer,
    a_log: B::GlobalBuffer,
    dt_bias: B::GlobalBuffer,
    norm: B::GlobalBuffer,
    conv_kernel: <B::Kernels as Kernels>::Qwen38GdnConvKernel,
    step_kernel: <B::Kernels as Kernels>::Qwen38GdnStepKernel,
    channels: u32,
    key_heads: u32,
    value_heads: u32,
    taps: u32,
    epsilon: f32,
    sigmoid_gate: u32,
}

pub(super) struct GdnState<B: Backend> {
    pub recurrent: B::GlobalBuffer,
    pub convolution: B::GlobalBuffer,
}

impl<B: Backend> Gdn<B> {
    pub fn new(
        context: &B::Context,
        cfg: &Qwen38ArchitectureConfig,
        layer: u32,
        loader: &Qwen38WeightLoader<B>,
    ) -> Result<Self, Qwen38Error<B>> {
        if cfg.linear_key_head_dim != 128
            || cfg.linear_value_head_dim != 128
            || cfg.linear_num_key_heads == 0
            || !cfg.linear_num_value_heads.is_multiple_of(cfg.linear_num_key_heads)
            || cfg.linear_conv_kernel_dim < 2
        {
            return Err(Qwen38Error::Invalid("unsupported Qwen GDN geometry"));
        }
        let key_dim =
            cfg.linear_num_key_heads.checked_mul(128).ok_or(Qwen38Error::Invalid("GDN dimension overflow"))?;
        let value_dim =
            cfg.linear_num_value_heads.checked_mul(128).ok_or(Qwen38Error::Invalid("GDN dimension overflow"))?;
        let channels = key_dim
            .checked_mul(2)
            .and_then(|n| n.checked_add(value_dim))
            .ok_or(Qwen38Error::Invalid("GDN dimension overflow"))?;
        let p = format!("model.language_model.layers.{layer}.linear_attn");
        let linear = |name: &str, input, output| {
            loader.linear(&format!("{p}.{name}"), input, output, DataType::BF16, DataType::BF16)
        };
        Ok(Self {
            qkv: linear("in_proj_qkv", cfg.hidden_size, channels)?,
            z: linear("in_proj_z", cfg.hidden_size, value_dim)?,
            a: linear("in_proj_a", cfg.hidden_size, cfg.linear_num_value_heads)?,
            b: linear("in_proj_b", cfg.hidden_size, cfg.linear_num_value_heads)?,
            out: linear("out_proj", value_dim, cfg.hidden_size)?,
            convolution: loader
                .control_f32(&format!("{p}.conv1d.weight"), &[channels, 1, cfg.linear_conv_kernel_dim])?,
            a_log: loader.control_f32(&format!("{p}.A_log"), &[cfg.linear_num_value_heads])?,
            dt_bias: loader.control_f32(&format!("{p}.dt_bias"), &[cfg.linear_num_value_heads])?,
            norm: loader.control_f32(&format!("{p}.norm.weight"), &[128])?,
            conv_kernel: <B::Kernels as Kernels>::Qwen38GdnConvKernel::new(context).map_err(Qwen38Error::Backend)?,
            step_kernel: <B::Kernels as Kernels>::Qwen38GdnStepKernel::new(context).map_err(Qwen38Error::Backend)?,
            channels,
            key_heads: cfg.linear_num_key_heads,
            value_heads: cfg.linear_num_value_heads,
            taps: cfg.linear_conv_kernel_dim,
            epsilon: cfg.rms_norm_eps,
            sigmoid_gate: u32::from(cfg.output_gate_type.as_deref().unwrap_or(cfg.hidden_act.as_str()) == "sigmoid"),
        })
    }

    pub fn create_state(
        &self,
        context: &B::Context,
    ) -> Result<GdnState<B>, B::Error> {
        let mut recurrent = context.create_buffer(self.value_heads as usize * 128 * 128 * 4)?;
        let mut convolution = context.create_buffer(self.channels as usize * (self.taps - 1) as usize * 2)?;
        recurrent.as_slice_mut::<u8>().fill(0);
        convolution.as_slice_mut::<u8>().fill(0);
        Ok(GdnState {
            recurrent,
            convolution,
        })
    }

    pub fn encode_batch(
        &self,
        input: impl BufferRef<Backend = B>,
        rows: u32,
        state: &mut GdnState<B>,
        commands: &mut Encoding<B>,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let qkv = project_batch(&self.qkv, input, rows, commands)?;
        let z = project_batch(&self.z, input, rows, commands)?;
        let a = project_batch(&self.a, input, rows, commands)?;
        let b = project_batch(&self.b, input, rows, commands)?;
        let mut core = commands.allocate_scratch_for_shape(&[rows, self.value_heads, 128], DataType::BF16)?;
        // Preserve the qualified step arithmetic and causal state order while
        // batching the surrounding projections through standard GEMM.
        for row in 0..rows as usize {
            let mut convolution = commands.allocate_scratch_for_shape(&[self.channels], DataType::BF16)?;
            let qkv_span = row * self.channels as usize * 2..(row + 1) * self.channels as usize * 2;
            let control_span = row * self.value_heads as usize * 2..(row + 1) * self.value_heads as usize * 2;
            let value_span = row * self.value_heads as usize * 128 * 2..(row + 1) * self.value_heads as usize * 128 * 2;
            self.conv_kernel.encode(
                (&qkv).subrange(qkv_span),
                &self.convolution,
                &mut state.convolution,
                &mut convolution,
                self.channels,
                self.taps,
                commands,
            );
            self.step_kernel.encode(
                &convolution,
                (&z).subrange(value_span.clone()),
                (&a).subrange(control_span.clone()),
                (&b).subrange(control_span),
                &self.a_log,
                &self.dt_bias,
                &self.norm,
                &mut state.recurrent,
                (&mut core).subrange_mut(value_span),
                self.key_heads,
                self.value_heads,
                self.epsilon,
                self.sigmoid_gate,
                commands,
            );
        }
        project_batch(&self.out, &core, rows, commands)
    }
}
