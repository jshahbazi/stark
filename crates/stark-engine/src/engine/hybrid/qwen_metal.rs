// SPDX-License-Identifier: Apache-2.0
//! Persistent Qwen GDN F32 state. Commands are encoded by `weight_metal` so
//! projections, recurrent updates and output projection share one completion.

use metal::{MTLBuffer, MTLDevice, MTLDeviceExt, MTLResourceOptions};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::{Error, Result};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

pub(super) struct Geometry {
    pub key_heads: usize,
    pub key_dim: usize,
    pub value_heads: usize,
    pub value_dim: usize,
    pub conv_kernel: usize,
    pub eps: f32,
    pub sigmoid_output: bool,
}

pub(super) struct Gdn {
    pub geometry: Geometry,
    pub parameters: [u32; 8],
    pub controls: Buffer,
    pub history: Buffer,
    /// Private `[head, key, value]` F32 layout gives coalesced GPU state access.
    pub recurrent: Buffer,
    pub convolved: Buffer,
    pub result: Buffer,
    history_count: usize,
    recurrent_count: usize,
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn qwen_gdn_state_initializes_and_clears_all_recurrent_history() {
        let geometry = || Geometry {
            key_heads: 2,
            key_dim: 4,
            value_heads: 6,
            value_dim: 8,
            conv_kernel: 4,
            eps: 1e-6,
            sigmoid_output: true,
        };
        let width = 2 * 2 * 4 + 6 * 8;
        let convolution = vec![0.2; width * 4];
        let mut state = Gdn::new(geometry(), &convolution, &[0.1; 6], &[0.3; 6], &[1.0; 8]).unwrap();
        let zeroed = |state: &Gdn| unsafe {
            std::slice::from_raw_parts(state.history.contents().as_ptr().cast::<f32>(), state.history_count)
                .iter()
                .chain(std::slice::from_raw_parts(
                    state.recurrent.contents().as_ptr().cast::<f32>(),
                    state.recurrent_count,
                ))
                .all(|value| *value == 0.0)
        };
        assert!(zeroed(&state));
        // Exclusive owned CPU/GPU buffers; no command has been submitted.
        unsafe {
            std::slice::from_raw_parts_mut(state.history.contents().as_ptr().cast::<f32>(), state.history_count)
                .fill(1.0);
            std::slice::from_raw_parts_mut(state.recurrent.contents().as_ptr().cast::<f32>(), state.recurrent_count)
                .fill(2.0);
        }
        assert!(!zeroed(&state));
        state.reset();
        assert!(zeroed(&state));
        assert!(Gdn::new(geometry(), &convolution[..convolution.len() - 1], &[0.1; 6], &[0.3; 6], &[1.0; 8]).is_err());
        assert!(Gdn::new(geometry(), &convolution, &[f32::NAN; 6], &[0.3; 6], &[1.0; 8]).is_err());
        let mut invalid = geometry();
        invalid.value_heads = 5;
        assert!(Gdn::new(invalid, &convolution, &[0.1; 6], &[0.3; 6], &[1.0; 8]).is_err());
    }
}

impl Gdn {
    pub(super) fn new(
        geometry: Geometry,
        convolution: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        norm: &[f32],
    ) -> Result<Self> {
        let g = &geometry;
        if g.key_heads == 0
            || g.key_heads > 256
            || g.value_heads == 0
            || g.value_heads > 256
            || !g.value_heads.is_multiple_of(g.key_heads)
            || !(1..=1024).contains(&g.key_dim)
            || !(1..=1024).contains(&g.value_dim)
            || !(2..=32).contains(&g.conv_kernel)
            || !g.eps.is_finite()
            || g.eps <= 0.0
        {
            return Err(Error::Invalid("Metal Qwen GDN geometry"));
        }
        let width = 2 * g.key_heads * g.key_dim + g.value_heads * g.value_dim;
        let recurrent_count = g.value_heads * g.value_dim * g.key_dim;
        if recurrent_count > 16_777_216
            || width > 8_000_000
            || convolution.len() != width * g.conv_kernel
            || a_log.len() != g.value_heads
            || dt_bias.len() != g.value_heads
            || norm.len() != g.value_dim
        {
            return Err(Error::Invalid("Metal Qwen GDN control geometry"));
        }
        let all_controls: Vec<f32> = convolution.iter().chain(a_log).chain(dt_bias).chain(norm).copied().collect();
        if all_controls.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invalid("Metal Qwen GDN control values"));
        }
        let device = <dyn MTLDevice>::system_default().ok_or(Error::Unsupported("Metal device unavailable"))?;
        let allocate = |count| {
            device
                .new_buffer(count * 4, MTLResourceOptions::STORAGE_MODE_SHARED)
                .ok_or(Error::Invalid("Metal Qwen GDN allocation failed"))
        };
        let history_count = width * (g.conv_kernel - 1);
        let parameters = [
            g.key_heads as u32,
            g.key_dim as u32,
            g.value_heads as u32,
            g.value_dim as u32,
            g.conv_kernel as u32,
            g.eps.to_bits(),
            u32::from(g.sigmoid_output),
            width as u32,
        ];
        let controls = device
            .new_buffer_with_data(bytemuck::cast_slice(&all_controls), MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or(Error::Invalid("Metal Qwen GDN control allocation failed"))?;
        let mut result = Self {
            history: allocate(history_count)?,
            recurrent: allocate(recurrent_count)?,
            convolved: allocate(width)?,
            result: allocate(g.value_heads * g.value_dim)?,
            geometry,
            parameters,
            controls,
            history_count,
            recurrent_count,
        };
        result.reset();
        Ok(result)
    }

    pub(super) fn reset(&mut self) {
        // All commands complete before the owner regains this mutable state.
        // The model serializes requests, including failure recovery and reset.
        unsafe {
            self.history.contents().as_ptr().cast::<f32>().write_bytes(0, self.history_count);
            self.recurrent.contents().as_ptr().cast::<f32>().write_bytes(0, self.recurrent_count);
        }
    }
}
