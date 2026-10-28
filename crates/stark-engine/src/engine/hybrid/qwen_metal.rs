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
    use std::ptr::NonNull;

    use metal::{
        MTLCommandBuffer, MTLCommandBufferExt, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
        MTLComputeCommandEncoder, MTLComputePipelineState, MTLLibraryExt, MTLSize,
    };
    use stark_engine_macros::stark_test;

    use super::*;

    struct RecurrenceHarness {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        staged: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        fused: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    }

    impl RecurrenceHarness {
        fn new() -> Self {
            let device = <dyn MTLDevice>::system_default().unwrap();
            let queue = device.new_command_queue().unwrap();
            let source = format!(
                "#include <metal_stdlib>\nusing namespace metal;\n{}\n{}",
                include_str!("feed_forward.metal"),
                include_str!("qwen_gdn.metal"),
            );
            let options = metal::MTLCompileOptions::new();
            options.set_math_mode(metal::MTLMathMode::Safe);
            let library = device.new_library_with_source(&source, Some(&options)).unwrap();
            let pipeline = |name| {
                device.new_compute_pipeline_state_with_function(&library.new_function_with_name(name).unwrap()).unwrap()
            };
            let staged = pipeline("hybrid_qwen_gdn_recurrent");
            let fused = pipeline("hybrid_qwen_gdn_recurrent_fused");
            Self {
                device,
                queue,
                staged,
                fused,
            }
        }

        fn run(
            &self,
            state: &mut Gdn,
            fused: bool,
            seed: usize,
            repeats: usize,
        ) -> (Vec<f32>, Vec<f32>, f64) {
            let g = &state.geometry;
            let data = |count: usize, salt: usize| {
                (0..count).map(|i| (((i * 19 + seed * 7 + salt) % 67) as f32 - 33.0) * 0.017).collect::<Vec<_>>()
            };
            let upload = |values: &[f32]| {
                self.device
                    .new_buffer_with_data(bytemuck::cast_slice(values), MTLResourceOptions::STORAGE_MODE_SHARED)
                    .unwrap()
            };
            let width = 2 * g.key_heads * g.key_dim + g.value_heads * g.value_dim;
            let mixed = upload(&data(width, 3));
            let z = upload(&data(g.value_heads * g.value_dim, 5));
            let a = upload(&data(g.value_heads, 7));
            let b = upload(&data(g.value_heads, 11));
            let failure = upload(&[0.0]);
            let command = self.queue.command_buffer().unwrap();
            let encoder = command.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(if fused {
                &self.fused
            } else {
                &self.staged
            });
            for (index, buffer) in
                [&mixed, &z, &a, &b, &state.controls, &state.recurrent, &failure, &state.result].into_iter().enumerate()
            {
                encoder.set_buffer(Some(buffer), 0, index);
            }
            encoder.set_bytes(NonNull::from(&state.parameters).cast(), std::mem::size_of_val(&state.parameters), 8);
            for _ in 0..repeats {
                encoder.dispatch_threadgroups(
                    MTLSize {
                        width: g.value_heads,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: g.key_dim.max(g.value_dim).next_power_of_two(),
                        height: 1,
                        depth: 1,
                    },
                );
            }
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            assert_eq!(unsafe { *failure.contents().as_ptr().cast::<u32>() }, 0);
            let read = |buffer: &Buffer, count| unsafe {
                std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<f32>(), count).to_vec()
            };
            (
                read(&state.result, g.value_heads * g.value_dim),
                read(&state.recurrent, state.recurrent_count),
                command.gpu_end_time() - command.gpu_start_time(),
            )
        }
    }

    fn recurrence_state(
        key_dim: usize,
        value_dim: usize,
        value_heads: usize,
        sigmoid_output: bool,
    ) -> Gdn {
        let geometry = Geometry {
            key_heads: value_heads / 3,
            key_dim,
            value_heads,
            value_dim,
            conv_kernel: 4,
            eps: 1e-6,
            sigmoid_output,
        };
        let width = 2 * geometry.key_heads * key_dim + value_heads * value_dim;
        Gdn::new(
            geometry,
            &vec![0.02; width * 4],
            &vec![-0.3; value_heads],
            &vec![0.1; value_heads],
            &vec![1.1; value_dim],
        )
        .unwrap()
    }

    #[stark_test]
    fn qwen_gdn_blocked_recurrence_matches_staged_state_and_output_over_multiple_steps() {
        let harness = RecurrenceHarness::new();
        for (key_dim, value_dim, value_heads) in [(7, 13, 6), (128, 128, 48)] {
            for sigmoid_output in [false, true] {
                let mut staged = recurrence_state(key_dim, value_dim, value_heads, sigmoid_output);
                let mut fused = recurrence_state(key_dim, value_dim, value_heads, sigmoid_output);
                for step in 0..8 {
                    let expected = harness.run(&mut staged, false, step, 1);
                    let actual = harness.run(&mut fused, true, step, 1);
                    assert!(actual.0 == expected.0, "blocked GDN output changed ordered arithmetic");
                    assert!(actual.1 == expected.1, "blocked GDN recurrent state changed ordered arithmetic");
                }
                staged.reset();
                fused.reset();
                let expected = harness.run(&mut staged, false, 0, 1);
                let actual = harness.run(&mut fused, true, 0, 1);
                assert!(actual.0 == expected.0, "blocked GDN output differs after reset");
                assert!(actual.1 == expected.1, "blocked GDN state differs after reset");
            }
        }
    }

    #[stark_test]
    #[ignore]
    fn qwen_gdn_blocked_recurrence_microbenchmark() {
        let harness = RecurrenceHarness::new();
        let mut staged = recurrence_state(128, 128, 48, true);
        let mut fused = recurrence_state(128, 128, 48, true);
        for _ in 0..3 {
            staged.reset();
            fused.reset();
            let expected = harness.run(&mut staged, false, 1, 256);
            let actual = harness.run(&mut fused, true, 1, 256);
            assert!(actual.0 == expected.0, "blocked GDN benchmark output differs");
            assert!(actual.1 == expected.1, "blocked GDN benchmark state differs");
            eprintln!(
                "qwen_gdn_recurrence_gpu_us staged={:.3} blocked={:.3}",
                expected.2 * 1e6 / 256.0,
                actual.2 * 1e6 / 256.0
            );
        }
    }

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
