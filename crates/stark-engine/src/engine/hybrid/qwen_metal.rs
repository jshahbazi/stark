// SPDX-License-Identifier: Apache-2.0
//! Persistent Qwen GDN F32 state. Commands are encoded by `weight_metal` so
//! projections, recurrent updates and output projection share one completion.

use metal::{MTLBuffer, MTLDevice, MTLDeviceExt, MTLResource, MTLResourceOptions};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::{Error, Result};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

const EDGE0_RAW_NEGATIVE_A: u32 = 2;
const EDGE0_TILED_VALUE_HEADS: u32 = 4;

#[derive(Clone)]
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

/// Private RAM only. Immutable controls and overwrite-before-read convolved/
/// result buffers are excluded. The model-level owner fences before capture.
pub(super) struct Checkpoint {
    parameters: [u32; 8],
    history: Box<[f32]>,
    recurrent: Box<[f32]>,
}

impl Checkpoint {
    pub(super) fn bytes(&self) -> usize {
        (self.history.len() + self.recurrent.len()) * 4
    }
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

    #[stark_test]
    fn private_gdn_checkpoint_roundtrip_preserves_bits_and_rejects_bad_state() {
        use crate::engine::hybrid::weight_metal::{DecodeGraph, DeviceVector};

        let state = Gdn::new_edge0(
            Geometry {
                key_heads: 1,
                key_dim: 2,
                value_heads: 2,
                value_dim: 2,
                conv_kernel: 2,
                eps: 1e-5,
                sigmoid_output: true,
            },
            &[0.0; 16],
            &[-0.5, -1.0],
            &[0.0; 2],
            &[1.0; 2],
        )
        .unwrap();
        let values = [f32::from_bits(0x8000_0000), f32::from_bits(1), 0.25, -0.75, 1.0, -2.0, 0.5, 0.0];
        unsafe {
            state.history.contents().as_ptr().cast::<f32>().copy_from_nonoverlapping(values.as_ptr(), values.len());
            state.recurrent.contents().as_ptr().cast::<f32>().copy_from_nonoverlapping(values.as_ptr(), values.len());
        }
        let snapshot = state.capture_checkpoint().unwrap();
        let mut graph = DecodeGraph::new().unwrap();
        let overwrite = DecodeGraph::upload(&[7.0; 8]).unwrap();
        for buffer in [&state.history, &state.recurrent] {
            let destination = DeviceVector {
                buffer: buffer.clone(),
                offset: 0,
                len: 8,
            };
            graph.copy_into(&overwrite, &destination).unwrap();
        }
        graph.finish(&[]).unwrap();
        let restored = state.stage_checkpoint(&snapshot).unwrap();
        let actual = restored.capture_checkpoint().unwrap();
        assert!(
            actual.history.iter().zip(values).all(|(a, b)| a.to_bits() == b.to_bits()),
            "GDN history checkpoint bits"
        );
        assert!(
            actual.recurrent.iter().zip(values).all(|(a, b)| a.to_bits() == b.to_bits()),
            "GDN recurrent checkpoint bits"
        );
        assert!(!std::ptr::eq(&*restored.history, &*state.history), "GDN restore aliases mutable history");
        assert!(!std::ptr::eq(&*restored.recurrent, &*state.recurrent), "GDN restore aliases mutable recurrence");
        assert!(std::ptr::eq(&*restored.controls, &*state.controls), "GDN immutable control ownership");
        let mut wrong = restored.capture_checkpoint().unwrap();
        wrong.parameters[6] ^= EDGE0_TILED_VALUE_HEADS;
        assert!(state.stage_checkpoint(&wrong).is_err());
        wrong = restored.capture_checkpoint().unwrap();
        wrong.recurrent[0] = f32::NAN;
        assert!(state.stage_checkpoint(&wrong).is_err());
        wrong = restored.capture_checkpoint().unwrap();
        wrong.history = Box::new([]);
        assert!(state.stage_checkpoint(&wrong).is_err());
        unsafe {
            state.history.contents().as_ptr().cast::<f32>().write(f32::NAN);
        }
        assert!(state.capture_checkpoint().is_err());
    }

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

    // Independent F64 post-convolution recurrence. Its state stays in
    // [head,value,key] order, rather than the device's [head,key,value] layout.
    fn edge0_reference_step(
        g: &Geometry,
        state: &mut [f64],
        negative_a: &[f32],
        dt: &[f32],
        norm: &[f32],
        seed: usize,
    ) -> Vec<f64> {
        let data = |count: usize, salt: usize| {
            (0..count).map(|i| f64::from((((i * 19 + seed * 7 + salt) % 67) as f32 - 33.0) * 0.017)).collect::<Vec<_>>()
        };
        let key_width = g.key_heads * g.key_dim;
        let mixed = data(2 * key_width + g.value_heads * g.value_dim, 3);
        let z = data(g.value_heads * g.value_dim, 5);
        let a = data(g.value_heads, 7);
        let b = data(g.value_heads, 11);
        let mut output = vec![0.0; z.len()];
        for head in 0..g.value_heads {
            let key_head = head % g.key_heads;
            let mut q = mixed[key_head * g.key_dim..(key_head + 1) * g.key_dim].to_vec();
            let mut k = mixed[key_width + key_head * g.key_dim..key_width + (key_head + 1) * g.key_dim].to_vec();
            let qs = 1.0 / ((q.iter().map(|v| v * v).sum::<f64>() + 1e-6) * g.key_dim as f64).sqrt();
            let ks = 1.0 / (k.iter().map(|v| v * v).sum::<f64>() + 1e-6).sqrt();
            for i in 0..g.key_dim {
                q[i] *= qs;
                k[i] *= ks;
            }
            let argument = a[head] + f64::from(dt[head]);
            let softplus = argument.max(0.0) + (-argument.abs()).exp().ln_1p();
            let decay = (f64::from(negative_a[head]) * softplus).exp();
            let beta = 1.0 / (1.0 + (-b[head]).exp());
            let mut values = vec![0.0; g.value_dim];
            for (channel, value) in values.iter_mut().enumerate() {
                let offset = (head * g.value_dim + channel) * g.key_dim;
                let row = &mut state[offset..offset + g.key_dim];
                for entry in row.iter_mut() {
                    *entry *= decay;
                }
                let prediction = row.iter().zip(&k).map(|(s, k)| s * k).sum::<f64>();
                let delta = (mixed[2 * key_width + head * g.value_dim + channel] - prediction) * beta;
                for i in 0..g.key_dim {
                    row[i] += k[i] * delta;
                }
                *value = row.iter().zip(&q).map(|(s, q)| s * q).sum();
            }
            let inverse =
                1.0 / (values.iter().map(|v| v * v).sum::<f64>() / g.value_dim as f64 + f64::from(g.eps)).sqrt();
            for channel in 0..g.value_dim {
                let index = head * g.value_dim + channel;
                let gate = 1.0 / (1.0 + (-z[index]).exp());
                let gate = if g.sigmoid_output {
                    gate
                } else {
                    z[index] * gate
                };
                output[index] = values[channel] * inverse * f64::from(norm[channel]) * gate;
            }
        }
        output
    }

    #[stark_test]
    fn edge0_gdn_gpu_preserves_negative_a_tiled_heads_and_f32_state() {
        let harness = RecurrenceHarness::new();
        for (key_dim, value_dim, value_heads) in [(7, 13, 6), (128, 128, 48)] {
            for sigmoid_output in [false, true] {
                let geometry = || Geometry {
                    key_heads: value_heads / 3,
                    key_dim,
                    value_heads,
                    value_dim,
                    conv_kernel: 4,
                    eps: 1e-6,
                    sigmoid_output,
                };
                let width = 2 * (value_heads / 3) * key_dim + value_heads * value_dim;
                let negative_a: Vec<f32> = (0..value_heads).map(|h| -0.1 - h as f32 * 0.025).collect();
                let dt: Vec<f32> = (0..value_heads).map(|h| (h as f32 - 3.0) * 0.015).collect();
                let norm: Vec<f32> = (0..value_dim).map(|c| 0.8 + (c % 11) as f32 * 0.03).collect();
                let make = || Gdn::new_edge0(geometry(), &vec![0.02; width * 4], &negative_a, &dt, &norm).unwrap();
                let mut staged = make();
                let mut fused = make();
                assert_eq!(staged.parameters[6], 6 | u32::from(sigmoid_output));
                let mut reference = vec![0.0; value_heads * key_dim * value_dim];
                for step in 0..8 {
                    let expected = edge0_reference_step(&geometry(), &mut reference, &negative_a, &dt, &norm, step);
                    let plain = harness.run(&mut staged, false, step, 1);
                    let blocked = harness.run(&mut fused, true, step, 1);
                    assert!(plain.0 == blocked.0, "Edge0 blocked output changed ordered F32 arithmetic");
                    assert!(plain.1 == blocked.1, "Edge0 blocked state changed ordered F32 arithmetic");
                    for (&actual, &expected) in plain.0.iter().zip(&expected) {
                        assert!(actual.is_finite() && (f64::from(actual) - expected).abs() <= 2e-5);
                    }
                    for head in 0..value_heads {
                        for key in 0..key_dim {
                            for value in 0..value_dim {
                                let actual = plain.1[(head * key_dim + key) * value_dim + value];
                                let expected = reference[(head * value_dim + value) * key_dim + key];
                                assert!(actual.is_finite() && (f64::from(actual) - expected).abs() <= 2e-5);
                            }
                        }
                    }
                }
                staged.reset();
                let after_reset = harness.run(&mut staged, false, 0, 1);
                let mut fresh = make();
                let fresh_result = harness.run(&mut fresh, false, 0, 1);
                assert!(after_reset.0 == fresh_result.0 && after_reset.1 == fresh_result.1);
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
        assert_eq!(state.parameters[6], 1);
        let edge0 = Gdn::new_edge0(geometry(), &convolution, &[-0.1; 6], &[0.3; 6], &[1.0; 8]).unwrap();
        assert_eq!(edge0.parameters[6], 7);
        assert!(Gdn::new_edge0(geometry(), &convolution, &[0.1; 6], &[0.3; 6], &[1.0; 8]).is_err());
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
    pub(super) fn capture_checkpoint(&self) -> Result<Checkpoint> {
        let read = |buffer: &Buffer, count: usize| -> Result<Box<[f32]>> {
            if count.checked_mul(4) != Some(buffer.length()) {
                return Err(Error::Invalid("GDN checkpoint source span"));
            }
            let values = unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<f32>(), count) };
            if values.iter().any(|value| !value.is_finite()) {
                return Err(Error::Invalid("GDN checkpoint nonfinite state"));
            }
            Ok(values.to_vec().into_boxed_slice())
        };
        Ok(Checkpoint {
            parameters: self.parameters,
            history: read(&self.history, self.history_count)?,
            recurrent: read(&self.recurrent, self.recurrent_count)?,
        })
    }

    /// Allocate and populate replacement owners without touching this state.
    /// A caller can stage every layer before its infallible model-level swap.
    pub(super) fn stage_checkpoint(
        &self,
        checkpoint: &Checkpoint,
    ) -> Result<Self> {
        if checkpoint.parameters != self.parameters
            || checkpoint.history.len() != self.history_count
            || checkpoint.recurrent.len() != self.recurrent_count
            || checkpoint.history.iter().chain(checkpoint.recurrent.iter()).any(|value| !value.is_finite())
        {
            return Err(Error::Invalid("GDN checkpoint geometry or values"));
        }
        let device = self.history.device();
        let copy = |values: &[f32]| {
            device
                .new_buffer_with_data(bytemuck::cast_slice(values), MTLResourceOptions::STORAGE_MODE_SHARED)
                .ok_or(Error::Invalid("GDN checkpoint replacement allocation"))
        };
        let zeros = |count| {
            device
                .new_buffer_with_data(&vec![0_u8; count * 4], MTLResourceOptions::STORAGE_MODE_SHARED)
                .ok_or(Error::Invalid("GDN checkpoint scratch allocation"))
        };
        Ok(Self {
            geometry: self.geometry.clone(),
            parameters: self.parameters,
            controls: self.controls.clone(),
            history: copy(&checkpoint.history)?,
            recurrent: copy(&checkpoint.recurrent)?,
            convolved: zeros(self.parameters[7] as usize)?,
            result: zeros(self.geometry.value_heads * self.geometry.value_dim)?,
            history_count: self.history_count,
            recurrent_count: self.recurrent_count,
        })
    }

    pub(super) fn new(
        geometry: Geometry,
        convolution: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        norm: &[f32],
    ) -> Result<Self> {
        Self::new_with_policy(geometry, convolution, a_log, dt_bias, norm, false)
    }

    /// Original GGUF controls already contain -exp(A_log), with value heads
    /// tiled across key heads. Keep both transformations in their stored form.
    pub(super) fn new_edge0(
        geometry: Geometry,
        convolution: &[f32],
        negative_a: &[f32],
        dt_bias: &[f32],
        norm: &[f32],
    ) -> Result<Self> {
        Self::new_with_policy(geometry, convolution, negative_a, dt_bias, norm, true)
    }

    fn new_with_policy(
        geometry: Geometry,
        convolution: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        norm: &[f32],
        edge0: bool,
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
        if all_controls.iter().any(|value| !value.is_finite()) || (edge0 && a_log.iter().any(|value| *value > 0.0)) {
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
            u32::from(g.sigmoid_output)
                | if edge0 {
                    EDGE0_RAW_NEGATIVE_A | EDGE0_TILED_VALUE_HEADS
                } else {
                    0
                },
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
