//! Persistent F32 GLM KDA decode state. No weight expansion or request caching.
//! Original Stark implementation of the scalar contract in `glm.rs`.

use std::{ptr::NonNull, sync::OnceLock};

use metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLDeviceExt, MTLLibraryExt,
    MTLResourceOptions, MTLSize,
};
#[cfg(test)]
use metal::{MTLCommandBuffer, MTLCommandBufferExt, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue};
use objc2::{rc::Retained, runtime::ProtocolObject};

#[cfg(test)]
use super::profile;
use super::{Error, Result};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

struct Executor {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    #[cfg(test)]
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static EXECUTOR: OnceLock<std::result::Result<Executor, &'static str>> = OnceLock::new();

fn executor() -> Result<&'static Executor> {
    EXECUTOR
        .get_or_init(|| {
            let device = <dyn MTLDevice>::system_default().ok_or("Metal device unavailable")?;
            #[cfg(test)]
            let queue = device.new_command_queue().ok_or("Metal KDA queue unavailable")?;
            let options = metal::MTLCompileOptions::new();
            options.set_math_mode(metal::MTLMathMode::Safe);
            let library = device
                .new_library_with_source(include_str!("glm_metal.metal"), Some(&options))
                .map_err(|_| "Metal KDA kernel compilation failed")?;
            let function = library.new_function_with_name("glm_kda_decode").ok_or("Metal KDA function unavailable")?;
            let pipeline =
                device.new_compute_pipeline_state_with_function(&function).map_err(|_| "Metal KDA pipeline failed")?;
            if pipeline.max_total_threads_per_threadgroup() < 256 {
                return Err("Metal KDA execution width is unsupported");
            }
            Ok(Executor {
                device,
                #[cfg(test)]
                queue,
                pipeline,
            })
        })
        .as_ref()
        .map_err(|message| Error::Unsupported(message))
}

pub(super) struct Kda {
    parameters: [u32; 6],
    input: Buffer,
    weights: Buffer,
    state: Buffer,
    output: Buffer,
    failure: Buffer,
    inner: usize,
    heads: usize,
    state_values: usize,
}

pub(super) struct Geometry {
    pub heads: usize,
    pub dim: usize,
    pub kernel: usize,
    pub rms_eps: f32,
    pub lower_bound: f32,
}

impl Kda {
    pub(super) fn new(
        geometry: Geometry,
        convolutions: [&[f32]; 3],
        dt: &[f32],
        norm: &[f32],
        a: &[f32],
    ) -> Result<Self> {
        let Geometry {
            heads,
            dim,
            kernel,
            rms_eps,
            lower_bound,
        } = geometry;
        let inner = heads.checked_mul(dim).ok_or(Error::Invalid("Metal KDA dimensions"))?;
        if heads == 0
            || heads > 256
            || dim == 0
            || dim > 256
            || !(2..=32).contains(&kernel)
            || !rms_eps.is_finite()
            || rms_eps <= 0.0
            || !lower_bound.is_finite()
            || lower_bound >= 0.0
            || convolutions.iter().any(|values| values.len() != inner * kernel)
            || dt.len() != inner
            || norm.len() != dim
            || a.len() != heads
        {
            return Err(Error::Invalid("Metal KDA dimensions"));
        }
        let all_weights: Vec<_> = convolutions.into_iter().flatten().chain(dt).chain(norm).chain(a).copied().collect();
        if all_weights.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("Metal KDA weights"));
        }
        let e = executor()?;
        let allocate = |length| {
            e.device
                .new_buffer(length, MTLResourceOptions::STORAGE_MODE_SHARED)
                .ok_or(Error::Invalid("Metal KDA allocation failed"))
        };
        let state_values = 3 * inner * (kernel - 1) + inner * dim;
        let state = allocate(state_values * 4)?;
        let weights = e
            .device
            .new_buffer_with_data(bytemuck::cast_slice(&all_weights), MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or(Error::Invalid("Metal KDA allocation failed"))?;
        let mut result = Self {
            parameters: [
                heads as u32,
                dim as u32,
                kernel as u32,
                inner as u32,
                rms_eps.to_bits(),
                lower_bound.to_bits(),
            ],
            input: allocate((5 * inner + heads) * 4)?,
            weights,
            state,
            output: allocate(inner * 4)?,
            failure: allocate(4)?,
            inner,
            heads,
            state_values,
        };
        result.reset();
        Ok(result)
    }

    pub(super) fn reset(&mut self) {
        // Every submitted KDA command is awaited before returning to its owner.
        // The model serializes decisions, so reset has exclusive buffer access.
        unsafe { self.state.contents().as_ptr().cast::<f32>().write_bytes(0, self.state_values) };
        unsafe { self.failure.contents().as_ptr().cast::<u32>().write(0) };
    }

    pub(super) fn inner(&self) -> usize {
        self.inner
    }
    pub(super) fn heads(&self) -> usize {
        self.heads
    }
    pub(super) fn dim(&self) -> usize {
        self.parameters[1] as usize
    }
    pub(super) fn input_buffer(&self) -> &Buffer {
        &self.input
    }
    pub(super) fn output_buffer(&self) -> &Buffer {
        &self.output
    }

    /// Append one state update to the caller's command. All projections must
    /// already be encoded into input_buffer in q/k/v/forget/gate/beta order.
    pub(super) fn encode(
        &mut self,
        encoding: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    ) -> Result<()> {
        // The model serializes steps and awaits each command before reusing
        // these buffers. This status is initialized before this command runs.
        unsafe { self.failure.contents().as_ptr().cast::<u32>().write(0) };
        encoding.set_compute_pipeline_state(&executor()?.pipeline);
        for (index, buffer) in [&self.input, &self.weights, &self.state, &self.output].into_iter().enumerate() {
            encoding.set_buffer(Some(buffer), 0, index);
        }
        encoding.set_bytes(NonNull::from(&self.parameters).cast(), std::mem::size_of_val(&self.parameters), 4);
        encoding.set_buffer(Some(&self.failure), 0, 5);
        encoding.dispatch_threadgroups(
            MTLSize {
                width: self.heads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.dim(),
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    /// Called only after the command containing encode has completed.
    pub(super) fn check_completed(&self) -> Result<()> {
        // GPU writes are visible after completion of the owning command.
        if unsafe { self.failure.contents().as_ptr().cast::<u32>().read() } != 0 {
            return Err(Error::Invalid("nonfinite Metal KDA intermediate"));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn step(
        &mut self,
        projections: [&[f32]; 5],
        beta: &[f32],
    ) -> Result<Vec<f32>> {
        if projections.iter().any(|v| v.len() != self.inner)
            || beta.len() != self.heads
            || projections.iter().copied().flatten().chain(beta).any(|v| !v.is_finite())
        {
            return Err(Error::Invalid("Metal KDA projection dimensions or values"));
        }
        let _scope = profile::Scope::new("glm.kda.metal");
        for (part, values) in projections.into_iter().chain(std::iter::once(beta)).enumerate() {
            // Checked disjoint input spans are fully initialized before submit.
            unsafe {
                self.input
                    .contents()
                    .as_ptr()
                    .cast::<f32>()
                    .add(part * self.inner)
                    .copy_from_nonoverlapping(values.as_ptr(), values.len());
            }
        }
        objc2::rc::autoreleasepool(|_| {
            let e = executor()?;
            let command = e.queue.command_buffer().ok_or(Error::Invalid("Metal KDA command allocation failed"))?;
            let encoding =
                command.compute_command_encoder().ok_or(Error::Invalid("Metal KDA encoder allocation failed"))?;
            self.encode(&encoding)?;
            encoding.end_encoding();
            command.commit();
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(Error::Invalid("Metal KDA execution failed"));
            }
            self.check_completed()?;
            profile::record("glm.kda.metal.gpu", command.gpu_end_time() - command.gpu_start_time(), 0);
            // Completion makes the initialized output visible to the CPU.
            let values =
                unsafe { std::slice::from_raw_parts(self.output.contents().as_ptr().cast::<f32>(), self.inner) }
                    .to_vec();
            if values.iter().any(|v| !v.is_finite()) {
                return Err(Error::Invalid("nonfinite Metal KDA output"));
            }
            Ok(values)
        })
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn gpu_kda_rejects_nonfinite_controls_that_gates_could_mask() {
        let mut state = Kda::new(
            Geometry {
                heads: 1,
                dim: 4,
                kernel: 2,
                rms_eps: 1e-5,
                lower_bound: -5.0,
            },
            [&[0.0; 8]; 3],
            &[0.0; 4],
            &[1.0; 4],
            &[-1.0],
        )
        .unwrap();
        for index in [3 * state.inner(), 4 * state.inner(), 5 * state.inner()] {
            state.reset();
            let mut input = vec![0.0; 5 * state.inner() + state.heads()];
            input[index] = f32::INFINITY;
            // Simulate a projection written by an earlier GPU dispatch. The
            // zero recurrent value would otherwise conceal these controls.
            unsafe {
                state.input.contents().as_ptr().cast::<f32>().copy_from_nonoverlapping(input.as_ptr(), input.len());
            }
            objc2::rc::autoreleasepool(|_| {
                let command = executor().unwrap().queue.command_buffer().unwrap();
                let encoding = command.compute_command_encoder().unwrap();
                state.encode(&encoding).unwrap();
                encoding.end_encoding();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                assert!(state.check_completed().is_err());
            });
        }
        state.reset();
        assert!(state.check_completed().is_ok());
        assert_eq!(state.step([&[0.0; 4]; 5], &[0.0]).unwrap(), [0.0; 4]);
    }
}
