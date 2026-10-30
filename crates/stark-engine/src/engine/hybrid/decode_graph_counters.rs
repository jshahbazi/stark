// SPDX-License-Identifier: Apache-2.0
//! Opt-in stage timestamps. One token still commits and completes one command.
//! Counter payloads remain transient; only fixed-name aggregate durations leave.

use metal::{
    MTL_COUNTER_ERROR_VALUE, MTLCommonCounterSet, MTLComputePassDescriptor, MTLCounterSampleBuffer,
    MTLCounterSampleBufferDescriptor, MTLCounterSampleBufferExt, MTLCounterSamplingPoint, MTLCounterSetExt,
    MTLStorageMode,
};

use super::*;

// The local M5 Max accepts 4096 samples and rejects 4097 with Invalid.
// This bounds the diagnostic to 2048 encoders within one token command.
const SAMPLE_CAPACITY: usize = 4096;
const OTHER: &str = "native.metal.encoder.other";

struct Interval {
    start: usize,
    end: usize,
    category: &'static str,
    active: std::cell::Cell<bool>,
}

pub(super) struct EncoderTiming {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    samples: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    intervals: Vec<Interval>,
    #[cfg(test)]
    labeled: usize,
    reference: Option<(u64, u64)>,
}

impl EncoderTiming {
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Result<Self> {
        if !device.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary) {
            return Err(Error::Unsupported("encoder timestamp boundaries unavailable"));
        }
        let timestamp_name = MTLCommonCounterSet::Timestamp.name();
        let counter_set = device
            .counter_sets()
            .ok_or(Error::Unsupported("GPU counters unavailable"))?
            .into_iter()
            .find(|set| set.name() == timestamp_name)
            .ok_or(Error::Unsupported("GPU timestamp counter unavailable"))?;
        let descriptor = MTLCounterSampleBufferDescriptor::new();
        descriptor.set_counter_set(Some(&counter_set));
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_sample_count(SAMPLE_CAPACITY);
        descriptor.set_label("Stark aggregate encoder profile");
        let samples = device
            .new_counter_sample_buffer_with_descriptor(&descriptor)
            .map_err(|_| Error::Invalid("encoder timestamp buffer allocation failed"))?;
        Ok(Self {
            device,
            samples,
            intervals: Vec::new(),
            #[cfg(test)]
            labeled: 0,
            reference: None,
        })
    }

    pub fn encoder(
        &mut self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        dispatch: MTLDispatchType,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
        let start = self
            .intervals
            .len()
            .checked_mul(2)
            .filter(|&index| index + 1 < SAMPLE_CAPACITY)
            .ok_or(Error::Invalid("encoder timestamp capacity exceeded"))?;
        let descriptor = MTLComputePassDescriptor::new();
        descriptor.set_dispatch_type(dispatch);
        let attachment = descriptor.sample_buffer_attachments().object_at_indexed_subscript(0);
        attachment.set_sample_buffer(Some(&self.samples));
        attachment.set_start_of_encoder_sample_index(start);
        attachment.set_end_of_encoder_sample_index(start + 1);
        let encoder = command
            .compute_command_encoder_with_descriptor(&descriptor)
            .ok_or(Error::Invalid("timestamped encoder allocation failed"))?;
        self.intervals.push(Interval {
            start,
            end: start + 1,
            category: OTHER,
            active: std::cell::Cell::new(false),
        });
        Ok(encoder)
    }

    pub fn mark_dispatch(&self) {
        if let Some(interval) = self.intervals.last() {
            interval.active.set(true);
        }
    }

    #[cfg(test)]
    pub fn label_completed(
        &mut self,
        category: &'static str,
    ) {
        for interval in &mut self.intervals[self.labeled..] {
            interval.category = category;
        }
        self.labeled = self.intervals.len();
    }

    pub fn start_clock(&mut self) {
        let (mut cpu, mut gpu) = (0, 0);
        self.device.sample_timestamps_gpu_timestamp(&mut cpu, &mut gpu);
        self.reference = Some((cpu, gpu));
    }

    pub fn complete(
        &self,
        command_seconds: f64,
    ) -> Result<()> {
        let (mut cpu_end, mut gpu_end) = (0, 0);
        self.device.sample_timestamps_gpu_timestamp(&mut cpu_end, &mut gpu_end);
        let (cpu_start, gpu_start) = self.reference.ok_or(Error::Invalid("encoder timestamp clock missing"))?;
        if cpu_end <= cpu_start || gpu_end <= gpu_start || !command_seconds.is_finite() || command_seconds <= 0.0 {
            return Err(Error::Invalid("encoder timestamp clock invalid"));
        }
        // Metal's paired CPU timestamps are nanoseconds. GPU counter ticks
        // require this measured clock ratio, rather than assuming nanoseconds.
        let seconds_per_tick = (cpu_end - cpu_start) as f64 / (gpu_end - gpu_start) as f64 * 1e-9;
        let count = self.intervals.len() * 2;
        let bytes =
            self.samples.resolve_counter_range(0..count).ok_or(Error::Invalid("encoder timestamp resolve failed"))?;
        if bytes.len() != count * std::mem::size_of::<u64>() {
            return Err(Error::Invalid("encoder timestamp resolved size invalid"));
        }
        let values = bytes.as_chunks::<8>().0.iter().map(|chunk| u64::from_ne_bytes(*chunk)).collect::<Vec<_>>();
        let mut durations = Vec::with_capacity(self.intervals.len());
        let mut total = 0.0;
        for interval in &self.intervals {
            // An end-of-phase checkpoint can open an otherwise empty encoder.
            // Metal may omit its samples; it contributes no dispatched work.
            if !interval.active.get() {
                continue;
            }
            let start = values[interval.start];
            let end = values[interval.end];
            if start == 0
                || end == 0
                || start == MTL_COUNTER_ERROR_VALUE
                || end == MTL_COUNTER_ERROR_VALUE
                || end < start
            {
                return Err(Error::Invalid("encoder timestamp sample invalid"));
            }
            let seconds = (end - start) as f64 * seconds_per_tick;
            total += seconds;
            durations.push((interval.category, seconds));
        }
        // Encoder stages share this one command's GPU interval. Allow 1% or
        // one microsecond for clock calibration and counter granularity.
        if !total.is_finite() || total > command_seconds + (command_seconds * 0.01).max(1e-6) {
            return Err(Error::Invalid("encoder timestamps exceed command interval"));
        }
        for (category, seconds) in durations {
            profile::record(category, seconds, 0);
        }
        profile::record("native.metal.encoder.gpu", total, bytes.len());
        profile::record("native.metal.encoder.command_gap", (command_seconds - total).max(0.0), 0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    #[ignore]
    fn encoder_timing_preserves_values_and_uses_one_command_completion() {
        assert!(DecodeGraph::new_with_timing(false, true, true).is_err());
        for single in [false, true] {
            profile::reset();
            profile::enable(true);
            let mut graph = DecodeGraph::new_with_timing(single, false, true).unwrap();
            graph.serial_projection_levels = false;
            let values = (0..128).map(|i| i as f32 - 16.0).collect::<Vec<_>>();
            let input = graph.upload_into(&values).unwrap();
            let ones = graph.upload_into(&[1.0; 128]).unwrap();
            let doubled = graph.add(&input, &input).unwrap();
            graph.diagnostic_checkpoint("native.metal.encoder.test_first").unwrap();
            graph.begin_projection_level().unwrap();
            let left = graph.add(&doubled, &ones).unwrap();
            let right = graph.add(&doubled, &doubled).unwrap();
            graph.end_projection_level().unwrap();
            let result = graph.add(&left, &right).unwrap();
            graph.diagnostic_checkpoint("native.metal.encoder.test_second").unwrap();
            let actual = graph.finish(&[result]).unwrap().remove(0);
            profile::enable(false);
            assert!(actual.iter().zip(values).all(|(&a, value)| a == value * 6.0 + 1.0));
            let counters = profile::snapshot();
            assert_eq!(counters["native.metal.submit"].calls, 1);
            assert_eq!(counters["native.metal.wait"].calls, 1);
            assert!(counters["native.metal.encoder.test_first"].calls > 0);
            assert!(counters["native.metal.encoder.test_second"].calls > 0);
            let command_seconds = counters["native.metal.gpu"].seconds;
            assert!(
                counters["native.metal.encoder.gpu"].seconds <= command_seconds + (command_seconds * 0.01).max(1e-6)
            );
        }
        let mut invalid = DecodeGraph::new_with_timing(false, false, true).unwrap();
        let poisoned = DecodeGraph::zeros(1).unwrap();
        unsafe { poisoned.buffer.contents().as_ptr().cast::<f32>().write(f32::NAN) };
        invalid.check_finite(&poisoned).unwrap();
        invalid.diagnostic_checkpoint("native.metal.encoder.test_invalid").unwrap();
        assert!(invalid.finish(&[]).is_err(), "encoder timing hid a sticky failure");
    }
}
