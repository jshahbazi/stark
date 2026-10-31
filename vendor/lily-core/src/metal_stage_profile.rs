// SPDX-License-Identifier: Apache-2.0
//! Original Stark diagnostic counters for the vendored Metal 4 decode graph.
//!
//! Precise timestamps preserve the graph's command buffer and explicit
//! dependency levels, but may perturb GPU scheduling. These measurements
//! attribute time; uninstrumented worker runs establish throughput.

use std::{collections::BTreeMap, sync::Mutex, time::Instant};

use anyhow::{Result, anyhow, ensure};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSRange;
use objc2_metal::{
    MTL4CounterHeap, MTL4CounterHeapDescriptor, MTL4CounterHeapType, MTLDevice,
};
use serde::Serialize;

use super::PassTiming;

// 582 entries bound the graph; selected WR1 pairs remove one entry each.
// Fixed diagnostic storage remains bounded; unsupported traces fail closed.
const CAPACITY: usize = 1024;
const MAX_RECORDED_PASSES: usize = 512;

/// Fixed categories only: no tensor names, token IDs or request data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Embedding,
    Ple,
    Hc,
    Gdn,
    GdnInputProjections,
    GdnInputProjectionConv,
    GdnConv,
    GdnRecurrence,
    GdnRecurrenceNorm,
    GdnNorm,
    GdnOutProjection,
    Qsa,
    Moe,
    MoeInputProjections,
    MoeRouteSharedActivation,
    MoeExpertGateUpSharedDown,
    MoeExpertDownCombine,
    Head,
    Draw,
}

impl Phase {
    /// Coarse totals remain inclusive sums of the detailed dependency groups.
    pub const fn coarse(self) -> Self {
        match self {
            Self::GdnInputProjections
            | Self::GdnInputProjectionConv
            | Self::GdnConv
            | Self::GdnRecurrence
            | Self::GdnRecurrenceNorm
            | Self::GdnNorm
            | Self::GdnOutProjection => Self::Gdn,
            Self::MoeInputProjections
            | Self::MoeRouteSharedActivation
            | Self::MoeExpertGateUpSharedDown
            | Self::MoeExpertDownCombine => Self::Moe,
            other => other,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Sample {
    pub phase: Phase,
    pub intervals: usize,
    pub gpu_seconds: f64,
}

/// One verified, completed diagnostic decode pass.
#[derive(Clone, Debug, Serialize)]
pub struct PassProfile {
    pub samples: Vec<Sample>,
    pub timestamp_count: usize,
    pub command_buffers: usize,
    pub measured_seconds: f64,
    pub command_gpu_seconds: f64,
    pub seconds_per_tick: f64,
    pub resolve_seconds: f64,
}

static PASSES: Mutex<Vec<PassProfile>> = Mutex::new(Vec::new());

/// Drain aggregate diagnostic profiles. No profiles exist by default.
pub fn take() -> Vec<PassProfile> {
    PASSES.lock().map(|mut passes| std::mem::take(&mut *passes)).unwrap_or_default()
}

pub(super) fn record(profile: PassProfile) -> Result<()> {
    let mut passes =
        PASSES.lock().map_err(|_| anyhow!("stage profile lock poisoned"))?;
    ensure!(passes.len() < MAX_RECORDED_PASSES, "stage profile queue full");
    passes.push(profile);
    Ok(())
}

pub(super) struct Trace {
    pub heap: Retained<ProtocolObject<dyn MTL4CounterHeap>>,
    phases: Vec<Option<Phase>>,
    finished: bool,
    seconds_per_tick: f64,
}

impl Trace {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let descriptor = MTL4CounterHeapDescriptor::new();
        descriptor.setType(MTL4CounterHeapType::Timestamp);
        // SAFETY: the fixed positive entry count is also checked before every
        // write and resolve below. The heap belongs to exactly one pass.
        unsafe { descriptor.setCount(CAPACITY) };
        let heap = device
            .newCounterHeapWithDescriptor_error(&descriptor)
            .map_err(|_| anyhow!("stage timestamp heap unavailable"))?;
        ensure!(heap.count() == CAPACITY, "stage timestamp heap capacity");
        ensure!(
            device.sizeOfCounterHeapEntry(MTL4CounterHeapType::Timestamp) == 8,
            "stage timestamp entry size"
        );
        let frequency = device.queryTimestampFrequency();
        ensure!(frequency > 0, "stage timestamp frequency");
        Ok(Self {
            heap,
            phases: Vec::new(),
            finished: false,
            seconds_per_tick: 1.0 / frequency as f64,
        })
    }

    pub fn push(&mut self, phase: Phase) -> Result<usize> {
        ensure!(!self.finished, "stage profile already finished");
        // Reserve one entry for the terminal timestamp.
        ensure!(self.phases.len() < CAPACITY - 1, "stage profile capacity exceeded");
        let index = self.phases.len();
        self.phases.push(Some(phase));
        Ok(index)
    }

    pub fn finish(&mut self) -> Result<usize> {
        ensure!(
            !self.finished && !self.phases.is_empty(),
            "stage profile terminal state"
        );
        ensure!(self.phases.len() < CAPACITY, "stage profile capacity exceeded");
        let index = self.phases.len();
        self.phases.push(None);
        self.finished = true;
        Ok(index)
    }

    pub fn resolve(
        self,
        timing: PassTiming,
        command_buffers: usize,
    ) -> Result<PassProfile> {
        let started = Instant::now();
        ensure!(self.finished, "stage profile missing terminal timestamp");
        ensure!(command_buffers == 1, "stage profile changed command buffer count");
        // SAFETY: all timestamps have completed verified feedback before this
        // call. Each pass has its own heap, and the range is bounded by CAPACITY.
        let data = unsafe {
            self.heap.resolveCounterRange(NSRange::new(0, self.phases.len()))
        }
        .ok_or_else(|| anyhow!("stage timestamp resolution failed"))?;
        let bytes = data.to_vec();
        ensure!(bytes.len() == self.phases.len() * 8, "stage timestamp result size");
        let timestamps = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| u64::from_ne_bytes(*chunk))
            .collect::<Vec<_>>();
        let mut result = profile_from_ticks(
            &self.phases,
            &timestamps,
            timing,
            self.seconds_per_tick,
            command_buffers,
        )?;
        result.resolve_seconds = started.elapsed().as_secs_f64();
        Ok(result)
    }
}

fn profile_from_ticks(
    phases: &[Option<Phase>],
    ticks: &[u64],
    timing: PassTiming,
    scale: f64,
    command_buffers: usize,
) -> Result<PassProfile> {
    ensure!(
        phases.len() == ticks.len()
            && (2..=CAPACITY).contains(&ticks.len())
            && phases.last() == Some(&None)
            && phases[..phases.len() - 1].iter().all(Option::is_some),
        "stage timestamp shape"
    );
    ensure!(scale.is_finite() && scale > 0.0, "stage timestamp scale");
    ensure!(command_buffers == 1, "stage profile changed command buffer count");
    ensure!(
        timing.gpu_start_secs.is_finite()
            && timing.gpu_end_secs.is_finite()
            && timing.gpu_start_secs > 0.0
            && timing.gpu_end_secs > timing.gpu_start_secs,
        "stage command timestamp span"
    );
    ensure!(
        ticks[0] > 0 && ticks.windows(2).all(|pair| pair[1] >= pair[0]),
        "stage timestamp order"
    );
    // Compare elapsed spans only. queryTimestampFrequency defines the GPU
    // tick scale without promising that its absolute epoch matches feedback.
    let measured_seconds = (ticks[ticks.len() - 1] - ticks[0]) as f64 * scale;
    let command_gpu_seconds = timing.gpu_end_secs - timing.gpu_start_secs;
    ensure!(
        measured_seconds.is_finite()
            && measured_seconds > 0.0
            && measured_seconds <= command_gpu_seconds * 1.1 + 0.0001
            && measured_seconds >= command_gpu_seconds * 0.5,
        "stage timestamp duration"
    );
    let mut grouped = BTreeMap::<Phase, (usize, f64)>::new();
    for (phase, pair) in phases.iter().zip(ticks.windows(2)) {
        let phase =
            phase.ok_or_else(|| anyhow!("stage timestamp terminal position"))?;
        let entry = grouped.entry(phase).or_default();
        entry.0 += 1;
        entry.1 += (pair[1] - pair[0]) as f64 * scale;
    }
    Ok(PassProfile {
        samples: grouped
            .into_iter()
            .map(|(phase, (intervals, gpu_seconds))| Sample {
                phase,
                intervals,
                gpu_seconds,
            })
            .collect(),
        timestamp_count: ticks.len(),
        command_buffers,
        measured_seconds,
        command_gpu_seconds,
        seconds_per_tick: scale,
        resolve_seconds: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_ticks_aggregate_repeated_categories() {
        let result = profile_from_ticks(
            &[Some(Phase::Hc), Some(Phase::Gdn), Some(Phase::Hc), None],
            &[1000, 1100, 1400, 1500],
            PassTiming { gpu_start_secs: 0.9, gpu_end_secs: 1.6 },
            0.001,
            1,
        )
        .unwrap();
        assert_eq!(result.samples.len(), 2);
        assert_eq!(result.samples[0].phase, Phase::Hc);
        assert_eq!(result.samples[0].intervals, 2);
        assert!((result.samples[0].gpu_seconds - 0.2).abs() < 1e-12);
        assert!((result.measured_seconds - 0.5).abs() < 1e-12);
    }

    #[test]
    fn detailed_checkpoint_trace_fits_and_preserves_coarse_totals() {
        // Unpaired official upper bound: 48 layers,36 GDN,12 attention,one PLE.
        // Six coarse intervals/layer, four more/GDN and three more/MoE;
        // embedding, final HC, head, draw, PLE and terminal add six entries.
        let (layers, gdn, attention) = (48, 36, 12);
        assert_eq!(gdn + attention, layers);
        let budget = |inject: bool,
                      head: bool,
                      ple: bool,
                      split_norm: bool,
                      fused_conv: bool| {
            // Two HC reads + four MoE groups per layer. QSA remains coarse.
            let gdn_groups = 4 + usize::from(split_norm) - usize::from(fused_conv);
            4 + usize::from(head)
                + usize::from(ple)
                + layers * (6 + 2 * usize::from(inject))
                + gdn * gdn_groups
                + attention
        };
        for inject in [false, true] {
            for head in [false, true] {
                for ple in [false, true] {
                    for split_norm in [false, true] {
                        for fused_conv in [false, true] {
                            assert!(
                                budget(inject, head, ple, split_norm, fused_conv)
                                    <= CAPACITY
                            );
                        }
                    }
                }
            }
        }
        let count = budget(true, true, true, true, false);
        assert_eq!(count, 582);
        let mut phases = vec![Some(Phase::GdnRecurrence); count - 1];
        phases.push(None);
        let ticks = (0..count).map(|i| 1000 + i as u64).collect::<Vec<_>>();
        let result = profile_from_ticks(
            &phases,
            &ticks,
            PassTiming { gpu_start_secs: 1.0, gpu_end_secs: 1.6 },
            0.001,
            1,
        )
        .unwrap();
        assert_eq!(result.timestamp_count, count);
        assert_eq!(result.samples.len(), 1);
        assert_eq!(result.samples[0].phase.coarse(), Phase::Gdn);
        assert_eq!(result.samples[0].intervals, count - 1);
        assert_eq!(Phase::MoeExpertGateUpSharedDown.coarse(), Phase::Moe);
        assert_eq!(Phase::Hc.coarse(), Phase::Hc);
    }

    #[test]
    fn malformed_or_uncalibrated_stage_ticks_are_rejected() {
        let timing = PassTiming { gpu_start_secs: 1.0, gpu_end_secs: 1.1 };
        let phases = [Some(Phase::Head), None];
        for ticks in [&[0, 1][..], &[1001, 1000][..], &[1000][..], &[10, 11][..]] {
            assert!(profile_from_ticks(&phases, ticks, timing, 0.001, 1).is_err());
        }
        assert!(
            profile_from_ticks(&phases, &[1000, 1050], timing, f64::NAN, 1).is_err()
        );
        assert!(
            profile_from_ticks(
                &[Some(Phase::Head), Some(Phase::Draw)],
                &[1000, 1050],
                timing,
                0.001,
                1
            )
            .is_err()
        );
    }
}
