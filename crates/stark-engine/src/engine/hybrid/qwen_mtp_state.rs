// SPDX-License-Identifier: Apache-2.0
//! Private, RAM-only target/head/pending-wide transactions. Every failure
//! fences the head and resets the target. Successful restoration creates a
//! new branch tag and binds only the wide restored from this sealed snapshot.

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use super::*;

#[cfg(test)]
#[path = "qwen_mtp_state_actual_tests.rs"]
mod actual_tests;

pub(super) const HEAD_DYNAMIC_BYTES: usize = 512 * 1024 * 1024;

// Three graph segments, bounded host outputs, immutable norms and live K/V.
// The 16 MiB reserve includes the compact eight-row target prefix wide
// (at most 320 KiB), dropped before admitting the next prefix batch.
// The remaining admission is shared by snapshots and staged replacements.
pub(super) fn head_live_bytes(g: HeadGeometry) -> usize {
    3 * SCRATCH_BYTES + 16 * 1024 * 1024 + g.context * (2 * g.kv_heads * g.head_dim + 2) * 4
}

pub(super) fn head_replacement_bytes(g: HeadGeometry) -> usize {
    // Host staging plus independent replacement device owners coexist.
    g.context * (4 * g.kv_heads * g.head_dim + 1) * 4
}

struct LeaseCore {
    admission: Arc<AtomicBool>,
    used: AtomicUsize,
    cap: usize,
    drain: fn() -> bool,
}

impl Drop for LeaseCore {
    fn drop(&mut self) {
        // A failed drain leaves admission fenced for the loaded model. It
        // cannot admit a second head over uncertain in-flight GPU resources.
        if (self.drain)() {
            self.admission.store(false, Ordering::Release);
        }
    }
}

pub(super) struct HeadLease(Arc<LeaseCore>);

impl HeadLease {
    pub(super) fn new(admission: Arc<AtomicBool>) -> Self {
        Self(Arc::new(LeaseCore {
            admission,
            used: AtomicUsize::new(0),
            cap: HEAD_DYNAMIC_BYTES,
            drain: || new_graph().and_then(|graph| graph.finish(&[])).is_ok(),
        }))
    }

    pub(super) fn reserve(
        &self,
        bytes: usize,
    ) -> Result<Reservation> {
        self.0
            .used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|&total| total <= self.0.cap)
            })
            .map_err(|_| Error::Invalid("private MTP live resource allowance"))?;
        Ok(Reservation {
            lease: self.0.clone(),
            bytes,
        })
    }

    fn owns(
        &self,
        model: &QwenModel,
    ) -> bool {
        model.weights.owns_edge0_mtp_head(&self.0.admission)
    }
}

/// Every live snapshot retains both its byte charge and admission. Dropping a
/// head while snapshots survive cannot admit a second concurrent resource set.
pub(in crate::engine::hybrid) struct Reservation {
    lease: Arc<LeaseCore>,
    bytes: usize,
}

impl Reservation {
    fn shrink(
        &mut self,
        bytes: usize,
    ) -> Result<()> {
        if bytes > self.bytes {
            return Err(Error::Invalid("private MTP reservation growth"));
        }
        self.lease.used.fetch_sub(self.bytes - bytes, Ordering::AcqRel);
        self.bytes = bytes;
        Ok(())
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.lease.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

impl Head {
    pub(super) fn reserve(
        &self,
        bytes: usize,
    ) -> Result<Option<Reservation>> {
        self.lease.as_ref().map(|lease| lease.reserve(bytes)).transpose()
    }

    pub(in crate::engine::hybrid) fn fence_target(
        &mut self,
        model: &mut QwenModel,
    ) {
        self.fence();
        model.reset();
    }

    /// Reuse the existing admitted weights after a target reset. A surviving
    /// checkpoint remains charged but becomes unusable through the head epoch.
    /// Positive target boundaries require the combined restore operation.
    pub(in crate::engine::hybrid) fn rebind_reset(
        &mut self,
        model: &mut QwenModel,
    ) -> Result<()> {
        let result = (|| {
            if model.position != 0 || !self.lease.as_ref().is_some_and(|lease| lease.owns(model)) {
                return Err(Error::Invalid("private MTP reset binding"));
            }
            let _reservation = self.reserve(head_replacement_bytes(self.geometry))?;
            self.reset()?;
            self.target_tag = Some(model.checkpoint_state_tag(checkpoint::StagingPolicy::Edge0MetalSingleStepV1)?);
            Ok(())
        })();
        if result.is_err() {
            self.fence_target(model);
        }
        result
    }
}

pub(in crate::engine::hybrid) struct CombinedCheckpoint {
    target: checkpoint::Checkpoint,
    head: HeadCheckpoint,
    tag: checkpoint::StateTag,
    row: usize,
    wide: Box<[f32]>,
    _resource_reservation: Option<Reservation>,
}

fn valid_boundary(
    position: usize,
    cell: usize,
    row: usize,
    width: usize,
    g: HeadGeometry,
) -> bool {
    row.checked_add(1) == Some(position) && row == cell && position <= g.context && width == g.streams * g.hidden
}

impl CombinedCheckpoint {
    pub(in crate::engine::hybrid) fn capture(
        model: &mut QwenModel,
        head: &mut Head,
        pending: &WideResidual,
    ) -> Result<Self> {
        let result = (|| {
            let policy = checkpoint::StagingPolicy::Edge0MetalSingleStepV1;
            if head.fenced
                || !head.lease.as_ref().is_some_and(|lease| lease.owns(model))
                || !head.target_tag.as_ref().is_some_and(|tag| tag.same(&pending.tag))
                || !model.checkpoint_state_tag_matches(&pending.tag, policy)?
                || !valid_boundary(model.position, head.next_cell, pending.row, pending.values.len, head.geometry)
            {
                return Err(Error::Invalid("private MTP combined capture boundary"));
            }
            // Reserve before any target host copy. The bounded controller
            // keeps one combined snapshot; additional simultaneous snapshots
            // may be refused by this conservative shared resource allowance.
            let wide_bytes = pending.values.len * 4;
            let mut reservation = head.reserve(checkpoint::MAX_CHECKPOINT_BYTES + wide_bytes * 2)?;
            let target = model.capture_checkpoint(policy)?;
            let saved_head = head.capture_checkpoint()?;
            let mut rows = new_graph()?.finish(std::slice::from_ref(&pending.values))?;
            if rows.len() != 1 {
                return Err(Error::Invalid("private MTP combined wide inventory"));
            }
            let wide = rows.pop().ok_or(Error::Invalid("private MTP combined wide missing"))?;
            if wide.len() != pending.values.len || wide.iter().any(|value| !value.is_finite()) {
                return Err(Error::Invalid("private MTP combined wide values"));
            }
            if let Some(reservation) = &mut reservation {
                reservation.shrink(target.bytes() + wide_bytes)?;
            }
            Ok(Self {
                target,
                head: saved_head,
                tag: pending.tag.clone(),
                row: pending.row,
                wide: wide.into_boxed_slice(),
                _resource_reservation: reservation,
            })
        })();
        if result.is_err() {
            head.fence_target(model);
        }
        result
    }

    pub(in crate::engine::hybrid) fn restore(
        &self,
        model: &mut QwenModel,
        head: &mut Head,
    ) -> Result<WideResidual> {
        let result = (|| {
            if !head.lease.as_ref().is_some_and(|lease| lease.owns(model))
                || !self.target.contains_state_tag(&self.tag)
                || !self.head.target_tag.as_ref().is_some_and(|tag| tag.same(&self.tag))
                || !valid_boundary(
                    self.target.position(),
                    self.head.next_cell,
                    self.row,
                    self.wide.len(),
                    head.geometry,
                )
                || self.wide.iter().any(|value| !value.is_finite())
            {
                return Err(Error::Invalid("private MTP combined restore boundary"));
            }
            let bytes = model
                .checkpoint_replacement_bytes(&self.target)?
                .checked_add(head_replacement_bytes(head.geometry))
                .and_then(|bytes| bytes.checked_add(self.wide.len() * 4))
                .ok_or(Error::Invalid("private MTP combined restore inventory"))?;
            let _reservation = head.reserve(bytes)?;
            let wide_reservation = head.reserve(self.wide.len() * 4)?;
            // Head and pending replacements are independent. Target restoration
            // stages all recurrent/graph owners and completes its queue fence
            // before its swap. Only infallible owner assignments follow below.
            let replacement = head.stage_checkpoint(&self.head)?;
            let values = DecodeGraph::upload(&self.wide)?;
            new_graph()?.finish(&[])?;
            let policy = checkpoint::StagingPolicy::Edge0MetalSingleStepV1;
            model.restore_checkpoint(&self.target, policy)?;
            let tag = model.checkpoint_state_tag(policy)?;
            head.cache = replacement;
            head.next_cell = self.head.next_cell;
            head.target_tag = Some(tag.clone());
            Ok(WideResidual {
                values,
                tag,
                row: self.row,
                _resource_reservation: wide_reservation,
            })
        })();
        if result.is_err() {
            head.fence_target(model);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    fn lease(
        cap: usize,
        drained: bool,
    ) -> (HeadLease, Arc<AtomicBool>) {
        let admission = Arc::new(AtomicBool::new(true));
        (
            HeadLease(Arc::new(LeaseCore {
                admission: admission.clone(),
                used: AtomicUsize::new(0),
                cap,
                drain: if drained {
                    || true
                } else {
                    || false
                },
            })),
            admission,
        )
    }

    #[stark_test]
    fn edge0_mtp_lease_counts_live_snapshots_and_waits_for_last_owner() {
        let (lease, admission) = lease(100, true);
        let live = lease.reserve(60).unwrap();
        let mut saved = lease.reserve(40).unwrap();
        assert!(lease.reserve(1).is_err());
        assert!(lease.reserve(usize::MAX).is_err());
        saved.shrink(20).unwrap();
        assert!(saved.shrink(21).is_err());
        let replacement = lease.reserve(20).unwrap();
        drop(live);
        assert!(lease.0.used.load(Ordering::Acquire) == 40);
        drop(lease);
        assert!(admission.load(Ordering::Acquire));
        assert!(admission.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err());
        drop(saved);
        assert!(admission.load(Ordering::Acquire));
        drop(replacement);
        assert!(!admission.load(Ordering::Acquire));
        assert!(admission.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok());
    }

    #[stark_test]
    fn edge0_mtp_failed_drain_keeps_admission_fenced() {
        let (lease, admission) = lease(100, false);
        drop(lease);
        assert!(admission.load(Ordering::Acquire));
    }

    #[stark_test]
    fn edge0_mtp_combined_boundary_requires_exact_pending_target_row() {
        let g = HeadGeometry {
            hidden: 64,
            intermediate: 64,
            streams: 4,
            low_rank: 4,
            heads: 2,
            kv_heads: 1,
            head_dim: 8,
            rotary: 4,
            experts: 2,
            active: 1,
            vocabulary: 8,
            context: 32,
            rope_theta: 10000.0,
        };
        assert!(valid_boundary(1, 0, 0, 256, g));
        assert!(valid_boundary(32, 31, 31, 256, g));
        for (position, cell, row, width) in [
            (0, 0, 0, 256),
            (32, 30, 31, 256),
            (33, 32, 32, 256),
            (32, 30, 30, 256),
            (32, 31, 31, 255),
            (0, usize::MAX, usize::MAX, 256),
        ] {
            assert!(!valid_boundary(position, cell, row, width, g));
        }
    }
}
