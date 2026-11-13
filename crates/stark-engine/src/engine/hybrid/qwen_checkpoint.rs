// SPDX-License-Identifier: Apache-2.0
//! Private native-target checkpoint staging. No public MTP/prefix capability
//! is enabled. Checkpoints contain exact F32 state in RAM only; they have no
//! Debug/serialization implementation. The bounded token history is private
//! model state; no prompt string or vocabulary readout is captured or logged.
//!
//! Integration must charge checkpoint RAM plus complete replacement mutable
//! GPU owners before enabling MTP. They are deliberately absent from today's
//! request planner because no production path calls these private primitives.

use std::sync::Arc;

use super::{Geometry, Mixer, QwenModel, decode_graph};
use crate::engine::hybrid::{Error, Result, qwen_metal, weight_metal::DecodeGraph};

const MAX_CHECKPOINT_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum StagingPolicy {
    /// Target verification uses ordinary single-token steps. Batched-prefill
    /// tolerance is not an exact target-distribution verification policy.
    /// Checkpoints require position > 0 after a completed target advance;
    /// the uninitialized position-zero state is outside this staging policy.
    Edge0MetalSingleStepV1,
}

#[derive(Clone, PartialEq, Eq)]
struct PolicyKey {
    staging: StagingPolicy,
    context: usize,
    sigmoid_output: bool,
    selectors: u64,
}

// These are actual selectors read by qwen_graph.rs, decode_graph.rs and
// weight_metal.rs. They cover GDN staged/fused choice, QSA/PLE scheduling,
// normalization/reduction policy and packed projection/transport diagnostics.
// Values follow the owning source's semantics, not mere variable presence.
const SELECTORS: &[(&str, bool)] = &[
    ("STARK_QWEN38_STAGED_GDN", false),
    ("STARK_QWEN38_SCALAR_ATTENTION", false),
    ("STARK_QWEN38_SCALAR_QSA_PREPARE", false),
    ("STARK_QWEN38_SCALAR_QSA_SELECTION", false),
    ("STARK_QWEN38_SCALAR_PLE", false),
    ("STARK_QWEN38_SERIAL_LEVELS", false),
    ("STARK_QWEN38_PHASE_TIMING", false),
    ("STARK_QWEN38_ENCODER_TIMING", false),
    ("STARK_HYBRID_ORDERED_RMS", false),
    ("STARK_HYBRID_BYTE_LOADS", false),
    ("STARK_HYBRID_VECTOR8", false),
    ("STARK_HYBRID_SCALAR_UINT_LOADS", false),
    ("STARK_HYBRID_FACTORED_AFFINE", false),
    ("STARK_HYBRID_CONCURRENT_FFN", true),
    ("STARK_HYBRID_SEPARATE_FINITE", false),
    ("STARK_GLM53_BATCH_FINITE", false),
    ("STARK_GLM53_FINITE_SELECTIVE", false),
    ("STARK_GLM53_SERIAL_FINITE_ORDER", false),
    ("STARK_GLM53_PARALLEL_RMS", false),
    ("STARK_GLM53_RMS_PREFETCH", false),
    ("STARK_GLM53_PACKED_FAST_MATH", false),
    ("STARK_GLM53_Q8_GROUPED8", false),
    ("STARK_GLM53_Q8_SHAPES", false),
    ("STARK_GLM53_Q4K_BLOCK", false),
    ("STARK_GLM53_KQUANT_FACTORED", false),
    ("STARK_GLM53_Q4_GROUPED32", false),
    ("STARK_GLM53_Q5_GROUPED32", false),
    ("STARK_GLM53_Q6_GROUPED16", false),
    ("STARK_GLM53_IQ_GPU_BOUND", false),
    ("STARK_GLM53_IQ_TYPED16", false),
    ("STARK_GLM53_IQ_SIGNED_FACTOR", false),
    ("STARK_GLM53_IQ_MULTIROW", false),
    ("STARK_GLM53_IQ4_VECTOR", false),
];

impl PolicyKey {
    fn read(
        g: &Geometry,
        staging: StagingPolicy,
    ) -> Result<Self> {
        if !g.edge0 || !g.edge0_metal || g.context == 0 || g.context > 512 {
            return Err(Error::Unsupported("private Edge0 checkpoint staging policy"));
        }
        let mut selectors = 0_u64;
        for (index, &(name, default_on)) in SELECTORS.iter().enumerate() {
            let value = std::env::var_os(name);
            let selected = if default_on {
                value.is_none_or(|value| value != "0")
            } else {
                value.is_some_and(|value| value == "1")
            };
            selectors |= u64::from(selected) << index;
        }
        Ok(Self {
            staging,
            context: g.context,
            sigmoid_output: g.sigmoid_output,
            selectors,
        })
    }
}

pub(super) struct Stamp {
    identity: Arc<()>,
    epoch: u64,
}

impl Stamp {
    pub(super) fn new() -> Self {
        Self {
            identity: Arc::new(()),
            epoch: 0,
        }
    }

    pub(super) fn invalidate(&mut self) {
        if let Some(next) = self.epoch.checked_add(1) {
            self.epoch = next;
        } else {
            self.identity = Arc::new(());
            self.epoch = 0;
        }
    }

    fn matches(
        &self,
        identity: &Arc<()>,
        epoch: u64,
    ) -> bool {
        Arc::ptr_eq(&self.identity, identity) && self.epoch == epoch
    }
}

pub(crate) struct Checkpoint {
    identity: Arc<()>,
    epoch: u64,
    policy: PolicyKey,
    position: usize,
    history: Box<[u32]>,
    recurrent: Vec<Option<qwen_metal::Checkpoint>>,
    graph: Option<decode_graph::Checkpoint>,
    bytes: usize,
}

impl Checkpoint {
    pub(crate) fn position(&self) -> usize {
        self.position
    }
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}

fn fence() -> Result<()> {
    // Target calls are synchronous already. This explicit queue completion
    // also documents the exclusive mutable-owner boundary for host reads.
    DecodeGraph::new()?.finish(&[]).map(|_| ())
}

fn valid_boundary(
    position: usize,
    history: &[u32],
    g: &Geometry,
) -> bool {
    position > 0
        && position <= g.context
        && g.ngram_size >= 2
        && history.len() == position.min(g.ngram_size - 1)
        && history.iter().all(|&token| (token as usize) < g.vocab)
}

impl QwenModel {
    /// Capture a completed positive-position target boundary under the staging policy.
    pub(super) fn capture_checkpoint(
        &mut self,
        staging: StagingPolicy,
    ) -> Result<Checkpoint> {
        let result = (|| {
            let policy = PolicyKey::read(&self.geometry, staging)?;
            if !self.graph_enabled || !valid_boundary(self.position, &self.history, &self.geometry) {
                return Err(Error::Invalid("checkpoint target state boundary"));
            }
            fence()?;
            let graph =
                self.graph.as_ref().map(|state| state.capture_checkpoint(&self.geometry, self.position)).transpose()?;
            if graph.is_none() {
                return Err(Error::Invalid("checkpoint graph state missing"));
            }
            let mut bytes = self.history.len() * 4;
            if let Some(graph) = &graph {
                bytes = bytes.checked_add(graph.bytes()).ok_or(Error::Invalid("checkpoint size overflow"))?;
            }
            let mut recurrent = Vec::with_capacity(self.layers.len());
            for layer in &self.layers {
                let saved = match &layer.mixer {
                    Mixer::Delta(state) => {
                        let value = state.metal.as_ref().map(|state| state.capture_checkpoint()).transpose()?;
                        if value.is_none() {
                            return Err(Error::Invalid("checkpoint recurrent state missing"));
                        }
                        value
                    },
                    Mixer::Attention(_) => None,
                };
                if let Some(state) = &saved {
                    bytes = bytes.checked_add(state.bytes()).ok_or(Error::Invalid("checkpoint size overflow"))?;
                }
                if bytes > MAX_CHECKPOINT_BYTES {
                    return Err(Error::Invalid("checkpoint private RAM cap"));
                }
                recurrent.push(saved);
            }
            let stamp = self.checkpoint_stamp.get_or_insert_with(Stamp::new);
            Ok(Checkpoint {
                identity: stamp.identity.clone(),
                epoch: stamp.epoch,
                policy,
                position: self.position,
                history: self.history.clone().into_boxed_slice(),
                recurrent,
                graph,
                bytes,
            })
        })();
        if result.is_err() {
            // Capture failure invalidates previously issued handles too.
            let _ = fence();
            self.reset();
        }
        result
    }

    /// Restore a completed positive-position boundary from this loaded model and epoch.
    pub(super) fn restore_checkpoint(
        &mut self,
        checkpoint: &Checkpoint,
        staging: StagingPolicy,
    ) -> Result<()> {
        let result = (|| {
            let policy = PolicyKey::read(&self.geometry, staging)?;
            if !self.graph_enabled
                || !self
                    .checkpoint_stamp
                    .as_ref()
                    .is_some_and(|stamp| stamp.matches(&checkpoint.identity, checkpoint.epoch))
                || checkpoint.policy != policy
                || checkpoint.recurrent.len() != self.layers.len()
                || checkpoint.bytes > MAX_CHECKPOINT_BYTES
                || !valid_boundary(checkpoint.position, &checkpoint.history, &self.geometry)
            {
                return Err(Error::Invalid("checkpoint model, epoch or state policy mismatch"));
            }
            fence()?;
            let graph = match (&self.graph, &checkpoint.graph) {
                (Some(source), Some(saved)) => {
                    Some(source.stage_checkpoint(saved, &self.geometry, checkpoint.position)?)
                },
                _ => return Err(Error::Invalid("checkpoint graph restore inventory")),
            };
            let mut replacements = Vec::with_capacity(self.layers.len());
            for (layer, saved) in self.layers.iter().zip(&checkpoint.recurrent) {
                replacements.push(match (&layer.mixer, saved) {
                    (Mixer::Delta(current), Some(saved)) => Some(
                        current
                            .metal
                            .as_ref()
                            .ok_or(Error::Invalid("checkpoint recurrent owner missing"))?
                            .stage_checkpoint(saved)?,
                    ),
                    (Mixer::Attention(_), None) => None,
                    _ => return Err(Error::Invalid("checkpoint recurrent restore inventory")),
                });
            }
            let history = checkpoint.history.to_vec();
            fence()?;
            // Everything fallible completed on independent replacement owners.
            // No callback or GPU submission can observe a partially swapped set.
            for (layer, replacement) in self.layers.iter_mut().zip(replacements) {
                if let Mixer::Delta(state) = &mut layer.mixer {
                    state.metal = replacement;
                }
            }
            self.graph = graph;
            self.position = checkpoint.position;
            self.history = history;
            Ok(())
        })();
        if result.is_err() {
            let _ = fence();
            self.reset();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use stark_engine_macros::stark_test;

    use super::*;
    use crate::engine::hybrid::{Forward, Model};

    /// Bounded actual-target state lifecycle only. Requires the registered
    /// prepared artifact and an externally serialized, resource-watched run.
    /// Token IDs and full readouts remain private in RAM throughout the test.
    #[stark_test]
    // Ignored by default: requires the prepared artifact and a coordinated
    // Metal resource gate, beyond the model-free foundation suite.
    #[ignore]
    fn private_edge0_actual_checkpoint_readouts_and_failure_fences() {
        assert!(
            std::env::var("STARK_EDGE0_REFERENCE").ok().as_deref() == Some("1")
                && std::env::var("STARK_EDGE0_METAL").ok().as_deref() == Some("1")
                && std::env::var("STARK_EDGE0_RESIDENT").ok().as_deref() != Some("1"),
            "checkpoint_actual_policy"
        );
        let path = PathBuf::from(
            std::env::var_os("STARK_EDGE0_SOURCE_PATH").unwrap_or_else(|| panic!("checkpoint_actual_source_required")),
        );
        let mut model = Model::load(&path).unwrap_or_else(|_| panic!("checkpoint_actual_load"));
        let text = "A quiet morning brings clear light through the window. ".repeat(8);
        let encoding = model.tokenizer().encode(text, false).unwrap_or_else(|_| panic!("checkpoint_actual_encode"));
        let tokens = encoding.get_ids().get(..36).unwrap_or_else(|| panic!("checkpoint_actual_corpus_length"));
        let Forward::Qwen(target) = &mut model.forward else {
            panic!("checkpoint_actual_backend");
        };
        assert!(
            target.geometry.edge0 && target.geometry.edge0_metal && target.graph_enabled,
            "checkpoint_actual_target_policy"
        );
        let policy = StagingPolicy::Edge0MetalSingleStepV1;
        target.ingest_prefix(&tokens[..32], || Ok(())).unwrap_or_else(|_| panic!("checkpoint_actual_initial_ingest"));
        let checkpoint = target.capture_checkpoint(policy).unwrap_or_else(|_| panic!("checkpoint_actual_capture"));
        assert!(checkpoint.position() == 32 && target.position == 32, "checkpoint_actual_boundary");
        let mut readouts = Vec::with_capacity(4);
        for &token in &tokens[32..] {
            let row = target.step(token).unwrap_or_else(|_| panic!("checkpoint_actual_initial_step"));
            assert!(
                row.len() == target.geometry.vocab && row.iter().all(|value| value.is_finite()),
                "checkpoint_actual_readout_shape"
            );
            readouts.push(row);
        }
        target.restore_checkpoint(&checkpoint, policy).unwrap_or_else(|_| panic!("checkpoint_actual_restore"));
        assert!(target.position == 32, "checkpoint_actual_restored_position");
        for (&token, expected) in tokens[32..].iter().zip(&readouts) {
            let row = target.step(token).unwrap_or_else(|_| panic!("checkpoint_actual_replay_step"));
            assert!(
                row.len() == expected.len()
                    && row.iter().zip(expected).all(|(value, expected)| value.to_bits() == expected.to_bits()),
                "checkpoint_actual_replay_bits"
            );
        }
        let mut invalid =
            target.capture_checkpoint(policy).unwrap_or_else(|_| panic!("checkpoint_actual_invalid_capture"));
        invalid.position = 0;
        invalid.history = Box::<[u32]>::default();
        assert!(target.restore_checkpoint(&invalid, policy).is_err(), "checkpoint_actual_invalid_admitted");
        assert!(
            target.position == 0
                && target.history.is_empty()
                && target.graph.is_none()
                && target
                    .checkpoint_stamp
                    .as_ref()
                    .is_some_and(|stamp| !stamp.matches(&checkpoint.identity, checkpoint.epoch)),
            "checkpoint_actual_failure_fence"
        );
        assert!(target.restore_checkpoint(&checkpoint, policy).is_err(), "checkpoint_actual_stale_admitted");
        assert!(target.position == 0 && target.graph.is_none(), "checkpoint_actual_stale_fence");
        target.ingest_prefix(&tokens[..32], || Ok(())).unwrap_or_else(|_| panic!("checkpoint_actual_recovery_ingest"));
        for (&token, expected) in tokens[32..].iter().zip(&readouts) {
            let row = target.step(token).unwrap_or_else(|_| panic!("checkpoint_actual_recovery_step"));
            assert!(
                row.len() == expected.len()
                    && row.iter().zip(expected).all(|(value, expected)| value.to_bits() == expected.to_bits()),
                "checkpoint_actual_recovery_bits"
            );
        }
        assert!(target.position == 36, "checkpoint_actual_recovery_position");
    }

    #[stark_test]
    fn private_checkpoint_stamps_reject_cross_model_reset_and_policy_changes() {
        let mut first = Stamp::new();
        let other = Stamp::new();
        let id = first.identity.clone();
        let epoch = first.epoch;
        assert!(first.matches(&id, epoch));
        assert!(!other.matches(&id, epoch));
        first.invalidate();
        assert!(!first.matches(&id, epoch));
        first.epoch = u64::MAX;
        let old_id = first.identity.clone();
        first.invalidate();
        assert!(!first.matches(&old_id, 0));
        let policy = PolicyKey {
            staging: StagingPolicy::Edge0MetalSingleStepV1,
            context: 512,
            sigmoid_output: true,
            selectors: 0,
        };
        let mut changed = policy.clone();
        changed.selectors ^= 1;
        assert!(policy != changed);
        changed = policy.clone();
        changed.sigmoid_output = false;
        assert!(policy != changed);
        changed = policy.clone();
        changed.context = 256;
        assert!(policy != changed);
    }

    #[stark_test]
    fn private_checkpoint_boundary_rejects_zero_and_requires_exact_positive_tail() {
        let g = Geometry {
            context: 512,
            vocab: 20,
            ngram_size: 3,
            ..Geometry::default()
        };
        // Capture and restore share this admission check. Rejecting the empty
        // boundary prevents restoring it from discarding the initialized GPU
        // owners needed by another positive checkpoint in the same epoch.
        assert!(!valid_boundary(0, &[], &g));
        assert!(!valid_boundary(0, &[2], &g));
        assert!(valid_boundary(1, &[2], &g));
        assert!(valid_boundary(2, &[2, 3], &g));
        assert!(valid_boundary(9, &[2, 3], &g));
        assert!(valid_boundary(512, &[2, 3], &g));
        assert!(!valid_boundary(1, &[], &g));
        assert!(!valid_boundary(1, &[2, 3], &g));
        assert!(!valid_boundary(9, &[2], &g));
        assert!(!valid_boundary(9, &[2, 20], &g));
        assert!(!valid_boundary(513, &[2, 3], &g));
    }
}
