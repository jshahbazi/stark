// SPDX-License-Identifier: Apache-2.0
//! Private Edge0 head diagnostic. Equations follow Edge0-AI/Edge0
//! b7900cbb322926ccebdcd0c0c5a1280497ecff22, src/core/mtp.cpp.
//! Stored weights stay unchanged; activations, reductions and K/V are F32.
//! This component neither samples nor advertises MTP or prefix reuse.
//! Dense draft attention is bounded to 512 cells. No sparse indexer is used.

#![allow(dead_code, reason = "private executable-head diagnostic awaiting numerical qualification")]

#[cfg(test)]
#[path = "qwen_mtp_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "qwen_mtp_actual_tests.rs"]
mod actual_tests;

use std::{collections::BTreeMap, sync::Arc};

use super::{Geometry, QwenModel, checkpoint, edge0};
use crate::engine::hybrid::{
    Error, Result,
    weight_metal::{DecodeGraph, DeviceVector},
    weights::Weights,
};

const SCRATCH_BYTES: usize = 64 * 1024 * 1024;
const EPS: f32 = 1e-6;

#[derive(Clone, Copy)]
pub(in crate::engine::hybrid) struct HeadGeometry {
    pub hidden: usize,
    pub intermediate: usize,
    pub streams: usize,
    pub low_rank: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary: usize,
    pub experts: usize,
    pub active: usize,
    pub vocabulary: usize,
    pub context: usize,
    pub rope_theta: f32,
}

impl HeadGeometry {
    pub(in crate::engine::hybrid) fn validate(self) -> Result<Self> {
        if self.hidden == 0
            || self.hidden > 2560
            || !self.hidden.is_multiple_of(64)
            || self.intermediate == 0
            || self.intermediate > 640
            || !self.intermediate.is_multiple_of(64)
            || self.streams != 4
            || self.low_rank == 0
            || self.low_rank > 320
            || self.heads == 0
            || self.heads > 24
            || self.kv_heads == 0
            || self.kv_heads > 2
            || !self.heads.is_multiple_of(self.kv_heads)
            || self.head_dim == 0
            || self.head_dim > 256
            || self.rotary == 0
            || self.rotary > self.head_dim
            || !self.rotary.is_multiple_of(2)
            || self.experts == 0
            || self.experts > 512
            || self.active == 0
            || self.active > self.experts
            || self.vocabulary == 0
            || self.vocabulary > 248_320
            || self.context == 0
            || self.context > 512
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
        {
            return Err(Error::Invalid("private MTP head geometry"));
        }
        Ok(self)
    }

    fn target(g: &Geometry) -> Result<Self> {
        if !g.edge0 || !g.edge0_metal || g.dim != 2560 || g.streams != 4 || g.expert_dim != 640 {
            return Err(Error::Unsupported("private MTP target geometry"));
        }
        Self {
            hidden: g.dim,
            intermediate: g.expert_dim,
            streams: g.streams,
            low_rank: 320,
            heads: g.heads,
            kv_heads: g.kv_heads,
            head_dim: g.head_dim,
            rotary: g.rotary_dim,
            experts: g.experts,
            active: g.active_experts,
            vocabulary: g.vocab,
            context: g.context,
            rope_theta: g.rope_theta,
        }
        .validate()
    }
}

/// The same orchestration accepts checked actual mappings or sealed tiny raw
/// fixture weights. Gate/up returns the fused SwiGLU result, not packed rows.
pub(in crate::engine::hybrid) trait HeadWeights {
    fn norm(
        &self,
        name: &str,
    ) -> Result<Vec<f32>>;
    fn shape(
        &self,
        name: &str,
    ) -> Result<[usize; 2]>;
    fn project(
        &self,
        graph: &mut DecodeGraph,
        name: &str,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector>;
    fn expert_gate_up(
        &self,
        graph: &mut DecodeGraph,
        expert: usize,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector>;
    fn expert_down(
        &self,
        graph: &mut DecodeGraph,
        expert: usize,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector>;
    fn embedding(
        &self,
        token: u32,
    ) -> Result<Vec<f32>>;
    fn readout(
        &self,
        graph: &mut DecodeGraph,
        input: &DeviceVector,
    ) -> Result<DeviceVector>;
}

pub(in crate::engine::hybrid) struct ActualWeights<'a> {
    pub weights: &'a Weights,
    pub embedding: &'a str,
    pub readout: &'a str,
}

impl HeadWeights for ActualWeights<'_> {
    fn norm(
        &self,
        name: &str,
    ) -> Result<Vec<f32>> {
        self.weights.edge0_mtp_vector(name)
    }
    fn shape(
        &self,
        name: &str,
    ) -> Result<[usize; 2]> {
        Ok(self.weights.edge0_mtp_dense(name)?.shape())
    }
    fn project(
        &self,
        graph: &mut DecodeGraph,
        name: &str,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        self.weights.encode_edge0_mtp_dense(graph, name, input, tokens)
    }
    fn expert_gate_up(
        &self,
        graph: &mut DecodeGraph,
        expert: usize,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        self.weights.encode_edge0_mtp_gate_up(graph, expert, input, tokens)
    }
    fn expert_down(
        &self,
        graph: &mut DecodeGraph,
        expert: usize,
        input: &DeviceVector,
        tokens: usize,
    ) -> Result<DeviceVector> {
        self.weights.encode_edge0_mtp_down(graph, expert, input, tokens)
    }
    fn embedding(
        &self,
        token: u32,
    ) -> Result<Vec<f32>> {
        self.weights.row(self.embedding, token as usize)
    }
    fn readout(
        &self,
        graph: &mut DecodeGraph,
        input: &DeviceVector,
    ) -> Result<DeviceVector> {
        self.weights.encode_edge0_named_batch(graph, self.readout, input, 1)
    }
}

/// Dedicated device storage; never retains a target graph scratch arena.
/// No Debug/serialization implementation exposes private activations.
pub(in crate::engine::hybrid) struct WideResidual {
    pub(super) values: DeviceVector,
    pub(super) tag: checkpoint::StateTag,
    pub(super) row: usize,
}

pub(in crate::engine::hybrid) struct HeadOutput {
    pub wide: Option<DeviceVector>,
    pub logits: Option<Vec<f32>>,
    pub routed_ids: Vec<usize>,
    /// Explicit diagnostic-only host captures, private in RAM.
    pub stages: BTreeMap<String, Vec<f32>>,
}

struct Cache {
    keys: DeviceVector,
    values: DeviceVector,
    dummy_index: DeviceVector,
    visible: DeviceVector,
}

pub(in crate::engine::hybrid) struct Head {
    geometry: HeadGeometry,
    norms: BTreeMap<String, DeviceVector>,
    cache: Cache,
    next_cell: usize,
    fenced: bool,
    target_tag: Option<checkpoint::StateTag>,
    identity: Arc<()>,
    epoch: u64,
    selectors: u64,
}

/// Exact valid-prefix K/V snapshot. No scratch, visibility mask, immutable
/// norm or raw weights are serialized; this remains private RAM state.
pub(in crate::engine::hybrid) struct HeadCheckpoint {
    identity: Arc<()>,
    epoch: u64,
    geometry: [usize; 14],
    selectors: u64,
    next_cell: usize,
    keys: Box<[f32]>,
    values: Box<[f32]>,
}

impl HeadCheckpoint {
    pub(in crate::engine::hybrid) fn bytes(&self) -> usize {
        (self.keys.len() + self.values.len()) * 4
    }
}

fn new_graph() -> Result<DecodeGraph> {
    let mut graph = DecodeGraph::new()?;
    graph.set_scratch_limit(SCRATCH_BYTES)?;
    Ok(graph)
}

fn checked(
    values: DeviceVector,
    length: usize,
) -> Result<DeviceVector> {
    if values.len != length {
        return Err(Error::Invalid("private MTP projected shape"));
    }
    Ok(values)
}

fn trace(
    stages: &mut BTreeMap<String, DeviceVector>,
    enabled: bool,
    name: &str,
    values: &DeviceVector,
) {
    if enabled {
        stages.insert(name.to_owned(), values.clone());
    }
}

fn join(
    graph: &mut DecodeGraph,
    rows: &[DeviceVector],
) -> Result<DeviceVector> {
    let width = rows.first().ok_or(Error::Invalid("private MTP empty rows"))?.len;
    if rows.iter().any(|row| row.len != width) {
        return Err(Error::Invalid("private MTP row shape"));
    }
    let result = graph.scratch(width * rows.len())?;
    for (index, row) in rows.iter().enumerate() {
        graph.copy_into(row, &result.slice(index * width, width)?)?;
    }
    Ok(result)
}

impl Head {
    pub(in crate::engine::hybrid) fn new(
        g: HeadGeometry,
        weights: &impl HeadWeights,
    ) -> Result<Self> {
        let g = g.validate()?;
        let mut norms = BTreeMap::new();
        for (prefix, injecting) in
            [("attn_hyper_connection", true), ("mlp_hyper_connection", true), ("hyper_connection_mixer", false)]
        {
            for (suffix, expected) in [
                ("hc_norm.weight", [1, g.streams * g.hidden]),
                ("input_mix_weight_down.weight", [g.low_rank, g.streams * g.hidden]),
                ("input_mix_weight_up.weight", [g.streams * g.hidden, g.low_rank]),
            ] {
                if weights.shape(&format!("{prefix}.{suffix}"))? != expected {
                    return Err(Error::Invalid("private MTP HC weight shape"));
                }
            }
            if injecting
                && weights.shape(&format!("{prefix}.block_inject_weight.weight"))? != [g.streams, g.streams * g.hidden]
            {
                return Err(Error::Invalid("private MTP injection shape"));
            }
        }
        for (name, expected) in [
            ("fc_embedding.weight", [g.hidden, g.hidden]),
            ("fc_hidden.weight", [g.hidden, g.hidden]),
            ("self_attn.q_proj.weight", [2 * g.heads * g.head_dim, g.hidden]),
            ("self_attn.k_proj.weight", [g.kv_heads * g.head_dim, g.hidden]),
            ("self_attn.v_proj.weight", [g.kv_heads * g.head_dim, g.hidden]),
            ("self_attn.o_proj.weight", [g.hidden, g.heads * g.head_dim]),
            ("mlp.gate.weight", [g.experts, g.hidden]),
            ("mlp.shared_expert_gate.weight", [1, g.hidden]),
            ("mlp.shared_expert.gate_proj.weight", [g.intermediate, g.hidden]),
            ("mlp.shared_expert.up_proj.weight", [g.intermediate, g.hidden]),
            ("mlp.shared_expert.down_proj.weight", [g.hidden, g.intermediate]),
        ] {
            if weights.shape(name)? != expected {
                return Err(Error::Invalid("private MTP weight shape"));
            }
        }
        for (name, length) in [
            ("pre_fc_norm_hidden.weight", g.streams * g.hidden),
            ("pre_fc_norm_embedding.weight", g.hidden),
            ("attn_hyper_connection.hc_norm.weight", g.streams * g.hidden),
            ("mlp_hyper_connection.hc_norm.weight", g.streams * g.hidden),
            ("hyper_connection_mixer.hc_norm.weight", g.streams * g.hidden),
            ("self_attn.q_norm.weight", g.head_dim),
            ("self_attn.k_norm.weight", g.head_dim),
        ] {
            let values = weights.norm(name)?;
            if values.len() != length || values.iter().any(|v| !v.is_finite()) {
                return Err(Error::Invalid("private MTP norm values"));
            }
            norms.insert(name.to_owned(), DecodeGraph::upload(&values)?);
        }
        Ok(Self {
            geometry: g,
            norms,
            cache: Cache {
                keys: DecodeGraph::zeros(g.context * g.kv_heads * g.head_dim)?,
                values: DecodeGraph::zeros(g.context * g.kv_heads * g.head_dim)?,
                dummy_index: DecodeGraph::zeros(g.context)?,
                visible: DecodeGraph::upload(&vec![f32::from_bits(1); g.context])?,
            },
            next_cell: 0,
            fenced: false,
            target_tag: None,
            identity: Arc::new(()),
            epoch: 0,
            selectors: checkpoint::state_selector_bits(),
        })
    }

    pub(in crate::engine::hybrid) fn for_target(model: &mut QwenModel) -> Result<Self> {
        let g = HeadGeometry::target(&model.geometry)?;
        // Admission precedes head allocation and any MTP raw projection.
        model.weights.prepare_edge0_mtp_head(512 * 1024 * 1024)?;
        let mut head = Self::new(
            g,
            &ActualWeights {
                weights: &model.weights,
                embedding: &model.embedding,
                readout: &model.readout,
            },
        )?;
        head.target_tag = Some(model.checkpoint_state_tag(checkpoint::StagingPolicy::Edge0MetalSingleStepV1)?);
        Ok(head)
    }

    pub(in crate::engine::hybrid) fn next_cell(&self) -> usize {
        self.next_cell
    }
    pub(in crate::engine::hybrid) fn is_fenced(&self) -> bool {
        self.fenced
    }

    fn checkpoint_geometry(&self) -> [usize; 14] {
        let g = self.geometry;
        [
            g.hidden,
            g.intermediate,
            g.streams,
            g.low_rank,
            g.heads,
            g.kv_heads,
            g.head_dim,
            g.rotary,
            g.experts,
            g.active,
            g.vocabulary,
            g.context,
            g.rope_theta.to_bits() as usize,
            EPS.to_bits() as usize,
        ]
    }

    fn policy_matches(&self) -> bool {
        self.selectors == checkpoint::state_selector_bits()
    }

    fn fence(&mut self) {
        self.fenced = true;
        self.next_cell = 0;
        if let Some(epoch) = self.epoch.checked_add(1) {
            self.epoch = epoch;
        } else {
            self.identity = Arc::new(());
            self.epoch = 0;
        }
        let _ = new_graph().and_then(|graph| graph.finish(&[]));
    }

    /// Clear draft state and invalidate every previously issued handle. A
    /// target reset changes its tag. Current one-shot diagnostic admission
    /// requires a model reload for a fresh binding; resource leasing and
    /// transactional target/head rebinding are not integrated yet.
    pub(in crate::engine::hybrid) fn reset(&mut self) -> Result<()> {
        self.fence();
        if !self.policy_matches() {
            return Err(Error::Invalid("private MTP reset policy"));
        }
        let g = self.geometry;
        let replacement = Cache {
            keys: DecodeGraph::zeros(g.context * g.kv_heads * g.head_dim)?,
            values: DecodeGraph::zeros(g.context * g.kv_heads * g.head_dim)?,
            dummy_index: DecodeGraph::zeros(g.context)?,
            visible: self.cache.visible.clone(),
        };
        new_graph()?.finish(&[])?;
        self.cache = replacement;
        self.fenced = false;
        Ok(())
    }

    pub(in crate::engine::hybrid) fn capture_checkpoint(&mut self) -> Result<HeadCheckpoint> {
        let result = (|| {
            if self.fenced || !self.policy_matches() {
                return Err(Error::Invalid("private MTP checkpoint policy"));
            }
            let used = self.next_cell * self.geometry.kv_heads * self.geometry.head_dim;
            let rows = if used == 0 {
                new_graph()?.finish(&[])?;
                Vec::new()
            } else {
                new_graph()?.finish(&[self.cache.keys.slice(0, used)?, self.cache.values.slice(0, used)?])?
            };
            let (keys, values) = if used == 0 {
                (Box::<[f32]>::default(), Box::<[f32]>::default())
            } else {
                if rows.len() != 2 || rows.iter().any(|row| row.len() != used || row.iter().any(|v| !v.is_finite())) {
                    return Err(Error::Invalid("private MTP checkpoint values"));
                }
                let mut rows = rows.into_iter();
                (
                    rows.next().ok_or(Error::Invalid("private MTP checkpoint keys"))?.into_boxed_slice(),
                    rows.next().ok_or(Error::Invalid("private MTP checkpoint values"))?.into_boxed_slice(),
                )
            };
            Ok(HeadCheckpoint {
                identity: self.identity.clone(),
                epoch: self.epoch,
                geometry: self.checkpoint_geometry(),
                selectors: self.selectors,
                next_cell: self.next_cell,
                keys,
                values,
            })
        })();
        if result.is_err() {
            self.fence();
        }
        result
    }

    pub(in crate::engine::hybrid) fn restore_checkpoint(
        &mut self,
        saved: &HeadCheckpoint,
    ) -> Result<()> {
        let result = (|| {
            if self.fenced
                || !self.policy_matches()
                || !Arc::ptr_eq(&self.identity, &saved.identity)
                || self.epoch != saved.epoch
                || saved.geometry != self.checkpoint_geometry()
                || saved.selectors != self.selectors
                || saved.next_cell > self.geometry.context
            {
                return Err(Error::Invalid("private MTP checkpoint ownership or policy"));
            }
            let width = self.geometry.kv_heads * self.geometry.head_dim;
            let used = saved.next_cell * width;
            if saved.keys.len() != used
                || saved.values.len() != used
                || saved.keys.iter().chain(saved.values.iter()).any(|v| !v.is_finite())
            {
                return Err(Error::Invalid("private MTP checkpoint prefix"));
            }
            // Allocate and populate complete independent owners before swapping
            // either live cache. A failed allocation cannot partially restore.
            let mut keys = vec![0.0; self.geometry.context * width];
            let mut values = vec![0.0; self.geometry.context * width];
            keys[..used].copy_from_slice(&saved.keys);
            values[..used].copy_from_slice(&saved.values);
            let replacement = Cache {
                keys: DecodeGraph::upload(&keys)?,
                values: DecodeGraph::upload(&values)?,
                dummy_index: DecodeGraph::zeros(self.geometry.context)?,
                visible: self.cache.visible.clone(),
            };
            new_graph()?.finish(&[])?;
            self.cache = replacement;
            self.next_cell = saved.next_cell;
            Ok(())
        })();
        if result.is_err() {
            self.fence();
        }
        result
    }

    fn norm(
        &self,
        name: &str,
    ) -> Result<&DeviceVector> {
        self.norms.get(name).ok_or(Error::Invalid("private MTP norm missing"))
    }

    #[expect(clippy::too_many_arguments, reason = "explicit injected weights, HC operands and private trace sink")]
    fn hc(
        &self,
        graph: &mut DecodeGraph,
        weights: &impl HeadWeights,
        prefix: &str,
        label: &str,
        residual: &DeviceVector,
        injecting: bool,
        stages: &mut BTreeMap<String, DeviceVector>,
        tracing: bool,
    ) -> Result<(DeviceVector, DeviceVector)> {
        let g = self.geometry;
        let normalized = graph.rms_norm(residual, self.norm(&format!("{prefix}.hc_norm.weight"))?, g.hidden, EPS)?;
        let low = checked(
            weights.project(graph, &format!("{prefix}.input_mix_weight_down.weight"), &normalized, 1)?,
            g.low_rank,
        )?;
        graph.kernel(
            "hybrid_graph_qwen_hc_activate",
            &[&low],
            &[g.low_rank as u32, g.streams as u32],
            g.low_rank,
            256,
        )?;
        let up = checked(
            weights.project(graph, &format!("{prefix}.input_mix_weight_up.weight"), &low, 1)?,
            g.streams * g.hidden,
        )?;
        let injection = if injecting {
            checked(
                weights.project(graph, &format!("{prefix}.block_inject_weight.weight"), &normalized, 1)?,
                g.streams,
            )?
        } else {
            graph.upload_into(&[0.0])?
        };
        let out = graph.scratch(
            g.hidden
                + if injecting {
                    g.streams
                } else {
                    0
                },
        )?;
        graph.kernel(
            "hybrid_graph_qwen_hc_mix",
            &[&up, &normalized, &injection, &out],
            &[g.hidden as u32, g.streams as u32, u32::from(injecting)],
            g.hidden.max(g.streams),
            256,
        )?;
        let mixed = out.slice(0, g.hidden)?;
        for (suffix, value) in [("normalized", &normalized), ("low", &low), ("mixed", &mixed)] {
            trace(stages, tracing, &format!("{label}_{suffix}"), value);
        }
        if injecting {
            trace(stages, tracing, &format!("{label}_inject"), &injection);
        }
        Ok((
            mixed,
            if injecting {
                out.slice(g.hidden, g.streams)?
            } else {
                injection
            },
        ))
    }

    fn inject(
        &self,
        graph: &mut DecodeGraph,
        residual: &DeviceVector,
        output: &DeviceVector,
        gates: &DeviceVector,
    ) -> Result<DeviceVector> {
        let g = self.geometry;
        let result = graph.scratch(g.streams * g.hidden)?;
        graph.kernel(
            "qwen_graph_inject",
            &[residual, output, gates, &result],
            &[g.streams as u32, g.hidden as u32],
            g.streams * g.hidden,
            256,
        )?;
        Ok(result)
    }

    pub(in crate::engine::hybrid) fn catch_up(
        &mut self,
        weights: &impl HeadWeights,
        cell: usize,
        wide: &DeviceVector,
        token: u32,
        tracing: bool,
    ) -> Result<HeadOutput> {
        self.execute(weights, cell, wide, token, false, tracing)
    }
    pub(in crate::engine::hybrid) fn step(
        &mut self,
        weights: &impl HeadWeights,
        cell: usize,
        wide: &DeviceVector,
        token: u32,
        tracing: bool,
    ) -> Result<HeadOutput> {
        self.execute(weights, cell, wide, token, true, tracing)
    }

    /// Actual callers must bind every residual to the same loaded target,
    /// live branch and state policy. Successful target restore changes that
    /// branch, deliberately refusing this bound head until a future combined
    /// target/head/pending-wide restore can rebind all three transactionally.
    /// Synthetic tests use the explicit raw component API.
    pub(in crate::engine::hybrid) fn target_step(
        &mut self,
        model: &QwenModel,
        wide: &WideResidual,
        token: u32,
        tracing: bool,
    ) -> Result<HeadOutput> {
        self.target_execute(model, wide, token, true, tracing)
    }

    pub(in crate::engine::hybrid) fn target_catch_up(
        &mut self,
        model: &QwenModel,
        wide: &WideResidual,
        token: u32,
        tracing: bool,
    ) -> Result<HeadOutput> {
        self.target_execute(model, wide, token, false, tracing)
    }

    fn target_execute(
        &mut self,
        model: &QwenModel,
        wide: &WideResidual,
        token: u32,
        full: bool,
        tracing: bool,
    ) -> Result<HeadOutput> {
        let policy = checkpoint::StagingPolicy::Edge0MetalSingleStepV1;
        if !model.checkpoint_state_tag_matches(&wide.tag, policy).unwrap_or(false)
            || !self.target_tag.as_ref().is_some_and(|tag| tag.same(&wide.tag))
        {
            self.fence();
            return Err(Error::Invalid("private MTP target residual ownership"));
        }
        if wide.row >= model.position || wide.row != self.next_cell {
            self.fence();
            return Err(Error::Invalid("private MTP target residual cell"));
        }
        self.execute(
            &ActualWeights {
                weights: &model.weights,
                embedding: &model.embedding,
                readout: &model.readout,
            },
            wide.row,
            &wide.values,
            token,
            full,
            tracing,
        )
    }

    fn execute(
        &mut self,
        weights: &impl HeadWeights,
        cell: usize,
        wide: &DeviceVector,
        token: u32,
        full: bool,
        tracing: bool,
    ) -> Result<HeadOutput> {
        let result = self.execute_inner(weights, cell, wide, token, full, tracing);
        if result.is_err() {
            self.fence();
        }
        result
    }

    fn execute_inner(
        &mut self,
        weights: &impl HeadWeights,
        cell: usize,
        wide: &DeviceVector,
        token: u32,
        full: bool,
        tracing: bool,
    ) -> Result<HeadOutput> {
        let g = self.geometry;
        if self.fenced
            || !self.policy_matches()
            || cell != self.next_cell
            || cell >= g.context
            || wide.len != g.streams * g.hidden
            || token as usize >= g.vocabulary
        {
            return Err(Error::Invalid("private MTP cell boundary"));
        }
        let mut graph = new_graph()?;
        graph.check_finite(wide)?;
        let mut stages = BTreeMap::new();
        let embedding = weights.embedding(token)?;
        if embedding.len() != g.hidden {
            return Err(Error::Invalid("private MTP embedding shape"));
        }
        let embedding = graph.upload_into(&embedding)?;
        let en = graph.rms_norm(&embedding, self.norm("pre_fc_norm_embedding.weight")?, g.hidden, EPS)?;
        // Unlike HC norms, this one RMS spans all four streams.
        let hn = graph.rms_norm(wide, self.norm("pre_fc_norm_hidden.weight")?, g.streams * g.hidden, EPS)?;
        let ep = checked(weights.project(&mut graph, "fc_embedding.weight", &en, 1)?, g.hidden)?;
        let hp = checked(weights.project(&mut graph, "fc_hidden.weight", &hn, g.streams)?, g.streams * g.hidden)?;
        let ep_wide = join(&mut graph, &vec![ep.clone(); g.streams])?;
        let mut residual = graph.add(&hp, &ep_wide)?;
        for (name, value) in [
            ("embedding_norm", &en),
            ("hidden_norm", &hn),
            ("embedding_projected", &ep),
            ("hidden_projected", &hp),
            ("input_residual", &residual),
        ] {
            trace(&mut stages, tracing, name, value);
        }
        let (mixed, attention_inject) =
            self.hc(&mut graph, weights, "attn_hyper_connection", "attn_hc", &residual, true, &mut stages, tracing)?;
        let keys =
            checked(weights.project(&mut graph, "self_attn.k_proj.weight", &mixed, 1)?, g.kv_heads * g.head_dim)?;
        let values =
            checked(weights.project(&mut graph, "self_attn.v_proj.weight", &mixed, 1)?, g.kv_heads * g.head_dim)?;
        // Front-only cells avoid Q projection, attention output and the MLP.
        // The existing prepare kernel receives a finite zero Q and a one-value
        // dummy index; neither is part of semantic draft cache state.
        let qfull = if full {
            checked(weights.project(&mut graph, "self_attn.q_proj.weight", &mixed, 1)?, 2 * g.heads * g.head_dim)?
        } else {
            graph.upload_into(&vec![0.0; 2 * g.heads * g.head_dim])?
        };
        let query = graph.scratch(g.heads * g.head_dim)?;
        let gates = graph.scratch(g.heads * g.head_dim)?;
        let dummy = graph.upload_into(&[0.0])?;
        for value in [&self.cache.keys, &self.cache.values, &self.cache.dummy_index] {
            graph.prepare_external_write(value)?;
        }
        graph.kernel(
            "qwen_graph_qsa_prepare",
            &[
                &dummy,
                &qfull,
                &keys,
                &values,
                self.norm("self_attn.q_norm.weight")?,
                self.norm("self_attn.k_norm.weight")?,
                &self.cache.dummy_index,
                &query,
                &self.cache.keys,
                &self.cache.values,
                &gates,
            ],
            &[
                (cell + 1) as u32,
                0,
                1,
                g.heads as u32,
                g.head_dim as u32,
                g.kv_heads as u32,
                g.rotary as u32,
                cell as u32,
                EPS.to_bits(),
                g.rope_theta.to_bits(),
            ],
            g.heads + g.kv_heads + 1,
            1,
        )?;
        let key = self.cache.keys.slice(cell * g.kv_heads * g.head_dim, g.kv_heads * g.head_dim)?;
        let value = self.cache.values.slice(cell * g.kv_heads * g.head_dim, g.kv_heads * g.head_dim)?;
        trace(&mut stages, tracing, "key", &key);
        trace(&mut stages, tracing, "value", &value);
        graph.check_finite(&key)?;
        graph.check_finite(&value)?;
        if !full {
            let captured = finish(graph, &stages, &[])?;
            self.next_cell += 1;
            return Ok(HeadOutput {
                wide: None,
                logits: None,
                routed_ids: Vec::new(),
                stages: captured,
            });
        }
        trace(&mut stages, tracing, "query_full", &qfull);
        trace(&mut stages, tracing, "query", &query);
        let scores = graph.upload_into(&vec![0.0; g.heads * g.context])?;
        let probabilities = graph.upload_into(&vec![0.0; g.heads * g.context])?;
        let p = [(cell + 1) as u32, g.heads as u32, g.head_dim as u32, g.kv_heads as u32, g.context as u32];
        graph.kernel(
            "qwen_graph_qsa_scores",
            &[&query, &self.cache.keys, &self.cache.visible, &scores],
            &p,
            g.heads * (cell + 1),
            64,
        )?;
        graph.kernel(
            "qwen_graph_qsa_softmax",
            &[&scores, &self.cache.visible, &probabilities],
            &[(cell + 1) as u32, g.heads as u32, g.context as u32],
            g.heads,
            1,
        )?;
        let gated = graph.scratch(g.heads * g.head_dim)?;
        graph.kernel(
            "qwen_graph_qsa_output",
            &[&probabilities, &self.cache.values, &gates, &self.cache.visible, &gated],
            &p,
            g.heads * g.head_dim,
            256,
        )?;
        trace(&mut stages, tracing, "attention_scores", &scores);
        trace(&mut stages, tracing, "attention_probabilities", &probabilities);
        if tracing {
            let zero = graph.upload_into(&vec![0.0; g.heads * g.head_dim])?;
            let half = graph.scratch(g.heads * g.head_dim)?;
            graph.kernel(
                "qwen_graph_qsa_output",
                &[&probabilities, &self.cache.values, &zero, &self.cache.visible, &half],
                &p,
                g.heads * g.head_dim,
                256,
            )?;
            let raw = graph.add(&half, &half)?;
            trace(&mut stages, true, "attention", &raw);
        }
        trace(&mut stages, tracing, "gated_attention", &gated);
        let output = checked(weights.project(&mut graph, "self_attn.o_proj.weight", &gated, 1)?, g.hidden)?;
        trace(&mut stages, tracing, "attention_output", &output);
        residual = self.inject(&mut graph, &residual, &output, &attention_inject)?;
        trace(&mut stages, tracing, "post_attention", &residual);
        let (mixed, mlp_inject) =
            self.hc(&mut graph, weights, "mlp_hyper_connection", "mlp_hc", &residual, true, &mut stages, tracing)?;
        let logits = checked(weights.project(&mut graph, "mlp.gate.weight", &mixed, 1)?, g.experts)?;
        trace(&mut stages, tracing, "router_logits", &logits);
        // Only the bounded router row crosses the normal execution boundary.
        let row = std::mem::replace(&mut graph, new_graph()?)
            .finish(&[logits])?
            .pop()
            .ok_or(Error::Invalid("private MTP router row missing"))?;
        let (ids, weights_row) = edge0::route(&row, g.active)?;
        let route_weights = graph.upload_into(&weights_row)?;
        trace(&mut stages, tracing, "router_weights", &route_weights);
        let mut parts = Vec::with_capacity(g.active);
        let mut activations = Vec::with_capacity(g.active);
        for &expert in &ids {
            let activation = checked(weights.expert_gate_up(&mut graph, expert, &mixed, 1)?, g.intermediate)?;
            parts.push(checked(weights.expert_down(&mut graph, expert, &activation, 1)?, g.hidden)?);
            activations.push(activation);
        }
        let activations = join(&mut graph, &activations)?;
        let parts = join(&mut graph, &parts)?;
        trace(&mut stages, tracing, "routed_activation", &activations);
        trace(&mut stages, tracing, "routed_parts", &parts);
        let sg =
            checked(weights.project(&mut graph, "mlp.shared_expert.gate_proj.weight", &mixed, 1)?, g.intermediate)?;
        let su = checked(weights.project(&mut graph, "mlp.shared_expert.up_proj.weight", &mixed, 1)?, g.intermediate)?;
        let sa = graph.swiglu(&sg, &su, None)?;
        let shared = checked(weights.project(&mut graph, "mlp.shared_expert.down_proj.weight", &sa, 1)?, g.hidden)?;
        let scalar = checked(weights.project(&mut graph, "mlp.shared_expert_gate.weight", &mixed, 1)?, 1)?;
        let scalar_out = graph.scratch(1)?;
        graph.kernel("qwen_graph_sigmoid", &[&scalar, &scalar_out], &[1], 1, 1)?;
        let output = graph.weighted_sum_device(&parts, &route_weights, &shared, &scalar_out)?;
        for (name, value) in [
            ("shared_gate", &sg),
            ("shared_up", &su),
            ("shared_activation", &sa),
            ("shared_scalar", &scalar_out),
            ("moe_output", &output),
        ] {
            trace(&mut stages, tracing, name, value);
        }
        if tracing {
            let empty = graph.upload_into(&vec![0.0; g.hidden])?;
            let zero = graph.upload_into(&[0.0])?;
            let scaled = graph.weighted_sum_device(&empty, &zero, &shared, &scalar_out)?;
            trace(&mut stages, true, "shared_output", &scaled);
        }
        residual = self.inject(&mut graph, &residual, &output, &mlp_inject)?;
        trace(&mut stages, tracing, "final_residual", &residual);
        // A compact owner avoids retaining a whole scratch generation for the
        // next draft cell after this command completes.
        let next = DecodeGraph::zeros(g.streams * g.hidden)?;
        graph.copy_into(&residual, &next)?;
        let (mixed, _) =
            self.hc(&mut graph, weights, "hyper_connection_mixer", "final_hc", &residual, false, &mut stages, tracing)?;
        let readout = checked(weights.readout(&mut graph, &mixed)?, g.vocabulary)?;
        trace(&mut stages, tracing, "readout", &readout);
        let mut captured = finish(graph, &stages, &[readout])?;
        let logits = captured.remove("__output_0").ok_or(Error::Invalid("private MTP readout missing"))?;
        self.next_cell += 1;
        Ok(HeadOutput {
            wide: Some(next),
            logits: Some(logits),
            routed_ids: ids,
            stages: captured,
        })
    }
}

fn finish(
    graph: DecodeGraph,
    stages: &BTreeMap<String, DeviceVector>,
    outputs: &[DeviceVector],
) -> Result<BTreeMap<String, Vec<f32>>> {
    let mut names = stages.keys().cloned().collect::<Vec<_>>();
    let mut values = stages.values().cloned().collect::<Vec<_>>();
    for (i, value) in outputs.iter().enumerate() {
        names.push(format!("__output_{i}"));
        values.push(value.clone());
    }
    let rows = graph.finish(&values)?;
    if rows.len() != names.len() || rows.iter().any(|row| row.iter().any(|v| !v.is_finite())) {
        return Err(Error::Invalid("private MTP output values"));
    }
    Ok(names.into_iter().zip(rows).collect())
}
