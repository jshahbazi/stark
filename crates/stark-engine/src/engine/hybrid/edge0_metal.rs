// SPDX-License-Identifier: Apache-2.0
//! Opt-in Edge0 latent device execution. Only router score rows cross the
//! device/host boundary; source F64 routing remains authoritative. All other
//! dependent activations stay on Metal. Raw stored weights are never repacked.

use std::collections::{BTreeMap, BTreeSet};

use super::{super::edge0, DecodeGraph, DeviceVector, Geometry, Weights};
use crate::engine::hybrid::{
    Error, Result,
    edge0_source::{LATENT_WIDTH, PROJECTION_GROUPS},
    profile,
    weight_metal::Edge0Schedule,
    weights::Edge0Matrix,
};

pub(super) const MAX_BATCH: usize = 8;
pub(super) const SEGMENT_SCRATCH: usize = 64 * 1024 * 1024;

pub(super) fn new_graph() -> Result<DecodeGraph> {
    let mut graph = DecodeGraph::new()?;
    graph.set_scratch_limit(SEGMENT_SCRATCH)?;
    Ok(graph)
}

#[derive(Clone)]
pub(super) struct Controls {
    stored_s: DeviceVector,
}

impl Controls {
    pub(super) fn new(
        weights: &Weights,
        layer: usize,
    ) -> Result<Self> {
        let scales = weights.edge0_column_scales(layer)?;
        if scales.len() != LATENT_WIDTH as usize || scales.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(Error::Invalid("Edge0 graph column scales"));
        }
        Ok(Self {
            stored_s: DecodeGraph::upload(scales)?,
        })
    }
}

#[derive(Clone)]
struct RoutePlan {
    expert_jobs: BTreeMap<usize, Vec<(usize, usize)>>,
    group_jobs: BTreeMap<usize, Vec<(usize, usize)>>,
    groups: BTreeSet<usize>,
}

fn plan_routes(
    selected: &[Vec<usize>],
    active: usize,
    experts: usize,
    mut label: impl FnMut(usize) -> Result<usize>,
) -> Result<RoutePlan> {
    if selected.is_empty() || selected.len() > MAX_BATCH || active == 0 || active > experts {
        return Err(Error::Invalid("Edge0 grouped route dimensions"));
    }
    let mut plan = RoutePlan {
        expert_jobs: BTreeMap::new(),
        group_jobs: BTreeMap::new(),
        groups: BTreeSet::new(),
    };
    for (token, ids) in selected.iter().enumerate() {
        if ids.len() != active {
            return Err(Error::Invalid("Edge0 grouped route width"));
        }
        let mut seen = BTreeSet::new();
        for (route, &expert) in ids.iter().enumerate() {
            if expert >= experts || !seen.insert(expert) {
                return Err(Error::Invalid("Edge0 grouped expert route"));
            }
            let group = label(expert)?;
            if group >= PROJECTION_GROUPS {
                return Err(Error::Invalid("Edge0 graph projection label"));
            }
            plan.groups.insert(group);
            plan.expert_jobs.entry(expert).or_default().push((token, route));
            plan.group_jobs.entry(group).or_default().push((token, route));
        }
    }
    Ok(plan)
}

pub(super) fn concatenate(
    graph: &mut DecodeGraph,
    rows: &[DeviceVector],
) -> Result<DeviceVector> {
    let width = rows.first().ok_or(Error::Invalid("Edge0 empty device batch"))?.len;
    if rows.len() > MAX_BATCH || rows.iter().any(|row| row.len != width) {
        return Err(Error::Invalid("Edge0 device batch geometry"));
    }
    let output = graph.scratch(width.checked_mul(rows.len()).ok_or(Error::Invalid("Edge0 batch overflow"))?)?;
    for (index, row) in rows.iter().enumerate() {
        graph.copy_into(row, &output.slice(index * width, width)?)?;
    }
    Ok(output)
}

/// Finish the preceding GPU segment before reading only the bounded score
/// rows. Other vector owners remain alive and are rebound into the new graph.
/// The wait/readback metric includes completion of preceding device work; it
/// is not a measurement of pure host routing overhead.
fn routes(
    graph: &mut DecodeGraph,
    weights: &Weights,
    scores: DeviceVector,
    tokens: usize,
    experts: usize,
    selected: usize,
) -> Result<(Vec<Vec<usize>>, DeviceVector)> {
    if !(1..=MAX_BATCH).contains(&tokens)
        || experts == 0
        || experts > 512
        || selected == 0
        || selected > experts
        || scores.len != tokens * experts
    {
        return Err(Error::Invalid("Edge0 router readback bound"));
    }
    let started = std::time::Instant::now();
    let previous = std::mem::replace(graph, new_graph()?);
    weights.edge0_record_graph_scratch(previous.scratch_backing_bytes());
    let rows = previous.finish(&[scores])?.pop().ok_or(Error::Invalid("Edge0 router scores missing"))?;
    profile::record("native.edge0.router.boundary_wait_readback", started.elapsed().as_secs_f64(), rows.len() * 4);
    let started = std::time::Instant::now();
    let mut ids = Vec::with_capacity(tokens);
    let mut probabilities = Vec::with_capacity(tokens * selected);
    for row in rows.chunks_exact(experts) {
        let (chosen, weights) = edge0::route(row, selected)?;
        ids.push(chosen);
        probabilities.extend(weights);
    }
    profile::record("native.edge0.router.f64", started.elapsed().as_secs_f64(), tokens * experts);
    let probabilities = graph.upload_into(&probabilities)?;
    Ok((ids, probabilities))
}

#[expect(clippy::too_many_arguments, reason = "separate checked model, layer, device operands and batch geometry")]
pub(super) fn encode(
    graph: &mut DecodeGraph,
    weights: &Weights,
    layer: usize,
    prefix: &str,
    inputs: &DeviceVector,
    g: &Geometry,
    controls: &Controls,
    tokens: usize,
) -> Result<DeviceVector> {
    if !g.edge0 || !(1..=MAX_BATCH).contains(&tokens) || inputs.len != tokens * g.dim {
        return Err(Error::Invalid("Edge0 graph latent batch"));
    }
    graph.begin_projection_level()?;
    let scores = weights.encode_edge0_named_batch(graph, &format!("{prefix}.gate.weight"), inputs, tokens)?;
    let shared_gate =
        weights.encode_edge0_named_batch(graph, &format!("{prefix}.shared_expert_gate.weight"), inputs, tokens)?;
    let shared_values =
        weights.encode_edge0_named_batch(graph, &format!("{prefix}.shared_expert.gate_proj.weight"), inputs, tokens)?;
    let shared_up =
        weights.encode_edge0_named_batch(graph, &format!("{prefix}.shared_expert.up_proj.weight"), inputs, tokens)?;
    graph.end_projection_level()?;
    graph.edge0_flush_verification()?;
    graph.edge0_stage("edge0.stage.router_proj", 1)?;
    let (selected, probabilities) = routes(graph, weights, scores, tokens, g.experts, g.active_experts)?;

    let plan = plan_routes(&selected, g.active_experts, g.experts, |expert| weights.edge0_group(layer, expert))?;
    let schedule = DecodeGraph::edge0_schedule();
    let (_, expanded) = if schedule == Edge0Schedule::Grouped {
        expand_grouped(graph, weights, layer, inputs, g, controls, tokens, plan)?
    } else {
        let direct = expand_direct(graph, weights, layer, inputs, g, controls, tokens, &plan, schedule.token_rows())?;
        if schedule.verify() {
            let grouped = expand_grouped(graph, weights, layer, inputs, g, controls, tokens, plan)?;
            graph.edge0_require_identical(&direct.0, &grouped.0)?;
            graph.edge0_require_identical(&direct.1, &grouped.1)?;
        }
        direct
    };
    let shared_values = graph.swiglu(&shared_values, &shared_up, None)?;
    let shared_down = weights.encode_edge0_named_batch(
        graph,
        &format!("{prefix}.shared_expert.down_proj.weight"),
        &shared_values,
        tokens,
    )?;
    graph.edge0_stage("edge0.stage.shared", 1)?;
    let result = graph.edge0_weighted_sum(
        &expanded,
        &probabilities,
        &shared_down,
        &shared_gate,
        g.dim,
        g.active_experts,
        tokens,
    )?;
    graph.check_finite(&result)?;
    graph.edge0_stage("edge0.stage.combine", 1)?;
    profile::record("native.edge0.latent.batch_tokens", 0.0, tokens);
    Ok(result)
}

/// Direct-route schedule. Every (token, route) row runs one-token kernels on
/// its own slots: no latent row is copied, D runs only for the (group, token)
/// pairs some route uses, and D writes its stored column-scale quotient in
/// the same dispatch. Route slots, per-route U and every F32 reduction match
/// the grouped schedule. Returns the latent expert rows and the expanded rows.
///
/// Chained rows carry no separate finite scan. Each kernel records sticky
/// failure for every value it writes, and a nonfinite input makes every row
/// result nonfinite. `inputs` was scanned by the router projection level,
/// whose command completed before routing.
///
/// `shared` selects the token-batched form of the same schedule: token rows
/// of one D, expert or U matrix share dispatches that decode each weight
/// once. Every row keeps its slot and its one-token F32 reduction.
#[expect(clippy::too_many_arguments, reason = "separate checked model, layer, device operands and batch geometry")]
fn expand_direct(
    graph: &mut DecodeGraph,
    weights: &Weights,
    layer: usize,
    inputs: &DeviceVector,
    g: &Geometry,
    controls: &Controls,
    tokens: usize,
    plan: &RoutePlan,
    shared: bool,
) -> Result<(DeviceVector, DeviceVector)> {
    let latent_width = LATENT_WIDTH as usize;
    let route_rows = tokens.checked_mul(g.active_experts).ok_or(Error::Invalid("Edge0 route count overflow"))?;
    let mut group_inputs: Vec<Option<DeviceVector>> = vec![None; PROJECTION_GROUPS];
    // D rows read only `inputs` and write disjoint rows: one projection level.
    graph.begin_projection_level()?;
    for (&group, jobs) in &plan.group_jobs {
        // Token rows no route of this group selects stay unwritten and unread.
        let projected = graph.scratch(tokens * latent_width)?;
        let mut written = [false; MAX_BATCH];
        let rows = jobs.iter().map(|&(token, _)| token).filter(|&token| !std::mem::replace(&mut written[token], true));
        let matrix = Edge0Matrix::ProjectionDown {
            layer,
            group,
        };
        if shared {
            // The same token rows share dispatches that decode D once.
            let rows = rows.collect::<Vec<_>>();
            weights.encode_edge0_matvec_tokens_into(
                graph,
                matrix,
                inputs,
                &rows,
                &projected,
                &rows,
                Some(&controls.stored_s),
            )?;
        } else {
            for token in rows {
                weights.encode_edge0_matvec_single_into(
                    graph,
                    matrix,
                    &inputs.slice(token * g.dim, g.dim)?,
                    &projected.slice(token * latent_width, latent_width)?,
                    Some(&controls.stored_s),
                )?;
            }
        }
        group_inputs[group] = Some(projected);
    }
    graph.end_projection_level()?;
    graph.edge0_stage("edge0.stage.proj_down", 1)?;
    let expert_outputs = graph.scratch(route_rows * latent_width)?;
    for (&expert, jobs) in &plan.expert_jobs {
        let group = weights.edge0_group(layer, expert)?;
        let source = group_inputs[group].as_ref().ok_or(Error::Invalid("Edge0 group activation missing"))?;
        if shared {
            // One expert's token rows share its gate/up and down dispatches.
            let rows = jobs.iter().map(|&(token, _)| token).collect::<Vec<_>>();
            let slots = jobs.iter().map(|&(token, route)| token * g.active_experts + route).collect::<Vec<_>>();
            let ordered = (0..jobs.len()).collect::<Vec<_>>();
            let hidden = weights.encode_edge0_gate_up_swiglu_tokens(graph, layer, expert, source, &rows)?;
            graph.edge0_stage("edge0.stage.experts.gate_up", 2)?;
            weights.encode_edge0_matvec_tokens_into(
                graph,
                Edge0Matrix::ExpertDown {
                    layer,
                    expert,
                },
                &hidden,
                &ordered,
                &expert_outputs,
                &slots,
                None,
            )?;
            graph.edge0_stage("edge0.stage.experts.down", 2)?;
            graph.resource_boundary()?;
            continue;
        }
        for &(token, route) in jobs {
            let hidden = weights.encode_edge0_gate_up_swiglu_single(
                graph,
                layer,
                expert,
                &source.slice(token * latent_width, latent_width)?,
            )?;
            graph.edge0_stage("edge0.stage.experts.gate_up", 2)?;
            weights.encode_edge0_matvec_single_into(
                graph,
                Edge0Matrix::ExpertDown {
                    layer,
                    expert,
                },
                &hidden,
                &expert_outputs.slice((token * g.active_experts + route) * latent_width, latent_width)?,
                None,
            )?;
            graph.edge0_stage("edge0.stage.experts.down", 2)?;
        }
        graph.resource_boundary()?;
    }
    graph.edge0_stage("edge0.stage.experts", 1)?;
    let expanded = graph.scratch(route_rows * g.dim)?;
    // U rows read completed expert rows and write disjoint rows: one level.
    // Every group's U lies in the layer's dense bank, as its D rows above do.
    graph.begin_projection_level()?;
    for (&group, jobs) in &plan.group_jobs {
        if shared {
            // Each route still receives U independently, in its own slot.
            let slots = jobs.iter().map(|&(token, route)| token * g.active_experts + route).collect::<Vec<_>>();
            weights.encode_edge0_matvec_tokens_into(
                graph,
                Edge0Matrix::ProjectionUp {
                    layer,
                    group,
                },
                &expert_outputs,
                &slots,
                &expanded,
                &slots,
                None,
            )?;
            continue;
        }
        for &(token, route) in jobs {
            let slot = token * g.active_experts + route;
            // Each original route still receives U independently.
            weights.encode_edge0_matvec_single_into(
                graph,
                Edge0Matrix::ProjectionUp {
                    layer,
                    group,
                },
                &expert_outputs.slice(slot * latent_width, latent_width)?,
                &expanded.slice(slot * g.dim, g.dim)?,
                None,
            )?;
        }
    }
    graph.end_projection_level()?;
    graph.resource_boundary()?;
    graph.edge0_stage("edge0.stage.proj_up", 1)?;
    Ok((expert_outputs, expanded))
}

/// Grouped schedule: batched D per selected group, expert jobs grouped by
/// expert and U jobs grouped by projection group, with copied latent rows.
#[expect(clippy::too_many_arguments, reason = "separate checked model, layer, device operands and batch geometry")]
fn expand_grouped(
    graph: &mut DecodeGraph,
    weights: &Weights,
    layer: usize,
    inputs: &DeviceVector,
    g: &Geometry,
    controls: &Controls,
    tokens: usize,
    plan: RoutePlan,
) -> Result<(DeviceVector, DeviceVector)> {
    let latent_width = LATENT_WIDTH as usize;
    let mut group_inputs: Vec<Option<DeviceVector>> = vec![None; PROJECTION_GROUPS];
    for group in plan.groups {
        let projected = weights.encode_edge0_matvec_batch(
            graph,
            Edge0Matrix::ProjectionDown {
                layer,
                group,
            },
            inputs,
            tokens,
        )?;
        graph.edge0_stage("edge0.stage.proj_down.matvec", 2)?;
        group_inputs[group] = Some(graph.edge0_column_scale(&projected, &controls.stored_s, latent_width, tokens)?);
        graph.edge0_stage("edge0.stage.proj_down.scale", 2)?;
    }
    graph.edge0_stage("edge0.stage.proj_down", 1)?;
    let route_rows = tokens.checked_mul(g.active_experts).ok_or(Error::Invalid("Edge0 route count overflow"))?;
    let expert_outputs = graph.scratch(route_rows * latent_width)?;
    for (expert, jobs) in plan.expert_jobs {
        // Each expert occurs at most once per token, so its input batch stays
        // within MAX_BATCH. Jobs are grouped without changing route slots.
        if jobs.len() > MAX_BATCH {
            return Err(Error::Invalid("Edge0 expert batch bound"));
        }
        let group = weights.edge0_group(layer, expert)?;
        let source = group_inputs[group].as_ref().ok_or(Error::Invalid("Edge0 group activation missing"))?;
        let rows = jobs
            .iter()
            .map(|&(token, _)| source.slice(token * latent_width, latent_width))
            .collect::<Result<Vec<_>>>()?;
        let input = concatenate(graph, &rows)?;
        graph.edge0_stage("edge0.stage.experts.gather", 2)?;
        let hidden = weights.encode_edge0_gate_up_swiglu_batch(graph, layer, expert, &input, jobs.len())?;
        graph.edge0_stage("edge0.stage.experts.gate_up", 2)?;
        let down = weights.encode_edge0_matvec_batch(
            graph,
            Edge0Matrix::ExpertDown {
                layer,
                expert,
            },
            &hidden,
            jobs.len(),
        )?;
        graph.edge0_stage("edge0.stage.experts.down", 2)?;
        for (index, &(token, route)) in jobs.iter().enumerate() {
            graph.copy_into(
                &down.slice(index * latent_width, latent_width)?,
                &expert_outputs.slice((token * g.active_experts + route) * latent_width, latent_width)?,
            )?;
        }
        graph.edge0_stage("edge0.stage.experts.scatter", 2)?;
        graph.resource_boundary()?;
    }
    graph.edge0_stage("edge0.stage.experts", 1)?;

    let expanded = graph.scratch(route_rows * g.dim)?;
    for (group, jobs) in plan.group_jobs {
        for batch in jobs.chunks(MAX_BATCH) {
            let rows = batch
                .iter()
                .map(|&(token, route)| {
                    expert_outputs.slice((token * g.active_experts + route) * latent_width, latent_width)
                })
                .collect::<Result<Vec<_>>>()?;
            let input = concatenate(graph, &rows)?;
            graph.edge0_stage("edge0.stage.proj_up.gather", 2)?;
            // Each original route receives U independently. Batching reuses
            // its bytes; it never sums latent rows before U.
            let output = weights.encode_edge0_matvec_batch(
                graph,
                Edge0Matrix::ProjectionUp {
                    layer,
                    group,
                },
                &input,
                batch.len(),
            )?;
            graph.edge0_stage(
                if batch.len() == 1 {
                    "edge0.stage.proj_up.matvec_single"
                } else {
                    "edge0.stage.proj_up.matvec_batch"
                },
                2,
            )?;
            for (index, &(token, route)) in batch.iter().enumerate() {
                graph.copy_into(
                    &output.slice(index * g.dim, g.dim)?,
                    &expanded.slice((token * g.active_experts + route) * g.dim, g.dim)?,
                )?;
            }
            graph.edge0_stage("edge0.stage.proj_up.scatter", 2)?;
        }
        graph.resource_boundary()?;
    }
    graph.edge0_stage("edge0.stage.proj_up", 1)?;
    Ok((expert_outputs, expanded))
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn edge0_prefill_grouping_preserves_every_original_route_slot() {
        let labels = [0, 1, 0, 2, 1];
        let routes = vec![vec![3, 1, 4], vec![0, 1, 2], vec![4, 2, 0]];
        let plan = plan_routes(&routes, 3, labels.len(), |expert| Ok(labels[expert])).unwrap();
        assert_eq!(plan.expert_jobs[&1], [(0, 1), (1, 1)]);
        assert_eq!(plan.expert_jobs[&4], [(0, 2), (2, 0)]);
        assert_eq!(plan.group_jobs[&1], [(0, 1), (0, 2), (1, 1), (2, 0)]);
        assert_eq!(plan.groups.into_iter().collect::<Vec<_>>(), [0, 1, 2]);
        let mut slots = plan.group_jobs.values().flatten().copied().collect::<Vec<_>>();
        slots.sort_unstable();
        assert_eq!(slots, (0..3).flat_map(|token| (0..3).map(move |route| (token, route))).collect::<Vec<_>>());
        assert_eq!(plan.expert_jobs.values().map(Vec::len).sum::<usize>(), 9);
    }

    #[stark_test]
    fn edge0_prefill_grouping_rejects_duplicate_or_unbounded_jobs() {
        assert!(plan_routes(&[vec![0, 0]], 2, 2, |_| Ok(0)).is_err());
        assert!(plan_routes(&[vec![2]], 1, 2, |_| Ok(0)).is_err());
        assert!(plan_routes(&[vec![0]], 1, 2, |_| Ok(PROJECTION_GROUPS)).is_err());
        assert!(plan_routes(&vec![vec![0]; MAX_BATCH + 1], 1, 2, |_| Ok(0)).is_err());
        let plan = plan_routes(&vec![vec![1, 0]; MAX_BATCH], 2, 2, |_| Ok(0)).unwrap();
        assert!(plan.expert_jobs.values().all(|jobs| jobs.len() == MAX_BATCH));
        assert_eq!(plan.group_jobs[&0].len(), 2 * MAX_BATCH);
    }
}
