// SPDX-License-Identifier: Apache-2.0
//! Batched independent MLA projections around chronological cache operations.

use super::{
    super::{
        weight_metal::{DecodeGraph, DeviceVector},
        weights::BoundedBankGroup,
    },
    *,
};

#[expect(clippy::too_many_arguments, reason = "one admitted MLA layer with explicit state and token ownership")]
pub(super) fn encode_mla_prefill(
    w: &Weights,
    c: &Config,
    graph: &mut DecodeGraph,
    input: &DeviceVector,
    p: &str,
    state: &mut super::super::glm_metal::Attention,
    banks: &BoundedBankGroup<'_>,
    tokens: usize,
    use_mma: bool,
) -> Result<Option<DeviceVector>> {
    if !std::env::var_os("STARK_GLM53_MLA_PREFILL_BATCH").is_some_and(|value| value == "1") {
        return Ok(None);
    }
    if !(1..=128).contains(&tokens) || c.hidden.checked_mul(tokens) != Some(input.len) {
        return Err(Error::Invalid("MLA prefill input geometry"));
    }
    if [
        c.hidden,
        c.q_rank,
        c.kv_rank,
        c.index_dim,
        c.index_heads,
        c.heads.saturating_mul(c.qk_dim),
        c.heads.saturating_mul(c.kv_rank),
        c.heads.saturating_mul(c.value_dim),
        c.index_heads.saturating_mul(c.index_dim),
    ]
    .iter()
    .any(|&width| width.checked_mul(tokens).is_none_or(|n| n > 8_000_000))
    {
        return Ok(None);
    }
    let project = |graph: &mut DecodeGraph, name: &str, x: &DeviceVector| {
        banks.encode_same_weight_batch(graph, &format!("{p}.{name}"), x, tokens, use_mma)
    };
    let q = project(graph, "attn_q_a.weight", input)?;
    let latent = project(graph, "attn_kv_a_mqa.weight", input)?;
    let key = project(graph, "indexer.attn_k.weight", input)?;
    let gate = project(graph, "indexer_compressor_gate.weight", input)?;
    let weights = project(graph, "indexer.proj.weight", input)?;
    graph.diagnostic_checkpoint("native.glm.prefill.gpu.mla_input_projections")?;
    let qnorm = DecodeGraph::upload(&w.vector(&format!("{p}.attn_q_a_norm.weight"))?)?;
    let q = graph.rms_norm(&q, &qnorm, c.q_rank, c.rms_eps)?;
    let query = project(graph, "attn_q_b.weight", &q)?;
    let index_query = project(graph, "indexer.attn_q_b.weight", &q)?;
    graph.diagnostic_checkpoint("native.glm.prefill.gpu.mla_query_projections")?;
    let kvnorm = DecodeGraph::upload(&w.vector(&format!("{p}.attn_kv_a_norm.weight"))?)?;
    let latent = graph.rms_norm(&latent, &kvnorm, c.kv_rank, c.rms_eps)?;
    let keynorm = DecodeGraph::upload(&w.vector(&format!("{p}.indexer.k_norm.weight"))?)?;
    let keybias = DecodeGraph::upload(&w.vector(&format!("{p}.indexer.k_norm.bias"))?)?;
    let key = graph.mla_index_norm_batch(&key, &keynorm, &keybias, c.index_dim, tokens)?;
    graph.diagnostic_checkpoint("native.glm.prefill.gpu.mla_norm_append")?;
    // Absorption is independent of cache position. All query rows may be
    // projected before chronological cache updates without observing future KV.
    let query = w.encode_mla_head_batch(graph, &format!("{p}.attn_k_b.weight"), &query, tokens)?;
    graph.diagnostic_checkpoint("native.glm.prefill.gpu.mla_key_projection")?;
    let attended = graph.scratch(tokens * c.heads * c.kv_rank)?;
    for token in 0..tokens {
        state.append(
            graph,
            &latent.slice(token * c.kv_rank, c.kv_rank)?,
            &key.slice(token * c.index_dim, c.index_dim)?,
            &gate.slice(token * c.index_dim, c.index_dim)?,
        )?;
        let row = state.attend(
            graph,
            &query.slice(token * c.heads * c.kv_rank, c.heads * c.kv_rank)?,
            &index_query.slice(token * c.index_heads * c.index_dim, c.index_heads * c.index_dim)?,
            &weights.slice(token * c.index_heads, c.index_heads)?,
        )?;
        graph.copy_into(&row, &attended.slice(token * c.heads * c.kv_rank, c.heads * c.kv_rank)?)?;
    }
    graph.diagnostic_checkpoint("native.glm.prefill.gpu.mla_attend")?;
    let output = w.encode_mla_head_batch(graph, &format!("{p}.attn_v_b.weight"), &attended, tokens)?;
    graph.diagnostic_checkpoint("native.glm.prefill.gpu.mla_value_projection")?;
    let output = project(graph, "attn_output.weight", &output)?;
    graph.diagnostic_checkpoint("native.glm.prefill.gpu.mla_output_projection")?;
    super::super::profile::record("native.glm.prefill.mla_batch.tokens", 0.0, tokens);
    Ok(Some(output))
}

#[cfg(test)]
#[path = "glm_mla_prefill_tests.rs"]
mod tests;
