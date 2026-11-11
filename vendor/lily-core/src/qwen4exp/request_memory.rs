// Modified by Stark: allocation-only request admission; see UPSTREAM.md.
//! Conservative request inventory derived from the state and scratch constructors.

use super::*;
use crate::qwen4exp::config::LayerType;
use objc2_metal::MTLDevice;

#[derive(Default)]
struct Inventory {
    bytes: u128,
    max_buffer: u128,
}

impl Inventory {
    fn add(&mut self, shape: &[u128], element_bytes: u128, copies: u128) -> Option<()> {
        let bytes = shape.iter().try_fold(element_bytes, |n, &d| n.checked_mul(d))?;
        // Cover independent Metal-buffer alignment as well as payload.
        let aligned = bytes.checked_add(255)? / 256 * 256;
        self.bytes = self.bytes.checked_add(aligned.checked_mul(copies)?)?;
        if copies != 0 {
            self.max_buffer = self.max_buffer.max(aligned);
        }
        Some(())
    }
}

fn kv_row_bytes(format: KvFormat, dim: u128) -> Option<u128> {
    match format {
        KvFormat::Bf16 => dim.checked_mul(2),
        KvFormat::Q8 => dim.checked_add(dim.checked_div(32)?.checked_mul(2)?),
        KvFormat::Q8Affine64 => dim.checked_add(dim.checked_div(64)?.checked_mul(4)?),
    }
}

fn hc(s: &mut Inventory, c: &Qwen4ExpConfig, m: u128) -> Option<()> {
    let (h, g, low) = (c.hidden_size as u128, c.hc_count as u128, c.hc_lowrank as u128);
    let wide = h.checked_mul(g)?;
    for (width, copies) in [(wide, 3), (low, 2), (h, 1), (g, 1)] {
        s.add(&[m, width], 2, copies)?;
    }
    s.add(&[m, g], 4, 1)?;
    // Bounds the optional stock residual and its <=4-row partials.
    s.add(&[m.min(4), wide], 4, 1)
}

fn qsa(
    s: &mut Inventory,
    c: &Qwen4ExpConfig,
    capacity: usize,
    rows: usize,
) -> Option<()> {
    let (ratio, k) = (c.indexer.compress_ratio, c.indexer.block_topk());
    if ratio == 0 || rows == 0 {
        return None;
    }
    let slots = qsa::split_scratch_slots(rows, k, ratio) as u128;
    s.add(&[rows as u128, capacity.div_ceil(ratio).max(1) as u128], 4, 1)?;
    s.add(&[rows as u128, k as u128], 4, 1)?;
    s.add(&[rows as u128], 4, 1)?;
    s.add(&[slots, c.num_attention_heads as u128, c.head_dim as u128], 4, 1)?;
    s.add(&[slots, c.num_attention_heads as u128, 2], 4, 1)
}

fn ple(s: &mut Inventory, c: &Qwen4ExpConfig, m: u128) -> Option<()> {
    let Some(p) = &c.ple else {
        return Some(());
    };
    let (h, wide, emb) = (
        c.hidden_size as u128,
        c.hidden_size as u128 * c.hc_count as u128,
        p.embed_dim as u128,
    );
    s.add(&[m, p.ngram_heads() as u128], 4, 3)?;
    s.add(&[m, emb], 2, 1)?;
    s.add(&[m, wide], 2, 5)?;
    s.add(&[m, h], 2, 1)?;
    // Two paged-table staging owners. F32 gathered rows conservatively
    // bound the Q4 codes, BF16 scales/biases and sequential row ids.
    s.add(&[m, emb], 4, 2)
}

fn moe(s: &mut Inventory, c: &Qwen4ExpConfig, m: u128, prefill: bool) -> Option<()> {
    let (h, e, k, inter) = (
        c.hidden_size as u128,
        c.num_experts as u128,
        c.num_experts_per_tok as u128,
        c.moe_intermediate_size as u128,
    );
    let routed = m.checked_mul(k)?;
    s.add(&[m, e], 4, 1)?;
    s.add(&[m, k], 4, 4)?;
    s.add(&[m, h], 2, 1)?;
    s.add(&[m], 4, 1)?;
    s.add(&[65536], 2, 1)?;
    if prefill {
        s.add(&[m.min(4), e], 4, 1)?;
        s.add(&[m.min(4)], 4, 1)?;
        s.add(&[e + 1], 4, 4)?;
        s.add(&[routed], 4, 1)?;
        // ceil(S/tile_rows) + min(E,S) <= S + min(E,S).
        let tiles = routed.checked_add(e.min(routed))?;
        s.add(&[tiles, inter.div_ceil(64), 4], 4, 1)?;
        s.add(&[tiles, h.div_ceil(64), 4], 4, 1)?;
        s.add(&[routed, h], 2, 2)?;
        s.add(&[routed, inter], 2, 3)?;
    } else {
        s.add(&[k, inter], 2, 1)?;
    }
    Some(())
}

fn scratch(
    c: &Qwen4ExpConfig,
    format: KvFormat,
    capacity: usize,
    rows: usize,
    mtp: bool,
) -> Option<Inventory> {
    let mut s = Inventory::default();
    let (h, wide, inter) = (
        c.hidden_size as u128,
        c.hidden_size as u128 * c.hc_count as u128,
        c.shared_expert_intermediate_size as u128,
    );
    let (nq, nkv, hd, nh) = (
        c.num_attention_heads as u128,
        c.num_key_value_heads as u128,
        c.head_dim as u128,
        c.indexer.n_heads as u128,
    );
    let (heads, hk) =
        (c.linear_num_value_heads as u128, c.linear_num_key_heads as u128);
    let dim_v = heads.checked_mul(c.linear_value_head_dim as u128)?;
    let conv = hk
        .checked_mul(c.linear_key_head_dim as u128)?
        .checked_mul(2)?
        .checked_add(dim_v)?;
    // Decode constructor: count each owning buffer once (fused projections
    // supply views for z/a/b, gate/up and the new K/V rows).
    for (width, copies) in [
        (h, 2),
        (wide, 1),
        (conv + dim_v + 2 * heads, 1),
        ((2 * nq + 2 * nkv) * hd, 1),
        (2 * inter, 1),
        (inter, 1),
        (conv, 1),
        (dim_v, 2),
        (nq * hd, 4),
        ((2 * nh + 1) * INDEXER_D as u128, 1),
    ] {
        s.add(&[width], 2, copies)?;
    }
    hc(&mut s, c, 1)?;
    ple(&mut s, c, 1)?;
    qsa(&mut s, c, capacity, 1)?;
    moe(&mut s, c, 1, false)?;
    s.add(&[c.vocab_size as u128], 4, 3)?;
    s.add(
        &[
            nq,
            sdpa_split_scratch_splits(capacity.min(c.indexer.dense_limit())) as u128,
            hd + 2,
        ],
        4,
        1,
    )?;
    // Sampler control arrays, token slots, synchronization and <=256-byte
    // metadata owners are bounded independently of the vocabulary.
    s.add(&[1024 * 1024], 1, 1)?;
    let ple_emb = c.ple.as_ref().map_or(0, |p| p.embed_dim as u128);
    let largest_projection = [
        conv * h,
        dim_v * h,
        heads * h,
        h * dim_v,
        2 * nq * hd * h,
        nkv * hd * h,
        h * nq * hd,
        c.num_experts as u128 * h,
        c.moe_intermediate_size as u128 * h,
        inter * h,
        c.hc_lowrank as u128 * wide,
        c.hc_count as u128 * wide,
        (nh + 1) * INDEXER_D as u128 * h,
        wide * ple_emb,
        h * ple_emb,
    ]
    .into_iter()
    .max()?;
    s.add(&[largest_projection], 2, 1)?;
    if mtp {
        let gdn = c
            .layer_types
            .iter()
            .filter(|t| matches!(t, LayerType::LinearAttention))
            .count() as u128;
        s.add(
            &[
                MAX_DRAFTS as u128,
                c.linear_num_value_heads as u128,
                GDN_HEAD_DIM as u128,
                GDN_HEAD_DIM as u128,
            ],
            4,
            gdn,
        )?;
        let conv = (2 * c.linear_num_key_heads as u128 * c.linear_key_head_dim as u128)
            + c.linear_num_value_heads as u128 * c.linear_value_head_dim as u128;
        s.add(&[(MAX_DRAFTS + 1) as u128, conv], 2, gdn)?;
        s.add(&[(MAX_DRAFTS + 1) as u128, c.hc_width() as u128], 2, 1)?;
        s.add(&[(MAX_DRAFTS + 1) as u128, c.vocab_size as u128], 4, 1)?;
        s.add(&[4 * 1024 * 1024], 1, 1)?;
    }
    if rows == 0 {
        return Some(s);
    }
    let m = rows as u128;
    hc(&mut s, c, m)?;
    ple(&mut s, c, m)?;
    moe(&mut s, c, m, true)?;
    qsa(&mut s, c, capacity, QSA_QUERY_BATCH.min(rows))?;
    for (width, copies) in [
        (h, 2),
        (wide, if mtp { 3 } else { 1 }),
        (inter, 3),
        (conv, 2),
        (dim_v, 3),
        (heads, 2),
        (2 * hk * GDN_HEAD_DIM as u128, 1),
        (2 * nq * hd, 1),
        (nq * hd, 4),
        (nkv * hd, 2),
        ((2 * nh + 1) * INDEXER_D as u128, 1),
    ] {
        s.add(&[m, width], 2, copies)?;
    }
    // Chunked GDN staging: QK norm above, decay/beta, W/U/P/G.
    s.add(&[m, heads], 4, 3)?;
    s.add(&[m, heads, GDN_HEAD_DIM as u128], 2, 2)?;
    s.add(&[m, heads, crate::kernels::gdn::GDN_CHUNK as u128], 2, 1)?;
    let stack = (conv + dim_v + 2 * heads).max((2 * nq + 2 * nkv) * hd).max(2 * inter);
    s.add(&[m, stack], 2, 1)?;
    s.add(&[m], 4, 1)?;
    s.add(&[m + c.indexer.compress_ratio as u128, 3], 4, 1)?;
    if format != KvFormat::Bf16 {
        s.add(&[nkv, c.indexer.dense_limit() as u128, hd], 2, 2)?;
    }
    Some(s)
}

impl Qwen4ExpModel {
    /// Request-owned streaming vision buffers plus host staging. Charge the
    /// complete new scratch even on reuse; existing allocations are resident
    /// in the OS headroom observation. No quadratic attention buffer exists.
    pub fn image_request_memory_bytes(
        &self,
        context: &MetalContext,
        patches: usize,
    ) -> Option<usize> {
        let inventory = vision_request_inventory(self.config.vision.as_ref()?, patches)?;
        if inventory.max_buffer > context.device().maxBufferLength() as u128 {
            return None;
        }
        usize::try_from(inventory.bytes).ok()
    }

    /// Peak text-request inventory without allocating tensors. Caches use the
    /// loaded KV format and include a loaded draft head even while disabled.
    pub fn request_memory_bytes(
        &self,
        context: &MetalContext,
        capacity: usize,
        input_tokens: usize,
    ) -> Option<usize> {
        let (bytes, max_buffer) = request_inventory(
            &self.config,
            self.kv_format,
            self.weights.mtp.is_some(),
            self.prefill_chunk,
            capacity,
            input_tokens,
        )?;
        (max_buffer <= context.device().maxBufferLength()).then_some(bytes)
    }

    /// Additional bytes a text request may need beyond allocations already
    /// resident in this loaded session. Context and prefill growth are charged
    /// at their replacement peak; retained state is not charged again.
    pub fn additional_request_memory_bytes(
        &self,
        context: &MetalContext,
        capacity: usize,
        input_tokens: usize,
        scratch: &Scratch,
    ) -> Option<usize> {
        let target_capacity = round_capacity(capacity).ok()?;
        let input_tokens = u32::try_from(input_tokens).ok()?;
        if input_tokens == 0 || target_capacity < input_tokens as usize {
            return None;
        }
        let mtp = self.weights.mtp.is_some();
        let format = self.kv_format;
        let (current_rows, resident_prefill_capacity) =
            scratch.resident_prefill_geometry(self.config.indexer.compress_ratio);
        let resident_capacity = round_capacity(
            scratch.resident_context_capacity(self.config.indexer.compress_ratio),
        )
        .ok()?;
        let bytes = additional_request_memory(
            &self.config,
            format,
            mtp,
            self.prefill_chunk,
            target_capacity,
            input_tokens as usize,
            ResidentAllocations {
                context_capacity: resident_capacity,
                prefill_rows: current_rows,
                prefill_capacity: resident_prefill_capacity,
            },
        )?;
        let (_, target_max_buffer) = request_inventory(
            &self.config,
            format,
            mtp,
            self.prefill_chunk,
            target_capacity,
            input_tokens as usize,
        )?;
        if target_max_buffer as u64 > context.device().maxBufferLength() as u64 {
            return None;
        }
        Some(bytes)
    }
}

fn vision_request_inventory(c: &crate::qwen4exp::config::VisionConfig, patches: usize) -> Option<Inventory> {
    if patches == 0 || patches > 65_536 || !patches.is_multiple_of(4) {
        return None;
    }
    let (n, h, i, out) = (patches as u128, c.hidden_size as u128,
        c.intermediate_size as u128, c.out_hidden_size as u128);
    let mut s = Inventory::default();
    // VisionScratch::new: pixels, x/h/qkv/attn/tmp, mlp, merged, rotary.
    s.add(&[n, c.patch_dim() as u128], 2, 1)?;
    s.add(&[n, h], 2, 7)?;
    s.add(&[n, i], 2, 1)?;
    s.add(&[n / 4, out], 2, 2)?; // scratch merged plus owned image rows
    s.add(&[n, 36], 4, 2)?;
    // Prepared BF16 patches, F32 upload, temporary BF16 upload and CPU rotary.
    s.add(&[n, c.patch_dim() as u128], 2, 4)?;
    s.add(&[n, 36], 4, 3)?;
    // Allocation/command metadata margin; available-memory admission adds its
    // separate host reserve. Checked products also bind the largest buffer.
    s.add(&[64 * 1024 * 1024], 1, 1)?;
    Some(s)
}

#[derive(Clone, Copy)]
struct ResidentAllocations {
    context_capacity: usize,
    prefill_rows: usize,
    prefill_capacity: usize,
}

fn additional_request_memory(
    c: &Qwen4ExpConfig,
    format: KvFormat,
    mtp: bool,
    prefill_chunk: usize,
    capacity: usize,
    input_tokens: usize,
    resident: ResidentAllocations,
) -> Option<usize> {
    let capacity = round_capacity(capacity).ok()?;
    let resident_capacity = round_capacity(resident.context_capacity).ok()?;
    let rows = input_tokens
        .max(if mtp { MAX_DRAFTS + 1 } else { 1 })
        .min(prefill_chunk)
        .checked_next_power_of_two()?
        .min(prefill_chunk);
    let old_rows = resident.prefill_rows.min(prefill_chunk);
    let old_prefill_capacity = resident.prefill_capacity.max(resident_capacity);
    let target_prefill_capacity = capacity.max(old_prefill_capacity);
    let target_rows =
        if capacity > resident_capacity { rows } else { rows.max(old_rows) };
    let target_scratch = scratch(c, format, target_prefill_capacity, target_rows, mtp)?;
    let scratch_bytes = if capacity > resident_capacity {
        // Decode scratch is rebuilt before state growth. The old owner remains
        // live while the complete replacement is constructed.
        target_scratch.bytes
    } else {
        let old_scratch = scratch(c, format, resident_capacity, old_rows, mtp)?;
        target_scratch.bytes.checked_sub(old_scratch.bytes)?
    };
    let caches = if capacity > resident_capacity {
        cache_bytes(c, format, mtp, capacity)? as u128
    } else {
        0
    };
    let snapshots = (snapshot_bytes(c, mtp)? as u128).checked_mul(2)?;
    let workspace = (input_tokens as u128)
        .checked_mul(16)?
        .checked_add((target_prefill_capacity as u128).checked_mul(16)?)?
        .checked_add((c.vocab_size as u128).checked_mul(64)?)?
        .checked_add(8 * 1024 * 1024)?;
    usize::try_from(
        caches
            .checked_add(scratch_bytes)?
            .checked_add(snapshots)?
            .checked_add(workspace)?,
    )
    .ok()
}

fn cache_bytes(
    c: &Qwen4ExpConfig,
    format: KvFormat,
    mtp: bool,
    capacity: usize,
) -> Option<usize> {
    let capacity = round_capacity(capacity).ok()? as u128;
    let attention =
        c.layer_types.iter().filter(|t| matches!(t, LayerType::FullAttention)).count()
            as u128
            + u128::from(mtp);
    let heads = c.num_key_value_heads as u128;
    let dim = c.head_dim as u128;
    heads.checked_mul(capacity)?.checked_mul(dim).filter(|&n| n <= u32::MAX as u128)?;
    let per_token = 2u128
        .checked_mul(heads)?
        .checked_mul(kv_row_bytes(format, dim)?)?
        .checked_add(INDEXER_D as u128 * 2)?;
    let blocks = (block_key_rows(capacity as usize, c.indexer.compress_ratio) as u128)
        .checked_mul(INDEXER_D as u128)?
        .checked_mul(2)?;
    let bytes = (attention.checked_mul(capacity)?.checked_mul(per_token)?)
        .checked_add(attention.checked_mul(blocks)?)?;
    usize::try_from(bytes).ok()
}

fn snapshot_bytes(c: &Qwen4ExpConfig, mtp: bool) -> Option<usize> {
    let state = (c.linear_num_value_heads as u128)
        .checked_mul(GDN_HEAD_DIM as u128)?
        .checked_mul(GDN_HEAD_DIM as u128)?
        .checked_mul(4)?;
    let conv = (c.gdn_conv_channels() as u128)
        .checked_mul(c.linear_conv_kernel_dim.checked_sub(1)? as u128)?
        .checked_mul(2)?;
    let ple = c.ple.as_ref().map_or(Some(0), |p| {
        (c.hc_width() as u128).checked_mul(p.conv_state_len() as u128)?.checked_mul(2)
    })?;
    let hidden = if mtp { (c.hc_width() as u128).checked_mul(2)? } else { 0 };
    let gdn_layers = c
        .layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::LinearAttention))
        .count() as u128;
    usize::try_from(
        gdn_layers
            .checked_mul(state.checked_add(conv)?)?
            .checked_add(ple)?
            .checked_add(hidden)?,
    )
    .ok()
}

fn request_inventory(
    c: &Qwen4ExpConfig,
    kv_format: KvFormat,
    mtp: bool,
    prefill_chunk: usize,
    capacity: usize,
    input_tokens: usize,
) -> Option<(usize, usize)> {
    // Loaded configs pass validate_flash_next: all widths/layer counts have
    // fixed small bounds, and the only large axis below is capacity <=MAX_SEQ.
    // Width expressions therefore fit u128; Inventory checks every product,
    // copy count, alignment and accumulated total before converting to usize.
    let capacity = round_capacity(capacity).ok()?;
    if input_tokens == 0 || input_tokens > capacity || c.indexer.compress_ratio == 0 {
        return None;
    }
    let attention =
        c.layer_types.iter().filter(|t| matches!(t, LayerType::FullAttention)).count()
            as u128
            + u128::from(mtp);
    let gdn = c
        .layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::LinearAttention))
        .count() as u128;
    let mut state = Inventory::default();
    let (heads, dim) = (c.num_key_value_heads as u128, c.head_dim as u128);
    heads
        .checked_mul(capacity as u128)?
        .checked_mul(dim)
        .filter(|&n| n <= u32::MAX as u128)?;
    state.add(
        &[heads, capacity as u128, dim],
        if kv_format == KvFormat::Bf16 { 2 } else { 1 },
        2 * attention,
    )?;
    if kv_format != KvFormat::Bf16 {
        // Quantized values and scale/bias pairs are distinct Metal buffers.
        state.add(
            &[heads, capacity as u128],
            kv_row_bytes(kv_format, dim)?.checked_sub(dim)?,
            2 * attention,
        )?;
    }
    state.add(&[capacity as u128, INDEXER_D as u128], 2, attention)?;
    state.add(
        &[
            block_key_rows(capacity, c.indexer.compress_ratio) as u128,
            INDEXER_D as u128,
        ],
        2,
        attention,
    )?;
    // One live state and two simultaneously retained/replaced snapshots.
    state.add(
        &[c.linear_num_value_heads as u128, GDN_HEAD_DIM as u128, GDN_HEAD_DIM as u128],
        4,
        3 * gdn,
    )?;
    let conv = (2 * c.linear_num_key_heads as u128 * c.linear_key_head_dim as u128)
        + c.linear_num_value_heads as u128 * c.linear_value_head_dim as u128;
    state.add(&[conv, c.linear_conv_kernel_dim.checked_sub(1)? as u128], 2, 4 * gdn)?;
    if let Some(p) = &c.ple {
        state.add(
            &[c.hidden_size as u128, c.hc_count as u128, p.conv_state_len() as u128],
            2,
            4,
        )?;
    }
    if mtp {
        state.add(&[c.hidden_size as u128, c.hc_count as u128], 2, 3)?;
    }
    let rows = input_tokens
        .max(if mtp { MAX_DRAFTS + 1 } else { 1 })
        .min(prefill_chunk)
        .checked_next_power_of_two()?
        .min(prefill_chunk);
    let scratch = scratch(c, kv_format, capacity, rows, mtp)?;
    // Growth may keep old/new cache and scratch owners alive together.
    let bytes = state
        .bytes
        .checked_add(scratch.bytes)?
        .checked_mul(2)?
        .checked_add((input_tokens as u128).checked_mul(16)?)?
        .checked_add((capacity as u128).checked_mul(16)?)?
        .checked_add((c.vocab_size as u128).checked_mul(64)?)?
        .checked_add(8 * 1024 * 1024)?;
    Some((
        usize::try_from(bytes).ok()?,
        usize::try_from(state.max_buffer.max(scratch.max_buffer)).ok()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen4exp::config::{
        EosTokenIds, GateAct, IndexerConfig, QuantizationConfig, RopeParameters,
        StoragePolicy,
    };

    #[test]
    fn retained_vision_inventory_covers_source_ceiling_and_rejects_invalid_extents() {
        let v = crate::qwen4exp::config::VisionConfig {
            depth: 27, hidden_size: 1152, num_heads: 16, intermediate_size: 4304,
            patch_size: 16, temporal_patch_size: 2, spatial_merge_size: 2,
            in_channels: 3, num_position_embeddings: 2304, out_hidden_size: 2560,
            hidden_act: "gelu_pytorch_tanh".into(), deepstack_visual_indexes: vec![],
            image_token_id: 1, video_token_id: 2, vision_start_token_id: 3,
            vision_end_token_id: 4, tensors: 0,
        };
        let small = vision_request_inventory(&v, 4096).unwrap();
        let large = vision_request_inventory(&v, 65_536).unwrap();
        assert_eq!(large.max_buffer, 65_536 * 4304 * 2);
        assert_eq!(large.bytes - 64 * 1024 * 1024,
            16 * (small.bytes - 64 * 1024 * 1024));
        assert!(large.bytes < 3 * 1024 * 1024 * 1024);
        for invalid in [0, 1, 65_537, 65_540, usize::MAX] {
            assert!(vision_request_inventory(&v, invalid).is_none());
        }
    }

    fn geometry() -> Qwen4ExpConfig {
        Qwen4ExpConfig {
            hidden_size: 2560,
            num_hidden_layers: 2,
            layer_types: vec![LayerType::LinearAttention, LayerType::FullAttention],
            vocab_size: 4096,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 1_048_576,
            eos_token_id: EosTokenIds::One(0),
            num_experts: 16,
            num_experts_per_tok: 2,
            moe_intermediate_size: 512,
            shared_expert_intermediate_size: 1024,
            norm_topk_prob: true,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 128,
            rope_parameters: RopeParameters {
                rope_theta: 10000.0,
                partial_rotary_factor: 0.25,
                mrope_section: None,
                mrope_interleaved: None,
            },
            linear_num_key_heads: 4,
            linear_num_value_heads: 8,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            gdn_gate: GateAct::Sigmoid,
            hc_count: 4,
            hc_lowrank: 320,
            ple: None,
            indexer: IndexerConfig {
                n_heads: 4,
                head_dim: 128,
                budget: 2048,
                compress_ratio: 4,
            },
            quantization: QuantizationConfig { group_size: 64, bits: 4 },
            storage: StoragePolicy::Q4,
            mtp: None,
            vision: None,
        }
    }

    #[test]
    fn cache_formats_include_scales_and_inventory_rejects_overflow() {
        assert_eq!(kv_row_bytes(KvFormat::Bf16, 128), Some(256));
        assert_eq!(kv_row_bytes(KvFormat::Q8, 128), Some(136));
        assert_eq!(kv_row_bytes(KvFormat::Q8Affine64, 128), Some(136));
        let mut inventory = Inventory::default();
        inventory.add(&[2, 17], 2, 2).unwrap();
        assert_eq!(inventory.bytes, 512);
        assert_eq!(inventory.max_buffer, 256);
        assert!(inventory.add(&[u128::MAX, 2], 4, 1).is_none());
    }

    #[test]
    fn request_inventory_grows_with_context_prompt_and_loaded_head() {
        let cfg = geometry();
        let short =
            request_inventory(&cfg, KvFormat::Q8, false, 4096, 16384, 4).unwrap();
        let long =
            request_inventory(&cfg, KvFormat::Q8, false, 4096, 131072, 4).unwrap();
        assert!(long > short);
        assert!(
            request_inventory(&cfg, KvFormat::Bf16, false, 4096, 16384, 4).unwrap()
                > short
        );
        assert!(
            request_inventory(&cfg, KvFormat::Q8, true, 4096, 16384, 4).unwrap()
                > short
        );
        assert!(
            request_inventory(&cfg, KvFormat::Q8, false, 4096, 16384, 4096).unwrap()
                > short
        );
        // The request estimate must cover the existing steady state plus two
        // snapshots; cache/scratch replacement is charged separately above it.
        let steady = session_bytes(&cfg, false, KvFormat::Q8, 16384, 2).unwrap();
        assert!(short.0 > steady as usize);
        assert!(short.1 < short.0);
        assert!(long.1 >= short.1);
        assert!(
            request_inventory(&cfg, KvFormat::Q8, false, 4096, MAX_SEQ + 1, 4)
                .is_none()
        );
    }

    #[test]
    fn incremental_request_admission_does_not_recharge_retained_state() {
        let cfg = geometry();
        let retained = additional_request_memory(
            &cfg,
            KvFormat::Q8,
            false,
            4096,
            16384,
            4,
            ResidentAllocations {
                context_capacity: 16384,
                prefill_rows: 2048,
                prefill_capacity: 16384,
            },
        )
        .unwrap();
        let cold =
            request_inventory(&cfg, KvFormat::Q8, false, 4096, 16384, 4).unwrap().0;
        assert!(retained < cold);

        // A tiny turn after a large prefill keeps the resident row allocation;
        // it must not add the already allocated cache or prefill buffers again.
        let tiny_after_large = additional_request_memory(
            &cfg,
            KvFormat::Q8,
            false,
            4096,
            16384,
            1,
            ResidentAllocations {
                context_capacity: 16384,
                prefill_rows: 2048,
                prefill_capacity: 16384,
            },
        )
        .unwrap();
        assert!(tiny_after_large < cold);
        assert!(tiny_after_large <= retained);
    }

    #[test]
    fn incremental_request_admission_charges_context_growth_overlap() {
        let cfg = geometry();
        let retained = additional_request_memory(
            &cfg,
            KvFormat::Q8,
            false,
            4096,
            16384,
            4,
            ResidentAllocations {
                context_capacity: 16384,
                prefill_rows: 1024,
                prefill_capacity: 16384,
            },
        )
        .unwrap();
        let growth = additional_request_memory(
            &cfg,
            KvFormat::Q8,
            false,
            4096,
            32768,
            4,
            ResidentAllocations {
                context_capacity: 16384,
                prefill_rows: 1024,
                prefill_capacity: 16384,
            },
        )
        .unwrap();
        assert!(growth > retained);
        assert!(growth >= cache_bytes(&cfg, KvFormat::Q8, false, 32768).unwrap());
    }
}
