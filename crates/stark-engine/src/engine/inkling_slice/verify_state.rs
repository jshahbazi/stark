// SPDX-License-Identifier: Apache-2.0
//! Short source verifier and rollback of overwritten ring rows.

use super::{prefill::VerifyExperts, *};
use crate::engine::inkling_slice::dspark::DraftBlock;

pub const MAX_VERIFY_ROWS: usize = 4;
const KV_ROW_BYTES: usize = KV * size_of::<u16>();
const CONV_ROW_BYTES: [usize; 4] = [KV_ROW_BYTES, KV_ROW_BYTES, WIDTH * 2, WIDTH * 2];
const CONV_OFFSETS: [usize; 4] = [0, KV_ROW_BYTES, 2 * KV_ROW_BYTES, 2 * KV_ROW_BYTES + WIDTH * 2];
const CONV_BYTES: usize = 2 * KV_ROW_BYTES + 4 * WIDTH;
const SAVED_VISIT_BYTES: usize = 2 * MAX_VERIFY_ROWS * KV_ROW_BYTES + 3 * CONV_BYTES;

pub(super) struct VerifyWorkspace {
    saved: Buffer,
    anchors: Buffer,
    attention_parts: Option<Buffer>,
    pub(super) local_rows_per_group: usize,
    pub(super) global_share: (usize, usize),
    lanes: PrefillLanes,
    draft: DraftBlock,
}

impl VerifyWorkspace {
    fn new(
        runtime: &mut Runtime,
        visits: usize,
    ) -> Result<Self> {
        let mut lanes = PrefillLanes::for_m5_rows(runtime, MAX_VERIFY_ROWS)?;
        lanes.enable_verify_groups(runtime)?;
        packed_prefill::register_verify_nvfp4_matvec(runtime)?;
        packed_prefill::register_parallel_route(runtime)?;
        packed_prefill::register(runtime, include_str!("verify_state.metal"), &["ink_save_anchor_convolution"])?;
        let shared_attention =
            std::env::var_os("STARK_INKLING_EXPERIMENT_SHARED_LOCAL").as_deref() == Some(std::ffi::OsStr::new("1"));
        let shared_global =
            std::env::var_os("STARK_INKLING_EXPERIMENT_SHARED_GLOBAL").as_deref() == Some(std::ffi::OsStr::new("1"));
        if shared_attention || shared_global {
            packed_prefill::register(
                runtime,
                include_str!("verify_attention.metal"),
                &["ink_verify_local_parts", "ink_verify_global_parts"],
            )?;
        }
        let draft = DraftBlock::new(runtime)?;
        let alloc = |bytes| {
            runtime
                .device
                .new_buffer(bytes, MTLResourceOptions::STORAGE_MODE_PRIVATE)
                .ok_or("Inkling verifier rollback allocation failed")
        };
        Ok(Self {
            saved: alloc(visits * SAVED_VISIT_BYTES)?,
            anchors: alloc(visits * CONV_BYTES)?,
            attention_parts: (shared_attention || shared_global)
                .then(|| alloc(MAX_VERIFY_ROWS * HEADS * 64 * 130 * 4))
                .transpose()?,

            global_share: if shared_global {
                (3, 1)
            } else {
                (0, 0)
            },
            local_rows_per_group: if shared_attention {
                2
            } else {
                0
            },
            lanes,
            draft,
        })
    }

    fn save(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        session: &Session,
        rows: usize,
    ) -> Result<()> {
        let blit = command.blit_command_encoder().ok_or("Inkling verifier save encoder allocation failed")?;
        for (index, visit) in session.visits.iter().enumerate() {
            let base = index * SAVED_VISIT_BYTES;
            for (plane, dst) in [&visit.keys, &visit.values].into_iter().enumerate() {
                for row in 0..rows {
                    let position = session.position + row;
                    if position >= visit.capacity {
                        let saved = base + (plane * MAX_VERIFY_ROWS + row) * KV_ROW_BYTES;
                        let cache = (position % visit.capacity) * KV_ROW_BYTES;
                        blit.copy_buffer_to_buffer(dst, cache, &self.saved, saved, KV_ROW_BYTES);
                    }
                }
            }
            for (i, buffer) in visit.conv.iter().enumerate() {
                let saved = base + 2 * MAX_VERIFY_ROWS * KV_ROW_BYTES + CONV_OFFSETS[i] * 3;
                blit.copy_buffer_to_buffer(buffer, 0, &self.saved, saved, CONV_ROW_BYTES[i] * 3);
            }
        }
        blit.end_encoding();
        Ok(())
    }

    fn restore_suffix(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        session: &Session,
        rows: usize,
        kept: usize,
    ) -> Result<()> {
        if kept == rows {
            return Ok(());
        }
        let blit = command.blit_command_encoder().ok_or("Inkling verifier undo encoder allocation failed")?;
        for (index, visit) in session.visits.iter().enumerate() {
            let base = index * SAVED_VISIT_BYTES;
            for row in kept..rows {
                let position = session.position + row;
                if position >= visit.capacity {
                    for (plane, dst) in [&visit.keys, &visit.values].into_iter().enumerate() {
                        let saved = base + (plane * MAX_VERIFY_ROWS + row) * KV_ROW_BYTES;
                        let cache = (position % visit.capacity) * KV_ROW_BYTES;
                        blit.copy_buffer_to_buffer(&self.saved, saved, dst, cache, KV_ROW_BYTES);
                    }
                }
                for (i, buffer) in visit.conv.iter().enumerate() {
                    let stride = CONV_ROW_BYTES[i];
                    let slot = (position % 3) * stride;
                    // Row 3 overwrites the accepted anchor's ring slot. Other
                    // rejected rows overwrite disjoint slots from before the call.
                    if row == 3 {
                        let saved = index * CONV_BYTES + CONV_OFFSETS[i];
                        blit.copy_buffer_to_buffer(&self.anchors, saved, buffer, slot, stride);
                    } else {
                        let saved = base + 2 * MAX_VERIFY_ROWS * KV_ROW_BYTES + CONV_OFFSETS[i] * 3 + slot;
                        blit.copy_buffer_to_buffer(&self.saved, saved, buffer, slot, stride);
                    }
                }
            }
        }
        blit.end_encoding();
        Ok(())
    }

    pub(super) fn shares_local_attention(&self) -> bool {
        self.local_rows_per_group > 0 && self.attention_parts.is_some()
    }

    pub(super) fn encode_shared_local(
        &self,
        gpu: &mut Stream<'_>,
        index: usize,
        layer: &Layer,
        visit: &Visit,
        position: usize,
        rows: &[&Scratch],
    ) {
        let parts = self.attention_parts.as_ref().expect("shared local verifier workspace");
        for (row, scratch) in rows.iter().enumerate() {
            layer.attention_write_cache(gpu, scratch, visit, [(position + row) as u32, LOCAL as u32, LOCAL as u32, 64]);
            gpu.one(
                "ink_prefill_relative_table_source",
                &[&scratch.r, &layer.relative, &scratch.table],
                HEADS as u32,
                (LOCAL / 32, HEADS.div_ceil(16)),
                32,
            );
        }
        let [a, b, c, d] = std::array::from_fn(|index| rows[index.min(rows.len() - 1)]);
        gpu.dispatch(
            "ink_verify_local_parts",
            &[
                binding(&a.q),
                binding(&b.q),
                binding(&c.q),
                binding(&d.q),
                binding(&a.table),
                binding(&b.table),
                binding(&c.table),
                binding(&d.table),
                binding(&a.sink),
                binding(&b.sink),
                binding(&c.sink),
                binding(&d.sink),
                binding(&visit.keys),
                binding(&visit.values),
                (&self.saved, index * SAVED_VISIT_BYTES),
                binding(parts),
            ],
            [position as u32, rows.len() as u32, self.local_rows_per_group as u32, 0],
            (16, HEADS * rows.len().div_ceil(self.local_rows_per_group)),
            128,
        );
        for (row, scratch) in rows.iter().enumerate() {
            gpu.dispatch(
                "ink_attention_source_gate",
                &[(parts, row * HEADS * 64 * 130 * 4), binding(&scratch.attn_gate), binding(&scratch.attended)],
                [64u32, 64],
                (8, HEADS),
                128,
            );
        }
    }

    pub(super) fn shares_global_attention(&self) -> bool {
        self.global_share.0 != 0 && self.attention_parts.is_some()
    }

    pub(super) fn encode_shared_global(
        &self,
        gpu: &mut Stream<'_>,
        layer: &Layer,
        visit: &Visit,
        position: usize,
        rows: &[&Scratch],
    ) {
        let parts = self.attention_parts.as_ref().expect("shared global verifier workspace");
        let (heads_per_group, rows_per_group) = self.global_share;
        gpu.phase(PrefillPhase::KeyValue);
        for (row, scratch) in rows.iter().enumerate() {
            layer.attention_write_cache(gpu, scratch, visit, [(position + row) as u32, visit.capacity as u32, 0, 64]);
        }
        gpu.phase(PrefillPhase::RelativeTable);
        for scratch in rows {
            gpu.one(
                "ink_prefill_relative_table_source",
                &[&scratch.r, &layer.relative, &scratch.table],
                HEADS as u32,
                (LOCAL / 32, HEADS.div_ceil(16)),
                32,
            );
        }
        gpu.phase(PrefillPhase::GlobalAttention);
        let [a, b, c, d] = std::array::from_fn(|index| rows[index.min(rows.len() - 1)]);
        gpu.dispatch(
            "ink_verify_global_parts",
            &[
                binding(&a.q),
                binding(&b.q),
                binding(&c.q),
                binding(&d.q),
                binding(&a.table),
                binding(&b.table),
                binding(&c.table),
                binding(&d.table),
                binding(&a.sink),
                binding(&b.sink),
                binding(&c.sink),
                binding(&d.sink),
                binding(&visit.keys),
                binding(&visit.values),
                binding(parts),
            ],
            [position as u32, rows.len() as u32, rows_per_group as u32, heads_per_group as u32],
            (16, HEADS / heads_per_group * rows.len().div_ceil(rows_per_group)),
            128,
        );
        for (row, scratch) in rows.iter().enumerate() {
            gpu.dispatch(
                "ink_attention_source_gate",
                &[(parts, row * HEADS * 64 * 130 * 4), binding(&scratch.attn_gate), binding(&scratch.attended)],
                [64u32, 64],
                (8, HEADS),
                128,
            );
        }
    }

    pub(super) fn save_anchor(
        &self,
        gpu: &mut Stream<'_>,
        index: usize,
        scratch: &Scratch,
        experts: Binding<'_>,
    ) {
        gpu.dispatch(
            "ink_save_anchor_convolution",
            &[
                binding(&scratch.k),
                binding(&scratch.v),
                binding(&scratch.attn_raw),
                experts,
                (&self.anchors, index * CONV_BYTES),
            ],
            (),
            ((2 * KV + 2 * WIDTH) / 128, 1),
            128,
        );
    }
}

/// A completed target command with provisional cache writes. The caller
/// commits one or more verified inputs before the session can be used again.
pub struct Verification<'a> {
    session: Option<&'a mut Session>,
    rows: usize,
}

impl<'a> Verification<'a> {
    pub fn logits(
        &self,
        row: usize,
    ) -> Result<&[f32]> {
        if row >= self.rows {
            return Err("Inkling verifier readout is outside the block");
        }
        // The command completed and this transaction has exclusive access to
        // the session. The logits cannot change while the slice is borrowed.
        let values = unsafe {
            let session = self.session.as_ref().ok_or("Inkling verifier is already committed")?;
            let base = session.logits.contents().as_ptr().cast::<f32>();
            std::slice::from_raw_parts(base.add(row * PADDED_VOCABULARY), VOCABULARY)
        };
        if values.iter().any(|x| !x.is_finite()) {
            return Err("Inkling verifier produced nonfinite logits");
        }
        Ok(values)
    }

    pub fn commit(
        mut self,
        kept: usize,
    ) -> Result<&'a mut Session> {
        if kept == 0 || kept > self.rows {
            return Err("Inkling verifier commit is outside the block");
        }
        let session = self.session.as_deref_mut().ok_or("Inkling verifier is already committed")?;
        let workspace = session.verifier.as_ref().ok_or("Inkling verifier workspace unavailable")?;
        let draft = session.active_dspark().ok_or("Inkling verifier draft unavailable")?;
        let position = session.position;
        execute_prepared(
            &session.runtime,
            None,
            |command| workspace.restore_suffix(command, session, self.rows, kept),
            |gpu| {
                for (row, taps) in workspace.lanes.verify_taps(session, self.rows).take(kept).enumerate() {
                    draft.commit(gpu, taps, position + row);
                }
            },
        )?;
        session.position += kept;
        session.features_ready = false;
        self.session.take().ok_or("Inkling verifier is already committed")
    }
}

impl Drop for Verification<'_> {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            session.reset();
        }
    }
}

impl Session {
    pub const SOURCE_VERIFIER_RESERVE_BYTES: usize = 96 * 1024 * 1024;

    /// Enable source-mode draft prefill and reserve the short verifier.
    /// Select the mode at a request boundary after setting the context.
    pub fn set_source_verifier(
        &mut self,
        enabled: bool,
    ) -> Result<()> {
        if self.position != 0 {
            return Err("Inkling verifier mode must be selected before prefill");
        }
        if enabled {
            if !self.has_dspark() || !self.supports_source_text_prefill(self.context) {
                return Err("Inkling source verifier requires the source M5 path and DSpark");
            }
            if self.verifier.is_none() {
                let workspace = VerifyWorkspace::new(&mut self.runtime, self.visits.len())?;
                self.packed_prefill
                    .as_mut()
                    .ok_or("Inkling packed prefill unavailable")?
                    .enable_draft_taps(&mut self.runtime)?;
                self.verifier = Some(workspace);
            }
        }
        self.source_verifier = enabled;
        self.apply_prefill_scope();
        Ok(())
    }

    pub fn propose_verify(
        &self,
        anchor: u32,
        depth: usize,
    ) -> Result<Vec<u32>> {
        if !self.source_verifier
            || !self.runtime.source_text_active
            || self.position == 0
            || depth == 0
            || depth >= MAX_VERIFY_ROWS
            || anchor as usize >= VOCABULARY
            || self.context - self.position < depth + 1
        {
            return Err("Inkling draft proposal exceeds the verifier extent");
        }
        let workspace = self.verifier.as_ref().ok_or("Inkling verifier workspace unavailable")?;
        let draft = self.active_dspark().ok_or("Inkling verifier draft unavailable")?;
        if depth == 1 {
            return draft
                .propose(&self.runtime, anchor, self.position, &self.embedding, &self.embed_norm, &self.unembedding)
                .map(|token| vec![token]);
        }
        draft.propose_many(
            &self.runtime,
            anchor,
            self.position,
            &self.embedding,
            &self.embed_norm,
            &self.unembedding,
            depth,
            &workspace.draft,
        )
    }

    pub fn begin_verify(
        &mut self,
        tokens: &[u32],
    ) -> Result<Verification<'_>> {
        self.begin_verify_profiled(tokens, None).map(|(verification, _)| verification)
    }

    pub(super) fn begin_verify_profiled(
        &mut self,
        tokens: &[u32],
        profile: Option<&mut PrefillTiming>,
    ) -> Result<(Verification<'_>, Timing)> {
        self.check_append(tokens.len())?;
        if !self.source_verifier
            || !self.runtime.source_text_active
            || self.active_dspark().is_none()
            || !(2..=MAX_VERIFY_ROWS).contains(&tokens.len())
            || tokens.iter().any(|&token| token as usize >= VOCABULARY)
        {
            return Err("Inkling short verifier requires source draft mode and two to four tokens");
        }
        let workspace = self.verifier.as_ref().ok_or("Inkling verifier workspace unavailable")?;
        let parallel = self.runtime.parallel_route;
        let matvec = self.runtime.verify_nvfp4_matvec;
        self.runtime.parallel_route = true;
        self.runtime.verify_nvfp4_matvec = true;
        let command = execute_prepared(
            &self.runtime,
            profile,
            |command| workspace.save(command, self, tokens.len()),
            |gpu| self.encode_verify_transaction(gpu, &workspace.lanes, tokens, VerifyExperts::Grouped, workspace),
        );
        self.runtime.verify_nvfp4_matvec = matvec;
        self.runtime.parallel_route = parallel;
        match command {
            Ok(timing) => Ok((
                Verification {
                    session: Some(self),
                    rows: tokens.len(),
                },
                timing,
            )),
            Err(error) => {
                self.reset();
                Err(error)
            },
        }
    }
}
