// SPDX-License-Identifier: Apache-2.0
//! Complete, private F32 attention snapshots for Flint generation prefixes.
//!
//! Full-attention rows and sliding windows are copied. This preserves ordinary
//! reset semantics and does not depend on rows surviving a later request.

use super::*;

const PREFILL_CHUNK_TOKENS: usize = 1024;

pub struct PrefixSnapshot {
    pub(super) instance: Arc<()>,
    pub(super) position: usize,
    pub(super) states: Vec<AttentionState>,
}

impl PrefixSnapshot {
    pub fn position(&self) -> usize {
        self.position
    }
}

fn valid_states(
    config: &Config,
    position: usize,
    states: &[AttentionState],
) -> bool {
    let width = config.kv_heads * config.dim;
    states.len() == config.layers.len()
        && states.iter().zip(&config.layers).all(|(state, layer)| {
            let rows = if layer.sliding {
                position.min(config.window)
            } else {
                position
            };
            state.keys.len() == rows
                && state.values.len() == rows
                && state
                    .keys
                    .iter()
                    .chain(&state.values)
                    .all(|row| row.len() == width && row.iter().all(|value| value.is_finite()))
        })
}

impl FlintModel {
    /// A publication grid over unchanged, sequential F32 attention execution.
    pub fn prefill_chunk_tokens(&self) -> usize {
        PREFILL_CHUNK_TOKENS
    }

    /// Capture after a complete token at the start of the final prompt chunk.
    pub fn capture_prefix_at(
        &mut self,
        position: usize,
    ) -> Result<()> {
        let chunk = self.prefill_chunk_tokens();
        if position <= self.position
            || position >= self.config.context
            || !position.is_multiple_of(chunk)
            || !self.position.is_multiple_of(chunk)
        {
            self.reset();
            return Err(Error::Invalid("flint_prefix_capture_position"));
        }
        self.capture_at = Some(position);
        self.captured = None;
        Ok(())
    }

    pub fn take_prefix_snapshot(&mut self) -> Option<PrefixSnapshot> {
        self.captured.take()
    }

    pub(super) fn capture_prefix_if_ready(&mut self) -> Result<()> {
        if self.capture_at != Some(self.position) {
            return Ok(());
        }
        self.capture_at = None;
        if !valid_states(&self.config, self.position, &self.states) {
            return Err(Error::Invalid("flint_prefix_capture_state"));
        }
        let _scope = super::super::profile::Scope::new("flint.prefix_snapshot");
        // Matrix operations return completed host vectors. There is no GPU
        // writer or shared cache alias left in these deques when a token ends.
        self.captured = Some(PrefixSnapshot {
            instance: Arc::clone(&self.prefix_instance),
            position: self.position,
            states: self.states.clone(),
        });
        Ok(())
    }

    /// Restore the entire attention state of this loaded model after reset.
    /// The worker independently binds the snapshot to the preceding request's
    /// token prefix, context, host scope and state-affecting options.
    pub fn restore_prefix(
        &mut self,
        snapshot: &PrefixSnapshot,
    ) -> Result<()> {
        let result = self.restore_prefix_inner(snapshot);
        if result.is_err() {
            self.reset();
        }
        result
    }

    fn restore_prefix_inner(
        &mut self,
        snapshot: &PrefixSnapshot,
    ) -> Result<()> {
        if self.position != 0
            || !Arc::ptr_eq(&self.prefix_instance, &snapshot.instance)
            || snapshot.position == 0
            || snapshot.position >= self.config.context
            || !snapshot.position.is_multiple_of(self.prefill_chunk_tokens())
            || !valid_states(&self.config, snapshot.position, &snapshot.states)
        {
            return Err(Error::Invalid("flint_prefix_restore_state"));
        }
        let _scope = super::super::profile::Scope::new("flint.prefix_restore");
        self.states.clone_from(&snapshot.states);
        self.position = snapshot.position;
        Ok(())
    }
}
