// SPDX-License-Identifier: Apache-2.0
//! In-memory GLM recurrent snapshots. Completed MLA/indexer prefix rows stay
//! in the loaded model; only recurrent state and the incomplete pool are copied.

use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(1);

pub(super) fn next_instance() -> u64 {
    NEXT_INSTANCE.fetch_add(1, Ordering::Relaxed)
}

enum LayerSnapshot {
    Kda {
        #[cfg(not(feature = "metal"))]
        q_conv: Vec<f32>,
        #[cfg(not(feature = "metal"))]
        k_conv: Vec<f32>,
        #[cfg(not(feature = "metal"))]
        v_conv: Vec<f32>,
        #[cfg(not(feature = "metal"))]
        matrix: Vec<f32>,
        #[cfg(feature = "metal")]
        metal: super::super::glm_metal::KdaSnapshot,
    },
    // The CPU reference owns full vectors rather than a visible-cache cursor.
    // Metal's reference graph also uses these vectors for MLA execution.
    Attention(AttentionState),
}

pub struct PrefixSnapshot {
    instance: u64,
    position: usize,
    layers: Vec<LayerSnapshot>,
    #[cfg(feature = "metal")]
    attention: Vec<Option<super::super::glm_metal::AttentionSnapshot>>,
}

impl PrefixSnapshot {
    pub fn position(&self) -> usize {
        self.position
    }
}

impl GlmModel {
    /// A reuse grid independent of scalar or qualified 4/8-token prefill.
    pub fn prefill_chunk_tokens(&self) -> usize {
        1024
    }

    pub fn capture_prefix_at(
        &mut self,
        position: usize,
    ) -> Result<()> {
        #[cfg(feature = "metal")]
        if self.is_fenced() {
            return Err(Error::Invalid("GLM prefix capture requires a healthy model"));
        }
        let chunk = self.prefill_chunk_tokens();
        if position <= self.position
            || position >= self.config.max_context
            || !position.is_multiple_of(chunk)
            || !self.position.is_multiple_of(chunk)
        {
            return Err(Error::Invalid("GLM prefix capture position"));
        }
        self.capture_at = Some(position);
        self.captured = None;
        Ok(())
    }

    pub fn take_prefix_snapshot(&mut self) -> Option<PrefixSnapshot> {
        self.captured.take()
    }

    /// Called only with exclusive state ownership after every GPU writer is
    /// fenced. Neither cache buffers nor content ever leave worker memory.
    pub(super) fn capture_prefix_if_ready(&mut self) -> Result<()> {
        if self.capture_at != Some(self.position) {
            return Ok(());
        }
        self.capture_at = None;
        let _scope = super::super::profile::Scope::new("glm.prefix_snapshot");
        let layers = self
            .states
            .iter()
            .map(|state| match state {
                LayerState::Kda(state) => Ok(LayerSnapshot::Kda {
                    #[cfg(not(feature = "metal"))]
                    q_conv: state.q_conv.clone(),
                    #[cfg(not(feature = "metal"))]
                    k_conv: state.k_conv.clone(),
                    #[cfg(not(feature = "metal"))]
                    v_conv: state.v_conv.clone(),
                    #[cfg(not(feature = "metal"))]
                    matrix: state.matrix.clone(),
                    #[cfg(feature = "metal")]
                    metal: state
                        .metal
                        .as_ref()
                        .ok_or(Error::Invalid("GLM prefix KDA state unavailable"))?
                        .snapshot()?,
                }),
                LayerState::Attention(state) => Ok(LayerSnapshot::Attention(state.clone())),
            })
            .collect::<Result<Vec<_>>>()?;
        #[cfg(feature = "metal")]
        let attention = self.decode.as_ref().map_or_else(
            || Ok(Vec::new()),
            |decode| {
                decode
                    .attention
                    .iter()
                    .map(|state| state.as_ref().map(|state| state.snapshot()).transpose())
                    .collect::<Result<Vec<_>>>()
            },
        )?;
        self.captured = Some(PrefixSnapshot {
            instance: self.prefix_instance,
            position: self.position,
            layers,
            #[cfg(feature = "metal")]
            attention,
        });
        Ok(())
    }

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
        #[cfg(feature = "metal")]
        if self.is_fenced() {
            return Err(Error::Invalid("GLM prefix restore requires a healthy model"));
        }
        if self.position != 0
            || snapshot.instance != self.prefix_instance
            || snapshot.position == 0
            || snapshot.position >= self.config.max_context
            || !snapshot.position.is_multiple_of(self.prefill_chunk_tokens())
            || snapshot.layers.len() != self.states.len()
        {
            return Err(Error::Invalid("GLM prefix restore admission"));
        }
        let _scope = super::super::profile::Scope::new("glm.prefix_restore");
        for (state, saved) in self.states.iter_mut().zip(&snapshot.layers) {
            match (state, saved) {
                (
                    LayerState::Kda(state),
                    LayerSnapshot::Kda {
                        #[cfg(not(feature = "metal"))]
                        q_conv,
                        #[cfg(not(feature = "metal"))]
                        k_conv,
                        #[cfg(not(feature = "metal"))]
                        v_conv,
                        #[cfg(not(feature = "metal"))]
                        matrix,
                        #[cfg(feature = "metal")]
                        metal,
                    },
                ) => {
                    #[cfg(not(feature = "metal"))]
                    {
                        state.q_conv.clone_from(q_conv);
                        state.k_conv.clone_from(k_conv);
                        state.v_conv.clone_from(v_conv);
                        state.matrix.clone_from(matrix);
                    }
                    #[cfg(feature = "metal")]
                    state.metal.as_mut().ok_or(Error::Invalid("GLM prefix KDA state unavailable"))?.restore(metal)?;
                },
                (LayerState::Attention(state), LayerSnapshot::Attention(saved)) => state.clone_from(saved),
                _ => return Err(Error::Invalid("GLM prefix layer state mismatch")),
            }
        }
        #[cfg(feature = "metal")]
        match (&mut self.decode, snapshot.attention.as_slice()) {
            (None, []) => {},
            (Some(decode), saved) if decode.attention.len() == saved.len() => {
                for (state, saved) in decode.attention.iter_mut().zip(saved) {
                    match (state, saved) {
                        (None, None) => {},
                        (Some(state), Some(saved)) => state.restore(saved)?,
                        _ => return Err(Error::Invalid("GLM prefix attention state mismatch")),
                    }
                }
            },
            _ => return Err(Error::Invalid("GLM prefix decode state mismatch")),
        }
        self.position = snapshot.position;
        #[cfg(feature = "metal")]
        {
            self.last_greedy = None;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    /// Private small source-bound weights exercise full GLM state, including
    /// sparse indexer pool completion, capture during bulk ingest and decode.
    #[stark_test]
    #[ignore]
    fn glm_prefix_snapshot_restores_full_forward_and_rejects_other_instances() {
        let path = std::env::var_os("STARK_GLM_REFERENCE_FIXTURE_PATH")
            .map(std::path::PathBuf::from)
            .expect("private GLM fixture path required");
        let config: Value =
            serde_json::from_slice(&std::fs::read(path.join("config.json")).expect("private GLM config read failed"))
                .expect("private GLM config invalid");
        let seed: Vec<u32> =
            serde_json::from_slice(&std::fs::read(path.join("tokens.json")).expect("private GLM tokens read failed"))
                .expect("private GLM tokens invalid");
        assert!(!seed.is_empty());
        let create = || {
            let mut model = GlmModel::new(Weights::open(&path).expect("private GLM weights invalid"), &config)
                .expect("private GLM admission failed");
            model.config.max_context = 4096;
            model
        };
        let mut model = create();
        let chunk = model.prefill_chunk_tokens();
        let prompt: Vec<u32> = seed.iter().copied().cycle().take(chunk + 17).collect();
        let longer: Vec<u32> = prompt.iter().copied().chain(seed.iter().rev().copied().cycle().take(13)).collect();
        let run = |model: &mut GlmModel, input: &[u32], saved: Option<&PrefixSnapshot>, capture| {
            model.reset();
            let start = if let Some(saved) = saved {
                model.restore_prefix(saved).expect("private GLM prefix restore failed");
                saved.position()
            } else {
                0
            };
            if let Some(position) = capture {
                model.capture_prefix_at(position).expect("private GLM prefix capture admission failed");
            }
            #[cfg(feature = "metal")]
            model
                .ingest_prefix(&input[start..input.len() - 1], &mut || Ok(()))
                .expect("private GLM bulk prefill failed");
            #[cfg(not(feature = "metal"))]
            for &token in &input[start..input.len() - 1] {
                model.ingest(token).expect("private GLM prefill failed");
            }
            let mut logits = model.step(*input.last().unwrap()).expect("private GLM readout failed");
            logits.extend(model.step(seed[0]).expect("private GLM decode failed"));
            (logits, model.take_prefix_snapshot())
        };
        let same =
            |a: &[f32], b: &[f32]| a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits());
        let (plain, none) = run(&mut model, &prompt, None, None);
        assert!(none.is_none());
        let (split, snapshot) = run(&mut model, &prompt, None, Some(chunk));
        assert!(same(&plain, &split), "GLM prefix capture changed fresh logits");
        let snapshot = snapshot.expect("GLM prefix snapshot unavailable");
        assert_eq!(snapshot.position(), chunk);
        let (fresh, _) = run(&mut model, &longer, None, None);
        let (resumed, _) = run(&mut model, &longer, Some(&snapshot), None);
        assert!(same(&fresh, &resumed), "GLM prefix restore changed logits");
        let mut other = create();
        assert!(other.restore_prefix(&snapshot).is_err(), "GLM accepted another loaded model's snapshot");
        model.reset();
        assert!(model.capture_prefix_at(chunk + 1).is_err());
        assert!(model.capture_prefix_at(0).is_err());
        model.ingest(seed[0]).unwrap();
        assert!(model.restore_prefix(&snapshot).is_err());
        assert_eq!(model.position, 0);
        assert!(model.take_prefix_snapshot().is_none());
    }
}
