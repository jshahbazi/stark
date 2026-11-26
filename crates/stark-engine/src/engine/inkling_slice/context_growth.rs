// SPDX-License-Identifier: Apache-2.0
//! Preserve live state while replacing context-sized physical caches.

use super::*;

impl Session {
    /// Grow the physical cache capacity while retaining the committed prefix.
    ///
    /// Replacement caches are allocated before the old caches are released.
    /// A single blit copies the live target and active DSpark state, so a failed
    /// allocation or copy leaves the old context and state installed.
    pub fn grow_context(
        &mut self,
        context: usize,
    ) -> Result<()> {
        if context <= self.context {
            return Err("Inkling context growth must increase capacity");
        }
        Self::check_context(&self.runtime, context)?;

        let visits = Self::new_visits(&self.runtime, &self.layers, context)?;
        let draft_cache = self
            .dspark
            .as_ref()
            .map(|draft| draft.allocate_context_cache(&self.runtime.device, context))
            .transpose()?;

        let command = self.runtime.queue.command_buffer().ok_or("Inkling context growth command allocation failed")?;
        let encoder = command.blit_command_encoder().ok_or("Inkling context growth encoder allocation failed")?;

        for (old, new) in self.visits.iter().zip(&visits) {
            let kv_bytes = self.position.min(old.capacity) * KV * 2;
            encoder.copy_buffer_to_buffer(&old.keys, 0, &new.keys, 0, kv_bytes);
            encoder.copy_buffer_to_buffer(&old.values, 0, &new.values, 0, kv_bytes);
            for (old_conv, new_conv) in old.conv.iter().zip(&new.conv) {
                encoder.copy_buffer_to_buffer(old_conv, 0, new_conv, 0, old_conv.length());
            }
        }

        if self.dspark_enabled
            && let (Some(draft), Some(next)) = (&self.dspark, &draft_cache)
        {
            draft.encode_context_cache_copy(next, &encoder, self.position);
        }

        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err("Inkling context growth copy failed");
        }

        self.visits = visits;
        if let (Some(draft), Some(next)) = (&mut self.dspark, draft_cache) {
            draft.install_context_cache(next);
        }
        self.context = context;
        self.apply_prefill_scope();
        self.ensure_source_workspace()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use stark_engine_macros::stark_test;

    use super::*;

    const INITIAL_CONTEXT: usize = 8;
    const LARGER_CONTEXT: usize = 12;
    const PREFIX: [u32; 4] = [1, 7, 19, 23];
    const CONTINUATION: u32 = 31;
    const PROPOSAL_ANCHOR: u32 = 47;

    fn state_digest(
        session: &Session,
        dspark: bool,
    ) -> Result<String> {
        if dspark {
            Ok(transaction_validation::state_digest(session)?.0)
        } else {
            Ok(target_validation::state_digest(session))
        }
    }

    fn continue_and_record(
        session: &mut Session,
        dspark: bool,
    ) -> Result<(Vec<f32>, Option<u32>, Vec<u16>, String)> {
        let logits = session.step(CONTINUATION)?;
        let proposal = if dspark {
            Some(session.draft_propose(PROPOSAL_ANCHOR)?.ok_or("Inkling growth test expected a DSpark proposal")?)
        } else {
            None
        };
        let features = session.draft_features().ok_or("Inkling growth test expected captured features")?;
        let state = state_digest(session, dspark)?;
        Ok((logits, proposal, features, state))
    }

    fn compare_growth(
        session: &mut Session,
        dspark: bool,
    ) -> Result<()> {
        session.set_context(INITIAL_CONTEXT)?;
        session.set_dspark_enabled(dspark)?;
        session.capture_draft_features(true);
        session.ingest_many(&PREFIX[..PREFIX.len() - 1])?;
        session.step(PREFIX[PREFIX.len() - 1])?;
        let position = session.position();
        let features_before_growth =
            session.draft_features().ok_or("Inkling growth test expected captured features")?;

        session.grow_context(LARGER_CONTEXT)?;
        assert_eq!(session.context(), LARGER_CONTEXT);
        assert_eq!(session.position(), position);
        assert_eq!(session.draft_features().as_deref(), Some(features_before_growth.as_slice()));
        let grown = continue_and_record(session, dspark)?;

        // Rebuild the same prefix in this one session to get an independent
        // larger cache without holding a second full model resident.
        session.set_context(INITIAL_CONTEXT)?;
        session.set_context(LARGER_CONTEXT)?;
        session.ingest_many(&PREFIX[..PREFIX.len() - 1])?;
        session.step(PREFIX[PREFIX.len() - 1])?;
        let fresh = continue_and_record(session, dspark)?;
        if !transaction_validation::same_logits(&grown.0, &fresh.0) {
            return Err("Inkling live growth changed continuation logits");
        }
        if grown.1 != fresh.1 {
            return Err("Inkling live growth changed the DSpark proposal");
        }
        if grown.2 != fresh.2 {
            return Err("Inkling live growth changed captured target features");
        }
        if grown.3 != fresh.3 {
            return Err("Inkling live growth changed committed cache state");
        }
        Ok(())
    }

    // Run with STARK_TEST_INKLING_PATH and STARK_TEST_INKLING_DSPARK_PATH set
    // to the private target and DSpark SafeTensors files.
    #[stark_test]
    #[ignore]
    fn live_growth_matches_fresh_larger_target_and_dspark_caches() {
        let target = PathBuf::from(std::env::var_os("STARK_TEST_INKLING_PATH").expect("set private Inkling target"));
        let draft = PathBuf::from(
            std::env::var_os("STARK_TEST_INKLING_DSPARK_PATH").expect("set private Inkling DSpark checkpoint"),
        );
        let mut session = Session::load(&target, INITIAL_CONTEXT).expect("load Inkling target");
        session.load_dspark(&draft).expect("load Inkling DSpark");

        compare_growth(&mut session, false).expect("target-only live context growth");
        compare_growth(&mut session, true).expect("active DSpark live context growth");
    }
}
