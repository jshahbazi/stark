use super::*;
use crate::backends::common::{
    BufferMut,
    kernel::{Kernels, radix_top_k_small::RadixTopKSmall},
};

pub(crate) const DRAFT_CANDIDATES: u32 = 64;

pub(crate) struct CoarseDraftReadout<B: Backend> {
    linear: crate::encodable_block::linear::LinearMatmul<B>,
    shortlist: <B::Kernels as Kernels>::RadixTopKSmall,
}

impl<B: Backend> Embedding<B> {
    pub(crate) fn coarse_draft_readout(
        &self,
        context: &B::Context,
        bits: u32,
    ) -> Result<Option<CoarseDraftReadout<B>>, EmbeddingError<B>> {
        if self.data_type != DataType::BF16 || self.vocab_size < 65536 || self.logit_transform.is_some() {
            return Ok(None);
        }
        let EmbeddingTying::Untied {
            output,
            ..
        } = &self.tying
        else {
            return Ok(None);
        };
        let Some(linear) = output.coarse_affine(context, bits)? else {
            return Ok(None);
        };
        let shortlist =
            <B::Kernels as Kernels>::RadixTopKSmall::new_with_data_type(context, self.vocab_size, DataType::F32)
                .map_err(EmbeddingError::BackendError)?;
        Ok(Some(CoarseDraftReadout {
            linear,
            shortlist,
        }))
    }
}

impl<B: Backend> CoarseDraftReadout<B> {
    pub(crate) fn encode_scores(
        &self,
        input: impl BufferRef<Backend = B>,
        command: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        self.linear.encode_with_a(
            MatmulA::FullPrecision {
                values: input,
                offset: 0,
            },
            1,
            None::<Gather<&B::ScratchBuffer>>,
            command,
        )
    }

    pub(crate) fn encode(
        &self,
        embedding: &Embedding<B>,
        input: impl BufferRef<Backend = B>,
        command: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(B::ScratchBuffer, B::ScratchBuffer), EmbeddingError<B>> {
        let scores = self.encode_scores(input, command).map_err(EmbeddingError::BackendError)?;
        let mut ids = command.allocate_scratch(DRAFT_CANDIDATES as usize * 4).map_err(EmbeddingError::BackendError)?;
        let mut ranked =
            command.allocate_scratch(DRAFT_CANDIDATES as usize * 4).map_err(EmbeddingError::BackendError)?;
        self.shortlist
            .encode(&scores, (&mut ids).reborrow(), &mut ranked, 1, DRAFT_CANDIDATES, command)
            .map_err(EmbeddingError::BackendError)?;
        let exact = embedding.encode_readout(1, input, DRAFT_CANDIDATES, Some(&ids), true, command)?;
        Ok((ids, exact))
    }
}
