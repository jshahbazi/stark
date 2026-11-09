// SPDX-License-Identifier: Apache-2.0
//! GLM-specific still-image components, independently admitted from decoder
//! shards. Components do not advertise image execution before fusion and
//! actual-model lifecycle qualification are complete.

pub(super) mod artifact;
pub(super) mod preprocess;
mod tower;

use std::{path::Path, sync::Arc};

pub(super) use tower::VisionTower;

use super::{Error, PreparedHybridImage, Result, image};

const TOKENS: image::Tokens = image::Tokens {
    start: 154830,
    end: 154831,
    pad: 154854,
    video: 154855,
};

/// The tower, source receipt and opaque patches share one loaded owner. No
/// image embedding is cached or exposed to the host process.
pub(super) struct Images {
    pub artifact: artifact::Artifact,
    tower: VisionTower,
    owner: Arc<()>,
}

impl Images {
    pub fn load(
        path: &Path,
        artifact: artifact::Artifact,
    ) -> Result<Self> {
        let tower = VisionTower::load(&path.join(artifact::SIDECAR))?;
        artifact.unchanged(path)?;
        Ok(Self {
            artifact,
            tower,
            owner: Arc::new(()),
        })
    }

    pub fn prepare(
        &self,
        encoded: &[u8],
    ) -> Result<PreparedHybridImage> {
        Ok(PreparedHybridImage {
            owner: self.owner.clone(),
            prepared: image::Patches::Glm(
                preprocess::prepare(encoded).map_err(|_| Error::Invalid("GLM image preprocessing failed"))?,
            ),
        })
    }

    fn bound(
        &self,
        image: &PreparedHybridImage,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.owner, &image.owner) || !matches!(&image.prepared, image::Patches::Glm(_)) {
            return Err(Error::Invalid("model-bound GLM image"));
        }
        Ok(())
    }

    pub fn prompt(
        &self,
        tokens: &[u32],
        image: &PreparedHybridImage,
        capacity: u32,
    ) -> Result<Box<[u32]>> {
        self.bound(image)?;
        reject_video(tokens)?;
        TOKENS.expand(tokens, image.image_token_count() as usize, capacity as usize)
    }

    pub fn span(
        &self,
        tokens: &[u32],
        image: &PreparedHybridImage,
    ) -> Result<usize> {
        self.bound(image)?;
        reject_video(tokens)?;
        TOKENS.span(tokens, image.image_token_count() as usize)
    }

    pub fn encode(
        &self,
        image: PreparedHybridImage,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<f32>> {
        self.bound(&image)?;
        let image::Patches::Glm(patches) = image.prepared else {
            unreachable!()
        };
        self.tower.forward(&patches, cancel)
    }
}

fn reject_video(tokens: &[u32]) -> Result<()> {
    if tokens.iter().any(|token| matches!(token, 154832 | 154833 | 154855)) {
        return Err(Error::Invalid("GLM still-image request contains video controls"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn still_image_contract_rejects_every_video_control() {
        for token in [154832, 154833, 154855] {
            assert!(reject_video(&[154830, 154854, 154831, token]).is_err());
        }
        assert!(reject_video(&[154830, 154854, 154831]).is_ok());
    }
}
