// SPDX-License-Identifier: Apache-2.0
//! GLM-specific still-image components, independently admitted from decoder
//! shards. Components do not advertise image execution before fusion and
//! actual-model lifecycle qualification are complete.

mod preprocess;
mod tower;

pub(super) use tower::VisionTower;
