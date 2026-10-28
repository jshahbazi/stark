//! Private, owned image inputs with explicit Qwen or Gemma position semantics.
//!
//! A projected image is bound to the loaded decoder that produced it. Validation
//! here is entirely on the host and must finish before any device allocation or
//! state mutation. These inputs intentionally have no serialization or Debug
//! implementation: image features and token sequences are request data.

use std::{ops::Range, sync::Arc};

use half::bf16;

use crate::{backends::common::InvalidInput, data_type::DataType};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct ImageGrid {
    pub(crate) temporal: u32,
    pub(crate) height: u32,
    pub(crate) width: u32,
    pub(crate) merge_size: u32,
}

#[derive(Clone, Copy)]
pub(crate) struct ImageTokenIds {
    pub(crate) image: u64,
    pub(crate) vision_start: u64,
    pub(crate) vision_end: u64,
    pub(crate) video: u64,
}

pub(crate) struct ProjectedImage {
    pub(crate) model_identity: Arc<()>,
    pub(crate) model_dim: u32,
    pub(crate) data_type: DataType,
    pub(crate) grid: ImageGrid,
    /// Index of the first image placeholder, after its vision-start token.
    pub(crate) token_start: usize,
    /// Row-major projected features, with one model-width row per placeholder.
    pub(crate) features: Box<[bf16]>,
}

pub(crate) struct MultimodalPrefill {
    pub(crate) tokens: Box<[u64]>,
    pub(crate) image: ProjectedImage,
    /// Spatial rotary coordinates for Qwen; sequential triples for Gemma host
    /// validation only (Gemma dispatch uses ordinary scalar text RoPE).
    pub(crate) positions: Box<[[u32; 3]]>,
    /// Applied only to rotary positions for subsequent text, never KV offsets.
    pub(crate) position_delta: i64,
    pub(crate) mode: MultimodalMode,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MultimodalMode {
    Qwen35Spatial,
    Gemma4Sequential,
}

impl ImageGrid {
    fn merged_shape(self) -> Result<(u32, u32, u32), InvalidInput> {
        if self.temporal != 1 {
            return Err(InvalidInput("only one static image is supported"));
        }
        if self.height == 0
            || self.width == 0
            || self.merge_size == 0
            || !self.height.is_multiple_of(self.merge_size)
            || !self.width.is_multiple_of(self.merge_size)
        {
            return Err(InvalidInput("invalid image grid or spatial merge"));
        }
        let patches = self
            .temporal
            .checked_mul(self.height)
            .and_then(|count| count.checked_mul(self.width))
            .ok_or(InvalidInput("image patch count overflow"))?;
        let merge_area =
            self.merge_size.checked_mul(self.merge_size).ok_or(InvalidInput("image merge area overflow"))?;
        let height = self.height / self.merge_size;
        let width = self.width / self.merge_size;
        let rows = height.checked_mul(width).ok_or(InvalidInput("image feature row count overflow"))?;
        if patches / merge_area != rows {
            return Err(InvalidInput("inconsistent image grid"));
        }
        Ok((height, width, rows))
    }
}

impl MultimodalPrefill {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new(
        tokens: Box<[u64]>,
        image: ProjectedImage,
        special_ids: ImageTokenIds,
        expected_identity: &Arc<()>,
        expected_model_dim: u32,
        expected_dtype: DataType,
        vocab_size: u32,
        context_capacity: u32,
    ) -> Result<Self, InvalidInput> {
        if !Arc::ptr_eq(&image.model_identity, expected_identity) {
            return Err(InvalidInput("projected image belongs to a different loaded model"));
        }
        if expected_model_dim == 0 || image.model_dim != expected_model_dim {
            return Err(InvalidInput("projected image width differs from decoder width"));
        }
        if expected_dtype != DataType::BF16 || image.data_type != expected_dtype {
            return Err(InvalidInput("projected image requires matching BF16 decoder data"));
        }
        let token_count = u32::try_from(tokens.len()).map_err(|_| InvalidInput("multimodal token count overflow"))?;
        if token_count == 0 || token_count > context_capacity {
            return Err(InvalidInput("multimodal prompt exceeds context capacity"));
        }
        let (height, width, rows) = image.grid.merged_shape()?;
        let feature_count =
            rows.checked_mul(image.model_dim).ok_or(InvalidInput("projected image element count overflow"))?;
        let feature_bytes = feature_count
            .checked_mul(u32::try_from(size_of::<bf16>()).expect("BF16 size fits u32"))
            .ok_or(InvalidInput("projected image byte count overflow"))?;
        let feature_bytes =
            usize::try_from(feature_bytes).map_err(|_| InvalidInput("projected image size overflow"))?;
        if feature_bytes > isize::MAX as usize || image.features.len() != feature_count as usize {
            return Err(InvalidInput("projected image feature shape differs from placeholders"));
        }
        if image.features.iter().any(|value| !value.is_finite()) {
            return Err(InvalidInput("projected image contains nonfinite features"));
        }
        let ids = [special_ids.image, special_ids.vision_start, special_ids.vision_end, special_ids.video];
        if ids.iter().enumerate().any(|(index, id)| *id >= u64::from(vocab_size) || ids[..index].contains(id)) {
            return Err(InvalidInput("invalid or overlapping image special token IDs"));
        }
        if tokens.iter().any(|token| *token >= u64::from(vocab_size)) {
            return Err(InvalidInput("multimodal token is outside decoder vocabulary"));
        }
        let image_start = image.token_start;
        let image_end =
            image_start.checked_add(rows as usize).ok_or(InvalidInput("image placeholder span overflow"))?;
        let wrapper_start = image_start.checked_sub(1).ok_or(InvalidInput("missing image vision-start wrapper"))?;
        if tokens.get(wrapper_start) != Some(&special_ids.vision_start)
            || tokens.get(image_end) != Some(&special_ids.vision_end)
        {
            return Err(InvalidInput("image span must have immediate vision wrappers"));
        }
        for (index, &token) in tokens.iter().enumerate() {
            let is_image = (image_start..image_end).contains(&index);
            if is_image && token != special_ids.image {
                return Err(InvalidInput("image placeholder span is not contiguous"));
            }
            if (token == special_ids.image && !is_image)
                || (token == special_ids.vision_start && index != wrapper_start)
                || (token == special_ids.vision_end && index != image_end)
                || token == special_ids.video
            {
                return Err(InvalidInput("surplus or unsupported vision special token"));
            }
        }

        // Transformers 2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397,
        // Qwen3_5Model.get_rope_index: text groups advance all three axes;
        // the image group advances the temporal/height/width axes separately.
        let base = u32::try_from(image_start).map_err(|_| InvalidInput("image position overflow"))?;
        let tail_base = base.checked_add(height.max(width)).ok_or(InvalidInput("image rotary position overflow"))?;
        let mut positions = Vec::new();
        positions.try_reserve_exact(tokens.len()).map_err(|_| InvalidInput("multimodal position allocation failed"))?;
        for index in 0..image_start {
            let position = u32::try_from(index).map_err(|_| InvalidInput("text rotary position overflow"))?;
            positions.push([position; 3]);
        }
        for row in 0..height {
            for column in 0..width {
                positions.push([
                    base,
                    base.checked_add(row).ok_or(InvalidInput("image height position overflow"))?,
                    base.checked_add(column).ok_or(InvalidInput("image width position overflow"))?,
                ]);
            }
        }
        for offset in 0..(tokens.len() - image_end) {
            let offset = u32::try_from(offset).map_err(|_| InvalidInput("text rotary position overflow"))?;
            let position = tail_base.checked_add(offset).ok_or(InvalidInput("text rotary position overflow"))?;
            positions.push([position; 3]);
        }
        let last_position =
            positions.iter().flatten().copied().max().ok_or(InvalidInput("empty multimodal position sequence"))?;
        let position_delta = i64::from(last_position)
            .checked_add(1)
            .and_then(|position| position.checked_sub(i64::from(token_count)))
            .ok_or(InvalidInput("multimodal position delta overflow"))?;
        Ok(Self {
            tokens,
            image,
            positions: positions.into_boxed_slice(),
            position_delta,
            mode: MultimodalMode::Qwen35Spatial,
        })
    }

    /// Gemma image slots consume ordinary sequential text positions. Only PLE
    /// token lookup uses token 0; decoder/cache token IDs remain the real IDs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new_gemma4(
        tokens: Box<[u64]>,
        image: ProjectedImage,
        special_ids: ImageTokenIds,
        audio_token_id: u64,
        expected_identity: &Arc<()>,
        expected_model_dim: u32,
        expected_dtype: DataType,
        vocab_size: u32,
        context_capacity: u32,
    ) -> Result<Self, InvalidInput> {
        if audio_token_id >= u64::from(vocab_size)
            || [special_ids.image, special_ids.vision_start, special_ids.vision_end, special_ids.video]
                .contains(&audio_token_id)
            || tokens.contains(&audio_token_id)
        {
            return Err(InvalidInput("unsupported or overlapping Gemma audio token"));
        }
        // Share ownership, dtype, finite-feature, exact wrapper/span, vocabulary
        // and checked-grid validation with the existing image path.
        let mut input = Self::try_new(
            tokens,
            image,
            special_ids,
            expected_identity,
            expected_model_dim,
            expected_dtype,
            vocab_size,
            context_capacity,
        )?;
        for (index, position) in input.positions.iter_mut().enumerate() {
            *position = [u32::try_from(index).map_err(|_| InvalidInput("Gemma image position overflow"))?; 3];
        }
        input.position_delta = 0;
        input.mode = MultimodalMode::Gemma4Sequential;
        Ok(input)
    }

    pub(crate) fn uses_spatial_positions(&self) -> bool {
        self.mode == MultimodalMode::Qwen35Spatial
    }

    /// Return the chunk-local token rows and corresponding feature rows to copy.
    /// Chunk boundaries are physical cache/token offsets, independent of M-RoPE.
    pub(crate) fn chunk_feature_range(
        &self,
        chunk_start: usize,
        chunk_len: usize,
    ) -> Result<Option<(Range<usize>, Range<usize>)>, InvalidInput> {
        let chunk_end = chunk_start.checked_add(chunk_len).ok_or(InvalidInput("prefill chunk range overflow"))?;
        if chunk_end > self.tokens.len() {
            return Err(InvalidInput("prefill chunk exceeds prompt"));
        }
        let image_start = self.image.token_start;
        let image_rows = self.image.features.len() / self.image.model_dim as usize;
        let image_end = image_start.checked_add(image_rows).ok_or(InvalidInput("image feature range overflow"))?;
        let start = chunk_start.max(image_start);
        let end = chunk_end.min(image_end);
        if start >= end {
            return Ok(None);
        }
        Ok(Some((start - chunk_start..end - chunk_start, start - image_start..end - image_start)))
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;
    use crate::encodable_block::transformer::scalar_continuation_position;

    const SPECIAL_IDS: ImageTokenIds = ImageTokenIds {
        image: 90,
        vision_start: 91,
        vision_end: 92,
        video: 93,
    };

    fn image(
        identity: &Arc<()>,
        grid: ImageGrid,
        token_start: usize,
    ) -> ProjectedImage {
        let rows = grid.temporal * (grid.height / grid.merge_size) * (grid.width / grid.merge_size);
        ProjectedImage {
            model_identity: identity.clone(),
            model_dim: 4,
            data_type: DataType::BF16,
            grid,
            token_start,
            features: vec![bf16::from_f32(0.5); rows as usize * 4].into_boxed_slice(),
        }
    }

    fn non_square_grid() -> ImageGrid {
        ImageGrid {
            temporal: 1,
            height: 4,
            width: 6,
            merge_size: 2,
        }
    }

    fn input(
        identity: &Arc<()>,
        tokens: &[u64],
        projected: ProjectedImage,
    ) -> Result<MultimodalPrefill, InvalidInput> {
        MultimodalPrefill::try_new(tokens.into(), projected, SPECIAL_IDS, identity, 4, DataType::BF16, 100, 32)
    }

    #[stark_test]
    fn static_image_positions_match_pinned_reference_groups() {
        let identity = Arc::new(());
        let prefill =
            input(&identity, &[1, 2, 91, 90, 90, 90, 90, 90, 90, 92, 3], image(&identity, non_square_grid(), 3))
                .unwrap();
        // Independently expanded text/image/text groups from the pinned Python
        // get_rope_index and get_vision_position_ids reference.
        assert_eq!(
            prefill.positions.as_ref(),
            &[
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [3, 3, 4],
                [3, 3, 5],
                [3, 4, 3],
                [3, 4, 4],
                [3, 4, 5],
                [6, 6, 6],
                [7, 7, 7],
            ]
        );
        assert_eq!(prefill.position_delta, -3);
        let continuation =
            [0, 1, 2].map(|height| scalar_continuation_position(11, height, prefill.position_delta).unwrap());
        assert_eq!(continuation, [8, 9, 10]);
    }

    #[stark_test]
    fn gemma_images_keep_sequential_positions_and_zero_continuation_delta() {
        let identity = Arc::new(());
        let grid = ImageGrid {
            temporal: 1,
            height: 6,
            width: 9,
            merge_size: 3,
        };
        let tokens = [1, 91, 90, 90, 90, 90, 90, 90, 92, 2];
        let prefill = MultimodalPrefill::try_new_gemma4(
            tokens.into(),
            image(&identity, grid, 2),
            SPECIAL_IDS,
            94,
            &identity,
            4,
            DataType::BF16,
            100,
            32,
        )
        .unwrap();
        assert!(!prefill.uses_spatial_positions());
        assert_eq!(prefill.position_delta, 0);
        assert_eq!(prefill.positions.as_ref(), &(0..tokens.len() as u32).map(|index| [index; 3]).collect::<Vec<_>>());
        assert_eq!(prefill.chunk_feature_range(0, 4).unwrap(), Some((2..4, 0..2)));
        assert_eq!(prefill.chunk_feature_range(4, 4).unwrap(), Some((0..4, 2..6)));
        assert_eq!(prefill.chunk_feature_range(8, 2).unwrap(), None);
        assert_eq!(prefill.tokens.as_ref(), &tokens);
    }

    #[stark_test]
    fn gemma_images_reject_audio_forged_spans_and_foreign_features() {
        let identity = Arc::new(());
        let grid = ImageGrid {
            temporal: 1,
            height: 6,
            width: 9,
            merge_size: 3,
        };
        let valid = [1, 91, 90, 90, 90, 90, 90, 90, 92, 2];
        for audio in [90, 91, 92, 93, 100] {
            assert!(
                MultimodalPrefill::try_new_gemma4(
                    valid.into(),
                    image(&identity, grid, 2),
                    SPECIAL_IDS,
                    audio,
                    &identity,
                    4,
                    DataType::BF16,
                    100,
                    32
                )
                .is_err()
            );
        }
        for (index, token) in [(0, 94), (9, 94), (0, 90), (1, 2), (3, 2), (8, 2)] {
            let mut tokens = valid;
            tokens[index] = token;
            assert!(
                MultimodalPrefill::try_new_gemma4(
                    tokens.into(),
                    image(&identity, grid, 2),
                    SPECIAL_IDS,
                    94,
                    &identity,
                    4,
                    DataType::BF16,
                    100,
                    32
                )
                .is_err()
            );
        }
        assert!(
            MultimodalPrefill::try_new_gemma4(
                valid.into(),
                image(&Arc::new(()), grid, 2),
                SPECIAL_IDS,
                94,
                &identity,
                4,
                DataType::BF16,
                100,
                32
            )
            .is_err()
        );
        assert!(
            MultimodalPrefill::try_new_gemma4(
                valid.into(),
                image(&identity, grid, 2),
                SPECIAL_IDS,
                94,
                &identity,
                4,
                DataType::BF16,
                100,
                9
            )
            .is_err()
        );
    }

    #[stark_test]
    fn image_fusion_ranges_cross_multiple_prefill_chunks() {
        let identity = Arc::new(());
        let prefill =
            input(&identity, &[1, 2, 91, 90, 90, 90, 90, 90, 90, 92, 3], image(&identity, non_square_grid(), 3))
                .unwrap();
        assert_eq!(prefill.chunk_feature_range(0, 3).unwrap(), None);
        assert_eq!(prefill.chunk_feature_range(0, 4).unwrap(), Some((3..4, 0..1)));
        assert_eq!(prefill.chunk_feature_range(4, 4).unwrap(), Some((0..4, 1..5)));
        assert_eq!(prefill.chunk_feature_range(8, 3).unwrap(), Some((0..1, 5..6)));
        assert_eq!(prefill.chunk_feature_range(9, 2).unwrap(), None);
        assert_eq!(prefill.chunk_feature_range(4, 0).unwrap(), None);
        assert!(prefill.chunk_feature_range(10, 2).is_err());
        assert!(prefill.chunk_feature_range(usize::MAX, 1).is_err());
    }

    #[stark_test]
    fn one_row_and_one_column_images_preserve_reference_delta() {
        let identity = Arc::new(());
        for (height, width) in [(2, 8), (8, 2), (2, 2)] {
            let grid = ImageGrid {
                temporal: 1,
                height,
                width,
                merge_size: 2,
            };
            let count = (height / 2 * (width / 2)) as usize;
            let mut tokens = vec![91];
            tokens.extend(std::iter::repeat_n(90, count));
            tokens.push(92);
            let prefill = input(&identity, &tokens, image(&identity, grid, 1)).unwrap();
            assert_eq!(prefill.position_delta, 0);
            assert_eq!(prefill.positions.last().copied(), Some([tokens.len() as u32 - 1; 3]));
        }
    }

    #[stark_test]
    fn rejects_forged_or_extra_vision_special_tokens() {
        let identity = Arc::new(());
        let valid = [1, 91, 90, 90, 90, 90, 90, 90, 92, 2];
        for (index, replacement) in [(0, 90), (0, 91), (0, 92), (0, 93), (1, 2), (2, 3), (7, 3), (8, 2), (9, 90)] {
            let mut tokens = valid;
            tokens[index] = replacement;
            assert!(input(&identity, &tokens, image(&identity, non_square_grid(), 2)).is_err());
        }
        assert!(input(&identity, &valid, image(&identity, non_square_grid(), 0)).is_err());
        assert!(input(&identity, &valid, image(&identity, non_square_grid(), usize::MAX)).is_err());
        assert!(input(&identity, &valid[..8], image(&identity, non_square_grid(), 2)).is_err());
    }

    #[stark_test]
    fn rejects_foreign_model_dtype_shape_and_nonfinite_features() {
        let identity = Arc::new(());
        let tokens = [91, 90, 90, 90, 90, 90, 90, 92];
        assert!(input(&identity, &tokens, image(&Arc::new(()), non_square_grid(), 1)).is_err());
        let mut projected = image(&identity, non_square_grid(), 1);
        projected.model_dim = 3;
        assert!(input(&identity, &tokens, projected).is_err());
        let mut projected = image(&identity, non_square_grid(), 1);
        projected.data_type = DataType::F16;
        assert!(input(&identity, &tokens, projected).is_err());
        let mut projected = image(&identity, non_square_grid(), 1);
        projected.features = projected.features[..23].into();
        assert!(input(&identity, &tokens, projected).is_err());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut projected = image(&identity, non_square_grid(), 1);
            projected.features[0] = bf16::from_f32(value);
            assert!(input(&identity, &tokens, projected).is_err());
        }
    }

    #[stark_test]
    fn rejects_invalid_grid_and_checked_shape_arithmetic() {
        let identity = Arc::new(());
        let tokens = [91, 90, 90, 90, 90, 90, 90, 92];
        for grid in [
            ImageGrid {
                temporal: 0,
                ..non_square_grid()
            },
            ImageGrid {
                temporal: 2,
                ..non_square_grid()
            },
            ImageGrid {
                height: 0,
                ..non_square_grid()
            },
            ImageGrid {
                width: 0,
                ..non_square_grid()
            },
            ImageGrid {
                merge_size: 0,
                ..non_square_grid()
            },
            ImageGrid {
                height: 3,
                ..non_square_grid()
            },
            ImageGrid {
                width: 5,
                ..non_square_grid()
            },
            ImageGrid {
                temporal: 1,
                height: u32::MAX,
                width: 2,
                merge_size: 1,
            },
            ImageGrid {
                temporal: 1,
                height: 65536,
                width: 65536,
                merge_size: 65536,
            },
        ] {
            let mut projected = image(&identity, non_square_grid(), 1);
            projected.grid = grid;
            assert!(input(&identity, &tokens, projected).is_err());
        }
        let mut projected = image(&identity, non_square_grid(), 1);
        projected.model_dim = u32::MAX;
        assert!(
            MultimodalPrefill::try_new(
                tokens.into(),
                projected,
                SPECIAL_IDS,
                &identity,
                u32::MAX,
                DataType::BF16,
                100,
                32
            )
            .is_err()
        );
        let mut projected = image(&identity, non_square_grid(), 1);
        projected.model_dim = u32::MAX / 8;
        assert!(
            MultimodalPrefill::try_new(
                tokens.into(),
                projected,
                SPECIAL_IDS,
                &identity,
                u32::MAX / 8,
                DataType::BF16,
                100,
                32
            )
            .is_err()
        );
    }

    #[stark_test]
    fn validates_vocabulary_special_ids_and_context_capacity() {
        let identity = Arc::new(());
        let tokens = [91, 90, 90, 90, 90, 90, 90, 92];
        for ids in [
            ImageTokenIds {
                image: 100,
                ..SPECIAL_IDS
            },
            ImageTokenIds {
                vision_start: 90,
                ..SPECIAL_IDS
            },
            ImageTokenIds {
                vision_end: 91,
                ..SPECIAL_IDS
            },
            ImageTokenIds {
                video: 92,
                ..SPECIAL_IDS
            },
        ] {
            assert!(
                MultimodalPrefill::try_new(
                    tokens.into(),
                    image(&identity, non_square_grid(), 1),
                    ids,
                    &identity,
                    4,
                    DataType::BF16,
                    100,
                    32
                )
                .is_err()
            );
        }
        let mut invalid_tokens = tokens;
        invalid_tokens[0] = 100;
        assert!(input(&identity, &invalid_tokens, image(&identity, non_square_grid(), 1)).is_err());
        for capacity in [0, 7] {
            assert!(
                MultimodalPrefill::try_new(
                    tokens.into(),
                    image(&identity, non_square_grid(), 1),
                    SPECIAL_IDS,
                    &identity,
                    4,
                    DataType::BF16,
                    100,
                    capacity
                )
                .is_err()
            );
        }
        assert!(
            MultimodalPrefill::try_new(
                tokens.into(),
                image(&identity, non_square_grid(), 1),
                SPECIAL_IDS,
                &identity,
                4,
                DataType::F16,
                100,
                32
            )
            .is_err()
        );
    }
}
