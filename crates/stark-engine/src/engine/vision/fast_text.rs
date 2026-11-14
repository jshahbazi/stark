//! The ordinary source-converted Qwen text graph, sharing raw admitted weights.
use super::*;

pub(super) fn config(config: &LanguageModelConfig) -> Result<LanguageModelConfig, serde_json::Error> {
    fn ordinary_text(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(fields) => {
                // Both conversion modes use identical norm cast_f32 payloads:
                // affine scales are already baked, while BF16 uses the same
                // explicit scale_offset=1. Preserve offsets and weight specs.
                if fields.get("type").and_then(serde_json::Value::as_str) == Some("Qwen35MRoPEConfig") {
                    fields.insert("type".into(), "UnscaledRoPEConfig".into());
                    fields.remove("mrope_section");
                    fields.remove("interleaved");
                }
                if let Some(reference) = fields.get_mut("reference_bf16") {
                    *reference = false.into();
                }
                if let Some(upcast) = fields.get_mut("upcast_mode") {
                    *upcast = "only_normalization".into();
                }
                for child in fields.values_mut() {
                    ordinary_text(child);
                }
            },
            serde_json::Value::Array(children) => children.iter_mut().for_each(ordinary_text),
            _ => {},
        }
    }
    let mut value = serde_json::to_value(config)?;
    ordinary_text(&mut value);
    serde_json::from_value(value)
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    #[ignore]
    fn private_canonical_text_config_matches_source_conversion() {
        let image = std::env::var("STARK_QWEN_IMAGE_MODEL").expect("private image artifact required");
        let text = std::env::var("STARK_QWEN_TEXT_MODEL").expect("private text artifact required");
        let image: serde_json::Value =
            serde_json::from_reader(File::open(Path::new(&image).join("config.json")).unwrap()).unwrap();
        let text: LanguageModelConfig =
            serde_json::from_reader(File::open(Path::new(&text).join("config.json")).unwrap()).unwrap();
        let image: ArtifactConfig = serde_json::from_value(image).unwrap();
        let image: LanguageModelConfig = serde_json::from_value(serde_json::json!({
            "type": "LanguageModelConfig", "decoder_config": image.decoder_config,
            "token_codec_config": image.token_codec_config, "generation_config": image.generation_config,
            "mtp_config": image.mtp_config,
        }))
        .unwrap();
        let actual = serde_json::to_value(config(&image).unwrap()).unwrap();
        let expected = serde_json::to_value(text).unwrap();
        assert!(actual["decoder_config"] == expected["decoder_config"], "private target graph mismatch");
        assert!(actual["mtp_config"] == expected["mtp_config"], "private predictor graph mismatch");
    }

    #[cfg(backend = "metal")]
    #[stark_test]
    #[ignore]
    fn private_fast_text_state_identity_and_resident_storage() {
        use crate::backends::metal::Metal;
        let image = std::env::var("STARK_QWEN_IMAGE_MODEL").expect("private image artifact required");
        let engine = Engine::<Metal>::new().unwrap();
        let mut model = engine.load_vision_language_model(Path::new(&image)).unwrap();
        assert!(model.text_language_model.is_some());
        let mut image_state = model.create_empty_state(Some(128), 42).unwrap();
        let mut text_state = model.create_text_state(Some(128), 42).unwrap();
        let options = model.default_stream_options();
        assert!(model.stream_text(&[100], &mut image_state, options).is_err());
        assert!(!model.is_fenced());
        let options = model.default_stream_options();
        assert!(model.stream(&[100], &mut text_state, options).is_err());
        assert!(!model.is_fenced());
        let session = model.text_model().prefill_dense_mtp_controlled(&[100, 101], 128, &mut || false).unwrap();
        let mut foreign_session = session;
        assert!(matches!(
            model.language_model.restore_dense_mtp_prefix(&mut foreign_session, 1),
            Err(DenseMtpError::Invalid)
        ));
        assert!(!foreign_session.is_fenced());
        assert!(!model.is_fenced());
        let stats = model.parameter_storage_statistics();
        assert!(stats.shared_matrix_bytes > 10_000_000_000);
        assert!(stats.resident_bytes < 21_000_000_000);
        let receipt = std::env::var("STARK_QWEN_STORAGE_RECEIPT").expect("private aggregate receipt required");
        let file = OpenOptions::new().write(true).create_new(true).open(receipt).unwrap();
        serde_json::to_writer(file, &stats).unwrap();
    }

    #[cfg(backend = "metal")]
    #[stark_test]
    #[ignore]
    fn private_family_fast_text_state_and_expert_storage() {
        use crate::backends::metal::Metal;
        let image = std::env::var("STARK_QWEN_IMAGE_MODEL").expect("private image artifact required");
        let path = Path::new(&image);
        let engine = Engine::<Metal>::new().unwrap();
        let mut model = engine.load_vision_language_model(path).unwrap();
        assert!(model.text_language_model.is_some());
        let mut image_state = model.create_empty_state(Some(128), 42).unwrap();
        let mut text_state = model.create_text_state(Some(128), 42).unwrap();
        let options = model.default_stream_options();
        assert!(model.stream_text(&[100], &mut image_state, options).is_err());
        assert!(!model.is_fenced());
        let options = model.default_stream_options();
        assert!(model.stream(&[100], &mut text_state, options).is_err());
        assert!(!model.is_fenced());
        let stats = model.parameter_storage_statistics();
        let payload = usize::try_from(std::fs::metadata(path.join("model.safetensors")).unwrap().len()).unwrap();
        assert!(stats.shared_matrix_bytes > payload / 2);
        assert!(stats.resident_bytes < payload + 256 * 1024 * 1024);
        assert_eq!(
            stats.resident_bytes,
            stats.shared_matrix_bytes + stats.unshared_parameter_bytes + stats.representation_copy_bytes
        );
        if matches!(model.family_profile, family::FamilyProfile::MoeV4) {
            assert!(stats.shared_expert_matrix_bytes > payload / 2, "expert bank was duplicated");
        }
        let receipt = std::env::var("STARK_QWEN_STORAGE_RECEIPT").expect("private aggregate receipt required");
        let file = OpenOptions::new().write(true).create_new(true).open(receipt).unwrap();
        serde_json::to_writer(file, &stats).unwrap();
    }
}
