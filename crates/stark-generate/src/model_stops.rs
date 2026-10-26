/// Honor both generation stops and the chat codec's terminal token. Some source
/// checkpoints declare different end-of-text and end-of-turn identifiers.
pub(crate) fn model_stop_token_ids(
    configured: &[u64],
    chat_eos: Option<&str>,
    vocab_size: usize,
    token_to_id: impl Fn(&str) -> Option<u32>,
    contains_id: impl Fn(u32) -> bool,
) -> Result<Vec<u32>, &'static str> {
    let mut stops = Vec::new();
    for &id in configured {
        if id >= vocab_size as u64 || id > u32::MAX as u64 || !contains_id(id as u32) {
            return Err("invalid_model_config");
        }
        if !stops.contains(&(id as u32)) {
            stops.push(id as u32);
        }
    }
    if let Some(eos) = chat_eos.filter(|eos| !eos.is_empty()) {
        let id = token_to_id(eos).ok_or("invalid_model_config")?;
        if id as usize >= vocab_size || !contains_id(id) {
            return Err("invalid_model_config");
        }
        if !stops.contains(&id) {
            stops.push(id);
        }
    }
    Ok(stops)
}

#[cfg(test)]
mod tests {
    use super::model_stop_token_ids;

    fn resolve(
        configured: &[u64],
        eos: Option<&str>,
    ) -> Result<Vec<u32>, &'static str> {
        model_stop_token_ids(configured, eos, 5, |token| (token == "<|im_end|>").then_some(4), |id| id < 5)
    }

    #[test]
    fn chat_end_of_turn_supplements_end_of_text() {
        assert_eq!(resolve(&[2], Some("<|im_end|>")), Ok(vec![2, 4]));
    }

    #[test]
    fn preserves_existing_stops_without_duplicates() {
        assert_eq!(resolve(&[2, 4, 2], Some("<|im_end|>")), Ok(vec![2, 4]));
        assert_eq!(resolve(&[2], None), Ok(vec![2]));
        assert_eq!(resolve(&[2], Some("")), Ok(vec![2]));
    }

    #[test]
    fn rejects_unresolvable_or_invalid_stops() {
        assert_eq!(resolve(&[2], Some("unknown")), Err("invalid_model_config"));
        assert_eq!(resolve(&[5], None), Err("invalid_model_config"));
        assert_eq!(resolve(&[u64::MAX], None), Err("invalid_model_config"));
        assert_eq!(model_stop_token_ids(&[2], None, 5, |_| None, |_| false), Err("invalid_model_config"));
        assert_eq!(model_stop_token_ids(&[], Some("eos"), 5, |_| Some(5), |_| true), Err("invalid_model_config"));
    }
}
