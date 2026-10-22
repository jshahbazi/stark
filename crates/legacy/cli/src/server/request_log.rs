use std::time::Instant;

use stark::types::session::chat::ChatReplyStats;

/// Per-request tracing: one event when the request arrives, one when it ends,
/// correlated by a short tag derived from the request id.
pub struct RequestLog {
    started: Instant,
}

impl RequestLog {
    pub fn start(
        stream: bool,
        messages: usize,
        tools: usize,
        reasoning_effort: Option<&str>,
    ) -> Self {
        let stream = if stream {
            "stream"
        } else {
            "blocking"
        };
        tracing::info!(
            "received: {messages} messages, {}, {tools} tools, reasoning_effort={}",
            stream,
            reasoning_effort_category(reasoning_effort),
        );
        Self {
            started: Instant::now(),
        }
    }

    pub fn finish(
        &self,
        finish_reason: &str,
        stats: Option<&ChatReplyStats>,
        notes: Vec<String>,
    ) {
        let mut parts =
            vec![format!("completed in {:.2}s, finish={finish_reason}", self.started.elapsed().as_secs_f64())];
        if let Some(stats) = stats {
            let cached = stats.tokens_count_input_cached.unwrap_or(0);
            let prefilled = stats.tokens_count_input.unwrap_or(0);
            parts.push(format!(
                "prompt {} tok ({} cached, {} prefilled @ {} tok/s)",
                cached + prefilled,
                cached,
                prefilled,
                stats.prefill_tokens_per_second.map_or("-".to_string(), |tps| format!("{tps:.1}")),
            ));
            if let Some(ttft) = stats.time_to_first_token {
                parts.push(format!("ttft {ttft:.2}s"));
            }
            if let Some(generated) = stats.tokens_count_output {
                parts.push(format!(
                    "decode {generated} tok @ {} tok/s",
                    stats.generate_tokens_per_second.map_or("-".to_string(), |tps| format!("{tps:.1}")),
                ));
            }
            if let Some(speculator) = &stats.speculator_stats {
                parts.push(format!(
                    "spec {:.2} tok/pass ({} passes)",
                    speculator.tokens_per_forward_pass, speculator.num_decode_forward_passes
                ));
            }
        }
        parts.extend(notes);
        tracing::info!("{}", parts.join(", "));
    }

    pub fn fail(
        &self,
        _error: &str,
    ) {
        tracing::info!("failed in {:.2}s", self.started.elapsed().as_secs_f64());
    }
}

fn reasoning_effort_category(value: Option<&str>) -> &'static str {
    match value {
        None | Some("unspecified") => "unspecified",
        Some("default") => "default",
        Some("disabled") => "disabled",
        Some("minimal") => "minimal",
        Some("low") => "low",
        Some("medium") => "medium",
        Some("high") => "high",
        Some("xhigh") => "xhigh",
        Some(_) => "invalid",
    }
}

#[cfg(test)]
mod tests {
    use super::reasoning_effort_category;

    #[test]
    fn reasoning_diagnostics_allow_only_fixed_categories() {
        assert_eq!(reasoning_effort_category(Some("high")), "high");
        assert_eq!(reasoning_effort_category(Some("high\nforged private-marker")), "invalid");
        assert_eq!(reasoning_effort_category(Some("\u{1b}[2J")), "invalid");
    }
}
