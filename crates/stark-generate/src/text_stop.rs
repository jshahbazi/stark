//! Streaming stop-sequence filtering for decoded text.

#[derive(Debug, Default)]
pub(crate) struct StopFilter {
    stops: Vec<String>,
    buffer: String,
    output: String,
    stopped: bool,
}

impl StopFilter {
    pub(crate) fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            ..Self::default()
        }
    }

    /// Append decoded text, returning newly released text and whether a stop matched.
    pub(crate) fn push(
        &mut self,
        text: &str,
    ) -> (String, bool) {
        if self.stopped {
            return (String::new(), true);
        }
        self.buffer.push_str(text);
        if let Some((index, _)) = self
            .stops
            .iter()
            .filter_map(|stop| self.buffer.find(stop).map(|index| (index, stop)))
            .min_by_key(|(index, stop)| (index + stop.len(), *index))
        {
            let released = self.buffer[..index].to_owned();
            self.output.push_str(&released);
            self.buffer.clear();
            self.stopped = true;
            return (released, true);
        }
        let flush_at = self
            .buffer
            .char_indices()
            .filter_map(|(index, _)| {
                let suffix = &self.buffer[index..];
                self.stops.iter().any(|stop| stop.starts_with(suffix)).then_some(index)
            })
            .next()
            .unwrap_or(self.buffer.len());
        let released = self.buffer[..flush_at].to_owned();
        self.buffer.drain(..flush_at);
        self.output.push_str(&released);
        (released, false)
    }

    pub(crate) fn finish(&mut self) -> String {
        if self.stopped {
            return String::new();
        }
        let released = std::mem::take(&mut self.buffer);
        self.output.push_str(&released);
        released
    }

    pub(crate) fn text(&self) -> &str {
        &self.output
    }
}

#[cfg(test)]
mod tests {
    use super::StopFilter;

    fn collect(
        stops: Vec<&str>,
        chunks: &[&str],
    ) -> (String, bool) {
        let mut filter = StopFilter::new(stops.into_iter().map(str::to_owned).collect());
        let mut stopped = false;
        for chunk in chunks {
            let (_, hit) = filter.push(chunk);
            stopped |= hit;
        }
        filter.finish();
        (filter.text().to_owned(), stopped)
    }

    #[test]
    fn splits_stop_across_chunks() {
        assert_eq!(collect(vec!["<end>"], &["hello <", "en", "d>ignored"]), ("hello ".into(), true));
    }

    #[test]
    fn finds_earliest_match_and_ignores_later_text() {
        assert_eq!(collect(vec!["END", "STOP"], &["aST", "OPbENDc"]), ("a".into(), true));
    }

    #[test]
    fn unicode_chunks_and_stop_are_safe() {
        assert_eq!(collect(vec!["世界"], &["你", "好世", "界后"]), ("你好".into(), true));
    }

    #[test]
    fn prefix_only_is_released_on_finish() {
        assert_eq!(collect(vec!["abc"], &["xab"]), ("xab".into(), false));
    }

    #[test]
    fn overlapping_stops_use_first_completed_match() {
        assert_eq!(collect(vec!["abcd", "bc"], &["a", "b", "cd"]), ("a".into(), true));
    }

    #[test]
    fn empty_stops_are_passthrough() {
        assert_eq!(collect(vec![], &["a", "世界"]), ("a世界".into(), false));
    }

    #[test]
    fn chunked_and_single_input_match() {
        assert_eq!(collect(vec!["XYZ"], &["abXYZtail"]), collect(vec!["XYZ"], &["ab", "X", "Y", "Z", "tail"]));
    }
}
