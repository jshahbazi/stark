use crate::{
    Parser,
    extraction::{ExtractionParser, ExtractionParserConfig, ExtractionParserState},
    framing::FramingParser,
    reduction::ReductionParser,
    token_stream::{TokenStreamParserConfig, TokenStreamParserError},
    types::Token,
};

pub struct TokenStreamParser {
    framing: FramingParser,
    reduction: ReductionParser,
    extraction: ExtractionParser,
    tokens: usize,
}

impl Parser for TokenStreamParser {
    type Config = TokenStreamParserConfig;
    type Input = Token;
    type Output = ();
    type State = ExtractionParserState;
    type Error = TokenStreamParserError;

    fn new(config: Self::Config) -> Result<Self, Self::Error> {
        config.reduction.validate()?;
        config.transformation.validate()?;
        let sections_compose_groups = config.reduction.collect_sections_compose_groups();
        let framing = match FramingParser::new(config.framing_config()) {
            Ok(parser) => parser,
            Err(infallible) => match infallible {},
        };
        let reduction = ReductionParser::new(config.reduction)?;
        let extraction = match ExtractionParser::new(ExtractionParserConfig {
            schema: Some(config.transformation),
            sections_compose_groups,
        }) {
            Ok(parser) => parser,
            Err(infallible) => match infallible {},
        };
        Ok(Self {
            framing,
            reduction,
            extraction,
            tokens: 0,
        })
    }

    #[tracing::instrument(skip_all)]
    fn push(
        &mut self,
        input: &Token,
    ) -> Result<(), TokenStreamParserError> {
        self.push_inner(input, true)
    }

    fn state(&self) -> &ExtractionParserState {
        self.extraction.state()
    }

    fn reset(&mut self) {
        self.framing.reset();
        self.reduction.reset();
        self.extraction.reset();
        self.tokens = 0;
    }
}

impl TokenStreamParser {
    fn push_inner(
        &mut self,
        input: &Token,
        extract: bool,
    ) -> Result<(), TokenStreamParserError> {
        // Check before framing clones or retains input and before reduction mutates.
        if self.tokens >= 262_144 {
            return Err(crate::reduction::ReductionParserError::ResourceLimit.into());
        }
        self.tokens += 1;
        let event = match self.framing.push(input) {
            Ok(event) => event,
            Err(infallible) => match infallible {},
        };
        self.reduction.push(&event)?;
        if extract {
            self.flush_extraction();
        }
        Ok(())
    }

    /// Framing and reduction only: extraction output is recomputed from the whole reduction
    /// state on every push, which is quadratic over bulk loads. Call `flush_extraction` once
    /// after the last `push_bulk`.
    pub fn push_bulk(
        &mut self,
        input: &Token,
    ) -> Result<(), TokenStreamParserError> {
        self.push_inner(input, false)
    }

    pub fn flush_extraction(&mut self) {
        match self.extraction.push(self.reduction.state()) {
            Ok(()) => {},
            Err(infallible) => match infallible {},
        }
    }

    /// Exposes `value` to every transformation pipeline as `$<name>`.
    pub fn set_variable(
        &mut self,
        name: &str,
        value: serde_json::Value,
    ) {
        self.extraction.set_variable(name, value);
    }

    pub fn framing(&self) -> &FramingParser {
        &self.framing
    }

    pub fn reduction(&self) -> &ReductionParser {
        &self.reduction
    }

    pub fn extraction(&self) -> &ExtractionParser {
        &self.extraction
    }
}
