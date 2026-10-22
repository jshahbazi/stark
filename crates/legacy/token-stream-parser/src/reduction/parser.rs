use crate::{
    Parser,
    framing::{FramingParserOutput, FramingParserSection},
    reduction::{ReductionParserConfig, ReductionParserError, ReductionParserGroupStack, ReductionParserState},
    types::Token,
};

pub struct ReductionParser {
    state: ReductionParserState,
    stack: ReductionParserGroupStack,
    tokens: usize,
}

impl Parser for ReductionParser {
    type Config = ReductionParserConfig;
    type Input = FramingParserOutput;
    type Output = ();
    type State = ReductionParserState;
    type Error = ReductionParserError;

    fn new(config: Self::Config) -> Result<Self, Self::Error> {
        config.validate()?;
        Ok(Self {
            state: ReductionParserState::new(),
            stack: ReductionParserGroupStack::new(config.groups),
            tokens: 0,
        })
    }

    #[tracing::instrument(skip_all)]
    fn push(
        &mut self,
        input: &FramingParserOutput,
    ) -> Result<(), ReductionParserError> {
        let added = match input {
            FramingParserOutput::Added(FramingParserSection::Text(tokens)) => tokens.len(),
            _ => 1,
        };
        let tokens = self.tokens.checked_add(added).ok_or(ReductionParserError::ResourceLimit)?;
        if tokens > super::stack::MAX_TOKENS {
            return Err(ReductionParserError::ResourceLimit);
        }
        self.tokens = tokens;
        match input {
            FramingParserOutput::Added(section) => match section {
                FramingParserSection::Marker(token) => self.handle_frame_marker(token.clone()),
                FramingParserSection::Text(tokens) => self.handle_frame_text_new(tokens.clone()),
            },
            FramingParserOutput::Extended(token) => self.handle_frame_text_extend(token.clone()),
        }
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn reset(&mut self) {
        self.state.sections.clear();
        self.stack.clear();
        self.tokens = 0;
    }
}

impl ReductionParser {
    #[tracing::instrument(skip_all)]
    fn handle_frame_marker(
        &mut self,
        token: Token,
    ) -> Result<(), ReductionParserError> {
        // Close token for an ancestor group
        if self.stack.close_groups_matching_token(&token, &mut self.state)? {
            return Ok(());
        }

        // Open token for a child or root group
        if self.stack.open_group_matching_token(&token, &mut self.state)? {
            return Ok(());
        }

        // No match, add marker as content to current group
        self.stack.open_greedy_group_if_needed(&mut self.state)?;
        self.stack.append_frame_marker(token, &mut self.state)
    }

    #[tracing::instrument(skip_all, fields(tokens = tokens.len()))]
    fn handle_frame_text_new(
        &mut self,
        tokens: Vec<Token>,
    ) -> Result<(), ReductionParserError> {
        self.stack.open_greedy_group_if_needed(&mut self.state)?;
        self.stack.append_frame_text(tokens, &mut self.state)
    }

    #[tracing::instrument(skip_all)]
    fn handle_frame_text_extend(
        &mut self,
        token: Token,
    ) -> Result<(), ReductionParserError> {
        self.stack.open_greedy_group_if_needed(&mut self.state)?;
        self.stack.extend_frame_text(token, &mut self.state)
    }
}

#[cfg(test)]
mod resource_tests {
    use super::*;
    use crate::reduction::{ReductionParserGroup, ReductionParserSection};

    #[test]
    fn repeated_roots_remain_siblings_and_reset_is_safe() {
        let mut parser = ReductionParser::new(ReductionParserConfig {
            groups: vec![ReductionParserGroup::Bounded {
                name: "root".into(),
                open_token: "<root>".into(),
                close_tokens: vec!["</root>".into()],
                closes_on_sibling: false,
                groups: vec![],
            }],
        })
        .unwrap();
        let marker = FramingParserOutput::Added(FramingParserSection::Marker(Token {
            id: 1,
            value: "<root>".into(),
            is_special: true,
        }));
        for _ in 0..1000 {
            parser.push(&marker).unwrap();
        }
        assert_eq!(parser.state().sections.len(), 1000);
        assert!(
            parser.state().sections.iter().all(
                |section| matches!(section, ReductionParserSection::Group { sections, .. } if sections.is_empty())
            )
        );
        parser.reset();
        parser.push(&marker).unwrap();
        assert_eq!(parser.state().sections.len(), 1);
    }

    #[test]
    fn token_budget_rejects_before_retaining_input() {
        let mut parser = ReductionParser::new(ReductionParserConfig {
            groups: vec![],
        })
        .unwrap();
        let token = Token {
            id: 1,
            value: "x".into(),
            is_special: false,
        };
        let oversized =
            FramingParserOutput::Added(FramingParserSection::Text(vec![token; super::super::stack::MAX_TOKENS + 1]));
        assert!(matches!(parser.push(&oversized), Err(ReductionParserError::ResourceLimit)));
        assert!(parser.state().sections.is_empty());
    }
}
