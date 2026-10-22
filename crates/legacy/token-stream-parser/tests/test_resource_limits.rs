use token_stream_parser::{
    Parser, ParserState,
    token_stream::{TokenStreamParser, TokenStreamParserConfig},
    types::Token,
};

#[test]
fn shipped_configs_accept_normal_framed_input_without_a_model() {
    let config_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../hanashi/configs/parsing");
    for entry in std::fs::read_dir(config_path).unwrap() {
        let entry = entry.unwrap();
        if entry.path().extension().is_none_or(|extension| extension != "json") {
            continue;
        }
        let config: TokenStreamParserConfig = serde_json::from_slice(&std::fs::read(entry.path()).unwrap()).unwrap();
        let mut parser = TokenStreamParser::new(config).unwrap();
        parser
            .push(&Token {
                id: 1,
                value: "synthetic ordinary text".into(),
                is_special: false,
            })
            .unwrap();
        assert_eq!(parser.framing().state().tokens().len(), 1);
        parser.reset();
        assert!(parser.framing().state().tokens().is_empty());
    }
}

#[test]
fn parser_construction_rejects_cyclic_transformations() {
    let config: TokenStreamParserConfig = serde_json::from_value(serde_json::json!({
        "reduction": {"groups": []}, "transformation": {"root": [{"type":"call", "name":"root"}]}
    }))
    .unwrap();
    assert!(TokenStreamParser::new(config).is_err());
}
