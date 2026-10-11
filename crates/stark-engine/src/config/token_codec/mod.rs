use stark_engine_macros::stark_config_abstract;

pub mod chat_codec;
pub mod raw_text_codec;

#[stark_config_abstract(chat_codec::ChatCodecConfig, raw_text_codec::RawTextCodecConfig)]
pub struct TokenCodecConfig;
