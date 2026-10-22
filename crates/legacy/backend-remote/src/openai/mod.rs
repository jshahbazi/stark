mod bridging;
mod config;
mod error;
mod instance;
mod stream_state;
mod tool_call_state;

use std::{pin::Pin, sync::Arc};

use async_openai::{Client, config::OpenAIConfig};
pub use config::{ApiType, Config};
pub use error::Error;
pub use instance::Instance;
use shoji::{
    traits::{
        Backend as BackendTrait,
        backend::{Error as BackendError, chat_message},
    },
    types::session::chat::ChatConfig,
};

pub struct Backend {
    config: Config,
    client: Arc<Client<OpenAIConfig>>,
}

/// Override every SDK credential default before using a provider-selected URL.
/// Local compatible providers intentionally have no cloud credentials.
pub fn explicit_client_config(
    endpoint: &str,
    api_key: Option<&str>,
) -> OpenAIConfig {
    OpenAIConfig::new()
        .with_api_base(endpoint)
        .with_api_key(api_key.unwrap_or_default())
        .with_org_id("")
        .with_project_id("")
}

impl Backend {
    pub fn new(config: Config) -> Result<Self, Error> {
        let mut openai_config = explicit_client_config(&config.api_endpoint, config.api_key.as_deref());
        if let Some(headers) = config.headers.as_ref() {
            for (key, value) in headers {
                let name =
                    reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|error| Error::UnableToCreate {
                        message: error.to_string(),
                    })?;
                openai_config =
                    openai_config.with_header(name, value.as_str()).map_err(|error| Error::UnableToCreate {
                        message: error.to_string(),
                    })?;
            }
        }

        Ok(Self {
            config,
            client: Arc::new(Client::with_config(openai_config)),
        })
    }
}

impl BackendTrait for Backend {
    fn identifier(&self) -> String {
        self.config.identifier.clone()
    }

    fn version(&self) -> String {
        "default".to_string()
    }

    fn as_chat_via_message_capable(&self) -> Option<&dyn chat_message::Backend> {
        Some(self)
    }
}

impl chat_message::Backend for Backend {
    fn instance(
        &self,
        reference: String,
        _config: ChatConfig,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn chat_message::Instance>, BackendError>> + Send + '_>> {
        let client = self.client.clone();
        let api_type = self.config.api_type.clone();
        Box::pin(
            async move { Ok(Box::new(Instance::new(client, reference, api_type)) as Box<dyn chat_message::Instance>) },
        )
    }
}

#[cfg(test)]
mod tests {
    use async_openai::config::Config as _;

    use super::explicit_client_config;

    #[test]
    fn provider_credentials_are_explicit() {
        let local = explicit_client_config("http://127.0.0.1:11434/v1", None);
        let headers = local.headers();
        assert_eq!(headers[reqwest::header::AUTHORIZATION], "Bearer ");
        assert!(!headers.contains_key("OpenAI-Organization"));
        assert!(!headers.contains_key("OpenAI-Project"));
        let cloud = explicit_client_config("https://example.invalid/v1", Some("synthetic-key"));
        assert_eq!(cloud.headers()[reqwest::header::AUTHORIZATION], "Bearer synthetic-key");
    }
}
