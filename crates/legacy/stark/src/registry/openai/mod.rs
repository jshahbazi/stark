mod config;

use std::{future::Future, pin::Pin};

use async_openai::{Client, config::OpenAIConfig};
pub use config::Config;
use fancy_regex::Regex;
use serde::Deserialize;
use shoji::{
    traits::Registry as RegistryTrait,
    types::model::{Model as ShojiModel, ModelAccessibility, ModelSpecialization},
};

#[derive(Debug, Deserialize)]
struct Model {
    id: String,
}

#[derive(Debug, Deserialize)]
struct ListModelsResponse {
    data: Vec<Model>,
}

use crate::registry::RegistryError;

pub struct Registry {
    config: Config,
    client: Client<OpenAIConfig>,
    model_filter: Option<Regex>,
}

impl Registry {
    pub fn new(config: Config) -> Result<Self, RegistryError> {
        let mut openai_config =
            backend_remote::openai::explicit_client_config(&config.api_endpoint, config.api_key.as_deref());
        if let Some(headers) = config.headers.as_ref() {
            for (key, value) in headers {
                let name = reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|error| {
                    RegistryError::UnableToCreate {
                        message: error.to_string(),
                    }
                })?;
                openai_config =
                    openai_config.with_header(name, value.as_str()).map_err(|error| RegistryError::UnableToCreate {
                        message: error.to_string(),
                    })?;
            }
        }

        let client = Client::with_config(openai_config);

        let model_filter = config.model_filter_pattern.as_deref().map(Regex::new).transpose().map_err(|error| {
            RegistryError::UnableToCreate {
                message: error.to_string(),
            }
        })?;

        Ok(Self {
            config,
            client,
            model_filter,
        })
    }
}

impl RegistryTrait for Registry {
    type Error = RegistryError;

    fn identifier(&self) -> String {
        self.config.identifier.clone()
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<ShojiModel>, RegistryError>> + Send + '_>> {
        Box::pin(async {
            let response: ListModelsResponse =
                self.client.models().list_byot().await.map_err(|error| RegistryError::UnableToGetModels {
                    message: error.to_string(),
                })?;
            let mut identifiers = response.data.into_iter().map(|model| model.id).collect::<Vec<_>>();
            identifiers.sort();

            let models = identifiers
                .into_iter()
                .filter(|identifier| {
                    self.model_filter.as_ref().is_none_or(|regex| regex.is_match(identifier).unwrap_or(false))
                })
                .map(|identifier| {
                    ShojiModel::external(
                        identifier.clone(),
                        self.config.identifier.clone(),
                        self.config.name.clone(),
                        self.config.identifier.clone(),
                        self.config.name.clone(),
                        "default".to_string(),
                        vec![ModelSpecialization::Chat {}],
                        ModelAccessibility::Remote {
                            repository: None,
                        },
                        None,
                    )
                })
                .collect();

            Ok(models)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambient_credentials_are_not_forwarded_to_local_providers() {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "registry::openai::tests::environment_child_local_listener"])
            .env("STARK_CREDENTIAL_TEST_CHILD", "1")
            .env("OPENAI_API_KEY", "synthetic-cloud-key")
            .env("OPENAI_ADMIN_KEY", "synthetic-admin-key")
            .env("OPENAI_ORG_ID", "synthetic-organization")
            .env("OPENAI_PROJECT_ID", "synthetic-project")
            .output()
            .unwrap();
        assert!(result.status.success(), "local credential isolation child test failed");
    }

    #[tokio::test]
    async fn environment_child_local_listener() {
        if std::env::var("STARK_CREDENTIAL_TEST_CHILD").as_deref() != Ok("1") {
            return;
        }
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let request = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            loop {
                let mut buffer = [0u8; 1024];
                let read = socket.read(&mut buffer).await.unwrap();
                assert!(read > 0 && request.len() + read <= 8192);
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let request = String::from_utf8(request).unwrap();
            assert!(request.starts_with("GET /v1/models "));
            for private in ["synthetic-cloud-key", "synthetic-admin-key", "synthetic-organization", "synthetic-project"]
            {
                assert!(!request.contains(private));
            }
            let lower = request.to_ascii_lowercase();
            assert!(!lower.contains("openai-organization:"));
            assert!(!lower.contains("openai-project:"));
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"data\":[]}").await.unwrap();
        });
        let registry = Registry::new(Config {
            identifier: "local".into(),
            name: "Local".into(),
            api_endpoint: format!("http://{address}/v1"),
            api_key: None,
            api_type: config::ApiType::Completions,
            model_filter_pattern: None,
            headers: None,
        })
        .unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(5), registry.models())
                .await
                .unwrap()
                .unwrap()
                .is_empty()
        );
        request.await.unwrap();
    }
}
