use std::{io, net::IpAddr, path::PathBuf, pin::Pin, sync::Arc, task::Poll, time::Duration};

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use rocket::{
    Config, Request, Response,
    config::LogLevel,
    data::Data,
    fairing::{Fairing, Info, Kind},
    response::Body,
    routes,
};
use stark::{
    engine::{Engine, EngineConfig},
    types::session::chat::ChatConfig,
};
use tokio::{
    io::{AsyncRead, ReadBuf},
    sync::{Mutex, Semaphore},
};

use crate::{
    common::thinking::ThinkingSupport,
    server::{
        ServerState, handle_chat_completions, handle_models,
        log::{self, FileLogger},
        request_info::RequestInfo,
    },
};

pub struct ResponseBodyLogger<'r> {
    body: Body<'r>,
    prefix: String,
    bytes: u64,
    logged: bool,
    span: tracing::Span,
}

impl<'r> ResponseBodyLogger<'r> {
    pub fn new(
        body: Body<'r>,
        prefix: String,
        _is_json: bool,
        span: tracing::Span,
    ) -> Self {
        Self {
            body,
            prefix,
            bytes: 0,
            logged: false,
            span,
        }
    }

    fn log(
        &mut self,
        suffix: &str,
    ) {
        if self.logged {
            return;
        }
        self.logged = true;

        self.span.in_scope(|| tracing::debug!("{} bytes={}{}", self.prefix, self.bytes, suffix));
    }
}

impl AsyncRead for ResponseBodyLogger<'_> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let span = this.span.clone();
        let _entered = span.enter();
        let filled_before = buffer.filled().len();

        match Pin::new(&mut this.body).poll_read(cx, buffer) {
            Poll::Ready(Ok(())) => {
                let filled_after = buffer.filled().len();
                if filled_after == filled_before {
                    this.log("");
                } else {
                    this.bytes = this.bytes.saturating_add((filled_after - filled_before) as u64);
                }
                Poll::Ready(Ok(()))
            },
            Poll::Ready(Err(error)) => {
                this.log(" [read error]");
                Poll::Ready(Err(error))
            },
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for ResponseBodyLogger<'_> {
    fn drop(&mut self) {
        self.log(" [incomplete]");
    }
}

#[cfg(test)]
mod body_logger_tests {
    use tokio::io::AsyncReadExt;

    use super::*;

    #[tokio::test]
    async fn response_logging_retains_only_a_count_and_preserves_transport_bytes() {
        let payload = b"synthetic private reply\n".repeat(65_536);
        let mut response = Response::build().streamed_body(std::io::Cursor::new(payload.clone())).finalize();
        let body = response.body_mut().take();
        let mut logger = ResponseBodyLogger::new(body, "response".into(), false, tracing::Span::none());
        let mut transported = Vec::new();
        logger.read_to_end(&mut transported).await.unwrap();
        assert_eq!(transported, payload);
        assert_eq!(logger.bytes, payload.len() as u64);
    }
}

struct RequestLoggingFairing;

#[rocket::async_trait]
impl Fairing for RequestLoggingFairing {
    fn info(&self) -> Info {
        Info {
            name: "Request and response logger",
            kind: Kind::Request | Kind::Response,
        }
    }

    async fn on_request(
        &self,
        request: &mut Request<'_>,
        _data: &mut Data<'_>,
    ) {
        let _ = request.local_cache(|| RequestInfo::new(request.method(), request.uri().path().as_str().to_string()));
    }

    async fn on_response<'r>(
        &self,
        request: &'r Request<'_>,
        response: &mut Response<'r>,
    ) {
        if matches!(request.uri().path().as_str(), "/logs") {
            return;
        }

        let request_info =
            request.local_cache(|| RequestInfo::new(request.method(), request.uri().path().as_str().to_string()));
        let _entered = request_info.span.enter();
        let prefix = format!("<-- {} {}", response.status(), request_info.uri);
        if response.body().is_none() {
            tracing::debug!("{prefix} body=<empty>\n");
        } else {
            let is_json = response.content_type().is_some_and(|content_type| content_type.is_json());
            let body = response.body_mut().take();
            response.set_streamed_body(ResponseBodyLogger::new(body, prefix, is_json, request_info.span.clone()));
        }
    }
}

pub async fn run_server(
    model: String,
    host: String,
    port: u16,
    prefix_cache: bool,
    verbose_file_logs: bool,
    allow_unauthenticated_local: bool,
) -> Result<()> {
    let address: IpAddr = host.parse().with_context(|| format!("Invalid host: {host}"))?;
    let access = super::access::ServerAccess::new(
        address,
        std::env::var("STARK_SERVER_API_KEY").ok(),
        allow_unauthenticated_local,
    )
    .map_err(anyhow::Error::msg)?;
    let file_logger = if verbose_file_logs {
        let cache_dir = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
            .or_else(|| dirs::home_dir().map(|path| path.join(".cache")))
            .context("Failed to resolve cache directory")?;
        let logs_dir_path = cache_dir.join("mirai").join("server").join("logs");
        Some(FileLogger::new(logs_dir_path)?)
    } else {
        None
    };
    log::init(tracing::Level::INFO, file_logger.clone(), tracing::Level::DEBUG)?;

    let engine_config = EngineConfig::default().with_application_identifier("com.trymirai.cli".to_string());
    let engine = Engine::new(engine_config).await.context("Failed to create engine")?;

    let resolved = engine.model(model.clone()).await?.with_context(|| format!("Model not found: {model}"))?;

    let spinner = ProgressBar::new_spinner();
    spinner.enable_steady_tick(Duration::from_millis(100));
    spinner.set_style(ProgressStyle::default_spinner().template("{spinner:.green} {msg}").unwrap());
    spinner.set_message(format!("Preparing model: {}", resolved.identifier));
    let downloader = engine.download(&resolved).await.context("Failed to start model download")?;
    while let Some(update) = downloader.next().await {
        spinner.set_message(format!("Downloading {}: {:.0}%", resolved.identifier, update.progress() * 100.0));
    }
    spinner.finish_with_message(format!("Loaded: {}", resolved.identifier));

    let thinking_support = ThinkingSupport::for_model(&resolved);
    let session =
        engine.chat(resolved.clone(), ChatConfig::default()).await.context("Failed to create chat session")?;
    let model_name = resolved.identifier.clone();
    let state = ServerState {
        model_name: model_name.clone(),
        session: Arc::new(Mutex::new(session)),
        thinking_support,
        prefix_cache,
        admission: Arc::new(Semaphore::new(1)),
    };

    let config = Config {
        port,
        address,
        log_level: LogLevel::Off,
        ..Config::default()
    };

    let prefix_cache_enabled = if prefix_cache {
        "enabled"
    } else {
        "disabled"
    };
    if let Some(ref file_logger) = file_logger {
        tracing::info!("🗃️  Logs will be written to file: {}", file_logger.file_path().display())
    }
    tracing::info!(
        concat!(
            "🚀 OpenAI-compatible server v{version} for model: {model_name}\n",
            "🌐 Available at: http://{host}:{port}\n",
            "🗄️  Prefix cache: {prefix_cache_enabled}\n",
            "📝 Endpoints:\n",
            "   POST /v1/chat/completions (or /chat/completions)\n",
            "   GET  /v1/models           (or /models)\n",
        ),
        version = Engine::version(),
        model_name = model_name,
        host = host,
        port = port,
        prefix_cache_enabled = prefix_cache_enabled,
    );

    let rocket = rocket::custom(config)
        .manage(state)
        .manage(access)
        .attach(RequestLoggingFairing)
        .mount("/", routes![handle_chat_completions, handle_models])
        .mount("/v1", routes![handle_chat_completions, handle_models]);

    if let Err(error) = rocket.launch().await {
        bail!("Server error: {error}");
    }

    Ok(())
}
