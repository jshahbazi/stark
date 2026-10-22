use tokio::sync::mpsc;

use super::{TelemetryContext, body::build_body, events::Events, record::TelemetryRecord};
use crate::api::Client;

pub(super) async fn run(
    client: Client,
    context: TelemetryContext,
    mut receiver: mpsc::Receiver<TelemetryRecord>,
) {
    while let Some(record) = receiver.recv().await {
        if client.send::<Events>(&build_body(&context, &record)).await.is_err() {
            tracing::warn!("telemetry event dropped");
        }
    }
}
