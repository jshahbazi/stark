use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct TelemetryDevice {
    pub os_name: Option<String>,
    pub cpu_name: Option<String>,
    pub memory_total: i64,
    pub is_environment_sandboxed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TelemetryContext {
    pub engine_session_id: String,
    // The upstream Mirai telemetry endpoint expects this serialized key.
    #[serde(rename = "uzu_version")]
    pub stark_version: String,
    pub toolchain_version: String,
    pub device: TelemetryDevice,
}

// The constructor lives only in non-wasm code so `uuid` is never referenced on wasm.
#[cfg(not(target_family = "wasm"))]
impl TelemetryContext {
    pub fn new(
        stark_version: String,
        toolchain_version: String,
        device: TelemetryDevice,
    ) -> Self {
        Self {
            engine_session_id: uuid::Uuid::new_v4().to_string(),
            stark_version,
            toolchain_version,
            device,
        }
    }
}
