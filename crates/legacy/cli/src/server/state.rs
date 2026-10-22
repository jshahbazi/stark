use std::sync::Arc;

use stark::session::chat::ChatSession;
use tokio::sync::{Mutex, Semaphore};

use crate::common::thinking::ThinkingSupport;

pub struct ServerState {
    pub model_name: String,
    pub session: Arc<Mutex<ChatSession>>,
    pub thinking_support: ThinkingSupport,
    pub prefix_cache: bool,
    // Admission occurs before reading the request body. One active request
    // bounds retained bodies and prevents a queue behind the shared session.
    pub admission: Arc<Semaphore>,
}
