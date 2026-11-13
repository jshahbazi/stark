// Included inside generation::tests so the existing model-free backend fixtures
// exercise the production request dispatcher.

#[test]
fn memory_diagnostics_readiness_is_explicit() {
    let disabled = readiness(&mut worker());
    assert!(disabled.get("memory_diagnostics").is_none());
    assert!(!disabled["operations"].as_array().unwrap().contains(&json!("memory_diagnostics")));

    let mut enabled = worker();
    enabled.enable_memory_diagnostics();
    let ready = readiness(&mut enabled);
    assert_eq!(
        ready["memory_diagnostics"],
        json!({"schema":"self-task-memory-v1","counter_scope":"worker_task","operation":"memory_diagnostics"})
    );
    assert!(ready["operations"].as_array().unwrap().contains(&json!("memory_diagnostics")));
}

fn memory_diagnostics_backend_counts(backend: &FakeBackend) -> (usize, usize, usize, usize, usize, usize, usize) {
    (
        backend.idle_calls,
        backend.generation_calls,
        backend.tokenization_calls,
        backend.image_preparations,
        backend.image_generations,
        backend.state_allocations,
        backend.plans.len(),
    )
}

fn assert_memory_diagnostics_reply(reply: &Value, enabled: bool) {
    if enabled {
        assert_eq!(reply["schema"], "self-task-memory-v1");
        assert_eq!(reply["counter_scope"], "worker_task");
        assert!(reply.get("error").is_none());
    } else {
        assert_eq!(reply, &json!({"error":"memory_diagnostics_unavailable"}));
    }
}

#[test]
fn memory_diagnostics_preserves_prefix_without_backend_work() {
    for enabled in [false, true] {
        let mut worker = prefix_worker(true);
        if enabled {
            worker.enable_memory_diagnostics();
        }
        assert_eq!(reused(&mut worker, 3_000), 0);
        let before = memory_diagnostics_backend_counts(&worker.backend);
        for _ in 0..2 {
            let reply = post(&mut worker, json!({"operation":"memory_diagnostics"}));
            assert_memory_diagnostics_reply(&reply, enabled);
            assert_eq!(memory_diagnostics_backend_counts(&worker.backend), before);
            assert!(worker.backend.prefix_cache_retained());
        }
        assert_eq!(reused(&mut worker, 3_000), 2048);
    }
}

#[test]
fn memory_diagnostics_preserves_prepared_image_without_backend_work() {
    for enabled in [false, true] {
        let root = crate::media::tests::PrivateRoot::new();
        let mut worker = image_worker(&root);
        if enabled {
            worker.enable_memory_diagnostics();
        }
        let prepared = post(&mut worker, image_prepare(&root));
        let before = memory_diagnostics_backend_counts(&worker.backend);
        let id = worker.prepared.as_ref().unwrap().id.clone();
        let reply = post(&mut worker, json!({"operation":"memory_diagnostics"}));
        assert_memory_diagnostics_reply(&reply, enabled);
        assert_eq!(memory_diagnostics_backend_counts(&worker.backend), before);
        assert_eq!(worker.prepared.as_ref().unwrap().id, id);
        assert_eq!(worker.backend.prepared_drops.load(Ordering::SeqCst), 0);
        assert_eq!(post(&mut worker, image_generate(&prepared))["finish_reason"], "eos");
        assert_eq!(worker.backend.prepared_drops.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn memory_diagnostics_rejects_unknown_fields_without_echoing_values() {
    for enabled in [false, true] {
        let mut worker = prefix_worker(true);
        if enabled {
            worker.enable_memory_diagnostics();
        }
        assert_eq!(reused(&mut worker, 3_000), 0);
        let before = memory_diagnostics_backend_counts(&worker.backend);
        let reply = post(&mut worker, json!({"operation":"memory_diagnostics","private_marker":"do-not-echo"}));
        assert_eq!(reply, json!({"error":"invalid_request"}));
        assert_eq!(memory_diagnostics_backend_counts(&worker.backend), before);
        assert!(!worker.backend.prefix_cache_retained());
    }
}
